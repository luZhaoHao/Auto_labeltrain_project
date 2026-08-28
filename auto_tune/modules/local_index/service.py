"""Service layer for the local SQLite stable index (Studio S2 Core).

Owns domain projection (dataset/experiment), path constraints, idempotent
registration, bounded read-only legacy-import, and stable error mapping for the
API/UI. SQLite failures surface as ``LocalIndexError``; they never change the
training/run-state facts and never block report generation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from auto_tune.modules.input_safety import (
    InputSafetyError,
    InputSafetyPolicy,
    validate_directory_path,
)

from .database import connect_database, initialize_database
from .detail import (
    build_dataset_experiments,
    build_experiment_detail,
    compute_best_experiment,
    group_best_by_task,
)
from .models import (
    ArtifactRecord,
    DatasetRecord,
    ExperimentQuery,
    ExperimentRecord,
    ImportFailure,
    ImportSummary,
    LocalIndexConfig,
    LocalIndexConfigError,
    LocalIndexError,
    LocalIndexPersistenceError,
)
from .reconciliation import (
    MAX_BACKFILL_RECORDS,
    audit_index,
    backfill_startup,
    rebuild_index,
)
from .repository import LocalIndexRepository
from auto_tune.modules.presentation.experiment_views import (
    ExperimentNotFoundError,
    build_audit_view,
    build_report_view,
)

MAX_LEGACY_FILE_BYTES = 16 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
MAX_RECENT_TRAINING_CANDIDATES = 20

_LEGACY_STATUS_MAP = {
    "done": "completed",
    "error": "failed",
    "aborted": "cancelled",
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_path(raw: str) -> str:
    """Normalize a path for indexing; only absolute paths are accepted."""
    if not isinstance(raw, str) or not raw:
        raise LocalIndexConfigError("dataset path must be a non-empty string")
    if not os.path.isabs(raw):
        raise LocalIndexConfigError("dataset path must be absolute")
    return os.path.normcase(os.path.normpath(os.path.abspath(raw)))


def _normalize_reference_path(raw: object) -> str | None:
    """Normalize a file reference (absolute or cwd-relative) for matching.

    ``data_yaml_path`` from a production ``latest_dataset.json`` may be a
    relative ``log/...`` value; both sides of the dataset<->experiment lookup
    use this same normalization so the foreign key resolves to the snapshot.
    """
    if not isinstance(raw, str) or not raw:
        return None
    return os.path.normcase(os.path.normpath(os.path.abspath(raw)))


def _dataset_id_for(canonical_path: str, snapshot_id: str | None) -> str:
    if snapshot_id:
        material = f"snapshot:{snapshot_id}"
    else:
        material = f"path:{canonical_path}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _map_status(status: str) -> str:
    if status in ("starting", "running", "completed", "failed", "cancelled", "interrupted", "unknown"):
        return status
    return _LEGACY_STATUS_MAP.get(status, "unknown")


class _LegacyFileTooLarge(Exception):
    pass


class _ImportFileError(Exception):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _safe_failure_message(path: Path, error_code: str) -> str:
    return f"{os.path.basename(str(path))}: {error_code}"


def _read_bounded(path: Path) -> tuple[bytes, str]:
    """Read a legacy JSON file with bounded, chunked reads.

    The declared size is only a fast-path guard: the stream is actually read in
    1 MiB chunks and abandoned as soon as the accumulated total exceeds
    ``MAX_LEGACY_FILE_BYTES``, so a file that grows during the read is still
    bounded. The SHA-256 is computed incrementally over the same chunks.
    Returns ``(data, sha256_hexdigest)``.
    """
    if os.path.getsize(path) > MAX_LEGACY_FILE_BYTES:
        raise _LegacyFileTooLarge()
    sha = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    with open(path, "rb") as fh:
        while True:
            # The single-file and total budgets coincide for a legacy file; each
            # read is capped at the chunk size with a one-byte allowance to
            # detect a file that grew past the limit mid-read.
            read_len = min(_CHUNK_SIZE, MAX_LEGACY_FILE_BYTES - total + 1)
            chunk = fh.read(read_len)
            if not chunk:
                break
            sha.update(chunk)
            total += len(chunk)
            if total > MAX_LEGACY_FILE_BYTES:
                raise _LegacyFileTooLarge()
            chunks.append(chunk)
    return b"".join(chunks), sha.hexdigest()


def _legacy_status(item: dict) -> str:
    error = item.get("error")
    if error:
        if "取消" in str(error):
            return "cancelled"
        return "failed"
    if item.get("train_name") or item.get("result_mAP50") is not None:
        return "completed"
    probe = item.get("probe_decision") or {}
    if probe.get("verdict") == "continue":
        return "completed"
    return "unknown"


def _legacy_analysis_status(item: dict) -> str:
    if item.get("result_mAP50") is not None or item.get("result_mAP50_95") is not None:
        return "completed"
    return "unknown"


def _legacy_metrics(item: dict) -> dict:
    mapping = {
        "result_mAP50": "mAP50",
        "result_mAP50_95": "mAP50_95",
        "result_precision": "precision",
        "result_recall": "recall",
    }
    metrics: dict = {}
    for source, target in mapping.items():
        value = item.get(source)
        if value is not None:
            metrics[target] = value
    return metrics


def _legacy_to_experiment_record(item: dict, index: int) -> dict:
    run_name = item.get("train_name")
    timestamp = item.get("timestamp") or item.get("finished_at") or ""
    if run_name:
        run_id = f"legacy-tuning:{run_name}"
    else:
        run_id = f"legacy-tuning:{index}:{timestamp}"
    return {
        "run_id": run_id,
        "source": "tuning",
        "run_name": run_name,
        "status": _legacy_status(item),
        "analysis_status": _legacy_analysis_status(item),
        "metrics": _legacy_metrics(item),
        "finished_at": timestamp,
        "params": {},
        "decision": item.get("decision"),
        "probe_decision": item.get("probe_decision"),
        "_legacy": True,
    }


class LocalIndexService:
    """Application-facing facade over the local index storage/repository."""

    def __init__(self, config: LocalIndexConfig) -> None:
        self.config = config
        self._initialized = False

    def initialize(self) -> int:
        version = initialize_database(self.config)
        self._initialized = True
        return version

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _open(self):
        self._ensure_initialized()
        return connect_database(self.config)

    # ── dataset projection ──

    def index_dataset(self, payload: dict) -> DatasetRecord:
        """Project and upsert a dataset from a ``latest_dataset.json``-shaped payload."""
        if not isinstance(payload, dict):
            raise LocalIndexConfigError("dataset payload must be a mapping")
        canonical_raw = payload.get("source_dataset_path") or payload.get("dataset_path") or ""
        canonical_path = _normalize_path(canonical_raw)
        data_yaml_path = _normalize_reference_path(payload.get("data_yaml_path"))
        snapshot_id = payload.get("snapshot_id") or None
        dataset_id = payload.get("dataset_id") or _dataset_id_for(canonical_path, snapshot_id)

        if payload.get("snapshot_valid") is True:
            validation_status = "valid"
        elif payload.get("snapshot_valid") is False:
            validation_status = "invalid"
        elif snapshot_id:
            # Registration only happens after the S1.2 validation gate.
            validation_status = "valid"
        else:
            validation_status = "unverified"

        now = _utc_now_iso()
        record = DatasetRecord(
            dataset_id=dataset_id,
            display_name=payload.get("display_name") or os.path.basename(canonical_path.rstrip("/\\")),
            canonical_path=canonical_path,
            data_yaml_path=data_yaml_path,
            snapshot_id=snapshot_id,
            snapshot_digest=payload.get("snapshot_manifest_digest") or None,
            validation_status=validation_status,
            created_at=payload.get("snapshot_created_at") or payload.get("upload_time") or now,
            updated_at=now,
            last_used_at=payload.get("last_used_at") or now,
        )
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            repo.upsert_dataset(record)
            return record
        finally:
            conn.close()

    def list_datasets(self) -> list[dict]:
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            return [asdict(d) for d in repo.list_datasets()]
        finally:
            conn.close()

    def get_dataset(self, dataset_id: str) -> dict | None:
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            found = repo.get_dataset(dataset_id)
            return asdict(found) if found is not None else None
        finally:
            conn.close()

    # ── experiment projection ──

    def index_experiment(
        self, record: dict, runtime_run_id: str | None = None, dataset_id: str | None = None
    ) -> dict:
        """Project and upsert one experiment; ``runtime_run_id`` is the S1.5 key.

        The original JSON ``record["run_id"]`` is preserved in ``params_json``
        compatibility metadata when the SQLite key differs. An explicit
        ``dataset_id`` (the P2 frozen resolution identity) is authoritative: the
        dataset row must exist or a stable ``LocalIndexPersistenceError`` is
        raised — it is never replaced by a params.data path match. Only when no
        explicit ``dataset_id`` is supplied does the legacy data_yaml lookup
        apply (a missing row stays NULL honestly).
        """
        if not isinstance(record, dict):
            raise LocalIndexPersistenceError("experiment record must be a mapping")

        sqlite_run_id = runtime_run_id if runtime_run_id else (record.get("run_id") or "")
        if not sqlite_run_id:
            raise LocalIndexPersistenceError("experiment record requires run_id")

        source = record.get("source") or "unknown"
        if source not in ("manual", "tuning"):
            source = "unknown"
        status = _map_status(str(record.get("status") or "unknown"))
        params_raw = record.get("params") if isinstance(record.get("params"), dict) else {}
        params = dict(params_raw)
        if runtime_run_id and record.get("run_id"):
            params.setdefault("_legacy_record_run_id", record["run_id"])
        for meta_key, meta_value in (
            ("_epochs", record.get("epochs")),
            ("_tuning", record.get("tuning")),
            ("_decision", record.get("decision")),
            ("_probe_decision", record.get("probe_decision")),
            ("_audit_path", record.get("audit_path")),
        ):
            if meta_value is not None:
                params.setdefault(meta_key, meta_value)

        data_path = params.get("data")
        artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
        artifact_records = []
        for kind, path in (
            ("report", artifacts.get("report_path")),
            ("run_dir", artifacts.get("run_dir")),
            ("audit", record.get("audit_path")),
        ):
            if path:
                artifact_records.append(ArtifactRecord(
                    run_id=sqlite_run_id, kind=kind, path=str(path),
                ))

        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            if dataset_id:
                # Explicit identity is authoritative: a missing dataset row must
                # fail instead of silently associating the experiment with a
                # different dataset matched by params.data.
                if repo.get_dataset(dataset_id) is None:
                    raise LocalIndexPersistenceError(
                        "reference dataset not found in local index"
                    )
            else:
                lookup_path = _normalize_reference_path(data_path)
                if lookup_path is not None:
                    dataset = repo.find_dataset_by_data_yaml(lookup_path)
                    dataset_id = dataset.dataset_id if dataset is not None else None

            exp_record = ExperimentRecord(
                run_id=sqlite_run_id,
                source=source,
                run_name=record.get("run_name"),
                dataset_id=dataset_id,
                status=status,
                phase=record.get("phase"),
                model_name=params.get("model"),
                task_type=params.get("task"),
                started_at=record.get("started_at"),
                finished_at=record.get("finished_at"),
                params=params,
                metrics=record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
                analysis_status=record.get("analysis_status"),
                error=record.get("error") if isinstance(record.get("error"), dict) else None,
                updated_at=record.get("updated_at") or _utc_now_iso(),
            )
            repo.upsert_experiment(exp_record, artifacts=tuple(artifact_records))
            return repo.get_experiment(sqlite_run_id) or {}
        finally:
            conn.close()

    def list_experiments(self, query: ExperimentQuery) -> list[dict]:
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            return repo.list_experiments(query)
        finally:
            conn.close()

    def find_reference_experiments(self, run_name: str) -> list[dict]:
        """Return every indexed experiment matching one run_name (exact).

        Used by the reference-dataset resolver to detect a unique dataset
        association or an ambiguity. Storage failures surface as stable
        ``LocalIndexError`` values; they are never disguised as empty results.
        """
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            return repo.list_experiments_by_run_name(run_name)
        finally:
            conn.close()

    def query_experiments(self, query: ExperimentQuery) -> dict:
        """Paginated, searchable/sortable query returning a stable page.

        Pagination is done in SQL (LIMIT/OFFSET); the full record set is never
        loaded into memory.
        """
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            items = repo.list_experiments(query)
            total = repo.count_experiments_matching(query)
            return {
                "items": items,
                "total": total,
                "limit": query.limit,
                "offset": query.offset,
            }
        finally:
            conn.close()

    # ── recent training runs (Bugfix P3) ──

    def recent_training_runs(
        self, limit: int = 4, policy: InputSafetyPolicy | None = None
    ) -> list[dict]:
        """Verified recent completed detect runs for the analysis shortcut.

        Candidates are bounded to the most recent completed ``detect``
        experiments that carry a ``run_dir`` artifact. Every candidate's run_dir
        is live-validated through the S1.4 input_safety policy (allowed roots,
        links/reparse points, permissions, resolution) and must still contain
        ``args.yaml`` and ``results.csv``. Invalid records are skipped until
        ``limit`` are collected or the bounded candidates are exhausted; the
        client-supplied path is never accepted (run_dir comes only from the
        registered artifact). Storage failures surface as stable
        ``LocalIndexError`` values, never as empty results.
        """
        bounded = max(1, min(int(limit), 4))
        safe_policy = policy if policy is not None else InputSafetyPolicy()
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            candidates = repo.list_recent_completed_detect(MAX_RECENT_TRAINING_CANDIDATES)
        finally:
            conn.close()

        items: list[dict] = []
        for exp in candidates:
            if len(items) >= bounded:
                break
            run_dir = self._verified_run_dir(exp, safe_policy)
            if run_dir is None:
                continue
            metrics = exp.get("metrics") if isinstance(exp.get("metrics"), dict) else {}
            items.append({
                "run_id": exp.get("run_id"),
                "run_name": exp.get("run_name"),
                "source": exp.get("source"),
                "model_name": exp.get("model_name"),
                "finished_at": exp.get("finished_at"),
                "map50": metrics.get("mAP50"),
                "run_dir": run_dir,
            })
        return items

    @staticmethod
    def _verified_run_dir(exp: dict, policy: InputSafetyPolicy) -> str | None:
        """Return the live-validated run_dir of one experiment, or None to skip.

        A tuning runtime run_id accumulates one ``run_dir`` artifact per
        iteration, so the first artifact in storage order is NOT authoritative.
        Only ``kind == 'run_dir'`` artifact paths are considered (never a client
        value); each must pass input_safety validation and contain ``args.yaml``
        + ``results.csv``. Selection then prefers the single directory whose
        basename equals ``run_name`` (normalized cross-platform). With no exact
        match, a lone valid run_dir is kept for legacy records; several valid
        run_dirs with no exact match are skipped rather than guessed.
        """
        valid: list[str] = []
        for artifact in exp.get("artifacts") or ():
            if not isinstance(artifact, dict):
                continue
            if artifact.get("kind") != "run_dir":
                continue
            raw = artifact.get("path")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                root = validate_directory_path(raw, policy)
            except (InputSafetyError, OSError):
                continue
            run_dir = str(root)
            if not os.path.exists(os.path.join(run_dir, "args.yaml")):
                continue
            if not os.path.exists(os.path.join(run_dir, "results.csv")):
                continue
            valid.append(run_dir)
        if not valid:
            return None
        run_name = exp.get("run_name")
        target = (
            os.path.normcase(os.path.normpath(run_name))
            if isinstance(run_name, str) and run_name else None
        )
        if target is not None:
            matches = [
                rd for rd in valid
                if os.path.normcase(os.path.normpath(os.path.basename(rd))) == target
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                # Two artifacts share the matching basename: ambiguous, do not guess.
                return None
        if len(valid) == 1:
            return valid[0]
        return None

    # ── comparison (S2.4) ──

    def compare_experiments(self, run_ids: list[str], baseline_run_id: str) -> dict:
        """Compare 2-5 unique experiments against one baseline (read-only)."""
        from .compare import COMPARE_MAX_RUNS, COMPARE_MIN_RUNS, compare_experiments
        from .models import LocalIndexCompareError, LocalIndexNotFoundError

        if not isinstance(run_ids, list) or not all(isinstance(r, str) and r for r in run_ids):
            raise LocalIndexCompareError("run_ids must be a list of non-empty strings")
        unique = list(dict.fromkeys(run_ids))
        if len(unique) < COMPARE_MIN_RUNS or len(unique) > COMPARE_MAX_RUNS:
            raise LocalIndexCompareError(
                f"comparison requires {COMPARE_MIN_RUNS}-{COMPARE_MAX_RUNS} unique experiments"
            )
        if not isinstance(baseline_run_id, str) or baseline_run_id not in unique:
            raise LocalIndexCompareError("baseline_run_id must be one of the selected run_ids")

        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            rows = repo.list_experiments_by_ids(unique)
            found = {row["run_id"]: row for row in rows}
            missing = [rid for rid in unique if rid not in found]
            if missing:
                raise LocalIndexNotFoundError("some experiments were not found")
            return compare_experiments([found[rid] for rid in unique], baseline_run_id)
        finally:
            conn.close()

    # ── diagnostics / maintenance (S2.3) ──

    def diagnostics(self) -> dict:
        from .diagnostics import build_diagnostics

        return build_diagnostics(self.config)

    def checkpoint(self) -> dict:
        """Run a WAL checkpoint; a busy result is reported honestly, not hidden."""
        conn = self._open()
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except LocalIndexError:
            raise
        except Exception as exc:
            from .models import LocalIndexPersistenceError

            raise LocalIndexPersistenceError("checkpoint failed") from exc
        finally:
            conn.close()
        busy, log_pages, checkpointed = (int(row[0]), int(row[1]), int(row[2])) if row else (1, 0, 0)
        status = "ok" if busy == 0 else "busy"
        return {
            "status": status,
            "busy": busy,
            "log_pages": log_pages,
            "checkpointed_pages": checkpointed,
        }

    def backup(self) -> dict:
        """Create a manual backup and record a maintenance event."""
        from .database import create_backup

        self._ensure_initialized()
        dest = create_backup(self.config)
        conn = self._open()
        try:
            LocalIndexRepository(conn).record_maintenance_event("backup", {"status": "ok"})
        finally:
            conn.close()
        return {
            "status": "ok",
            "backup_filename": os.path.basename(str(dest)),
            "backup_count": len([p for p in Path(self.config.backup_dir).glob(
                f"{Path(self.config.database_path).name}.*.bak"
            )]) if Path(self.config.backup_dir).is_dir() else 1,
        }

    def get_experiment(self, run_id: str) -> dict | None:
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            return repo.get_experiment(run_id)
        finally:
            conn.close()

    # ── detail / dataset association (S2.2) ──

    def get_experiment_detail(self, run_id: str) -> dict | None:
        """Extended, sanitized experiment detail with dataset + artifact manifest."""
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            experiment = repo.get_experiment(run_id)
            if experiment is None:
                return None
            dataset = None
            if experiment.get("dataset_id"):
                row = repo.get_dataset(experiment["dataset_id"])
                dataset = asdict(row) if row is not None else None
            return build_experiment_detail(experiment, dataset)
        finally:
            conn.close()

    def _artifact_root(self) -> tuple[str, ...]:
        """Controlled root for project artifacts: the fact-log directory.

        The log dir next to the index database is where report/audit fact files
        live (both in production ``log/`` and under an isolated tmp log in
        tests). This is the project's own artifact boundary — the dataset
        ``allowed_roots`` policy is deliberately NOT reused here so internal log
        reports stay readable.
        """
        return (os.path.realpath(os.path.abspath(str(Path(self.config.database_path).parent))),)

    def get_report_view(self, run_id: str) -> dict:
        """Bounded, read-only report display model for one SQLite run_id.

        The report must be a registered artifact of this exact experiment and
        its content must reference the same run_name. SQLite failures surface as
        stable ``LocalIndexError`` values; projection problems raise the P5
        ``ExperimentViewError`` family.
        """
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            experiment = repo.get_experiment(run_id)
            dataset = None
            if experiment is not None and experiment.get("dataset_id"):
                row = repo.get_dataset(experiment["dataset_id"])
                dataset = asdict(row) if row is not None else None
        finally:
            conn.close()
        if experiment is None:
            raise ExperimentNotFoundError("experiment not found")
        return build_report_view(experiment, dataset, artifact_roots=self._artifact_root())

    def get_audit_view(self, run_id: str) -> dict:
        """Bounded, read-only tuning-audit display model for one SQLite run_id."""
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            experiment = repo.get_experiment(run_id)
            dataset = None
            if experiment is not None and experiment.get("dataset_id"):
                row = repo.get_dataset(experiment["dataset_id"])
                dataset = asdict(row) if row is not None else None
        finally:
            conn.close()
        if experiment is None:
            raise ExperimentNotFoundError("experiment not found")
        return build_audit_view(experiment, dataset, artifact_roots=self._artifact_root())

    def get_dataset_experiments(self, dataset_id: str, limit: int = 10) -> dict | None:
        """Dataset association summary: count, best facts, recent experiments.

        All counts/best facts are computed with dataset-scoped SQL queries; the
        full experiment projection is never loaded into memory per dataset.
        """
        bounded = max(1, min(int(limit), 100))
        conn = self._open()
        try:
            repo = LocalIndexRepository(conn)
            row = repo.get_dataset(dataset_id)
            if row is None:
                return None
            dataset = asdict(row)
            recent = repo.list_experiments_by_dataset(dataset_id, limit=bounded)
            count = repo.count_experiments_by_dataset(dataset_id)
            best_rows = repo.best_experiments_by_dataset(dataset_id)
            best = compute_best_experiment(best_rows, dataset_id)
            best_by_task = group_best_by_task(best_rows, dataset_id)
            return build_dataset_experiments(dataset, count, recent, best, best_by_task)
        finally:
            conn.close()

    # ── bounded, read-only, idempotent legacy import ──

    def import_legacy_files(self, paths) -> ImportSummary:
        imported = 0
        skipped = 0
        failed = 0
        failures: list[ImportFailure] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                failed += 1
                failures.append(ImportFailure(
                    str(path), "LEGACY_IMPORT_MISSING", _safe_failure_message(path, "LEGACY_IMPORT_MISSING")
                ))
                continue
            try:
                data, digest = _read_bounded(path)
            except _LegacyFileTooLarge:
                failed += 1
                failures.append(ImportFailure(
                    str(path), "LEGACY_IMPORT_TOO_LARGE", _safe_failure_message(path, "LEGACY_IMPORT_TOO_LARGE")
                ))
                continue
            conn = self._open()
            try:
                repo = LocalIndexRepository(conn)
                if repo.has_legacy_import(str(path), digest):
                    skipped += 1
                    continue
                try:
                    count = self._import_payload(repo, data, path)
                except _ImportFileError as exc:
                    failed += 1
                    failures.append(ImportFailure(
                        str(path), exc.error_code, _safe_failure_message(path, exc.error_code)
                    ))
                    repo.record_legacy_import(str(path), digest, "failed", 0, exc.error_code)
                    continue
                imported += 1
                repo.record_legacy_import(str(path), digest, "imported", count, None)
            finally:
                conn.close()
        return ImportSummary(
            source_files=len(paths),
            imported=imported,
            skipped=skipped,
            failed=failed,
            failures=tuple(failures),
        )

    def _import_payload(self, repo: LocalIndexRepository, data: bytes, path: Path) -> int:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _ImportFileError("LEGACY_IMPORT_INVALID_JSON") from exc
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise _ImportFileError("LEGACY_IMPORT_INVALID_JSON") from exc

        count = 0
        if isinstance(parsed, dict) and isinstance(parsed.get("experiments"), list):
            for item in parsed["experiments"]:
                if not isinstance(item, dict):
                    continue
                self.index_experiment(item)
                count += 1
            return count
        if isinstance(parsed, list):
            for index, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                self.index_experiment(_legacy_to_experiment_record(item, index))
                count += 1
            return count
        raise _ImportFileError("LEGACY_IMPORT_INVALID_SCHEMA")

    # ── reconciliation: audit / rebuild / startup backfill ──

    def _fact_log_dir(self, log_dir: str | None) -> str:
        """Resolve the fact-files directory next to the index database."""
        if log_dir is not None:
            return log_dir
        return str(Path(self.config.database_path).parent)

    def audit(self, log_dir: str | None = None) -> dict:
        """Run a read-only reconciliation audit and return a stable dict.

        The audit itself never initializes, migrates, or writes to the database;
        persistence of an audit summary is an explicit separate operation.
        """
        from dataclasses import asdict

        result = audit_index(self.config, log_dir=self._fact_log_dir(log_dir))
        return asdict(result)

    def persist_audit(self, log_dir: str | None = None) -> dict:
        """Explicitly run an audit AND persist a bounded summary event.

        Called only from a CSRF-gated POST endpoint, never from the read-only
        GET audit route.
        """
        result = self.audit(log_dir=log_dir)
        try:
            conn = self._open()
            try:
                LocalIndexRepository(conn).record_maintenance_event("audit", {
                    "status": "ok" if result.get("error_code") is None else "failed",
                    "error_code": result.get("error_code"),
                    "scanned": result.get("scanned", 0),
                    "counts": result.get("counts", {}),
                })
            finally:
                conn.close()
        except LocalIndexError:
            pass
        return result

    def rebuild(self, log_dir: str | None = None) -> dict:
        """Atomically rebuild the index (backup -> temp -> publish)."""
        from dataclasses import asdict

        result = rebuild_index(self.config, log_dir=self._fact_log_dir(log_dir))
        return asdict(result)

    def backfill_startup(self, log_dir: str | None = None, max_records: int = 50) -> dict:
        """Catch up bounded recent JSON facts; idempotent and honest."""
        from dataclasses import asdict

        bounded = max(1, min(int(max_records), MAX_BACKFILL_RECORDS))
        result = backfill_startup(
            self.config, log_dir=self._fact_log_dir(log_dir), max_records=bounded
        )
        return asdict(result)

    # ── status ──

    def status(self) -> dict:
        result = {
            "enabled": True,
            "available": False,
            "schema_version": None,
            "dataset_count": 0,
            "experiment_count": 0,
            "last_import": None,
            "error_code": None,
        }
        try:
            if not self._initialized:
                self.initialize()
            conn = self._open()
            try:
                repo = LocalIndexRepository(conn)
                result["schema_version"] = repo.schema_version()
                result["dataset_count"] = len(repo.list_datasets())
                result["experiment_count"] = repo.count_experiments()
                result["last_import"] = repo.latest_legacy_import_at()
            finally:
                conn.close()
            result["available"] = True
        except LocalIndexError as exc:
            result["error_code"] = exc.error_code
            result["available"] = False
        return result
