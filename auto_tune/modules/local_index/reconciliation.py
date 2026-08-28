"""Reconciliation: read-only audit, atomic rebuild and bounded startup backfill.

SQLite is a rebuildable projection over the controlled fact files
(experiment_history.json, tuning_history.json, latest_dataset.json). This module
compares the projection against those facts (audit), reconstructs it atomically
(rebuild), and catches up bounded recent facts on startup (backfill). It never
mutates the fact files, never guesses an unknown terminal state into a
completion, and never stores more than a bounded summary in the index.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .database import (
    check_database_integrity,
    connect_database,
    connect_database_readonly,
    create_backup,
)
from ..dataset_snapshot.service import _canonical_json_bytes, _is_reparse_point
from .models import (
    AuditIssue,
    AuditResult,
    BackfillResult,
    LocalIndexConfig,
    LocalIndexCorruptError,
    LocalIndexError,
    LocalIndexPersistenceError,
    LocalIndexRebuildInProgress,
    RebuildResult,
)
from .repository import LocalIndexRepository
from ..train_analyzer.experiment_history import ExperimentHistoryError, ExperimentHistoryStore

MAX_ISSUES = 100
MAX_BACKFILL_RECORDS = 200
REBUILD_LOCK_TTL_SECONDS = 1800
MAX_FACT_FILE_BYTES = 16 * 1024 * 1024
_ISSUE_DETAIL_LIMIT = 200

# Bounded snapshot manifest scan: only immediate children of the fixed
# ``dataset_snapshots`` root are considered, manifest reads are chunked (never
# trusting stat size), and both the entry count and total read capacity are
# capped so a hostile/misbehaving snapshot tree can never be scanned unbounded.
MAX_SNAPSHOT_SCAN_ENTRIES = 500
MAX_SNAPSHOT_MANIFEST_BYTES = 1024 * 1024  # 1 MiB per manifest
MAX_SNAPSHOT_TOTAL_BYTES = 16 * 1024 * 1024  # 16 MiB across all manifests
_MANIFEST_READ_CHUNK = 1024 * 1024

_MANIFEST_STATUS_VALID = "valid"
_MANIFEST_STATUS_MISSING = "missing"
_MANIFEST_STATUS_CORRUPT = "corrupt"
_MANIFEST_STATUS_TOO_LARGE = "too_large"
_MANIFEST_STATUS_UNREADABLE = "unreadable"


def _new_service(config: LocalIndexConfig):
    """Build a service lazily to avoid an import cycle (service -> reconciliation)."""
    from .service import LocalIndexService
    return LocalIndexService(config)


class _FactFileTooLarge(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.filename = filename


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded(text: object) -> str:
    value = str(text)
    return value if len(value) <= _ISSUE_DETAIL_LIMIT else value[:_ISSUE_DETAIL_LIMIT] + "..."


def rebuild_lock_path(config: LocalIndexConfig) -> Path:
    """Path of the single-instance rebuild lock file."""
    return Path(str(config.database_path) + ".rebuild.lock")


# ── controlled fact readers (bounded) ──


def _fact_experiments(log_dir) -> list[dict]:
    log_dir = Path(log_dir)
    exp_path = log_dir / "experiment_history.json"
    legacy_path = log_dir / "tuning_history.json"
    for path in (exp_path, legacy_path):
        if path.is_file() and path.stat().st_size > MAX_FACT_FILE_BYTES:
            raise _FactFileTooLarge(path.name)
    store = ExperimentHistoryStore(str(exp_path), legacy_tuning_path=str(legacy_path))
    try:
        return store.list_experiments(include_legacy=True)
    except ExperimentHistoryError as exc:
        # A corrupt/oversized fact file is a stable domain error, never a leak
        # of the traceback or the underlying path.
        raise LocalIndexCorruptError("fact history is corrupt") from exc


def _fact_dataset(log_dir) -> dict | None:
    path = Path(log_dir) / "latest_dataset.json"
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_FACT_FILE_BYTES:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _fact_key(fact: dict) -> str:
    run_id = fact.get("run_id")
    if run_id:
        return str(run_id)
    return f"{fact.get('source')}:{fact.get('run_name')}"


def _metric_equal(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    return str(left) == str(right)


def _normalize_reference(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    return os.path.normcase(os.path.normpath(os.path.abspath(raw)))


def _resolve_dataset_id(repo: LocalIndexRepository, data_path: object) -> str | None:
    normalized = _normalize_reference(data_path)
    if normalized is None:
        return None
    dataset = repo.find_dataset_by_data_yaml(normalized)
    return dataset.dataset_id if dataset is not None else None


def _add_issue(issues: list, counts: dict, code: str, subject: str, detail: str) -> None:
    counts[code] = counts.get(code, 0) + 1
    if len(issues) < MAX_ISSUES:
        issues.append(AuditIssue(code, _bounded(subject), _bounded(detail)))


def _build_key_map(repo: LocalIndexRepository) -> dict:
    """Map a fact key to the existing index row (preserving runtime keys).

    The fact key is ``params._legacy_record_run_id`` when the SQLite row was
    indexed under an S1.5 runtime identity, otherwise the row's own run_id.
    """
    key_map: dict = {}
    for row in repo.list_all_experiments():
        params = row.get("params") or {}
        key = params.get("_legacy_record_run_id") or row.get("run_id")
        if key:
            key_map.setdefault(str(key), row)
    return key_map


def _build_current_key_map(config: LocalIndexConfig) -> dict:
    """Best-effort fact-key map from the current database (readable only)."""
    if not Path(config.database_path).is_file():
        return {}
    try:
        conn = connect_database(config)
        try:
            return _build_key_map(LocalIndexRepository(conn))
        finally:
            conn.close()
    except LocalIndexError:
        return {}


def _compare_fact(
    repo: LocalIndexRepository,
    key_map: dict,
    fact: dict,
    issues: list,
    counts: dict,
) -> None:
    key = _fact_key(fact)
    row = key_map.get(key)
    if row is None:
        _add_issue(issues, counts, "MISSING_IN_INDEX", key, "fact record is not indexed")
        return
    if fact.get("status") and row.get("status") != fact["status"]:
        _add_issue(issues, counts, "STATUS_MISMATCH", key,
                   f"index={row.get('status')} fact={fact.get('status')}")
    fact_metrics = fact.get("metrics") or {}
    row_metrics = row.get("metrics") or {}
    for metric_key in sorted(set(fact_metrics) | set(row_metrics)):
        if not _metric_equal(fact_metrics.get(metric_key), row_metrics.get(metric_key)):
            _add_issue(issues, counts, "METRICS_MISMATCH", key, "metrics differ")
            break
    fact_dataset_id = _resolve_dataset_id(repo, (fact.get("params") or {}).get("data"))
    if fact_dataset_id and row.get("dataset_id") != fact_dataset_id:
        _add_issue(issues, counts, "DATASET_MISMATCH", key, "dataset association differs")
    for artifact in repo.raw_artifact_rows(row["run_id"]):
        live = "exists" if os.path.exists(artifact["path"]) else "missing"
        if artifact["exists_state"] and artifact["exists_state"] != live:
            _add_issue(issues, counts, "ARTIFACT_STATE_MISMATCH", key,
                       f"artifact {artifact['kind']} state differs")


# ── audit ──


def audit_index(config: LocalIndexConfig, log_dir="log", max_issues: int = MAX_ISSUES) -> AuditResult:
    """Read-only audit: compare the projection against the controlled facts.

    The audit never initializes, migrates, or creates the database, never opens
    a read-write connection, and never records a maintenance event. A missing
    database reports every fact as MISSING_IN_INDEX (the projection is empty)
    without creating the file. A corrupt or unavailable database, or an
    oversized fact file, returns a stable error code instead of pretending the
    index is healthy. Fact files are never modified.
    """
    timestamp = _utc_now_iso()
    counts: dict = {"scanned": 0}
    issues: list = []

    def _result(error_code: str | None, extra_counts: dict | None = None) -> AuditResult:
        merged = dict(counts)
        if extra_counts:
            merged.update(extra_counts)
        return AuditResult(
            timestamp=timestamp, error_code=error_code, scanned=counts["scanned"],
            counts=merged, issues=tuple(issues[:max_issues]),
        )

    if not Path(config.database_path).is_file():
        # No projection exists: every fact is missing. Read facts read-only and
        # never create or open the database.
        try:
            facts = _fact_experiments(log_dir)
        except _FactFileTooLarge:
            return _result("LOCAL_INDEX_AUDIT_TOO_LARGE", {"AUDIT_FILE_TOO_LARGE": 1})
        except LocalIndexError as exc:
            return _result(exc.error_code, {exc.error_code: 1})
        for fact in facts:
            counts["scanned"] += 1
            _add_issue(issues, counts, "MISSING_IN_INDEX", _fact_key(fact),
                       "fact record is not indexed")
        return _result(None)

    try:
        check_database_integrity(config, readonly=True)
    except LocalIndexError as exc:
        return _result(exc.error_code, {exc.error_code: 1})

    try:
        facts = _fact_experiments(log_dir)
    except _FactFileTooLarge:
        return _result("LOCAL_INDEX_AUDIT_TOO_LARGE", {"AUDIT_FILE_TOO_LARGE": 1})
    except LocalIndexError as exc:
        return _result(exc.error_code, {exc.error_code: 1})

    try:
        conn = connect_database_readonly(config)
    except LocalIndexError as exc:
        return _result(exc.error_code, {exc.error_code: 1})
    try:
        repo = LocalIndexRepository(conn)
        key_map = _build_key_map(repo)
        fact_keys: set = set()
        for fact in facts:
            fact_keys.add(_fact_key(fact))
            counts["scanned"] += 1
            _compare_fact(repo, key_map, fact, issues, counts)
        for key, row in key_map.items():
            if key in fact_keys or row.get("status") in ("starting", "running"):
                continue
            _add_issue(issues, counts, "EXTRA_IN_INDEX", key,
                       "indexed record has no fact backing")
        for row in key_map.values():
            dataset_id = row.get("dataset_id")
            if dataset_id and repo.get_dataset(dataset_id) is None:
                _add_issue(issues, counts, "ORPHAN_DATASET", row["run_id"],
                           "experiment references an unknown dataset")
    except LocalIndexError as exc:
        return _result(exc.error_code)
    finally:
        conn.close()
    return _result(None)


# ── rebuild ──


def _temp_path_for(config: LocalIndexConfig) -> Path:
    return Path(config.database_path).with_name(
        f".{Path(config.database_path).name}.rebuild-{uuid.uuid4().hex}.tmp"
    )


def _cleanup_temp(temp_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(temp_path) + suffix).unlink()
        except OSError:
            pass


def _acquire_lock(lock_path: Path) -> bool:
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} at={_utc_now_iso()}\n".encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - lock_path.stat().st_mtime > REBUILD_LOCK_TTL_SECONDS:
                lock_path.unlink()
                return _acquire_lock(lock_path)
        except OSError:
            pass
        return False
    except OSError:
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


def _record_maintenance(config: LocalIndexConfig, kind: str, summary: dict) -> None:
    """Best-effort bounded maintenance summary; failures are never fatal."""
    if not Path(config.database_path).is_file():
        return
    try:
        conn = connect_database(config)
        try:
            LocalIndexRepository(conn).record_maintenance_event(kind, summary)
        finally:
            conn.close()
    except LocalIndexError:
        pass


def _snapshot_root_normalized(log_dir) -> str:
    """Normalized prefix of the fixed snapshot-manifest root directory."""
    return _normalize_reference(str(Path(log_dir) / "dataset_snapshots"))


def _is_under_snapshot_root(normalized_path: str | None, root_norm: str) -> bool:
    if not normalized_path:
        return False
    return normalized_path.startswith(root_norm + os.sep)


def _manifest_digest_ok(manifest: dict) -> bool:
    """Verify the canonical manifest digest (same rule the snapshot writer uses)."""
    digest = manifest.get("manifest_digest")
    if not isinstance(digest, str) or not digest:
        return False
    payload = {
        key: value for key, value in manifest.items()
        if key not in ("created_at", "source_root", "manifest_digest")
    }
    try:
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() == digest
    except Exception:
        return False


def _snapshot_identity_matches(manifest_payload: dict, snapshot_id, snapshot_digest) -> bool:
    """A latest_dataset snapshot claim is consistent with a validated manifest.

    Path consistency is implied by the ``manifest_map`` lookup (the manifest key
    is the normalized ``data_yaml_path``); any snapshot_id/digest the fact
    declared must also agree with the manifest's verified identity.
    """
    if snapshot_id and manifest_payload.get("snapshot_id") != snapshot_id:
        return False
    if snapshot_digest and manifest_payload.get("snapshot_manifest_digest") != snapshot_digest:
        return False
    return True


def _read_manifest_bounded(manifest_path: Path, remaining_budget: int) -> tuple[bytes | None, str]:
    """Read a manifest with chunked reads (never trusting stat size alone).

    Returns ``(raw_bytes, status)`` where status is VALID, TOO_LARGE (single
    file exceeds the per-file limit or the remaining total budget) or
    UNREADABLE (filesystem error). The stream is actually read so a file that
    grows during the read is still bounded.
    """
    try:
        if manifest_path.stat().st_size > MAX_SNAPSHOT_MANIFEST_BYTES:
            return None, _MANIFEST_STATUS_TOO_LARGE
    except OSError:
        return None, _MANIFEST_STATUS_UNREADABLE
    chunks: list[bytes] = []
    total = 0
    try:
        with open(manifest_path, "rb") as fh:
            while True:
                # Each read is bounded by the smaller of the chunk size, the
                # per-file limit remaining and the total budget remaining; the
                # "+1" bytes are the only allowance for detecting an over-limit
                # file (a file that grew mid-read never exceeds the cap by more
                # than one read).
                read_len = min(
                    _MANIFEST_READ_CHUNK,
                    MAX_SNAPSHOT_MANIFEST_BYTES - total + 1,
                    remaining_budget - total + 1,
                )
                chunk = fh.read(read_len)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_SNAPSHOT_MANIFEST_BYTES or total > remaining_budget:
                    return None, _MANIFEST_STATUS_TOO_LARGE
                chunks.append(chunk)
    except OSError:
        return None, _MANIFEST_STATUS_UNREADABLE
    return b"".join(chunks), _MANIFEST_STATUS_VALID


def scan_snapshot_manifests(log_dir) -> tuple[dict, dict, bool]:
    """Scan snapshot manifests once per rebuild with strict bounds.

    Only immediate children of ``dataset_snapshots`` are considered (fixed
    depth, no recursive scan). Returns ``(manifest_map, status_counts, exceeded)``
    where ``manifest_map`` maps the normalized ``data.yaml`` path to the parsed
    valid payload, ``status_counts`` counts valid/missing/corrupt/too_large/
    unreadable, and ``exceeded`` is True when the entry-count or total-read
    capacity was hit (the scan stops at the bound). No full path ever leaks in
    the counts.
    """
    root = Path(log_dir) / "dataset_snapshots"
    manifest_map: dict = {}
    status_counts: dict = {}
    exceeded = False
    if not root.is_dir():
        return manifest_map, status_counts, exceeded
    if _is_reparse_point(root):
        # The snapshot root itself is a link: refuse to scan anything through it.
        status_counts[_MANIFEST_STATUS_UNREADABLE] = 1
        return manifest_map, status_counts, exceeded
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return manifest_map, status_counts, True
    try:
        root_resolved = root.resolve()
    except OSError:
        return manifest_map, status_counts, True
    scanned = 0
    total_read = 0
    for snap_dir in entries:
        if scanned >= MAX_SNAPSHOT_SCAN_ENTRIES:
            exceeded = True
            break
        scanned += 1
        if _is_reparse_point(snap_dir):
            # symlinks / junctions / reparse points are never followed: reading
            # their manifest could reach content outside the snapshot root.
            status_counts[_MANIFEST_STATUS_UNREADABLE] = status_counts.get(_MANIFEST_STATUS_UNREADABLE, 0) + 1
            continue
        if not snap_dir.is_dir():
            continue
        try:
            resolved = snap_dir.resolve()
        except OSError:
            status_counts[_MANIFEST_STATUS_UNREADABLE] = status_counts.get(_MANIFEST_STATUS_UNREADABLE, 0) + 1
            continue
        if _normalize_reference(str(resolved.parent)) != _normalize_reference(str(root_resolved)):
            # Defense in depth: the resolved snapshot directory must stay an
            # immediate child of the root; content outside it is never read.
            status_counts[_MANIFEST_STATUS_UNREADABLE] = status_counts.get(_MANIFEST_STATUS_UNREADABLE, 0) + 1
            continue
        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.is_file():
            status_counts[_MANIFEST_STATUS_MISSING] = status_counts.get(_MANIFEST_STATUS_MISSING, 0) + 1
            continue
        remaining = MAX_SNAPSHOT_TOTAL_BYTES - total_read
        if remaining <= 0:
            exceeded = True
            break
        raw, status = _read_manifest_bounded(manifest_path, remaining)
        if status != _MANIFEST_STATUS_VALID:
            status_counts[status] = status_counts.get(status, 0) + 1
            # A per-file over-limit/unreadable manifest consumes no budget; the
            # total capacity is enforced by ``remaining`` on the next iteration.
            continue
        total_read += len(raw)
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            status_counts[_MANIFEST_STATUS_CORRUPT] = status_counts.get(_MANIFEST_STATUS_CORRUPT, 0) + 1
            continue
        if not isinstance(manifest, dict) or not _manifest_digest_ok(manifest):
            status_counts[_MANIFEST_STATUS_CORRUPT] = status_counts.get(_MANIFEST_STATUS_CORRUPT, 0) + 1
            continue
        snapshot_dir = manifest_path.parent
        data_yaml = os.path.join(str(snapshot_dir), "data.yaml")
        payload = {
            "source_dataset_path": manifest.get("source_root") or str(snapshot_dir),
            "data_yaml_path": manifest.get("data_yaml_path") or data_yaml,
            "snapshot_id": manifest.get("snapshot_id") or snapshot_dir.name,
            "snapshot_manifest_digest": manifest.get("manifest_digest"),
            "snapshot_created_at": manifest.get("created_at"),
            "snapshot_valid": True,
        }
        norm = _normalize_reference(payload["data_yaml_path"])
        manifest_map[norm] = payload
        status_counts[_MANIFEST_STATUS_VALID] = status_counts.get(_MANIFEST_STATUS_VALID, 0) + 1
    return manifest_map, status_counts, exceeded


def _project_facts(service: LocalIndexService, key_map: dict, log_dir,
                   manifest_map: dict, snap_root_norm: str) -> dict:
    """Re-project controlled facts into ``service`` preserving runtime keys.

    Datasets are restored from the latest_dataset fact AND the bounded snapshot
    manifest scan, then completed with unverified dataset records derived from
    each experiment's own ``params.data`` so a historical association is never
    silently cleared. Experiments that carry no data reference are counted in
    ``dataset_unresolved``. Experiments whose data path lies inside a snapshot
    whose identity cannot be recovered (missing/corrupt/oversized manifest) are
    counted in ``snapshot_unresolved`` and never quietly downgraded to a normal
    unverified dataset.
    """
    counts = {
        "datasets": 0, "experiments": 0, "artifacts": 0,
        "dataset_unresolved": 0, "snapshot_unresolved": 0,
    }
    known: set = set()

    dataset_payload = _fact_dataset(log_dir)
    if dataset_payload is not None:
        dy = _normalize_reference(dataset_payload.get("data_yaml_path"))
        snapshot_id = dataset_payload.get("snapshot_id") or None
        snapshot_digest = (
            dataset_payload.get("snapshot_manifest_digest")
            or dataset_payload.get("manifest_digest")
            or None
        )
        references_snapshot = bool(snapshot_id or snapshot_digest) or (
            dy is not None and _is_under_snapshot_root(dy, snap_root_norm)
        )
        if references_snapshot:
            # A snapshot cannot be vouched for by latest_dataset alone: a valid
            # association requires a matching manifest whose path, snapshot_id
            # and digest are consistent. A broken reference stays out of
            # ``known`` so the experiment loop reports it snapshot_unresolved
            # instead of silently fabricating a dataset for it.
            manifest_payload = manifest_map.get(dy) if dy is not None else None
            if manifest_payload is not None and _snapshot_identity_matches(
                manifest_payload, snapshot_id, snapshot_digest
            ):
                service.index_dataset(manifest_payload)
                counts["datasets"] += 1
                if dy:
                    known.add(dy)
        else:
            service.index_dataset(dataset_payload)
            counts["datasets"] += 1
            if dy:
                known.add(dy)

    for snap_payload in manifest_map.values():
        dy = _normalize_reference(snap_payload.get("data_yaml_path"))
        if dy and dy in known:
            continue
        service.index_dataset(snap_payload)
        counts["datasets"] += 1
        if dy:
            known.add(dy)

    try:
        facts = _fact_experiments(log_dir)
    except _FactFileTooLarge:
        raise LocalIndexCorruptError("fact history file too large") from None

    for fact in facts:
        norm = _normalize_reference((fact.get("params") or {}).get("data"))
        if not norm:
            counts["dataset_unresolved"] += 1
            continue
        if norm in known:
            continue
        if _is_under_snapshot_root(norm, snap_root_norm):
            # Snapshot identity cannot be restored; report honestly instead of
            # silently creating an unverified dataset.
            counts["snapshot_unresolved"] += 1
            continue
        service.index_dataset({
            "source_dataset_path": os.path.dirname(norm),
            "data_yaml_path": norm,
        })
        counts["datasets"] += 1
        known.add(norm)

    for fact in facts:
        row = key_map.get(_fact_key(fact))
        runtime_run_id = row["run_id"] if row is not None else None
        service.index_experiment(fact, runtime_run_id=runtime_run_id)
        counts["experiments"] += 1

    conn = service._open()
    try:
        counts["artifacts"] = LocalIndexRepository(conn).count_artifacts()
    finally:
        conn.close()
    return counts


def rebuild_index(config: LocalIndexConfig, log_dir="log") -> RebuildResult:
    """Atomically rebuild the index: backup -> temp rebuild -> publish.

    A controlled backup is taken first. The new database is fully built in a
    same-directory temp file, passes ``quick_check``, then replaces the original
    via ``os.replace``. Any failure keeps the original database untouched and
    removes the temp file. Rebuild is single-instance (lock file gate).
    """
    lock = rebuild_lock_path(config)
    if not _acquire_lock(lock):
        raise LocalIndexRebuildInProgress("a rebuild is already running")
    timestamp = _utc_now_iso()
    temp_path: Path | None = None
    try:
        backup_created = False
        if Path(config.database_path).is_file():
            create_backup(config)
            backup_created = True
        # The snapshot manifest scan runs once per rebuild; the cached mapping
        # is reused for every experiment (no per-experiment rescan).
        manifest_map, manifest_statuses, scan_exceeded = scan_snapshot_manifests(log_dir)
        snap_root_norm = _snapshot_root_normalized(log_dir)
        temp_path = _temp_path_for(config)
        temp_cfg = replace(config, database_path=temp_path)
        temp_service = _new_service(temp_cfg)
        temp_service.initialize()
        key_map = _build_current_key_map(config)
        counts = _project_facts(temp_service, key_map, log_dir, manifest_map, snap_root_norm)
        check_database_integrity(temp_cfg)
        os.replace(temp_path, config.database_path)
        temp_path = None
        snapshot_issues = dict(manifest_statuses)
        if scan_exceeded:
            snapshot_issues["exceeded"] = True
        # Events are recorded only after the publish succeeds, so a failed
        # rebuild leaves the original database byte-for-byte untouched.
        _record_maintenance(config, "backup", {"status": "ok"})
        _record_maintenance(config, "rebuild", {
            "status": "ok",
            "datasets": counts["datasets"],
            "experiments": counts["experiments"],
            "artifacts": counts["artifacts"],
            "dataset_unresolved": counts.get("dataset_unresolved", 0),
            "snapshot_unresolved": counts.get("snapshot_unresolved", 0),
            "snapshot_issues": snapshot_issues,
            "backup_created": backup_created,
        })
        return RebuildResult(
            timestamp=timestamp, error_code=None, backup_created=backup_created,
            datasets=counts["datasets"], experiments=counts["experiments"],
            artifacts=counts["artifacts"],
            dataset_unresolved=counts.get("dataset_unresolved", 0),
            snapshot_unresolved=counts.get("snapshot_unresolved", 0),
            snapshot_issues=snapshot_issues,
        )
    except LocalIndexError:
        raise
    except OSError as exc:
        raise LocalIndexPersistenceError("rebuild failed") from exc
    finally:
        if temp_path is not None:
            _cleanup_temp(temp_path)
        _release_lock(lock)


# ── startup backfill ──


def backfill_startup(config: LocalIndexConfig, log_dir="log", max_records: int = 50) -> BackfillResult:
    """Catch up bounded recent JSON facts after startup/index outages.

    Only the newest ``max_records`` facts are considered. Already-indexed facts
    are skipped (idempotent); unknown terminal states are indexed as unknown and
    never guessed to completed. A single corrupted/oversized fact never blocks
    the other records from being backfilled.
    """
    timestamp = _utc_now_iso()
    try:
        facts = _fact_experiments(log_dir)
    except _FactFileTooLarge:
        return BackfillResult(timestamp=timestamp, error_code="LOCAL_INDEX_BACKFILL_TOO_LARGE")
    facts = list(facts[:max_records])
    scanned = len(facts)
    indexed = 0
    skipped = 0
    failures = 0
    try:
        service = _new_service(config)
        service.initialize()
        key_map = _build_current_key_map(config)
        for fact in facts:
            key = _fact_key(fact)
            if key in key_map:
                skipped += 1
                continue
            try:
                service.index_experiment(fact, runtime_run_id=key_map.get(key))
                indexed += 1
                key_map[key] = service.get_experiment(key) or {}
            except LocalIndexError:
                failures += 1
    except LocalIndexError as exc:
        return BackfillResult(
            timestamp=timestamp, error_code=exc.error_code, scanned=scanned,
            indexed=indexed, skipped=skipped, failures=failures,
        )
    result = BackfillResult(
        timestamp=timestamp, error_code=None, scanned=scanned,
        indexed=indexed, skipped=skipped, failures=failures,
    )
    # Persist a bounded maintenance summary so the UI/diagnostics can surface
    # the most recent startup backfill outcome; never fatal.
    _record_maintenance(config, "backfill", {
        "status": "ok",
        "error_code": None,
        "scanned": result.scanned,
        "indexed": result.indexed,
        "skipped": result.skipped,
        "failures": result.failures,
    })
    return result
