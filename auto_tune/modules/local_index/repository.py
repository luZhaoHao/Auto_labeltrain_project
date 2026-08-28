"""Parameterized CRUD/query repository for the local index (Studio S2 Core).

Business code never builds raw SQL outside this layer. All statements are
parameterized; constraint/lock failures are translated to stable
``LocalIndexPersistenceError`` values that never carry raw SQL or full paths.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from dataclasses import asdict
from typing import Any

from .database import transaction
from .models import (
    ArtifactRecord,
    DatasetRecord,
    ExperimentQuery,
    ExperimentRecord,
    LocalIndexCorruptError,
    LocalIndexError,
    LocalIndexPersistenceError,
)

_SQLITE_ERROR_CODES = (
    sqlite3.OperationalError,
    sqlite3.ProgrammingError,
    sqlite3.DatabaseError,
)

# Backend whitelist: sort field -> SQL expression. Never accepts client SQL.
_SORT_COLUMNS: dict[str, str] = {
    "finished_at": "COALESCE(finished_at, started_at, updated_at)",
    "started_at": "COALESCE(started_at, updated_at)",
    "updated_at": "updated_at",
    "name": "run_name",
    "mAP50": "json_extract(metrics_json, '$.mAP50')",
    "mAP50_95": "json_extract(metrics_json, '$.mAP50_95')",
    "precision": "json_extract(metrics_json, '$.precision')",
    "recall": "json_extract(metrics_json, '$.recall')",
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _translate(fn):
    """Execute ``fn`` and map sqlite failures to stable domain errors."""
    try:
        return fn()
    except LocalIndexCorruptError:
        raise
    except sqlite3.IntegrityError as exc:
        raise LocalIndexPersistenceError("database constraint violation") from exc
    except _SQLITE_ERROR_CODES as exc:
        raise LocalIndexPersistenceError("database unavailable") from exc


def _decode_json(raw: str | None, what: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise LocalIndexCorruptError(f"invalid {what} stored in database") from exc


class LocalIndexRepository:
    """A single connection-bound repository for datasets/experiments/artifacts.

    Write methods start their own transaction unless one is already active on
    the connection, so multi-statement service writes stay atomic.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def _write(self, fn):
        if self._conn.in_transaction:
            return _translate(fn)
        try:
            with transaction(self._conn):
                return _translate(fn)
        except LocalIndexError:
            raise
        except _SQLITE_ERROR_CODES as exc:
            raise LocalIndexPersistenceError("database unavailable") from exc

    # ── datasets ──

    def upsert_dataset(self, record: DatasetRecord) -> DatasetRecord:
        now = record.updated_at or _utc_now_iso()
        values = (
            record.dataset_id,
            record.display_name,
            record.canonical_path,
            record.data_yaml_path,
            record.snapshot_id,
            record.snapshot_digest,
            record.validation_status,
            record.created_at or now,
            now,
            record.last_used_at,
        )

        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO datasets(
                    dataset_id, display_name, canonical_path, data_yaml_path,
                    snapshot_id, snapshot_digest, validation_status,
                    created_at, updated_at, last_used_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    canonical_path=excluded.canonical_path,
                    data_yaml_path=excluded.data_yaml_path,
                    snapshot_id=excluded.snapshot_id,
                    snapshot_digest=excluded.snapshot_digest,
                    validation_status=excluded.validation_status,
                    updated_at=excluded.updated_at,
                    last_used_at=excluded.last_used_at
                """,
                values,
            )

        self._write(_do)
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        row = _translate(
            lambda: self._conn.execute(
                "SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        )
        return _dataset_from_row(row) if row is not None else None

    def find_dataset_by_data_yaml(self, path: str) -> DatasetRecord | None:
        row = _translate(
            lambda: self._conn.execute(
                "SELECT * FROM datasets WHERE data_yaml_path=?", (path,)
            ).fetchone()
        )
        return _dataset_from_row(row) if row is not None else None

    def list_datasets(self) -> list[DatasetRecord]:
        rows = _translate(
            lambda: self._conn.execute(
                "SELECT * FROM datasets"
                " ORDER BY COALESCE(last_used_at, updated_at) DESC"
            ).fetchall()
        )
        return [_dataset_from_row(r) for r in rows]

    # ── experiments / artifacts ──

    def upsert_experiment(
        self, record: ExperimentRecord, artifacts: tuple[ArtifactRecord, ...] = ()
    ) -> ExperimentRecord:
        now = record.updated_at or _utc_now_iso()
        params_json = json.dumps(record.params, ensure_ascii=False, default=str)
        metrics_json = json.dumps(record.metrics, ensure_ascii=False, default=str)
        error_json = json.dumps(record.error, ensure_ascii=False, default=str) if record.error is not None else "null"

        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO experiments(
                    run_id, source, run_name, dataset_id, status, phase,
                    model_name, task_type, started_at, finished_at,
                    params_json, metrics_json, analysis_status, error_json, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    source=excluded.source,
                    run_name=excluded.run_name,
                    dataset_id=excluded.dataset_id,
                    status=excluded.status,
                    phase=excluded.phase,
                    model_name=excluded.model_name,
                    task_type=excluded.task_type,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    params_json=excluded.params_json,
                    metrics_json=excluded.metrics_json,
                    analysis_status=excluded.analysis_status,
                    error_json=excluded.error_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.run_id, record.source, record.run_name, record.dataset_id,
                    record.status, record.phase, record.model_name, record.task_type,
                    record.started_at, record.finished_at, params_json, metrics_json,
                    record.analysis_status, error_json, now,
                ),
            )
            for artifact in artifacts:
                exists_state = "exists" if os.path.exists(artifact.path) else "missing"
                self._conn.execute(
                    """
                    INSERT INTO artifacts(run_id, kind, path, exists_state, created_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(run_id, kind, path) DO UPDATE SET
                        exists_state=excluded.exists_state,
                        created_at=excluded.created_at
                    """,
                    (
                        artifact.run_id, artifact.kind, artifact.path,
                        exists_state, artifact.created_at or _utc_now_iso(),
                    ),
                )

        self._write(_do)
        return record

    def get_experiment(self, run_id: str) -> dict | None:
        row = _translate(
            lambda: self._conn.execute(
                "SELECT * FROM experiments WHERE run_id=?", (run_id,)
            ).fetchone()
        )
        if row is None:
            return None
        return _experiment_to_dict(self._conn, row)

    def _build_where(self, query: ExperimentQuery) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if query.dataset_id is not None:
            where.append("dataset_id=?")
            params.append(query.dataset_id)
        if query.source is not None:
            where.append("source=?")
            params.append(query.source)
        if query.status is not None:
            where.append("status=?")
            params.append(query.status)
        if query.search:
            escaped = (
                query.search
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{escaped}%"
            where.append("(run_name LIKE ? ESCAPE '\\' OR model_name LIKE ? ESCAPE '\\')")
            params.extend([pattern, pattern])
        return where, params

    def _sort_sql(self, query: ExperimentQuery) -> str:
        column = _SORT_COLUMNS[query.sort]
        direction = "ASC" if query.order == "asc" else "DESC"
        # run_id tiebreak keeps pagination deterministic.
        return f"{column} {direction}, run_id {direction}"

    def list_experiments(self, query: ExperimentQuery) -> list[dict]:
        where, params = self._build_where(query)
        sql = "SELECT * FROM experiments"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {self._sort_sql(query)} LIMIT ? OFFSET ?"
        params.extend([int(query.limit), int(query.offset)])
        rows = _translate(lambda: self._conn.execute(sql, params).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def count_experiments_matching(self, query: ExperimentQuery) -> int:
        where, params = self._build_where(query)
        sql = "SELECT COUNT(*) FROM experiments"
        if where:
            sql += " WHERE " + " AND ".join(where)
        row = _translate(lambda: self._conn.execute(sql, params).fetchone())
        return int(row[0] or 0) if row is not None else 0

    def list_experiments_by_dataset(self, dataset_id: str, limit: int = 10) -> list[dict]:
        bounded = max(1, min(int(limit), 100))
        rows = _translate(lambda: self._conn.execute(
            "SELECT * FROM experiments WHERE dataset_id=?"
            " ORDER BY COALESCE(finished_at, started_at, updated_at) DESC, run_id DESC LIMIT ?",
            (dataset_id, bounded),
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def count_experiments_by_dataset(self, dataset_id: str) -> int:
        row = _translate(lambda: self._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE dataset_id=?", (dataset_id,)
        ).fetchone())
        return int(row[0] or 0) if row is not None else 0

    def best_experiments_by_dataset(self, dataset_id: str) -> list[dict]:
        """Completed experiments with an mAP50 metric for one dataset.

        Scoped by ``dataset_id`` (indexed) in SQL so the best-experiment fact is
        never computed from a full projection scan.
        """
        rows = _translate(lambda: self._conn.execute(
            "SELECT * FROM experiments WHERE dataset_id=? AND status='completed'"
            " AND json_extract(metrics_json, '$.mAP50') IS NOT NULL"
            " ORDER BY COALESCE(finished_at, started_at, updated_at) DESC, run_id DESC",
            (dataset_id,),
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def list_all_experiments(self) -> list[dict]:
        """Full-scan experiment projection used only by reconciliation.

        Reconciliation (audit/rebuild) inherently needs the complete projection
        to compare against the fact files; this is never used for pagination.
        """
        rows = _translate(lambda: self._conn.execute(
            "SELECT * FROM experiments"
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def list_experiments_by_run_name(self, run_name: str) -> list[dict]:
        """Return every experiment whose run_name matches exactly (parameterized).

        Used by the reference-dataset resolver to detect unique vs. ambiguous
        dataset associations for one run name. Empty associations are surfaced
        honestly (``dataset_id`` stays ``None``), never fabricated.
        """
        rows = _translate(lambda: self._conn.execute(
            "SELECT * FROM experiments WHERE run_name=?", (run_name,)
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def list_experiments_by_ids(self, run_ids: list[str]) -> list[dict]:
        """Fetch experiments by explicit run_id list (bounded, parameterized).

        Used by detail/compare so the caller never supplies raw SQL fragments.
        """
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        rows = _translate(lambda: self._conn.execute(
            f"SELECT * FROM experiments WHERE run_id IN ({placeholders})",
            list(run_ids),
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def list_recent_completed_detect(self, limit: int = 20) -> list[dict]:
        """Bounded candidates for the recent-training shortcut (Bugfix P3).

        Only completed ``detect`` experiments carrying a ``run_dir`` artifact are
        candidates. Ordering is ``finished_at DESC, run_id DESC`` so the tiebreak
        is stable across calls; the scan is bounded by ``limit`` and never walks
        the whole experiment table.
        """
        bounded = max(1, min(int(limit), 100))
        rows = _translate(lambda: self._conn.execute(
            "SELECT e.* FROM experiments e"
            " WHERE e.status='completed' AND e.task_type='detect'"
            " AND EXISTS (SELECT 1 FROM artifacts a"
            "             WHERE a.run_id=e.run_id AND a.kind='run_dir')"
            " ORDER BY COALESCE(e.finished_at, e.started_at, e.updated_at) DESC, e.run_id DESC"
            " LIMIT ?",
            (bounded,),
        ).fetchall())
        return [_experiment_to_dict(self._conn, r) for r in rows]

    def raw_artifact_rows(self, run_id: str) -> list[dict]:
        """Return artifact rows with the *stored* ``exists_state`` (no live probe).

        The audit compares the stored state against the live filesystem so stale
        rows are reported instead of silently corrected on read.
        """
        rows = _translate(lambda: self._conn.execute(
            "SELECT kind, path, exists_state, created_at FROM artifacts"
            " WHERE run_id=? ORDER BY kind, path",
            (run_id,),
        ).fetchall())
        return [{
            "kind": r["kind"],
            "path": r["path"],
            "exists_state": r["exists_state"],
            "created_at": r["created_at"],
        } for r in rows]

    # ── status helpers ──

    def schema_version(self) -> int:
        row = _translate(
            lambda: self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        )
        return int(row[0] or 0) if row is not None else 0

    def count_experiments(self) -> int:
        row = _translate(
            lambda: self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        )
        return int(row[0] or 0) if row is not None else 0

    def count_artifacts(self) -> int:
        row = _translate(
            lambda: self._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
        )
        return int(row[0] or 0) if row is not None else 0

    # ── maintenance events (Schema v2; reconciliation/backup/checkpoint) ──

    def record_maintenance_event(self, kind: str, summary: dict) -> None:
        payload = json.dumps(summary, ensure_ascii=False, default=str)
        now = _utc_now_iso()

        def _do() -> None:
            self._conn.execute(
                "INSERT INTO maintenance_events(kind, summary_json, created_at)"
                " VALUES (?,?,?)",
                (kind, payload, now),
            )

        self._write(_do)

    def recent_maintenance_events(self, limit: int) -> list[dict]:
        bounded = max(1, min(int(limit), 50))
        rows = _translate(lambda: self._conn.execute(
            "SELECT kind, summary_json, created_at FROM maintenance_events"
            " ORDER BY created_at DESC, event_id DESC LIMIT ?",
            (bounded,),
        ).fetchall())
        result = []
        for row in rows:
            try:
                summary = json.loads(row["summary_json"])
            except (ValueError, TypeError):
                summary = {}
            result.append({
                "kind": row["kind"],
                "summary": summary,
                "created_at": row["created_at"],
            })
        return result

    def latest_legacy_import_at(self) -> str | None:
        row = _translate(
            lambda: self._conn.execute(
                "SELECT MAX(imported_at) FROM legacy_imports"
            ).fetchone()
        )
        return row[0] if row is not None else None

    def latest_legacy_import(self) -> dict | None:
        row = _translate(lambda: self._conn.execute(
            "SELECT source_path, result, imported_count, error_code, imported_at"
            " FROM legacy_imports ORDER BY imported_at DESC, import_id DESC LIMIT 1"
        ).fetchone())
        if row is None:
            return None
        return {
            "source_path": row["source_path"],
            "result": row["result"],
            "imported_count": row["imported_count"],
            "error_code": row["error_code"],
            "imported_at": row["imported_at"],
        }

    # ── legacy imports ──

    def has_legacy_import(self, source_path: str, content_sha256: str) -> bool:
        row = _translate(
            lambda: self._conn.execute(
                "SELECT 1 FROM legacy_imports WHERE source_path=? AND content_sha256=? LIMIT 1",
                (source_path, content_sha256),
            ).fetchone()
        )
        return row is not None

    def record_legacy_import(
        self,
        source_path: str,
        content_sha256: str,
        result: str,
        imported_count: int,
        error_code: str | None = None,
    ) -> None:
        def _do() -> None:
            self._conn.execute(
                """
                INSERT INTO legacy_imports(
                    source_path, content_sha256, result, imported_count, error_code, imported_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(source_path, content_sha256) DO UPDATE SET
                    result=excluded.result,
                    imported_count=excluded.imported_count,
                    error_code=excluded.error_code,
                    imported_at=excluded.imported_at
                """,
                (source_path, content_sha256, result, imported_count, error_code, _utc_now_iso()),
            )

        self._write(_do)


def _dataset_from_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=row["dataset_id"],
        display_name=row["display_name"],
        canonical_path=row["canonical_path"],
        data_yaml_path=row["data_yaml_path"],
        snapshot_id=row["snapshot_id"],
        snapshot_digest=row["snapshot_digest"],
        validation_status=row["validation_status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_used_at=row["last_used_at"],
    )


def _artifact_rows(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = _translate(
        lambda: conn.execute(
            "SELECT kind, path, exists_state, created_at FROM artifacts"
            " WHERE run_id=? ORDER BY kind, path",
            (run_id,),
        ).fetchall()
    )
    return [{
        "kind": r["kind"],
        "path": r["path"],
        "exists_state": _live_exists_state(r["path"], r["exists_state"]),
        "created_at": r["created_at"],
    } for r in rows]


def _live_exists_state(path: str, stored: str) -> str:
    return "exists" if os.path.exists(path) else "missing"


def _experiment_to_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    return {
        "run_id": row["run_id"],
        "source": row["source"],
        "run_name": row["run_name"],
        "dataset_id": row["dataset_id"],
        "status": row["status"],
        "phase": row["phase"],
        "model_name": row["model_name"],
        "task_type": row["task_type"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "params": _decode_json(row["params_json"], "params"),
        "metrics": _decode_json(row["metrics_json"], "metrics"),
        "analysis_status": row["analysis_status"],
        "error": _decode_json(row["error_json"], "error"),
        "updated_at": row["updated_at"],
        "artifacts": _artifact_rows(conn, row["run_id"]),
    }
