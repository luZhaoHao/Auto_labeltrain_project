"""Index health diagnostics (Studio S2.3).

Reports schema version, database/WAL sizes, record counts, quick_check, backup
count, the most recent import and maintenance events, and stable error codes.
A corrupt or locked database is reported honestly (``available=False`` with a
stable ``error_code``); diagnostics never echo raw SQL or full local paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from .database import check_database_integrity, connect_database_readonly, read_journal_mode
from .models import LocalIndexConfig, LocalIndexError
from .repository import LocalIndexRepository

_MAX_RECENT_EVENTS = 10


def _collect_error_codes(events: list[dict], current: str | None) -> list[str]:
    codes: list[str] = []
    if current:
        codes.append(current)
    for event in events:
        summary = event.get("summary") or {}
        code = summary.get("error_code")
        if code and code not in codes:
            codes.append(str(code))
    return codes


def build_diagnostics(config: LocalIndexConfig) -> dict:
    """Build the bounded diagnostics report for the local index."""
    result: dict = {
        "available": False,
        "schema_version": None,
        "database_size_bytes": 0,
        "wal_size_bytes": 0,
        "dataset_count": 0,
        "experiment_count": 0,
        "artifact_count": 0,
        "backup_count": 0,
        "quick_check": "unavailable",
        "journal_mode": None,
        "recent_import": None,
        "recent_events": [],
        "recent_error_codes": [],
        "error_code": None,
    }
    db_path = Path(config.database_path)
    try:
        if db_path.is_file():
            result["database_size_bytes"] = db_path.stat().st_size
        wal_path = Path(str(db_path) + "-wal")
        if wal_path.is_file():
            result["wal_size_bytes"] = wal_path.stat().st_size
    except OSError:
        result["error_code"] = "LOCAL_INDEX_UNAVAILABLE"
        return result

    if config.backup_dir.is_dir():
        try:
            result["backup_count"] = len([
                entry for entry in config.backup_dir.iterdir()
                if entry.is_file() and entry.name.startswith(f"{db_path.name}.")
                and entry.name.endswith(".bak")
            ])
        except OSError:
            result["backup_count"] = 0

    # Run quick_check before connecting so a corrupt file is reported as
    # LOCAL_INDEX_CORRUPT (not a generic connect failure). Diagnostics is a
    # read-only GET: the integrity check and connection are both ``mode=ro`` so
    # the database bytes, mtime and WAL/SHM/journal files are never touched.
    if db_path.is_file():
        try:
            check_database_integrity(config, readonly=True)
        except LocalIndexError as exc:
            result["error_code"] = exc.error_code
            result["quick_check"] = "corrupt"
            result["recent_error_codes"] = [exc.error_code]
            return result

    try:
        conn = connect_database_readonly(config)
    except LocalIndexError as exc:
        result["error_code"] = exc.error_code
        result["recent_error_codes"] = _collect_error_codes([], exc.error_code)
        return result
    try:
        repo = LocalIndexRepository(conn)
        result["schema_version"] = repo.schema_version()
        result["dataset_count"] = len(repo.list_datasets())
        result["experiment_count"] = repo.count_experiments()
        result["artifact_count"] = repo.count_artifacts()
        result["recent_import"] = repo.latest_legacy_import()
        result["recent_events"] = repo.recent_maintenance_events(_MAX_RECENT_EVENTS)
        # Journal mode is read from the file header, not ``PRAGMA journal_mode``,
        # so the read-only connection (immutable mode) never creates companion
        # -wal/-shm files and never misreports the mode.
        result["journal_mode"] = read_journal_mode(db_path)
        quick_row = conn.execute("PRAGMA quick_check").fetchone()
        result["quick_check"] = "ok" if quick_row and quick_row[0] == "ok" else "corrupt"
        if result["quick_check"] != "ok":
            result["error_code"] = "LOCAL_INDEX_CORRUPT"
    except LocalIndexError as exc:
        result["error_code"] = exc.error_code
        result["quick_check"] = "unavailable"
        result["recent_events"] = []
        result["recent_import"] = None
    finally:
        conn.close()
    result["available"] = result["error_code"] is None
    result["recent_error_codes"] = _collect_error_codes(
        result["recent_events"], result["error_code"]
    )
    return result
