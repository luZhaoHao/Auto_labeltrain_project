"""Storage layer for the local SQLite stable index (Studio S2 Core).

Owns connection lifecycle, PRAGMAs, the schema-v1 migrations, integrity checks
and bounded backups. The database is a rebuildable query projection only; it is
never used to mutate the JSON/audit fact files or the S1.5 run state.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import os
import sqlite3
from pathlib import Path

from .models import (
    LocalIndexConfig,
    LocalIndexConfigError,
    LocalIndexCorruptError,
    LocalIndexError,
    LocalIndexMigrationError,
    LocalIndexPersistenceError,
)

# Schema v1 fixed by the S2 Core spec. Field names are a compatibility boundary;
# do not rename columns without a forward migration.
_SCHEMA_V1_STATEMENTS = [
    (
        "CREATE TABLE schema_migrations ("
        "  version INTEGER PRIMARY KEY,"
        "  checksum TEXT NOT NULL,"
        "  applied_at TEXT NOT NULL"
        ")"
    ),
    (
        "CREATE TABLE datasets ("
        "  dataset_id TEXT PRIMARY KEY,"
        "  display_name TEXT NOT NULL,"
        "  canonical_path TEXT NOT NULL,"
        "  data_yaml_path TEXT,"
        "  snapshot_id TEXT UNIQUE,"
        "  snapshot_digest TEXT,"
        "  validation_status TEXT NOT NULL,"
        "  created_at TEXT NOT NULL,"
        "  updated_at TEXT NOT NULL,"
        "  last_used_at TEXT"
        ")"
    ),
    "CREATE UNIQUE INDEX datasets_canonical_path_uq ON datasets(canonical_path)",
    (
        "CREATE TABLE experiments ("
        "  run_id TEXT PRIMARY KEY,"
        "  source TEXT NOT NULL,"
        "  run_name TEXT,"
        "  dataset_id TEXT REFERENCES datasets(dataset_id) ON DELETE SET NULL,"
        "  status TEXT NOT NULL,"
        "  phase TEXT,"
        "  model_name TEXT,"
        "  task_type TEXT,"
        "  started_at TEXT,"
        "  finished_at TEXT,"
        "  params_json TEXT NOT NULL,"
        "  metrics_json TEXT NOT NULL,"
        "  analysis_status TEXT,"
        "  error_json TEXT,"
        "  updated_at TEXT NOT NULL"
        ")"
    ),
    (
        "CREATE TABLE artifacts ("
        "  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  run_id TEXT NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,"
        "  kind TEXT NOT NULL,"
        "  path TEXT NOT NULL,"
        "  exists_state TEXT NOT NULL,"
        "  created_at TEXT NOT NULL,"
        "  UNIQUE(run_id, kind, path)"
        ")"
    ),
    (
        "CREATE TABLE legacy_imports ("
        "  import_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  source_path TEXT NOT NULL,"
        "  content_sha256 TEXT NOT NULL,"
        "  result TEXT NOT NULL,"
        "  imported_count INTEGER NOT NULL,"
        "  error_code TEXT,"
        "  imported_at TEXT NOT NULL,"
        "  UNIQUE(source_path, content_sha256)"
        ")"
    ),
]

# Schema v2 (Studio S2.1): bounded maintenance-event log for audit/rebuild/
# backup/checkpoint summaries. The v1 tables are never rewritten; the migration
# only adds this table so the v1 checksum and published statements stay intact.
_SCHEMA_V2_STATEMENTS = [
    (
        "CREATE TABLE maintenance_events ("
        "  event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  kind TEXT NOT NULL,"
        "  summary_json TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")"
    ),
    "CREATE INDEX maintenance_events_created_at ON maintenance_events(created_at)",
]


def _migration_checksum(statements: list[str]) -> str:
    return hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()


MIGRATIONS: list[dict] = [
    {
        "version": 1,
        "checksum": _migration_checksum(_SCHEMA_V1_STATEMENTS),
        "statements": list(_SCHEMA_V1_STATEMENTS),
    },
    {
        "version": 2,
        "checksum": _migration_checksum(_SCHEMA_V2_STATEMENTS),
        "statements": list(_SCHEMA_V2_STATEMENTS),
    },
]


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_path(value: object, base: Path, default: str) -> Path:
    raw = value if isinstance(value, str) and value else default
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path


def load_local_index_config(config: dict, base_dir: Path | None = None) -> LocalIndexConfig:
    """Resolve and validate the ``local_index`` config section.

    Relative paths are resolved against ``base_dir`` (defaults to cwd). Invalid
    bounds raise ``LocalIndexConfigError`` before any file is touched.
    """
    if not isinstance(config, dict):
        raise LocalIndexConfigError("local_index config must be a mapping")
    section = config.get("local_index")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise LocalIndexConfigError("local_index must be a mapping")

    base = Path(base_dir) if base_dir is not None else Path.cwd()
    database_path = _resolve_path(section.get("database_path"), base, "log/auto_tune.db")
    backup_dir = _resolve_path(section.get("backup_dir"), base, "log/db_backups")

    backup_max_files = section.get("backup_max_files", 3)
    if isinstance(backup_max_files, bool) or not isinstance(backup_max_files, int) or not (1 <= backup_max_files <= 10):
        raise LocalIndexConfigError("backup_max_files must be an integer in [1,10]")

    busy_timeout_ms = section.get("busy_timeout_ms", 5000)
    if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or not (100 <= busy_timeout_ms <= 60000):
        raise LocalIndexConfigError("busy_timeout_ms must be an integer in [100,60000]")

    if database_path == backup_dir:
        raise LocalIndexConfigError("database_path and backup_dir must differ")

    return LocalIndexConfig(
        database_path=database_path,
        backup_dir=backup_dir,
        backup_max_files=backup_max_files,
        busy_timeout_ms=busy_timeout_ms,
    )


def _storage_error(message: str) -> LocalIndexPersistenceError:
    """Build a stable persistence error that never carries SQL or a full path."""
    return LocalIndexPersistenceError(message)


def connect_database(config: LocalIndexConfig) -> sqlite3.Connection:
    """Open a connection with the project PRAGMAs applied.

    Only the parent directory is created; an existing corrupt file is never
    truncated (integrity failures surface at ``initialize_database``). Expected
    native ``OSError``/``sqlite3.Error`` failures here (mkdir, connect, PRAGMA)
    are translated to ``LocalIndexPersistenceError`` so callers always receive
    a stable domain error.
    """
    path = Path(config.database_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=config.busy_timeout_ms / 1000.0)
    except LocalIndexError:
        raise
    except OSError as exc:
        raise _storage_error("storage unavailable") from exc
    except sqlite3.Error as exc:
        raise _storage_error("database unavailable") from exc
    try:
        conn.row_factory = sqlite3.Row
        # Autocommit mode: transactions are started explicitly via BEGIN IMMEDIATE.
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(config.busy_timeout_ms)}")
    except sqlite3.Error as exc:
        conn.close()
        raise _storage_error("database unavailable") from exc
    return conn


def connect_database_readonly(config: LocalIndexConfig) -> sqlite3.Connection:
    """Open a read-only connection (``mode=ro``) for audit/diagnostics.

    The URI uses ``immutable=1`` so SQLite never creates or mutates ``-wal``/
    ``-shm`` companion files — a plain ``mode=ro`` open of a WAL-mode database
    would create ``-shm`` even for reads. The database is expected to be
    checkpointed when idle (write connections checkpoint on close), so the
    immutable read is honest for a snapshot-in-time diagnostic. A missing file
    is reported as unavailable, not created.
    """
    path = Path(config.database_path)
    if not path.is_file():
        raise _storage_error("database unavailable")
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True,
            timeout=config.busy_timeout_ms / 1000.0,
        )
    except OSError as exc:
        raise _storage_error("storage unavailable") from exc
    except sqlite3.Error as exc:
        raise _storage_error("database unavailable") from exc
    try:
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        conn.execute(f"PRAGMA busy_timeout={int(config.busy_timeout_ms)}")
    except sqlite3.Error as exc:
        conn.close()
        raise _storage_error("database unavailable") from exc
    return conn


def read_journal_mode(database_path) -> str | None:
    """Read the journal mode from the SQLite file header without opening a
    connection (no WAL/SHM/journal side effects on a read-only diagnostic).

    Header bytes 18-19 carry the file format read/write versions: ``w`` (2)
    means WAL, ``d`` (1) means the legacy journal. ``None`` means the header
    could not be read.
    """
    try:
        with open(database_path, "rb") as fh:
            header = fh.read(20)
    except OSError:
        return None
    if len(header) < 19:
        return None
    if chr(header[18]) == "w":
        return "wal"
    if chr(header[18]) == "d":
        return "delete"
    return None


@contextlib.contextmanager
def transaction(connection: sqlite3.Connection):
    """Run statements inside a single ``BEGIN IMMEDIATE`` transaction.

    Any exception rolls the transaction back; a failure during BEGIN itself
    propagates without rollback (there is nothing to roll back).
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def check_database_integrity(config: LocalIndexConfig, readonly: bool = False) -> None:
    """Run ``PRAGMA quick_check`` on an existing database.

    A corrupt file raises ``LocalIndexCorruptError`` and is never modified.
    A missing file is not created here. With ``readonly=True`` the connection is
    opened via ``connect_database_readonly`` (``mode=ro``) so a GET-style check
    never creates or mutates the database, WAL/SHM/journal files, or bytes.
    """
    path = Path(config.database_path)
    if not path.is_file():
        return
    try:
        conn = connect_database_readonly(config) if readonly else sqlite3.connect(str(path))
    except LocalIndexError:
        raise
    except sqlite3.Error as exc:
        raise _storage_error("database unavailable") from exc
    try:
        row = conn.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            raise LocalIndexCorruptError("database failed quick_check")
    except LocalIndexError:
        raise
    except sqlite3.DatabaseError as exc:
        raise LocalIndexCorruptError("database is corrupt") from exc
    finally:
        conn.close()


def _current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)
    except sqlite3.DatabaseError:
        return 0


def _stored_checksum(conn: sqlite3.Connection, version: int) -> str | None:
    row = conn.execute(
        "SELECT checksum FROM schema_migrations WHERE version=?", (version,)
    ).fetchone()
    return row[0] if row is not None else None


def _backup_name(db_name: str, old_version: int) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{db_name}.v{old_version}.{stamp}.bak"


def _create_backup(config: LocalIndexConfig, conn: sqlite3.Connection, old_version: int) -> Path:
    """Snapshot the current database before a migration (SQLite backup API).

    The backup is written to a temp file then atomically published. A failure
    here must abort the migration so the source is never half-migrated.
    """
    backup_dir = Path(config.backup_dir)
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _storage_error("storage unavailable") from exc
    name = _backup_name(Path(config.database_path).name, old_version)
    tmp = backup_dir / f".{name}.tmp"
    dest = backup_dir / name
    try:
        dest_conn = sqlite3.connect(str(tmp))
        try:
            conn.backup(dest_conn)
        finally:
            dest_conn.close()
        os.replace(tmp, dest)
    except LocalIndexError:
        raise
    except Exception as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        # The exception text may embed the tmp/dest paths; keep the message generic.
        raise _storage_error("backup failed") from exc
    _prune_backups(config)
    return dest


def _prune_backups(config: LocalIndexConfig) -> None:
    """Keep only the newest ``backup_max_files`` backups for this database.

    Only direct children of ``backup_dir`` strictly matching this database's
    backup prefix are considered; nothing is removed recursively. Both migration
    backups (``<db>.v<N>...bak``) and manual backups (``<db>.manual...bak``)
    share the same bounded budget.
    """
    backup_dir = Path(config.backup_dir)
    prefix = f"{Path(config.database_path).name}."
    if not backup_dir.is_dir():
        return
    try:
        candidates = [
            entry
            for entry in backup_dir.iterdir()
            if entry.is_file() and entry.name.startswith(prefix) and entry.name.endswith(".bak")
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        # Pruning is best-effort; a transient FS error never aborts a migration.
        return
    for stale in candidates[config.backup_max_files:]:
        try:
            stale.unlink()
        except OSError:
            pass


def create_backup(config: LocalIndexConfig) -> Path:
    """Snapshot the current database into ``backup_dir`` (manual/rebuild path).

    The backup is written to a same-dir temp file then atomically published via
    ``os.replace`` so an interrupted copy never leaves a half-written backup. A
    failure raises ``LocalIndexPersistenceError`` and leaves the source database
    untouched. Pruning keeps only the newest ``backup_max_files`` backups.
    """
    path = Path(config.database_path)
    if not path.is_file():
        raise _storage_error("backup failed")
    backup_dir = Path(config.backup_dir)
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _storage_error("storage unavailable") from exc
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{path.name}.manual.{stamp}.bak"
    tmp = backup_dir / f".{name}.tmp"
    dest = backup_dir / name
    try:
        src_conn = sqlite3.connect(str(path))
        try:
            dest_conn = sqlite3.connect(str(tmp))
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        finally:
            src_conn.close()
        os.replace(tmp, dest)
    except LocalIndexError:
        raise
    except Exception as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise _storage_error("backup failed") from exc
    _prune_backups(config)
    return dest


def initialize_database(config: LocalIndexConfig) -> int:
    """Migrate the database to the latest schema version and return it.

    Idempotent: a fresh database, a re-initialized database, and an upgrade all
    succeed with the same final version. Migration statements run inside a
    single transaction per version; a failure rolls back and raises
    ``LocalIndexMigrationError``. A corrupt existing file raises
    ``LocalIndexCorruptError`` without touching the original bytes.
    """
    path = Path(config.database_path)
    check_database_integrity(config)

    conn = connect_database(config)
    try:
        current = _current_version(conn)
        latest = max((m["version"] for m in MIGRATIONS), default=0)

        for migration in MIGRATIONS:
            if migration["version"] <= current:
                stored = _stored_checksum(conn, migration["version"])
                if stored is not None and stored != migration["checksum"]:
                    raise LocalIndexMigrationError(
                        f"checksum mismatch for applied migration v{migration['version']}"
                    )

        if current > 0 and current < latest:
            _create_backup(config, conn, current)

        for migration in sorted(MIGRATIONS, key=lambda m: m["version"]):
            if migration["version"] <= current:
                continue
            try:
                with transaction(conn):
                    for statement in migration["statements"]:
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, checksum, applied_at) VALUES (?,?,?)",
                        (migration["version"], migration["checksum"], _utc_now_iso()),
                    )
            except LocalIndexError:
                raise
            except Exception as exc:
                raise LocalIndexMigrationError(
                    f"migration v{migration['version']} failed and was rolled back"
                ) from exc
        return latest
    except LocalIndexError:
        raise
    except OSError as exc:
        raise _storage_error("storage unavailable") from exc
    except sqlite3.Error as exc:
        raise _storage_error("database unavailable") from exc
    finally:
        conn.close()
