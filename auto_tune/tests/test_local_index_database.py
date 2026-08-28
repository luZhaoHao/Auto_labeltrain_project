"""Task 1: SQLite storage, Schema v1, migrations and fault protection (Studio S2 Core)."""

import os
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from auto_tune.modules.local_index.database import (
    MIGRATIONS,
    connect_database,
    initialize_database,
    load_local_index_config,
    transaction,
)
from auto_tune.modules.local_index.models import (
    LocalIndexConfig,
    LocalIndexConfigError,
    LocalIndexCorruptError,
    LocalIndexMigrationError,
    LocalIndexPersistenceError,
)


def _cfg(tmp_path, **kw):
    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 5000),
    )


# ── Step 1: config resolution and domain errors ──


def test_load_local_index_config_resolves_paths(tmp_path):
    cfg = load_local_index_config({"local_index": {
        "database_path": "log/auto_tune.db",
        "backup_dir": "log/db_backups",
        "backup_max_files": 3,
        "busy_timeout_ms": 5000,
    }}, base_dir=tmp_path)
    assert cfg.database_path == tmp_path / "log" / "auto_tune.db"
    assert cfg.backup_dir == tmp_path / "log" / "db_backups"
    assert cfg.backup_max_files == 3
    assert cfg.busy_timeout_ms == 5000


def test_load_local_index_config_defaults(tmp_path):
    cfg = load_local_index_config({}, base_dir=tmp_path)
    assert cfg.database_path == tmp_path / "log" / "auto_tune.db"
    assert cfg.backup_dir == tmp_path / "log" / "db_backups"
    assert cfg.backup_max_files == 3
    assert cfg.busy_timeout_ms == 5000


@pytest.mark.parametrize("bad", [0, 11, -1, True, "3", 3.0])
def test_load_local_index_config_rejects_bad_backup_max_files(tmp_path, bad):
    with pytest.raises(LocalIndexConfigError):
        load_local_index_config({"local_index": {"backup_max_files": bad}}, base_dir=tmp_path)


@pytest.mark.parametrize("bad", [0, 99, 60001, True, "5000", 5000.0])
def test_load_local_index_config_rejects_bad_busy_timeout(tmp_path, bad):
    with pytest.raises(LocalIndexConfigError):
        load_local_index_config({"local_index": {"busy_timeout_ms": bad}}, base_dir=tmp_path)


def test_load_local_index_config_rejects_same_db_and_backup_dir(tmp_path):
    with pytest.raises(LocalIndexConfigError):
        load_local_index_config(
            {"local_index": {
                "database_path": "log/auto_tune.db",
                "backup_dir": "log/auto_tune.db",
            }}, base_dir=tmp_path
        )


def test_local_index_config_is_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FrozenInstanceError):
        cfg.busy_timeout_ms = 1


# ── Step 4: schema, PRAGMAs and idempotent initialization ──


def test_connect_sets_pragmas(tmp_path):
    cfg = _cfg(tmp_path, busy_timeout_ms=5000)
    conn = connect_database(cfg)
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_initialize_creates_schema_v2(tmp_path):
    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2
    conn = connect_database(cfg)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"schema_migrations", "datasets", "experiments", "artifacts",
                "legacy_imports", "maintenance_events"} <= names
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        assert row[0] == 2
    finally:
        conn.close()


def test_initialize_idempotent_repeat(tmp_path):
    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2
    assert initialize_database(cfg) == 2
    conn = connect_database(cfg)
    try:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2
    finally:
        conn.close()


def test_foreign_keys_enforced(tmp_path):
    cfg = _cfg(tmp_path)
    initialize_database(cfg)
    conn = connect_database(cfg)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                conn.execute(
                    "INSERT INTO experiments(run_id, source, params_json, metrics_json, updated_at)"
                    " VALUES ('t:1', 'manual', '{}', '{}', '2026-01-01T00:00:00Z')"
                )
    finally:
        conn.close()


# ── Step 6: corrupt db, migration rollback, bounded backups ──


def test_corrupt_database_rejected_bytes_unchanged(tmp_path):
    db = tmp_path / "auto_tune.db"
    payload = b"\x00\x01\x02not-a-sqlite-header-" * 8
    db.write_bytes(payload)
    cfg = LocalIndexConfig(
        database_path=db, backup_dir=tmp_path / "bk",
        backup_max_files=3, busy_timeout_ms=200,
    )
    with pytest.raises(LocalIndexCorruptError):
        initialize_database(cfg)
    assert db.read_bytes() == payload
    assert not (tmp_path / "bk").exists()


def test_failed_migration_rolls_back_to_latest(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2

    import auto_tune.modules.local_index.database as dbmod

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS) + [{
        "version": 3,
        "checksum": "sha-of-v3",
        "statements": [
            "CREATE TABLE bogus_v3 (id INTEGER PRIMARY KEY)",
            "THIS IS NOT VALID SQL",
        ],
    }])
    with pytest.raises(LocalIndexMigrationError):
        initialize_database(cfg)

    conn = connect_database(cfg)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "bogus_v3" not in tables
        assert "datasets" in tables
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2
    finally:
        conn.close()


def test_migration_creates_backup_before_migrate(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2

    import auto_tune.modules.local_index.database as dbmod

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS) + [{
        "version": 3,
        "checksum": "sha-of-v3",
        "statements": ["CREATE TABLE v3_only (id INTEGER PRIMARY KEY)"],
    }])
    assert initialize_database(cfg) == 3
    backups = list(cfg.backup_dir.glob("auto_tune.db.v2.*.bak"))
    assert len(backups) == 1


def test_backup_prunes_to_max_files(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backup_max_files=2)
    assert initialize_database(cfg) == 2

    backup_dir = cfg.backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.getmtime(cfg.database_path)
    for i in range(3):
        p = backup_dir / f"auto_tune.db.v1.2026082{i}T000000Z.bak"
        p.write_bytes(b"old-backup")
        os.utime(p, (base, base - i))

    import auto_tune.modules.local_index.database as dbmod

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS) + [{
        "version": 3,
        "checksum": "sha-of-v3",
        "statements": ["CREATE TABLE v3_only (id INTEGER PRIMARY KEY)"],
    }])
    assert initialize_database(cfg) == 3
    backups = list(cfg.backup_dir.glob("auto_tune.db.*.bak"))
    assert len(backups) == 2


def test_backup_failure_skips_migration(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2

    import auto_tune.modules.local_index.database as dbmod

    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS) + [{
        "version": 3,
        "checksum": "sha-of-v3",
        "statements": ["CREATE TABLE v3_only (id INTEGER PRIMARY KEY)"],
    }])

    def boom(config, conn, old_version):
        raise LocalIndexPersistenceError("backup failed")

    monkeypatch.setattr(dbmod, "_create_backup", boom)
    with pytest.raises(LocalIndexPersistenceError):
        initialize_database(cfg)

    conn = connect_database(cfg)
    try:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "v3_only" not in tables
    finally:
        conn.close()


# ── S2 Core 返修：原生 OSError / sqlite3.Error 在存储边界转为稳定领域错误 ──


def test_connect_wraps_mkdir_permission_error(tmp_path, monkeypatch):
    def _deny(self, *args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "mkdir", _deny)
    with pytest.raises(LocalIndexPersistenceError) as excinfo:
        connect_database(_cfg(tmp_path))
    assert excinfo.value.error_code == "LOCAL_INDEX_UNAVAILABLE"
    assert str(tmp_path) not in str(excinfo.value)


def test_connect_wraps_sqlite_connect_operational_error(tmp_path, monkeypatch):
    import auto_tune.modules.local_index.database as dbmod

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(dbmod.sqlite3, "connect", _boom)
    with pytest.raises(LocalIndexPersistenceError) as excinfo:
        connect_database(_cfg(tmp_path))
    assert excinfo.value.error_code == "LOCAL_INDEX_UNAVAILABLE"


def test_connect_wraps_pragma_operational_error(tmp_path, monkeypatch):
    import auto_tune.modules.local_index.database as dbmod

    real_connect = sqlite3.connect

    class _PragmaFailingConnection:
        def __init__(self, real):
            self._real = real

        def close(self):
            self._real.close()

        def execute(self, sql, *args):
            if isinstance(sql, str) and sql.startswith("PRAGMA"):
                raise sqlite3.OperationalError("attempt to write a readonly database")
            return self._real.execute(sql, *args)

    def _fake_connect(*args, **kwargs):
        return _PragmaFailingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(dbmod.sqlite3, "connect", _fake_connect)
    with pytest.raises(LocalIndexPersistenceError) as excinfo:
        connect_database(_cfg(tmp_path))
    assert excinfo.value.error_code == "LOCAL_INDEX_UNAVAILABLE"


def test_initialize_wraps_backup_dir_mkdir_error(tmp_path, monkeypatch):
    import auto_tune.modules.local_index.database as dbmod

    cfg = _cfg(tmp_path)
    assert initialize_database(cfg) == 2
    monkeypatch.setattr(dbmod, "MIGRATIONS", list(MIGRATIONS) + [{
        "version": 3,
        "checksum": "sha-of-v3",
        "statements": ["CREATE TABLE v3_only (id INTEGER PRIMARY KEY)"],
    }])

    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a dir")
    blocked_cfg = LocalIndexConfig(
        database_path=cfg.database_path,
        backup_dir=blocker / "db_backups",
        backup_max_files=3,
        busy_timeout_ms=5000,
    )
    with pytest.raises(LocalIndexPersistenceError) as excinfo:
        initialize_database(blocked_cfg)
    assert excinfo.value.error_code == "LOCAL_INDEX_UNAVAILABLE"
    # The source database is untouched: still v2, no partial v3 migration.
    conn = connect_database(cfg)
    try:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "v3_only" not in tables
    finally:
        conn.close()


def test_artifact_query_translates_closed_connection_error(tmp_path):
    """The repository artifact query must also convert native sqlite errors to a
    stable domain error so a locked/unavailable DB never escapes as raw sqlite3.Error."""
    from auto_tune.modules.local_index.repository import _artifact_rows

    cfg = _cfg(tmp_path)
    initialize_database(cfg)
    conn = connect_database(cfg)
    try:
        conn.execute(
            "INSERT INTO experiments(run_id, source, status, params_json, metrics_json, updated_at)"
            " VALUES ('manual:a', 'manual', 'completed', '{}', '{}', '2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO artifacts(run_id, kind, path, exists_state, created_at)"
            " VALUES ('manual:a', 'report', '/tmp/x.json', 'exists', '2026-01-01T00:00:00Z')"
        )
    finally:
        conn.close()
    with pytest.raises(LocalIndexPersistenceError):
        _artifact_rows(conn, "manual:a")


def test_initialize_preserves_local_index_error_unwrapped(tmp_path, monkeypatch):
    """A domain error already raised inside the boundary is never re-wrapped."""
    import auto_tune.modules.local_index.database as dbmod

    cfg = _cfg(tmp_path)

    def _boom_config(_cfg):
        raise LocalIndexCorruptError("already corrupt")

    monkeypatch.setattr(dbmod, "check_database_integrity", _boom_config)
    with pytest.raises(LocalIndexCorruptError):
        initialize_database(cfg)
