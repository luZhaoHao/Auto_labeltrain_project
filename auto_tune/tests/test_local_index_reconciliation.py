"""Studio S2.1: reconciliation (read-only audit / atomic rebuild / bounded startup backfill).

SQLite remains a rebuildable projection. These tests prove the reconciliation
machinery is read-only on the fact files, atomic on publish, bounded on issue
summaries and backfill scope, and honest about unknown terminal states.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from auto_tune.modules.dataset_snapshot.service import _canonical_json_bytes
from auto_tune.modules.local_index.database import connect_database, initialize_database
from auto_tune.modules.local_index.models import (
    ArtifactRecord,
    ExperimentQuery,
    ExperimentRecord,
    LocalIndexConfig,
    LocalIndexCorruptError,
    LocalIndexPersistenceError,
    LocalIndexRebuildInProgress,
)
from auto_tune.modules.local_index.reconciliation import (
    MAX_ISSUES,
    audit_index,
    backfill_startup,
    rebuild_index,
    rebuild_lock_path,
)
from auto_tune.modules.local_index.repository import LocalIndexRepository
from auto_tune.modules.local_index.service import LocalIndexService
from auto_tune.modules.train_analyzer.experiment_history import ExperimentHistoryStore


def _cfg(tmp_path, **kw):
    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 200),
    )


def _service(cfg):
    svc = LocalIndexService(cfg)
    svc.initialize()
    return svc


def _write_latest_dataset(log_dir, ds_path, snapshot_id="snap1", data_yaml=None,
                          snapshot_digest=None, snapshot_valid=True):
    log_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = data_yaml or os.path.join(str(ds_path), "data.yaml")
    payload = {
        "dataset_path": str(ds_path),
        "data_yaml_path": data_yaml,
        "snapshot_id": snapshot_id,
        "snapshot_valid": snapshot_valid,
    }
    if snapshot_digest is not None:
        payload["snapshot_manifest_digest"] = snapshot_digest
    (log_dir / "latest_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    return data_yaml


def _write_history(log_dir, experiments, legacy=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "experiment_history.json").write_text(
        json.dumps({"schema_version": "1.0", "experiments": experiments}, ensure_ascii=False),
        encoding="utf-8",
    )
    if legacy is not None:
        (log_dir / "tuning_history.json").write_text(json.dumps(legacy), encoding="utf-8")


def _exp(run_id, run_name=None, source="manual", status="completed", metrics=None,
         data_yaml=None, params=None, analysis_status="completed",
         finished_at="2026-08-01T00:00:00Z"):
    rec = {
        "run_id": run_id,
        "run_name": run_name or run_id.rsplit(":", 1)[-1],
        "source": source,
        "status": status,
        "analysis_status": analysis_status,
        "metrics": dict(metrics or {}),
        "params": dict(params or {}),
        "finished_at": finished_at,
    }
    if data_yaml:
        rec["params"]["data"] = data_yaml
    return rec


def _seed_dataset(svc, ds_path, snapshot_id="snap1"):
    return svc.index_dataset({
        "source_dataset_path": str(ds_path),
        "data_yaml_path": os.path.join(str(ds_path), "data.yaml"),
        "snapshot_id": snapshot_id,
        "snapshot_valid": True,
    })


# ─────────────────────────────────────────────────────────────
# 1. audit: read-only reconciliation report
# ─────────────────────────────────────────────────────────────


def test_audit_clean_returns_ok(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [_exp("manual:train1", data_yaml=data_yaml)])
    svc.index_experiment(_exp("manual:train1", data_yaml=data_yaml))

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code is None
    assert result.counts["scanned"] == 1
    assert result.issues == ()


def test_audit_missing_in_index(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [_exp("manual:train1", data_yaml=data_yaml)])
    # Index holds no experiment.

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["MISSING_IN_INDEX"] == 1
    assert result.issues[0].code == "MISSING_IN_INDEX"
    assert result.issues[0].subject == "manual:train1"


def test_audit_extra_in_index(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["EXTRA_IN_INDEX"] == 1
    assert result.issues[0].code == "EXTRA_IN_INDEX"


def test_audit_ignores_non_terminal_as_extra(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:running1", status="running"))
    _write_history(tmp_path / "log", [])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert "EXTRA_IN_INDEX" not in result.counts


def test_audit_status_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1", status="completed"))
    _write_history(tmp_path / "log", [_exp("manual:train1", status="failed")])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["STATUS_MISMATCH"] == 1


def test_audit_metrics_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1", metrics={"mAP50": 0.5}))
    _write_history(tmp_path / "log", [_exp("manual:train1", metrics={"mAP50": 0.6})])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["METRICS_MISMATCH"] == 1


def test_audit_dataset_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds1 = tmp_path / "ds1"
    ds2 = tmp_path / "ds2"
    d1 = _seed_dataset(svc, ds1, snapshot_id="s1")
    _seed_dataset(svc, ds2, snapshot_id="s2")
    svc.index_experiment(_exp("manual:train1", data_yaml=os.path.join(str(ds1), "data.yaml")))
    assert svc.get_experiment("manual:train1")["dataset_id"] == d1.dataset_id
    _write_latest_dataset(tmp_path / "log", ds1)
    _write_history(tmp_path / "log", [
        _exp("manual:train1", data_yaml=os.path.join(str(ds2), "data.yaml")),
    ])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["DATASET_MISMATCH"] == 1


def test_audit_artifact_state_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    report = tmp_path / "log" / "train1_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    svc.index_experiment(_exp("manual:train1"), )
    # Re-seed the artifact through the repository so the stored state is "exists".
    conn = connect_database(cfg)
    try:
        repo = LocalIndexRepository(conn)
        repo.upsert_experiment(
            ExperimentRecord(
                run_id="manual:train1", source="manual", status="completed",
                params={}, metrics={}, updated_at="2026-08-01T00:00:00Z",
            ),
            artifacts=(ArtifactRecord(run_id="manual:train1", kind="report", path=str(report)),),
        )
    finally:
        conn.close()
    report.unlink()  # live state is now "missing", stored state still "exists"

    _write_history(tmp_path / "log", [_exp("manual:train1")])
    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["ARTIFACT_STATE_MISMATCH"] == 1


def test_audit_orphan_dataset_foreign_key(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.initialize()
    # Insert an experiment referencing an unknown dataset with FK temporarily off.
    conn = connect_database(cfg)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO experiments(run_id, source, dataset_id, status, params_json, metrics_json, updated_at)"
            " VALUES ('manual:orphan', 'manual', 'missing-dataset', 'completed', '{}', '{}', '2026-08-01T00:00:00Z')"
        )
        conn.commit()
    finally:
        conn.close()
    _write_history(tmp_path / "log", [])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.counts["ORPHAN_DATASET"] == 1


def test_audit_corrupt_database_stable_error_no_path_leak(tmp_path):
    from dataclasses import asdict

    cfg = _cfg(tmp_path)
    db = cfg.database_path
    db.write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    _write_history(tmp_path / "log", [])

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code == "LOCAL_INDEX_CORRUPT"
    blob = json.dumps(asdict(result))
    assert str(tmp_path) not in blob
    assert "sqlite" not in blob.lower()


def test_audit_issue_list_bounded(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    experiments = [_exp(f"manual:missing_{i}",
                        finished_at=f"2026-08-01T00:{i % 60:02d}:{i // 60:02d}Z")
                   for i in range(150)]
    _write_history(tmp_path / "log", experiments)

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert len(result.issues) == MAX_ISSUES
    assert result.counts["MISSING_IN_INDEX"] == 150


def test_audit_does_not_modify_fact_files(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [_exp("manual:train1", data_yaml=data_yaml)])
    hist = tmp_path / "log" / "experiment_history.json"
    latest = tmp_path / "log" / "latest_dataset.json"
    hist_bytes = hist.read_bytes()
    latest_bytes = latest.read_bytes()

    audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert hist.read_bytes() == hist_bytes
    assert latest.read_bytes() == latest_bytes


# ─────────────────────────────────────────────────────────────
# 2. rebuild: backup → temp rebuild → quick_check → atomic publish
# ─────────────────────────────────────────────────────────────


def test_rebuild_success_backup_created_and_records_preserved(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [
        _exp("manual:train1", data_yaml=data_yaml),
        _exp("tuning:s1:at1", source="tuning", data_yaml=data_yaml,
             finished_at="2026-08-02T00:00:00Z"),
    ])
    svc.index_experiment(_exp("manual:train1", data_yaml=data_yaml))

    result = rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code is None
    assert result.backup_created is True
    assert result.datasets >= 1
    assert result.experiments == 2
    assert list(cfg.backup_dir.glob("auto_tune.db.manual.*.bak"))

    got = LocalIndexService(cfg).list_experiments(ExperimentQuery(limit=100))
    assert {r["run_id"] for r in got} == {"manual:train1", "tuning:s1:at1"}


def test_rebuild_preserves_s1_5_runtime_key(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"), runtime_run_id="manual:uuid-1")
    assert svc.get_experiment("manual:uuid-1") is not None
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    result = rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code is None
    got = LocalIndexService(cfg).get_experiment("manual:uuid-1")
    assert got is not None
    assert got["params"]["_legacy_record_run_id"] == "manual:train1"


def test_rebuild_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [_exp("manual:train1", data_yaml=data_yaml)])

    first = rebuild_index(cfg, log_dir=str(tmp_path / "log"))
    second = rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert first.error_code is None and second.error_code is None
    assert second.experiments == first.experiments == 1
    assert second.datasets == first.datasets


def test_rebuild_backup_failure_keeps_original(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    before = cfg.database_path.read_bytes()

    import auto_tune.modules.local_index.reconciliation as recmod

    def boom(config):
        raise LocalIndexPersistenceError("backup failed")

    monkeypatch.setattr(recmod, "create_backup", boom)
    with pytest.raises(LocalIndexPersistenceError):
        rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert cfg.database_path.read_bytes() == before
    assert not list(Path(cfg.database_path).parent.glob("*.rebuild-*.tmp"))


def test_rebuild_temp_build_failure_keeps_original(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    before = cfg.database_path.read_bytes()

    import auto_tune.modules.local_index.reconciliation as recmod

    def boom(service, key_map, log_dir, manifest_map, snap_root_norm):
        raise LocalIndexCorruptError("projection failed")

    monkeypatch.setattr(recmod, "_project_facts", boom)
    with pytest.raises(LocalIndexCorruptError):
        rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert cfg.database_path.read_bytes() == before
    assert not list(Path(cfg.database_path).parent.glob("*.rebuild-*.tmp"))


def test_rebuild_quick_check_failure_keeps_original(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    before = cfg.database_path.read_bytes()

    import auto_tune.modules.local_index.reconciliation as recmod

    real_check = recmod.check_database_integrity

    def selective(config):
        if config.database_path != cfg.database_path:
            raise LocalIndexCorruptError("temp db corrupt")
        real_check(config)

    monkeypatch.setattr(recmod, "check_database_integrity", selective)
    with pytest.raises(LocalIndexCorruptError):
        rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert cfg.database_path.read_bytes() == before


def test_rebuild_replace_failure_keeps_original(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    before = cfg.database_path.read_bytes()

    import auto_tune.modules.local_index.reconciliation as recmod

    real_replace = recmod.os.replace

    def boom(src, dst):
        if str(dst) == str(cfg.database_path):
            raise OSError("replace denied")
        return real_replace(src, dst)

    monkeypatch.setattr(recmod.os, "replace", boom)
    with pytest.raises(LocalIndexPersistenceError):
        rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    assert cfg.database_path.read_bytes() == before


def test_rebuild_single_instance_gate(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    _write_history(tmp_path / "log", [])
    lock = rebuild_lock_path(cfg)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("lock held")

    with pytest.raises(LocalIndexRebuildInProgress):
        rebuild_index(cfg, log_dir=str(tmp_path / "log"))


def test_rebuild_records_maintenance_event(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    _write_history(tmp_path / "log", [])

    rebuild_index(cfg, log_dir=str(tmp_path / "log"))

    conn = connect_database(cfg)
    try:
        repo = LocalIndexRepository(conn)
        events = repo.recent_maintenance_events(5)
    finally:
        conn.close()
    assert any(e["kind"] == "rebuild" for e in events)
    assert any(e["kind"] == "backup" for e in events)


# ─────────────────────────────────────────────────────────────
# 3. startup backfill: bounded recent facts, idempotent, honest
# ─────────────────────────────────────────────────────────────


def test_backfill_indexes_recent_missing(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    result = backfill_startup(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code is None
    assert result.scanned == 1
    assert result.indexed == 1
    assert result.failures == 0
    assert LocalIndexService(cfg).get_experiment("manual:train1") is not None


def test_backfill_is_bounded_to_recent(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    experiments = [_exp(f"manual:train_{i}", finished_at=f"2026-08-01T00:{i:02d}:00Z")
                   for i in range(60)]
    _write_history(tmp_path / "log", experiments)

    result = backfill_startup(cfg, log_dir=str(tmp_path / "log"), max_records=50)

    assert result.scanned == 50
    assert result.indexed == 50
    # The 10 oldest records are not indexed.
    got = {r["run_id"] for r in LocalIndexService(cfg).list_experiments(ExperimentQuery(limit=100))}
    assert "manual:train_0" not in got
    assert "manual:train_59" in got


def test_backfill_does_not_delete_existing(tmp_path):
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    svc.index_experiment(_exp("manual:keep", metrics={"mAP50": 0.5}))
    _write_history(tmp_path / "log", [
        _exp("manual:keep", metrics={"mAP50": 0.5}),
        _exp("manual:new1"),
    ])

    result = backfill_startup(cfg, log_dir=str(tmp_path / "log"))

    got = {r["run_id"]: r for r in LocalIndexService(cfg).list_experiments(ExperimentQuery(limit=100))}
    assert result.indexed == 1  # only the genuinely new record
    assert "manual:keep" in got
    assert "manual:new1" in got


def test_backfill_unknown_terminal_not_guessed_completed(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    _write_history(tmp_path / "log", [_exp("manual:mystery", status="unknown")])

    backfill_startup(cfg, log_dir=str(tmp_path / "log"))

    got = LocalIndexService(cfg).get_experiment("manual:mystery")
    assert got is not None
    assert got["status"] == "unknown"


def test_backfill_tail_records_eventually_indexed(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    # Training completed while the index was unavailable: only JSON facts exist.
    _write_history(tmp_path / "log", [_exp("manual:tail1")])

    result = backfill_startup(cfg, log_dir=str(tmp_path / "log"))

    assert result.indexed == 1
    assert LocalIndexService(cfg).get_experiment("manual:tail1") is not None


def test_backfill_idempotent_no_duplicates(tmp_path):
    cfg = _cfg(tmp_path)
    _service(cfg)
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    backfill_startup(cfg, log_dir=str(tmp_path / "log"))
    second = backfill_startup(cfg, log_dir=str(tmp_path / "log"))

    assert second.indexed == 0
    assert second.skipped == 1
    rows = LocalIndexService(cfg).list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 1


# ─────────────────────────────────────────────────────────────
# 4. API: GET audit (read-only), POST rebuild (CSRF-gated)
# ─────────────────────────────────────────────────────────────


def _use_tmp_log(monkeypatch, tmp_path):
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    return log_dir


def _client():
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    return TestClient(app_mod.app)


def test_audit_route_get_returns_stable_shape(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    resp = _client().get("/api/local-index/audit")

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) >= {"timestamp", "error_code", "scanned", "counts", "issues"}
    assert data["counts"]["MISSING_IN_INDEX"] == 1
    assert data["issues"][0]["code"] == "MISSING_IN_INDEX"


def test_audit_route_corrupt_db_stable_error_code(tmp_path, monkeypatch):
    # Like the status route, a diagnostic audit reports corruption honestly in
    # the body with a stable error code instead of pretending the index is fine.
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "auto_tune.db").write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)

    resp = _client().get("/api/local-index/audit")

    assert resp.status_code == 200
    assert resp.json()["error_code"] == "LOCAL_INDEX_CORRUPT"
    assert resp.json()["counts"]["LOCAL_INDEX_CORRUPT"] == 1


def test_rebuild_route_requires_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/local-index/rebuild", json={})
    assert resp.status_code == 403


def test_rebuild_route_with_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    from auto_tune.ui import app as app_mod

    headers = {
        "X-CSRF-Token": app_mod._CSRF_TOKEN,
        "Origin": "http://testserver",
    }
    resp = _client().post("/api/local-index/rebuild", headers=headers, json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error_code"] is None
    assert data["experiments"] == 1
    rows = _client().get("/api/experiments").json()["items"]
    assert {r["run_id"] for r in rows} == {"manual:train1"}


# ─────────────────────────────────────────────────────────────
# 返修 1: 生产启动补录 — startup 生命周期接线（非 service 方法级）
# ─────────────────────────────────────────────────────────────


def test_startup_backfill_wired_into_app_lifecycle(tmp_path, monkeypatch):
    """The app startup hook backfills recent facts and persists a summary that
    diagnostics can read — a production wiring test, not a service-only test."""
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    _write_history(tmp_path / "log", [
        _exp("manual:train1", finished_at="2026-08-01T00:00:00Z"),
        _exp("tuning:s1:at1", source="tuning", finished_at="2026-08-02T00:00:00Z"),
    ])

    with _client() as client:
        rows = client.get("/api/experiments?limit=50").json()["items"]
        assert {r["run_id"] for r in rows} == {"manual:train1", "tuning:s1:at1"}

        diag = client.get("/api/local-index/diagnostics").json()
        assert any(e["kind"] == "backfill" for e in diag["recent_events"])


def test_startup_backfill_non_fatal_when_index_unavailable(tmp_path, monkeypatch):
    """A corrupt database at startup must not block the app from serving."""
    from auto_tune.ui import app as app_mod

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "auto_tune.db").write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    with _client() as client:
        # The app still starts and serves the dashboard; the index is reported
        # honestly as corrupt, never as a startup crash.
        resp = client.get("/api/local-index/status")
        assert resp.status_code == 200
        assert resp.json()["error_code"] == "LOCAL_INDEX_CORRUPT"


# ─────────────────────────────────────────────────────────────
# 返修 2: GET audit 真正只读 + 显式 POST 持久化 audit 摘要
# ─────────────────────────────────────────────────────────────


def _db_state(tmp_path):
    db = tmp_path / "log" / "auto_tune.db"
    conn = connect_database(LocalIndexConfig(
        database_path=db,
        backup_dir=tmp_path / "log" / "db_backups",
        busy_timeout_ms=200,
    ))
    try:
        repo = LocalIndexRepository(conn)
        table_counts = {
            "datasets": conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "experiments": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            "artifacts": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            "maintenance_events": conn.execute("SELECT COUNT(*) FROM maintenance_events").fetchone()[0],
        }
    finally:
        conn.close()
    files = sorted(
        str(p.relative_to(tmp_path))
        for p in tmp_path.rglob("*")
        if p.is_file() and p.name.startswith("auto_tune.db")
    )
    stat = db.stat()
    return {
        "bytes": db.read_bytes(),
        "mtime_ns": stat.st_mtime_ns,
        "counts": table_counts,
        "files": files,
    }


def test_audit_get_is_truly_read_only(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    before = _db_state(tmp_path)

    resp = _client().get("/api/local-index/audit")

    assert resp.status_code == 200
    assert resp.json()["error_code"] is None
    after = _db_state(tmp_path)
    assert after["bytes"] == before["bytes"]
    assert after["mtime_ns"] == before["mtime_ns"]
    assert after["counts"] == before["counts"]
    assert after["files"] == before["files"]
    # The read-only GET must never persist an audit maintenance event.
    assert before["counts"]["maintenance_events"] == after["counts"]["maintenance_events"] == 0


def test_audit_get_does_not_create_missing_database(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    _write_history(tmp_path / "log", [_exp("manual:train1")])

    resp = _client().get("/api/local-index/audit")

    assert resp.status_code == 200
    data = resp.json()
    assert data["counts"]["MISSING_IN_INDEX"] == 1
    # No database file may be created by a read-only GET.
    assert not (tmp_path / "log" / "auto_tune.db").exists()
    assert not list(Path(tmp_path).rglob("auto_tune.db*"))


def test_audit_record_route_requires_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/local-index/audit/record", json={})
    assert resp.status_code == 403


def test_audit_record_route_persists_summary(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment(_exp("manual:train1"))
    _write_history(tmp_path / "log", [_exp("manual:train1")])
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}

    resp = _client().post("/api/local-index/audit/record", headers=headers, json={})

    assert resp.status_code == 200
    conn = connect_database(LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
        busy_timeout_ms=200,
    ))
    try:
        events = LocalIndexRepository(conn).recent_maintenance_events(5)
    finally:
        conn.close()
    assert any(e["kind"] == "audit" for e in events)


# ─────────────────────────────────────────────────────────────
# 返修 5: 损坏事实文件 → 稳定领域错误（不泄漏 traceback/SQL/绝对路径）
# ─────────────────────────────────────────────────────────────


def test_audit_corrupt_fact_history_stable_error_no_path_leak(tmp_path, monkeypatch):
    from dataclasses import asdict

    _use_tmp_log(monkeypatch, tmp_path)
    (tmp_path / "log" / "experiment_history.json").write_text(
        "{ not valid json ", encoding="utf-8")

    result = audit_index(_cfg(tmp_path), log_dir=str(tmp_path / "log"))

    assert result.error_code == "LOCAL_INDEX_CORRUPT"
    blob = json.dumps(asdict(result))
    assert str(tmp_path) not in blob
    assert "Traceback" not in blob


def test_rebuild_corrupt_fact_history_keeps_db_and_stable_error(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment(_exp("manual:train1"))
    before = (log_dir / "auto_tune.db").read_bytes()
    (log_dir / "experiment_history.json").write_text("{ broken ", encoding="utf-8")

    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    resp = _client().post("/api/local-index/rebuild", headers=headers, json={})

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "LOCAL_INDEX_CORRUPT"
    assert str(tmp_path) not in json.dumps(resp.json())
    assert (log_dir / "auto_tune.db").read_bytes() == before


# ─────────────────────────────────────────────────────────────
# 返修 3: 多数据集重建 — 恢复历史实验的数据集/快照关联
# ─────────────────────────────────────────────────────────────


def _write_snapshot(log_dir, snap_name, source_root=None, snapshot_id=None,
                    corrupt_digest=False, missing_manifest=False):
    """Write a snapshot manifest with a canonical digest (or a wrong/corrupt
    digest when ``corrupt_digest``), optionally without the manifest file."""
    snap_dir = log_dir / "dataset_snapshots" / snap_name
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "snapshot_id": snapshot_id or f"{snap_name}-id",
        "created_at": "2026-08-01T00:00:00Z",
        "source_root": str(source_root or log_dir / "sources" / snap_name),
        "samples": [],
        "train_count": 0,
        "val_count": 0,
        "background_count": 0,
    }
    digest_payload = {
        k: v for k, v in manifest.items()
        if k not in ("created_at", "source_root", "manifest_digest")
    }
    manifest["manifest_digest"] = (
        "wrong-digest" if corrupt_digest
        else hashlib.sha256(_canonical_json_bytes(digest_payload)).hexdigest()
    )
    if not missing_manifest:
        (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (snap_dir / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    return str(snap_dir / "data.yaml")


def test_rebuild_restores_multiple_dataset_associations(tmp_path):
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB")
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
        _exp("tuning:s1:at1", source="tuning", data_yaml=data_a,
             finished_at="2026-08-03T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.dataset_unresolved == 0
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:a"]["dataset_id"]
    assert by_id["manual:b"]["dataset_id"]
    assert by_id["tuning:s1:at1"]["dataset_id"] == by_id["manual:a"]["dataset_id"]
    assert by_id["manual:b"]["dataset_id"] != by_id["manual:a"]["dataset_id"]
    dataset_ids = {d["dataset_id"] for d in svc.list_datasets()}
    assert by_id["manual:a"]["dataset_id"] in dataset_ids
    assert by_id["manual:b"]["dataset_id"] in dataset_ids


def test_rebuild_reports_dataset_unresolved_when_no_data_path(tmp_path):
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    _write_history(log_dir, [
        _exp("manual:a", params={}, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=os.path.join(str(log_dir), "dataset_snapshots", "snapA", "data.yaml"),
             finished_at="2026-08-02T00:00:00Z"),
    ])
    _write_snapshot(log_dir, "snapA")

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    # The experiment with no data reference is reported explicitly, not silently.
    assert result.dataset_unresolved == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"]
    assert by_id["manual:a"]["dataset_id"] is None


# ─────────────────────────────────────────────────────────────
# 返修 2: 诚实处理损坏/缺失 snapshot manifest + snapshot_unresolved
# ─────────────────────────────────────────────────────────────


def test_rebuild_corrupt_manifest_reports_snapshot_unresolved(tmp_path):
    """A corrupt manifest must not silently downgrade the experiment to an
    unverified dataset; the identity is reported unresolved and no dataset is
    created for it."""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", corrupt_digest=True)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("corrupt") == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:a"]["dataset_id"]
    assert by_id["manual:b"]["dataset_id"] is None
    # No unverified dataset was fabricated for the corrupt snapshot identity.
    dataset_ids = {d["dataset_id"] for d in svc.list_datasets()}
    assert by_id["manual:a"]["dataset_id"] in dataset_ids


def test_rebuild_missing_manifest_reports_snapshot_unresolved(tmp_path):
    """A snapshot directory without a manifest cannot restore its identity."""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", missing_manifest=True)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("missing") == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"] is None


def test_rebuild_invalid_manifest_json_reports_snapshot_unresolved(tmp_path):
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    snap_dir = log_dir / "dataset_snapshots" / "snapB"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "manifest.json").write_text("{ not json ", encoding="utf-8")
    (snap_dir / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    data_b = str(snap_dir / "data.yaml")
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("corrupt") == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"] is None


def test_rebuild_snapshot_partial_recovery_is_idempotent(tmp_path):
    """A second rebuild preserves the partially-recovered associations and the
    honest unresolved counts (no drift, no duplicate datasets)."""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", corrupt_digest=True)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    first = rebuild_index(cfg, log_dir=str(log_dir))
    second = rebuild_index(cfg, log_dir=str(log_dir))

    assert first.error_code is None and second.error_code is None
    assert second.snapshot_unresolved == first.snapshot_unresolved == 1
    assert second.datasets == first.datasets
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:a"]["dataset_id"]
    assert by_id["manual:b"]["dataset_id"] is None
    assert len(svc.list_datasets()) == 1  # only snapA, never a fabricated duplicate


# ─────────────────────────────────────────────────────────────
# 返修 3: audit 全链路严格只读（不打开任何读写连接）
# ─────────────────────────────────────────────────────────────


def test_audit_never_opens_readwrite_connection(tmp_path, monkeypatch):
    """GET audit from quick_check to the projection queries must use mode=ro
    exclusively; any read-write connect call is forbidden."""
    cfg = _cfg(tmp_path)
    svc = _service(cfg)
    ds = tmp_path / "ds"
    _seed_dataset(svc, ds)
    data_yaml = _write_latest_dataset(tmp_path / "log", ds)
    _write_history(tmp_path / "log", [_exp("manual:train1", data_yaml=data_yaml)])
    svc.index_experiment(_exp("manual:train1", data_yaml=data_yaml))

    import auto_tune.modules.local_index.database as dbmod
    import auto_tune.modules.local_index.reconciliation as recmod

    def _forbid_rw(config):
        raise AssertionError("audit must not open a read-write connection")

    monkeypatch.setattr(dbmod, "connect_database", _forbid_rw)
    monkeypatch.setattr(recmod, "connect_database", _forbid_rw)
    real_connect = dbmod.sqlite3.connect

    def _ro_only(*args, **kwargs):
        if not kwargs.get("uri") and not (args and "mode=ro" in str(args[0])):
            raise AssertionError("audit opened a raw read-write sqlite3.connect")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dbmod.sqlite3, "connect", _ro_only)

    result = audit_index(cfg, log_dir=str(tmp_path / "log"))

    assert result.error_code is None
    assert result.counts["scanned"] == 1


def test_audit_readonly_integrity_does_not_touch_wal(tmp_path):
    """A read-only audit must leave the database byte set untouched: no WAL/SHM
    companion file is created or removed and the database bytes are equal."""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    _service(cfg)
    _write_history(log_dir, [_exp("manual:train1")])
    db = cfg.database_path

    def _companions():
        return sorted(
            str(p.relative_to(tmp_path))
            for p in tmp_path.rglob("auto_tune.db*")
            if p.is_file()
        )

    before_bytes = db.read_bytes()
    before_files = _companions()
    result = audit_index(cfg, log_dir=str(log_dir))
    assert result.error_code is None
    assert db.read_bytes() == before_bytes
    assert _companions() == before_files


# ─────────────────────────────────────────────────────────────
# 返修 4: 快照扫描有界化（固定深度 / 数量 / 单文件 / 总量 / 分块读取）
# ─────────────────────────────────────────────────────────────


def test_scan_snapshot_manifests_ignores_deep_directories(tmp_path, monkeypatch):
    """Only immediate children of dataset_snapshots are scanned; a manifest
    nested one level deeper is never read."""
    import auto_tune.modules.local_index.reconciliation as recmod

    log_dir = tmp_path / "log"
    (log_dir / "dataset_snapshots").mkdir(parents=True)
    data_a = _write_snapshot(log_dir, "snapA")
    deep = log_dir / "dataset_snapshots" / "nested" / "inner"
    deep.mkdir(parents=True)
    (deep / "manifest.json").write_text("{}", encoding="utf-8")
    (deep / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    calls = []

    real_read = recmod._read_manifest_bounded

    def spy(path, remaining):
        calls.append(str(path))
        return real_read(path, remaining)

    monkeypatch.setattr(recmod, "_read_manifest_bounded", spy)

    manifest_map, status_counts, exceeded = recmod.scan_snapshot_manifests(log_dir)

    assert data_a is not None
    assert status_counts["valid"] == 1
    # The nested dir is seen as a missing-manifest immediate child; its inner
    # manifest is never read.
    assert status_counts["missing"] == 1
    assert all("nested" not in p for p in calls)
    assert exceeded is False


def test_scan_snapshot_manifests_bounded_entry_count(tmp_path, monkeypatch):
    import auto_tune.modules.local_index.reconciliation as recmod

    monkeypatch.setattr(recmod, "MAX_SNAPSHOT_SCAN_ENTRIES", 3)
    log_dir = tmp_path / "log"
    for i in range(6):
        _write_snapshot(log_dir, f"snap{i}")
    manifest_map, status_counts, exceeded = recmod.scan_snapshot_manifests(log_dir)
    assert exceeded is True
    assert status_counts["valid"] <= 3


def test_scan_snapshot_manifests_growing_file_still_bounded(tmp_path, monkeypatch):
    """A file that reports a small stat size but contains more than the per-file
    limit (grew during read) is caught by the chunked reader."""
    import auto_tune.modules.local_index.reconciliation as recmod

    import pathlib

    manifest_path = tmp_path / "manifest.json"
    payload = b"x" * (recmod.MAX_SNAPSHOT_MANIFEST_BYTES + 1)
    manifest_path.write_bytes(payload)

    real_stat = pathlib.Path.stat

    def fake_stat(self):
        class _Fake:
            st_size = 16
        return _Fake()

    monkeypatch.setattr(pathlib.Path, "stat", fake_stat)
    raw, status = recmod._read_manifest_bounded(manifest_path, recmod.MAX_SNAPSHOT_TOTAL_BYTES)
    assert status == "too_large"
    assert raw is None


def test_scan_snapshot_manifests_single_file_over_limit(tmp_path):
    import auto_tune.modules.local_index.reconciliation as recmod

    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    big = log_dir / "dataset_snapshots" / "snapB"
    big.mkdir(parents=True, exist_ok=True)
    (big / "manifest.json").write_bytes(b" " * (recmod.MAX_SNAPSHOT_MANIFEST_BYTES + 1))
    (big / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    data_b = str(big / "data.yaml")
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(_cfg(tmp_path), log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("too_large") == 1


# ─────────────────────────────────────────────────────────────
# 返修3: 严格实际读取预算 — 总预算只剩少量字节时仍按 min 截断，不固定读 1 MiB
# ─────────────────────────────────────────────────────────────


def test_read_manifest_bounded_respects_tiny_total_budget(tmp_path, monkeypatch):
    """总预算只剩少量字节：read 长度取 min(chunk, 单文件剩余+1, 总预算剩余+1)，
    实际读取绝不超过预算，返回 too_large 且不读取整个文件。"""
    import auto_tune.modules.local_index.reconciliation as recmod

    import pathlib

    manifest_path = tmp_path / "manifest.json"
    payload = b"x" * 4096  # 4 KiB, well under per-file 1 MiB but over a tiny budget
    manifest_path.write_bytes(payload)

    real_stat = pathlib.Path.stat

    def fake_stat(self):
        class _Fake:
            st_size = 4096
        return _Fake()

    monkeypatch.setattr(pathlib.Path, "stat", fake_stat)
    # Remaining total budget is only 8 bytes; the reader must stop at 8 (plus the
    # single probe byte) instead of reading the whole 4 KiB file.
    raw, status = recmod._read_manifest_bounded(manifest_path, remaining_budget=8)
    assert status == "too_large"
    assert raw is None


def test_read_manifest_bounded_never_reads_fixed_1mb(tmp_path, monkeypatch):
    """不得固定 read(1 MiB) 后再判断：每次 read 由 min 上限约束，实际 read 请求
    长度受单文件剩余+1 与总预算剩余+1 限制，绝不一次请求整个 chunk。"""
    import auto_tune.modules.local_index.reconciliation as recmod

    import pathlib

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"x" * 100)

    real_stat = pathlib.Path.stat

    def fake_stat(self):
        class _Fake:
            st_size = 100
        return _Fake()

    monkeypatch.setattr(pathlib.Path, "stat", fake_stat)
    import builtins

    real_open = builtins.open

    # A 100-byte file with a 100-byte budget: read_len = min(1MiB, 100-0+1, 100-0+1)
    # = 101, never 1 MiB. The file is fully consumed in one read, status valid.
    class _Reader:
        def __init__(self, fh):
            self._fh = fh
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()
            return False

        def read(self, n):
            self.calls.append(n)
            return self._fh.read(n)

        def __getattr__(self, item):
            return getattr(self._fh, item)

    opened = {}

    def fake_open(*a, **kw):
        fh = real_open(*a, **kw)
        reader = _Reader(fh)
        opened["reader"] = reader
        return reader

    monkeypatch.setattr(builtins, "open", fake_open)
    raw, status = recmod._read_manifest_bounded(manifest_path, remaining_budget=100)
    assert status == "valid"
    reader = opened["reader"]
    assert reader.calls, "manifest was never read"
    assert max(reader.calls) <= 101  # budget + 1 probe byte, not the 1 MiB chunk
    assert max(reader.calls) < 1024 * 1024


# ─────────────────────────────────────────────────────────────
# 返修4: 拒绝快照目录链接（symlink / Windows junction / reparse point）
# ─────────────────────────────────────────────────────────────


def _make_dir_link(target, link_path) -> bool:
    """Create a directory symlink/junction; return False when the platform
    refuses (e.g. missing privileges), so the caller can skip."""
    import subprocess

    if os.name == "nt":
        # Windows: try a real directory symlink, then a junction via cmd.
        try:
            os.symlink(str(target), str(link_path), target_is_directory=True)
            return True
        except (OSError, NotImplementedError):
            try:
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
                    capture_output=True,
                )
                return result.returncode == 0
            except OSError:
                return False
    try:
        os.symlink(str(target), str(link_path), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def test_scan_snapshot_manifests_rejects_symlink_escape(tmp_path, monkeypatch):
    """A snapshot directory that is a symlink/junction pointing outside the root
    is counted unreadable and its manifest is never read."""
    import auto_tune.modules.local_index.reconciliation as recmod

    log_dir = tmp_path / "log"
    root = log_dir / "dataset_snapshots"
    root.mkdir(parents=True)
    _write_snapshot(log_dir, "snapA")

    # An external directory holding a *valid* manifest that must never be read.
    external = tmp_path / "external"
    external.mkdir()
    ext_manifest = {
        "schema_version": "1.0",
        "snapshot_id": "evil-id",
        "created_at": "2026-08-01T00:00:00Z",
        "source_root": str(external),
        "samples": [], "train_count": 0, "val_count": 0, "background_count": 0,
    }
    digest_payload = {k: v for k, v in ext_manifest.items()
                      if k not in ("created_at", "source_root", "manifest_digest")}
    ext_manifest["manifest_digest"] = hashlib.sha256(
        _canonical_json_bytes(digest_payload)).hexdigest()
    (external / "manifest.json").write_text(json.dumps(ext_manifest), encoding="utf-8")
    (external / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")

    link = root / "evilsnap"
    if not _make_dir_link(external, link):
        pytest.skip("symlink/junction not permitted on this platform")

    calls = []
    real_read = recmod._read_manifest_bounded

    def spy(path, remaining):
        calls.append(str(path))
        return real_read(path, remaining)

    monkeypatch.setattr(recmod, "_read_manifest_bounded", spy)
    manifest_map, status_counts, exceeded = recmod.scan_snapshot_manifests(log_dir)

    assert status_counts.get("unreadable") == 1
    assert not any("external" in p or "evilsnap" in p for p in calls)
    # snapA still scanned; the escape link contributed nothing.
    assert status_counts["valid"] == 1


def test_scan_snapshot_manifests_rejects_snapshot_root_link(tmp_path):
    """When the snapshot root itself is a link, nothing under it is scanned and
    the failure is a stable unreadable count."""
    import auto_tune.modules.local_index.reconciliation as recmod

    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    external = tmp_path / "ext_root"
    external.mkdir()

    link_root = log_dir / "dataset_snapshots"
    if not _make_dir_link(external, link_root):
        pytest.skip("symlink/junction not permitted on this platform")

    manifest_map, status_counts, exceeded = recmod.scan_snapshot_manifests(log_dir)
    assert status_counts.get("unreadable") == 1
    assert manifest_map == {}
    assert exceeded is False


# ─────────────────────────────────────────────────────────────
# 返修1: latest_dataset 与 manifest 的信任顺序（专项测试）
# latest_dataset 若声明 snapshot 身份（snapshot_id / snapshot_digest /
# 路径位于 dataset_snapshots 根下），必须经 manifest_map 找到路径、snapshot_id、
# digest 一致的 valid manifest 才能建立有效数据集；损坏/缺失/digest 不一致的
# manifest 一律不建立 dataset_id，实验计入 snapshot_unresolved，且不得被
# latest_dataset 的 known 集合绕过。
# ─────────────────────────────────────────────────────────────


def test_rebuild_latest_dataset_corrupt_manifest_not_trusted(tmp_path):
    """latest_dataset 指向损坏 manifest：不建立 dataset，引用实验 snapshot_unresolved。"""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", corrupt_digest=True)
    _write_latest_dataset(log_dir, log_dir / "dataset_snapshots" / "snapB",
                          snapshot_id="snapB-id", data_yaml=data_b)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("corrupt") == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"] is None
    assert by_id["manual:a"]["dataset_id"]
    # The corrupt snapshot identity is never fabricated as a dataset.
    assert len(svc.list_datasets()) == 1


def test_rebuild_latest_dataset_missing_manifest_not_trusted(tmp_path):
    """latest_dataset 指向缺失 manifest：不建立 dataset，引用实验 snapshot_unresolved。"""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", missing_manifest=True)
    _write_latest_dataset(log_dir, log_dir / "dataset_snapshots" / "snapB",
                          snapshot_id="snapB-id", data_yaml=data_b)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 1
    assert result.snapshot_issues.get("missing") == 1
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"] is None
    assert len(svc.list_datasets()) == 1


def test_rebuild_latest_dataset_digest_mismatch_not_trusted(tmp_path):
    """latest_dataset 声明的 digest 与有效 manifest 不一致：声明不被采用，数据集
    身份以 valid manifest 为准（manifest 是权威），不产生虚假的错 digest 数据集。"""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB")
    _write_latest_dataset(log_dir, log_dir / "dataset_snapshots" / "snapB",
                          snapshot_id="snapB-id", data_yaml=data_b,
                          snapshot_digest="fake-digest")
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 0
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    # The experiment still associates through the valid manifest; the dataset's
    # recorded digest is the manifest's verified digest, never the fake claim.
    assert by_id["manual:b"]["dataset_id"]
    ds = svc.get_dataset(by_id["manual:b"]["dataset_id"])
    assert ds is not None and ds["snapshot_digest"] != "fake-digest"
    assert all(d["snapshot_digest"] != "fake-digest" for d in svc.list_datasets())


def test_rebuild_latest_dataset_valid_manifest_establishes(tmp_path):
    """正常 latest_dataset + valid manifest：建立有效数据集和实验关联。"""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB")
    _write_latest_dataset(log_dir, log_dir / "dataset_snapshots" / "snapB",
                          snapshot_id="snapB-id", data_yaml=data_b)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    result = rebuild_index(cfg, log_dir=str(log_dir))

    assert result.error_code is None
    assert result.snapshot_unresolved == 0
    assert result.dataset_unresolved == 0
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"]
    assert by_id["manual:a"]["dataset_id"]


def test_rebuild_latest_dataset_broken_manifest_idempotent(tmp_path):
    """二次重建幂等：损坏 latest_dataset 引用不产生漂移、不重复数据集。"""
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "log"
    data_a = _write_snapshot(log_dir, "snapA")
    data_b = _write_snapshot(log_dir, "snapB", corrupt_digest=True)
    _write_latest_dataset(log_dir, log_dir / "dataset_snapshots" / "snapB",
                          snapshot_id="snapB-id", data_yaml=data_b)
    _write_history(log_dir, [
        _exp("manual:a", data_yaml=data_a, finished_at="2026-08-01T00:00:00Z"),
        _exp("manual:b", data_yaml=data_b, finished_at="2026-08-02T00:00:00Z"),
    ])

    first = rebuild_index(cfg, log_dir=str(log_dir))
    second = rebuild_index(cfg, log_dir=str(log_dir))

    assert first.error_code is None and second.error_code is None
    assert second.snapshot_unresolved == first.snapshot_unresolved == 1
    assert second.snapshot_issues == first.snapshot_issues
    svc = LocalIndexService(cfg)
    by_id = {r["run_id"]: r for r in svc.list_experiments(ExperimentQuery(limit=100))}
    assert by_id["manual:b"]["dataset_id"] is None
    assert len(svc.list_datasets()) == 1  # only snapA, never a fabricated duplicate
