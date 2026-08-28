"""Task 5: local-index query / import API and stable error mapping."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.local_index import (
    LocalIndexConfig,
    LocalIndexPersistenceError,
    LocalIndexService,
)
from auto_tune.ui import app as app_mod


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


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_dataset({
        "source_dataset_path": "C:/data/ds1",
        "data_yaml_path": "C:/data/ds1/data.yaml",
        "snapshot_id": "snap1", "snapshot_valid": True,
    })
    svc.index_dataset({
        "source_dataset_path": "C:/data/ds2",
        "data_yaml_path": "C:/data/ds2/data.yaml",
        "snapshot_id": "snap2", "snapshot_valid": True,
    })
    svc.index_experiment({
        "run_id": "manual:1", "run_name": "train1", "source": "manual",
        "status": "completed", "params": {"data": "C:/data/ds1/data.yaml"},
        "metrics": {"mAP50": 0.5},
    })
    svc.index_experiment({
        "run_id": "tuning:uuid:at1", "run_name": "at1", "source": "tuning",
        "status": "failed", "params": {"data": "C:/data/ds2/data.yaml"},
        "metrics": {"mAP50": 0.6},
    })
    return {"log_dir": log_dir, "service": svc}


def _client():
    return TestClient(app_mod.app)


def test_status_route(seeded):
    resp = _client().get("/api/local-index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["schema_version"] == 2
    assert data["dataset_count"] == 2
    assert data["experiment_count"] == 2


def test_datasets_route(seeded):
    resp = _client().get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    names = {d["display_name"] for d in data["datasets"]}
    assert names == {"ds1", "ds2"}


def test_datasets_route_invalid_limit(seeded):
    resp = _client().get("/api/datasets?limit=0")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_dataset_detail_route(seeded):
    service = seeded["service"]
    dataset_id = service.list_datasets()[0]["dataset_id"]
    resp = _client().get(f"/api/datasets/{dataset_id}")
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == dataset_id


def test_dataset_detail_not_found(seeded):
    resp = _client().get("/api/datasets/nonexistent")
    assert resp.status_code == 404


def test_experiments_route_filters(seeded):
    client = _client()
    assert len(client.get("/api/experiments?source=manual").json()["items"]) == 1
    assert len(client.get("/api/experiments?source=tuning").json()["items"]) == 1
    assert len(client.get("/api/experiments?status=failed").json()["items"]) == 1
    service = seeded["service"]
    dataset_id = service.list_datasets()[0]["dataset_id"]
    rows = client.get(f"/api/experiments?dataset_id={dataset_id}").json()["items"]
    assert len(rows) == 1


def test_experiments_pagination_reaches_beyond_100(tmp_path, monkeypatch):
    """返修 4: server-side pagination must make every record reachable, even
    beyond the old 100-record browser-render ceiling."""
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    for i in range(105):
        svc.index_experiment({
            "run_id": f"manual:t{i:03d}", "run_name": f"train{i:03d}",
            "source": "manual", "status": "completed", "params": {},
            "metrics": {"mAP50": 0.5},
            "finished_at": f"2026-08-01T00:{i // 60:02d}:{i % 60:02d}Z",
        })

    client = _client()
    seen = set()
    offset = 0
    limit = 25
    while True:
        data = client.get(f"/api/experiments?limit={limit}&offset={offset}").json()
        items = data["items"]
        seen.update(item["run_id"] for item in items)
        assert data["total"] == 105
        if offset + len(items) >= data["total"]:
            break
        assert len(items) == limit
        offset += limit

    assert len(seen) == 105


@pytest.mark.parametrize("query", [
    "limit=0", "limit=501", "source=bogus", "status=bogus",
])
def test_experiments_route_invalid_query(seeded, query):
    resp = _client().get(f"/api/experiments?{query}")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_experiment_detail_route_urlencoded_run_id(seeded):
    # run_id contains colons; both plain and URL-encoded forms resolve.
    resp = _client().get("/api/experiments/manual:1")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "manual:1"
    resp2 = _client().get("/api/experiments/tuning%3Auuid%3Aat1")
    assert resp2.status_code == 200
    assert resp2.json()["run_id"] == "tuning:uuid:at1"


def test_experiment_detail_not_found(seeded):
    resp = _client().get("/api/experiments/does_not_exist")
    assert resp.status_code == 404


def test_corrupt_database_returns_503_for_query(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    db = log_dir / "auto_tune.db"
    db.write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    resp = _client().get("/api/experiments")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "LOCAL_INDEX_CORRUPT"
    assert "sqlite" not in json.dumps(resp.json()).lower()


def test_corrupt_database_status_honest(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "auto_tune.db").write_bytes(b"\x00\x01\x02 not sqlite at all " * 8)
    resp = _client().get("/api/local-index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert data["error_code"] == "LOCAL_INDEX_CORRUPT"


def test_lock_failure_returns_503_unavailable(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    real_builder = app_mod._local_index_service

    def locked_builder():
        svc = real_builder()
        svc.query_experiments = lambda query: (_ for _ in ()).throw(LocalIndexPersistenceError("locked"))  # type: ignore[method-assign]
        return svc

    monkeypatch.setattr(app_mod, "_local_index_service", locked_builder)
    resp = _client().get("/api/experiments")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "LOCAL_INDEX_UNAVAILABLE"


def test_import_legacy_route_with_csrf(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "experiment_history.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:old1", "run_name": "old1", "source": "manual",
             "status": "completed", "params": {}, "metrics": {}},
        ],
    }), encoding="utf-8")
    (log_dir / "tuning_history.json").write_text(json.dumps([
        {"train_name": "autotune_old", "result_mAP50": 0.4, "timestamp": "2026-08-01T00:00:00Z"},
    ]), encoding="utf-8")

    headers = {
        "X-CSRF-Token": app_mod._CSRF_TOKEN,
        "Origin": "http://testserver",
    }
    resp = _client().post("/api/local-index/import-legacy", headers=headers, json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["imported"] == 2
    assert data["failed"] == 0
    # Experiments are now queryable from the index.
    rows = _client().get("/api/experiments").json()["items"]
    assert {r["run_id"] for r in rows} >= {"manual:old1", "legacy-tuning:autotune_old"}


def test_import_legacy_route_rejects_missing_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/local-index/import-legacy", json={})
    assert resp.status_code == 403


def test_native_storage_error_returns_503_unavailable(tmp_path, monkeypatch):
    """A native OSError at the storage boundary maps to 503 + LOCAL_INDEX_UNAVAILABLE,
    never an unhandled 500."""
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a dir")
    svc = LocalIndexService(LocalIndexConfig(
        database_path=blocker / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
    ))
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: svc)
    resp = _client().get("/api/experiments")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "LOCAL_INDEX_UNAVAILABLE"
    assert "sqlite" not in json.dumps(resp.json()).lower()


# ── 返修 1: 统一 API 出站投影 — 原始 HTTP JSON 永不泄漏任何路径形式 ──


def _seed_leaky_dataset(tmp_path, monkeypatch):
    """Seed datasets + experiments carrying Windows/Linux/relative/nested paths."""
    from auto_tune.ui import app as app_mod

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    # Lowercase path segments: service normalization is normcase (lowercase on
    # Windows), so the canonical/display names are lowercase.
    windows_ds = r"D:\data\project1\datasets\dsa"
    linux_ds = "/data/projects/dsb"
    svc.index_dataset({
        "source_dataset_path": windows_ds,
        "data_yaml_path": windows_ds + r"\data.yaml",
        "snapshot_id": "snapA", "snapshot_valid": True,
    })
    svc.index_dataset({
        "source_dataset_path": linux_ds,
        "data_yaml_path": linux_ds + "/data.yaml",
        "snapshot_id": "snapB", "snapshot_valid": True,
    })
    run_dir = r"D:\detect\train1"
    report = r"C:\log\train1_report.json"
    audit = r"D:\log\tuning_audit_x.json"
    svc.index_experiment({
        "run_id": "manual:1", "run_name": "train1", "source": "manual",
        "status": "completed",
        "params": {
            "data": windows_ds + r"\data.yaml",
            "model": "yolov8n.pt",
            "nested": {"labels_path": r"E:\data\labels\train"},
            "rel_path": "log/detect/train1/results.csv",
        },
        "metrics": {"mAP50": 0.5},
        "artifacts": {"run_dir": run_dir, "report_path": report},
        "audit_path": audit,
        "finished_at": "2026-08-01T00:00:00Z",
    })
    svc.index_experiment({
        "run_id": "tuning:u:at1", "run_name": "at1", "source": "tuning",
        "status": "completed",
        "params": {"data": linux_ds + "/data.yaml", "model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.6},
        "finished_at": "2026-08-02T00:00:00Z",
    })
    return svc


def test_api_raw_json_never_leaks_any_path_form(tmp_path, monkeypatch):
    """返修 1: every local-index HTTP endpoint must return basenames only — no
    absolute Windows/Linux paths, no separator-relative paths, no nested paths
    under params/artifacts/decision/tuning."""
    from auto_tune.ui import app as app_mod

    svc = _seed_leaky_dataset(tmp_path, monkeypatch)
    client = _client()

    windows_ds = r"D:\data\project1\datasets\dsa"
    linux_ds = "/data/projects/dsb"
    run_dir = r"D:\detect\train1"
    report = r"C:\log\train1_report.json"
    audit = r"D:\log\tuning_audit_x.json"
    leaks = [
        windows_ds,
        linux_ds,
        run_dir,
        report,
        audit,
        r"E:\data\labels\train",
        "log/detect/train1/results.csv",
    ]

    responses = {}
    responses["experiments_list"] = client.get("/api/experiments").json()
    ds_list = client.get("/api/datasets").json()
    responses["datasets_list"] = ds_list
    ds_a = next(d for d in ds_list["datasets"] if d["display_name"] == "dsa")
    ds_b = next(d for d in ds_list["datasets"] if d["display_name"] == "dsb")
    responses["dataset_detail"] = client.get(f"/api/datasets/{ds_a['dataset_id']}").json()
    responses["dataset_experiments"] = client.get(
        f"/api/datasets/{ds_a['dataset_id']}/experiments").json()
    responses["experiment_detail"] = client.get("/api/experiments/manual:1").json()
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    resp = client.post("/api/experiments/compare", headers=headers, json={
        "run_ids": ["manual:1", "tuning:u:at1"], "baseline_run_id": "manual:1"})
    assert resp.status_code == 200, resp.text
    responses["compare"] = resp.json()

    blob = json.dumps(responses)
    for path in leaks:
        assert path not in blob, f"path leaked in API response: {path}"
    # Basenames / controlled ids are still available for the UI.
    assert responses["experiments_list"]["items"][0]["params"]["data"] == "data.yaml"
    assert responses["experiment_detail"]["params"]["data"] == "data.yaml"
    assert ds_a["canonical_path"] == "dsa"
    assert ds_a["data_yaml_path"] == "data.yaml"
    # The compare parameter diff excludes path-typed params entirely.
    assert "data" not in json.dumps(responses["compare"]["parameters"])


def test_api_datasets_and_detail_projection_is_recursive(tmp_path, monkeypatch):
    """返修 1: dataset projection redacts canonical/data_yaml paths to basenames,
    and experiment params/artifacts are redacted recursively."""
    svc = _seed_leaky_dataset(tmp_path, monkeypatch)
    client = _client()
    ds_list = client.get("/api/datasets").json()["datasets"]
    for ds in ds_list:
        blob = json.dumps(ds)
        assert "D:\\data" not in blob and "/data/" not in blob
        assert ds["canonical_path"] == ds["display_name"]
    detail = client.get("/api/experiments/manual:1").json()
    blob = json.dumps(detail)
    assert r"E:\data\labels\train" not in blob
    assert detail["params"]["nested"]["labels_path"] == "train"
    assert detail["params"]["rel_path"] == "results.csv"


# ── 返修2: 维护 API 出站投影 — audit/audit-record/rebuild/diagnostics 原始 HTTP JSON ──


def _seed_maintenance_paths(tmp_path, monkeypatch):
    """Seed facts so every maintenance API response carries a path-bearing value:
    a legacy run_name whose run_id embeds an absolute Windows path (audit
    subject), a legacy source file (recent_import.source_path), and snapshot
    issues keyed from a scan (rebuild snapshot_issues)."""
    from auto_tune.ui import app as app_mod

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    win_legacy_name = r"D:\data\proj1\train_legacy"
    (log_dir / "tuning_history.json").write_text(json.dumps([
        {"train_name": win_legacy_name, "result_mAP50": 0.4,
         "timestamp": "2026-08-01T00:00:00Z"},
    ]), encoding="utf-8")
    (log_dir / "experiment_history.json").write_text(json.dumps({
        "schema_version": "1.0", "experiments": [
            {"run_id": "manual:ok", "run_name": "ok", "source": "manual",
             "status": "completed", "params": {"data": r"D:\data\ds1\data.yaml"},
             "metrics": {"mAP50": 0.5}},
        ],
    }), encoding="utf-8")
    return svc, win_legacy_name


def test_maintenance_api_raw_json_never_leaks_any_path(tmp_path, monkeypatch):
    """返修 2: audit / audit-record / rebuild / diagnostics 的原始 HTTP JSON 不得泄漏
    Windows 绝对路径、Linux 绝对路径或相对路径（issues.subject、legacy source_path、
    snapshot_issues、嵌套列表/字典均递归脱敏）。audit 在 import 之前调用，因此
    legacy run_id 含路径的 subject 会出现在 MISSING_IN_INDEX 里。"""
    from auto_tune.ui import app as app_mod

    svc, win_legacy_name = _seed_maintenance_paths(tmp_path, monkeypatch)
    client = _client()
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}

    # Audit BEFORE import: the legacy record is not indexed yet, so its run_id
    # (embedding an absolute Windows path) surfaces as a MISSING_IN_INDEX subject.
    audit = client.get("/api/local-index/audit").json()
    audit_record = client.post(
        "/api/local-index/audit/record", headers=headers, json={}).json()

    imp = client.post("/api/local-index/import-legacy", headers=headers, json={})
    assert imp.status_code == 200, imp.text
    rebuild = client.post(
        "/api/local-index/rebuild", headers=headers, json={}).json()
    diagnostics = client.get("/api/local-index/diagnostics").json()

    blob = json.dumps({"audit": audit, "audit_record": audit_record,
                       "rebuild": rebuild, "diagnostics": diagnostics})
    for leak in (win_legacy_name, r"D:\data\proj1", r"D:\data\ds1",
                 str(tmp_path), "/data/", "\\data\\proj1"):
        assert leak not in blob, f"path leaked in maintenance API: {leak}"
    # The legacy run_id subject survives only as its basename.
    assert "train_legacy" in blob


def test_maintenance_api_subject_and_source_path_are_basenames(tmp_path, monkeypatch):
    """返修 2: audit issue subject containing a path is reduced to a basename;
    diagnostics recent_import.source_path is a basename; rebuild snapshot_issues
    carries only bounded status keys."""
    from auto_tune.ui import app as app_mod

    svc, win_legacy_name = _seed_maintenance_paths(tmp_path, monkeypatch)
    client = _client()
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}

    # Audit before import: legacy run_id subject must be basename-only.
    audit = client.get("/api/local-index/audit").json()
    subjects = [(i.get("subject") or "") for i in audit.get("issues", [])]
    assert subjects, "expected at least one MISSING_IN_INDEX subject"
    for subj in subjects:
        assert "D:" not in subj and "/data/" not in subj, subj
        if "legacy-tuning" in subj:
            assert subj.endswith(os.path.basename(win_legacy_name)), subj

    client.post("/api/local-index/import-legacy", headers=headers, json={})
    rebuild = client.post(
        "/api/local-index/rebuild", headers=headers, json={}).json()
    for key in rebuild.get("snapshot_issues", {}):
        assert key in ("valid", "missing", "corrupt", "too_large",
                       "unreadable", "exceeded"), key

    diagnostics = client.get("/api/local-index/diagnostics").json()
    recent = diagnostics.get("recent_import") or {}
    src = recent.get("source_path")
    if src is not None:
        assert src == os.path.basename(src), src


# ── 服务器级重建：latest_dataset 指向损坏/缺失 manifest 时 POST /rebuild 稳定返回 ──


def _seed_corrupt_latest_dataset(tmp_path, monkeypatch, corrupt=True):
    """latest_dataset.json 引用 dataset_snapshots 下损坏/缺失 manifest 的快照，
    加上引用该快照的实验。返回 (log_dir, data_yaml_of_broken)。"""
    from auto_tune.modules.dataset_snapshot.service import _canonical_json_bytes
    import hashlib

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    snap_dir = log_dir / "dataset_snapshots" / "snapB"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0", "snapshot_id": "snapB-id",
        "created_at": "2026-08-01T00:00:00Z", "source_root": str(log_dir / "s" / "snapB"),
        "samples": [], "train_count": 0, "val_count": 0, "background_count": 0,
    }
    digest_payload = {k: v for k, v in manifest.items()
                      if k not in ("created_at", "source_root", "manifest_digest")}
    manifest["manifest_digest"] = (
        "wrong-digest" if corrupt
        else hashlib.sha256(_canonical_json_bytes(digest_payload)).hexdigest()
    )
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (snap_dir / "data.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    data_b = str(snap_dir / "data.yaml")
    # latest_dataset.json references the broken snapshot with a snapshot_id.
    (log_dir / "latest_dataset.json").write_text(json.dumps({
        "dataset_path": str(snap_dir),
        "data_yaml_path": data_b,
        "snapshot_id": "snapB-id",
        "snapshot_valid": True,
    }), encoding="utf-8")
    (log_dir / "experiment_history.json").write_text(json.dumps({
        "schema_version": "1.0", "experiments": [
            {"run_id": "manual:b", "run_name": "b", "source": "manual",
             "status": "completed", "params": {"data": data_b},
             "metrics": {"mAP50": 0.5}},
        ],
    }), encoding="utf-8")
    return log_dir, data_b


@pytest.mark.parametrize("corrupt", [True, False])
def test_server_rebuild_damaged_latest_dataset_manifest(tmp_path, monkeypatch, corrupt):
    """服务器级 POST /api/local-index/rebuild：latest_dataset 指向损坏(corrupt=True)
    或有效(corrupt=False) manifest 时返回稳定结果；损坏时不建立该 dataset 且
    snapshot_unresolved 计数正确，且原始 JSON 不泄漏临时路径。"""
    from auto_tune.ui import app as app_mod

    log_dir, data_b = _seed_corrupt_latest_dataset(tmp_path, monkeypatch, corrupt=corrupt)
    client = _client()
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}

    resp = client.post("/api/local-index/rebuild", headers=headers, json={})
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["error_code"] is None
    if corrupt:
        assert result["snapshot_unresolved"] == 1
        assert result["snapshot_issues"].get("corrupt") == 1
    else:
        assert result["snapshot_unresolved"] == 0
        assert result["snapshot_issues"].get("valid") == 1
    # Server JSON never leaks the temporary log path.
    blob = json.dumps(result)
    assert str(log_dir) not in blob

    # Rebuild is idempotent at the HTTP layer.
    resp2 = client.post("/api/local-index/rebuild", headers=headers, json={})
    assert resp2.status_code == 200
    assert resp2.json()["snapshot_unresolved"] == result["snapshot_unresolved"]
    assert resp2.json()["snapshot_issues"] == result["snapshot_issues"]
