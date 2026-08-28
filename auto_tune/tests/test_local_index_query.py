"""Studio S2.3: search, whitelisted sorting, stable pagination, diagnostics.

The query API is parameterized end-to-end (no SQL fragments from the client),
returns a stable ``{items,total,limit,offset}`` page, and diagnostics + the
read/write maintenance endpoints (checkpoint/backup) never allow clearing or
deleting history.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.local_index.models import (
    ExperimentQuery,
    LocalIndexConfig,
    LocalIndexPersistenceError,
)
from auto_tune.modules.local_index.service import LocalIndexService


def _cfg(tmp_path, **kw):
    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 200),
    )


def _service(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    return svc


def _exp(run_id, source="manual", status="completed", run_name=None, model="yolov8n.pt",
         metrics=None, finished_at="2026-08-01T00:00:00Z", data_yaml=None):
    rec = {
        "run_id": run_id,
        "run_name": run_name or run_id.rsplit(":", 1)[-1],
        "source": source,
        "status": status,
        "analysis_status": "completed",
        "metrics": dict(metrics or {}),
        "params": {"model": model},
        "finished_at": finished_at,
    }
    if data_yaml:
        rec["params"]["data"] = data_yaml
    return rec


def _seed(svc, count=5):
    for i in range(count):
        svc.index_experiment(_exp(
            f"manual:train_{i}",
            run_name=f"train_{i}",
            metrics={"mAP50": 0.1 * i, "mAP50_95": 0.05 * i},
            finished_at=f"2026-08-0{i + 1}T00:00:00Z",
        ))


# ── ExperimentQuery validation ──


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 101}, {"limit": True}, {"offset": -1}, {"offset": True},
    {"source": "bogus"}, {"status": "bogus"}, {"sort": "id"}, {"sort": "mAP"},
    {"order": "sideways"}, {"search": 5},
])
def test_experiment_query_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        ExperimentQuery(**kwargs)


def test_experiment_query_defaults():
    q = ExperimentQuery()
    assert q.limit == 25
    assert q.offset == 0
    assert q.sort == "finished_at"
    assert q.order == "desc"


def test_experiment_query_accepts_whitelisted_sort():
    for field in ("finished_at", "started_at", "updated_at", "name", "mAP50",
                  "mAP50_95", "precision", "recall"):
        assert ExperimentQuery(sort=field).sort == field


# ── search ──


def test_search_matches_run_name_and_model(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", run_name="defect_v1", model="yolov8s.pt"))
    svc.index_experiment(_exp("manual:b", run_name="other", model="yolov8n.pt"))

    rows = svc.query_experiments(ExperimentQuery(search="defect"))["items"]
    assert [r["run_id"] for r in rows] == ["manual:a"]
    rows = svc.query_experiments(ExperimentQuery(search="yolov8s"))["items"]
    assert [r["run_id"] for r in rows] == ["manual:a"]


def test_search_combines_with_filters(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:a", run_name="defect", source="manual"))
    svc.index_experiment(_exp("tuning:u1:at1", run_name="defect", source="tuning"))
    svc.index_experiment(_exp("manual:b", run_name="other"))

    rows = svc.query_experiments(ExperimentQuery(search="defect", source="tuning"))["items"]
    assert [r["run_id"] for r in rows] == ["tuning:u1:at1"]


def test_search_like_wildcards_escaped(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:lit", run_name="abc%def"))
    svc.index_experiment(_exp("manual:other", run_name="abcdef"))

    rows = svc.query_experiments(ExperimentQuery(search="abc%def"))["items"]
    assert [r["run_id"] for r in rows] == ["manual:lit"]


def test_search_underscore_escaped(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:under", run_name="a_b"))
    svc.index_experiment(_exp("manual:no", run_name="aXb"))

    rows = svc.query_experiments(ExperimentQuery(search="a_b"))["items"]
    assert [r["run_id"] for r in rows] == ["manual:under"]


# ── sort / order ──


def test_sort_by_name_asc(tmp_path):
    svc = _service(tmp_path)
    _seed(svc)
    rows = svc.query_experiments(ExperimentQuery(sort="name", order="asc"))["items"]
    names = [r["run_name"] for r in rows]
    assert names == sorted(names)


def test_sort_by_map50_desc(tmp_path):
    svc = _service(tmp_path)
    _seed(svc)
    rows = svc.query_experiments(ExperimentQuery(sort="mAP50", order="desc"))["items"]
    values = [r["metrics"]["mAP50"] for r in rows]
    assert values == sorted(values, reverse=True)


def test_sort_deterministic_tiebreak(tmp_path):
    svc = _service(tmp_path)
    svc.index_experiment(_exp("manual:b", run_name="b", finished_at="2026-08-01T00:00:00Z"))
    svc.index_experiment(_exp("manual:a", run_name="a", finished_at="2026-08-01T00:00:00Z"))
    rows = svc.query_experiments(ExperimentQuery(sort="finished_at", order="asc"))["items"]
    assert [r["run_id"] for r in rows] == ["manual:a", "manual:b"]


# ── pagination ──


def test_pagination_returns_stable_page(tmp_path):
    svc = _service(tmp_path)
    _seed(svc, count=30)
    page = svc.query_experiments(ExperimentQuery(limit=25, offset=0))

    assert page["total"] == 30
    assert page["limit"] == 25
    assert page["offset"] == 0
    assert len(page["items"]) == 25

    second = svc.query_experiments(ExperimentQuery(limit=25, offset=25))
    assert len(second["items"]) == 5
    assert second["total"] == 30
    # Pages are disjoint.
    ids1 = {r["run_id"] for r in page["items"]}
    ids2 = {r["run_id"] for r in second["items"]}
    assert not (ids1 & ids2)
    assert ids1 | ids2 == {f"manual:train_{i}" for i in range(30)}


def test_pagination_offset_beyond_total_returns_empty(tmp_path):
    svc = _service(tmp_path)
    _seed(svc, count=3)
    page = svc.query_experiments(ExperimentQuery(limit=25, offset=100))
    assert page["items"] == []
    assert page["total"] == 3


# ── diagnostics ──


def test_diagnostics_reports_health(tmp_path):
    svc = _service(tmp_path)
    _seed(svc, count=4)
    diag = svc.diagnostics()

    assert diag["available"] is True
    assert diag["schema_version"] == 2
    assert diag["experiment_count"] == 4
    assert diag["quick_check"] == "ok"
    assert diag["database_size_bytes"] > 0
    assert diag["backup_count"] == 0


def test_diagnostics_reports_recent_rebuild_event(tmp_path):
    svc = _service(tmp_path)
    from auto_tune.modules.local_index.reconciliation import rebuild_index
    rebuild_index(svc.config, log_dir=str(tmp_path))
    diag = svc.diagnostics()

    assert diag["available"] is True
    assert any(e["kind"] == "rebuild" for e in diag["recent_events"])


def test_diagnostics_corrupt_db_stable(tmp_path):
    svc = _service(tmp_path)
    svc.initialize()
    Path(svc.config.database_path).write_bytes(b"\x00\x01\x02 not sqlite " * 8)
    diag = svc.diagnostics()

    assert diag["available"] is False
    assert diag["error_code"] == "LOCAL_INDEX_CORRUPT"
    blob = json.dumps(diag)
    assert str(tmp_path) not in blob


def test_diagnostics_no_clear_or_delete_contract():
    from auto_tune.ui import app as app_mod

    for route in app_mod.app.routes:
        path = getattr(route, "path", "")
        if "local-index" in path or "experiments" in path:
            assert "clear" not in path.lower()
            assert "delete" not in path.lower()


# ── checkpoint / backup API ──


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
    from auto_tune.ui import app as app_mod

    return TestClient(app_mod.app)


def test_checkpoint_requires_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/local-index/checkpoint", json={})
    assert resp.status_code == 403


def test_checkpoint_with_csrf(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    resp = _client().post("/api/local-index/checkpoint", headers=headers, json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_backup_requires_csrf(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().post("/api/local-index/backup", json={})
    assert resp.status_code == 403


def test_backup_with_csrf_creates_backup(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    resp = _client().post("/api/local-index/backup", headers=headers, json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    backups = list((log_dir / "db_backups").glob("auto_tune.db.manual.*.bak"))
    assert len(backups) == 1


# ── API: paginated experiments + diagnostics ──


def test_api_experiments_paginated_shape(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    _seed(svc, count=30)

    resp = _client().get("/api/experiments")

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"items", "total", "limit", "offset"}
    assert data["limit"] == 25
    assert data["total"] == 30
    assert len(data["items"]) == 25


def test_api_experiments_search_and_sort(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    _seed(svc, count=3)

    resp = _client().get("/api/experiments?search=train_2&sort=name&order=asc")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1


@pytest.mark.parametrize("query", [
    "limit=0", "limit=101", "offset=-1", "sort=id",
    "order=up", "source=bogus", "status=bogus",
])
def test_api_experiments_invalid_query_stable_400(tmp_path, monkeypatch, query):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get(f"/api/experiments?{query}")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_api_experiments_sort_injection_rejected(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get("/api/experiments?sort=name%3BDROP%20TABLE%20experiments")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_api_diagnostics_route(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get("/api/local-index/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) >= {"schema_version", "database_size_bytes", "quick_check",
                         "experiment_count", "backup_count", "recent_events"}
