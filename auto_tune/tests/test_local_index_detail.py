"""Studio S2.2: experiment detail, dataset association and artifact availability.

The detail projection shows facts (identity/source/status/times/model/task/
params/metrics/tuning/error) plus a controlled artifact manifest. The manifest
never leaks full business paths: it returns kind/name/status only, with honest
exists/missing/unregistered/unavailable states.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.local_index.detail import (
    _manifest_state,
    _path_state,
    build_artifact_manifest,
    build_dataset_experiments,
    build_experiment_detail,
)
from auto_tune.modules.local_index.models import ExperimentQuery, LocalIndexConfig
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


def _seed_dataset(svc, ds_path, snapshot_id="snap1", data_yaml=None):
    return svc.index_dataset({
        "source_dataset_path": str(ds_path),
        "data_yaml_path": data_yaml or os.path.join(str(ds_path), "data.yaml"),
        "snapshot_id": snapshot_id,
        "snapshot_valid": True,
    })


def _seed_experiment(svc, run_id, source="manual", status="completed", metrics=None,
                     run_dir=None, report_path=None, audit_path=None, data_yaml=None,
                     params=None, tuning=None, decision=None, probe_decision=None,
                     runtime_run_id=None, finished_at="2026-08-01T00:00:00Z",
                     model_name="yolov8n.pt", task_type="detect"):
    rec = {
        "run_id": run_id,
        "run_name": run_id.rsplit(":", 1)[-1],
        "source": source,
        "status": status,
        "analysis_status": "completed",
        "metrics": dict(metrics or {}),
        "params": dict(params or {}),
        "finished_at": finished_at,
        "artifacts": {"run_dir": run_dir, "report_path": report_path},
        "audit_path": audit_path,
    }
    if data_yaml:
        rec["params"]["data"] = data_yaml
    if task_type:
        rec["params"].setdefault("task", task_type)
    if model_name and "model" not in rec["params"]:
        rec["params"]["model"] = model_name
    if tuning is not None:
        rec["tuning"] = tuning
    if decision is not None:
        rec["decision"] = decision
    if probe_decision is not None:
        rec["probe_decision"] = probe_decision
    return svc.index_experiment(rec, runtime_run_id=runtime_run_id)


def _make_run_dir(tmp_path, name, files=()):
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        path = run_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return str(run_dir)


def _yolo_run_dir(tmp_path, name, *, results_csv=True, args_yaml=True,
                  best_pt=False, last_pt=False, weights_dir=True):
    """Standard YOLO Detect run directory (weights live under ``weights/``)."""
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if results_csv:
        (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    if args_yaml:
        (run_dir / "args.yaml").write_text("epochs: 2\n", encoding="utf-8")
    if best_pt or last_pt or weights_dir:
        (run_dir / "weights").mkdir(exist_ok=True)
    if best_pt:
        (run_dir / "weights" / "best.pt").write_bytes(b"best")
    if last_pt:
        (run_dir / "weights" / "last.pt").write_bytes(b"last")
    return str(run_dir)


def _manifest_of(svc, run_id="manual:train1"):
    detail = svc.get_experiment_detail(run_id)
    return {a["kind"]: a for a in detail["artifacts"]}


# ─────────────────────────────────────────────────────────────
# detail projection
# ─────────────────────────────────────────────────────────────


def test_detail_manual_facts(tmp_path):
    svc = _service(tmp_path)
    _seed_experiment(svc, "manual:train1", run_dir=_make_run_dir(tmp_path, "train1"),
                     metrics={"mAP50": 0.5}, params={"model": "yolov8n.pt", "epochs": 100})

    detail = svc.get_experiment_detail("manual:train1")

    assert detail["run_id"] == "manual:train1"
    assert detail["source"] == "manual"
    assert detail["status"] == "completed"
    assert detail["model_name"] == "yolov8n.pt"
    assert detail["task_type"] == "detect"
    assert detail["metrics"]["mAP50"] == 0.5
    assert detail["params"]["epochs"] == 100
    assert detail["tuning"] is None


def test_detail_tuning_facts(tmp_path):
    svc = _service(tmp_path)
    audit = tmp_path / "log" / "tuning_audit_s1.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("{}", encoding="utf-8")
    tuning = {"guardrails": {"valid": True}, "action": "keep"}
    _seed_experiment(svc, "tuning:uuid1", source="tuning", audit_path=str(audit),
                     tuning=tuning, decision={"diagnosis": "overfit"},
                     probe_decision={"verdict": "continue"})

    detail = svc.get_experiment_detail("tuning:uuid1")

    assert detail["tuning"]["decision"]["diagnosis"] == "overfit"
    assert detail["tuning"]["probe_decision"]["verdict"] == "continue"
    assert detail["tuning"]["guardrails"]["valid"] is True
    assert detail["tuning"]["audit_filename"] == "tuning_audit_s1.json"


def test_detail_dataset_association(tmp_path):
    svc = _service(tmp_path)
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    _seed_dataset(svc, ds, snapshot_id="snap9", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:train1", data_yaml=data_yaml)

    detail = svc.get_experiment_detail("manual:train1")

    assert detail["dataset"] is not None
    assert detail["dataset"]["snapshot_id"] == "snap9"
    assert detail["dataset"]["display_name"] == "ds"
    assert "canonical_path" not in detail["dataset"]


def test_detail_no_full_path_leak(tmp_path):
    svc = _service(tmp_path)
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    audit = tmp_path / "log" / "tuning_audit_s1.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("{}", encoding="utf-8")
    run_dir = _make_run_dir(tmp_path, "train1", files=("results.csv", "args.yaml"))
    report = tmp_path / "log" / "train1_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    _seed_experiment(svc, "tuning:uuid1", source="tuning", run_dir=run_dir,
                     report_path=str(report), audit_path=str(audit), data_yaml=data_yaml)

    detail = svc.get_experiment_detail("tuning:uuid1")
    blob = json.dumps(detail)

    assert str(tmp_path) not in blob
    assert str(run_dir) not in blob
    assert str(report) not in blob
    assert str(audit) not in blob
    assert detail["params"].get("data") == "data.yaml"


def test_detail_artifact_manifest_statuses(tmp_path):
    svc = _service(tmp_path)
    # 标准 YOLO 目录：results.csv/args.yaml 在根，权重在 weights/ 子目录
    run_dir = _yolo_run_dir(tmp_path, "train1", best_pt=True)
    report = tmp_path / "log" / "train1_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    _seed_experiment(svc, "manual:train1", run_dir=run_dir, report_path=str(report))

    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}

    assert by_kind["report"]["status"] == "exists"
    assert by_kind["report"]["name"] == "train1_report.json"
    assert by_kind["run_dir"]["status"] == "exists"
    assert by_kind["results_csv"]["status"] == "exists"
    assert by_kind["args_yaml"]["status"] == "exists"
    assert by_kind["best_pt"]["status"] == "exists"
    assert by_kind["last_pt"]["status"] == "missing"


# ── P2 返修：权重必须从受控 run_dir 的 weights/ 子目录判断 ─────────────
#
# 真实 YOLO Detect 运行把权重放在 ``run_dir/weights/``，而 results.csv 与
# args.yaml 在 ``run_dir`` 根。修复前四项统一按根目录文件判断，导致真实存在
# 的 best.pt/last.pt 被标为 missing。以下反例用 tmp_path 构造真实目录结构。


def test_manifest_finds_weights_under_the_yolo_weights_dir(tmp_path):
    svc = _service(tmp_path)
    run_dir = _yolo_run_dir(tmp_path, "train1", best_pt=True, last_pt=True)
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)

    by_kind = _manifest_of(svc)
    assert by_kind["results_csv"]["status"] == "exists"
    assert by_kind["args_yaml"]["status"] == "exists"
    assert by_kind["best_pt"]["status"] == "exists"
    assert by_kind["last_pt"]["status"] == "exists"
    assert by_kind["best_pt"]["name"] == "best.pt"
    assert by_kind["last_pt"]["name"] == "last.pt"


def test_manifest_reports_only_the_missing_weight(tmp_path):
    svc = _service(tmp_path)
    run_dir = _yolo_run_dir(tmp_path, "train1", best_pt=True, last_pt=False)
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)

    by_kind = _manifest_of(svc)
    assert by_kind["best_pt"]["status"] == "exists"
    assert by_kind["last_pt"]["status"] == "missing"


def test_manifest_reports_both_weights_missing_without_a_weights_dir(tmp_path):
    svc = _service(tmp_path)
    run_dir = _yolo_run_dir(tmp_path, "train1", weights_dir=False)
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)

    by_kind = _manifest_of(svc)
    assert by_kind["results_csv"]["status"] == "exists"
    assert by_kind["args_yaml"]["status"] == "exists"
    assert by_kind["best_pt"]["status"] == "missing"
    assert by_kind["last_pt"]["status"] == "missing"


def test_manifest_keeps_unregistered_without_a_registered_run_dir(tmp_path):
    svc = _service(tmp_path)
    _seed_experiment(svc, "manual:train1")

    by_kind = _manifest_of(svc)
    for kind in ("results_csv", "args_yaml", "best_pt", "last_pt"):
        assert by_kind[kind]["status"] == "unregistered", kind


def test_manifest_weights_error_is_unavailable_and_isolated(tmp_path, monkeypatch):
    """读取权重失败只影响该项，其他产物状态不被改变。"""
    svc = _service(tmp_path)
    run_dir = _yolo_run_dir(tmp_path, "train1", best_pt=True, last_pt=True)
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)
    from auto_tune.modules.local_index import detail as detail_mod

    real_exists = os.path.exists

    def fake_exists(path):
        if os.path.basename(str(path)) in ("best.pt", "last.pt"):
            raise OSError("permission denied")
        return real_exists(path)

    monkeypatch.setattr(detail_mod.os.path, "exists", fake_exists)
    by_kind = _manifest_of(svc)
    assert by_kind["best_pt"]["status"] == "unavailable"
    assert by_kind["last_pt"]["status"] == "unavailable"
    assert by_kind["results_csv"]["status"] == "exists"
    assert by_kind["args_yaml"]["status"] == "exists"


def test_manifest_weight_projection_never_leaks_paths(tmp_path):
    svc = _service(tmp_path)
    run_dir = _yolo_run_dir(tmp_path, "train1", best_pt=True, last_pt=True)
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)

    detail = svc.get_experiment_detail("manual:train1")
    blob = json.dumps(detail, ensure_ascii=False)
    assert run_dir not in blob
    assert str(tmp_path) not in blob
    assert "weights" not in blob
    for artifact in detail["artifacts"]:
        assert set(artifact) == {"kind", "name", "status"}


def test_detail_artifact_unregistered(tmp_path):
    svc = _service(tmp_path)
    _seed_experiment(svc, "manual:train1")

    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}

    assert by_kind["run_dir"]["status"] == "unregistered"
    assert by_kind["results_csv"]["status"] == "unregistered"
    assert by_kind["args_yaml"]["status"] == "unregistered"
    assert by_kind["audit"]["status"] == "unregistered"


def test_detail_artifact_unavailable(tmp_path, monkeypatch):
    svc = _service(tmp_path)
    run_dir = _make_run_dir(tmp_path, "train1", files=("results.csv",))
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)
    from auto_tune.modules.local_index import detail as detail_mod

    real_exists = os.path.exists

    def fake_exists(path):
        if str(path).endswith("results.csv"):
            raise OSError("permission denied")
        return real_exists(path)

    monkeypatch.setattr(detail_mod.os.path, "exists", fake_exists)
    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}
    assert by_kind["results_csv"]["status"] == "unavailable"


def test_detail_snapshot_manifest_status(tmp_path):
    svc = _service(tmp_path)
    snap_dir = tmp_path / "snapshots" / "snap1"
    snap_dir.mkdir(parents=True)
    (snap_dir / "manifest.json").write_text("{}", encoding="utf-8")
    data_yaml = os.path.join(str(snap_dir), "data.yaml")
    _seed_dataset(svc, snap_dir, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:train1", data_yaml=data_yaml)

    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}
    assert by_kind["manifest"]["status"] == "exists"

    # Remove the manifest -> honest missing.
    (snap_dir / "manifest.json").unlink()
    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}
    assert by_kind["manifest"]["status"] == "missing"


def test_detail_snapshot_manifest_unregistered_for_plain_dir(tmp_path):
    svc = _service(tmp_path)
    ds = tmp_path / "plain_ds"
    ds.mkdir(parents=True)
    data_yaml = os.path.join(str(ds), "data.yaml")
    _seed_dataset(svc, ds, snapshot_id=None, data_yaml=data_yaml)
    _seed_experiment(svc, "manual:train1", data_yaml=data_yaml)

    detail = svc.get_experiment_detail("manual:train1")
    by_kind = {a["kind"]: a for a in detail["artifacts"]}
    assert by_kind["manifest"]["status"] == "unregistered"


def test_detail_does_not_read_arbitrary_path(tmp_path):
    svc = _service(tmp_path)
    # A fabricated run_dir that does not exist must surface honest statuses,
    # never an arbitrary read or a crash.
    _seed_experiment(svc, "manual:train1", run_dir=str(tmp_path / "nope"))

    detail = svc.get_experiment_detail("manual:train1")

    assert detail is not None
    by_kind = {a["kind"]: a for a in detail["artifacts"]}
    assert by_kind["run_dir"]["status"] == "missing"
    assert by_kind["results_csv"]["status"] == "missing"


def test_detail_not_found(tmp_path):
    svc = _service(tmp_path)
    assert svc.get_experiment_detail("manual:nope") is None


# ─────────────────────────────────────────────────────────────
# dataset association summary
# ─────────────────────────────────────────────────────────────


def test_dataset_experiments_summary(tmp_path):
    svc = _service(tmp_path)
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    dataset = _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:a", data_yaml=data_yaml, metrics={"mAP50": 0.4},
                     finished_at="2026-08-01T00:00:00Z")
    _seed_experiment(svc, "manual:b", data_yaml=data_yaml, metrics={"mAP50": 0.8},
                     finished_at="2026-08-02T00:00:00Z")
    _seed_experiment(svc, "manual:c", data_yaml=data_yaml, metrics={"mAP50": 0.6},
                     finished_at="2026-08-03T00:00:00Z", status="failed")

    summary = svc.get_dataset_experiments(dataset.dataset_id, limit=2)

    assert summary["experiment_count"] == 3
    assert summary["best"]["metric"] == "mAP50"
    assert summary["best"]["value"] == 0.8
    assert summary["best"]["run_id"] == "manual:b"
    assert [r["run_id"] for r in summary["recent_experiments"]] == ["manual:c", "manual:b"]


def test_dataset_experiments_best_requires_completed_with_metric(tmp_path):
    svc = _service(tmp_path)
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    dataset = _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:f1", data_yaml=data_yaml, status="failed")
    _seed_experiment(svc, "manual:nm", data_yaml=data_yaml, metrics={})

    summary = svc.get_dataset_experiments(dataset.dataset_id)

    assert summary["experiment_count"] == 2
    assert summary["best"] is None


def test_dataset_experiments_unknown_dataset(tmp_path):
    svc = _service(tmp_path)
    assert svc.get_dataset_experiments("does-not-exist") is None


def test_dataset_best_mixed_tasks_returns_null_and_grouped(tmp_path):
    """返修 7: Detect + Classify on one dataset must never yield a single mixed
    'best experiment'; best is null and the per-task facts are grouped."""
    svc = _service(tmp_path)
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    dataset = _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:d1", data_yaml=data_yaml, task_type="detect",
                     metrics={"mAP50": 0.9}, finished_at="2026-08-01T00:00:00Z")
    _seed_experiment(svc, "manual:c1", data_yaml=data_yaml, task_type="classify",
                     metrics={"mAP50": 0.7}, finished_at="2026-08-02T00:00:00Z")

    summary = svc.get_dataset_experiments(dataset.dataset_id)

    assert summary["best"] is None
    assert summary["best_by_task"]["detect"]["run_id"] == "manual:d1"
    assert summary["best_by_task"]["classify"]["run_id"] == "manual:c1"


def test_build_dataset_experiments_projection_shape(tmp_path):
    dataset = {"dataset_id": "d1", "display_name": "ds", "snapshot_id": "s1",
               "validation_status": "valid", "last_used_at": "2026-08-01T00:00:00Z",
               "data_yaml_path": "C:/x/data.yaml"}
    experiments = [{
        "run_id": "manual:1", "run_name": "t1", "source": "manual",
        "status": "completed", "finished_at": "2026-08-01T00:00:00Z",
        "metrics": {"mAP50": 0.5}, "model_name": "yolov8n.pt", "task_type": "detect",
    }]
    out = build_dataset_experiments(dataset, experiment_count=5, recent_experiments=experiments, best=None)
    assert out["dataset_id"] == "d1"
    assert "data_yaml_path" not in out
    assert out["experiment_count"] == 5
    assert out["recent_experiments"][0]["mAP50"] == 0.5


def test_path_state_and_manifest_state(tmp_path):
    exists_file = tmp_path / "x.json"
    exists_file.write_text("{}", encoding="utf-8")
    assert _path_state(str(exists_file)) == "exists"
    assert _path_state(str(tmp_path / "missing.json")) == "missing"
    assert _path_state(None) == "unregistered"
    assert _path_state("") == "unregistered"
    plain = {"data_yaml_path": "C:/plain/data.yaml"}
    assert _manifest_state(plain) == "unregistered"
    snapshot = {"data_yaml_path": str(tmp_path / "s" / "data.yaml"), "snapshot_id": "s1"}
    assert _manifest_state(snapshot) == "missing"


def test_build_artifact_manifest_never_leaks_abs_path(tmp_path):
    run_dir = _make_run_dir(tmp_path, "train1", files=("results.csv",))
    manifest = build_artifact_manifest(
        {"run_id": "manual:1", "artifacts": [
            {"kind": "run_dir", "path": run_dir, "exists_state": "exists"},
        ]}, None)
    blob = json.dumps(manifest)
    assert str(tmp_path) not in blob


# ─────────────────────────────────────────────────────────────
# API routes
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
    from auto_tune.ui import app as app_mod

    return TestClient(app_mod.app)


def test_api_experiment_detail_extended(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    run_dir = _make_run_dir(tmp_path, "train1", files=("results.csv", "args.yaml"))
    _seed_experiment(svc, "manual:train1", run_dir=run_dir)

    resp = _client().get("/api/experiments/manual:train1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "manual:train1"
    assert {a["kind"] for a in data["artifacts"]} >= {
        "report", "run_dir", "results_csv", "args_yaml", "best_pt", "last_pt",
    }


def test_api_dataset_experiments_route(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    dataset = _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "manual:train1", data_yaml=data_yaml, metrics={"mAP50": 0.5})

    resp = _client().get(f"/api/datasets/{dataset.dataset_id}/experiments")

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_count"] == 1
    assert data["best"]["value"] == 0.5


def test_api_dataset_experiments_not_found(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get("/api/datasets/nope/experiments")
    assert resp.status_code == 404


def test_api_dataset_experiments_invalid_limit(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    dataset = _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    resp = _client().get(f"/api/datasets/{dataset.dataset_id}/experiments?limit=0")
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "INVALID_QUERY"


def test_api_detail_never_leaks_full_paths(tmp_path, monkeypatch):
    """GET /api/experiments/{run_id} must not leak params.data, report_path,
    run_dir or audit_path full paths (返修 6)."""
    from auto_tune.ui import app as app_mod

    _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    run_dir = _make_run_dir(tmp_path, "train1", files=("results.csv",))
    report = tmp_path / "log" / "train1_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")
    audit = tmp_path / "log" / "tuning_audit_s1.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("{}", encoding="utf-8")
    ds = tmp_path / "ds"
    data_yaml = os.path.join(str(ds), "data.yaml")
    _seed_dataset(svc, ds, snapshot_id="snap1", data_yaml=data_yaml)
    _seed_experiment(svc, "tuning:uuid1", source="tuning", run_dir=run_dir,
                     report_path=str(report), audit_path=str(audit), data_yaml=data_yaml)

    resp = _client().get("/api/experiments/tuning:uuid1")

    assert resp.status_code == 200
    blob = json.dumps(resp.json())
    assert str(run_dir) not in blob
    assert str(report) not in blob
    assert str(audit) not in blob
    assert str(data_yaml) not in blob
    # Basename form is still available for the UI.
    assert resp.json()["params"]["data"] == "data.yaml"
