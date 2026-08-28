"""Bugfix P5: run_id-bound report/audit view API routes.

The two narrow endpoints accept only a run_id path parameter, resolve the
experiment by that exact run_id, and serve the registered report/audit artifact
as a minimal display model. Error mapping is stable (404/409/413/503); original
report/audit fact files are never rewritten; no external service is contacted.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.local_index import (
    LocalIndexConfig,
    LocalIndexPersistenceError,
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


def _client():
    return TestClient(app_mod.app)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _report_payload(run_name="train1"):
    return {
        "module": "train_analyzer", "version": "1.0",
        "analysis_timestamp": "2026-08-01T00:10:00Z",
        "detect_dir": "detect/train1", "total_runs": 1,
        "runs": {
            run_name: {
                "name": run_name,
                "args": {"model": "yolov8n.pt", "epochs": 50, "batch": 16,
                         "imgsz": 640, "optimizer": "AdamW", "lr0": 0.001,
                         "device": "0"},
                "results": {"total_epochs": 50, "best_epoch": 43,
                            "final_metrics": {
                                "metrics/mAP50(B)": 0.8121,
                                "metrics/mAP50-95(B)": 0.3676,
                                "metrics/precision(B)": 0.7769,
                                "metrics/recall(B)": 0.6571}},
                "issues": [{"issue": "overfitting", "severity": "high",
                            "description": "val loss rises"}],
            }
        },
        "summary": {"best_overall_run": run_name},
        "comparison": {"best_run": run_name},
    }


def _audit_payload(session_id="sess1", status="completed", iterations=None,
                   final_summary=None, final_summary_status="generated"):
    return {
        "schema_version": "1.0", "session_id": session_id, "status": status,
        "started_at": "2026-08-01T00:00:00Z", "finished_at": "2026-08-01T02:00:00Z",
        "reference_run": "train52", "reference_dataset": None, "max_retries": 3,
        "iterations": list(iterations or [{
            "iteration": 1, "status": "completed",
            "started_at": "2026-08-01T00:00:00Z", "finished_at": "2026-08-01T01:00:00Z",
            "baseline": {"reference_run": "train52", "params": {}, "metrics": {}},
            "decision": {"raw_response": None, "diagnosis": "overfit",
                         "action": "reduce lr",
                         "hyperparameter_changes": {"lr0": 0.001},
                         "training_overrides": {}},
            "guardrails": {"valid": True, "warnings": [], "errors": [],
                           "clamped": {}, "sanitized_changes": {"lr0": 0.001}},
            "perception": {"status": None},
            "execution": {"actual_params": {"lr0": 0.001, "batch": 16},
                          "args_yaml_path": None, "command": [],
                          "train_name": "autotune_x_iter01"},
            "result": {"before_metrics": {"mAP50": 0.8},
                       "after_metrics": {"mAP50": 0.8316, "mAP50_95": 0.3759,
                                         "precision": 0.8989, "recall": 0.7623},
                       "metric_delta": {"mAP50": 0.0316},
                       "probe": {}, "analysis": None},
            "error": None,
        }]),
        "error": None,
        "final_summary": final_summary,
        "final_summary_status": final_summary_status,
    }


def _seed_manual_report(log_dir, run_name="train1", run_id="manual:train1"):
    svc = app_mod._local_index_service()
    svc.initialize()
    report = _write_json(log_dir / f"{run_name}_report.json", _report_payload(run_name))
    svc.index_experiment({
        "run_id": run_id, "run_name": run_name, "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.8121, "mAP50_95": 0.3676,
                    "precision": 0.7769, "recall": 0.6571},
        "artifacts": {"report_path": report, "run_dir": None},
    }, runtime_run_id=run_id)
    return svc


def _seed_tuning_audit(log_dir, session_id="sess1", run_id="tuning:u1"):
    svc = app_mod._local_index_service()
    svc.initialize()
    audit = _write_json(log_dir / f"tuning_audit_{session_id}.json",
                        _audit_payload(session_id))
    svc.index_experiment({
        "run_id": run_id, "run_name": "autotune_x_iter01", "source": "tuning",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.8316},
        "artifacts": {"report_path": None, "run_dir": None},
        "audit_path": audit,
    }, runtime_run_id=run_id)
    return svc


# ── 27. 只接受 run_id 路径参数 ──


def test_report_view_api_accepts_only_run_id_path_param(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "manual:train1"


def test_audit_view_api_accepts_only_run_id_path_param(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_tuning_audit(log_dir)
    resp = _client().get("/api/experiments/tuning:u1/audit-view")
    assert resp.status_code == 200
    assert resp.json()["session"]["session_id"] == "sess1"


# ── 28. 404/409/413/503 映射稳定 ──


def test_report_view_experiment_not_found(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get("/api/experiments/manual:nope/report-view")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "EXPERIMENT_NOT_FOUND"


def test_audit_view_experiment_not_found(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)
    resp = _client().get("/api/experiments/manual:nope/audit-view")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "EXPERIMENT_NOT_FOUND"


def test_report_view_missing_artifact_404(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:t", "run_name": "t", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": None, "run_dir": None},
    }, runtime_run_id="manual:t")
    resp = _client().get("/api/experiments/manual:t/report-view")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "REPORT_NOT_AVAILABLE"


def test_audit_view_manual_training_404(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:t", "run_name": "t", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": None, "run_dir": None},
    }, runtime_run_id="manual:t")
    resp = _client().get("/api/experiments/manual:t/audit-view")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "AUDIT_NOT_AVAILABLE"


def test_report_view_corrupt_json_409(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    report = log_dir / "train1_report.json"
    report.write_text("{bad", encoding="utf-8")
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:train1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": str(report), "run_dir": None},
    }, runtime_run_id="manual:train1")
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "REPORT_INVALID"


def test_report_view_too_large_413(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    from auto_tune.modules.presentation import experiment_views as views_mod

    monkeypatch.setattr(views_mod, "MAX_ARTIFACT_BYTES", 64)
    report = log_dir / "train1_report.json"
    report.write_text("x" * 200, encoding="utf-8")
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:train1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": str(report), "run_dir": None},
    }, runtime_run_id="manual:train1")
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 413
    assert resp.json()["error_code"] == "ARTIFACT_TOO_LARGE"


def test_report_view_identity_mismatch_409(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    report = _write_json(log_dir / "other_report.json", _report_payload("other"))
    svc = app_mod._local_index_service()
    svc.initialize()
    svc.index_experiment({
        "run_id": "manual:train1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": report, "run_dir": None},
    }, runtime_run_id="manual:train1")
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "ARTIFACT_IDENTITY_MISMATCH"


def test_local_index_unavailable_503(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)

    def broken(*args, **kwargs):
        raise LocalIndexPersistenceError("db down")

    monkeypatch.setattr(app_mod.LocalIndexService, "get_report_view", broken)
    monkeypatch.setattr(app_mod.LocalIndexService, "get_audit_view", broken)
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "LOCAL_INDEX_UNAVAILABLE"
    resp = _client().get("/api/experiments/manual:train1/audit-view")
    assert resp.status_code == 503


# ── 29. API Schema 固定 ──


def test_report_view_api_schema_fixed(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    body = _client().get("/api/experiments/manual:train1/report-view").json()
    assert set(body) == {
        "run_id", "run_name", "source", "status", "task_type", "model_name",
        "dataset", "timing", "metrics", "epochs", "parameters", "issues",
        "ai_analysis", "vision_analysis", "artifacts", "analysis_status",
    }
    assert set(body["timing"]) == {"started_at", "finished_at", "duration_seconds"}
    assert set(body["epochs"]) == {"configured", "completed", "best"}


def test_audit_view_api_schema_fixed(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_tuning_audit(log_dir)
    body = _client().get("/api/experiments/tuning:u1/audit-view").json()
    assert set(body) == {
        "run_id", "session", "reference_dataset", "iterations", "truncated",
        "best_result", "final_summary",
    }
    assert set(body["session"]) == {
        "session_id", "reference_run", "status", "termination_reason",
        "total_iterations", "best_iteration",
    }
    assert set(body["iterations"][0]) == {
        "iteration", "status", "diagnosis", "action", "suggested_parameters",
        "guarded_parameters", "executed_parameters", "guardrails", "metrics",
        "metric_delta", "run_name", "error_code",
    }


# ── 30. 原有 /api/experiments 契约不变 ──


def test_existing_experiments_contract_unchanged(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    resp = _client().get("/api/experiments/manual:train1")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("run_id", "source", "run_name", "status", "analysis_status",
                "params", "metrics", "dataset", "artifacts"):
        assert key in data
    assert data["run_id"] == "manual:train1"


# ── 31. 原始报告和审计文件不被改写 ──


def test_view_api_does_not_rewrite_fact_files(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    _seed_tuning_audit(log_dir)
    report_path = log_dir / "train1_report.json"
    audit_path = log_dir / "tuning_audit_sess1.json"
    report_before = report_path.read_bytes()
    audit_before = audit_path.read_bytes()

    _client().get("/api/experiments/manual:train1/report-view")
    _client().get("/api/experiments/tuning:u1/audit-view")

    assert report_path.read_bytes() == report_before
    assert audit_path.read_bytes() == audit_before


# ── 32. 请求不产生外部网络调用 ──


def test_view_modules_have_no_network_imports():
    from auto_tune.modules.presentation import experiment_views as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "http.client", "aiohttp"):
        assert forbidden not in source


def test_view_api_success_no_network(tmp_path, monkeypatch):
    # The successful view API calls complete without any external client; no
    # network stub is needed and the fact files stay untouched.
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    _seed_tuning_audit(log_dir)
    assert _client().get("/api/experiments/manual:train1/report-view").status_code == 200
    assert _client().get("/api/experiments/tuning:u1/audit-view").status_code == 200


# ── 报告/审计 部分数据在服务端仍可读（不套用数据集 allowed_roots）──


def test_view_api_reads_project_log_root_not_dataset_roots(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_manual_report(log_dir)
    resp = _client().get("/api/experiments/manual:train1/report-view")
    assert resp.status_code == 200
    assert resp.json()["run_name"] == "train1"


# ── 返修：畸形审计 Schema → 409 AUDIT_INVALID（不是 500）──


def _seed_malformed_audit(log_dir, malformed_fields):
    svc = app_mod._local_index_service()
    svc.initialize()
    payload = _audit_payload("sess1")
    for path, value in malformed_fields:
        node = payload
        for seg in path[:-1]:
            node = node[seg]
        node[path[-1]] = value
    audit = _write_json(log_dir / "tuning_audit_sess1.json", payload)
    svc.index_experiment({
        "run_id": "tuning:u1", "run_name": "iter01", "source": "tuning",
        "status": "failed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {},
        "artifacts": {"report_path": None, "run_dir": None},
        "audit_path": audit,
    }, runtime_run_id="tuning:u1")


@pytest.mark.parametrize("path,value", [
    (("iterations", 0, "decision", "hyperparameter_changes"), ["lr0"]),
    (("iterations", 0, "decision", "training_overrides"), "epochs=80"),
    (("iterations", 0, "guardrails", "sanitized_changes"), 42),
    (("iterations", 0, "execution", "actual_params"), "lr0=0.001"),
    (("iterations", 0, "guardrails", "clamped"), ["lr0"]),
    (("iterations", 0, "result", "after_metrics"), "mAP50=0.5"),
    (("iterations", 0, "result", "metric_delta"), 7),
])
def test_api_audit_malformed_field_returns_409(path, value, tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_malformed_audit(log_dir, [(path, value)])
    resp = _client().get("/api/experiments/tuning:u1/audit-view")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "AUDIT_INVALID"


def test_api_audit_warnings_string_returns_409(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_malformed_audit(
        log_dir,
        [((("iterations", 0, "guardrails", "warnings")), "D:/secret/data.yaml")],
    )
    resp = _client().get("/api/experiments/tuning:u1/audit-view")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "AUDIT_INVALID"


def test_api_audit_error_message_never_leaks(tmp_path, monkeypatch):
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    _seed_malformed_audit(log_dir, [(("error",), {
        "stage": "execute", "error_type": "training_failed",
        "message": "failed at D:/secret/data.yaml with api_key=abc\nTraceback ...",
    })])
    resp = _client().get("/api/experiments/tuning:u1/audit-view")
    assert resp.status_code == 200
    blob = json.dumps(resp.json())
    assert "D:/secret/data.yaml" not in blob
    assert "api_key=abc" not in blob
    assert "Traceback" not in blob
    assert resp.json()["session"]["termination_reason"] == "training_failed"
