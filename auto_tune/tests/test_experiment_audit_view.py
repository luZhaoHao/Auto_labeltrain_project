"""Bugfix P5: bounded, read-only, run_id-bound tuning audit view projection.

The audit view projects a stored tuning_audit_*.json into a minimal display
model. Iterations are ordered by their real ``iteration`` field, the three
parameter buckets stay separate, the best round comes from stored facts only,
and more than MAX_AUDIT_ITERATIONS rounds are truncated with ``truncated=True``.
"""

import json
import os
from pathlib import Path

import pytest

from auto_tune.modules.presentation.experiment_views import (
    ArtifactIdentityMismatchError,
    ArtifactTooLargeError,
    ArtifactUnavailableError,
    AuditInvalidError,
    AuditNotAvailableError,
    build_audit_view,
)


def _roots(tmp_path) -> tuple:
    return (str(Path(tmp_path) / "log"),)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _iteration(iteration=1, status="completed", train_name="autotune_x_iter01",
               diagnosis="overfit", action="reduce lr", changes=None, overrides=None,
               sanitized=None, clamped=None, actual=None, metrics=None,
               metric_delta=None, error=None):
    return {
        "iteration": iteration,
        "status": status,
        "started_at": "2026-08-01T00:00:00Z",
        "finished_at": "2026-08-01T01:00:00Z",
        "baseline": {"reference_run": "train52", "params": {}, "metrics": {}},
        "decision": {
            "raw_response": None,
            "diagnosis": diagnosis,
            "action": action,
            "hyperparameter_changes": dict(changes or {"lr0": 0.001}),
            "training_overrides": dict(overrides or {}),
        },
        "guardrails": {
            "valid": True,
            "warnings": ["lr0 clamped"],
            "errors": [],
            "clamped": dict(clamped or {"lr0": 0.001}),
            "sanitized_changes": dict(sanitized or {"lr0": 0.001}),
        },
        "perception": {"status": None},
        "execution": {
            "actual_params": dict(actual or {"lr0": 0.001, "batch": 16}),
            "args_yaml_path": "/secret/detect/autotune_x_iter01/args.yaml",
            "command": ["yolo", "train", "model=yolov8n.pt", "data=/secret/ds/data.yaml"],
            "train_name": train_name,
        },
        "result": {
            "before_metrics": {"mAP50": 0.8},
            "after_metrics": dict(metrics or {"mAP50": 0.8316, "mAP50_95": 0.3759,
                                              "precision": 0.8989, "recall": 0.7623}),
            "metric_delta": dict(metric_delta or {"mAP50": 0.0316}),
            "probe": {"verdict": "continue", "reason": None, "suggestion": None},
            "analysis": None,
        },
        "error": error,
    }


def _audit_payload(session_id="sess1", status="completed", reference_run="train52",
                   iterations=None, reference_dataset=None, final_summary=None,
                   final_summary_status="generated", error=None):
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "status": status,
        "started_at": "2026-08-01T00:00:00Z",
        "finished_at": "2026-08-01T02:00:00Z",
        "reference_run": reference_run,
        "reference_dataset": reference_dataset,
        "max_retries": 3,
        "iterations": list(iterations or [_iteration()]),
        "error": error,
        "final_summary": final_summary,
        "final_summary_status": final_summary_status,
    }


def _experiment(run_id="tuning:u1", run_name="autotune_x_iter01", source="tuning",
                audit_path=None, dataset_id=None, status="completed",
                artifacts=None):
    artifacts = list(artifacts or [])
    if audit_path:
        artifacts.append({"kind": "audit", "path": audit_path, "exists_state": "exists"})
    return {
        "run_id": run_id,
        "run_name": run_name,
        "source": source,
        "status": status,
        "phase": "terminal",
        "model_name": "yolov8n.pt",
        "task_type": "detect",
        "started_at": "2026-08-01T00:00:00Z",
        "finished_at": "2026-08-01T02:00:00Z",
        "updated_at": "2026-08-01T02:00:00Z",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.8316},
        "analysis_status": "completed",
        "error": None,
        "dataset_id": dataset_id,
        "artifacts": artifacts,
    }


# ── 16.5 Q1.1/Q1.2: 1.0 / 1.1 / 1.2 audits all read safely ──


@pytest.mark.parametrize("schema_version", ["1.0", "1.1", "1.2"])
def test_audit_view_accepts_old_and_new_schema(schema_version, tmp_path):
    iteration = _iteration()
    if schema_version in ("1.1", "1.2"):
        iteration["fact_package"] = {
            "schema_version": "1.0", "fact_package_id": "sha256:abc",
            "task": "detect", "reference_run": "train52", "sources": {}, "facts": [],
        }
        iteration["decision_validation"] = {
            "valid": True, "error_code": None, "error_detail": None,
            "retried": False, "referenced_fact_ids": [],
        }
    if schema_version == "1.2":
        iteration["semantic_validation"] = {
            "valid": True, "error_code": None, "reason_code": None,
            "retried": False, "parameters": [],
        }
    payload = _audit_payload("sess1", iterations=[iteration])
    payload["schema_version"] = schema_version
    audit_path = _write(tmp_path / "log" / "tuning_audit_sess1.json", payload)
    view = build_audit_view(
        _experiment(audit_path=audit_path), artifact_roots=_roots(tmp_path)
    )
    assert view["session"]["session_id"] == "sess1"
    assert view["iterations"][0]["suggested_parameters"] == {"lr0": 0.001}


@pytest.mark.parametrize("schema_version", ["1.0", "1.1", "1.2"])
def test_audit_view_old_records_show_no_semantic_validation(schema_version, tmp_path):
    """旧记录没有 semantic_validation 时投影为 None（UI 显示"未执行语义校验"）。"""
    iteration = _iteration()
    if schema_version in ("1.1", "1.2"):
        iteration["fact_package"] = {
            "schema_version": "1.0", "fact_package_id": "sha256:abc",
            "task": "detect", "reference_run": "train52", "sources": {}, "facts": [],
        }
        iteration["decision_validation"] = {
            "valid": True, "error_code": None, "error_detail": None,
            "retried": False, "referenced_fact_ids": [],
        }
    if schema_version == "1.2":
        # 1.2 但缺少 semantic_validation（历史记录）→ 同样显示未执行
        pass
    payload = _audit_payload("sess1", iterations=[iteration])
    payload["schema_version"] = schema_version
    audit_path = _write(tmp_path / "log" / "tuning_audit_sess1.json", payload)
    view = build_audit_view(
        _experiment(audit_path=audit_path), artifact_roots=_roots(tmp_path)
    )
    assert view["iterations"][0]["semantic_validation"] is None


def test_audit_view_projects_semantic_validation_minimally(tmp_path):
    iteration = _iteration()
    iteration["fact_package"] = {
        "schema_version": "1.0", "fact_package_id": "sha256:abc",
        "task": "detect", "reference_run": "train52", "sources": {}, "facts": [],
    }
    iteration["decision_validation"] = {
        "valid": False, "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
        "error_detail": "semantic failed", "retried": True, "referenced_fact_ids": [],
    }
    iteration["semantic_validation"] = {
        "valid": False, "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
        "reason_code": "NO_SUPPORTING_RULE", "retried": True,
        "parameters": [{
            "parameter": "lr0", "current_value": 0.01, "suggested_value": 0.006,
            "change_direction": "decrease", "rule_ids": [],
            "supporting_fact_ids": [], "conflicting_fact_ids": [],
            "neutral_fact_ids": ["training.metrics.mAP50"],
        }],
    }
    payload = _audit_payload("sess1", iterations=[iteration])
    payload["schema_version"] = "1.2"
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", payload)
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    sv = view["iterations"][0]["semantic_validation"]
    assert sv["valid"] is False
    assert sv["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert sv["reason_code"] == "NO_SUPPORTING_RULE"
    assert sv["parameter"] == "lr0"
    # 只投影状态/原因/参数；原始响应、完整事实包、逐参数内部与敏感信息不出现
    blob = json.dumps(view)
    assert "raw_response" not in blob
    assert "fact_package" not in blob
    assert "supporting_fact_ids" not in blob
    assert "api_key" not in blob.lower()
    assert "authorization" not in blob.lower()
    assert "training.metrics.mAP50" not in blob


def test_audit_view_never_leaks_fact_package_sources_paths(tmp_path):
    """A malicious fact_package.sources value must never reach the client."""
    iteration = _iteration()
    iteration["fact_package"] = {
        "schema_version": "1.0", "fact_package_id": "sha256:abc",
        "task": "detect", "reference_run": "train52",
        "sources": {
            "dataset_report": "D:/secret/ds/dataset_report_1.json",
            "training_report": "C:/Users/evil/train52_report.json",
        },
        "facts": [],
    }
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json",
                        _audit_payload("sess1", iterations=[iteration]))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    assert "D:/secret/ds" not in blob
    assert "C:/Users/evil" not in blob
    assert "dataset_report_1.json" not in blob
    assert "train52_report.json" not in blob


# ── 17. tuning run_id 精确绑定 audit ──


def test_audit_view_bound_by_run_id(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload("sess1"))
    exp = _experiment(run_id="tuning:u1", run_name="autotune_x_iter01",
                      audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["run_id"] == "tuning:u1"
    assert view["session"]["session_id"] == "sess1"
    assert view["session"]["reference_run"] == "train52"
    assert view["session"]["status"] == "completed"
    assert view["session"]["total_iterations"] == 1
    assert view["truncated"] is False


# ── 18. 普通训练不提供审计 ──


def test_audit_view_manual_training_raises_not_available(tmp_path):
    exp = _experiment(run_id="manual:train1", run_name="train1", source="manual")

    with pytest.raises(AuditNotAvailableError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


def test_audit_view_tuning_without_audit_artifact_raises_not_available(tmp_path):
    exp = _experiment(run_id="tuning:u1", run_name="autotune_x_iter01",
                      audit_path=None)

    with pytest.raises(AuditNotAvailableError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


# ── 19. 迭代按真实轮次排序 ──


def test_audit_iterations_sorted_by_real_iteration(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1",
        iterations=[_iteration(3, train_name="iter03"),
                    _iteration(1, train_name="iter01"),
                    _iteration(2, train_name="iter02")],
    ))
    exp = _experiment(run_id="tuning:u1", run_name="autotune_x_iter03",
                      audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert [it["iteration"] for it in view["iterations"]] == [1, 2, 3]
    assert [it["run_name"] for it in view["iterations"]] == ["iter01", "iter02", "iter03"]


# ── 20. suggested/guarded/executed 不混淆 ──


def test_audit_parameter_buckets_stay_separate(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1",
        iterations=[_iteration(
            changes={"lr0": 0.001, "batch": 32},
            overrides={"epochs": 80},
            sanitized={"lr0": 0.0005, "batch": 16},
            actual={"lr0": 0.0005, "batch": 16, "imgsz": 640},
        )],
    ))
    exp = _experiment(run_id="tuning:u1", run_name="autotune_x_iter01",
                      audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    it = view["iterations"][0]

    assert it["suggested_parameters"] == {"lr0": 0.001, "batch": 32, "epochs": 80}
    assert it["guarded_parameters"] == {"lr0": 0.0005, "batch": 16}
    assert it["executed_parameters"] == {"lr0": 0.0005, "batch": 16, "imgsz": 640}
    # Buckets never share a value that the audit did not record.
    assert it["suggested_parameters"].get("imgsz") is None


# ── 21. 缺失参数不猜测 ──


def test_audit_missing_parameter_buckets_are_empty_not_guessed(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1",
        iterations=[{
            "iteration": 1,
            "status": "failed",
            "started_at": "2026-08-01T00:00:00Z",
            "finished_at": "2026-08-01T01:00:00Z",
            "baseline": {"reference_run": "train52", "params": {}, "metrics": {}},
            "decision": {"raw_response": None, "diagnosis": None, "action": None,
                         "hyperparameter_changes": {}, "training_overrides": {}},
            "guardrails": {"valid": None, "warnings": [], "errors": [],
                           "clamped": {}, "sanitized_changes": {}},
            "perception": {"status": None},
            "execution": {"actual_params": {}, "args_yaml_path": None,
                          "command": [], "train_name": "iter01"},
            "result": {"before_metrics": {}, "after_metrics": {},
                       "metric_delta": {}, "probe": {}, "analysis": None},
            "error": {"stage": "execute", "error_type": "training_failed",
                      "message": "boom", "fatal": True, "timestamp": "2026-08-01T01:00:00Z"},
        }],
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    it = view["iterations"][0]

    assert it["suggested_parameters"] == {}
    assert it["guarded_parameters"] == {}
    assert it["executed_parameters"] == {}
    assert it["metrics"] == {}
    assert it["metric_delta"] == {}
    assert it["error_code"] == "training_failed"


# ── 22. 失败、取消、部分完成状态保留 ──


def test_audit_preserves_failed_and_cancelled_states(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1", status="cancelled",
        iterations=[_iteration(1, status="completed", train_name="iter01"),
                    _iteration(2, status="failed", train_name="iter02",
                               error={"stage": "execute", "error_type": "training_failed",
                                      "message": "x", "fatal": True,
                                      "timestamp": "2026-08-01T01:00:00Z"})],
        final_summary=None, final_summary_status="skipped",
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter02", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["session"]["status"] == "cancelled"
    assert [it["status"] for it in view["iterations"]] == ["completed", "failed"]
    assert view["iterations"][1]["error_code"] == "training_failed"
    assert view["final_summary"]["status"] == "skipped"


# ── 23. 最佳轮次不重新计算 ──


def test_audit_best_iteration_comes_from_stored_facts(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1",
        iterations=[_iteration(1, train_name="iter01", metrics={"mAP50": 0.9}),
                    _iteration(2, train_name="iter02", metrics={"mAP50": 0.8})],
        final_summary={
            "best_iteration": 2,
            "best_train_name": "iter02",
            "best_metrics": {"mAP50": 0.8, "mAP50_95": 0.3, "precision": 0.7, "recall": 0.6},
        },
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter02", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    # Even though iteration 1 has higher mAP50, the stored best fact (2) wins.
    assert view["best_result"]["iteration"] == 2
    assert view["best_result"]["run_name"] == "iter02"
    assert view["best_result"]["metrics"]["mAP50"] == 0.8


def test_audit_best_iteration_none_when_not_stored(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1", final_summary=None, final_summary_status="skipped",
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["best_result"] is None


# ── 24. 超过 50 轮截断并标记 ──


def test_audit_truncates_past_limit(tmp_path, monkeypatch):
    from auto_tune.modules.presentation import experiment_views as views_mod

    monkeypatch.setattr(views_mod, "MAX_AUDIT_ITERATIONS", 2)
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1", iterations=[_iteration(i, train_name=f"iter{i:02d}")
                             for i in range(1, 5)],
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter04", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["truncated"] is True
    assert len(view["iterations"]) == 2
    assert [it["iteration"] for it in view["iterations"]] == [1, 2]
    assert view["session"]["total_iterations"] == 4


# ── 25. 损坏、过大、缺失、身份冲突稳定失败 ──


def test_audit_corrupt_json_invalid(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = str(log_dir / "tuning_audit_sess1.json")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "tuning_audit_sess1.json").write_text("not json", encoding="utf-8")
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    with pytest.raises(Exception) as excinfo:
        build_audit_view(exp, artifact_roots=_roots(tmp_path))
    assert getattr(excinfo.value, "error_code", "") == "AUDIT_INVALID"


def test_audit_too_large(tmp_path, monkeypatch):
    from auto_tune.modules.presentation import experiment_views as views_mod

    monkeypatch.setattr(views_mod, "MAX_ARTIFACT_BYTES", 64)
    log_dir = Path(tmp_path) / "log"
    audit_path = str(log_dir / "tuning_audit_sess1.json")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "tuning_audit_sess1.json").write_text("x" * 200, encoding="utf-8")
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    with pytest.raises(ArtifactTooLargeError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


def test_audit_missing_file_honest_error(tmp_path):
    log_dir = Path(tmp_path) / "log"
    missing = str(log_dir / "tuning_audit_sess1.json")
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=missing)

    with pytest.raises(ArtifactUnavailableError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


def test_audit_identity_mismatch(tmp_path):
    log_dir = Path(tmp_path) / "log"
    # The artifact filename says "other" but the content session_id is
    # "different": the two cannot be reconciled, so it must be rejected.
    audit_path = _write(log_dir / "tuning_audit_other.json", _audit_payload("different"))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    with pytest.raises(ArtifactIdentityMismatchError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


# ── 26. 不泄漏绝对路径、完整命令、日志或凭据 ──


def test_audit_view_never_leaks_paths_commands_logs_credentials(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload("sess1"))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    assert "/secret/" not in blob
    assert "args_yaml_path" not in blob
    assert "command" not in blob
    assert "raw_response" not in blob
    assert "api_key" not in blob.lower()
    assert "password" not in blob.lower()
    assert "token" not in blob.lower()
    assert "traceback" not in blob.lower()


def test_audit_reference_dataset_projection(tmp_path):
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1", reference_dataset={
            "reference_run": "train52",
            "dataset_id": "ds_abc",
            "snapshot_id": "snap9",
            "data_yaml_path": "/secret/ds/data.yaml",
            "resolution_source": "sqlite",
            "dataset_display_name": "myds",
        },
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path,
                      dataset_id="ds_abc")

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["reference_dataset"]["dataset_id"] == "ds_abc"
    assert view["reference_dataset"]["display_name"] == "myds"
    assert view["reference_dataset"]["snapshot_id"] == "snap9"
    assert view["reference_dataset"]["resolution_source"] == "sqlite"
    assert "/secret/" not in json.dumps(view)
    assert "data_yaml_path" not in json.dumps(view["reference_dataset"])


def test_audit_view_no_framework_dependency():
    from auto_tune.modules.presentation import experiment_views as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("fastapi", "jinja2", "sqlite3"):
        assert forbidden not in source.lower()


# ── 返修：畸形内部 Schema 不得 500 ──

_OBJECT_FIELDS = [
    ("decision", "hyperparameter_changes"),
    ("decision", "training_overrides"),
    ("guardrails", "sanitized_changes"),
    ("guardrails", "clamped"),
    ("execution", "actual_params"),
    ("result", "after_metrics"),
    ("result", "metric_delta"),
]


@pytest.mark.parametrize("path", _OBJECT_FIELDS)
@pytest.mark.parametrize("bad_value", [["lr0"], "lr0=0.001", 42])
def test_audit_malformed_object_field_raises_invalid(path, bad_value, tmp_path):
    it = _iteration()
    node = it
    for seg in path[:-1]:
        node = node[seg]
    node[path[-1]] = bad_value
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json",
                        _audit_payload("sess1", iterations=[it]))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    with pytest.raises(AuditInvalidError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


def test_audit_warnings_string_rejected(tmp_path):
    it = _iteration()
    it["guardrails"]["warnings"] = "D:/secret/data.yaml"
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json",
                        _audit_payload("sess1", iterations=[it]))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    with pytest.raises(AuditInvalidError):
        build_audit_view(exp, artifact_roots=_roots(tmp_path))


def test_audit_legitimate_warnings_still_work(tmp_path):
    it = _iteration()
    it["guardrails"]["warnings"] = ["lr0 clamped to 0.001", "strong augmentation may overfit"]
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json",
                        _audit_payload("sess1", iterations=[it]))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))

    assert view["iterations"][0]["guardrails"]["warnings"] == [
        "lr0 clamped to 0.001", "strong augmentation may overfit"
    ]


def test_audit_warnings_redact_paths_credentials_traceback(tmp_path):
    it = _iteration()
    it["guardrails"]["warnings"] = [
        "path D:/secret/data.yaml used",
        "api_key=abc sent",
        "Traceback (most recent call last): boom",
    ]
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json",
                        _audit_payload("sess1", iterations=[it]))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    assert "D:/secret/data.yaml" not in blob
    assert "api_key=abc" not in blob
    assert "Traceback" not in blob
    assert "secret" not in blob.lower()
    warnings = view["iterations"][0]["guardrails"]["warnings"]
    assert len(warnings) == 3


def test_audit_termination_reason_never_projects_raw_message(tmp_path):
    it = _iteration()
    it["error"] = {
        "stage": "execute", "error_type": "training_failed",
        "message": "failed at D:/secret/data.yaml with api_key=abc\nTraceback ...",
        "fatal": True, "timestamp": "2026-08-01T01:00:00Z",
    }
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1",
        status="failed",
        iterations=[it],
        error={
            "stage": "execute", "error_type": "training_failed",
            "message": "failed at D:/secret/data.yaml with api_key=abc\nTraceback ...",
        },
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    # Only the stable error type is shown; the raw message (with paths,
    # credentials and a traceback) never reaches the client.
    assert view["session"]["termination_reason"] == "training_failed"
    assert "D:/secret/data.yaml" not in blob
    assert "api_key=abc" not in blob
    assert "Traceback" not in blob
    assert "failed at D:/" not in blob


def test_audit_termination_reason_fixed_copy_when_no_stable_code(tmp_path):
    it = _iteration()
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", _audit_payload(
        "sess1", status="failed", iterations=[it],
        error={"stage": "execute", "message": "boom D:/secret/data.yaml", "fatal": True},
    ))
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    # No stable error_type/error_code: a fixed safe copy (or empty) is shown,
    # never the raw message.
    assert view["session"]["termination_reason"] is not None
    assert view["session"]["termination_reason"] != "boom D:/secret/data.yaml"
    assert "D:/secret/data.yaml" not in blob


# ── Q1.2 返修一：decision_attempts 是内部追踪，不进只读视图投影 ─────────────


def test_audit_view_does_not_project_decision_attempts(tmp_path):
    """attempt 细节不进入 API 投影；顶层仍是最终那次结果。"""
    iteration = _iteration()
    iteration["fact_package"] = {
        "schema_version": "1.0", "fact_package_id": "sha256:abc",
        "task": "detect", "reference_run": "train52", "sources": {}, "facts": [],
    }
    # 顶层 = 最终采用结果（第二次 keep_params）
    iteration["decision_validation"] = {
        "valid": True, "error_code": None, "error_detail": None,
        "retried": True, "referenced_fact_ids": [],
    }
    iteration["semantic_validation"] = {
        "valid": True, "error_code": None, "reason_code": None,
        "retried": True, "parameters": [],
    }
    iteration["decision_attempts"] = [
        {
            "attempt": 1, "retried": False,
            "decision": {"schema_version": "1.0", "fact_package_id": "sha256:abc",
                         "diagnosis": "first-attempt-diagnosis", "action": "adjust",
                         "hyperparameter_changes": {"lr0": 0.006},
                         "training_overrides": {}, "evidence_ids": {"lr0": ["training.metrics.mAP50"]}},
            "decision_validation": {"valid": False, "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
                                    "error_detail": "semantic failed", "referenced_fact_ids": []},
            "semantic_validation": {"valid": False, "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
                                    "reason_code": "NO_SUPPORTING_RULE", "parameters": []},
        },
        {
            "attempt": 2, "retried": True,
            "decision": {"schema_version": "1.0", "fact_package_id": "sha256:abc",
                         "diagnosis": "second-attempt-diagnosis", "action": "keep_params",
                         "hyperparameter_changes": {}, "training_overrides": {}, "evidence_ids": {}},
            "decision_validation": {"valid": True, "error_code": None, "error_detail": None,
                                    "referenced_fact_ids": []},
            "semantic_validation": {"valid": True, "error_code": None, "reason_code": None,
                                    "parameters": []},
        },
    ]
    payload = _audit_payload("sess1", iterations=[iteration])
    payload["schema_version"] = "1.2"
    log_dir = Path(tmp_path) / "log"
    audit_path = _write(log_dir / "tuning_audit_sess1.json", payload)
    exp = _experiment(run_id="tuning:u1", run_name="iter01", audit_path=audit_path)

    view = build_audit_view(exp, artifact_roots=_roots(tmp_path))
    blob = json.dumps(view)

    # attempts 不投影，含诊断/证据细节也不泄漏
    assert "decision_attempts" not in blob
    assert "first-attempt-diagnosis" not in blob
    assert "second-attempt-diagnosis" not in blob
    assert "training.metrics.mAP50" not in blob
    assert "0.006" not in blob
    # 顶层仍投影最终结果（只投影状态/原因/参数，不暴露 retried 等内部细节）
    sv = view["iterations"][0]["semantic_validation"]
    assert sv["valid"] is True
    assert sv["error_code"] is None
    assert sv["reason_code"] is None
    assert "retried" not in sv
