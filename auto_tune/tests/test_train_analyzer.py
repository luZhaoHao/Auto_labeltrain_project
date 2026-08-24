"""Tests for train_analyzer orchestrator."""

import os
import yaml
import pytest
from auto_tune.modules.train_analyzer.analyzer import analyze_training_results


CSV_CONTENT = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
    "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
    "1,2.0,3.0,2.5,0.1,0.2,0.05,0.01,2.1,3.1,2.6\n"
    "2,1.5,2.5,2.0,0.2,0.3,0.10,0.03,1.6,2.6,2.1\n"
    "3,1.2,2.0,1.8,0.3,0.4,0.15,0.05,1.3,2.1,1.9\n"
)


def _create_run(tmp_path, name: str):
    """Create a synthetic training run directory."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    # args.yaml
    args = {"model": "yolov8s.yaml", "epochs": 100, "patience": 15,
            "imgsz": [640, 640]}
    with open(run_dir / "args.yaml", "w") as f:
        yaml.dump(args, f)
    # results.csv
    (run_dir / "results.csv").write_text(CSV_CONTENT, encoding="utf-8")
    return str(run_dir)


def test_analyze_training_results_all_runs(tmp_path):
    _create_run(tmp_path, "train")
    _create_run(tmp_path, "train2")

    config = {
        "plateau_epochs": 5,
        "overfit_threshold": 0.15,
        "stale_threshold": 15,
        "min_acceptable_map": 0.5,
        "top_k_runs": 5,
        "compare_metric": "metrics/mAP50(B)",
    }
    result = analyze_training_results(str(tmp_path), config, run_name="__all__")

    assert result["module"] == "train_analyzer"
    assert result["total_runs"] == 2
    assert "train" in result["runs"]
    assert "train2" in result["runs"]
    assert result["comparison"]["total_runs"] == 2
    assert result["summary"]["total_runs_analyzed"] == 2


def test_analyze_training_results_latest_run(tmp_path):
    _create_run(tmp_path, "train")
    import time; time.sleep(0.1)
    _create_run(tmp_path, "train2")

    config = {"min_acceptable_map": 0.5, "stale_threshold": 15}
    result = analyze_training_results(str(tmp_path), config)

    assert result["total_runs"] == 1
    # Should auto-detect train2 as latest
    assert "train2" in result["runs"]
    assert result["comparison"]["total_runs"] == 1


def test_analyze_training_results_specific_run(tmp_path):
    _create_run(tmp_path, "train")
    _create_run(tmp_path, "train2")

    config = {"min_acceptable_map": 0.5, "stale_threshold": 15}
    result = analyze_training_results(str(tmp_path), config, run_name="train")

    assert result["total_runs"] == 1
    assert "train" in result["runs"]
    assert "train2" not in result["runs"]


def test_analyze_training_results_no_runs(tmp_path):
    result = analyze_training_results(str(tmp_path), {})
    assert "error" in result


def test_analyze_training_results_not_found(tmp_path):
    result = analyze_training_results(str(tmp_path / "nonexistent"), {})
    assert "error" in result


def test_analyze_training_results_single_run(tmp_path):
    _create_run(tmp_path, "train")
    config = {"min_acceptable_map": 0.5, "stale_threshold": 15}
    result = analyze_training_results(str(tmp_path), config)
    assert result["total_runs"] == 1
    # Should detect underfitting (mAP still improving) + low map
    issues = result["runs"]["train"].get("issues", [])
    assert len(issues) >= 1


def test_analyze_training_results_output_structure(tmp_path):
    _create_run(tmp_path, "train")
    config = {}
    result = analyze_training_results(str(tmp_path), config)
    assert "module" in result
    assert "version" in result
    assert "analysis_timestamp" in result
    assert "detect_dir" in result
    assert "total_runs" in result
    assert "runs" in result
    assert "comparison" in result
    assert "summary" in result


def test_analyze_training_results_specific_run_not_found(tmp_path):
    result = analyze_training_results(str(tmp_path), {}, run_name="train_ghost")
    assert "error" in result


def test_analyzer_uses_nested_train_analyzer_thresholds(tmp_path):
    """Catches full config being passed where the analyzer expects its subsection."""
    _create_run(tmp_path, "train")
    config = {
        "project": {"name": "test"},
        "train_analyzer": {
            "min_acceptable_map": 0.1,
            "stale_threshold": 15,
            "plateau_epochs": 10,
        },
    }

    result = analyze_training_results(
        str(tmp_path), config, run_name="train", enable_llm=False, enable_vision=False
    )
    issue_types = {issue["type"] for issue in result["runs"]["train"]["issues"]}

    assert "low_final_map" not in issue_types
    assert result["project"]["name"] == "test"


def test_orchestrator_reports_early_stop_using_run_args(tmp_path):
    """Catches detect_early_stopping receiving run_data at the wrong level."""
    _create_run(tmp_path, "train")

    result = analyze_training_results(
        str(tmp_path), {"train_analyzer": {}}, run_name="train",
        enable_llm=False, enable_vision=False,
    )
    early = result["runs"]["train"]["curve_analysis"]["early_stopping"]

    assert early["actual_epochs"] == 3
    assert early["planned_epochs"] == 100
    assert early["stopped_early"] is True


# --- S1.3 security boundary for the text diagnosis LLM call ------------------


class _FakeLLMResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = (
            payload
            if payload is not None
            else {"choices": [{"message": {"content": "ok"}}]}
        )
        self.text = text

    def json(self):
        return self._payload


def _yaml_key_config():
    return {
        "llm": {
            "api_key": "yaml-secret-must-not-be-used",
            "model": "deepseek-v4-flash",
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "allow_private_endpoint": False,
        }
    }


def test_llm_uses_resolved_credential_and_ignores_yaml_key(monkeypatch):
    import auto_tune.modules.train_analyzer.llm_analyzer as llm_module

    captured = {}
    purposes = []

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeLLMResponse()

    monkeypatch.setattr(
        llm_module,
        "resolve_credential",
        lambda purpose: purposes.append(purpose) or "resolved-secret",
    )
    monkeypatch.setattr(
        llm_module,
        "validate_endpoint",
        lambda endpoint, allow_private: "https://resolved.example/v1/chat/completions",
    )
    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    result = llm_module.call_deepseek("prompt", _yaml_key_config())

    assert result == "ok"
    assert purposes == ["text"]
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer resolved-secret"
    assert captured["kwargs"]["allow_redirects"] is False
    assert captured["kwargs"]["timeout"] == (10, 120)
    assert "yaml-secret-must-not-be-used" not in repr(captured)


def test_llm_missing_credential_never_calls_network(monkeypatch):
    import auto_tune.modules.train_analyzer.llm_analyzer as llm_module

    called = []
    monkeypatch.setattr(llm_module, "resolve_credential", lambda purpose: None)
    monkeypatch.setattr(
        llm_module, "validate_endpoint", lambda e, a: called.append("endpoint") or e
    )
    monkeypatch.setattr(
        llm_module.requests, "post", lambda **kwargs: called.append("post") or _FakeLLMResponse()
    )

    with pytest.raises(RuntimeError) as excinfo:
        llm_module.call_deepseek("prompt", _yaml_key_config())

    assert "credential_missing" in str(excinfo.value)
    assert called == []


def test_llm_401_error_is_safe_and_never_leaks_body(monkeypatch):
    import auto_tune.modules.train_analyzer.llm_analyzer as llm_module

    def fake_post(url, **kwargs):
        return _FakeLLMResponse(
            status_code=401,
            payload={"error": {"message": "resolved-secret provider-private-body"}},
            text="provider-private-body raw",
        )

    monkeypatch.setattr(llm_module, "resolve_credential", lambda purpose: "resolved-secret")
    monkeypatch.setattr(
        llm_module, "validate_endpoint", lambda e, a: "https://resolved.example/v1"
    )
    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    with pytest.raises(RuntimeError) as excinfo:
        llm_module.call_deepseek("prompt", _yaml_key_config())

    message = str(excinfo.value)
    assert "authentication_failed" in message
    assert "401" in message
    assert "resolved-secret" not in message
    assert "provider-private-body" not in message


def test_llm_real_endpoint_policy_blocks_private(monkeypatch):
    import auto_tune.modules.train_analyzer.llm_analyzer as llm_module

    monkeypatch.setattr(llm_module, "resolve_credential", lambda purpose: "resolved-secret")
    cfg = _yaml_key_config()
    cfg["llm"]["endpoint"] = "https://127.0.0.1/v1/chat/completions"

    with pytest.raises(RuntimeError) as excinfo:
        llm_module.call_deepseek("prompt", cfg)

    assert "endpoint_rejected" in str(excinfo.value)
