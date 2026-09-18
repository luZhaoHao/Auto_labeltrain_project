"""Regression tests for safe parameter assembly and audit failure policy."""

import json
import threading
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine import loop
from auto_tune.modules.agent_engine.decision_facts import FactPackageError
from auto_tune.modules.agent_engine.loop import (
    TuningHistory,
    _failure,
    _metric_delta,
    _read_reference_before_metrics,
    run_tuning_loop,
    sanitize_and_merge_tuning_params,
)


def _valid_decision() -> dict:
    return {
        "diagnosis": "test diagnosis",
        "action": "apply changes",
        "hyperparameter_changes": {"lr0": 0.001},
        "training_overrides": {},
        "raw_response": '{"hyperparameter_changes": {"lr0": 0.001}}',
        "error": None,
    }


def _valid_fact_package() -> dict:
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "task": "detect",
        "reference_run": "train38",
        "sources": {
            "dataset_report": "dataset_report_1.json",
            "training_report": "train38_report.json",
            "metrics": "results.csv",
            "params": "args.yaml",
        },
        "facts": [{"fact_id": "training.params.lr0", "value": 0.01, "source": "params"}],
    }


def _mock_fact_package(monkeypatch):
    """Existing loop tests inject a hand-built perception without ``sources``;
    Q1.1 freezes the fact package before the decision, so those tests mock the
    builder to isolate the orchestration under test."""
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_tuning_fact_package",
        lambda *a, **k: _valid_fact_package(),
    )


def test_failure_helper_returns_exact_dict():
    assert _failure("decision", "decision_schema_error", "invalid JSON") == {
        "stage": "decision",
        "error_type": "decision_schema_error",
        "message": "invalid JSON",
        "fatal": True,
    }


def test_decision_error_is_fatal_and_never_launches_training(tmp_path, monkeypatch):
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": "not json",
            "error": "Failed to parse JSON from LLM response",
        },
    )
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert launched == []
    assert len(result["iterations"]) == 1
    assert result["failure"]["stage"] == "decision"
    assert result["failure"]["error_type"] == "decision_schema_error"
    assert result["failure"]["fatal"] is True
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["iterations"][0]["decision"]["raw_response"] == "not json"


def test_guardrail_rejection_is_fatal_and_never_launches_training(tmp_path, monkeypatch):
    _mock_fact_package(monkeypatch)
    from auto_tune.modules.agent_engine.guardrails import GuardResult

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: _valid_decision(),
    )
    fake_guard = GuardResult(valid=False, errors=["over-regularization"])
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
        lambda *args, **kwargs: (None, fake_guard),
    )
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert launched == []
    assert len(result["iterations"]) == 1
    assert result["failure"]["error_type"] == "guardrail_rejected"
    assert result["failure"]["fatal"] is True
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["iterations"][0]["guardrails"]["errors"] == ["over-regularization"]


def test_preflight_error_blocks_launch(tmp_path, monkeypatch):
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: _valid_decision(),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.validate_training_preflight",
        lambda *args, **kwargs: ["data yaml missing"],
    )
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert launched == []
    assert result["failure"]["error_type"] == "preflight_error"
    assert "预检" in result["iterations"][0]["error"]
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"


def test_audit_persistence_error_blocks_launch(tmp_path, monkeypatch):
    import auto_tune.modules.agent_engine.audit as audit_module

    def boom(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(audit_module, "atomic_write_json", boom)
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert launched == []
    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["iterations"] == []


def test_metric_delta_computes_difference():
    assert _metric_delta({"mAP50": 0.7}, {"mAP50": 0.72}) == {"mAP50": 0.02}


def test_metric_delta_skips_missing_values():
    assert _metric_delta({"mAP50": 0.7}, {"mAP50_95": 0.41}) == {}


def test_metric_delta_keeps_zero():
    assert _metric_delta({"precision": 0.5}, {"precision": 0.5}) == {"precision": 0.0}


@pytest.mark.parametrize("scenario", [
    "decision_failure",
    "guardrail_failure",
    "preflight_failure",
    "cancellation",
    "dry_run",
])
def test_audit_reaches_terminal_status_in_all_paths(tmp_path, monkeypatch, scenario):
    """Every return path must leave the audit session terminal, never 'running'."""
    _mock_fact_package(monkeypatch)
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    cancel_event = threading.Event()
    skip_execute = False

    if scenario == "decision_failure":
        monkeypatch.setattr(
            "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
            lambda *args, **kwargs: {
                "diagnosis": None, "action": None,
                "hyperparameter_changes": {}, "training_overrides": {},
                "raw_response": "not json", "error": "Failed to parse JSON",
            },
        )
    elif scenario == "guardrail_failure":
        from auto_tune.modules.agent_engine.guardrails import GuardResult
        monkeypatch.setattr(
            "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
            lambda *args, **kwargs: _valid_decision(),
        )
        monkeypatch.setattr(
            "auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
            lambda *args, **kwargs: (None, GuardResult(valid=False, errors=["boom"])),
        )
    elif scenario == "preflight_failure":
        monkeypatch.setattr(
            "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
            lambda *args, **kwargs: _valid_decision(),
        )
        monkeypatch.setattr(
            "auto_tune.modules.agent_engine.loop.validate_training_preflight",
            lambda *args, **kwargs: ["data yaml missing"],
        )
    elif scenario == "cancellation":
        cancel_event.set()
    elif scenario == "dry_run":
        skip_execute = True

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
        skip_execute=skip_execute,
        cancel_event=cancel_event,
    )

    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] in {"completed", "failed", "cancelled"}
    assert audit["status"] != "running"

    if scenario == "cancellation":
        assert audit["status"] == "cancelled"


def test_unexpected_exception_is_fatal_single_iteration(tmp_path, monkeypatch):
    """A fatal iteration exception must stop the loop immediately, not retry."""
    def boom(**kwargs):
        raise RuntimeError("perception exploded")

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        boom,
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert len(result["iterations"]) == 1
    assert result["failure"]["stage"] == "loop"
    assert result["failure"]["error_type"] == "iteration_exception"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["error"]["error_type"] == "iteration_exception"
    assert audit["error"]["error_type"] != "retries_exhausted"


def test_loop_before_metrics_read_from_reference_results_csv(tmp_path, monkeypatch):
    """Before metrics must come from detect/<ref>/results.csv, not perception."""
    _mock_fact_package(monkeypatch)
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    detect_dir = tmp_path / "detect"
    ref_dir = detect_dir / "train38"
    ref_dir.mkdir(parents=True)
    (ref_dir / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\nepochs: 100\n",
        encoding="utf-8",
    )
    (ref_dir / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.00266, 0.57692, 0.04768, 0.00815\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    # perception deliberately carries NO training metrics: results.csv is authoritative.
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command", lambda *a, **k: ["yolo", "train", "epochs=1"])

    class FakeProc:
        def poll(self):
            return 0  # completed

        def terminate(self):
            pass

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", lambda *a, **k: FakeProc())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training", lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None, **kw):
        return {
            "run_id": f"manual:{run_name}",
            "run_name": run_name,
            "source": "manual",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.06, "mAP50_95": 0.02, "precision": 0.01, "recall": 0.60},
            "epochs": 1,
            "artifacts": {"report_path": str(tmp_path / "x_report.json")},
            "analysis_error": None,
            "history_error": None,
            "error": None,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train38",
        log_dir=str(tmp_path),
        auto_analyze=True,
    )

    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["baseline"]["reference_run"] == "train38"
    assert it["baseline"]["params"]["lr0"] == 0.01
    assert it["baseline"]["params"]["batch"] == 16
    assert it["baseline"]["params"]["epochs"] == 100
    assert "_old_batch" not in it["baseline"]["params"]
    assert "_old_batch" not in it["execution"]["actual_params"]
    assert it["baseline"]["metrics"] == {
        "mAP50": 0.04768,
        "mAP50_95": 0.00815,
        "precision": 0.00266,
        "recall": 0.57692,
    }
    assert it["baseline"]["metrics_source"]["type"] == "results_csv"
    assert it["baseline"]["metrics_source"]["epoch_scope"] == "final"
    assert it["result"]["before_metrics"] == it["baseline"]["metrics"]
    assert it["result"]["after_metrics"]["mAP50"] == 0.06
    # metric_delta[key] == after[key] - before[key]
    assert it["result"]["metric_delta"]["mAP50"] == round(0.06 - 0.04768, 10)
    assert it["result"]["metric_delta"]["mAP50_95"] == round(0.02 - 0.00815, 10)
    assert it["result"]["metric_delta"]["precision"] == round(0.01 - 0.00266, 10)
    assert it["result"]["metric_delta"]["recall"] == round(0.60 - 0.57692, 10)


def test_loop_reference_baseline_ignores_global_best(tmp_path, monkeypatch):
    """Baseline must bind to reference_run, not the global-best run summary."""
    _mock_fact_package(monkeypatch)
    detect_dir = tmp_path / "detect"
    for name, map50 in (("train38", 0.04768), ("train39", 0.99)):
        run = detect_dir / name
        run.mkdir(parents=True)
        (run / "args.yaml").write_text("model: yolov8n.pt\ndata: test_data.yaml\n", encoding="utf-8")
        (run / "results.csv").write_text(
            f"epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
            f"0, 0.1, 0.5, {map50}, 0.01\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    # Global summary points at train39 (global best) — must NOT be used for train38.
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **k: {"dataset": {"total_images": 10}, "training": {"best_mAP50": 0.99}},
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        skip_execute=True,
    )

    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    baseline = audit["iterations"][0]["baseline"]
    assert baseline["reference_run"] == "train38"
    assert baseline["metrics"]["mAP50"] == 0.04768
    assert baseline["metrics"]["mAP50"] != 0.99


def test_loop_reference_metrics_without_module_b_report(tmp_path, monkeypatch):
    """Stale/missing Module B report must not block reading reference results.csv."""
    _mock_fact_package(monkeypatch)
    detect_dir = tmp_path / "detect"
    ref_dir = detect_dir / "train38"
    ref_dir.mkdir(parents=True)
    (ref_dir / "args.yaml").write_text("model: yolov8n.pt\ndata: test_data.yaml\n", encoding="utf-8")
    (ref_dir / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.00266, 0.57692, 0.04768, 0.00815\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    # No Module B report: perception has no training metrics at all.
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        skip_execute=True,
    )

    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    baseline = audit["iterations"][0]["baseline"]
    assert baseline["metrics"]["mAP50"] == 0.04768
    assert baseline["metrics"]["mAP50_95"] == 0.00815


def test_read_reference_before_metrics_keeps_zero(tmp_path):
    """Real zero metrics must be kept, not treated as missing."""
    detect_dir = tmp_path / "detect"
    run = detect_dir / "train38"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.0, 0.5, 0.0, 0.01\n",
        encoding="utf-8",
    )
    metrics, source = _read_reference_before_metrics("train38", str(detect_dir))
    assert metrics["precision"] == 0.0
    assert metrics["mAP50"] == 0.0
    assert source["error"] is None


def test_read_reference_before_metrics_no_reference(tmp_path):
    metrics, source = _read_reference_before_metrics(None, str(tmp_path))
    assert metrics == {}
    assert source["error"] == "no_reference_run"


def test_read_reference_before_metrics_missing_csv(tmp_path):
    detect_dir = tmp_path / "detect"
    run = detect_dir / "train38"
    run.mkdir(parents=True)
    metrics, source = _read_reference_before_metrics("train38", str(detect_dir))
    assert metrics == {}
    assert source["error"] == "results_csv_missing"


def test_tuning_training_output_forwarded_to_log_and_sse(tmp_path, monkeypatch):
    """S1.1: candidate training output lands in training.log and emits training_log SSE.

    Uses auto_analyze=True so the loop's wait path joins the output forwarder
    before finalizing, making the log assertion deterministic.
    """
    _mock_fact_package(monkeypatch)
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    detect_dir = tmp_path / "detect"
    ref_dir = detect_dir / "train38"
    ref_dir.mkdir(parents=True)
    (ref_dir / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nepochs: 1\nbatch: 16\n",
        encoding="utf-8",
    )
    (ref_dir / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.1, 0.2, 0.3, 0.05\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command", lambda *a, **k: ["yolo", "train", "epochs=1"])

    launched = {}

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    def fake_launch(train_name, args_path, merged_params, command=None):
        out_dir = str(Path(args_path).parent)
        launched["dir"] = out_dir
        with open(Path(out_dir) / "yolo_train.log", "w", encoding="utf-8") as f:
            f.write("  1/100  1.20G  1.234  0.456  0.789\n")
            f.write("all 10 50 0.123 0.456 0 0.111\n")
            f.write("1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]\n")
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.monitor_training",
        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"),
    )

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None,
                      audit_path=None, started_at=None, finished_at=None, tuning_context=None, **kw):
        return {
            "run_id": f"tuning:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.3, "mAP50_95": 0.05},
            "epochs": 1,
            "artifacts": {"report_path": str(tmp_path / "r.json")},
            "analysis_error": None,
            "history_error": None,
            "error": None,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)

    events = []

    def on_progress(iteration, message, **kwargs):
        events.append({"iteration": iteration, "message": message, "kwargs": kwargs})

    run_tuning_loop(
        {"probe": {"max_retries": 3}, "train_analyzer": {}},
        reference_run="train38",
        log_dir=str(tmp_path),
        on_progress=on_progress,
        auto_analyze=True,
    )

    log_text = (Path(launched["dir"]) / "training.log").read_text(encoding="utf-8")
    assert "  1/100  1.20G  1.234  0.456  0.789" in log_text
    assert "all 10 50 0.123 0.456 0 0.111" in log_text
    assert "5/10" in log_text

    training_events = [e for e in events if e["kwargs"].get("event") == "training_log"]
    kinds = [e["kwargs"]["log_kind"] for e in training_events]
    assert "epoch" in kinds
    assert "validation" in kinds
    epoch_event = next(e for e in training_events if e["kwargs"]["log_kind"] == "epoch")
    assert epoch_event["kwargs"]["epoch"] == 1
    assert epoch_event["kwargs"]["total_epochs"] == 100
    assert epoch_event["kwargs"]["detail"] == "  1/100  1.20G  1.234  0.456  0.789"


def _finalizer_loop_setup(tmp_path, monkeypatch, fake_finalize):
    """Common setup: train38 reference + mocked perception/decision/probe/finalizer."""
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    detect_dir = tmp_path / "detect"
    ref_dir = detect_dir / "train38"
    ref_dir.mkdir(parents=True)
    (ref_dir / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\n", encoding="utf-8"
    )
    (ref_dir / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.00266, 0.57692, 0.04768, 0.00815\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command", lambda *a, **k: ["yolo", "train", "epochs=1"])

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", lambda *a, **k: FakeProc())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training", lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)


def test_loop_calls_shared_finalizer_once_with_tuning_identity(tmp_path, monkeypatch):
    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None, **kw):
        calls.append({
            "run_dir": run_dir,
            "run_name": run_name,
            "source": source,
            "training_status": training_status,
            "session_id": session_id,
            "audit_path": audit_path,
            "started_at": started_at,
            "finished_at": finished_at,
        })
        return {
            "run_id": f"tuning:{session_id}:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.06, "mAP50_95": 0.02, "precision": 0.01, "recall": 0.60},
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": str(tmp_path / "x_report.json")},
            "analysis_error": None,
            "history_error": None,
            "error": None,
        }

    import datetime as _dt
    import types

    class FakeDateTime(_dt.datetime):
        seq = iter([
            _dt.datetime(2026, 8, 14, 7, 0, 0, tzinfo=_dt.timezone.utc),
            _dt.datetime(2026, 8, 14, 7, 1, 0, tzinfo=_dt.timezone.utc),
        ])

        @classmethod
        def now(cls, tz=None):
            return next(cls.seq)

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.datetime",
        types.SimpleNamespace(datetime=FakeDateTime, timezone=_dt.timezone),
    )

    _finalizer_loop_setup(tmp_path, monkeypatch, fake_finalize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        auto_analyze=True,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["source"] == "tuning"
    assert call["training_status"] == "completed"
    assert call["run_name"].startswith("autotune_")
    assert call["session_id"] is not None
    assert call["audit_path"] == result["audit_path"]
    # started_at captured before launch, finished_at after → duration > 0
    assert call["started_at"] == "2026-08-14T07:00:00Z"
    assert call["finished_at"] == "2026-08-14T07:01:00Z"
    assert call["started_at"] < call["finished_at"]
    assert result["iterations"][0]["result_mAP50"] == 0.06
    assert result["iterations"][0]["result_best_epoch"] == 2
    assert result["final_result"]["module_b_analyzed"] is True


def test_loop_finalizer_metrics_reach_audit_without_renaming(tmp_path, monkeypatch):
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None, **kw):
        return {
            "run_id": f"tuning:{session_id}:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.06, "mAP50_95": 0.02, "precision": 0.01, "recall": 0.60},
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": str(tmp_path / "x_report.json")},
            "analysis_error": None,
            "history_error": None,
            "error": None,
        }

    _finalizer_loop_setup(tmp_path, monkeypatch, fake_finalize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        auto_analyze=True,
    )

    assert result["iterations"][0]["result_best_epoch"] == 2
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["result"]["before_metrics"]["mAP50"] == 0.04768
    assert it["result"]["after_metrics"]["mAP50"] == 0.06
    assert it["result"]["after_metrics"]["mAP50_95"] == 0.02
    assert it["result"]["metric_delta"]["mAP50"] == round(0.06 - 0.04768, 10)
    assert it["result"]["metric_delta"]["mAP50_95"] == round(0.02 - 0.00815, 10)


def test_loop_analysis_failure_keeps_training_completed(tmp_path, monkeypatch):
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None, **kw):
        return {
            "run_id": f"tuning:{session_id}:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "failed",
            "metrics": {},
            "epochs": {"configured": 100, "completed": None, "best": None},
            "artifacts": {"report_path": None},
            "analysis_error": {"stage": "analysis", "error_type": "analysis_failed", "message": "bad csv", "timestamp": "x"},
            "history_error": None,
            "error": None,
        }

    _finalizer_loop_setup(tmp_path, monkeypatch, fake_finalize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        auto_analyze=True,
    )

    assert result["iterations"][0]["error"] is None
    assert result["final_result"]["module_b_analyzed"] is False
    assert result["final_result"]["analysis_status"] == "failed"
    assert result["final_result"]["analysis_error"]["error_type"] == "analysis_failed"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "completed"
    assert audit["iterations"][0]["result"]["analysis"]["error_type"] == "analysis_failed"


def test_probe_retry_audit_write_failure_blocks_next_iteration(tmp_path, monkeypatch):
    """A probe RETRY whose audit write fails must not continue to the next round."""
    _mock_fact_package(monkeypatch)
    from auto_tune.modules.agent_engine.audit import TuningAuditSession
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command", lambda *a, **k: ["yolo", "train"])

    launched = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

    def fake_launch(*a, **k):
        launched.append(True)
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.monitor_training",
        lambda *a, **k: ProbeDecision(ProbeDecision.RETRY, "low mAP"),
    )

    def boom_fail(self, iteration, stage, error_type, message, fatal=True):
        raise OSError("disk full")

    monkeypatch.setattr(TuningAuditSession, "fail_iteration", boom_fail)

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["failure"]["stage"] == "audit"
    assert len(result["iterations"]) == 1
    assert len(launched) == 1


def test_probe_abort_audit_write_failure_blocks_next_iteration(tmp_path, monkeypatch):
    """A probe ABORT whose audit write fails must not be treated as retryable."""
    _mock_fact_package(monkeypatch)
    from auto_tune.modules.agent_engine.audit import TuningAuditSession
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters", lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight", lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command", lambda *a, **k: ["yolo", "train"])

    launched = []

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

    def fake_launch(*a, **k):
        launched.append(True)
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.monitor_training",
        lambda *a, **k: ProbeDecision(ProbeDecision.ABORT, "loss exploded"),
    )

    def boom_fail(self, iteration, stage, error_type, message, fatal=True):
        raise OSError("disk full")

    monkeypatch.setattr(TuningAuditSession, "fail_iteration", boom_fail)

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert len(result["iterations"]) == 1
    assert len(launched) == 1


def test_loop_merges_only_sanitized_values():
    """Catches raw LLM values being merged after Guardrails warned about them."""
    base = {"lr0": 0.01, "batch": 16, "optimizer": "SGD", "model": "yolov8n.pt"}

    merged, guard = sanitize_and_merge_tuning_params(
        base,
        {"lr0": 2.0},
        {"batch": 0, "optimizer": "AdamW"},
        {"total_images": 230},
    )

    assert guard.valid is True
    assert merged["lr0"] == 0.1
    assert merged["batch"] == 1
    assert merged["optimizer"] == "AdamW"
    assert merged["model"] == "yolov8n.pt"


def test_loop_does_not_return_merged_params_when_guardrails_reject():
    """Catches invalid semantic combinations continuing to execution."""
    merged, guard = sanitize_and_merge_tuning_params(
        {"optimizer": "SGD"},
        {"lr0": 0.0025},
        {"optimizer": "auto"},
        {},
    )

    assert guard.valid is False
    assert merged is None


def test_history_feedback_contains_real_metric_delta():
    """Catches the next LLM iteration receiving result='unknown'."""
    history = TuningHistory()
    history.add_attempt({
        "decision": {"hyperparameter_changes": {"lr0": 0.002}},
        "result_mAP50": 0.72,
        "result_mAP50_95": 0.41,
        "before_metrics": {"mAP50": 0.70, "mAP50_95": 0.39},
        "probe_decision": {"verdict": "continue"},
        "error": None,
    })

    feedback = history.get_previous_changes()[0]

    assert feedback["changes"] == {"lr0": 0.002}
    assert feedback["after_metrics"]["mAP50"] == 0.72
    assert feedback["metric_delta"] == {"mAP50": 0.02, "mAP50_95": 0.02}
    assert feedback["probe_verdict"] == "continue"


# ── S1.5: explicit run-state callback ──


def test_loop_on_state_reports_phases_dry_run(tmp_path, monkeypatch):
    _mock_fact_package(monkeypatch)
    states = []

    def on_state(phase, event_type, message, process_identity=None):
        states.append(phase)

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: _valid_decision(),
    )

    run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run=None,
        log_dir=str(tmp_path),
        skip_execute=True,
        on_state=on_state,
    )

    assert "preparing" in states
    assert "analyzing" in states
    assert "finalizing" in states


def test_loop_on_state_binds_process_identity(tmp_path, monkeypatch):
    import os as _os

    from auto_tune.modules.run_state.models import ProcessIdentity

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      session_id=None, audit_path=None, started_at=None,
                      finished_at=None, tuning_context=None, **kw):
        return {
            "run_id": f"tuning:{session_id}:{run_name}",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "analysis_status": "completed",
            "metrics": {"mAP50": 0.06, "mAP50_95": 0.02, "precision": 0.01, "recall": 0.60},
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": str(tmp_path / "x_report.json")},
            "analysis_error": None,
            "history_error": None,
            "error": None,
        }

    _finalizer_loop_setup(tmp_path, monkeypatch, fake_finalize)

    class FakeProcWithPid:
        def __init__(self):
            self.pid = _os.getpid()

        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *a, **k: FakeProcWithPid(),
    )

    executing = []

    def on_state(phase, event_type, message, process_identity=None):
        if event_type == "executing":
            executing.append(process_identity)

    run_tuning_loop(
        {"probe": {"max_retries": 1}},
        reference_run="train38",
        log_dir=str(tmp_path),
        auto_analyze=True,
        on_state=on_state,
    )

    assert executing and executing[0] is not None
    assert isinstance(executing[0], ProcessIdentity)
    assert executing[0].pid == _os.getpid()
    assert executing[0].process_create_token


# ── Q1.1 Task 4: fact package freezes before LLM; failures never launch ──────


def _loop_perception(reference_run="train54"):
    return {
        "dataset": {"total_images": 10},
        "training": {"reference_run": reference_run},
        "sources": {
            "dataset_report": {"status": "available", "basename": "dataset_report_1.json"},
            "training_report": {"status": "available",
                                "basename": f"{reference_run}_report.json"},
        },
    }


_DEFAULT_FINAL_METRICS = {
    "precision": 0.00266, "recall": 0.57692, "mAP50": 0.04768, "mAP50_95": 0.00815,
}


def _reference_run(tmp_path, name="train54", final_metrics=None):
    """参考运行目录。

    ``final_metrics`` 为 None 时沿用既有默认指标；显式传入的 ``None`` 值写出空
    单元格，表示该指标**缺失**（不是 0），供「基线未知」场景使用。
    """
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\nepochs: 100\n",
        encoding="utf-8",
    )
    metrics = dict(_DEFAULT_FINAL_METRICS if final_metrics is None else final_metrics)

    def _cell(value):
        return "" if value is None else f"{value}"

    (ref / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        f"0, {_cell(metrics['precision'])}, {_cell(metrics['recall'])}, "
        f"{_cell(metrics['mAP50'])}, {_cell(metrics['mAP50_95'])}\n",
        encoding="utf-8",
    )
    return detect


def _loop_basic_mocks(monkeypatch, tmp_path, reference_run="train54",
                      final_metrics=None):
    detect = _reference_run(tmp_path, reference_run, final_metrics)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception",
                        lambda **k: _loop_perception(reference_run))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])
    return detect


def _valid_tuning_decision():
    return {
        "diagnosis": "test diagnosis",
        "action": "apply changes",
        "hyperparameter_changes": {"lr0": 0.001},
        "training_overrides": {},
        "raw_response": '{"lr0": 0.001}',
        "error": None,
        "retried": False,
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "evidence_ids": {"lr0": ["training.params.lr0"]},
        "validation": {
            "valid": True, "error_code": None, "error_detail": None,
            "retried": False, "referenced_fact_ids": ["training.params.lr0"],
        },
    }


def _contract_failure_decision(error_code, retried=True):
    return {
        "diagnosis": None, "action": None,
        "hyperparameter_changes": {}, "training_overrides": {},
        "raw_response": None, "error": error_code, "retried": retried,
        "schema_version": None, "fact_package_id": "sha256:test",
        "evidence_ids": {},
        "validation": {
            "valid": False, "error_code": error_code, "error_detail": "detail",
            "retried": retried, "referenced_fact_ids": [],
        },
    }


def _semantic_failure_decision(error_code="DECISION_SEMANTIC_UNSUPPORTED", retried=True):
    return {
        "diagnosis": None, "action": None,
        "hyperparameter_changes": {}, "training_overrides": {},
        "raw_response": None, "error": error_code, "retried": retried,
        "schema_version": None, "fact_package_id": "sha256:test",
        "evidence_ids": {},
        "validation": {
            "valid": False, "error_code": error_code, "error_detail": "semantic failed",
            "retried": retried, "referenced_fact_ids": [],
        },
        "semantic_validation": {
            "valid": False, "error_code": error_code, "reason_code": "NO_SUPPORTING_RULE",
            "retried": retried,
            "parameters": [{
                "parameter": "lr0", "current_value": 0.01, "suggested_value": 0.006,
                "change_direction": "decrease", "rule_ids": [],
                "supporting_fact_ids": [], "conflicting_fact_ids": [],
                "neutral_fact_ids": ["training.metrics.mAP50"],
            }],
        },
    }


def _valid_semantic_decision():
    decision = _valid_tuning_decision()
    decision["semantic_validation"] = {
        "valid": True, "error_code": None, "reason_code": None,
        "retried": False, "parameters": [],
    }
    return decision


def test_fact_package_failure_stops_before_llm_and_training(tmp_path, monkeypatch):
    calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_tuning_fact_package",
        lambda *a, **k: (_ for _ in ()).throw(FactPackageError("reference mismatch")),
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: calls.append("llm"))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: calls.append("train"))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert calls == []
    assert result["failure"]["error_type"] == "fact_package_invalid"
    assert result["failure"]["stage"] == "facts"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["iterations"][0]["error"]["error_type"] == "fact_package_invalid"


def test_fact_package_written_before_llm_call(tmp_path, monkeypatch):
    """The fact package must be persisted to the audit before the LLM is called."""
    from auto_tune.modules.agent_engine.audit import TuningAuditSession

    order = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    orig_update = TuningAuditSession.update_iteration

    def spy_update(self, iteration, **fields):
        for key in fields:
            order.append(key)
        return orig_update(self, iteration, **fields)

    monkeypatch.setattr(TuningAuditSession, "update_iteration", spy_update)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: order.append("LLM") or _valid_tuning_decision())

    run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True,
    )

    assert order.index("fact_package") < order.index("LLM")


def test_fact_package_audit_write_failure_stops_before_llm(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.audit import TuningAuditSession

    calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: calls.append("llm"))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: calls.append("train"))

    orig_update = TuningAuditSession.update_iteration

    def fail_on_fact_package(self, iteration, **fields):
        if "fact_package" in fields:
            raise OSError("disk full")
        return orig_update(self, iteration, **fields)

    monkeypatch.setattr(TuningAuditSession, "update_iteration", fail_on_fact_package)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert calls == []
    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["failure"]["stage"] == "audit"


@pytest.mark.parametrize("error_code,error_type", [
    ("DECISION_FACT_PACKAGE_MISMATCH", "decision_fact_package_mismatch"),
    ("DECISION_EVIDENCE_MISSING", "decision_evidence_missing"),
    ("DECISION_EVIDENCE_UNKNOWN", "decision_evidence_unknown"),
])
def test_contract_failure_never_builds_command_or_launches(
    tmp_path, monkeypatch, error_code, error_type
):
    cmd_calls = []
    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _contract_failure_decision(error_code))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert cmd_calls == []
    assert train_calls == []
    assert result["failure"]["error_type"] == error_type
    assert result["failure"]["stage"] == "decision"


def test_second_retry_failure_never_launches(tmp_path, monkeypatch):
    """A contract failure that already retried once is terminal: no command,
    no launch."""
    cmd_calls = []
    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _contract_failure_decision(
                            "DECISION_EVIDENCE_UNKNOWN", retried=True))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert cmd_calls == []
    assert train_calls == []
    assert result["failure"]["error_type"] == "decision_evidence_unknown"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["decision_validation"]["error_code"] == "DECISION_EVIDENCE_UNKNOWN"
    assert it["decision_validation"]["retried"] is True


def test_keep_params_path_builds_fact_package_without_llm(tmp_path, monkeypatch):
    llm_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: llm_calls.append(1))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True, keep_params=True,
    )

    assert llm_calls == []
    assert result["failure"] is None
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["fact_package"]["fact_package_id"] == "sha256:test"
    assert it["decision"]["action"] == "keep_params"
    assert it["decision"]["fact_package_id"] == "sha256:test"
    assert it["decision"]["hyperparameter_changes"] == {}
    assert it["decision"]["training_overrides"] == {}
    assert it["decision"]["evidence_ids"] == {}
    assert it["decision_validation"]["valid"] is True


def test_valid_evidence_still_enters_guardrails(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.guardrails import GuardResult

    guard_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _valid_tuning_decision())

    def fake_sanitize(base, changes, overrides, dataset_info):
        guard_calls.append(1)
        return {"lr0": 0.001}, GuardResult(valid=True, warnings=[], errors=[])

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        fake_sanitize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True,
    )

    assert guard_calls == [1]
    assert result["failure"] is None
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["fact_package"]["fact_package_id"] == "sha256:test"
    assert it["decision"]["evidence_ids"] == {"lr0": ["training.params.lr0"]}
    assert it["decision_validation"]["valid"] is True
    assert it["baseline"]["reference_run"] == "train54"
    assert it["baseline"]["metrics"]["mAP50"] == 0.04768


# ── F1.1-B 阶段五：多轮调优的基线更新与停止条件由代码确定 ──────────────────
#
# 旧实现每轮无条件把 reference_run 换成刚训练完的运行：变差的运行也会成为新
# 基线，模型看不到「上次那招没用」，于是同一组建议被反复提出（真实审计里
# lr0 被逐轮腰斩 0.01→0.005→0.0025→0.00125，无人叫停）。


def _auto_loop_decision(changes=None, action="adjust"):
    decision = _valid_tuning_decision()
    decision["action"] = action
    decision["hyperparameter_changes"] = dict(changes or {"lr0": 0.005})
    decision["evidence_ids"] = {k: ["training.params.lr0"]
                                for k in decision["hyperparameter_changes"]}
    return decision


def _auto_loop_harness(monkeypatch, tmp_path, decisions, metrics, max_rounds=2,
                       final_metrics=None, keep_params=False):
    """驱动 auto_loop：逐轮返回给定决策与最终指标，记录每轮真正启动的训练。

    ``final_metrics`` 控制参考运行 results.csv 的最终指标，因此也决定进入调优前
    的原参考基线综合分。
    """
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    launched = []
    _loop_basic_mocks(monkeypatch, tmp_path, final_metrics=final_metrics)
    _mock_fact_package(monkeypatch)
    decisions_iter, metrics_iter = iter(decisions), iter(metrics)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: next(decisions_iter))

    def _finalize(run_dir, run_name, source, config, **kw):
        launched.append(run_name)
        return {
            "run_id": f"tuning:{kw.get('session_id')}:{run_name}",
            "run_name": run_name, "source": "tuning", "status": "completed",
            "analysis_status": "completed", "metrics": next(metrics_iter),
            "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": None},
            "analysis_error": None, "history_error": None,
            "index_error": None, "error": None,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", _finalize)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: FakeProc())

    result = run_tuning_loop(
        {"probe": {"max_retries": max_rounds}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True, auto_loop=True,
        keep_params=keep_params,
    )
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    return result, audit, launched


def test_auto_loop_keeps_the_better_baseline_when_a_round_regresses(tmp_path, monkeypatch):
    """第二轮变差时，基线仍是第一轮；并且就此结束，不再继续下探。

    这正是真实审计里 lr0 被逐轮腰斩（0.006→0.004→0.002）而指标不动的场景：
    变差的运行不得接替基线，循环也不该继续消耗训练。
    """
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005}),
                   _auto_loop_decision({"lr0": 0.002})],
        metrics=[{"mAP50": 0.60, "mAP50_95": 0.30},
                 {"mAP50": 0.10, "mAP50_95": 0.05}],
        max_rounds=5,
    )

    assert len(launched) == 2
    assert audit["iterations"][1]["perception"]["reference_run"] == launched[0]
    verdict = audit["iterations"][1]["round_verdict"]
    assert verdict["improved"] is False
    assert verdict["best_run"] == launched[0]
    assert verdict["next_reference_run"] == launched[0]
    assert result["stop_reason"] == "no_improvement"
    assert result["baseline_run"] == launched[0]


def test_auto_loop_improvement_moves_the_baseline_forward(tmp_path, monkeypatch):
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005}),
                   _auto_loop_decision({"lr0": 0.002})],
        metrics=[{"mAP50": 0.30, "mAP50_95": 0.10},
                 {"mAP50": 0.70, "mAP50_95": 0.40}],
        max_rounds=2,
    )

    assert audit["iterations"][1]["perception"]["reference_run"] == launched[0]
    verdict = audit["iterations"][1]["round_verdict"]
    assert verdict["improved"] is True
    assert verdict["best_run"] == launched[1]
    assert verdict["next_reference_run"] == launched[1]


def test_auto_loop_stops_when_the_same_change_is_repeated(tmp_path, monkeypatch):
    """第二轮给出与已执行轮次相同的实际参数：训练前就停止，不浪费一次训练。

    旧实现在训练完成后才判定重复，因此永远会白白多跑一轮。
    """
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005}),
                   _auto_loop_decision({"lr0": 0.005})],
        metrics=[{"mAP50": 0.60, "mAP50_95": 0.30}],
        max_rounds=5,
    )

    assert len(launched) == 1
    assert result["stop_reason"] == "repeated_change"
    assert audit["stop_reason"] == "repeated_change"
    verdict = audit["iterations"][1]["round_verdict"]
    assert verdict["repeated_change"] is True
    assert verdict["phase"] == "pre_training"


def test_auto_loop_stops_on_parameter_oscillation_before_training(tmp_path, monkeypatch):
    """同一个参数先增后减（来回摆动）时必须在训练前终止。"""
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.02}),
                   _auto_loop_decision({"lr0": 0.005})],
        metrics=[{"mAP50": 0.30, "mAP50_95": 0.10}],
        max_rounds=5,
    )

    assert len(launched) == 1
    assert result["stop_reason"] == "oscillation"
    verdict = audit["iterations"][1]["round_verdict"]
    assert verdict["oscillating_params"] == ["lr0"]
    assert verdict["phase"] == "pre_training"


def test_auto_loop_keep_params_launches_no_training(tmp_path, monkeypatch):
    """LLM 返回 keep_params 是正常终态：不建目录、不构造命令、不启动训练。"""
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({}, action="keep_params")],
        metrics=[{"mAP50": 0.60, "mAP50_95": 0.30}],
        max_rounds=4,
    )

    assert launched == []
    assert result["stop_reason"] == "keep_params"
    assert audit["status"] == "completed"
    assert audit["stop_reason"] == "keep_params"

    iteration = audit["iterations"][0]
    assert iteration["round_verdict"]["stop_reason"] == "keep_params"
    assert iteration["round_verdict"]["phase"] == "pre_training"
    # 审计闭环：决策、护栏结果与当前最佳基线都能追溯到
    assert iteration["decision"]["action"] == "keep_params"
    assert iteration["guardrails"]["valid"] is True
    assert iteration["baseline"]["reference_run"] == "train54"
    # 没有构造训练命令，也没有分配训练目录
    assert iteration["execution"]["command"] == []
    assert iteration["execution"]["train_name"] is None
    # 没有创建任何调优运行目录（detect/ 下只剩原参考运行）
    assert sorted(p.name for p in (tmp_path / "detect").iterdir()) == ["train54"]


def test_auto_loop_repeated_after_clamping_stops_before_training(tmp_path, monkeypatch):
    """重复必须按护栏处理后的实际执行值判断，而不是按模型原始建议值。

    两轮建议的 lr0 不同（1e-6 / 1e-7），但都会被夹紧到同一个下界 1e-5：
    实际准备执行的参数完全相同，因此第二轮不得再启动训练。
    """
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 1e-6}),
                   _auto_loop_decision({"lr0": 1e-7})],
        metrics=[{"mAP50": 0.60, "mAP50_95": 0.30}],
        max_rounds=5,
    )

    assert len(launched) == 1
    assert audit["iterations"][0]["guardrails"]["clamped"]["lr0"] == 1e-5
    assert audit["iterations"][1]["guardrails"]["clamped"]["lr0"] == 1e-5
    assert result["stop_reason"] == "repeated_change"
    assert audit["iterations"][1]["round_verdict"]["phase"] == "pre_training"


def test_auto_loop_still_trains_a_genuinely_new_change(tmp_path, monkeypatch):
    """真正的新参数建议仍照常启动训练（预训练判定不得误伤正常轮次）。"""
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005}),
                   _auto_loop_decision({"lr0": 0.004}),
                   _auto_loop_decision({"lr0": 0.003})],
        metrics=[{"mAP50": 0.30, "mAP50_95": 0.10},
                 {"mAP50": 0.40, "mAP50_95": 0.15},
                 {"mAP50": 0.50, "mAP50_95": 0.20}],
        max_rounds=3,
    )

    assert len(launched) == 3
    assert [it["round_verdict"]["phase"] for it in audit["iterations"]] == [
        "post_training", "post_training", "post_training"]


def test_manual_train_with_original_params_still_launches_training(tmp_path, monkeypatch):
    """用户主动选择「按原参数训练」时仍必须训练一次。

    预训练停止只针对 LLM 在 auto_loop 里返回的 action=keep_params；显式的
    keep_params 模式代表「用原参数跑一次」，且不调用 LLM，契约不变。
    """
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[],
        metrics=[{"mAP50": 0.60, "mAP50_95": 0.30}],
        max_rounds=1,
        keep_params=True,
    )

    assert len(launched) == 1
    assert result["stop_reason"] == "keep_params"


# ── Codex 复核 P1：进入调优前的原参考运行就是初始最佳基线 ──────────────────

_HIGH_REFERENCE_METRICS = {
    "mAP50": 0.90, "mAP50_95": 0.90, "precision": 0.90, "recall": 0.90,
}
_LOW_REFERENCE_METRICS = {
    "mAP50": 0.10, "mAP50_95": 0.05, "precision": 0.10, "recall": 0.10,
}


def test_reference_run_is_the_initial_baseline_when_the_first_round_regresses(
        tmp_path, monkeypatch):
    """原参考综合分 0.90、首轮 0.16：基线仍是原参考运行，整体最佳不是首轮。"""
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005})],
        metrics=[{"mAP50": 0.20, "mAP50_95": 0.10}],
        max_rounds=3,
        final_metrics=_HIGH_REFERENCE_METRICS,
    )

    assert len(launched) == 1
    verdict = audit["iterations"][0]["round_verdict"]
    assert verdict["improved"] is False
    assert verdict["best_run"] == "train54"
    assert verdict["best_score"] == pytest.approx(0.90)
    assert verdict["next_reference_run"] == "train54"
    assert result["stop_reason"] == "no_improvement"
    assert result["baseline_run"] == "train54"
    assert result["kept_reference_baseline"] is True

    # 收尾报告不得把变差的首轮描述成整体最佳
    summary = result["final_summary"]
    assert summary["kept_reference_baseline"] is True
    assert summary["reference_baseline"]["run_name"] == "train54"
    assert summary["reference_baseline"]["score"] == pytest.approx(0.90)
    text = Path(result["final_summary_path"]).read_text(encoding="utf-8")
    assert "原参考运行" in text
    assert "train54" in text


def test_first_round_better_than_the_reference_replaces_the_baseline(tmp_path, monkeypatch):
    """原参考 0.0875、首轮 0.93：只有这时首轮才接替基线。"""
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005})],
        metrics=[{"mAP50": 0.95, "mAP50_95": 0.90}],
        max_rounds=1,
        final_metrics=_LOW_REFERENCE_METRICS,
    )

    verdict = audit["iterations"][0]["round_verdict"]
    assert verdict["improved"] is True
    assert verdict["best_run"] == launched[0]
    assert result["reference_baseline"]["run_name"] == "train54"
    assert result["reference_baseline"]["score"] == pytest.approx(0.0875)
    assert result["kept_reference_baseline"] is False


def test_reference_without_metrics_keeps_the_initial_baseline_unknown(
        tmp_path, monkeypatch):
    """原参考指标缺失：基线分数保持未知（None），绝不补造 0。

    未知基线之后沿用既有的「首个可测分数即基线」语义（此时没有任何可比较的
    历史最佳），但初始化本身必须是 None 而不是 0。
    """
    missing = {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None}
    result, audit, launched = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005})],
        metrics=[{"mAP50": 0.30, "mAP50_95": 0.10}],
        max_rounds=1,
        final_metrics=missing,
    )

    assert result["reference_baseline"] == {"run_name": "train54", "score": None}

    verdict = audit["iterations"][0]["round_verdict"]
    assert verdict["improved"] is True
    assert verdict["best_run"] == launched[0]


def test_auto_loop_regression_does_not_promote_the_worse_run(tmp_path, monkeypatch):
    """变差的运行不得成为被记录的最佳运行。"""
    result, audit, _ = _auto_loop_harness(
        monkeypatch, tmp_path,
        decisions=[_auto_loop_decision({"lr0": 0.005}),
                   _auto_loop_decision({"lr0": 0.002})],
        metrics=[{"mAP50": 0.80, "mAP50_95": 0.50},
                 {"mAP50": 0.05, "mAP50_95": 0.01}],
        max_rounds=2,
    )

    assert result["best_iteration"] == 1
    assert result["best_metrics"]["mAP50"] == 0.80


def test_legal_chain_builds_command_once_and_audits_params(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    cmd_calls = []
    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _valid_tuning_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda name, args_path, merged: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or FakeProc())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run",
                        lambda run_dir, run_name, source, config, **kw: {
                            "run_id": f"tuning:{kw.get('session_id')}:{run_name}",
                            "run_name": run_name,
                            "source": "tuning", "status": "completed",
                            "analysis_status": "completed",
                            "metrics": {"mAP50": 0.06, "mAP50_95": 0.02},
                            "epochs": {"configured": 100, "completed": 3, "best": 2},
                            "artifacts": {"report_path": None},
                            "analysis_error": None, "history_error": None,
                            "index_error": None, "error": None,
                        })

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert len(cmd_calls) == 1
    assert len(train_calls) == 1
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    assert it["execution"]["command"] == ["yolo", "train"]
    assert it["execution"]["actual_params"]["lr0"] == 0.001
    assert it["execution"]["actual_params"]["epochs"] == 100
    assert it["decision_validation"]["valid"] is True
    assert it["fact_package"]["fact_package_id"] == "sha256:test"


# ── Q1.2：语义校验在 Guardrails 前、失败零启动 ──────────────────────────────


def test_semantic_failure_never_enters_guardrails_or_launches(tmp_path, monkeypatch):
    guard_calls = []
    cmd_calls = []
    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _semantic_failure_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        lambda *a, **k: guard_calls.append(1) or ({}, None))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert guard_calls == []
    assert cmd_calls == []
    assert train_calls == []
    assert result["failure"]["error_type"] == "decision_semantic_error"
    assert result["failure"]["stage"] == "decision"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    sv = audit["iterations"][0]["semantic_validation"]
    assert sv["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert sv["reason_code"] == "NO_SUPPORTING_RULE"


def test_second_semantic_failure_is_terminal_and_zero_launch(tmp_path, monkeypatch):
    cmd_calls = []
    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _semantic_failure_decision(retried=True))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert cmd_calls == []
    assert train_calls == []
    assert result["failure"]["error_type"] == "decision_semantic_error"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["iterations"][0]["semantic_validation"]["retried"] is True


def test_q1_contract_failure_leaves_semantic_validation_none(tmp_path, monkeypatch):
    """Q1.1 失败时不进入 Q1.2：审计 semantic_validation 保持 None。"""
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _contract_failure_decision(
                            "DECISION_EVIDENCE_UNKNOWN", retried=True))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert result["failure"]["error_type"] == "decision_evidence_unknown"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["iterations"][0]["semantic_validation"] is None


def test_valid_semantic_decision_still_enters_guardrails(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.guardrails import GuardResult

    guard_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _valid_semantic_decision())

    def fake_sanitize(base, changes, overrides, dataset_info):
        guard_calls.append(1)
        return {"lr0": 0.001}, GuardResult(valid=True, warnings=[], errors=[])

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        fake_sanitize)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True,
    )

    assert guard_calls == [1]
    assert result["failure"] is None
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["iterations"][0]["semantic_validation"]["valid"] is True


def test_semantic_validation_audit_write_failure_blocks_launch(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.audit import TuningAuditSession

    train_calls = []
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _valid_semantic_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())
    orig_update = TuningAuditSession.update_iteration

    def fail_on_semantic(self, iteration, **fields):
        if "semantic_validation" in fields:
            raise OSError("disk full")
        return orig_update(self, iteration, **fields)

    monkeypatch.setattr(TuningAuditSession, "update_iteration", fail_on_semantic)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert train_calls == []
    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["failure"]["stage"] == "audit"


# ── Q1.2 返修一：每次模型响应校验后立即持久化 decision_attempts ─────────────


def _bad_lr0_tuning_json():
    """证据是真实事实（training.params.lr0）但无 lr0 语义规则 → Q1.2 失败。"""
    return json.dumps({
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "diagnosis": "降低学习率",
        "action": "adjust",
        "hyperparameter_changes": {"lr0": 0.006},
        "training_overrides": {},
        "evidence_ids": {"lr0": ["training.params.lr0"]},
    }, ensure_ascii=False)


def _keep_params_tuning_json():
    return json.dumps({
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "diagnosis": "保持参数",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
        "evidence_ids": {},
    }, ensure_ascii=False)


def test_decision_attempts_persisted_before_continue(tmp_path, monkeypatch):
    """第一次响应校验后先写 attempt，纠错响应后再写第二个 attempt；顶层仍为最终结果。"""
    from auto_tune.modules.agent_engine import decision_agent as da_mod
    from auto_tune.modules.agent_engine.guardrails import GuardResult

    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    replies = iter([_bad_lr0_tuning_json(), _keep_params_tuning_json()])
    llm_calls = []
    monkeypatch.setattr(da_mod, "call_decision_llm",
                        lambda *a, **k: llm_calls.append(1) or next(replies))
    guard_calls = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        lambda *a, **k: guard_calls.append(1)
                        or ({"lr0": 0.001}, GuardResult(valid=True, warnings=[], errors=[])))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True,
    )

    assert result["failure"] is None
    assert len(llm_calls) == 2  # 无第三次调用
    assert guard_calls == [1]   # 成功进入 Guardrails
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    it = audit["iterations"][0]
    attempts = it["decision_attempts"]
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["retried"] is False
    assert attempts[1]["retried"] is True
    assert attempts[0]["decision"]["fact_package_id"] == "sha256:test"
    assert attempts[1]["decision"]["fact_package_id"] == "sha256:test"
    assert attempts[0]["semantic_validation"]["valid"] is False
    assert attempts[0]["semantic_validation"]["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert attempts[1]["decision_validation"]["valid"] is True
    # 顶层字段仍是最终采用/最终失败那次（第二次 keep_params）
    assert it["decision"]["action"] == "keep_params"
    assert it["decision_validation"]["valid"] is True
    assert it["semantic_validation"]["valid"] is True


def test_decision_attempt1_audit_write_failure_blocks_launch(tmp_path, monkeypatch):
    """attempt 1 审计写入失败：不发起纠错、不进 Guardrails、不启动训练，磁盘 attempts 为空。"""
    from auto_tune.modules.agent_engine import decision_agent as da_mod
    from auto_tune.modules.agent_engine.audit import TuningAuditSession

    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    replies = iter([_bad_lr0_tuning_json(), _keep_params_tuning_json()])
    llm_calls = []
    monkeypatch.setattr(da_mod, "call_decision_llm",
                        lambda *a, **k: llm_calls.append(1) or next(replies))
    guard_calls = []
    cmd_calls = []
    train_calls = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        lambda *a, **k: guard_calls.append(1) or ({}, None))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())
    orig_update = TuningAuditSession.update_iteration

    def fail_on_attempts(self, iteration, **fields):
        if "decision_attempts" in fields:
            raise OSError("disk full")
        return orig_update(self, iteration, **fields)

    monkeypatch.setattr(TuningAuditSession, "update_iteration", fail_on_attempts)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert len(llm_calls) == 1
    assert guard_calls == []
    assert cmd_calls == []
    assert train_calls == []
    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["failure"]["stage"] == "audit"
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["iterations"][0]["error"]["error_type"] == "audit_persistence_error"
    assert audit["iterations"][0]["decision_attempts"] == []


def test_attempt2_write_failure_disk_keeps_only_attempt_1(tmp_path, monkeypatch):
    """attempt 2 写盘失败后磁盘审计只保留 attempt 1，且终态为 audit_persistence_error。"""
    from auto_tune.modules.agent_engine import decision_agent as da_mod
    from auto_tune.modules.agent_engine.audit import TuningAuditSession

    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    replies = iter([_bad_lr0_tuning_json(), _keep_params_tuning_json()])
    llm_calls = []
    monkeypatch.setattr(da_mod, "call_decision_llm",
                        lambda *a, **k: llm_calls.append(1) or next(replies))
    guard_calls = []
    cmd_calls = []
    train_calls = []
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.sanitize_and_merge_tuning_params",
                        lambda *a, **k: guard_calls.append(1) or ({}, None))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: cmd_calls.append(1) or ["yolo", "train"])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: train_calls.append(1) or object())
    orig_update = TuningAuditSession.update_iteration

    def fail_on_second_attempt(self, iteration, **fields):
        if "decision_attempts" in fields and len(fields["decision_attempts"]) == 2:
            raise OSError("disk full")
        return orig_update(self, iteration, **fields)

    monkeypatch.setattr(TuningAuditSession, "update_iteration", fail_on_second_attempt)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), auto_analyze=True,
    )

    assert result["failure"]["error_type"] == "audit_persistence_error"
    assert result["failure"]["stage"] == "audit"
    assert len(llm_calls) == 2
    assert guard_calls == []
    assert cmd_calls == []
    assert train_calls == []

    # 重新读取磁盘 JSON：attempt 2 必须不在其中
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    attempts = audit["iterations"][0]["decision_attempts"]
    assert [a["attempt"] for a in attempts] == [1]
    assert all(a["attempt"] != 2 for a in attempts)
    # attempt 1 内容保持完整
    assert attempts[0]["decision"]["fact_package_id"] == "sha256:test"
    assert attempts[0]["decision"]["diagnosis"] == "降低学习率"
    assert attempts[0]["semantic_validation"]["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert attempts[0]["semantic_validation"]["reason_code"] == "NO_SUPPORTING_RULE"
    assert audit["iterations"][0]["error"]["error_type"] == "audit_persistence_error"


def test_keep_params_path_records_empty_decision_attempts(tmp_path, monkeypatch):
    """keep_params 不调用 LLM，decision_attempts 保持空列表。"""
    _loop_basic_mocks(monkeypatch, tmp_path)
    _mock_fact_package(monkeypatch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call LLM")))

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train54",
        log_dir=str(tmp_path), skip_execute=True, keep_params=True,
    )

    assert result["failure"] is None
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["iterations"][0]["decision_attempts"] == []


# ── F1.1-A Task 4：大模型调优继承参考运行的初始权重 ─────────────────


def test_llm_tuning_inherits_the_reference_run_model():
    """执行参数里的 model 必须就是参考运行 args.yaml 里的那一个。

    大模型调优没有独立权重选择器，也不存在任何请求字段能覆盖它；合并只
    在参考运行的 base_args 之上应用护栏校验过的超参数变更。
    """
    reference_args = {
        "model": "detect/train63/weights/best.pt",
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "optimizer": "SGD",
    }

    merged, guard = sanitize_and_merge_tuning_params(
        reference_args, {"lr0": 0.005}, {"batch": 8}, {"total_images": 230})

    assert guard.valid is True
    assert merged["model"] == reference_args["model"]
    assert merged["lr0"] == 0.005
    assert merged["batch"] == 8
    # 参考运行里没有的模型字段绝不会被凭空造出来，也不会变成默认权重名
    assert merged["model"] != "yolov8n.pt"


def test_llm_tuning_never_executes_a_swapped_model():
    """即使 LLM 违规给出 model 变更，也在结构契约边界被判不合法（零启动）。

    model 只有「继承参考运行」这一种来源：它已不是可调参数，因此违规响应在
    TuningDecision v1 解析阶段就以 unknown parameter 被拒绝，既不会进入 Q1.2
    语义校验，也不可能进入执行参数；参考运行的基线不会被新权重悄悄沿用。
    """
    from auto_tune.modules.agent_engine.decision_contract import (
        DecisionContractError,
        parse_tuning_decision_response,
    )
    from auto_tune.modules.agent_engine.decision_semantics import (
        validate_decision_semantics,
    )
    from auto_tune.modules.agent_engine.parameter_registry import (
        get_tunable_parameter_names,
    )
    from auto_tune.modules.agent_engine.semantic_rules import (
        get_semantic_parameter_set,
    )

    assert "model" not in get_tunable_parameter_names()
    assert "model" not in get_semantic_parameter_set()

    reference_model = "detect/train63/weights/best.pt"
    package = {
        "schema_version": "1.0", "fact_package_id": "sha256:test",
        "task": "detect", "reference_run": "train63", "sources": {},
        "facts": [
            {"fact_id": "training.issue.underfitting", "value": True,
             "source": "issue"},
            {"fact_id": "training.params.model", "value": reference_model,
             "source": "params"},
        ],
    }
    decision = {
        "schema_version": "1.0", "fact_package_id": "sha256:test",
        "diagnosis": "d", "action": "adjust",
        "hyperparameter_changes": {"model": "yolo11n.pt"},
        "training_overrides": {},
        "evidence_ids": {"model": ["training.issue.underfitting"]},
    }

    # 第一道边界：结构化契约直接拒绝，违规响应不产生任何 decision
    with pytest.raises(DecisionContractError) as excinfo:
        parse_tuning_decision_response(json.dumps(decision))
    assert excinfo.value.error_code == "DECISION_SCHEMA_INVALID"
    assert "model" in excinfo.value.detail

    # 纵深防御：即便绕过解析器构造出该 decision，语义层也没有任何 model 规则
    result = validate_decision_semantics(decision, package)
    assert result["valid"] is False
    assert result["parameter"] == "model"

    # 执行参数里的 model 只能来自参考运行
    executed, guard = sanitize_and_merge_tuning_params(
        {"model": reference_model, "batch": 16, "imgsz": 640, "lr0": 0.01},
        {}, {}, {"total_images": 230})
    assert guard.valid is True
    assert executed["model"] == reference_model
    assert executed["model"] != "yolo11n.pt"
