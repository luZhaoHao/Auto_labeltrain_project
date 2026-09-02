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


def _reference_run(tmp_path, name="train54"):
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text(
        "model: yolov8n.pt\ndata: test_data.yaml\nlr0: 0.01\nbatch: 16\nepochs: 100\n",
        encoding="utf-8",
    )
    (ref / "results.csv").write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "0, 0.00266, 0.57692, 0.04768, 0.00815\n",
        encoding="utf-8",
    )
    return detect


def _loop_basic_mocks(monkeypatch, tmp_path, reference_run="train54"):
    detect = _reference_run(tmp_path, reference_run)
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
