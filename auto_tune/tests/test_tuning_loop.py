"""Regression tests for safe parameter assembly and audit failure policy."""

import json
import threading
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine import loop
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


def test_failure_helper_returns_exact_dict():
    assert _failure("decision", "decision_schema_error", "invalid JSON") == {
        "stage": "decision",
        "error_type": "decision_schema_error",
        "message": "invalid JSON",
        "fatal": True,
    }


def test_decision_error_is_fatal_and_never_launches_training(tmp_path, monkeypatch):
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

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None):
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

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None):
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
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None):
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
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, session_id=None, audit_path=None, started_at=None, finished_at=None, tuning_context=None):
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
