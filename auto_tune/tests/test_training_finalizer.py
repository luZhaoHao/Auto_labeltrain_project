"""Tests for the shared training finalizer (Module B + KPI + history)."""

import json
from pathlib import Path

from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run


def _make_run(tmp_path, name, csv_content="epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.3,0.4,0.2,0.1\n"):
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True)
    (run_dir / "args.yaml").write_text("model: yolov8n.pt\nlr0: 0.01\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(csv_content, encoding="utf-8")
    return str(run_dir)


def test_successful_finalization(tmp_path):
    run_dir = _make_run(tmp_path, "train1")
    log_dir = tmp_path / "log"

    result = finalize_training_run(
        run_dir, "train1", "manual", {"train_analyzer": {}}, log_dir=str(log_dir)
    )

    assert result["status"] == "completed"
    assert result["analysis_status"] == "completed"
    assert result["metrics"] == {
        "mAP50": 0.2,
        "mAP50_95": 0.1,
        "precision": 0.3,
        "recall": 0.4,
    }
    assert result["epochs"] == {"configured": None, "completed": 1, "best": 1}
    assert Path(result["artifacts"]["report_path"]).exists()
    assert result["run_id"] == "manual:train1"
    history = json.loads((log_dir / "experiment_history.json").read_text("utf-8"))
    assert len(history["experiments"]) == 1
    assert history["experiments"][0]["run_id"] == "manual:train1"


def test_finalize_completed_with_bad_csv(tmp_path):
    run_dir = _make_run(tmp_path, "train2", csv_content="not,a,csv\n")

    result = finalize_training_run(
        run_dir, "train2", "manual", {"train_analyzer": {}}, log_dir=str(tmp_path / "log")
    )

    assert result["status"] == "completed"
    assert result["analysis_status"] == "failed"
    assert result["analysis_error"]["stage"] == "analysis"
    assert result["analysis_error"]["error_type"] == "analysis_failed"
    assert result["analysis_error"]["timestamp"].endswith("Z")


def test_finalize_failed_training_skips_analysis(tmp_path):
    run_dir = _make_run(tmp_path, "train3")

    result = finalize_training_run(
        run_dir, "train3", "manual", {}, log_dir=str(tmp_path / "log"),
        training_status="failed",
    )

    assert result["status"] == "failed"
    assert result["analysis_status"] == "skipped"


def test_finalize_cancelled_training(tmp_path):
    run_dir = _make_run(tmp_path, "train4")

    result = finalize_training_run(
        run_dir, "train4", "manual", {}, log_dir=str(tmp_path / "log"),
        training_status="cancelled",
    )

    assert result["status"] == "cancelled"
    assert result["analysis_status"] == "skipped"


def test_finalize_keeps_zero_and_missing_metrics(tmp_path):
    run_dir = _make_run(
        tmp_path, "train5",
        csv_content="epoch,metrics/precision(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.0,0.2,0.1\n",
    )

    result = finalize_training_run(
        run_dir, "train5", "manual", {}, log_dir=str(tmp_path / "log")
    )

    assert result["metrics"]["precision"] == 0.0
    assert "recall" not in result["metrics"]


def test_finalize_history_persistence_failure(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, "train6")

    import auto_tune.modules.train_analyzer.experiment_history as eh

    real = eh.atomic_write_json

    def flaky(path, payload):
        if "experiment_history" in str(path):
            raise OSError("disk full")
        return real(path, payload)

    monkeypatch.setattr(eh, "atomic_write_json", flaky)

    result = finalize_training_run(
        run_dir, "train6", "manual", {}, log_dir=str(tmp_path / "log")
    )

    assert result["status"] == "completed"
    assert result["analysis_status"] == "completed"
    assert result["history_error"]["error_type"] == "history_persistence_error"


def test_finalize_epochs_structure_with_best(tmp_path):
    run_dir = tmp_path / "detect" / "train7"
    run_dir.mkdir(parents=True)
    (run_dir / "args.yaml").write_text("model: yolov8n.pt\nepochs: 100\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B)\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8"
    )

    result = finalize_training_run(
        str(run_dir), "train7", "manual", {}, log_dir=str(tmp_path / "log")
    )

    assert result["epochs"] == {"configured": 100, "completed": 3, "best": 3}


def test_finalize_records_training_error(tmp_path):
    run_dir = _make_run(tmp_path, "train8")
    training_error = {
        "stage": "training",
        "error_type": "training_process_failed",
        "message": "训练进程退出码 1",
        "timestamp": "2026-08-01T00:00:00Z",
    }

    result = finalize_training_run(
        str(run_dir), "train8", "manual", {}, log_dir=str(tmp_path / "log"),
        training_status="failed",
        training_error=training_error,
    )

    assert result["status"] == "failed"
    assert result["analysis_status"] == "skipped"
    assert result["error"] == training_error


def test_finalize_tuning_stores_structured_tuning_context(tmp_path):
    run_dir = _make_run(tmp_path, "autotune_1")
    tuning_context = {
        "decision": {
            "diagnosis": "学习率偏高",
            "action": "adjust",
            "hyperparameter_changes": {"lr0": 0.001},
            "training_overrides": {"workers": 2},
        },
        "guardrails": {
            "valid": True,
            "warnings": ["lr0 从 0.005 约束到 0.001"],
            "errors": [],
            "clamped": {"lr0": 0.001},
        },
    }

    result = finalize_training_run(
        str(run_dir), "autotune_1", "tuning", {}, log_dir=str(tmp_path / "log"),
        session_id="s1", tuning_context=tuning_context,
    )

    assert result["tuning"] == tuning_context
    history = json.loads((tmp_path / "log" / "experiment_history.json").read_text("utf-8"))
    stored = history["experiments"][0]
    assert stored["tuning"] == tuning_context
    assert stored["tuning"]["decision"]["diagnosis"] == "学习率偏高"
    assert stored["tuning"]["guardrails"]["valid"] is True


def test_finalize_manual_has_no_tuning_field(tmp_path):
    run_dir = _make_run(tmp_path, "train_m")

    result = finalize_training_run(
        str(run_dir), "train_m", "manual", {}, log_dir=str(tmp_path / "log")
    )

    assert "tuning" not in result
    history = json.loads((tmp_path / "log" / "experiment_history.json").read_text("utf-8"))
    assert "tuning" not in history["experiments"][0]
