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
