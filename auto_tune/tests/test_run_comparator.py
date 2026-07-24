"""Tests for run_comparator.py."""

import pytest
from auto_tune.modules.train_analyzer.run_comparator import compare_runs, summarize_runs


def _make_run(name, map50=0.5, epochs=10, best_epoch=8):
    return {
        "name": name,
        "args": {"model": "yolov8s.yaml", "imgsz": [640, 640], "optimizer": "auto"},
        "results": {
            "total_epochs": epochs,
            "best_epoch": best_epoch,
            "final_metrics": {
                "metrics/mAP50(B)": map50,
                "metrics/mAP50-95(B)": map50 * 0.3,
                "metrics/precision(B)": 0.6,
                "metrics/recall(B)": 0.5,
            },
        },
    }


def test_compare_runs():
    runs = [
        _make_run("train", map50=0.3),
        _make_run("train2", map50=0.8),
        _make_run("train3", map50=0.5),
    ]
    result = compare_runs(runs, {"compare_metric": "mAP50", "top_k_runs": 5})
    assert result["best_run"] == "train2"
    assert result["total_runs"] == 3
    assert result["top_runs"][0]["name"] == "train2"
    assert result["top_runs"][0]["final_mAP50"] == 0.8


def test_compare_runs_top_k():
    runs = [_make_run(f"train{i}", map50=i * 0.1) for i in range(10)]
    result = compare_runs(runs, {"compare_metric": "mAP50", "top_k_runs": 3})
    assert len(result["top_runs"]) == 3


def test_compare_runs_empty():
    result = compare_runs([], {})
    assert result["total_runs"] == 0
    assert result["best_run"] is None


def test_summarize_runs():
    runs = [
        _make_run("train", map50=0.3),
        _make_run("train2", map50=0.8),
        _make_run("train3", map50=0.5),
    ]
    for r in runs:
        r["_issues"] = [{"type": "low_final_map"}]
    runs[1]["_issues"] = []

    result = summarize_runs(runs, {})
    assert result["total_runs_analyzed"] == 3
    assert result["best_overall_run"] == "train2"
    assert result["best_mAP50"] == 0.8
    assert result["average_mAP50"] == pytest.approx(0.5333, rel=0.01)
    assert result["runs_with_issues"] == 2
    assert result["common_issues"][0]["type"] == "low_final_map"


def test_summarize_runs_empty():
    result = summarize_runs([], {})
    assert result["total_runs_analyzed"] == 0


def test_summarize_runs_no_issues():
    runs = [_make_run("train", map50=0.9)]
    runs[0]["_issues"] = []
    result = summarize_runs(runs, {})
    assert result["runs_with_issues"] == 0
    assert result["common_issues"] == []
