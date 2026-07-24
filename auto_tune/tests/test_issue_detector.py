"""Tests for issue_detector.py."""

import pytest
from auto_tune.modules.train_analyzer.issue_detector import detect_issues


def _make_run_data(name="train", epochs=10,
                   train_box=None, val_box=None, mAP=None,
                   map_final=None):
    """Build a run_data dict with synthetic results."""
    n = epochs
    train_box = train_box or [2.0 - i * 0.15 for i in range(n)]
    val_box = val_box or [2.1 - i * 0.1 for i in range(n)]
    mAP = mAP or [0.05 + i * 0.02 for i in range(n)]
    if map_final is not None:
        mAP[-1] = map_final

    return {
        "name": name,
        "args": {"epochs": 100, "patience": 15},
        "results": {
            "columns": {
                "epoch": list(range(1, n + 1)),
                "train/box_loss": train_box,
                "train/cls_loss": [3.0 - i * 0.2 for i in range(n)],
                "train/dfl_loss": [2.0 - i * 0.1 for i in range(n)],
                "val/box_loss": val_box,
                "val/cls_loss": [3.1 - i * 0.15 for i in range(n)],
                "val/dfl_loss": [2.1 - i * 0.08 for i in range(n)],
                "metrics/mAP50(B)": mAP,
                "metrics/mAP50-95(B)": [v * 0.3 for v in mAP],
                "metrics/precision(B)": [0.5 + i * 0.02 for i in range(n)],
                "metrics/recall(B)": [0.5 + i * 0.02 for i in range(n)],
            },
            "total_epochs": n,
        },
    }


def test_detect_no_issues():
    run_data = _make_run_data(epochs=10)
    issues = detect_issues(run_data, {"min_acceptable_map": 0.5, "stale_threshold": 15})
    # Should have underfitting (still improving) and low map
    types = [i["type"] for i in issues]
    assert "underfitting" in types
    assert "low_final_map" in types


def test_detect_low_final_map():
    run_data = _make_run_data(epochs=10, map_final=0.05)
    issues = detect_issues(run_data, {"min_acceptable_map": 0.5})
    types = [i["type"] for i in issues]
    assert "low_final_map" in types


def test_detect_overfitting():
    run_data = _make_run_data(
        epochs=10,
        train_box=[2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        val_box=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8],
    )
    issues = detect_issues(run_data, {})
    types = [i["type"] for i in issues]
    assert "overfitting" in types


def test_detect_nan_loss():
    run_data = _make_run_data(epochs=5)
    run_data["results"]["columns"]["train/box_loss"][2] = None
    issues = detect_issues(run_data, {})
    types = [i["type"] for i in issues]
    assert "nan_loss" in types


def test_detect_parse_error():
    run_data = {
        "name": "bad_run",
        "args": {},
        "results": {"error": "results.csv not found"},
    }
    issues = detect_issues(run_data, {})
    assert len(issues) == 1
    assert issues[0]["type"] == "parse_error"


def test_detect_underfitting():
    # mAP still improving at last epoch
    mAP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.47, 0.49]
    run_data = _make_run_data(epochs=10, mAP=mAP)
    issues = detect_issues(run_data, {"min_acceptable_map": 0.5})
    types = [i["type"] for i in issues]
    assert "underfitting" in types or "low_final_map" in types


def test_detect_early_stop_too_soon():
    # Improving mAP but stopped early
    mAP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.47, 0.49]
    run_data = _make_run_data(epochs=10, mAP=mAP)
    run_data["args"] = {"epochs": 100, "patience": 15}
    run_data["results"]["total_epochs"] = 10
    issues = detect_issues(run_data, {"min_acceptable_map": 0.5, "stale_threshold": 15})
    types = [i["type"] for i in issues]
    assert "early_stop_too_soon" in types
