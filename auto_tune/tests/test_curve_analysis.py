"""Tests for curve_analysis.py."""

import pytest
from auto_tune.modules.train_analyzer.curve_analysis import (
    analyze_loss_curves, analyze_metric_curves, detect_early_stopping,
)


def _make_results(epochs, train_box=None, train_cls=None, val_box=None, mAP=None):
    """Helper to build a fake results dict."""
    n = len(epochs)
    return {
        "columns": {
            "epoch": list(epochs),
            "train/box_loss": train_box if train_box else [1.0] * n,
            "train/cls_loss": train_cls if train_cls else [2.0] * n,
            "train/dfl_loss": [1.5] * n,
            "val/box_loss": val_box if val_box else [1.1] * n,
            "val/cls_loss": [2.1] * n,
            "val/dfl_loss": [1.6] * n,
            "metrics/mAP50(B)": mAP if mAP else [0.1 + i * 0.02 for i in range(n)],
            "metrics/mAP50-95(B)": [v * 0.3 for v in (mAP if mAP else [0.1 + i * 0.02 for i in range(n)])],
            "metrics/precision(B)": [0.5] * n,
            "metrics/recall(B)": [0.5] * n,
        },
        "total_epochs": n,
    }


def test_analyze_loss_curves_descending():
    results = _make_results(
        epochs=[1, 2, 3, 4, 5],
        train_box=[2.0, 1.8, 1.5, 1.2, 1.0],
        val_box=[2.1, 1.9, 1.6, 1.3, 1.1],
    )
    analysis = analyze_loss_curves(results, {"plateau_epochs": 5, "overfit_threshold": 0.15})
    assert analysis["train_box"]["trend"] == "descending"
    assert analysis["val_box"]["trend"] == "descending"
    assert analysis["overfitting_detected"] is False


def test_analyze_loss_curves_overfitting():
    results = _make_results(
        epochs=list(range(1, 11)),
        train_box=[2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        val_box=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4, 1.6, 1.8],
    )
    analysis = analyze_loss_curves(results, {"plateau_epochs": 5, "overfit_threshold": 0.15})
    assert analysis["train_box"]["trend"] == "descending"
    assert analysis["val_box"]["trend"] == "rising"
    assert analysis["overfitting_detected"] is True


def test_analyze_loss_curves_plateau():
    results = _make_results(
        epochs=[1, 2, 3, 4, 5],
        train_box=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    analysis = analyze_loss_curves(results, {"plateau_epochs": 5, "overfit_threshold": 0.15})
    assert analysis["train_box"]["trend"] == "plateaued"


def test_analyze_metric_curves_improving():
    mAP = [0.1, 0.15, 0.2, 0.25, 0.3]
    results = _make_results(epochs=[1, 2, 3, 4, 5], mAP=mAP)
    analysis = analyze_metric_curves(results, {"stale_threshold": 15})
    assert analysis["mAP50"]["trend"] == "improving"
    assert analysis["mAP50"]["best"] == 0.3
    assert analysis["mAP50"]["best_epoch"] == 5


def test_analyze_metric_curves_saturated():
    mAP = [0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    results = _make_results(epochs=list(range(1, 19)), mAP=mAP)
    analysis = analyze_metric_curves(results, {"stale_threshold": 15})
    assert analysis["mAP50"]["trend"] == "saturated"
    assert analysis["mAP50"]["best"] == 0.5


def test_analyze_metric_curves_empty():
    results = {"columns": {}}
    analysis = analyze_metric_curves(results, {})
    assert analysis["mAP50"]["trend"] == "unknown"
    assert analysis["mAP50"]["best"] is None


def test_detect_early_stopping_not_early():
    """Run completed all planned epochs, still improving."""
    mAP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.36, 0.37, 0.38]
    results = _make_results(epochs=list(range(1, 11)), mAP=mAP)
    results["total_epochs"] = 10
    results["args"] = {"epochs": 10, "patience": 5}
    es = detect_early_stopping(results, {})
    assert es["stopped_early"] is False
    assert es["was_improving"] is True


def test_detect_early_stopping_too_soon():
    """Early stopped while still improving."""
    mAP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.43, 0.44]
    results = _make_results(epochs=list(range(1, 11)), mAP=mAP)
    results["total_epochs"] = 10
    results["args"] = {"epochs": 100, "patience": 15}
    es = detect_early_stopping(results, {})
    assert es["stopped_early"] is True
    assert es["was_improving"] is True
    assert es["improvement_potential"] == "high"
