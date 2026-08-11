"""Training issue detection from YOLO results."""

from .curve_analysis import analyze_loss_curves, analyze_metric_curves


def _check_nan_loss(results: dict) -> bool:
    """Check if any loss columns contain NaN values."""
    columns = results.get("columns", {})
    for col_name in ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                      "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        vals = columns.get(col_name, [])
        if any(v is None for v in vals):
            return True
    return False


def _check_overfitting(curve_analysis: dict, config: dict) -> bool:
    """Check if val loss rises while train loss descends."""
    return curve_analysis.get("overfitting_detected", False)


def _check_underfitting(metric_analysis: dict, results: dict, config: dict) -> bool:
    """Check if metrics are still improving at the last epoch."""
    map_analysis = metric_analysis.get("mAP50", {})
    if map_analysis.get("trend") == "improving":
        return True
    # Also check if slope is positive in recent epochs
    return False


def _check_plateau(metric_analysis: dict, config: dict) -> bool:
    """Check if metrics have plateaued."""
    map_analysis = metric_analysis.get("mAP50", {})
    return map_analysis.get("trend") == "saturated"


def _check_stale(metric_analysis: dict, config: dict) -> bool:
    """Check if no improvement for stale_threshold epochs."""
    map_analysis = metric_analysis.get("mAP50", {})
    best_epoch = map_analysis.get("best_epoch")
    results_obj = metric_analysis.get("_results", {})
    total_epochs = results_obj.get("total_epochs", 0) if isinstance(results_obj, dict) else 0
    if best_epoch and total_epochs:
        stale_threshold = config.get("stale_threshold", 15)
        return (total_epochs - best_epoch) >= stale_threshold
    return False


def _check_unstable_training(curve_analysis: dict, config: dict) -> bool:
    """Check for large oscillations in val loss."""
    for key in ["val_box", "val_cls"]:
        trend = curve_analysis.get(key, {}).get("trend", "")
        if trend == "rising":
            return True
    return False


def _check_early_stop_too_soon(early_stop: dict) -> bool:
    """Check if early stopping happened while still improving."""
    return early_stop.get("improvement_potential") == "high"


def _check_low_final_map(metric_analysis: dict, config: dict) -> bool:
    """Check if final mAP50 is too low."""
    min_map = config.get("min_acceptable_map", 0.5)
    map_analysis = metric_analysis.get("mAP50", {})
    final_val = map_analysis.get("final")
    if final_val is not None and final_val < min_map:
        return True
    return False


def detect_issues(run_data: dict, config: dict) -> list[dict]:
    """Run all issue detectors on a single training run.

    Args:
        run_data: dict from load_training_run() — has "args", "results" keys.
        config: train_analyzer config dict.

    Returns:
        list of issue dicts: [{"type": "overfitting", "severity": "high", "detail": "..."}, ...]
    """
    issues = []
    results = run_data.get("results", {})
    if "error" in results:
        return [{"type": "parse_error", "severity": "high", "detail": results["error"]}]

    # Compute analyses
    curve_analysis = analyze_loss_curves(results, config)
    metric_analysis = analyze_metric_curves(results, config)

    # NaN loss
    if _check_nan_loss(results):
        issues.append({"type": "nan_loss", "severity": "high",
                       "detail": "Loss values contain NaN — training may have diverged."})

    # Overfitting
    if _check_overfitting(curve_analysis, config):
        issues.append({"type": "overfitting", "severity": "medium",
                       "detail": "Validation loss is rising while training loss descends."})

    # Underfitting
    if _check_underfitting(metric_analysis, results, config):
        issues.append({"type": "underfitting", "severity": "medium",
                       "detail": "mAP50 is still trending upward at the last epoch — consider more epochs."})

    # Plateau
    if _check_plateau(metric_analysis, config):
        issues.append({"type": "plateau", "severity": "low",
                       "detail": "mAP50 has saturated — further training unlikely to improve."})

    # Low final mAP
    if _check_low_final_map(metric_analysis, config):
        issues.append({"type": "low_final_map", "severity": "high",
                       "detail": f"Final mAP50 below threshold ({config.get('min_acceptable_map', 0.5)})."})

    # Unstable training
    if _check_unstable_training(curve_analysis, config):
        issues.append({"type": "unstable_training", "severity": "medium",
                       "detail": "Validation loss shows rising trend — potential divergence."})

    # Early stopping
    early_stop = curve_analysis.get("early_stopping", None)
    if early_stop is None:
        from .curve_analysis import detect_early_stopping
        early_stop = detect_early_stopping(results, config)

    if _check_early_stop_too_soon(early_stop):
        issues.append({"type": "early_stop_too_soon", "severity": "medium",
                       "detail": f"Early stopped at epoch {early_stop.get('actual_epochs', '?')} while mAP50 was still improving. Recommended: {early_stop.get('recommended_epochs', '?')} epochs."})

    return issues
