"""Loss and metric curve analysis for YOLO training results."""

import numpy as np


def _moving_average(values: list, window: int) -> list:
    """Compute moving average, handling None values."""
    arr = np.array([v if v is not None else np.nan for v in values])
    if window < 1 or len(arr) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(np.nan_to_num(arr), kernel, mode="valid")
    # Pad front to match original length
    pad = [np.nan] * (window - 1)
    return list(pad) + list(smoothed)


def _trend_slope(values: list, lookback: int = 10) -> float:
    """Compute linear slope over the last N non-None values."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 3:
        return 0.0
    recent = valid[-min(lookback, len(valid)):]
    xs = np.arange(len(recent))
    ys = np.array([v for _, v in recent])
    if np.std(ys) == 0:
        return 0.0
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def analyze_loss_curves(results: dict, config: dict) -> dict:
    """Analyze training and validation loss curve trends.

    Returns:
        dict with per-loss-column analysis:
            train_box: {"trend": "descending"|"plateaued"|"rising", "slope": float}
            train_cls: ...
            val_box: ...
            val_cls: ...
            overfitting_detected: bool
    """
    columns = results.get("columns", {})
    plateau_epochs = config.get("plateau_epochs", 10)
    overfit_threshold = config.get("overfit_threshold", 0.15)

    loss_cols = {
        "train_box": "train/box_loss",
        "train_cls": "train/cls_loss",
        "train_dfl": "train/dfl_loss",
        "val_box": "val/box_loss",
        "val_cls": "val/cls_loss",
        "val_dfl": "val/dfl_loss",
    }

    analysis = {}
    for short_name, col_name in loss_cols.items():
        vals = columns.get(col_name, [])
        if not vals or all(v is None for v in vals):
            analysis[short_name] = {"trend": "unknown", "slope": 0.0}
            continue

        slope = _trend_slope(vals)
        if abs(slope) < 0.001:
            trend = "plateaued"
        elif slope < 0:
            trend = "descending"
        else:
            trend = "rising"
        analysis[short_name] = {"trend": trend, "slope": round(slope, 6)}

    # Overfitting detection: val loss rising in last plateau_epochs while train still descending
    overfitting = False
    for val_key, train_key in [("val_box", "train_box"), ("val_cls", "train_cls")]:
        vt = analysis.get(val_key, {}).get("trend", "")
        tt = analysis.get(train_key, {}).get("trend", "")
        if vt == "rising" and tt == "descending":
            overfitting = True
            break

    analysis["overfitting_detected"] = overfitting
    return analysis


def analyze_metric_curves(results: dict, config: dict) -> dict:
    """Analyze performance metric curves (mAP, precision, recall).

    Returns:
        dict with per-metric analysis:
            mAP50: {"trend": "improving"|"saturated"|"oscillating", "slope": float, "best": float, "best_epoch": int}
            ...
    """
    columns = results.get("columns", {})
    stale_threshold = config.get("stale_threshold", 15)

    metric_cols = {
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
    }

    analysis = {}
    for short_name, col_name in metric_cols.items():
        vals = columns.get(col_name, [])
        if not vals or all(v is None for v in vals):
            analysis[short_name] = {"trend": "unknown", "slope": 0.0, "best": None, "best_epoch": None}
            continue

        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            analysis[short_name] = {"trend": "unknown", "slope": 0.0, "best": None, "best_epoch": None}
            continue

        best_val = max(v for _, v in valid)
        best_idx = max(valid, key=lambda x: x[1])[0]
        epoch_col = columns.get("epoch", [])
        best_epoch = int(epoch_col[best_idx]) if epoch_col and best_idx < len(epoch_col) else best_idx + 1

        slope = _trend_slope(vals)
        last_non_none = [v for v in vals if v is not None]

        # Check for stagnation
        recent_vals = [v for _, v in valid if _ >= len(valid) - stale_threshold]
        if len(recent_vals) >= stale_threshold and max(recent_vals) <= best_val * 0.99:
            trend = "saturated"
        elif slope > 0.001:
            trend = "improving"
        elif slope < -0.001:
            trend = "degrading"
        else:
            trend = "saturated"

        # Detect oscillation: alternating up/down in last 10 values
        osc_count = 0
        last_vals = [v for v in vals if v is not None][-10:]
        for i in range(2, len(last_vals)):
            if (last_vals[i] - last_vals[i - 1]) * (last_vals[i - 1] - last_vals[i - 2]) < 0:
                osc_count += 1
        is_oscillating = osc_count >= 4

        analysis[short_name] = {
            "trend": trend,
            "slope": round(slope, 6),
            "best": round(best_val, 4),
            "best_epoch": best_epoch,
            "final": round(last_non_none[-1], 4) if last_non_none else None,
            "oscillating": is_oscillating,
        }

    return analysis


def detect_early_stopping(results: dict, config: dict) -> dict:
    """Analyze if early stopping was appropriate.

    Returns:
        dict with:
            stopped_early: bool
            was_improving: bool (was the metric still trending up at stop?)
            improvement_potential: str ("high"|"low"|"none")
            recommended_epochs: int or None
    """
    columns = results.get("columns", {})
    total_epochs = results.get("total_epochs", 0)
    map_col = columns.get("metrics/mAP50(B)", [])
    valid_map = [(i, v) for i, v in enumerate(map_col) if v is not None]

    if len(valid_map) < 5:
        return {"stopped_early": False, "was_improving": False,
                "improvement_potential": "unknown", "recommended_epochs": None}

    # Check slope in last 10 epochs
    last_vals = [v for _, v in valid_map[-10:]]
    if len(last_vals) < 3:
        return {"stopped_early": False, "was_improving": False,
                "improvement_potential": "unknown", "recommended_epochs": None}

    xs = np.arange(len(last_vals))
    slope = np.polyfit(xs, last_vals, 1)[0]

    was_improving = bool(slope > 0.005)
    args = results.get("args", {})
    planned_epochs = args.get("epochs", 100) if isinstance(args, dict) else 100
    patience = args.get("patience", 15) if isinstance(args, dict) else 15
    stopped_early = total_epochs < planned_epochs

    if not stopped_early:
        return {"stopped_early": False, "was_improving": was_improving,
                "improvement_potential": "low" if was_improving else "none",
                "recommended_epochs": total_epochs}

    if was_improving:
        potential = "high"
        recommended = total_epochs + max(20, patience * 2)
    else:
        potential = "low"
        recommended = total_epochs

    return {
        "stopped_early": True,
        "was_improving": was_improving,
        "improvement_potential": potential,
        "planned_epochs": planned_epochs,
        "actual_epochs": total_epochs,
        "recommended_epochs": recommended,
    }
