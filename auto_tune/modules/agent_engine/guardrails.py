"""Guardrails — hyperparameter validation, bound clamping, and conflict detection.

Every LLM-suggested hyperparameter change passes through here before execution.
Invalid values are clamped with a warning; conflicting combinations raise errors.
"""

from dataclasses import dataclass, field
import math
from typing import Any

from .parameter_registry import PARAMETER_REGISTRY

# ── Parameter bounds ──────────────────────────────────────────────
BOUNDS: dict[str, tuple[float, float]] = {
    "lr0": (1e-5, 0.1),
    "lrf": (1e-5, 0.1),
    "momentum": (0.6, 0.99),
    "weight_decay": (0.0, 0.1),
    "warmup_epochs": (0.0, 10.0),
    "warmup_momentum": (0.0, 0.99),
    "warmup_bias_lr": (0.0, 0.2),
    "box": (1.0, 20.0),
    "cls": (0.1, 5.0),
    "dfl": (0.5, 5.0),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 1.0),
    "scale": (0.0, 1.0),
    "shear": (0.0, 45.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "dropout": (0.0, 0.5),
    "batch": (1, 256),
    "epochs": (1, 1000),
    "patience": (0, 200),
}

INT_PARAMS = {"batch", "epochs", "patience", "warmup_epochs", "close_mosaic"}

OPTIMIZER_CHOICES = {"SGD", "AdamW", "Adam", "auto", "Adamax", "NAdam", "RAdam"}

# Known YOLO args.yaml parameters that should pass through without warnings
YOLO_INTERNAL_PARAMS = {
    "task", "mode", "model", "data", "time", "imgsz", "save", "save_period",
    "cache", "device", "workers", "project", "name", "exist_ok", "pretrained",
    "optimizer", "verbose", "seed", "deterministic", "single_cls", "rect",
    "cos_lr", "close_mosaic", "resume", "amp", "fraction", "profile", "freeze",
    "multi_scale", "overlap_mask", "mask_ratio", "val", "split", "save_json",
    "save_hybrid", "conf", "iou", "max_det", "half", "dnn", "plots", "source",
    "vid_stride", "stream_buffer", "visualize", "augment", "agnostic_nms",
    "classes", "retina_masks", "embed", "show", "save_frames", "save_txt",
    "save_conf", "save_crop", "show_labels", "show_conf", "show_boxes",
    "line_width", "format", "keras", "optimize", "int8", "dynamic", "simplify",
    "opset", "workspace", "nms", "pose", "kobj", "label_smoothing", "nbs",
    "bgr", "shear", "perspective", "flipud", "fliplr", "auto_augment",
    "erasing", "crop_fraction", "cfg", "tracker", "save_dir",
    "_old_batch", "_skip", "time", "close_mosaic",
}


@dataclass
class GuardResult:
    """Result of guardrail validation."""
    valid: bool = True
    clamped: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


def validate_and_clamp(params: dict[str, Any], dataset_info: dict | None = None) -> GuardResult:
    """Validate and clamp hyperparameter changes.

    Args:
        params: dict of parameter name -> suggested value.
        dataset_info: optional dict with dataset metadata (total_images, etc.)
                      used for context-aware constraints.

    Returns:
        GuardResult with clamped values and any warnings/errors.
    """
    result = GuardResult()
    clamped = dict(params)

    # 1. Bound check & clamp
    for key, value in list(clamped.items()):
        if key not in BOUNDS:
            if key not in YOLO_INTERNAL_PARAMS:
                result.warnings.append(f"Unknown parameter '{key}' — passed through unchanged")
            continue

        lo, hi = BOUNDS[key]
        if key in INT_PARAMS:
            value = round(value)
        if value < lo:
            clamped[key] = lo if key not in INT_PARAMS else int(lo)
            result.warnings.append(f"'{key}'={value} clamped to lower bound {lo}")
            result.clamped[key] = clamped[key]
        elif value > hi:
            clamped[key] = hi if key not in INT_PARAMS else int(hi)
            result.warnings.append(f"'{key}'={value} clamped to upper bound {hi}")
            result.clamped[key] = clamped[key]

    # 2. Type checks
    optimizer = clamped.get("optimizer")
    if optimizer is not None and optimizer not in OPTIMIZER_CHOICES:
        result.warnings.append(
            f"Optimizer '{optimizer}' not in {sorted(OPTIMIZER_CHOICES)}, reset to 'auto'"
        )
        clamped["optimizer"] = "auto"

    # 3. Constraint: AdamW + high lr
    if optimizer and optimizer.upper() == "ADAMW":
        lr = clamped.get("lr0", 0.01)
        if lr > 0.005:
            result.warnings.append(
                f"AdamW with lr0={lr} > 0.005 may cause instability; consider lr0 <= 0.001"
            )

    # 4. Constraint: batch vs lr scaling hint
    old_batch = clamped.get("_old_batch")
    new_batch = clamped.get("batch")
    if old_batch and new_batch and old_batch != new_batch and "lr0" not in clamped:
        ratio = new_batch / old_batch
        if abs(ratio - 1.0) > 0.1:
            result.warnings.append(
                f"batch_size changed {old_batch} -> {new_batch}, "
                f"consider scaling lr0 by ~{ratio:.2f}x (linear scaling rule)"
            )

    # 5. Constraint: over-regularization
    dropout = clamped.get("dropout", 0.0)
    wd = clamped.get("weight_decay", 0.0005)
    if dropout > 0.2 and wd > 0.001:
        result.errors.append(
            f"Over-regularization: dropout={dropout} > 0.2 AND weight_decay={wd} > 0.001. "
            "Reduce at least one of them."
        )

    # 6. Constraint: strong augmentation + small dataset
    if dataset_info:
        n_images = dataset_info.get("total_images", 0)
        if 0 < n_images < 500:
            strong_aug = []
            if clamped.get("mosaic", 0) > 0.5:
                strong_aug.append(f"mosaic={clamped['mosaic']}")
            if clamped.get("mixup", 0) > 0.3:
                strong_aug.append(f"mixup={clamped['mixup']}")
            if clamped.get("degrees", 0) > 15:
                strong_aug.append(f"degrees={clamped['degrees']}")
            if strong_aug:
                result.warnings.append(
                    f"Small dataset ({n_images} images) with strong augmentation "
                    f"({' , '.join(strong_aug)}) may cause underfitting. "
                    "Consider reducing augmentation strength."
                )

    # 7. Validate JSON-serializable types
    for key, value in list(clamped.items()):
        if not isinstance(value, (int, float, str, bool)):
            result.errors.append(f"'{key}' has unsupported type {type(value).__name__}")

    result.valid = len(result.errors) == 0
    result.params = {k: v for k, v in clamped.items() if not k.startswith("_")}
    return result


def _normalize_value(key: str, value: Any, result: GuardResult) -> Any:
    """Normalize one LLM value according to the central parameter registry."""
    spec = PARAMETER_REGISTRY[key]
    if spec.kind in {"int", "float"}:
        if isinstance(value, bool):
            result.errors.append(f"'{key}' must be numeric, not bool")
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            result.errors.append(f"'{key}' must be numeric")
            return None
        if not math.isfinite(normalized):
            result.errors.append(f"'{key}' must be finite")
            return None
        normalized = max(spec.minimum, min(spec.maximum, normalized))
        if normalized != float(value):
            result.warnings.append(f"'{key}'={value} clamped to {normalized}")
            result.clamped[key] = int(normalized) if spec.kind == "int" else normalized
        return int(round(normalized)) if spec.kind == "int" else normalized
    if spec.kind == "choice":
        if not isinstance(value, str) or value not in spec.choices:
            result.errors.append(f"'{key}' must be one of {sorted(spec.choices)}")
            return None
        return value
    if spec.kind == "bool":
        if not isinstance(value, bool):
            result.errors.append(f"'{key}' must be bool")
            return None
        return value
    if spec.kind == "string":
        if not isinstance(value, str) or not value.strip():
            result.errors.append(f"'{key}' must be a non-empty string")
            return None
        return value.strip()
    result.errors.append(f"Unsupported parameter type for '{key}'")
    return None


def sanitize_tuning_parameters(
    hyperparameter_changes: dict[str, Any],
    training_overrides: dict[str, Any] | None = None,
    dataset_info: dict | None = None,
) -> GuardResult:
    """Validate both LLM parameter sections and return the only executable values."""
    result = GuardResult()
    candidate = dict(hyperparameter_changes or {})
    candidate.update(training_overrides or {})
    normalized: dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in PARAMETER_REGISTRY:
            result.errors.append(f"Unknown parameter '{key}'")
            continue
        converted = _normalize_value(key, value, result)
        if converted is not None:
            normalized[key] = converted

    if normalized.get("optimizer") == "auto" and ({"lr0", "momentum"} & normalized.keys()):
        result.errors.append("optimizer=auto causes Ultralytics to ignore explicit lr0/momentum")

    if normalized.get("optimizer", "").upper() == "ADAMW" and normalized.get("lr0", 0.0) > 0.005:
        result.warnings.append("AdamW with lr0 > 0.005 may be unstable")

    if dataset_info and 0 < dataset_info.get("total_images", 0) < 500:
        if normalized.get("mosaic", 0) > 0.5:
            result.warnings.append("Small dataset with mosaic > 0.5 may underfit")

    result.params = normalized
    result.valid = not result.errors
    return result


def merge_params(base_args: dict, changes: dict) -> dict:
    """Merge validated hyperparameter changes into base args.

    Args:
        base_args: original args.yaml as dict.
        changes: validated hyperparameter changes (from LLM + guardrails).

    Returns:
        Merged dict safe to write to args.yaml.
    """
    merged = dict(base_args)
    # Remove internal keys before merge
    clean = {k: v for k, v in changes.items() if not k.startswith("_")}
    merged.update(clean)
    return merged
