"""H1.1 Detect HPO 条件搜索空间与候选映射。

版本化协议：按固定顺序 suggest；SGD momentum 与 AdamW beta1 使用独立采样键，
统一映射为训练候选的 ``momentum``。``validate_candidate`` 严格校验，绝不静默
clamp、不重采样掩盖非法候选。
"""

from typing import Any

from .models import (
    EVALUATION_WEIGHTS,
    LEGACY_MODE,
    LEGACY_OBJECTIVE,
    OBJECTIVE_BY_MODE,
    SEARCH_SPACE_VERSION,
    HpoError,
)

CANDIDATE_KEYS = frozenset({
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
})

OPTIMIZER_CHOICES = ("SGD", "AdamW")

# (low, high) per optimizer for the mapped training momentum key.
MOMENTUM_RANGE = {
    "SGD": (0.8, 0.98),
    "AdamW": (0.85, 0.95),
}

LR0_RANGE = (1e-5, 0.005)
LRF_RANGE = (0.01, 0.1)
WEIGHT_DECAY_RANGE = (0.0, 0.001)
WARMUP_MAX_CAP = 5


def suggest_candidate(trial: Any, *, epochs: int) -> tuple[dict, dict]:
    """按版本化顺序在 ``trial`` 上采样，返回 (sampled_params, candidate_params)。

    ``sampled_params`` 保存 Optuna 原名；``candidate_params`` 恰好为六个训练键。
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise HpoError("HPO_INVALID_CONFIG", "epochs must be a positive int")

    optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZER_CHOICES))
    lr0 = trial.suggest_float("lr0", LR0_RANGE[0], LR0_RANGE[1], log=True)
    lrf = trial.suggest_float("lrf", LRF_RANGE[0], LRF_RANGE[1], log=True)
    lo, hi = MOMENTUM_RANGE[optimizer]
    momentum_key = "momentum_sgd" if optimizer == "SGD" else "beta1_adamw"
    momentum = trial.suggest_float(momentum_key, lo, hi)
    weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAY_RANGE[0],
                                       WEIGHT_DECAY_RANGE[1])
    warmup = trial.suggest_int("warmup_epochs", 0, min(WARMUP_MAX_CAP, epochs - 1))

    sampled = {
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        momentum_key: momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup,
    }
    candidate = {
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup,
    }
    return sampled, candidate


OBJECTIVE_CODE = LEGACY_OBJECTIVE
OBJECTIVE_LABEL = "验证集最佳 epoch 的 mAP50-95"

# 评价模式的用户可读说明（实际目标版本码与权重来自 models.py 的版本化契约）。
EVALUATION_MODE_LABELS = {
    LEGACY_MODE: "旧版 mAP50-95（单指标）",
    "quick": "快速（mAP50 × 0.10 + mAP50-95 × 0.90）",
    "comprehensive": (
        "全面（mAP50 × 0.10 + mAP50-95 × 0.50 + Precision × 0.20 + Recall × 0.20）"),
}


def evaluation_mode_label(mode: str) -> str:
    """评价模式的中文说明；未知模式回退为版本码本身（不伪造）。"""
    return EVALUATION_MODE_LABELS.get(mode, str(mode))


def search_space_summary(*, epochs: int, sampler: str,
                         evaluation_mode: str = LEGACY_MODE) -> dict:
    """只读摘要：供 UI 展示搜索配置与评价方式。

    完全由本模块的运行时常量与条件化规则导出，不重新定义范围、不改变采样行为；
    仅供展示（前端不再自行硬编码上下限）。``momentum`` 与 ``warmup_epochs``
    标记为条件参数：前者按 optimizer 分支，后者上限随 epochs 变化。评价模式与
    权重同样来自版本化契约，前端只负责显示与选择。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        epochs = 1
    if evaluation_mode not in EVALUATION_WEIGHTS:
        evaluation_mode = LEGACY_MODE
    return {
        "search_space_version": SEARCH_SPACE_VERSION,
        "sampler": sampler,
        "evaluation_mode": evaluation_mode,
        "evaluation_mode_label": evaluation_mode_label(evaluation_mode),
        "objective": OBJECTIVE_BY_MODE[evaluation_mode],
        "objective_label": EVALUATION_MODE_LABELS[evaluation_mode],
        "direction": "maximize",
        "score": {
            "objective": OBJECTIVE_BY_MODE[evaluation_mode],
            "weights": dict(EVALUATION_WEIGHTS[evaluation_mode]),
            "tie_break": "earliest_best_epoch",
        },
        "parameters": {
            "optimizer": {
                "kind": "choice",
                "choices": list(OPTIMIZER_CHOICES),
                "conditional": True,
                "condition_label": "SGD/AdamW 各自动量区间",
            },
            "lr0": {"kind": "float", "low": LR0_RANGE[0], "high": LR0_RANGE[1],
                    "log": True, "conditional": False},
            "lrf": {"kind": "float", "low": LRF_RANGE[0], "high": LRF_RANGE[1],
                    "log": True, "conditional": False},
            "momentum": {
                "kind": "float",
                "conditional": True,
                "condition_label": "按 optimizer 选择区间",
                "ranges": {name: {"low": lo, "high": hi}
                           for name, (lo, hi) in MOMENTUM_RANGE.items()},
            },
            "weight_decay": {"kind": "float", "low": WEIGHT_DECAY_RANGE[0],
                             "high": WEIGHT_DECAY_RANGE[1], "conditional": False},
            "warmup_epochs": {
                "kind": "int", "low": 0,
                "high": min(WARMUP_MAX_CAP, epochs - 1),
                "conditional": True,
                "condition_label": "上限 = min(5, epochs-1)",
                "epochs_dependent": True,
            },
        },
    }


def _finite_real(key: str, value: Any) -> None:
    import math
    if isinstance(value, bool) or type(value) not in (int, float):
        raise HpoError("HPO_INVALID_CONFIG",
                       f"candidate '{key}' must be a native int/float")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HpoError("HPO_INVALID_CONFIG",
                       f"candidate '{key}' must be a finite number") from exc
    if not math.isfinite(f):
        raise HpoError("HPO_INVALID_CONFIG",
                       f"candidate '{key}' must be finite")


def validate_candidate(params: dict, *, epochs: int) -> dict:
    """严格校验映射后的六个训练键候选；非法即抛 HPO_INVALID_CONFIG。

    只读、不修改输入，返回传入字典的副本。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise HpoError("HPO_INVALID_CONFIG", "epochs must be a positive int")
    if not isinstance(params, dict) or set(params) != CANDIDATE_KEYS:
        raise HpoError("HPO_INVALID_CONFIG",
                       "candidate must contain exactly the six training keys")

    optimizer = params["optimizer"]
    if not isinstance(optimizer, str) or optimizer not in OPTIMIZER_CHOICES:
        raise HpoError("HPO_INVALID_CONFIG",
                       f"optimizer must be one of {list(OPTIMIZER_CHOICES)}")

    for key in ("lr0", "lrf", "momentum", "weight_decay"):
        _finite_real(key, params[key])
    value = params["lr0"]
    if not (LR0_RANGE[0] <= float(value) <= LR0_RANGE[1]):
        raise HpoError("HPO_INVALID_CONFIG", "lr0 out of search-space range")
    value = params["lrf"]
    if not (LRF_RANGE[0] <= float(value) <= LRF_RANGE[1]):
        raise HpoError("HPO_INVALID_CONFIG", "lrf out of search-space range")
    value = params["weight_decay"]
    if not (WEIGHT_DECAY_RANGE[0] <= float(value) <= WEIGHT_DECAY_RANGE[1]):
        raise HpoError("HPO_INVALID_CONFIG", "weight_decay out of search-space range")

    lo, hi = MOMENTUM_RANGE[optimizer]
    momentum = float(params["momentum"])
    if not (lo <= momentum <= hi):
        raise HpoError("HPO_INVALID_CONFIG",
                       f"momentum out of {optimizer} search-space range {lo}..{hi}")

    warmup = params["warmup_epochs"]
    if isinstance(warmup, bool) or type(warmup) is not int:
        raise HpoError("HPO_INVALID_CONFIG", "warmup_epochs must be an int")
    upper = min(WARMUP_MAX_CAP, epochs - 1)
    if not (0 <= warmup <= upper):
        raise HpoError("HPO_INVALID_CONFIG",
                       f"warmup_epochs must be within 0..{upper}")

    return dict(params)
