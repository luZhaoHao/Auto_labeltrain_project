"""Q1.2 — Detect 语义规则注册表（唯一运行时来源）。

本注册表同时是：
- 语义校验器（decision_semantics.py）的规则来源；
- 自动调优初始提示词「允许关系摘要」的来源；
- 语义纠错提示「该参数允许的证据类型/方向/幅度」的来源。

提示词不得手写第二套事实—参数映射。rule_id / fact_id / 方向令牌 / 幅度限制
全部来自下方固定常量，不拼接任意 LLM 输入。首版只支持 YOLOv8 Detect；
Classify 后续使用独立规则集合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parameter_registry import get_tunable_parameter_names

# 方向令牌
DIRECTION_INCREASE = "increase"
DIRECTION_DECREASE = "decrease"
DIRECTION_ENABLE = "enable"  # cos_lr: false -> true

# 幅度限制类型
LIMIT_RATIO_RANGE = "ratio_range"       # new/current 在 [min_ratio, max_ratio]
LIMIT_MAX_MULTIPLE = "max_multiple"     # new <= max_ratio * current（当前为 0 时用 zero_cap）
LIMIT_MIN_FRACTION = "min_fraction"     # new >= min_ratio * current（当前必须 > 0）
LIMIT_MAX_ADD = "max_add"               # new <= current + max_add
LIMIT_BOOL = "bool"                     # 仅布尔转换，无幅度


@dataclass(frozen=True)
class ChangeLimit:
    """结构化单轮变化幅度限制。边界值允许，超过边界拒绝。"""

    kind: str
    min_ratio: float | None = None
    max_ratio: float | None = None
    max_add: float | None = None
    zero_cap: float | None = None
    description: str = ""


@dataclass(frozen=True)
class SemanticRule:
    """一条不可变 Detect 语义规则。"""

    rule_id: str
    fact_id: str
    parameter: str
    allowed_direction: str
    change_limit: ChangeLimit
    description: str = ""
    fact_value: Any | None = None  # 精确事实值条件；None 表示不限定值


def _rule(
    rule_id: str,
    fact_id: str,
    parameter: str,
    allowed_direction: str,
    change_limit: ChangeLimit,
    description: str = "",
    fact_value: Any | None = None,
) -> SemanticRule:
    return SemanticRule(rule_id, fact_id, parameter, allowed_direction,
                        change_limit, description, fact_value)


DETECT_SEMANTIC_RULES: tuple[SemanticRule, ...] = (
    # ── 过拟合：增大 weight_decay / 减少 epochs ──
    _rule("detect.overfitting.weight_decay.increase.v1",
          "training.issue.overfitting", "weight_decay", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=4.0, zero_cap=0.001,
                      description="当前值>0 时新值不超过 4 倍；当前值为 0 时新值不超过 0.001"),
          "过拟合时应增大 weight_decay"),
    _rule("detect.overfitting.epochs.decrease.v1",
          "training.issue.overfitting", "epochs", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_MIN_FRACTION, min_ratio=0.5,
                      description="新值不低于当前值的 50%"),
          "过拟合时应减少 epochs"),
    _rule("detect.val_box_loss.rising.weight_decay.increase.v1",
          "training.curve.val_box_loss", "weight_decay", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=4.0, zero_cap=0.001,
                      description="当前值>0 时新值不超过 4 倍；当前值为 0 时新值不超过 0.001"),
          "val_box_loss 上升时应增大 weight_decay", fact_value="rising"),
    _rule("detect.val_box_loss.rising.epochs.decrease.v1",
          "training.curve.val_box_loss", "epochs", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_MIN_FRACTION, min_ratio=0.5,
                      description="新值不低于当前值的 50%"),
          "val_box_loss 上升时应减少 epochs", fact_value="rising"),
    _rule("detect.val_cls_loss.rising.weight_decay.increase.v1",
          "training.curve.val_cls_loss", "weight_decay", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=4.0, zero_cap=0.001,
                      description="当前值>0 时新值不超过 4 倍；当前值为 0 时新值不超过 0.001"),
          "val_cls_loss 上升时应增大 weight_decay", fact_value="rising"),
    _rule("detect.val_cls_loss.rising.epochs.decrease.v1",
          "training.curve.val_cls_loss", "epochs", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_MIN_FRACTION, min_ratio=0.5,
                      description="新值不低于当前值的 50%"),
          "val_cls_loss 上升时应减少 epochs", fact_value="rising"),
    # ── 欠拟合：减小 weight_decay / 增大 epochs ──
    _rule("detect.underfitting.weight_decay.decrease.v1",
          "training.issue.underfitting", "weight_decay", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_MIN_FRACTION, min_ratio=0.25,
                      description="当前值必须大于 0，新值不低于当前值的 25%"),
          "欠拟合时应减小 weight_decay"),
    _rule("detect.underfitting.epochs.increase.v1",
          "training.issue.underfitting", "epochs", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=2.0,
                      description="新值不超过当前值的 2 倍"),
          "欠拟合时应增大 epochs"),
    # ── plateau：降低 lr0 / 开启 cos_lr ──
    _rule("detect.plateau.lr0.decrease.v1",
          "training.issue.plateau", "lr0", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_RATIO_RANGE, min_ratio=0.25, max_ratio=0.80,
                      description="新值为当前值的 25%–80%"),
          "mAP 停滞时应降低 lr0"),
    _rule("detect.plateau.cos_lr.enable.v1",
          "training.issue.plateau", "cos_lr", DIRECTION_ENABLE,
          ChangeLimit(LIMIT_BOOL, description="只允许 false→true"),
          "mAP 停滞时可开启 cos_lr"),
    # 注意：不存在以 training.curve.mAP50 为事实的规则。TrainAnalyzer 只把
    # analyze_loss_curves 的结果并入报告 curve_analysis，mAP50 趋势从未进入
    # 报告，因此该事实永不进入事实包；挂在它上面的规则无法触发，却会被
    # build_semantic_rule_summary 当作可用关系写进提示词，诱导模型引用不存在
    # 的事实（DECISION_EVIDENCE_UNKNOWN）。mAP50 饱和场景已由 plateau 覆盖。
    # ── 训练不稳定 / NaN：降低 lr0 / 增大 warmup_epochs ──
    _rule("detect.unstable_training.lr0.decrease.v1",
          "training.issue.unstable_training", "lr0", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_RATIO_RANGE, min_ratio=0.25, max_ratio=0.80,
                      description="新值为当前值的 25%–80%"),
          "训练不稳定时应降低 lr0"),
    _rule("detect.unstable_training.warmup_epochs.increase.v1",
          "training.issue.unstable_training", "warmup_epochs", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_ADD, max_add=3.0,
                      description="新值不超过当前值加 3"),
          "训练不稳定时应增大 warmup_epochs"),
    _rule("detect.nan_loss.lr0.decrease.v1",
          "training.issue.nan_loss", "lr0", DIRECTION_DECREASE,
          ChangeLimit(LIMIT_RATIO_RANGE, min_ratio=0.25, max_ratio=0.80,
                      description="新值为当前值的 25%–80%"),
          "出现 NaN loss 时应降低 lr0"),
    _rule("detect.nan_loss.warmup_epochs.increase.v1",
          "training.issue.nan_loss", "warmup_epochs", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_ADD, max_add=3.0,
                      description="新值不超过当前值加 3"),
          "出现 NaN loss 时应增大 warmup_epochs"),
    # ── 过早早停：增大 patience ──
    _rule("detect.early_stop_too_soon.patience.increase.v1",
          "training.issue.early_stop_too_soon", "patience", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=2.0, zero_cap=20,
                      description="当前值>0 时不超过 2 倍；当前值为 0 时新值不超过 20"),
          "过早早停时应增大 patience"),
    # ── 数据集：小目标 / 长尾 / 中心偏置 ──
    _rule("detect.tiny_bbox.imgsz.increase.v1",
          "dataset.issue.tiny_bbox_high_ratio", "imgsz", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=2.0,
                      description="新值不超过当前值 2 倍"),
          "小目标占比高时应增大 imgsz"),
    _rule("detect.tiny_bbox.box.increase.v1",
          "dataset.issue.tiny_bbox_high_ratio", "box", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=2.0,
                      description="新值不超过当前值 2 倍"),
          "小目标占比高时应增大 box"),
    _rule("detect.long_tail.cls.increase.v1",
          "dataset.issue.long_tail_class", "cls", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_MULTIPLE, max_ratio=2.0,
                      description="新值不超过当前值 2 倍"),
          "类别长尾时应增大 cls"),
    _rule("detect.center_spatial_bias.translate.increase.v1",
          "dataset.issue.center_spatial_bias", "translate", DIRECTION_INCREASE,
          ChangeLimit(LIMIT_MAX_ADD, max_add=0.2,
                      description="新值不超过当前值加 0.2"),
          "中心空间偏置时应增大 translate"),
)


def get_semantic_rules() -> tuple[SemanticRule, ...]:
    return DETECT_SEMANTIC_RULES


def get_semantic_parameter_set() -> frozenset[str]:
    """返回当前版本具有语义允许关系的参数集合。"""
    return frozenset(rule.parameter for rule in DETECT_SEMANTIC_RULES)


def get_semantic_evidence_fact_ids() -> frozenset[str]:
    """返回可以被引用为证据的 fact_id 集合（规则注册表里出现过的那些）。

    事实包里还有大量真实但**不可作证据**的事实：参考指标、bbox/图像比例、
    计数、参数当前值。它们只作只读背景——没有任何规则以它们为前提，拿它们
    支持参数修改会被语义校验以 NO_SUPPORTING_RULE 拒绝。
    """
    return frozenset(rule.fact_id for rule in DETECT_SEMANTIC_RULES)


_DIRECTION_TEXT = {
    DIRECTION_INCREASE: "增加",
    DIRECTION_DECREASE: "减少",
    DIRECTION_ENABLE: "false→true",
}


def build_semantic_rule_summary() -> str:
    """从注册表生成初始提示词允许关系摘要（不手写第二套映射）。"""
    lines = [
        "## 允许的超参数修改关系（本批语义规则，唯一事实—参数映射）",
        "只能依据下列关系修改参数；每条修改必须引用下列事实并遵循方向与幅度：",
    ]
    for rule in DETECT_SEMANTIC_RULES:
        value_cond = f"（值={rule.fact_value}）" if rule.fact_value is not None else ""
        direction_text = _DIRECTION_TEXT.get(rule.allowed_direction, rule.allowed_direction)
        lines.append(
            f"- {rule.fact_id}{value_cond} → 允许 {rule.parameter} {direction_text}"
            f"（幅度：{rule.change_limit.description}）"
        )
    supported = sorted(get_semantic_parameter_set())
    unsupported = sorted(get_tunable_parameter_names() - get_semantic_parameter_set())
    lines.append(f"- 可修改参数：{', '.join(supported)}")
    lines.append(f"- 禁止修改参数：{', '.join(unsupported)}")
    lines.append(
        "  「禁止修改参数」在本批没有任何受支持的语义关系：把它们写进"
        " hyperparameter_changes 或 training_overrides 都会以"
        " DECISION_SEMANTIC_UNSUPPORTED 失败，本轮随即终止且不会启动任何训练。"
    )
    return "\n".join(lines)


def build_parameter_rule_summary(parameter: str) -> str:
    """从注册表生成某参数在纠错提示中的允许关系摘要。"""
    lines = [f"参数 {parameter} 的允许关系："]
    matched = [rule for rule in DETECT_SEMANTIC_RULES if rule.parameter == parameter]
    if not matched:
        lines.append("- 该参数没有任何受支持的语义关系，禁止自动修改。")
    for rule in matched:
        value_cond = f"（值={rule.fact_value}）" if rule.fact_value is not None else ""
        direction_text = _DIRECTION_TEXT.get(rule.allowed_direction, rule.allowed_direction)
        lines.append(
            f"- 证据 {rule.fact_id}{value_cond} → 方向 {direction_text}"
            f"（幅度：{rule.change_limit.description}）"
        )
    return "\n".join(lines)
