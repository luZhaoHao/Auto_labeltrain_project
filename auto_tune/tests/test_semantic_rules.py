"""Q1.2 — Detect 语义规则注册表与提示词摘要。

注册表是语义校验器和自动调优提示词的唯一运行时来源：提示词不得手写第二套
事实—参数映射。rule_id / fact_id / 方向令牌 / 幅度限制必须稳定且来自固定常量，
不得拼接任意 LLM 输入。
"""

import pytest

from auto_tune.modules.agent_engine.semantic_rules import (
    DETECT_SEMANTIC_RULES,
    build_parameter_rule_summary,
    build_semantic_rule_summary,
    get_semantic_parameter_set,
)

VALID_DIRECTIONS = frozenset({"increase", "decrease", "enable"})
VALID_LIMIT_KINDS = frozenset({"ratio_range", "max_multiple", "min_fraction", "max_add", "bool"})


def test_registry_is_non_empty_and_stable():
    assert DETECT_SEMANTIC_RULES
    rule_ids = [r.rule_id for r in DETECT_SEMANTIC_RULES]
    assert len(rule_ids) == len(set(rule_ids))  # unique stable ids


@pytest.mark.parametrize("rule", DETECT_SEMANTIC_RULES, ids=lambda r: r.rule_id)
def test_every_rule_is_well_formed(rule):
    assert rule.rule_id
    assert rule.fact_id
    assert rule.parameter
    assert rule.allowed_direction in VALID_DIRECTIONS
    assert rule.change_limit.kind in VALID_LIMIT_KINDS
    # rule_id / fact_id / parameter are fixed registry constants, never dynamic
    assert "\n" not in rule.rule_id and "\n" not in rule.fact_id and "\n" not in rule.parameter


def test_detect_rule_count_matches_spec():
    assert len(DETECT_SEMANTIC_RULES) == 21


def test_supported_parameter_set():
    supported = get_semantic_parameter_set()
    for param in ("weight_decay", "epochs", "lr0", "cos_lr", "warmup_epochs",
                  "patience", "imgsz", "box", "cls", "translate"):
        assert param in supported
    for param in ("model", "optimizer", "batch", "mosaic", "mixup", "copy_paste"):
        assert param not in supported


def test_summary_contains_every_rule_relation():
    summary = build_semantic_rule_summary()
    for rule in DETECT_SEMANTIC_RULES:
        assert rule.fact_id in summary
        assert rule.parameter in summary


def test_summary_marks_unsupported_parameters():
    summary = build_semantic_rule_summary()
    for param in ("model", "optimizer", "batch"):
        assert param in summary


def test_parameter_summary_contains_only_that_parameter_rules():
    wd_summary = build_parameter_rule_summary("weight_decay")
    assert "training.issue.overfitting" in wd_summary
    assert "training.issue.underfitting" in wd_summary
    assert "training.curve.val_box_loss" in wd_summary
    assert "epochs" not in wd_summary
    lr0_summary = build_parameter_rule_summary("lr0")
    assert "training.issue.plateau" in lr0_summary
    assert "weight_decay" not in lr0_summary


def test_summary_is_derived_from_registry_not_hardcoded():
    # The summary must be a pure function of the registry: removing a rule
    # removes its relation from the summary.
    import auto_tune.modules.agent_engine.semantic_rules as mod

    orig = DETECT_SEMANTIC_RULES
    try:
        reduced = tuple(r for r in orig if "unstable_training" not in r.rule_id)
        mod.DETECT_SEMANTIC_RULES = reduced
        assert "training.issue.unstable_training" not in build_semantic_rule_summary()
    finally:
        mod.DETECT_SEMANTIC_RULES = orig
