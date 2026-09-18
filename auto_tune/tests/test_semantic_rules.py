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
    # 21 条规则中有 2 条挂在 training.curve.mAP50 上；该曲线事实从不进入报告，
    # 规则无法触发且会误导提示词，已移除，见
    # test_no_rule_targets_a_curve_fact_the_training_report_never_supplies。
    assert len(DETECT_SEMANTIC_RULES) == 19


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
    # 有注册表资格但本轮未开放语义关系的参数，会在提示词中被显式列出
    for param in ("optimizer", "batch"):
        assert param in summary
    # model 已不是可调参数（大模型调优只能继承参考运行权重）：既不在允许列表，
    # 也不再作为「未开放参数」出现在提示词里
    assert "model" not in summary


def test_evidence_eligible_facts_are_exactly_the_rule_facts():
    from auto_tune.modules.agent_engine.semantic_rules import (
        get_semantic_evidence_fact_ids,
    )

    assert get_semantic_evidence_fact_ids() == frozenset(
        rule.fact_id for rule in DETECT_SEMANTIC_RULES)


def test_parameter_summary_contains_only_that_parameter_rules():
    wd_summary = build_parameter_rule_summary("weight_decay")
    assert "training.issue.overfitting" in wd_summary
    assert "training.issue.underfitting" in wd_summary
    assert "training.curve.val_box_loss" in wd_summary
    assert "epochs" not in wd_summary
    lr0_summary = build_parameter_rule_summary("lr0")
    assert "training.issue.plateau" in lr0_summary
    assert "weight_decay" not in lr0_summary


# ── 事实可达性：规则只能挂在训练报告真会提供的曲线事实上 ────────────────────

# perception 把 Module B 的曲线键投影成事实名；这张表就是投影关系本身。
_CURVE_FACT_SOURCES = {
    "val_box_loss": "val_box",
    "val_cls_loss": "val_cls",
    "mAP50": "mAP50",
}


def _module_b_curve_keys() -> set:
    """真实 Module B 报告 ``curve_analysis`` 里会出现的键。

    由真实生产者 ``analyze_loss_curves`` 算出，再并入 analyzer 追加的
    ``early_stopping``；不手工抄一份键名，否则测试会与生产漂移。
    """
    from auto_tune.modules.train_analyzer.curve_analysis import analyze_loss_curves

    n = 20
    results = {"columns": {
        "epoch": [float(i + 1) for i in range(n)],
        "train/box_loss": [2.0 - 0.05 * i for i in range(n)],
        "train/cls_loss": [4.0 - 0.08 * i for i in range(n)],
        "train/dfl_loss": [2.0 - 0.03 * i for i in range(n)],
        "val/box_loss": [2.2 - 0.02 * i for i in range(n)],
        "val/cls_loss": [4.2 - 0.01 * i for i in range(n)],
        "val/dfl_loss": [2.1 - 0.02 * i for i in range(n)],
    }}
    return set(analyze_loss_curves(results, {})) | {"early_stopping"}


def test_no_rule_targets_a_curve_fact_the_training_report_never_supplies():
    """规则不得挂在 Module B 报告永不提供的曲线事实上。

    ``TrainAnalyzer`` 只把 ``analyze_loss_curves`` 的结果并入报告
    ``curve_analysis``（``analyze_metric_curves`` 的 mAP50 趋势从未进入报告），
    所以 ``training.curve.mAP50`` 永远不进入事实包。挂在该事实上的规则是不可
    触发的死规则，而 ``build_semantic_rule_summary`` 会把它们当可用关系写进
    提示词——模型照做就会引用不存在的事实，得到 DECISION_EVIDENCE_UNKNOWN。

    该测试从**真实生产者**推导可达曲线事实，因此将来若把 mAP50 趋势接通，
    对应规则会自动重新变为合法。
    """
    provided = _module_b_curve_keys()
    reachable = {fact for fact, key in _CURVE_FACT_SOURCES.items() if key in provided}
    dead = sorted(
        rule.rule_id for rule in DETECT_SEMANTIC_RULES
        if rule.fact_id.startswith("training.curve.")
        and rule.fact_id.rsplit(".", 1)[-1] not in reachable
    )
    assert dead == []


def test_curve_facts_are_limited_to_curves_the_report_supplies():
    from auto_tune.modules.agent_engine.decision_facts import CURVE_FIELDS

    provided = _module_b_curve_keys()
    assert CURVE_FIELDS == {fact for fact, key in _CURVE_FACT_SOURCES.items()
                            if key in provided}


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
