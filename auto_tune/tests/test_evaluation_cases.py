"""F1.1-B 阶段一 — 离线案例集的确定性与生产保真度。

案例集是修改前/修改后质量对比的基准，它本身必须先被锁死：

- 事实包必须由真实生产链路生成，且对同一案例稳定可复现；
- 事实包只能绑定当前参考运行，另一个 run 的事实不得渗入；
- before 指标必须等于该参考运行 results.csv 的末轮指标（与生产同口径）；
- 案例不得造出生产不会出现的事实（当前指 Module B 从不产出的 mAP 曲线趋势）。
"""

import copy
import os

import pytest

from auto_tune.evaluation.cases import get_case, get_cases
from auto_tune.evaluation.runner import prepare_case

_CASES = get_cases()


def _facts(prepared):
    return {f["fact_id"]: f["value"] for f in prepared.fact_package["facts"]}


def test_case_set_is_stable_and_unique():
    ids = [case.case_id for case in _CASES]
    assert len(ids) == 15
    assert len(set(ids)) == 15
    assert ids == [case.case_id for case in get_cases()]
    assert len(get_cases("core")) == 9
    assert len(get_cases("hard")) == 6


def test_hard_cases_admit_multiple_legal_actions(tmp_path):
    """困难案例必须真的「合法性容易满足」：至少两个参数有可用语义关系。

    只有一个合法答案时，通过了也说明不了判断力。困难案例的设计前提就是
    多条规则族同时可用，于是语义校验必然放行，质量只能由期望来区分。
    """
    from auto_tune.modules.agent_engine.semantic_rules import get_semantic_rules

    for case in get_cases("hard"):
        prepared = prepare_case(case, os.path.join(str(tmp_path), case.case_id))
        facts = {f["fact_id"]: f["value"] for f in prepared.fact_package["facts"]}
        legal = set()
        for rule in get_semantic_rules():
            if rule.fact_id not in facts:
                continue
            if rule.fact_value is not None and facts[rule.fact_id] != rule.fact_value:
                continue
            legal.add(rule.parameter)
        assert len(legal) >= 2, (case.case_id, sorted(legal))


def test_hard_cases_declare_a_measurable_quality_expectation():
    """困难案例必须给出可度量的质量维度，否则只能得到合法性数字。"""
    for case in get_cases("hard"):
        expectation = case.expectation
        assert (expectation.counter_params or expectation.evidence_scope
                or expectation.incoherent_pairs
                or expectation.preferred_ratios), case.case_id


def test_every_case_builds_a_bound_fact_package(tmp_path):
    for case in _CASES:
        prepared = prepare_case(case, os.path.join(str(tmp_path), case.case_id))
        assert prepared.blocking_code is None, case.case_id
        assert prepared.fact_error is None, case.case_id
        assert prepared.fact_package["reference_run"] == case.reference_run
        assert prepared.fact_package["fact_package_id"].startswith("sha256:")


def test_fact_package_is_deterministic_per_case(tmp_path):
    for case in _CASES:
        first = prepare_case(case, os.path.join(str(tmp_path), case.case_id, "a"))
        second = prepare_case(case, os.path.join(str(tmp_path), case.case_id, "b"))
        assert first.fact_package == second.fact_package, case.case_id


def test_before_metrics_match_case_final_metrics(tmp_path):
    for case in _CASES:
        prepared = prepare_case(case, os.path.join(str(tmp_path), case.case_id))
        expected = {k: v for k, v in case.final_metrics.items() if v is not None}
        assert prepared.before_metrics == expected, case.case_id


def test_fact_package_is_bound_to_the_reference_run(tmp_path):
    case = get_case("reference_vs_history_conflict")
    prepared = prepare_case(case, os.path.join(str(tmp_path), case.case_id))
    facts = _facts(prepared)

    # 参考运行的过拟合事实必须存在
    assert facts["training.issue.overfitting"] is True
    # 另一个 run 的指标不得出现在事实包里
    assert facts["training.metrics.mAP50"] == case.final_metrics["mAP50"]
    assert facts["training.params.weight_decay"] == case.args["weight_decay"]
    # 事实包的训练指标只能有一份，不能混入历史最佳 run 的数值
    assert "training.metrics.mAP50_95" in facts


def test_no_case_fabricates_the_map50_curve_fact(tmp_path):
    """Module B 的 curve_analysis 从不包含 mAP50 键（204/204 真实报告）。

    事实包里出现 ``training.curve.mAP50`` 就说明案例造了生产不会出现的事实，
    该事实会反过来让依赖它的语义规则在评估中「显得可用」。
    """
    for case in _CASES:
        prepared = prepare_case(case, os.path.join(str(tmp_path), case.case_id))
        facts = _facts(prepared)
        assert "training.curve.mAP50" not in facts, case.case_id
        run = case.training_report["runs"][case.reference_run]
        authored = run["curve_analysis"]["val_box"]["trend"]
        assert ("training.curve.val_box_loss" in facts) is bool(authored), case.case_id


def test_sparse_case_has_strictly_narrower_facts(tmp_path):
    sparse = prepare_case(
        get_case("insufficient_facts"), os.path.join(str(tmp_path), "sparse"))
    rich = prepare_case(
        get_case("healthy"), os.path.join(str(tmp_path), "rich"))
    sparse_ids = set(_facts(sparse))
    rich_ids = set(_facts(rich))
    assert sparse_ids < rich_ids
    for missing in ("dataset.label_rate", "dataset.quality_score",
                    "dataset.bbox_analysis.tiny_bbox_ratio",
                    "dataset.image_quality.blur_ratio"):
        assert missing not in sparse_ids
    assert not any(fid.startswith(("training.issue.", "dataset.issue."))
                   for fid in sparse_ids)


def test_case_definition_is_not_mutated_by_preparation(tmp_path):
    case = get_case("overfitting")
    before = copy.deepcopy((case.dataset_report, case.training_report, case.args))
    prepare_case(case, os.path.join(str(tmp_path), case.case_id))
    assert (case.dataset_report, case.training_report, case.args) == before


@pytest.mark.parametrize("case_id", [case.case_id for case in _CASES])
def test_expectation_is_internally_consistent(case_id):
    case = get_case(case_id)
    expectation = case.expectation
    if expectation.should_keep_params:
        assert not expectation.primary_params
        assert not expectation.allowed_directions
    else:
        assert expectation.primary_params
        assert set(expectation.allowed_directions) == set(expectation.primary_params)
