"""F1.1-B — 评分的「格式合法」与「建议质量」必须分开。

语义校验逐参数、逐引用事实判定：只要每条证据对得上它支持的那个参数就通过。
于是主因无关、证据掺杂、组合自相矛盾、幅度偏激的建议都能合法通过。这些测试
锁住最关键的一点——**这些建议的格式合法性都是 True**，因此「合法率 100%」
不能用来证明建议质量；质量必须由独立维度量出来。
"""

import json
import os

import pytest

from auto_tune.evaluation import metrics as metrics_mod
from auto_tune.evaluation.cases import get_case
from auto_tune.evaluation.runner import prepare_case

_QUALITY_RATE_KEYS = (
    "main_cause_hit_rate",
    "counter_param_rate",
    "ineligible_evidence_rate",
    "off_scope_evidence_rate",
    "incoherent_combo_rate",
    "amplitude_within_preference_rate",
)


def _fake_record(case, changes, evidence, root):
    """构造一个「已通过格式校验」的回放记录，只用于检验评分层。"""
    prepared = prepare_case(case, root)
    facts = {f["fact_id"]: f["value"] for f in prepared.fact_package["facts"]}
    decision = {
        "action": "adjust",
        "hyperparameter_changes": dict(changes),
        "training_overrides": {},
        "evidence_ids": dict(evidence),
        "retried": False,
    }
    return {
        "case_id": case.case_id,
        "run_index": 1,
        "outcome": "valid",
        "error_code": None,
        "would_start_training": True,
        "fact_ids": sorted(facts),
        "facts": facts,
        "decision": decision,
        "attempts": [{
            "attempt": 1, "retried": False,
            "decision": {k: v for k, v in decision.items() if k != "action" and k != "retried"},
            "decision_validation": {"valid": True, "error_code": None},
            "semantic_validation": {
                "valid": True,
                "parameters": [
                    {"parameter": parameter, "supporting_fact_ids": ids,
                     "reason_code": None, "error_code": None}
                    for parameter, ids in evidence.items()
                ],
            },
        }],
    }


def _evaluate(case_id, changes, evidence, tmp_path, suffix):
    case = get_case(case_id)
    record = _fake_record(case, changes, evidence,
                          os.path.join(str(tmp_path), suffix))
    return metrics_mod.evaluate_run(record, case)


def test_directionally_wrong_but_legal_advice_passes_format_and_fails_quality(tmp_path):
    """欠拟合主因下降 lr0：plateau 规则允许，格式合法，但方向是错的。"""
    evaluation = _evaluate(
        "direction_trap", {"lr0": 0.005}, {"lr0": ["training.issue.plateau"]},
        tmp_path, "direction")

    assert evaluation["final_valid"] is True
    assert evaluation["main_cause_hit"] is False
    assert evaluation["counter_param_hit"] == ["lr0"]


def test_evidence_pollution_passes_format_and_fails_quality(tmp_path):
    """引用无规则的数据集比例/指标当证据：结构上对得上，但与该参数无关。"""
    evaluation = _evaluate(
        "evidence_pollution", {"lr0": 0.006},
        {"lr0": ["training.issue.plateau",
                 "dataset.bbox_analysis.tiny_bbox_ratio",
                 "training.metrics.mAP50"]},
        tmp_path, "pollution")

    assert evaluation["final_valid"] is True
    assert evaluation["ineligible_evidence"] == [
        "dataset.bbox_analysis.tiny_bbox_ratio", "training.metrics.mAP50"]
    assert evaluation["off_scope_evidence"] == []


def test_incoherent_combo_passes_format_and_fails_quality(tmp_path):
    """减少 epochs 同时增大 patience：两项都合法，组合自相矛盾。"""
    evaluation = _evaluate(
        "incoherent_combo", {"epochs": 60, "patience": 90},
        {"epochs": ["training.issue.overfitting"],
         "patience": ["training.issue.early_stop_too_soon"]},
        tmp_path, "combo")

    assert evaluation["final_valid"] is True
    assert evaluation["incoherent_combo"] == [["epochs", "patience"]]


def test_over_aggressive_amplitude_passes_format_and_fails_quality(tmp_path):
    """一次砍到规则下界：合法，但超出稳健幅度区间。"""
    evaluation = _evaluate(
        "amplitude_aggressive", {"lr0": 0.0025},
        {"lr0": ["training.issue.plateau"]}, tmp_path, "amplitude")

    assert evaluation["final_valid"] is True
    assert evaluation["amplitude_aggressive"] == ["lr0"]


def test_amplitude_within_preference_is_not_flagged(tmp_path):
    evaluation = _evaluate(
        "amplitude_aggressive", {"lr0": 0.006},
        {"lr0": ["training.issue.plateau"]}, tmp_path, "amplitude_ok")

    assert evaluation["amplitude_aggressive"] == []


def test_picking_the_counter_param_is_flagged(tmp_path):
    """多 issue 场景里去改不对症的参数：合法，但没命中主因。"""
    evaluation = _evaluate(
        "multi_issue_priority", {"imgsz": 900},
        {"imgsz": ["dataset.issue.tiny_bbox_high_ratio"]}, tmp_path, "priority")

    assert evaluation["final_valid"] is True
    assert evaluation["main_cause_hit"] is False
    assert evaluation["counter_param_hit"] == ["imgsz"]
    # 该引用本身是 imgsz 的**合法**证据（有规则），不算不合规证据；
    # 只算「越出本案例期望的证据范围」。
    assert evaluation["ineligible_evidence"] == []
    assert evaluation["off_scope_evidence"] == ["dataset.issue.tiny_bbox_high_ratio"]


def test_quality_summary_separates_the_two_axes(tmp_path):
    """汇总里格式合法率与质量指标必须并存，且同一批数据给出不同结论。"""
    evaluations = [
        _evaluate("direction_trap", {"lr0": 0.005},
                  {"lr0": ["training.issue.plateau"]}, tmp_path, "s1"),
        _evaluate("direction_trap", {"epochs": 120, "weight_decay": 0.00025},
                  {"epochs": ["training.issue.underfitting"],
                   "weight_decay": ["training.issue.underfitting"]},
                  tmp_path, "s2"),
    ]

    summary = metrics_mod.compute_summary(evaluations)

    assert summary["final_valid_rate"] == 1.0          # 两次都合法
    assert summary["main_cause_hit_rate"] == 0.5       # 只有一次命中主因
    assert summary["counter_param_rate"] == 0.5        # 一次改成了反向参数
    assert summary["advice_quality_score"] < 1.0


def test_quality_rates_only_count_accepted_runs(tmp_path):
    """被拒绝的建议不进质量分母——它们根本不会执行。"""
    accepted = _evaluate("direction_trap", {"epochs": 120},
                         {"epochs": ["training.issue.underfitting"]},
                         tmp_path, "ok")
    rejected = dict(accepted)
    rejected["final_valid"] = False
    rejected["outcome"] = "contract_failed"
    rejected["main_cause_hit"] = False
    rejected["counter_param_hit"] = ["lr0"]

    summary = metrics_mod.compute_summary([accepted, rejected])

    assert summary["accepted_runs"] == 1
    assert summary["main_cause_hit_rate"] == 1.0
    assert summary["counter_param_rate"] == 0.0


# ── Codex 复核 P3：确定性统计错误 ─────────────────────────────────────────


def _evaluate_action(case_id, action, tmp_path, suffix):
    """构造一条指向给定 action 的回放判定（用于 keep_params 统计）。"""
    case = get_case(case_id)
    record = _fake_record(case, {}, {}, os.path.join(str(tmp_path), suffix))
    record["decision"]["action"] = action
    return metrics_mod.evaluate_run(record, case)


def _rejected(evaluation):
    rejected = dict(evaluation)
    rejected["final_valid"] = False
    rejected["outcome"] = "contract_failed"
    return rejected


def test_all_rejected_runs_report_unknown_quality_instead_of_crashing(tmp_path):
    """全部建议被拒绝时质量分母为空：必须返回 None，不能做 1 - None。"""
    evaluations = [
        _rejected(_evaluate("direction_trap", {"lr0": 0.005},
                            {"lr0": ["training.issue.plateau"]},
                            tmp_path, f"rejected{index}"))
        for index in range(2)
    ]

    summary = metrics_mod.compute_summary(evaluations)

    assert summary["accepted_runs"] == 0
    assert summary["advice_quality_score"] is None
    for key in _QUALITY_RATE_KEYS:
        assert summary[key] is None, key


def test_case_table_survives_all_rejected_runs(tmp_path):
    """逐案例汇总走同一段代码，同样不得崩溃。"""
    rejected = _rejected(_evaluate("healthy", {}, {}, tmp_path, "case_table"))

    rows = metrics_mod.case_table([rejected])

    assert rows[0]["advice_quality_score"] is None


def test_keep_params_correct_rate_never_exceeds_one(tmp_path):
    """分子必须取自分母的同一集合。

    「非 keep 案例误报 keep」当前会被计入分子却不在分母里，于是单案例正确
    也能算出 2.0 —— 比率必须恒在 [0, 1]。
    """
    correct = _evaluate_action("healthy", "keep_params", tmp_path, "keep_ok")
    irrelevant = _evaluate_action("overfitting", "keep_params", tmp_path, "keep_elsewhere")

    summary = metrics_mod.compute_summary([correct, irrelevant])

    assert summary["keep_params_correct_rate"] == 1.0


def test_keep_params_correct_rate_is_unknown_without_keep_cases(tmp_path):
    evaluation = _evaluate_action("overfitting", "keep_params", tmp_path, "no_keep_case")

    summary = metrics_mod.compute_summary([evaluation])

    assert summary["keep_params_correct_rate"] is None


def test_keep_params_correct_rate_is_zero_when_the_keep_case_is_missed(tmp_path):
    evaluation = _evaluate_action("healthy", "adjust", tmp_path, "missed_keep")

    summary = metrics_mod.compute_summary([evaluation])

    assert summary["keep_params_correct_rate"] == 0.0


def test_no_summary_rate_escapes_the_unit_interval(tmp_path):
    evaluations = [
        _evaluate_action("healthy", "keep_params", tmp_path, "bounds_keep"),
        _evaluate_action("overfitting", "keep_params", tmp_path, "bounds_other"),
        _rejected(_evaluate("direction_trap", {"lr0": 0.005},
                            {"lr0": ["training.issue.plateau"]},
                            tmp_path, "bounds_rejected")),
    ]

    summary = metrics_mod.compute_summary(evaluations)

    for key, value in summary.items():
        if key.endswith("_rate") and value is not None:
            assert 0.0 <= value <= 1.0, (key, value)


# ── Codex 复核 P3（续）：综合分的方向必须与各维度的语义一致 ─────────────────


def _quality_evaluation(*, main_cause_hit, counter_param_hit, ineligible_evidence,
                        incoherent_combo, amplitude_aggressive,
                        amplitude_judged=True, off_scope_evidence=None,
                        final_valid=True):
    """只控制五个质量维度的最小判定，用于检验综合分聚合方向。

    这里刻意直接给出 compute_summary 真正读取的键：比率由这些取值唯一决定，
    因此可以精确构造「全优」「全差」样本，无需迁就案例集。
    """
    return {
        "case_id": "synthetic",
        "run_index": 1,
        "outcome": "valid",
        "error_code": None,
        "action": "adjust",
        "changes": {"lr0": 0.005},
        "evidence_ids": {"lr0": ["training.issue.plateau"]},
        "retried": False,
        "first_response_valid": True,
        "final_valid": final_valid,
        "keep_params_expected": False,
        "keep_params_correct": False,
        "primary_match": True,
        "primary_params": ["lr0"],
        "out_of_primary": [],
        "direction_matches": {"lr0": True},
        "directions": {"lr0": "decrease"},
        "supporting_evidence_ok": True,
        "magnitude_limited": [],
        "params_with_support": ["lr0"],
        "unknown_params": [],
        "unknown_evidence": [],
        "would_start_training": True,
        "main_cause_hit": main_cause_hit,
        "counter_param_hit": list(counter_param_hit),
        "ineligible_evidence": list(ineligible_evidence),
        "off_scope_evidence": list(off_scope_evidence or []),
        "incoherent_combo": list(incoherent_combo),
        "amplitude_aggressive": list(amplitude_aggressive),
        "amplitude_judged": amplitude_judged,
        "illegal_launch": False,
        "unknown_param_pass": False,
        "wrong_evidence_pass": False,
        "direction_reversed_pass": False,
        "forced_change_on_keep_case": False,
    }


def _score(**dimensions):
    return metrics_mod.compute_summary([_quality_evaluation(**dimensions)])[
        "advice_quality_score"]


def test_advice_quality_score_is_one_when_every_dimension_is_best():
    """五个维度全部最优时必须得到 1.0，不能因为正向维度被反向计算而掉到 0.6。"""
    summary = metrics_mod.compute_summary([_quality_evaluation(
        main_cause_hit=True, counter_param_hit=[], ineligible_evidence=[],
        incoherent_combo=[], amplitude_aggressive=[])])

    assert summary["main_cause_hit_rate"] == 1.0
    assert summary["amplitude_within_preference_rate"] == 1.0
    assert summary["advice_quality_score"] == 1.0


def test_advice_quality_score_is_zero_when_every_dimension_is_worst():
    summary = metrics_mod.compute_summary([_quality_evaluation(
        main_cause_hit=False, counter_param_hit=["lr0"],
        ineligible_evidence=["training.metrics.mAP50"],
        incoherent_combo=[["epochs", "patience"]], amplitude_aggressive=["lr0"])])

    assert summary["main_cause_hit_rate"] == 0.0
    assert summary["amplitude_within_preference_rate"] == 0.0
    assert summary["advice_quality_score"] == 0.0


def test_raising_the_main_cause_hit_rate_raises_the_composite():
    """正向指标：命中率提高只能让综合分更高。"""
    common = dict(counter_param_hit=[], ineligible_evidence=[],
                  incoherent_combo=[], amplitude_aggressive=[])
    miss = _score(main_cause_hit=False, **common)
    hit = _score(main_cause_hit=True, **common)

    assert hit > miss


def test_raising_the_amplitude_rate_raises_the_composite():
    """正向指标：幅度落在稳健区间的比例提高只能让综合分更高。"""
    common = dict(main_cause_hit=True, counter_param_hit=[],
                  ineligible_evidence=[], incoherent_combo=[])
    aggressive = _score(amplitude_aggressive=["lr0"], **common)
    stable = _score(amplitude_aggressive=[], **common)

    assert stable > aggressive


def test_raising_a_bad_rate_lowers_the_composite():
    """负向指标：改用反向参数/引用无规则事实/组合矛盾的比例提高必须更低分。"""
    common = dict(main_cause_hit=True, amplitude_aggressive=[])
    clean = _score(counter_param_hit=[], ineligible_evidence=[],
                   incoherent_combo=[], **common)
    counters = _score(counter_param_hit=["lr0"], ineligible_evidence=[],
                      incoherent_combo=[], **common)
    polluting = _score(counter_param_hit=[], ineligible_evidence=["training.metrics.mAP50"],
                       incoherent_combo=[], **common)
    incoherent = _score(counter_param_hit=[], ineligible_evidence=[],
                        incoherent_combo=[["epochs", "patience"]], **common)

    assert counters < clean
    assert polluting < clean
    assert incoherent < clean


def test_dimensions_without_a_denominator_are_excluded_not_defaulted():
    """没有分母的维度必须从综合分中排除，不得补成 0 或 1。"""
    summary = metrics_mod.compute_summary([_quality_evaluation(
        main_cause_hit=True, counter_param_hit=[], ineligible_evidence=[],
        incoherent_combo=[], amplitude_aggressive=[], amplitude_judged=False)])

    assert summary["amplitude_within_preference_rate"] is None
    # 只剩四个有分母的维度，且它们全部最优
    assert summary["advice_quality_score"] == 1.0


def test_report_command_completes_on_all_rejected_records(tmp_path, capsys):
    """`--report` 对全拒绝记录必须正常完成并显示 —，不能抛异常。"""
    from auto_tune.scripts import evaluate_tuning_decisions as script

    out_dir = tmp_path / "rejected_report"
    out_dir.mkdir()
    for index, case_id in enumerate(("direction_trap", "healthy")):
        case = get_case(case_id)
        record = _fake_record(case, {}, {}, str(tmp_path / f"ws{index}"))
        record["outcome"] = "contract_failed"
        record["error_code"] = "DECISION_SCHEMA_INVALID"
        record["decision"] = None
        record["attempts"] = []
        record["would_start_training"] = False
        (out_dir / f"{case_id}__run1.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8")

    exit_code = script.main(["--report", str(out_dir)])

    assert exit_code == 0
    assert "—" in capsys.readouterr().out
