"""F1.1-B 阶段一/四 — 由回放记录计算客观质量指标与硬性安全门槛。

只读回放记录与案例期望：不调用 LLM、不重跑校验、不修改记录。方向判定刻意
复用生产的 ``_compute_direction``，保证度量口径与系统实际执行的语义一致。

两类判定严格分开：

- **硬性安全门槛**（plan §四 第八节）：只有「本不该启动训练却仍会启动」才算
  违规。它针对的是校验器的健全性，与场景的临床期望无关。
- **决策质量指标**：决策是否命中该场景的主要矛盾。越出 ``primary_params``
  只说明建议与主因无关，是「修改必要性 / 证据相关性」的质量信号，不是安全
  事故——给定案例事实，注册表可能同时开放其它合法关系。

人工审核矩阵（证据相关性 / 参数方向 / 幅度合理性 / 修改必要性 / keep_params）
不在此模块：由 Codex 与艾卡按固定矩阵人工打分。
"""

from __future__ import annotations

from auto_tune.modules.agent_engine.decision_semantics import _compute_direction
from auto_tune.modules.agent_engine.parameter_registry import PARAMETER_REGISTRY
from auto_tune.modules.agent_engine.semantic_rules import get_semantic_evidence_fact_ids

from .cases import EvalCase
from .runner import OUTCOME_VALID

# 有语义关系的 fact_id 才允许作为证据；其余（比例、指标、计数、参数当前值）
# 无论看起来多相关都只能作只读背景。
_ELIGIBLE_EVIDENCE = get_semantic_evidence_fact_ids()


def _first_attempt(record: dict) -> dict | None:
    attempts = record.get("attempts") or []
    return attempts[0] if attempts else None


def _last_attempt(record: dict) -> dict | None:
    attempts = record.get("attempts") or []
    return attempts[-1] if attempts else None


def _attempt_changes(attempt: dict | None) -> dict:
    if not attempt:
        return {}
    decision = attempt.get("decision") or {}
    return {
        **decision.get("hyperparameter_changes", {}),
        **decision.get("training_overrides", {}),
    }


def _attempt_evidence(attempt: dict | None) -> dict:
    if not attempt:
        return {}
    return (attempt.get("decision") or {}).get("evidence_ids") or {}


def _semantic_details(attempt: dict | None) -> dict:
    """参数 → 该参数的语义校验明细（含 supporting_fact_ids / reason_code）。"""
    if not attempt:
        return {}
    semantic = attempt.get("semantic_validation") or {}
    return {
        detail.get("parameter"): detail
        for detail in (semantic.get("parameters") or [])
        if isinstance(detail, dict) and detail.get("parameter")
    }


def evaluate_run(record: dict, case: EvalCase) -> dict:
    """对单次回放给出一组可聚合的客观判定。"""
    expectation = case.expectation
    facts = record.get("facts") or {}
    first, last = _first_attempt(record), _last_attempt(record)
    effective = record.get("decision") or {}

    action = effective.get("action")
    changes = {
        **effective.get("hyperparameter_changes", {}),
        **effective.get("training_overrides", {}),
    }
    final_valid = record.get("outcome") == OUTCOME_VALID
    first_valid = bool(first and (first.get("decision_validation") or {}).get("valid"))

    # 方向指标只统计「本场景对该参数有临床意见」的那些参数；越出主因的参数
    # 由 out_of_primary 单独表达，不混入方向正确率。
    directions: dict[str, str | None] = {}
    direction_matches: dict[str, bool] = {}
    for parameter, suggested in changes.items():
        current = facts.get(f"training.params.{parameter}")
        directions[parameter] = _compute_direction(current, suggested) if current is not None else None
    for parameter, expected in expectation.allowed_directions.items():
        if parameter in changes:
            direction_matches[parameter] = directions[parameter] == expected

    changed = set(changes)
    if expectation.should_keep_params:
        primary_match = action == "keep_params"
    else:
        primary_match = (
            bool(changed)
            and changed <= set(expectation.primary_params)
            and all(direction_matches.values())
        )

    details = _semantic_details(last)
    supporting = [p for p in changes if (details.get(p) or {}).get("supporting_fact_ids")]
    limited = [p for p, d in details.items() if d.get("reason_code") == "CHANGE_LIMIT_EXCEEDED"]
    with_support = [
        p for p in changes
        if p in details and details[p].get("error_code") != "DECISION_SEMANTIC_UNSUPPORTED"
    ]

    unknown_params = sorted(changed - set(PARAMETER_REGISTRY))
    fact_ids = set(record.get("fact_ids") or [])
    unknown_evidence = sorted(
        {fid for ids in _attempt_evidence(last).values() for fid in ids} - fact_ids
    )

    # ── 建议质量维度（合法性之外） ──────────────────────────────────────────
    # 语义校验逐参数、逐引用事实判定：只要每条证据对得上它支持的那个参数就通过。
    # 因此 valid=True 只证明合法，不证明正确。下面四项把「主因是否命中、有没有
    # 改成方向相反的参数、证据是否掺杂无关事实、组合是否自相矛盾、幅度是否偏激」
    # 单独量出来。
    cited = sorted({fid for ids in _attempt_evidence(last).values() for fid in ids})
    # 「证据不合规」是客观事实：引用了根本没有语义关系的 fact_id（比例、指标、
    # 参数当前值）。而「越出本案例的期望证据范围」是另一回事——例如多 issue
    # 场景里为 imgsz 引用 dataset.issue.tiny_bbox_high_ratio，那是该参数的**合法**
    # 证据，只是不服务于主因。两者必须分开，否则会把正确引用误报成污染。
    ineligible_evidence = sorted(fid for fid in cited if fid not in _ELIGIBLE_EVIDENCE)
    scope = expectation.evidence_scope
    off_scope_evidence = (
        sorted(fid for fid in cited
               if not any(fid.startswith(prefix) for prefix in scope)
               and fid not in ineligible_evidence)
        if scope else []
    )

    if expectation.should_keep_params:
        main_cause_hit = action == "keep_params"
    else:
        main_cause_hit = bool(changed & set(expectation.primary_params))

    counter_hit = sorted(changed & set(expectation.counter_params))

    incoherent = [
        list(pair) for pair in expectation.incoherent_pairs
        if pair[0] in changed and pair[1] in changed
    ]

    aggressive = []
    for parameter, (low, high) in expectation.preferred_ratios.items():
        if parameter not in changes:
            continue
        current = facts.get(f"training.params.{parameter}")
        suggested = changes[parameter]
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            continue
        if not isinstance(suggested, (int, float)) or isinstance(suggested, bool):
            continue
        if current == 0:
            continue
        ratio = suggested / current
        if not (low <= ratio <= high):
            aggressive.append(parameter)

    # 硬性安全门槛：只有「本不该启动训练却仍会启动」才算违规。
    reversed_on_primary = [
        p for p, ok in direction_matches.items() if not ok and p in expectation.primary_params
    ]
    would_start = bool(record.get("would_start_training"))
    forced_change = expectation.should_keep_params and action != "keep_params"

    return {
        "case_id": record["case_id"],
        "run_index": record.get("run_index"),
        "outcome": record.get("outcome"),
        "error_code": record.get("error_code"),
        "action": action,
        "changes": changes,
        "evidence_ids": effective.get("evidence_ids", {}),
        "retried": bool(effective.get("retried")),
        "first_response_valid": first_valid,
        "final_valid": final_valid,
        "keep_params_expected": expectation.should_keep_params,
        "keep_params_correct": (action == "keep_params"),
        "primary_match": primary_match,
        "primary_params": sorted(expectation.primary_params),
        "out_of_primary": sorted(changed - set(expectation.primary_params)),
        "direction_matches": direction_matches,
        "directions": directions,
        "supporting_evidence_ok": bool(supporting) if changes else None,
        "magnitude_limited": limited,
        "params_with_support": with_support,
        "unknown_params": unknown_params,
        "unknown_evidence": unknown_evidence,
        "would_start_training": would_start,
        # 建议质量（与合法性分开）
        "main_cause_hit": main_cause_hit,
        "counter_param_hit": counter_hit,
        "ineligible_evidence": ineligible_evidence,
        "off_scope_evidence": off_scope_evidence,
        "incoherent_combo": incoherent,
        "amplitude_aggressive": aggressive,
        "amplitude_judged": bool(expectation.preferred_ratios),
        "illegal_launch": bool(
            would_start and (unknown_params or unknown_evidence or reversed_on_primary
                             or forced_change)
        ),
        "unknown_param_pass": bool(would_start and unknown_params),
        "wrong_evidence_pass": bool(would_start and unknown_evidence),
        "direction_reversed_pass": bool(would_start and reversed_on_primary),
        "forced_change_on_keep_case": bool(would_start and forced_change),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_summary(evaluations: list[dict]) -> dict:
    """聚合客观指标与硬性安全门槛计数。"""
    runs = len(evaluations)
    adjusted = [e for e in evaluations if e["changes"]]
    primary_judged = [e for e in evaluations
                      if e["primary_match"] is not None]
    direction_flags = [
        ok for e in evaluations for ok in e["direction_matches"].values()
    ]
    magnitude_judged = [
        e for e in evaluations if e["params_with_support"] or e["magnitude_limited"]
    ]
    magnitude_ok = [
        e for e in magnitude_judged
        if not e["magnitude_limited"] and e["params_with_support"]
    ]

    # 建议质量：只统计「合法通过」的那些轮次——被拒绝的建议根本不会执行，
    # 把它们混进来会掩盖真正通过校验的错误建议。
    accepted = [e for e in evaluations if e["final_valid"]]
    amplitude_judged = [e for e in accepted if e["amplitude_judged"]]
    keep_expected = [e for e in evaluations if e["keep_params_expected"]]
    quality = {
        "accepted_runs": len(accepted),
        "main_cause_hit_rate": _rate(
            sum(1 for e in accepted if e["main_cause_hit"]), len(accepted)),
        "counter_param_rate": _rate(
            sum(1 for e in accepted if e["counter_param_hit"]), len(accepted)),
        "ineligible_evidence_rate": _rate(
            sum(1 for e in accepted if e["ineligible_evidence"]), len(accepted)),
        "off_scope_evidence_rate": _rate(
            sum(1 for e in accepted if e["off_scope_evidence"]), len(accepted)),
        "incoherent_combo_rate": _rate(
            sum(1 for e in accepted if e["incoherent_combo"]), len(accepted)),
        "amplitude_within_preference_rate": _rate(
            sum(1 for e in amplitude_judged if not e["amplitude_aggressive"]),
            len(amplitude_judged)),
    }
    # 各质量维度的方向不同，转换必须与语义一致：负向指标（越低越好）取
    # 1-rate，正向指标（越高越好）直接用原值。两者都取 1-rate 会让「主因命中
    # 率越高、幅度越稳健」反而降低综合分。
    #
    # accepted=[] 时没有任何质量分母，各率都是 None（未知）。未知就是未知：
    # 既不补 0 也不补 1，因此只吸收有分母的那些维度。
    bad_rates = (
        quality["counter_param_rate"],
        quality["ineligible_evidence_rate"],
        quality["incoherent_combo_rate"],
    )
    good_rates = (
        quality["main_cause_hit_rate"],
        quality["amplitude_within_preference_rate"],
    )
    dimension_scores = [
        *[1 - rate for rate in bad_rates if rate is not None],
        *[rate for rate in good_rates if rate is not None],
    ]
    quality["advice_quality_score"] = _rate(
        round(sum(dimension_scores) / len(dimension_scores), 6), 1
    ) if dimension_scores else None

    return {
        "runs": runs,
        "first_response_valid_rate": _rate(
            sum(1 for e in evaluations if e["first_response_valid"]), runs),
        "final_valid_rate": _rate(
            sum(1 for e in evaluations if e["final_valid"]), runs),
        **quality,
        "retried_runs": sum(1 for e in evaluations if e["retried"]),
        "evidence_param_match_rate": _rate(
            sum(1 for e in adjusted if e["supporting_evidence_ok"]), len(adjusted)),
        "direction_correct_rate": _rate(
            sum(1 for ok in direction_flags if ok), len(direction_flags)),
        "magnitude_reasonable_rate": _rate(len(magnitude_ok), len(magnitude_judged)),
        # 分子只取自分母集合：非 keep 场景误报 keep 不属于「keep 判断正确」，
        # 否则分母 1 而分子 2，比率会超过 1。
        "keep_params_correct_rate": _rate(
            sum(1 for e in keep_expected if e["keep_params_correct"]),
            len(keep_expected)),
        "primary_match_rate": _rate(
            sum(1 for e in primary_judged if e["primary_match"]), len(primary_judged)),
        # ── 硬性安全门槛（任一非零即 F1.1-B 不通过） ──
        "illegal_launch_count": sum(1 for e in evaluations if e["illegal_launch"]),
        "unknown_param_pass_count": sum(1 for e in evaluations if e["unknown_param_pass"]),
        "wrong_evidence_pass_count": sum(1 for e in evaluations if e["wrong_evidence_pass"]),
        "direction_reversed_pass_count": sum(
            1 for e in evaluations if e["direction_reversed_pass"]),
        "forced_change_on_keep_case_count": sum(
            1 for e in evaluations if e["forced_change_on_keep_case"]),
        "would_start_training_count": sum(
            1 for e in evaluations if e["would_start_training"]),
    }


def case_table(evaluations: list[dict]) -> list[dict]:
    """逐案例汇总，用于修改前/修改后对比表。"""
    by_case: dict[str, list[dict]] = {}
    for evaluation in evaluations:
        by_case.setdefault(evaluation["case_id"], []).append(evaluation)
    return [
        {"case_id": case_id, **compute_summary(items)}
        for case_id, items in sorted(by_case.items())
    ]


def failure_breakdown(evaluations: list[dict]) -> dict:
    """按稳定错误码统计失败，供阶段二「质量下降来源」分类。"""
    breakdown: dict[str, int] = {}
    for evaluation in evaluations:
        if evaluation["final_valid"]:
            continue
        code = evaluation["error_code"] or evaluation["outcome"]
        breakdown[code] = breakdown.get(code, 0) + 1
    return dict(sorted(breakdown.items(), key=lambda item: (-item[1], item[0])))
