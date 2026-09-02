"""Q1.2 — 确定性语义校验：证据—参数—方向—幅度。

校验器只读取语义规则注册表和冻结事实包，不修改 decision / fact_package，
不解释开放式自然语言，不调用第二个模型。返回结果可直接 JSON 序列化，且不含
原始 LLM 响应、凭据、绝对路径、命令或未受控异常正文。

校验顺序（规格 §6）：
1. 读取当前参数值，缺失则失败；
2. 按注册类型规范化建议值，无法比较则失败；
3. 计算方向或布尔转换；
4. 划分证据为支持 / 冲突 / 中性；
5. 至少一条支持证据；
6. 任一冲突证据即整体失败；
7. 支持关系存在后再校验单轮幅度；
8. 所有参数均通过才通过整个决策。
"""

from __future__ import annotations

from typing import Any

from .decision_facts import _normalize_parameter_fact
from .parameter_registry import PARAMETER_REGISTRY
from .semantic_rules import (
    DETECT_SEMANTIC_RULES,
    ChangeLimit,
    SemanticRule,
    get_semantic_rules,
)

# ── 稳定错误码 ──────────────────────────────────────────────────────────────
DECISION_SEMANTIC_UNSUPPORTED = "DECISION_SEMANTIC_UNSUPPORTED"
DECISION_SEMANTIC_DIRECTION_CONFLICT = "DECISION_SEMANTIC_DIRECTION_CONFLICT"
DECISION_SEMANTIC_CHANGE_TOO_LARGE = "DECISION_SEMANTIC_CHANGE_TOO_LARGE"
DECISION_SEMANTIC_CURRENT_VALUE_MISSING = "DECISION_SEMANTIC_CURRENT_VALUE_MISSING"
DECISION_SEMANTIC_EVIDENCE_CONFLICT = "DECISION_SEMANTIC_EVIDENCE_CONFLICT"

SEMANTIC_ERROR_CODES = frozenset({
    DECISION_SEMANTIC_UNSUPPORTED,
    DECISION_SEMANTIC_DIRECTION_CONFLICT,
    DECISION_SEMANTIC_CHANGE_TOO_LARGE,
    DECISION_SEMANTIC_CURRENT_VALUE_MISSING,
    DECISION_SEMANTIC_EVIDENCE_CONFLICT,
})

# ── 受控 reason_code ────────────────────────────────────────────────────────
NO_SUPPORTING_RULE = "NO_SUPPORTING_RULE"
UNCHANGED_VALUE = "UNCHANGED_VALUE"
DIRECTION_NOT_SUPPORTED = "DIRECTION_NOT_SUPPORTED"
CHANGE_LIMIT_EXCEEDED = "CHANGE_LIMIT_EXCEEDED"
CURRENT_VALUE_MISSING = "CURRENT_VALUE_MISSING"
CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"

# 方向令牌（与 semantic_rules 对齐）
_INCREASE = "increase"
_DECREASE = "decrease"
_ENABLE = "enable"
_DISABLE = "disable"
_UNCHANGED = "unchanged"


def _check_change_limit(limit: ChangeLimit, current, suggested) -> bool:
    """按规则幅度限制判断单轮变化是否可接受。边界值允许，超过边界拒绝。"""
    kind = limit.kind
    if kind == "bool":
        return True  # 方向已在上层校验，bool 无幅度
    if kind == "ratio_range":
        if current == 0:
            return False
        return (limit.min_ratio is None or suggested >= limit.min_ratio * current) and (
            limit.max_ratio is None or suggested <= limit.max_ratio * current)
    if kind == "max_multiple":
        if current == 0:
            return limit.zero_cap is not None and suggested <= limit.zero_cap
        return limit.max_ratio is None or suggested <= limit.max_ratio * current
    if kind == "min_fraction":
        if current <= 0:
            return False
        return limit.min_ratio is None or suggested >= limit.min_ratio * current
    if kind == "max_add":
        return limit.max_add is None or suggested <= current + limit.max_add
    return False


def _compute_direction(current, suggested) -> str:
    """计算 increase / decrease / enable / disable / unchanged。"""
    if isinstance(current, bool) and isinstance(suggested, bool):
        if current is False and suggested is True:
            return _ENABLE
        if current is True and suggested is False:
            return _DISABLE
        return _UNCHANGED
    if suggested > current:
        return _INCREASE
    if suggested < current:
        return _DECREASE
    return _UNCHANGED


def _rule_matches(rule: SemanticRule, fact_value: Any) -> bool:
    if rule.fact_value is None:
        return True
    return fact_value == rule.fact_value


def _validate_parameter(parameter: str, suggested_raw: Any, facts: dict,
                        evidence_ids: list[str]) -> dict:
    """校验单个修改参数，返回逐参数结构化结果。"""
    current = facts.get(f"training.params.{parameter}")

    def _result(valid, error_code, reason_code, current, suggested,
                direction, rule_ids, supporting, conflicting, neutral):
        return {
            "valid": valid,
            "error_code": error_code,
            "reason_code": reason_code,
            "parameter": parameter,
            "rule_ids": sorted(rule_ids),
            "supporting_fact_ids": sorted(supporting),
            "conflicting_fact_ids": sorted(conflicting),
            "neutral_fact_ids": sorted(neutral),
            "current_value": current,
            "suggested_value": suggested,
            "change_direction": direction,
        }

    if current is None:
        return _result(False, DECISION_SEMANTIC_CURRENT_VALUE_MISSING, CURRENT_VALUE_MISSING,
                       None, suggested_raw, None, [], [], [], [])

    spec = PARAMETER_REGISTRY.get(parameter)
    if spec is None:
        return _result(False, DECISION_SEMANTIC_UNSUPPORTED, NO_SUPPORTING_RULE,
                       current, suggested_raw, None, [], [], [], [])
    suggested = _normalize_parameter_fact(parameter, suggested_raw, spec)
    if suggested is None:
        return _result(False, DECISION_SEMANTIC_UNSUPPORTED, NO_SUPPORTING_RULE,
                       current, suggested_raw, None, [], [], [], [])

    direction = _compute_direction(current, suggested)
    if direction == _UNCHANGED:
        return _result(False, DECISION_SEMANTIC_UNSUPPORTED, UNCHANGED_VALUE,
                       current, suggested, direction, [], [], [], [])

    evidence_ids = [fid for fid in (evidence_ids or [])]
    # 分类证据：支持 / 冲突 / 中性
    fired_rule_ids: set[str] = set()
    supporting_facts: set[str] = set()
    conflicting_facts: set[str] = set()
    neutral_facts: set[str] = set()
    for fid in evidence_ids:
        fact_value = facts.get(fid)
        matches = [r for r in get_semantic_rules()
                   if r.fact_id == fid and r.parameter == parameter
                   and _rule_matches(r, fact_value)]
        if not matches:
            neutral_facts.add(fid)
            continue
        for rule in matches:
            fired_rule_ids.add(rule.rule_id)
            if rule.allowed_direction == direction:
                supporting_facts.add(fid)
            else:
                conflicting_facts.add(fid)

    if supporting_facts and conflicting_facts:
        return _result(False, DECISION_SEMANTIC_EVIDENCE_CONFLICT, CONFLICTING_EVIDENCE,
                       current, suggested, direction, fired_rule_ids,
                       supporting_facts, conflicting_facts, neutral_facts)
    if not supporting_facts and conflicting_facts:
        return _result(False, DECISION_SEMANTIC_DIRECTION_CONFLICT, DIRECTION_NOT_SUPPORTED,
                       current, suggested, direction, fired_rule_ids,
                       supporting_facts, conflicting_facts, neutral_facts)
    if not supporting_facts:
        return _result(False, DECISION_SEMANTIC_UNSUPPORTED, NO_SUPPORTING_RULE,
                       current, suggested, direction, [], [], [], neutral_facts)

    # 支持关系存在：校验幅度（使用首个支持规则的幅度限制）
    limit = next(r.change_limit for r in get_semantic_rules()
                 if r.rule_id in fired_rule_ids
                 and r.allowed_direction == direction)
    if not _check_change_limit(limit, current, suggested):
        return _result(False, DECISION_SEMANTIC_CHANGE_TOO_LARGE, CHANGE_LIMIT_EXCEEDED,
                       current, suggested, direction, fired_rule_ids,
                       supporting_facts, conflicting_facts, neutral_facts)

    return _result(True, None, None, current, suggested, direction,
                   fired_rule_ids, supporting_facts, conflicting_facts, neutral_facts)


def validate_decision_semantics(decision: dict, fact_package: dict) -> dict:
    """返回结构化语义校验结果；失败时不修改 decision 与 fact_package。

    keep_params（无参数变化）直接通过，不要求训练参数当前值。
    """
    changes = dict(decision.get("hyperparameter_changes") or {})
    changes.update(decision.get("training_overrides") or {})
    if decision.get("action") == "keep_params" or not changes:
        return {
            "valid": True,
            "error_code": None,
            "reason_code": None,
            "parameter": None,
            "rule_ids": [],
            "supporting_fact_ids": [],
            "conflicting_fact_ids": [],
            "neutral_fact_ids": [],
            "current_value": None,
            "suggested_value": None,
            "change_direction": None,
            "parameters": [],
        }

    facts = {f["fact_id"]: f["value"] for f in fact_package["facts"]}
    evidence_by_parameter = decision.get("evidence_ids") or {}
    parameters_results: list[dict] = []
    first_failure: dict | None = None
    for parameter, suggested_raw in changes.items():
        pr = _validate_parameter(parameter, suggested_raw, facts,
                                 evidence_by_parameter.get(parameter, []))
        parameters_results.append(pr)
        if not pr["valid"] and first_failure is None:
            first_failure = pr

    if first_failure is None:
        return {
            "valid": True,
            "error_code": None,
            "reason_code": None,
            "parameter": None,
            "rule_ids": [],
            "supporting_fact_ids": [],
            "conflicting_fact_ids": [],
            "neutral_fact_ids": [],
            "current_value": None,
            "suggested_value": None,
            "change_direction": None,
            "parameters": parameters_results,
        }

    return {
        "valid": False,
        "error_code": first_failure["error_code"],
        "reason_code": first_failure["reason_code"],
        "parameter": first_failure["parameter"],
        "rule_ids": first_failure["rule_ids"],
        "supporting_fact_ids": first_failure["supporting_fact_ids"],
        "conflicting_fact_ids": first_failure["conflicting_fact_ids"],
        "neutral_fact_ids": first_failure["neutral_fact_ids"],
        "current_value": first_failure["current_value"],
        "suggested_value": first_failure["suggested_value"],
        "change_direction": first_failure["change_direction"],
        "parameters": parameters_results,
    }
