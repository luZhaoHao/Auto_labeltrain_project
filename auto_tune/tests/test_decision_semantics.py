"""Q1.2 — 确定性语义校验：证据—参数—方向—幅度。

校验器只读取注册表与事实包，不修改 decision / fact_package，不解释自然语言。
所有结果字段可直接 JSON 序列化，且不含原始 LLM 响应、凭据或路径。
"""

import json

import pytest

from auto_tune.modules.agent_engine.decision_semantics import (
    DECISION_SEMANTIC_CHANGE_TOO_LARGE,
    DECISION_SEMANTIC_CURRENT_VALUE_MISSING,
    DECISION_SEMANTIC_DIRECTION_CONFLICT,
    DECISION_SEMANTIC_EVIDENCE_CONFLICT,
    DECISION_SEMANTIC_UNSUPPORTED,
    validate_decision_semantics,
)
from auto_tune.modules.agent_engine.semantic_rules import DETECT_SEMANTIC_RULES

_BASE_FACTS = {
    "training.issue.overfitting": True,
    "training.issue.underfitting": True,
    "training.issue.plateau": True,
    "training.issue.unstable_training": True,
    "training.issue.nan_loss": True,
    "training.issue.early_stop_too_soon": True,
    "training.curve.val_box_loss": "rising",
    "training.curve.val_cls_loss": "rising",
    "training.curve.mAP50": "saturated",
    "dataset.issue.tiny_bbox_high_ratio": True,
    "dataset.issue.long_tail_class": True,
    "dataset.issue.center_spatial_bias": True,
    "training.params.weight_decay": 0.0005,
    "training.params.epochs": 100,
    "training.params.patience": 20,
    "training.params.lr0": 0.01,
    "training.params.warmup_epochs": 3,
    "training.params.imgsz": 640,
    "training.params.box": 7.5,
    "training.params.cls": 0.5,
    "training.params.translate": 0.0,
    "training.params.cos_lr": False,
    "training.params.batch": 16,
    "training.params.model": "yolov8n.pt",
    "training.params.optimizer": "AdamW",
    "training.params.mosaic": 1.0,
    "training.params.mixup": 0.0,
    "training.params.copy_paste": 0.0,
}

_PARAM_CURRENT = {
    "weight_decay": 0.0005, "epochs": 100, "patience": 20, "lr0": 0.01,
    "warmup_epochs": 3, "imgsz": 640, "box": 7.5, "cls": 0.5,
    "translate": 0.0, "cos_lr": False,
}


def _package(fact_values=None):
    facts = dict(_BASE_FACTS)
    if fact_values:
        facts.update(fact_values)
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "task": "detect",
        "reference_run": "train54",
        "sources": {},
        "facts": [{"fact_id": k, "value": v, "source": "params"}
                  for k, v in facts.items() if v is not None],
    }


def _decision(changes=None, overrides=None, evidence=None, action="adjust"):
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "diagnosis": "d",
        "action": action,
        "hyperparameter_changes": changes or {},
        "training_overrides": overrides or {},
        "evidence_ids": evidence or {},
    }


def _legal_suggestion(parameter, direction):
    current = _PARAM_CURRENT[parameter]
    if direction == "enable":
        return True
    if parameter == "lr0":
        return round(current * 0.6, 6) if direction == "decrease" else round(current * 1.2, 6)
    if parameter == "weight_decay":
        return round(current * 2.0, 6) if direction == "increase" else round(current * 0.5, 6)
    if parameter == "epochs":
        return int(current * 1.5) if direction == "increase" else int(current * 0.6)
    if parameter == "patience":
        return int(current * 1.5) if direction == "increase" else int(current * 0.5)
    if parameter == "warmup_epochs":
        return current + 1 if direction == "increase" else max(current - 1, 0)
    if parameter == "imgsz":
        return int(current * 1.5) if direction == "increase" else int(current * 0.5)
    if parameter in ("box", "cls"):
        return round(current * 1.5, 6) if direction == "increase" else round(current * 0.5, 6)
    if parameter == "translate":
        return round(current + 0.1, 6) if direction == "increase" else round(current - 0.1, 6)
    return current


def _validate(package=None, **decision_kwargs):
    return validate_decision_semantics(_decision(**decision_kwargs), package or _package())


# ── 每条注册规则的合法动作 ──────────────────────────────────────────────────


@pytest.mark.parametrize("rule", DETECT_SEMANTIC_RULES, ids=lambda r: r.rule_id)
def test_every_rule_legal_action(rule):
    suggested = _legal_suggestion(rule.parameter, rule.allowed_direction)
    result = _validate(
        changes={rule.parameter: suggested},
        evidence={rule.parameter: [rule.fact_id]},
    )
    assert result["valid"] is True, rule.rule_id
    assert result["error_code"] is None
    assert result["parameter"] is None


# ── 每条注册规则的相反动作 ──────────────────────────────────────────────────


@pytest.mark.parametrize("rule", DETECT_SEMANTIC_RULES, ids=lambda r: r.rule_id)
def test_every_rule_reverse_direction_rejected(rule):
    package = _package()
    if rule.parameter == "cos_lr":
        package = _package({"training.params.cos_lr": True})  # current True → suggested False = disable
        suggested = False
    else:
        opp = "decrease" if rule.allowed_direction == "increase" else "increase"
        suggested = _legal_suggestion(rule.parameter, opp)
    result = _validate(package,
                       changes={rule.parameter: suggested},
                       evidence={rule.parameter: [rule.fact_id]})
    assert result["valid"] is False, rule.rule_id
    assert result["error_code"] == DECISION_SEMANTIC_DIRECTION_CONFLICT, rule.rule_id
    assert result["reason_code"] == "DIRECTION_NOT_SUPPORTED"
    assert result["parameter"] == rule.parameter


# ── 幅度限制：边界与越界 ────────────────────────────────────────────────────


def _wd_increase(current):
    return _validate(changes={"weight_decay": current * 4.0},
                     evidence={"weight_decay": ["training.issue.overfitting"]},
                     package=_package({"training.params.weight_decay": current}))


def test_weight_decay_increase_boundary_four_x():
    assert _wd_increase(0.0005)["valid"] is True


def test_weight_decay_increase_over_four_x_rejected():
    result = _validate(changes={"weight_decay": 0.0005 * 4.0 + 0.0001},
                       evidence={"weight_decay": ["training.issue.overfitting"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE
    assert result["reason_code"] == "CHANGE_LIMIT_EXCEEDED"


def test_weight_decay_increase_zero_current_cap():
    result = _validate(changes={"weight_decay": 0.001},
                       evidence={"weight_decay": ["training.issue.overfitting"]},
                       package=_package({"training.params.weight_decay": 0.0}))
    assert result["valid"] is True


def test_weight_decay_increase_zero_current_over_cap_rejected():
    result = _validate(changes={"weight_decay": 0.0011},
                       evidence={"weight_decay": ["training.issue.overfitting"]},
                       package=_package({"training.params.weight_decay": 0.0}))
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_weight_decay_decrease_from_zero_cannot_be_legal():
    # 当前值为 0 时无法产生真正的减少：任意非负建议值要么方向相反（>0），要么未变化（==0）。
    result = _validate(changes={"weight_decay": 0.0001},
                       evidence={"weight_decay": ["training.issue.underfitting"]},
                       package=_package({"training.params.weight_decay": 0.0}))
    assert result["valid"] is False


def test_lr0_decrease_lower_boundary():
    result = _validate(changes={"lr0": 0.01 * 0.25},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["valid"] is True


def test_lr0_decrease_upper_boundary():
    result = _validate(changes={"lr0": 0.01 * 0.80},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["valid"] is True


def test_lr0_decrease_below_25_percent_rejected():
    result = _validate(changes={"lr0": 0.01 * 0.24},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_lr0_decrease_above_80_percent_rejected():
    result = _validate(changes={"lr0": 0.01 * 0.81},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_epochs_decrease_lower_boundary():
    result = _validate(changes={"epochs": 50},
                       evidence={"epochs": ["training.issue.overfitting"]})
    assert result["valid"] is True


def test_epochs_decrease_below_half_rejected():
    result = _validate(changes={"epochs": 49},
                       evidence={"epochs": ["training.issue.overfitting"]})
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_epochs_increase_upper_boundary():
    result = _validate(changes={"epochs": 200},
                       evidence={"epochs": ["training.issue.underfitting"]})
    assert result["valid"] is True


def test_epochs_increase_over_double_rejected():
    result = _validate(changes={"epochs": 201},
                       evidence={"epochs": ["training.issue.underfitting"]})
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_warmup_epochs_increase_add_boundary():
    result = _validate(changes={"warmup_epochs": 6},
                       evidence={"warmup_epochs": ["training.issue.unstable_training"]})
    assert result["valid"] is True


def test_warmup_epochs_increase_over_add_rejected():
    result = _validate(changes={"warmup_epochs": 7},
                       evidence={"warmup_epochs": ["training.issue.unstable_training"]})
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


def test_patience_increase_zero_current_cap():
    result = _validate(changes={"patience": 20},
                       evidence={"patience": ["training.issue.early_stop_too_soon"]},
                       package=_package({"training.params.patience": 0}))
    assert result["valid"] is True


def test_patience_increase_zero_current_over_cap_rejected():
    result = _validate(changes={"patience": 21},
                       evidence={"patience": ["training.issue.early_stop_too_soon"]},
                       package=_package({"training.params.patience": 0}))
    assert result["error_code"] == DECISION_SEMANTIC_CHANGE_TOO_LARGE


# ── 整数 / 浮点 / 布尔比较 ──────────────────────────────────────────────────


def test_int_parameter_comparison():
    result = _validate(changes={"epochs": 150},
                       evidence={"epochs": ["training.issue.underfitting"]})
    assert result["parameters"][0]["change_direction"] == "increase"
    assert result["parameters"][0]["current_value"] == 100
    assert result["parameters"][0]["suggested_value"] == 150


def test_float_parameter_comparison():
    result = _validate(changes={"lr0": 0.006},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["parameters"][0]["change_direction"] == "decrease"
    assert result["parameters"][0]["current_value"] == 0.01


def test_bool_parameter_enable_transition():
    result = _validate(changes={"cos_lr": True},
                       evidence={"cos_lr": ["training.issue.plateau"]})
    assert result["valid"] is True
    assert result["parameters"][0]["change_direction"] == "enable"


def test_bool_parameter_disable_transition_rejected():
    result = _validate(changes={"cos_lr": False},
                       evidence={"cos_lr": ["training.issue.plateau"]},
                       package=_package({"training.params.cos_lr": True}))
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_DIRECTION_CONFLICT


def test_cos_lr_non_bool_value_unsupported():
    result = _validate(changes={"cos_lr": "true"},
                       evidence={"cos_lr": ["training.issue.plateau"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_UNSUPPORTED


# ── 当前参数事实缺失 ────────────────────────────────────────────────────────


def test_current_value_missing():
    package = _package({"training.params.lr0": None})
    result = _validate(package,
                       changes={"lr0": 0.006},
                       evidence={"lr0": ["training.issue.plateau"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_CURRENT_VALUE_MISSING
    assert result["reason_code"] == "CURRENT_VALUE_MISSING"
    assert result["parameter"] == "lr0"


# ── 证据组合：支持 + 中性 / 支持 + 冲突 / 全部中性 ──────────────────────────


def test_support_plus_neutral_is_valid():
    result = _validate(changes={"weight_decay": 0.001},
                       evidence={"weight_decay": [
                           "training.issue.overfitting", "training.metrics.mAP50"]})
    assert result["valid"] is True
    pr = result["parameters"][0]
    assert pr["supporting_fact_ids"] == ["training.issue.overfitting"]
    assert "training.metrics.mAP50" in pr["neutral_fact_ids"]


def test_support_plus_conflict_rejected():
    result = _validate(changes={"weight_decay": 0.001},
                       evidence={"weight_decay": [
                           "training.issue.overfitting", "training.issue.underfitting"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_EVIDENCE_CONFLICT
    assert result["reason_code"] == "CONFLICTING_EVIDENCE"


def test_all_neutral_rejected():
    result = _validate(changes={"weight_decay": 0.001},
                       evidence={"weight_decay": ["training.metrics.mAP50"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_UNSUPPORTED
    assert result["reason_code"] == "NO_SUPPORTING_RULE"


# ── 多参数：单项失败整体拒绝 ────────────────────────────────────────────────


def test_multi_parameter_single_failure_rejects_all():
    result = _validate(changes={"weight_decay": 0.001, "lr0": 0.006},
                       evidence={"weight_decay": ["training.issue.overfitting"],
                                 "lr0": ["training.metrics.mAP50"]})
    assert result["valid"] is False
    assert result["parameter"] == "lr0"
    assert result["reason_code"] == "NO_SUPPORTING_RULE"


def test_no_partial_success_fields_when_failed():
    result = _validate(changes={"weight_decay": 0.001, "lr0": 0.006},
                       evidence={"weight_decay": ["training.issue.overfitting"],
                                 "lr0": ["training.metrics.mAP50"]})
    assert result["valid"] is False
    assert result["error_code"] is not None


# ── keep_params 直接通过 ────────────────────────────────────────────────────


def test_keep_params_passes_directly():
    result = _validate(action="keep_params")
    assert result["valid"] is True
    assert result["error_code"] is None
    assert result["parameters"] == []


# ── 未开放参数拒绝 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("param", ["model", "optimizer", "batch", "mosaic", "mixup", "copy_paste"])
def test_unopened_parameter_rejected(param):
    suggested = {"model": "yolov8s.pt", "optimizer": "SGD", "batch": 32,
                 "mosaic": 0.8, "mixup": 0.1, "copy_paste": 0.3}[param]
    result = _validate(changes={param: suggested},
                       evidence={param: ["training.issue.overfitting"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_UNSUPPORTED
    assert result["reason_code"] == "NO_SUPPORTING_RULE"


# ── 建议值未变化拒绝 ────────────────────────────────────────────────────────


def test_unchanged_value_rejected():
    result = _validate(changes={"weight_decay": 0.0005},
                       evidence={"weight_decay": ["training.issue.overfitting"]})
    assert result["valid"] is False
    assert result["error_code"] == DECISION_SEMANTIC_UNSUPPORTED
    assert result["reason_code"] == "UNCHANGED_VALUE"


# ── 结果结构与安全 ──────────────────────────────────────────────────────────


def test_result_is_json_serializable_and_safe():
    result = _validate(changes={"lr0": 0.006},
                       evidence={"lr0": ["training.issue.plateau"]})
    blob = json.dumps(result, ensure_ascii=False)
    assert "api_key" not in blob.lower()
    assert "authorization" not in blob.lower()


def test_result_does_not_mutate_decision_or_package():
    decision = _decision(changes={"lr0": 0.006},
                         evidence={"lr0": ["training.issue.plateau"]})
    package = _package()
    before_decision = json.dumps(decision, ensure_ascii=False)
    before_package = json.dumps(package, ensure_ascii=False)
    validate_decision_semantics(decision, package)
    assert json.dumps(decision, ensure_ascii=False) == before_decision
    assert json.dumps(package, ensure_ascii=False) == before_package
