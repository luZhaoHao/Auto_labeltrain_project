"""Q1.1 Task 2 — TuningDecision v1 结构合同与证据引用校验。

TuningDecision v1 是自动调优专用的 LLM 响应合同。parse 只校验字段结构，
validate_decision_evidence 负责事实包绑定和证据引用完整性。参数取值范围与
组合约束继续由 parameter_registry / guardrails 处理，不在此模块重复。
"""

import json

import pytest

from auto_tune.modules.agent_engine.decision_contract import (
    DECISION_SCHEMA_VERSION,
    ROOT_FIELDS,
    DecisionContractError,
    parse_tuning_decision_response,
    validate_decision_evidence,
)


PACKAGE = {
    "schema_version": "1.0",
    "fact_package_id": "sha256:abc",
    "task": "detect",
    "reference_run": "train54",
    "sources": {},
    "facts": [
        {"fact_id": "training.issue.overfitting", "value": True, "source": "training_report"},
        {"fact_id": "training.params.weight_decay", "value": 0.0005, "source": "params"},
    ],
}


def _decision(**overrides):
    value = {
        "schema_version": "1.0",
        "fact_package_id": "sha256:abc",
        "diagnosis": "存在过拟合",
        "action": "adjust",
        "hyperparameter_changes": {"weight_decay": 0.002},
        "training_overrides": {},
        "evidence_ids": {"weight_decay": ["training.issue.overfitting"]},
    }
    value.update(overrides)
    return value


def _parse(**overrides):
    return parse_tuning_decision_response(json.dumps(_decision(**overrides)))


def _contract_error_code(fn):
    with pytest.raises(DecisionContractError) as excinfo:
        fn()
    return excinfo.value.error_code


# ── 合法决定 ─────────────────────────────────────────────────────────────────


def test_valid_decision_is_normalized():
    parsed = parse_tuning_decision_response(json.dumps(_decision()))
    validated = validate_decision_evidence(parsed, PACKAGE)
    assert validated["schema_version"] == DECISION_SCHEMA_VERSION
    assert validated["fact_package_id"] == "sha256:abc"
    assert validated["evidence_ids"] == {"weight_decay": ["training.issue.overfitting"]}
    assert set(ROOT_FIELDS) == {
        "schema_version", "fact_package_id", "diagnosis", "action",
        "hyperparameter_changes", "training_overrides", "evidence_ids",
    }


def test_keep_params_allows_only_empty_params_and_evidence():
    parsed = parse_tuning_decision_response(json.dumps(_decision(
        action="keep_params", hyperparameter_changes={}, training_overrides={},
        evidence_ids={},
    )))
    validated = validate_decision_evidence(parsed, PACKAGE)
    assert validated["action"] == "keep_params"
    assert validated["evidence_ids"] == {}


@pytest.mark.parametrize("overrides", [
    {"action": "keep_params", "hyperparameter_changes": {}, "training_overrides": {},
     "evidence_ids": {}},
    {"hyperparameter_changes": {"lr0": 0.001}, "training_overrides": {},
     "evidence_ids": {"lr0": ["training.issue.overfitting"]}},
])
def test_evidence_exactly_matches_changes(overrides):
    parsed = _parse(**overrides)
    validated = validate_decision_evidence(parsed, PACKAGE)
    assert validated["hyperparameter_changes"] == overrides["hyperparameter_changes"]


# ── 结构合同失败 ─────────────────────────────────────────────────────────────


def test_extra_root_field_rejected():
    assert _contract_error_code(lambda: _parse(extra_field="x")) == "DECISION_SCHEMA_INVALID"


@pytest.mark.parametrize("schema_version", [None, "", "2.0", 1.0, 1])
def test_wrong_schema_version_rejected(schema_version):
    code = _contract_error_code(lambda: _parse(schema_version=schema_version))
    assert code == "DECISION_SCHEMA_INVALID"


@pytest.mark.parametrize("action", [None, "", "delete", "ADJUST", 3])
def test_invalid_action_rejected(action):
    assert _contract_error_code(lambda: _parse(action=action)) == "DECISION_SCHEMA_INVALID"


def test_duplicate_parameter_buckets_rejected():
    code = _contract_error_code(lambda: _parse(
        hyperparameter_changes={"weight_decay": 0.002},
        training_overrides={"weight_decay": 0.003},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


def test_unknown_parameter_rejected():
    code = _contract_error_code(lambda: _parse(
        hyperparameter_changes={"not_a_param": 1},
        evidence_ids={"not_a_param": ["training.issue.overfitting"]},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


def test_more_than_three_changes_rejected():
    code = _contract_error_code(lambda: _parse(
        hyperparameter_changes={"lr0": 0.001, "box": 8.0, "cls": 0.7},
        training_overrides={"epochs": 200},
        evidence_ids={"lr0": ["training.params.lr0"], "box": ["training.params.box"],
                      "cls": ["training.params.cls"], "epochs": ["training.params.epochs"]},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


@pytest.mark.parametrize("action", ["adjust", "keep_params", None])
def test_non_object_parameter_bucket_rejected(action):
    code = _contract_error_code(lambda: _parse(
        action=action,
        hyperparameter_changes=[1],
        training_overrides={},
        evidence_ids={},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


@pytest.mark.parametrize("evidence", [[], "lr0", 42, None])
def test_non_object_evidence_rejected(evidence):
    code = _contract_error_code(lambda: _parse(evidence_ids=evidence))
    assert code == "DECISION_SCHEMA_INVALID"


@pytest.mark.parametrize("diagnosis", [None, "", "   "])
def test_empty_diagnosis_rejected(diagnosis):
    assert _contract_error_code(lambda: _parse(diagnosis=diagnosis)) == "DECISION_SCHEMA_INVALID"


def test_keep_params_with_params_rejected():
    code = _contract_error_code(lambda: _parse(
        action="keep_params", hyperparameter_changes={"lr0": 0.001},
        training_overrides={}, evidence_ids={"lr0": ["training.params.lr0"]},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


def test_keep_params_with_evidence_rejected():
    code = _contract_error_code(lambda: _parse(
        action="keep_params", hyperparameter_changes={}, training_overrides={},
        evidence_ids={"weight_decay": ["training.issue.overfitting"]},
    ))
    assert code == "DECISION_SCHEMA_INVALID"


def test_not_json_raises_schema_invalid():
    assert _contract_error_code(
        lambda: parse_tuning_decision_response("not json {{")
    ) == "DECISION_SCHEMA_INVALID"


# ── 证据引用校验失败 ─────────────────────────────────────────────────────────


def test_mismatched_package_id_rejected():
    parsed = _parse(fact_package_id="sha256:other")
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_FACT_PACKAGE_MISMATCH"


def test_missing_evidence_key_rejected():
    parsed = _parse(evidence_ids={})
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_MISSING"


def test_extra_evidence_key_rejected():
    parsed = _parse(evidence_ids={
        "weight_decay": ["training.issue.overfitting"],
        "lr0": ["training.params.lr0"],
    })
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_MISSING"


def test_empty_evidence_list_rejected():
    parsed = _parse(evidence_ids={"weight_decay": []})
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_MISSING"


def test_duplicate_fact_id_in_evidence_rejected():
    parsed = _parse(evidence_ids={"weight_decay": [
        "training.issue.overfitting", "training.issue.overfitting",
    ]})
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_MISSING"


def test_unknown_fact_id_rejected():
    parsed = _parse(evidence_ids={"weight_decay": ["invented.fact"]})
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_UNKNOWN"


def test_non_string_fact_id_rejected():
    parsed = _parse(evidence_ids={"weight_decay": [42]})
    code = _contract_error_code(lambda: validate_decision_evidence(parsed, PACKAGE))
    assert code == "DECISION_EVIDENCE_MISSING"
