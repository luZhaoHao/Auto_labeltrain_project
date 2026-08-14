"""Tests for the structured LLM decision boundary."""

import json

import pytest

from auto_tune.modules.agent_engine.decision_agent import parse_decision_response


def test_valid_decision_is_normalized():
    raw = json.dumps({
        "diagnosis": "学习率偏高",
        "action": "降低学习率",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {"optimizer": "AdamW"},
    })

    result = parse_decision_response(raw)

    assert result["error"] is None
    assert result["hyperparameter_changes"] == {"lr0": 0.002}


@pytest.mark.parametrize("payload", [
    {"diagnosis": "x", "action": "x", "hyperparameter_changes": [1]},
    {"diagnosis": "x", "action": "x", "hyperparameter_changes": {"unknown": 1}},
    {"diagnosis": 3, "action": "x", "hyperparameter_changes": {}},
    {"diagnosis": "x", "action": "x", "hyperparameter_changes": {}, "training_overrides": {}},
])
def test_malformed_or_ambiguous_decision_is_rejected(payload):
    result = parse_decision_response(json.dumps(payload))

    assert result["error"]


def test_keep_params_is_the_only_valid_empty_change_action():
    raw = json.dumps({
        "diagnosis": "指标稳定",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
    })

    result = parse_decision_response(raw)

    assert result["error"] is None
    assert result["action"] == "keep_params"


def test_normal_mode_limits_each_iteration_to_three_changes():
    raw = json.dumps({
        "diagnosis": "x",
        "action": "调整多个参数",
        "hyperparameter_changes": {
            "lr0": 0.002,
            "box": 8.0,
            "cls": 0.7,
            "mosaic": 0.5,
        },
        "training_overrides": {},
    })

    result = parse_decision_response(raw)

    assert "最多修改 3 个" in result["error"]

