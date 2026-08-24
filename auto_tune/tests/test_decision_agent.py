"""Tests for the structured LLM decision boundary."""

import json

import pytest

from auto_tune.modules.agent_engine import decision_agent
from auto_tune.modules.agent_engine.decision_agent import (
    call_decision_llm,
    parse_decision_response,
)
from auto_tune.modules.security.endpoint_policy import EndpointPolicyError


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = (
            payload
            if payload is not None
            else {"choices": [{"message": {"content": "ok"}}]}
        )
        self.text = text

    def json(self):
        return self._payload


def _yaml_key_config():
    return {
        "llm": {
            "api_key": "yaml-secret-must-not-be-used",
            "model": "deepseek-v4-flash",
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "allow_private_endpoint": False,
        }
    }


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


# --- S1.3 security boundary for the decision LLM call ------------------------


def test_decision_uses_resolved_credential_and_ignores_yaml_key(monkeypatch):
    captured = {}
    purposes = []

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse()

    monkeypatch.setattr(
        decision_agent,
        "resolve_credential",
        lambda purpose: purposes.append(purpose) or "resolved-secret",
    )
    monkeypatch.setattr(
        decision_agent,
        "validate_endpoint",
        lambda endpoint, allow_private: "https://resolved.example/v1/chat/completions",
    )
    monkeypatch.setattr(decision_agent.requests, "post", fake_post)

    result = call_decision_llm("prompt", _yaml_key_config())

    assert result == "ok"
    assert purposes == ["text"]
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer resolved-secret"
    assert captured["kwargs"]["allow_redirects"] is False
    assert captured["kwargs"]["timeout"] == (10, 120)
    assert "yaml-secret-must-not-be-used" not in repr(captured)


def test_decision_missing_credential_never_calls_network(monkeypatch):
    called = []

    monkeypatch.setattr(decision_agent, "resolve_credential", lambda purpose: None)
    monkeypatch.setattr(
        decision_agent,
        "validate_endpoint",
        lambda endpoint, allow_private: called.append("endpoint") or endpoint,
    )
    monkeypatch.setattr(
        decision_agent.requests,
        "post",
        lambda **kwargs: called.append("post") or _FakeResponse(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        call_decision_llm("prompt", _yaml_key_config())

    assert "credential_missing" in str(excinfo.value)
    assert called == []


def test_decision_401_error_is_safe_and_never_leaks_body(monkeypatch):
    def fake_post(url, **kwargs):
        return _FakeResponse(
            status_code=401,
            payload={
                "error": {
                    "message": "Incorrect key resolved-secret provider-private-body"
                }
            },
            text="provider-private-body raw",
        )

    monkeypatch.setattr(decision_agent, "resolve_credential", lambda purpose: "resolved-secret")
    monkeypatch.setattr(
        decision_agent,
        "validate_endpoint",
        lambda endpoint, allow_private: "https://resolved.example/v1",
    )
    monkeypatch.setattr(decision_agent.requests, "post", fake_post)

    with pytest.raises(RuntimeError) as excinfo:
        call_decision_llm("prompt", _yaml_key_config())

    message = str(excinfo.value)
    assert "authentication_failed" in message
    assert "401" in message
    assert "resolved-secret" not in message
    assert "provider-private-body" not in message


def test_decision_endpoint_policy_rejection_is_safe(monkeypatch):
    def bad_endpoint(endpoint, allow_private):
        raise EndpointPolicyError("endpoint resolves to a private address")

    called = []
    monkeypatch.setattr(decision_agent, "resolve_credential", lambda purpose: "resolved-secret")
    monkeypatch.setattr(decision_agent, "validate_endpoint", bad_endpoint)
    monkeypatch.setattr(
        decision_agent.requests, "post", lambda **kwargs: called.append(1) or _FakeResponse()
    )

    with pytest.raises(RuntimeError) as excinfo:
        call_decision_llm("prompt", _yaml_key_config())

    assert "endpoint_rejected" in str(excinfo.value)
    assert called == []


def test_decision_real_endpoint_policy_blocks_private(monkeypatch):
    """End-to-end wiring: real validate_endpoint rejects a private target."""
    monkeypatch.setattr(decision_agent, "resolve_credential", lambda purpose: "resolved-secret")
    cfg = _yaml_key_config()
    cfg["llm"]["endpoint"] = "https://127.0.0.1/v1/chat/completions"

    with pytest.raises(RuntimeError) as excinfo:
        call_decision_llm("prompt", cfg)

    assert "endpoint_rejected" in str(excinfo.value)

