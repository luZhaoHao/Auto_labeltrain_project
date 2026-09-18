"""Tests for the structured LLM decision boundary."""

import json
import re

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


# --- Q1.1 Task 3: auto-tuning TuningDecision v1 contract + one retry ---------


def _tuning_config():
    return {"llm": {"model": "deepseek-v4-flash", "endpoint": "https://api.deepseek.com/v1/chat/completions"}}


def _fact_package():
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:fixed",
        "task": "detect",
        "reference_run": "train54",
        "sources": {},
        "facts": [
            {"fact_id": "training.params.lr0", "value": 0.01, "source": "params"},
            {"fact_id": "training.issue.overfitting", "value": True, "source": "training_report"},
            {"fact_id": "training.issue.plateau", "value": True, "source": "training_report"},
            {"fact_id": "training.metrics.mAP50", "value": 0.6, "source": "metrics"},
        ],
    }


def _tuning_json(evidence_ids=None, changes=None, action="adjust", package_id="sha256:fixed"):
    return json.dumps({
        "schema_version": "1.0",
        "fact_package_id": package_id,
        "diagnosis": "参考训练存在过拟合",
        "action": action,
        "hyperparameter_changes": changes if changes is not None else {"lr0": 0.006},
        "training_overrides": {},
        "evidence_ids": evidence_ids if evidence_ids is not None else {"lr0": ["training.issue.plateau"]},
    }, ensure_ascii=False)


def test_tuning_decision_retries_once_on_unknown_evidence(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["invented.fact"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    replies = iter([bad, good])
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: next(replies))

    result = decision_agent.decide_hyperparameters(package, _tuning_config())

    assert result["error"] is None
    assert result["retried"] is True
    assert result["validation"]["valid"] is True
    assert result["evidence_ids"] == {"lr0": ["training.issue.plateau"]}
    assert result["fact_package_id"] == "sha256:fixed"
    assert result["schema_version"] == "1.0"


def test_tuning_decision_second_evidence_failure_is_stable(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["invented.fact"]})
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: bad)

    result = decision_agent.decide_hyperparameters(package, _tuning_config())

    assert result["error"] == "DECISION_EVIDENCE_UNKNOWN"
    assert result["retried"] is True
    assert result["validation"]["valid"] is False
    assert result["validation"]["error_code"] == "DECISION_EVIDENCE_UNKNOWN"
    assert result["hyperparameter_changes"] == {}
    assert result["evidence_ids"] == {}


def test_tuning_decision_mismatched_package_is_stable(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(package_id="sha256:other")
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: bad)

    result = decision_agent.decide_hyperparameters(package, _tuning_config())

    assert result["error"] == "DECISION_FACT_PACKAGE_MISMATCH"
    assert result["retried"] is True
    assert result["validation"]["valid"] is False


def test_tuning_decision_first_pass_success_no_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda *a, **k: calls.append(1) or _tuning_json(),
    )
    result = decision_agent.decide_hyperparameters(_fact_package(), _tuning_config())
    assert result["error"] is None
    assert result["retried"] is False
    assert len(calls) == 1


def test_tuning_decision_transport_error_no_retry(monkeypatch):
    calls = []

    def boom(prompt, config, **kwargs):
        calls.append(1)
        raise RuntimeError("DeepSeek API error: network_failed")

    monkeypatch.setattr(decision_agent, "call_decision_llm", boom)
    result = decision_agent.decide_hyperparameters(_fact_package(), _tuning_config())

    assert result["error"] == "DeepSeek API error: network_failed"
    assert result["retried"] is False
    assert result["validation"]["valid"] is False
    assert len(calls) == 1  # provider errors never retry


def test_tuning_prompt_embeds_fact_package_not_free_text(monkeypatch):
    prompt = decision_agent.build_tuning_decision_prompt(_fact_package())
    assert "training.params.lr0" in prompt
    assert "training.issue.overfitting" in prompt
    assert "sha256:fixed" in prompt
    assert "fact_package_id" in prompt
    assert "证据" in prompt or "fact_id" in prompt
    # no free-form perception summary and no hard-coded values
    assert "图片总数" not in prompt


# ── Q1.2：语义失败复用一次纠错，二次失败终态 ────────────────────────────────


def test_tuning_decision_semantic_failure_retries_once(monkeypatch):
    package = _fact_package()
    # mAP50 是真实事实但无 lr0 语义规则 → 全部中性 → NO_SUPPORTING_RULE
    bad = _tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    replies = iter([bad, good])
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: next(replies))

    result = decision_agent.decide_hyperparameters(package, _tuning_config())

    assert result["error"] is None
    assert result["retried"] is True
    assert result["validation"]["valid"] is True
    assert result["semantic_validation"]["valid"] is True
    assert result["semantic_validation"]["retried"] is True


def test_tuning_decision_second_semantic_failure_is_terminal(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]})
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: bad)

    result = decision_agent.decide_hyperparameters(package, _tuning_config())

    assert result["error"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert result["retried"] is True
    assert result["validation"]["valid"] is False
    assert result["semantic_validation"]["valid"] is False
    assert result["semantic_validation"]["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert result["semantic_validation"]["reason_code"] == "NO_SUPPORTING_RULE"
    assert result["semantic_validation"]["retried"] is True


def test_tuning_prompt_semantic_summary_comes_from_registry():
    prompt = decision_agent.build_tuning_decision_prompt(_fact_package())
    assert "允许的超参数修改关系" in prompt
    assert "training.issue.plateau" in prompt
    assert "禁止修改参数" in prompt
    assert "lr0" in prompt
    # 语义摘要必须来自注册表：weight_decay 规则与超参列表都出现在提示词中
    assert "weight_decay" in prompt


def test_generate_suggestion_keeps_old_four_field_contract(monkeypatch):
    old = json.dumps({
        "diagnosis": "学习率偏高",
        "action": "降低学习率",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {},
    })
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda p, c, **k: old)
    result = decision_agent.generate_suggestion("summary", None, _tuning_config())
    assert result["error"] is None
    assert result["hyperparameter_changes"] == {"lr0": 0.002}
    assert "fact_package_id" not in result
    assert "evidence_ids" not in result


# ── Q1.1 返修：事实包是自动调优提示词的唯一来源 ─────────────────────────────


def test_tuning_prompt_unaffected_by_any_history():
    """The same fact package must always produce the same prompt: history must
    never change the auto-tuning LLM input."""
    package = _fact_package()
    first = decision_agent.build_tuning_decision_prompt(package)
    second = decision_agent.build_tuning_decision_prompt(package)
    assert first == second
    assert "历史调参" not in first
    assert "尝试 1" not in first
    # prompt content comes only from the fact package
    assert "training.params.lr0" in first
    assert "training.issue.overfitting" in first
    assert "fact_package_id" in first


# ── 提示词可调集合必须与语义规则开放集合一致 ───────────────────────────────


def _prompt_parameter_lists(prompt: str):
    """从提示词里取出「可修改 / 禁止修改」两份参数清单。"""
    opened = closed = None
    for line in prompt.splitlines():
        if line.startswith("- 可修改参数："):
            opened = {p.strip() for p in line.split("：", 1)[1].split(",") if p.strip()}
        elif line.startswith("- 禁止修改参数："):
            closed = {p.strip() for p in line.split("：", 1)[1].split(",") if p.strip()}
    return opened, closed


def test_tuning_prompt_advertises_only_semantically_open_parameters():
    """提示词宣布的可调集合必须等于语义规则开放集合。

    宣布全部可调参数而其中大多数没有任何语义关系，等于把一批必然失败的选择
    摆到模型面前：模型一旦选中就是 DECISION_SEMANTIC_UNSUPPORTED /
    NO_SUPPORTING_RULE，本轮直接失败（生产真实审计中已出现同类失败）。
    未开放参数必须被显式标注为禁止修改，而不是只在末尾附带一句。
    """
    from auto_tune.modules.agent_engine.parameter_registry import (
        get_tunable_parameter_names,
    )
    from auto_tune.modules.agent_engine.semantic_rules import get_semantic_parameter_set

    opened_param_set = set(get_semantic_parameter_set())
    all_tunable = set(get_tunable_parameter_names())

    opened, closed = _prompt_parameter_lists(
        decision_agent.build_tuning_decision_prompt(_fact_package()))

    assert opened == opened_param_set
    assert closed == all_tunable - opened_param_set
    assert not (opened & closed)


def test_tuning_prompt_states_that_closed_parameters_fail_the_round():
    prompt = decision_agent.build_tuning_decision_prompt(_fact_package())
    assert "禁止修改参数" in prompt
    # 规则 1 不得再把「全部可调参数」说成允许修改
    assert "只允许调整以下参数" not in prompt


def test_build_tuning_decision_prompt_rejects_previous_attempts():
    with pytest.raises(TypeError):
        decision_agent.build_tuning_decision_prompt(_fact_package(), previous_attempts=[])


def test_decide_hyperparameters_rejects_previous_attempts(monkeypatch):
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: _tuning_json())
    with pytest.raises(TypeError):
        decision_agent.decide_hyperparameters(
            _fact_package(), _tuning_config(), previous_attempts=[]
        )


# ── 返修一：每次模型响应校验后立即持久化 attempt ────────────────────────────


def test_tuning_decision_persists_attempt_1_before_correction(monkeypatch):
    """第一次响应校验后先写 attempt，成功后才发起唯一一次纠错调用。"""
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    replies = iter([bad, good])
    order = []
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda *a, **k: order.append("llm") or next(replies),
    )
    attempts = []

    def on_attempt(attempt):
        order.append(f"attempt{attempt['attempt']}")
        attempts.append(attempt)

    result = decision_agent.decide_hyperparameters(package, _tuning_config(), on_attempt=on_attempt)

    assert result["error"] is None
    assert result["retried"] is True
    first_llm = order.index("llm")
    second_llm = order.index("llm", first_llm + 1)
    assert first_llm < order.index("attempt1") < second_llm < order.index("attempt2")
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert [a["retried"] for a in attempts] == [False, True]


def test_tuning_decision_attempt_record_content(monkeypatch):
    """attempt 记录含完整 TuningDecision、Q1.1 与 Q1.2 结果，且两次同一 fact_package_id。"""
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    replies = iter([bad, good])
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: next(replies))
    attempts = []

    result = decision_agent.decide_hyperparameters(package, _tuning_config(), on_attempt=attempts.append)

    assert len(attempts) == 2
    a1, a2 = attempts
    assert a1["attempt"] == 1 and a1["retried"] is False
    assert a1["decision"]["fact_package_id"] == "sha256:fixed"
    assert a1["decision"]["schema_version"] == "1.0"
    assert a1["decision"]["evidence_ids"] == {"lr0": ["training.metrics.mAP50"]}
    assert a1["decision_validation"]["valid"] is False
    assert a1["decision_validation"]["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert a1["semantic_validation"]["valid"] is False
    assert a1["semantic_validation"]["error_code"] == "DECISION_SEMANTIC_UNSUPPORTED"
    assert a1["semantic_validation"]["reason_code"] == "NO_SUPPORTING_RULE"
    assert a2["attempt"] == 2 and a2["retried"] is True
    assert a2["decision"]["fact_package_id"] == "sha256:fixed"
    assert a2["decision"]["evidence_ids"] == {"lr0": ["training.issue.plateau"]}
    assert a2["decision_validation"]["valid"] is True
    assert a2["semantic_validation"]["valid"] is True
    assert a1["decision"]["fact_package_id"] == a2["decision"]["fact_package_id"] == "sha256:fixed"
    assert result["validation"]["retried"] is True


def test_tuning_decision_attempt_persistence_failure_blocks_correction(monkeypatch):
    """attempt 1 审计写入失败：不允许发起纠错调用。"""
    package = _fact_package()
    replies = iter([_tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]}),
                    _tuning_json()])
    llm_calls = []
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda *a, **k: llm_calls.append(1) or next(replies),
    )

    def boom(attempt):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        decision_agent.decide_hyperparameters(package, _tuning_config(), on_attempt=boom)

    assert len(llm_calls) == 1


def test_tuning_decision_attempt_2_persistence_failure_stops(monkeypatch):
    """attempt 2 审计写入失败：即便响应有效也不放行。"""
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["training.metrics.mAP50"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    replies = iter([bad, good])
    llm_calls = []
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda *a, **k: llm_calls.append(1) or next(replies),
    )

    def boom(attempt):
        if attempt["attempt"] == 2:
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        decision_agent.decide_hyperparameters(package, _tuning_config(), on_attempt=boom)

    assert len(llm_calls) == 2


def test_tuning_decision_first_pass_records_single_attempt(monkeypatch):
    """首次即通过时只持久化一次 attempt。"""
    package = _fact_package()
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: _tuning_json())
    attempts = []

    result = decision_agent.decide_hyperparameters(package, _tuning_config(), on_attempt=attempts.append)

    assert result["error"] is None
    assert result["retried"] is False
    assert len(attempts) == 1
    assert attempts[0]["attempt"] == 1
    assert attempts[0]["retried"] is False
    assert attempts[0]["decision_validation"]["valid"] is True
    assert attempts[0]["semantic_validation"]["valid"] is True


# ── 返修二：提示词不再禁止模型自行输出新参数值 ──────────────────────────────


def test_initial_prompt_does_not_forbid_new_parameter_values():
    prompt = decision_agent.build_tuning_decision_prompt(_fact_package())
    assert "不要提供替代参数值" not in prompt
    assert "不要提供或猜测替代参数值" not in prompt
    # 系统不替 LLM 推荐/计算具体修正值；LLM 须依据同一事实包与规则自行输出
    assert "系统不会为你提供或猜测替代参数值" in prompt
    assert "自行输出完整且合法的 TuningDecision v1" in prompt


def test_semantic_correction_prompt_does_not_forbid_new_values():
    prompt = decision_agent.build_semantic_correction_prompt(
        "p", _fact_package(), "DECISION_SEMANTIC_CHANGE_TOO_LARGE", "lr0", "CHANGE_LIMIT_EXCEEDED",
    )
    assert "不要提供或猜测替代参数值" not in prompt
    assert "不要提供替代参数值" not in prompt
    assert "系统不会提供具体修正值" in prompt
    assert "重新独立输出完整且合法的 TuningDecision v1" in prompt


def test_semantic_correction_prompt_has_no_computed_substitute_value():
    prompt = decision_agent.build_semantic_correction_prompt(
        "p", _fact_package(), "DECISION_SEMANTIC_CHANGE_TOO_LARGE", "lr0", "CHANGE_LIMIT_EXCEEDED",
    )
    # 系统不得给出具体建议值（0.006/0.005 不在事实包或规则摘要中）
    assert "0.006" not in prompt
    assert "0.005" not in prompt


def test_semantic_correction_prompt_carries_rules_and_package_id():
    prompt = decision_agent.build_semantic_correction_prompt(
        "ORIGINAL", _fact_package(), "DECISION_SEMANTIC_CHANGE_TOO_LARGE", "lr0", "CHANGE_LIMIT_EXCEEDED",
    )
    assert "稳定错误码: DECISION_SEMANTIC_CHANGE_TOO_LARGE" in prompt
    assert "失败参数: lr0" in prompt
    assert "失败原因: CHANGE_LIMIT_EXCEEDED" in prompt
    # 允许的证据关系 / 方向 / 幅度来自规则注册表
    assert "参数 lr0 的允许关系" in prompt
    assert "方向" in prompt
    assert "幅度" in prompt
    # 同一 fact_package_id
    assert "sha256:fixed" in prompt
    # 要求重新输出完整 TuningDecision v1，并回带原始任务
    assert "重新独立输出完整且合法的 TuningDecision v1" in prompt
    assert "ORIGINAL" in prompt


# ── F1.1-A 复审 Task 2：model 不再是 LLM 可调参数 ──────────────────────────

_MODEL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])model(?![A-Za-z0-9_])")


def test_legacy_decision_prompt_does_not_offer_model_as_a_tunable_parameter():
    """旧决策提示词不得再把 model 列入 training_overrides / 可调参数。"""
    prompt = decision_agent.build_decision_prompt("感知摘要（不含权重字样）")
    assert _MODEL_TOKEN_RE.search(prompt) is None


def test_structured_decision_prompt_does_not_offer_model_as_a_tunable_parameter():
    """新版结构化决策提示词（允许参数列表 + 语义摘要）不得出现 model。"""
    prompt = decision_agent.build_tuning_decision_prompt(_fact_package())
    allowed = [line for line in prompt.splitlines() if line.startswith("- 可修改参数：")]
    assert allowed, "提示词缺少可修改参数列表"
    assert "model" not in allowed[0]
    assert _MODEL_TOKEN_RE.search(prompt) is None


def test_legacy_parser_rejects_a_model_change():
    """旧解析入口同样在结构边界拒绝 model（不进入语义校验）。"""
    raw = json.dumps({
        "diagnosis": "换模型",
        "action": "换更大模型",
        "hyperparameter_changes": {"model": "yolo11n.pt"},
        "training_overrides": {},
    })
    result = parse_decision_response(raw)
    assert result["error"] == "Unknown parameter(s): model"


# ── F1.1-A 第二轮复审 Task 1：提示词不得再诱导“换更大模型” ────────────────────

# 第一轮复审已复现：model 虽已脱离可调参数注册表，旧提示词规则8 仍写着“换更大模型”，
# 于是 LLM 按提示输出 model，又在结构契约处以未知参数被拒，导致整轮调优失败。
_MODEL_SWAP_PHRASES = (
    "换更大模型", "换模型", "更换模型", "更大模型", "更换权重", "切换权重",
    "换权重的模型",
)


def _model_fact_package():
    """真实 FactPackage：参考运行的 model 作为只读事实进入事实包。"""
    package = _fact_package()
    package["facts"] = list(package["facts"]) + [
        {"fact_id": "training.params.model", "value": "yolov8n.pt", "source": "params"},
    ]
    return package


def _underfitting_rule(prompt: str) -> str:
    """抽取“规则8：欠拟合”整段（到下一个同级/更高级小节为止）。"""
    start = prompt.index("### 规则8")
    rest = prompt[start:]
    boundaries = [pos for pos in (rest.find("\n### ", 1), rest.find("\n## ", 1))
                  if pos != -1]
    return rest if not boundaries else rest[:min(boundaries)]


def test_legacy_decision_prompt_never_suggests_swapping_the_model():
    """旧决策提示词不得再给出任何“换更大模型 / 更换权重”的正向调优动作。"""
    prompt = decision_agent.build_decision_prompt("感知摘要")
    for phrase in _MODEL_SWAP_PHRASES:
        assert phrase not in prompt, f"旧提示词仍在建议 {phrase}"


def test_legacy_underfitting_rule_only_offers_supported_parameters():
    """欠拟合规则只能建议允许且有语义规则支撑的参数，并声明权重不可修改。"""
    rule = _underfitting_rule(decision_agent.build_decision_prompt("感知摘要"))
    for phrase in _MODEL_SWAP_PHRASES:
        assert phrase not in rule
    # 模型 / 初始权重属于参考运行的不可变条件
    assert "禁止" in rule
    assert "模型" in rule or "权重" in rule
    # 只保留有语义规则支撑的方向：epochs 增大、weight_decay 减小，小目标时 imgsz/box 增大
    for parameter in ("epochs", "weight_decay", "imgsz", "box"):
        assert parameter in rule
    # 欠拟合在语义规则里没有任何 lr0 关系，方向相反的建议更不允许出现
    assert "lr0" not in rule


def test_structured_decision_prompt_shows_the_reference_model_as_a_read_only_fact():
    """含真实 training.params.model 事实时：事实可见、参数列表不含它、写入被明确禁止。"""
    prompt = decision_agent.build_tuning_decision_prompt(_model_fact_package())
    # 参考模型仍是只读事实，不得为了让 token 检查通过而把它从事实包里删掉
    assert "training.params.model" in prompt
    assert "yolov8n.pt" in prompt

    allowed = [line for line in prompt.splitlines() if line.startswith("- 可修改参数：")]
    assert allowed, "提示词缺少可修改参数列表"
    assert "model" not in allowed[0]

    forbid = [line for line in prompt.splitlines()
              if "禁止" in line
              and "hyperparameter_changes" in line
              and "training_overrides" in line]
    assert forbid, "结构化提示词必须明确禁止把模型写入两个修改对象"
    assert any("模型" in line or "权重" in line for line in forbid)


def test_neither_decision_prompt_suggests_swapping_the_model():
    """两套仍在使用的提示词都不得出现任何“建议换模型”的语义。"""
    prompts = {
        "legacy": decision_agent.build_decision_prompt("感知摘要"),
        "structured": decision_agent.build_tuning_decision_prompt(_model_fact_package()),
    }
    for name, prompt in prompts.items():
        for phrase in _MODEL_SWAP_PHRASES:
            assert phrase not in prompt, f"{name} 提示词仍在建议 {phrase}"


# ── 结构化调参请求的 JSON 输出约束与文本模型默认值 ──────────────────────────


def _capture_payloads(monkeypatch, contents):
    """真实走完 call_decision_llm，只把每次请求的 JSON body 记下来。"""
    payloads = []
    replies = iter(contents)

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return _FakeResponse(
            payload={"choices": [{"message": {"content": next(replies)}}]}
        )

    monkeypatch.setattr(decision_agent, "resolve_credential", lambda purpose: "resolved-secret")
    monkeypatch.setattr(
        decision_agent,
        "validate_endpoint",
        lambda endpoint, allow_private: "https://resolved.example/v1/chat/completions",
    )
    monkeypatch.setattr(decision_agent.requests, "post", fake_post)
    return payloads


def test_tuning_decision_request_carries_json_response_format(monkeypatch):
    payloads = _capture_payloads(monkeypatch, [_tuning_json()])

    result = decision_agent.decide_hyperparameters(_fact_package(), _tuning_config())

    assert result["error"] is None
    assert len(payloads) == 1
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_tuning_correction_retry_request_carries_json_response_format(monkeypatch):
    bad = _tuning_json(evidence_ids={"lr0": ["invented.fact"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.issue.plateau"]})
    payloads = _capture_payloads(monkeypatch, [bad, good])

    result = decision_agent.decide_hyperparameters(_fact_package(), _tuning_config())

    assert result["error"] is None
    assert result["retried"] is True
    assert len(payloads) == 2
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[1]["response_format"] == {"type": "json_object"}


def test_plain_text_decision_call_has_no_json_response_format(monkeypatch):
    """终局摘要等纯文本调用不得被强制成 JSON 输出。"""
    payloads = _capture_payloads(monkeypatch, ["本轮 lr0 调整有效"])

    assert call_decision_llm("请用中文给出文本总结", _yaml_key_config()) == "本轮 lr0 调整有效"

    assert "response_format" not in payloads[0]


def test_decision_llm_default_text_model_is_deepseek_flash(monkeypatch):
    payloads = _capture_payloads(monkeypatch, ["ok"])

    call_decision_llm("prompt", {"llm": {"endpoint": "https://api.deepseek.com/v1/x"}})

    assert payloads[0]["model"] == "deepseek-flash"


def test_decision_llm_configured_model_still_wins(monkeypatch):
    payloads = _capture_payloads(monkeypatch, ["ok"])

    call_decision_llm("prompt", _yaml_key_config())

    assert payloads[0]["model"] == "deepseek-v4-flash"


_SUGGESTION_JSON = json.dumps({
    "diagnosis": "学习率偏高",
    "action": "降低学习率",
    "hyperparameter_changes": {"lr0": 0.002},
    "training_overrides": {},
}, ensure_ascii=False)


def test_generate_suggestion_request_carries_json_response_format(monkeypatch):
    payloads = _capture_payloads(monkeypatch, [_SUGGESTION_JSON])

    result = decision_agent.generate_suggestion("summary", None, _tuning_config())

    assert result["error"] is None
    assert len(payloads) == 1
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_generate_suggestion_json_fix_retry_carries_json_response_format(monkeypatch):
    payloads = _capture_payloads(monkeypatch, ["完全不是 JSON {{", _SUGGESTION_JSON])

    result = decision_agent.generate_suggestion("summary", None, _tuning_config())

    assert result["error"] is None
    assert result["retried"] is True
    assert len(payloads) == 2
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[1]["response_format"] == {"type": "json_object"}
    # 纠错重试仍是同一个任务，只是要求重新输出合法 JSON
    assert "JSON" in payloads[1]["messages"][1]["content"]
