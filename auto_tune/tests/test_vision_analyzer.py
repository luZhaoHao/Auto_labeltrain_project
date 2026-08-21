"""Tests for the S1.3 security boundary around the vision LLM call."""

from auto_tune.modules.train_analyzer import vision_analyzer


class _FakeVisionResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = (
            payload
            if payload is not None
            else {"choices": [{"message": {"content": "视觉分析结果"}}]}
        )
        self.text = text

    def json(self):
        return self._payload


def _yaml_key_config():
    return {
        "vision": {
            "api_key": "yaml-secret-must-not-be-used",
            "model": "qwen-vl-plus",
            "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "allow_private_endpoint": False,
        }
    }


def test_vision_uses_resolved_credential_and_ignores_yaml_key(monkeypatch):
    captured = {}
    purposes = []

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeVisionResponse()

    monkeypatch.setattr(vision_analyzer, "_encode_image", lambda path: "aW1n")
    monkeypatch.setattr(
        vision_analyzer,
        "resolve_credential",
        lambda purpose: purposes.append(purpose) or "resolved-secret",
    )
    monkeypatch.setattr(
        vision_analyzer,
        "validate_endpoint",
        lambda endpoint, allow_private: "https://resolved.example/v1/chat/completions",
    )
    monkeypatch.setattr(vision_analyzer.requests, "post", fake_post)

    result = vision_analyzer._call_vision_api("dummy.png", _yaml_key_config(), "prompt")

    assert purposes == ["vision"]
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer resolved-secret"
    assert captured["kwargs"]["allow_redirects"] is False
    assert captured["kwargs"]["timeout"] == (10, 120)
    assert "yaml-secret-must-not-be-used" not in repr(captured)
    assert result["error"] is None
    assert "视觉分析结果" in result["analysis"]


def test_vision_missing_credential_returns_safe_error(monkeypatch):
    called = []
    monkeypatch.setattr(vision_analyzer, "resolve_credential", lambda purpose: None)
    monkeypatch.setattr(vision_analyzer, "_encode_image", lambda path: "aW1n")
    monkeypatch.setattr(
        vision_analyzer,
        "validate_endpoint",
        lambda endpoint, allow_private: called.append("endpoint") or endpoint,
    )
    monkeypatch.setattr(
        vision_analyzer.requests, "post", lambda **kwargs: called.append("post") or _FakeVisionResponse()
    )

    result = vision_analyzer._call_vision_api("dummy.png", _yaml_key_config(), "prompt")

    assert result["error"] == "credential_missing"
    assert called == []


def test_vision_401_error_is_safe_and_never_leaks_body(monkeypatch):
    def fake_post(url, **kwargs):
        return _FakeVisionResponse(
            status_code=401,
            payload={"error": {"message": "resolved-secret provider-private-body"}},
            text="provider-private-body raw",
        )

    monkeypatch.setattr(vision_analyzer, "_encode_image", lambda path: "aW1n")
    monkeypatch.setattr(vision_analyzer, "resolve_credential", lambda purpose: "resolved-secret")
    monkeypatch.setattr(
        vision_analyzer, "validate_endpoint", lambda e, a: "https://resolved.example/v1"
    )
    monkeypatch.setattr(vision_analyzer.requests, "post", fake_post)

    result = vision_analyzer._call_vision_api("dummy.png", {"vision": {}}, "prompt")

    assert "authentication_failed" in result["error"]
    assert "401" in result["error"]
    assert "resolved-secret" not in repr(result)
    assert "provider-private-body" not in repr(result)


def test_vision_real_endpoint_policy_blocks_private(monkeypatch):
    monkeypatch.setattr(vision_analyzer, "_encode_image", lambda path: "aW1n")
    monkeypatch.setattr(vision_analyzer, "resolve_credential", lambda purpose: "resolved-secret")

    result = vision_analyzer._call_vision_api(
        "dummy.png", {"vision": {"endpoint": "http://127.0.0.1:8000/v1/chat/completions"}}, "prompt"
    )

    assert result["error"] == "endpoint_rejected"
