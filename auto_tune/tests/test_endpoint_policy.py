"""Tests for S1.3 endpoint policy and shared redaction.

No real network calls are made: DNS is injected with fake resolvers and
private address literals resolve locally without the network.
"""

import socket

import pytest

from auto_tune.modules.security.endpoint_policy import (
    EndpointPolicyError,
    validate_endpoint,
)
from auto_tune.modules.security.redaction import (
    REDACTED,
    redact_sensitive,
    safe_provider_error,
)


@pytest.fixture
def public_dns():
    def resolver(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 443))
        ]

    return resolver


def _ipv6_sockaddr(addr):
    return (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (addr, 443, 0, 0))


# --- endpoint policy --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/key",
        "https://user:pass@example.com/v1/chat/completions",
        "https://example.com/v1?key=x",
        "https://example.com/v1#fragment",
        "http://public.example/v1/chat/completions",
    ],
)
def test_public_endpoint_rejects_unsafe_urls(url, public_dns):
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False, resolver=public_dns)


def test_default_provider_endpoints_accepted(public_dns):
    assert (
        validate_endpoint(
            "https://api.deepseek.com/v1/chat/completions", False, resolver=public_dns
        )
        == "https://api.deepseek.com/v1/chat/completions"
    )
    assert (
        validate_endpoint(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            False,
            resolver=public_dns,
        )
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_custom_https_public_endpoint_accepted(public_dns):
    result = validate_endpoint(
        "https://my-gateway.example.com/v1/chat/completions",
        False,
        resolver=public_dns,
    )
    assert result == "https://my-gateway.example.com/v1/chat/completions"


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.8", "169.254.1.2", "::1"])
def test_private_targets_require_explicit_switch(host):
    url = (
        f"http://[{host}]/v1/chat/completions"
        if ":" in host
        else f"http://{host}/v1/chat/completions"
    )
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False)
    assert validate_endpoint(url, True).startswith("http")


def test_https_private_target_also_requires_switch():
    url = "https://10.0.0.8/v1/chat/completions"
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False)
    assert validate_endpoint(url, True).startswith("https")


def test_private_switch_never_permits_public_http(public_dns):
    url = "http://public.example/v1/chat/completions"
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, True, resolver=public_dns)


def test_mixed_dns_public_and_private_rejected():
    def resolver(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", port or 443)),
        ]

    with pytest.raises(EndpointPolicyError):
        validate_endpoint(
            "https://example.com/v1/chat/completions", False, resolver=resolver
        )


def test_dns_failure_is_rejected(public_dns):
    def resolver(host, port, *args, **kwargs):
        raise socket.gaierror("no such host")

    with pytest.raises(EndpointPolicyError):
        validate_endpoint(
            "https://unknown.example/v1/chat/completions", False, resolver=resolver
        )


@pytest.mark.parametrize("url", ["https://example.com:99999/x", "https://example.com:abc/x"])
def test_invalid_port_rejected(url, public_dns):
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False, resolver=public_dns)


@pytest.mark.parametrize("url", ["https://", "https:///path", ""])
def test_missing_hostname_rejected(url, public_dns):
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(url, False, resolver=public_dns)


def test_non_string_endpoint_rejected(public_dns):
    with pytest.raises(EndpointPolicyError):
        validate_endpoint(12345, False, resolver=public_dns)


# --- redaction --------------------------------------------------------------


def test_redaction_handles_keys_headers_and_free_text_without_mutation():
    source = {
        "Authorization": "Bearer top-secret",
        "nested": {"client_secret_value": "top-secret"},
        "message": "provider echoed Bearer top-secret",
    }
    result = redact_sensitive(source, known_secrets=("top-secret",))
    assert "top-secret" not in repr(result)
    assert source["Authorization"] == "Bearer top-secret"


def test_free_text_replaces_known_secret():
    result = redact_sensitive(
        "provider body echoed key top-secret-abc123", known_secrets=("top-secret-abc123",)
    )
    assert "top-secret-abc123" not in result
    assert REDACTED in result


def test_auth_header_text_replaced():
    text = "request failed: Authorization: Bearer sk-abc123456789"
    result = redact_sensitive(text)
    assert "sk-abc123456789" not in result
    assert "Authorization" not in result


def test_bearer_token_text_replaced():
    result = redact_sensitive("echoed Bearer sk-abcdefghijklmnop")
    assert "sk-abcdefghijklmnop" not in result


def test_normal_text_is_preserved():
    text = "训练完成，mAP50 达到 0.85，没有发现异常 Basic 用法"
    assert redact_sensitive(text) == text


def test_redaction_recurses_and_does_not_mutate():
    source = {"headers": {"Set-Cookie": "sid=abc"}, "items": ["Bearer sk-1234567890abc"]}
    result = redact_sensitive(source)
    assert "abc" not in repr(result).replace("sid=", "")
    assert "sk-1234567890abc" not in repr(result)
    assert source["headers"]["Set-Cookie"] == "sid=abc"


def test_redaction_key_variants():
    result = redact_sensitive(
        {
            "api_key": "x",
            "AccessToken": "y",
            "Refresh_Token": "z",
            "Proxy-Authorization": "w",
            "plain": "keep",
        }
    )
    assert set(result.values()) == {REDACTED, "keep"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, "provider_failed"),
        (400, "provider_failed"),
        (401, "authentication_failed"),
        (403, "authentication_failed"),
        (404, "endpoint_rejected"),
        (429, "rate_limited"),
        (500, "provider_failed"),
        (503, "provider_failed"),
    ],
)
def test_safe_provider_error_mapping(status, expected):
    assert safe_provider_error(status) == expected
