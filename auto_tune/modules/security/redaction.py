"""Shared redaction and safe provider error classification.

Structured objects are redacted recursively by sensitive key names; free text
gets a limited auth-header/token-pattern replacement plus known-secret literal
replacement.  Redaction never mutates the source object.
"""

from __future__ import annotations

import re

REDACTED = "***REDACTED***"

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "cookie",
)

# Standalone "Bearer <token>" / "Basic <token>" shapes; requires a token of at
# least 8 characters so ordinary English words like "Basic HTML" are not hit.
_AUTH_TOKEN_PATTERN = re.compile(r"(?i)(Bearer|Basic)\s+[A-Za-z0-9._~+/=;-]{8,}")

# Explicit header-style text such as "Authorization: Bearer sk-...".
_HEADER_PATTERN = re.compile(
    r"(?i)(Authorization|Proxy-Authorization|Cookie|Set-Cookie)\s*[:=]\s*\S+"
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _redact_text(text: str, known_secrets: tuple[str, ...]) -> str:
    for secret in known_secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    text = _AUTH_TOKEN_PATTERN.sub(REDACTED, text)
    text = _HEADER_PATTERN.sub(REDACTED, text)
    return text


def redact_sensitive(value: object, known_secrets: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_sensitive(item, known_secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item, known_secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item, known_secrets) for item in value)
    if isinstance(value, str):
        return _redact_text(value, known_secrets)
    return value


def safe_provider_error(status_code: int) -> str:
    """Map an HTTP status code to one of the fixed safe error categories."""
    if status_code in (401, 403):
        return "authentication_failed"
    if status_code == 429:
        return "rate_limited"
    if status_code == 404:
        return "endpoint_rejected"
    return "provider_failed"
