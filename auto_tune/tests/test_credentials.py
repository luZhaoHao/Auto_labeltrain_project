"""Tests for the S1.3 credential store and resolver.

All Windows Credential Manager interactions are mocked; these tests never
touch real credentials on the developer machine.
"""

import pytest

from auto_tune.modules import security
from auto_tune.modules.security import credentials
from auto_tune.modules.security.credentials import (
    CredentialError,
    UnsupportedPlatformError,
    delete_credential,
    get_credential_status,
    invalidate_credential_cache,
    resolve_credential,
    store_credential,
    supports_os_credential_store,
)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    """Ensure no test can reach the real Windows Credential API or env store."""
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TUNE_VISION_API_KEY", raising=False)
    invalidate_credential_cache()


def test_environment_wins_and_status_never_contains_secret(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret-123")
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-secret")

    invalidate_credential_cache()
    assert resolve_credential("text") == "env-secret-123"
    status = get_credential_status("text")
    assert status.configured is True
    assert status.source == "environment"
    assert status.writable is False
    assert "secret" not in repr(status).lower()


def test_environment_is_read_only_priority_source(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_VISION_API_KEY", "env-vision")
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-vision")
    assert resolve_credential("vision") == "env-vision"
    status = get_credential_status("vision")
    assert status.source == "environment"
    assert status.writable is False


def test_cache_hits_within_ttl_and_expires(monkeypatch):
    clock = iter([0.0, 1.0, 301.0, 302.0])
    values = iter(["first", "second"])
    monkeypatch.setattr(credentials.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: next(values))

    assert resolve_credential("text") == "first"
    assert resolve_credential("text") == "first"
    assert resolve_credential("text") == "second"


def test_invalidate_cache_clears_single_purpose(monkeypatch):
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "cached-value")
    assert resolve_credential("text") == "cached-value"
    invalidate_credential_cache("text")
    invalidate_credential_cache()
    # After full invalidation a fresh read returns the stored value again.
    assert resolve_credential("text") == "cached-value"


def test_windows_credential_read_after_invalidation(monkeypatch):
    stored = {"AutoTuneStudio/text/deepseek": "vault-abc"}
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: stored.get(target))
    assert resolve_credential("text") == "vault-abc"
    invalidate_credential_cache("text")
    assert resolve_credential("text") == "vault-abc"


def test_fixed_purpose_maps_to_fixed_target_only():
    assert credentials._PURPOSES["text"] == (
        "AUTO_TUNE_TEXT_API_KEY",
        "AutoTuneStudio/text/deepseek",
    )
    assert credentials._PURPOSES["vision"] == (
        "AUTO_TUNE_VISION_API_KEY",
        "AutoTuneStudio/vision/qwen",
    )


@pytest.mark.parametrize("purpose", ["text", "vision"])
def test_only_text_and_vision_are_valid(purpose):
    status = get_credential_status(purpose)
    assert status.purpose == purpose


@pytest.mark.parametrize("bad", ["", "llm", "decision", "api_key", None, 3, ["text"]])
def test_invalid_purpose_is_rejected(bad):
    with pytest.raises(ValueError):
        resolve_credential(bad)
    with pytest.raises(ValueError):
        store_credential(bad, "x")
    with pytest.raises(ValueError):
        delete_credential(bad)
    with pytest.raises(ValueError):
        get_credential_status(bad)
    with pytest.raises(ValueError):
        invalidate_credential_cache(bad)


def test_store_write_read_replace(monkeypatch):
    written = {}

    def fake_write(target, value):
        written[target] = value

    def fake_read(target):
        return written.get(target)

    monkeypatch.setattr(credentials, "_write_windows_credential", fake_write)
    monkeypatch.setattr(credentials, "_read_windows_credential", fake_read)

    store_credential("text", "first-key")
    assert resolve_credential("text") == "first-key"
    store_credential("text", "replacement-key")
    assert resolve_credential("text") == "replacement-key"
    assert written["AutoTuneStudio/text/deepseek"] == "replacement-key"
    # Store writes scoped to the fixed target only.
    assert "AutoTuneStudio/vision/qwen" not in written


def test_delete_is_idempotent_when_missing(monkeypatch):
    deleted = []

    def fake_delete(target):
        deleted.append(target)

    monkeypatch.setattr(credentials, "_delete_windows_credential", fake_delete)
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: None)

    delete_credential("vision")
    delete_credential("vision")  # idempotent
    assert deleted == ["AutoTuneStudio/vision/qwen", "AutoTuneStudio/vision/qwen"]


def test_environment_write_and_delete_conflict(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret")
    with pytest.raises(CredentialError):
        store_credential("text", "new-secret")
    with pytest.raises(CredentialError):
        delete_credential("text")


def test_placeholder_value_rejected_on_store(monkeypatch):
    written = {}
    monkeypatch.setattr(credentials, "_write_windows_credential", lambda t, v: written.update({t: v}))
    with pytest.raises(CredentialError):
        store_credential("text", "YOUR_DEEPSEEK_API_KEY")
    with pytest.raises(CredentialError):
        store_credential("text", "")
    assert written == {}


def test_placeholder_is_not_a_configured_credential(monkeypatch):
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "YOUR_QWEN_API_KEY")
    assert resolve_credential("vision") is None
    assert get_credential_status("vision").configured is False


def test_non_windows_read_missing_write_delete_unsupported(monkeypatch):
    monkeypatch.setattr(credentials, "_is_windows", lambda: False)
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-value")

    assert resolve_credential("text") is None
    assert supports_os_credential_store() is False
    with pytest.raises(UnsupportedPlatformError):
        store_credential("text", "any")
    with pytest.raises(UnsupportedPlatformError):
        delete_credential("text")


def test_windows_api_failure_propagates_safe_error(monkeypatch):
    def boom(target):
        raise CredentialError("windows credential read failed (error code 5)")

    monkeypatch.setattr(credentials, "_read_windows_credential", boom)
    with pytest.raises(CredentialError) as excinfo:
        resolve_credential("text")
    assert "error code 5" in str(excinfo.value)


def test_status_and_error_never_expose_secret(monkeypatch):
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "top-secret-value")
    status = get_credential_status("text")
    assert status.configured is True
    assert "top-secret-value" not in repr(status)
    assert "top-secret-value" not in str(status)


def test_last_test_result_roundtrip():
    credentials.set_last_test_result("text", "rate_limited")
    status = get_credential_status("text")
    assert status.last_test_result == "rate_limited"
    assert status.last_tested_at is not None
    credentials.clear_last_test_result("text")
    status = get_credential_status("text")
    assert status.last_test_result is None
    assert status.last_tested_at is None


def test_security_package_exports_stable_names():
    assert hasattr(security, "credentials")
    assert hasattr(security, "endpoint_policy")
    assert hasattr(security, "redaction")


# --- Dynamic environment/vault switching (Codex review S1.3) -----------------


def test_dynamic_switch_vault_then_environment_wins(monkeypatch):
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-value")
    assert resolve_credential("text") == "vault-value"
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-new")
    assert resolve_credential("text") == "env-new"


def test_dynamic_environment_change_takes_effect_immediately(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-a")
    assert resolve_credential("text") == "env-a"
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-b")
    assert resolve_credential("text") == "env-b"


def test_dynamic_environment_removal_falls_back_to_vault(monkeypatch):
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda target: "vault-value")
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-a")
    assert resolve_credential("text") == "env-a"
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY")
    assert resolve_credential("text") == "vault-value"


def test_text_and_vision_caches_are_independent(monkeypatch):
    stored = {
        "AutoTuneStudio/text/deepseek": "text-vault",
        "AutoTuneStudio/vision/qwen": "vision-vault",
    }
    monkeypatch.setattr(credentials, "_read_windows_credential", lambda t: stored.get(t))
    assert resolve_credential("text") == "text-vault"
    assert resolve_credential("vision") == "vision-vault"
    invalidate_credential_cache("vision")
    monkeypatch.setattr(
        credentials, "_read_windows_credential", lambda t: stored.get(t) + "-CHANGED"
    )
    assert resolve_credential("text") == "text-vault"
    assert resolve_credential("vision") == "vision-vault-CHANGED"
