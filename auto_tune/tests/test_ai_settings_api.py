"""Tests for the S1.3 safe AI-settings / credential / migration API.

Windows Credential Manager access is fully mocked; no test touches real
credentials, real YAML configs, or the network.
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from auto_tune.modules.security import credentials
from auto_tune.ui import app as app_mod


def _write_base_config(path: Path) -> None:
    path.write_text(
        "llm:\n"
        "  provider: deepseek\n"
        "  model: deepseek-chat\n"
        "  endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "  enabled: true\n"
        "  allow_private_endpoint: false\n"
        "vision:\n"
        "  provider: qwen\n"
        "  model: qwen-vl-plus\n"
        "  endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions\n"
        "  enabled: true\n"
        "  allow_private_endpoint: false\n",
        encoding="utf-8",
    )


def _auth_headers(**extra):
    headers = {
        "X-CSRF-Token": app_mod._CSRF_TOKEN,
        "Origin": "http://testserver",
    }
    headers.update(extra)
    return headers


def _delete(client, url, confirm=True):
    return client.request(
        "DELETE",
        url,
        content=json.dumps({"confirm": confirm}),
        headers=_auth_headers(**{"Content-Type": "application/json"}),
    )


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """No real environment variables, Windows credentials, config or network."""
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TUNE_VISION_API_KEY", raising=False)
    credentials.invalidate_credential_cache()

    store: dict[str, str] = {}
    monkeypatch.setattr(
        credentials, "_write_windows_credential", lambda t, v: store.update({t: v})
    )
    monkeypatch.setattr(
        credentials, "_read_windows_credential", lambda t: store.get(t)
    )
    monkeypatch.setattr(
        credentials, "_delete_windows_credential", lambda t: store.pop(t, None)
    )

    cfg_path = tmp_path / "config.yaml"
    _write_base_config(cfg_path)
    monkeypatch.setattr(app_mod, "config_path", cfg_path)
    monkeypatch.setattr(
        app_mod, "APP_CONFIG", yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    )
    return {"store": store, "cfg_path": cfg_path}


def _client():
    return TestClient(app_mod.app)


# ── Read / update settings (spec 12.1, 12.2) ────────────────────────────────


def test_get_ai_settings_never_contains_secret(_isolated_env):
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "top-secret-vault"
    client = _client()

    resp = client.get("/api/ai-settings")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"text", "vision"}
    text = body["text"]
    assert text["purpose"] == "text"
    assert text["configured"] is True
    assert text["source"] == "windows_credential_manager"
    assert text["writable"] is True
    assert "top-secret-vault" not in repr(body)


def test_get_ai_settings_environment_source_read_only(_isolated_env, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret")
    client = _client()

    body = client.get("/api/ai-settings").json()["text"]

    assert body["source"] == "environment"
    assert body["configured"] is True
    assert body["writable"] is False
    assert "env-secret" not in repr(body)


def test_update_settings_rejects_api_key_and_extra_fields(_isolated_env):
    client = _client()
    for payload in [
        {"api_key": "secret"},
        {"credential_ref": "AutoTuneStudio/text/deepseek"},
        {"foo": 1},
        {"model": "x", "surprise": True},
    ]:
        resp = client.put("/api/ai-settings/text", json=payload, headers=_auth_headers())
        assert resp.status_code == 400, payload


def test_update_settings_invalid_purpose_rejected(_isolated_env):
    client = _client()
    resp = client.put("/api/ai-settings/llm", json={"model": "x"}, headers=_auth_headers())
    assert resp.status_code in (400, 404)


def test_update_settings_unsafe_endpoint_rejected(_isolated_env):
    client = _client()
    resp = client.put(
        "/api/ai-settings/text",
        json={"endpoint": "https://127.0.0.1/v1/chat/completions"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_update_settings_accepts_whitelist_and_persists(_isolated_env):
    client = _client()
    resp = client.put(
        "/api/ai-settings/text",
        json={
            "enabled": True,
            "provider": "deepseek",
            "model": "deepseek-chat",
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "allow_private_endpoint": False,
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    saved = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert saved["llm"]["model"] == "deepseek-chat"


def test_update_settings_requires_same_origin(_isolated_env):
    client = _client()
    resp = client.put(
        "/api/ai-settings/text",
        json={"model": "x"},
        headers=_auth_headers(Origin="http://evil.example"),
    )
    assert resp.status_code in (400, 403)


def test_update_settings_requires_csrf(_isolated_env):
    client = _client()
    resp = client.put(
        "/api/ai-settings/text",
        json={"model": "x"},
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code in (400, 403)


# ── Credential lifecycle (spec 12.3, 12.4, 12.5) ─────────────────────────────


def test_credential_environment_write_conflict(_isolated_env, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret")
    client = _client()
    resp = client.put(
        "/api/credentials/text",
        json={"key": "new-secret", "test_before_replace": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 409


def test_credential_delete_environment_conflict(_isolated_env, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_VISION_API_KEY", "env-secret")
    client = _client()
    resp = _delete(client, "/api/credentials/vision")
    assert resp.status_code == 409


def test_credential_test_before_replace_success_stores(_isolated_env, monkeypatch):
    monkeypatch.setattr(
        app_mod, "_probe_connection", lambda purpose, api_key_override=None: "success"
    )
    client = _client()

    resp = client.put(
        "/api/credentials/text",
        json={"key": "brand-new-key", "test_before_replace": True},
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "brand-new-key"
    status = credentials.get_credential_status("text")
    assert status.last_test_result == "success"


def test_credential_test_before_replace_failure_keeps_old(_isolated_env, monkeypatch):
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "old-key"
    credentials.invalidate_credential_cache()
    monkeypatch.setattr(
        app_mod,
        "_probe_connection",
        lambda purpose, api_key_override=None: "authentication_failed",
    )
    client = _client()

    resp = client.put(
        "/api/credentials/text",
        json={"key": "brand-new-key", "test_before_replace": True},
        headers=_auth_headers(),
    )

    assert resp.status_code == 400
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "old-key"


def test_credential_store_untested_when_not_probed(_isolated_env):
    client = _client()

    resp = client.put(
        "/api/credentials/text",
        json={"key": "offline-key", "test_before_replace": False},
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "offline-key"
    assert credentials.get_credential_status("text").last_test_result == "untested"


@pytest.mark.parametrize("bad_key", ["", "   ", "ctrl\x01char", "x" * 600, "YOUR_DEEPSEEK_API_KEY"])
def test_credential_key_validation(_isolated_env, bad_key):
    client = _client()
    resp = client.put(
        "/api/credentials/text",
        json={"key": bad_key, "test_before_replace": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_credential_delete_requires_confirmation(_isolated_env):
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "some-key"
    credentials.invalidate_credential_cache()
    client = _client()

    no_confirm = _delete(client, "/api/credentials/text", confirm=False)
    assert no_confirm.status_code == 400
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "some-key"

    with_confirm = _delete(client, "/api/credentials/text", confirm=True)
    assert with_confirm.status_code == 200
    assert "AutoTuneStudio/text/deepseek" not in _isolated_env["store"]


def test_credential_test_endpoint_returns_safe_category(_isolated_env, monkeypatch):
    monkeypatch.setattr(
        app_mod, "_probe_connection", lambda purpose, api_key_override=None: "rate_limited"
    )
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "some-key"
    credentials.invalidate_credential_cache()
    client = _client()

    resp = client.post(
        "/api/credentials/text/test", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 200
    assert resp.json()["result"] == "rate_limited"
    assert "rate_limited" not in repr(_isolated_env["store"])


def test_credential_test_missing_credential_is_error(_isolated_env):
    client = _client()
    resp = client.post("/api/credentials/text/test", json={}, headers=_auth_headers())
    assert resp.status_code == 400


# ── Legacy migration (spec 12.6, 17) ────────────────────────────────────────


def _write_legacy_config(path: Path, text: str = "legacy-secret"):
    path.write_text(
        "llm:\n"
        f"  api_key: {text}\n"
        "  enabled: true\n"
        "  endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "vision:\n"
        "  provider: qwen\n"
        "  model: qwen-vl-plus\n"
        "  endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions\n"
        "  enabled: true\n",
        encoding="utf-8",
    )


def test_migration_removes_yaml_key_only_after_verified_store(_isolated_env):
    _write_legacy_config(_isolated_env["cfg_path"])
    client = _client()

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 200
    content = _isolated_env["cfg_path"].read_text(encoding="utf-8")
    assert "legacy-secret" not in content
    cfg = yaml.safe_load(content)
    assert "api_key" not in cfg["llm"]
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "legacy-secret"


def test_migration_store_failure_keeps_yaml(_isolated_env, monkeypatch):
    _write_legacy_config(_isolated_env["cfg_path"])
    client = _client()

    def boom(target, value):
        raise credentials.CredentialError("windows credential write failed (error code 5)")

    monkeypatch.setattr(credentials, "_write_windows_credential", boom)

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 500
    assert "api_key" in yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))["llm"]


def test_migration_readback_mismatch_keeps_yaml(_isolated_env, monkeypatch):
    """A store that fails to confirm the write must not modify the YAML.

    The store is empty for the pre-migration existing check, then returns a
    mismatching value on the post-write read-back verification.
    """
    _write_legacy_config(_isolated_env["cfg_path"])
    client = _client()

    read_calls = {"n": 0}

    def flaky_read(target):
        read_calls["n"] += 1
        return None if read_calls["n"] == 1 else "DIFFERENT"

    monkeypatch.setattr(credentials, "_read_windows_credential", flaky_read)

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 500
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert cfg["llm"].get("api_key") == "legacy-secret"


def test_migration_yaml_write_failure_keeps_config(_isolated_env, monkeypatch):
    _write_legacy_config(_isolated_env["cfg_path"])
    client = _client()

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("auto_tune.ui.app.os.replace", fail_replace)

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 500
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert cfg["llm"].get("api_key") == "legacy-secret"
    # The secure credential is retained even when the YAML rewrite fails.
    assert _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "legacy-secret"


def test_migration_environment_active_rejected(_isolated_env, monkeypatch):
    _write_legacy_config(_isolated_env["cfg_path"])
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-secret")
    client = _client()

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 409
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert cfg["llm"].get("api_key") == "legacy-secret"


def test_migration_placeholder_is_not_migrated(_isolated_env):
    _write_legacy_config(_isolated_env["cfg_path"], text="YOUR_DEEPSEEK_API_KEY")
    client = _client()

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 400
    assert "AutoTuneStudio/text/deepseek" not in _isolated_env["store"]


def test_migration_no_legacy_key_rejected(_isolated_env):
    client = _client()
    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )
    assert resp.status_code == 400


def test_migration_cross_origin_rejected(_isolated_env):
    _write_legacy_config(_isolated_env["cfg_path"])
    client = _client()
    resp = client.post(
        "/api/credentials/text/migrate",
        json={},
        headers=_auth_headers(Origin="http://evil.example"),
    )
    assert resp.status_code in (400, 403)
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert cfg["llm"].get("api_key") == "legacy-secret"


def test_migration_conflicts_with_existing_secure_credential(_isolated_env):
    """Existing Windows secure credential must never be overwritten by migration."""
    _write_legacy_config(_isolated_env["cfg_path"])
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "existing-secure-key"
    credentials.invalidate_credential_cache()
    client = _client()

    resp = client.post(
        "/api/credentials/text/migrate", json={}, headers=_auth_headers()
    )

    assert resp.status_code == 409
    assert "existing-secure-key" not in resp.text
    # Both sides keep their original values: the secure credential is untouched
    # and the legacy YAML key is not removed.
    assert (
        _isolated_env["store"]["AutoTuneStudio/text/deepseek"] == "existing-secure-key"
    )
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    assert cfg["llm"].get("api_key") == "legacy-secret"


# ── Two independent services: which capability is usable right now ──────────
#
# DeepSeek serves text diagnosis and LLM tuning; Qwen serves vision analysis.
# The two credentials are independent: neither is required for the other, and
# neither is required for an ordinary training run or an HPO study. When a
# service is enabled but its key was never saved the page has to say so, with
# the service name, instead of letting the capability fail later with a bare
# ``credential_missing``.


def _settings(client, purpose):
    return client.get("/api/ai-settings").json()[purpose]


def test_an_enabled_service_without_a_key_reports_the_gap(_isolated_env):
    body = _settings(_client(), "text")

    assert body["service"] == "deepseek"
    assert body["enabled"] is True
    assert body["configured"] is False
    assert body["missing_credential"] is True


def test_each_service_names_its_own_provider(_isolated_env):
    body = _client().get("/api/ai-settings").json()

    assert body["text"]["service"] == "deepseek"
    assert body["vision"]["service"] == "qwen"


def test_a_saved_key_clears_its_own_gap_only(_isolated_env):
    client = _client()
    _isolated_env["store"]["AutoTuneStudio/text/deepseek"] = "text-secret"
    credentials.invalidate_credential_cache()

    text = _settings(client, "text")
    vision = _settings(client, "vision")

    assert text["configured"] is True and text["missing_credential"] is False
    assert vision["configured"] is False and vision["missing_credential"] is True, (
        "configuring DeepSeek must not claim that vision analysis works")


def test_a_disabled_service_is_not_reported_as_a_missing_key(_isolated_env):
    cfg = yaml.safe_load(_isolated_env["cfg_path"].read_text(encoding="utf-8"))
    cfg["llm"]["enabled"] = False
    cfg["vision"]["enabled"] = False
    _isolated_env["cfg_path"].write_text(yaml.safe_dump(cfg), encoding="utf-8")
    app_mod.APP_CONFIG = cfg

    body = _client().get("/api/ai-settings").json()

    assert body["text"]["missing_credential"] is False, (
        "a disabled service is not advertised, so it has no gap to report")
    assert body["vision"]["missing_credential"] is False


def test_testing_a_service_without_a_key_names_that_service(_isolated_env):
    client = _client()

    resp = client.post("/api/credentials/vision/test", json={}, headers=_auth_headers())

    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "credential_missing"
    assert body["service"] == "qwen"
    assert "qwen" in resp.text.lower()


# ── The settings page contract (zh + en) ────────────────────────────────────


def _render_page(lang: str = "zh", **overrides) -> str:
    from auto_tune.modules.presentation import build_experiment_labels
    from auto_tune.ui.i18n import make_translator

    translator = make_translator(lang)
    ai_config = {
        "text": {"purpose": "text", "enabled": True, "provider": "deepseek",
                 "model": "deepseek-flash", "endpoint": "", "default_endpoint": "",
                 "allow_private_endpoint": False, "migration_required": False},
        "vision": {"purpose": "vision", "enabled": True, "provider": "qwen",
                   "model": "qwen-vl-plus", "endpoint": "", "default_endpoint": "",
                   "allow_private_endpoint": False, "migration_required": False},
    }
    context = dict(
        active_page="projects",
        experiment_history=[],
        experiment_history_source="sqlite",
        experiment_index_warning=None,
        dataset_index=[],
        tuning_history=[],
        dataset=None,
        training=None,
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
        ai_config=ai_config,
        csrf_token="token",
    )
    context.update(overrides)
    return app_mod._jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang=lang,
        experiment_labels=build_experiment_labels(translator),
        **context,
    )


_Zh = "zh"
_En = "en"

_TRAINING_CONTEXT = {
    "summary": {
        "total_runs_analyzed": 1,
        "best_mAP50": 0.4,
        "best_overall_run": "train1",
        "average_mAP50": 0.4,
        "runs_with_issues": 0,
    },
    "runs": {},
    "suggestion": None,
}


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (_Zh, "DeepSeek：文本诊断与大模型调参"),
        (_Zh, "Qwen：训练结果视觉分析"),
        (_En, "DeepSeek · Text Diagnosis & LLM Tuning"),
        (_En, "Qwen · Training-Result Vision Analysis"),
    ],
)
def test_the_page_names_each_service_and_its_capability(lang, expected):
    assert expected in _render_page(lang)


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (_Zh, "两个 Key 相互独立"),
        (_Zh, "普通训练与 HPO 不依赖任何 API Key"),
        (_En, "needs no key at all"),
    ],
)
def test_the_page_states_that_the_two_keys_are_independent(lang, expected):
    assert expected in _render_page(lang)


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        (_Zh, "尚未保存 DeepSeek API Key，文本诊断与大模型调参不可用"),
        (_Zh, "请在“AI 服务配置”中保存 DeepSeek API Key"),
        (_Zh, "尚未保存 Qwen API Key，视觉分析不可用；不使用时可关闭视觉服务"),
        (_En, "No DeepSeek API Key saved yet"),
        (_En, "Save the Qwen API Key in AI Service Settings"),
    ],
)
def test_the_page_carries_the_missing_key_warning_for_both_services(lang, expected):
    assert expected in _render_page(lang)


def test_the_missing_key_warning_is_a_prominent_element_wired_to_live_status():
    page = _render_page(_Zh)

    assert "data-ai-missing-warning" in page, "the warning needs its own element"
    warning = next(
        line for line in page.splitlines() if "data-ai-missing-warning" in line
    )
    assert "#B91C1C" in warning, "the warning must stand out"
    assert "missing_credential" in page, "the warning follows the live status"
    assert "凭据" in page, "each card states its own credential status"


def test_each_card_states_its_own_credential_state():
    page = _render_page(_Zh)

    assert "已配置" in page and "未配置" in page
    assert "_aiField(card, 'status')" in page


def test_a_stored_model_analysis_error_is_projected_to_its_service():
    """A stored ``credential_missing`` must reach the operator as an actionable
    sentence naming the service, never as the bare code."""
    page = _render_page(
        _Zh,
        training=_TRAINING_CONTEXT,
        llm_analysis={"train1": {"error": "DeepSeek API error: credential_missing"}},
        vision_analysis={
            "train1": {
                "confusion_matrix_analysis": {"error": "credential_missing"},
                "error_crop_analysis": {"error": "credential_missing"},
            }
        },
    )

    assert "DeepSeek API error: credential_missing" not in page
    assert "尚未保存 DeepSeek API Key，文本诊断与大模型调参不可用" in page
    assert "尚未保存 Qwen API Key，视觉分析不可用" in page
    assert "请在“AI 服务配置”中保存 Qwen API Key" in page


def test_a_stored_model_analysis_error_is_projected_in_english():
    page = _render_page(
        _En,
        training=_TRAINING_CONTEXT,
        llm_analysis={"train1": {"error": "DeepSeek API error: credential_missing"}},
        vision_analysis={
            "train1": {"confusion_matrix_analysis": {"error": "credential_missing"}}
        },
    )

    assert "DeepSeek API error: credential_missing" not in page
    assert "No DeepSeek API Key saved yet" in page
    assert "No Qwen API Key saved yet" in page


def test_any_other_analysis_error_is_still_shown_verbatim():
    """Only the credential gap is re-worded; every other failure stays as it was
    recorded, so a real provider error is never hidden."""
    page = _render_page(
        _Zh,
        training=_TRAINING_CONTEXT,
        llm_analysis={"train1": {"error": "provider exploded"}},
    )

    assert "provider exploded" in page
