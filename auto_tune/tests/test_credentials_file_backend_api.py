"""The Studio page on Linux/Docker: saving, replacing and deleting a key.

The container has no OS credential store, so the settings page must be able to
persist a key to the mounted credential file and report honestly where the key
comes from. The file backend itself is covered by
``test_credentials_file_backend.py``; this suite drives the HTTP contract and
the page contract the operator actually uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from auto_tune.modules.security import credentials
from auto_tune.ui import app as app_mod

TEXT_KEY = "text-key-aaaaaaaa"
VISION_KEY = "vision-key-bbbbbbbb"


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
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    headers.update(extra)
    return headers


def _client() -> TestClient:
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def _container_environment(monkeypatch, tmp_path):
    """A Linux container: no Windows store, a controlled credential file."""
    monkeypatch.delenv("AUTO_TUNE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TUNE_VISION_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_is_windows", lambda: False)
    path = tmp_path / "secrets" / "credentials.json"
    monkeypatch.setenv("AUTO_TUNE_CREDENTIALS_PATH", str(path))
    credentials.invalidate_credential_cache()

    cfg_path = tmp_path / "config.yaml"
    _write_base_config(cfg_path)
    monkeypatch.setattr(app_mod, "config_path", cfg_path)
    monkeypatch.setattr(
        app_mod, "APP_CONFIG", yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    )
    yield path
    credentials.invalidate_credential_cache()


def _put(client, purpose, key, test_before_replace=False):
    return client.put(
        f"/api/credentials/{purpose}",
        json={"key": key, "test_before_replace": test_before_replace},
        headers=_auth_headers(),
    )


def _delete(client, purpose, confirm=True):
    return client.request(
        "DELETE",
        f"/api/credentials/{purpose}",
        content=json.dumps({"confirm": confirm}),
        headers=_auth_headers(**{"Content-Type": "application/json"}),
    )


# ── status: where the key comes from ────────────────────────────────────────


def test_a_first_start_reports_not_configured_and_writable(_container_environment):
    body = _client().get("/api/ai-settings").json()["text"]

    assert body["configured"] is False
    assert body["source"] == "missing"
    assert body["writable"] is True, "the page must accept a first key"


def test_a_saved_key_is_reported_as_the_container_credential_file(_container_environment):
    client = _client()
    assert _put(client, "text", TEXT_KEY).status_code == 200

    body = client.get("/api/ai-settings").json()["text"]

    assert body["configured"] is True
    assert body["source"] == "file_credential_store"
    assert body["writable"] is True


def test_the_settings_response_never_contains_the_key(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)

    assert TEXT_KEY not in client.get("/api/ai-settings").text


def test_the_environment_is_reported_as_read_only(_container_environment, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-key-cccccccc")

    body = _client().get("/api/ai-settings").json()["text"]

    assert body["source"] == "environment"
    assert body["configured"] is True
    assert body["writable"] is False
    assert "env-key-cccccccc" not in json.dumps(body)


def test_a_corrupt_store_is_reported_as_not_writable(_container_environment):
    _container_environment.parent.mkdir(parents=True, exist_ok=True)
    _container_environment.write_text("{ not json", encoding="utf-8")

    body = _client().get("/api/ai-settings").json()["text"]

    assert body["configured"] is False
    assert body["writable"] is False, "nothing may be offered that could clobber the file"


# ── save / replace / delete through the page ────────────────────────────────


def test_a_saved_key_is_immediately_usable(_container_environment):
    client = _client()

    assert _put(client, "text", TEXT_KEY).status_code == 200

    assert credentials.resolve_credential("text") == TEXT_KEY
    assert json.loads(_container_environment.read_text(encoding="utf-8")) == {"text": TEXT_KEY}


def test_saving_one_purpose_keeps_the_other(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)

    assert _put(client, "vision", VISION_KEY).status_code == 200

    stored = json.loads(_container_environment.read_text(encoding="utf-8"))
    assert stored == {"text": TEXT_KEY, "vision": VISION_KEY}
    assert credentials.resolve_credential("text") == TEXT_KEY


def test_replacing_one_purpose_keeps_the_other(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)
    _put(client, "vision", VISION_KEY)

    assert _put(client, "text", "text-key-cccccccc").status_code == 200

    stored = json.loads(_container_environment.read_text(encoding="utf-8"))
    assert stored == {"text": "text-key-cccccccc", "vision": VISION_KEY}


def test_deleting_one_purpose_keeps_the_other(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)
    _put(client, "vision", VISION_KEY)

    assert _delete(client, "text").status_code == 200

    assert json.loads(_container_environment.read_text(encoding="utf-8")) == {
        "vision": VISION_KEY
    }
    body = client.get("/api/ai-settings").json()
    assert body["text"]["configured"] is False
    assert body["vision"]["configured"] is True


def test_deleting_the_last_purpose_leaves_no_file(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)

    assert _delete(client, "text").status_code == 200

    assert not _container_environment.exists()
    assert credentials.resolve_credential("text") is None


def test_an_environment_key_can_not_be_saved_over_or_deleted(
    _container_environment, monkeypatch
):
    monkeypatch.setenv("AUTO_TUNE_TEXT_API_KEY", "env-key-cccccccc")
    client = _client()

    assert _put(client, "text", TEXT_KEY).status_code == 409
    assert _delete(client, "text").status_code == 409
    assert not _container_environment.exists()


def test_an_invalid_key_is_rejected_before_anything_is_written(_container_environment):
    client = _client()
    _put(client, "text", TEXT_KEY)

    resp = _put(client, "vision", "YOUR_QWEN_API_KEY")

    assert resp.status_code == 400
    assert json.loads(_container_environment.read_text(encoding="utf-8")) == {"text": TEXT_KEY}


def test_a_corrupt_store_refuses_a_save_without_leaking(_container_environment):
    _container_environment.parent.mkdir(parents=True, exist_ok=True)
    _container_environment.write_text("{ not json", encoding="utf-8")
    client = _client()

    resp = _put(client, "text", TEXT_KEY)

    assert resp.status_code == 500
    assert TEXT_KEY not in resp.text
    assert str(_container_environment) not in resp.text
    assert _container_environment.read_text(encoding="utf-8") == "{ not json"


# ── the connection test ─────────────────────────────────────────────────────


def test_testing_without_a_key_says_no_key_was_saved(_container_environment):
    """The hint must not suggest a provider request was already made."""
    resp = _client().post("/api/credentials/text/test", json={}, headers=_auth_headers())

    assert resp.status_code == 400
    assert resp.json()["reason"] == "credential_missing"
    assert "no API key saved yet" in resp.json()["error"]


def test_a_failed_test_before_saving_keeps_the_previous_key(
    _container_environment, monkeypatch
):
    client = _client()
    _put(client, "text", TEXT_KEY)
    monkeypatch.setattr(
        app_mod,
        "_probe_connection",
        lambda purpose, api_key_override=None: "authentication_failed",
    )

    resp = _put(client, "text", "text-key-cccccccc", test_before_replace=True)

    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "credential_test_failed", (
        "a connection test failure must be distinguishable from a write failure")
    assert body["category"] == "authentication_failed"
    assert "text-key-cccccccc" not in resp.text
    assert json.loads(_container_environment.read_text(encoding="utf-8")) == {"text": TEXT_KEY}


def test_a_successful_test_before_saving_replaces_the_key(_container_environment, monkeypatch):
    monkeypatch.setattr(
        app_mod, "_probe_connection", lambda purpose, api_key_override=None: "success"
    )
    client = _client()

    resp = _put(client, "text", TEXT_KEY, test_before_replace=True)

    assert resp.status_code == 200
    assert credentials.resolve_credential("text") == TEXT_KEY


# ── the page contract ───────────────────────────────────────────────────────


def _render_page() -> str:
    from auto_tune.modules.presentation import build_experiment_labels
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return app_mod._jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        experiment_labels=build_experiment_labels(translator),
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
        ai_config={},
        csrf_token="",
    )


def test_the_page_names_the_container_credential_file():
    page = _render_page()

    assert "file_credential_store:" in page, "the source must have a label"
    assert "容器持久凭据文件" in page


def test_the_page_distinguishes_a_connection_test_failure():
    page = _render_page()

    assert "credential_test_failed" in page
    assert "连接测试失败" in page


def test_the_page_explains_a_missing_key():
    page = _render_page()

    assert "credential_missing" in page
    assert "尚未保存 DeepSeek API Key" in page
    assert "尚未保存 Qwen API Key" in page
