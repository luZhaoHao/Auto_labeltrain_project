"""F1.2-B: delivery runtime boundary (startup bind + config path) and `/healthz`.

The delivery layer only resolves where the server listens and which
configuration file the product reads. It must not change any training,
HPO, tuning, snapshot or persistence semantics, and the desktop defaults
must stay `127.0.0.1:8000`.
"""

from pathlib import Path

import pytest

from auto_tune.delivery.runtime import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    resolve_config_path,
    resolve_server_bind,
)


# ── Task 1: controlled startup address / port ───────────────────────────────


def test_resolve_server_bind_keeps_desktop_defaults(monkeypatch):
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    assert resolve_server_bind() == ("127.0.0.1", 8000)
    assert (DEFAULT_HOST, DEFAULT_PORT) == ("127.0.0.1", 8000)


def test_resolve_server_bind_accepts_container_values(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_HOST", "0.0.0.0")
    monkeypatch.setenv("AUTO_TUNE_PORT", "18000")
    assert resolve_server_bind() == ("0.0.0.0", 18000)


def test_resolve_server_bind_honours_explicit_defaults(monkeypatch):
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    assert resolve_server_bind("0.0.0.0", 9000) == ("0.0.0.0", 9000)


def test_resolve_server_bind_environment_overrides_explicit_defaults(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_HOST", "0.0.0.0")
    monkeypatch.setenv("AUTO_TUNE_PORT", "18000")
    assert resolve_server_bind("127.0.0.1", 8000) == ("0.0.0.0", 18000)


@pytest.mark.parametrize("host", ["", "   ", "\t"])
def test_resolve_server_bind_rejects_empty_host(monkeypatch, host):
    monkeypatch.setenv("AUTO_TUNE_HOST", host)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    with pytest.raises(ValueError):
        resolve_server_bind()


def test_resolve_server_bind_strips_surrounding_host_whitespace(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_HOST", "  0.0.0.0  ")
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    assert resolve_server_bind() == ("0.0.0.0", 8000)


@pytest.mark.parametrize("port", ["", "   ", "abc", "80.5", "0", "-1", "65536", "1e3", "8000 "])
def test_resolve_server_bind_rejects_invalid_port(monkeypatch, port):
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.setenv("AUTO_TUNE_PORT", port)
    with pytest.raises(ValueError):
        resolve_server_bind()


@pytest.mark.parametrize("port", ["1", "65535"])
def test_resolve_server_bind_accepts_port_range_edges(monkeypatch, port):
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.setenv("AUTO_TUNE_PORT", port)
    assert resolve_server_bind()[1] == int(port)


def test_resolve_server_bind_error_does_not_echo_the_raw_value(monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_PORT", "super-secret-token")
    with pytest.raises(ValueError) as excinfo:
        resolve_server_bind()
    assert "super-secret-token" not in str(excinfo.value)


# ── Task 1: controlled configuration path ───────────────────────────────────


def test_resolve_config_path_uses_explicit_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_TUNE_CONFIG_PATH", raising=False)
    default = tmp_path / "default.yaml"
    assert resolve_config_path(default) == default


def test_resolve_config_path_uses_controlled_environment_path(tmp_path, monkeypatch):
    target = tmp_path / "config.yaml"
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(target))
    assert resolve_config_path(tmp_path / "default.yaml") == target.resolve()


def test_resolve_config_path_returns_an_absolute_path_for_relative_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", "data/config/config.yaml")
    resolved = resolve_config_path(tmp_path / "unused.yaml")
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "data" / "config" / "config.yaml").resolve()


def test_resolve_config_path_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", "~/config/config.yaml")
    resolved = resolve_config_path(tmp_path / "unused.yaml")
    assert "~" not in str(resolved)
    assert resolved.is_absolute()


def test_resolve_config_path_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_DELIVERY_TEST_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "AUTO_TUNE_CONFIG_PATH",
        "${AUTO_TUNE_DELIVERY_TEST_ROOT}/config/config.yaml",
    )
    resolved = resolve_config_path(tmp_path / "unused.yaml")
    assert str(resolved).startswith(str(tmp_path))
    assert resolved.name == "config.yaml"


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_config_path_rejects_blank_environment_value(tmp_path, monkeypatch, value):
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", value)
    with pytest.raises(ValueError):
        resolve_config_path(tmp_path / "unused.yaml")


def test_package_default_config_path_is_the_repository_template_location():
    from auto_tune.delivery import runtime

    assert runtime.PACKAGE_CONFIG_PATH == Path(runtime.__file__).resolve().parent.parent / "config.yaml"


# ── Task 2: minimal operational probe ───────────────────────────────────────


def test_healthz_is_a_minimal_operational_probe():
    from fastapi.testclient import TestClient

    from auto_tune.ui.app import app

    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "product": "auto-tune-studio"}


def test_healthz_does_not_leak_paths_versions_or_environment():
    from fastapi.testclient import TestClient

    from auto_tune.ui.app import app

    response = TestClient(app).get("/healthz")
    body = response.text
    assert set(response.json().keys()) == {"status", "product"}
    for leaked in ("config.yaml", "D:", "C:", "/opt/", "3.10", "python", "venv"):
        assert leaked not in body


def test_healthz_is_not_linked_from_the_user_interface():
    from auto_tune.ui.app import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/healthz" in paths
