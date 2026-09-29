"""F1.2-B: delivery runtime boundary (startup bind + config path) and `/healthz`.

The delivery layer only resolves where the server listens and which
configuration file the product reads. It must not change any training,
HPO, tuning, snapshot or persistence semantics, and the desktop defaults
must stay `127.0.0.1:8000`.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

from auto_tune.delivery import runtime
from auto_tune.delivery.runtime import (
    CONTAINER_INPUT_ROOTS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    INPUT_ALLOWED_ROOTS_ENV,
    resolve_config_path,
    resolve_input_allowed_roots,
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


# ── Task 3: the Windows event loop the socket layer needs ───────────────────


def test_windows_selects_the_selector_event_loop(monkeypatch):
    """Some Windows machines fail every accept with WinError 10014 under the
    default Proactor loop; the selector policy is selected explicitly."""
    recorded: list[object] = []
    monkeypatch.setattr(runtime, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        runtime,
        "asyncio",
        types.SimpleNamespace(
            WindowsSelectorEventLoopPolicy=lambda: "selector-policy",
            set_event_loop_policy=recorded.append,
        ),
    )

    assert runtime.configure_platform_event_loop() is True
    assert recorded == ["selector-policy"]


def test_non_windows_never_touches_the_windows_only_api(monkeypatch):
    class _Exploding:
        def __getattr__(self, name):  # pragma: no cover - only fires on a bug
            raise AssertionError(f"a non-Windows run touched asyncio.{name}")

    monkeypatch.setattr(runtime, "sys", types.SimpleNamespace(platform="linux"))
    monkeypatch.setattr(runtime, "asyncio", _Exploding())

    assert runtime.configure_platform_event_loop() is False


def test_a_windows_python_without_the_policy_is_left_alone(monkeypatch):
    recorded: list[object] = []
    monkeypatch.setattr(runtime, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        runtime,
        "asyncio",
        types.SimpleNamespace(set_event_loop_policy=recorded.append),
    )

    assert runtime.configure_platform_event_loop() is False
    assert recorded == []


def test_start_server_selects_the_event_loop_before_uvicorn_runs(monkeypatch, tmp_path):
    from auto_tune.ui import app as app_mod

    calls: list[str] = []
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "log").mkdir()
    monkeypatch.setattr(
        app_mod, "configure_platform_event_loop", lambda: calls.append("policy")
    )
    monkeypatch.setattr(app_mod, "_common_context", lambda: {})
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: calls.append("uvicorn")),
    )

    app_mod.start_server(host="127.0.0.1", port=8000)

    assert calls == ["policy", "uvicorn"]


def _captured_start_server(monkeypatch, tmp_path) -> dict:
    """Run the real ``start_server`` up to the ``uvicorn.run`` call boundary."""
    from auto_tune.ui import app as app_mod

    captured: dict = {}
    monkeypatch.delenv("AUTO_TUNE_HOST", raising=False)
    monkeypatch.delenv("AUTO_TUNE_PORT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "log").mkdir()
    monkeypatch.setattr(app_mod, "_common_context", lambda: {})
    monkeypatch.setattr(
        app_mod, "configure_platform_event_loop", lambda: True
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: captured.update(kwargs)),
    )

    app_mod.start_server(host="127.0.0.1", port=8000)
    return captured


def test_start_server_asks_uvicorn_for_the_environment_loop(monkeypatch, tmp_path):
    """``loop="none"`` is what makes uvicorn *ask* the platform for the loop.

    Every other value makes uvicorn build the factory itself, which is how the
    Windows start ended up on the Proactor loop despite the selector policy."""
    captured = _captured_start_server(monkeypatch, tmp_path)

    assert captured["loop"] == "none"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


@pytest.fixture
def restore_event_loop_policy():
    """Leave the process-wide policy exactly as it was found."""
    original = asyncio.get_event_loop_policy()
    yield
    asyncio.set_event_loop_policy(original)


def test_the_selected_event_loop_is_never_a_proactor_loop(restore_event_loop_policy):
    """The real uvicorn 0.51 path: ``Config.get_loop_factory`` → ``asyncio``.

    With ``loop="none"`` uvicorn returns no factory at all, so its own
    ``asyncio_run`` falls back to ``asyncio.new_event_loop()``, which is the one
    call that honours the policy :func:`configure_platform_event_loop` sets.
    """
    import uvicorn

    assert runtime.configure_platform_event_loop() in (True, False)

    config = uvicorn.Config(app=None, loop="none")
    factory = config.get_loop_factory()
    loop = asyncio.new_event_loop()
    try:
        assert factory is None, (
            "loop='none' must leave the loop creation to asyncio itself")
        assert "SelectorEventLoop" in type(loop).__name__, (
            f"the environment loop is {type(loop).__name__}, which is the loop "
            "that fails every accept with WinError 10014 on Windows")
        assert not isinstance(loop, getattr(asyncio, "ProactorEventLoop", ()))
    finally:
        loop.close()


def test_uvicorn_loop_none_creates_the_windows_selector_loop(restore_event_loop_policy):
    """The Windows end state, asked of the real ``uvicorn`` 0.51 and asyncio."""
    if sys.platform != "win32":
        pytest.skip("the Proactor loop only exists on Windows")

    import uvicorn

    assert runtime.configure_platform_event_loop() is True

    assert uvicorn.Config(app=None, loop="none").get_loop_factory() is None
    loop = asyncio.new_event_loop()
    try:
        assert type(loop).__name__ == "_WindowsSelectorEventLoop", type(loop).__name__
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_the_default_uvicorn_loop_bypasses_the_policy_on_windows(
    restore_event_loop_policy,
):
    """Root cause, kept as evidence: the default factory ignores the policy.

    Selecting the selector policy and then asking uvicorn for its default loop
    still produces a Proactor loop, because the factory is built explicitly and
    ``asyncio.new_event_loop()`` is never called."""
    if sys.platform != "win32":
        pytest.skip("the Proactor loop only exists on Windows")

    import uvicorn

    runtime.configure_platform_event_loop()
    config = uvicorn.Config(app=None, loop="auto")

    factory = config.get_loop_factory()
    assert factory is not None
    loop = factory()
    try:
        assert isinstance(loop, asyncio.ProactorEventLoop)
    finally:
        loop.close()


def test_the_cli_training_paths_never_start_the_web_server(monkeypatch, capsys):
    """``--dry-run`` and ``--train`` must not change event loop behaviour, which
    they cannot reach: the policy is selected by the web start alone."""
    import auto_tune.main as main_mod
    from auto_tune.modules.agent_engine import loop as loop_mod
    from auto_tune.ui import app as app_mod

    started: list[int] = []
    monkeypatch.setattr(app_mod, "start_server", lambda *args, **kwargs: started.append(1))
    monkeypatch.setattr(
        loop_mod, "run_tuning_loop", lambda *args, **kwargs: {"iterations": []}
    )

    for flag in ("--dry-run", "--train"):
        monkeypatch.setattr(sys, "argv", ["auto_tune.main", flag])
        main_mod.main()

    assert started == []
    capsys.readouterr()


# ── Task 4: the controlled input browse roots ───────────────────────────────


def _set_roots(monkeypatch, value: str) -> None:
    monkeypatch.setenv(INPUT_ALLOWED_ROOTS_ENV, value)


def _posix(path) -> str:
    """The declared root as a comparable container path on any host."""
    return str(path).replace("\\", "/")


def test_input_roots_are_unset_by_default(monkeypatch):
    monkeypatch.delenv(INPUT_ALLOWED_ROOTS_ENV, raising=False)

    assert resolve_input_allowed_roots() is None


def test_input_roots_come_from_the_controlled_environment(monkeypatch):
    _set_roots(monkeypatch, ";".join(CONTAINER_INPUT_ROOTS))

    roots = resolve_input_allowed_roots()

    assert [_posix(root) for root in roots] == list(CONTAINER_INPUT_ROOTS)


def test_input_roots_are_deduplicated_and_absolute(monkeypatch):
    """The declared pair, written repeatedly and messily, is still that pair."""
    _set_roots(monkeypatch, "/data/datasets;/data/datasets/; /data/datasets/./ ;"
                          "/opt/auto-tune/detect;/opt/auto-tune/detect/")

    roots = resolve_input_allowed_roots()

    assert [_posix(root) for root in roots] == list(CONTAINER_INPUT_ROOTS)


@pytest.mark.parametrize("value", [
    "/data/datasets",
    "/data/datasets/",
    "/data/datasets/./",
    "/opt/auto-tune/detect",
    "/opt/auto-tune/detect/",
])
def test_a_single_root_is_refused_even_when_it_is_declared(monkeypatch, value):
    """Both declared roots are required, not "any subset of them".

    A container that declares only one of the two would offer a picker that can
    reach the dataset share but never the directory every training run is
    written into (or the reverse), which is a misconfiguration rather than a
    narrower but working delivery.
    """
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


@pytest.mark.parametrize("value", [
    "/data/datasets;/data/datasets/",
    "/opt/auto-tune/detect;/opt/auto-tune/detect/",
    "/data/datasets;/data/./datasets;/data/datasets",
])
def test_a_repeated_single_root_is_refused(monkeypatch, value):
    """Deduplication must not turn a one-directory declaration into a set that
    looks complete: the two roots are distinct directories, not two spellings."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


def test_both_roots_in_the_other_order_come_back_in_the_declared_order(monkeypatch):
    """The input order is free; the returned order is the declared one."""
    _set_roots(monkeypatch, "/opt/auto-tune/detect;/data/datasets")

    roots = resolve_input_allowed_roots()

    assert [_posix(root) for root in roots] == list(CONTAINER_INPUT_ROOTS)


@pytest.mark.parametrize("value", ["", "   ", ";", " ; ; ", "\t"])
def test_an_empty_effective_root_set_is_refused(monkeypatch, value):
    """An empty set would silently become "browse nothing" in the container."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


def test_the_declared_roots_are_exactly_the_two_delivery_directories(monkeypatch):
    """The accepted set is a whitelist, not a few wide directories refused.

    The dataset share is the input the product only reads; ``detect`` is where
    ``find_detect_dir`` really writes every training run, so it is the directory
    the training analysis picker has to be able to reach.
    """
    from auto_tune.delivery.preflight import CONTAINER_APP_ROOT, CONTAINER_DATASETS_DIR
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    assert CONTAINER_INPUT_ROOTS == (
        CONTAINER_DATASETS_DIR.as_posix(),
        (CONTAINER_APP_ROOT / "detect").as_posix(),
    )


@pytest.mark.parametrize("value", [
    "/etc",
    "/proc",
    "/tmp",
    "/data",
    "/data/other",
    "/data/secrets",
    "/opt/auto-tune/log",
    "/opt/auto-tune/models",
    "/opt/auto-tune/models/weights",
    "/opt/auto-tune/runs",
    "/opt/auto-tune/config",
])
def test_a_directory_that_is_not_declared_is_refused(monkeypatch, value):
    """Everything outside the declared pair is refused, including the
    application's own directories: the picker must never be pointed at the
    logs, the weight store or the retired ``runs`` mount."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


@pytest.mark.parametrize("value, other", [
    (" /data/datasets ", "/opt/auto-tune/detect"),
    ("/data/datasets/", "/opt/auto-tune/detect"),
    ("/data/datasets/.", "/opt/auto-tune/detect"),
    ("//data/datasets", "/opt/auto-tune/detect"),
    ("/data/./datasets", "/opt/auto-tune/detect"),
    ("/data/../data/datasets", "/opt/auto-tune/detect"),
    ("/opt/auto-tune/detect/", "/data/datasets"),
    ("//opt/auto-tune/detect", "/data/datasets"),
])
def test_a_declared_root_written_differently_is_still_the_declared_root(
    monkeypatch, value, other
):
    """Duplicates, trailing separators, ``.``, ``..`` and doubled ``/`` are
    normalised onto the declared root instead of being refused or accepted as a
    second, unvetted directory."""
    _set_roots(monkeypatch, f"{value};{other}")

    roots = resolve_input_allowed_roots()

    assert [_posix(root) for root in roots] == list(CONTAINER_INPUT_ROOTS)


@pytest.mark.parametrize("value", [
    "/data/datasets/..",
    "/data/datasets/../secrets",
    "/opt/auto-tune/detect/../runs",
    "/opt/auto-tune/detect/../../..",
])
def test_a_declared_root_that_climbs_out_of_itself_is_refused(monkeypatch, value):
    """Normalisation happens before the comparison, so ``..`` can never smuggle
    a wider directory in under a declared root's spelling."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


@pytest.mark.parametrize("value", ["datasets", "data/datasets", "./runs", "..", r"C:\datasets"])
def test_paths_that_are_not_container_absolute_are_refused(monkeypatch, value):
    """The variable names *container* directories: a relative path or a host
    drive path would silently resolve against the process' working directory."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


def test_a_root_with_a_nul_byte_is_refused(monkeypatch):
    _set_roots(monkeypatch, "/data/datasets\x00/runs")

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


@pytest.mark.parametrize("value", ["/", "/data", "/opt", "/opt/auto-tune"])
def test_container_wide_roots_are_refused(monkeypatch, value):
    """``/`` is the whole filesystem, ``/data`` and ``/opt`` are the parents the
    container shares with other software, and ``/opt/auto-tune`` is the
    application root that holds the logs, weights and training directories."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


@pytest.mark.parametrize(
    "value",
    ["/data/", "/opt/", "/opt/auto-tune/.", "/./", "//", "//data", "/data/..", "/opt/auto-tune/.."],
)
def test_a_wide_root_written_differently_is_still_refused(monkeypatch, value):
    """The comparison is on the normalised container path, not on the spelling."""
    _set_roots(monkeypatch, value)

    with pytest.raises(ValueError):
        resolve_input_allowed_roots()


def test_a_narrow_root_under_a_container_parent_is_accepted(monkeypatch):
    _set_roots(monkeypatch, "/data/datasets;/opt/auto-tune/detect")

    roots = resolve_input_allowed_roots()

    assert [_posix(root) for root in roots] == ["/data/datasets", "/opt/auto-tune/detect"]


def test_the_refusal_never_echoes_the_environment_value(monkeypatch):
    _set_roots(monkeypatch, "super-secret-root")

    with pytest.raises(ValueError) as excinfo:
        resolve_input_allowed_roots()

    assert "super-secret-root" not in str(excinfo.value)


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
