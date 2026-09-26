"""F1.2-C: the shared delivery preflight used by the container start-up.

Every check here is a *delivery* concern — where the persistent directories
live, whether they are actually mounted, whether they can be written, whether
the controlled runtime is importable and whether the GPU the product requires
is visible. Nothing in this module may change training, HPO or tuning
semantics, and every check is driven through injected probes so these tests
never need a real container, a real GPU or the real product directories.
"""

from pathlib import Path

import pytest

from auto_tune.delivery import preflight
from auto_tune.delivery.preflight import (
    CONTAINER_APP_ROOT,
    CONTAINER_DATASETS_DIR,
    DELIVERY_CONFIG_MISSING,
    DELIVERY_DEPENDENCY_MISSING,
    DELIVERY_DIR_NOT_MOUNTED,
    DELIVERY_DIR_NOT_WRITABLE,
    DELIVERY_GPU_UNAVAILABLE,
    DELIVERY_TEMPLATE_MISSING,
    DeliveryError,
    bootstrap_config,
    check_dependencies,
    check_mounts,
    check_writable,
    ensure_directories,
    require_gpu,
    resolve_paths,
    run,
)

TEMPLATE = """project:
  name: template
llm:
  endpoint: https://example.invalid/v1/chat/completions
"""


# ── controlled layout ───────────────────────────────────────────────────────


def test_resolve_paths_uses_the_container_delivery_layout():
    paths = resolve_paths(app_root=Path("/opt/auto-tune"),
                          config_path=Path("/data/config/config.yaml"),
                          datasets_dir=Path("/data/datasets"),
                          template_path=Path("/opt/auto-tune/auto_tune/config.template.yaml"))

    assert paths.app_root == Path("/opt/auto-tune")
    assert paths.config_path == Path("/data/config/config.yaml")
    assert paths.config_dir == Path("/data/config")
    assert paths.datasets_dir == Path("/data/datasets")
    assert paths.persistent_dirs == {
        "config": Path("/data/config"),
        "datasets": Path("/data/datasets"),
        "log": Path("/opt/auto-tune/log"),
        "detect": Path("/opt/auto-tune/detect"),
        "runs": Path("/opt/auto-tune/runs"),
        "models/weights": Path("/opt/auto-tune/models/weights"),
    }


def test_container_defaults_are_the_documented_mount_points():
    assert (CONTAINER_APP_ROOT, CONTAINER_DATASETS_DIR) == (
        Path("/opt/auto-tune"), Path("/data/datasets"))


def test_resolve_paths_reads_only_the_controlled_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_TUNE_APP_ROOT", str(tmp_path / "app"))
    monkeypatch.setenv("AUTO_TUNE_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(tmp_path / "config" / "config.yaml"))

    paths = resolve_paths()

    assert paths.app_root == (tmp_path / "app").resolve()
    assert paths.datasets_dir == (tmp_path / "datasets").resolve()
    assert paths.config_path == (tmp_path / "config" / "config.yaml").resolve()
    assert paths.persistent_dirs["log"] == (tmp_path / "app" / "log").resolve()


def test_resolve_paths_keeps_the_packaged_configuration_when_nothing_is_set(monkeypatch):
    for name in ("AUTO_TUNE_APP_ROOT", "AUTO_TUNE_DATASETS_DIR",
                 "AUTO_TUNE_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)

    paths = resolve_paths()

    from auto_tune.delivery.runtime import PACKAGE_CONFIG_PATH

    assert paths.config_path == PACKAGE_CONFIG_PATH
    assert paths.app_root == PACKAGE_CONFIG_PATH.parent
    assert paths.persistent_dirs["log"] == paths.app_root / "log"


# ── directory initialisation and write permission ───────────────────────────


def _layout(tmp_path):
    return resolve_paths(app_root=tmp_path / "app",
                         config_path=tmp_path / "data" / "config" / "config.yaml",
                         datasets_dir=tmp_path / "data" / "datasets",
                         template_path=tmp_path / "config.template.yaml")


def test_ensure_directories_creates_the_missing_delivery_layout(tmp_path):
    paths = _layout(tmp_path)

    created = ensure_directories(paths)

    assert set(created) == set(paths.persistent_dirs.values())
    for directory in paths.persistent_dirs.values():
        assert directory.is_dir()


def test_ensure_directories_is_idempotent(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    assert ensure_directories(paths) == []


def test_ensure_directories_refuses_a_path_occupied_by_a_file(tmp_path):
    paths = _layout(tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "log").write_text("not a directory", encoding="utf-8")

    with pytest.raises(DeliveryError) as excinfo:
        ensure_directories(paths)

    assert excinfo.value.code == DELIVERY_DIR_NOT_WRITABLE


def test_check_writable_reports_a_directory_the_runtime_cannot_write(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    def deny(path, mode):
        return not str(path).endswith("detect")

    with pytest.raises(DeliveryError) as excinfo:
        check_writable(paths, probe=deny)

    assert excinfo.value.code == DELIVERY_DIR_NOT_WRITABLE
    assert "detect" in str(excinfo.value)


def test_check_writable_passes_for_a_prepared_layout(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    assert check_writable(paths) is None


# ── mount contract: never write into the image layer ────────────────────────


def test_check_mounts_accepts_a_layout_where_every_directory_is_mounted(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    assert check_mounts(paths, probe=lambda path: True) is None


def test_check_mounts_rejects_a_persistent_directory_that_is_not_mounted(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    def probe(path):
        return not str(path).endswith("runs")

    with pytest.raises(DeliveryError) as excinfo:
        check_mounts(paths, probe=probe)

    assert excinfo.value.code == DELIVERY_DIR_NOT_MOUNTED
    assert "runs" in str(excinfo.value)


def test_check_mounts_names_every_unmounted_directory(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)

    with pytest.raises(DeliveryError) as excinfo:
        check_mounts(paths, probe=lambda path: False)

    message = str(excinfo.value)
    for name in ("config", "datasets", "log", "detect", "runs", "models"):
        assert name in message


def test_check_mounts_ships_with_a_real_mount_probe(tmp_path):
    """The shipping default probes the real mount table, not a stub."""
    paths = _layout(tmp_path)
    ensure_directories(paths)

    assert preflight.mount_probe is preflight.os.path.ismount


# ── configuration bootstrap ─────────────────────────────────────────────────


def test_missing_configuration_is_reported_before_startup(tmp_path):
    paths = _layout(tmp_path)
    paths.template_path.write_text(TEMPLATE, encoding="utf-8")

    with pytest.raises(DeliveryError) as excinfo:
        bootstrap_config(paths, create=False)

    assert excinfo.value.code == DELIVERY_CONFIG_MISSING


def test_bootstrap_creates_the_configuration_from_the_sanitized_template(tmp_path):
    paths = _layout(tmp_path)
    ensure_directories(paths)
    paths.template_path.write_text(TEMPLATE, encoding="utf-8")

    assert bootstrap_config(paths, create=True) is True
    assert paths.config_path.read_text(encoding="utf-8") == TEMPLATE


def test_bootstrap_never_overwrites_an_existing_configuration(tmp_path):
    paths = _layout(tmp_path)
    paths.config_dir.mkdir(parents=True)
    paths.config_path.write_text("project:\n  name: operator\n", encoding="utf-8")
    paths.template_path.write_text(TEMPLATE, encoding="utf-8")

    assert bootstrap_config(paths, create=True) is False
    assert paths.config_path.read_text(encoding="utf-8") == "project:\n  name: operator\n"


def test_bootstrap_reports_a_missing_template(tmp_path):
    paths = _layout(tmp_path)

    with pytest.raises(DeliveryError) as excinfo:
        bootstrap_config(paths, create=True)

    assert excinfo.value.code == DELIVERY_TEMPLATE_MISSING


def test_bootstrap_does_not_create_the_configuration_when_it_is_not_requested(tmp_path):
    paths = _layout(tmp_path)
    paths.template_path.write_text(TEMPLATE, encoding="utf-8")

    with pytest.raises(DeliveryError):
        bootstrap_config(paths, create=False)

    assert not paths.config_path.exists()


# ── runtime dependencies ────────────────────────────────────────────────────


def test_check_dependencies_names_the_component_that_is_missing():
    def finder(name):
        return None if name == "onnxruntime" else object()

    with pytest.raises(DeliveryError) as excinfo:
        check_dependencies(finder=finder)

    assert excinfo.value.code == DELIVERY_DEPENDENCY_MISSING
    assert "onnxruntime" in str(excinfo.value)


def test_check_dependencies_passes_when_every_component_is_available():
    assert check_dependencies(finder=lambda name: object()) is None


def test_check_dependencies_covers_training_and_onnx_export_components():
    assert set(preflight.REQUIRED_MODULES.values()) == {
        "torch", "ultralytics", "onnx", "onnxruntime"}


def test_check_dependencies_passes_in_the_accepted_conda_environment():
    assert check_dependencies() is None


def test_dependency_error_message_never_leaks_a_filesystem_path():
    with pytest.raises(DeliveryError) as excinfo:
        check_dependencies(finder=lambda name: None)

    message = str(excinfo.value)
    for leaked in ("\\", "/", "site-packages", ".exe", "C:", "D:"):
        assert leaked not in message


# ── GPU requirement ─────────────────────────────────────────────────────────


def test_require_gpu_rejects_a_runtime_without_a_visible_cuda_device():
    with pytest.raises(DeliveryError) as excinfo:
        require_gpu(probe=lambda: False)

    assert excinfo.value.code == DELIVERY_GPU_UNAVAILABLE


def test_gpu_error_states_that_there_is_no_cpu_fallback():
    with pytest.raises(DeliveryError) as excinfo:
        require_gpu(probe=lambda: False)

    message = str(excinfo.value)
    assert "GPU" in message
    assert "CPU" in message


def test_require_gpu_passes_when_the_delivery_gpu_is_visible():
    assert require_gpu(probe=lambda: True) is None


# ── command line contract used by the container entrypoint ──────────────────


def _container_layout(tmp_path, monkeypatch, *, mounted: bool = True,
                      gpu: bool = True, config: bool = False):
    """Prepare a delivery layout shaped exactly like the compose mounts."""
    app_root = tmp_path / "app"
    config_path = tmp_path / "data" / "config" / "config.yaml"
    datasets = tmp_path / "data" / "datasets"
    template = app_root / "auto_tune" / "config.template.yaml"
    monkeypatch.setenv("AUTO_TUNE_APP_ROOT", str(app_root))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AUTO_TUNE_DATASETS_DIR", str(datasets))
    monkeypatch.setenv("AUTO_TUNE_TEMPLATE_PATH", str(template))
    monkeypatch.setattr(preflight, "mount_probe", lambda path: mounted)
    monkeypatch.setattr(preflight, "gpu_available", lambda: gpu)

    paths = resolve_paths()
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(TEMPLATE, encoding="utf-8")
    if config:
        paths.app_root.mkdir(parents=True, exist_ok=True)
        (paths.app_root / "log").mkdir(parents=True, exist_ok=True)
        ensure_directories(paths)
        config_path.write_text(TEMPLATE, encoding="utf-8")
    return paths


def test_run_initialises_the_delivery_layout_and_exits_zero(tmp_path, monkeypatch, capsys):
    _container_layout(tmp_path, monkeypatch, config=False)

    code = run(["--require-mounts", "--bootstrap-config"])

    assert code == 0
    captured = capsys.readouterr()
    assert "[delivery]" in captured.out
    assert "Traceback" not in captured.err


def test_run_refuses_an_unmounted_delivery_directory_with_a_stable_code(tmp_path, monkeypatch, capsys):
    _container_layout(tmp_path, monkeypatch, mounted=False, config=True)

    code = run(["--require-mounts"])

    assert code == 1
    assert DELIVERY_DIR_NOT_MOUNTED in capsys.readouterr().err


def test_run_refuses_to_create_a_configuration_into_the_image_layer(tmp_path, monkeypatch, capsys):
    """An unmounted config directory would silently swallow the operator's
    configuration on the next container recreation, so nothing may be written."""
    paths = _container_layout(tmp_path, monkeypatch, mounted=False, config=False)

    code = run(["--require-mounts", "--bootstrap-config"])

    assert code == 1
    assert DELIVERY_DIR_NOT_MOUNTED in capsys.readouterr().err
    assert not paths.config_path.exists()


def test_run_refuses_a_runtime_without_the_required_gpu(tmp_path, monkeypatch, capsys):
    _container_layout(tmp_path, monkeypatch, gpu=False, config=True)

    code = run(["--require-mounts", "--require-gpu"])

    assert code == 1
    assert DELIVERY_GPU_UNAVAILABLE in capsys.readouterr().err


def test_run_refuses_a_runtime_missing_a_controlled_dependency(tmp_path, monkeypatch, capsys):
    _container_layout(tmp_path, monkeypatch, config=True)

    def fail(*args, **kwargs):
        raise DeliveryError(DELIVERY_DEPENDENCY_MISSING, "缺少运行组件：onnx。")

    monkeypatch.setattr(preflight, "check_dependencies", fail)

    code = run(["--require-mounts"])

    assert code == 1
    assert DELIVERY_DEPENDENCY_MISSING in capsys.readouterr().err


def test_run_reports_a_missing_configuration_without_a_traceback(tmp_path, monkeypatch, capsys):
    _container_layout(tmp_path, monkeypatch, config=False)

    code = run(["--require-mounts"])

    assert code == 1
    err = capsys.readouterr().err
    assert DELIVERY_CONFIG_MISSING in err
    assert "Traceback" not in err


def test_run_never_prints_configuration_or_credential_contents(tmp_path, monkeypatch, capsys):
    paths = _container_layout(tmp_path, monkeypatch, config=True)
    paths.config_path.write_text(
        "llm:\n  api_key: super-secret-token-value\n", encoding="utf-8")

    run(["--require-mounts"])

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "super-secret-token-value" not in combined
    assert "api_key" not in combined.lower()
