"""F1.2-D 返修：the private runtime proves it is the locked CUDA runtime.

The Windows installer builds its Python runtime from the package's offline
bundle and then asks that interpreter to answer for itself: exact interpreter
path, Python 3.10, the pinned CUDA PyTorch and Torchvision, CUDA 12.1, a visible
GPU and the delivery's importable components. A CPU wheel or a different CUDA
build must be refused here — this is the check that makes "no CPU fallback" a
verified fact rather than a promise.

The checks live in the shared delivery preflight (the container uses the same
module), and the machine answers are injectable, so nothing here needs a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_tune.delivery import preflight

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "windows" / "package-manifest.json"


def _details(**overrides) -> dict:
    """A runtime that satisfies every lock, with named facts overridden."""
    facts = {
        "executable": r"E:\AutoTuneStudio\runtime\py310\python.exe",
        "python_version": "3.10.16",
        "torch_version": "2.5.1+cu121",
        "torchvision_version": "0.20.1+cu121",
        "cuda_version": "12.1",
        "cuda_available": True,
        "missing_modules": [],
    }
    facts.update(overrides)
    return facts


def _check(details) -> None:
    preflight.check_offline_runtime(
        expect_python=details["executable"] if details["executable"] else None,
        details=lambda: details)


def _reason(details) -> str:
    with pytest.raises(preflight.DeliveryError) as failure:
        _check(details)
    assert failure.value.code == preflight.DELIVERY_RUNTIME_MISMATCH
    return failure.value.message


def test_the_locked_runtime_is_accepted():
    preflight.check_offline_runtime(expect_python=_details()["executable"], details=_details)


def test_the_locked_versions_are_the_ones_the_package_ships():
    """One runtime, one lock: the checked versions are the pinned wheel versions."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    versions = {wheel["purpose"]: wheel["version"]
                for wheel in manifest["runtime"]["torch"]["wheels"]}

    assert preflight.TORCH_VERSION == versions["cuda-torch"]
    assert preflight.TORCHVISION_VERSION == versions["cuda-torchvision"]
    assert manifest["python_version"] == "3.10"
    assert preflight.PYTHON_VERSION == manifest["python_version"]


def test_a_cpu_torch_build_is_refused():
    message = _reason(_details(torch_version="2.5.1"))

    assert "torch" in message
    assert "2.5.1+cu121" in message


def test_a_torchvision_build_from_another_index_is_refused():
    message = _reason(_details(torchvision_version="0.20.1"))

    assert "torchvision" in message


def test_a_torch_built_for_another_cuda_is_refused():
    message = _reason(_details(cuda_version="12.4"))

    assert "12.1" in message


def test_a_runtime_that_cannot_see_a_gpu_is_refused():
    message = _reason(_details(cuda_available=False))

    assert message


def test_a_runtime_without_a_cuda_build_is_refused():
    message = _reason(_details(cuda_version="", torch_version="2.5.1"))

    assert message


def test_another_python_version_is_refused():
    message = _reason(_details(python_version="3.11.9"))

    assert "3.10" in message


def test_another_interpreter_is_refused():
    """The runtime must be the private interpreter the installation built."""
    with pytest.raises(preflight.DeliveryError) as failure:
        preflight.check_offline_runtime(expect_python=r"E:\AutoTuneStudio\runtime\py310\python.exe",
                                        details=lambda: _details(
                                            executable=r"D:\Program Files\anaconda3\envs\auto_tune\python.exe"))
    assert failure.value.code == preflight.DELIVERY_RUNTIME_MISMATCH
    assert "anaconda3" not in failure.value.message, "no foreign path is repeated"


def test_a_missing_component_is_refused():
    message = _reason(_details(missing_modules=["optuna", "onnxruntime"]))

    assert "optuna" in message
    assert "onnxruntime" in message


def test_the_required_components_cover_the_product_capabilities():
    modules = set(preflight.OFFLINE_MODULES)

    assert {"fastapi", "ultralytics", "optuna", "onnx", "onnxruntime"} <= modules


def test_no_absolute_path_is_ever_reported():
    """The message travels into the operator's log; it stays path-free."""
    for details in (_details(executable=""), _details(torch_version="2.5.1"),
                    _details(python_version="3.11.9"), _details(cuda_available=False)):
        message = _reason(details)
        assert ":\\" not in message
        assert "/" not in message


# ── the command line the installer runs ─────────────────────────────────────


def _layout(tmp_path, monkeypatch):
    """A delivery layout in a temporary tree, exactly like the container one."""
    app_root = tmp_path / "app"
    config_path = tmp_path / "data" / "config" / "config.yaml"
    template = app_root / "auto_tune" / "config.template.yaml"
    monkeypatch.setenv("AUTO_TUNE_APP_ROOT", str(app_root))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AUTO_TUNE_DATASETS_DIR", str(tmp_path / "data" / "datasets"))
    monkeypatch.setenv("AUTO_TUNE_TEMPLATE_PATH", str(template))
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("project:\n  name: x\n", encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project:\n  name: x\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "check_dependencies", lambda finder=None: None)


def test_the_cli_requires_a_matching_runtime(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details", lambda: _details())

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 0


def test_the_cli_fails_on_a_foreign_runtime(tmp_path, monkeypatch, capsys):
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: _details(torch_version="2.5.1"))

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 1
    printed = capsys.readouterr()
    assert preflight.DELIVERY_RUNTIME_MISMATCH in printed.err
    assert "E:\\AutoTuneStudio" not in printed.err, "the log stays path-free"


def test_the_offline_check_is_opt_in(tmp_path, monkeypatch):
    """The container preflight stays as accepted: the new check is opt-in."""
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: pytest.fail("the offline check must be opt-in"))

    assert preflight.run([]) == 0


# ── the runtime-only path the installer runs before it installs anything ─────


def _bare_layout(tmp_path, monkeypatch):
    """A delivery layout in which nothing has been created yet.

    The Windows installer asks the private runtime to prove itself the moment
    the runtime has been built — before the program payload, the configuration
    and the persistent directories exist. None of them may be required by that
    check, and none of them may be produced by it.
    """
    app_root = tmp_path / "app"
    config_path = tmp_path / "data" / "config" / "config.yaml"
    template = app_root / "auto_tune" / "config.template.yaml"
    monkeypatch.setenv("AUTO_TUNE_APP_ROOT", str(app_root))
    monkeypatch.setenv("AUTO_TUNE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AUTO_TUNE_DATASETS_DIR", str(tmp_path / "data" / "datasets"))
    monkeypatch.setenv("AUTO_TUNE_TEMPLATE_PATH", str(template))
    return tmp_path


def test_the_runtime_only_check_needs_no_configuration_or_payload(tmp_path, monkeypatch):
    root = _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details", lambda: _details())

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 0

    assert list(root.rglob("*")) == [], (
        "the runtime-only check must not prepare a single directory or file")


def test_the_runtime_only_check_creates_no_directory_or_file(tmp_path, monkeypatch):
    root = _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details", lambda: _details())

    assert preflight.run(["--require-offline-runtime"]) == 0

    assert not (root / "app").exists()
    assert not (root / "data").exists()


def test_the_runtime_only_check_never_reads_the_delivery_layout(tmp_path, monkeypatch):
    """The dedicated path is taken on the flag itself, not as a side effect."""
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details", lambda: _details())

    def forbidden(*args, **kwargs):
        pytest.fail("the runtime-only check touched the delivery layout")

    monkeypatch.setattr(preflight, "resolve_paths", forbidden)
    monkeypatch.setattr(preflight, "ensure_directories", forbidden)
    monkeypatch.setattr(preflight, "check_writable", forbidden)
    monkeypatch.setattr(preflight, "bootstrap_config", forbidden)
    monkeypatch.setattr(preflight, "check_dependencies", forbidden)

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 0


def test_the_runtime_only_check_reports_a_mismatch_without_a_configuration(
        tmp_path, monkeypatch, capsys):
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: _details(torch_version="2.5.1"))

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 1

    printed = capsys.readouterr()
    assert preflight.DELIVERY_RUNTIME_MISMATCH in printed.err
    assert preflight.DELIVERY_CONFIG_MISSING not in printed.err, (
        "a first installation has no configuration yet; that is not the failure")


def test_the_runtime_only_check_still_refuses_a_foreign_interpreter(
        tmp_path, monkeypatch, capsys):
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: _details(executable=r"D:\other\python.exe"))

    assert preflight.run(["--require-offline-runtime",
                          "--expect-python", _details()["executable"]]) == 1
    assert preflight.DELIVERY_RUNTIME_MISMATCH in capsys.readouterr().err


# ── the ordinary start-up preflight keeps its full behaviour ────────────────


def test_the_plain_startup_preflight_still_requires_the_configuration(
        tmp_path, monkeypatch, capsys):
    _bare_layout(tmp_path, monkeypatch)

    assert preflight.run([]) == 1
    assert preflight.DELIVERY_CONFIG_MISSING in capsys.readouterr().err


def test_the_gpu_startup_preflight_still_runs_the_full_layout_check(
        tmp_path, monkeypatch, capsys):
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "require_gpu", lambda probe=None: None)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: pytest.fail("the full layout check comes first"))

    assert preflight.run(["--require-gpu"]) == 1
    assert preflight.DELIVERY_CONFIG_MISSING in capsys.readouterr().err


def test_combining_the_gpu_flag_keeps_the_full_startup_preflight(
        tmp_path, monkeypatch, capsys):
    """Asking for the GPU makes it the formal start-up preflight, not a runtime
    check: the layout, the configuration and the components are all still
    required, so the runtime-only shortcut must not apply."""
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "require_gpu", lambda probe=None: None)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: pytest.fail("the full layout check comes first"))

    assert preflight.run(["--require-gpu", "--require-offline-runtime"]) == 1
    assert preflight.DELIVERY_CONFIG_MISSING in capsys.readouterr().err


def test_combining_the_bootstrap_flag_keeps_the_full_startup_preflight(
        tmp_path, monkeypatch, capsys):
    _bare_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: pytest.fail("the full layout check comes first"))

    assert preflight.run(["--bootstrap-config", "--require-offline-runtime"]) == 1
    assert preflight.DELIVERY_TEMPLATE_MISSING in capsys.readouterr().err


def test_a_full_startup_preflight_still_verifies_the_private_runtime(
        tmp_path, monkeypatch, capsys):
    """The strictness of the check is unchanged: the container subset still runs
    it when it is asked for, and still refuses a CPU torch."""
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "mount_probe", lambda path: True)
    monkeypatch.setattr(preflight, "offline_runtime_details",
                        lambda: _details(torch_version="2.5.1"))

    assert preflight.run(["--require-mounts", "--require-offline-runtime"]) == 1
    assert preflight.DELIVERY_RUNTIME_MISMATCH in capsys.readouterr().err
