"""Tests for portable YOLO process invocation."""

from pathlib import Path

from auto_tune.modules.agent_engine import executor
from auto_tune.modules.agent_engine.executor import validate_training_preflight


def test_resolver_finds_conda_scripts_executable_when_path_is_missing(tmp_path, monkeypatch):
    """Catches direct python startup failing because yolo is absent from PATH."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    executable = scripts / "yolo.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(executor.shutil, "which", lambda name: None)
    monkeypatch.setattr(executor.sys, "prefix", str(tmp_path))

    assert executor.resolve_yolo_executable() == str(executable)


def test_build_command_uses_resolved_executable(tmp_path, monkeypatch):
    """Catches command construction reverting to an unresolved bare yolo name."""
    monkeypatch.setattr(executor, "resolve_yolo_executable", lambda: r"C:\env\Scripts\yolo.exe")
    args_path = tmp_path / "train1" / "args.yaml"
    args_path.parent.mkdir()

    command = executor.build_yolo_command("train1", str(args_path), {"epochs": 1})

    assert command[:2] == [r"C:\env\Scripts\yolo.exe", "train"]


def test_preflight_rejects_missing_reference_directory(tmp_path):
    errors = validate_training_preflight(
        reference_run="train38",
        reference_dir=str(tmp_path / "missing"),
        merged_params={"model": "yolov8n.pt", "data": str(tmp_path / "data.yaml")},
    )
    assert any("reference" in error.lower() for error in errors)


def test_preflight_rejects_missing_local_data_yaml(tmp_path):
    errors = validate_training_preflight(
        reference_run=None,
        reference_dir=None,
        merged_params={"model": "yolov8n.pt", "data": str(tmp_path / "missing.yaml")},
    )
    assert any("data" in error.lower() for error in errors)


def test_preflight_accepts_existing_yaml_and_builtin_weight_name(tmp_path, monkeypatch):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo",
    )

    errors = validate_training_preflight(
        reference_run=None,
        reference_dir=None,
        merged_params={"model": "yolov8n.pt", "data": str(data_yaml)},
    )
    assert errors == []


def test_launch_training_uses_supplied_command_without_rebuilding(tmp_path, monkeypatch):
    args_path = tmp_path / "args.yaml"
    args_path.write_text("epochs: 1\n", encoding="utf-8")
    supplied = ["yolo", "train", "epochs=1"]
    captured = {}

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.build_yolo_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.subprocess.Popen", fake_popen)

    proc = executor.launch_training("train1", str(args_path), {"epochs": 1}, command=supplied)

    assert isinstance(proc, FakeProcess)
    assert captured["command"] == supplied

