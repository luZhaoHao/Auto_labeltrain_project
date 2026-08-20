"""Tests for portable YOLO process invocation."""

import os
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


def _make_training_process(tmp_path, content=b""):
    train_dir = tmp_path / "detect" / "autotune_1"
    train_dir.mkdir(parents=True)
    (train_dir / "yolo_train.log").write_bytes(content)
    return executor.TrainingProcess("autotune_1", str(train_dir), object())


def test_drain_output_consumes_each_line_once(tmp_path):
    proc = _make_training_process(tmp_path, b"first\nsecond\nthird\n")

    first = proc.drain_output()
    assert first == ["first", "second", "third"]
    assert proc.drain_output() == []
    # No callback also drains (backward-compatible call).
    assert proc.drain_output(on_line=lambda line: None) == []


def test_drain_output_utf8_decode_replaces_invalid_bytes(tmp_path):
    proc = _make_training_process(tmp_path, b"ok\nbad\xffline\nend\n")

    lines = proc.drain_output()
    assert lines[0] == "ok"
    assert lines[1] == "bad�line"
    assert lines[2] == "end"


def test_drain_output_calls_on_line_for_each_consumed_line(tmp_path):
    proc = _make_training_process(tmp_path, b"a\nb\n")
    seen = []

    proc.drain_output(on_line=seen.append)

    assert seen == ["a", "b"]


def test_drain_output_preserves_partial_tail_and_completes_it(tmp_path):
    proc = _make_training_process(tmp_path, b"line1\nline2")
    assert proc.drain_output() == ["line1"]  # line2 is a partial tail

    # Subprocess keeps writing: the tail must complete, not be lost/duplicated.
    with open(os.path.join(proc.train_dir, "yolo_train.log"), "ab") as f:
        f.write(b"more\n")
    assert proc.drain_output() == ["line2more"]

    # A final line after process end is still drained exactly once.
    with open(os.path.join(proc.train_dir, "yolo_train.log"), "ab") as f:
        f.write(b"tail\n")
    assert proc.drain_output() == ["tail"]
    assert proc.drain_output() == []


def test_drain_output_missing_log_returns_empty(tmp_path):
    proc = executor.TrainingProcess("x", str(tmp_path / "missing"), object())
    assert proc.drain_output() == []


def test_launch_training_closes_log_file_after_popen(tmp_path, monkeypatch):
    args_path = tmp_path / "args.yaml"
    args_path.write_text("epochs: 1\n", encoding="utf-8")
    captured = {}

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        captured["stdout"] = kwargs.get("stdout")
        return FakeProcess()

    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.subprocess.Popen", fake_popen)

    executor.launch_training("train1", str(args_path), {"epochs": 1}, command=["yolo", "train"])

    # The parent's log-file handle must be released right after Popen returns
    # (the child keeps its own inherited handle).
    assert captured["stdout"].closed is True


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

