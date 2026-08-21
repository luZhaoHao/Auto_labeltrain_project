"""API / latest / training-boundary tests for immutable dataset snapshots (Studio S1.2)."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.dataset_snapshot import (
    SnapshotConflictError,
    SnapshotInsufficientSpaceError,
    SnapshotIOError,
    SnapshotValidationError,
)
from auto_tune.ui import app as app_mod


def _make_source(tmp_path, n=4, names=None):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    for i in range(n):
        (source / f"img{i}.jpg").write_bytes(f"image-{i}".encode())
        (source / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    names = names or {0: "defect"}
    lines = ["names:"]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    lines.append(f"nc: {len(names)}")
    (source / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source


def _use_tmp_log(monkeypatch, tmp_path):
    """Redirect app log/ I/O and snapshot storage into tmp_path/log."""
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    monkeypatch.setattr(app_mod, "LATEST_DATASET_PATH", log_dir / "latest_dataset.json")
    monkeypatch.setattr(app_mod, "DATASET_SNAPSHOT_ROOT", log_dir / "dataset_snapshots")
    return log_dir


def _register_source(monkeypatch, tmp_path, split=False):
    source = _make_source(tmp_path)
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "latest_dataset.json").write_text(
        json.dumps({"dataset_path": str(source), "split": split}), encoding="utf-8")
    return source


def _fake_finalize(*a, **k):
    return {
        "run_id": "manual:t", "run_name": "t", "source": "manual",
        "status": "completed", "analysis_status": "skipped", "metrics": {},
        "artifacts": {"report_path": None}, "error": None,
        "analysis_error": None, "history_error": None,
    }


class _FakeStdout:
    async def readline(self):
        return b""


class _FakeProc:
    returncode = 0
    stdout = _FakeStdout()

    async def wait(self):
        return 0


def _capture_subprocess(monkeypatch, captured, tmp_path):
    async def fake_subprocess_exec(*args, **kwargs):
        captured["cmd"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo",
    )
    monkeypatch.setattr(app_mod, "finalize_training_run", _fake_finalize)


# ── Task 5: split API replaces destructive move, atomic latest registration ──


def test_split_legacy_request_succeeds(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["reused"] is False
    assert len(data["snapshot_id"]) == 64
    assert data["snapshot_path"]
    assert data["manifest_path"]
    assert data["data_yaml_path"]
    assert data["train_count"] == 3
    assert data["val_count"] == 1
    assert data["background_count"] == 0
    assert data["total_bytes"] > 0


def test_split_reuses_identical_snapshot(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    first = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42}).json()
    second = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42}).json()
    assert second["status"] == "success"
    assert second["reused"] is True
    assert second["snapshot_id"] == first["snapshot_id"]


@pytest.mark.parametrize("body", [
    {"val_ratio": 0.0, "seed": 42},
    {"val_ratio": 1.0, "seed": 42},
    {"val_ratio": "0.2", "seed": 42},
    {"val_ratio": 0.2, "seed": True},
    {"val_ratio": 0.2, "seed": 42.0},
])
def test_split_invalid_params_return_400(tmp_path, monkeypatch, body):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json=body)
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "SNAPSHOT_VALIDATION_FAILED"


@pytest.mark.parametrize("error,status", [
    (SnapshotValidationError("bad"), 400),
    (SnapshotConflictError("conflict"), 409),
    (SnapshotInsufficientSpaceError("full"), 507),
    (SnapshotIOError("io"), 500),
])
def test_split_error_mapping(tmp_path, monkeypatch, error, status):
    _register_source(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise error

    monkeypatch.setattr(app_mod, "create_dataset_snapshot", boom)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == status
    assert resp.json()["error_code"] == error.error_code


def test_split_service_failure_does_not_update_latest(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    latest = app_mod.LATEST_DATASET_PATH
    before = latest.read_bytes()

    def boom(*a, **k):
        raise SnapshotValidationError("no")

    monkeypatch.setattr(app_mod, "create_dataset_snapshot", boom)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == 400
    assert latest.read_bytes() == before


def test_latest_old_json_readable_not_rewritten_snapshot_valid_false(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    latest = app_mod.LATEST_DATASET_PATH
    before = latest.read_bytes()
    client = TestClient(app_mod.app)
    data = client.get("/api/dataset/latest").json()["dataset"]
    assert data["snapshot_valid"] is False
    assert latest.read_bytes() == before


def test_latest_new_snapshot_valid(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    data = client.get("/api/dataset/latest").json()["dataset"]
    assert data["snapshot_valid"] is True
    assert data["snapshot_id"]
    assert data["data_yaml_path"]


def test_latest_corrupted_snapshot_valid_false_source_survives(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    Path(info["data_yaml_path"]).write_text("broken: true\n", encoding="utf-8")
    data = client.get("/api/dataset/latest").json()["dataset"]
    assert data["snapshot_valid"] is False
    assert data["dataset_path"]


def test_latest_atomic_write_failure_then_reuse(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    latest = app_mod.LATEST_DATASET_PATH
    before = latest.read_bytes()
    real_write = app_mod._write_json_atomic
    calls = {"n": 0}

    def flaky_write(path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_write(path, payload)

    monkeypatch.setattr(app_mod, "_write_json_atomic", flaky_write)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == 500
    assert resp.json()["error_code"] == "SNAPSHOT_IO_FAILED"
    assert latest.read_bytes() == before
    # The published snapshot can be reused and registered by the next request.
    resp2 = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp2.status_code == 200
    assert resp2.json()["reused"] is True
    assert latest.exists()


def test_common_context_corrupted_snapshot_does_not_crash(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    Path(info["data_yaml_path"]).write_text("broken\n", encoding="utf-8")
    ctx = app_mod._common_context()
    assert ctx["latest_dataset"] is not None
    assert ctx["latest_dataset"]["snapshot_valid"] is False


# ── Task 6: resolver + training/tuning must use validated snapshots ──


def test_resolver_returns_snapshot_data_yaml(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    latest = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    resolved = app_mod._resolve_validated_snapshot_data_yaml(latest)
    assert resolved == Path(latest["data_yaml_path"]).resolve()
    assert resolved.is_absolute()
    assert resolved.is_file()


def test_resolver_rejects_old_latest_without_snapshot(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    latest = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    with pytest.raises(SnapshotValidationError):
        app_mod._resolve_validated_snapshot_data_yaml(latest)


def test_resolver_rejects_corrupted_snapshot(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    Path(info["data_yaml_path"]).write_text("broken\n", encoding="utf-8")
    latest = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    with pytest.raises(SnapshotValidationError):
        app_mod._resolve_validated_snapshot_data_yaml(latest)


def test_training_start_uses_snapshot_data_yaml(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    snapshot_data_yaml = json.loads(
        app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))["data_yaml_path"]

    captured = {}
    _capture_subprocess(monkeypatch, captured, tmp_path)
    app_mod._running_training.clear()
    try:
        resp = client.post("/api/training/start", json={"epochs": 1, "model": "yolov8n.pt"})
        assert resp.status_code == 200
        cmd = captured.get("cmd") or []
        data_args = [a for a in cmd if str(a).startswith("data=")]
        assert data_args, f"no data= arg in {cmd}"
        assert data_args[0] == f"data={os.path.abspath(snapshot_data_yaml)}"
    finally:
        app_mod._running_training.clear()


def test_training_start_manual_data_yaml_preserved(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    manual = str(tmp_path / "manual.yaml")
    Path(manual).write_text("path: .\n", encoding="utf-8")

    captured = {}
    _capture_subprocess(monkeypatch, captured, tmp_path)
    app_mod._running_training.clear()
    try:
        resp = client.post("/api/training/start", json={"data_yaml": manual, "epochs": 1})
        assert resp.status_code == 200
        cmd = captured.get("cmd") or []
        data_args = [a for a in cmd if str(a).startswith("data=")]
        assert data_args, f"no data= arg in {cmd}"
        assert data_args[0] == f"data={os.path.abspath(manual)}"
    finally:
        app_mod._running_training.clear()


def test_training_start_blocked_when_snapshot_invalid(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    Path(info["data_yaml_path"]).write_text("broken\n", encoding="utf-8")

    called = {"n": 0}

    async def never_launch(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("training must not launch with an invalid snapshot")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never_launch)
    resp = client.post("/api/training/start", json={"epochs": 1})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "SNAPSHOT_VALIDATION_FAILED"
    assert called["n"] == 0


def test_tuning_start_blocked_when_snapshot_invalid(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    Path(info["data_yaml_path"]).write_text("broken\n", encoding="utf-8")
    resp = client.post("/tuning/start", json={"mode": "dry_run"})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "SNAPSHOT_VALIDATION_FAILED"


# ── P1: tuning must bind the validated snapshot data.yaml into the loop config ──


def test_tuning_start_forwards_snapshot_data_yaml(tmp_path, monkeypatch):
    _register_source(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    info = json.loads(app_mod.LATEST_DATASET_PATH.read_text(encoding="utf-8"))
    snapshot_data_yaml = os.path.abspath(info["data_yaml_path"])

    captured = {}

    def fake_run_tuning_loop(config, **kwargs):
        captured["config"] = config
        captured["skip_execute"] = kwargs.get("skip_execute")
        return {
            "final_result": None, "best_iteration": None,
            "best_train_name": None, "best_metrics": None,
            "iterations": [], "eval_mode": "comprehensive",
        }

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.run_tuning_loop", fake_run_tuning_loop
    )
    resp = client.post("/tuning/start", json={"mode": "dry_run", "max_retries": 1})
    assert resp.status_code == 200
    assert captured["config"]["training"]["data_yaml"] == snapshot_data_yaml
    # 注入必须使用副本，不得直接修改共享 APP_CONFIG
    assert captured["config"] is not app_mod.APP_CONFIG


def test_tuning_start_rejects_missing_latest(tmp_path, monkeypatch):
    _use_tmp_log(monkeypatch, tmp_path)

    def never(*a, **k):
        raise AssertionError("tuning must not start without a valid snapshot")

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.run_tuning_loop", never)
    resp = TestClient(app_mod.app).post("/tuning/start", json={"mode": "train", "max_retries": 1})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "SNAPSHOT_VALIDATION_FAILED"


def test_tuning_start_rejects_old_latest_without_snapshot(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    _use_tmp_log(monkeypatch, tmp_path)
    app_mod.LATEST_DATASET_PATH.write_text(
        json.dumps({"dataset_path": str(source), "split": False}), encoding="utf-8"
    )

    def never(*a, **k):
        raise AssertionError("tuning must not start without a valid snapshot")

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.run_tuning_loop", never)
    resp = TestClient(app_mod.app).post("/tuning/start", json={"mode": "train", "max_retries": 1})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "SNAPSHOT_VALIDATION_FAILED"
