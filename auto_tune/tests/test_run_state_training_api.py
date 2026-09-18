"""S1.5 Task 3: ordinary training wired into the unified run-state contract.

Covers the pre-launch persistence gate, per-SSE run identity/event_seq,
terminal-state consistency, and the stopped/failed paths.
"""

import asyncio
import json
import os

import pytest

from auto_tune.modules.run_state.events import EventBroker
from auto_tune.modules.run_state.models import RunStatePersistenceError
from auto_tune.modules.run_state.service import (
    new_run_state,
    project_public_state,
    read_run_state,
    update_run_state,
)


def _redirect_log(monkeypatch, tmp_path):
    """Redirect os.path.join('log', ...) to a temp dir, returning that dir."""
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    return log_dir


def _monkeypatch_subprocess(monkeypatch, tmp_path, lines, returncode=0, pid=99999):
    """Patch create_subprocess_exec + executor helpers with a fake subprocess."""
    class FakeStdout:
        def __init__(self):
            self._lines = [l.encode("utf-8", errors="replace") + b"\n" for l in lines]
            self._i = 0

        async def readline(self):
            if self._i >= len(self._lines):
                return b""
            line = self._lines[self._i]
            self._i += 1
            return line

    class FakeProc:
        def __init__(self):
            self.returncode = returncode
            self.stdout = FakeStdout()
            self.pid = pid

        async def wait(self):
            return returncode

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo",
    )


def _fake_finalize(monkeypatch, tmp_path, status="completed"):
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name, "source": "manual",
            "status": status, "analysis_status": "skipped", "metrics": {},
            "artifacts": {"report_path": None}, "error": training_error,
            "analysis_error": None, "history_error": None,
        }
    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)


def _sse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ── Step 1: pre-launch persistence gate ──


def test_start_persist_failure_rejects_before_subprocess(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    launched = []

    async def fake_subprocess_exec(*args, **kwargs):
        launched.append(True)
        raise AssertionError("must not create a subprocess")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"),
    )

    def boom(*args, **kwargs):
        raise RunStatePersistenceError("disk full")

    monkeypatch.setattr("auto_tune.ui.app.write_run_state", boom)

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "epochs": 1,
        })
        assert resp.status_code == 500
        assert "RUN_STATE_PERSIST_FAILED" in resp.text
        assert launched == []
    finally:
        app_mod._running_training.clear()


# ── Step 2: stream identity, monotonic seq, terminal consistency ──


def test_start_streams_run_id_phase_seq_and_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    _monkeypatch_subprocess(monkeypatch, tmp_path, [
        "  1/2  1.20G  1.234  0.456  0.789",
        "all 10 50 0.123 0.456 0 0.111",
    ], returncode=0)
    _fake_finalize(monkeypatch, tmp_path, status="completed")

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "epochs": 1,
        })
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        stamped = [e for e in events if "event_seq" in e]
        assert stamped, "expected events carrying event_seq"
        run_ids = {e["run_id"] for e in stamped}
        assert len(run_ids) == 1
        run_id = run_ids.pop()
        assert run_id.startswith("manual:")
        # training_log events carry strictly increasing seqs
        log_seqs = [e["event_seq"] for e in stamped if e.get("event") == "training_log"]
        assert log_seqs
        assert log_seqs == sorted(log_seqs)
        assert len(set(log_seqs)) == len(log_seqs)
        # every stamped event carries a phase
        assert all(e.get("phase") for e in stamped)

        # terminal state persisted (never deleted)
        state = read_run_state(log_dir / "training_running.json", run_kind="manual")
        assert state is not None
        assert state.run_id == run_id
        assert state.status == "completed"
        assert state.phase == "terminal"
        # the persisted terminal event advances seq beyond the streamed events
        assert state.last_event is not None
        assert state.last_event.seq >= max(log_seqs)

        # status API returns the same run identity and terminal state
        api = client.get("/api/training/running").json()
        assert api["run_id"] == run_id
        assert api["status"] == "completed"
        assert api["phase"] == "terminal"
        assert api["running"] is False
    finally:
        app_mod._running_training.clear()


def test_start_failure_writes_failed_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    _monkeypatch_subprocess(monkeypatch, tmp_path, ["line one"], returncode=1)
    _fake_finalize(monkeypatch, tmp_path, status="failed")

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "epochs": 1,
        })
        assert resp.status_code == 200
        assert "训练失败" in resp.text
        state = read_run_state(log_dir / "training_running.json", run_kind="manual")
        assert state is not None
        assert state.status == "failed"
        assert state.phase == "terminal"
        api = client.get("/api/training/running").json()
        assert api["status"] == "failed"
        assert api["running"] is False
    finally:
        app_mod._running_training.clear()


# ── Step 4: stop → cancelled/terminal, file never deleted ──


def test_stop_controller_lost_returns_409(tmp_path, monkeypatch):
    """After a restart there is no live controller; stop must not fake a stop
    nor rewrite the running record to cancelled."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(
        state_file, state, status="running", phase="training",
        pid=99999, process_create_token=None,
    )

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/stop")
        assert resp.status_code == 409
        assert "CONTROLLER_LOST" in resp.text
        loaded = read_run_state(state_file, run_kind="manual")
        assert loaded.status != "cancelled"
        assert loaded.status != "running"
    finally:
        app_mod._running_training.clear()


def test_alive_process_reports_running_not_deleted(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod
    import auto_tune.modules.run_state.manual_controller as mc

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    run_state = new_run_state("manual", run_name="train9")
    run_state = update_run_state(
        state_file, run_state, status="running", phase="training", pid=os.getpid(),
    )
    broker = EventBroker(run_state.run_id)
    controller = mc.ManualRunController(
        run_state=run_state, state_file=str(state_file), cmd=["yolo"],
        params={}, train_name="train9", train_dir=str(tmp_path),
        data_yaml=str(tmp_path / "data.yaml"), model="yolov8n.pt", epochs=1,
        log_path=str(tmp_path / "training.log"), finalize_cb=None,
        broker=broker, manager=app_mod._RUN_MANAGER,
    )
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()
        assert data["running"] is True
        assert data["status"] == "running"
        assert data["run_id"] == run_state.run_id
        # A running record is never deleted just because a client polls it.
        assert state_file.exists()
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)
