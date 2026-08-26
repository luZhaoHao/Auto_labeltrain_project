"""S1.5 Task 4: auto-tuning wired into the unified run-state contract.

Covers the tuning run identity/phase/terminal contract, the cancelled path,
and conservative reconciliation after controller loss / PID reuse.
"""

import json
import os

from auto_tune.modules.run_state.events import EventBroker
from auto_tune.modules.run_state.models import RunStatePersistenceError
from auto_tune.modules.run_state.process_identity import capture_process_identity
from auto_tune.modules.run_state.service import (
    new_run_state,
    read_run_state,
    update_run_state,
)
from auto_tune.modules.run_state.tuning_controller import TuningRunController


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


def _valid_decision():
    return {
        "diagnosis": "test diagnosis",
        "action": "apply changes",
        "hyperparameter_changes": {"lr0": 0.001},
        "training_overrides": {},
        "raw_response": '{"hyperparameter_changes": {"lr0": 0.001}}',
        "error": None,
    }


def _monkeypatch_loop_inputs(monkeypatch):
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: _valid_decision(),
    )


def _sse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _start_payload(**overrides):
    payload = {
        "reference_run": None,
        "max_retries": 1,
        "mode": "dry_run",
        "auto_analyze": False,
        "auto_loop": False,
        "eval_mode": "comprehensive",
    }
    payload.update(overrides)
    return payload


# ── Step 1: identity, phases, terminal consistency ──


def test_tuning_dry_run_single_run_id_and_completed_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)
    _monkeypatch_loop_inputs(monkeypatch)

    app_mod._running_training.clear()
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_start_payload())
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        stamped = [e for e in events if "run_id" in e]
        assert stamped
        run_ids = {e["run_id"] for e in stamped}
        assert len(run_ids) == 1
        run_id = run_ids.pop()
        assert run_id.startswith("tuning:")
        # every stamped event carries a phase
        assert all(e.get("phase") for e in stamped)

        state = read_run_state(log_dir / "tuning_running.json", run_kind="tuning")
        assert state is not None
        assert state.run_id == run_id
        assert state.status == "completed"
        assert state.phase == "terminal"

        api = client.get("/api/tuning/status").json()
        assert api["run_id"] == run_id
        assert api["status"] == "completed"
        assert api["running"] is False
    finally:
        app_mod._running_training.clear()
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()


def test_tuning_exception_writes_failed_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)

    def boom(**kwargs):
        raise RuntimeError("perception exploded")

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception", boom)

    app_mod._running_training.clear()
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_start_payload())
        assert resp.status_code == 200
        assert any(e["status"] in ("error", "failed") for e in _sse_events(resp.text))

        state = read_run_state(log_dir / "tuning_running.json", run_kind="tuning")
        assert state is not None
        assert state.status == "failed"
        assert state.phase == "terminal"

        api = client.get("/api/tuning/status").json()
        assert api["status"] == "failed"
        assert api["running"] is False
    finally:
        app_mod._running_training.clear()
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()


def test_tuning_stop_writes_cancelled_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "tuning_running.json"
    run_state = new_run_state("tuning")
    broker = EventBroker(run_state.run_id)

    def blocking_loop(on_progress, on_state, cancel_event):
        import time as _time
        while not cancel_event.is_set():
            _time.sleep(0.02)
        return {"error": "用户取消"}

    controller = TuningRunController(
        run_state=run_state, state_file=str(state_file), broker=broker,
        manager=app_mod._RUN_MANAGER, loop_runner=blocking_loop,
    )
    app_mod._RUN_MANAGER.register(controller)
    controller.start()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/stop")
        assert resp.status_code == 200
        assert state_file.exists()  # never deleted
        loaded = read_run_state(state_file, run_kind="tuning")
        assert loaded.status == "cancelled"
        assert loaded.phase == "terminal"
        api = client.get("/api/tuning/status").json()
        assert api["status"] == "cancelled"
        assert api["running"] is False
    finally:
        controller.cancel_event.set()
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_tuning_persist_failure_rejects_before_launch(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)

    def boom(*args, **kwargs):
        raise RunStatePersistenceError("disk full")

    monkeypatch.setattr("auto_tune.ui.app.write_run_state", boom)

    app_mod._running_training.clear()
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_start_payload())
        assert resp.status_code == 500
        assert "RUN_STATE_PERSIST_FAILED" in resp.text
    finally:
        app_mod._running_training.clear()
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()


# ── Step 5: restart / PID reuse reconciliation ──


def test_tuning_status_controller_lost_after_restart(tmp_path, monkeypatch):
    """After losing the in-memory controller, a persisted running record whose
    PID identity still matches must project to interrupted/controller_lost and
    be written back."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "tuning_running.json"
    token = capture_process_identity(os.getpid()).process_create_token
    state = new_run_state("tuning")
    update_run_state(
        state_file, state, status="running", phase="training",
        pid=os.getpid(), process_create_token=token,
    )

    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/tuning/status").json()
        assert data["status"] == "interrupted"
        assert data["terminal_reason"] == "controller_lost"
        assert data["running"] is False
        # The conservative result is written back.
        persisted = read_run_state(state_file, run_kind="tuning")
        assert persisted.status == "interrupted"
        assert persisted.phase == "terminal"
    finally:
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()


def test_tuning_status_pid_reused(tmp_path, monkeypatch):
    """A persisted running record whose PID identity no longer matches must
    project to interrupted/pid_reused."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "tuning_running.json"
    state = new_run_state("tuning")
    update_run_state(
        state_file, state, status="running", phase="training",
        pid=os.getpid(), process_create_token="windows-filetime:0",
    )

    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/tuning/status").json()
        assert data["status"] == "interrupted"
        assert data["terminal_reason"] == "pid_reused"
        assert data["running"] is False
    finally:
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()


def test_tuning_status_no_record_unknown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/tuning/status").json()
        assert data["status"] == "unknown"
        assert data["running"] is False
        assert data["run_id"] is None
    finally:
        app_mod._tuning_loop_active = False
        app_mod._tuning_cancel_event.clear()
