"""S1.5 reconnect fix: background controller decoupled from the SSE request.

The training subprocess, stdout consumption, log writes, state updates, and
terminal persistence run in a ``ManualRunController`` / ``TuningRunController``
that own their ``EventBroker``. SSE is only a subscriber. These tests prove:

- disconnecting a subscriber never cancels the controller (stdout is still
  consumed, training.log still grows, terminal state is accurate);
- resubscribe via ``after_seq`` replays missed buffered events;
- every published event gets a unique, strictly increasing ``event_seq``;
- duplicate start while active returns 409 RUN_ALREADY_ACTIVE;
- a stopped/terminal run cannot reconnect as an active stream;
- stop never rewrites an existing terminal state.

The subscriber-disconnect scenarios drive the controller directly through the
same broker/controller mechanism the SSE endpoint uses (httpx cannot do a
partial read of a long-lived SSE stream in the TestClient).
"""

import asyncio
import json
import os
import threading
import time

from auto_tune.modules.run_state.events import EventBroker
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.modules.run_state.service import (
    new_run_state,
    read_run_state,
    update_run_state,
)


def _redirect_log(monkeypatch, tmp_path):
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


def _sse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


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


class ControllableStdout:
    """Async readline fed from the test thread; blocks until feed/finish."""

    def __init__(self):
        self._lines = []
        self._lock = threading.Lock()
        self._finished = False
        self._wake = threading.Event()

    async def readline(self):
        while True:
            with self._lock:
                if self._lines:
                    return self._lines.pop(0)
                if self._finished:
                    return b""
            await asyncio.get_running_loop().run_in_executor(None, self._wake.wait)
            self._wake.clear()

    def feed(self, line_bytes):
        with self._lock:
            self._lines.append(line_bytes)
        self._wake.set()

    def finish(self):
        with self._lock:
            self._finished = True
        self._wake.set()


def _make_manual_controller(tmp_path, monkeypatch, rc=0):
    """Build a ManualRunController wired to a controllable fake subprocess.

    The fake ``create_subprocess_exec`` is installed via pytest monkeypatch so
    it stays in effect while the controller task runs inside ``asyncio.run``.
    """
    import auto_tune.modules.run_state.manual_controller as mc

    run_state = new_run_state("manual", run_name="train1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    stdout = ControllableStdout()

    class FakeProc:
        def __init__(self):
            self.stdout = stdout
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = rc
            return rc

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)
    controller = mc.ManualRunController(
        run_state=run_state,
        state_file=state_file,
        cmd=["yolo", "train"],
        params={"batch": 16, "imgsz": 640},
        train_name="train1",
        train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt",
        epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None,
        broker=broker,
        manager=manager,
    )
    manager.register(controller)
    return controller, broker, manager, stdout


def _make_tuning_controller(tmp_path, result=None, error=None):
    """Build a TuningRunController whose loop emits a few events then returns."""
    from auto_tune.modules.run_state.tuning_controller import TuningRunController

    run_state = new_run_state("tuning")
    state_file = os.path.join(str(tmp_path), "tuning_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()

    def loop_runner(on_progress, on_state, cancel_event):
        on_state("preparing", "tuning_start", "调优会话开始")
        on_progress(1, "迭代 1 开始", step="iteration_start")
        on_state("analyzing", "perception", "感知层")
        on_state("finalizing", "finalizing", "收尾")
        if cancel_event.is_set():
            return {"error": "用户取消"}
        if error is not None:
            raise RuntimeError(error)
        return result if result is not None else {"final_result": {"train_name": "dry_run"}}

    controller = TuningRunController(
        run_state=run_state,
        state_file=state_file,
        broker=broker,
        manager=manager,
        loop_runner=loop_runner,
    )
    manager.register(controller)
    return controller, broker, manager


# ── EventBroker unit contract ──


def test_event_broker_bounded_ring_and_unique_seq():
    broker = EventBroker("manual:test", max_events=5)
    published = []
    for i in range(10):
        published.append(broker.publish({"message": f"m{i}"}))

    seqs = [e["event_seq"] for e in published]
    assert seqs == list(range(1, 11))
    recent = broker.recent()
    assert len(recent) == 5
    assert recent[0]["message"] == "m5"
    assert [e["event_seq"] for e in recent] == [6, 7, 8, 9, 10]


def test_event_broker_subscribe_replays_after_seq():
    broker = EventBroker("manual:test", max_events=20)
    for i in range(6):
        broker.publish({"message": f"m{i}"})

    q, replay = broker.subscribe(after_seq=3)
    try:
        assert [e["event_seq"] for e in replay] == [4, 5, 6]
        assert [e["message"] for e in replay] == ["m3", "m4", "m5"]
    finally:
        broker.unsubscribe(q)


# ── SSE disconnect must not stop the background controller ──


def test_manual_controller_continues_after_subscriber_disconnect(tmp_path, monkeypatch):
    """After the (SSE) subscriber disconnects, the controller keeps consuming
    stdout, keeps growing training.log, and writes an accurate terminal state —
    it must not degrade to interrupted."""
    controller, broker, _manager, stdout = _make_manual_controller(tmp_path, monkeypatch)

    async def scenario():
        task = asyncio.create_task(controller._run())
        q, replay = broker.subscribe(0)
        got = []
        while len(got) < 3:  # read the three banner events, then "disconnect"
            try:
                got.append(q.get_nowait())
            except Exception:
                await asyncio.sleep(0.02)
        broker.unsubscribe(q)  # SSE connection dropped

        stdout.feed(b"  1/2  1.20G  1.234  0.456  0.789\n")
        stdout.feed(b"all 10 50 0.123 0.456 0 0.111\n")
        await asyncio.sleep(0.3)
        stdout.finish()
        await asyncio.wait_for(task, timeout=10)
        return got

    got = asyncio.run(scenario())
    assert len(got) >= 3
    state = read_run_state(controller.state_file, run_kind="manual")
    assert state is not None
    assert state.status == "completed"
    assert state.phase == "terminal"
    assert state.terminal_reason is None  # never interrupted
    # stdout was still consumed after the disconnect.
    log = (tmp_path / "training.log").read_text(encoding="utf-8")
    assert "  1/2  1.20G  1.234  0.456  0.789" in log
    assert "all 10 50" in log


def test_manual_controller_failure_after_disconnect_writes_failed(tmp_path, monkeypatch):
    controller, broker, _manager, stdout = _make_manual_controller(tmp_path, monkeypatch, rc=1)

    async def scenario():
        task = asyncio.create_task(controller._run())
        q, replay = broker.subscribe(0)
        got = []
        while len(got) < 3:
            try:
                got.append(q.get_nowait())
            except Exception:
                await asyncio.sleep(0.02)
        broker.unsubscribe(q)
        stdout.finish()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())
    state = read_run_state(controller.state_file, run_kind="manual")
    assert state.status == "failed"
    assert state.phase == "terminal"


def test_tuning_controller_continues_after_disconnect(tmp_path):
    controller, broker, _manager = _make_tuning_controller(
        tmp_path, result={"final_result": {"train_name": "x"}}
    )
    controller.start()
    q, replay = broker.subscribe(0)
    got = []
    deadline = time.time() + 5
    while len(got) < 2 and time.time() < deadline:
        try:
            got.append(q.get_nowait())
        except Exception:
            time.sleep(0.02)
    broker.unsubscribe(q)  # SSE disconnected

    deadline = time.time() + 10
    while time.time() < deadline:
        state = read_run_state(controller.state_file, run_kind="tuning")
        if state is not None and state.status != "running":
            break
        time.sleep(0.05)
    assert state is not None
    assert state.status == "completed"
    assert state.phase == "terminal"


# ── resubscribe via after_seq replays missed events ──


def test_resubscribe_after_seq_replays_missed_events(tmp_path, monkeypatch):
    controller, broker, _manager, stdout = _make_manual_controller(tmp_path, monkeypatch)

    async def scenario():
        task = asyncio.create_task(controller._run())
        q1, replay = broker.subscribe(0)
        got = []
        while len(got) < 3:
            try:
                got.append(q1.get_nowait())
            except Exception:
                await asyncio.sleep(0.02)
        after_seq = max(e["event_seq"] for e in got)
        broker.unsubscribe(q1)

        # events produced while "offline"
        stdout.feed(b"  1/2  1.20G  1.234  0.456  0.789\n")
        await asyncio.sleep(0.3)

        q2, replay2 = broker.subscribe(after_seq=after_seq)
        missed = list(replay2)
        broker.unsubscribe(q2)
        assert missed, "expected replayed events after after_seq"
        assert all(e["event_seq"] > after_seq for e in missed)
        seqs = [e["event_seq"] for e in missed]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

        stdout.finish()
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(scenario())


# ── every event gets a unique strictly-increasing seq (HTTP, full read) ──


def test_manual_http_events_seq_strictly_increasing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod
    import auto_tune.modules.run_state.manual_controller as mc

    _redirect_log(monkeypatch, tmp_path)
    detect_dir = tmp_path / "detect"
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir", lambda: str(detect_dir))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable", lambda: "yolo")
    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run",
                        lambda *a, **k: {"run_id": "manual:x", "run_name": "x",
                                         "source": "manual", "status": "completed",
                                         "analysis_status": "skipped", "metrics": {},
                                         "artifacts": {"report_path": None}, "error": None,
                                         "analysis_error": None, "history_error": None})

    class FixedStdout:
        def __init__(self):
            self._lines = [b"  1/2  1.20G  1.234  0.456  0.789\n"]
            self._i = 0

        async def readline(self):
            if self._i >= len(self._lines):
                return b""
            line = self._lines[self._i]
            self._i += 1
            return line

    class FakeProc:
        def __init__(self):
            self.stdout = FixedStdout()
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt", "epochs": 1,
        })
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        stamped = [e for e in events if "event_seq" in e]
        assert stamped
        seqs = [e["event_seq"] for e in stamped]
        assert seqs == list(range(min(seqs), min(seqs) + len(seqs))), seqs
        assert len(set(seqs)) == len(seqs)
        assert all(e.get("run_id") and e.get("phase") for e in stamped)
    finally:
        app_mod._running_training.clear()


def test_tuning_http_events_seq_strictly_increasing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)
    _monkeypatch_loop_inputs(monkeypatch)

    app_mod._running_training.clear()
    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json={
            "reference_run": None, "max_retries": 1, "mode": "dry_run",
            "auto_analyze": False, "auto_loop": False, "eval_mode": "comprehensive",
        })
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        stamped = [e for e in events if "event_seq" in e]
        assert stamped
        seqs = [e["event_seq"] for e in stamped]
        assert seqs == list(range(min(seqs), min(seqs) + len(seqs))), seqs
        assert len(set(seqs)) == len(seqs)
        assert all(e.get("run_id") and e.get("phase") for e in stamped)
    finally:
        app_mod._running_training.clear()
        app_mod._tuning_cancel_event.clear()


# ── duplicate start while active → 409 ──


def test_duplicate_manual_start_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    controller, broker, manager, _stdout = _make_manual_controller(tmp_path, monkeypatch)
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt", "epochs": 1,
        })
        assert resp.status_code == 409
        assert "RUN_ALREADY_ACTIVE" in resp.text
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_duplicate_tuning_start_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    controller, broker, manager = _make_tuning_controller(tmp_path)
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json={
            "reference_run": None, "max_retries": 1, "mode": "dry_run",
            "auto_analyze": False, "auto_loop": False, "eval_mode": "comprehensive",
        })
        assert resp.status_code == 409
        assert "RUN_ALREADY_ACTIVE" in resp.text
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


# ── stop semantics: terminal states are never rewritten to cancelled ──


def test_manual_stop_does_not_rewrite_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status="completed", phase="terminal")

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/stop")
        assert resp.status_code == 200
        assert resp.json()["status_message"] == "completed"
        loaded = read_run_state(state_file, run_kind="manual")
        assert loaded.status == "completed"  # unchanged
    finally:
        app_mod._running_training.clear()


def test_stop_terminate_failure_not_fake_success(tmp_path, monkeypatch):
    """If the subprocess cannot be terminated, stop must return an error —
    never a fake 'stopped'."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    controller, broker, _manager, _stdout = _make_manual_controller(tmp_path, monkeypatch)

    class BadProc:
        returncode = None

        def terminate(self):
            raise OSError("access denied")

        def kill(self):
            raise OSError("access denied")

    controller._proc = BadProc()
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/stop")
        assert resp.status_code == 500
        assert "STOP_FAILED" in resp.text
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_remove_status_file_is_true_noop(tmp_path, monkeypatch):
    """The legacy _remove_status_file must never delete the status file."""
    from auto_tune.ui import app as app_mod

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    f = log_dir / "training_running.json"
    f.write_text('{"x": 1}', encoding="utf-8")
    import os as real_os
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    app_mod._remove_status_file()
    assert f.exists()


def test_tuning_stop_does_not_rewrite_terminal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "tuning_running.json"
    state = new_run_state("tuning")
    update_run_state(state_file, state, status="failed", phase="terminal")

    app_mod._tuning_cancel_event.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/stop")
        assert resp.status_code == 200
        assert resp.json()["status_message"] == "failed"
        loaded = read_run_state(state_file, run_kind="tuning")
        assert loaded.status == "failed"  # unchanged
    finally:
        app_mod._tuning_cancel_event.clear()


# ── a stopped run cannot reconnect as an active stream ──


def test_stopped_run_stream_not_active(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status="cancelled", phase="terminal")

    try:
        client = TestClient(app_mod.app)
        resp = client.get(f"/api/runs/{state.run_id}/stream")
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        assert events
        terminal = events[-1]
        assert terminal.get("status") == "cancelled"
        assert terminal.get("run_id") == state.run_id
    finally:
        app_mod._running_training.clear()


def test_run_stream_after_restart_reconciles_interrupted(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status="running", phase="training",
                     pid=os.getpid(), process_create_token="windows-filetime:0")

    try:
        client = TestClient(app_mod.app)
        resp = client.get(f"/api/runs/{state.run_id}/stream")
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        assert events
        terminal = events[-1]
        # No controller after "restart": the process is never re-adopted.
        assert terminal.get("status") in ("interrupted", "unknown")
        assert terminal.get("terminal_reason") in ("controller_lost", "pid_reused", "process_identity_unverifiable")
    finally:
        app_mod._running_training.clear()


# ── Startup stop race: stop must win even before/while the subprocess is created ──


def _build_race_controller(tmp_path):
    """A manual controller with a controllable create_subprocess_exec."""
    import auto_tune.modules.run_state.manual_controller as mc

    run_state = new_run_state("manual", run_name="race1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    controller = mc.ManualRunController(
        run_state=run_state, state_file=state_file,
        cmd=["yolo", "train"], params={"batch": 16, "imgsz": 640},
        train_name="race1", train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt", epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None, broker=broker, manager=manager,
    )
    manager.register(controller)
    return controller, broker, manager, state_file


def test_manual_stop_before_subprocess_created_skips_creation(tmp_path, monkeypatch):
    """A stop requested before the background task reaches
    create_subprocess_exec must skip creating any subprocess and end the run
    cancelled/terminal — with no fake STOP_FAILED."""
    import auto_tune.modules.run_state.manual_controller as mc

    controller, _broker, _manager, state_file = _build_race_controller(tmp_path)
    created = []

    class FakeStdout:
        async def readline(self):
            return b""

    class FakeProc:
        def __init__(self):
            self.stdout = FakeStdout()
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        created.append(True)
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    async def scenario():
        controller.start()
        # Stop before the task ever gets to run (no await in between).
        ok = controller.request_stop()
        assert ok is True
        await asyncio.wait_for(asyncio.shield(controller._task), timeout=10)
        return ok

    ok = asyncio.run(scenario())
    assert ok is True  # never a fake STOP_FAILED
    assert created == []  # no subprocess was ever created
    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "cancelled"
    assert state.phase == "terminal"


def test_manual_stop_during_subprocess_creation_race(tmp_path, monkeypatch):
    """Stop arriving while create_subprocess_exec is in flight must not start
    training nor report a fake STOP_FAILED: the fresh process is terminated and
    the run ends cancelled/terminal."""
    import auto_tune.modules.run_state.manual_controller as mc

    controller, _broker, _manager, state_file = _build_race_controller(tmp_path)

    entered = threading.Event()
    released = asyncio.Event()
    created = []

    class FakeStdout:
        async def readline(self):
            return b""

    class FakeProc:
        def __init__(self):
            self.stdout = FakeStdout()
            self.pid = os.getpid()
            self.returncode = None
            self.terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        async def wait(self):
            return self.returncode

    async def paused_subprocess_exec(*args, **kwargs):
        entered.set()
        await released.wait()
        proc = FakeProc()
        created.append(proc)
        return proc

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", paused_subprocess_exec)

    async def scenario():
        task = asyncio.create_task(controller._run())
        deadline = asyncio.get_running_loop().time() + 5
        while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert entered.is_set(), "create_subprocess_exec never entered"
        ok = controller.request_stop()  # stop while creation is paused
        released.set()
        await asyncio.wait_for(task, timeout=10)
        return ok

    ok = asyncio.run(scenario())
    assert ok is True  # never a fake STOP_FAILED
    assert len(created) == 1
    assert created[0].terminated is True  # training did not keep running
    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "cancelled"
    assert state.phase == "terminal"


def test_manual_stop_too_late_keeps_natural_completion(tmp_path, monkeypatch):
    """A run that completed naturally (returncode 0) must stay completed even
    if a stop request arrived too late to take effect on the process."""
    import auto_tune.modules.run_state.manual_controller as mc

    run_state = new_run_state("manual", run_name="late1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    stdout = ControllableStdout()

    class FakeProc:
        def __init__(self):
            self.stdout = stdout
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    controller = mc.ManualRunController(
        run_state=run_state, state_file=state_file,
        cmd=["yolo", "train"], params={"batch": 16, "imgsz": 640},
        train_name="late1", train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt", epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None, broker=broker, manager=manager,
    )
    manager.register(controller)

    async def scenario():
        task = asyncio.create_task(controller._run())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while controller._proc is None and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert controller._proc is not None
        # The process has already exited naturally (returncode 0) before the
        # stop request arrives; the stop can no longer take effect.
        controller._proc.returncode = 0
        ok = controller.request_stop()
        assert ok is True
        stdout.finish()  # EOF → the read loop breaks and terminal is computed
        await asyncio.wait_for(task, timeout=10)
        return ok

    asyncio.run(scenario())
    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "completed"  # not wrongly cancelled
    assert state.phase == "terminal"


def test_manual_stop_unapplied_nonzero_exit_is_failed(tmp_path, monkeypatch):
    """When a stop request never actually terminates the process and the
    process then exits non-zero on its own, the run must be failed — not
    cancelled. Only ``_stop_applied`` may write cancelled."""
    import auto_tune.modules.run_state.manual_controller as mc

    run_state = new_run_state("manual", run_name="fail1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    stdout = ControllableStdout()

    class FakeProc:
        def __init__(self):
            self.stdout = stdout
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 1
            return 1

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    controller = mc.ManualRunController(
        run_state=run_state, state_file=state_file,
        cmd=["yolo", "train"], params={"batch": 16, "imgsz": 640},
        train_name="fail1", train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt", epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None, broker=broker, manager=manager,
    )
    manager.register(controller)

    async def scenario():
        task = asyncio.create_task(controller._run())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while controller._proc is None and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert controller._proc is not None
        # The process has already exited non-zero on its own before the stop
        # request arrives; the stop can no longer take effect.
        controller._proc.returncode = 1
        ok = controller.request_stop()
        assert ok is True
        assert controller._stop_applied is False
        stdout.finish()  # EOF → the read loop breaks and terminal is computed
        await asyncio.wait_for(task, timeout=10)
        return ok

    asyncio.run(scenario())
    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "failed"  # not cancelled
    assert state.phase == "terminal"


# ── Terminal retention: finished controllers keep their broker for after_seq replay ──


def test_retained_manual_controller_replays_missed_events_after_completion(tmp_path, monkeypatch):
    """A client that disconnects before completion must still receive the real
    buffered events (7,8,9 + terminal + finalizer) when it reconnects after the
    run finished — not just a synthetic terminal built from the state file."""
    import auto_tune.modules.run_state.manual_controller as mc

    run_state = new_run_state("manual", run_name="replay1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    stdout = ControllableStdout()

    class FakeProc:
        def __init__(self):
            self.stdout = stdout
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    def fake_finalize(controller, returncode):
        return {
            "status": "done", "level": "success",
            "message": f"训练完成: {controller.train_name}",
            "run_id": controller.run_id, "phase": "terminal",
            "event": "finalize_result",
        }

    controller = mc.ManualRunController(
        run_state=run_state, state_file=state_file,
        cmd=["yolo", "train"], params={"batch": 16, "imgsz": 640},
        train_name="replay1", train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt", epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=fake_finalize, broker=broker, manager=manager,
    )
    manager.register(controller)

    async def scenario():
        task = asyncio.create_task(controller._run())
        q1, _replay1 = broker.subscribe(0)
        # Feed two lines so the controller publishes up through seq 6, then the
        # subscriber "disconnects".
        stdout.feed(b"  1/2  1.20G  1.234  0.456  0.789\n")
        stdout.feed(b"all 10 50 0.123 0.456 0 0.111\n")
        got = []
        while max((e["event_seq"] for e in got), default=0) < 6:
            try:
                got.append(q1.get_nowait())
            except Exception:
                await asyncio.sleep(0.02)
        broker.unsubscribe(q1)

        # the controller publishes 7, 8, 9 and completes while "offline"
        stdout.feed(b"  2/2  1.10G  1.111  0.333  0.555\n")
        await asyncio.sleep(0.3)
        stdout.finish()
        await asyncio.wait_for(task, timeout=10)
        return 6  # the seq the disconnected client had seen

    after_seq = asyncio.run(scenario())

    # Reconnect after completion: the retained broker must replay the real
    # events > 6 (including terminal + finalizer), with no synthetic extra seq.
    retained = manager.get(controller.run_id)
    assert retained is not None
    assert retained.is_done()
    replay = retained.broker.replay_after(after_seq)
    seqs = [e["event_seq"] for e in replay]
    assert len(seqs) >= 3
    assert seqs == list(range(after_seq + 1, after_seq + 1 + len(seqs))), seqs
    # The last replayed event is the real last published event — not a
    # fabricated seq (last_seq + 1) built from the state file.
    assert max(seqs) == broker.last_seq
    # Real terminal and ordinary-training finalizer results are in the replay.
    assert any(e.get("status") == "completed" for e in replay)
    assert any(e.get("event") == "finalize_result" for e in replay)


def test_tuning_controller_retained_after_completion(tmp_path):
    """A finished tuning controller stays queryable through its retained broker."""
    controller, broker, manager = _make_tuning_controller(
        tmp_path, result={"final_result": {"train_name": "x"}}
    )
    controller.start()
    q, replay = broker.subscribe(0)
    got = list(replay)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            got.append(q.get_nowait())
        except Exception:
            time.sleep(0.02)
        if controller.is_done():
            break
    while True:  # drain events that landed while the thread finished
        try:
            got.append(q.get_nowait())
        except Exception:
            break
    assert controller.is_done()
    assert manager.get(controller.run_id) is controller
    after_seq = max(e["event_seq"] for e in got) - 1
    replay_events = broker.replay_after(after_seq)
    assert replay_events
    assert [e["event_seq"] for e in replay_events] == list(
        range(after_seq + 1, broker.last_seq + 1)
    )
    assert replay_events[-1]["status"] in ("completed", "failed", "cancelled")


# ── Retention is bounded: count + TTL ──


def test_run_manager_retention_bounded_by_count(tmp_path):
    from auto_tune.modules.run_state.manual_controller import ManualRunController

    manager = RunManager(retain_max=3, retain_ttl=3600)
    for i in range(6):
        rs = new_run_state("manual", run_name=f"t{i}")
        broker = EventBroker(rs.run_id)
        controller = ManualRunController(
            run_state=rs, state_file=str(tmp_path / f"s{i}.json"), cmd=[],
            params={}, train_name=f"t{i}", train_dir=str(tmp_path),
            data_yaml=str(tmp_path / "data.yaml"), model="yolov8n.pt", epochs=1,
            log_path=str(tmp_path / f"t{i}.log"), finalize_cb=None,
            broker=broker, manager=manager,
        )
        manager.retain(controller.run_id, controller)
    assert manager.retained_count() == 3


def test_run_manager_retention_ttl_expires(tmp_path):
    from auto_tune.modules.run_state.manual_controller import ManualRunController

    manager = RunManager(retain_max=10, retain_ttl=0.05)
    rs = new_run_state("manual", run_name="ttl1")
    broker = EventBroker(rs.run_id)
    controller = ManualRunController(
        run_state=rs, state_file=str(tmp_path / "s.json"), cmd=[],
        params={}, train_name="ttl1", train_dir=str(tmp_path),
        data_yaml=str(tmp_path / "data.yaml"), model="yolov8n.pt", epochs=1,
        log_path=str(tmp_path / "ttl1.log"), finalize_cb=None,
        broker=broker, manager=manager,
    )
    manager.retain(controller.run_id, controller)
    assert manager.get(controller.run_id) is not None
    time.sleep(0.1)
    # expired retention is no longer returned and is dropped
    assert manager.get(controller.run_id) is None
    assert manager.retained_count() == 0
    # prune() also evicts expired entries without new traffic
    manager.retain(controller.run_id, controller)
    time.sleep(0.1)
    manager.prune()
    assert manager.retained_count() == 0


def test_event_broker_replay_truncated_signal():
    broker = EventBroker("manual:test", max_events=3)
    for i in range(6):
        broker.publish({"message": f"m{i}"})
    # buffer holds only seqs 4,5,6; events before that are gone
    assert broker.replay_truncated(after_seq=0)
    assert broker.replay_truncated(after_seq=2)
    assert not broker.replay_truncated(after_seq=3)
    assert not broker.replay_truncated(after_seq=6)


_JS_TERMINAL_STATUSES = frozenset(
    {"done", "error", "cancelled", "completed", "failed", "interrupted"}
)


def _js_is_terminal(data):
    """Mirror of single_page.html ``_isTerminal(data)``."""
    return data.get("status") in _JS_TERMINAL_STATUSES


def test_replay_truncated_warning_is_not_terminal(tmp_path, monkeypatch):
    """After an active run's ring buffer has rolled, an after_seq reconnect
    emits a replay_truncated transport warning that must NOT look like a
    terminal event — the run is still going."""
    import auto_tune.modules.run_state.manual_controller as mc
    from auto_tune.ui import app as app_mod

    run_state = new_run_state("manual", run_name="roll1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id, max_events=3)  # tiny buffer → rolls fast
    manager = RunManager()
    stdout = ControllableStdout()

    class FakeProc:
        def __init__(self):
            self.stdout = stdout
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)

    controller = mc.ManualRunController(
        run_state=run_state, state_file=state_file,
        cmd=["yolo", "train"], params={"batch": 16, "imgsz": 640},
        train_name="roll1", train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt", epochs=2,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None, broker=broker, manager=manager,
    )
    manager.register(controller)

    async def scenario():
        task = asyncio.create_task(controller._run())
        loop = asyncio.get_running_loop()
        # wait until the controller has published the 3 banners + lifecycle
        deadline = loop.time() + 5
        while broker.last_seq < 4 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert broker.last_seq >= 4
        # one more line → seq 5; with max_events=3 the buffer now holds 3,4,5
        stdout.feed(b"  1/2  1.20G  1.234  0.456  0.789\n")
        deadline = loop.time() + 5
        while broker.last_seq < 5 and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert broker.last_seq >= 5
        assert broker.replay_truncated(after_seq=1)  # seq 2 was evicted

        # reconnect with after_seq=1 while the run is still active
        gen = app_mod._run_sse(broker, controller, after_seq=1)
        try:
            first = await gen.__anext__()
        finally:
            await gen.aclose()

        stdout.finish()  # clean up the still-running controller
        await asyncio.wait_for(task, timeout=10)
        return first

    first = asyncio.run(scenario())
    events = _sse_events(first)
    assert events
    data = events[0]
    assert data.get("event") == "replay_truncated"
    assert data.get("replay_truncated") is True
    # transport control message: no event_seq, exempt from the run seq series
    assert "event_seq" not in data
    # must NOT look like a terminal event (active run is still going)
    assert data.get("status") not in _JS_TERMINAL_STATUSES
    assert data.get("phase") != "terminal"
    assert _js_is_terminal(data) is False


# ── HTTP: reconnect after completion replays real events; synthetic terminal is honest ──


def test_http_reconnect_replays_real_events_after_completion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod
    import auto_tune.modules.run_state.manual_controller as mc

    _redirect_log(monkeypatch, tmp_path)

    class FixedStdout:
        def __init__(self):
            self._lines = [
                b"  1/2  1.20G  1.234  0.456  0.789\n",
                b"all 10 50 0.123 0.456 0 0.111\n",
            ]
            self._i = 0

        async def readline(self):
            if self._i >= len(self._lines):
                return b""
            line = self._lines[self._i]
            self._i += 1
            return line

    class FakeProc:
        def __init__(self):
            self.stdout = FixedStdout()
            self.pid = os.getpid()
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(mc.asyncio, "create_subprocess_exec", fake_subprocess_exec)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo")
    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run",
                        lambda *a, **k: {"run_id": "manual:x", "run_name": "x",
                                         "source": "manual", "status": "completed",
                                         "analysis_status": "skipped", "metrics": {},
                                         "artifacts": {"report_path": None}, "error": None,
                                         "analysis_error": None, "history_error": None})

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt", "epochs": 1,
        })
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        stamped = [e for e in events if "event_seq" in e]
        assert stamped
        run_id = stamped[0]["run_id"]
        seqs = [e["event_seq"] for e in stamped]
        last_seq = max(seqs)

        # Reconnect after completion, missing the last two events: the retained
        # controller must replay those real events — never a synthetic +1.
        resp2 = client.get(f"/api/runs/{run_id}/stream?after_seq={last_seq - 2}")
        assert resp2.status_code == 200
        events2 = _sse_events(resp2.text)
        stamped2 = [e for e in events2 if "event_seq" in e]
        assert stamped2, "expected replay of missed events after completion"
        seqs2 = [e["event_seq"] for e in stamped2]
        assert seqs2 == [last_seq - 1, last_seq], seqs2
        assert max(seqs2) == last_seq  # no synthetic last_seq + 1
    finally:
        app_mod._running_training.clear()


def test_run_stream_synthetic_terminal_flags_replay_truncated(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _redirect_log(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status="cancelled", phase="terminal")

    try:
        client = TestClient(app_mod.app)
        resp = client.get(f"/api/runs/{state.run_id}/stream")
        assert resp.status_code == 200
        events = _sse_events(resp.text)
        assert events
        terminal = events[-1]
        assert terminal.get("status") == "cancelled"
        assert terminal.get("run_id") == state.run_id
        # No broker available → the persisted terminal honestly says replay
        # is incomplete instead of pretending the full stream was delivered.
        assert terminal.get("replay_truncated") is True
    finally:
        app_mod._running_training.clear()
