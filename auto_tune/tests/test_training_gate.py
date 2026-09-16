"""H1.3 Task 1: unified training-slot reservation + persisted restart gate.

Unit-level tests only — no YOLO subprocess, no network LLM, no browser. A
controlled local process is used for one identity check; everything else is
deterministic in-memory / tmp_path state.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from auto_tune.modules.run_state import gate as gate_mod
from auto_tune.modules.run_state.gate import (
    SlotDecision,
    evaluate_persisted_run_state,
    hpo_storage_has_live_process,
)
from auto_tune.modules.run_state.manager import (
    RESERVABLE_KINDS,
    RunManager,
    TrainingBusyError,
    TrainingGateError,
)
from auto_tune.modules.run_state.process_identity import capture_process_identity
from auto_tune.modules.run_state.service import (
    new_run_state,
    with_status_phase,
)


class _StubController:
    run_kind = "tuning"

    def __init__(self, run_id):
        self.run_id = run_id
        self._done = False

    def is_active(self):
        return not self._done


def _capture_live_pid() -> int:
    """Return the PID of a live child process (still running during the test)."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    return proc.pid


def _pid_gone() -> int:
    """Return a PID guaranteed to be gone."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    return proc.pid


# ── RunManager reservation: lifecycle ──────────────────────────────


def test_reserve_release_lifecycle():
    manager = RunManager()
    assert RESERVABLE_KINDS == frozenset({"manual", "tuning", "hpo"})
    token = manager.reserve("manual", "manual:abc")
    assert token and isinstance(token, str)
    assert manager.reservation_owner() == ("manual", "manual:abc")
    assert manager.release(token) is True
    assert manager.reservation_owner() is None
    # releasing again is a no-op, not an error
    assert manager.release(token) is False


@pytest.mark.parametrize("kind", ["manual", "tuning", "hpo"])
def test_reserve_each_kind_sequential(kind):
    manager = RunManager()
    token = manager.reserve(kind, f"{kind}:x")
    assert manager.reservation_owner() == (kind, f"{kind}:x")
    manager.release(token)


def test_second_reserve_is_busy():
    manager = RunManager()
    manager.reserve("tuning", "tuning:1")
    with pytest.raises(TrainingBusyError):
        manager.reserve("hpo", "hpo_x")
    with pytest.raises(TrainingBusyError):
        manager.reserve("manual", "manual:2")


def test_foreign_token_cannot_release():
    manager = RunManager()
    token = manager.reserve("manual", "manual:1")
    with pytest.raises(TrainingGateError):
        manager.release("other-token")
    assert manager.reservation_owner() == ("manual", "manual:1")
    assert manager.release(token) is True


def test_unknown_kind_rejected_for_reserve_and_lookup():
    manager = RunManager()
    with pytest.raises(TrainingGateError):
        manager.reserve("classify", "classify:1")
    with pytest.raises(TrainingGateError):
        manager.active_for_kind("bogus")  # must NOT fall back to active_tuning


def test_active_controller_blocks_reserve_any_kind():
    manager = RunManager()
    controller = _StubController("tuning:active")
    manager.register(controller)
    try:
        with pytest.raises(TrainingBusyError):
            manager.reserve("manual", "manual:x")
        with pytest.raises(TrainingBusyError):
            manager.reserve("hpo", "hpo_x")
        assert manager.active_train() is controller
    finally:
        manager.unregister(controller.run_id)


def test_retained_done_controller_does_not_block_reserve():
    manager = RunManager()
    controller = _StubController("tuning:done")
    controller._done = True
    manager.register(controller)
    try:
        token = manager.reserve("manual", "manual:x")
        assert manager.release(token) is True
    finally:
        manager.unregister(controller.run_id)


# ── barrier: simultaneous start/resume/LLM only one acquires ──────


def test_barrier_concurrent_reserve_only_one_wins():
    manager = RunManager()
    n = 8
    outcomes = []
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        try:
            token = manager.reserve("manual" if i % 2 else "tuning",
                                    f"kind{i}:{i}")
            outcomes.append(("ok", i))
        except TrainingBusyError:
            outcomes.append(("busy", i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    ok = [o for o in outcomes if o[0] == "ok"]
    busy = [o for o in outcomes if o[0] == "busy"]
    assert len(ok) == 1
    assert len(busy) == n - 1


# ── persisted restart gate ─────────────────────────────────────────


def test_persisted_none_or_terminal_allows():
    assert evaluate_persisted_run_state(None).blocked is False
    state = new_run_state("manual", run_name="train1")
    terminal = with_status_phase(
        state, status="completed", phase="terminal", terminal_reason="completed")
    decision = evaluate_persisted_run_state(terminal)
    assert decision.blocked is False
    assert decision.persist is None


def test_running_without_identity_blocks():
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(state, status="running", phase="training")
    decision = evaluate_persisted_run_state(state)
    assert decision.blocked is True
    assert decision.persist is None


def test_running_match_blocks():
    pid = _capture_live_pid()
    try:
        identity = capture_process_identity(pid)
        if identity is None:
            pytest.skip("platform cannot capture identity for the live child")
        state = new_run_state("tuning")
        state = with_status_phase(
            state, status="running", phase="training",
            pid=pid, process_create_token=identity.process_create_token)
        decision = evaluate_persisted_run_state(state)
        assert decision.blocked is True
        assert decision.persist is None
    finally:
        try:
            os.kill(pid, 0)
        except OSError:
            pass


def test_running_missing_allows_and_reconciles():
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=_pid_gone(), process_create_token="token")
    decision = evaluate_persisted_run_state(state)
    # The child is already gone → MISSING → not blocked, terminal reconciled.
    assert decision.blocked is False
    assert decision.persist is not None
    assert decision.persist.status == "interrupted"
    assert decision.persist.terminal_reason == "process_missing"


def test_running_mismatch_allows_and_reconciles():
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=_pid_gone(), process_create_token="token")
    decision = evaluate_persisted_run_state(state)
    assert decision.blocked is False
    assert decision.persist is not None


def test_unverifiable_blocks(monkeypatch):
    from auto_tune.modules.run_state.process_identity import IdentityMatch

    monkeypatch.setattr(gate_mod, "compare_process_identity",
                        lambda expected: IdentityMatch.UNVERIFIABLE)
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training", pid=os.getpid(),
        process_create_token="windows-filetime:x")
    decision = evaluate_persisted_run_state(state)
    assert decision.blocked is True
    assert decision.persist is None


def test_foreign_pid_is_never_killed(monkeypatch):
    """MISMATCH must only reconcile the record, never signal a kill."""
    from auto_tune.modules.run_state.process_identity import IdentityMatch

    monkeypatch.setattr(gate_mod, "compare_process_identity",
                        lambda expected: IdentityMatch.MISMATCH)
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=12345, process_create_token="old-token")
    decision = evaluate_persisted_run_state(state)
    assert decision.blocked is False
    assert decision.persist is not None
    assert decision.persist.status == "interrupted"


# ── one identity observation per gate decision (stability) ─────────


def _counting_identity(monkeypatch, match_name):
    """Patch every identity reader with a call-counting fake."""
    from auto_tune.modules.run_state import process_identity as identity_mod
    from auto_tune.modules.run_state.process_identity import IdentityMatch

    calls = []

    def fake(expected):
        calls.append(expected)
        return IdentityMatch[match_name]

    monkeypatch.setattr(gate_mod, "compare_process_identity", fake)
    monkeypatch.setattr(identity_mod, "compare_process_identity", fake)
    return calls


@pytest.mark.parametrize("match_name,expected_reason", [
    ("MISSING", "process_missing"),
    ("MISMATCH", "pid_reused"),
])
def test_terminal_reason_uses_one_identity_observation(
        monkeypatch, match_name, expected_reason):
    calls = _counting_identity(monkeypatch, match_name)
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=12345, process_create_token="token")
    decision = evaluate_persisted_run_state(state)
    # Exactly one read: the gate decision and the reconciled terminal reason
    # must come from the same observation (a PID reused between two reads used
    # to flip MISSING into MISMATCH).
    assert len(calls) == 1
    assert decision.blocked is False
    assert decision.persist is not None
    assert decision.persist.status == "interrupted"
    assert decision.persist.terminal_reason == expected_reason


@pytest.mark.parametrize("match_name", ["MATCH", "UNVERIFIABLE"])
def test_conservative_block_uses_one_identity_observation(monkeypatch, match_name):
    calls = _counting_identity(monkeypatch, match_name)
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=os.getpid(), process_create_token="token")
    decision = evaluate_persisted_run_state(state)
    assert len(calls) == 1
    assert decision.blocked is True
    assert decision.persist is None


# ── HPO persisted live-process scan ────────────────────────────────


def _write_execution(root: Path, study: str, payload: dict) -> Path:
    study_dir = root / study
    study_dir.mkdir(parents=True, exist_ok=True)
    target = study_dir / "execution.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _execution_record(status: str, attempt=None) -> dict:
    return {
        "schema_version": "hpo-execution-v1",
        "study_id": "hpo_x",
        "status": status,
        "attempts": [attempt] if attempt is not None else [],
    }


def test_hpo_scan_empty_root_free(tmp_path):
    assert hpo_storage_has_live_process(tmp_path / "nope") is False
    assert hpo_storage_has_live_process(tmp_path) is False


def test_hpo_scan_terminal_free(tmp_path):
    _write_execution(tmp_path, "hpo_a", _execution_record("COMPLETED"))
    _write_execution(tmp_path, "hpo_b", _execution_record("PAUSED"))
    assert hpo_storage_has_live_process(tmp_path) is False


def test_hpo_scan_running_no_pid_blocks(tmp_path):
    attempt = {"phase": "RUNNING", "pid": None}
    _write_execution(tmp_path, "hpo_a",
                     _execution_record("RUNNING", attempt))
    assert hpo_storage_has_live_process(tmp_path) is True


def test_hpo_scan_running_corrupt_blocks(tmp_path):
    target = tmp_path / "hpo_a"
    target.mkdir()
    (target / "execution.json").write_text("{not json", encoding="utf-8")
    assert hpo_storage_has_live_process(tmp_path) is True


def test_hpo_scan_running_gone_pid_free(tmp_path):
    pid = _pid_gone()
    attempt = {"phase": "RUNNING", "pid": pid,
               "process_create_token": "token"}
    _write_execution(tmp_path, "hpo_a",
                     _execution_record("RUNNING", attempt))
    # gone PID → MISSING → not blocking
    assert hpo_storage_has_live_process(tmp_path) is False


# ── HTTP-level cross-kind slot gate (via the app endpoints) ───────


class _StubTrainController:
    """Minimal controller stub that looks active without any subprocess."""

    def __init__(self, kind, run_id):
        self.run_kind = kind
        self.run_id = run_id

    def is_active(self):
        return True

    def is_done(self):
        return False


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


def _post_ordinary_start(client, payload=None):
    return client.post("/api/training/start", json=payload or {})


@pytest.mark.parametrize("kind", ["tuning", "hpo"])
def test_ordinary_start_blocked_by_other_active_kind(tmp_path, monkeypatch, kind):
    """A live tuning or HPO run must block an ordinary training start (409)."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    controller = _StubTrainController(kind, f"{kind}:stub")
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        resp = _post_ordinary_start(client)
        assert resp.status_code == 409
        assert "RUN_ALREADY_ACTIVE" in resp.text
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_manual_start_releases_slot_on_persist_failure(tmp_path, monkeypatch):
    """Initialization failure after reserving must release the slot (no leak)."""
    from fastapi.testclient import TestClient
    from auto_tune.modules.run_state.models import RunStatePersistenceError
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
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
            "data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt",
            "epochs": 1,
        })
        assert resp.status_code == 500
        assert "RUN_STATE_PERSIST_FAILED" in resp.text
        assert app_mod._RUN_MANAGER.reservation_owner() is None
        assert app_mod._RUN_MANAGER.active_train() is None
    finally:
        app_mod._running_training.clear()


# ── reserve-before-create ordering for ordinary training (rework #3) ──


class _BlockingProc:
    """Subprocess stand-in that stays alive until the test releases it."""

    def __init__(self, release: threading.Event):
        self.pid = os.getpid()
        self.returncode = 0
        self._release = release

        class _Stdout:
            async def readline(_self):
                await asyncio.to_thread(release.wait, 15)
                return b""

        self.stdout = _Stdout()

    async def wait(self):
        await asyncio.to_thread(self._release.wait, 15)
        return self.returncode


class _ImmediateProc(_BlockingProc):
    def __init__(self):
        event = threading.Event()
        event.set()
        super().__init__(event)


def _patch_training_runtime(monkeypatch, detect_dir, launched, release=None):
    """Fake the YOLO executable/subprocess/finalizer for one training start."""
    from auto_tune.ui import app as app_mod

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(detect_dir))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo")

    async def fake_exec(*args, **kwargs):
        launched.append(list(args))
        return _BlockingProc(release)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    def fake_finalize(run_dir, run_name, source_kind, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name,
            "source": "manual", "status": "completed", "analysis_status": "skipped",
            "metrics": {}, "artifacts": {"report_path": None}, "error": None,
            "analysis_error": None, "history_error": None,
        }

    monkeypatch.setattr(app_mod, "finalize_training_run", fake_finalize)
    app_mod._running_training.clear()
    return app_mod


def _training_payload(tmp_path):
    return {"data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt",
            "epochs": 1}


@pytest.mark.parametrize("kind", ["tuning", "hpo"])
def test_ordinary_start_busy_leaves_no_directory_trace(tmp_path, monkeypatch, kind):
    """A busy slot must not create a trainN dir, args.yaml or a process."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    (detect / "train1").mkdir()
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(detect))
    controller = _StubTrainController(kind, f"{kind}:stub")
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json=_training_payload(tmp_path))
        assert resp.status_code == 409
        assert sorted(p.name for p in detect.iterdir()) == ["train1"]
        assert app_mod._RUN_MANAGER.reservation_owner() is None
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_ordinary_start_blocked_by_bare_reservation(tmp_path, monkeypatch):
    """reserve→register gap: an unregistered reservation still owns the slot."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(detect))
    token = app_mod._RUN_MANAGER.reserve("hpo", "hpo_reserved")
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json=_training_payload(tmp_path))
        assert resp.status_code == 409
        assert list(detect.iterdir()) == []
        assert app_mod._RUN_MANAGER.reservation_owner() == ("hpo", "hpo_reserved")
    finally:
        app_mod._RUN_MANAGER.release(token)


def test_concurrent_ordinary_starts_create_one_run(tmp_path, monkeypatch):
    """Two simultaneous ordinary starts: only one reserves, creates and runs."""
    from fastapi.testclient import TestClient

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    launched = []
    release = threading.Event()
    app_mod = _patch_training_runtime(monkeypatch, detect, launched, release)
    barrier = threading.Barrier(2)
    codes = []
    lock = threading.Lock()

    def worker():
        client = TestClient(app_mod.app)
        payload = _training_payload(tmp_path)
        barrier.wait(timeout=10)
        code = client.post("/api/training/start", json=payload).status_code
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 10
    while time.time() < deadline and not launched:
        time.sleep(0.02)
    assert len(launched) == 1
    release.set()
    for t in threads:
        t.join(timeout=20)
    try:
        assert sorted(codes) == [200, 409]
        assert sorted(p.name for p in detect.iterdir()) == ["train1"]
        assert app_mod._RUN_MANAGER.reservation_owner() is None
    finally:
        app_mod._running_training.clear()


def test_existing_train_path_is_never_overwritten(tmp_path, monkeypatch):
    """A taken name is skipped; an existing entry is never clobbered."""
    from fastapi.testclient import TestClient

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    (detect / "train1").write_text("legacy placeholder", encoding="utf-8")
    launched = []
    app_mod = _patch_training_runtime(monkeypatch, detect, launched)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json=_training_payload(tmp_path))
        assert resp.status_code == 200
        assert len(launched) == 1
        assert (detect / "train1").read_text(encoding="utf-8") == "legacy placeholder"
        assert (detect / "train2" / "args.yaml").is_file()
    finally:
        app_mod._running_training.clear()


def test_manual_start_releases_slot_on_dir_creation_failure(tmp_path, monkeypatch):
    """Directory failure after reserving: slot released, nothing started."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    launched = []
    _patch_training_runtime(monkeypatch, detect, launched)

    def boom(detect_dir, train_dir):
        raise OSError("disk full")

    monkeypatch.setattr(app_mod, "_create_train_dirs", boom)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json=_training_payload(tmp_path))
        assert resp.status_code == 500
        assert resp.json()["error_code"] == "TRAIN_DIR_CREATE_FAILED"
        assert app_mod._RUN_MANAGER.reservation_owner() is None
        assert app_mod._RUN_MANAGER.active_train() is None
        assert launched == []
        assert not (tmp_path / "log" / "training_running.json").exists()
    finally:
        app_mod._running_training.clear()


# ── reconcile write-back failure must block new work ───────────────
#
# MISSING/MISMATCH may only be released after the orphaned record has been
# *successfully* reconciled to a terminal state. If that write-back fails the
# old run may still be alive, so nothing may start — the failure must never be
# swallowed into "free slot".

_ORPHAN_FILES = {"manual": "training_running.json",
                 "tuning": "tuning_running.json"}


def _write_orphan_state(tmp_path, run_kind):
    """Persist a `running` record whose process identity will read as gone."""
    from auto_tune.modules.run_state.service import write_run_state as _svc_write

    state = new_run_state(run_kind, run_name="train1")
    state = with_status_phase(
        state, status="running", phase="training",
        pid=12345, process_create_token="stale-token")
    path = tmp_path / "log" / _ORPHAN_FILES[run_kind]
    _svc_write(str(path), state)


def _fail_reconcile_write_back(monkeypatch):
    """Fail only the reconciled terminal write-back, keep normal writes working."""
    from auto_tune.modules.run_state.models import RunStatePersistenceError
    from auto_tune.ui import app as app_mod

    real = app_mod._persist_run_state

    def flaky(state_file, state):
        if state.status == "interrupted":
            raise RunStatePersistenceError(
                r"disk full writing C:\secret\logs\training_running.json")
        real(state_file, state)

    monkeypatch.setattr(app_mod, "_persist_run_state", flaky)


def _stub_reference_dataset():
    """Minimal resolution stub for the tuning pre-flight checks."""
    from types import SimpleNamespace

    return SimpleNamespace(
        snapshot_id="a" * 64, reference_run="train1",
        dataset_display_name="stub-dataset", resolution_source="sqlite")


def _cleanup_runs(app_mod):
    """Test-only safety net: leave no controller or reservation behind."""
    for controller in app_mod._RUN_MANAGER.snapshot():
        app_mod._RUN_MANAGER.unregister(controller.run_id)
    app_mod._RUN_MANAGER._reservation = None
    app_mod._running_training.clear()


def _tuning_payload(mode, reference_run="train1"):
    return {"reference_run": reference_run, "mode": mode, "max_retries": 1,
            "auto_analyze": False, "auto_loop": False}


@pytest.mark.parametrize("run_kind", ["manual", "tuning"])
@pytest.mark.parametrize("match_name", ["MISSING", "MISMATCH"])
def test_reconcile_write_back_failure_blocks_ordinary_training(
        tmp_path, monkeypatch, run_kind, match_name):
    """No reservation, no trainN, no args.yaml, no process when the old
    record cannot be proven converged."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    detect = tmp_path / "detect"
    detect.mkdir()
    (detect / "train1").mkdir()
    launched = []
    _patch_training_runtime(monkeypatch, detect, launched)
    _counting_identity(monkeypatch, match_name)
    _write_orphan_state(tmp_path, run_kind)
    _fail_reconcile_write_back(monkeypatch)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json=_training_payload(tmp_path))
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_code"] == "RUN_STATE_RECONCILE_FAILED"
        assert set(body) == {"error", "error_code", "next_action"}
        assert "disk full" not in resp.text
        assert "secret" not in resp.text
        assert launched == []
        assert app_mod._RUN_MANAGER.reservation_owner() is None
        assert app_mod._RUN_MANAGER.active_train() is None
        assert sorted(p.name for p in detect.iterdir()) == ["train1"]
        assert not (detect / "train1" / "args.yaml").exists()
    finally:
        _cleanup_runs(app_mod)


@pytest.mark.parametrize("run_kind", ["manual", "tuning"])
def test_reconcile_write_back_failure_blocks_llm_tuning_start(
        tmp_path, monkeypatch, run_kind):
    """The real LLM tuning entry (mode=full) is blocked too."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "find_detect_dir",
                        lambda *a, **k: str(tmp_path / "detect"))
    monkeypatch.setattr(app_mod, "resolve_reference_dataset",
                        lambda *a, **k: _stub_reference_dataset())

    def boom(*args, **kwargs):
        raise AssertionError("the tuning loop must not be reached")

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.run_tuning_loop", boom)
    _counting_identity(monkeypatch, "MISSING")
    _write_orphan_state(tmp_path, run_kind)
    _fail_reconcile_write_back(monkeypatch)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_tuning_payload("full"))
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "RUN_STATE_RECONCILE_FAILED"
        assert "disk full" not in resp.text
        assert app_mod._RUN_MANAGER.reservation_owner() is None
        assert app_mod._RUN_MANAGER.active_tuning() is None
    finally:
        _cleanup_runs(app_mod)


@pytest.mark.parametrize("run_kind", ["manual", "tuning"])
@pytest.mark.parametrize("match_name", ["MISSING", "MISMATCH"])
def test_reconcile_write_back_success_still_releases_the_run(
        tmp_path, monkeypatch, run_kind, match_name):
    """A converged record must still allow new work (no permanent lock-out)."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "find_detect_dir",
                        lambda *a, **k: str(tmp_path / "detect"))
    monkeypatch.setattr(app_mod, "resolve_reference_dataset",
                        lambda *a, **k: _stub_reference_dataset())

    captured = {}
    written = []

    real_persist = app_mod._persist_run_state

    def spy_persist(state_file, state):
        written.append(state)
        real_persist(state_file, state)

    monkeypatch.setattr(app_mod, "_persist_run_state", spy_persist)

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.run_tuning_loop", fake_loop)
    _counting_identity(monkeypatch, match_name)
    _write_orphan_state(tmp_path, run_kind)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_tuning_payload("full"))
        assert resp.status_code == 200
        assert "RUN_STATE_RECONCILE_FAILED" not in resp.text
        assert captured, "the tuning loop must run once the slot is free"
        expected_reason = "process_missing" if match_name == "MISSING" else "pid_reused"
        assert any(s.status == "interrupted" and s.terminal_reason == expected_reason
                   for s in written), "the orphaned record must be reconciled first"
    finally:
        _cleanup_runs(app_mod)


def test_reconcile_write_back_failure_does_not_change_dry_run(tmp_path, monkeypatch):
    """dry_run keeps its existing semantics (it never holds a real slot)."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "find_detect_dir",
                        lambda *a, **k: str(tmp_path / "detect"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.run_tuning_loop",
        lambda **kwargs: {})
    _counting_identity(monkeypatch, "MISSING")
    _write_orphan_state(tmp_path, "manual")
    _fail_reconcile_write_back(monkeypatch)
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json=_tuning_payload("dry_run", None))
        assert resp.status_code == 200
        assert "RUN_STATE_RECONCILE_FAILED" not in resp.text
        assert app_mod._RUN_MANAGER.reservation_owner() is None
    finally:
        _cleanup_runs(app_mod)
