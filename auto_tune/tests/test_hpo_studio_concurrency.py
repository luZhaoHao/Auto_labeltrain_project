"""H1.3 third-round repair: Studio HPO read/worker contention + error durability.

The reported release blocker was: a real Studio start returned 202, then the
2s status/list polling raced the worker's short storage transactions. Because
every HpoService/ExecutionStore transaction acquires a **non-blocking** lock,
a concurrent read made the worker itself raise ``HPO_STUDY_BUSY`` /
``HPO_EXECUTION_BUSY``; the worker died, the controller unregistered, and the
error vanished — leaving the page on ``READY 0/2`` with no visible failure.

These counterexamples use the **real** StudyStore/ExecutionStore/HpoService/
HpoRunner over ``tmp_path`` (so the real lock files and real recovery matrix are
exercised) with a controlled fake training adapter: no YOLO, no network, no
browser. Contention is made deterministic with events/barriers instead of
sleeps.
"""

import contextlib
import json
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    ExecutionConfig,
    HpoError,
    HpoRunner,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.hpo.execution_adapter import CollectedOutcome
from auto_tune.modules.hpo.execution_models import MetricDiagnostics
from auto_tune.modules.model_store import ModelStore
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.ui import hpo_controller as controller_mod
from auto_tune.ui.hpo_api import create_hpo_router

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"

# A short, deterministic backoff so retries are exercised without wall-clock
# waiting. The production constants are exercised by the real acceptance run;
# these tests coordinate with events and only need the retry to be *possible*.
_FAST_BACKOFF = (0.01, 0.02, 0.05)
_FAST_WINDOW = 5.0


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    # Tolerate the pre-repair module so each counterexample fails for its own
    # behavioural reason instead of at fixture setup.
    if hasattr(controller_mod, "BUSY_RETRY_BACKOFF"):
        monkeypatch.setattr(controller_mod, "BUSY_RETRY_BACKOFF", _FAST_BACKOFF)
        monkeypatch.setattr(controller_mod, "BUSY_RETRY_WINDOW_SECONDS",
                            _FAST_WINDOW)


# ── controlled fake training adapter (never launches YOLO) ────────


class FakeProc:
    _seq = 7000

    def __init__(self, rc, pid=None):
        FakeProc._seq += 1
        self.pid = pid if pid is not None else FakeProc._seq
        self._rc = rc
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakeAdapter:
    """Same seam the H1.2 suite uses: real runner, fake subprocess boundary."""

    launch_count = 0
    returncodes = [0]
    collect_result = None
    finalize_result = None

    def __init__(self, output_root, log_root):
        self.output_root = Path(output_root)
        self.log_root = Path(log_root)

    def prepare(self, study, trial, config):
        from auto_tune.modules.agent_engine.executor import build_yolo_command
        run_relpath = f"{study.study_id}/{trial.trial_id}"
        fixed = {"task": "detect", "workers": 0, "resume": False,
                 "deterministic": True, "patience": 0, "val": True,
                 "save": True, "plots": False, "amp": False}
        effective = dict(fixed)
        effective["model"] = study.model_binding.model_path
        effective["data"] = study.snapshot_binding.data_yaml_path
        effective["epochs"] = study.config.epochs
        effective["seed"] = study.config.seed
        effective["batch"] = config.batch
        effective["imgsz"] = config.imgsz
        effective["device"] = config.device
        effective.update(trial.candidate_params)
        run_dir = Path(self.output_root) / run_relpath
        command = build_yolo_command(trial.trial_id,
                                     str(run_dir / "args.yaml"),
                                     dict(effective))
        return {
            "trial_number": trial.number,
            "trial_id": trial.trial_id,
            "candidate_params": dict(trial.candidate_params),
            "effective_params": effective,
            "command": command,
            "run_relpath": run_relpath,
            "args_sha256": "0" * 64,
        }

    def validate_launch(self, prepared):
        return None

    def launch(self, prepared):
        FakeAdapter.launch_count += 1
        rc = FakeAdapter.returncodes[
            FakeAdapter.launch_count - 1
            if FakeAdapter.launch_count <= len(FakeAdapter.returncodes)
            else -1]
        return FakeProc(rc)

    def collect(self, study, attempt):
        if FakeAdapter.collect_result is not None:
            return FakeAdapter.collect_result
        evidence = Evidence(
            run_id=attempt.run_id,
            artifact_relpath=f"{study.study_id}/"
                             f"{attempt.run_relpath.split('/')[-1]}/results.csv",
            artifact_sha256="0" * 64, epoch=1)
        return CollectedOutcome(
            result=ResultInput(state="SUCCESS", value=0.7, evidence=evidence),
            diagnostics=MetricDiagnostics(total_rows=1, excluded_rows=0))

    def finalize(self, study, attempt):
        return dict(FakeAdapter.finalize_result
                    if FakeAdapter.finalize_result is not None
                    else {"status": "completed"})

    def detect_oom(self, run_dir):
        return False


@pytest.fixture
def reset_fake():
    FakeAdapter.launch_count = 0
    FakeAdapter.returncodes = [0]
    FakeAdapter.collect_result = None
    FakeAdapter.finalize_result = None
    yield
    FakeAdapter.launch_count = 0
    FakeAdapter.collect_result = None
    FakeAdapter.finalize_result = None


# ── stack: real stores + real runner + real router ────────────────


class Stack:
    BUDGET = 2

    def __init__(self, tmp_path, runner=None):
        source = tmp_path / "source"
        source.mkdir()
        for n in range(4):
            Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
            (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                             encoding="utf-8")
        self.snapshot = create_dataset_snapshot(
            source, tmp_path / "snapshots", val_ratio=0.5, seed=42,
            class_names={0: "part"})
        self.model = tmp_path / "fixture.pt"
        self.model.write_bytes(b"hpo-concurrency-test-not-a-real-model")
        # 新建研究只接受受控模型标识（客户端不再提交任何路径）
        self.store = ModelStore(tmp_path / "models" / "weights",
                                legacy_roots=[tmp_path],
                                max_upload_bytes=1024, max_models=50)
        self.model_id = {row.name: row.model_id
                         for row in self.store.list_models()}["fixture.pt"]
        self.root = tmp_path / "hpo"
        self.service = HpoService(self.root)
        self.real_runner = HpoRunner(self.root, tmp_path / "out", tmp_path / "log")
        self.runner = runner if runner is not None else self.real_runner
        self.manager = RunManager()
        self.client = self._client()

    def _client(self):
        snapshot, model = self.snapshot, self.model

        def resolve_snapshot(sid):
            if sid == snapshot.snapshot_id:
                return Path(snapshot.snapshot_path)
            raise HpoError("HPO_INVALID_CONFIG", "bad snapshot")

        def resolve_model(model_id):
            return self.store.resolve(model_id)

        router = create_hpo_router(
            service=self.service, runner=self.runner, manager=self.manager,
            resolve_snapshot=resolve_snapshot, resolve_model=resolve_model,
            assert_training_slot_free=lambda: None)
        app = FastAPI()
        app.include_router(router, prefix="/api/hpo")
        return TestClient(app)

    def create(self):
        resp = self.client.post("/api/hpo/studies", json={
            "snapshot_id": self.snapshot.snapshot_id,
            "model_id": self.model_id,
            "study_config": {"budget": self.BUDGET, "epochs": 1},
            "execution_config": {"batch": 1, "imgsz": 64, "device": "cpu",
                                 "timeout_seconds": 120},
        })
        assert resp.status_code == 201, resp.text
        return resp.json()["study_id"]

    def status(self, study_id, **params):
        return self.client.get(f"/api/hpo/studies/{study_id}", params=params or None)


@pytest.fixture
def stack(tmp_path, reset_fake, monkeypatch):
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    return Stack(tmp_path)


def _wait(pred, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


_BUSY_CODES = ("HPO_STUDY_BUSY", "HPO_EXECUTION_BUSY")


def _read_status(runner, study_id, timeout=30.0):
    """Read the execution record, retrying only short transaction conflicts.

    The test acts as an ordinary Studio client here: a transient conflict means
    "ask again shortly", never "the training broke".
    """
    deadline = time.time() + timeout
    while True:
        try:
            return runner.status(study_id)
        except HpoError as exc:
            if exc.code not in _BUSY_CODES or time.time() >= deadline:
                raise
            time.sleep(0.01)


def _wait_status(runner, study_id, status, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _read_status(runner, study_id).status == status:
            return True
        time.sleep(0.02)
    return _read_status(runner, study_id).status == status


def _status_until(client, study_id, pred, timeout=20.0):
    last = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        last = client.get(f"/api/hpo/studies/{study_id}").json()
        if pred(last):
            return last
        time.sleep(0.02)
    return last


# ── 1. a reader holding a short transaction must not kill the worker ──


def test_reader_lock_contention_does_not_kill_background_worker(
        stack, tmp_path, monkeypatch):
    """A read holding a short study transaction must not defeat the worker.

    The mechanism is deliberately not asserted (yield or bounded retry both
    satisfy it); what must hold is that the worker survives the held read, still
    exhausts the budget, loses no trial slot, and leaves no error behind.
    """
    # Small positive backoff: the coordination below is the event pair, not the
    # delay, but the worker must not spin while it waits for the reader.
    monkeypatch.setattr(controller_mod, "BUSY_RETRY_BACKOFF", (0.01, 0.02, 0.05))
    study_id = stack.create()

    from auto_tune.modules.hpo import storage as storage_mod
    original_read = storage_mod.StudyStore.read
    original_locked = storage_mod.StudyStore.locked

    reader_inside = threading.Event()
    release_reader = threading.Event()
    gate_armed = threading.Event()
    arm_reader = threading.Event()

    def _is_worker():
        return threading.current_thread().name.startswith("hpo-runner-")

    @contextlib.contextmanager
    def gated_locked(self, sid):
        # Hold the worker just before its first study transaction so the reader
        # is guaranteed to own the lock first: a deterministic barrier, not a
        # race won by timing.
        if not gate_armed.is_set() and _is_worker():
            gate_armed.set()
            assert reader_inside.wait(15), "reader never entered the study lock"
        with original_locked(self, sid):
            yield

    def blocking_read(self, sid):
        # Only the first read of the polling round stalls, and only for the
        # read-only path; the worker proceeds once the reader releases.
        if arm_reader.is_set() and not _is_worker():
            arm_reader.clear()
            reader_inside.set()
            assert release_reader.wait(15), "test did not release the reader"
        return original_read(self, sid)

    monkeypatch.setattr(storage_mod.StudyStore, "read", blocking_read)
    monkeypatch.setattr(storage_mod.StudyStore, "locked", gated_locked)

    reader_results = {}

    def reader():
        try:
            reader_results["status"] = stack.client.get(
                f"/api/hpo/studies/{study_id}")
            reader_results["list"] = stack.client.get("/api/hpo/studies")
        except Exception as exc:  # pragma: no cover - surfaced by assertion
            reader_results["error"] = repr(exc)

    start = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert start.status_code == 202, start.text
    controller = stack.manager.get(study_id)
    assert controller is not None

    arm_reader.set()
    reader_thread = threading.Thread(target=reader, name="studio-poll-reader")
    reader_thread.start()
    assert reader_inside.wait(10), "reader never entered the study transaction"

    # While the polling request owns the study transaction the worker must stay
    # alive and unerrored. The read is held open deliberately so an
    # implementation that treated the conflict as terminal would already have
    # died here.
    assert controller.is_active() is True
    assert controller.error_code is None
    assert FakeAdapter.launch_count == 0

    release_reader.set()
    reader_thread.join(15)
    assert "error" not in reader_results, reader_results
    assert reader_results["status"].status_code == 200
    assert reader_results["list"].status_code == 200

    assert _wait_status(stack.real_runner, study_id, "COMPLETED")
    record = _read_status(stack.real_runner, study_id)
    assert record.status == "COMPLETED"
    assert controller.error_code is None
    assert FakeAdapter.launch_count == Stack.BUDGET
    # each trial is launched exactly once (never a duplicate launch)
    numbers = [a.trial_number for a in record.attempts]
    assert numbers == [0, 1]
    assert all(a.phase == "FINALIZED" for a in record.attempts)

    body = stack.status(study_id).json()
    assert body["execution_status"] == "COMPLETED"
    assert body["error_code"] is None
    assert body["terminal_count"] == Stack.BUDGET
    # a read must not cost a trial slot either: the whole budget still succeeds
    assert body["success_count"] == Stack.BUDGET


# ── 2. bounded retry absorbs a finite pre-run busy ────────────────


class _CountingRunner:
    """Real runner facade that injects a finite number of busy failures."""

    def __init__(self, real, busy_count, code="HPO_STUDY_BUSY"):
        self.real = real
        self.busy_left = busy_count
        self.code = code
        self.run_calls = 0
        self.resume_calls = 0

    def _maybe_busy(self):
        if self.busy_left > 0:
            self.busy_left -= 1
            raise HpoError(self.code, "injected transient busy")

    def prepare(self, study_id, config):
        return self.real.prepare(study_id, config)

    def status(self, study_id):
        return self.real.status(study_id)

    def run(self, study_id, *, stop_event=None):
        self.run_calls += 1
        self._maybe_busy()
        return self.real.run(study_id, stop_event=stop_event)

    def resume(self, study_id, *, stop_event=None):
        self.resume_calls += 1
        self._maybe_busy()
        return self.real.resume(study_id, stop_event=stop_event)


def test_finite_busy_before_run_is_absorbed_and_run_called_once(tmp_path, stack):
    study_id = stack.create()
    counting = _CountingRunner(stack.real_runner, busy_count=3)
    stack.runner = counting
    stack.client = stack._client()

    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert resp.status_code == 202, resp.text
    controller = stack.manager.get(study_id)
    assert controller is not None

    assert _wait_status(stack.real_runner, study_id, "COMPLETED")
    assert controller.error_code is None
    # every injected failure was absorbed (an extra real collision with this
    # test's own polling is legitimate and also absorbed)
    assert controller.busy_retries >= 3
    # retries must never replay run(): they go through the idempotent resume
    assert counting.run_calls == 1
    assert counting.resume_calls == controller.busy_retries
    assert FakeAdapter.launch_count == Stack.BUDGET


def test_exhausted_busy_is_visible_not_silent_ready(tmp_path, stack, monkeypatch):
    monkeypatch.setattr(controller_mod, "BUSY_RETRY_WINDOW_SECONDS", 0.2)
    monkeypatch.setattr(controller_mod, "BUSY_RETRY_BACKOFF", (0.01, 0.02, 0.05))
    study_id = stack.create()
    counting = _CountingRunner(stack.real_runner, busy_count=99,
                               code="HPO_EXECUTION_BUSY")
    stack.runner = counting
    stack.client = stack._client()

    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert resp.status_code == 202, resp.text

    controller = stack.manager.get(study_id)
    assert controller is not None
    assert _wait(lambda: controller.is_done(), timeout=10)
    # the retry budget is bounded by time: it stops and reports, never hangs
    assert controller.busy_retries >= 1
    assert counting.run_calls == 1
    assert counting.resume_calls == controller.busy_retries

    body = _status_until(stack.client, study_id,
                         lambda b: b.get("error_code") is not None)
    assert body["error_code"] == "HPO_EXECUTION_BUSY"
    assert body["next_action"]
    # no training was launched, and the failure fact survives a refresh
    assert FakeAdapter.launch_count == 0
    assert stack.status(study_id).json()["error_code"] == "HPO_EXECUTION_BUSY"


# ── 3. side-effect phase boundaries never double-launch ───────────


def _inject_busy_on_phase(monkeypatch, phase, code="HPO_EXECUTION_BUSY"):
    """Make exactly one execution commit of ``phase`` fail with a busy error."""
    from auto_tune.modules.hpo import execution_storage as exec_storage
    original_write = exec_storage.ExecutionStore.write
    fired = threading.Event()

    def flaky_write(self, record):
        if not fired.is_set() and record.attempts \
                and record.attempts[-1].phase == phase:
            fired.set()
            raise HpoError(code, "injected busy at phase " + phase)
        return original_write(self, record)

    monkeypatch.setattr(exec_storage.ExecutionStore, "write", flaky_write)
    return fired


def test_busy_at_prepared_phase_launches_each_trial_once(stack, monkeypatch):
    study_id = stack.create()
    fired = _inject_busy_on_phase(monkeypatch, "PREPARED")

    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert resp.status_code == 202, resp.text
    assert _wait(fired.is_set)
    assert _wait_status(stack.real_runner, study_id, "COMPLETED")

    record = _read_status(stack.real_runner, study_id)
    # the interrupted claim is retried through the idempotent ask, so the first
    # trial is still launched exactly once and the budget is not exceeded
    assert FakeAdapter.launch_count == Stack.BUDGET
    assert [a.trial_number for a in record.attempts] == [0, 1]
    assert all(a.phase == "FINALIZED" for a in record.attempts)


@pytest.mark.parametrize("phase,expected_status,expected_launches", [
    # The intent commit failed, so the write-ahead fact proves no launch ever
    # happened: retrying through the idempotent ask must launch once per trial
    # and still finish the budget.
    ("LAUNCH_INTENT", "COMPLETED", 2),
    # The launch already happened and the commit that would have recorded the
    # live process failed: the outcome is unprovable, so the worker must stop in
    # the existing BLOCKED recovery state with exactly one launch.
    ("RUNNING", "BLOCKED", 1),
])
def test_side_effect_phase_conflict_never_double_launches(
        stack, monkeypatch, phase, expected_status, expected_launches):
    study_id = stack.create()
    fired = _inject_busy_on_phase(monkeypatch, phase)

    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert resp.status_code == 202, resp.text
    assert _wait(fired.is_set)
    assert _wait_status(stack.real_runner, study_id, expected_status)

    final = _read_status(stack.real_runner, study_id)
    assert final.status == expected_status
    assert FakeAdapter.launch_count == expected_launches

    if expected_status == "COMPLETED":
        assert [a.trial_number for a in final.attempts] == [0, 1]
        assert all(a.phase == "FINALIZED" for a in final.attempts)
        return

    # Unknown launch outcome: no second launch, BLOCKED has no force-continue.
    assert len(final.attempts) == 1
    assert final.attempts[0].error_code in ("HPO_RECOVERY_REQUIRED",
                                            "HPO_PROCESS_STILL_ACTIVE")
    body = stack.status(study_id).json()
    assert body["error_code"] == "HPO_RECOVERY_REQUIRED"
    assert body["next_action"]
    again = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert again.status_code == 409
    assert again.json()["error_code"] == "HPO_RECOVERY_REQUIRED"
    assert FakeAdapter.launch_count == expected_launches


# ── 4. unrecoverable background failure stays visible and redacted ──


class _ExplodingRunner:
    SECRET = r"C:\secret\execution.json token=abc123 --command yolo train"

    def __init__(self, real):
        self.real = real
        self.run_calls = 0

    def prepare(self, study_id, config):
        return self.real.prepare(study_id, config)

    def status(self, study_id):
        return self.real.status(study_id)

    def run(self, study_id, *, stop_event=None):
        self.run_calls += 1
        raise RuntimeError(self.SECRET)

    def resume(self, study_id, *, stop_event=None):
        raise RuntimeError(self.SECRET)


def test_unrecoverable_failure_is_stable_redacted_and_survives_refresh(
        stack):
    study_id = stack.create()
    exploding = _ExplodingRunner(stack.real_runner)
    stack.runner = exploding
    stack.client = stack._client()

    started = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert started.status_code == 202, started.text

    resp = _status_until(stack.client, study_id,
                         lambda b: b.get("error_code") is not None)
    assert resp["error_code"] == "HPO_EXECUTION_ERROR"
    assert resp["next_action"]
    # never echo the raw exception text, path, command or token
    refresh = stack.client.get(f"/api/hpo/studies/{study_id}")
    assert refresh.status_code == 200
    assert "secret" not in refresh.text.lower()
    assert "token=abc123" not in refresh.text
    assert "yolo" not in refresh.text.lower()
    assert "traceback" not in refresh.text.lower()
    # the failure is provably side-effect free: nothing launched, record READY
    assert FakeAdapter.launch_count == 0
    assert resp["execution_status"] == "READY"
    assert exploding.run_calls == 1


# ── 5. reservation lifecycle on success and failure ───────────────


def test_reservation_released_only_when_worker_truly_ends(stack):
    study_id = stack.create()
    release = threading.Event()

    class _SlowRunner:
        def __init__(self, real):
            self.real = real
            self.entered = threading.Event()

        def prepare(self, study_id, config):
            return self.real.prepare(study_id, config)

        def status(self, study_id):
            return self.real.status(study_id)

        def run(self, study_id, *, stop_event=None):
            self.entered.set()
            assert release.wait(15)
            return self.real.run(study_id, stop_event=stop_event)

        def resume(self, study_id, *, stop_event=None):
            return self.real.resume(study_id, stop_event=stop_event)

    slow = _SlowRunner(stack.real_runner)
    stack.runner = slow
    stack.client = stack._client()

    assert stack.client.post(f"/api/hpo/studies/{study_id}/start",
                             json={}).status_code == 202
    assert slow.entered.wait(10)
    assert stack.manager.reservation_owner() == ("hpo", study_id)

    release.set()
    assert _wait_status(stack.real_runner, study_id, "COMPLETED")
    assert _wait(lambda: stack.manager.reservation_owner() is None)


def test_reservation_released_after_background_failure(stack):
    study_id = stack.create()
    exploding = _ExplodingRunner(stack.real_runner)
    stack.runner = exploding
    stack.client = stack._client()

    assert stack.client.post(f"/api/hpo/studies/{study_id}/start",
                             json={}).status_code == 202
    assert _wait(lambda: stack.manager.reservation_owner() is None)
    controller = stack.manager.get(study_id)
    assert controller is not None
    assert controller.is_active() is False
    # the terminal failure stays readable through the manager, but must never
    # count as an active controller / live slot
    assert stack.manager.active_hpo() is None
    assert stack.manager.active_train() is None


# ── 6. full API lifecycle under dense concurrent reads ────────────


def test_api_lifecycle_with_dense_polling_completes_once(tmp_path, stack):
    study_id = stack.create()
    counting = _CountingRunner(stack.real_runner, busy_count=0)
    stack.runner = counting
    stack.client = stack._client()

    stop_readers = threading.Event()
    reader_errors = []

    def hammer():
        while not stop_readers.is_set():
            try:
                stack.client.get(f"/api/hpo/studies/{study_id}")
                stack.client.get("/api/hpo/studies?offset=0&limit=20")
            except Exception as exc:  # pragma: no cover
                reader_errors.append(repr(exc))
                return
            # Far denser than the real 2s page poll, but not a lock-starvation
            # benchmark: the point is sustained read pressure, not CPU denial.
            time.sleep(0.02)

    readers = [threading.Thread(target=hammer, name=f"studio-hammer-{i}")
               for i in range(3)]
    for reader in readers:
        reader.start()
    try:
        assert stack.client.post(f"/api/hpo/studies/{study_id}/start",
                                 json={}).status_code == 202
        assert _wait_status(stack.real_runner, study_id, "COMPLETED", timeout=60)
    finally:
        stop_readers.set()
        for reader in readers:
            reader.join(15)

    assert reader_errors == []
    assert counting.run_calls == 1
    assert FakeAdapter.launch_count == Stack.BUDGET

    body = stack.status(study_id).json()
    assert body["execution_status"] == "COMPLETED"
    assert body["error_code"] is None
    # Sustained polling must not cost trial slots: the writer yields to the
    # readers' short transactions instead of being restarted by them.
    assert body["success_count"] == Stack.BUDGET
    assert body["claimed_count"] == Stack.BUDGET
    assert body["terminal_count"] == Stack.BUDGET
    assert body["success_count"] == Stack.BUDGET
    assert len(body["trials"]) == Stack.BUDGET
    assert [t["trial_number"] for t in body["trials"]] == [1, 2]
    assert body["ranking"] and body["ranking"][0]["rank"] == 1
    assert body["can_resume"] is False


# ── 7. the lock policy itself: readers fast-fail, writers wait ────


def test_writer_session_waits_out_a_reader_while_readers_fail_fast(stack):
    """A short read must not be able to defeat the execution session."""
    from auto_tune.modules.hpo import storage as storage_mod
    study_id = stack.create()
    store = storage_mod.StudyStore(stack.root)

    reader_inside = threading.Event()
    release_reader = threading.Event()

    def reader():
        with store.locked(study_id):
            reader_inside.set()
            assert release_reader.wait(10)

    reader_thread = threading.Thread(target=reader, name="studio-reader")
    reader_thread.start()
    assert reader_inside.wait(10)
    try:
        # A plain read keeps the historical contract: an explicit, retryable
        # busy code instead of blocking the request.
        with pytest.raises(HpoError) as err:
            with store.locked(study_id):
                pass
        assert err.value.code == "HPO_STUDY_BUSY"

        finished = threading.Event()
        outcome = {}

        def writer():
            with storage_mod.writer_waits_for_lock(3.0):
                with store.locked(study_id):
                    outcome["ok"] = True
            finished.set()

        writer_thread = threading.Thread(target=writer, name="hpo-runner-test")
        writer_thread.start()
        # still waiting, not failed and not past the lock
        assert finished.wait(0.3) is False
        release_reader.set()
        assert finished.wait(10) is True
        writer_thread.join(10)
        assert outcome.get("ok") is True
    finally:
        release_reader.set()
        reader_thread.join(10)


def test_writer_wait_is_bounded_and_still_reports_busy(stack):
    """A holder that never releases must not hang the session forever."""
    from auto_tune.modules.hpo import storage as storage_mod
    study_id = stack.create()
    store = storage_mod.StudyStore(stack.root)

    reader_inside = threading.Event()
    release_reader = threading.Event()

    def holder():
        with store.locked(study_id):
            reader_inside.set()
            assert release_reader.wait(10)

    holder_thread = threading.Thread(target=holder, name="studio-reader")
    holder_thread.start()
    assert reader_inside.wait(10)
    try:
        with storage_mod.writer_waits_for_lock(0.2):
            with pytest.raises(HpoError) as err:
                with store.locked(study_id):
                    pass
        assert err.value.code == "HPO_STUDY_BUSY"
    finally:
        release_reader.set()
        holder_thread.join(10)


# ── 8. transient busy in the history list is not "unreadable" ─────


def test_history_marks_transient_busy_separately_from_corruption(stack):
    good = stack.create()
    bad = stack.root / ("hpo_" + "3" * 32)
    bad_id = bad.name
    bad.mkdir()
    (bad / "study.json").write_text("{bad", encoding="utf-8")

    from auto_tune.modules.hpo import storage as storage_mod
    original_read = storage_mod.StudyStore.read

    def busy_read(self, sid):
        if sid == good:
            raise HpoError("HPO_STUDY_BUSY", "study is busy")
        return original_read(self, sid)

    # A scoped patch only: the stack fixtures must stay in place afterwards.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage_mod.StudyStore, "read", busy_read)
        rows = {r["study_id"]: r for r in
                stack.client.get("/api/hpo/studies").json()["studies"]}

    assert rows[good]["readable"] is False
    assert rows[good].get("transient") is True     # short-lived, will recover
    assert rows[bad_id]["readable"] is False
    assert rows[bad_id].get("transient") is False     # real corruption stays fatal

    # next refresh recovers the transient row to a normal record
    rows2 = {r["study_id"]: r for r in
             stack.client.get("/api/hpo/studies").json()["studies"]}
    assert rows2[good]["readable"] is True
    assert rows2[good]["execution_status"] == "READY"
    assert rows2[bad_id]["readable"] is False


# ── 8. front-end: serialized reads + transient rendering ──────────


def test_js_serializes_detail_and_history_reads():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    # a single in-flight refresh round
    assert "state.refreshing" in script
    # history is fetched only after the detail read settles
    assert ".then(hpoRefreshHistory)" in script or ".then(function () { hpoRefreshHistory" in script
    # the manual refresh path goes through the serialized round
    refresh = script.split("window.hpoRefresh = function", 1)[1].split("};", 1)[0]
    assert "refreshRound" in refresh


def test_js_renders_transient_history_rows_without_calling_them_corrupt():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    assert "row.transient" in script
    assert "暂忙" in script
    assert ".innerHTML" not in script


def test_js_keeps_single_timer_and_last_trusted_state():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    assert "if (state.timer) return;" in script
    assert "显示最近一次可信状态" in script
    assert "记录不可读取" in script
