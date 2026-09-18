"""H1.3 Task 4: progress / stop / resume / history contract for HPO.

HTTP + static-JS assertions at tmp_path. Real HpoService/HpoRunner are used for
the corrupted/old-record rejection checks (no training is ever started); the
lifecycle HTTP tests use a fake runner that never launches YOLO.
"""

import json
import re
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
from auto_tune.modules.model_store import ModelStore
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.ui.hpo_api import create_hpo_router

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _rid():
    return uuid.uuid4().hex


def _make_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-lifecycle-test-not-a-real-model")
    return snapshot, model


class FakeExec:
    def __init__(self, study_id, config):
        self.study_id = study_id
        self.config = config
        self.status = "READY"
        self.revision = 0
        self.stop_reason = None
        self.attempts = []

    def set(self, status, stop_reason=None):
        self.status = status
        self.stop_reason = stop_reason
        self.revision += 1


class FakeRunner:
    def __init__(self):
        self._execs = {}
        self._lock = threading.Lock()
        self.run_calls = 0
        self.resume_calls = 0

    def _get(self, study_id):
        record = self._execs.get(study_id)
        if record is None:
            raise HpoError("HPO_NOT_FOUND", "no execution")
        return record

    def prepare(self, study_id, config):
        existing = self._execs.get(study_id)
        if existing is not None:
            return existing
        record = FakeExec(study_id, config)
        self._execs[study_id] = record
        return record

    def status(self, study_id):
        return self._get(study_id)

    def _spin(self, record, stop_event):
        record.set("RUNNING")
        deadline = time.time() + 10
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                record.set("PAUSED", "user_stopped")
                return
            time.sleep(0.005)
        record.set("PAUSED", "timeout")

    def run(self, study_id, *, stop_event=None):
        with self._lock:
            self.run_calls += 1
        self._spin(self._get(study_id), stop_event)

    def resume(self, study_id, *, stop_event=None):
        with self._lock:
            self.resume_calls += 1
        self._spin(self._get(study_id), stop_event)


class Stack:
    def __init__(self, tmp_path):
        self.snapshot, self.model = _make_inputs(tmp_path)
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.runner = FakeRunner()
        self.manager = RunManager()
        snapshot, model = self.snapshot, self.model
        # 新建研究只接受受控模型标识（客户端不再提交任何路径）
        self.store = ModelStore(tmp_path / "models" / "weights",
                                legacy_roots=[tmp_path],
                                max_upload_bytes=1024, max_models=50)
        self.model_id = {row.name: row.model_id
                         for row in self.store.list_models()}["fixture.pt"]

        def resolve_snapshot(sid):
            if sid == snapshot.snapshot_id:
                return Path(snapshot.snapshot_path)
            raise HpoError("HPO_INVALID_CONFIG", "bad snapshot")

        def resolve_model(model_id):
            return self.store.resolve(model_id)

        router = create_hpo_router(service=self.service, runner=self.runner,
                                   manager=self.manager,
                                   resolve_snapshot=resolve_snapshot,
                                   resolve_model=resolve_model,
                                   assert_training_slot_free=lambda: None)
        app = FastAPI()
        app.include_router(router, prefix="/api/hpo")
        self.client = TestClient(app)

    def create(self):
        return self.client.post("/api/hpo/studies", json={
            "snapshot_id": self.snapshot.snapshot_id,
            "model_id": self.model_id,
            "study_config": {}, "execution_config": {},
        }).json()["study_id"]


@pytest.fixture
def stack(tmp_path):
    return Stack(tmp_path)


def _wait(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ── state projections ─────────────────────────────────────────────


@pytest.mark.parametrize("status,can_resume", [
    ("READY", False),
    ("RUNNING", False),
    ("PAUSED", True),
    ("INTERRUPTED", True),
    ("COMPLETED", False),
    ("BLOCKED", False),
])
def test_status_projection_for_each_state(stack, status, can_resume):
    study_id = stack.create()
    stack.runner._get(study_id).set(status)
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["execution_status"] == status
    assert body["can_resume"] is can_resume
    assert body["control_active"] is False
    assert body["can_stop"] is False


def test_completed_all_failed_has_no_result(stack):
    study_id = stack.create()
    service = stack.service
    for _ in range(2):
        trial = service.ask(study_id, request_id=_rid())
        service.tell(study_id, trial.number,
                     ResultInput(state="FAILED", reason_code="training_failed"))
    stack.runner._get(study_id).set("COMPLETED")
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["terminal_count"] == 2
    assert body["success_count"] == 0
    assert body["ranking"] == []
    assert body["has_success"] is False
    assert body["can_resume"] is False
    best = stack.client.get(f"/api/hpo/studies/{study_id}/best-config")
    assert best.status_code == 409
    assert best.json()["error_code"] == "HPO_NO_SUCCESS"


def test_blocked_has_no_force_continue(stack):
    study_id = stack.create()
    stack.runner._get(study_id).set("BLOCKED")
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["error_code"] == "HPO_RECOVERY_REQUIRED"
    assert body["can_resume"] is False
    assert stack.client.post(f"/api/hpo/studies/{study_id}/resume", json={}).status_code == 409
    assert stack.client.post(f"/api/hpo/studies/{study_id}/start", json={}).status_code == 409


def test_stop_accepts_without_claiming_stopped(stack):
    study_id = stack.create()
    stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert _wait(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "RUNNING")
    stop = stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert stop.status_code == 202
    assert stop.json()["stop_requested"] is True
    assert stop.json()["stopped"] is False  # request received ≠ process exited
    assert _wait(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")


def test_restart_does_not_auto_resume(stack):
    study_id = stack.create()
    stack.runner._get(study_id).set("RUNNING")  # persisted RUNNING, no controller
    before = stack.runner.resume_calls
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["execution_status"] == "RUNNING"
    assert stack.runner.resume_calls == before  # a GET never resumes
    explicit = stack.client.post(f"/api/hpo/studies/{study_id}/resume", json={})
    assert explicit.status_code == 202
    assert _wait(lambda: stack.runner.resume_calls == before + 1)
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert _wait(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")


# ── old/corrupt execution records: stable reject, bytes unchanged ──


def _real_stack(tmp_path):
    snapshot, model = _make_inputs(tmp_path)
    root = tmp_path / "hpo"
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2),
                                 snapshot_dir=snapshot.snapshot_path,
                                 model_path=model)
    runner = HpoRunner(root, tmp_path / "out", tmp_path / "log")
    runner.prepare(study.study_id, ExecutionConfig())
    return root, study.study_id, runner


def _exec_path(root, study_id):
    return root / study_id / "execution.json"


def test_old_record_missing_command_executable_rejected_bytes_unchanged(tmp_path):
    root, study_id, runner = _real_stack(tmp_path)
    target = _exec_path(root, study_id)
    data = json.loads(target.read_text(encoding="utf-8"))
    data["status"] = "RUNNING"
    data["attempts"] = [{
        "trial_number": 0,
        "trial_id": f"{study_id}_t0000",
        "request_id": "0" * 32,
        "run_id": "tuning:" + uuid.uuid4().hex,
        "phase": "PREPARED",
        "candidate_params": {},
        "effective_params": {},
        "command": ["yolo", "train"],
        "run_relpath": "run0",
        "args_sha256": "0" * 64,
        # command_executable intentionally absent (pre-acceptance old record)
    }]
    before = json.dumps(data, sort_keys=True).encode("utf-8")
    target.write_bytes(before)
    with pytest.raises(HpoError) as err:
        runner.status(study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert target.read_bytes() == before  # never rewritten


def test_polluted_execution_record_rejected_bytes_unchanged(tmp_path):
    root, study_id, runner = _real_stack(tmp_path)
    target = _exec_path(root, study_id)
    data = json.loads(target.read_text(encoding="utf-8"))
    data["config"]["epochs"] = 999  # config is not a frozen execution field
    before = json.dumps(data, sort_keys=True).encode("utf-8")
    target.write_bytes(before)
    with pytest.raises(HpoError):
        runner.status(study_id)
    assert target.read_bytes() == before


# ── front-end lifecycle guarantees (static) ────────────────────────


def test_js_single_timer_and_unload_cleanup():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    assert "if (state.timer) return;" in script          # single timer
    assert "beforeunload" in script and "stopPolling" in script
    assert "clearInterval" in script


def test_js_refresh_does_not_restart_and_keeps_trusted_state():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    # refresh only fetches; it never POSTs start/resume
    refresh = script.split("window.hpoRefresh = function", 1)[1].split("};", 1)[0]
    assert "/start" not in refresh and "/resume" not in refresh
    # network error keeps last state and tells the user it is retrying
    assert "显示最近一次可信状态" in script


def test_js_history_uses_textcontent_and_shows_unreadable():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    assert "记录不可读取" in script
    assert ".textContent" in script


def test_history_bad_entry_visible_via_api(stack, tmp_path):
    stack.create()
    bad = stack.root / ("hpo_" + "2" * 32)
    bad.mkdir()
    (bad / "study.json").write_text("{bad", encoding="utf-8")
    rows = {e["study_id"]: e for e in
            stack.client.get("/api/hpo/studies").json()["studies"]}
    assert bad.name in rows and rows[bad.name]["readable"] is False
