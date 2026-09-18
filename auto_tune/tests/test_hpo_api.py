"""H1.3 Task 2: HPO HTTP API — strict input, start/query, slot behavior.

A real HpoService over a tmp_path storage root provides the study facts; the
execution runner is a fake that never starts a YOLO subprocess or calls a
network LLM (credential resolution is patched to fail loudly when invoked).
Each test builds an isolated FastAPI app + fresh RunManager.
"""

import json
import os
import re
import subprocess
import sys
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
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.model_store import ModelStore, ModelStoreError
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.modules.run_state.process_identity import capture_process_identity
from auto_tune.modules.run_state.service import (
    new_run_state,
    read_run_state,
    with_status_phase,
    write_run_state,
)
from auto_tune.ui.hpo_api import (
    _ERR_TEMPLATE,
    _NEXT_ACTION,
    create_hpo_router,
)


def _rid():
    return uuid.uuid4().hex


def _evidence(value_epoch=1):
    return Evidence(run_id=f"run-{_rid()}", artifact_relpath="results.csv",
                    artifact_sha256="0" * 64, epoch=value_epoch)


def _make_inputs(tmp_path, source_name="source", count=4, seed=42):
    source = tmp_path / source_name
    source.mkdir()
    for n in range(count):
        Image.new("RGB", (16, 16)).save(source / f"img_{n:04d}.jpg")
        (source / f"img_{n:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                 encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=seed,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-api-test-not-a-real-model")
    return snapshot, model


# A legal published snapshot whose manifest is far larger than the old fixed
# 64 KiB read cap (~500 bytes per sample → 160 samples ≈ 80 KiB).
_LARGE_SNAPSHOT_SAMPLES = 160


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
        with self._lock:
            record = self._execs.get(study_id)
        if record is None:
            raise HpoError("HPO_NOT_FOUND", f"study {study_id} has no execution")
        return record

    def prepare(self, study_id, config):
        with self._lock:
            existing = self._execs.get(study_id)
            if existing is not None:
                if existing.config != config:
                    raise HpoError("HPO_EXECUTION_CONFLICT",
                                   "execution config differs")
                return existing
            record = FakeExec(study_id, config)
            self._execs[study_id] = record
            return record

    def status(self, study_id):
        return self._get(study_id)

    def _spin_until_stop(self, record, stop_event):
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
        record = self._get(study_id)
        self._spin_until_stop(record, stop_event)
        return record

    def resume(self, study_id, *, stop_event=None):
        with self._lock:
            self.resume_calls += 1
        record = self._get(study_id)
        self._spin_until_stop(record, stop_event)
        return record


class Stack:
    def __init__(self, tmp_path, gate=None, snapshot=None, model=None,
                 list_snapshots=None):
        if snapshot is None or model is None:
            snapshot, model = _make_inputs(tmp_path)
        self.snapshot, self.model = snapshot, model
        # 新建研究只接受受控模型标识：fixture 权重就是受控库里的一个普通文件
        self.store = ModelStore(tmp_path / "models" / "weights",
                                legacy_roots=[tmp_path],
                                max_upload_bytes=1024, max_models=50)
        self.model_id = {row.name: row.model_id
                         for row in self.store.list_models()}["fixture.pt"]
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.runner = FakeRunner()
        self.manager = RunManager()
        # Published-snapshot projection a test may override (the authoritative
        # snapshot_id → dataset_name source used by the history listing).
        self.published_rows = None
        self._list_snapshots = list_snapshots
        self.client = self._client(tmp_path, gate=gate)

    def _client(self, tmp_path, gate=None):
        snapshot = self.snapshot
        model = self.model
        stack = self

        def resolve_snapshot(snapshot_id):
            if snapshot_id == snapshot.snapshot_id:
                return Path(snapshot.snapshot_path)
            raise HpoError("HPO_INVALID_CONFIG", "快照不存在或已失效")

        def resolve_model(model_id):
            """受控模型标识 -> 冻结路径；测试里就是受控库中的 fixture 权重。"""
            return stack.store.resolve(model_id)

        def published_snapshots():
            from auto_tune.ui.app import list_published_snapshots

            if stack._list_snapshots is not None:
                return stack._list_snapshots()
            if stack.published_rows is not None:
                return stack.published_rows
            return list_published_snapshots(tmp_path / "snapshots")

        router = create_hpo_router(
            service=self.service, runner=self.runner, manager=self.manager,
            resolve_snapshot=resolve_snapshot, resolve_model=resolve_model,
            assert_training_slot_free=gate or (lambda: None),
            list_snapshots=published_snapshots)
        app = FastAPI()
        app.include_router(router, prefix="/api/hpo")
        return TestClient(app)

    def create(self, **overrides):
        payload = {
            "snapshot_id": self.snapshot.snapshot_id,
            "model_id": self.model_id,
            "study_config": {},
            "execution_config": {},
        }
        payload.update(overrides)
        return self.client.post("/api/hpo/studies", json=payload)


@pytest.fixture
def stack(tmp_path):
    return Stack(tmp_path)


def _row(stack, study_id):
    """The history-listing row of one study (single-page listing)."""
    rows = stack.client.get("/api/hpo/studies?offset=0&limit=10").json()["studies"]
    return {r["study_id"]: r for r in rows}[study_id]


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ── create: prepare only, never train ─────────────────────────────


def test_create_201_and_read_only_status_fields(stack):
    resp = stack.create()
    assert resp.status_code == 201
    study_id = resp.json()["study_id"]
    assert study_id.startswith("hpo_")
    assert stack.runner.run_calls == 0

    status = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert status["study_id"] == study_id
    assert status["execution_status"] == "READY"
    assert status["budget"] == 10
    assert status["terminal_count"] == 0
    assert status["success_count"] == 0
    assert status["control_active"] is False
    assert status["can_stop"] is False
    assert status["can_resume"] is False
    assert status["trials"] == []
    assert status["ranking"] == []


@pytest.mark.parametrize("field,payload", [
    ("budget", {"study_config": {"budget": True}}),
    ("budget", {"study_config": {"budget": 0}}),
    ("budget", {"study_config": {"budget": 101}}),
    ("budget", {"study_config": {"budget": "10"}}),
    ("epochs", {"study_config": {"epochs": 0}}),
    ("seed", {"study_config": {"seed": -1}}),
    ("batch", {"execution_config": {"batch": True}}),
    ("batch", {"execution_config": {"batch": 300}}),
    ("imgsz", {"execution_config": {"imgsz": 100}}),
    ("imgsz", {"execution_config": {"imgsz": "640"}}),
    ("extra", {"foo": 1}),
    ("extra nested", {"study_config": {"unknown_field": 1}}),
    ("roots/executable", {"execution_config": {"output_root": "/tmp/x"}}),
    ("roots/executable", {"execution_config": {"command": ["yolo", "train"]}}),
])
def test_create_rejects_invalid_input_zero_study(stack, field, payload):
    resp = stack.create(**payload)
    assert resp.status_code == 422, field
    body = resp.json()
    assert body["error_code"] and body["error"] and body["next_action"]
    assert list((stack.root).glob("hpo_*")) == []


def test_create_unknown_snapshot_rejected_no_latest_fallback(stack):
    # A wrong/explicit snapshot id must not silently fall back to the global
    # latest dataset; the HPO module has no concept of it and the resolver
    # rejects the id, so nothing is created and nothing is trained.
    resp = stack.create(snapshot_id="0" * 64)
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "HPO_INVALID_CONFIG"
    assert list(stack.root.glob("hpo_*")) == []


def test_create_missing_model_rejected(stack):
    resp = stack.create(model_id="sha256:" + "0" * 64)
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "MODEL_NOT_FOUND"
    assert resp.json()["field"] == "model_id"
    assert str(stack.root) not in resp.text
    assert list(stack.root.glob("hpo_*")) == []


def test_create_rejects_a_client_supplied_model_path(stack):
    """客户端路径字段在严格契约下直接拒绝，绝不作为权重来源。"""
    resp = stack.create(model_path=str(stack.model))
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_HPO_FIELD"
    assert resp.json()["field"] == "model_path"
    assert list(stack.root.glob("hpo_*")) == []


def test_create_freezes_the_controlled_weight_identity(stack):
    """新研究冻结的是受控文件的规范化路径、字节数与 SHA-256（既有绑定契约）。"""
    import hashlib

    body = stack.create().json()
    study = stack.service.load_study(body["study_id"])
    binding = study.model_binding
    assert Path(binding.model_path) == Path(stack.model).resolve()
    assert binding.model_bytes == Path(stack.model).stat().st_size
    assert binding.model_sha256 == hashlib.sha256(
        Path(stack.model).read_bytes()).hexdigest()


def test_old_studies_keep_their_frozen_binding(stack):
    """旧研究继续按既有冻结值校验，不按当前模型库重新绑定。"""
    study_id = stack.create().json()["study_id"]
    study = stack.service.load_study(study_id)
    frozen_path = Path(study.model_binding.model_path)

    # 当前模型库里不再有这个名字，旧研究仍按冻结路径与哈希读取
    frozen_path.unlink()
    reopened = stack.service.load_study(study_id)
    assert Path(reopened.model_binding.model_path) == frozen_path
    assert reopened.model_binding.model_sha256 == study.model_binding.model_sha256


def test_llm_credentials_never_resolved(stack, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM/credential resolver must not be called")

    monkeypatch.setattr("auto_tune.modules.security.credentials.resolve_credential",
                        boom, raising=False)
    resp = stack.create()
    assert resp.status_code == 201
    study_id = resp.json()["study_id"]
    status = stack.client.get(f"/api/hpo/studies/{study_id}")
    assert status.status_code == 200


# ── start / stop / resume / same-study idempotency ────────────────


def test_start_202_then_stop_converges_to_paused(stack):
    study_id = stack.create().json()["study_id"]
    start = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert start.status_code == 202
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "RUNNING")
    status = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert status["control_active"] is True
    assert status["can_stop"] is True

    stop = stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert stop.status_code == 202
    assert stop.json()["stop_requested"] is True
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")
    after = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert after["can_resume"] is True
    assert after["control_active"] is False


def test_start_rejects_extra_body_fields(stack):
    study_id = stack.create().json()["study_id"]
    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={"a": 1})
    assert resp.status_code == 422
    assert stack.runner.run_calls == 0


def test_same_study_duplicate_start_no_second_worker(stack):
    study_id = stack.create().json()["study_id"]
    first = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert first.status_code == 202
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "RUNNING")
    # same study while its controller is active → 200 + current state, no worker
    second = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert second.status_code == 200
    assert stack.runner.run_calls == 1
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")


def test_second_different_study_start_409(stack):
    first = stack.create().json()["study_id"]
    second = stack.create().json()["study_id"]
    assert first != second
    stack.client.post(f"/api/hpo/studies/{first}/start", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{first}").json()["execution_status"] == "RUNNING")
    resp = stack.client.post(f"/api/hpo/studies/{second}/start", json={})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
    assert stack.runner.run_calls == 1
    stack.client.post(f"/api/hpo/studies/{first}/stop", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{first}").json()["execution_status"] == "PAUSED")


def test_start_on_completed_is_idempotent_200(stack):
    study_id = stack.create().json()["study_id"]
    record = stack.runner._get(study_id)
    record.set("COMPLETED")
    resp = stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert resp.status_code == 200
    assert stack.runner.run_calls == 0


def test_stop_without_controller_on_running_409(stack):
    """Restart-like: RUNNING persisted execution with no controller → 409."""
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("RUNNING")
    resp = stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_RECOVERY_REQUIRED"


def test_stop_terminal_returns_200(stack):
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("COMPLETED")
    resp = stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert resp.status_code == 200
    assert resp.json()["stopped"] is True


def test_resume_on_paused_spawns_resume_worker(stack):
    study_id = stack.create().json()["study_id"]
    stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "RUNNING")
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")
    before = stack.runner.resume_calls
    resume = stack.client.post(f"/api/hpo/studies/{study_id}/resume", json={})
    assert resume.status_code == 202
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "RUNNING")
    assert stack.runner.resume_calls == before + 1
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})
    assert _wait_until(lambda: stack.client.get(
        f"/api/hpo/studies/{study_id}").json()["execution_status"] == "PAUSED")


def test_resume_on_blocked_409(stack):
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("BLOCKED")
    resp = stack.client.post(f"/api/hpo/studies/{study_id}/resume", json={})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_RECOVERY_REQUIRED"


# ── GET only reads: no mutation, no start/finalize ────────────────


def test_get_routes_do_not_mutate_or_launch(stack):
    study_id = stack.create().json()["study_id"]
    before_rev = stack.service.load_study(study_id).revision
    study_file = stack.root / study_id / "study.json"
    before_bytes = study_file.read_bytes()
    assert stack.client.get(f"/api/hpo/studies/{study_id}").status_code == 200
    assert stack.client.get("/api/hpo/studies").status_code == 200
    assert stack.client.get(f"/api/hpo/studies/{study_id}/best-config").status_code == 409
    after = stack.service.load_study(study_id)
    assert after.revision == before_rev
    assert study_file.read_bytes() == before_bytes
    assert stack.runner.run_calls == 0
    assert stack.runner.resume_calls == 0


# ── list: pagination + corrupt studies stay visible ───────────────


def test_list_pagination_and_stable_ordering(stack):
    ids = [stack.create().json()["study_id"] for _ in range(3)]
    page1 = stack.client.get("/api/hpo/studies?offset=0&limit=2").json()
    assert page1["count"] == 3
    assert len(page1["studies"]) == 2
    page2 = stack.client.get("/api/hpo/studies?offset=2&limit=2").json()
    assert len(page2["studies"]) == 1
    # newest first ordering across the two pages; every study appears once
    ordered = page1["studies"] + page2["studies"]
    creations = [s["created_at"] for s in ordered]
    assert creations == sorted(creations, reverse=True)
    assert sorted(s["study_id"] for s in ordered) == sorted(ids)


def test_list_bad_record_visible_not_silently_dropped(stack, tmp_path):
    stack.create()
    bad = stack.root / ("hpo_" + "1" * 32)
    bad.mkdir()
    (bad / "study.json").write_text("{not json", encoding="utf-8")
    data = stack.client.get("/api/hpo/studies").json()
    entries = {e["study_id"]: e for e in data["studies"]}
    assert bad.name in entries
    assert entries[bad.name]["readable"] is False
    assert entries[bad.name]["execution_error_code"] == "HPO_CORRUPT_STUDY"


@pytest.mark.parametrize("query", [
    "offset=-1&limit=20", "offset=0&limit=0", "offset=0&limit=101",
    "offset=true&limit=20",
])
def test_list_invalid_query_422(stack, query):
    resp = stack.client.get(f"/api/hpo/studies?{query}")
    assert resp.status_code == 422


# ── best-config: only revalidated rank-1 success ──────────────────


def test_best_config_requires_success_and_returns_rank_first(stack):
    study_id = stack.create().json()["study_id"]
    service = stack.service
    for i, value in enumerate((0.5, 0.9)):
        trial = service.ask(study_id, request_id=_rid())
        service.tell(study_id, trial.number,
                     ResultInput(state="SUCCESS", value=value,
                                 evidence=_evidence(value_epoch=1)))
    resp = stack.client.get(f"/api/hpo/studies/{study_id}/best-config")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["source"]["study_id"] == study_id
    assert payload["source"]["trial_number"] == 1  # the 0.9 trial ranks first
    assert payload["value"] == 0.9
    assert set(payload["search"]) == {
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"}
    assert payload["fixed"]["epochs"] == 30
    assert payload["fixed"]["batch"] == 16
    assert payload["fixed"]["imgsz"] == 640


# ── unified persisted restart gate on start/resume (rework #1) ────


def _redirect_log(monkeypatch, tmp_path):
    """Redirect every fact path rooted at ``log/`` into a tmp log directory."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir.joinpath(*parts[1:]))
        return real_join(*parts)

    monkeypatch.setattr(os.path, "join", fake_join)
    return log_dir


def _gated_stack(tmp_path, monkeypatch):
    """A stack whose router uses the real application-wide training-slot gate."""
    from auto_tune.ui import app as app_mod

    stack = Stack(tmp_path)
    _redirect_log(monkeypatch, tmp_path)
    stack.client = stack._client(
        tmp_path, gate=app_mod._assert_training_slot_free)
    return stack


def _live_child():
    """A real live child process used only to capture a matching identity."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    identity = capture_process_identity(proc.pid)
    if identity is None:
        proc.terminate()
        pytest.skip("platform cannot capture identity for the live child")
    return proc, identity


def _gone_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    return proc.pid


def _write_state_file(path: Path, run_kind: str, *, status: str,
                      pid=None, token=None):
    state = new_run_state(run_kind)
    state = with_status_phase(
        state, status=status, phase="training",
        pid=pid, process_create_token=token)
    write_run_state(str(path), state)
    return state


def _write_execution(log_dir: Path, study: str, text: str) -> None:
    target = log_dir / "hpo" / "studies" / study
    target.mkdir(parents=True, exist_ok=True)
    (target / "execution.json").write_text(text, encoding="utf-8")


def _start(stack, study_id):
    return stack.client.post(f"/api/hpo/studies/{study_id}/start", json={})


def _resume(stack, study_id):
    return stack.client.post(f"/api/hpo/studies/{study_id}/resume", json={})


def test_hpo_start_blocked_by_persisted_manual_match(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    proc, identity = _live_child()
    try:
        _write_state_file(tmp_path / "log" / "training_running.json", "manual",
                          status="running", pid=proc.pid,
                          token=identity.process_create_token)
        resp = _start(stack, study_id)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        assert stack.runner.run_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        proc.terminate()


def test_hpo_start_blocked_by_persisted_manual_unverifiable(tmp_path, monkeypatch):
    from auto_tune.modules.run_state import gate as gate_mod
    from auto_tune.modules.run_state.process_identity import IdentityMatch

    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    monkeypatch.setattr(gate_mod, "compare_process_identity",
                        lambda expected: IdentityMatch.UNVERIFIABLE)
    _write_state_file(tmp_path / "log" / "training_running.json", "manual",
                      status="running", pid=os.getpid(), token="unverifiable")
    resp = _start(stack, study_id)
    assert resp.status_code == 409
    assert stack.runner.run_calls == 0
    assert stack.manager.reservation_owner() is None


def test_hpo_start_blocked_by_persisted_tuning_match(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    proc, identity = _live_child()
    try:
        _write_state_file(tmp_path / "log" / "tuning_running.json", "tuning",
                          status="running", pid=proc.pid,
                          token=identity.process_create_token)
        resp = _start(stack, study_id)
        assert resp.status_code == 409
        assert stack.runner.run_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        proc.terminate()


def test_hpo_start_blocked_by_other_study_live_execution(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    proc, identity = _live_child()
    try:
        _write_execution(tmp_path / "log", "hpo_otherstudy", json.dumps({
            "schema_version": "hpo-execution-v1",
            "study_id": "hpo_otherstudy",
            "status": "RUNNING",
            "attempts": [{
                "phase": "RUNNING",
                "pid": proc.pid,
                "process_create_token": identity.process_create_token,
            }],
        }))
        resp = _start(stack, study_id)
        assert resp.status_code == 409
        assert stack.runner.run_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        proc.terminate()


def test_hpo_start_blocked_by_corrupt_execution(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    _write_execution(tmp_path / "log", "hpo_broken", "{not json")
    resp = _start(stack, study_id)
    assert resp.status_code == 409
    assert stack.runner.run_calls == 0
    assert stack.manager.reservation_owner() is None


def test_hpo_resume_blocked_by_persisted_manual_match(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("PAUSED", "user_stopped")
    proc, identity = _live_child()
    try:
        _write_state_file(tmp_path / "log" / "training_running.json", "manual",
                          status="running", pid=proc.pid,
                          token=identity.process_create_token)
        resp = _resume(stack, study_id)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        assert stack.runner.resume_calls == 0
        assert stack.runner.run_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        proc.terminate()


def test_hpo_resume_blocked_by_other_study_live_execution(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("PAUSED", "user_stopped")
    proc, identity = _live_child()
    try:
        _write_execution(tmp_path / "log", "hpo_otherstudy", json.dumps({
            "status": "RUNNING",
            "attempts": [{"phase": "RUNNING", "pid": proc.pid,
                          "process_create_token": identity.process_create_token}]}))
        resp = _resume(stack, study_id)
        assert resp.status_code == 409
        assert stack.runner.resume_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        proc.terminate()


def test_blocked_start_creates_no_controller_or_directories(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    studies_before = sorted(p.name for p in stack.root.iterdir())
    proc, identity = _live_child()
    try:
        _write_state_file(tmp_path / "log" / "training_running.json", "manual",
                          status="running", pid=proc.pid,
                          token=identity.process_create_token)
        assert _start(stack, study_id).status_code == 409
        # no worker, no controller registered anywhere, no slot taken, no dirs
        assert stack.runner.run_calls == 0
        assert stack.manager.active_hpo() is None
        assert stack.manager.reservation_owner() is None
        assert app_mod._RUN_MANAGER.reservation_owner() is None
        assert sorted(p.name for p in stack.root.iterdir()) == studies_before
    finally:
        proc.terminate()


def test_missing_persisted_state_reconciles_then_allows_start(tmp_path, monkeypatch):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    state_file = tmp_path / "log" / "training_running.json"
    _write_state_file(state_file, "manual", status="running",
                      pid=_gone_pid(), token="stale-token")

    resp = _start(stack, study_id)
    assert resp.status_code == 202
    assert stack.runner.run_calls == 1
    reconciled = read_run_state(str(state_file), run_kind="manual")
    assert reconciled.status == "interrupted"
    assert reconciled.terminal_reason == "process_missing"
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})


def test_reservation_without_controller_blocks_hpo_start(tmp_path, monkeypatch):
    """The reserve→register atomic gap must already count as occupied."""
    from auto_tune.ui import app as app_mod

    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    token = app_mod._RUN_MANAGER.reserve("manual", "manual:reserved-not-registered")
    try:
        resp = _start(stack, study_id)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        assert stack.runner.run_calls == 0
        assert stack.manager.reservation_owner() is None
    finally:
        app_mod._RUN_MANAGER.release(token)


def test_two_study_concurrent_start_only_one_runs(stack):
    first = stack.create().json()["study_id"]
    second = stack.create().json()["study_id"]
    barrier = threading.Barrier(2)
    codes = []
    lock = threading.Lock()

    def worker(study_id):
        barrier.wait(timeout=10)
        code = _start(stack, study_id).status_code
        with lock:
            codes.append(code)

    threads = [threading.Thread(target=worker, args=(sid,))
               for sid in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert sorted(codes) == [202, 409]
    assert stack.runner.run_calls == 1
    for study_id in (first, second):
        stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})


# ── controlled error bodies: no underlying detail may leak (rework #2) ──

_CONTROLLED_TEXTS = {text for _, text in _ERR_TEMPLATE.values()} | {
    "执行出现未知错误。", "处理请求时发生未知错误。",
}

_DIRTY = [
    r"C:\secret\execution.json",
    "/etc/secret/config.yaml",
    "Traceback (most recent call last)",
    "sk-abcdef123456",
    r"yolo train data=C:\private\data.yaml",
    "<script>alert('xss')</script>",
]


def _assert_redacted(resp, *dirty):
    assert resp.status_code >= 400
    body = resp.json()
    assert re.fullmatch(r"HPO_[A-Z0-9_]+", body["error_code"])
    assert body["error"] in _CONTROLLED_TEXTS
    assert body["next_action"] in _NEXT_ACTION.values()
    assert set(body) == {"error_code", "error", "next_action"}
    for token in dirty:
        assert token not in resp.text


@pytest.mark.parametrize("dirty", _DIRTY)
def test_hpo_error_routes_never_echo_hpoerror_message(stack, dirty):
    study_id = stack.create().json()["study_id"]
    service = stack.service
    trial = service.ask(study_id, request_id=_rid())
    service.tell(study_id, trial.number,
                 ResultInput(state="SUCCESS", value=0.5, evidence=_evidence()))

    def boom(_study_id):
        raise HpoError("HPO_CORRUPT_EXECUTION", dirty)

    stack.runner.status = boom
    _assert_redacted(stack.client.get(f"/api/hpo/studies/{study_id}"), dirty)
    _assert_redacted(stack.client.get(f"/api/hpo/studies/{study_id}/best-config"), dirty)
    _assert_redacted(_start(stack, study_id), dirty)
    _assert_redacted(stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={}), dirty)
    _assert_redacted(_resume(stack, study_id), dirty)
    # list stays a readable 200 and only exposes the stable code
    listing = stack.client.get("/api/hpo/studies")
    assert listing.status_code == 200
    assert dirty not in listing.text
    entries = {e["study_id"]: e for e in listing.json()["studies"]}
    assert entries[study_id]["execution_error_code"] == "HPO_CORRUPT_EXECUTION"


@pytest.mark.parametrize("dirty", _DIRTY)
def test_hpo_error_routes_never_echo_unexpected_exception(stack, dirty):
    study_id = stack.create().json()["study_id"]

    def boom(_study_id):
        raise RuntimeError(dirty)

    stack.runner.status = boom
    _assert_redacted(stack.client.get(f"/api/hpo/studies/{study_id}"), dirty)
    _assert_redacted(_start(stack, study_id), dirty)
    _assert_redacted(_resume(stack, study_id), dirty)


@pytest.mark.parametrize("dirty", _DIRTY)
def test_polluted_error_code_never_reaches_the_client(stack, dirty):
    study_id = stack.create().json()["study_id"]

    def boom(_study_id):
        raise HpoError(dirty, "boom")

    stack.runner.status = boom
    resp = stack.client.get(f"/api/hpo/studies/{study_id}")
    _assert_redacted(resp, dirty)
    assert resp.json()["error_code"] == "HPO_EXECUTION_ERROR"


def test_redaction_never_turns_an_error_into_success(stack):
    study_id = stack.create().json()["study_id"]
    stack.runner.status = lambda _study_id: (_ for _ in ()).throw(
        RuntimeError(r"C:\secret\execution.json failed"))
    assert stack.client.get(f"/api/hpo/studies/{study_id}").status_code == 500
    assert _start(stack, study_id).status_code == 500
    assert stack.runner.run_calls == 0


# ── reconcile write-back failure must block start/resume ────────────
#
# A MISSING/MISMATCH record may only be released once it has been written
# back as terminal. If that write-back fails the old process may still be
# alive, so the HPO worker must not start either.


def _pin_identity(monkeypatch, match_name):
    """Pin the persisted-identity comparison to one outcome (once per read)."""
    from auto_tune.modules.run_state import gate as gate_mod
    from auto_tune.modules.run_state import process_identity as identity_mod
    from auto_tune.modules.run_state.process_identity import IdentityMatch

    calls = []

    def fake(expected):
        calls.append(expected)
        return IdentityMatch[match_name]

    monkeypatch.setattr(gate_mod, "compare_process_identity", fake)
    monkeypatch.setattr(identity_mod, "compare_process_identity", fake)
    return calls


def _fail_reconcile_write_back(monkeypatch):
    """The gate's terminal write-back fails; nothing else writes states."""
    from auto_tune.modules.run_state.models import RunStatePersistenceError
    from auto_tune.ui import app as app_mod

    def boom(*args, **kwargs):
        raise RunStatePersistenceError(
            r"disk full writing C:\secret\logs\training_running.json")

    monkeypatch.setattr(app_mod, "write_run_state", boom)


_RUN_STATE_FILES = {"manual": "training_running.json",
                    "tuning": "tuning_running.json"}


def _orphan_manual_state(tmp_path, run_kind="manual"):
    _write_state_file(tmp_path / "log" / _RUN_STATE_FILES[run_kind], run_kind,
                      status="running", pid=12345,
                      token="stale-token")


@pytest.mark.parametrize("run_kind", ["manual", "tuning"])
@pytest.mark.parametrize("match_name", ["MISSING", "MISMATCH"])
def test_hpo_start_blocked_when_reconcile_write_back_fails(
        tmp_path, monkeypatch, run_kind, match_name):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    _pin_identity(monkeypatch, match_name)
    _orphan_manual_state(tmp_path, run_kind)
    _fail_reconcile_write_back(monkeypatch)

    resp = _start(stack, study_id)
    assert resp.status_code == 503
    body = resp.json()
    assert body["error_code"] == "RUN_STATE_RECONCILE_FAILED"
    assert set(body) == {"error", "error_code", "next_action"}
    assert "disk full" not in resp.text
    assert "secret" not in resp.text
    assert stack.runner.run_calls == 0
    assert stack.manager.reservation_owner() is None
    assert stack.manager.active_hpo() is None


@pytest.mark.parametrize("match_name", ["MISSING", "MISMATCH"])
def test_hpo_resume_blocked_when_reconcile_write_back_fails(
        tmp_path, monkeypatch, match_name):
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    stack.runner._get(study_id).set("PAUSED", "user_stopped")
    _pin_identity(monkeypatch, match_name)
    _orphan_manual_state(tmp_path)
    _fail_reconcile_write_back(monkeypatch)

    resp = _resume(stack, study_id)
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "RUN_STATE_RECONCILE_FAILED"
    assert "disk full" not in resp.text
    assert stack.runner.resume_calls == 0
    assert stack.runner.run_calls == 0
    assert stack.manager.reservation_owner() is None


def test_hpo_start_allowed_when_reconcile_write_back_succeeds(tmp_path, monkeypatch):
    """The recovery path must not be permanently locked by this guard."""
    stack = _gated_stack(tmp_path, monkeypatch)
    study_id = stack.create().json()["study_id"]
    _pin_identity(monkeypatch, "MISSING")
    _orphan_manual_state(tmp_path)

    resp = _start(stack, study_id)
    assert resp.status_code == 202
    assert stack.runner.run_calls == 1
    state = read_run_state(str(tmp_path / "log" / "training_running.json"),
                          run_kind="manual")
    assert state.status == "interrupted"
    assert state.terminal_reason == "process_missing"
    stack.client.post(f"/api/hpo/studies/{study_id}/stop", json={})


# ── 第四轮：评价模式与组成指标的只读投影 ────────────────────────────

def _success(stack, study_id, value, evidence):
    trial = stack.service.ask(study_id, request_id=uuid.uuid4().hex)
    stack.service.tell(study_id, trial.number,
                       ResultInput(state="SUCCESS", value=value,
                                   evidence=evidence))
    return trial


def test_status_payload_reports_the_studys_evaluation_mode(stack):
    study_id = stack.create(
        study_config={"evaluation_mode": "comprehensive"}).json()["study_id"]
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["evaluation_mode"] == "comprehensive"
    assert body["objective"] == "comprehensive_composite_best_epoch_v1"
    assert body["search_space"]["evaluation_mode"] == "comprehensive"
    assert body["search_space"]["objective"] == \
        "comprehensive_composite_best_epoch_v1"
    # 评价方式说明必须来自服务端权重，不能仍写着“mAP50-95 单指标”
    score = body["search_space"]["score"]
    assert score["weights"] == {
        "metrics/mAP50(B)": 0.10, "metrics/mAP50-95(B)": 0.50,
        "metrics/precision(B)": 0.20, "metrics/recall(B)": 0.20}


def test_default_create_uses_the_comprehensive_mode(stack):
    """不传评价模式时按当前产品规则默认全面模式，绝不新建 legacy 研究。"""
    study_id = stack.create().json()["study_id"]
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["evaluation_mode"] == "comprehensive"
    assert body["objective"] == "comprehensive_composite_best_epoch_v1"
    assert body["search_space"]["score"]["weights"] == {
        "metrics/mAP50(B)": 0.10, "metrics/mAP50-95(B)": 0.50,
        "metrics/precision(B)": 0.20, "metrics/recall(B)": 0.20}


def test_create_rejects_the_legacy_mode_with_zero_studies(stack):
    """手工提交 legacy_map50_95：零创建 + 稳定字段错误，绝不新建 legacy 研究。"""
    resp = stack.create(study_config={"evaluation_mode": "legacy_map50_95"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "INVALID_HPO_FIELD"
    assert body["field"] == "evaluation_mode"
    assert body["reason_code"] == "FIELD_VALUE"
    assert body["error"] and body["next_action"]
    # 不泄漏选项原文/底层异常文本
    for marker in ("legacy_map50_95", "Pydantic", "Input should", "Traceback"):
        assert marker not in resp.text
    assert list(stack.root.glob("hpo_*")) == []
    assert stack.runner.run_calls == 0


def test_old_records_without_a_mode_still_load_and_keep_legacy_semantics(stack):
    """旧 hpo-study-v1 记录没有评价模式：读取/展示继续按旧单指标语义，且不被重写。"""
    study_id = stack.create().json()["study_id"]
    study_path = stack.root / study_id / "study.json"
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["config"].pop("evaluation_mode", None)
    raw["config"].pop("objective", None)
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    before = study_path.read_bytes()

    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    assert body["evaluation_mode"] == "legacy_map50_95"
    assert body["objective"] == "val_map50_95_best_epoch_v1"
    assert body["search_space"]["score"]["weights"] == {
        "metrics/mAP50-95(B)": 1.0}
    # 只读：旧 JSON 绝不被读回写、补齐或重算
    assert study_path.read_bytes() == before

    rows = stack.client.get("/api/hpo/studies?offset=0&limit=10").json()["studies"]
    row = {r["study_id"]: r for r in rows}[study_id]
    assert row["evaluation_mode"] == "legacy_map50_95"
    assert row["objective"] == "val_map50_95_best_epoch_v1"


def test_best_payload_exposes_mode_composite_and_component_metrics(stack):
    study_id = stack.create(
        study_config={"evaluation_mode": "quick", "epochs": 5}).json()["study_id"]
    _success(stack, study_id, 0.6, Evidence(
        run_id="run-q", artifact_relpath="results.csv",
        artifact_sha256="1" * 64, epoch=2,
        evaluation_mode="quick", objective="quick_composite_best_epoch_v1",
        metrics={"metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.6,
                 "metrics/precision(B)": 0.1, "metrics/recall(B)": 0.1}))
    body = stack.client.get(f"/api/hpo/studies/{study_id}").json()
    best = body["best"]
    assert best["value"] == 0.6
    assert best["evaluation_mode"] == "quick"
    assert best["objective"] == "quick_composite_best_epoch_v1"
    assert best["metrics"]["metrics/mAP50(B)"] == 0.6
    assert best["metrics"]["metrics/recall(B)"] == 0.1
    assert body["ranking"][0]["value"] == 0.6
    assert body["ranking"][0]["metrics"]["metrics/mAP50-95(B)"] == 0.6


def test_best_config_reports_the_evaluation_mode_and_components(stack):
    study_id = stack.create(
        study_config={"evaluation_mode": "quick"}).json()["study_id"]
    _success(stack, study_id, 0.5, Evidence(
        run_id="run-q", artifact_relpath="results.csv",
        artifact_sha256="1" * 64, epoch=3,
        evaluation_mode="quick", objective="quick_composite_best_epoch_v1",
        metrics={"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.5}))
    payload = stack.client.get(
        f"/api/hpo/studies/{study_id}/best-config").json()
    assert payload["value"] == 0.5
    assert payload["evaluation_mode"] == "quick"
    assert payload["objective"] == "quick_composite_best_epoch_v1"
    assert payload["metrics"]["metrics/mAP50(B)"] == 0.5
    # 旧字段仍然存在（固定配置验证接口兼容）
    assert set(payload["fixed"]) == {"epochs", "batch", "imgsz", "device"}


def test_study_list_reports_mode_and_dataset_name(stack):
    study_id = stack.create(
        study_config={"evaluation_mode": "quick"}).json()["study_id"]
    rows = stack.client.get("/api/hpo/studies?offset=0&limit=10").json()["studies"]
    row = {r["study_id"]: r for r in rows}[study_id]
    assert row["evaluation_mode"] == "quick"
    assert row["objective"] == "quick_composite_best_epoch_v1"
    # 数据集名称来自受控快照 manifest 的 source_root 基名，绝不返回绝对路径
    assert row["dataset_name"] == "source"


def test_study_list_never_uses_a_short_id_as_the_dataset_name(stack, tmp_path):
    """旧记录解析不出数据集名称时诚实缺失，绝不用 study/snapshot 短身份冒充。"""
    study_id = stack.create().json()["study_id"]
    # 让权威快照事实不可读（模拟被移动/删除的旧快照）
    snapshot_dir = stack.root.parent / "snapshots" / stack.snapshot.snapshot_id
    assert snapshot_dir.is_dir()
    import shutil
    shutil.rmtree(snapshot_dir)

    rows = stack.client.get("/api/hpo/studies?offset=0&limit=10").json()["studies"]
    row = {r["study_id"]: r for r in rows}[study_id]
    assert row["readable"] is True          # 研究记录本身仍可读
    assert row["dataset_name"] is None      # 诚实缺失，由界面显示“数据集不可用”
    assert row["dataset_name"] != study_id[:8]
    assert row["dataset_name"] != stack.snapshot.snapshot_id[:8]
    assert str(tmp_path) not in json.dumps(row)
    assert str(stack.root) not in str(row)


# ── 最终浏览器阻断项第二轮 Task 3：旧 study 的数据集名称 ─────────────


def _large_stack(tmp_path):
    """A legal published snapshot whose manifest is ~80 KiB (> the old 64 KiB cap)."""
    snapshot, model = _make_inputs(tmp_path, source_name="dataset_cegai_914v2",
                                   count=_LARGE_SNAPSHOT_SAMPLES, seed=7)
    return Stack(tmp_path, snapshot=snapshot, model=model)


def test_old_study_resolves_its_name_from_a_large_published_snapshot(tmp_path):
    """真实存在的旧快照（manifest 大于 64 KiB）必须解析出真实数据集名称。"""
    stack = _large_stack(tmp_path)
    manifest = Path(stack.snapshot.snapshot_path) / "manifest.json"
    assert manifest.stat().st_size > 65536        # 反例的前提：合法但很大
    study_id = stack.create().json()["study_id"]

    row = _row(stack, study_id)
    assert row["readable"] is True
    assert row["dataset_name"] == "dataset_cegai_914v2"
    # 名称只来自权威快照事实，不返回任何物理路径/内部身份
    assert "source_root" not in row
    assert "snapshot_path" not in row
    assert str(tmp_path) not in json.dumps(row)


def test_snapshot_projection_is_built_once_for_the_whole_page(tmp_path):
    """多条 study 绑定同一快照：一次 history 请求只构建一次映射。"""
    calls = []

    def listing():
        from auto_tune.ui.app import list_published_snapshots
        calls.append(1)
        return list_published_snapshots(tmp_path / "snapshots")

    stack = Stack(tmp_path, list_snapshots=listing)
    study_ids = {stack.create().json()["study_id"] for _ in range(3)}
    calls.clear()

    rows = stack.client.get("/api/hpo/studies?offset=0&limit=10").json()["studies"]
    assert {r["study_id"] for r in rows} == study_ids
    assert {r["dataset_name"] for r in rows} == {"source"}
    assert len(calls) == 1


def test_dataset_name_requires_an_exact_full_identity_match(tmp_path):
    """只接受完整 snapshot_id 精确匹配：短身份/大小写/空名称一律诚实缺失。"""
    stack = Stack(tmp_path)
    study_id = stack.create().json()["study_id"]
    full = stack.snapshot.snapshot_id
    stack.published_rows = [
        {"snapshot_id": full[:8], "dataset_name": "short-id-must-not-match"},
        {"snapshot_id": full.upper(), "dataset_name": "case-must-not-match"},
        {"snapshot_id": full, "dataset_name": ""},
        {"snapshot_id": full, "dataset_name": None},
        {"dataset_name": "no-identity"},
        "not-a-row",
    ]
    assert _row(stack, study_id)["dataset_name"] is None
    assert _row(stack, study_id)["dataset_name"] != full[:8]


def test_missing_or_unlistable_snapshots_are_honestly_missing(tmp_path):
    """快照不存在或权威列表不可用时诚实返回 None，且不影响研究记录本身。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    empty = Stack(tmp_path / "a", list_snapshots=lambda: [])
    empty_id = empty.create().json()["study_id"]
    assert _row(empty, empty_id)["dataset_name"] is None

    def boom():
        raise RuntimeError("snapshot root unavailable: E:/secret/path")

    broken = Stack(tmp_path / "b", list_snapshots=boom)
    broken_id = broken.create().json()["study_id"]
    resp = broken.client.get("/api/hpo/studies?offset=0&limit=10")
    assert resp.status_code == 200
    row = {r["study_id"]: r for r in resp.json()["studies"]}[broken_id]
    assert row["readable"] is True
    assert row["dataset_name"] is None
    assert "secret" not in resp.text


# ── F1.1-A Task 5：试验状态轨道与最近关键事件（纯只读投影）──────────


class _Attempt:
    """状态投影只读 attempt 的 phase/trial_number；不需要完整审计记录。"""

    def __init__(self, phase, trial_number):
        self.phase = phase
        self.trial_number = trial_number


_REASON_CODE = {
    "FAILED": "training_failed",
    "CANCELLED": "user_stopped",
    "INTERRUPTED": "process_interrupted",
}


def _run_trials(stack, study_id, states):
    """按给定终态创建试验：state 为 None 时只 ask（保持 PENDING）。"""
    numbers = []
    for state in states:
        trial = stack.service.ask(study_id, request_id=uuid.uuid4().hex)
        numbers.append(trial.number)
        if state == "SUCCESS":
            stack.service.tell(study_id, trial.number,
                               ResultInput(state="SUCCESS", value=0.5,
                                           evidence=_evidence(1)))
        elif state is not None:
            stack.service.tell(study_id, trial.number,
                               ResultInput(state=state,
                                           reason_code=_REASON_CODE[state]))
    return numbers


def _status(stack, study_id):
    resp = stack.client.get(f"/api/hpo/studies/{study_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_status_payload_exposes_the_trial_state_rail(stack):
    study_id = stack.create(study_config={"budget": 3}).json()["study_id"]
    _run_trials(stack, study_id, ["SUCCESS", None])
    stack.runner._get(study_id).attempts = [_Attempt("RUNNING", 1)]

    body = _status(stack, study_id)
    assert body["trial_states"] == [
        {"trial_number": 1, "state": "SUCCESS"},
        {"trial_number": 2, "state": "RUNNING"},
        {"trial_number": 3, "state": "WAITING"},
    ]


def test_trial_state_rail_covers_every_budget_slot_and_keeps_real_states(stack):
    study_id = stack.create(study_config={"budget": 5}).json()["study_id"]
    _run_trials(stack, study_id, ["SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"])

    body = _status(stack, study_id)
    assert [row["trial_number"] for row in body["trial_states"]] == [1, 2, 3, 4, 5]
    assert [row["state"] for row in body["trial_states"]] == [
        "SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED", "WAITING"]


def test_trial_state_rail_does_not_fabricate_a_started_trial(stack):
    study_id = stack.create(study_config={"budget": 2}).json()["study_id"]
    body = _status(stack, study_id)
    assert body["trial_states"] == [
        {"trial_number": 1, "state": "WAITING"},
        {"trial_number": 2, "state": "WAITING"},
    ]
    # 没有任何试验事实时绝不伪造“已开始”的试验事件
    assert all(row["kind"] != "trial_finished" for row in body["recent_events"])
    assert all(row["kind"] != "trial_running" for row in body["recent_events"])


def test_recent_events_are_bounded_deterministic_and_path_free(stack):
    study_id = stack.create(study_config={"budget": 4}).json()["study_id"]
    _run_trials(stack, study_id, ["SUCCESS", "SUCCESS", "FAILED"])
    stack.runner._get(study_id).set("PAUSED", "user_stopped")

    first = _status(stack, study_id)
    events = first["recent_events"]
    assert len(events) == 3
    for event in events:
        assert set(event) == {"event_id", "kind", "trial_number", "state",
                              "message"}
        assert isinstance(event["message"], str) and event["message"]
        assert not any(token in event["message"]
                       for token in ("C:\\", "E:/", "/", "Traceback"))
        assert event["event_id"] == event["event_id"].strip()
    # 事件 ID 不含时间：同一持久事实必须得到完全相同的投影
    assert first == _status(stack, study_id)
    # 研究终态事件排在最后（最近事件）
    assert events[-1]["kind"] == "study_status"
    assert events[-1]["state"] == "PAUSED"
    assert "暂停" in events[-1]["message"]


def test_progress_projection_is_read_only(stack):
    study_id = stack.create(study_config={"budget": 2}).json()["study_id"]
    _run_trials(stack, study_id, ["SUCCESS"])
    before = sorted((p.name, p.stat().st_size)
                    for p in (stack.root / study_id).rglob("*") if p.is_file())
    for _ in range(3):
        _status(stack, study_id)
    after = sorted((p.name, p.stat().st_size)
                   for p in (stack.root / study_id).rglob("*") if p.is_file())
    assert before == after
