"""H1.3 Task 5: best-configuration reuse via a fixed-config verification run.

The HPO source is real (HpoService/HpoRunner over tmp_path); the ordinary
training subprocess is faked so no YOLO runs. Verifies the six search params
and all fixed conditions reach the actual command/args, the bound initial
weights are used (not best.pt), and the HPO study/execution/ranking are never
modified.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    ExecutionConfig,
    HpoRunner,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.run_state.manager import RunManager

_SEARCH_KEYS = ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")


def _rid():
    return uuid.uuid4().hex


def _evidence(epoch=1):
    return Evidence(run_id=f"run-{_rid()}", artifact_relpath="results.csv",
                    artifact_sha256="0" * 64, epoch=epoch)


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
    model.write_bytes(b"hpo-reuse-test-not-a-real-model")
    return snapshot, model


class Source:
    """Real HPO source study with deterministic SUCCESS trials."""

    def __init__(self, tmp_path, values=(0.5, 0.9)):
        self.snapshot, self.model = _make_inputs(tmp_path)
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.study = self.service.create_study(
            StudyConfig(budget=5, epochs=10), snapshot_dir=self.snapshot.snapshot_path,
            model_path=self.model)
        self.runner = HpoRunner(self.root, tmp_path / "out", tmp_path / "log")
        self.runner.prepare(self.study.study_id,
                            ExecutionConfig(batch=4, imgsz=64, device="cpu",
                                            timeout_seconds=120))
        for value in values:
            trial = self.service.ask(self.study.study_id, request_id=_rid())
            self.service.tell(self.study.study_id, trial.number,
                              ResultInput(state="SUCCESS", value=value,
                                          evidence=_evidence(1)))
        trials = self.service.load_study(self.study.study_id).trials
        self.top = trials[-1] if trials else None

    @property
    def study_id(self):
        return self.study.study_id


class FakeProc:
    def __init__(self):
        class Stdout:
            async def readline(self):
                return b""

        self.stdout = Stdout()
        self.pid = os.getpid()
        self.returncode = 0

    async def wait(self):
        return 0


def _patch_app(monkeypatch, tmp_path, source):
    from auto_tune.ui import app as app_mod

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(os.path, "join", fake_join)
    monkeypatch.setattr(app_mod, "_hpo_service", source.service)
    monkeypatch.setattr(app_mod, "_hpo_runner", source.runner)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"))
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo")

    launched = []

    async def fake_subprocess_exec(*args, **kwargs):
        launched.append(list(args))
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    def fake_finalize(run_dir, run_name, source_kind, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name,
            "source": "manual", "status": "completed", "analysis_status": "skipped",
            "metrics": {}, "artifacts": {"report_path": None}, "error": None,
            "analysis_error": None, "history_error": None,
        }

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)
    app_mod._running_training.clear()
    return app_mod, launched


def _post(app_mod, payload):
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    return client.post("/api/training/start", json=payload)


@pytest.fixture
def source(tmp_path):
    return Source(tmp_path)


def test_reuse_builds_command_from_authoritative_params(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study = source.service.load_study(source.study_id)
    study_before = (source.root / source.study_id / "study.json").read_bytes()
    exec_before = (source.root / source.study_id / "execution.json").read_bytes()

    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": source.top.trial_id}})
    assert resp.status_code == 200
    assert launched, "verification training must start exactly once"
    cmd = launched[0]

    # six search params + fixed conditions actually reach the command
    top_params = source.top.candidate_params
    for key in _SEARCH_KEYS:
        assert any(tok == f"{key}={top_params[key]}" for tok in cmd), key
    assert f"model={study.model_binding.model_path}" in cmd
    assert "best.pt" not in " ".join(cmd)
    assert f"data={os.path.abspath(study.snapshot_binding.data_yaml_path)}" in cmd
    assert "epochs=10" in cmd
    assert "seed=42" in cmd
    assert "batch=4" in cmd
    assert "imgsz=64" in cmd
    assert "device=cpu" in cmd
    assert "task=detect" in cmd and "workers=0" in cmd

    # args.yaml mirrors the same effective config
    args_path = tmp_path / "detect" / "train1" / "args.yaml"
    import yaml

    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    for key in _SEARCH_KEYS:
        assert args[key] == top_params[key]
    assert args["model"] == study.model_binding.model_path
    assert args["batch"] == 4 and args["imgsz"] == 64 and args["device"] == "cpu"

    # source metadata recorded in the run dir
    src_meta = json.loads(
        (tmp_path / "detect" / "train1" / "hpo_source.json").read_text(encoding="utf-8"))
    assert src_meta["study_id"] == source.study_id
    assert src_meta["trial_id"] == source.top.trial_id

    # HPO facts untouched
    assert (source.root / source.study_id / "study.json").read_bytes() == study_before
    assert (source.root / source.study_id / "execution.json").read_bytes() == exec_before


def test_reuse_no_success_zero_start(tmp_path, monkeypatch):
    source = Source(tmp_path, values=())
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": source.study.study_id + "_t0000"}})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_NO_SUCCESS"
    assert launched == []


def test_reuse_non_rank_first_zero_start(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study = source.service.load_study(source.study_id)
    first_trial_id = study.trials[0].trial_id  # the 0.5 trial, not rank #1
    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": first_trial_id}})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
    assert launched == []


def test_reuse_forged_source_zero_start(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    forged = source.study_id + "_t9999"
    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": forged}})
    assert resp.status_code == 409
    assert launched == []


def test_reuse_mixed_params_rejected(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, {
        "source_hpo": {"study_id": source.study_id, "trial_id": source.top.trial_id},
        "epochs": 5,
    })
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "MIXED_TRAINING_PARAMS"
    assert launched == []


def test_reuse_binding_drift_zero_start(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    source.model.write_bytes(b"changed-model-content")  # initial weights changed
    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": source.top.trial_id}})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_BINDING_MISMATCH"
    assert launched == []


def test_reuse_tie_prefers_smaller_number(tmp_path, monkeypatch):
    source = Source(tmp_path, values=(0.5, 0.5))
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    study = source.service.load_study(source.study_id)
    top_id = study.trials[0].trial_id  # same score → smaller number wins
    ok = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": top_id}})
    assert ok.status_code == 200
    assert launched
    other = study.trials[1].trial_id
    again = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id, "trial_id": other}})
    assert again.status_code == 409


def test_reuse_blocked_by_other_active_training(tmp_path, monkeypatch, source):
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)

    class Stub:
        run_kind = "tuning"
        run_id = "tuning:stub"

        def is_active(self):
            return True

        def is_done(self):
            return False

    stub = Stub()
    app_mod._RUN_MANAGER.register(stub)
    try:
        resp = _post(app_mod, {"source_hpo": {
            "study_id": source.study_id, "trial_id": source.top.trial_id}})
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        assert launched == []
    finally:
        app_mod._RUN_MANAGER.unregister(stub.run_id)


def test_verification_blocked_by_inflight_training_zero_side_effects(
        tmp_path, monkeypatch, source):
    """An in-flight ordinary training must leave the Detect tree untouched."""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)

    class InFlightManual:
        run_kind = "manual"
        run_id = "manual:inflight"

        def is_active(self):
            return True

        def is_done(self):
            return False

    stub = InFlightManual()
    app_mod._RUN_MANAGER.register(stub)
    try:
        resp = _post(app_mod, {"source_hpo": {
            "study_id": source.study_id, "trial_id": source.top.trial_id}})
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "RUN_ALREADY_ACTIVE"
        assert launched == []
        assert not (tmp_path / "detect").exists()
        assert list(tmp_path.rglob("hpo_source.json")) == []
    finally:
        app_mod._RUN_MANAGER.unregister(stub.run_id)


def test_verification_busy_or_invalid_source_writes_nothing(tmp_path, monkeypatch, source):
    """Invalid/forged source: no run dir, no args.yaml, no source metadata."""
    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    resp = _post(app_mod, {"source_hpo": {
        "study_id": source.study_id,
        "trial_id": source.study_id + "_t9999"}})
    assert resp.status_code == 409
    assert launched == []
    assert not (tmp_path / "detect").exists()
    assert list(tmp_path.rglob("hpo_source.json")) == []
    assert list(tmp_path.rglob("args.yaml")) == []
    assert app_mod._RUN_MANAGER.reservation_owner() is None


def test_verification_blocked_when_reconcile_write_back_fails(
        tmp_path, monkeypatch, source):
    """A failed terminal write-back blocks the verification run too.

    The verification entry shares the ordinary-training gate, so an orphaned
    record that cannot be written back must stop it before any run directory,
    source metadata or process is created.
    """
    from auto_tune.modules.run_state import gate as gate_mod
    from auto_tune.modules.run_state import process_identity as identity_mod
    from auto_tune.modules.run_state.models import RunStatePersistenceError
    from auto_tune.modules.run_state.process_identity import IdentityMatch
    from auto_tune.modules.run_state.service import new_run_state, with_status_phase
    from auto_tune.modules.run_state.service import write_run_state as _svc_write

    app_mod, launched = _patch_app(monkeypatch, tmp_path, source)
    state = new_run_state("manual", run_name="train1")
    state = with_status_phase(state, status="running", phase="training",
                              pid=12345, process_create_token="stale-token")
    _svc_write(str(tmp_path / "log" / "training_running.json"), state)

    fake = lambda expected: IdentityMatch.MISSING  # noqa: E731
    monkeypatch.setattr(gate_mod, "compare_process_identity", fake)
    monkeypatch.setattr(identity_mod, "compare_process_identity", fake)

    def boom(*args, **kwargs):
        raise RunStatePersistenceError(
            r"disk full writing C:\secret\logs\training_running.json")

    monkeypatch.setattr(app_mod, "write_run_state", boom)
    try:
        resp = _post(app_mod, {"source_hpo": {
            "study_id": source.study_id, "trial_id": source.top.trial_id}})
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_code"] == "RUN_STATE_RECONCILE_FAILED"
        assert set(body) == {"error", "error_code", "next_action"}
        assert "disk full" not in resp.text
        assert "secret" not in resp.text
        assert launched == []
        assert not (tmp_path / "detect").exists()
        assert list(tmp_path.rglob("hpo_source.json")) == []
        assert app_mod._RUN_MANAGER.reservation_owner() is None
    finally:
        for controller in app_mod._RUN_MANAGER.snapshot():
            app_mod._RUN_MANAGER.unregister(controller.run_id)
        app_mod._RUN_MANAGER._reservation = None
