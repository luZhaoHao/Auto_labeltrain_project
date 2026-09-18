"""H1.3 追加返修 Task 3: 受控产物路由（Trial best.pt/last.pt 与正式训练 best.pt）。

只有合法 study/trial/train 身份 + 白名单产物名才能下载；服务端从冻结
``output_root`` 与权威 attempt 的 ``run_relpath``（或受控 train 目录）重建路径，
拒绝路径穿越、链接/重解析点、越根、身份不一致，也拒绝伪造 record 绕过。
"""

import json
import os
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
from auto_tune.modules.model_store import ModelStoreError
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.ui.hpo_api import create_hpo_router
from auto_tune.ui.hpo_training import create_hpo_training_router


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
    model.write_bytes(b"hpo-artifacts-test-not-a-real-model")
    return snapshot, model


class _Attempt:
    def __init__(self, trial_number, trial_id, run_relpath, phase="FINALIZED"):
        self.trial_number = trial_number
        self.trial_id = trial_id
        self.run_relpath = run_relpath
        self.phase = phase


class _Roots:
    def __init__(self, output_root):
        # the frozen contract stores an absolute string (ExecutionRoots)
        self.output_root = str(output_root)


class _ExecView:
    def __init__(self, study_id, output_root, attempts, status="COMPLETED"):
        self.study_id = study_id
        self.status = status
        self.config = ExecutionConfig(batch=4, imgsz=64, device="cpu")
        self.roots = _Roots(output_root)
        self.attempts = attempts
        self.revision = 1
        self.stop_reason = None


class RunnerDouble:
    """Read-only runner stand-in exposing roots + attempts."""

    def __init__(self, exec_view):
        self._view = exec_view
        self.run_calls = 0

    def status(self, study_id):
        assert study_id == self._view.study_id
        return self._view

    def prepare(self, study_id, config):
        return self._view


def _write_history(log_dir, run_name, *, status="completed", metrics=None,
                   train_dir=None):
    """把一条统一历史事实写进 ``log_dir``（正式训练状态的**权威**来源）。"""
    from auto_tune.modules.train_analyzer.experiment_history import (
        ExperimentHistoryStore,
        make_run_id,
    )

    record = {
        "run_id": make_run_id("manual", run_name),
        "run_name": run_name,
        "source": "manual",
        "status": status,
        "analysis_status": "completed",
        "started_at": "2026-09-16T00:00:00Z",
        "finished_at": "2026-09-16T00:10:00Z",
        "params": {},
        "metrics": ({"mAP50": 0.71, "mAP50_95": 0.44}
                    if metrics is None else metrics),
        "epochs": {"configured": 2, "completed": 2, "best": 2},
        "artifacts": {"report_path": None,
                      "run_dir": str(train_dir) if train_dir else None},
    }
    ExperimentHistoryStore(
        str(Path(log_dir) / "experiment_history.json")).upsert(record)


class _LiveController:
    """最小活动普通训练控制器替身：只提供投影需要的只读事实。"""

    def __init__(self, run_id, train_name, status="running"):
        self.run_id = run_id
        self.run_kind = "manual"
        self.train_name = train_name
        state = type("_State", (), {})()
        state.status = status
        state.run_id = run_id
        self.run_state = state
        self._active = True

    def is_active(self):
        return self._active


class Stack:
    def __init__(self, tmp_path):
        self.snapshot, self.model = _make_inputs(tmp_path)
        self.root = tmp_path / "storage"
        self.service = HpoService(self.root)
        self.study = self.service.create_study(
            StudyConfig(budget=3, epochs=5),
            snapshot_dir=self.snapshot.snapshot_path, model_path=self.model)
        trial = self.service.ask(self.study.study_id, request_id=_rid())
        self.service.tell(self.study.study_id, trial.number,
                          ResultInput(state="SUCCESS", value=0.9,
                                      evidence=_evidence(1)))
        self.trial = self.service.load_study(self.study.study_id).trials[0]

        self.output_root = tmp_path / "out"
        self.out_dir = self.output_root / self.study.study_id / self.trial.trial_id
        (self.out_dir / "weights").mkdir(parents=True)
        (self.out_dir / "weights" / "best.pt").write_bytes(b"trial-best-weights")
        (self.out_dir / "weights" / "last.pt").write_bytes(b"trial-last-weights")
        (self.out_dir / "results.csv").write_text("epoch,x\n1,2\n", encoding="utf-8")

        attempts = [_Attempt(0, self.trial.trial_id,
                             f"{self.study.study_id}/{self.trial.trial_id}")]
        self.exec_view = _ExecView(self.study.study_id, self.output_root, attempts)
        self.runner = RunnerDouble(self.exec_view)
        self.manager = RunManager()
        self.detect_dir = tmp_path / "detect"
        self.log_dir = tmp_path / "log"
        self.log_dir.mkdir(exist_ok=True)
        self.client = self._client()

    def _client(self):
        snapshot, model = self.snapshot, self.model
        service, runner, manager = self.service, self.runner, self.manager

        def resolve_snapshot(sid):
            if sid == snapshot.snapshot_id:
                return Path(snapshot.snapshot_path)
            raise HpoError("HPO_INVALID_CONFIG", "bad snapshot")

        def resolve_model(model_id):
            """受控模型标识 -> 冻结路径；这些用例只读产物，不做权重解析。"""
            raise ModelStoreError("MODEL_NOT_FOUND", "未找到该受控权重，请重新选择。")

        app = FastAPI()
        app.include_router(create_hpo_router(
            service=service, runner=runner, manager=manager,
            resolve_snapshot=resolve_snapshot, resolve_model=resolve_model,
            assert_training_slot_free=lambda: None), prefix="/api/hpo")
        app.include_router(create_hpo_training_router(
            service=lambda: service, runner=lambda: runner, manager=lambda: manager,
            detect_dir=lambda: str(self.detect_dir), log_dir=str(self.log_dir),
            build_deps=lambda: None), prefix="/api/hpo")
        return TestClient(app)

    def trial_url(self, name, study_id=None, trial_id=None):
        return (f"/api/hpo/studies/{study_id or self.study.study_id}"
                f"/trials/{trial_id or self.trial.trial_id}/artifacts/{name}")

    def formal_url(self, train_name, name, study_id=None):
        return (f"/api/hpo/studies/{study_id or self.study.study_id}"
                f"/formal-runs/{train_name}/artifacts/{name}")

    def make_formal_run(self, train_name="train1", study_id=None, mode="formal",
                        content=b"formal-best-weights", create_weights=True,
                        status="completed", history=True, runtime_run_id=None,
                        metrics=None):
        """建一条受控正式训练目录；默认附上 ``completed`` 的统一历史事实。

        ``history=False`` 表示没有任何运行事实（状态缺失），``status`` 用于构造
        非完成态事实；两者都只影响权威状态来源，不影响目录与来源 metadata。
        """
        train_dir = self.detect_dir / train_name
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "hpo_source.json").write_text(json.dumps({
            "mode": mode,
            "study_id": study_id or self.study.study_id,
            "trial_id": self.trial.trial_id,
            "trial_number": 0,
            "training_config": {"epochs": 2, "batch": 4, "imgsz": 96,
                                "device": "cpu"},
            "train_name": train_name,
            "runtime_run_id": runtime_run_id,
        }), encoding="utf-8")
        if create_weights:
            (train_dir / "weights").mkdir(exist_ok=True)
            (train_dir / "weights" / "best.pt").write_bytes(content)
        if history:
            _write_history(self.log_dir, train_name, status=status,
                           metrics=metrics, train_dir=train_dir)
        return train_dir


@pytest.fixture
def stack(tmp_path):
    return Stack(tmp_path)


# ── Trial 产物 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name,content", [
    ("best.pt", b"trial-best-weights"),
    ("last.pt", b"trial-last-weights"),
])
def test_trial_artifact_downloads_whitelisted_file(stack, name, content):
    resp = stack.client.get(stack.trial_url(name))
    assert resp.status_code == 200
    assert resp.content == content
    disposition = resp.headers.get("content-disposition", "")
    assert name in disposition
    assert str(stack.output_root) not in disposition


@pytest.mark.parametrize("name", [
    "results.csv", "args.yaml", "run_state.json", "best.pt.bak",
    "..%2F..%2Freport.json", "weights%2Fbest.pt", "train_best.pt", "",
])
def test_trial_artifact_rejects_non_whitelisted_names(stack, name):
    resp = stack.client.get(stack.trial_url(name))
    assert resp.status_code in (404, 405, 409, 422), name
    assert b"trial-best-weights" not in resp.content


def test_trial_artifact_unknown_study_and_trial(stack):
    assert stack.client.get(stack.trial_url(
        "best.pt", study_id="hpo_" + "0" * 32)).status_code in (404, 409)
    forged = (f"{stack.study.study_id}_t9999")
    resp = stack.client.get(stack.trial_url("best.pt", trial_id=forged))
    assert resp.status_code in (404, 409)
    assert b"trial-best-weights" not in resp.content


def test_trial_artifact_cross_study_trial_identity_rejected(stack):
    other = f"{stack.study.study_id}_t0000"
    resp = stack.client.get(stack.trial_url(
        "best.pt", study_id="hpo_" + "1" * 32, trial_id=other))
    assert resp.status_code in (404, 409)


def test_trial_artifact_missing_file_and_missing_attempt(stack):
    (stack.out_dir / "weights" / "best.pt").unlink()
    resp = stack.client.get(stack.trial_url("best.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"].startswith("HPO_ARTIFACT")

    stack.exec_view.attempts = []
    resp = stack.client.get(stack.trial_url("last.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_AVAILABLE"


def test_trial_artifact_malicious_run_relpath_never_served(stack, tmp_path):
    secret = tmp_path / "secret_best.pt"
    secret.write_bytes(b"secret-weights")
    # point the attempt outside the controlled output root
    stack.exec_view.attempts = [_Attempt(
        0, stack.trial.trial_id, "../secret_best.pt")]
    resp = stack.client.get(stack.trial_url("best.pt"))
    assert resp.status_code in (404, 409)
    assert b"secret-weights" not in resp.content


def test_trial_artifact_escape_attempt_within_relpath_not_served(stack, tmp_path):
    """A run_relpath that keeps the prefix but escapes via .. is refused."""
    outside = tmp_path / "outside"
    (outside / "weights").mkdir(parents=True)
    (outside / "weights" / "best.pt").write_bytes(b"escaped-weights")
    stack.exec_view.attempts = [_Attempt(
        0, stack.trial.trial_id,
        f"{stack.study.study_id}/{stack.trial.trial_id}/../../outside")]
    resp = stack.client.get(stack.trial_url("best.pt"))
    assert resp.status_code in (404, 409)
    assert b"escaped-weights" not in resp.content


def test_trial_artifact_symlink_not_served(stack, tmp_path):
    link = stack.out_dir / "weights" / "best.pt"
    real = stack.out_dir / "weights" / "real.pt"
    link.replace(real)
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        link.write_bytes(b"trial-best-weights")  # restore for other tests
        pytest.skip("symlink creation not permitted on this platform")
    resp = stack.client.get(stack.trial_url("best.pt"))
    assert resp.status_code in (404, 409, 422)
    assert resp.json()["error_code"].startswith("HPO_ARTIFACT")
    assert b"trial-best-weights" not in resp.content


def test_forged_execution_record_with_traversal_is_rejected(tmp_path):
    """A hand-written execution.json cannot bypass the identity rebuild."""
    snapshot, model = _make_inputs(tmp_path)
    root = tmp_path / "storage"
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2),
                                 snapshot_dir=snapshot.snapshot_path,
                                 model_path=model)
    runner = HpoRunner(root, tmp_path / "out", tmp_path / "log")
    runner.prepare(study.study_id, ExecutionConfig())
    target = root / study.study_id / "execution.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["status"] = "COMPLETED"
    data["attempts"] = [{
        "trial_number": 0,
        "trial_id": f"{study.study_id}_t0000",
        "request_id": "0" * 32,
        "run_id": "tuning:" + uuid.uuid4().hex,
        "phase": "FINALIZED",
        "candidate_params": {}, "effective_params": {},
        "command": ["yolo", "train"], "command_executable": "yolo",
        "run_relpath": "../../secret",          # traversal attempt
        "args_sha256": "0" * 64,
        "result": {"state": "SUCCESS", "value": 0.5,
                   "evidence": {"run_id": "r", "artifact_relpath": "results.csv",
                                "artifact_sha256": "0" * 64, "epoch": 1}},
    }]
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HpoError) as err:
        runner.status(study.study_id)
    assert err.value.code in ("HPO_CORRUPT_EXECUTION", "HPO_CORRUPT_STUDY")


# ── 正式训练产物 ───────────────────────────────────────────────────


def test_formal_artifact_downloads_bound_run_best_pt(stack):
    stack.make_formal_run("train1", content=b"formal-best-weights")
    resp = stack.client.get(stack.formal_url("train1", "best.pt"))
    assert resp.status_code == 200
    assert resp.content == b"formal-best-weights"
    disposition = resp.headers.get("content-disposition", "")
    assert "best.pt" in disposition
    # the download name never reveals the controlled directory layout
    assert "weights" not in disposition and str(stack.detect_dir) not in disposition


def test_formal_artifact_requires_matching_study_binding(stack):
    stack.make_formal_run("train1", study_id="hpo_" + "2" * 32)
    resp = stack.client.get(stack.formal_url("train1", "best.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"


def test_formal_artifact_rejects_verification_mode(stack):
    stack.make_formal_run("train1", mode="verification")
    resp = stack.client.get(stack.formal_url("train1", "best.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"


def test_formal_artifact_missing_metadata_or_file(stack):
    resp = stack.client.get(stack.formal_url("train9", "best.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_AVAILABLE"

    stack.make_formal_run("train1", create_weights=False)
    resp = stack.client.get(stack.formal_url("train1", "best.pt"))
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_AVAILABLE"


@pytest.mark.parametrize("train_name", [
    "train1/../../etc", "..", ".", "train", "train1x", "autotune_1",
    "train1%2F..%2F..%2Fsecret", "C%3A", "train-1",
])
def test_formal_artifact_rejects_traversal_train_names(stack, train_name):
    resp = stack.client.get(stack.formal_url(train_name, "best.pt"))
    assert resp.status_code in (404, 409, 422), train_name


@pytest.mark.parametrize("name", ["results.csv", "args.yaml", "hpo_source.json",
                                  "best.pt.bak", "..%2Fhpo_source.json"])
def test_formal_artifact_rejects_non_whitelisted_names(stack, name):
    stack.make_formal_run("train1")
    resp = stack.client.get(stack.formal_url("train1", name))
    assert resp.status_code in (404, 409, 422), name
    assert b"formal-best-weights" not in resp.content


def test_formal_artifact_unknown_study_not_served(stack):
    stack.make_formal_run("train1")
    resp = stack.client.get(stack.formal_url(
        "train1", "best.pt", study_id="hpo_" + "0" * 32))
    assert resp.status_code in (404, 409)


def test_formal_artifact_requested_arbitrary_path_is_refused(stack, tmp_path):
    """The client can never pass a path; only an identity + whitelisted name."""
    secret = tmp_path / "outside.pt"
    secret.write_bytes(b"outside-weights")
    for bad in ("..%2Foutside.pt", "%2E%2E%2Foutside.pt", "C%3A%5Coutside.pt"):
        resp = stack.client.get(stack.formal_url("train1", bad))
        assert resp.status_code in (404, 409, 422), bad
        assert b"outside-weights" not in resp.content


# ── 第四轮：受控“打开调优结果文件夹” ────────────────────────────────

def _folder_url(study_id=None):
    return (f"/api/hpo/studies/{study_id or 'hpo_' + '0' * 32}"
            "/best/open-folder")


class OpenerRecorder:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, path):
        self.calls.append(str(path))
        if self.error is not None:
            raise self.error


@pytest.fixture
def folder_stack(stack):
    """Attach an injectable opener recorder to the artifacts stack."""
    recorder = OpenerRecorder()
    stack.opener = recorder
    service, runner, manager = stack.service, stack.runner, stack.manager
    app = FastAPI()
    app.include_router(create_hpo_training_router(
        service=lambda: service, runner=lambda: runner, manager=lambda: manager,
        detect_dir=lambda: str(stack.detect_dir), log_dir=str(stack.log_dir),
        build_deps=lambda: None, opener=lambda: recorder), prefix="/api/hpo")
    stack.folder_client = TestClient(app)
    return stack


def test_open_folder_reveals_only_the_authoritative_rank1_run_dir(folder_stack):
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["opened"] is True
    assert body["study_id"] == folder_stack.study.study_id
    assert body["trial_id"] == folder_stack.trial.trial_id
    # 只打开权威 rank-1 试验的受控运行目录，且调用参数来自服务端解析
    assert folder_stack.opener.calls == [str(folder_stack.out_dir)]
    # 响应里绝不出现任何绝对路径
    assert str(folder_stack.output_root) not in resp.text
    assert str(folder_stack.out_dir) not in resp.text


def test_open_folder_rejects_a_client_supplied_path(folder_stack):
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder",
        json={"path": str(folder_stack.out_dir)})
    assert resp.status_code == 422
    assert folder_stack.opener.calls == []


@pytest.mark.parametrize("status", ["READY", "RUNNING", "PAUSED", "BLOCKED"])
def test_open_folder_requires_a_completed_study(folder_stack, status):
    folder_stack.exec_view.status = status
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_SOURCE_INVALID"
    assert folder_stack.opener.calls == []


def test_open_folder_requires_a_successful_trial(tmp_path):
    """没有成功试验时不能打开任何目录（零系统调用）。"""
    from auto_tune.modules.hpo import ResultInput

    snapshot, model = _make_inputs(tmp_path)
    service = HpoService(tmp_path / "storage")
    study = service.create_study(
        StudyConfig(budget=3, epochs=5),
        snapshot_dir=snapshot.snapshot_path, model_path=model)
    trial = service.ask(study.study_id, request_id=_rid())
    service.tell(study.study_id, trial.number,
                 ResultInput(state="FAILED", reason_code="training_failed"))

    output_root = tmp_path / "out"
    run_dir = output_root / study.study_id / trial.trial_id
    run_dir.mkdir(parents=True)
    exec_view = _ExecView(study.study_id, output_root,
                          [_Attempt(0, trial.trial_id,
                                    f"{study.study_id}/{trial.trial_id}")])
    recorder = OpenerRecorder()
    app = FastAPI()
    app.include_router(create_hpo_training_router(
        service=lambda: service, runner=lambda: RunnerDouble(exec_view),
        manager=lambda: RunManager(), detect_dir=lambda: str(tmp_path / "detect"),
        log_dir=str(tmp_path / "log"), build_deps=lambda: None,
        opener=lambda: recorder), prefix="/api/hpo")
    client = TestClient(app)
    resp = client.post(
        f"/api/hpo/studies/{study.study_id}/best/open-folder", json={})
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_NO_SUCCESS"
    assert recorder.calls == []


@pytest.mark.parametrize("study_id", [
    "not-a-study", "hpo_short", "hpo_" + "0" * 32,
])
def test_open_folder_unknown_or_illegal_study_is_zero_system_calls(folder_stack, study_id):
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{study_id}/best/open-folder", json={})
    assert resp.status_code in (404, 422)
    assert folder_stack.opener.calls == []


@pytest.mark.parametrize("relpath", [
    "../../../outside",
    "hpo_ffffffffffffffffffffffffffffffff/t0000",
])
def test_open_folder_rejects_identity_and_path_escapes(folder_stack, relpath):
    folder_stack.exec_view.attempts = [_Attempt(0, folder_stack.trial.trial_id, relpath)]
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert resp.status_code in (409, 422)
    assert resp.json()["error_code"] in ("HPO_ARTIFACT_IDENTITY_MISMATCH",
                                         "HPO_ARTIFACT_INVALID")
    assert folder_stack.opener.calls == []
    assert str(folder_stack.output_root) not in resp.text


def test_open_folder_reports_an_unsupported_platform_without_leaking_paths(folder_stack):
    folder_stack.opener.error = NotImplementedError("no shell")
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error_code"] == "HPO_ARTIFACT_OPEN_UNSUPPORTED"
    assert set(body) == {"error_code", "error", "next_action"}
    assert str(folder_stack.output_root) not in resp.text
    assert len(folder_stack.opener.calls) == 1


def test_open_folder_reports_a_shell_failure_with_a_stable_error(folder_stack):
    folder_stack.opener.error = OSError("explorer exploded")
    resp = folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "HPO_ARTIFACT_OPEN_FAILED"
    assert "exploded" not in resp.text
    assert str(folder_stack.output_root) not in resp.text


def test_open_folder_is_read_only_for_the_study_facts(folder_stack):
    study_path = folder_stack.root / folder_stack.study.study_id / "study.json"
    before = study_path.read_bytes()
    folder_stack.folder_client.post(
        f"/api/hpo/studies/{folder_stack.study.study_id}/best/open-folder", json={})
    assert study_path.read_bytes() == before


# ── 第五轮：受控“打开正式训练结果文件夹” ─────────────────────────────
#
# 语义与“打开调优结果文件夹”完全不同：后者打开 HPO 搜索阶段 rank-1 试验目录，
# 这里打开的是**当前研究关联的正式训练**的受控 ``detect/trainN`` 目录。身份只能
# 来自 路径参数(study_id/train_name)，路径一律由服务端从受控 detect 根重建。

def _formal_folder_url(study_id, train_name):
    return f"/api/hpo/studies/{study_id}/formal-runs/{train_name}/open-folder"


@pytest.fixture
def formal_folder_stack(stack):
    """正式训练“打开结果文件夹”栈：注入可记录 opener，绝不真的启动资源管理器。

    它是 ``stack`` 的同一个实例（测试同时请求两个夹具时拿到同一对象），只是额外
    挂上注入 opener 的客户端，因此断言仍然作用在同一个受控 detect/study 上。
    """
    recorder = OpenerRecorder()
    stack.formal_opener = recorder
    service, runner, manager = stack.service, stack.runner, stack.manager
    app = FastAPI()
    app.include_router(create_hpo_training_router(
        service=lambda: service, runner=lambda: runner, manager=lambda: manager,
        detect_dir=lambda: str(stack.detect_dir), log_dir=str(stack.log_dir),
        build_deps=lambda: None, opener=lambda: recorder), prefix="/api/hpo")
    stack.formal_folder_client = TestClient(app)
    assert stack.formal_opener is recorder
    return stack


def _open_formal_folder(stack, train_name, study_id=None, body=None):
    return stack.formal_folder_client.post(
        _formal_folder_url(study_id or stack.study.study_id, train_name),
        json={} if body is None else body)


def test_open_formal_folder_opens_only_the_controlled_train_dir(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["opened"] is True
    assert body["study_id"] == stack.study.study_id
    assert body["train_name"] == "train1"
    assert body == {"study_id": stack.study.study_id, "train_name": "train1",
                    "opened": True, "error_code": None, "error": None,
                    "next_action": None}
    # 只打开服务端从受控 detect 根重建的目录
    assert stack.formal_opener.calls == [str(stack.detect_dir / "train1")]
    # 绝不打开搜索阶段的 rank-1 试验目录，也绝不回显任何绝对路径
    assert str(stack.out_dir) not in resp.text
    assert str(stack.detect_dir) not in resp.text
    assert str(stack.output_root) not in resp.text


def test_open_formal_folder_never_targets_the_search_rank1_dir(
        stack, formal_folder_stack):
    """正式训练入口只认 detect/trainN；搜索阶段的 rank-1 目录不属于它。"""
    stack.make_formal_run("train1")
    assert stack.detect_dir.resolve() != stack.out_dir.resolve()
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 200
    assert stack.formal_opener.calls == [str(stack.detect_dir / "train1")]
    assert str(stack.out_dir) not in stack.formal_opener.calls[0]


def test_open_formal_folder_requires_a_controlled_linked_run_only(
        stack, formal_folder_stack):
    """同名目录不属于当前研究时拒绝：另一研究的正式训练绝不代开。"""
    other_study = "hpo_" + "9" * 32
    stack.make_formal_run("train1")
    stack.make_formal_run("train2", study_id=other_study)
    resp = _open_formal_folder(stack, "train2")
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_rejects_a_client_supplied_path(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    for body in ({"path": str(stack.detect_dir / "train1")},
                 {"run_dir": str(stack.detect_dir / "train1")},
                 {"train_name": "train1"},
                 {"study_id": stack.study.study_id}):
        resp = _open_formal_folder(stack, "train1", body=body)
        assert resp.status_code == 422, body
        assert stack.formal_opener.calls == []
        assert str(stack.detect_dir) not in resp.text


@pytest.mark.parametrize("train_name", [
    "train1/../../etc", "..", ".", "train", "train1x", "autotune_1",
    "train-1", "C%3A", "%2E%2E%2Fsecret",
])
def test_open_formal_folder_rejects_traversal_train_names(
        stack, formal_folder_stack, train_name):
    stack.make_formal_run("train1")
    resp = _open_formal_folder(stack, train_name)
    assert resp.status_code in (404, 405, 409, 422), train_name
    assert stack.formal_opener.calls == []
    assert str(stack.detect_dir) not in resp.text


@pytest.mark.parametrize("train_name", ["train", "train1x", "..", "."])
def test_open_formal_folder_illegal_names_are_refused_by_the_resolver(stack,
                                                                     train_name):
    """即使绕过路由匹配，解析层也必须拒绝非法训练编号（零系统调用）。"""
    from auto_tune.ui.hpo_training import HpoArtifactError, open_formal_run_folder

    stack.make_formal_run("train1")
    recorder = OpenerRecorder()
    with pytest.raises(HpoArtifactError) as err:
        open_formal_run_folder(stack.detect_dir, stack.study.study_id, train_name,
                               status_resolver=lambda name: "completed",
                               opener=recorder)
    assert err.value.code == "HPO_ARTIFACT_INVALID"
    assert recorder.calls == []


def test_open_formal_folder_rejects_a_mismatched_study_binding(
        stack, formal_folder_stack):
    stack.make_formal_run("train1", study_id="hpo_" + "2" * 32)
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_rejects_verification_mode_metadata(
        stack, formal_folder_stack):
    stack.make_formal_run("train1", mode="verification")
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_rejects_metadata_recording_another_directory(
        stack, formal_folder_stack):
    """目录名与来源记录声明的 train_name 不一致 → 不是这条正式训练。"""
    stack.make_formal_run("train5")
    meta_path = stack.detect_dir / "train5" / "hpo_source.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["train_name"] = "train1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    resp = _open_formal_folder(stack, "train5")
    assert resp.status_code in (404, 409)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_IDENTITY_MISMATCH"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_missing_metadata_or_directory(
        stack, formal_folder_stack):
    resp = _open_formal_folder(stack, "train9")           # no such directory
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_AVAILABLE"
    assert stack.formal_opener.calls == []

    (stack.detect_dir / "train9").mkdir(parents=True)     # directory, no metadata
    resp = _open_formal_folder(stack, "train9")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_AVAILABLE"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_unknown_or_illegal_study_is_zero_system_calls(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    for study_id in ("not-a-study", "hpo_short", "hpo_" + "0" * 32):
        resp = _open_formal_folder(stack, "train1", study_id=study_id)
        assert resp.status_code in (404, 422), study_id
        assert stack.formal_opener.calls == []


def test_open_formal_folder_refuses_a_link_like_train_dir(
        stack, formal_folder_stack):
    """链接/junction/reparse 目录绝不打开（否则可逃出受控 detect 根）。"""
    stack.make_formal_run("train1")
    link = stack.detect_dir / "train2"
    try:
        os.symlink(stack.detect_dir / "train1", link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("directory symlinks are not permitted on this platform")
    resp = _open_formal_folder(stack, "train2")
    assert resp.status_code in (404, 409, 422)
    assert resp.json()["error_code"].startswith("HPO_ARTIFACT")
    assert stack.formal_opener.calls == []


def test_open_formal_folder_can_be_opened_without_a_best_pt(
        stack, formal_folder_stack):
    """正式训练已完成但 best.pt 缺失时，结果目录仍然可以打开。"""
    stack.make_formal_run("train1", create_weights=False)
    assert not (stack.detect_dir / "train1" / "weights" / "best.pt").is_file()
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 200
    assert stack.formal_opener.calls == [str(stack.detect_dir / "train1")]


def test_open_formal_folder_reports_a_shell_failure_with_a_stable_error(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    stack.formal_opener.error = OSError("explorer exploded")
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "HPO_ARTIFACT_OPEN_FAILED"
    assert set(body) == {"error_code", "error", "next_action"}
    assert "exploded" not in resp.text
    assert "OSError" not in resp.text
    assert str(stack.detect_dir) not in resp.text
    assert len(stack.formal_opener.calls) == 1


def test_open_formal_folder_reports_an_unsupported_platform(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    stack.formal_opener.error = NotImplementedError("no shell")
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409
    body = resp.json()
    assert body["error_code"] == "HPO_ARTIFACT_OPEN_UNSUPPORTED"
    assert set(body) == {"error_code", "error", "next_action"}
    assert str(stack.detect_dir) not in resp.text


def test_open_formal_folder_is_read_only_for_the_run_facts(
        stack, formal_folder_stack):
    stack.make_formal_run("train1")
    train_dir = stack.detect_dir / "train1"
    study_path = stack.root / stack.study.study_id / "study.json"
    before = (study_path.read_bytes(),
              (train_dir / "hpo_source.json").read_bytes())
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 200
    # 打开目录只读：研究与来源事实逐字节不变，目录内容也不被改写
    assert (study_path.read_bytes(),
            (train_dir / "hpo_source.json").read_bytes()) == before
    assert sorted(p.name for p in train_dir.iterdir()) == ["hpo_source.json",
                                                          "weights"]


def test_best_pt_download_still_refuses_a_mismatched_metadata_train_name(stack):
    """共享的受控目录校验对下载入口同样生效（不放宽既有安全边界）。"""
    stack.make_formal_run("train5")
    meta_path = stack.detect_dir / "train5" / "hpo_source.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["train_name"] = "train1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    resp = stack.client.get(stack.formal_url("train5", "best.pt"))
    assert resp.status_code in (404, 409)
    assert b"formal-best-weights" not in resp.content


# ── 第六轮：打开正式训练目录的服务端完成态门控 ───────────────────────
#
# 根因：路由此前只校验研究存在与受控目录身份，完成态**只靠前端限制**；直接 POST
# 就能打开 running/失败/状态未知的正式训练目录。门控必须由服务端用与
# ``/formal-runs`` 完全相同的权威投影判定，客户端提交的状态一律不作数。

def test_open_formal_folder_requires_a_completed_formal_run(
        stack, formal_folder_stack):
    """完成态之外的权威状态一律拒绝，且 opener 零调用。"""
    stack.make_formal_run("train1", status="running")
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert set(body) == {"error_code", "error", "next_action"}
    assert body["error"].strip() and body["next_action"].strip()
    assert "完成" in body["error"] and "完成" in body["next_action"]
    assert stack.formal_opener.calls == []
    assert str(stack.detect_dir) not in resp.text


@pytest.mark.parametrize("status", [
    "running", "starting", "accepted", "stopping", "failed", "cancelled",
    "interrupted", "unknown", "",
])
def test_open_formal_folder_rejects_every_non_completed_status(
        stack, formal_folder_stack, status):
    stack.make_formal_run("train1", status=status)
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409, (status, resp.text)
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_rejects_a_run_without_a_status_fact(
        stack, formal_folder_stack):
    """投影里找不到该 train_name 的合法运行事实（状态缺失）时拒绝。"""
    stack.make_formal_run("train1", history=False)
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_rejects_a_failed_run_that_left_weights(
        stack, formal_folder_stack):
    """失败/停止的运行可能留下 best.pt，目录存在与权重存在都不等于完成。"""
    train_dir = stack.make_formal_run("train1", status="failed")
    assert (train_dir / "weights" / "best.pt").is_file()
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_uses_the_live_controller_status(
        stack, formal_folder_stack):
    """活动控制器在跑时，权威状态是 running（历史里的 completed 不作数）。"""
    stack.make_formal_run("train1", status="completed")
    controller = _LiveController("manual:" + str(uuid.uuid4()), "train1",
                                 status="running")
    stack.manager.register(controller)
    resp = _open_formal_folder(stack, "train1")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert stack.formal_opener.calls == []


def test_open_formal_folder_never_borrows_another_runs_status(
        stack, formal_folder_stack):
    """同名以外的已完成运行不得把状态借给未完成的请求行。"""
    stack.make_formal_run("train1", status="completed")
    stack.make_formal_run("train2", status="running")
    resp = _open_formal_folder(stack, "train2")
    assert resp.status_code == 409, resp.text
    assert resp.json()["error_code"] == "HPO_ARTIFACT_NOT_COMPLETED"
    assert stack.formal_opener.calls == []

    ok = _open_formal_folder(stack, "train1")     # 相邻的已完成运行不受影响
    assert ok.status_code == 200, ok.text
    assert stack.formal_opener.calls == [str(stack.detect_dir / "train1")]


def test_open_formal_folder_rejects_a_run_missing_from_the_projection(
        stack, formal_folder_stack):
    """投影中不存在该 train_name 时绝不按同名或最新记录猜测。"""
    stack.make_formal_run("train1", status="completed")
    resp = _open_formal_folder(stack, "train9")
    assert resp.status_code in (404, 409), resp.text
    assert stack.formal_opener.calls == []


def test_open_formal_run_folder_gates_completion_at_the_resolver_layer(
        stack, formal_folder_stack):
    """绕过 HTTP 直接调用解析层同样受完成态门控约束（零系统调用）。"""
    from auto_tune.ui.hpo_training import HpoArtifactError, open_formal_run_folder

    stack.make_formal_run("train1", status="running")
    recorder = OpenerRecorder()
    for status in ("running", "failed", "stopping", "unknown", None, ""):
        with pytest.raises(HpoArtifactError) as err:
            open_formal_run_folder(
                stack.detect_dir, stack.study.study_id, "train1",
                status_resolver=lambda name, s=status: s, opener=recorder)
        assert err.value.code == "HPO_ARTIFACT_NOT_COMPLETED", status
        assert recorder.calls == []
    # 只有完成态才进入系统调用
    open_formal_run_folder(
        stack.detect_dir, stack.study.study_id, "train1",
        status_resolver=lambda name: "completed", opener=recorder)
    assert recorder.calls == [str(stack.detect_dir / "train1")]


def test_open_formal_folder_ignores_a_client_supplied_status(
        stack, formal_folder_stack):
    """客户端提交的状态字段一律拒绝：请求体只能是空 JSON 对象。"""
    stack.make_formal_run("train1", status="running")
    for body in ({"status": "completed"}, {"state": "completed"},
                 {"opened": True}, {"run_dir": "train1"}):
        resp = _open_formal_folder(stack, "train1", body=body)
        assert resp.status_code == 422, body
        assert stack.formal_opener.calls == []
        assert str(stack.detect_dir) not in resp.text
