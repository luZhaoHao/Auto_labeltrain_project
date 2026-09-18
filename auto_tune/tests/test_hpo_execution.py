"""H1.2 顺序执行与幂等收尾测试 — 假适配器/进程，无真实训练/LLM/网络。

覆盖预算不超额、失败后续跑、全失败、run 幂等、run(PAUSED) 冲突、prepare 配置
冲突、status 不触发执行、finalizer analysis/history/index 三类失败、写盘注入
零后续 launch、确认进程退出等。
"""

import threading
import uuid
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    HpoError,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.hpo.execution import HpoRunner, execution_request_id
from auto_tune.modules.hpo.execution_adapter import CollectedOutcome
from auto_tune.modules.hpo.execution_models import ExecutionConfig, MetricDiagnostics
from auto_tune.modules.hpo.models import ResultPayload
from auto_tune.modules.hpo.storage import StudyStore


def _rid():
    return uuid.uuid4().hex


@pytest.fixture
def hpo_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-exec-test-not-a-real-model")
    return snapshot.snapshot_path, model


class FakeProc:
    _seq = 4000

    def __init__(self, rc, pid=None):
        FakeProc._seq += 1
        self.pid = pid if pid is not None else FakeProc._seq
        self._rc = rc
        self.terminated = False
        self.killed = False
        self._waited = False

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        self._waited = True
        return self._rc

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakeAdapter:
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
        # 假适配器不写真实 args.yaml；完整启动校验由真实 ExecutionAdapter 覆盖。
        return None

    def launch(self, prepared):
        FakeAdapter.launch_count += 1
        rc = FakeAdapter.returncodes[
            FakeAdapter.launch_count - 1
            if FakeAdapter.launch_count <= len(FakeAdapter.returncodes)
            else -1]
        return FakeProc(rc)

    def collect(self, study, attempt):
        if FakeAdapter.collect_result is None:
            value = 0.7
            evidence = Evidence(
                run_id=attempt.run_id,
                artifact_relpath=f"{study.study_id}/"
                                 f"{attempt.run_relpath.split('/')[-1]}/results.csv",
                artifact_sha256="0" * 64, epoch=1)
            return CollectedOutcome(
                result=ResultInput(state="SUCCESS", value=value, evidence=evidence),
                diagnostics=MetricDiagnostics(total_rows=1, excluded_rows=0))
        return FakeAdapter.collect_result

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
    FakeAdapter.finalize_result = None


@pytest.fixture
def ctx(tmp_path, hpo_inputs, monkeypatch, reset_fake):
    snapshot, model = hpo_inputs
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    service = HpoService(tmp_path / "hpo")
    study = service.create_study(StudyConfig(budget=2, epochs=30),
                                 snapshot_dir=snapshot, model_path=model)
    runner = HpoRunner(tmp_path / "hpo", tmp_path / "out", tmp_path / "log")
    runner.prepare(study.study_id, ExecutionConfig(batch=2, imgsz=64,
                                                   device="cpu",
                                                   timeout_seconds=300))
    return service, study, runner, tmp_path


def _cfg(**kw):
    data = dict(ExecutionConfig(batch=2, imgsz=64, device="cpu",
                                timeout_seconds=300).model_dump())
    data.update(kw)
    return ExecutionConfig(**data)


# ── budget 不超额：首败次成 ──────────────────────────────────────

def test_run_first_failed_then_success_budget_exact(ctx):
    service, study, runner, tmp_path = ctx
    FakeAdapter.returncodes = [1, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    assert FakeAdapter.launch_count == 2
    loaded = service.load_study(study.study_id)
    assert [t.state for t in loaded.trials] == ["FAILED", "SUCCESS"]
    assert loaded.trials[0].result.reason_code == "training_failed"
    assert loaded.trials[1].result.value == 0.7
    assert len(record.attempts) == 2
    assert all(a.phase == "FINALIZED" for a in record.attempts)
    # 不超过预算：没有第三次 ask/launch。
    assert FakeAdapter.launch_count == len(loaded.trials)


def test_run_all_failed_budget_consumed(ctx):
    service, study, runner, tmp_path = ctx
    FakeAdapter.returncodes = [1, 2]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    assert FakeAdapter.launch_count == 2
    loaded = service.load_study(study.study_id)
    assert [t.state for t in loaded.trials] == ["FAILED", "FAILED"]


# ── run 幂等 / status 只读 ───────────────────────────────────────

def test_run_second_time_is_idempotent_no_new_launch(ctx):
    service, study, runner, tmp_path = ctx
    FakeAdapter.returncodes = [1, 0]
    first = runner.run(study.study_id)
    launches = FakeAdapter.launch_count
    second = runner.run(study.study_id)
    assert FakeAdapter.launch_count == launches
    assert second.status == "COMPLETED"
    assert second.model_dump(mode="json") == first.model_dump(mode="json")


def test_status_triggers_no_execution(ctx):
    service, study, runner, tmp_path = ctx
    before = FakeAdapter.launch_count
    record = runner.status(study.study_id)
    assert record.status == "READY"
    assert FakeAdapter.launch_count == before


def test_run_on_paused_requires_resume(ctx):
    service, study, runner, tmp_path = ctx
    stop = threading.Event()
    stop.set()
    paused = runner.run(study.study_id, stop_event=stop)
    assert paused.status == "PAUSED"
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_EXECUTION_CONFLICT"


def test_resume_after_paused_runs_remaining_budget(ctx):
    service, study, runner, tmp_path = ctx
    stop = threading.Event()
    # 第一次 run 在第一次 ask 前停止 → 无 trial 消耗。
    stop.set()
    paused = runner.run(study.study_id, stop_event=stop)
    assert paused.status == "PAUSED"
    FakeAdapter.returncodes = [1, 0]
    stop.clear()
    resumed = runner.resume(study.study_id)
    assert resumed.status == "COMPLETED"
    assert FakeAdapter.launch_count == 2


# ── prepare 冲突 ─────────────────────────────────────────────────

def test_prepare_rejects_changed_config(ctx):
    service, study, runner, tmp_path = ctx
    with pytest.raises(HpoError) as err:
        runner.prepare(study.study_id, _cfg(batch=8))
    assert err.value.code == "HPO_EXECUTION_CONFLICT"


def test_prepare_rejects_study_with_terminal_trials(tmp_path, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(tmp_path / "hpo")
    study = service.create_study(StudyConfig(budget=1, epochs=30),
                                 snapshot_dir=snapshot, model_path=model)
    trial = service.ask(study.study_id, request_id=_rid())
    service.tell(study.study_id, trial.number,
                 ResultInput(state="FAILED", reason_code="training_failed"))
    runner = HpoRunner(tmp_path / "hpo", tmp_path / "out", tmp_path / "log")
    with pytest.raises(HpoError) as err:
        runner.prepare(study.study_id, _cfg())
    assert err.value.code == "HPO_EXECUTION_CONFLICT"


# ── finalizer 三类失败 ───────────────────────────────────────────

def test_finalizer_history_error_blocks_budget(ctx):
    service, study, runner, tmp_path = ctx
    FakeAdapter.finalize_result = {"status": "completed",
                                   "history_error": {"error_type": "history_persistence_error"}}
    FakeAdapter.returncodes = [0]
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    # 阻断：不得启动第二个 trial。
    assert FakeAdapter.launch_count == 1


def test_finalizer_analysis_and_index_errors_do_not_change_result(ctx):
    service, study, runner, tmp_path = ctx
    FakeAdapter.finalize_result = {
        "status": "completed",
        "analysis_error": {"error_type": "analysis_failed"},
        "index_error": {"error_type": "local_index_persistence_error"},
    }
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "SUCCESS"
    assert loaded.trials[0].result.value == 0.7
    assert record.attempts[0].phase == "FINALIZED"
    assert record.attempts[0].finalizer_record["analysis_error"] is not None
    assert record.attempts[0].finalizer_record["index_error"] is not None


# ── 写盘注入：零后续 launch ──────────────────────────────────────

def test_persistence_failure_at_launch_intent_zero_launch(ctx, monkeypatch):
    service, study, runner, tmp_path = ctx
    from auto_tune.modules.hpo import execution as exec_module

    def _should_fail(record):
        for a in record.attempts:
            if a.phase == "LAUNCH_INTENT":
                return True
        return False

    original = exec_module.ExecutionStore.write

    def failing_write(self, record):
        if _should_fail(record):
            raise HpoError("HPO_PERSISTENCE_ERROR", "injected")
        return original(self, record)

    monkeypatch.setattr(exec_module.ExecutionStore, "write", failing_write)
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    # LAUNCH_INTENT 写失败发生在 launch 之前 → 零 launch。
    assert FakeAdapter.launch_count == 0


def test_persistence_failure_at_result_ready_stops_before_tell_and_launch(
        ctx, monkeypatch):
    service, study, runner, tmp_path = ctx
    from auto_tune.modules.hpo import execution as exec_module
    original = exec_module.ExecutionStore.write

    def failing_write(self, record):
        for a in record.attempts:
            if a.phase == "RESULT_READY":
                raise HpoError("HPO_PERSISTENCE_ERROR", "injected")
        return original(self, record)

    monkeypatch.setattr(exec_module.ExecutionStore, "write", failing_write)
    FakeAdapter.returncodes = [1, 0]
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    # RESULT_READY 落盘失败在 tell 之前 → 不启动第二个 trial。
    assert FakeAdapter.launch_count == 1
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "PENDING"


def test_tell_conflict_is_corrupt_execution(ctx, monkeypatch):
    service, study, runner, tmp_path = ctx
    from auto_tune.modules.hpo import execution as exec_module
    original_tell = exec_module.HpoService.tell

    def conflicting_tell(self, study_id, trial_number, result):
        raise HpoError("HPO_RESULT_CONFLICT", "external terminal")

    monkeypatch.setattr(exec_module.HpoService, "tell", conflicting_tell)
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
