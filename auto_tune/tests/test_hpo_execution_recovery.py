"""H1.2 停止、超时与恢复矩阵测试 — 假进程/身份，无真实训练/LLM/网络。

覆盖规格第 6 节恢复矩阵：PREPARED 可启动一次、LAUNCH_INTENT 无可核验身份阻断、
RUNNING MATCH/UNVERIFIABLE 阻断、MISSING/MISMATCH 中断不杀复用 PID、尾部 PENDING
无 attempt 沿用编号、RESULT_READY/TOLD 幂等提交不重训、超时继续预算、用户停止
PAUSED、正常退出后停止收尾不启动下一个。
"""

import subprocess
import sys
import threading
import time
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
    utc_now_iso,
)
from auto_tune.modules.hpo.execution import HpoRunner, execution_request_id
from auto_tune.modules.hpo.execution_adapter import CollectedOutcome
from auto_tune.modules.hpo.execution_models import (
    ExecutionAttempt,
    ExecutionConfig,
    MetricDiagnostics,
)
from auto_tune.modules.hpo.models import ResultPayload
from auto_tune.modules.run_state.process_identity import (
    IdentityMatch,
    ProcessIdentity,
    capture_process_identity,
    compare_process_identity,
)


def _rid():
    return uuid.uuid4().hex


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-recovery-test-not-a-real-model")
    return snapshot.snapshot_path, model


class FakeProc:
    def __init__(self, rc, pid=None):
        self._seq = getattr(FakeProc, "_seq", 6000) + 1
        FakeProc._seq = self._seq
        self.pid = pid if pid is not None else 6000 + self._seq
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


class LiveProc:
    """保持运行直到 terminate/kill；不自动退出。"""

    def __init__(self, pid=None):
        self._seq = getattr(LiveProc, "_seq", 8000) + 1
        LiveProc._seq = self._seq
        self.pid = pid if pid is not None else 8000 + self._seq
        self._rc = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def terminate(self):
        self.terminated = True
        self._rc = -15

    def kill(self):
        self.killed = True
        if self._rc is None:
            self._rc = -9


class FakeAdapter:
    launch_count = 0
    procs = []  # 每次 launch 弹出一个进程工厂
    on_launch = None

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
        if FakeAdapter.on_launch is not None:
            FakeAdapter.on_launch()
        if FakeAdapter.procs:
            return FakeAdapter.procs.pop(0)
        return FakeProc(0)

    def collect(self, study, attempt):
        evidence = Evidence(
            run_id=attempt.run_id,
            artifact_relpath=f"{study.study_id}/"
                             f"{attempt.run_relpath.split('/')[-1]}/results.csv",
            artifact_sha256="0" * 64, epoch=1)
        return CollectedOutcome(
            result=ResultInput(state="SUCCESS", value=0.66, evidence=evidence),
            diagnostics=MetricDiagnostics(total_rows=1, excluded_rows=0))

    def finalize(self, study, attempt):
        return {"status": "completed"}

    def detect_oom(self, run_dir):
        return False


@pytest.fixture
def reset_fake():
    FakeAdapter.launch_count = 0
    FakeAdapter.procs = []
    FakeAdapter.on_launch = None
    yield
    FakeAdapter.launch_count = 0
    FakeAdapter.procs = []
    FakeAdapter.on_launch = None


def _make_env(tmp_path, snapshot, model, budget):
    service = HpoService(tmp_path / "hpo")
    study = service.create_study(StudyConfig(budget=budget, epochs=30),
                                 snapshot_dir=snapshot, model_path=model)
    runner = HpoRunner(tmp_path / "hpo", tmp_path / "out", tmp_path / "log")
    runner.prepare(study.study_id, ExecutionConfig(
        batch=2, imgsz=64, device="cpu", timeout_seconds=3600))
    return service, study, runner, tmp_path


@pytest.fixture
def base(tmp_path, inputs, monkeypatch, reset_fake):
    snapshot, model = inputs
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    return _make_env(tmp_path, snapshot, model, budget=2)


@pytest.fixture
def mini(tmp_path, inputs, monkeypatch, reset_fake):
    snapshot, model = inputs
    monkeypatch.setattr("auto_tune.modules.hpo.execution.ExecutionAdapter",
                        FakeAdapter)
    monkeypatch.setattr("auto_tune.modules.hpo.execution._capture_process_identity",
                        lambda pid: f"tok:{pid}")
    return _make_env(tmp_path, snapshot, model, budget=1)


def _authoritative_effective(study, config, trial):
    from auto_tune.modules.agent_engine.executor import build_yolo_command
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
    return effective, build_yolo_command


def _attempt(study, trial, phase, *, runner, run_id=None, result=None,
             returncode=None, pid=None, token=None,
             termination_reason=None, finished_at=None):
    trial_id = trial.trial_id
    rid = run_id or f"tuning:{uuid.uuid4()}"
    candidate = dict(trial.candidate_params)
    config = runner.status(study.study_id).config
    effective, builder = _authoritative_effective(study, config, trial)
    run_dir = runner._output_root / f"{study.study_id}/{trial_id}"
    command = builder(trial_id, str(run_dir / "args.yaml"), dict(effective))
    return ExecutionAttempt(
        trial_number=trial.number,
        trial_id=trial_id,
        request_id=trial.request_id,
        run_id=rid,
        phase=phase,
        candidate_params=candidate,
        effective_params=effective,
        command=command,
        command_executable=command[0],
        run_relpath=f"{study.study_id}/{trial_id}",
        args_sha256="0" * 64,
        pid=pid,
        process_create_token=token,
        started_at="2026-09-08T00:00:00.000000+00:00",
        finished_at=finished_at,
        returncode=returncode,
        termination_reason=termination_reason,
        result=result,
        finalizer_record=None,
        error_code=None,
        error_message=None,
    )


def _synthetic_evidence(study, number, run_id):
    return Evidence(
        run_id=run_id,
        artifact_relpath=f"{study.study_id}/"
                         f"{study.study_id}_t{number:04d}/results.csv",
        artifact_sha256="0" * 64, epoch=1)


def _publish_state(base, runner, status="PAUSED", attempts=None):
    service, study, r, tmp_path = base
    record = runner.status(study.study_id)
    from auto_tune.modules.hpo.execution_storage import ExecutionStore as ES
    from auto_tune.modules.hpo.execution import HpoRunner as HR
    from auto_tune.modules.run_state.service import new_run_state as nrs
    from dataclasses import replace
    # 为 PREPARED/RUNNING attempt 同步补 run_state 文件（正常流程由注册写入）。
    for attempt in (attempts or []):
        if attempt.phase in ("PREPARED", "RUNNING"):
            state_path = runner._state_path(attempt.run_relpath)
            state = nrs("tuning")
            state = replace(state, run_id=attempt.run_id,
                            started_at=attempt.started_at)
            from auto_tune.modules.run_state.service import write_run_state
            state_path.parent.mkdir(parents=True, exist_ok=True)
            write_run_state(state_path, state)
    store = ES(tmp_path / "hpo")
    next_record = record.model_copy(update={
        "status": status,
        "attempts": list(attempts or []),
        "revision": record.revision + 1,
        "updated_at": utc_now_iso(),
    })
    with store.locked(study.study_id):
        store.write(next_record)
    return next_record


def _ask_pending(service, study, number):
    trial = service.ask(study.study_id,
                        request_id=execution_request_id(study.study_id, number))
    assert trial.state == "PENDING"
    return trial


# ── RUNNING 身份矩阵 ─────────────────────────────────────────────

@pytest.mark.parametrize("match,code", [
    (IdentityMatch.MATCH, "HPO_PROCESS_STILL_ACTIVE"),
    (IdentityMatch.UNVERIFIABLE, "HPO_RECOVERY_REQUIRED"),
])
def test_resume_running_match_or_unverifiable_blocks(base, monkeypatch, match, code):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)
    attempt = _attempt(study, trial, "RUNNING", runner=runner, pid=4242, token="tok:4242")
    _publish_state(base, runner, status="RUNNING", attempts=[attempt])
    monkeypatch.setattr("auto_tune.modules.hpo.execution.compare_process_identity",
                        lambda identity: match)
    with pytest.raises(HpoError) as err:
        runner.resume(study.study_id)
    assert err.value.code == code
    assert FakeAdapter.launch_count == 0


def test_resume_running_missing_interrupts_and_continues_budget(base, monkeypatch):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)
    attempt = _attempt(study, trial, "RUNNING", runner=runner, pid=4242, token="tok:4242")
    _publish_state(base, runner, status="RUNNING", attempts=[attempt])
    monkeypatch.setattr("auto_tune.modules.hpo.execution.compare_process_identity",
                        lambda identity: IdentityMatch.MISSING)
    FakeAdapter.procs = [FakeProc(0), FakeProc(0)]
    record = runner.resume(study.study_id)
    assert record.status == "COMPLETED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "INTERRUPTED"
    assert loaded.trials[0].result.reason_code == "process_interrupted"
    assert loaded.trials[1].state == "SUCCESS"
    # 中断的 trial 不启动；后续预算 trial 正常启动。
    assert FakeAdapter.launch_count == 1


def test_resume_running_mismatch_does_not_kill_and_interrupts(base, monkeypatch):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)
    attempt = _attempt(study, trial, "RUNNING", runner=runner, pid=4242, token="tok:4242")
    _publish_state(base, runner, status="RUNNING", attempts=[attempt])
    killed = []
    monkeypatch.setattr("auto_tune.modules.hpo.execution.compare_process_identity",
                        lambda identity: IdentityMatch.MISMATCH)
    record = runner.resume(study.study_id)
    assert record.status == "COMPLETED" or record.status == "PAUSED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "INTERRUPTED"


# ── LAUNCH_INTENT / PREPARED 恢复 ───────────────────────────────

def test_resume_launch_intent_without_identity_blocks(base, monkeypatch):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)
    attempt = _attempt(study, trial, "LAUNCH_INTENT", runner=runner)
    _publish_state(base, runner, status="RUNNING", attempts=[attempt])
    with pytest.raises(HpoError) as err:
        runner.resume(study.study_id)
    assert err.value.code == "HPO_RECOVERY_REQUIRED"


def test_resume_prepared_launches_once(mini, monkeypatch):
    service, study, runner, tmp_path = mini
    trial = _ask_pending(service, study, 0)
    attempt = _attempt(study, trial, "PREPARED", runner=runner)
    _publish_state(mini, runner, status="PAUSED", attempts=[attempt])
    FakeAdapter.procs = [FakeProc(0)]
    record = runner.resume(study.study_id)
    assert FakeAdapter.launch_count == 1
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "SUCCESS"


def test_resume_tail_pending_without_attempt_claims_same_number(base):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)  # PENDING 但 execution 无 attempt
    FakeAdapter.procs = [FakeProc(1), FakeProc(0)]
    record = runner.resume(study.study_id)
    assert record.status == "COMPLETED"
    loaded = service.load_study(study.study_id)
    # 沿用编号 0（失败），再跑 1（成功）→ 恰好两次 launch。
    assert len(loaded.trials) == 2
    assert FakeAdapter.launch_count == 2
    assert loaded.trials[0].state == "FAILED"
    assert loaded.trials[1].state == "SUCCESS"


# ── RESULT_READY / TOLD 幂等提交 ────────────────────────────────

def test_resume_result_ready_tells_and_does_not_retrain(mini):
    service, study, runner, tmp_path = mini
    trial = _ask_pending(service, study, 0)
    run_id = f"tuning:{uuid.uuid4()}"
    result = ResultInput(state="SUCCESS", value=0.6,
                         evidence=_synthetic_evidence(study, 0, run_id))
    attempt = _attempt(study, trial, "RESULT_READY", runner=runner, run_id=run_id, result=result,
                       returncode=0, pid=4242, token="tok:4242")
    _publish_state(mini, runner, status="RUNNING", attempts=[attempt])
    record = runner.resume(study.study_id)
    assert record.status == "COMPLETED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "SUCCESS"
    assert loaded.trials[0].result.value == 0.6
    assert FakeAdapter.launch_count == 0


# ── 超时 / 用户停止 ─────────────────────────────────────────────

def test_timeout_terminates_confirms_exit_and_continues_budget(base):
    service, study, runner, tmp_path = base
    # 直接改写已绑定执行记录的 config 为 1s 超时（避免 prepare 配置冲突）。
    from auto_tune.modules.hpo.execution_storage import ExecutionStore as ES
    current = runner.status(study.study_id)
    store = ES(tmp_path / "hpo")
    next_record = current.model_copy(update={
        "config": ExecutionConfig(batch=2, imgsz=64, device="cpu",
                                  timeout_seconds=1),
        "revision": current.revision + 1,
        "updated_at": utc_now_iso(),
    })
    with store.locked(study.study_id):
        store.write(next_record)
    live = LiveProc()
    FakeAdapter.procs = [live, FakeProc(0)]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "FAILED"
    assert loaded.trials[0].result.reason_code == "timeout"
    assert live.terminated
    assert loaded.trials[1].state == "SUCCESS"
    assert FakeAdapter.launch_count == 2


def test_user_stop_pauses_and_no_more_asks(base, monkeypatch):
    service, study, runner, tmp_path = base
    live = LiveProc()
    FakeAdapter.procs = [live]
    launched = threading.Event()
    FakeAdapter.on_launch = launched.set
    stop = threading.Event()
    results = []

    def run_in_thread():
        results.append(runner.run(study.study_id, stop_event=stop))

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    # 等 launch 后再发停止信号（不并发读执行锁）。
    assert launched.wait(timeout=15)
    stop.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    record = results[0]
    assert record.status == "PAUSED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "CANCELLED"
    assert loaded.trials[0].result.reason_code == "user_stopped"
    assert len(loaded.trials) == 1
    assert FakeAdapter.launch_count == 1


def test_normal_exit_then_stop_finishes_and_pauses(base, monkeypatch):
    service, study, runner, tmp_path = base
    trial = _ask_pending(service, study, 0)
    run_id = f"tuning:{uuid.uuid4()}"
    attempt = _attempt(study, trial, "EXITED", runner=runner, run_id=run_id, returncode=0,
                       pid=4242, token="tok:4242")
    _publish_state(base, runner, status="PAUSED", attempts=[attempt])
    stop = threading.Event()
    stop.set()
    record = runner.resume(study.study_id, stop_event=stop)
    assert record.status == "PAUSED"
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "SUCCESS"
    # 正常退出后停止：不再 ask。
    assert FakeAdapter.launch_count == 0


# ── 真实 token 核验（受控本地短进程，不 kill 任意 PID）──────────

def test_real_process_identity_capture_and_match(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(20)"])
    try:
        identity = capture_process_identity(proc.pid)
        assert identity is not None
        assert identity.pid == proc.pid
        assert identity.process_create_token
        match = compare_process_identity(
            ProcessIdentity(proc.pid, identity.process_create_token))
        assert match is IdentityMatch.MATCH
    finally:
        proc.terminate()
        proc.wait(timeout=10)
