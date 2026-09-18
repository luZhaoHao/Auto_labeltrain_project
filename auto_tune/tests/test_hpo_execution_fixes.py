"""H1.2 返修 R1–R7 正式回归测试 — 从 Codex 独立反例提炼，不依赖 log 目录。

普通 pytest 不启动真实 YOLO、不调用网络 LLM；活动进程清理用确实保持运行的受控
替身进程验证。真实短训练由 Codex 在返修后重新验收时执行。
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_tune.modules.hpo import HpoError
from auto_tune.modules.hpo.execution import HpoRunner
from auto_tune.modules.hpo.execution_adapter import ExecutionAdapter
from auto_tune.modules.hpo.metrics import extract_objective
from auto_tune.modules.run_state.service import read_run_state
from auto_tune.scripts.verify_hpo_execution import main as cli_main

from auto_tune.tests.test_hpo_execution import (
    FakeAdapter,
    FakeProc,
    ctx,
    hpo_inputs,
    reset_fake,
)

import auto_tune.modules.hpo.execution as execution
from auto_tune.modules.run_state.service import new_run_state as _nrs


def _run_state_finished(attempt):
    return attempt.finished_at


# ── R1 活动进程清理 ──────────────────────────────────────────────

def test_running_write_failure_terminates_owned_process(ctx, monkeypatch):
    service, study, runner, root = ctx
    proc = FakeProc(None)  # 保持运行；被清理时应 terminate/kill。
    monkeypatch.setattr(FakeAdapter, "launch", lambda self, prepared: proc)
    original = runner._commit

    def failing(record):
        if record.attempts and record.attempts[-1].phase == "RUNNING":
            raise HpoError("HPO_PERSISTENCE_ERROR", "injected disk failure")
        return original(record)

    monkeypatch.setattr(runner, "_commit", failing)
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert proc.terminated or proc.killed, \
        "owned process left running after persistence failure"
    # 未确认退出不得启动下一个 trial。
    assert FakeAdapter.launch_count <= 1


def test_monitor_persist_termination_failure_cleans_process(ctx, monkeypatch):
    service, study, runner, root = ctx
    from auto_tune.tests.test_hpo_execution_recovery import LiveProc

    live = LiveProc()
    stop = threading.Event()

    def launch(self, prepared):
        stop.set()
        return live

    monkeypatch.setattr(FakeAdapter, "launch", launch)
    original = execution.write_run_state

    def failing(path, state):
        if state.phase == "stopping":
            raise OSError("injected stopping-state disk failure")
        return original(path, state)

    monkeypatch.setattr(execution, "write_run_state", failing)
    with pytest.raises(HpoError):
        runner.run(study.study_id, stop_event=stop)
    assert live.killed or live.terminated, \
        "owned process left running after stopping-state write failure"


# ── R5 终态 RunState 写盘失败阻断预算 ────────────────────────────

def test_terminal_state_write_failure_blocks_next_trial(ctx, monkeypatch):
    service, study, runner, root = ctx
    original = execution.write_run_state

    def failing(path, state):
        if state.phase == "terminal":
            raise OSError("injected terminal state disk failure")
        return original(path, state)

    monkeypatch.setattr(execution, "write_run_state", failing)
    with pytest.raises(HpoError) as err:
        runner.run(study.study_id)
    assert "persist" in err.value.message.lower() \
        or "state" in err.value.message.lower() \
        or "write" in err.value.message.lower()
    assert FakeAdapter.launch_count == 1


def test_normal_success_run_state_has_finished_at(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    attempt = record.attempts[0]
    state = read_run_state(root / "out" / attempt.run_relpath / "run_state.json",
                           run_kind="tuning")
    assert state.status == "completed"
    assert state.finished_at, "terminal run_state missing finished_at"


def test_exit_zero_after_stop_is_cancelled_in_run_state(ctx, monkeypatch):
    service, study, runner, root = ctx
    stop = threading.Event()

    class StopProc(FakeProc):
        def terminate(self):
            self.terminated = True
            self._rc = 0

    proc = StopProc(None)

    def launch(self, prepared):
        stop.set()
        return proc

    monkeypatch.setattr(FakeAdapter, "launch", launch)
    record = runner.run(study.study_id, stop_event=stop)
    attempt = record.attempts[0]
    assert attempt.result.state == "CANCELLED"
    state = read_run_state(root / "out" / attempt.run_relpath / "run_state.json",
                           run_kind="tuning")
    assert state.status == "cancelled", \
        f"trial cancelled but run_state is {state.status}"


def test_timeout_run_state_status_failed(ctx, monkeypatch):
    service, study, runner, root = ctx
    from auto_tune.tests.test_hpo_execution_recovery import LiveProc
    from auto_tune.modules.hpo.execution_storage import ExecutionStore as ES
    current = runner.status(study.study_id)
    store = ES(root / "hpo")
    next_record = current.model_copy(update={
        "config": execution.ExecutionConfig(batch=2, imgsz=64, device="cpu",
                                            timeout_seconds=1),
        "revision": current.revision + 1,
        "updated_at": execution.utc_now_iso(),
    })
    with store.locked(study.study_id):
        store.write(next_record)
    live = LiveProc()
    monkeypatch.setattr(FakeAdapter, "launch",
                        lambda self, prepared: live)
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    attempt = record.attempts[0]
    assert attempt.result.state == "FAILED"
    state = read_run_state(root / "out" / attempt.run_relpath
                           / "run_state.json", run_kind="tuning")
    assert state.status == "failed"
    assert state.finished_at


# ── R4 实际 args 缺字段不可 SUCCESS ──────────────────────────────

def test_actual_args_missing_required_fields_is_not_success(tmp_path):
    run = tmp_path / "trial"
    run.mkdir()
    (run / "args.yaml").write_text("{}", encoding="utf-8")
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.9\n", encoding="utf-8")
    adapter = ExecutionAdapter(tmp_path, tmp_path / "log")
    attempt = SimpleNamespace(run_relpath="trial", run_id="tuning:test",
                              effective_params={"lr0": 0.001, "epochs": 1})
    outcome = adapter.collect(
        SimpleNamespace(config=SimpleNamespace(epochs=1)), attempt)
    assert outcome.result is None, \
        "empty args.yaml accepted as verified actual parameters"


def test_actual_args_drift_is_not_success(tmp_path):
    run = tmp_path / "trial"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.9\n", encoding="utf-8")
    import yaml as _yaml
    (run / "args.yaml").write_text(
        _yaml.safe_dump({"task": "detect", "model": "m.pt", "data": "d.yaml",
                         "epochs": 1, "seed": 42, "batch": 1, "imgsz": 64,
                         "device": "cpu", "workers": 0, "resume": False,
                         "deterministic": True, "patience": 0, "val": True,
                         "save": True, "plots": False, "amp": False,
                         "optimizer": "SGD", "lr0": 999.0, "lrf": 0.05,
                         "momentum": 0.9, "weight_decay": 0.0005,
                         "warmup_epochs": 1}), encoding="utf-8")
    adapter = ExecutionAdapter(tmp_path, tmp_path / "log")
    attempt = SimpleNamespace(
        run_relpath="trial", run_id="tuning:test",
        effective_params={"task": "detect", "model": "m.pt", "data": "d.yaml",
                          "epochs": 1, "seed": 42, "batch": 1, "imgsz": 64,
                          "device": "cpu", "workers": 0, "resume": False,
                          "deterministic": True, "patience": 0, "val": True,
                          "save": True, "plots": False, "amp": False,
                          "optimizer": "SGD", "lr0": 0.001, "lrf": 0.05,
                          "momentum": 0.9, "weight_decay": 0.0005,
                          "warmup_epochs": 1})
    outcome = adapter.collect(
        SimpleNamespace(config=SimpleNamespace(epochs=1)), attempt)
    assert outcome.result is None
    assert outcome.reason_code == "invalid_params"


# ── R2 launch 启动前 args 校验 ───────────────────────────────────

def test_launch_checks_prepared_args_hash(tmp_path, monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as module
    run = tmp_path / "trial"
    run.mkdir()
    (run / "args.yaml").write_text("lr0: 999\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    prepared = dict(run_relpath="trial", trial_id="trial",
                    args_sha256="0" * 64, effective_params={"lr0": .001},
                    command=["test-only-command"])
    with pytest.raises(HpoError):
        ExecutionAdapter(tmp_path, tmp_path / "log").launch(prepared)
    assert not calls


def test_launch_missing_args_yaml_never_recreates(tmp_path, monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as module
    run = tmp_path / "trial"
    run.mkdir()
    calls = []
    monkeypatch.setattr(module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    prepared = dict(run_relpath="trial", trial_id="trial",
                    args_sha256="0" * 64, effective_params={"lr0": .001},
                    command=["yolo", "train"])
    with pytest.raises(HpoError):
        ExecutionAdapter(tmp_path, tmp_path / "log").launch(prepared)
    assert not calls


# ── R3 冻结根路径与链接边界 ──────────────────────────────────────

def test_runner_rejects_changed_bound_output_root(ctx):
    service, study, runner, root = ctx
    changed = HpoRunner(root / "hpo", root / "different-out",
                        root / "different-log")
    with pytest.raises(HpoError) as caught:
        changed.run(study.study_id)
    assert caught.value.code == "HPO_EXECUTION_CONFLICT"
    assert FakeAdapter.launch_count == 0


def test_metric_reader_rejects_symlink_ancestor(tmp_path):
    target = tmp_path / "outside"
    run = target / "trial"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.9\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(HpoError):
        extract_objective(link / "trial", artifact_root=link,
                          run_id="tuning:test", epochs=1)


# ── R2 执行语义损坏拒绝 ──────────────────────────────────────────

@pytest.mark.parametrize("mutation", ["candidate", "command", "completed_pending"])
def test_execution_semantic_corruption_is_rejected(ctx, mutation):
    service, study, runner, root = ctx
    runner._finish_session(runner.status(study.study_id), "RUNNING", None)
    runner._claim_next(service.load_study(study.study_id),
                       runner.status(study.study_id), None)
    path = root / "hpo" / study.study_id / "execution.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "candidate":
        data["attempts"][0]["candidate_params"]["lr0"] = 999.0
    elif mutation == "command":
        data["attempts"][0]["command"] = ["unrelated-executable"]
    else:
        data["status"] = "COMPLETED"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HpoError):
        runner.status(study.study_id)


# ── R6 环境漂移幂等收尾 ──────────────────────────────────────────

def test_environment_drift_allows_already_told_finalizer_replay(ctx,
                                                                monkeypatch):
    service, study, runner, root = ctx
    FakeAdapter.finalize_result = {"history_error": {"error": "disk"}}
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    assert runner.status(study.study_id).attempts[0].phase == "TOLD"
    FakeAdapter.finalize_result = {"status": "completed"}
    original = execution._current_execution_environment

    def drifted():
        result = original()
        result["torch_version"] += "-changed"
        return result

    monkeypatch.setattr(execution, "_current_execution_environment", drifted)
    try:
        runner.resume(study.study_id)
    except HpoError:
        pass  # 新训练可能被阻断，但已提交事实必须完成重放。
    assert runner.status(study.study_id).attempts[0].phase == "FINALIZED"


# ── R7 验收 CLI 退出码 ───────────────────────────────────────────

def test_cli_all_failed_must_exit_nonzero(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [1]
    rc = cli_main(["--snapshot-dir", study.snapshot_binding.snapshot_path,
                   "--model-path", study.model_binding.model_path,
                   "--storage-root", str(root / "cli"),
                   "--output-root", str(root / "cli-out"),
                   "--log-root", str(root / "cli-log"),
                   "--budget", "1"])
    assert rc != 0


def test_cli_partial_failure_must_exit_nonzero(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [1, 0]
    rc = cli_main(["--snapshot-dir", study.snapshot_binding.snapshot_path,
                   "--model-path", study.model_binding.model_path,
                   "--storage-root", str(root / "cli2"),
                   "--output-root", str(root / "cli2-out"),
                   "--log-root", str(root / "cli2-log"),
                   "--budget", "2"])
    assert rc != 0


def test_cli_all_success_returns_zero(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    rc = cli_main(["--snapshot-dir", study.snapshot_binding.snapshot_path,
                   "--model-path", study.model_binding.model_path,
                   "--storage-root", str(root / "cli3"),
                   "--output-root", str(root / "cli3-out"),
                   "--log-root", str(root / "cli3-log"),
                   "--budget", "2"])
    assert rc == 0
