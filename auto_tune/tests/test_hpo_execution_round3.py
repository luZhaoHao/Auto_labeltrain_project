"""H1.2 第三轮返修 R2a-1/R2a-2/R6 正式回归测试 — 从 Codex 独立反例提炼。

不依赖 log 目录。覆盖：launch 对“摘要正确但内容与 effective 不一致”拒绝零启动；
缺字段/非法类型/多余键拒绝；启动前预验失败与发布 LAUNCH_INTENT 后临近复验失败均
保留稳定错误码并阻断后续取样/启动；TOLD 幂等收尾不依赖当前 YOLO 解析（抛错/路径
变化），纯 status 不要求当前 YOLO，历史命令污染在解析器不可用时仍被拒绝。
"""

import hashlib
import json
import threading

import pytest
import yaml

from auto_tune.modules.agent_engine import executor as executor_module
from auto_tune.modules.hpo import HpoError
from auto_tune.modules.hpo.execution_adapter import ExecutionAdapter

from auto_tune.tests.test_hpo_execution import (
    FakeAdapter,
    ctx,
    hpo_inputs,
    reset_fake,
)


def _read_execution(root, study_id):
    path = root / "hpo" / study_id / "execution.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_prepared(service, study, runner):
    runner._finish_session(runner.status(study.study_id), "RUNNING", None)
    runner._claim_next(service.load_study(study.study_id),
                       runner.status(study.study_id), None)


def _poison_args(ctx, attempt, *, mutate):
    """把磁盘 args.yaml 改为 mutate(effective) 并返回 prepared；摘要对新字节正确。"""
    service, study, runner, root = ctx
    effective = dict(attempt.effective_params)
    altered = mutate(dict(effective))
    raw = yaml.safe_dump(altered).encode("utf-8")
    args = root / "out" / attempt.run_relpath / "args.yaml"
    args.write_bytes(raw)
    return {
        "trial_id": attempt.trial_id,
        "run_relpath": attempt.run_relpath,
        "effective_params": effective,
        "args_sha256": hashlib.sha256(raw).hexdigest(),
        "command": list(attempt.command),
    }


# ── R2a-1：启动前 args 内容语义校验 ──────────────────────────────

def test_launch_rejects_semantically_wrong_args_with_matching_digest(ctx,
                                                                    monkeypatch):
    """完整合法 effective/command，仅污染磁盘 args.epochs 并给正确摘要仍拒绝。"""
    import auto_tune.modules.hpo.execution_adapter as adapter_module
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    attempt = runner.status(study.study_id).attempts[0]
    prepared = _poison_args(ctx, attempt, mutate=lambda p: dict(p, epochs=999))
    calls = []
    monkeypatch.setattr(adapter_module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    with pytest.raises(HpoError) as caught:
        ExecutionAdapter(root / "out", root / "log").launch(prepared)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
    assert not calls


def test_launch_rejects_missing_field_with_matching_digest(ctx, monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as adapter_module
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    attempt = runner.status(study.study_id).attempts[0]
    prepared = _poison_args(
        ctx, attempt,
        mutate=lambda p: {k: v for k, v in p.items() if k != "optimizer"})
    calls = []
    monkeypatch.setattr(adapter_module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    with pytest.raises(HpoError) as caught:
        ExecutionAdapter(root / "out", root / "log").launch(prepared)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
    assert not calls


@pytest.mark.parametrize("bad_value", ["5", True])
def test_launch_rejects_illegal_type_with_matching_digest(ctx, monkeypatch,
                                                          bad_value):
    import auto_tune.modules.hpo.execution_adapter as adapter_module
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    attempt = runner.status(study.study_id).attempts[0]
    prepared = _poison_args(ctx, attempt,
                            mutate=lambda p: dict(p, warmup_epochs=bad_value))
    calls = []
    monkeypatch.setattr(adapter_module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    with pytest.raises(HpoError) as caught:
        ExecutionAdapter(root / "out", root / "log").launch(prepared)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
    assert not calls


def test_launch_accepts_consistent_full_args(ctx, monkeypatch):
    """合法且完整的 args/effective/command 应通过并实际调用 launch_training。"""
    import auto_tune.modules.hpo.execution_adapter as adapter_module
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    attempt = runner.status(study.study_id).attempts[0]
    prepared = _poison_args(ctx, attempt, mutate=lambda p: dict(p))
    captured = {}
    marker = object()
    monkeypatch.setattr(
        adapter_module, "launch_training",
        lambda *a, **kw: captured.update(command=kw["command"]) or marker)
    returned = ExecutionAdapter(root / "out", root / "log").launch(prepared)
    assert returned is marker
    assert captured["command"] == list(attempt.command)


# ── R2a-2：校验损坏不得降级为普通训练失败 ────────────────────────

def test_prelaunch_validation_failure_blocks_without_launch(ctx, monkeypatch):
    """预验即失败：validate_launch 抛损坏 → 保留错误码、零 launch、无后续取样。"""
    service, study, runner, root = ctx
    calls = []

    def reject(self, prepared):
        calls.append(prepared["trial_id"])
        raise HpoError("HPO_CORRUPT_EXECUTION", "prepared args hash mismatch")

    monkeypatch.setattr(FakeAdapter, "validate_launch", reject)
    with pytest.raises(HpoError) as caught:
        runner.run(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
    assert len(calls) == 1
    assert FakeAdapter.launch_count == 0
    assert len(service.load_study(study.study_id).trials) == 1


def test_nearlaunch_revalidation_failure_blocks_remaining_budget(ctx,
                                                                 monkeypatch):
    """预验通过、发布 LAUNCH_INTENT 后临近启动复验失败 → 保留错误码、阻断取样。"""
    service, study, runner, root = ctx
    calls = []

    def reject(self, prepared):
        calls.append(prepared["trial_id"])
        raise HpoError("HPO_CORRUPT_EXECUTION", "prepared args hash mismatch")

    monkeypatch.setattr(FakeAdapter, "launch", reject)
    with pytest.raises(HpoError) as caught:
        runner.run(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
    assert len(calls) == 1
    assert len(service.load_study(study.study_id).trials) == 1


def test_launch_validation_hpo_error_is_not_failed_trial(ctx, monkeypatch):
    """launch 抛 HpoError 不得登记 FAILED/training_failed 也不得继续预算。"""
    service, study, runner, root = ctx
    # 第一次 trial 在“临近启动复验”被阻断；验证 study 中没有 FAILED 终态被伪造。
    monkeypatch.setattr(
        FakeAdapter, "launch",
        lambda self, prepared: (_ for _ in ()).throw(
            HpoError("HPO_CORRUPT_EXECUTION", "prepared args hash mismatch")))
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    loaded = service.load_study(study.study_id)
    assert loaded.trials[0].state == "PENDING"
    assert len(loaded.trials) == 1


# ── R6：历史收尾不依赖当前 YOLO 可执行文件 ───────────────────────

def test_told_replay_does_not_require_current_yolo_resolver(ctx, monkeypatch):
    service, study, runner, root = ctx
    FakeAdapter.finalize_result = {"history_error": {"error": "disk"}}
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    assert _read_execution(root, study.study_id)["attempts"][0]["phase"] == "TOLD"
    FakeAdapter.finalize_result = {"status": "completed"}
    launches = FakeAdapter.launch_count

    def missing():
        raise FileNotFoundError("YOLO unavailable after environment drift")

    monkeypatch.setattr(executor_module, "resolve_yolo_executable", missing)
    try:
        runner.resume(study.study_id)
    except (HpoError, FileNotFoundError):
        pass  # 新训练可能被阻断，但已发布 TOLD 必须先重放收尾。
    data = _read_execution(root, study.study_id)
    assert data["attempts"][0]["phase"] == "FINALIZED"
    assert FakeAdapter.launch_count == launches


def test_told_replay_survives_yolo_resolver_path_change(ctx, monkeypatch):
    """当前 PATH 指向另一 YOLO 时历史记录不得被当作损坏。"""
    service, study, runner, root = ctx
    FakeAdapter.finalize_result = {"history_error": {"error": "disk"}}
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    FakeAdapter.finalize_result = {"status": "completed"}
    launches = FakeAdapter.launch_count
    monkeypatch.setattr(executor_module, "resolve_yolo_executable",
                        lambda: "C:/elsewhere/yolo.EXE")
    stop = threading.Event()
    stop.set()  # 完成已发布收尾后暂停，不认领新 trial。
    record = runner.resume(study.study_id, stop_event=stop)
    data = _read_execution(root, study.study_id)
    assert data["attempts"][0]["phase"] == "FINALIZED"
    assert FakeAdapter.launch_count == launches
    assert record.status == "PAUSED"


def test_pure_status_on_terminal_record_needs_no_current_yolo(ctx,
                                                              monkeypatch):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"

    def missing():
        raise FileNotFoundError("YOLO unavailable after environment drift")

    monkeypatch.setattr(executor_module, "resolve_yolo_executable", missing)
    viewed = runner.status(study.study_id)  # 不应抛错
    assert viewed.status == "COMPLETED"
    assert all(a.phase == "FINALIZED" for a in viewed.attempts)


def test_historical_command_pollution_still_rejected_without_resolver(
        ctx, monkeypatch):
    """即使解析器不可用，历史命令污染仍必须拒绝（不取消命令完整性校验）。"""
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command"].append("lr0=999")
    path = root / "hpo" / study.study_id / "execution.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def missing():
        raise FileNotFoundError("YOLO unavailable after environment drift")

    monkeypatch.setattr(executor_module, "resolve_yolo_executable", missing)
    with pytest.raises(HpoError) as caught:
        runner.status(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"
