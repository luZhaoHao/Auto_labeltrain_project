"""H1.2 第四轮返修 R2a-3 正式回归测试 — 不依赖 log 目录。

覆盖：FINALIZED attempt 单独替换 command[0] 时 status 拒绝（不得用被验证字段自身
作为依据）；命令尾部污染在解析器不可用时仍拒绝；环境漂移下 TOLD 幂等收尾保留冻结
executable 依据；替换 executable 的新启动被当前环境校验拦截（HPO_PREFLIGHT_FAILED、
零 launch）；缺冻结依据（command_executable）的旧记录被明确拒绝而不从 command[0]
自动补齐。
"""

import hashlib
import json

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


def _write_execution(root, study_id, data):
    path = root / "hpo" / study_id / "execution.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _finish_budget(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    return service, study, runner, root


# ── command[0] 单独污染（历史 executable 自校验）─────────────────

def test_historical_command0_substitution_is_rejected(ctx):
    """已完成 attempt 单独替换 command[0] 不得再被 status 接受。"""
    service, study, runner, root = _finish_budget(ctx)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command"][0] = "C:/unrelated/not-a-training-program.exe"
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError) as caught:
        runner.status(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"


def test_command_executable_field_mutation_alone_is_rejected(ctx):
    """篡改冻结依据 command_executable（command[0] 未变）同样拒绝。"""
    service, study, runner, root = _finish_budget(ctx)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command_executable"] = "C:/unrelated/x.EXE"
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError) as caught:
        runner.status(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"


def test_legacy_record_missing_command_executable_is_not_derived(ctx):
    """缺少冻结依据的旧记录被拒绝，不从 command[0] 自动补齐后放行。"""
    service, study, runner, root = _finish_budget(ctx)
    data = _read_execution(root, study.study_id)
    del data["attempts"][0]["command_executable"]
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError) as caught:
        runner.status(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"


# ── 命令尾部污染在解析器不可用时仍拒绝 ──────────────────────────

def test_command_tail_pollution_rejected_without_resolver(ctx, monkeypatch):
    service, study, runner, root = _finish_budget(ctx)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command"].append("lr0=999")
    _write_execution(root, study.study_id, data)

    def missing():
        raise FileNotFoundError("YOLO unavailable after environment drift")

    monkeypatch.setattr(executor_module, "resolve_yolo_executable", missing)
    with pytest.raises(HpoError) as caught:
        runner.status(study.study_id)
    assert caught.value.code == "HPO_CORRUPT_EXECUTION"


# ── 环境漂移下历史收尾（冻结依据不依赖当前 YOLO）────────────────

def test_told_replay_keeps_frozen_executable_under_env_drift(ctx,
                                                             monkeypatch):
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
        pass  # 新训练可能被阻断，但已发布 TOLD 必须先补齐 FINALIZED。
    data = _read_execution(root, study.study_id)
    assert data["attempts"][0]["phase"] == "FINALIZED"
    assert FakeAdapter.launch_count == launches


# ── 新启动拦截当前环境 executable 漂移/替换 ──────────────────────

def test_substituted_executable_cannot_launch_in_current_environment(
        ctx, monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as module
    service, study, runner, root = ctx
    runner._finish_session(runner.status(study.study_id), "RUNNING", None)
    runner._claim_next(service.load_study(study.study_id),
                       runner.status(study.study_id), None)
    attempt = runner.status(study.study_id).attempts[0]
    raw = yaml.safe_dump(dict(attempt.effective_params)).encode("utf-8")
    (root / "out" / attempt.run_relpath / "args.yaml").write_bytes(raw)
    command = list(attempt.command)
    command[0] = "C:/unrelated/not-a-training-program.exe"
    calls = []
    monkeypatch.setattr(module, "launch_training",
                        lambda *a, **kw: calls.append(kw))
    with pytest.raises(HpoError) as caught:
        ExecutionAdapter(root / "out", root / "log").launch(dict(
            trial_id=attempt.trial_id, run_relpath=attempt.run_relpath,
            effective_params=dict(attempt.effective_params), command=command,
            args_sha256=hashlib.sha256(raw).hexdigest()))
    assert caught.value.code == "HPO_PREFLIGHT_FAILED"
    assert not calls


# ── 合法完整记录不受回归影响 ─────────────────────────────────────

def test_legitimate_completed_record_status_still_ok(ctx):
    service, study, runner, root = _finish_budget(ctx)
    viewed = runner.status(study.study_id)
    assert viewed.status == "COMPLETED"
    assert all(a.phase == "FINALIZED" for a in viewed.attempts)
    for attempt in viewed.attempts:
        assert attempt.command_executable == attempt.command[0]
