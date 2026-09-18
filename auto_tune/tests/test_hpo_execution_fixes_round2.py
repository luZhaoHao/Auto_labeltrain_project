"""H1.2 第二轮返修 R2a–R2c 正式回归测试 — 不依赖 log 目录。

覆盖：命令参数污染（追加 lr0=999）status/run/resume 拒绝且零启动；effective 固定
条件单字段污染（epochs/seed/batch/imgsz/device）与 effective/command 同时污染拒绝；
launch 与 args 哈希语义不一致零调用；空/未耗尽预算的伪 COMPLETED 拒绝。合法输入
与预算用尽后的真 COMPLETED 仍通过。
"""

import json
import hashlib

import pytest
import yaml

from auto_tune.modules.agent_engine.executor import build_yolo_command
from auto_tune.modules.hpo import HpoError
from auto_tune.modules.hpo.execution_adapter import ExecutionAdapter
from auto_tune.scripts.verify_hpo_execution import main as cli_main

from auto_tune.tests.test_hpo_execution import (
    FakeAdapter,
    ctx,
    hpo_inputs,
    reset_fake,
)


def _full_effective() -> dict:
    """完整且一致的 planned effective：固定项 + 绑定 + 执行配置 + 六搜索参数。"""
    params = {"task": "detect", "workers": 0, "resume": False,
              "deterministic": True, "patience": 0, "val": True,
              "save": True, "plots": False, "amp": False,
              "model": "C:/hpo/model.pt", "data": "C:/hpo/data.yaml",
              "epochs": 30, "seed": 42, "batch": 2, "imgsz": 64,
              "device": "cpu", "optimizer": "AdamW", "lr0": 0.0004,
              "lrf": 0.05, "momentum": 0.9, "weight_decay": 0.0005,
              "warmup_epochs": 2}
    return params


def _read_execution(root, study_id):
    path = root / "hpo" / study_id / "execution.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_execution(root, study_id, data):
    path = root / "hpo" / study_id / "execution.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _claim_prepared(service, study, runner):
    runner._finish_session(runner.status(study.study_id), "RUNNING", None)
    runner._claim_next(service.load_study(study.study_id),
                       runner.status(study.study_id), None)


@pytest.mark.parametrize("field,value", [
    ("epochs", 999),
    ("seed", 12345),
    ("batch", 999),
    ("imgsz", 32),
    ("device", "0,1"),
])
def test_effective_single_field_pollution_rejected_status_run_resume(
        ctx, field, value):
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["effective_params"][field] = value
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError):
        runner.status(study.study_id)
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    with pytest.raises(HpoError):
        runner.resume(study.study_id)
    assert FakeAdapter.launch_count == 0


def test_effective_and_command_together_polluted_rejected(ctx):
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    data = _read_execution(root, study.study_id)
    # 同时污染 effective.epochs 与命令里的 epochs 保持“自洽”也须拒绝。
    data["attempts"][0]["effective_params"]["epochs"] = 999
    data["attempts"][0]["command"] = [
        tok if not tok.startswith("epochs=") else "epochs=999"
        for tok in data["attempts"][0]["command"]]
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError):
        runner.status(study.study_id)
    assert FakeAdapter.launch_count == 0


def test_command_parameter_pollution_rejected_status(ctx):
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command"].append("lr0=999")
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError):
        runner.status(study.study_id)


def test_command_subcommand_or_extra_override_rejected(ctx):
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    data = _read_execution(root, study.study_id)
    data["attempts"][0]["command"].insert(1, "segment")
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError):
        runner.status(study.study_id)


def test_unmutated_claimed_record_status_ok(ctx):
    service, study, runner, root = ctx
    _claim_prepared(service, study, runner)
    runner.status(study.study_id)  # 不应抛错


def test_launch_rejects_command_inconsistent_with_hashed_args(tmp_path,
                                                             monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as module
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
                        lambda: "yolo-test")
    run = tmp_path / "trial"
    run.mkdir()
    effective = _full_effective()
    raw = yaml.safe_dump(effective).encode("utf-8")
    (run / "args.yaml").write_bytes(raw)
    good = build_yolo_command("trial", str(run / "args.yaml"),
                              dict(effective))
    # 命令尾参数被污染（lr0=999），与 args 内容/摘要不一致。
    poisoned = [tok if not tok.startswith("lr0=") else "lr0=999"
                for tok in good]
    calls = []
    monkeypatch.setattr(module, "launch_training",
                        lambda *a, **kw: calls.append((a, kw)))
    prepared = dict(run_relpath="trial", trial_id="trial",
                    args_sha256=hashlib.sha256(raw).hexdigest(),
                    effective_params=effective, command=poisoned)
    with pytest.raises(HpoError):
        ExecutionAdapter(tmp_path, tmp_path / "log").launch(prepared)
    assert not calls


def test_launch_accepts_command_rebuilt_from_effective(tmp_path, monkeypatch):
    import auto_tune.modules.hpo.execution_adapter as module
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
                        lambda: "yolo-test")
    run = tmp_path / "trial"
    run.mkdir()
    effective = _full_effective()
    raw = yaml.safe_dump(effective).encode("utf-8")
    (run / "args.yaml").write_bytes(raw)
    rebuilt = build_yolo_command("trial", str(run / "args.yaml"),
                                 dict(effective))
    calls = []
    proc = object()
    monkeypatch.setattr(module, "launch_training",
                        lambda *a, **kw: calls.append((a, kw)) or proc)
    prepared = dict(run_relpath="trial", trial_id="trial",
                    args_sha256=hashlib.sha256(raw).hexdigest(),
                    effective_params=effective, command=rebuilt)
    returned = ExecutionAdapter(tmp_path, tmp_path / "log").launch(prepared)
    assert returned is proc
    assert calls and calls[0][1]["command"] == rebuilt


# ── R2c：COMPLETED 必须预算耗尽 ─────────────────────────────────

def test_empty_premature_completed_rejected(ctx):
    service, study, runner, root = ctx
    data = _read_execution(root, study.study_id)  # 刚 prepare，无 trial
    data["status"] = "COMPLETED"
    _write_execution(root, study.study_id, data)
    with pytest.raises(HpoError):
        runner.status(study.study_id)
    with pytest.raises(HpoError):
        runner.run(study.study_id)
    assert FakeAdapter.launch_count == 0


def test_full_budget_completed_ok_and_cli_success(ctx):
    service, study, runner, root = ctx
    FakeAdapter.returncodes = [0, 0]
    record = runner.run(study.study_id)
    assert record.status == "COMPLETED"
    runner.status(study.study_id)  # 合法预算耗尽不抛错
    rc = cli_main(["--snapshot-dir", study.snapshot_binding.snapshot_path,
                   "--model-path", study.model_binding.model_path,
                   "--storage-root", str(root / "cli"),
                   "--output-root", str(root / "cli-out"),
                   "--log-root", str(root / "cli-log"),
                   "--budget", "2"])
    assert rc == 0
