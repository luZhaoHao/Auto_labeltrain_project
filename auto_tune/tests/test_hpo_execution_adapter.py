"""H1.2 严格训练适配器测试 — 合成快照/CSV，假进程，无真实训练/LLM/网络。

覆盖：候选护栏拒绝（GuardrailRejection）、预检缺失资源/GPU 不存在、专属目录与
args 写入、命令与审计一致、launch 使用已发布命令、collect 参数漂移/真实指标、
finalizer 三种独立失败（analysis/history/index）与无启动不调用。
"""

import os
import uuid
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.modules.agent_engine.executor import build_yolo_command
from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import Evidence, HpoError, HpoService, ResultInput, StudyConfig
from auto_tune.modules.hpo.execution_adapter import (
    CollectedOutcome,
    ExecutionAdapter,
    GuardrailRejection,
)
from auto_tune.modules.hpo.execution_models import (
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionRecord,
    ExecutionRoots,
)


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
    model.write_bytes(b"hpo-adapter-test-not-a-real-model")
    return snapshot.snapshot_path, model


@pytest.fixture
def study_fixture(tmp_path, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(tmp_path / "hpo")
    study = service.create_study(StudyConfig(budget=2, epochs=30),
                                 snapshot_dir=snapshot, model_path=model)
    trial = service.ask(study.study_id, request_id=_rid())
    return service, study, trial, tmp_path


def _adapter(tmp_path):
    return ExecutionAdapter(tmp_path / "out", tmp_path / "log")


def _cfg(**kw):
    data = dict(ExecutionConfig(device="cpu", batch=2, imgsz=64).model_dump())
    data.update(kw)
    return ExecutionConfig(**data)


# ── executor task/amp 命令映射 ───────────────────────────────────

def test_explicit_detect_and_amp_command(monkeypatch, tmp_path):
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
                        lambda: "yolo-test")
    cmd = build_yolo_command("trial", str(tmp_path / "trial" / "args.yaml"),
                             {"task": "detect", "amp": False, "epochs": 1})
    assert "task=detect" in cmd
    assert "amp=False" in cmd


# ── validate_binding 只读复验 ────────────────────────────────────

def test_validate_binding_rejects_changed_model_with_pending(study_fixture):
    service, study, trial, tmp_path = study_fixture
    assert service.validate_binding(study.study_id) is None
    model = Path(study.model_binding.model_path)
    model.write_bytes(b"changed-model-bytes")
    with pytest.raises(HpoError) as err:
        service.validate_binding(study.study_id)
    assert err.value.code == "HPO_BINDING_MISMATCH"


# ── check_trial 护栏拒绝 ─────────────────────────────────────────

class _StubTrial:
    def __init__(self, params):
        self.candidate_params = params


def test_check_trial_rejects_guardrail_clamped_candidate():
    adapter = ExecutionAdapter(Path("C:/out"), Path("C:/log"))
    bad = {"optimizer": "SGD", "lr0": 1e-4, "lrf": 0.05, "momentum": 0.9,
           "weight_decay": 0.5, "warmup_epochs": 2}
    with pytest.raises(GuardrailRejection):
        adapter.check_trial(_StubTrial(bad))


def test_check_trial_accepts_in_space_candidate():
    adapter = ExecutionAdapter(Path("C:/out"), Path("C:/log"))
    ok = {"optimizer": "AdamW", "lr0": 1e-4, "lrf": 0.05, "momentum": 0.9,
          "weight_decay": 0.0005, "warmup_epochs": 2}
    assert adapter.check_trial(_StubTrial(ok)) is not None


# ── prepare 目录/命令/args ───────────────────────────────────────

def test_prepare_writes_dedicated_dir_and_builds_command(study_fixture, tmp_path):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)
    assert run_dir.is_dir()
    assert (run_dir / "args.yaml").is_file()
    assert prepared["run_relpath"] == f"{study.study_id}/{trial.trial_id}"
    assert len(prepared["args_sha256"]) == 64
    assert prepared["candidate_params"] == trial.candidate_params
    command = prepared["command"]
    assert "yolo" in Path(command[0]).name.lower()
    assert f"name={trial.trial_id}" in command
    assert f"project={run_dir.parent}" in command
    assert "task=detect" in command and "amp=False" in command
    # effective 参数包含固定项与候选六键
    eff = prepared["effective_params"]
    assert eff["epochs"] == study.config.epochs
    assert eff["workers"] == 0
    for key in ("optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs"):
        assert eff[key] == trial.candidate_params[key]


def test_search_trials_keep_yolo_plots_disabled(study_fixture, tmp_path):
    """搜索阶段护栏：内存参数、落盘 args.yaml 与 YOLO 命令三处都必须关闭绘图。

    每个短试验都带 plots=True 会生成大量重复图表；正式训练改用 plots=True 时
    绝不能顺带打开搜索阶段的绘图（见 test_hpo_formal_training.py 的正式训练用例）。
    """
    import yaml

    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)

    assert prepared["effective_params"]["plots"] is False
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    assert args["plots"] is False
    assert "plots=False" in prepared["command"]
    assert not [arg for arg in prepared["command"] if arg.startswith("plots=")
                and arg != "plots=False"]


def test_prepare_rejects_gpu_unavailable(study_fixture, tmp_path, monkeypatch):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    monkeypatch.setattr("auto_tune.modules.hpo.execution_adapter._gpu_available",
                        lambda index: False)
    with pytest.raises(HpoError) as err:
        adapter.prepare(study, trial, _cfg(device="0"))
    assert err.value.code == "HPO_PREFLIGHT_FAILED"


def test_prepare_rejects_missing_local_model(study_fixture, tmp_path):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    os.remove(study.model_binding.model_path)
    with pytest.raises(HpoError) as err:
        adapter.prepare(study, trial, _cfg())
    assert err.value.code == "HPO_PREFLIGHT_FAILED"


# ── launch 使用已发布命令 ────────────────────────────────────────

def test_launch_uses_exact_persisted_command(study_fixture, tmp_path, monkeypatch):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_launch(train_name, args_path, merged_params, command):
        captured["command"] = command
        captured["train_name"] = train_name
        return FakeProcess()

    monkeypatch.setattr("auto_tune.modules.hpo.execution_adapter.launch_training",
                        fake_launch)
    proc = adapter.launch(prepared)
    assert isinstance(proc, FakeProcess)
    assert captured["command"] == prepared["command"] == prepared["command"]
    assert captured["train_name"] == trial.trial_id


# ── collect 真实指标 / 参数漂移 ──────────────────────────────────

def _write_results(run_dir, text):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.csv").write_text(text, encoding="utf-8")


def test_collect_success_uses_best_epoch(study_fixture, tmp_path):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)
    _write_results(run_dir, "epoch,metrics/mAP50-95(B)\n1,0.2\n2,0.9\n3,0.9\n")
    # Ultralytics 会在 args.yaml 中记录实际执行参数；这里把实际 args 覆盖成一致。
    (run_dir / "args.yaml").write_text(
        _args_yaml(study, trial, prepared), encoding="utf-8")

    attempt = _fake_attempt(study, prepared)
    outcome = adapter.collect(study, attempt)
    assert isinstance(outcome, CollectedOutcome)
    assert outcome.result is not None
    assert outcome.result.state == "SUCCESS"
    assert outcome.result.value == 0.9
    assert outcome.result.evidence.epoch == 2
    assert outcome.result.evidence.run_id == attempt.run_id
    assert outcome.diagnostics.excluded_rows == 0
    assert outcome.error_code is None


def test_collect_drifted_params_is_invalid_params(study_fixture, tmp_path):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)
    _write_results(run_dir, "epoch,metrics/mAP50-95(B)\n1,0.5\n")
    args = _load_args_yaml(prepared)
    args["lr0"] = 0.999
    (run_dir / "args.yaml").write_text(_dump_yaml(args), encoding="utf-8")
    attempt = _fake_attempt(study, prepared)
    outcome = adapter.collect(study, attempt)
    assert outcome.result is None
    assert outcome.reason_code == "invalid_params"


def test_collect_no_valid_metric_is_training_failed(study_fixture, tmp_path):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)
    _write_results(run_dir, "epoch,metrics/mAP50-95(B)\n1,\n")
    (run_dir / "args.yaml").write_text(
        _args_yaml(study, trial, prepared), encoding="utf-8")
    attempt = _fake_attempt(study, prepared)
    outcome = adapter.collect(study, attempt)
    assert outcome.result is None
    assert outcome.reason_code == "training_failed"
    assert outcome.error_code == "HPO_INVALID_METRICS"


# ── finalize 三种失败 ────────────────────────────────────────────

def test_finalize_surfaces_analysis_history_index_errors(study_fixture, tmp_path, monkeypatch):
    service, study, trial, root = study_fixture
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())

    cases = {
        "analysis_error": {"analysis_error": {"error_type": "a"}},
        "history_error": {"history_error": {"error_type": "h"}},
        "index_error": {"index_error": {"error_type": "i"}},
    }
    for kind, marker in cases.items():
        def fake_finalize(*args, **kwargs):
            return {"status": "completed", **marker}

        monkeypatch.setattr(
            "auto_tune.modules.hpo.execution_adapter.finalize_training_run",
            fake_finalize)
        attempt = _fake_attempt(study, prepared, state="SUCCESS")
        record = adapter.finalize(study, attempt)
        assert record[kind] is not None


# ── helper ───────────────────────────────────────────────────────

def _fake_attempt(study, prepared, state="SUCCESS"):
    from auto_tune.modules.hpo.execution_models import ExecutionAttempt
    run_id = f"tuning:{uuid.uuid4()}"
    result = None
    if state == "SUCCESS":
        result = ResultInput(state="SUCCESS", value=0.9,
                             evidence=Evidence(
                                 run_id=run_id,
                                 artifact_relpath=f"{study.study_id}/"
                                                  f"{prepared['run_relpath'].split('/')[-1]}/results.csv",
                                 artifact_sha256="0" * 64, epoch=1))
    attempt = ExecutionAttempt(
        trial_number=int(prepared["trial_number"]),
        trial_id=prepared["trial_id"],
        request_id="0" * 32,
        run_id=run_id,
        phase="EXITED",
        candidate_params=dict(prepared["candidate_params"]),
        effective_params=dict(prepared["effective_params"]),
        command=list(prepared["command"]),
        command_executable=list(prepared["command"])[0],
        run_relpath=prepared["run_relpath"],
        args_sha256=prepared["args_sha256"],
        returncode=0,
        result=result,
        started_at="2026-09-08T00:00:00.000000+00:00",
        finished_at="2026-09-08T00:01:00.000000+00:00",
    )
    return attempt


def _args_yaml(study, trial, prepared):
    return _dump_yaml(dict(prepared["effective_params"]))


def _load_args_yaml(prepared):
    import yaml
    run_dir = Path(prepared.get("_run_dir")) if "_run_dir" in prepared else None
    return dict(prepared["effective_params"])


def _dump_yaml(data):
    import yaml
    return yaml.safe_dump(data, sort_keys=False)


# ── 第四轮：collect 使用研究的评价模式 ──────────────────────────────

_QUICK_CSV_HEADER = ("epoch,metrics/precision(B),metrics/recall(B),"
                     "metrics/mAP50(B),metrics/mAP50-95(B)")


def test_collect_uses_the_study_evaluation_mode_and_keeps_components(
        tmp_path, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(tmp_path / "hpo")
    study = service.create_study(
        StudyConfig(budget=2, epochs=30, evaluation_mode="quick"),
        snapshot_dir=snapshot, model_path=model)
    trial = service.ask(study.study_id, request_id=_rid())
    adapter = _adapter(tmp_path)
    prepared = adapter.prepare(study, trial, _cfg())
    run_dir = adapter.run_dir(study, trial.trial_id)
    _write_results(run_dir,
                   _QUICK_CSV_HEADER + "\n"
                   "1,0.9,0.9,0.20,0.40\n"
                   "2,0.1,0.1,0.60,0.60\n"
                   "3,0.1,0.1,0.60,0.60\n")
    (run_dir / "args.yaml").write_text(
        _args_yaml(study, trial, prepared), encoding="utf-8")

    outcome = adapter.collect(study, _fake_attempt(study, prepared))
    assert outcome.result is not None
    assert outcome.result.value == pytest.approx(0.60)
    evidence = outcome.result.evidence
    assert evidence.epoch == 2                       # 同分取最早
    assert evidence.evaluation_mode == "quick"
    assert evidence.objective == "quick_composite_best_epoch_v1"
    assert evidence.metrics["metrics/precision(B)"] == 0.1
    assert evidence.metrics["metrics/mAP50-95(B)"] == 0.6
