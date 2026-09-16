"""H1.1 严格输入契约测试 — StudyConfig/记录模型/ResultInput 的强制校验。

合成数据，无真实训练/LLM。
"""

import pytest
from pydantic import ValidationError

from auto_tune.modules.hpo.models import (
    EnvironmentSnapshot,
    Evidence,
    HpoError,
    ModelBinding,
    ResultInput,
    ResultPayload,
    SnapshotBinding,
    StudyConfig,
    StudyRecord,
    TrialRecord,
)


# ── StudyConfig 严格性 ──────────────────────────────────────────────

@pytest.mark.parametrize("patch", [
    {"budget": True}, {"budget": "10"}, {"budget": 0}, {"budget": 101},
    {"seed": -1}, {"seed": 2 ** 31}, {"epochs": 0}, {"epochs": 1001},
    {"task": "classify"}, {"model_family": "yolov5"},
    {"sampler": "other"}, {"sampler_protocol": "other"},
    {"search_space_version": "unknown"}, {"objective": "unknown"},
    {"direction": "minimize"}, {"schema_version": "hpo-study-v2"},
    {"extra": 1},
])
def test_reject_invalid_config(patch):
    with pytest.raises(ValidationError):
        StudyConfig(**patch)


def test_default_config_is_valid_and_fixed():
    cfg = StudyConfig()
    assert cfg.schema_version == "hpo-study-v1"
    assert cfg.task == "detect"
    assert cfg.model_family == "yolov8"
    assert cfg.sampler == "tpe"
    assert cfg.sampler_protocol == "rebuild-per-trial-v1"
    assert cfg.search_space_version == "detect-hpo-v1"
    assert cfg.objective == "val_map50_95_best_epoch_v1"
    assert cfg.direction == "maximize"
    assert cfg.budget == 10
    assert cfg.seed == 42
    assert cfg.epochs == 30


@pytest.mark.parametrize("patch", [
    {"budget": 10.0}, {"budget": 10.5}, {"seed": 42.0}, {"epochs": 30.0},
])
def test_integer_fields_reject_float(patch):
    with pytest.raises(ValidationError):
        StudyConfig(**patch)


def test_valid_bounds_accepted():
    cfg = StudyConfig(budget=1, seed=0, epochs=1, sampler="random")
    assert cfg.budget == 1
    assert cfg.seed == 0
    assert cfg.epochs == 1
    assert cfg.sampler == "random"


# ── 数值输入一律拒绝 bool/str/numpy/非有限 ─────────────────────────

def test_result_value_rejects_bool_str_nan_inf_out_of_range():
    evidence = Evidence(run_id="run", artifact_relpath="val/pred/0.png",
                        artifact_sha256="0" * 64, epoch=1)
    for value in [True, "0.5", float("nan"), float("inf"), -0.1, 1.5, 2]:
        with pytest.raises(ValidationError):
            ResultPayload(value=value, evidence=evidence, reason_code=None)


def test_evidence_metric_key_fixed():
    with pytest.raises(ValidationError):
        Evidence(run_id="r", artifact_relpath="a/b.png",
                 artifact_sha256="0" * 64, epoch=1, metric_key="other")


# ── Evidence artifact_relpath 安全路径 ──────────────────────────────

@pytest.mark.parametrize("relpath", [
    "", "../escape.png", "a\\b.png", "/abs.png", "C:/abs.png",
    "a//b.png", "a/./b.png", "a/../b.png", ".", "a/",
])
def test_evidence_rejects_unsafe_artifact_relpath(relpath):
    with pytest.raises(ValidationError):
        Evidence(run_id="r", artifact_relpath=relpath,
                 artifact_sha256="0" * 64, epoch=1)


@pytest.mark.parametrize("relpath", [
    "val/pred/0.png", "runs/detect/run1/0.jpg", "metrics/x",
])
def test_evidence_accepts_safe_artifact_relpath(relpath):
    ev = Evidence(run_id="r", artifact_relpath=relpath,
                  artifact_sha256="0" * 64, epoch=1)
    assert ev.artifact_relpath == relpath


def test_evidence_rejects_bad_sha256_and_epoch():
    with pytest.raises(ValidationError):
        Evidence(run_id="r", artifact_relpath="a.png",
                 artifact_sha256="GG", epoch=1)
    with pytest.raises(ValidationError):
        Evidence(run_id="r", artifact_relpath="a.png",
                 artifact_sha256="0" * 64, epoch=0)
    with pytest.raises(ValidationError):
        Evidence(run_id="r", artifact_relpath="a.png",
                 artifact_sha256="0" * 64, epoch=True)


# ── ResultInput ─────────────────────────────────────────────────────

def test_result_input_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ResultInput(state="FAILED", reason_code="oom", unexpected=1)


def test_result_input_state_must_be_terminal():
    with pytest.raises(ValidationError):
        ResultInput(state="PENDING")


def test_result_input_reason_code_must_be_known():
    with pytest.raises(ValidationError):
        ResultInput(state="FAILED", reason_code="mystery")


# ── 绑定 / 环境 / 记录结构 ─────────────────────────────────────────

def _binding_fixture():
    snapshot = SnapshotBinding(snapshot_id="a" * 64,
                               manifest_digest="b" * 64,
                               snapshot_path=r"C:\snap\s1",
                               data_yaml_path=r"C:\snap\s1\data.yaml")
    model = ModelBinding(model_path=r"C:\models\y.pt", model_bytes=123,
                         model_sha256="c" * 64)
    env = EnvironmentSnapshot(python_version="3.10.18",
                              optuna_version="4.5.0",
                              numpy_version="2.2.6",
                              ultralytics_version="8.3.253")
    return snapshot, model, env


def _record(snapshot, model, env, trials=None):
    return StudyRecord(
        study_id="hpo_" + "d" * 32,
        created_at="2026-09-07T00:00:00.000000+00:00",
        updated_at="2026-09-07T00:00:00.000000+00:00",
        revision=0,
        config=StudyConfig(),
        snapshot_binding=snapshot,
        model_binding=model,
        environment=env,
        trials=trials or [],
    )


def _revalidate(record, **updates):
    data = record.model_dump(mode="json")
    data.update(updates)
    return StudyRecord.model_validate(data)


def test_record_study_id_must_match_pattern():
    snapshot, model, env = _binding_fixture()
    record = _record(snapshot, model, env)
    with pytest.raises(ValidationError):
        _revalidate(record, study_id="../../evil")


def test_record_rejects_negative_revision():
    snapshot, model, env = _binding_fixture()
    record = _record(snapshot, model, env)
    with pytest.raises(ValidationError):
        _revalidate(record, revision=-1)


def test_trial_record_number_trial_id_request_id_validation():
    with pytest.raises(ValidationError):
        TrialRecord(number=-1, trial_id="hpo_t0000", request_id="0" * 32,
                    state="PENDING", sampled_params={"a": 1},
                    distributions={"a": {"name": "x"}},
                    candidate_params={"a": 1},
                    created_at="2026-09-07T00:00:00.000000+00:00")
    with pytest.raises(ValidationError):
        TrialRecord(number=0, trial_id="bad", request_id="0" * 32,
                    state="PENDING", sampled_params={"a": 1},
                    distributions={"a": {"name": "x"}},
                    candidate_params={"a": 1},
                    created_at="2026-09-07T00:00:00.000000+00:00")
    with pytest.raises(ValidationError):
        TrialRecord(number=0, trial_id="hpo_t0000", request_id="not-hex",
                    state="PENDING", sampled_params={"a": 1},
                    distributions={"a": {"name": "x"}},
                    candidate_params={"a": 1},
                    created_at="2026-09-07T00:00:00.000000+00:00")


def test_trial_record_state_must_be_known():
    with pytest.raises(ValidationError):
        TrialRecord(number=0, trial_id="hpo_t0000", request_id="0" * 32,
                    state="RUNNING", sampled_params={"a": 1},
                    distributions={"a": {"name": "x"}},
                    candidate_params={"a": 1},
                    created_at="2026-09-07T00:00:00.000000+00:00")


def test_trial_record_rejects_state_result_mismatch():
    """SUCCESS 必须有带 value/evidence 的 result；PENDING 不得有 result。"""
    base = dict(number=0, trial_id="hpo_t0000", request_id="0" * 32,
                sampled_params={"a": 1},
                distributions={"a": {"name": "x"}},
                candidate_params={"a": 1},
                created_at="2026-09-07T00:00:00.000000+00:00")
    with pytest.raises(ValidationError):
        TrialRecord(**base, state="SUCCESS", result=None)
    with pytest.raises(ValidationError):
        TrialRecord(**base, state="PENDING", result=ResultPayload(value=0.5))


def test_result_payload_success_requires_evidence():
    """SUCCESS 结果对象缺少 evidence 属于结构非法。"""
    with pytest.raises(ValidationError):
        ResultPayload(value=0.5, evidence=None, reason_code=None)


def test_hpo_error_code_and_message():
    err = HpoError("HPO_NOT_FOUND", "no such study")
    assert err.code == "HPO_NOT_FOUND"
    assert err.message == "no such study"
    assert str(err) == "no such study"


# ── 第四轮：评价模式契约与旧记录兼容 ────────────────────────────────

def test_study_config_defaults_to_legacy_map50_95():
    """旧记录没有评价模式字段，默认必须是旧的单指标语义。"""
    cfg = StudyConfig()
    assert cfg.evaluation_mode == "legacy_map50_95"
    assert cfg.objective == "val_map50_95_best_epoch_v1"


@pytest.mark.parametrize("mode,objective", [
    ("quick", "quick_composite_best_epoch_v1"),
    ("comprehensive", "comprehensive_composite_best_epoch_v1"),
    ("legacy_map50_95", "val_map50_95_best_epoch_v1"),
])
def test_study_config_derives_the_versioned_objective_from_the_mode(mode, objective):
    cfg = StudyConfig(evaluation_mode=mode)
    assert cfg.evaluation_mode == mode
    assert cfg.objective == objective


@pytest.mark.parametrize("mode,objective", [
    ("quick", "val_map50_95_best_epoch_v1"),
    ("comprehensive", "quick_composite_best_epoch_v1"),
    ("legacy_map50_95", "comprehensive_composite_best_epoch_v1"),
])
def test_study_config_rejects_a_mode_objective_mismatch(mode, objective):
    """评价模式与目标版本必须互相自洽，不能被伪造组合。"""
    with pytest.raises(ValidationError):
        StudyConfig(evaluation_mode=mode, objective=objective)


def test_study_config_rejects_unknown_evaluation_mode():
    with pytest.raises(ValidationError):
        StudyConfig(evaluation_mode="bogus")


@pytest.mark.parametrize("patch", [
    {"evaluation_mode": 1}, {"evaluation_mode": None},
    {"objective": "quick_composite_best_epoch_v1"},
])
def test_study_config_rejects_invalid_mode_shapes(patch):
    with pytest.raises(ValidationError):
        StudyConfig(**patch)


def test_legacy_record_json_without_evaluation_mode_still_loads():
    """旧 study.json 没有 evaluation_mode 字段，必须按旧语义可读。"""
    cfg = StudyConfig.model_validate({
        "schema_version": "hpo-study-v1", "task": "detect",
        "model_family": "yolov8", "sampler": "tpe",
        "sampler_protocol": "rebuild-per-trial-v1",
        "search_space_version": "detect-hpo-v1",
        "objective": "val_map50_95_best_epoch_v1",
        "direction": "maximize", "budget": 3, "seed": 7, "epochs": 5,
    })
    assert cfg.evaluation_mode == "legacy_map50_95"
    assert cfg.objective == "val_map50_95_best_epoch_v1"


def _quick_evidence(**overrides):
    data = {
        "run_id": "tuning:r", "artifact_relpath": "results.csv",
        "artifact_sha256": "0" * 64, "epoch": 2,
        "evaluation_mode": "quick",
        "objective": "quick_composite_best_epoch_v1",
        "metrics": {"metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.6,
                    "metrics/precision(B)": 0.1, "metrics/recall(B)": 0.1},
    }
    data.update(overrides)
    return Evidence(**data)


def test_evidence_records_evaluation_mode_objective_and_components():
    ev = _quick_evidence()
    assert ev.evaluation_mode == "quick"
    assert ev.objective == "quick_composite_best_epoch_v1"
    assert ev.metrics["metrics/mAP50(B)"] == 0.6
    # 旧记录（无这些字段）仍然合法
    legacy = Evidence(run_id="r", artifact_relpath="results.csv",
                      artifact_sha256="0" * 64, epoch=1)
    assert legacy.evaluation_mode is None and legacy.metrics is None


@pytest.mark.parametrize("metrics", [
    {"metrics/mAP50(B)": True},
    {"metrics/mAP50(B)": "0.5"},
    {"metrics/mAP50(B)": float("nan")},
    {"metrics/mAP50(B)": float("inf")},
    {"metrics/mAP50(B)": 1.5},
    {"metrics/mAP50(B)": -0.1},
    {"unknown/metric": 0.5},
    {},
])
def test_evidence_rejects_illegal_metric_payloads(metrics):
    with pytest.raises(ValidationError):
        _quick_evidence(metrics=metrics)


def test_evidence_rejects_a_mode_objective_mismatch():
    with pytest.raises(ValidationError):
        _quick_evidence(objective="val_map50_95_best_epoch_v1")
