"""H1.1 HpoService 组合与恢复幂等测试 — 合成快照/模型，无训练/LLM/网络。"""

import os
import uuid
from pathlib import Path

import pytest
from PIL import Image

from auto_tune.modules import hpo as hpo_module
from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import (
    Evidence,
    HpoError,
    HpoService,
    ResultInput,
    StudyConfig,
)
from auto_tune.modules.hpo.service import _current_environment


def _rid():
    return uuid.uuid4().hex


def _evidence(n=1, epoch=1):
    return Evidence(run_id=f"run-{n}", artifact_relpath=f"val/pred/{n}.png",
                    artifact_sha256="0" * 64, epoch=epoch)


@pytest.fixture
def hpo_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for n in range(4):
        Image.new("RGB", (16, 16)).save(source / f"{n}.jpg")
        (source / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                         encoding="utf-8")
    snapshot = create_dataset_snapshot(source, tmp_path / "snapshots",
                                       val_ratio=0.5, seed=42,
                                       class_names={0: "part"})
    model = tmp_path / "fixture.pt"
    model.write_bytes(b"hpo-contract-test-not-a-real-model")
    return snapshot.snapshot_path, model


@pytest.fixture
def root(tmp_path):
    return tmp_path / "hpo"


def _make_service(root):
    return HpoService(root)


# ── 计划给出的最小端到端用例 ───────────────────────────────────────

def test_pending_reload_and_budget(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=1), snapshot_dir=snapshot,
                                 model_path=model)
    key = _rid()
    trial = service.ask(study.study_id, request_id=key)
    restored = HpoService(root)
    assert restored.ask(study.study_id, request_id=key) == trial
    with pytest.raises(HpoError) as error:
        restored.ask(study.study_id, request_id=_rid())
    assert error.value.code == "HPO_PENDING_TRIAL"
    restored.tell(study.study_id, trial.number,
                  ResultInput(state="FAILED", reason_code="oom"))
    assert restored.ask(study.study_id, request_id=key).state == "FAILED"
    with pytest.raises(HpoError) as error:
        restored.ask(study.study_id, request_id=_rid())
    assert error.value.code == "HPO_BUDGET_EXHAUSTED"


# ── 配置 / 路径非法 → 零 study 发布 ────────────────────────────────

def test_create_rejects_invalid_config_and_publishes_nothing(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    with pytest.raises(HpoError) as err:
        service.create_study({"task": "classify"}, snapshot_dir=snapshot,
                             model_path=model)
    assert err.value.code == "HPO_INVALID_CONFIG"
    assert list(root.glob("hpo_*")) == []


def test_create_rejects_bad_snapshot_and_model(root, hpo_inputs, tmp_path):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    with pytest.raises(HpoError) as err:
        service.create_study(StudyConfig(), snapshot_dir=tmp_path / "nope",
                             model_path=model)
    assert err.value.code == "HPO_INVALID_CONFIG"
    assert list(root.glob("hpo_*")) == []

    with pytest.raises(HpoError) as err:
        service.create_study(StudyConfig(), snapshot_dir=snapshot,
                             model_path=tmp_path / "missing.pt")
    assert err.value.code == "HPO_INVALID_CONFIG"
    assert list(root.glob("hpo_*")) == []

    bad_suffix = tmp_path / "model.bin"
    bad_suffix.write_bytes(b"x")
    with pytest.raises(HpoError) as err:
        service.create_study(StudyConfig(), snapshot_dir=snapshot,
                             model_path=bad_suffix)
    assert err.value.code == "HPO_INVALID_CONFIG"
    assert list(root.glob("hpo_*")) == []


# ── 绑定变更 ───────────────────────────────────────────────────────

def test_model_modified_gives_binding_mismatch_old_json_unchanged(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    r1 = _rid()
    trial = service.ask(sid, request_id=r1)
    service.tell(sid, trial.number,
                 ResultInput(state="FAILED", reason_code="training_failed"))
    target = root / sid / "study.json"
    before = target.read_bytes()

    model.write_bytes(b"hpo-model-content-changed")
    with pytest.raises(HpoError) as err:
        service.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_BINDING_MISMATCH"
    assert target.read_bytes() == before


def test_snapshot_modified_gives_binding_mismatch(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    r1 = _rid()
    trial = service.ask(sid, request_id=r1)
    service.tell(sid, trial.number,
                 ResultInput(state="FAILED", reason_code="training_failed"))
    image = next(snapshot.glob("**/*.jpg"))
    image.write_bytes(b"tampered-image-content")
    with pytest.raises(HpoError) as err:
        service.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_BINDING_MISMATCH"


# ── 环境版本变化 ───────────────────────────────────────────────────

def test_env_mismatch_load_readable_ask_version_mismatch(root, hpo_inputs, monkeypatch):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id

    def _fake_env():
        return {**_current_environment(), "optuna_version": "99.0.0"}

    monkeypatch.setattr(hpo_module.service, "_current_environment", _fake_env)
    loaded = service.load_study(sid)
    assert loaded.study_id == sid
    with pytest.raises(HpoError) as err:
        service.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_VERSION_MISMATCH"


# ── 幂等 / 结果冲突 ────────────────────────────────────────────────

def test_tell_same_result_idempotent_revision_unchanged(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    result = ResultInput(state="SUCCESS", value=0.6, evidence=_evidence(1))
    first = service.tell(sid, trial.number, result)
    assert first.state == "SUCCESS"
    rev_after_first = service.load_study(sid).revision
    second = service.tell(sid, trial.number, result)
    assert second == first
    assert service.load_study(sid).revision == rev_after_first


def test_tell_conflicting_result_conflict(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    service.tell(sid, trial.number,
                 ResultInput(state="SUCCESS", value=0.6, evidence=_evidence(1)))
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     ResultInput(state="SUCCESS", value=0.7,
                                 evidence=_evidence(1)))
    assert err.value.code == "HPO_RESULT_CONFLICT"


def test_ask_after_terminal_returns_terminal_record(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    key = _rid()
    trial = service.ask(sid, request_id=key)
    service.tell(sid, trial.number,
                 ResultInput(state="CANCELLED", reason_code="user_stopped"))
    restored = service.ask(sid, request_id=key)
    assert restored.state == "CANCELLED"
    assert restored.result.reason_code == "user_stopped"


# ── INVALID_RESULT：保持 PENDING ───────────────────────────────────

def test_success_missing_evidence_stays_pending(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     {"state": "SUCCESS", "value": 0.6})
    assert err.value.code == "HPO_INVALID_RESULT"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"


@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf"), -0.2])
def test_success_bad_value_stays_pending(root, hpo_inputs, value):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     {"state": "SUCCESS", "value": value,
                      "evidence": _evidence(1)})
    assert err.value.code == "HPO_INVALID_RESULT"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"


def test_success_epoch_exceeds_config_stays_pending(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(epochs=10, budget=2),
                                 snapshot_dir=snapshot, model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     ResultInput(state="SUCCESS", value=0.6,
                                 evidence=_evidence(1, epoch=11)))
    assert err.value.code == "HPO_INVALID_RESULT"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"


@pytest.mark.parametrize("state,value,evidence,reason", [
    ("FAILED", 0.5, _evidence(1), "oom"),
    ("CANCELLED", 0.5, None, "user_stopped"),
    ("INTERRUPTED", 0.5, None, "process_interrupted"),
])
def test_failure_states_must_not_carry_value(root, hpo_inputs,
                                             state, value, evidence, reason):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     {"state": state, "value": value,
                      "evidence": evidence, "reason_code": reason})
    assert err.value.code == "HPO_INVALID_RESULT"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"


@pytest.mark.parametrize("state,reason", [
    ("FAILED", "user_stopped"),
    ("FAILED", "process_interrupted"),
    ("CANCELLED", "oom"),
    ("CANCELLED", "process_interrupted"),
    ("INTERRUPTED", "user_stopped"),
    ("INTERRUPTED", "timeout"),
])
def test_state_reason_code_must_match(root, hpo_inputs, state, reason):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    with pytest.raises(HpoError) as err:
        service.tell(sid, trial.number,
                     ResultInput(state=state, reason_code=reason))
    assert err.value.code == "HPO_INVALID_RESULT"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"


@pytest.mark.parametrize("state,reason", [
    ("FAILED", "timeout"),
    ("FAILED", "invalid_params"),
    ("CANCELLED", "user_stopped"),
    ("INTERRUPTED", "process_interrupted"),
])
def test_valid_failure_state_transitions(root, hpo_inputs, state, reason):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    done = service.tell(sid, trial.number,
                        ResultInput(state=state, reason_code=reason))
    assert done.state == state
    assert done.result.value is None
    assert done.result.evidence is None
    assert done.result.reason_code == reason


def test_tell_bad_trial_number_not_found(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    with pytest.raises(HpoError) as err:
        service.tell(study.study_id, 99,
                     ResultInput(state="FAILED", reason_code="oom"))
    assert err.value.code == "HPO_NOT_FOUND"


def test_ask_bad_request_id_invalid_config(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=1), snapshot_dir=snapshot,
                                 model_path=model)
    with pytest.raises(HpoError) as err:
        service.ask(study.study_id, request_id="not-hex!")
    assert err.value.code == "HPO_INVALID_CONFIG"


# ── 写盘失败：不返回、不复活、可重试 ───────────────────────────────

def test_ask_persistence_failure_publishes_nothing_then_retry(root, hpo_inputs, monkeypatch):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=3), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    with monkeypatch.context() as m:
        from auto_tune.modules.hpo import storage as hpo_storage

        def _fail_replace(src, dst):
            raise OSError("injected replace failure")

        m.setattr(hpo_storage.os, "replace", _fail_replace)
        with pytest.raises(HpoError) as err:
            service.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    restored = service.load_study(sid)
    assert len(restored.trials) == 0
    assert restored.revision == 0
    # 重试成功后发布第一个候选。
    trial = service.ask(sid, request_id=_rid())
    assert trial.number == 0
    assert service.load_study(sid).revision == 1


def test_tell_persistence_failure_keeps_pending_then_retry(root, hpo_inputs, monkeypatch):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=3), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    result = ResultInput(state="SUCCESS", value=0.6, evidence=_evidence(1))
    with monkeypatch.context() as m:
        from auto_tune.modules.hpo import storage as hpo_storage

        def _fail_replace(src, dst):
            raise OSError("injected replace failure")

        m.setattr(hpo_storage.os, "replace", _fail_replace)
        with pytest.raises(HpoError) as err:
            service.tell(sid, trial.number, result)
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert service.load_study(sid).trials[trial.number].state == "PENDING"
    done = service.tell(sid, trial.number, result)
    assert done.state == "SUCCESS"


# ── 并发 / 预算 ────────────────────────────────────────────────────

def test_two_services_one_pending_and_busy(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service1 = HpoService(root)
    study = service1.create_study(StudyConfig(budget=3), snapshot_dir=snapshot,
                                  model_path=model)
    sid = study.study_id
    service2 = HpoService(root)

    # 同进程锁非重入：持锁期间第二个服务立即 BUSY，不发布任何候选。
    with service1._store.locked(sid):
        with pytest.raises(HpoError) as err:
            service2.ask(sid, request_id=_rid())
        assert err.value.code == "HPO_STUDY_BUSY"
    assert len(service1.load_study(sid).trials) == 0

    # 顺序路径：pending 存在时新 request 为 PENDING_TRIAL。
    key = _rid()
    service1.ask(sid, request_id=key)
    with pytest.raises(HpoError) as err:
        service2.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_PENDING_TRIAL"
    assert len(service1.load_study(sid).trials) == 1


# ── 返回对象不可修改内部状态 ───────────────────────────────────────

def test_mutating_returned_record_does_not_affect_disk(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2), snapshot_dir=snapshot,
                                 model_path=model)
    sid = study.study_id
    loaded = service.load_study(sid)
    loaded.trials.append(object())
    loaded.config.budget = 100
    again = service.load_study(sid)
    assert len(again.trials) == 0
    assert again.config.budget == 2


# ── TPE 重载路径（≥5 SUCCESS 之后）────────────────────────────────

def test_tpe_reload_path_repeatable_without_extra_candidate(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(sampler="tpe", seed=7,
                                             budget=30, epochs=30),
                                 snapshot_dir=snapshot, model_path=model)
    sid = study.study_id
    for i in range(6):
        trial = service.ask(sid, request_id=_rid())
        service.tell(sid, trial.number,
                     ResultInput(state="SUCCESS", value=0.5 + i * 0.05,
                                 evidence=_evidence(i + 1)))
    key = _rid()
    candidate = service.ask(sid, request_id=key)
    assert candidate.number == 6
    revision_before = service.load_study(sid).revision

    # 重载后同 request 幂等返回，不新增候选、不改 revision。
    restored = HpoService(root)
    same = restored.ask(sid, request_id=key)
    assert same == candidate
    assert restored.load_study(sid).revision == revision_before
    with pytest.raises(HpoError) as err:
        restored.ask(sid, request_id=_rid())
    assert err.value.code == "HPO_PENDING_TRIAL"


def test_two_identical_studies_yield_same_next_candidate(root, hpo_inputs):
    snapshot, model = hpo_inputs
    config = StudyConfig(sampler="tpe", seed=7, budget=30, epochs=30)
    service_a = HpoService(root / "a")
    service_b = HpoService(root / "b")
    study_a = service_a.create_study(config, snapshot_dir=snapshot,
                                     model_path=model)
    study_b = service_b.create_study(config, snapshot_dir=snapshot,
                                     model_path=model)
    for svc, sid in ((service_a, study_a.study_id),
                     (service_b, study_b.study_id)):
        for i in range(5):
            trial = svc.ask(sid, request_id=_rid())
            svc.tell(sid, trial.number,
                     ResultInput(state="SUCCESS", value=0.5 + i * 0.05,
                                 evidence=_evidence(i + 1)))
    cand_a = service_a.ask(study_a.study_id, request_id=_rid())
    cand_b = service_b.ask(study_b.study_id, request_id=_rid())
    assert cand_a.candidate_params == cand_b.candidate_params
    assert cand_a.sampled_params == cand_b.sampled_params


# ── 第四轮：评价模式结果持久化与旧记录兼容 ──────────────────────────

def _quick_evidence(epoch=2):
    return Evidence(
        run_id="run-q", artifact_relpath="results.csv",
        artifact_sha256="1" * 64, epoch=epoch,
        evaluation_mode="quick", objective="quick_composite_best_epoch_v1",
        metrics={"metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.6,
                 "metrics/precision(B)": 0.1, "metrics/recall(B)": 0.1})


def _comprehensive_evidence(epoch=1):
    return Evidence(
        run_id="run-c", artifact_relpath="results.csv",
        artifact_sha256="2" * 64, epoch=epoch,
        evaluation_mode="comprehensive",
        objective="comprehensive_composite_best_epoch_v1",
        metrics={"metrics/mAP50(B)": 0.2, "metrics/mAP50-95(B)": 0.4,
                 "metrics/precision(B)": 0.9, "metrics/recall(B)": 0.9})


def test_new_mode_result_survives_json_reload_with_components(root, hpo_inputs):
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(
        StudyConfig(budget=2, epochs=5, evaluation_mode="quick"),
        snapshot_dir=snapshot, model_path=model)
    sid = study.study_id
    trial = service.ask(sid, request_id=_rid())
    service.tell(sid, trial.number,
                 ResultInput(state="SUCCESS", value=0.6,
                             evidence=_quick_evidence(2)))

    reloaded = HpoService(root).load_study(sid)
    assert reloaded.config.evaluation_mode == "quick"
    assert reloaded.config.objective == "quick_composite_best_epoch_v1"
    stored = reloaded.trials[trial.number]
    assert stored.result.value == 0.6
    assert stored.result.evidence.evaluation_mode == "quick"
    assert stored.result.evidence.epoch == 2
    assert stored.result.evidence.metrics["metrics/mAP50-95(B)"] == 0.6
    # 排名口径不变：按 result.value 降序
    from auto_tune.modules.hpo import rank_trials

    assert rank_trials(reloaded)[0].number == trial.number


def test_legacy_study_without_mode_reloads_and_ranks_by_map50_95(root, hpo_inputs):
    """旧 study.json 没有 evaluation_mode：按旧 mAP50-95 语义读取与排名。"""
    import json

    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=3, epochs=4),
                                 snapshot_dir=snapshot, model_path=model)
    sid = study.study_id
    for value in (0.30, 0.80, 0.55):
        trial = service.ask(sid, request_id=_rid())
        service.tell(sid, trial.number,
                     ResultInput(state="SUCCESS", value=value,
                                 evidence=_evidence(trial.number, epoch=1)))
    path = root / sid / "study.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["config"].pop("evaluation_mode", None)
    data["config"]["objective"] = "val_map50_95_best_epoch_v1"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    reloaded = HpoService(root).load_study(sid)
    assert reloaded.config.evaluation_mode == "legacy_map50_95"
    from auto_tune.modules.hpo import rank_trials

    ranked = rank_trials(reloaded)
    assert [t.number for t in ranked] == [1, 2, 0]
    assert ranked[0].result.value == 0.80


def test_success_without_mode_fields_still_accepted_for_legacy_studies(root, hpo_inputs):
    """旧语义的成功结果（只有 value+evidence）在旧研究里仍然可登记。"""
    snapshot, model = hpo_inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=2, epochs=3),
                                 snapshot_dir=snapshot, model_path=model)
    trial = service.ask(study.study_id, request_id=_rid())
    stored = service.tell(study.study_id, trial.number,
                          ResultInput(state="SUCCESS", value=0.42,
                                      evidence=_evidence(1, epoch=2)))
    assert stored.result.evidence.evaluation_mode is None
    assert stored.result.value == 0.42
