"""H1.2 确定性排名测试 — 合成 study，无训练/LLM/网络。

rank_trials 只对 SUCCESS 排序：key=(-value, trial.number)；0.0 合法；空列表表示
无优胜；严格重验证，污染模型拒绝；返回深拷贝不改原对象；不调用 Optuna best_trial。
"""

import uuid

import pytest
from PIL import Image

from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import Evidence, HpoError, HpoService, ResultInput, StudyConfig
from auto_tune.modules.hpo.ranking import rank_trials


def _rid():
    return uuid.uuid4().hex


def _evidence(n=1):
    return Evidence(run_id=f"tuning:{'0' * 8}-{'0' * 4}-4{'0' * 3}-{'0' * 4}-{'0' * 12}",
                    artifact_relpath=f"hpo_x/{n}/results.csv",
                    artifact_sha256="0" * 64, epoch=1)


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
    model.write_bytes(b"hpo-ranking-test-not-a-real-model")
    return snapshot.snapshot_path, model


def _make_service(root):
    return HpoService(root)


def _run_study(root, inputs, budget, values=None, states=None):
    snapshot, model = inputs
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=budget),
                                  snapshot_dir=snapshot, model_path=model)
    for number in range(budget):
        trial = service.ask(study.study_id, request_id=_rid())
        state = (states or ["SUCCESS"])[number] if states else "SUCCESS"
        if state == "SUCCESS":
            result = ResultInput(state="SUCCESS", value=values[number],
                                 evidence=_evidence(number + 1))
        else:
            result = ResultInput(state="FAILED", reason_code="training_failed")
        service.tell(study.study_id, trial.number, result)
    return service.load_study(study.study_id)


def test_rank_orders_success_by_value_desc_then_number(tmp_path, hpo_inputs):
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=4,
                        values=[0.5, 0.8, 0.8, 0.3])
    ordered = rank_trials(record)
    assert [t.number for t in ordered] == [1, 2, 0, 3]
    assert [t.result.value for t in ordered] == [0.8, 0.8, 0.5, 0.3]


def test_rank_zero_is_legal_and_sorted(tmp_path, hpo_inputs):
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=3,
                        values=[0.0, 0.4, 0.2])
    ordered = rank_trials(record)
    assert [t.number for t in ordered] == [1, 2, 0]
    assert ordered[2].result.value == 0.0


def test_rank_all_failed_is_empty(tmp_path, hpo_inputs):
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=3,
                        values=[None, None, None],
                        states=["FAILED", "FAILED", "FAILED"])
    assert rank_trials(record) == []


def test_rank_returns_deep_copies_original_unchanged(tmp_path, hpo_inputs):
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=2,
                        values=[0.5, 0.9])
    original_numbers = [t.number for t in record.trials]
    original_values = [t.result.value for t in record.trials]
    ordered = rank_trials(record)
    ordered[0].result.value = 0.001
    ordered[0].number = 999
    again = rank_trials(record)
    assert again[0].number == 1
    assert again[0].result.value == 0.9
    assert [t.number for t in record.trials] == original_numbers
    assert [t.result.value for t in record.trials] == original_values


def test_rank_polluted_value_rejected(tmp_path, hpo_inputs):
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=2,
                        values=[0.5, 0.9])
    record.trials[0].result.value = 2.0
    with pytest.raises(HpoError) as err:
        rank_trials(record)
    assert err.value.code == "HPO_CORRUPT_STUDY"


def test_rank_mutated_instance_revalidated(tmp_path, hpo_inputs):
    """模型实例的返回对象即使被赋值污染，rank 也重新 dump→validate 拒绝。"""
    record = _run_study(tmp_path / "hpo", hpo_inputs, budget=2,
                        values=[0.5, 0.9])
    record.trials[1].number = 7
    with pytest.raises(HpoError) as err:
        rank_trials(record)
    assert err.value.code == "HPO_CORRUPT_STUDY"
