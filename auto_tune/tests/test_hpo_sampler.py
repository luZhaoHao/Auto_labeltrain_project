"""H1.1 Optuna 历史重建与候选生成测试 — 合成记录，无训练/LLM/网络。"""

import json

import pytest

from auto_tune.modules.hpo.models import (
    EnvironmentSnapshot,
    Evidence,
    HpoError,
    ModelBinding,
    ResultPayload,
    SnapshotBinding,
    StudyConfig,
    TrialRecord,
)
from auto_tune.modules.hpo.sampler import sample_next
from auto_tune.modules.hpo.search_space import (
    CANDIDATE_KEYS,
    LR0_RANGE,
    LRF_RANGE,
    MOMENTUM_RANGE,
    WEIGHT_DECAY_RANGE,
    validate_candidate,
)

SID = "hpo_" + "9" * 32


def _distributions(optimizer):
    key = "momentum_sgd" if optimizer == "SGD" else "beta1_adamw"
    lo, hi = MOMENTUM_RANGE[optimizer]
    raw = {
        "optimizer": {"name": "CategoricalDistribution",
                      "attributes": {"choices": ["SGD", "AdamW"]}},
        "lr0": {"name": "FloatDistribution", "attributes": {"low": LR0_RANGE[0],
                                                            "high": LR0_RANGE[1],
                                                            "log": True}},
        "lrf": {"name": "FloatDistribution", "attributes": {"low": LRF_RANGE[0],
                                                            "high": LRF_RANGE[1],
                                                            "log": True}},
        key: {"name": "FloatDistribution", "attributes": {"low": lo, "high": hi,
                                                          "log": False}},
        "weight_decay": {"name": "FloatDistribution",
                         "attributes": {"low": WEIGHT_DECAY_RANGE[0],
                                        "high": WEIGHT_DECAY_RANGE[1],
                                        "log": False}},
        "warmup_epochs": {"name": "IntDistribution",
                          "attributes": {"low": 0, "high": 5, "step": 1,
                                         "log": False}},
    }
    return {k: json.dumps(v) for k, v in raw.items()}


def _sampled(optimizer, momentum):
    key = "momentum_sgd" if optimizer == "SGD" else "beta1_adamw"
    return {"optimizer": optimizer, "lr0": 1e-4, "lrf": 0.05,
            key: momentum, "weight_decay": 0.0005, "warmup_epochs": 2}


def _candidate(optimizer, momentum):
    return {"optimizer": optimizer, "lr0": 1e-4, "lrf": 0.05,
            "momentum": momentum, "weight_decay": 0.0005, "warmup_epochs": 2}


def _evidence():
    return Evidence(run_id="run1", artifact_relpath="val/pred/0.png",
                    artifact_sha256="0" * 64, epoch=2)


def _record(number, state, optimizer, momentum, value=None, reason=None,
            distributions=None, sampled=None, candidate=None):
    if distributions is None:
        distributions = _distributions(optimizer)
    if sampled is None:
        sampled = _sampled(optimizer, momentum)
    if candidate is None:
        candidate = _candidate(optimizer, momentum)
    result = None
    if state == "SUCCESS":
        result = ResultPayload(value=value, evidence=_evidence(),
                               reason_code=None)
    elif state in ("FAILED", "CANCELLED", "INTERRUPTED"):
        result = ResultPayload(value=None, evidence=None, reason_code=reason)
    return TrialRecord(
        number=number, trial_id=f"{SID}_t{number:04d}",
        request_id=f"{number:032x}", state=state,
        sampled_params=sampled, distributions=distributions,
        candidate_params=candidate,
        created_at="2026-09-07T00:00:00.000000+00:00",
        finished_at="2026-09-07T00:00:01.000000+00:00" if state != "PENDING" else None,
        result=result,
    )


def _mixed_history(n=6):
    """交替 SGD/AdamW 的 n 个 SUCCESS 记录。"""
    history = []
    for i in range(n):
        optimizer = "SGD" if i % 2 == 0 else "AdamW"
        value = 0.4 + 0.05 * i
        history.append(_record(i, "SUCCESS", optimizer, 0.9, value=value))
    return history


def _cfg(**overrides):
    return StudyConfig(**overrides)


# ── 可重现与条件分布 ───────────────────────────────────────────────

def test_tpe_deterministic_with_six_success_history():
    config = _cfg(sampler="tpe", seed=7, epochs=30)
    history = _mixed_history(6)
    first = sample_next(config, history)
    second = sample_next(config, history)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_random_sampler_deterministic():
    config = _cfg(sampler="random", seed=3, epochs=30)
    history = _mixed_history(3)
    first = sample_next(config, history)
    second = sample_next(config, history)
    assert first[0] == second[0]
    assert first[1] == second[1]


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_sampler_candidate_within_bounds_and_valid(sampler):
    config = _cfg(sampler=sampler, seed=11, epochs=30)
    history = _mixed_history(6 if sampler == "tpe" else 2)
    sampled, candidate, distributions = sample_next(config, history)
    assert set(candidate) == CANDIDATE_KEYS
    assert set(sampled) == set(distributions)
    assert validate_candidate(candidate, epochs=30) == candidate
    assert candidate["optimizer"] in ("SGD", "AdamW")
    lo, hi = MOMENTUM_RANGE[candidate["optimizer"]]
    assert lo <= candidate["momentum"] <= hi
    assert LR0_RANGE[0] <= candidate["lr0"] <= LR0_RANGE[1]
    assert LRF_RANGE[0] <= candidate["lrf"] <= LRF_RANGE[1]
    assert 0 <= candidate["warmup_epochs"] <= 5


def test_distribution_json_matches_optuna_roundtrip():
    config = _cfg(sampler="random", seed=1, epochs=30)
    sampled, _, distributions = sample_next(config, _mixed_history(2))
    from optuna.distributions import json_to_distribution
    rebuilt = {k: json_to_distribution(v) for k, v in distributions.items()}
    assert set(rebuilt) == set(sampled)


def test_epochs_one_forces_warmup_zero_via_ask():
    config = _cfg(sampler="random", seed=1, epochs=1)
    sampled, candidate, _ = sample_next(config, [])
    assert candidate["warmup_epochs"] == 0
    assert isinstance(candidate["warmup_epochs"], int)


# ── 失败历史处理 ───────────────────────────────────────────────────

def test_failed_history_is_deterministic_and_valid():
    config = _cfg(sampler="tpe", seed=7, epochs=30)
    history = _mixed_history(5)
    history.append(_record(5, "FAILED", "SGD", 0.9, reason="training_failed"))
    first = sample_next(config, history)
    second = sample_next(config, history)
    assert first[0] == second[0]
    assert validate_candidate(first[1], epochs=30) == first[1]


def test_cancelled_and_interrupted_histories_allowed():
    config = _cfg(sampler="tpe", seed=5, epochs=30)
    history = _mixed_history(3)
    history.append(_record(3, "CANCELLED", "AdamW", 0.9,
                           reason="user_stopped"))
    history.append(_record(4, "INTERRUPTED", "SGD", 0.9,
                           reason="process_interrupted"))
    history.append(_record(5, "SUCCESS", "SGD", 0.9, value=0.7))
    sampled, candidate, _ = sample_next(config, history)
    assert validate_candidate(candidate, epochs=30) == candidate


# ── 非法历史拒绝 ───────────────────────────────────────────────────

def test_pending_history_rejected():
    config = _cfg(sampler="random", seed=1, epochs=30)
    history = _mixed_history(2)
    pending = _record(2, "PENDING", "SGD", 0.9)
    history.append(pending)
    with pytest.raises(HpoError) as err:
        sample_next(config, history)
    assert err.value.code == "HPO_INVALID_CONFIG"


def test_out_of_order_history_rejected():
    config = _cfg(sampler="random", seed=1, epochs=30)
    history = _mixed_history(2)
    history.insert(0, history.pop())
    with pytest.raises(HpoError) as err:
        sample_next(config, history)
    assert err.value.code == "HPO_INVALID_CONFIG"


def test_duplicate_number_history_rejected():
    config = _cfg(sampler="random", seed=1, epochs=30)
    history = _mixed_history(2)
    history.append(_record(1, "SUCCESS", "AdamW", 0.9, value=0.6))
    with pytest.raises(HpoError) as err:
        sample_next(config, history)
    assert err.value.code == "HPO_INVALID_CONFIG"


def test_malformed_distribution_rejected():
    config = _cfg(sampler="random", seed=1, epochs=30)
    history = _mixed_history(2)
    bad = _record(2, "SUCCESS", "SGD", 0.9, value=0.7)
    bad.distributions["lr0"] = json.dumps({"name": "Bogus", "attributes": {}})
    history.append(bad)
    with pytest.raises(HpoError) as err:
        sample_next(config, history)
    assert err.value.code == "HPO_CORRUPT_STUDY"


def test_sample_next_does_not_mutate_history():
    config = _cfg(sampler="random", seed=1, epochs=30)
    history = _mixed_history(3)
    snapshot = [t.model_dump(mode="json") for t in history]
    sample_next(config, history)
    assert [t.model_dump(mode="json") for t in history] == snapshot
