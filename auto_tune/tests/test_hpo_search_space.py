"""H1.1 条件搜索空间与候选校验测试 — 合成 FixedTrial，无训练/LLM。"""

import pytest
from optuna.trial import FixedTrial

from auto_tune.modules.agent_engine.guardrails import validate_and_clamp
from auto_tune.modules.hpo.models import HpoError
from auto_tune.modules.hpo.search_space import (
    CANDIDATE_KEYS,
    suggest_candidate,
    validate_candidate,
)


def _sgd_fixed():
    return FixedTrial({
        "optimizer": "SGD",
        "lr0": 1e-4,
        "lrf": 0.05,
        "momentum_sgd": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 2,
    })


def _adamw_fixed():
    return FixedTrial({
        "optimizer": "AdamW",
        "lr0": 1e-4,
        "lrf": 0.05,
        "beta1_adamw": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 2,
    })


def _good_candidate(optimizer="SGD", **overrides):
    p = {"optimizer": optimizer, "lr0": 1e-4, "lrf": 0.05,
         "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 2}
    p.update(overrides)
    return p


# ── 映射：SGD / AdamW 两分支 ────────────────────────────────────────

def test_sgd_branch_maps_momentum_sgd():
    sampled, candidate = suggest_candidate(_sgd_fixed(), epochs=30)
    assert "momentum_sgd" in sampled
    assert "beta1_adamw" not in sampled
    assert candidate["momentum"] == sampled["momentum_sgd"] == 0.9
    assert set(candidate) == CANDIDATE_KEYS


def test_adamw_branch_maps_beta1_adamw():
    sampled, candidate = suggest_candidate(_adamw_fixed(), epochs=30)
    assert "beta1_adamw" in sampled
    assert "momentum_sgd" not in sampled
    assert candidate["momentum"] == sampled["beta1_adamw"] == 0.9
    assert set(candidate) == CANDIDATE_KEYS


def test_candidate_exactly_six_training_keys():
    for fixed in (_sgd_fixed(), _adamw_fixed()):
        sampled, candidate = suggest_candidate(fixed, epochs=30)
        assert set(candidate) == CANDIDATE_KEYS
        assert set(candidate) == {"optimizer", "lr0", "lrf", "momentum",
                                  "weight_decay", "warmup_epochs"}
        # sampled 使用条件原名（momentum_sgd / beta1_adamw），共 6 键。
        assert len(sampled) == 6
        assert "momentum" not in sampled
        assert {"optimizer", "lr0", "lrf", "weight_decay", "warmup_epochs"} <= set(sampled)


def test_sampled_params_preserve_values_and_types():
    sampled, candidate = suggest_candidate(_sgd_fixed(), epochs=30)
    assert sampled["optimizer"] == "SGD"
    assert sampled["warmup_epochs"] == 2 and isinstance(sampled["warmup_epochs"], int)
    assert isinstance(sampled["lr0"], float)


def test_candidate_not_silently_clamped():
    p = dict(optimizer="SGD", lr0=0.001, lrf=0.5, momentum=0.9,
             weight_decay=0.0005, warmup_epochs=0)
    with pytest.raises(HpoError) as error:
        validate_candidate(p, epochs=1)
    assert error.value.code == "HPO_INVALID_CONFIG"
    assert p["lrf"] == 0.5


# ── 条件范围：momentum 取决于 optimizer ─────────────────────────────

@pytest.mark.parametrize("optimizer,low,high,value", [
    ("SGD", 0.8, 0.98, 0.79),
    ("SGD", 0.8, 0.98, 0.99),
    ("AdamW", 0.85, 0.95, 0.84),
    ("AdamW", 0.85, 0.95, 0.96),
])
def test_validate_rejects_momentum_outside_optimizer_range(optimizer, low, high, value):
    with pytest.raises(HpoError):
        validate_candidate(_good_candidate(optimizer, momentum=value), epochs=30)


def test_validate_accepts_boundary_momentum():
    assert validate_candidate(_good_candidate("SGD", momentum=0.8), epochs=30)
    assert validate_candidate(_good_candidate("SGD", momentum=0.98), epochs=30)
    assert validate_candidate(_good_candidate("AdamW", momentum=0.85), epochs=30)
    assert validate_candidate(_good_candidate("AdamW", momentum=0.95), epochs=30)


# ── 各参数越界 / 类型拒绝 ───────────────────────────────────────────

@pytest.mark.parametrize("patch", [
    {"lr0": 0.0}, {"lr0": 0.006}, {"lr0": float("nan")}, {"lr0": float("inf")},
    {"lrf": 0.001}, {"lrf": 0.11},
    {"weight_decay": -1e-6}, {"weight_decay": 0.002},
])
def test_validate_rejects_out_of_range_numeric(patch):
    with pytest.raises(HpoError) as error:
        validate_candidate(_good_candidate(**patch), epochs=30)
    assert error.value.code == "HPO_INVALID_CONFIG"


@pytest.mark.parametrize("patch", [
    {"optimizer": True}, {"optimizer": "Adam"}, {"optimizer": "auto"},
    {"momentum": "0.9"}, {"momentum": True}, {"lr0": "1e-4"},
    {"weight_decay": True}, {"warmup_epochs": "2"},
])
def test_validate_rejects_bool_str_wrong_kind(patch):
    with pytest.raises(HpoError) as error:
        validate_candidate(_good_candidate(**patch), epochs=30)
    assert error.value.code == "HPO_INVALID_CONFIG"


@pytest.mark.parametrize("warmup", [0.5, 2.0, 6, -1, True])
def test_validate_rejects_bad_warmup(warmup):
    with pytest.raises(HpoError) as error:
        validate_candidate(_good_candidate(warmup_epochs=warmup), epochs=30)
    assert error.value.code == "HPO_INVALID_CONFIG"


def test_validate_rejects_unknown_or_missing_key():
    with pytest.raises(HpoError):
        validate_candidate(_good_candidate(extra_key=1), epochs=30)
    p = _good_candidate()
    del p["lrf"]
    with pytest.raises(HpoError):
        validate_candidate(p, epochs=30)


def test_validate_does_not_mutate_input():
    p = _good_candidate()
    original = dict(p)
    validate_candidate(p, epochs=30)
    assert p == original


def test_validate_returns_dict_copy():
    p = _good_candidate()
    out = validate_candidate(p, epochs=30)
    assert out == p
    assert out is not p


# ── epochs=1 时 warmup 只能为 0 ─────────────────────────────────────

def test_epochs_one_forces_warmup_zero():
    with pytest.raises(HpoError):
        validate_candidate(_good_candidate(warmup_epochs=1), epochs=1)
    assert validate_candidate(_good_candidate(warmup_epochs=0), epochs=1)


# ── 与现有护栏交集：合法候选不被 clamp ────────────────────────────

def test_valid_candidate_passes_guardrails_without_clamp():
    for candidate in (_good_candidate("SGD"), _good_candidate("AdamW")):
        result = validate_and_clamp(dict(candidate))
        assert result.valid is True
        assert result.clamped == {}
        assert result.params == candidate


def test_epochs_two_allows_warmup_up_to_one():
    assert validate_candidate(_good_candidate(warmup_epochs=1), epochs=2)
    with pytest.raises(HpoError):
        validate_candidate(_good_candidate(warmup_epochs=2), epochs=2)
