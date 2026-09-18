"""F1.1-B 阶段五 — 多轮调优确定性策略。

最佳结果、是否改善、基线更新与停止条件必须由代码确定：LLM 只提供诊断和参数
建议。这些测试锁定三件事——基线只在真改善后接替、失败的建议不会被无限重复、
停止原因是稳定的字符串常量（可写入审计）。
"""

import pytest

from auto_tune.modules.agent_engine.round_policy import (
    DEFAULT_MAX_CONSECUTIVE_NO_IMPROVEMENT,
    STOP_BUDGET_EXHAUSTED,
    STOP_KEEP_PARAMS,
    STOP_NO_IMPROVEMENT,
    STOP_OSCILLATION,
    STOP_REPEATED_CHANGE,
    STOP_REASONS,
    RoundState,
    change_signature,
    direction_of,
    evaluate_decision_before_training,
    evaluate_round,
)


def _adjust(**changes):
    return {
        "action": "adjust",
        "hyperparameter_changes": changes,
        "training_overrides": {},
    }


def _keep():
    return {"action": "keep_params", "hyperparameter_changes": {}, "training_overrides": {}}


def _round(state, decision, score, train_name="run1", previous=None, applied=None):
    return evaluate_round(
        state,
        train_name=train_name,
        score=score,
        decision=decision,
        previous_params=previous if previous is not None else {"lr0": 0.01},
        applied_params=applied if applied is not None else {"lr0": 0.005},
    )


def test_all_stop_reasons_are_stable_constants():
    assert STOP_REASONS == {
        STOP_KEEP_PARAMS, STOP_REPEATED_CHANGE, STOP_OSCILLATION,
        STOP_NO_IMPROVEMENT, STOP_BUDGET_EXHAUSTED,
    }


@pytest.mark.parametrize("old,new,expected", [
    (0.01, 0.02, "increase"), (0.02, 0.01, "decrease"),
    (False, True, "enable"), (True, False, "disable"),
    (0.01, 0.01, "unchanged"), (True, True, "unchanged"),
    (None, 0.01, "unchanged"),
])
def test_direction_of(old, new, expected):
    assert direction_of(old, new) == expected


def test_first_round_only_replaces_the_reference_baseline_when_strictly_better():
    """多轮调优的基线是**进入调优前的原参考运行**，不是第一个调优轮次。

    原参考 0.80、首轮 0.60：变差的首轮不得接替基线，并按既有无改善策略停止。
    """
    state, verdict = _round(RoundState(best_run="reference", best_score=0.80),
                            _adjust(lr0=0.005), 0.60, train_name="runA")

    assert verdict["improved"] is False
    assert state.best_run == "reference"
    assert state.best_score == 0.80
    assert verdict["best_run"] == "reference"
    assert verdict["next_reference_run"] == "reference"
    assert verdict["stop_reason"] == STOP_NO_IMPROVEMENT


def test_first_round_replaces_the_reference_baseline_when_strictly_better():
    state, verdict = _round(RoundState(best_run="reference", best_score=0.60),
                            _adjust(lr0=0.005), 0.80, train_name="runA")

    assert verdict["improved"] is True
    assert state.best_run == "runA"
    assert state.best_score == 0.80
    assert verdict["next_reference_run"] == "runA"
    assert verdict["stop_reason"] is None


def test_an_equal_score_never_replaces_the_reference_baseline():
    """「严格优于」才接替：与参考持平仍保留参考运行。"""
    state, verdict = _round(RoundState(best_run="reference", best_score=0.70),
                            _adjust(lr0=0.005), 0.70, train_name="runA")

    assert verdict["improved"] is False
    assert state.best_run == "reference"
    assert verdict["next_reference_run"] == "reference"


def test_an_unmeasured_reference_baseline_stays_unknown():
    """原参考指标缺失时基线分数保持未知（None），绝不补造 0。"""
    state = RoundState(best_run="reference", best_score=None)

    assert state.best_score is None

    state, verdict = _round(state, _adjust(lr0=0.005), None, train_name="runA")

    assert verdict["improved"] is False
    assert state.best_score is None
    assert state.best_run == "reference"


def test_better_round_replaces_the_baseline():
    state = RoundState(best_run="runA", best_score=0.5)
    state, verdict = _round(state, _adjust(lr0=0.004), 0.6, train_name="runB")

    assert verdict["improved"] is True
    assert state.best_run == "runB"
    assert verdict["next_reference_run"] == "runB"


def test_worse_round_keeps_the_previous_baseline():
    """指标下降时继续保留原最佳运行，绝不把变差的运行当新基线。"""
    state = RoundState(best_run="runA", best_score=0.6)
    state, verdict = _round(state, _adjust(lr0=0.004), 0.4, train_name="runB")

    assert verdict["improved"] is False
    assert state.best_run == "runA"
    assert state.best_score == 0.6
    assert verdict["next_reference_run"] == "runA"
    assert state.consecutive_no_improvement == 1


def test_missing_score_is_never_treated_as_improvement_nor_as_regression():
    """指标缺失是「无法判定」，既不算改善，也不算退化。

    Module B 分析失败属于部分成功；若把它计入无改善计数，一次读取失败就会
    误判为指标下滑并终止调优。
    """
    state, verdict = _round(RoundState(best_run="runA", best_score=0.6),
                            _adjust(lr0=0.004), None, train_name="runB")

    assert verdict["improved"] is False
    assert state.best_run == "runA"
    assert state.consecutive_no_improvement == 0
    assert verdict["stop_reason"] is None

    # 已有计数时保持不变（放大阈值，避免被停止条件遮住断言）
    state, _ = evaluate_round(
        RoundState(best_run="runA", best_score=0.6, consecutive_no_improvement=1),
        train_name="runC", score=None, decision=_adjust(lr0=0.003),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.003},
        max_consecutive_no_improvement=5)

    assert state.consecutive_no_improvement == 1


def test_first_round_without_score_falls_back_to_its_own_run():
    """指标缺失时无法确立基线；只能以本轮运行作基线，否则会重复分析同一 run。"""
    state, verdict = _round(RoundState(), _adjust(lr0=0.005), None, train_name="runA")

    assert state.best_run is None
    assert verdict["baseline_fallback"] is True
    assert verdict["next_reference_run"] == "runA"
    assert verdict["stop_reason"] is None


def test_repeating_an_ineffective_change_stops_the_loop():
    """不允许连续重复同一组无效参数。"""
    state, _ = _round(RoundState(), _adjust(lr0=0.005), 0.5, train_name="runA")

    state, verdict = _round(state, _adjust(lr0=0.005), 0.5, train_name="runB")

    assert verdict["repeated_change"] is True
    assert verdict["stop_reason"] == STOP_REPEATED_CHANGE


def test_a_different_change_is_not_a_repeat():
    state, _ = _round(RoundState(), _adjust(lr0=0.005), 0.5, train_name="runA")
    state, verdict = evaluate_round(
        state, train_name="runB", score=0.5, decision=_adjust(lr0=0.003),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.003},
        max_consecutive_no_improvement=2)

    assert verdict["repeated_change"] is False
    assert verdict["stop_reason"] is None


def test_keep_params_is_a_normal_terminal_state():
    state, verdict = _round(RoundState(best_run="runA", best_score=0.6), _keep(), 0.6)

    assert verdict["stop_reason"] == STOP_KEEP_PARAMS
    assert verdict["next_reference_run"] == "runA"


def test_oscillating_parameter_stops_the_loop():
    """同一个参数先增后减（来回摆动）时必须停止。"""
    state, _ = evaluate_round(
        RoundState(), train_name="runA", score=0.5, decision=_adjust(lr0=0.02),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.02})
    assert state.direction_history["lr0"] == ("increase",)

    state, verdict = evaluate_round(
        state, train_name="runB", score=0.5, decision=_adjust(lr0=0.005),
        previous_params={"lr0": 0.02}, applied_params={"lr0": 0.005})

    assert state.direction_history["lr0"] == ("increase", "decrease")
    assert verdict["oscillating_params"] == ["lr0"]
    assert verdict["stop_reason"] == STOP_OSCILLATION


def test_same_direction_twice_is_not_oscillation():
    state, _ = evaluate_round(
        RoundState(), train_name="runA", score=0.5, decision=_adjust(lr0=0.008),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.008})
    state, verdict = evaluate_round(
        state, train_name="runB", score=0.5, decision=_adjust(lr0=0.006),
        previous_params={"lr0": 0.008}, applied_params={"lr0": 0.006},
        max_consecutive_no_improvement=2)

    assert verdict["oscillating_params"] == []
    assert verdict["stop_reason"] is None


def test_consecutive_no_improvement_stops_the_loop():
    """本轮没超过历史最佳就结束——默认阈值必须能在默认预算内生效。

    真实多轮审计里 lr0 被逐轮下探而指标不动，正是靠这条守卫挡住。
    """
    assert DEFAULT_MAX_CONSECUTIVE_NO_IMPROVEMENT == 1

    state = RoundState(best_run="runA", best_score=0.9)
    state, verdict = _round(state, _adjust(lr0=0.005), 0.1, train_name="runB")

    assert verdict["improved"] is False
    assert verdict["stop_reason"] == STOP_NO_IMPROVEMENT
    assert verdict["next_reference_run"] == "runA"


def test_a_larger_threshold_tolerates_exploration_rounds():
    state = RoundState(best_run="runA", best_score=0.9)
    state, verdict = evaluate_round(
        state, train_name="runB", score=0.1, decision=_adjust(lr0=0.005),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.005},
        max_consecutive_no_improvement=2)

    assert verdict["stop_reason"] is None
    assert state.consecutive_no_improvement == 1


def test_improvement_resets_the_no_improvement_counter():
    state = RoundState(best_run="runA", best_score=0.5, consecutive_no_improvement=1)
    state, verdict = _round(state, _adjust(lr0=0.004), 0.7, train_name="runB")

    assert state.consecutive_no_improvement == 0
    assert verdict["stop_reason"] is None


def test_change_signature_is_order_independent():
    assert change_signature({"lr0": 0.005, "epochs": 90}) == change_signature(
        {"epochs": 90, "lr0": 0.005})
    assert change_signature({}) == ()


# ── Codex 复核 P2：训练前判定（keep_params / 重复变更 / 参数摆动） ──────────
#
# keep_params 与「重复同一组参数」「参数来回摆动」都不需要训练就能判定。旧实现
# 把它们拖到训练结束之后，等于每发现一次就白跑一轮训练。


def _pretrain(state, decision, previous=None, applied=None, train_name="runA"):
    return evaluate_decision_before_training(
        state, train_name=train_name, decision=decision,
        previous_params=previous if previous is not None else {"lr0": 0.01},
        applied_params=applied if applied is not None else {"lr0": 0.005},
    )


def test_keep_params_stops_before_training():
    verdict = _pretrain(RoundState(best_run="reference", best_score=0.8), _keep())

    assert verdict["stop_reason"] == STOP_KEEP_PARAMS
    assert verdict["phase"] == "pre_training"
    assert verdict["changes"] == {}
    assert verdict["best_run"] == "reference"


def test_repeating_an_executed_change_stops_before_training():
    state, _ = _round(RoundState(best_run="reference", best_score=0.5),
                      _adjust(lr0=0.005), 0.6, train_name="runA")

    verdict = _pretrain(state, _adjust(lr0=0.005), train_name="runB")

    assert verdict["repeated_change"] is True
    assert verdict["stop_reason"] == STOP_REPEATED_CHANGE


def test_repeat_is_judged_on_the_applied_values_not_the_requested_ones():
    """护栏夹紧后形成的重复必须识别出来：请求值不同、实际执行值相同。"""
    state, _ = _round(RoundState(best_run="reference", best_score=0.5),
                      _adjust(lr0=0.000001), 0.6, train_name="runA",
                      applied={"lr0": 1e-5})

    verdict = _pretrain(state, _adjust(lr0=0.0000001), train_name="runB",
                        applied={"lr0": 1e-5})

    assert verdict["repeated_change"] is True
    assert verdict["stop_reason"] == STOP_REPEATED_CHANGE


def test_oscillation_stops_before_training():
    state, _ = evaluate_round(
        RoundState(best_run="reference", best_score=0.5), train_name="runA",
        score=0.6, decision=_adjust(lr0=0.02),
        previous_params={"lr0": 0.01}, applied_params={"lr0": 0.02})

    verdict = _pretrain(state, _adjust(lr0=0.005), train_name="runB",
                        previous={"lr0": 0.02}, applied={"lr0": 0.005})

    assert verdict["oscillating_params"] == ["lr0"]
    assert verdict["stop_reason"] == STOP_OSCILLATION


def test_a_new_change_does_not_stop_before_training():
    state, _ = _round(RoundState(best_run="reference", best_score=0.5),
                      _adjust(lr0=0.005), 0.6, train_name="runA")

    verdict = _pretrain(state, _adjust(lr0=0.004), train_name="runB",
                        applied={"lr0": 0.004})

    assert verdict["stop_reason"] is None
    assert verdict["repeated_change"] is False


def test_pretraining_never_stops_on_no_improvement():
    """无改善依赖本轮真实训练指标，训练前没有该结论，不得据此停止。"""
    state = RoundState(best_run="reference", best_score=0.9,
                       consecutive_no_improvement=1)

    verdict = _pretrain(state, _adjust(lr0=0.005))

    assert verdict["stop_reason"] is None


def test_pretraining_does_not_mutate_the_round_state():
    state = RoundState(best_run="reference", best_score=0.5)
    before = (state.best_run, state.best_score, state.tried_signatures,
              dict(state.direction_history))

    _pretrain(state, _adjust(lr0=0.005))

    assert (state.best_run, state.best_score, state.tried_signatures,
            dict(state.direction_history)) == before


def test_round_state_is_not_mutated_in_place():
    state = RoundState(best_run="runA", best_score=0.5)
    before = (state.best_run, state.best_score, state.consecutive_no_improvement,
              state.tried_signatures, dict(state.direction_history))

    _round(state, _adjust(lr0=0.004), 0.4, train_name="runB")

    assert (state.best_run, state.best_score, state.consecutive_no_improvement,
            state.tried_signatures, dict(state.direction_history)) == before
