"""F1.1-B 阶段五 — 多轮调优的确定性策略（无 I/O、无 LLM）。

「最佳结果、是否改善、基线更新与停止条件由代码确定，LLM 只提供诊断和参数
建议」这一分工在这里落地：循环负责执行与审计落盘，本模块只做判定，且判定
完全由本轮真实参考运行的指标与参数推出，不看模型的自述。

基线规则：只有本轮指标**确实高于**历史最佳时才接替基线；指标下降时继续保留
原最佳运行。停止规则：重复同一组无效变更、参数来回摆动、连续多轮无改善，
或模型给出 ``keep_params``（正常终态）。所有停止原因都是稳定字符串常量，
由循环写入审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

STOP_KEEP_PARAMS = "keep_params"
STOP_REPEATED_CHANGE = "repeated_change"
STOP_OSCILLATION = "oscillation"
STOP_NO_IMPROVEMENT = "no_improvement"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"

STOP_REASONS = frozenset({
    STOP_KEEP_PARAMS, STOP_REPEATED_CHANGE, STOP_OSCILLATION,
    STOP_NO_IMPROVEMENT, STOP_BUDGET_EXHAUSTED,
})

# 连续无改善即停：默认 1，即「本轮没有超过历史最佳就结束」。
#
# 取 1 而非 2 是效率要求：`probe.max_retries` 默认 3，若容忍 2 轮无改善，就
# 需要 3 轮全部不改善才可能触发，在默认预算下这个守卫永远不会生效。真实
# 多轮审计（5 个会话 / 13 次轮次转换）显示：字面重复 = 0 次，真正的病态是
# lr0 单调下探（0.006→0.004→0.002、0.005→0.0025→0.00125）而每轮指标纹丝不
# 动——只有「没改善就停」能挡住它。代价是放弃「再试一轮也许更好」的探索，
# 由调用方按需调大。
DEFAULT_MAX_CONSECUTIVE_NO_IMPROVEMENT = 1

_INCREASE, _DECREASE, _ENABLE, _DISABLE, _UNCHANGED = (
    "increase", "decrease", "enable", "disable", "unchanged")
_REVERSE = {_INCREASE: _DECREASE, _DECREASE: _INCREASE,
            _ENABLE: _DISABLE, _DISABLE: _ENABLE}


def direction_of(old, new) -> str:
    """参数从 old 到 new 的变化方向；bool 只允许 false→true / true→false。"""
    if isinstance(old, bool) or isinstance(new, bool):
        if isinstance(old, bool) and isinstance(new, bool):
            if old is False and new is True:
                return _ENABLE
            if old is True and new is False:
                return _DISABLE
        return _UNCHANGED
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if new > old:
            return _INCREASE
        if new < old:
            return _DECREASE
    return _UNCHANGED


def change_signature(changes: dict) -> tuple:
    """变更集的规范签名：参数名与目标值排序后的元组（可哈希、可比较）。"""
    return tuple(sorted((str(k), repr(v)) for k, v in (changes or {}).items()))


def applied_changes(applied: dict) -> dict:
    """实际生效参数中的对外部分（剔除下划线前缀的内部字段）。"""
    return {k: v for k, v in (applied or {}).items() if not str(k).startswith("_")}


def requested_changes(decision: dict) -> dict:
    changes = dict((decision or {}).get("hyperparameter_changes") or {})
    changes.update((decision or {}).get("training_overrides") or {})
    return changes


def applied_change_signature(decision: dict, applied_params: dict) -> tuple:
    """本轮**实际会执行**的变更集签名。

    参数集合取自模型建议，目标值取护栏处理后的实际执行值：被夹紧到同一下界的
    两次不同建议实际会执行出完全相同的参数，因此必须判为同一变更。用请求值
    做签名会漏掉这类重复。
    """
    applied = applied_changes(applied_params)
    return change_signature({
        parameter: applied[parameter]
        for parameter in requested_changes(decision)
        if parameter in applied
    })


def _moved_directions(previous_params: dict, applied_params: dict) -> dict:
    """本轮相对上一轮真实发生了方向变化的参数。"""
    moved = {}
    for parameter, new_value in applied_changes(applied_params).items():
        if parameter not in previous_params:
            continue
        direction = direction_of(previous_params.get(parameter), new_value)
        if direction != _UNCHANGED:
            moved[parameter] = direction
    return moved


def _oscillating(state: "RoundState", moved: dict) -> list:
    """本轮方向与上一轮相反的参数（来回摆动）。"""
    return sorted(
        parameter for parameter, direction in moved.items()
        if state.direction_history.get(parameter)
        and _REVERSE.get(state.direction_history[parameter][-1]) == direction
    )


@dataclass(frozen=True)
class RoundState:
    """跨轮累积的确定性状态。全部字段不可变，逐轮 replace。"""

    best_run: str | None = None
    best_score: float | None = None
    consecutive_no_improvement: int = 0
    tried_signatures: frozenset = field(default_factory=frozenset)
    direction_history: dict = field(default_factory=dict)


def evaluate_round(
    state: RoundState,
    *,
    train_name: str,
    score: float | None,
    decision: dict,
    previous_params: dict,
    applied_params: dict,
    max_consecutive_no_improvement: int = DEFAULT_MAX_CONSECUTIVE_NO_IMPROVEMENT,
) -> tuple[RoundState, dict]:
    """判定一轮的结果，返回 ``(新状态, 判定)``。

    ``score`` 为本轮真实指标算出的综合分；缺失（None）时无法比较，按「未改善」
    处理，绝不假定改善。
    """
    action = (decision or {}).get("action")
    requested = requested_changes(decision)
    signature = applied_change_signature(decision, applied_params)

    moved = _moved_directions(previous_params, applied_params)

    improved = score is not None and (state.best_score is None or score > state.best_score)
    best_run = train_name if improved else state.best_run
    best_score = score if improved else state.best_score
    if improved:
        consecutive = 0
    elif score is None:
        # 指标缺失是「无法判定」，不是「变差了」：Module B 分析失败属于部分成功，
        # 不得据此推进无改善计数，否则一次读取失败就会误判为退化并终止调优。
        consecutive = state.consecutive_no_improvement
    else:
        consecutive = state.consecutive_no_improvement + 1

    oscillating = _oscillating(state, moved)

    tried = set(state.tried_signatures)
    repeated = bool(signature) and signature in tried
    tried.add(signature)

    if action == "keep_params":
        stop_reason = STOP_KEEP_PARAMS
    elif repeated:
        stop_reason = STOP_REPEATED_CHANGE
    elif oscillating:
        stop_reason = STOP_OSCILLATION
    elif consecutive >= max_consecutive_no_improvement:
        stop_reason = STOP_NO_IMPROVEMENT
    else:
        stop_reason = None

    history = dict(state.direction_history)
    for parameter, direction in moved.items():
        history[parameter] = (*history.get(parameter, ()), direction)

    # 首轮若因指标缺失而无法确立最佳运行，只能以本轮运行作为下一轮基线，
    # 否则下一轮会反复分析同一个参考运行。
    baseline_fallback = best_run is None
    next_reference_run = best_run if best_run is not None else train_name

    verdict = {
        "train_name": train_name,
        "phase": "post_training",
        "action": action,
        "changes": requested,
        "score": score,
        "improved": improved,
        "best_run": best_run,
        "best_score": best_score,
        "consecutive_no_improvement": consecutive,
        "baseline_fallback": baseline_fallback,
        "next_reference_run": next_reference_run,
        "repeated_change": repeated,
        "oscillating_params": oscillating,
        "stop_reason": stop_reason,
    }
    new_state = RoundState(
        best_run=best_run,
        best_score=best_score,
        consecutive_no_improvement=consecutive,
        tried_signatures=frozenset(tried),
        direction_history=history,
    )
    return new_state, verdict


def evaluate_decision_before_training(
    state: RoundState,
    *,
    train_name: str,
    decision: dict,
    previous_params: dict,
    applied_params: dict,
) -> dict:
    """训练前的确定性判定：``keep_params`` / 重复变更 / 参数摆动。

    只回答「这一轮还有没有必要训练」。改善与否取决于本轮的**真实训练指标**，
    必须等训练结束后由 :func:`evaluate_round` 判定，因此这里不涉及
    ``no_improvement``。

    判定使用的参数一律是护栏处理后的实际执行值，且不修改传入状态：
    未停止时状态必须保持原样，才能让训练后的判定只登记一次变更历史。
    """
    action = (decision or {}).get("action")
    requested = requested_changes(decision)
    signature = applied_change_signature(decision, applied_params)
    moved = _moved_directions(previous_params, applied_params)
    oscillating = _oscillating(state, moved)
    repeated = bool(signature) and signature in state.tried_signatures

    if action == "keep_params":
        stop_reason = STOP_KEEP_PARAMS
    elif repeated:
        stop_reason = STOP_REPEATED_CHANGE
    elif oscillating:
        stop_reason = STOP_OSCILLATION
    else:
        stop_reason = None

    return {
        "train_name": train_name,
        "phase": "pre_training",
        "action": action,
        "changes": requested,
        "score": None,
        "improved": False,
        "best_run": state.best_run,
        "best_score": state.best_score,
        "consecutive_no_improvement": state.consecutive_no_improvement,
        "baseline_fallback": state.best_run is None,
        "next_reference_run": state.best_run if state.best_run is not None else train_name,
        "repeated_change": repeated,
        "oscillating_params": oscillating,
        "stop_reason": stop_reason,
    }
