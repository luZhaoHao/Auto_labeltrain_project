"""H1.3 HPO HTTP API (``/api/hpo``) — configuration, start, query.

The module only reads/validates input and schedules or queries the already
accepted HpoService/HpoRunner. It never re-implements sampling, command
construction, ``tell`` or recovery: those stay the sole responsibility of the
H1.1/H1.2 modules. ``create`` prepares a study (no training); ``start``/``resume``
occupy the shared training slot and launch one background controller; all GET
routes are read-only and never call run/resume/tell/finalize.

Error bodies always carry ``error_code`` / ``error`` / ``next_action`` — safe,
displayable Chinese text with no absolute path or credential content.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auto_tune.modules.hpo import (
    ExecutionConfig,
    HpoError,
    HpoRunner,
    HpoService,
    StudyConfig,
    rank_trials,
    search_space_summary,
)
from auto_tune.modules.hpo.models import (
    COMPREHENSIVE_OBJECTIVE,
    ObjectiveCode,
)
from auto_tune.modules.hpo.search_space import evaluation_mode_label
from auto_tune.modules.run_state.manager import TrainingBusyError

from .hpo_controller import HpoController

_SEARCH_KEYS = ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")

# Stable HTTP mapping for HpoError codes. Unknown codes are a 500 by default.
_HPO_STATUS = {
    "HPO_INVALID_CONFIG": 422,
    "HPO_INVALID_EXECUTION_CONFIG": 422,
    "HPO_INVALID_RESULT": 422,
    "HPO_NOT_FOUND": 404,
    "HPO_PENDING_TRIAL": 409,
    "HPO_BUDGET_EXHAUSTED": 409,
    "HPO_RESULT_CONFLICT": 409,
    "HPO_EXECUTION_CONFLICT": 409,
    "HPO_STUDY_BUSY": 409,
    "HPO_EXECUTION_BUSY": 409,
    "HPO_RECOVERY_REQUIRED": 409,
    "HPO_VERSION_MISMATCH": 409,
    "HPO_BINDING_MISMATCH": 409,
    "HPO_NO_SUCCESS": 409,
    "HPO_CORRUPT_STUDY": 500,
    "HPO_CORRUPT_EXECUTION": 500,
    "HPO_PERSISTENCE_ERROR": 500,
    "HPO_INVALID_METRICS": 500,
}

_NEXT_ACTION = {
    "busy": "已有其他训练占用，请先停止或等待其完成后再操作。",
    "conflict": "该任务状态不允许此操作，请按界面提示先恢复或停止。",
    "recovery": "该任务处于需要处理的状态，请先检查并恢复；BLOCKED 不支持强制继续。",
    "input": "请检查并修正输入后重试。",
    "missing": "未找到该研究任务，请刷新列表重新选择。",
    "persist": "任务记录读写失败，系统未改变原有训练事实；请稍后重试或查看历史原始记录。",
    "no_success": "还没有可用的成功试验结果，无法复用最佳配置。",
}

_ERR_TEMPLATE = {
    "HPO_INVALID_CONFIG": ("input", "输入不合法。"),
    "HPO_INVALID_EXECUTION_CONFIG": ("input", "执行配置不合法。"),
    "HPO_INVALID_RESULT": ("input", "试验结果不合法。"),
    "HPO_NOT_FOUND": ("missing", "未找到该研究任务。"),
    "HPO_PENDING_TRIAL": ("conflict", "该研究已有进行中的候选。"),
    "HPO_BUDGET_EXHAUSTED": ("conflict", "试验预算已耗尽。"),
    "HPO_RESULT_CONFLICT": ("conflict", "试验结果已登记且不一致。"),
    "HPO_EXECUTION_CONFLICT": ("conflict", "执行状态冲突。"),
    "HPO_STUDY_BUSY": ("busy", "该研究任务暂忙，请稍后刷新重试。"),
    "HPO_EXECUTION_BUSY": ("busy", "执行器暂忙，请稍后重试。"),
    "HPO_RECOVERY_REQUIRED": ("recovery", "该执行需要恢复，不能直接继续。"),
    "HPO_VERSION_MISMATCH": ("conflict", "运行环境版本已改变。"),
    "HPO_BINDING_MISMATCH": ("conflict", "数据集快照或模型绑定已改变。"),
    "HPO_NO_SUCCESS": ("no_success", "没有可用成功结果。"),
    "HPO_CORRUPT_STUDY": ("persist", "研究任务记录已损坏，无法读取。"),
    "HPO_CORRUPT_EXECUTION": ("persist", "执行审计记录已损坏，无法读取。"),
    "HPO_PERSISTENCE_ERROR": ("persist", "记录持久化失败。"),
    "HPO_INVALID_METRICS": ("persist", "试验指标无法读取。"),
}

_STUDY_ID_KEYS = tuple(sorted(_ERR_TEMPLATE.keys()))

# Errors that only mean "another short transaction is in flight right now".
_TRANSIENT_BUSY_CODES = frozenset({"HPO_STUDY_BUSY", "HPO_EXECUTION_BUSY"})

# Bounded backoff for control requests that must first read the current facts.
# A concurrent read (the page's own 2s poll, another client, a manual GET) may
# hold a study transaction for the duration of one read; that must not turn a
# start/stop into a failure. Only these two transient codes are retried, and
# only finitely many times, so corruption still surfaces immediately.
_PREFLIGHT_BUSY_RETRY_DELAYS = (0.02, 0.05, 0.1)


def _read_with_busy_retry(read, delays=_PREFLIGHT_BUSY_RETRY_DELAYS):
    """Run a short read-only pre-flight, absorbing transient conflicts."""
    attempt = 0
    while True:
        try:
            return read()
        except HpoError as exc:
            if exc.code not in _TRANSIENT_BUSY_CODES or attempt >= len(delays):
                raise
            time.sleep(delays[attempt])
            attempt += 1


def _hpo_error_response(exc: HpoError) -> JSONResponse:
    """Controlled error body: stable code + fixed Chinese text + next action.

    ``exc.message`` is never echoed. Even a known ``HpoError`` is not assumed to
    carry display-safe text: it can embed absolute paths, file names, traceback
    fragments, commands or credentials from the layer that raised it. The
    template below is the single source of user-visible error text.
    """
    code = safe_hpo_error_code(exc.code)
    kind, message = _ERR_TEMPLATE.get(code, ("persist", _UNKNOWN_ERROR_TEXT))
    status = _HPO_STATUS.get(code, 500)
    return JSONResponse(
        {
            "error_code": code,
            "error": message,
            "next_action": _NEXT_ACTION[kind],
        },
        status_code=status,
    )


def _bad_request(message: str, error_code: str = "INVALID_HPO_INPUT") -> JSONResponse:
    return JSONResponse(
        {
            "error_code": error_code,
            "error": message,
            "next_action": _NEXT_ACTION["input"],
        },
        status_code=422,
    )


def _busy_response(message: str, error_code: str = "RUN_ALREADY_ACTIVE") -> JSONResponse:
    return JSONResponse(
        {
            "error_code": error_code,
            "error": message,
            "next_action": _NEXT_ACTION["busy"],
        },
        status_code=409,
    )


_UNKNOWN_ERROR_TEXT = "执行出现未知错误。"

# ── 安全字段错误投影 ──────────────────────────────────────────────
#
# Pydantic 的 loc/msg 属于底层实现细节（英文、可能带上下文对象），绝不能作为
# 用户可见文本返回。这里只把错误定位到**白名单字段名**并映射成稳定理由码与
# 固定中文修正方法；字段名本身来自客户端输入（extra 字段），因此也必须过滤成
# 形如标识符的安全短名，否则退化为通用名。

FIELD_ERROR_CODE = "INVALID_HPO_FIELD"

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

_FIELD_REASONS = (
    "FIELD_REQUIRED", "FIELD_TYPE", "FIELD_RANGE", "FIELD_MULTIPLE",
    "FIELD_VALUE", "FIELD_LENGTH", "FIELD_UNKNOWN", "FIELD_INVALID",
)

_FIELD_REASON_TEXT = {
    "FIELD_REQUIRED": "该字段为必填项，请填写后再提交。",
    "FIELD_TYPE": "字段类型不合法，请填写整数（不要用布尔值、小数或文本代替）。",
    "FIELD_RANGE": "数值超出允许范围，请按界面提示的上下限填写。",
    "FIELD_MULTIPLE": "图像尺寸必须是 32 的整数倍。",
    "FIELD_VALUE": "取值不在允许的选项内，请从界面提供的选项中选择。",
    "FIELD_LENGTH": "文本长度超出允许范围，请缩短后重试。",
    "FIELD_UNKNOWN": "不支持该字段，请移除后重试。",
    "FIELD_INVALID": "字段取值不合法，请检查后重试。",
}

_FIELD_TYPE_BY_ERROR = {
    "missing": "FIELD_REQUIRED",
    "string_too_short": "FIELD_REQUIRED",
    "string_too_long": "FIELD_LENGTH",
    "extra_forbidden": "FIELD_UNKNOWN",
    "greater_than": "FIELD_RANGE",
    "greater_than_equal": "FIELD_RANGE",
    "less_than": "FIELD_RANGE",
    "less_than_equal": "FIELD_RANGE",
    "int_type": "FIELD_TYPE",
    "int_parsing": "FIELD_TYPE",
    "int_from_float": "FIELD_TYPE",
    "float_type": "FIELD_TYPE",
    "float_parsing": "FIELD_TYPE",
    "finite_number": "FIELD_TYPE",
    "string_type": "FIELD_TYPE",
    "bool_type": "FIELD_TYPE",
    "dict_type": "FIELD_TYPE",
    "list_type": "FIELD_TYPE",
    "literal_error": "FIELD_VALUE",
    "enum": "FIELD_VALUE",
    "string_pattern_mismatch": "FIELD_VALUE",
}

# 自定义校验器抛 ValueError 时 pydantic 只给出 ``value_error``。底层 message
# 只用于**在服务端挑选稳定的理由码**，绝不进入响应；因此按内容特征匹配即可。
_VALUE_ERROR_MARKERS = (
    ("multiple of 32", "FIELD_MULTIPLE"),
    ("must be a native int", "FIELD_TYPE"),
    ("must be a native int or float", "FIELD_TYPE"),
    ("must be finite", "FIELD_TYPE"),
    ("device must be 'cpu'", "FIELD_VALUE"),
    ("device must be a string", "FIELD_TYPE"),
)


def _safe_field_name(exc: ValidationError) -> str:
    """Return a whitelisted-looking field name, never an arbitrary payload."""
    try:
        loc = exc.errors()[0].get("loc") or ()
        name = str(loc[-1]) if loc else "request"
    except Exception:
        return "request"
    return name if _FIELD_NAME_RE.match(name) else "request"


def _value_error_reason(exc: ValidationError) -> str:
    """Map a custom-validator ValueError to a whitelisted reason code."""
    try:
        ctx = exc.errors()[0].get("ctx") or {}
        raw = ctx.get("error")
        text = str(getattr(raw, "args", ("",))[0] if getattr(raw, "args", None) else "")
    except Exception:
        return "FIELD_INVALID"
    for marker, reason in _VALUE_ERROR_MARKERS:
        if marker in text:
            return reason
    return "FIELD_INVALID"


def field_error(field: str, reason: str) -> JSONResponse:
    """Field-located, fully redacted 422 for a rejected value of a known field."""
    safe_field = field if _FIELD_NAME_RE.match(str(field)) else "request"
    safe_reason = reason if reason in _FIELD_REASON_TEXT else "FIELD_INVALID"
    return JSONResponse(
        {
            "error_code": FIELD_ERROR_CODE,
            "field": safe_field,
            "reason_code": safe_reason,
            "error": _FIELD_REASON_TEXT[safe_reason],
            "next_action": _NEXT_ACTION["input"],
        },
        status_code=422,
    )


def field_error_response(exc: ValidationError) -> JSONResponse:
    """Field-located, fully redacted 422 for a rejected create/train request."""
    try:
        error_type = str(exc.errors()[0].get("type", ""))
    except Exception:
        error_type = ""
    field = _safe_field_name(exc)
    if error_type in _FIELD_TYPE_BY_ERROR:
        reason = _FIELD_TYPE_BY_ERROR[error_type]
    elif error_type == "value_error":
        reason = _value_error_reason(exc)
    else:
        reason = "FIELD_INVALID"
    return field_error(field, reason)

# Only our own HPO_* vocabulary may ever reach a client as an error_code, so a
# polluted / unexpected code cannot smuggle arbitrary text into a response.
_HPO_CODE_RE = re.compile(r"^HPO_[A-Z0-9_]{1,64}$")


def safe_hpo_error_code(code: object) -> str:
    """Return the stable HPO error code, or one fixed fallback code."""
    if isinstance(code, str) and _HPO_CODE_RE.match(code):
        return code
    return "HPO_EXECUTION_ERROR"


# ── Strict create payload ─────────────────────────────────────────


# 新建研究的评价模式边界。旧 ``hpo-study-v1`` 记录没有该字段，读取/恢复/排名必须
# 继续按旧单指标语义（``StudyConfig`` 的内部默认值保持不变）；但**新建**研究只允许
# 两种产品模式：``legacy_map50_95`` 会落在 Literal 之外，由字段级错误投影稳定拒绝，
# 因此当前创建 API 不可能新建一个 legacy 研究。
CreateEvaluationMode = Literal["quick", "comprehensive"]


class CreateStudyConfig(StudyConfig):
    """创建边界的评价模式：只能显式选择或默认为全面/快速。"""

    evaluation_mode: CreateEvaluationMode = "comprehensive"
    objective: ObjectiveCode = COMPREHENSIVE_OBJECTIVE


class CreateStudyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1, max_length=128)
    model_path: str = Field(min_length=1, max_length=1024)
    study_config: CreateStudyConfig = Field(default_factory=CreateStudyConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)


async def _read_empty_object(request: Request):
    """Return ``{}`` for an empty body; reject anything else (extra forbidden)."""
    raw = await request.body()
    text = raw.decode("utf-8") if raw else ""
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return _bad_request("请求体必须是 JSON 对象。")
    if not isinstance(obj, dict):
        return _bad_request("请求体必须是 JSON 对象。")
    if obj:
        return _bad_request("该接口只接受空 JSON 对象，不接受额外字段。")
    return {}


def _controller_for(manager, study_id):
    controller = manager.get(study_id)
    if controller is not None and getattr(controller, "run_kind", None) == "hpo" \
            and controller.is_active():
        return controller
    return None


def _terminal_controller_for(manager, study_id):
    """Return this study's *finished* HPO controller when it carries a failure.

    The worker records its terminal error on the controller and then moves it
    into the manager's bounded retention instead of unregistering it, so a
    status read (including one right after a browser refresh) can still report
    a stable ``error_code``/``next_action``. Reads never resurrect it as
    "active": only the error fact is projected.
    """
    controller = manager.get(study_id)
    if controller is None or getattr(controller, "run_kind", None) != "hpo":
        return None
    if controller.is_active() or not getattr(controller, "error_code", None):
        return None
    return controller


def _approved_for_formal(manager, study_id: str, *, execution_status: str | None,
                         has_success: bool = True) -> bool:
    """已完成、有成功结果、无活动 HPO 控制器的研究才可用于正式训练。

    这是 status 与 best-config 共用的**唯一**规则：两个只读投影必须给出同一个
    允许状态，前端只读取服务端字段，绝不自行推断。真正提交时仍会重新验证。
    """
    if execution_status != "COMPLETED" or not has_success:
        return False
    controller = manager.get(study_id) if manager is not None else None
    if controller is not None and getattr(controller, "run_kind", None) == "hpo" \
            and controller.is_active():
        return False
    return True


def _snapshot_dataset_names(list_snapshots) -> dict[str, str]:
    """一次 history 请求只构建一次的 ``snapshot_id → dataset_name`` 只读映射。

    名称只来自 router 已有的受控、已发布快照列表——不为每条 study 自行无界扫描
    样本清单，也不逐条重复读取 manifest。列表缺失/不可用时返回空映射，调用方
    诚实缺失；**绝不**回退到目录名、短 ID 或客户端路径。
    """
    if list_snapshots is None:
        return {}
    try:
        rows = list(list_snapshots())
    except Exception:  # defensive: a broken listing must not break history
        return {}
    names: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        snapshot_id = row.get("snapshot_id")
        name = row.get("dataset_name")
        if isinstance(snapshot_id, str) and snapshot_id \
                and isinstance(name, str) and name.strip():
            names.setdefault(snapshot_id, name)
    return names


def _dataset_display_name(study, names: dict[str, str]) -> str | None:
    """研究的数据集名称：必须与研究的**完整** ``snapshot_id`` 精确匹配。

    只接受权威快照投影里的完整身份；快照不存在、不可读、身份不匹配或缺少合法
    名称时返回 ``None``（界面显示“数据集不可用”），绝不按短 ID/目录名模糊匹配，
    也绝不把 ``source_root``/snapshot_path 等物理路径返回给客户端。
    """
    binding = getattr(study, "snapshot_binding", None)
    snapshot_id = getattr(binding, "snapshot_id", None)
    if not isinstance(snapshot_id, str) or not snapshot_id:
        return None
    name = names.get(snapshot_id)
    return name if isinstance(name, str) and name else None


def _execution_optional(runner, study_id):
    """Return the validated execution record, or None when not prepared yet.

    Transient busy and hard-corruption are surfaced as-is (never hidden). An
    unexpected failure is converted into a stable corrupt-record error with a
    fixed message — the original exception text (paths, tracebacks, commands)
    stays in the exception chain and never reaches the client.
    """
    try:
        return runner.status(study_id)
    except HpoError as exc:
        if exc.code == "HPO_NOT_FOUND":
            return None
        raise
    except Exception as exc:  # defensive: stable error for unexpected failures
        raise HpoError("HPO_CORRUPT_EXECUTION", "执行审计记录无法读取。") from exc


def _evidence_metrics(trial) -> dict | None:
    """被选 epoch 的原始组成指标（旧记录没有该字段时为 None）。"""
    result = trial.result if trial is not None else None
    if result is None or result.evidence is None:
        return None
    metrics = result.evidence.metrics
    return dict(metrics) if metrics else None


def _trial_display(trial) -> dict:
    result = trial.result
    value = None
    epoch = None
    reason_code = None
    if result is not None:
        value = result.value
        if result.evidence is not None:
            epoch = result.evidence.epoch
        reason_code = result.reason_code
    return {
        "trial_number": trial.number + 1,
        "number": trial.number,
        "trial_id": trial.trial_id,
        "state": trial.state,
        "value": value,
        "epoch": epoch,
        "reason_code": reason_code,
        "metrics": _evidence_metrics(trial),
        "created_at": trial.created_at,
        "finished_at": trial.finished_at,
    }


def _ranking_display(ranked) -> list[dict]:
    return [
        {
            "rank": idx + 1,
            "trial_number": trial.number + 1,
            "number": trial.number,
            "value": trial.result.value if trial.result is not None else None,
            "epoch": trial.result.evidence.epoch if trial.result is not None and trial.result.evidence is not None else None,
            "metrics": _evidence_metrics(trial),
        }
        for idx, trial in enumerate(ranked)
    ]


def _build_status_payload(service, runner, manager, study_id: str) -> dict:
    """Read-only status projection of one study (never starts or finalizes)."""
    study = service.load_study(study_id)
    execution = _execution_optional(runner, study_id)
    execution_status = "READY"
    execution_revision = 0
    stop_reason = None
    attempts = []
    if execution is not None:
        execution_status = execution.status
        execution_revision = execution.revision
        stop_reason = execution.stop_reason
        attempts = execution.attempts

    terminal = [t for t in study.trials if t.state != "PENDING"]
    success = [t for t in study.trials if t.state == "SUCCESS"]
    pending = [t for t in study.trials if t.state == "PENDING"]
    failed = [t for t in study.trials if t.state == "FAILED"]
    cancelled = [t for t in study.trials if t.state == "CANCELLED"]
    interrupted = [t for t in study.trials if t.state == "INTERRUPTED"]

    current_trial_number = None
    unfinished = [a for a in attempts if a.phase not in ("FINALIZED",)]
    if unfinished:
        current_trial_number = unfinished[-1].trial_number + 1
    elif pending:
        current_trial_number = study.trials[-1].number + 1

    controller = _controller_for(manager, study_id)
    control_active = controller is not None
    controller_error = None
    if controller is not None:
        controller_error = controller.error_code
    if controller_error is None:
        # A worker that already ended must not silently revert to READY: its
        # failure fact is retained until a new attempt (or retention expiry).
        finished = _terminal_controller_for(manager, study_id)
        if finished is not None:
            controller_error = finished.error_code

    ranked = rank_trials(study)

    can_resume = execution_status in ("PAUSED", "INTERRUPTED")
    can_stop = control_active and execution_status in ("READY", "RUNNING")

    error_code = controller_error
    if error_code is None and execution_status in ("BLOCKED",):
        error_code = "HPO_RECOVERY_REQUIRED"
    if error_code is not None:
        # A polluted / unexpected code must never reach the client verbatim.
        error_code = safe_hpo_error_code(error_code)
    next_action = None
    if error_code is not None:
        kind, _ = _ERR_TEMPLATE.get(error_code, ("persist", ""))
        next_action = _NEXT_ACTION[kind]

    return {
        "study_id": study_id,
        "execution_status": execution_status,
        "execution_revision": execution_revision,
        "stop_reason": stop_reason,
        "budget": study.config.budget,
        "claimed_count": len(terminal) + len(pending),
        "terminal_count": len(terminal),
        "success_count": len(success),
        "failed_count": len(failed),
        "cancelled_count": len(cancelled),
        "interrupted_count": len(interrupted),
        # 正在跑的候选数：未终态的 attempt 最多一条（顺序执行器契约）
        "running_count": len(unfinished),
        # 剩余槽位 = 预算 − 已占用（含 PENDING）；完成前不猜测成功数
        "remaining_count": max(0, study.config.budget - len(terminal) - len(pending)),
        "current_trial_number": current_trial_number,
        "evaluation_mode": study.config.evaluation_mode,
        "objective": study.config.objective,
        "control_active": control_active,
        "can_stop": can_stop,
        "can_resume": can_resume,
        "error_code": error_code,
        "next_action": next_action,
        "sampler": study.config.sampler,
        "seed": study.config.seed,
        "trials": [_trial_display(t) for t in study.trials],
        "ranking": _ranking_display(ranked),
        "has_success": bool(success),
        "snapshot_id": study.snapshot_binding.snapshot_id,
        "model_display": os.path.basename(study.model_binding.model_path),
        # ── 冻结详情（只读权威值，编辑草稿不得改动历史研究） ──
        "created_at": study.created_at,
        "revision": study.revision,
        "study_epochs": study.config.epochs,
        "snapshot_short_id": study.snapshot_binding.snapshot_id[:8],
        "batch": execution.config.batch if execution is not None else None,
        "imgsz": execution.config.imgsz if execution is not None else None,
        "device": execution.config.device if execution is not None else None,
        "timeout_seconds": (execution.config.timeout_seconds
                            if execution is not None else None),
        "search_space": _search_space_payload(study, execution),
        # ── 正式训练允许状态（与 best-config 同一服务端规则） ──
        "approved_for_formal_training": _approved_for_formal(
            manager, study_id, execution_status=execution_status,
            has_success=bool(success)),
        # ── 最佳结果与参数使用的只读入口 ──
        "best": _best_summary(service, study, runner),
    }


def _search_space_payload(study, execution) -> dict:
    """搜索配置与评价方式（来自运行时常量/规则），与固定条件明确区分。"""
    payload = search_space_summary(epochs=study.config.epochs,
                                   sampler=study.config.sampler,
                                   evaluation_mode=study.config.evaluation_mode)
    payload["fixed"] = {
        "epochs": study.config.epochs,
        "batch": execution.config.batch if execution is not None else None,
        "imgsz": execution.config.imgsz if execution is not None else None,
        "device": execution.config.device if execution is not None else None,
        "seed": study.config.seed,
        "snapshot": study.snapshot_binding.snapshot_id[:8],
        "model": os.path.basename(study.model_binding.model_path),
    }
    return payload


def _trial_source(study, trial) -> dict:
    return {
        "study_id": study.study_id,
        "trial_id": trial.trial_id,
        "trial_number": trial.number,
        "trial_display_number": trial.number + 1,
        "value": trial.result.value if trial.result is not None else None,
        "epoch": (trial.result.evidence.epoch if trial.result is not None
                  and trial.result.evidence is not None else None),
        # 评价模式/目标版本/组成指标：综合分数的口径必须随结果一起展示，
        # 不能把综合分数标成单独的 mAP50-95。
        "evaluation_mode": study.config.evaluation_mode,
        "objective": study.config.objective,
        "metrics": _evidence_metrics(trial),
    }


def _best_summary(service, study, runner) -> dict | None:
    """排名第一的成功试验摘要（无成功结果时为 None）。"""
    ranked = rank_trials(study)
    if not ranked:
        return None
    return _best_payload(service, runner, study, ranked)


def _best_payload(service, runner, study, ranked) -> dict:
    from .hpo_training import trial_artifacts_available

    top = ranked[0]
    payload = _trial_source(study, top)
    payload["search"] = {k: top.candidate_params.get(k) for k in _SEARCH_KEYS}
    payload["ranked_count"] = len(ranked)
    payload["is_tie"] = (len(ranked) > 1 and ranked[1].result is not None
                         and ranked[1].result.value == top.result.value)
    try:
        payload["artifacts"] = trial_artifacts_available(
            service, runner, study.study_id, top.trial_id)
    except Exception:
        payload["artifacts"] = {"best_pt_available": False,
                                "last_pt_available": False}
    return payload


# ── Router factory ────────────────────────────────────────────────


def create_hpo_router(*, service, runner, manager, resolve_snapshot, validate_model,
                      assert_training_slot_free, list_snapshots=None,
                      list_models=None) -> APIRouter:
    """Build the read-only/control HPO router bound to the given instances.

    ``resolve_snapshot(snapshot_id)`` returns the validated published snapshot
    directory (or raises ``HpoError``); ``validate_model(value)`` returns the
    validated absolute model path (or raises ``HpoError``). The three HPO roots
    are frozen inside ``service``/``runner`` at construction time and can never
    be changed by a request.

    ``assert_training_slot_free()`` is the caller's unified training-slot gate:
    it returns a 409 ``JSONResponse`` when the slot is occupied (live in-memory
    controller *or* a leftover persisted run whose process may still be alive)
    and ``None`` when a new real training may start. It is mandatory because the
    in-memory ``manager.reserve`` below can only see this service process; after
    a restart the persisted facts are the only protection against running two
    trainings at once. ``start``/``resume`` evaluate it before reserving,
    creating a controller or calling ``runner.run``/``resume``.

    ``list_snapshots()`` is the read-only published-snapshot listing used by the
    controlled snapshot selector; it is optional so a router can be built
    without one, and it never falls back to the global ``latest_dataset``.
    ``list_models()`` is the minimal local ``.pt`` listing: it must only scan
    controlled directories (never the whole system) and is likewise optional.
    """

    def _hpo_not_found_or_500(exc: Exception) -> JSONResponse:
        if isinstance(exc, HpoError):
            return _hpo_error_response(exc)
        return JSONResponse(
            {
                "error_code": "HPO_EXECUTION_ERROR",
                "error": "处理请求时发生未知错误。",
                "next_action": _NEXT_ACTION["persist"],
            },
            status_code=500,
        )

    async def _create(request: Request):
        try:
            payload = CreateStudyRequest.model_validate(await request.json())
        except ValidationError as exc:
            return field_error_response(exc)
        except (ValueError, TypeError, json.JSONDecodeError):
            return _bad_request("请求体不是合法的 JSON。")
        try:
            snapshot_dir = resolve_snapshot(payload.snapshot_id)
            model_path = validate_model(payload.model_path)
            study = service.create_study(
                payload.study_config, snapshot_dir=snapshot_dir, model_path=model_path)
            runner.prepare(study.study_id, payload.execution_config)
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)
        return JSONResponse(
            {
                "study_id": study.study_id,
                "execution_status": "READY",
                "error_code": None,
                "error": None,
                "next_action": None,
            },
            status_code=201,
        )

    async def _start(study_id: str, request: Request):
        body = await _read_empty_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            existing = _controller_for(manager, study_id)
            if existing is not None:
                return _status(study_id)
            # Absorb a concurrent read's short transaction conflict before
            # deciding anything: a poll must never make start/resume fail.
            execution = _read_with_busy_retry(
                lambda: _execution_optional(runner, study_id))
            if execution is None:
                return _hpo_error_response(
                    HpoError("HPO_EXECUTION_CONFLICT", "该研究尚未绑定执行配置，无法启动。"))
            if execution.status == "COMPLETED":
                return _status(study_id)
            if execution.status == "RUNNING":
                # persisted RUNNING with no live controller → cannot start again
                return _hpo_error_response(
                    HpoError("HPO_RECOVERY_REQUIRED",
                             "该研究已有未完成的执行记录，请检查并恢复。"))
            if execution.status in ("PAUSED", "INTERRUPTED"):
                return _hpo_error_response(
                    HpoError("HPO_EXECUTION_CONFLICT",
                             "该研究处于已暂停状态，请使用恢复继续。"))
            if execution.status == "BLOCKED":
                return _hpo_error_response(
                    HpoError("HPO_RECOVERY_REQUIRED",
                             "该研究处于 BLOCKED 状态，不支持强制继续。"))
            # Unified persisted gate BEFORE any reservation / controller /
            # runner side effect: restart-leftover manual, tuning or HPO
            # trainings (and unresolvable records) must block this start.
            gate = assert_training_slot_free()
            if gate is not None:
                return gate
            token, busy = _reserve(manager, "hpo", study_id)
            if busy is not None:
                return busy
            controller = HpoController(
                study_id=study_id, runner=runner, manager=manager,
                reservation_token=token)
            manager.register(controller)
            try:
                controller.start(resume=False)
            except Exception:
                _release(manager, token)
                manager.unregister(study_id)
                raise
            return JSONResponse(
                {
                    "study_id": study_id,
                    "started": True,
                    "execution_status": "RUNNING",
                    "error_code": None,
                    "error": None,
                    "next_action": None,
                },
                status_code=202,
            )
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)

    def _status(study_id: str):
        try:
            # One bounded retry window: a poll that lands inside the worker's
            # own short transaction should report the real state, not "busy".
            payload = _read_with_busy_retry(
                lambda: _build_status_payload(service, runner, manager, study_id))
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)
        return JSONResponse(payload)

    async def _stop(study_id: str, request: Request):
        body = await _read_empty_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            controller = _controller_for(manager, study_id)
            if controller is not None:
                controller.request_stop()
                return JSONResponse(
                    {
                        "study_id": study_id,
                        "stopped": False,
                        "stop_requested": True,
                        "message": "已提交停止请求，等待执行器收敛",
                        "error_code": None, "error": None, "next_action": None,
                    },
                    status_code=202,
                )
            execution = _read_with_busy_retry(
                lambda: _execution_optional(runner, study_id))
            if execution is not None and execution.status == "COMPLETED":
                return JSONResponse(
                    {
                        "study_id": study_id,
                        "stopped": True,
                        "stop_requested": False,
                        "message": "该任务已完成，无需停止",
                        "error_code": None, "error": None, "next_action": None,
                    },
                    status_code=200,
                )
            if execution is not None and execution.status == "RUNNING":
                return _hpo_error_response(
                    HpoError("HPO_RECOVERY_REQUIRED",
                             "控制器已丢失，无法停止原进程；请检查后按恢复流程处理。"))
            return JSONResponse(
                {
                    "study_id": study_id,
                    "stopped": True,
                    "stop_requested": False,
                    "message": "没有正在运行的执行",
                    "error_code": None, "error": None, "next_action": None,
                },
                status_code=200,
            )
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)

    async def _resume(study_id: str, request: Request):
        body = await _read_empty_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            existing = _controller_for(manager, study_id)
            if existing is not None:
                return _status(study_id)
            # Absorb a concurrent read's short transaction conflict before
            # deciding anything: a poll must never make start/resume fail.
            execution = _read_with_busy_retry(
                lambda: _execution_optional(runner, study_id))
            if execution is None:
                return _hpo_error_response(
                    HpoError("HPO_EXECUTION_CONFLICT", "该研究尚未绑定执行配置，无法恢复。"))
            if execution.status == "COMPLETED":
                return _status(study_id)
            if execution.status == "BLOCKED":
                return _hpo_error_response(
                    HpoError("HPO_RECOVERY_REQUIRED",
                             "BLOCKED 状态没有强制恢复入口，请先检查问题。"))
            # Same unified persisted gate as start: a resume is still a real
            # training launch and must not race a leftover live process.
            gate = assert_training_slot_free()
            if gate is not None:
                return gate
            token, busy = _reserve(manager, "hpo", study_id)
            if busy is not None:
                return busy
            controller = HpoController(
                study_id=study_id, runner=runner, manager=manager,
                reservation_token=token)
            manager.register(controller)
            try:
                controller.start(resume=True)
            except Exception:
                _release(manager, token)
                manager.unregister(study_id)
                raise
            return JSONResponse(
                {
                    "study_id": study_id,
                    "resumed": True,
                    "execution_status": "RUNNING",
                    "error_code": None, "error": None, "next_action": None,
                },
                status_code=202,
            )
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)

    def _list_studies(offset: int = Query(0), limit: int = Query(20)):
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return _bad_request("offset 必须是非负整数。")
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 100):
            return _bad_request("limit 必须是 1..100 的整数。")
        rows = []
        # 权威快照投影只构建一次：本页所有 study 共用同一个映射（不逐条扫描）
        dataset_names = _snapshot_dataset_names(list_snapshots)
        root = Path(service.storage_root)
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if not child.is_dir() or not child.name.startswith("hpo_"):
                    continue
                study_id = child.name
                try:
                    study = service.load_study(study_id)
                    execution_status = None
                    execution_error = None
                    try:
                        execution = _execution_optional(runner, study_id)
                        if execution is not None:
                            execution_status = execution.status
                    except HpoError as exc:
                        execution_error = exc.code
                    terminal = [t for t in study.trials if t.state != "PENDING"]
                    success = [t for t in study.trials if t.state == "SUCCESS"]
                    rows.append({
                        "study_id": study_id,
                        "readable": True,
                        "transient": execution_error in _TRANSIENT_BUSY_CODES,
                        "created_at": study.created_at,
                        "sampler": study.config.sampler,
                        "budget": study.config.budget,
                        "execution_status": execution_status,
                        "execution_error_code": execution_error,
                        "terminal_count": len(terminal),
                        "success_count": len(success),
                        # 研究级历史条目：数据集名称与评价模式都是用户可读事实。
                        # 名称只能来自权威已发布快照投影的完整身份；解析不出时诚实
                        # 返回 None（界面显示“数据集不可用”），绝不用短身份冒充。
                        "dataset_name": _dataset_display_name(study, dataset_names),
                        "evaluation_mode": study.config.evaluation_mode,
                        "objective": study.config.objective,
                        "objective_label": evaluation_mode_label(
                            study.config.evaluation_mode),
                    })
                except HpoError as exc:
                    # A short transaction conflict is a *temporary* state: the
                    # row must recover on the next refresh instead of looking
                    # like a corrupt record. Real corruption stays fatal.
                    rows.append({
                        "study_id": study_id,
                        "readable": False,
                        "transient": exc.code in _TRANSIENT_BUSY_CODES,
                        "created_at": None,
                        "sampler": None,
                        "budget": None,
                        "execution_status": None,
                        "execution_error_code": exc.code,
                        "terminal_count": None,
                        "success_count": None,
                        "dataset_name": None,
                        "evaluation_mode": None,
                        "objective": None,
                        "objective_label": None,
                    })
        # newest first, stable id tiebreak; unreadable (no created_at) at the end
        readable = [r for r in rows if r["created_at"]]
        unreadable = [r for r in rows if not r["created_at"]]
        readable.sort(key=lambda r: r["study_id"])
        readable.sort(key=lambda r: r["created_at"], reverse=True)
        ordered = readable + unreadable
        total = len(ordered)
        page = ordered[offset:offset + limit]
        return JSONResponse({"studies": page, "count": total, "offset": offset, "limit": limit})

    def _best_config(study_id: str):
        try:
            study = service.load_study(study_id)
            ranked = rank_trials(study)
            if not ranked:
                return _hpo_error_response(
                    HpoError("HPO_NO_SUCCESS", "该研究没有可用的成功试验结果。"))
            top = ranked[0]
            execution = _execution_optional(runner, study_id)
            if execution is None:
                return _hpo_error_response(
                    HpoError("HPO_EXECUTION_CONFLICT", "该研究尚未绑定执行配置。"))
            six = {k: top.candidate_params.get(k) for k in _SEARCH_KEYS}
            best = _best_payload(service, runner, study, ranked)
            payload = {
                "study_id": study_id,
                "source": {
                    "study_id": study_id,
                    "trial_id": top.trial_id,
                    "trial_number": top.number,
                },
                "value": top.result.value if top.result is not None else None,
                "epoch": top.result.evidence.epoch if top.result is not None and top.result.evidence is not None else None,
                "evaluation_mode": study.config.evaluation_mode,
                "objective": study.config.objective,
                "objective_label": evaluation_mode_label(study.config.evaluation_mode),
                "metrics": _evidence_metrics(top),
                "sampler": study.config.sampler,
                "seed": study.config.seed,
                "search": six,
                "fixed": {
                    "epochs": study.config.epochs,
                    "batch": execution.config.batch,
                    "imgsz": execution.config.imgsz,
                    "device": execution.config.device,
                },
                "snapshot_id": study.snapshot_binding.snapshot_id,
                "snapshot_short_id": study.snapshot_binding.snapshot_id[:8],
                "model_display": os.path.basename(study.model_binding.model_path),
                # 展示用只读扩展（不改变固定配置验证接口的语义）
                "is_tie": best["is_tie"],
                "ranked_count": best["ranked_count"],
                "trial_artifacts": best["artifacts"],
                "approved_for_formal_training": _approved_for_formal(
                    manager, study_id, execution_status=execution.status),
                "error_code": None, "error": None, "next_action": None,
            }
            return JSONResponse(payload)
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception as exc:  # defensive
            return _hpo_not_found_or_500(exc)

    def _trial_artifact(study_id: str, trial_id: str, name: str):
        """受控 Trial 权重下载：身份 + 白名单产物名，路径由服务端重建。"""
        from fastapi.responses import FileResponse

        from .hpo_training import (
            HpoArtifactError,
            artifact_error_response,
            resolve_trial_artifact,
        )

        try:
            path = resolve_trial_artifact(service, runner, study_id, trial_id, name)
        except HpoArtifactError as exc:
            return artifact_error_response(exc)
        except HpoError as exc:
            return _hpo_error_response(exc)
        except Exception:
            return artifact_error_response(
                HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE"))
        return FileResponse(str(path), filename=name,
                            media_type="application/octet-stream")

    def _local_models():
        """最小本地 .pt 选择接口：只扫描受控目录，不遍历系统磁盘。"""
        if list_models is None:
            return JSONResponse({"models": [], "count": 0})
        try:
            rows = [row for row in list(list_models()) if isinstance(row, dict)]
        except Exception:
            return JSONResponse(
                {"error_code": "HPO_LOCAL_MODEL_LIST_FAILED",
                 "error": "本地权重列表暂不可用。",
                 "next_action": _NEXT_ACTION["persist"]},
                status_code=503,
            )
        return JSONResponse({"models": rows, "count": len(rows)})

    def _snapshots():
        """受控快照选择器：只读已发布快照根，绝不用 latest 自动替代。"""
        if list_snapshots is None:
            return JSONResponse({"snapshots": [], "count": 0})
        try:
            rows = list(list_snapshots())
        except Exception as exc:  # defensive: never leak a path/traceback
            return JSONResponse(
                {"error_code": "HPO_SNAPSHOT_LIST_FAILED",
                 "error": "快照列表暂不可用。",
                 "next_action": _NEXT_ACTION["persist"]},
                status_code=503,
            )
        safe_rows = [row for row in rows if isinstance(row, dict)]
        return JSONResponse({"snapshots": safe_rows, "count": len(safe_rows)})

    router = APIRouter()
    router.add_api_route("/studies", _create, methods=["POST"], name="hpo_create")
    router.add_api_route("/studies", _list_studies, methods=["GET"], name="hpo_list")
    router.add_api_route("/studies/{study_id}", _status, methods=["GET"], name="hpo_status")
    router.add_api_route("/studies/{study_id}/start", _start, methods=["POST"], name="hpo_start")
    router.add_api_route("/studies/{study_id}/stop", _stop, methods=["POST"], name="hpo_stop")
    router.add_api_route("/studies/{study_id}/resume", _resume, methods=["POST"], name="hpo_resume")
    router.add_api_route("/studies/{study_id}/best-config", _best_config, methods=["GET"],
                         name="hpo_best_config")
    router.add_api_route("/snapshots", _snapshots, methods=["GET"], name="hpo_snapshots")
    router.add_api_route("/local-models", _local_models, methods=["GET"],
                         name="hpo_local_models")
    router.add_api_route("/studies/{study_id}/trials/{trial_id}/artifacts/{name}",
                         _trial_artifact, methods=["GET"], name="hpo_trial_artifact")
    return router


def _reserve(manager, kind: str, run_id: str):
    try:
        return manager.reserve(kind, run_id), None
    except TrainingBusyError:
        return None, _busy_response("已有训练在运行（普通训练/大模型调参/HPO），请先停止或等待完成。")


def _release(manager, token) -> None:
    if token is None:
        return
    try:
        manager.release(token)
    except Exception:
        pass
