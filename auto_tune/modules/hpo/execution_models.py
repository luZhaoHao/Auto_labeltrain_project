"""H1.2 HPO 顺序执行契约与审计记录模型。

执行事实独立于 H1.1 study/trial 事实：``execution.json``（hpo-execution-v1）
只描述“执行器如何把一个已采样的候选跑成终态”，不改 H1.1 的搜索/采样/预算/
幂等语义。所有模型 ``extra='forbid'``，数值拒绝 bool/str/NumPy/NaN/Inf，公共
入口（prepare/run/resume/status）必须先 dump 再 ``model_validate``，不信任已
实例化对象。
"""

import math
import os
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import (
    HpoError,
    REQUEST_ID_RE,
    ResultInput,
    SHA256_RE,
    StrictInt,
    StrictModel,
    UTC_ISO,
    utc_now_iso,
)

EXECUTION_SCHEMA_VERSION = "hpo-execution-v1"

STUDY_ID_RE = re.compile(r"^hpo_[0-9a-f]{32}$")

DEVICE_RE = re.compile(r"^(?:cpu|0|[1-9]|[1-5][0-9]|6[0-3])$")
RUN_ID_RE = re.compile(r"^tuning:[A-Za-z0-9-]{1,200}$")

ATTEMPT_PHASES = Literal[
    "PREPARED", "LAUNCH_INTENT", "RUNNING", "EXITED",
    "RESULT_READY", "TOLD", "FINALIZED",
]
RECORD_STATUSES = Literal[
    "READY", "RUNNING", "PAUSED", "INTERRUPTED", "COMPLETED", "BLOCKED",
]
STOP_REASON = Literal["user_stopped", "timeout", "audit_failure"]
TERMINATION_REASON = Literal["user_stopped", "timeout", "audit_failure"]


def _validate_device(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("device must be a string")
    if not DEVICE_RE.fullmatch(value):
        raise ValueError("device must be 'cpu' or a single GPU index 0..63")
    return value


def _validate_run_id(value: Any) -> str:
    if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value):
        raise ValueError("run_id must be 'tuning:<uuid>'")
    return value


def _validate_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("run_relpath must be a non-empty relative path")
    if len(value) > 256 or "\\" in value:
        raise ValueError("run_relpath must be a short POSIX relative path")
    if value.startswith("/") or value.startswith(".") or "//" in value:
        raise ValueError("run_relpath must be normalized without dot segments")
    return value


def _validate_flat_params(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("params must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("param key must be a non-empty string")
        if item is None:
            continue
        if isinstance(item, bool):
            continue
        if type(item) not in (int, float, str, list):
            raise ValueError(f"unsupported param value for '{key}'")
        if isinstance(item, list):
            for element in item:
                if element is None or isinstance(element, bool):
                    continue
                if type(element) not in (int, float, str):
                    raise ValueError(f"unsupported list value for '{key}'")
                if type(element) in (int, float) and not math.isfinite(float(element)):
                    raise ValueError(f"non-finite list value for '{key}'")
        if type(item) in (int, float) and not math.isfinite(float(item)):
            raise ValueError(f"non-finite param value for '{key}'")
    return value


def _validate_command(value: Any) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError("command must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("command items must be non-empty strings")
    if len(value) > 1024:
        raise ValueError("command too long")
    if any(len(item) > 8192 for item in value):
        raise ValueError("command item too long")
    return value


def _validate_absolute_path(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("root path must be a string")
    if not value or not os.path.isabs(value):
        raise ValueError("root path must be absolute")
    normalized = os.path.normpath(value)
    if normalized != value:
        raise ValueError("root path must be already normalized")
    return value


# ── ExecutionConfig ───────────────────────────────────────────────

class ExecutionConfig(StrictModel):
    batch: StrictInt = Field(default=16, ge=1, le=256)
    imgsz: StrictInt = Field(default=640, ge=32, le=2048)
    device: str = Field(default="cpu")
    timeout_seconds: StrictInt = Field(default=3600, ge=1, le=86400)

    @field_validator("imgsz")
    @classmethod
    def _imgsz_multiple(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: Any) -> str:
        return _validate_device(value)


# ── Roots / Environment ──────────────────────────────────────────

class ExecutionRoots(StrictModel):
    storage_root: str = Field(min_length=1, max_length=2048)
    output_root: str = Field(min_length=1, max_length=2048)
    log_root: str = Field(min_length=1, max_length=2048)

    @field_validator("storage_root", "output_root", "log_root")
    @classmethod
    def _abs(cls, value: Any) -> str:
        return _validate_absolute_path(value)


class ExecutionEnvironment(StrictModel):
    sys_executable: str = Field(min_length=1, max_length=1024)
    python_version: str = Field(min_length=1, max_length=64)
    torch_version: str = Field(max_length=64)
    cuda_version: str = Field(max_length=64)
    ultralytics_version: str = Field(max_length=64)
    optuna_version: str = Field(max_length=64)
    numpy_version: str = Field(max_length=64)


class MetricDiagnostics(StrictModel):
    total_rows: StrictInt = Field(ge=0)
    excluded_rows: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _bounded(self) -> "MetricDiagnostics":
        if self.excluded_rows > self.total_rows:
            raise ValueError("excluded_rows must not exceed total_rows")
        return self


# ── ExecutionAttempt ─────────────────────────────────────────────

class ExecutionAttempt(StrictModel):
    trial_number: StrictInt = Field(ge=0)
    trial_id: str = Field(pattern=r"^hpo_[0-9a-f]{32}_t\d{4}$")
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str = Field(min_length=1, max_length=256)
    phase: ATTEMPT_PHASES
    candidate_params: dict[str, Any]
    effective_params: dict[str, Any]
    command: list[str]
    # R2a-3：与 command 分离冻结的 executable 身份（命令构造时由 resolve_yolo_executable
    # 独立解析），供历史校验作为依据；绝不从待验证的 command[0] 推导。
    command_executable: str = Field(min_length=1, max_length=1024)
    run_relpath: str = Field(min_length=1, max_length=256)
    args_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_args: dict[str, Any] | None = None
    actual_args_sha256: str | None = None
    metric_diagnostics: MetricDiagnostics | None = None
    pid: StrictInt | None = Field(default=None, ge=1)
    process_create_token: str | None = Field(default=None, max_length=512)
    started_at: UTC_ISO | None = None
    finished_at: UTC_ISO | None = None
    returncode: int | None = None
    termination_reason: TERMINATION_REASON | None = None
    result: ResultInput | None = None
    finalizer_record: dict | None = None
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=2048)

    @field_validator("candidate_params", "effective_params", "actual_args")
    @classmethod
    def _params(cls, value: Any) -> Any:
        if value is None:
            return value
        return _validate_flat_params(value)

    @field_validator("command")
    @classmethod
    def _cmd(cls, value: Any) -> list:
        return _validate_command(value)

    @field_validator("run_relpath")
    @classmethod
    def _rel(cls, value: Any) -> str:
        return _validate_relpath(value)

    @field_validator("run_id")
    @classmethod
    def _rid(cls, value: Any) -> str:
        return _validate_run_id(value)

    @field_validator("returncode")
    @classmethod
    def _rc(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            if value is None:
                return None
            raise ValueError("returncode must be an int or None")
        if type(value) is not int:
            raise ValueError("returncode must be a native int or None")
        return value

    @field_validator("command_executable")
    @classmethod
    def _exec(cls, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError("command_executable must be a string")
        if not value or len(value) > 1024:
            raise ValueError("command_executable must be a non-empty short string")
        return value

    @model_validator(mode="after")
    def _identity(self) -> "ExecutionAttempt":
        if not self.trial_id.endswith(f"_t{self.trial_number:04d}"):
            raise ValueError("trial_id and trial_number mismatch")
        if self.command and self.command[0] != self.command_executable:
            # R2a-3：命令首 token 必须与冻结 executable 一致（交叉字段一致性），
            # 不能允许用被验证的 command[0] 自身作为依据。
            raise ValueError("command_executable must equal the command token [0]")
        if self.pid is not None:
            if self.process_create_token is None:
                raise ValueError("pid requires process_create_token")
        if self.phase in ("RESULT_READY", "TOLD", "FINALIZED") and self.result is None:
            raise ValueError(f"{self.phase} attempt requires a result")
        if self.phase == "EXITED" and self.returncode is None:
            raise ValueError("EXITED attempt requires returncode")
        if self.finished_at is not None and self.phase in ("PREPARED", "LAUNCH_INTENT", "RUNNING"):
            raise ValueError("active attempt must not have finished_at")
        if self.termination_reason is not None and self.phase in ("PREPARED",):
            raise ValueError("PREPARED attempt must not carry termination_reason")
        return self


# ── ExecutionRecord ──────────────────────────────────────────────

class ExecutionRecord(StrictModel):
    schema_version: Literal["hpo-execution-v1"] = EXECUTION_SCHEMA_VERSION
    study_id: str = Field(pattern=r"^hpo_[0-9a-f]{32}$")
    revision: StrictInt = Field(ge=0)
    created_at: UTC_ISO
    updated_at: UTC_ISO
    config: ExecutionConfig
    roots: ExecutionRoots
    environment: ExecutionEnvironment
    status: RECORD_STATUSES = "READY"
    stop_reason: STOP_REASON | None = None
    attempts: list[ExecutionAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _attempts(self) -> "ExecutionRecord":
        numbers = [attempt.trial_number for attempt in self.attempts]
        if len(numbers) != len(set(numbers)):
            raise ValueError("attempt trial_numbers must be unique")
        if numbers != sorted(numbers):
            raise ValueError("attempts must be ordered by trial_number")
        for attempt in self.attempts:
            expected_id = f"{self.study_id}_t{attempt.trial_number:04d}"
            if attempt.trial_id != expected_id:
                raise ValueError(f"attempt trial_id mismatch for number {attempt.trial_number}")
        if self.attempts:
            # Sequential executor: at most the last attempt may be unfinished.
            if any(a.phase != "FINALIZED" for a in self.attempts[:-1]):
                raise ValueError("only the last attempt may be unfinished")
        if self.status == "READY" and self.attempts:
            raise ValueError("READY record must not carry attempts")
        return self


# ── 完整校验 ─────────────────────────────────────────────────────

def validate_execution(record, *, expected_study_id: str) -> ExecutionRecord:
    """读/写/重建共用的执行事实完整校验。

    即使传入的是已构造的 ExecutionRecord 实例，也从它的数据重新验证，防止
    可变对象绕过入口（H1.1 R4 同类问题）。身份与文件目录不一致即损坏。
    """
    try:
        data = (record.model_dump(mode="python", warnings=False)
                if isinstance(record, ExecutionRecord) else record)
        checked = ExecutionRecord.model_validate(data)
    except Exception as exc:
        if isinstance(exc, HpoError):
            raise exc
        raise HpoError("HPO_CORRUPT_EXECUTION",
                       "execution record failed contract validation") from exc
    if checked.study_id != expected_study_id:
        raise HpoError("HPO_CORRUPT_EXECUTION",
                       "execution record identity differs from requested study")
    return checked
