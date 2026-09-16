"""H1.1 Detect HPO 严格输入契约与持久化记录模型。

所有模型使用 Pydantic v2，``extra='forbid'``；拒绝字符串转数字、bool 充当
整数/实数、NaN/Infinity 以及 NumPy 标量隐式转换。数值只允许原生 int/float。
"""

import math
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator

HPO_STUDY_SCHEMA = "hpo-study-v1"
SEARCH_SPACE_VERSION = "detect-hpo-v1"

# ── 评价模式（第四轮新增；旧记录没有该字段，缺失 = 旧单指标语义） ──────
#
# 每个模式只有一个确定性的最大化目标（不是 Pareto 多目标）。权重固定在后端
# 版本化契约里，普通用户不可编辑；目标版本码随模式一起冻结进 study/trial 事实。
EvaluationMode = Literal["legacy_map50_95", "quick", "comprehensive"]
LEGACY_MODE = "legacy_map50_95"
LEGACY_OBJECTIVE = "val_map50_95_best_epoch_v1"
QUICK_OBJECTIVE = "quick_composite_best_epoch_v1"
COMPREHENSIVE_OBJECTIVE = "comprehensive_composite_best_epoch_v1"
ObjectiveCode = Literal[
    "val_map50_95_best_epoch_v1",
    "quick_composite_best_epoch_v1",
    "comprehensive_composite_best_epoch_v1",
]

OBJECTIVE_BY_MODE = {
    LEGACY_MODE: LEGACY_OBJECTIVE,
    "quick": QUICK_OBJECTIVE,
    "comprehensive": COMPREHENSIVE_OBJECTIVE,
}

# results.csv 的原始指标列名（组成指标一律以列名为键，避免歧义）。
METRIC_MAP50 = "metrics/mAP50(B)"
METRIC_MAP50_95 = "metrics/mAP50-95(B)"
METRIC_PRECISION = "metrics/precision(B)"
METRIC_RECALL = "metrics/recall(B)"
METRIC_KEYS = (METRIC_MAP50, METRIC_MAP50_95, METRIC_PRECISION, METRIC_RECALL)

# 各模式的固定权重（后端版本化契约，用户不可编辑）。
EVALUATION_WEIGHTS = {
    LEGACY_MODE: {METRIC_MAP50_95: 1.0},
    "quick": {METRIC_MAP50: 0.10, METRIC_MAP50_95: 0.90},
    "comprehensive": {METRIC_MAP50: 0.10, METRIC_MAP50_95: 0.50,
                      METRIC_PRECISION: 0.20, METRIC_RECALL: 0.20},
}

STUDY_ID_RE = re.compile(r"^hpo_[0-9a-f]{32}$")
REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REASON_CODES = Literal[
    "training_failed", "timeout", "oom", "invalid_params",
    "user_stopped", "process_interrupted",
]

ALLOWED_REASON_BY_STATE = {
    "FAILED": frozenset({"training_failed", "timeout", "oom", "invalid_params"}),
    "CANCELLED": frozenset({"user_stopped"}),
    "INTERRUPTED": frozenset({"process_interrupted"}),
}

_UTC_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class HpoError(Exception):
    """HPO 稳定错误：code + 面向用户的 message（不含凭据）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def required_metric_keys(mode: str) -> tuple[str, ...]:
    """该评价模式评分所必需的原始指标列（按固定权重顺序）。"""
    weights = EVALUATION_WEIGHTS.get(mode)
    if weights is None:
        raise HpoError("HPO_INVALID_METRICS", "unknown evaluation mode")
    return tuple(weights)


def evaluation_score(mode: str, metrics: dict) -> float | None:
    """确定性综合分数：任一必需指标缺失/非法返回 ``None``，绝不补 0。"""
    weights = EVALUATION_WEIGHTS.get(mode)
    if weights is None or not isinstance(metrics, dict):
        return None
    total = 0.0
    for key, weight in weights.items():
        raw = metrics.get(key)
        if isinstance(raw, bool) or type(raw) not in (int, float):
            return None
        value = float(raw)
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            return None
        total += weight * value
    return total


def _reject_strict_int(value: Any) -> int:
    """仅接受原生 int；拒绝 bool/str/float/NumPy 标量。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("must be a native int (bool/float/str rejected)")
    return value


def _reject_real(value: Any) -> Any:
    """数值输入：仅原生 int/float、有限；拒绝 bool/str/NumPy/NaN/Inf。"""
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError("must be a native int or float")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("must be a finite number") from exc
    if not math.isfinite(f):
        raise ValueError("must be finite")
    return value


StrictInt = Annotated[int, BeforeValidator(_reject_strict_int)]
RealValue = Annotated[float, BeforeValidator(_reject_real)]


def _utc_ts_validator(value: str) -> str:
    if not isinstance(value, str) or not _UTC_TS_RE.fullmatch(value):
        raise ValueError("must be an ISO8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("must be an ISO8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("must be an ISO8601 UTC timestamp with UTC offset")
    return value


UTC_ISO = Annotated[str, BeforeValidator(_utc_ts_validator)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


# ── StudyConfig ─────────────────────────────────────────────────────

class StudyConfig(StrictModel):
    schema_version: Literal["hpo-study-v1"] = HPO_STUDY_SCHEMA
    task: Literal["detect"] = "detect"
    model_family: Literal["yolov8"] = "yolov8"
    sampler: Literal["tpe", "random"] = "tpe"
    sampler_protocol: Literal["rebuild-per-trial-v1"] = "rebuild-per-trial-v1"
    search_space_version: Literal["detect-hpo-v1"] = SEARCH_SPACE_VERSION
    evaluation_mode: EvaluationMode = LEGACY_MODE
    objective: ObjectiveCode = LEGACY_OBJECTIVE
    direction: Literal["maximize"] = "maximize"
    budget: StrictInt = Field(default=10, ge=1, le=100)
    seed: StrictInt = Field(default=42, ge=0, le=2147483647)
    epochs: StrictInt = Field(default=30, ge=1, le=1000)

    @model_validator(mode="before")
    @classmethod
    def _derive_objective(cls, data: Any) -> Any:
        """评价模式显式给出而目标版本缺省时，按模式派生版本码。

        旧记录（没有 evaluation_mode）保持旧目标码不动，因此旧语义默认值确定。
        """
        if isinstance(data, dict) and "objective" not in data:
            mode = data.get("evaluation_mode")
            if mode is not None:
                derived = OBJECTIVE_BY_MODE.get(mode)
                if derived is None:
                    # 非法模式交给字段校验报错，不伪造目标码
                    return data
                return {**data, "objective": derived}
        return data

    @model_validator(mode="after")
    def _objective_matches_mode(self) -> "StudyConfig":
        if self.objective != OBJECTIVE_BY_MODE[self.evaluation_mode]:
            raise ValueError("objective must match evaluation_mode")
        return self


# ── 绑定 / 环境 ────────────────────────────────────────────────────

class SnapshotBinding(StrictModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_path: str = Field(min_length=1, max_length=2048)
    data_yaml_path: str = Field(min_length=1, max_length=2048)


class ModelBinding(StrictModel):
    model_path: str = Field(min_length=1, max_length=2048)
    model_bytes: StrictInt = Field(ge=0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnvironmentSnapshot(StrictModel):
    python_version: str = Field(min_length=1, max_length=64)
    optuna_version: str = Field(min_length=1, max_length=64)
    numpy_version: str = Field(min_length=1, max_length=64)
    ultralytics_version: str = Field(min_length=1, max_length=64)


# ── 证据与结果 ─────────────────────────────────────────────────────

_RELPATH_REJECT_SEG = re.compile(r"(^|/)\.\.?(/|$)")
_RELPATH_DRIVE = re.compile(r"^[a-zA-Z]:[/\\]|^//|^\\\\")


def _validate_artifact_relpath(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact_relpath must be a non-empty relative path")
    if len(value) > 256:
        raise ValueError("artifact_relpath too long")
    if "\\" in value:
        raise ValueError("artifact_relpath must use forward slashes")
    if value.startswith("/") or _RELPATH_DRIVE.match(value):
        raise ValueError("artifact_relpath must be relative")
    if _RELPATH_REJECT_SEG.search(value) or value.startswith("."):
        raise ValueError("artifact_relpath must be normalized without . or ..")
    if value.endswith("/") or "//" in value:
        raise ValueError("artifact_relpath must not have empty segments")
    return value


class Evidence(StrictModel):
    run_id: str = Field(min_length=1, max_length=256)
    artifact_relpath: str = Field(min_length=1, max_length=256)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_key: Literal["metrics/mAP50-95(B)"] = "metrics/mAP50-95(B)"
    epoch: StrictInt = Field(ge=1)
    # 第四轮：自描述的评价模式、目标版本与被选 epoch 的原始组成指标。旧记录
    # 没有这些字段（None）即旧单指标语义，读取与排名保持不变。
    evaluation_mode: EvaluationMode | None = None
    objective: ObjectiveCode | None = None
    metrics: dict[str, RealValue] | None = None

    @field_validator("artifact_relpath")
    @classmethod
    def _relpath(cls, value: str) -> str:
        return _validate_artifact_relpath(value)

    @field_validator("run_id")
    @classmethod
    def _run_id_len(cls, value: str) -> str:
        return _validate_text(value, "run_id")

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict) or not value:
            raise ValueError("metrics must be a non-empty object")
        checked: dict[str, float] = {}
        for key, item in value.items():
            if key not in METRIC_KEYS:
                raise ValueError(f"unsupported metric key '{key}'")
            if isinstance(item, bool) or type(item) not in (int, float):
                raise ValueError(f"metric '{key}' must be a native int or float")
            number = float(item)
            if not math.isfinite(number) or not (0.0 <= number <= 1.0):
                raise ValueError(f"metric '{key}' must be finite within 0..1")
            checked[key] = number
        return checked

    @model_validator(mode="after")
    def _mode_objective_consistent(self) -> "Evidence":
        if self.evaluation_mode is None:
            if self.objective is not None:
                raise ValueError("objective requires an evaluation_mode")
            return self
        if self.objective != OBJECTIVE_BY_MODE[self.evaluation_mode]:
            raise ValueError("evidence objective must match its evaluation_mode")
        return self


def _validate_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > 256:
        raise ValueError(f"{name} too long (max 256)")
    return value


class ResultPayload(StrictModel):
    """已登记结果的存储形态：成功 value+evidence，失败 reason_code。"""

    value: RealValue | None = Field(default=None, ge=0.0, le=1.0)
    evidence: Evidence | None = None
    reason_code: REASON_CODES | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ResultPayload":
        has_value = self.value is not None
        has_evidence = self.evidence is not None
        has_reason = self.reason_code is not None
        if has_reason:
            if has_value or has_evidence:
                raise ValueError("failure result must not carry value/evidence")
        else:
            if not (has_value and has_evidence):
                raise ValueError("success result requires value and evidence")
        return self


class ResultInput(StrictModel):
    """tell 的外部输入。跨字段语义（epoch<=config.epochs、state↔reason、
    SUCCESS 必须携带完整证据）由 service 分层校验并映射为稳定错误码。"""

    state: Literal["SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"]
    value: RealValue | None = Field(default=None, ge=0.0, le=1.0)
    evidence: Evidence | None = None
    reason_code: REASON_CODES | None = None


# ── Trial / Study 记录 ─────────────────────────────────────────────

class TrialRecord(StrictModel):
    number: StrictInt = Field(ge=0)
    trial_id: str = Field(pattern=r"^hpo_[0-9a-f]{32}_t\d{4}$")
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    state: Literal["PENDING", "SUCCESS", "FAILED", "CANCELLED", "INTERRUPTED"]
    sampled_params: dict[str, Any]
    distributions: dict[str, str]
    candidate_params: dict[str, Any]
    created_at: UTC_ISO
    finished_at: UTC_ISO | None = None
    result: ResultPayload | None = None

    @model_validator(mode="after")
    def _wellformed(self) -> "TrialRecord":
        if set(self.sampled_params) != set(self.distributions):
            raise ValueError("sampled_params and distributions keys must match")
        for container in (self.sampled_params, self.candidate_params):
            for key, val in container.items():
                if isinstance(val, bool) or type(val) not in (int, float, str):
                    raise ValueError(f"unsupported sampled value for '{key}'")
                if type(val) in (int, float) and not math.isfinite(float(val)):
                    raise ValueError(f"non-finite sampled value for '{key}'")
        return self

    @model_validator(mode="after")
    def _state_result(self) -> "TrialRecord":
        if self.state == "PENDING":
            if self.result is not None:
                raise ValueError("PENDING trial must not have a result")
        else:
            if self.result is None:
                raise ValueError("terminal trial must carry a result")
        return self


class StudyRecord(StrictModel):
    study_id: str = Field(pattern=r"^hpo_[0-9a-f]{32}$")
    config: StudyConfig
    created_at: UTC_ISO
    updated_at: UTC_ISO
    revision: StrictInt = Field(ge=0)
    snapshot_binding: SnapshotBinding
    model_binding: ModelBinding
    environment: EnvironmentSnapshot
    trials: list[TrialRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _continuous_trials(self) -> "StudyRecord":
        for idx, trial in enumerate(self.trials):
            if trial.number != idx:
                raise ValueError("trials must be contiguous from number 0")
            expected_id = f"{self.study_id}_t{idx:04d}"
            if trial.trial_id != expected_id:
                raise ValueError(f"trial_id mismatch for number {idx}")
        return self
