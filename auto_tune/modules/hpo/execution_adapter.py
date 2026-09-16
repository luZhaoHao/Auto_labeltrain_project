"""H1.2 严格训练适配器 — 把已采样 trial 转成可执行训练并验证产物。

组合现有 ``agent_engine.executor`` 的预检/写 args/命令/launch、护栏与共享
``training_finalizer``。适配器不自行 ask/tell、不重试、不扩预算；Prepared 数据
写入 ExecutionAttempt 后，launch 用已发布的 attempt 数据重建字典并复验，不使
用未持久化缓存。输出目录固定为 ``output_root/<study_id>/<trial_id>/``。

护栏 clamp 即拒绝（GuardrailRejection），不能把 clamp 后结果当作原候选；预检/
环境失败抛 ``HPO_PREFLIGHT_FAILED``，不偷偷改 CPU、不自动下载资源。
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import uuid
from pathlib import Path

import yaml

from .execution_models import MetricDiagnostics
from .metrics import objective_evidence, read_objective
from .models import HpoError, ResultInput
from .storage import _is_reparse_point, reject_link_chain
from auto_tune.modules.agent_engine.executor import (
    build_yolo_command,
    launch_training,
    write_training_config,
)
from auto_tune.modules.agent_engine.guardrails import validate_and_clamp
from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run

FIXED_PARAMS = {
    "task": "detect",
    "workers": 0,
    "resume": False,
    "deterministic": True,
    "patience": 0,
    "val": True,
    "save": True,
    "plots": False,
    "amp": False,
}

# 启动前 args.yaml 必须完整覆盖：全部固定项 + 输入绑定 + 执行配置 + 六搜索参数。
PLANNED_PARAM_KEYS = (
    list(FIXED_PARAMS)
    + ["model", "data", "epochs", "seed", "batch", "imgsz", "device"]
    + ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"]
)

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


class GuardrailRejection(Exception):
    """候选违反既有护栏；调用方须零启动登记 FAILED/invalid_params。"""


class CollectedOutcome:
    """collect 结果：成功 result 或失败 reason_code + 具体 error_code。"""

    __slots__ = ("result", "reason_code", "error_code", "error_message",
                 "diagnostics", "actual_args", "actual_args_sha256")

    def __init__(self, result=None, reason_code=None, error_code=None,
                 error_message=None, diagnostics=None,
                 actual_args=None, actual_args_sha256=None):
        self.result = result
        self.reason_code = reason_code
        self.error_code = error_code
        self.error_message = error_message
        self.diagnostics = diagnostics
        self.actual_args = actual_args
        self.actual_args_sha256 = actual_args_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1048576)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalize_path_text(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))


def _param_value_matches(key, expected, actual) -> bool:
    """逐字段类型严格比较；不允许缺字段、错误路径类型、bool 数值、NaN/Inf。"""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) \
            and actual == expected
    if key in ("model", "data"):
        if not isinstance(expected, str) or not isinstance(actual, str):
            return False
        return _normalize_path_text(expected) == _normalize_path_text(actual)
    if key in ("optimizer", "task"):
        return isinstance(expected, str) and isinstance(actual, str) \
            and expected == actual
    if key == "device":
        if isinstance(actual, str):
            return expected == actual
        if isinstance(actual, int):
            return isinstance(expected, str) and expected.isdigit() \
                and str(actual) == expected
        return False
    if key == "imgsz":
        if isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]
        if type(expected) is int and type(actual) is int:
            return expected == actual
        if type(expected) in (int, float) and type(actual) in (int, float):
            return math.isclose(float(expected), float(actual),
                                rel_tol=1e-9, abs_tol=1e-12)
        return False
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        if type(actual) not in (int, float):
            return False
        value = float(actual)
        return math.isfinite(value) and math.isclose(
            expected, value, rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


class ExecutionAdapter:
    """把单个 trial 的准备/启动/收集/收尾封装成与持久化 attempt 对齐的操作。"""

    def __init__(self, output_root, log_root):
        self._output_root = Path(output_root)
        self._log_root = Path(log_root)

    # ── 路径 ──────────────────────────────────────────────────────

    def run_dir(self, study, trial_id: str) -> Path:
        return self._output_root / study.study_id / trial_id

    def _safe_study_root(self, study_id: str) -> Path:
        reject_link_chain(self._output_root, code="HPO_CORRUPT_EXECUTION")
        study_dir = self._output_root / study_id
        reject_link_chain(study_dir, code="HPO_CORRUPT_EXECUTION")
        return study_dir

    # ── 护栏 ──────────────────────────────────────────────────────

    def check_trial(self, trial) -> dict:
        """护栏校验候选：clamp/error 或参数变化均拒绝（不可采纳 clamp 结果）。"""
        candidate = trial.candidate_params
        guard = validate_and_clamp(dict(candidate))
        if not guard.valid or guard.clamped or guard.params != dict(candidate):
            raise GuardrailRejection(
                "candidate violates guardrails (clamp not accepted)")
        return dict(candidate)

    # ── prepare ───────────────────────────────────────────────────

    def prepare(self, study, trial, config) -> dict:
        self.check_trial(trial)
        config = ExecutionConfigValidator.ensure(config)
        run_relpath = f"{study.study_id}/{trial.trial_id}"
        merged = self._effective_params(study, trial, config)
        self._preflight(study, merged, config)

        trial_dir = self.run_dir(study, trial.trial_id)
        self._safe_study_root(study.study_id)
        reject_link_chain(trial_dir, code="HPO_CORRUPT_EXECUTION")
        trial_dir.mkdir(parents=True, exist_ok=False)
        try:
            args_path = write_training_config(
                None, dict(merged), str(trial_dir))
        except OSError as exc:
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           f"cannot write args.yaml: {exc}") from exc
        args_sha256 = _sha256_file(Path(args_path))
        command = self._build_command(trial.trial_id, Path(args_path), merged)
        return {
            "trial_number": trial.number,
            "trial_id": trial.trial_id,
            "candidate_params": dict(trial.candidate_params),
            "effective_params": merged,
            "command": command,
            "run_relpath": run_relpath,
            "args_sha256": args_sha256,
        }

    def _effective_params(self, study, trial, config) -> dict:
        params = dict(FIXED_PARAMS)
        params["model"] = study.model_binding.model_path
        params["data"] = study.snapshot_binding.data_yaml_path
        params["epochs"] = study.config.epochs
        params["seed"] = study.config.seed
        params["batch"] = config.batch
        params["imgsz"] = config.imgsz
        params["device"] = config.device
        params.update(trial.candidate_params)
        return params

    def _build_command(self, trial_id: str, args_path: Path, merged: dict) -> list:
        try:
            return build_yolo_command(trial_id, str(args_path), dict(merged))
        except Exception as exc:
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           f"cannot build yolo command: {exc}") from exc

    def _preflight(self, study, merged: dict, config) -> None:
        if config.device != "cpu" and not _gpu_available(int(config.device)):
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           f"device {config.device} is not an available GPU")
        model_path = merged["model"]
        if _is_reparse_point(Path(model_path)) or not os.path.isfile(model_path):
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           "bound model is no longer an accessible local file")
        data_path = merged["data"]
        if _is_reparse_point(Path(data_path)) or not os.path.isfile(data_path):
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           "bound data.yaml is no longer an accessible local file")

    # ── launch ────────────────────────────────────────────────────

    def _safe_parse_args_bytes(self, raw: bytes) -> dict:
        """对同一份已读字节做安全 YAML 解析；根必须为对象。"""
        try:
            text = raw.decode("utf-8")
            parsed = yaml.safe_load(text) or {}
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"prepared args.yaml is not valid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "prepared args.yaml root must be an object")
        return parsed

    def _assert_planned_args(self, expected: dict, parsed: dict) -> None:
        """R2a-1：args 内容与权威 effective 的完整语义复验。

        六搜索参数、全部固定项、输入绑定与必需字段都必须存在、类型合法且与
        effective 一致；缺字段、非法数值/类型、字段矛盾（含多余键）一律拒绝。
        类型/路径规范化规则见 :func:`_param_value_matches`。
        """
        if not isinstance(expected, dict):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "planned effective_params are missing")
        missing_plan = [k for k in PLANNED_PARAM_KEYS if k not in expected]
        if missing_plan:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"planned effective_params lack {missing_plan}")
        missing = sorted(set(expected) - set(parsed))
        extra = sorted(set(parsed) - set(expected))
        if missing or extra:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"on-disk args keyset deviates from the plan "
                           f"(missing={missing}, extra={extra})")
        for key in expected:
            if not _param_value_matches(key, expected[key], parsed[key]):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               f"on-disk args value for '{key}' deviates from "
                               "the planned effective params")

    def validate_launch(self, prepared: dict) -> None:
        """发布 LAUNCH_INTENT 前的完整启动校验；只读、不创建进程、不写盘。

        顺序：链接/目录/args 文件 → 字节摘要 → 同一份字节安全解析 → 内容语义
        复验（R2a-1）→ 冻结命令尾全等（R6，不解析当前 YOLO）→ 当前 executable
        可用性（新启动环境）。任一步失败抛稳定 HpoError；调用方必须保留原错误码、
        阻断取样与启动。
        """
        run_relpath = prepared.get("run_relpath")
        command = list(prepared.get("command") or [])
        effective = prepared.get("effective_params")
        if not run_relpath or not command:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "prepared launch data is incomplete")
        run_dir = self._output_root / run_relpath
        reject_link_chain(run_dir, code="HPO_CORRUPT_EXECUTION")
        if not run_dir.is_dir() or _is_reparse_point(run_dir):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "run directory is missing or a reparse point")
        args_path = run_dir / "args.yaml"
        reject_link_chain(args_path, code="HPO_CORRUPT_EXECUTION")
        if not args_path.is_file() or _is_reparse_point(args_path):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "run dir has no auditable args.yaml")
        try:
            raw = args_path.read_bytes()
        except OSError as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"cannot read prepared args.yaml: {exc}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        expected_sha = prepared.get("args_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 \
                or digest != expected_sha:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "prepared args hash mismatch; refusing launch")
        # R2a-1：对同一次读取的字节做安全解析与内容语义复验。
        parsed = self._safe_parse_args_bytes(raw)
        self._assert_planned_args(effective, parsed)
        trial_id = prepared.get("trial_id")
        if not trial_id:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "prepared trial_id is missing")
        # R6：冻结命令校验以已冻结 executable（command[0]）为锚点重建，不解析
        # 当前 YOLO；命令参数仍必须与冻结 effective 一致（不取消命令完整性）。
        rebuilt_frozen = build_yolo_command(
            trial_id, str(args_path), dict(effective), executable=command[0])
        if command != rebuilt_frozen:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "command deviates from the frozen expected command")
        # 新启动环境/executable 可用性：解析失败或与冻结命令不一致时零启动。
        try:
            current = build_yolo_command(trial_id, str(args_path),
                                         dict(effective))
        except Exception as exc:
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           f"cannot resolve yolo executable: {exc}") from exc
        if command != current:
            raise HpoError("HPO_PREFLIGHT_FAILED",
                           "current yolo executable differs from the frozen "
                           "launch command")

    def launch(self, prepared: dict):
        """使用已发布的 attempt 数据启动进程；不重算命令、不无审计重建 args。

        R2a-1/R2a-2：完整启动校验先于进程创建（:meth:`validate_launch`），校验
        失败抛稳定 HpoError 且已知没有启动；缺失/不匹配 args 一律拒绝零 launch，
        禁止静默重建 args 补救一次非法启动。
        """
        self.validate_launch(prepared)
        run_dir = self._output_root / prepared["run_relpath"]
        args_path = run_dir / "args.yaml"
        return launch_training(
            prepared["trial_id"], str(args_path),
            dict(prepared["effective_params"]),
            command=list(prepared["command"]),
        )

    # ── collect ───────────────────────────────────────────────────

    def collect(self, study, attempt) -> CollectedOutcome:
        run_dir = self._output_root / attempt.run_relpath
        try:
            reject_link_chain(run_dir, code="HPO_CORRUPT_EXECUTION")
            actual_args, actual_sha = self._read_actual_args(run_dir)
        except HpoError as exc:
            return CollectedOutcome(reason_code="invalid_params",
                                    error_code=exc.code,
                                    error_message=exc.message)
        outcome = CollectedOutcome(actual_args=actual_args,
                                   actual_args_sha256=actual_sha)
        if not self._params_match(attempt, actual_args):
            outcome.reason_code = "invalid_params"
            outcome.error_message = "actual args drifted from planned candidate"
            return outcome
        try:
            obj = read_objective(run_dir, artifact_root=self._output_root,
                                 run_id=attempt.run_id,
                                 epochs=study.config.epochs,
                                 evaluation_mode=study.config.evaluation_mode)
        except HpoError as exc:
            outcome.reason_code = "training_failed"
            outcome.error_code = exc.code
            outcome.error_message = exc.message
            return outcome
        outcome.result = ResultInput(state="SUCCESS", value=obj.value,
                                     evidence=objective_evidence(obj, attempt.run_id))
        outcome.diagnostics = obj.diagnostics
        return outcome

    def _read_actual_args(self, run_dir: Path):
        target = run_dir / "args.yaml"
        reject_link_chain(target, code="HPO_CORRUPT_EXECUTION")
        if not target.is_file() or _is_reparse_point(target):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "run dir has no readable args.yaml")
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"cannot read actual args.yaml: {exc}") from exc
        actual_sha = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
            parsed = yaml.safe_load(text) or {}
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"actual args.yaml is not valid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "actual args.yaml root must be an object")
        return parsed, actual_sha

    def _params_match(self, attempt, actual: dict) -> bool:
        """六搜索参数与所有固定条件必须存在、类型合法并与计划相符（R4）。

        缺字段、错误路径类型、bool 充当数值、NaN/Inf 一律判不匹配，不产生成功指标。
        """
        expected = attempt.effective_params
        for key in PLANNED_PARAM_KEYS:
            if key not in expected or key not in actual:
                return False
            if not _param_value_matches(key, expected[key], actual[key]):
                return False
        return True

    # ── finalize ──────────────────────────────────────────────────

    def finalize(self, study, attempt) -> dict:
        run_dir = self._output_root / attempt.run_relpath
        run_name = run_dir.name
        state = attempt.result.state if attempt.result is not None else "FAILED"
        status_map = {"SUCCESS": "completed", "FAILED": "failed",
                      "CANCELLED": "cancelled", "INTERRUPTED": "interrupted"}
        training_status = status_map.get(state, "failed")
        training_error = None
        if training_status != "completed":
            reason = (attempt.result.reason_code
                      if attempt.result is not None else None)
            training_error = {
                "error_type": reason or "failed",
                "error_code": attempt.error_code,
                "message": attempt.error_message,
            }
        tuning_context = {
            "strategy": "hpo",
            "algorithm": study.config.sampler,
            "study_id": study.study_id,
            "trial_number": attempt.trial_number,
            "request_id": attempt.request_id,
        }
        config = {"project": {"name": f"HPO {study.study_id}"},
                  "train_analyzer": {}}
        return finalize_training_run(
            str(run_dir),
            run_name,
            "tuning",
            config,
            log_dir=str(self._log_root),
            training_status=training_status,
            session_id=study.study_id,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            training_error=training_error,
            tuning_context=tuning_context,
            runtime_run_id=attempt.run_id,
            dataset_id=study.snapshot_binding.snapshot_id,
        )

    # ── OOM 探测 ──────────────────────────────────────────────────

    def detect_oom(self, run_dir: Path) -> bool:
        log_path = Path(run_dir) / "yolo_train.log"
        if not log_path.is_file():
            return False
        try:
            size = log_path.stat().st_size
            with open(log_path, "rb") as fh:
                if size > 1048576:
                    fh.seek(size - 1048576)
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return False
        return any(marker in tail for marker in OOM_MARKERS)


def _gpu_available(index: int) -> bool:
    """返回单个 GPU 索引当前是否可训练（预检用，不偷偷回退 CPU）。"""
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        count = torch.cuda.device_count()
    except Exception:
        return False
    return 0 <= int(index) < count


class ExecutionConfigValidator:
    """占位命名空间：把已实例化配置在入口 dump→validate（R4 同类防御）。"""

    @staticmethod
    def ensure(config):
        from .execution_models import ExecutionConfig
        if isinstance(config, ExecutionConfig):
            data = config.model_dump(mode="python", warnings=False)
            return ExecutionConfig.model_validate(data)
        return ExecutionConfig.model_validate(config)
