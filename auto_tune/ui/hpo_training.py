"""H1.3 追加返修：最佳参数的正式训练、受控产物与关联结果投影。

本模块只做三件事：

1. **受控产物解析**：Trial 的 ``best.pt``/``last.pt`` 由冻结的 ``output_root``
   与权威 attempt 的 ``run_relpath`` 重建；正式训练的权重由受控 train 目录重建。
   请求只能给身份（study_id/trial_id/train_name/白名单产物名），不能给路径。
2. **严格正式训练配置**：``FormalTrainingConfig`` 只允许 epochs/batch/imgsz/device，
   四个字段必填且范围与 HPO 创建一致；六个搜索参数永不接受客户端输入。
3. **事实投影**：正式训练的来源 metadata 写入受控 train 目录，关联结果按 study
   从已持久化事实重新投影，服务重启后仍可查询，且从不写回 HPO study/execution。

不实现第二种训练执行器：真正的提交复用普通训练的 ManualRunController、收尾回调和
共享原子门禁（见 :func:`submit_formal_training`）。
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from auto_tune.modules.hpo import HpoError, rank_trials
from auto_tune.modules.hpo.models import StrictInt
from auto_tune.modules.hpo.execution_adapter import FIXED_PARAMS
from auto_tune.modules.hpo.search_space import validate_candidate
from auto_tune.modules.local_index import LocalIndexError

SEARCH_KEYS = ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")

# 正式训练专用覆盖。搜索阶段的固定参数（``FIXED_PARAMS``）关闭绘图，因为每个短
# 试验都出图既慢又浪费磁盘；用最佳参数进行的**最终正式训练**是用户交付物，必须
# 生成 YOLO 标准静态结果图（results.png / confusion_matrix*.png / 各类曲线）。
# 只覆盖这一个键：既不建立第二套固定参数配置，也不改动搜索阶段的共享常量。
FORMAL_TRAINING_OVERRIDES = {"plots": True}

# 本轮只允许这两个产物名；客户端永远不能提交任意文件名。
ARTIFACT_NAMES = ("best.pt", "last.pt")

STUDY_ID_RE = re.compile(r"^hpo_[0-9a-f]{32}$")
TRIAL_ID_RE = re.compile(r"^hpo_[0-9a-f]{32}_t\d{4}$")
TRAIN_NAME_RE = re.compile(r"^train\d+$")
DEVICE_RE = re.compile(r"^(?:cpu|0|[1-9]|[1-5][0-9]|6[0-3])$")

FORMAL_SOURCE_FILENAME = "hpo_source.json"
FORMAL_SOURCE_MODE = "formal"
VERIFY_SOURCE_MODE = "verification"

# 关联投影最多扫描多少个 train 目录（读取有界，绝不遍历整个磁盘）。
_MAX_TRAIN_DIRS = 500

# S1.5 统一运行身份 ``<kind>:<uuid4>``。它**只**用于查询本地实验索引；
# JSON 历史身份 ``manual:<train_name>`` 永远不匹配，因此绝不可能被当作索引主键。
RUNTIME_RUN_ID_RE = re.compile(
    r"^(?:manual|tuning):"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# “查看结果”必须指向权威实验索引里的同一条运行。身份无法唯一确定时给出稳定原因
# 码，前端据此禁用入口并说明原因，而不是放一个点了必然 404 的按钮。
EXPERIMENT_NOT_INDEXED = "EXPERIMENT_NOT_INDEXED"
EXPERIMENT_AMBIGUOUS = "EXPERIMENT_AMBIGUOUS"
LOCAL_INDEX_UNAVAILABLE = "LOCAL_INDEX_UNAVAILABLE"


class HpoArtifactError(Exception):
    """受控产物解析失败；code 为稳定错误码，message 只在服务端使用。"""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


# ── 严格正式训练配置 ────────────────────────────────────────────────


class FormalTrainingConfig(BaseModel):
    """正式训练请求体：四项条件都必须携带，但只有 ``epochs`` 会被采纳。

    batch/imgsz/device 必须与当前研究的权威执行条件逐项相等（见
    :func:`resolve_hpo_formal_training`），否则零启动拒绝；保留这四个字段是为了
    兼容既有请求形态与字段级错误投影。数值使用 HPO 模块的严格类型（拒绝
    bool/字符串/浮点替代整数），设备沿用 ``cpu`` 或单个 GPU 编号 0..63。
    """

    model_config = ConfigDict(extra="forbid")

    epochs: StrictInt = Field(ge=1, le=1000)
    batch: StrictInt = Field(ge=1, le=256)
    imgsz: StrictInt = Field(ge=32, le=2048)
    device: str

    @field_validator("imgsz")
    @classmethod
    def _imgsz_multiple(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError("device must be a string")
        if not DEVICE_RE.fullmatch(value):
            raise ValueError("device must be 'cpu' or a single GPU index 0..63")
        return value


class TrainBestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(min_length=1, max_length=64)
    training_config: FormalTrainingConfig


@dataclass(frozen=True)
class VerifiedFormalTraining:
    """服务端从权威记录重建的正式训练事实。"""

    study_id: str
    trial_id: str
    trial_number: int
    value: float | None
    epoch: int | None
    search: dict           # 六个搜索参数（只读）
    original: dict         # 原研究的 epochs/batch/imgsz/device
    training_config: dict  # 实际生效的四项条件
    differences: dict      # 与原研究不同的条件
    effective: dict        # 完整训练参数（FIXED_PARAMS + 绑定 + 搜索参数 + 四项条件）

    def source_metadata(self, *, runtime_run_id: str | None = None,
                        train_name: str | None = None) -> dict:
        """写入 ``hpo_source.json`` 的来源事实（不含任何绝对路径）。

        ``runtime_run_id`` 是本次正式运行的统一运行身份（``manual:<uuid4>``，
        与实际执行/监控/SQLite 详情一致），``train_name`` 是受控 train 目录名。
        两者在**启动前**写盘，因此活跃、页面重载与终态投影恒为同一身份；它们与
        统一历史的 JSON run_id（``manual:<train_name>``）是两个不同的概念。
        """
        return {
            "mode": FORMAL_SOURCE_MODE,
            "study_id": self.study_id,
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "value": self.value,
            "epoch": self.epoch,
            "search_params": dict(self.search),
            "original_conditions": dict(self.original),
            "training_config": dict(self.training_config),
            "differences": dict(self.differences),
            "seed": self.effective.get("seed"),
            "runtime_run_id": runtime_run_id,
            "train_name": train_name,
        }


def parse_formal_request(payload: Any) -> TrainBestRequest:
    """严格解析 train-best 请求体；失败抛 ``HpoError(HPO_INVALID_CONFIG)``。"""
    try:
        return TrainBestRequest.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HpoError("HPO_INVALID_CONFIG", "正式训练请求体不合法") from exc


def resolve_hpo_formal_training(service, runner, manager, study_id: str,
                                trial_id: str,
                                training_config: FormalTrainingConfig,
                                ) -> VerifiedFormalTraining:
    """重验证 rank-1 SUCCESS、绑定、执行记录与已完成状态后重建完整参数。

    只采纳请求里的 ``epochs``；batch/imgsz/device 必须与研究的权威执行条件相等，
    失配即 ``HPO_INVALID_CONFIG`` 零启动。除 epochs 外的固定参数、初始权重、快照
    与 seed 一律取自权威记录。``warmup_epochs`` 等候选条件在**新的** epochs 下
    重新校验，不合法时安全拒绝（``HPO_SOURCE_INVALID``），绝不静默裁剪最佳参数。
    """
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        raise HpoError("HPO_SOURCE_INVALID", "研究 ID 格式不合法")
    if not isinstance(trial_id, str) or not TRIAL_ID_RE.fullmatch(trial_id):
        raise HpoError("HPO_SOURCE_INVALID", "来源试验 ID 格式不合法")

    study = service.load_study(study_id)  # HPO_NOT_FOUND for an unknown study

    ranked = rank_trials(study)
    if not ranked:
        raise HpoError("HPO_NO_SUCCESS", "该研究没有可用的成功试验结果")
    top = ranked[0]
    if top.trial_id != trial_id:
        raise HpoError("HPO_SOURCE_INVALID", "来源不是当前排名第一的成功试验")

    # 绑定/环境漂移在重建参数之前就拒绝。
    service.validate_binding(study_id)

    try:
        execution = runner.status(study_id)
    except HpoError as exc:
        if exc.code == "HPO_NOT_FOUND":
            raise HpoError("HPO_EXECUTION_CONFLICT",
                           "该研究尚未绑定执行配置") from exc
        raise
    execution_status = getattr(execution, "status", None)
    if execution_status != "COMPLETED":
        raise HpoError("HPO_SOURCE_INVALID",
                       "只有已完成的搜索研究才能用于正式训练")

    # 有活动的 HPO 控制器时不允许启动正式训练（避免与搜索争用同一批产物）。
    if manager is not None:
        controller = manager.get(study_id)
        if controller is not None and getattr(controller, "run_kind", None) == "hpo" \
                and controller.is_active():
            raise HpoError("HPO_SOURCE_INVALID", "该研究仍有进行中的搜索，无法启动正式训练")

    if not isinstance(training_config, FormalTrainingConfig):
        raise HpoError("HPO_INVALID_CONFIG", "正式训练配置不合法")

    config = training_config
    if any(isinstance(getattr(config, name), bool) or type(getattr(config, name)) is not int
           for name in ("epochs", "batch", "imgsz")):
        raise HpoError("HPO_INVALID_CONFIG", "正式训练条件必须是原生整数")
    if config.imgsz % 32 != 0 or not (32 <= config.imgsz <= 2048):
        raise HpoError("HPO_INVALID_CONFIG", "图像尺寸必须是 32 的整数倍且在 32..2048")
    if not (1 <= config.epochs <= 1000) or not (1 <= config.batch <= 256):
        raise HpoError("HPO_INVALID_CONFIG", "训练轮数或 Batch 超出允许范围")
    if not DEVICE_RE.fullmatch(config.device):
        raise HpoError("HPO_INVALID_CONFIG", "计算设备必须是 cpu 或 0..63 的单个 GPU 编号")

    candidate = {key: top.candidate_params.get(key) for key in SEARCH_KEYS}
    # 候选条件在正式 epochs 下重新校验；不合法 → 安全拒绝，不裁剪。
    try:
        validate_candidate(candidate, epochs=config.epochs)
    except HpoError as exc:
        raise HpoError(
            "HPO_SOURCE_INVALID",
            "最佳试验的搜索参数在新的训练轮数下不合法，未启动训练") from exc

    original = {
        "epochs": study.config.epochs,
        "batch": execution.config.batch,
        "imgsz": execution.config.imgsz,
        "device": execution.config.device,
    }
    # 除 epochs 外全部冻结：batch/imgsz/device 必须与研究的权威执行条件逐项相等，
    # 任何失配都是配置冲突（零启动），绝不静默覆盖客户端传值。
    for name in ("batch", "imgsz", "device"):
        if getattr(config, name) != original[name]:
            raise HpoError(
                "HPO_INVALID_CONFIG",
                f"正式训练的 {name} 必须与当前研究的执行条件一致")
    actual = {"epochs": config.epochs, "batch": original["batch"],
              "imgsz": original["imgsz"], "device": original["device"]}
    differences = {key: {"original": original[key], "requested": actual[key]}
                   for key in original if original[key] != actual[key]}

    effective = dict(FIXED_PARAMS)
    effective["model"] = study.model_binding.model_path
    effective["data"] = study.snapshot_binding.data_yaml_path
    effective["seed"] = study.config.seed
    effective.update(actual)
    effective.update(candidate)
    # 最后施加正式训练覆盖，保证 plots=True 是这一次重建的唯一权威取值
    # （四项条件与六个搜索参数都不含该键，覆盖不会被后续 update 顶掉）。
    effective.update(FORMAL_TRAINING_OVERRIDES)

    return VerifiedFormalTraining(
        study_id=study_id,
        trial_id=top.trial_id,
        trial_number=top.number,
        value=top.result.value if top.result is not None else None,
        epoch=(top.result.evidence.epoch if top.result is not None
               and top.result.evidence is not None else None),
        search=candidate,
        original=original,
        training_config=actual,
        differences=differences,
        effective=effective,
    )


# ── 受控路径解析 ───────────────────────────────────────────────────


def _is_link_like(path: Path) -> bool:
    """Reuse the audited HPO link/reparse check (symlinks and NT junctions)."""
    from auto_tune.modules.hpo.storage import _is_reparse_point

    try:
        if os.path.islink(path):
            return True
    except OSError:
        pass
    try:
        return bool(_is_reparse_point(Path(path)))
    except OSError:
        return False


def _reject_link_chain(path: Path) -> None:
    from auto_tune.modules.hpo.storage import reject_link_chain

    try:
        reject_link_chain(Path(path), code="HPO_ARTIFACT_INVALID")
    except HpoError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_INVALID",
                               "产物路径包含链接或重解析点") from exc


def _inside(root: Path, candidate: Path) -> bool:
    try:
        root_n = os.path.normcase(os.path.abspath(str(root)))
        cand_n = os.path.normcase(os.path.abspath(str(candidate)))
    except (OSError, ValueError):
        return False
    try:
        return os.path.commonpath([root_n, cand_n]) == root_n
    except ValueError:
        return False


def validate_artifact_name(name: str) -> str:
    if not isinstance(name, str) or name not in ARTIFACT_NAMES:
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "不支持的产物名")
    return name


def _require_file(path: Path, root: Path) -> Path:
    if not _inside(root, path):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "产物路径越出受控目录")
    _reject_link_chain(path)
    if not path.is_file():
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "产物文件不存在")
    return path


# ── 受控“打开调优结果文件夹” ───────────────────────────────────────


def _open_directory(path: str) -> None:
    """用系统文件管理器打开一个**服务端解析**的目录。

    独立小函数：自动化测试通过 monkeypatch 注入替换，绝不真的启动 Explorer。
    非 Windows 平台直接拒绝；失败向上抛，由调用方映射成稳定错误码。
    """
    if os.name != "nt":
        raise NotImplementedError("opening a folder is only supported on Windows")
    os.startfile(path)  # noqa: S606 — Windows-only, path is server-resolved


def resolve_best_run_dir(service, runner, study_id: str) -> tuple[str, Path]:
    """返回 ``(trial_id, 权威运行目录)``：当前 rank-1 SUCCESS 试验的运行目录。

    目录由冻结的 ``output_root`` 与权威 attempt 的 ``run_relpath`` 重建，并复用
    既有父链链接/重解析点检查；客户端永远不能提交目录或文件路径。
    """
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "研究 ID 格式不合法")

    study = service.load_study(study_id)  # HPO_NOT_FOUND for an unknown study
    ranked = rank_trials(study)
    if not ranked:
        raise HpoError("HPO_NO_SUCCESS", "该研究没有可用的成功试验结果")
    top = ranked[0]

    record = runner.status(study_id)  # HPO_NOT_FOUND → not prepared yet
    if getattr(record, "status", None) != "COMPLETED":
        raise HpoError("HPO_SOURCE_INVALID", "只有已完成的搜索研究才有结果目录")

    attempts = getattr(record, "attempts", None) or []
    matching = [a for a in attempts
                if getattr(a, "trial_number", None) == top.number]
    if not matching:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "该试验尚无执行记录")
    attempt = matching[-1]
    expected_rel = f"{study_id}/{top.trial_id}"
    if getattr(attempt, "trial_id", None) != top.trial_id \
            or getattr(attempt, "run_relpath", None) != expected_rel:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "执行记录身份不一致")

    roots = getattr(record, "roots", None)
    output_root_text = getattr(roots, "output_root", None)
    if not isinstance(output_root_text, str) or not output_root_text:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "输出根不可用")
    output_root = Path(output_root_text)
    run_dir = output_root / expected_rel
    if not _inside(output_root, run_dir):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "结果目录越出受控根")
    _reject_link_chain(run_dir)
    if not run_dir.is_dir() or _is_link_like(run_dir):
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "结果目录不存在")
    return top.trial_id, run_dir


def open_best_run_folder(service, runner, study_id: str, *, opener=None) -> dict:
    """在本地文件管理器中打开当前 rank-1 试验的结果目录。

    任何身份/状态/路径问题都在调用 opener 之前失败，因此这些情形都是**零系统调用**。
    返回值不含任何绝对路径。
    """
    trial_id, run_dir = resolve_best_run_dir(service, runner, study_id)
    open_fn = opener or _open_directory
    try:
        open_fn(str(run_dir))
    except NotImplementedError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_OPEN_UNSUPPORTED",
                               "当前平台不支持打开文件夹") from exc
    except OSError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_OPEN_FAILED",
                               "无法打开结果文件夹") from exc
    return {"study_id": study_id, "trial_id": trial_id, "opened": True}


def resolve_trial_artifact(service, runner, study_id: str, trial_id: str,
                           name: str) -> Path:
    """从冻结 output_root 与权威 attempt 的 run_relpath 重建 Trial 产物路径。"""
    validate_artifact_name(name)
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "研究 ID 格式不合法")
    if not isinstance(trial_id, str) or not TRIAL_ID_RE.fullmatch(trial_id):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "试验 ID 格式不合法")
    if not trial_id.startswith(study_id + "_t"):
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "试验不属于该研究")

    service.load_study(study_id)  # 研究身份与记录完整性
    try:
        number = int(trial_id.rsplit("_t", 1)[1])
    except (IndexError, ValueError) as exc:
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "试验 ID 格式不合法") from exc

    try:
        record = runner.status(study_id)
    except HpoError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", exc.message) from exc

    attempts = getattr(record, "attempts", None)
    if not attempts:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "该试验尚无执行记录")
    matching = [a for a in attempts if getattr(a, "trial_number", None) == number]
    if not matching:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "该试验尚无执行记录")
    attempt = matching[-1]
    if getattr(attempt, "trial_id", None) != trial_id:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "试验身份不一致")

    expected_rel = f"{study_id}/{trial_id}"
    if getattr(attempt, "run_relpath", None) != expected_rel:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "执行记录身份不一致")

    roots = getattr(record, "roots", None)
    output_root_text = getattr(roots, "output_root", None)
    if not isinstance(output_root_text, str) or not output_root_text:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "输出根不可用")
    output_root = Path(output_root_text)
    return _require_file(output_root / expected_rel / "weights" / name, output_root)


def trial_artifacts_available(service, runner, study_id: str, trial_id: str) -> dict:
    """只读探测两个白名单产物是否可下载；任何失败都降级为不可用。"""
    result = {"best_pt_available": False, "last_pt_available": False}
    for name, key in (("best.pt", "best_pt_available"),
                      ("last.pt", "last_pt_available")):
        try:
            resolve_trial_artifact(service, runner, study_id, trial_id, name)
            result[key] = True
        except (HpoArtifactError, HpoError, OSError):
            result[key] = False
    return result


# ── 正式训练来源 metadata / 关联投影 ────────────────────────────────


def _load_source_file(path: Path) -> tuple[dict | None, bool]:
    """Return ``(data_or_None, present)``.

    ``present`` is True when a source file exists but cannot be trusted/parsed,
    so the caller can distinguish "no formal history at all" from "a record is
    there but unreadable" and warn instead of silently reporting an empty list.
    """
    try:
        if not path.is_file():
            return None, False
    except OSError:
        return None, False
    if _is_link_like(path):
        return None, True
    try:
        raw = path.read_bytes()
    except OSError:
        return None, True
    if len(raw) > 65536:
        return None, True
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, True
    if not isinstance(data, dict):
        return None, True
    return data, True


def read_formal_source(train_dir: Path) -> dict | None:
    """读取受控 train 目录的来源 metadata；不合法或缺失时返回 None。"""
    data, _present = _load_source_file(Path(train_dir) / FORMAL_SOURCE_FILENAME)
    return data


def resolve_controlled_formal_run_dir(detect_dir, study_id: str,
                                      train_name: str) -> Path:
    """校验并返回当前研究关联的正式训练受控目录 ``detect/trainN``。

    “正式产物下载”与“打开结果文件夹”共用这一条受控目录校验：编号格式、严格位于
    受控 detect 根之下、父链无符号链接/重解析点、来源 metadata 为 formal 模式且
    绑定同一研究、目录与 metadata 记录的训练编号一致。客户端永远只能提交身份
    （study_id/train_name），物理路径一律由服务端重建。
    """
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "研究 ID 格式不合法")
    if not isinstance(train_name, str) or not TRAIN_NAME_RE.fullmatch(train_name):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "训练编号格式不合法")

    detect_root = Path(os.path.abspath(str(detect_dir)))
    train_dir = detect_root / train_name
    if not _inside(detect_root, train_dir):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "训练目录越出受控根")
    _reject_link_chain(train_dir)

    source = read_formal_source(train_dir)
    if not source:
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "缺少来源 metadata")
    if source.get("mode") != FORMAL_SOURCE_MODE:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "来源不是正式训练")
    if source.get("study_id") != study_id:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH", "来源研究不一致")
    # 新记录才写 train_name；旧记录缺失时按目录名（它就是启动时选定的编号）。
    recorded_train_name = source.get("train_name")
    if recorded_train_name is not None and recorded_train_name != train_name:
        raise HpoArtifactError("HPO_ARTIFACT_IDENTITY_MISMATCH",
                               "来源训练编号与目录不一致")
    if not train_dir.is_dir() or _is_link_like(train_dir):
        raise HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE", "训练目录不存在")
    return train_dir


def resolve_formal_artifact(detect_dir, study_id: str, train_name: str,
                            name: str) -> Path:
    """从已验证的来源 metadata 与受控 train 目录解析正式训练产物。"""
    validate_artifact_name(name)
    train_dir = resolve_controlled_formal_run_dir(detect_dir, study_id, train_name)
    detect_root = Path(os.path.abspath(str(detect_dir)))
    return _require_file(train_dir / "weights" / name, detect_root)


def resolve_formal_run_status(*, log_dir, detect_dir, study_id: str,
                              train_name: str, manager=None) -> str | None:
    """权威正式训练投影中该 ``train_name`` 的当前状态；无法唯一确定时返回 ``None``。

    复用 ``/formal-runs`` **同一个**扫描函数与同样的来源过滤（有界、只读），因此
    列表显示与打开入口永远不会出现状态漂移。``None`` 表示“投影里没有这一行”或
    “同一编号出现多行”，调用方一律按未完成处理，绝不猜测、绝不借用其他行状态。
    """
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        return None
    if not isinstance(train_name, str) or not TRAIN_NAME_RE.fullmatch(train_name):
        return None
    try:
        rows, _warnings, _truncated = _scan_formal_runs(
            log_dir=log_dir, detect_dir=detect_dir, study_id=study_id,
            manager=manager, index=None)
    except Exception:
        return None
    matched = [row for row in rows
               if isinstance(row, dict) and row.get("train_name") == train_name]
    if len(matched) != 1:
        return None
    status = matched[0].get("status")
    return status if isinstance(status, str) and status else None


def open_formal_run_folder(detect_dir, study_id: str, train_name: str, *,
                           status_resolver, opener=None) -> dict:
    """在本地文件管理器中打开**当前研究关联的正式训练**结果目录 ``detect/trainN``。

    与“打开调优结果文件夹”语义不同：后者打开 HPO 搜索阶段 rank-1 试验目录。身份
    只能来自路径参数，目录由服务端重建并逐项校验；任何身份/来源/路径问题都在调用
    opener 之前失败（零系统调用）。返回值不含任何绝对路径。

    ``status_resolver`` 是**必填**的权威状态来源（``train_name -> status``，由
    :func:`resolve_formal_run_status` 提供）：完成态门控由服务端独立执行，只有
    当前行的权威状态为 ``completed`` 才允许打开目录。前端的完成态限制只是体验层，
    绕过前端直接调用接口同样会被拒绝。门控在受控目录身份/来源校验**之后**、
    opener **之前**执行，因此非完成态与身份非法一样是零系统调用。
    """
    train_dir = resolve_controlled_formal_run_dir(detect_dir, study_id, train_name)
    if status_resolver(train_name) != "completed":
        raise HpoArtifactError("HPO_ARTIFACT_NOT_COMPLETED",
                               "该正式训练尚未成功完成")
    open_fn = opener or _open_directory
    try:
        open_fn(str(train_dir))
    except NotImplementedError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_OPEN_UNSUPPORTED",
                               "当前平台不支持打开文件夹") from exc
    except OSError as exc:
        raise HpoArtifactError("HPO_ARTIFACT_OPEN_FAILED",
                               "无法打开结果文件夹") from exc
    return {"study_id": study_id, "train_name": train_name, "opened": True}


def _bounded_dir_entries(root: Path) -> tuple[list[Path], bool]:
    """Return ``(newest_first_train_dirs, truncated)`` with a hard bound."""
    try:
        entries = [Path(entry) for entry in os.scandir(root)]
    except OSError:
        return [], False
    train_dirs = [p for p in entries if TRAIN_NAME_RE.fullmatch(p.name)
                  and p.is_dir() and not _is_link_like(p)]
    train_dirs.sort(key=lambda p: int(p.name[5:]), reverse=True)
    limited = train_dirs[:_MAX_TRAIN_DIRS]
    return limited, len(train_dirs) > len(limited)


def _last_results_row(train_dir: Path) -> dict:
    """有界读取 ``results.csv`` 末行；返回映射后的四个指标。"""
    path = Path(train_dir) / "results.csv"
    if _is_link_like(path):
        return {}
    try:
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            return {}
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip()
            last = ""
            for line in fh:
                if line.strip():
                    last = line.strip()
    except OSError:
        return {}
    if not header or not last:
        return {}
    columns = [c.strip() for c in header.split(",")]
    values = [v.strip() for v in last.split(",")]
    if len(values) != len(columns):
        return {}
    row = dict(zip(columns, values))
    mapping = {"metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50_95",
               "metrics/precision(B)": "precision", "metrics/recall(B)": "recall"}
    out = {}
    for source_key, target in mapping.items():
        raw = row.get(source_key)
        if raw in (None, ""):
            continue
        try:
            out[target] = float(raw)
        except ValueError:
            continue
    return out


def _live_manual_status(manager, train_name: str):
    """Return the manual controller for one train_name, preferring an active one.

    同一个 ``train_name`` 可能同时存在一条已收尾但尚未 ``retain()`` 的控制器和一条
    真正在跑的控制器（例如上一批运行刚结束就用同名目录重跑）。监控身份必须取自
    **活动**的那条，否则会指向错误的运行；两条都不活动时才返回任一条用于展示状态。
    """
    if manager is None:
        return None
    try:
        controllers = manager.snapshot()
    except Exception:
        return None
    inactive = None
    for controller in controllers:
        if getattr(controller, "run_kind", None) != "manual":
            continue
        if getattr(controller, "train_name", None) != train_name:
            continue
        if controller.is_active():
            return controller
        if inactive is None:
            inactive = controller
    return inactive


def _history_index(log_dir) -> dict:
    """run_name → 统一历史记录（JSON 事实源）；不可读时返回空映射。"""
    from auto_tune.modules.train_analyzer.experiment_history import (
        ExperimentHistoryStore,
    )

    path = os.path.join(str(log_dir), "experiment_history.json")
    if not os.path.isfile(path):
        return {}
    try:
        records = ExperimentHistoryStore(path).list_experiments(include_legacy=False)
    except Exception:
        return {}
    index = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        run_name = record.get("run_name")
        if isinstance(run_name, str) and run_name:
            index[run_name] = record
    return index


def _norm_path(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(value)))
    except (OSError, ValueError):
        return None


def _run_dir_artifacts(exp: dict) -> list[str]:
    paths: list[str] = []
    for artifact in exp.get("artifacts") or ():
        if not isinstance(artifact, dict) or artifact.get("kind") != "run_dir":
            continue
        raw = artifact.get("path")
        if isinstance(raw, str) and raw:
            paths.append(raw)
    return paths


def _matches_controlled_train_dir(exp: dict, train_dir: Path, train_name: str) -> bool:
    """实验身份必须同时匹配 run_name **与**这个受控 ``detect/trainN`` 目录。

    绝对路径按规范化结果精确比较；历史记录里唯一允许的相对形式是 ``detect/trainN``
    或裸 ``trainN``。任何其他目录一律不算匹配，因此同名运行不会跨目录串用。
    """
    if exp.get("run_name") != train_name:
        return False
    target = _norm_path(str(train_dir))
    if target is None:
        return False
    for raw in _run_dir_artifacts(exp):
        candidate = _norm_path(raw)
        if candidate is not None and candidate == target:
            return True
        try:
            is_absolute = os.path.isabs(raw)
        except (OSError, ValueError):
            continue
        if not is_absolute:
            try:
                relative = os.path.normcase(os.path.normpath(raw))
                bare = os.path.normcase(train_name)
                under_detect = os.path.normcase(os.path.join("detect", train_name))
            except (OSError, ValueError):
                continue
            if relative in (bare, under_detect):
                return True
    return False


def _declared_runtime_identity(value: Any) -> Any:
    """返回 metadata *声明* 的 runtime 身份；没有声明时返回 ``None``。

    ``None`` 与空/纯空白字符串表示“旧记录没有记录运行身份”，这是**唯一**允许按
    run_name 回退的情形。其余任何值（包括非字符串和格式非法的字符串）都是一次
    声明：必须据其判定，绝不静默当作缺失。
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _strict_runtime_identity(value: Any) -> str | None:
    """返回可直接用于运行查询/监控的 runtime 身份；否则返回 ``None``。

    S1.5 运行身份形如 ``<kind>:<uuid4>``。JSON 历史 ID（``manual:train1``）或其他
    非法值都不是可用的运行身份：它们既不能交给 ``/api/runs/{run_id}/stream``，
    也不能冒充实验索引主键。这里只做格式判定，绝不补造或改写身份。
    """
    if isinstance(value, str) and RUNTIME_RUN_ID_RE.fullmatch(value):
        return value
    return None


def resolve_formal_experiment_identity(*, index, train_dir, train_name: str,
                                       runtime_run_id: str | None
                                       ) -> tuple[str | None, str | None]:
    """解析一次正式训练对应的**权威实验详情身份**。

    返回 ``(experiment_run_id, reason_code)``。解析成功时 ``reason_code`` 为
    ``None``；否则为 :data:`EXPERIMENT_NOT_INDEXED` /
    :data:`EXPERIMENT_AMBIGUOUS` / :data:`LOCAL_INDEX_UNAVAILABLE`。

    身份只来自本地实验索引，两条分支互斥：

    * ``hpo_source.json`` **声明了** runtime 身份（字段存在且非空，不论值是否合法）：
      它就是这条正式训练声明的运行身份，只允许按该精确 ID 查询索引，且必须由**同一条
      记录**同时命中 run_name 与受控 ``detect/trainN`` 目录。未入索引、目录不符或值
      本身非法时一律明确不可用——**绝不改绑**到另一条同名同目录实验，也不补造 UUID。
    * metadata **真正缺少/为空**（仅此情形）：允许按 run_name 回退查询，回退必须唯一，
      0 条为 :data:`EXPERIMENT_NOT_INDEXED`，多条为 :data:`EXPERIMENT_AMBIGUOUS`。

    整个解析**只读**：不写来源 metadata、HPO 研究、执行审计、预算或排名。
    """
    if index is None:
        return None, LOCAL_INDEX_UNAVAILABLE

    declared = _declared_runtime_identity(runtime_run_id)
    try:
        if declared is not None:
            exact = _strict_runtime_identity(declared)
            if exact is None:
                return None, EXPERIMENT_NOT_INDEXED
            exp = index.get_experiment(exact)
            if isinstance(exp, dict) and _matches_controlled_train_dir(
                    exp, Path(train_dir), train_name):
                return exact, None
            return None, EXPERIMENT_NOT_INDEXED
        candidates = [
            exp for exp in index.find_reference_experiments(train_name)
            if isinstance(exp, dict)
            and _matches_controlled_train_dir(exp, Path(train_dir), train_name)
        ]
    except LocalIndexError:
        return None, LOCAL_INDEX_UNAVAILABLE
    except Exception:
        return None, LOCAL_INDEX_UNAVAILABLE

    if len(candidates) == 1:
        run_id = candidates[0].get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id, None
        return None, EXPERIMENT_NOT_INDEXED
    if not candidates:
        return None, EXPERIMENT_NOT_INDEXED
    return None, EXPERIMENT_AMBIGUOUS


def _scan_formal_runs(*, log_dir, detect_dir, study_id: str, manager=None,
                      index=None) -> tuple[list[dict], list[dict], bool]:
    """Strict, bounded, read-only scan of the controlled Detect directory.

    Returns ``(rows, warnings, truncated)``. Only metadata that validates as a
    ``formal`` source bound to this study becomes a row; a present-but-broken
    record is reported through a stable warning code so the caller never claims
    "no formal history" when the truth is "history exists but is unreadable".
    """
    if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
        raise HpoArtifactError("HPO_ARTIFACT_INVALID", "研究 ID 格式不合法")

    rows: list[dict] = []
    warnings: list[dict] = []
    history = _history_index(log_dir)
    train_dirs, truncated = _bounded_dir_entries(Path(detect_dir))
    if truncated:
        warnings.append({"code": "FORMAL_RUNS_TRUNCATED", "train_name": None})

    for train_dir in train_dirs:
        train_name = train_dir.name
        source, present = _load_source_file(train_dir / FORMAL_SOURCE_FILENAME)
        if source is None:
            if present:
                warnings.append({"code": "FORMAL_SOURCE_UNREADABLE",
                                 "train_name": train_name})
            continue
        # 非 formal（同条件验证等）不是本研究的正式训练，静默跳过。
        if source.get("mode") != FORMAL_SOURCE_MODE:
            continue
        source_study = source.get("study_id")
        trial_id = source.get("trial_id")
        recorded_train_name = source.get("train_name")
        if not (isinstance(source_study, str) and STUDY_ID_RE.fullmatch(source_study)
                and isinstance(trial_id, str) and TRIAL_ID_RE.fullmatch(trial_id)
                and trial_id.startswith(source_study + "_t")
                # 目录身份必须与来源记录一致（新记录才写 train_name；旧记录缺省跳过）
                and (recorded_train_name is None
                     or recorded_train_name == train_name)):
            warnings.append({"code": "FORMAL_SOURCE_INVALID",
                             "train_name": train_name})
            continue
        if source_study != study_id:
            continue

        record = history.get(train_name)
        controller = _live_manual_status(manager, train_name)

        status = "unknown"
        metrics: dict = {}
        epochs_kpi = None
        analysis_status = None
        started_at = None
        finished_at = None
        if controller is not None and controller.is_active():
            status = getattr(getattr(controller, "run_state", None), "status", None) or "running"
        elif record is not None:
            status = record.get("status") or "unknown"
            raw_metrics = record.get("metrics")
            metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
            epochs_kpi = record.get("epochs") if isinstance(record.get("epochs"), dict) else None
            analysis_status = record.get("analysis_status")
            started_at = record.get("started_at")
            finished_at = record.get("finished_at")
        if not metrics:
            # 统一历史缺失/未收尾时回退到运行目录的 results.csv 末行（run 事实）。
            metrics = _last_results_row(train_dir)
        # 只有真正完成且能读到指标的运行才算“有最终结果”，不把 202 当完成。
        result_available = status == "completed" and bool(metrics)

        # 最终权重入口只在“该 run 已完成、且隔离目录里确实存在受控 best.pt”时可用：
        # 文件存在本身不足以声称正式训练完成（失败/停止的运行也可能留下权重）。
        best_available = False
        best_path = train_dir / "weights" / "best.pt"
        if status == "completed" and best_path.is_file() and not _is_link_like(best_path):
            best_available = True

        # runtime 身份来自启动前写下的 metadata（活跃/重载/终态一致）；旧记录
        # 缺少该字段时明确标记缺失，并用活跃控制器回退，绝不补造 UUID。
        declared_runtime_run_id = _declared_runtime_identity(source.get("runtime_run_id"))
        live_runtime_run_id = None
        if controller is not None and controller.is_active():
            live_runtime_run_id = _declared_runtime_identity(
                getattr(getattr(controller, "run_state", None), "run_id", None))
        # 监控身份只认严格合法的 runtime ID：JSON 历史 ID（``manual:train1``）或
        # 其他非法值既不能查询 ``/api/runs/{run_id}``，也不是一次真实的运行身份。
        # metadata 缺失/非法时才回退到**活动**控制器持有的真实身份。
        runtime_run_id = (_strict_runtime_identity(declared_runtime_run_id)
                          or _strict_runtime_identity(live_runtime_run_id))
        history_run_id = None
        if record is not None:
            raw_history_id = record.get("run_id")
            if isinstance(raw_history_id, str) and raw_history_id:
                history_run_id = raw_history_id

        # “查看结果”必须使用权威实验索引里的详情身份，而不是 JSON 历史 ID：
        # 详情接口以索引主键（``manual:<uuid4>``）查询。详情解析**只**吃 metadata
        # 的原始声明：字段存在即是一次声明（含非法值 → 明确不可用，禁止按名称回退），
        # 字段真正缺失才走按 run_name 的唯一回退。活动控制器身份只服务“查看监控”，
        # 不能拿来顶替详情身份，也不会改变旧记录的兼容行为。
        experiment_run_id, identity_reason = resolve_formal_experiment_identity(
            index=index, train_dir=train_dir, train_name=train_name,
            runtime_run_id=declared_runtime_run_id)

        rows.append({
            "train_name": train_name,
            "runtime_run_id": runtime_run_id,
            "runtime_identity_missing": runtime_run_id is None,
            "history_run_id": history_run_id,
            "experiment_run_id": experiment_run_id,
            "experiment_identity_reason": identity_reason,
            "status": status,
            "analysis_status": analysis_status,
            "started_at": started_at,
            "finished_at": finished_at,
            "metrics": metrics,
            "epochs": epochs_kpi,
            "result_available": result_available,
            "best_pt_available": best_available,
            "source_trial_id": source.get("trial_id"),
            "source_trial_number": source.get("trial_number"),
            "training_config": source.get("training_config"),
            "search_params": source.get("search_params"),
            "differences": source.get("differences"),
            "source_readable": True,
        })
    return rows, warnings, truncated


def list_formal_runs(*, log_dir, detect_dir, study_id: str, manager=None,
                     index=None) -> list[dict]:
    """按 study 投影普通训练的正式运行事实（只读；不写 study/execution）。

    状态优先级：内存中仍在运行的普通训练控制器 → 统一历史的 JSON 事实 →
    ``unknown``（诚实呈现，不猜测终态）。指标来自统一历史，缺失时回退到
    ``results.csv`` 末行。``index`` 为权威实验索引（用于解析详情身份，只读）。
    只返回可严格校验的行；坏记录见 :func:`project_formal_runs`。
    """
    rows, _warnings, _truncated = _scan_formal_runs(
        log_dir=log_dir, detect_dir=detect_dir, study_id=study_id, manager=manager,
        index=index)
    return rows


def project_formal_runs(*, log_dir, detect_dir, study_id: str, manager=None,
                        index=None) -> dict:
    """``list_formal_runs`` 的完整投影：合法行 + 安全 warnings + 截断标记。"""
    rows, warnings, truncated = _scan_formal_runs(
        log_dir=log_dir, detect_dir=detect_dir, study_id=study_id, manager=manager,
        index=index)
    return {"runs": rows, "warnings": warnings, "truncated": truncated}


# ── 最小公共提交逻辑（复用于普通训练控制器/收尾/门禁） ──────────────


@dataclass(frozen=True)
class TrainingSubmitDeps:
    """app.py 注入的普通训练依赖；本模块不 import app，避免循环依赖。"""

    run_manager: Any
    detect_dir: Callable[[], str]
    resolve_executable: Callable[[], str]
    new_run_state: Callable[..., Any]
    broker_factory: Callable[[str], Any]
    manual_controller_cls: type
    finalize_cb: Callable[[Any, int], dict]
    pick_train_name: Callable[[str], str]
    create_train_dirs: Callable[[str, str], None]
    persist_run_state: Callable[[str, Any], None]
    reserve: Callable[[str, str], tuple]
    release: Callable[[Any], None]
    gate: Callable[[], Any]
    log_dir: str = "log"
    state_file: str = os.path.join("log", "training_running.json")
    invalidate_cache: Callable[[str], None] | None = None
    run_state_replace: Callable[[Any, str], Any] | None = None


def _json_error(status_code: int, error_code: str, message: str,
                next_action: str) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse({"error": message, "error_code": error_code,
                         "next_action": next_action}, status_code=status_code)


def submit_formal_training(deps: TrainingSubmitDeps,
                           verified: VerifiedFormalTraining) -> Any:  # noqa: C901
    """用普通训练控制器启动一次正式训练；返回 202 JSON 或受控错误。

    校验/占用失败必须零进程、零新训练目录：所有写盘都发生在拿到共享槽位之后，
    而且任何写盘失败都会释放槽位并返回稳定错误。**不创建第二种训练执行器**。
    """
    import yaml

    gate = deps.gate()
    if gate is not None:
        return gate

    detect_dir = deps.detect_dir()
    train_name = deps.pick_train_name(detect_dir)
    run_state = deps.new_run_state("manual", run_name=train_name)
    reservation_token, busy = deps.reserve("manual", run_state.run_id)
    if busy is not None:
        return busy

    train_dir = os.path.join(detect_dir, train_name)
    try:
        train_name = deps.pick_train_name(detect_dir)
        train_dir = os.path.join(detect_dir, train_name)
        deps.create_train_dirs(detect_dir, train_dir)
    except OSError:
        deps.release(reservation_token)
        return _json_error(
            500, "TRAIN_DIR_CREATE_FAILED", "训练目录创建失败，未启动训练",
            "请检查 Detect 目录权限、磁盘空间与同名占用后重试。")
    if train_name != run_state.run_name and deps.run_state_replace is not None:
        run_state = deps.run_state_replace(run_state, train_name)

    params = dict(verified.effective)
    params["data"] = os.path.abspath(params["data"])
    params["name"] = train_name
    params["project"] = os.path.abspath(detect_dir)
    params["exist_ok"] = "True"
    try:
        with open(os.path.join(train_dir, "args.yaml"), "w", encoding="utf-8") as fh:
            yaml.dump(params, fh, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
    except OSError:
        deps.release(reservation_token)
        return _json_error(
            500, "ARGS_PERSIST_FAILED", "训练参数文件写入失败，未启动训练",
            "系统未启动训练，请检查运行目录写入权限后重试。")

    source = verified.source_metadata(runtime_run_id=run_state.run_id,
                                      train_name=train_name)
    try:
        with open(os.path.join(train_dir, FORMAL_SOURCE_FILENAME), "w",
                  encoding="utf-8") as fh:
            json.dump(source, fh, ensure_ascii=False, indent=2)
    except OSError:
        deps.release(reservation_token)
        return _json_error(
            500, "SOURCE_METADATA_PERSIST_FAILED", "来源元数据写入失败，未启动训练",
            "系统未启动训练，请检查运行目录写入权限后重试。")

    from auto_tune.modules.run_state.models import RunStatePersistenceError

    try:
        deps.persist_run_state(deps.state_file, run_state)
    except RunStatePersistenceError:
        deps.release(reservation_token)
        return _json_error(
            500, "RUN_STATE_PERSIST_FAILED", "运行状态写入失败，未启动训练",
            "系统未启动训练，请稍后重试。")

    cmd = [deps.resolve_executable(), "train"]
    for key, value in params.items():
        cmd.append(f"{key}={value}")

    broker = deps.broker_factory(run_state.run_id)
    controller = deps.manual_controller_cls(
        run_state=run_state,
        state_file=deps.state_file,
        cmd=cmd,
        params=params,
        train_name=train_name,
        train_dir=train_dir,
        data_yaml=params["data"],
        model=params["model"],
        epochs=int(params["epochs"]),
        log_path=os.path.join(train_dir, "training.log"),
        finalize_cb=deps.finalize_cb,
        broker=broker,
        manager=deps.run_manager,
        reservation_token=reservation_token,
    )
    controller.source_hpo = {
        "mode": FORMAL_SOURCE_MODE,
        "study_id": verified.study_id,
        "trial_id": verified.trial_id,
        "trial_number": verified.trial_number,
    }
    deps.run_manager.register(controller)
    try:
        controller.start()
    except Exception:
        deps.release(reservation_token)
        raise
    if deps.invalidate_cache is not None:
        deps.invalidate_cache("load_data")

    return _json_accepted(run_state, train_name, verified)


def _json_accepted(run_state, train_name: str, verified: VerifiedFormalTraining) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse({
        "status": "accepted",
        "run_id": run_state.run_id,
        "train_name": train_name,
        "mode": FORMAL_SOURCE_MODE,
        "source": {
            "study_id": verified.study_id,
            "trial_id": verified.trial_id,
            "trial_number": verified.trial_number,
            "value": verified.value,
            "epoch": verified.epoch,
        },
        "training_config": dict(verified.training_config),
        "differences": dict(verified.differences),
        "links": {
            "monitor": "/api/training/running",
            "formal_runs": f"/api/hpo/studies/{verified.study_id}/formal-runs",
        },
        "error_code": None,
        "error": None,
        "next_action": None,
    }, status_code=202)


# ── 受控错误映射（绝不回显底层 message / 路径 / 凭据） ──────────────


_TRAINING_ERROR_STATUS = {
    "HPO_INVALID_CONFIG": 422,
    "HPO_INVALID_EXECUTION_CONFIG": 422,
    "HPO_INVALID_RESULT": 422,
    "HPO_NOT_FOUND": 404,
    "HPO_NO_SUCCESS": 409,
    "HPO_SOURCE_INVALID": 409,
    "HPO_BINDING_MISMATCH": 409,
    "HPO_VERSION_MISMATCH": 409,
    "HPO_EXECUTION_CONFLICT": 409,
    "HPO_PENDING_TRIAL": 409,
    "HPO_BUDGET_EXHAUSTED": 409,
    "HPO_RESULT_CONFLICT": 409,
    "HPO_STUDY_BUSY": 409,
    "HPO_EXECUTION_BUSY": 409,
    "HPO_RECOVERY_REQUIRED": 409,
    "HPO_CORRUPT_STUDY": 500,
    "HPO_CORRUPT_EXECUTION": 500,
    "HPO_PERSISTENCE_ERROR": 500,
}

# 409 必须区分来源失效/绑定失效/占用等，不能一律提示“已有训练”。
_TRAINING_ERROR_TEXT = {
    "HPO_INVALID_CONFIG": ("正式训练条件不合法，未启动训练。",
                           "请按界面提示的范围修正训练轮数、Batch、图像尺寸与计算设备。"),
    "HPO_INVALID_EXECUTION_CONFIG": ("正式训练条件不合法，未启动训练。",
                                     "请按界面提示的范围修正后重试。"),
    "HPO_INVALID_RESULT": ("来源结果不合法，未启动训练。", "请刷新研究结果后重试。"),
    "HPO_NOT_FOUND": ("未找到该研究任务，未启动训练。", "请刷新列表重新选择研究。"),
    "HPO_NO_SUCCESS": ("还没有可用的成功试验结果，无法用于正式训练。",
                       "请先完成至少一个成功试验。"),
    "HPO_SOURCE_INVALID": ("该最佳结果当前不可用于正式训练，未启动训练。",
                           "请确认研究已完成、来源为排名第一的成功试验，且搜索参数在新条件下仍然合法。"),
    "HPO_BINDING_MISMATCH": ("数据集快照或初始权重已改变，未启动训练。",
                             "请重新创建研究以绑定当前数据与权重，或恢复原始文件后重试。"),
    "HPO_VERSION_MISMATCH": ("运行环境版本已改变，未启动训练。",
                             "请恢复原运行环境后重试。"),
    "HPO_EXECUTION_CONFLICT": ("该研究的执行状态不允许启动正式训练。",
                               "请先按界面提示完成或恢复该研究。"),
    "HPO_STUDY_BUSY": ("该研究任务暂忙，未启动训练。", "请稍后刷新重试。"),
    "HPO_EXECUTION_BUSY": ("执行器暂忙，未启动训练。", "请稍后刷新重试。"),
    "HPO_RECOVERY_REQUIRED": ("该研究需要先处理恢复问题，未启动训练。",
                              "请先检查并恢复该研究；BLOCKED 不支持强制继续。"),
    "HPO_CORRUPT_STUDY": ("研究任务记录已损坏，无法启动正式训练。",
                          "系统未启动训练，请检查历史原始记录。"),
    "HPO_CORRUPT_EXECUTION": ("执行审计记录已损坏，无法启动正式训练。",
                              "系统未启动训练，请检查历史原始记录。"),
    "HPO_PERSISTENCE_ERROR": ("记录读写失败，系统未启动训练。",
                              "请稍后重试或检查日志目录写入权限。"),
}


_ARTIFACT_ERROR_STATUS = {
    "HPO_ARTIFACT_INVALID": 422,
    "HPO_ARTIFACT_IDENTITY_MISMATCH": 409,
    "HPO_ARTIFACT_NOT_AVAILABLE": 404,
    "HPO_ARTIFACT_NOT_COMPLETED": 409,
    "HPO_ARTIFACT_OPEN_UNSUPPORTED": 409,
    "HPO_ARTIFACT_OPEN_FAILED": 500,
}

_ARTIFACT_ERROR_TEXT = {
    "HPO_ARTIFACT_INVALID": ("产物请求不合法，未返回任何文件。",
                             "请刷新界面后从提供的入口重新下载。"),
    "HPO_ARTIFACT_IDENTITY_MISMATCH": ("该产物不属于当前研究或训练，未返回任何文件。",
                                       "请刷新界面后重新选择研究与试验。"),
    "HPO_ARTIFACT_NOT_AVAILABLE": ("该产物不存在或尚未生成。",
                                   "请确认对应训练已完成后再下载。"),
    "HPO_ARTIFACT_NOT_COMPLETED": ("该正式训练尚未成功完成，未打开结果文件夹。",
                                   "请等待正式训练成功完成后再重试。"),
    "HPO_ARTIFACT_OPEN_UNSUPPORTED": ("当前运行环境不支持自动打开结果文件夹。",
                                      "请手动到调优输出目录查看该试验的结果。"),
    "HPO_ARTIFACT_OPEN_FAILED": ("无法打开结果文件夹，系统未执行其它操作。",
                                 "请稍后重试，或手动到调优输出目录查看。"),
}


def artifact_error_response(exc: HpoArtifactError) -> Any:
    """受控产物错误响应；只回稳定错误码与固定中文，绝不回路径。"""
    code = getattr(exc, "code", None)
    if code not in _ARTIFACT_ERROR_STATUS:
        code = "HPO_ARTIFACT_NOT_AVAILABLE"
    status = _ARTIFACT_ERROR_STATUS[code]
    message, next_action = _ARTIFACT_ERROR_TEXT[code]
    return JSONResponse(
        {"error_code": code, "error": message, "next_action": next_action},
        status_code=status,
    )


def hpo_training_error_response(exc: HpoError) -> Any:
    """稳定、脱敏的正式训练错误响应；``exc.message`` 永不回显。"""
    from fastapi.responses import JSONResponse

    from .hpo_api import safe_hpo_error_code

    code = safe_hpo_error_code(getattr(exc, "code", None))
    status = _TRAINING_ERROR_STATUS.get(code, 500)
    message, next_action = _TRAINING_ERROR_TEXT.get(
        code, ("处理请求时发生未知错误，未启动训练。", "系统未启动训练，请稍后重试。"))
    return JSONResponse(
        {"error_code": code, "error": message, "next_action": next_action},
        status_code=status,
    )


# ── Router ─────────────────────────────────────────────────────────


def _deref(value):
    """Accept either a live instance or a zero-arg accessor.

    The app binds its HPO service/runner/manager as module globals that can be
    replaced (tests, restart). Reading them through an accessor at request time
    prevents this router from silently keeping a stale instance.
    """
    return value() if callable(value) and not isinstance(value, type) else value


def create_hpo_training_router(*, service, runner, manager, detect_dir, log_dir,
                               build_deps, index=None, opener=None) -> Any:
    """Build the formal-training + linked-results router (prefix ``/api/hpo``).

    ``service``/``runner``/``manager`` may be instances or zero-arg accessors;
    ``detect_dir()`` and ``build_deps()`` are resolved **per request** so tests
    and a restarted server always see the current controlled directories and the
    current app-level callables. Nothing here owns a controller lifecycle: the
    submission delegates to :func:`submit_formal_training`.

    ``opener`` is an optional callable used to reveal the rank-1 result folder.
    It exists so automated tests can inject a recorder instead of launching the
    system file manager; production leaves it ``None`` (the Windows shell).
    """

    def _studies_404(exc: HpoError) -> Any:
        return hpo_training_error_response(exc)

    async def _train_best(study_id: str, request: Request):
        from .hpo_api import field_error_response

        try:
            raw = await request.json()
        except (ValueError, TypeError, json.JSONDecodeError):
            return _json_error(422, "HPO_INVALID_CONFIG", "请求体不是合法的 JSON。",
                               "请检查请求后重试。")
        try:
            payload = TrainBestRequest.model_validate(raw)
        except ValidationError as exc:
            return field_error_response(exc)
        try:
            verified = resolve_hpo_formal_training(
                _deref(service), _deref(runner), _deref(manager), study_id,
                payload.trial_id, payload.training_config)
        except HpoError as exc:
            return hpo_training_error_response(exc)
        except Exception:  # defensive: never leak an internal traceback
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "处理请求时发生未知错误，未启动训练。",
                               "系统未启动训练，请稍后重试。")
        try:
            deps = build_deps()
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "训练环境暂不可用，未启动训练。",
                               "系统未启动训练，请稍后重试。")
        return submit_formal_training(deps, verified)

    def _formal_runs(study_id: str):
        try:
            _deref(service).load_study(study_id)
        except HpoError as exc:
            return _studies_404(exc)
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "读取研究任务时发生未知错误。",
                               "请稍后重试。")
        try:
            projection = project_formal_runs(
                log_dir=log_dir, detect_dir=detect_dir(),
                study_id=study_id, manager=_deref(manager),
                # 只读解析详情身份；accessor 在请求时求值，测试/重启后仍看到当前索引
                index=_deref(index) if index is not None else None)
        except HpoArtifactError as exc:
            if exc.code == "HPO_ARTIFACT_INVALID":
                return _json_error(422, "HPO_INVALID_CONFIG", "研究 ID 格式不合法。",
                                   "请刷新列表重新选择研究。")
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "读取正式训练关联结果时发生未知错误。",
                               "请稍后重试。")
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "读取正式训练关联结果时发生未知错误。",
                               "请稍后重试。")
        rows = projection["runs"]
        return JSONResponse({
            "study_id": study_id,
            "runs": rows,
            "count": len(rows),
            "warnings": projection["warnings"],
            "truncated": projection["truncated"],
        })

    def _formal_artifact(study_id: str, train_name: str, name: str):
        from fastapi.responses import FileResponse

        try:
            path = resolve_formal_artifact(detect_dir(), study_id, train_name, name)
        except HpoArtifactError as exc:
            return artifact_error_response(exc)
        except Exception:
            return artifact_error_response(
                HpoArtifactError("HPO_ARTIFACT_NOT_AVAILABLE"))
        return FileResponse(str(path), filename=name,
                            media_type="application/octet-stream")

    async def _open_best_folder(study_id: str, request: Request):
        """打开当前 rank-1 试验的结果目录；客户端只提交研究身份，绝不提交路径。"""
        from .hpo_api import _read_empty_object

        body = await _read_empty_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            result = open_best_run_folder(
                _deref(service), _deref(runner), study_id,
                opener=_deref(opener) if opener is not None else None)
        except HpoArtifactError as exc:
            return artifact_error_response(exc)
        except HpoError as exc:
            return hpo_training_error_response(exc)
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "打开结果文件夹时发生未知错误。",
                               "请稍后重试或手动到调优输出目录查看。")
        return JSONResponse({
            "study_id": result["study_id"],
            "trial_id": result["trial_id"],
            "opened": True,
            "error_code": None, "error": None, "next_action": None,
        })

    async def _open_formal_folder(study_id: str, train_name: str, request: Request):
        """打开当前研究关联的正式训练结果目录。

        与 ``/best/open-folder``（搜索阶段 rank-1 试验目录）语义不同：这里只接受
        研究身份 + 受控训练编号，目标目录由服务端从 detect 根重建并复验来源
        metadata；客户端提交任何路径字段都会被拒绝（零系统调用）。完成态门控由
        服务端执行：状态取自 ``/formal-runs`` 的同一套权威投影（统一历史 / 活动
        控制器 / 受控目录事实），客户端提交的状态一律不作数。
        """
        from .hpo_api import _read_empty_object

        body = await _read_empty_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            _deref(service).load_study(study_id)
        except HpoError as exc:
            return _studies_404(exc)
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "读取研究任务时发生未知错误。",
                               "请稍后重试。")

        def _authoritative_status(name: str) -> str | None:
            return resolve_formal_run_status(
                log_dir=log_dir, detect_dir=detect_dir(), study_id=study_id,
                train_name=name, manager=_deref(manager))

        try:
            result = open_formal_run_folder(
                detect_dir(), study_id, train_name,
                status_resolver=_authoritative_status,
                opener=_deref(opener) if opener is not None else None)
        except HpoArtifactError as exc:
            return artifact_error_response(exc)
        except Exception:
            return _json_error(500, "HPO_EXECUTION_ERROR",
                               "打开结果文件夹时发生未知错误。",
                               "请稍后重试，或手动到 Detect 目录查看。")
        return JSONResponse({
            "study_id": result["study_id"],
            "train_name": result["train_name"],
            "opened": True,
            "error_code": None, "error": None, "next_action": None,
        })

    router = APIRouter()
    router.add_api_route("/studies/{study_id}/train-best", _train_best,
                         methods=["POST"], name="hpo_train_best")
    router.add_api_route("/studies/{study_id}/best/open-folder", _open_best_folder,
                         methods=["POST"], name="hpo_open_best_folder")
    router.add_api_route(
        "/studies/{study_id}/formal-runs/{train_name}/open-folder",
        _open_formal_folder, methods=["POST"], name="hpo_open_formal_folder")
    router.add_api_route("/studies/{study_id}/formal-runs", _formal_runs,
                         methods=["GET"], name="hpo_formal_runs")
    router.add_api_route(
        "/studies/{study_id}/formal-runs/{train_name}/artifacts/{name}",
        _formal_artifact, methods=["GET"], name="hpo_formal_artifact")
    return router
