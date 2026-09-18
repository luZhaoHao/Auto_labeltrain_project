"""H1.2 真实 results.csv 目标指标读取（确定性、无 DataFrame 宽松推断）。

目标按研究的评价模式计算：旧模式取 ``metrics/mAP50-95(B)`` 的最大值，快速/全面
模式取固定权重综合分数的最大值；同分一律取最早 epoch。不等同 Ultralytics
best.pt 的 fitness 选择或训练 final 指标。结构非法（缺列/重复表头/epoch 非
1..epochs 严格递增）直接判无可信指标；**行内**必需指标缺失/非法只排除该行
（不补 0），全部行都不可用同样判无可信指标。一次读字节并对同一字节计算 SHA256。
"""

import csv
import hashlib
import io
import math
import os
import re
import stat
from pathlib import Path

from .execution_models import MetricDiagnostics
from .models import (
    EVALUATION_WEIGHTS,
    LEGACY_MODE,
    METRIC_KEYS,
    OBJECTIVE_BY_MODE,
    Evidence,
    HpoError,
    ResultInput,
    evaluation_score,
    required_metric_keys,
)
from .storage import reject_link_chain

MAX_RESULTS_CSV_BYTES = 5 * 1024 * 1024
_EPOCH_TOKEN = "epoch"
_INTEGER_RE = re.compile(r"^[0-9]+$")


class ObjectiveResult:
    """目标读取结果：综合分数、最早最佳 epoch、相对产物路径、SHA256、诊断、
    评价模式/目标版本以及被选 epoch 的原始组成指标。"""

    __slots__ = ("value", "epoch", "artifact_relpath", "artifact_sha256",
                 "diagnostics", "evaluation_mode", "objective", "metrics")

    def __init__(self, value, epoch, artifact_relpath, artifact_sha256,
                 diagnostics, evaluation_mode=LEGACY_MODE, objective=None,
                 metrics=None):
        self.value = value
        self.epoch = epoch
        self.artifact_relpath = artifact_relpath
        self.artifact_sha256 = artifact_sha256
        self.diagnostics = diagnostics
        self.evaluation_mode = evaluation_mode
        self.objective = objective or OBJECTIVE_BY_MODE[evaluation_mode]
        self.metrics = dict(metrics or {})


def _finite_metric(raw: str):
    """单元格 → [0,1] 内有限浮点；缺失/空值/NaN/Infinity/越界/非数字返回 None。"""
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        return None
    return value


def _invalid(message: str) -> HpoError:
    return HpoError("HPO_INVALID_METRICS", message)


def _is_reparse(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def read_objective(run_dir: Path, *, artifact_root: Path,
                   run_id: str, epochs: int,
                   evaluation_mode: str = LEGACY_MODE) -> ObjectiveResult:
    """从已退出进程的 ``run_dir/results.csv`` 读取目标，返回诊断对象。

    非法结构抛 ``HPO_INVALID_METRICS``；成功值必须来自至少一行有效目标。
    """
    if isinstance(epochs, bool) or type(epochs) is not int or epochs < 1:
        raise _invalid("epochs must be a positive int")
    if evaluation_mode not in EVALUATION_WEIGHTS:
        raise _invalid("unknown evaluation mode")
    required = required_metric_keys(evaluation_mode)
    run_dir = Path(run_dir)
    artifact_root = Path(artifact_root)
    # 拒绝 artifact_root/run_dir 及其原始父链上的 symlink/reparse（R3）。
    try:
        reject_link_chain(artifact_root, code="HPO_INVALID_METRICS")
        reject_link_chain(run_dir, code="HPO_INVALID_METRICS")
    except HpoError:
        raise
    try:
        abs_run = Path(os.path.abspath(run_dir))
        abs_root = Path(os.path.abspath(artifact_root))
        rel = abs_run.relative_to(abs_root)
    except ValueError as exc:
        raise _invalid("run_dir must be under artifact_root") from exc
    rel_dir = rel.as_posix()
    if not rel_dir or rel_dir == ".":
        rel_dir = ""
    if _is_reparse(run_dir):
        raise _invalid("run_dir must not be a reparse point")

    target = run_dir / "results.csv"
    reject_link_chain(target, code="HPO_INVALID_METRICS")
    if _is_reparse(target):
        raise _invalid("results.csv must not be a reparse point")
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise _invalid("results.csv unreadable") from exc
    if size > MAX_RESULTS_CSV_BYTES:
        raise _invalid(f"results.csv exceeds {MAX_RESULTS_CSV_BYTES} bytes")
    try:
        with open(target, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise _invalid("results.csv unreadable") from exc
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid("results.csv is not UTF-8") from exc

    reader = csv.reader(io.StringIO(text))
    header = None
    rows = []
    for row in reader:
        if not row or all(cell.strip() == "" for cell in row):
            continue
        if header is None:
            header = [cell.strip() for cell in row]
            continue
        rows.append(row)
    if header is None:
        raise _invalid("results.csv has no header")
    if len(header) != len(set(header)):
        raise _invalid("results.csv has duplicate header columns")
    try:
        epoch_idx = header.index(_EPOCH_TOKEN)
    except ValueError as exc:
        raise _invalid("results.csv missing the epoch column") from exc
    try:
        metric_idx = {key: header.index(key) for key in required}
    except ValueError as exc:
        raise _invalid(
            "results.csv is missing a metric required by the evaluation mode") from exc
    # 组成指标只保存 header 里真实存在的那几列；缺失的列不参与，也不补 0。
    optional_idx = {key: header.index(key) for key in METRIC_KEYS
                    if key in header}
    if not rows:
        raise _invalid("results.csv has no data rows")

    best_value = None
    best_epoch = None
    best_metrics = None
    previous_epoch = 0
    total_rows = 0
    excluded_rows = 0
    for row in rows:
        if len(row) <= epoch_idx:
            raise _invalid("results.csv row is shorter than the header")
        raw_epoch = row[epoch_idx].strip()
        if not _INTEGER_RE.fullmatch(raw_epoch):
            raise _invalid("results.csv epoch is not a positive integer")
        epoch = int(raw_epoch)
        if not (1 <= epoch <= epochs):
            raise _invalid("results.csv epoch out of configured range")
        if epoch <= previous_epoch:
            raise _invalid("results.csv epoch must be strictly increasing")
        previous_epoch = epoch

        total_rows += 1
        observed = {}
        for key, idx in optional_idx.items():
            value = _finite_metric(row[idx]) if len(row) > idx else None
            if value is not None:
                observed[key] = value
        # 必需指标缺失/非法 → 排除该行，绝不用 0 代替
        score = evaluation_score(evaluation_mode, observed)
        if score is None:
            excluded_rows += 1
            continue
        if best_value is None or score > best_value:
            best_value = score
            best_epoch = epoch
            best_metrics = observed

    if best_value is None:
        raise _invalid("results.csv has no valid objective value")
    relpath = ("results.csv" if not rel_dir
               else f"{rel_dir}/results.csv")
    diagnostics = MetricDiagnostics(total_rows=total_rows,
                                    excluded_rows=excluded_rows)
    return ObjectiveResult(best_value, best_epoch, relpath, artifact_sha256,
                           diagnostics, evaluation_mode=evaluation_mode,
                           metrics=best_metrics)


def objective_evidence(obj: ObjectiveResult, run_id: str) -> Evidence:
    """把目标读取结果投影成自描述证据（模式、目标版本、被选 epoch 原始指标）。"""
    return Evidence(
        run_id=run_id,
        artifact_relpath=obj.artifact_relpath,
        artifact_sha256=obj.artifact_sha256,
        epoch=obj.epoch,
        evaluation_mode=obj.evaluation_mode,
        objective=obj.objective,
        metrics=dict(obj.metrics) or None,
    )


def extract_objective(run_dir: Path, *, artifact_root: Path,
                      run_id: str, epochs: int,
                      evaluation_mode: str = LEGACY_MODE) -> ResultInput:
    """返回可直接 tell 的 SUCCESS ResultInput（含证据与最佳 epoch）。"""
    obj = read_objective(run_dir, artifact_root=artifact_root,
                         run_id=run_id, epochs=epochs,
                         evaluation_mode=evaluation_mode)
    return ResultInput(state="SUCCESS", value=obj.value,
                       evidence=objective_evidence(obj, run_id))
