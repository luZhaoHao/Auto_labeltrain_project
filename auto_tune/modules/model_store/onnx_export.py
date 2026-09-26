"""F1.2-A manual PT → ONNX export over the controlled weight library.

Uploading keeps a ``.pt`` opaque. This module is the one place a *trusted,
operator-confirmed* weight is actually loaded, and it does so in a bounded child
process: the child protects Studio's availability (a crash, a hang or a broken
checkpoint must not take the server down) — it is **not** a sandbox for a
malicious pickle.

Nothing here exposes a server path. The client only ever sends the ``model_id``
the library already publishes; the target is derived from that record
(``<stem>.onnx`` / ``<stem>.fp16.onnx`` beside the weight), and every failure
carries a stable code with a fixed message.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .service import (
    MODEL_CHANGED,
    MODEL_PATH_FORBIDDEN,
    ModelRecord,
    ModelStore,
    ModelStoreError,
    _CHUNK_SIZE,
    _hash_file,
    _is_plain_file,
)

__all__ = [
    "FP32",
    "FP16",
    "ExportInfo",
    "ExportJob",
    "OnnxExportError",
    "OnnxExportService",
    "build_worker_command",
    "build_worker_env",
    "run_export_worker",
    "run_worker_process",
]

FP32 = "fp32"
FP16 = "fp16"
_PRECISIONS = (FP32, FP16)
_PRECISION_SUFFIX = {FP32: "", FP16: ".fp16"}

DEFAULT_TIMEOUT_SECONDS = 600
EXPORT_WORKDIR_PREFIX = ".export-"

# 发布的校验事实：导出成功时与产物同目录写下的一份 JSON 副档。存在同名的
# ``.onnx`` 并不等于“已导出”——只有本服务校验通过、原子发布并写下该事实、
# 且产物此后未被替换/损坏的文件，才允许显示状态与提供下载。没有它就没有
# 导出（外部预存的同名文件一律不算）。这里不引入数据库，副档就是最小事实。
_RECORD_SUFFIX = ".export.json"
_RECORD_VERSION = "onnx-export-v1"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# 半精度默认关闭。2026-09-20 在本机（RTX 3060 Laptop / torch 2.5.1+cu121 /
# ultralytics 8.3.253 / onnx 1.17.0）对 yolov8n 实测：
#   · CPU 半精度（本模块固定的导出设备）经 onnxruntime.transformers 转换后，
#     图未拓扑排序，onnx.checker 直接拒绝 —— 本环境产不出合格半精度文件；
#   · 只有 device=0 的 CUDA 路径能产出 checker 通过、ONNX Runtime 可加载的
#     float16 模型，但它依赖可选 GPU，且在 CPU 交付形态下不可用。
# 因此首批不开放半精度：与其推出一个在 CPU 交付里必然失败、在 GPU 上数值偏差
# 明显放大的选项，不如保持隐藏。若要开放，需同时改变半精度导出设备选择并另做
# 一次 GPU 上的最小推理验收。
FP16_VERIFIED_DEFAULT = False

# 固定导出参数：页面不提供这些参数的输入入口，客户端也无法覆盖
EXPORT_KWARGS = {"imgsz": 640, "opset": 17, "dynamic": False, "simplify": False,
                 "nms": False}

_WORKER_MODULE = "auto_tune.modules.model_store.onnx_export_worker"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_EXPORT_INVALID_REQUEST = "MODEL_EXPORT_INVALID_REQUEST"
MODEL_EXPORT_INVALID_PRECISION = "MODEL_EXPORT_INVALID_PRECISION"
MODEL_EXPORT_TRUST_REQUIRED = "MODEL_EXPORT_TRUST_REQUIRED"
MODEL_EXPORT_UNSUPPORTED_ORIGIN = "MODEL_EXPORT_UNSUPPORTED_ORIGIN"
MODEL_EXPORT_CONFLICT = "MODEL_EXPORT_CONFLICT"
MODEL_EXPORT_BUSY = "MODEL_EXPORT_BUSY"
MODEL_EXPORT_TIMEOUT = "MODEL_EXPORT_TIMEOUT"
MODEL_EXPORT_FAILED = "MODEL_EXPORT_FAILED"
MODEL_EXPORT_PRECISION_UNSUPPORTED = "MODEL_EXPORT_PRECISION_UNSUPPORTED"
MODEL_EXPORT_INVALID_OUTPUT = "MODEL_EXPORT_INVALID_OUTPUT"
MODEL_EXPORT_UNAVAILABLE = "MODEL_EXPORT_UNAVAILABLE"
MODEL_EXPORT_NOT_FOUND = "MODEL_EXPORT_NOT_FOUND"

_STATUS_CODES = {
    MODEL_EXPORT_INVALID_REQUEST: 400,
    MODEL_EXPORT_INVALID_PRECISION: 400,
    MODEL_EXPORT_TRUST_REQUIRED: 400,
    MODEL_EXPORT_UNSUPPORTED_ORIGIN: 422,
    MODEL_EXPORT_CONFLICT: 409,
    MODEL_EXPORT_BUSY: 409,
    MODEL_EXPORT_TIMEOUT: 504,
    MODEL_EXPORT_FAILED: 500,
    MODEL_EXPORT_PRECISION_UNSUPPORTED: 422,
    MODEL_EXPORT_INVALID_OUTPUT: 500,
    MODEL_EXPORT_UNAVAILABLE: 503,
    MODEL_EXPORT_NOT_FOUND: 404,
}

# 子进程只能报告这些码；其余一律归并为通用失败，绝不把子进程文本透传给客户端
_MESSAGES = {
    MODEL_EXPORT_FAILED: "导出失败，未生成可用的 ONNX 文件。",
    MODEL_EXPORT_TIMEOUT: "导出超时，已中止本次导出。",
    MODEL_EXPORT_PRECISION_UNSUPPORTED: "当前环境无法生成半精度 ONNX 文件，本次导出未生效。",
    MODEL_EXPORT_INVALID_OUTPUT: "导出的 ONNX 文件未通过校验，本次导出未生效。",
    MODEL_EXPORT_UNAVAILABLE: "导出组件不可用，请检查运行环境。",
    MODEL_EXPORT_INVALID_PRECISION: "不支持的导出精度。",
    MODEL_EXPORT_INVALID_REQUEST: "导出请求格式不正确。",
    MODEL_EXPORT_TRUST_REQUIRED: "请先确认该权重来源可信，再发起导出。",
}


def stable_message(code: str) -> str:
    """The one fixed client-facing message for a stable export error code."""
    return _MESSAGES.get(code, _MESSAGES[MODEL_EXPORT_FAILED])


class OnnxExportError(ModelStoreError):
    """A stable, client-safe export failure. ``message`` never contains a path."""

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(code, message,
                         _STATUS_CODES.get(code, 400) if status_code is None
                         else status_code)


@dataclass(frozen=True)
class ExportInfo:
    """One committed export. ``target`` is server-side only, never projected."""

    model_id: str
    precision: str
    name: str
    size_bytes: int
    target: Path = field(repr=False, compare=False)

    def public_dict(self) -> dict:
        return {"model_id": self.model_id, "precision": self.precision,
                "name": self.name, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ExportJob:
    """Everything the bounded child is allowed to know.

    Every path here is derived server-side from a verified record; the child
    never receives a client-supplied path and never receives client-supplied
    export parameters.
    """

    python_executable: str
    source: Path
    workdir: Path
    output_name: str
    precision: str
    result_path: Path
    timeout_seconds: int


# ── 子进程调用 ─────────────────────────────────────────────────────


def build_worker_command(job: ExportJob) -> list[str]:
    return [job.python_executable, "-m", _WORKER_MODULE,
            "--source", str(job.source),
            "--workdir", str(job.workdir),
            "--precision", job.precision,
            "--result", str(job.result_path)]


def build_worker_env() -> dict:
    """Environment for the child: importable package, no automatic installs."""
    env = dict(os.environ)
    env["YOLO_AUTOINSTALL"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"
    inherited = [part for part in (env.get("PYTHONPATH") or "").split(os.pathsep)
                 if part]
    env["PYTHONPATH"] = os.pathsep.join([str(_PROJECT_ROOT)] + inherited)
    return env


def run_worker_process(command: list[str], *, cwd: str, timeout_seconds: int,
                       result_path: Path) -> dict:
    """Run the bounded child and return the result document it wrote."""
    try:
        completed = subprocess.run(command, cwd=cwd, env=build_worker_env(),
                                   capture_output=True,
                                   timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise OnnxExportError(MODEL_EXPORT_TIMEOUT,
                              _MESSAGES[MODEL_EXPORT_TIMEOUT]) from exc
    except OSError as exc:
        raise OnnxExportError(MODEL_EXPORT_UNAVAILABLE,
                              _MESSAGES[MODEL_EXPORT_UNAVAILABLE]) from exc
    if completed.returncode != 0:
        # 子进程崩溃（含加载权重时的硬失败）：只有稳定提示，没有堆栈
        raise OnnxExportError(MODEL_EXPORT_FAILED,
                              _MESSAGES[MODEL_EXPORT_FAILED])
    try:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OnnxExportError(MODEL_EXPORT_FAILED,
                              _MESSAGES[MODEL_EXPORT_FAILED]) from exc
    if not isinstance(payload, dict):
        raise OnnxExportError(MODEL_EXPORT_FAILED, _MESSAGES[MODEL_EXPORT_FAILED])
    return payload


def run_export_worker(job: ExportJob) -> dict:
    """The shipping runner: spawn the child from the project root."""
    return run_worker_process(build_worker_command(job), cwd=str(_PROJECT_ROOT),
                              timeout_seconds=job.timeout_seconds,
                              result_path=job.result_path)


# ── 导出服务 ───────────────────────────────────────────────────────


class OnnxExportService:
    """Identity-checked, single-flight, no-overwrite export of managed weights.

    ``store`` is the live :class:`ModelStore`, or a zero-argument accessor
    returning it, so a rebind of the module global is picked up without
    rebuilding the service. ``runner`` is the bounded child invocation; tests
    replace it so the identity/conflict rules can be exercised without a real
    checkpoint.
    """

    def __init__(self, store, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                 fp16_available: bool = False,
                 runner: Callable[[ExportJob], dict] | None = None,
                 python_executable: str | None = None):
        self._store_ref = store
        self._timeout_seconds = max(1, int(timeout_seconds))
        self._fp16_available = bool(fp16_available)
        self._runner = runner if runner is not None else run_export_worker
        self._python = python_executable or sys.executable
        self._active: set[str] = set()
        self._lock = threading.Lock()

    # ── 对外 ──

    @property
    def fp16_available(self) -> bool:
        return self._fp16_available

    def export(self, model_id, precision, *, source_trusted: bool = False) -> ExportInfo:
        """Export one managed weight, once, without ever overwriting a file.

        ``source_trusted`` is the operator's explicit confirmation that the
        uploaded weight comes from a trusted source. It has no default "yes":
        a missing or non-boolean-true value is refused before anything loads
        the checkpoint, so no caller can start the loading child process
        without saying so on the request itself.
        """
        precision = self._validated_precision(precision)
        record = self._managed_record(model_id)
        if source_trusted is not True:
            raise OnnxExportError(MODEL_EXPORT_TRUST_REQUIRED,
                                  _MESSAGES[MODEL_EXPORT_TRUST_REQUIRED])
        target = self._target(record, precision)
        key = "%s|%s" % (record.model_id, precision)
        with self._lock:
            if key in self._active:
                raise OnnxExportError(MODEL_EXPORT_BUSY,
                                      "该权重正在导出，请等待本次导出结束。")
            if _is_plain_file(target):
                raise OnnxExportError(
                    MODEL_EXPORT_CONFLICT,
                    "该权重已存在同精度的 ONNX 文件，请改名后重新上传权重。")
            self._active.add(key)
        workdir = None
        try:
            workdir = self._new_workdir(target)
            source = self._stage(record, workdir)
            self._invoke(record, precision, source, workdir, target.name)
            produced = self._checked_output(workdir, target.name)
            # 转换期间来源被替换/删除时绝不发布
            self._managed_record(record.model_id)
            self._publish(produced, target, model_id=record.model_id,
                          precision=precision)
            return ExportInfo(record.model_id, precision, target.name,
                              target.stat().st_size, target)
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
            with self._lock:
                self._active.discard(key)

    def status(self, model_id) -> dict:
        """Read-only availability, derived from disk so a refresh still sees it."""
        record = self._record(model_id)
        exportable = record.origin == "managed"
        exports = {}
        for precision in _PRECISIONS:
            info = self._existing(record, precision) if exportable else None
            exports[precision] = info.public_dict() if info else None
        return {"model_id": record.model_id, "origin": record.origin,
                "exportable": exportable,
                "fp16_available": self._fp16_available, "exports": exports}

    def resolve_download(self, model_id, precision) -> ExportInfo:
        """Resolve an already-committed export for download, or fail stably."""
        precision = self._validated_precision(precision)
        record = self._managed_record(model_id)
        info = self._existing(record, precision)
        if info is None:
            raise OnnxExportError(MODEL_EXPORT_NOT_FOUND,
                                  "未找到该权重的 ONNX 导出文件。")
        return info

    # ── 内部 ──

    def _store(self) -> ModelStore:
        return self._store_ref() if callable(self._store_ref) else self._store_ref

    @staticmethod
    def _translate(exc: ModelStoreError) -> OnnxExportError:
        if exc.code == MODEL_PATH_FORBIDDEN:
            return OnnxExportError(
                MODEL_EXPORT_UNSUPPORTED_ORIGIN,
                "兼容来源的权重不支持导出，请先上传到受控权重库。")
        return OnnxExportError(exc.code, exc.message, exc.status_code)

    def _record(self, model_id) -> ModelRecord:
        try:
            return self._store().resolve_record(model_id)
        except ModelStoreError as exc:
            raise self._translate(exc) from exc

    def _managed_record(self, model_id) -> ModelRecord:
        try:
            return self._store().resolve_managed(model_id)
        except ModelStoreError as exc:
            raise self._translate(exc) from exc

    def _validated_precision(self, precision) -> str:
        if not isinstance(precision, str) or precision not in _PRECISIONS:
            raise OnnxExportError(MODEL_EXPORT_INVALID_PRECISION,
                                  "不支持的导出精度。")
        if precision == FP16 and not self._fp16_available:
            raise OnnxExportError(
                MODEL_EXPORT_INVALID_PRECISION,
                "当前环境未通过半精度导出验证，已停用半精度。")
        return precision

    @staticmethod
    def _target(record: ModelRecord, precision: str) -> Path:
        suffix = _PRECISION_SUFFIX[precision]
        return record.path.with_name("%s%s.onnx" % (record.path.stem, suffix))

    def _existing(self, record: ModelRecord, precision: str):
        """The committed export for this source, or ``None``.

        A file with the right name is not enough: the target must carry the
        validation fact this service wrote when it published it, and the target
        must still match that fact byte for byte. Anything else — a file the
        operator dropped in place, an artifact that was replaced, truncated or
        edited after publication, a fact belonging to another weight — reads as
        "no export" instead of an unverified download.
        """
        target = self._target(record, precision)
        if not _is_plain_file(target):
            return None
        published = self._published_facts(record, precision, target)
        if published is None:
            return None
        return ExportInfo(record.model_id, precision, target.name, published,
                          target)

    @staticmethod
    def _published_facts(record: ModelRecord, precision: str,
                         target: Path) -> int | None:
        published = _read_record(target)
        if published is None:
            return None
        if (published.get("model_id") != record.model_id
                or published.get("precision") != precision
                or published.get("name") != target.name):
            return None
        try:
            size = target.stat().st_size
        except OSError:
            return None
        if size <= 0 or published.get("size_bytes") != size:
            return None
        digest = published.get("sha256")
        if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
            return None
        try:
            if _hash_file(target) != digest:
                return None
        except OSError:
            return None
        return size

    def _new_workdir(self, target: Path) -> Path:
        workdir = target.parent / (EXPORT_WORKDIR_PREFIX + uuid.uuid4().hex)
        try:
            workdir.mkdir(parents=True)
        except OSError as exc:
            raise OnnxExportError(MODEL_EXPORT_UNAVAILABLE,
                                  _MESSAGES[MODEL_EXPORT_UNAVAILABLE]) from exc
        return workdir

    def _stage(self, record: ModelRecord, workdir: Path) -> Path:
        """Copy the verified weight into the work directory, re-hashing as we go."""
        source = workdir / record.name
        digest = hashlib.sha256()
        try:
            with open(record.path, "rb") as src, open(source, "wb") as dst:
                while True:
                    chunk = src.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    dst.write(chunk)
        except OSError as exc:
            raise OnnxExportError(MODEL_EXPORT_FAILED,
                                  _MESSAGES[MODEL_EXPORT_FAILED]) from exc
        if digest.hexdigest() != record.sha256:
            raise OnnxExportError(MODEL_CHANGED, "受控权重文件已被替换，请重新选择。")
        return source

    def _invoke(self, record: ModelRecord, precision: str, source: Path,
                workdir: Path, output_name: str) -> None:
        job = ExportJob(python_executable=self._python, source=source,
                        workdir=workdir, output_name=output_name,
                        precision=precision, result_path=workdir / "result.json",
                        timeout_seconds=self._timeout_seconds)
        result = self._runner(job)
        if isinstance(result, dict) and result.get("ok") is True:
            return
        code = result.get("error_code") if isinstance(result, dict) else None
        if code not in _MESSAGES:
            code = MODEL_EXPORT_FAILED
        raise OnnxExportError(code, stable_message(code))

    def _checked_output(self, workdir: Path, expected_name: str) -> Path:
        produced = workdir / expected_name
        if not _is_plain_file(produced):
            raise OnnxExportError(MODEL_EXPORT_INVALID_OUTPUT,
                                  _MESSAGES[MODEL_EXPORT_INVALID_OUTPUT])
        try:
            if produced.stat().st_size <= 0:
                raise OnnxExportError(MODEL_EXPORT_INVALID_OUTPUT,
                                      _MESSAGES[MODEL_EXPORT_INVALID_OUTPUT])
        except OSError as exc:
            raise OnnxExportError(MODEL_EXPORT_INVALID_OUTPUT,
                                  _MESSAGES[MODEL_EXPORT_INVALID_OUTPUT]) from exc
        return produced

    @staticmethod
    def _publish(produced: Path, target: Path, *, model_id: str,
                 precision: str) -> None:
        try:
            # 原子创建最终名称：目标已存在时失败，绝不覆盖既有导出
            os.link(produced, target)
        except FileExistsError as exc:
            raise OnnxExportError(
                MODEL_EXPORT_CONFLICT,
                "该权重已存在同精度的 ONNX 文件，请改名后重新上传权重。") from exc
        except OSError as exc:
            raise OnnxExportError(MODEL_EXPORT_FAILED,
                                  _MESSAGES[MODEL_EXPORT_FAILED]) from exc
        try:
            _write_record(target, model_id=model_id, precision=precision)
        except OSError as exc:
            # 校验事实写不下去，就不能留下一个看起来已经完成的产物
            try:
                os.unlink(target)
            except OSError:
                pass
            raise OnnxExportError(MODEL_EXPORT_FAILED,
                                  _MESSAGES[MODEL_EXPORT_FAILED]) from exc


def _record_path(target: Path) -> Path:
    """本服务为一次发布写下的校验事实所在路径（与产物同目录）。"""
    return target.with_name(target.name + _RECORD_SUFFIX)


def _read_record(target: Path) -> dict | None:
    """读取并粗校验校验事实；缺失、不可读或版本不符都只是“没有导出”。"""
    path = _record_path(target)
    if not _is_plain_file(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 含 UnicodeDecodeError：任何读不动/解析不了的副档都只是“没有导出”
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("record_version") != _RECORD_VERSION:
        return None
    return payload


def _write_record(target: Path, *, model_id: str, precision: str) -> None:
    """原子写下校验事实：同目录临时文件 flush/fsync 后整体替换。"""
    try:
        size = target.stat().st_size
        digest = _hash_file(target)
    except OSError as exc:
        raise OSError("published artifact unreadable") from exc
    payload = {
        "record_version": _RECORD_VERSION,
        "model_id": model_id,
        "precision": precision,
        "name": target.name,
        "size_bytes": size,
        "sha256": digest,
    }
    temp = target.with_name(".%s.record-%s.tmp" % (target.name, uuid.uuid4().hex))
    try:
        with open(temp, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, _record_path(target))
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
