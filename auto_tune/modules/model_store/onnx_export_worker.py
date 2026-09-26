"""F1.2-A ONNX export child process.

The one place a ``.pt`` is ever loaded into memory. It is started by
:mod:`auto_tune.modules.model_store.onnx_export` with internally generated
paths, the frozen export arguments and a result file. Every failure is written
into that file as a stable code with a fixed message — never a traceback, never
a server path — and the process exits 0 whenever it managed to report, so the
parent can tell "the child reported a failure" from "the child crashed".

Boundedness, not sandboxing: this process isolates an export crash or hang from
the Studio server. It does not make an untrusted pickle safe. Ultralytics is
imported lazily, only once a source has been handed to the exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .onnx_export import (
    EXPORT_KWARGS,
    FP16,
    FP32,
    MODEL_EXPORT_FAILED,
    MODEL_EXPORT_INVALID_OUTPUT,
    MODEL_EXPORT_INVALID_PRECISION,
    MODEL_EXPORT_PRECISION_UNSUPPORTED,
)

__all__ = ["main", "run_export"]

_PRECISIONS = (FP32, FP16)
_FLOAT_TYPES = {"float32", "float16"}

# 固定导出设备：CPU 导出的行为在 Windows 与容器里一致，也不与训练争抢显存。
# 半精度在本栈上只有 CUDA 路径能产出合格模型（见 FP16_VERIFIED_DEFAULT），
# 因此半精度仍处停用状态，而不是悄悄换设备。
EXPORT_DEVICE = "cpu"

_MESSAGES = {
    MODEL_EXPORT_FAILED: "导出失败，未生成可用的 ONNX 文件。",
    MODEL_EXPORT_INVALID_OUTPUT: "导出的 ONNX 文件未通过校验，本次导出未生效。",
    MODEL_EXPORT_INVALID_PRECISION: "不支持的导出精度。",
    MODEL_EXPORT_PRECISION_UNSUPPORTED: "当前环境无法生成半精度 ONNX 文件，本次导出未生效。",
}


def _failure(code: str, message: str | None = None) -> dict:
    return {"ok": False, "error_code": code,
            "message": message or _MESSAGES.get(code, _MESSAGES[MODEL_EXPORT_FAILED])}


def _inside(workdir: Path, path: Path) -> bool:
    """The child only ever touches files directly inside its own work directory."""
    try:
        return path.resolve().parent == workdir.resolve()
    except OSError:
        return False


def run_export(workdir, source, precision, *, exporter=None,
               inspector=None) -> dict:
    """Export one staged copy and validate the result before reporting success.

    ``exporter`` and ``inspector`` are seams: the shipping defaults load
    Ultralytics and ONNX, tests substitute fakes so the frozen arguments and
    the validation rules can be checked without a real checkpoint.
    """
    workdir = Path(workdir)
    source = Path(source)
    if precision not in _PRECISIONS:
        return _failure(MODEL_EXPORT_INVALID_PRECISION)
    if not _inside(workdir, source) or not source.is_file():
        return _failure(MODEL_EXPORT_FAILED)

    exporter = exporter or _default_export
    inspector = inspector or _default_inspect
    half = precision == FP16
    try:
        requested = exporter(source, precision, **dict(EXPORT_KWARGS, half=half))
    except Exception:
        # 权重无法加载/导出：只回稳定码，绝不回传异常文本或堆栈
        return _failure(MODEL_EXPORT_FAILED)

    output = Path(requested) if requested is not None else None
    if (output is None or not _inside(workdir, output) or not output.is_file()
            or output.stat().st_size <= 0):
        return _failure(MODEL_EXPORT_INVALID_OUTPUT)

    try:
        dtypes = {str(value) for value in inspector(output)}
    except Exception:
        # 半精度路径上结构校验失败 = 本环境产不出可用半精度模型（Ultralytics 在
        # CPU 上经 onnxruntime.transformers 转换后图未拓扑排序，checker 会拒绝）
        return _failure(MODEL_EXPORT_PRECISION_UNSUPPORTED if precision == FP16
                        else MODEL_EXPORT_INVALID_OUTPUT)

    if precision == FP16:
        # 上游在半精度不可用时只打 warning 并退回 FP32：这里必须如实判失败
        if "float16" not in dtypes:
            return _failure(MODEL_EXPORT_PRECISION_UNSUPPORTED)
    elif "float32" not in dtypes or "float16" in dtypes:
        return _failure(MODEL_EXPORT_INVALID_OUTPUT)

    return {"ok": True, "precision": precision, "name": output.name,
            "size_bytes": output.stat().st_size,
            "weights_dtype": "float16" if precision == FP16 else "float32"}


def _default_export(source: Path, precision: str, **kwargs) -> Path:
    """Load the trusted checkpoint and run the Ultralytics exporter."""
    from ultralytics import YOLO

    model = YOLO(str(source))
    exported = model.export(format="onnx", device=EXPORT_DEVICE, **kwargs)
    return Path(exported)


def _default_inspect(path: Path) -> set[str]:
    """Structural check plus the float dtypes actually present in the weights."""
    import onnx

    onnx.checker.check_model(str(path))
    model = onnx.load(str(path))
    dtypes = set()
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            dtypes.add("float32")
        elif initializer.data_type == onnx.TensorProto.FLOAT16:
            dtypes.add("float16")
    return dtypes & _FLOAT_TYPES


def _write_result(result_path: Path, payload: dict) -> None:
    try:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False),
                               encoding="utf-8")
    except OSError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="F1.2-A bounded ONNX export")
    parser.add_argument("--source", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    try:
        payload = run_export(Path(args.workdir), Path(args.source), args.precision)
    except Exception:
        payload = _failure(MODEL_EXPORT_FAILED)
    _write_result(Path(args.result), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
