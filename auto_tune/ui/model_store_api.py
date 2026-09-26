"""F1.1-A / F1.2-A controlled-weight HTTP surface.

Listing, uploading and the manual PT → ONNX export. Every answer is built from
``ModelRecord.public_dict()`` / ``ExportInfo.public_dict()``, so a server-side
path, a traceback or an exception text can never reach a client. Any filesystem
work — including the bounded export child process — is blocking, hence the
threadpool hop that keeps the rest of the UI responsive.
"""

from __future__ import annotations

from typing import Callable
from urllib.parse import urlencode

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from auto_tune.modules.model_store import ModelStore, ModelStoreError
from auto_tune.modules.model_store.onnx_export import (
    MODEL_EXPORT_FAILED,
    MODEL_EXPORT_INVALID_REQUEST,
    OnnxExportService,
    stable_message,
)

__all__ = ["create_model_store_router"]


def _download_url(model_id: str, precision: str) -> str:
    return "/api/models/export/download?" + urlencode(
        {"model_id": model_id, "precision": precision})


def create_model_store_router(*, store, exports: OnnxExportService,
                              require_security: Callable[[Request], object]) -> APIRouter:
    """Build the ``/api/models`` router.

    ``store`` and ``exports`` are the live objects, or zero-argument accessors
    returning them — the module globals stay the truth, so a test or a reload
    can rebind one without rebuilding the router. ``require_security(request)``
    returns ``None`` when the request may proceed and a ready ``JSONResponse``
    when it must not; the caller keeps ownership of the CSRF/origin policy.
    """
    router = APIRouter(prefix="/api/models", tags=["models"])

    def _store() -> ModelStore:
        return store() if callable(store) else store

    def _exports() -> OnnxExportService:
        return exports() if callable(exports) else exports

    def _rejected(request: Request):
        verdict = require_security(request)
        return verdict if isinstance(verdict, JSONResponse) else None

    def _failure(exc: ModelStoreError) -> JSONResponse:
        return JSONResponse({"error_code": exc.code, "error": exc.message},
                            status_code=exc.status_code)

    @router.get("")
    def list_models():
        try:
            rows = _store().list_models()
        except ModelStoreError as exc:
            return JSONResponse({"error_code": exc.code, "error": exc.message,
                                 "models": []}, status_code=exc.status_code)
        except OSError:
            # A listing failure is isolated: a stable code and an explicitly
            # empty list, so a client keeps its already-valid options instead
            # of silently showing a shortened library.
            return JSONResponse(
                {"error_code": "MODEL_STORE_UNAVAILABLE",
                 "error": "受控权重库暂时不可用，请稍后重试。", "models": []},
                status_code=503)
        return {"models": [row.public_dict() for row in rows]}

    @router.post("/upload", status_code=201)
    async def upload_model(request: Request, file: UploadFile = File(...)):
        rejection = _rejected(request)
        if rejection is not None:
            return rejection
        try:
            record = await run_in_threadpool(
                _store().import_stream, file.filename or "", file.file)
        except ModelStoreError as exc:
            return JSONResponse({"error_code": exc.code, "error": exc.message},
                                status_code=exc.status_code)
        status = "created" if record.created_now else "exists"
        return JSONResponse({"status": status, "model": record.public_dict()},
                            status_code=201 if record.created_now else 200)

    # ── 手动导出 ONNX（F1.2-A）──────────────────────────────────────
    #
    # 只接受受控权重库已经公开的 model_id 与精度，没有路径参数、没有目标目录。
    # 导出本身在受控子进程里运行，这里只做一次线程池跳转，让其余页面请求继续
    # 被处理；结果落盘后刷新页面仍可从磁盘查询与下载。

    @router.get("/export")
    def export_status(model_id: str = ""):
        try:
            state = _exports().status(model_id)
        except ModelStoreError as exc:
            return _failure(exc)
        for precision, info in state["exports"].items():
            if info is not None:
                info["download_url"] = _download_url(state["model_id"], precision)
        return state

    @router.post("/export", status_code=201)
    async def export_model(request: Request):
        rejection = _rejected(request)
        if rejection is not None:
            return rejection
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(
                {"error_code": MODEL_EXPORT_INVALID_REQUEST,
                 "error": "导出请求格式不正确。"}, status_code=400)
        try:
            # source_trusted 原样交给服务层判定：缺省即“未确认”，不会默认放行
            info = await run_in_threadpool(_exports().export, body.get("model_id"),
                                           body.get("precision"),
                                           source_trusted=body.get("source_trusted"))
        except ModelStoreError as exc:
            return _failure(exc)
        except Exception:
            # 未预期的失败同样只回稳定错误码：绝不把堆栈或服务器路径交给客户端
            return JSONResponse({"error_code": MODEL_EXPORT_FAILED,
                                 "error": stable_message(MODEL_EXPORT_FAILED)},
                                status_code=500)
        payload = info.public_dict()
        payload["download_url"] = _download_url(info.model_id, info.precision)
        return JSONResponse({"status": "exported", "export": payload},
                            status_code=201)

    @router.get("/export/download")
    def download_export(model_id: str = "", precision: str = ""):
        try:
            info = _exports().resolve_download(model_id, precision)
        except ModelStoreError as exc:
            return _failure(exc)
        # 只流式返回受控目录内、由已登记导出派生出的那一个文件
        return FileResponse(info.target, filename=info.name,
                            media_type="application/octet-stream")

    return router
