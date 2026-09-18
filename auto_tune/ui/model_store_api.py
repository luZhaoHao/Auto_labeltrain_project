"""F1.1-A controlled-weight HTTP surface.

Two routes only: list the safe projection and import one operator-supplied
``.pt``. Both answers are built from ``ModelRecord.public_dict()``, so a
server-side path, a traceback or an exception text can never reach a client.
Any filesystem work is bounded and blocking, hence the threadpool hop.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from auto_tune.modules.model_store import ModelStore, ModelStoreError

__all__ = ["create_model_store_router"]


def create_model_store_router(*, store, require_security: Callable[[Request], object]) -> APIRouter:
    """Build the ``/api/models`` router.

    ``store`` is the live store, or a zero-argument accessor returning it — the
    module global stays the truth, so a test or a reload can rebind it without
    rebuilding the router. ``require_security(request)`` returns ``None`` when
    the request may proceed and a ready ``JSONResponse`` when it must not; the
    caller keeps ownership of the CSRF/origin policy.
    """
    router = APIRouter(prefix="/api/models", tags=["models"])

    def _store() -> ModelStore:
        return store() if callable(store) else store

    def _rejected(request: Request):
        verdict = require_security(request)
        return verdict if isinstance(verdict, JSONResponse) else None

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

    return router
