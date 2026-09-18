"""F1.1-A Task 3: the controlled-model HTTP surface.

The routes are exercised through the *real* application, so the CSRF/origin
gate, the routing table and the error projection are the shipping ones. Only
the store instance is rebound to a temporary controlled root.
"""

import io
import os
import pickle
import socket
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.model_store import ModelStore, ModelStoreError
from auto_tune.ui import app as app_mod
from auto_tune.ui.model_store_api import create_model_store_router

_ROOT = Path("models") / "weights"
_PAYLOAD = b"weights"


def _auth_headers(**extra):
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN,
               "Origin": "http://testserver"}
    headers.update(extra)
    return headers


class Api:
    def __init__(self, tmp_path, *, max_upload_bytes=1024, max_models=50,
                 root=None):
        self.tmp = tmp_path
        self.root = Path(root) if root is not None else tmp_path / _ROOT
        self.store = ModelStore(self.root, legacy_roots=[tmp_path],
                                max_upload_bytes=max_upload_bytes,
                                max_models=max_models)

    def upload(self, client, name="custom.pt", data=_PAYLOAD, headers=None):
        return client.post(
            "/api/models/upload",
            files={"file": (name, data, "application/octet-stream")},
            headers=_auth_headers() if headers is None else headers,
        )


@pytest.fixture
def api(tmp_path, monkeypatch):
    stack = Api(tmp_path)
    monkeypatch.setattr(app_mod, "_MODEL_STORE", stack.store)
    stack.client = TestClient(app_mod.app)
    return stack


@pytest.fixture
def isolated_router(tmp_path):
    """The router factory on its own app: no global state, no CSRF token."""
    store = ModelStore(tmp_path / _ROOT, legacy_roots=[tmp_path],
                       max_upload_bytes=1024, max_models=50)
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_model_store_router(
        store=store, require_security=lambda request: None))
    return store, TestClient(app)


# ── 列表投影 ───────────────────────────────────────────────────────


def test_list_models_returns_only_the_safe_projection(api):
    assert api.upload(api.client).status_code == 201

    response = api.client.get("/api/models")
    assert response.status_code == 200
    rows = response.json()["models"]
    assert len(rows) == 1
    assert set(rows[0]) == {"model_id", "name", "size_bytes", "sha256",
                            "origin", "available"}
    assert rows[0]["name"] == "custom.pt"
    assert rows[0]["origin"] == "managed"
    assert rows[0]["model_id"].startswith("sha256:")
    # 绝不泄露服务器路径
    assert "path" not in response.text.lower()
    assert str(api.tmp).replace("\\", "\\\\") not in response.text
    assert str(api.root) not in response.text


def test_list_models_of_the_router_factory_projection(isolated_router):
    store, client = isolated_router
    assert client.get("/api/models").json() == {"models": []}
    store.import_stream("yolov8n.pt", io.BytesIO(_PAYLOAD))
    body = client.get("/api/models").json()
    assert body["models"][0]["name"] == "yolov8n.pt"


def test_list_models_is_read_only(api):
    api.upload(api.client)
    before = sorted(p.name for p in api.root.iterdir())
    api.client.get("/api/models")
    assert sorted(p.name for p in api.root.iterdir()) == before


# ── 上传 ───────────────────────────────────────────────────────────


def test_upload_persists_the_file_and_projects_it(api):
    response = api.upload(api.client)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "created"
    assert body["model"]["name"] == "custom.pt"
    assert body["model"]["origin"] == "managed"
    assert set(body["model"]) == {"model_id", "name", "size_bytes", "sha256",
                                  "origin", "available"}
    assert (api.root / "custom.pt").read_bytes() == _PAYLOAD
    assert str(api.tmp) not in response.text
    assert "Traceback" not in response.text


def test_re_uploading_the_same_content_reports_it_already_exists(api):
    first = api.upload(api.client)
    second = api.upload(api.client)

    assert first.status_code == 201 and second.status_code == 200
    assert second.json()["status"] == "exists"
    assert second.json()["model"]["model_id"] == first.json()["model"]["model_id"]
    assert sorted(p.name for p in api.root.iterdir()) == ["custom.pt"]


def test_upload_rejects_a_different_file_with_the_same_name(api):
    first = api.upload(api.client)
    conflict = api.upload(api.client, data=b"different")

    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "MODEL_NAME_CONFLICT"
    assert (api.root / "custom.pt").read_bytes() == _PAYLOAD
    assert sorted(p.name for p in api.root.iterdir()) == ["custom.pt"]
    assert first.json()["model"]["model_id"] in api.client.get(
        "/api/models").text


def test_upload_requires_csrf_and_same_origin(api):
    no_csrf = api.upload(api.client, headers={"Origin": "http://testserver"})
    assert no_csrf.status_code == 403
    cross = api.upload(api.client, headers={
        "X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://evil.example"})
    assert cross.status_code == 403
    assert not api.root.exists()


@pytest.mark.parametrize("name,code,status", [
    ("payload.bin", "MODEL_UPLOAD_INVALID_TYPE", 400),
    ("payload.ptx", "MODEL_UPLOAD_INVALID_TYPE", 400),
    ("../escape.pt", "MODEL_UPLOAD_INVALID_NAME", 400),
    ("sub/escape.pt", "MODEL_UPLOAD_INVALID_NAME", 400),
    ("..", "MODEL_UPLOAD_INVALID_NAME", 400),
    ("anything\\sub\\escape.pt", "MODEL_UPLOAD_INVALID_NAME", 400),
])
def test_upload_rejects_illegal_names_and_types(api, name, code, status):
    response = api.upload(api.client, name=name)
    assert response.status_code == status
    assert response.json()["error_code"] == code
    # 稳定文案，不含异常原文、堆栈或服务器路径
    assert str(api.tmp) not in response.text
    assert "Traceback" not in response.text
    assert not api.root.exists() or sorted(p.name for p in api.root.iterdir()) == []


def test_upload_without_a_file_field_is_rejected(api):
    response = api.client.post("/api/models/upload", headers=_auth_headers())
    assert response.status_code == 422
    assert not api.root.exists()


def test_upload_rejects_an_empty_file(api):
    response = api.upload(api.client, data=b"")
    assert response.status_code == 400
    assert response.json()["error_code"] == "MODEL_UPLOAD_EMPTY"
    assert not api.root.exists() or sorted(p.name for p in api.root.iterdir()) == []


def test_upload_rejects_an_oversized_file_without_residue(tmp_path, monkeypatch):
    stack = Api(tmp_path, max_upload_bytes=8)
    monkeypatch.setattr(app_mod, "_MODEL_STORE", stack.store)
    client = TestClient(app_mod.app)

    response = stack.upload(client, data=b"x" * 4096)
    assert response.status_code == 413
    assert response.json()["error_code"] == "MODEL_UPLOAD_TOO_LARGE"
    assert sorted(p.name for p in stack.root.iterdir()) == []


def test_upload_reports_an_unavailable_store(tmp_path, monkeypatch):
    blocked = tmp_path / "weights"
    blocked.write_bytes(b"not a directory")
    stack = Api(tmp_path, root=blocked)
    monkeypatch.setattr(app_mod, "_MODEL_STORE", stack.store)
    client = TestClient(app_mod.app)

    response = stack.upload(client)
    assert response.status_code == 503
    assert response.json()["error_code"] == "MODEL_STORE_UNAVAILABLE"
    assert str(tmp_path) not in response.text


# ── 与训练提交边界的隔离 ───────────────────────────────────────────


def test_upload_survives_a_listing_failure(api, monkeypatch):
    """列表失败不得清空已经合法保存的权重，也不得污染上传结果。"""
    uploaded = api.upload(api.client)
    assert uploaded.status_code == 201

    def boom():
        raise OSError("listing broke")

    original = api.store.list_models
    monkeypatch.setattr(api.store, "list_models", boom)
    failed = api.client.get("/api/models")
    assert failed.status_code == 503
    assert failed.json()["error_code"] == "MODEL_STORE_UNAVAILABLE"
    assert failed.json()["models"] == []
    assert str(api.tmp) not in failed.text

    # 失败不改变磁盘事实：恢复后同一权重仍然合法可列
    monkeypatch.setattr(api.store, "list_models", original)
    assert api.client.get("/api/models").json()["models"][0]["name"] == "custom.pt"


def test_upload_never_deserialises_or_reaches_the_network(api, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("weights are opaque bytes")

    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)
    # 只封掉“主动发起连接/请求”的入口；socket.socket 本身被事件循环使用
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "load", boom, raising=False)

    response = api.upload(api.client, data=b"\x80\x04cbuiltins\neval\n.")
    assert response.status_code == 201
    assert (api.root / "custom.pt").read_bytes() == b"\x80\x04cbuiltins\neval\n."


# ── 与既有上传安全接口的边界 ───────────────────────────────────────


def test_model_upload_is_a_separate_surface_from_dataset_uploads(api):
    """权重上传不启用已停用的数据集/训练分析遗留上传入口。"""
    assert api.client.post("/upload/dataset", files={
        "file": ("d.zip", b"x", "application/zip")}).status_code in (404, 405, 410)
    assert api.client.post("/upload", files={
        "file": ("d.zip", b"x", "application/zip")}).status_code in (404, 405, 410)


def test_resolve_is_reverified_at_the_training_boundary(api):
    """上传后被替换时，训练提交边界必须拒绝启动。"""
    record = api.store.import_stream("custom.pt", io.BytesIO(_PAYLOAD))
    assert api.store.resolve(record.model_id) == record.path

    record.path.write_bytes(b"replaced")
    with pytest.raises(ModelStoreError) as exc:
        api.store.resolve(record.model_id)
    assert exc.value.code == "MODEL_CHANGED"

    os.remove(record.path)
    with pytest.raises(ModelStoreError) as exc:
        api.store.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"
