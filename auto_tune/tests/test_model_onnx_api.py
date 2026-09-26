"""F1.2-A Task 2: the internal export HTTP surface.

The routes are exercised through the real Studio application, so the CSRF/origin
gate, the routing table and the error projection are the shipping ones. Only the
store, the export service and its bounded child invocation are rebound to a
temporary controlled root.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.model_store import ModelStore
from auto_tune.modules.model_store.onnx_export import OnnxExportService
from auto_tune.ui import app as app_mod

_PAYLOAD = b"trusted-weights"
_ONNX_BYTES = b"onnx-bytes"
_ROOT = Path("models") / "weights"

# 用来把一个字段从请求体中彻底去掉（“缺失”与“传了 false”是两种拒绝）
_OMIT = object()


def _auth_headers(**extra):
    headers = {"X-CSRF-Token": app_mod._CSRF_TOKEN, "Origin": "http://testserver"}
    headers.update(extra)
    return headers


class StubRunner:
    def __init__(self, *, payload=_ONNX_BYTES):
        self.payload = payload
        self.jobs = []
        self.gate = None
        self.fail_with = None
        self.entered = threading.Event()

    def __call__(self, job):
        self.jobs.append(job)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail_with is not None:
            return self.fail_with
        (job.workdir / job.output_name).write_bytes(self.payload)
        return {"ok": True, "precision": job.precision, "name": job.output_name,
                "size_bytes": len(self.payload)}


class Api:
    def __init__(self, tmp_path, *, fp16_available=False, runner=None):
        self.tmp = tmp_path
        self.root = tmp_path / "weights"
        self.runner = runner or StubRunner()
        self.store = ModelStore(self.root, legacy_roots=[tmp_path],
                                max_upload_bytes=1024, max_models=50)
        self.service = OnnxExportService(self.store, runner=self.runner,
                                         fp16_available=fp16_available)
        self.client = TestClient(app_mod.app)
        self.other = TestClient(app_mod.app)

    def upload(self, name="yolov8n.pt", data=_PAYLOAD):
        return self.client.post(
            "/api/models/upload",
            files={"file": (name, data, "application/octet-stream")},
            headers=_auth_headers())

    def model_id(self, name="yolov8n.pt"):
        self.upload(name)
        rows = self.client.get("/api/models").json()["models"]
        return [row for row in rows if row["name"] == name][0]["model_id"]

    def export(self, model_id, precision="fp32", headers=None, **extra):
        # 默认带上操作人员对来源可信的显式确认；拒绝路径各自覆盖该字段。
        body = {"model_id": model_id, "precision": precision,
                "source_trusted": True}
        for key, value in extra.items():
            if value is _OMIT:
                body.pop(key, None)
            else:
                body[key] = value
        return self.client.post(
            "/api/models/export", json=body,
            headers=_auth_headers() if headers is None else headers)


@pytest.fixture
def api(tmp_path, monkeypatch):
    stack = Api(tmp_path)
    monkeypatch.setattr(app_mod, "_MODEL_STORE", stack.store)
    monkeypatch.setattr(app_mod, "_MODEL_EXPORT_SERVICE", stack.service)
    return stack


@pytest.fixture
def api_fp16(tmp_path, monkeypatch):
    stack = Api(tmp_path, fp16_available=True)
    monkeypatch.setattr(app_mod, "_MODEL_STORE", stack.store)
    monkeypatch.setattr(app_mod, "_MODEL_EXPORT_SERVICE", stack.service)
    return stack


# ── 导出 ───────────────────────────────────────────────────────────


def test_exporting_a_managed_weight_publishes_a_downloadable_file(api):
    model_id = api.model_id()

    response = api.export(model_id)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "exported"
    exported = body["export"]
    assert set(exported) == {"model_id", "precision", "name", "size_bytes",
                             "download_url"}
    assert exported["model_id"] == model_id
    assert exported["precision"] == "fp32"
    assert exported["name"] == "yolov8n.onnx"
    assert exported["size_bytes"] == len(_ONNX_BYTES)
    assert (api.root / "yolov8n.onnx").read_bytes() == _ONNX_BYTES
    # 原权重不受影响，响应不泄漏服务器路径或堆栈
    assert (api.root / "yolov8n.pt").read_bytes() == _PAYLOAD
    assert str(api.tmp) not in response.text
    assert str(api.root) not in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize("confirmation", [None, False, "true", 1, "", {}])
def test_export_requires_an_explicit_true_trust_confirmation(api, confirmation):
    """绕过页面直接 POST 也必须带上明确的可信来源确认。"""
    model_id = api.model_id()

    response = api.export(model_id, source_trusted=confirmation)

    assert response.status_code == 400
    assert response.json()["error_code"] == "MODEL_EXPORT_TRUST_REQUIRED"
    assert api.runner.jobs == []
    assert not (api.root / "yolov8n.onnx").exists()
    assert str(api.tmp) not in response.text


def test_export_without_the_trust_field_at_all_is_rejected(api):
    model_id = api.model_id()

    response = api.export(model_id, source_trusted=_OMIT)

    assert response.status_code == 400
    assert response.json()["error_code"] == "MODEL_EXPORT_TRUST_REQUIRED"
    assert api.runner.jobs == []
    assert not (api.root / "yolov8n.onnx").exists()


def test_a_preexisting_same_name_file_is_neither_reported_nor_downloaded(api):
    """Codex 复现的场景：任意字节的预存 probe.onnx 不得被当成成功导出。"""
    (api.root).mkdir(parents=True, exist_ok=True)
    model_id = api.model_id("probe.pt")
    (api.root / "probe.onnx").write_bytes(b"not-an-onnx")

    state = api.client.get("/api/models/export",
                           params={"model_id": model_id}).json()
    assert state["exports"] == {"fp32": None, "fp16": None}

    download = api.client.get("/api/models/export/download",
                              params={"model_id": model_id, "precision": "fp32"})
    assert download.status_code == 404
    assert download.json()["error_code"] == "MODEL_EXPORT_NOT_FOUND"
    assert download.content != b"not-an-onnx"

    # 预存目标依然冲突，且绝不会被覆盖
    conflict = api.export(model_id)
    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == "MODEL_EXPORT_CONFLICT"
    assert api.runner.jobs == []
    assert (api.root / "probe.onnx").read_bytes() == b"not-an-onnx"


def test_a_tampered_export_is_neither_reported_nor_downloaded(api):
    model_id = api.model_id()
    api.export(model_id)
    (api.root / "yolov8n.onnx").write_bytes(b"tampered-after-export")

    state = api.client.get("/api/models/export",
                           params={"model_id": model_id}).json()
    assert state["exports"]["fp32"] is None
    download = api.client.get("/api/models/export/download",
                              params={"model_id": model_id, "precision": "fp32"})
    assert download.status_code == 404
    assert download.content != b"tampered-after-export"


def test_a_validated_export_stays_downloadable_after_a_restart(api, monkeypatch):
    model_id = api.model_id()
    api.export(model_id)

    # 重启：只有磁盘事实，没有任何内存状态
    restarted = Api(api.tmp)
    restarted.store.list_models()
    monkeypatch.setattr(app_mod, "_MODEL_STORE", restarted.store)
    monkeypatch.setattr(app_mod, "_MODEL_EXPORT_SERVICE", restarted.service)

    state = restarted.client.get("/api/models/export",
                                 params={"model_id": model_id}).json()
    assert state["exports"]["fp32"]["name"] == "yolov8n.onnx"
    download = restarted.client.get(
        state["exports"]["fp32"]["download_url"])
    assert download.status_code == 200
    assert download.content == _ONNX_BYTES


def test_export_requires_csrf_and_a_same_origin_request(api):
    model_id = api.model_id()

    no_csrf = api.export(model_id, headers={"Origin": "http://testserver"})
    assert no_csrf.status_code == 403
    cross = api.export(model_id, headers={"X-CSRF-Token": app_mod._CSRF_TOKEN,
                                          "Origin": "http://evil.example"})
    assert cross.status_code == 403

    assert api.runner.jobs == []
    assert not (api.root / "yolov8n.onnx").exists()


def test_a_legacy_only_weight_cannot_be_exported(api):
    (api.tmp / "borrowed.pt").write_bytes(_PAYLOAD)
    legacy = [row for row in api.client.get("/api/models").json()["models"]
              if row["origin"] == "legacy"][0]

    response = api.export(legacy["model_id"])

    assert response.status_code == 422
    assert response.json()["error_code"] == "MODEL_EXPORT_UNSUPPORTED_ORIGIN"
    assert api.runner.jobs == []
    assert not (api.tmp / "borrowed.onnx").exists()
    assert str(api.tmp) not in response.text


@pytest.mark.parametrize("model_id", [
    "", "sha256:", "sha256:zz", "sha256:" + "0" * 64, None, 12345,
])
def test_a_fabricated_model_id_is_rejected(api, model_id):
    api.model_id()

    response = api.export(model_id)

    assert response.status_code in (404, 422)
    assert response.json()["error_code"] in ("MODEL_NOT_FOUND",
                                             "MODEL_EXPORT_UNSUPPORTED_ORIGIN")
    assert api.runner.jobs == []
    assert str(api.tmp) not in response.text


def test_a_weight_replaced_before_export_is_rejected(api):
    model_id = api.model_id()
    (api.root / "yolov8n.pt").write_bytes(b"swapped")

    response = api.export(model_id)

    assert response.status_code == 409
    assert response.json()["error_code"] == "MODEL_CHANGED"
    assert api.runner.jobs == []


def test_an_existing_export_is_a_conflict_and_is_not_overwritten(api):
    model_id = api.model_id()
    (api.root / "yolov8n.onnx").write_bytes(b"already-exported")

    response = api.export(model_id)

    assert response.status_code == 409
    assert response.json()["error_code"] == "MODEL_EXPORT_CONFLICT"
    assert (api.root / "yolov8n.onnx").read_bytes() == b"already-exported"
    assert api.runner.jobs == []


def test_a_duplicate_request_while_exporting_is_rejected(api):
    model_id = api.model_id()
    api.runner.gate = threading.Event()
    outcome = {}

    def first():
        outcome["response"] = api.export(model_id)

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert api.runner.entered.wait(10)
        duplicate = api.export(model_id)
        assert duplicate.status_code == 409
        assert duplicate.json()["error_code"] == "MODEL_EXPORT_BUSY"
    finally:
        api.runner.gate.set()
        thread.join(15)
    assert not thread.is_alive()
    assert outcome["response"].status_code == 201
    assert len(api.runner.jobs) == 1


def test_an_export_in_flight_does_not_block_other_requests(api):
    model_id = api.model_id()
    api.runner.gate = threading.Event()
    outcome = {}

    def first():
        outcome["response"] = api.export(model_id)

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert api.runner.entered.wait(10)
        started = time.monotonic()
        listing = api.other.get("/api/models")
        elapsed = time.monotonic() - started
        assert listing.status_code == 200
        assert listing.json()["models"], "导出进行中列表仍应可读"
        assert elapsed < 5
        assert outcome == {}, "此时导出尚未结束，说明其它请求未被阻塞"
    finally:
        api.runner.gate.set()
        thread.join(15)
    assert outcome["response"].status_code == 201


@pytest.mark.parametrize("precision", ["int8", "FP32", "", None, 32])
def test_an_unsupported_precision_is_rejected(api, precision):
    model_id = api.model_id()

    response = api.export(model_id, precision)

    assert response.status_code == 400
    assert response.json()["error_code"] == "MODEL_EXPORT_INVALID_PRECISION"
    assert api.runner.jobs == []


def test_fp16_cannot_be_exported_before_it_is_verified(api):
    model_id = api.model_id()

    response = api.export(model_id, "fp16")

    assert response.status_code == 400
    assert response.json()["error_code"] == "MODEL_EXPORT_INVALID_PRECISION"
    assert api.runner.jobs == []
    assert not (api.root / "yolov8n.fp16.onnx").exists()


def test_fp16_uses_its_own_file_once_verified(api_fp16):
    model_id = api_fp16.model_id()

    response = api_fp16.export(model_id, "fp16")

    assert response.status_code == 201
    assert response.json()["export"]["name"] == "yolov8n.fp16.onnx"
    assert (api_fp16.root / "yolov8n.fp16.onnx").exists()
    assert not (api_fp16.root / "yolov8n.onnx").exists()


def test_a_malformed_body_is_rejected_without_touching_anything(api):
    model_id = api.model_id()

    for body in (b"{not json", b"[1,2,3]", b'"text"'):
        response = api.client.post("/api/models/export", content=body,
                                   headers=_auth_headers())
        assert response.status_code == 400
        assert response.json()["error_code"]

    assert api.runner.jobs == []
    assert str(api.tmp) not in response.text


# ── 返修 3：应用装配不得让配置绕过半精度验收 ───────────────────────


def test_the_app_service_keeps_fp16_closed_even_if_the_config_enables_it(
        monkeypatch):
    """本批只验收 FP32：model_store.export.fp16=true 不得开放 UI/API。"""
    monkeypatch.setattr(app_mod, "APP_CONFIG",
                        {"model_store": {"export": {"fp16": True}}})

    service = app_mod._build_onnx_export_service()

    assert service.fp16_available is False


# ── 状态与下载 ─────────────────────────────────────────────────────


def test_export_status_is_read_only_and_survives_a_refresh(api):
    model_id = api.model_id()
    before = sorted(p.name for p in api.root.iterdir())

    empty = api.client.get("/api/models/export", params={"model_id": model_id})
    assert empty.status_code == 200
    assert empty.json() == {"model_id": model_id, "origin": "managed",
                            "exportable": True, "fp16_available": False,
                            "exports": {"fp32": None, "fp16": None}}
    assert sorted(p.name for p in api.root.iterdir()) == before

    api.export(model_id)
    state = api.client.get("/api/models/export", params={"model_id": model_id}).json()
    assert state["exports"]["fp32"]["name"] == "yolov8n.onnx"
    assert state["exports"]["fp32"]["size_bytes"] == len(_ONNX_BYTES)
    assert state["exports"]["fp32"]["download_url"].startswith(
        "/api/models/export/download?")
    assert state["exports"]["fp16"] is None


def test_export_status_reports_a_borrowed_weight_as_not_exportable(api):
    (api.tmp / "borrowed.pt").write_bytes(_PAYLOAD)
    legacy = [row for row in api.client.get("/api/models").json()["models"]
              if row["origin"] == "legacy"][0]

    response = api.client.get("/api/models/export",
                              params={"model_id": legacy["model_id"]})

    assert response.status_code == 200
    assert response.json()["exportable"] is False
    assert response.json()["exports"] == {"fp32": None, "fp16": None}


def test_export_status_rejects_an_unknown_model_id(api):
    api.model_id()

    response = api.client.get("/api/models/export",
                              params={"model_id": "sha256:" + "c" * 64})

    assert response.status_code == 404
    assert response.json()["error_code"] == "MODEL_NOT_FOUND"


def test_download_streams_exactly_the_committed_onnx(api):
    model_id = api.model_id()
    api.export(model_id)
    url = api.client.get("/api/models/export",
                         params={"model_id": model_id}).json()[
                             "exports"]["fp32"]["download_url"]

    response = api.client.get(url)

    assert response.status_code == 200
    assert response.content == _ONNX_BYTES
    assert "yolov8n.onnx" in response.headers.get("content-disposition", "")
    assert str(api.tmp) not in response.headers.get("content-disposition", "")


def test_download_of_a_missing_export_is_a_stable_404(api):
    model_id = api.model_id()

    response = api.client.get("/api/models/export/download",
                              params={"model_id": model_id, "precision": "fp32"})

    assert response.status_code == 404
    assert response.json()["error_code"] == "MODEL_EXPORT_NOT_FOUND"
    assert str(api.tmp) not in response.text


def test_download_never_serves_a_borrowed_weight(api):
    (api.tmp / "borrowed.pt").write_bytes(_PAYLOAD)
    (api.tmp / "borrowed.onnx").write_bytes(b"legacy-onnx")
    legacy = [row for row in api.client.get("/api/models").json()["models"]
              if row["origin"] == "legacy"][0]

    response = api.client.get("/api/models/export/download",
                              params={"model_id": legacy["model_id"],
                                      "precision": "fp32"})

    assert response.status_code == 422
    assert response.json()["error_code"] == "MODEL_EXPORT_UNSUPPORTED_ORIGIN"
    assert response.content != b"legacy-onnx"


def test_an_export_failure_is_projected_without_server_detail(api):
    model_id = api.model_id()
    api.runner.fail_with = {"ok": False, "error_code": "SOMETHING_NEW",
                            "message": str(api.tmp / "secret.pt")}

    response = api.export(model_id)

    assert response.status_code == 500
    assert response.json()["error_code"] == "MODEL_EXPORT_FAILED"
    assert str(api.tmp) not in response.text
    assert "secret" not in response.text
    assert not (api.root / "yolov8n.onnx").exists()


def test_an_unexpected_server_error_is_still_a_stable_projection(api):
    model_id = api.model_id()

    def boom(model_id, precision):
        raise RuntimeError("boom at " + str(api.tmp / "secret.pt"))

    api.service.export = boom
    response = api.export(model_id)

    assert response.status_code == 500
    assert set(response.json()) == {"error_code", "error"}
    assert response.json()["error_code"] == "MODEL_EXPORT_FAILED"
    assert str(api.tmp) not in response.text
    assert "boom" not in response.text
    assert "Traceback" not in response.text
    assert not (api.root / "yolov8n.onnx").exists()


def test_the_export_never_touches_anything_outside_the_weight_directory(api):
    model_id = api.model_id()
    before = sorted(p.name for p in api.tmp.iterdir())

    api.export(model_id)

    assert sorted(p.name for p in api.tmp.iterdir()) == before
    names = sorted(p.name for p in api.root.iterdir())
    assert "yolov8n.onnx" in names and "yolov8n.pt" in names
    # 受控目录里不留下临时工作目录或半成品
    assert [n for n in names
            if n.startswith(".export-") or n.endswith(".tmp")] == []
    job = api.runner.jobs[0]
    assert job.source.parent.parent == api.root
    assert job.source.parent.name.startswith(".export-")
    assert job.source.name == "yolov8n.pt"


def test_a_client_supplied_path_is_ignored_entirely(api):
    """导出只认 model_id；客户端多送的路径字段既不被采纳也不被回显。"""
    model_id = api.model_id()

    leaked = api.client.post("/api/models/export",
                             json={"model_id": model_id, "precision": "fp32",
                                   "source_trusted": True,
                                   "path": str(api.tmp), "output": "C:\\evil.onnx"},
                             headers=_auth_headers())

    assert leaked.status_code == 201
    assert str(api.tmp) not in leaked.text
    assert "evil" not in leaked.text
    assert api.runner.jobs[0].source.parent.parent == api.root


def test_the_api_process_never_deserialises_the_weight(api, monkeypatch):
    import pickle
    import sys

    def boom(*args, **kwargs):
        raise AssertionError("Studio 请求进程不得反序列化权重")

    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "load", boom, raising=False)

    model_id = api.model_id()
    assert api.export(model_id).status_code == 201
