"""F1.2-A Task 1: the ONNX export service and its file boundary.

The service is exercised through its real identity/conflict/single-flight rules;
only the bounded child invocation is replaced by a stub, so no test here needs a
real checkpoint or a GPU. The real-subprocess tests drive the shipping command
builder and the shipping child module only.
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from auto_tune.modules.model_store import ModelStore
from auto_tune.modules.model_store import onnx_export as oe
from auto_tune.modules.model_store import onnx_export_worker as worker

_PAYLOAD = b"trusted-weights"
_ONNX_BYTES = b"onnx-bytes"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _store(tmp_path):
    return ModelStore(tmp_path / "weights", legacy_roots=[tmp_path],
                      max_upload_bytes=1024, max_models=50)


def _managed(store, name="yolov8n.pt", data=_PAYLOAD):
    return store.import_stream(name, io.BytesIO(data))


class StubRunner:
    """Stand-in for the bounded child: writes what the real worker would write."""

    def __init__(self, *, payload=_ONNX_BYTES, result=None):
        self.payload = payload
        self.result = result
        self.jobs = []
        self.gate = None
        self.entered = threading.Event()

    def __call__(self, job):
        self.jobs.append(job)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.result is not None and self.result.get("ok") is False:
            return self.result
        (job.workdir / job.output_name).write_bytes(self.payload)
        return self.result or {"ok": True, "precision": job.precision,
                               "name": job.output_name,
                               "size_bytes": len(self.payload)}


def _service(store, runner=None, **kwargs):
    return oe.OnnxExportService(store, runner=runner or StubRunner(), **kwargs)


def _export(service, model_id, precision="fp32"):
    """一次来源可信确认齐全的正常导出；拒绝路径各自单独断言。"""
    return service.export(model_id, precision, source_trusted=True)


def _weights_dir(tmp_path):
    return tmp_path / "weights"


def _entries(path):
    return sorted(p.name for p in path.iterdir())


# ── 服务：受控来源与身份 ───────────────────────────────────────────


def test_managed_export_writes_the_onnx_beside_the_weight(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()

    info = _export(_service(store, runner), record.model_id)

    assert info.model_id == record.model_id
    assert info.precision == "fp32"
    assert info.name == "yolov8n.onnx"
    assert info.size_bytes == len(_ONNX_BYTES)
    assert (_weights_dir(tmp_path) / "yolov8n.onnx").read_bytes() == _ONNX_BYTES
    # 原权重字节不受影响；子进程只处理内部临时副本
    assert (_weights_dir(tmp_path) / "yolov8n.pt").read_bytes() == _PAYLOAD
    assert runner.jobs[0].source.parent == runner.jobs[0].workdir
    assert runner.jobs[0].source.name == "yolov8n.pt"


def test_a_weight_whose_content_is_also_borrowed_stays_managed(tmp_path):
    """同一内容的兼容副本不会把已上传权重降级成不可导出的来源。"""
    store = _store(tmp_path)
    record = _managed(store, name="uploaded.pt")
    (tmp_path / "borrowed-copy.pt").write_bytes(_PAYLOAD)
    rows = [row for row in store.list_models()
            if row.model_id == record.model_id]
    assert len(rows) == 1 and rows[0].origin == "managed"
    runner = StubRunner()

    info = _export(_service(store, runner), record.model_id)

    assert info.name == "uploaded.onnx"
    assert info.target.parent == _weights_dir(tmp_path)
    assert (_weights_dir(tmp_path) / "uploaded.onnx").read_bytes() == _ONNX_BYTES


def test_an_auto_named_upload_is_exported_under_its_actual_name(tmp_path):
    """同名不同内容上传后按 <stem>_<哈希前 12 位>.pt 落盘：导出与下载名跟随它。"""
    store = _store(tmp_path)
    _managed(store, name="best.pt")
    renamed = _managed(store, name="best.pt", data=b"other-weights")
    assert renamed.name.startswith("best_") and renamed.name != "best.pt"

    service = _service(store, StubRunner())
    info = _export(service, renamed.model_id)

    expected = renamed.name[:-len(".pt")] + ".onnx"
    assert info.name == expected
    assert (_weights_dir(tmp_path) / expected).read_bytes() == _ONNX_BYTES
    assert not (_weights_dir(tmp_path) / "best.onnx").exists()
    assert service.resolve_download(renamed.model_id, "fp32").name == expected
    assert (_weights_dir(tmp_path) / "best.pt").read_bytes() == _PAYLOAD


def test_legacy_only_weight_is_rejected(tmp_path):
    (tmp_path / "borrowed.pt").write_bytes(_PAYLOAD)
    store = _store(tmp_path)
    legacy = [row for row in store.list_models() if row.origin == "legacy"]
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), legacy[0].model_id)

    assert exc.value.code == "MODEL_EXPORT_UNSUPPORTED_ORIGIN"
    assert runner.jobs == []


@pytest.mark.parametrize("model_id", [
    "", "sha256:", "sha256:zz", "not-a-sha256:abcd", "sha256:" + "0" * 64,
    None, 17, "sha256:" + "a" * 63,
])
def test_fabricated_model_ids_are_rejected(tmp_path, model_id):
    store = _store(tmp_path)
    _managed(store)
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), model_id)

    assert exc.value.code in ("MODEL_NOT_FOUND", "MODEL_EXPORT_UNSUPPORTED_ORIGIN")
    assert runner.jobs == []


def test_a_weight_replaced_after_listing_is_rejected(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    (_weights_dir(tmp_path) / "yolov8n.pt").write_bytes(b"swapped")
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_CHANGED"
    assert runner.jobs == []


def test_a_source_swapped_between_resolve_and_copy_is_rejected(tmp_path, monkeypatch):
    """解析与复制之间的替换由复制阶段的哈希复核兜住。"""
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    real = store.resolve_managed

    def resolve_then_swap(model_id):
        row = real(model_id)
        (_weights_dir(tmp_path) / "yolov8n.pt").write_bytes(b"swapped-mid-flight")
        return row

    monkeypatch.setattr(store, "resolve_managed", resolve_then_swap)

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_CHANGED"
    assert runner.jobs == []
    assert not (_weights_dir(tmp_path) / "yolov8n.onnx").exists()


def test_a_source_changed_during_conversion_is_not_published(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    original = _weights_dir(tmp_path) / "yolov8n.pt"

    class SwappingRunner(StubRunner):
        def __call__(self, job):
            result = super().__call__(job)
            original.write_bytes(b"swapped-during-export")
            return result

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, SwappingRunner()), record.model_id)

    assert exc.value.code == "MODEL_CHANGED"
    assert not (_weights_dir(tmp_path) / "yolov8n.onnx").exists()
    assert original.read_bytes() == b"swapped-during-export"


# ── 服务：冲突、并发与精度 ─────────────────────────────────────────


def test_an_existing_target_is_a_conflict_and_is_never_overwritten(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    target = _weights_dir(tmp_path) / "yolov8n.onnx"
    target.write_bytes(b"already-exported")
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_CONFLICT"
    assert exc.value.status_code == 409
    assert target.read_bytes() == b"already-exported"
    assert runner.jobs == []


def test_a_concurrent_export_of_the_same_source_is_busy(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    runner.gate = threading.Event()
    service = _service(store, runner)
    outcome = {}

    def first():
        try:
            outcome["info"] = _export(service, record.model_id)
        except Exception as exc:  # pragma: no cover - 由下方断言暴露
            outcome["error"] = exc

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert runner.entered.wait(10)
        with pytest.raises(oe.OnnxExportError) as exc:
            _export(service, record.model_id)
        assert exc.value.code == "MODEL_EXPORT_BUSY"
        assert exc.value.status_code == 409
    finally:
        runner.gate.set()
        thread.join(10)
    assert not thread.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert len(runner.jobs) == 1
    assert (_weights_dir(tmp_path) / "yolov8n.onnx").exists()


def test_a_repeated_export_after_success_is_a_conflict(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    service = _service(store, runner)

    _export(service, record.model_id)
    with pytest.raises(oe.OnnxExportError) as exc:
        _export(service, record.model_id)

    assert exc.value.code == "MODEL_EXPORT_CONFLICT"
    assert len(runner.jobs) == 1


@pytest.mark.parametrize("precision", ["", "FP32", "int8", "float16", None, 32, "fp32 "])
def test_unsupported_precision_is_rejected(tmp_path, precision):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner, fp16_available=True), record.model_id,
                precision)

    assert exc.value.code == "MODEL_EXPORT_INVALID_PRECISION"
    assert exc.value.status_code == 400
    assert runner.jobs == []


def test_fp16_is_rejected_while_it_is_not_verified(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner, fp16_available=False), record.model_id,
                "fp16")

    assert exc.value.code == "MODEL_EXPORT_INVALID_PRECISION"
    assert runner.jobs == []
    assert not (_weights_dir(tmp_path) / "yolov8n.fp16.onnx").exists()


def test_fp16_export_uses_its_own_target_and_leaves_fp32_absent(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    service = _service(store, runner, fp16_available=True)

    info = _export(service, record.model_id, "fp16")

    assert info.name == "yolov8n.fp16.onnx"
    assert info.precision == "fp16"
    assert (_weights_dir(tmp_path) / "yolov8n.fp16.onnx").exists()
    assert not (_weights_dir(tmp_path) / "yolov8n.onnx").exists()
    assert runner.jobs[0].output_name == "yolov8n.fp16.onnx"
    assert runner.jobs[0].precision == "fp16"


def test_a_worker_failure_is_bounded_and_leaves_no_residue(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner(result={"ok": False, "error_code": "MODEL_EXPORT_FAILED",
                                "message": "/tmp/secret/path with traceback"})

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_FAILED"
    assert str(tmp_path) not in exc.value.message
    assert "secret" not in exc.value.message
    assert _entries(_weights_dir(tmp_path)) == ["yolov8n.pt"]


def test_a_worker_returning_an_unknown_code_is_a_stable_failure(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner(result={"ok": False, "error_code": "SOMETHING_NEW",
                                "message": "/tmp/secret/path"})

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_FAILED"
    assert "secret" not in exc.value.message
    assert "model_store" not in exc.value.message


def test_a_worker_that_writes_nothing_is_a_stable_failure(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)

    class Silent(StubRunner):
        def __call__(self, job):
            self.jobs.append(job)
            return {"ok": True, "name": job.output_name}

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, Silent()), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_INVALID_OUTPUT"
    assert _entries(_weights_dir(tmp_path)) == ["yolov8n.pt"]


def test_a_worker_naming_a_different_output_is_rejected(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)

    class WrongName(StubRunner):
        def __call__(self, job):
            self.jobs.append(job)
            (job.workdir / "elsewhere.onnx").write_bytes(_ONNX_BYTES)
            return {"ok": True, "name": "elsewhere.onnx"}

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, WrongName()), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_INVALID_OUTPUT"
    assert not (_weights_dir(tmp_path) / "elsewhere.onnx").exists()


def test_temporary_working_directories_are_always_removed(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    _export(_service(store, StubRunner()), record.model_id)

    assert not [p for p in _weights_dir(tmp_path).iterdir()
                if p.name.startswith(".export-")]

    second = tmp_path / "second"
    store2 = _store(second)
    record2 = _managed(store2, name="broken.pt")
    with pytest.raises(oe.OnnxExportError):
        _export(_service(store2, StubRunner(
            result={"ok": False, "error_code": "MODEL_EXPORT_FAILED",
                    "message": "导出失败。"})), record2.model_id)

    assert _entries(second / "weights") == ["broken.pt"]
    assert record2.path.read_bytes() == _PAYLOAD


# ── 服务：刷新后仍可查询与下载 ─────────────────────────────────────


def test_status_is_derived_from_disk_not_from_memory(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    _export(_service(store, StubRunner(payload=b"12345")), record.model_id)

    # 新实例（相当于刷新/重启）：没有导出内存状态，仍须从磁盘得到同一条导出
    fresh_store = _store(tmp_path)
    fresh_store.list_models()
    fresh = _service(fresh_store, StubRunner(), fp16_available=True)
    state = fresh.status(record.model_id)

    assert state["origin"] == "managed"
    assert state["exportable"] is True
    assert state["fp16_available"] is True
    assert state["exports"]["fp32"]["name"] == "yolov8n.onnx"
    assert state["exports"]["fp32"]["size_bytes"] == 5
    assert state["exports"]["fp32"]["precision"] == "fp32"
    assert state["exports"]["fp16"] is None


def test_status_reports_a_legacy_weight_as_not_exportable(tmp_path):
    (tmp_path / "borrowed.pt").write_bytes(_PAYLOAD)
    store = _store(tmp_path)
    legacy = [row for row in store.list_models() if row.origin == "legacy"][0]

    state = _service(store, StubRunner()).status(legacy.model_id)

    assert state["origin"] == "legacy"
    assert state["exportable"] is False
    assert state["exports"] == {"fp32": None, "fp16": None}


def test_status_rejects_an_unknown_model_id(tmp_path):
    store = _store(tmp_path)
    _managed(store)

    with pytest.raises(oe.OnnxExportError) as exc:
        _service(store, StubRunner()).status("sha256:" + "b" * 64)

    assert exc.value.code == "MODEL_NOT_FOUND"


def test_download_resolution_requires_the_managed_output_to_exist(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    service = _service(store, StubRunner())

    with pytest.raises(oe.OnnxExportError) as exc:
        service.resolve_download(record.model_id, "fp32")
    assert exc.value.code == "MODEL_EXPORT_NOT_FOUND"
    assert exc.value.status_code == 404

    info = _export(service, record.model_id)
    resolved = service.resolve_download(record.model_id, "fp32")
    assert resolved.target == info.target
    assert resolved.name == "yolov8n.onnx"
    assert resolved.target.read_bytes() == _ONNX_BYTES


def test_download_never_serves_a_legacy_source(tmp_path):
    (tmp_path / "borrowed.pt").write_bytes(_PAYLOAD)
    (tmp_path / "borrowed.onnx").write_bytes(b"legacy-onnx")
    store = _store(tmp_path)
    legacy = [row for row in store.list_models() if row.origin == "legacy"][0]

    with pytest.raises(oe.OnnxExportError) as exc:
        _service(store, StubRunner()).resolve_download(legacy.model_id, "fp32")

    assert exc.value.code == "MODEL_EXPORT_UNSUPPORTED_ORIGIN"


def test_no_error_message_leaks_a_server_path(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner(result={"ok": False, "error_code": "MODEL_EXPORT_FAILED",
                                "message": "导出失败。"})

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    text = exc.value.message + exc.value.code
    assert str(tmp_path) not in text
    assert "weights" not in text
    assert "\\" not in text and "/" not in text


# ── 返修 1：只有本服务校验并发布的产物才算“已导出” ────────────────────
#
# 同名文件本身不构成导出事实：预存文件、发布后被替换或损坏、发布事实缺失，
# 都不得显示成“已导出”，也不得提供下载。已有目标依然冲突，绝不覆盖。


def test_a_pre_existing_same_name_file_is_not_a_published_export(tmp_path):
    store = _store(tmp_path)
    record = _managed(store, name="probe.pt")
    (_weights_dir(tmp_path) / "probe.onnx").write_bytes(b"not-an-onnx")
    service = _service(store, StubRunner())

    assert service.status(record.model_id)["exports"] == {"fp32": None,
                                                           "fp16": None}
    with pytest.raises(oe.OnnxExportError) as exc:
        service.resolve_download(record.model_id, "fp32")
    assert exc.value.code == "MODEL_EXPORT_NOT_FOUND"
    assert exc.value.status_code == 404


def test_a_pre_existing_target_is_a_conflict_and_is_never_replaced(tmp_path):
    store = _store(tmp_path)
    record = _managed(store, name="probe.pt")
    target = _weights_dir(tmp_path) / "probe.onnx"
    target.write_bytes(b"not-an-onnx")
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, runner), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_CONFLICT"
    assert runner.jobs == []
    assert target.read_bytes() == b"not-an-onnx"


def test_an_export_replaced_after_publication_is_not_downloadable(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    service = _service(store, StubRunner())
    _export(service, record.model_id)
    assert service.resolve_download(record.model_id, "fp32").size_bytes == \
        len(_ONNX_BYTES)

    (_weights_dir(tmp_path) / "yolov8n.onnx").write_bytes(b"swapped-after-export")

    assert service.status(record.model_id)["exports"]["fp32"] is None
    with pytest.raises(oe.OnnxExportError) as exc:
        service.resolve_download(record.model_id, "fp32")
    assert exc.value.code == "MODEL_EXPORT_NOT_FOUND"


def test_an_export_corrupted_to_the_same_size_is_detected(tmp_path):
    """大小相同、内容不同：必须靠内容校验识别，不能只看文件大小。"""
    store = _store(tmp_path)
    record = _managed(store)
    service = _service(store, StubRunner(payload=b"0123456789"))
    _export(service, record.model_id)

    (_weights_dir(tmp_path) / "yolov8n.onnx").write_bytes(b"abcdefghij")

    assert service.status(record.model_id)["exports"]["fp32"] is None


def test_a_deleted_export_is_not_reported_and_not_downloadable(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    service = _service(store, StubRunner())
    _export(service, record.model_id)

    (_weights_dir(tmp_path) / "yolov8n.onnx").unlink()

    assert service.status(record.model_id)["exports"]["fp32"] is None
    with pytest.raises(oe.OnnxExportError) as exc:
        service.resolve_download(record.model_id, "fp32")
    assert exc.value.code == "MODEL_EXPORT_NOT_FOUND"


def test_a_validated_export_is_still_reachable_after_a_restart(tmp_path):
    """重启后只有磁盘事实：本服务发布过的产物仍须可查询、可下载。"""
    store = _store(tmp_path)
    record = _managed(store)
    _export(_service(store, StubRunner(payload=b"12345")), record.model_id)

    restarted_store = _store(tmp_path)
    restarted_store.list_models()
    restarted = _service(restarted_store, StubRunner())

    state = restarted.status(record.model_id)
    assert state["exports"]["fp32"]["name"] == "yolov8n.onnx"
    assert state["exports"]["fp32"]["size_bytes"] == 5
    assert restarted.resolve_download(record.model_id, "fp32").target.read_bytes() \
        == b"12345"


def test_an_export_does_not_follow_a_replaced_weight(tmp_path):
    """同名权重被换成不同内容后是另一个身份：旧导出不得被算作它的产物。"""
    store = _store(tmp_path)
    record = _managed(store)
    _export(_service(store, StubRunner()), record.model_id)

    (_weights_dir(tmp_path) / "yolov8n.pt").write_bytes(b"swapped-weight")
    fresh = _store(tmp_path)
    replaced = [row for row in fresh.list_models()
                if row.name == "yolov8n.pt"][0]
    assert replaced.model_id != record.model_id

    state = _service(fresh, StubRunner()).status(replaced.model_id)
    assert state["exports"] == {"fp32": None, "fp16": None}


def test_an_in_flight_export_is_never_reported_as_finished(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    runner.gate = threading.Event()
    service = _service(store, runner)
    thread = threading.Thread(target=_export, args=(service, record.model_id))
    thread.start()
    try:
        assert runner.entered.wait(10)
        # 转换期间的半成品既不在磁盘上，也不被当成已完成
        assert service.status(record.model_id)["exports"]["fp32"] is None
        with pytest.raises(oe.OnnxExportError):
            service.resolve_download(record.model_id, "fp32")
    finally:
        runner.gate.set()
        thread.join(10)
    assert not thread.is_alive()
    assert service.status(record.model_id)["exports"]["fp32"] is not None


def test_a_publication_that_cannot_record_its_fact_leaves_no_output(tmp_path):
    """校验事实写不下去时，绝不留下一个看起来已经完成的产物。"""
    store = _store(tmp_path)
    record = _managed(store)
    # 占用校验事实的路径：目录无法被原子替换覆盖，发布必须失败并回收产物
    (_weights_dir(tmp_path) / "yolov8n.onnx.export.json").mkdir()

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(_service(store, StubRunner()), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_FAILED"
    assert not (_weights_dir(tmp_path) / "yolov8n.onnx").exists()
    assert [p.name for p in _weights_dir(tmp_path).iterdir()
            if p.name.endswith(".tmp")] == []


# ── 返修 2：可信来源确认必须显式随请求提交 ─────────────────────────


@pytest.mark.parametrize("confirmation", [None, False, 0, 1, "", "true", "false",
                                          [], {}])
def test_an_export_without_an_explicit_true_confirmation_never_runs(
        tmp_path, confirmation):
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()
    service = _service(store, runner)

    with pytest.raises(oe.OnnxExportError) as exc:
        service.export(record.model_id, "fp32", source_trusted=confirmation)

    assert exc.value.code == "MODEL_EXPORT_TRUST_REQUIRED"
    assert exc.value.status_code == 400
    assert runner.jobs == []
    assert not (_weights_dir(tmp_path) / "yolov8n.onnx").exists()


def test_omitting_the_confirmation_argument_is_a_rejection_not_a_default_yes(
        tmp_path):
    """默认值必须是“未确认”：调用方漏传就等于没有确认。"""
    store = _store(tmp_path)
    record = _managed(store)
    runner = StubRunner()

    with pytest.raises(oe.OnnxExportError) as exc:
        _service(store, runner).export(record.model_id, "fp32")

    assert exc.value.code == "MODEL_EXPORT_TRUST_REQUIRED"
    assert runner.jobs == []


def test_the_confirmation_never_leaks_a_server_path(tmp_path):
    store = _store(tmp_path)
    record = _managed(store)

    with pytest.raises(oe.OnnxExportError) as exc:
        _service(store, StubRunner()).export(record.model_id, "fp32",
                                             source_trusted=False)

    assert str(tmp_path) not in exc.value.message
    assert "/" not in exc.value.message


# ── 子进程调用边界 ─────────────────────────────────────────────────


def _job(tmp_path, precision="fp32", **overrides):
    workdir = tmp_path / ".export-abc"
    workdir.mkdir(parents=True, exist_ok=True)
    values = dict(
        python_executable=sys.executable,
        source=workdir / "yolov8n.pt",
        workdir=workdir,
        output_name="yolov8n.onnx" if precision == "fp32" else "yolov8n.fp16.onnx",
        precision=precision,
        result_path=workdir / "result.json",
        timeout_seconds=15,
    )
    values.update(overrides)
    return oe.ExportJob(**values)


def test_the_child_command_is_built_from_internal_paths_only(tmp_path):
    job = _job(tmp_path)

    command = oe.build_worker_command(job)

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "auto_tune.modules.model_store.onnx_export_worker"]
    assert str(job.source) in command
    assert "--precision" in command and "fp32" in command
    # 客户端可控的导出参数与目标名都不会出现在命令里：固定参数由子进程自身持有，
    # 目标名只是父进程对产物的核对条件
    joined = " ".join(command)
    assert "imgsz" not in joined and "opset" not in joined
    assert job.output_name not in joined


def test_the_child_environment_disables_automatic_installs():
    env = oe.build_worker_env()

    assert env["YOLO_AUTOINSTALL"] == "false"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(_PROJECT_ROOT)


def test_a_child_process_failure_is_reported_without_the_traceback(tmp_path):
    with pytest.raises(oe.OnnxExportError) as exc:
        oe.run_worker_process([sys.executable, "-c", "raise SystemExit(3)"],
                              cwd=str(tmp_path), timeout_seconds=60,
                              result_path=tmp_path / "result.json")

    assert exc.value.code == "MODEL_EXPORT_FAILED"
    assert str(tmp_path) not in exc.value.message
    assert "Traceback" not in exc.value.message


def test_a_child_process_without_a_result_file_is_a_stable_failure(tmp_path):
    with pytest.raises(oe.OnnxExportError) as exc:
        oe.run_worker_process([sys.executable, "-c", "pass"],
                              cwd=str(tmp_path), timeout_seconds=60,
                              result_path=tmp_path / "result.json")

    assert exc.value.code == "MODEL_EXPORT_FAILED"


def test_a_hanging_child_is_killed_at_the_timeout(tmp_path):
    started = time.monotonic()

    with pytest.raises(oe.OnnxExportError) as exc:
        oe.run_worker_process(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            cwd=str(tmp_path), timeout_seconds=2,
            result_path=tmp_path / "result.json")

    assert exc.value.code == "MODEL_EXPORT_TIMEOUT"
    assert exc.value.status_code == 504
    assert time.monotonic() - started < 60


def test_a_child_result_file_is_parsed(tmp_path):
    result_path = tmp_path / "result.json"
    payload = {"ok": True, "precision": "fp32", "name": "yolov8n.onnx"}
    command = [sys.executable, "-c",
               "import json,sys;open(sys.argv[1],'w').write(json.dumps(%r))" % payload,
               str(result_path)]

    assert oe.run_worker_process(command, cwd=str(tmp_path), timeout_seconds=60,
                                 result_path=result_path) == payload


def test_a_corrupt_child_result_is_a_stable_failure(tmp_path):
    result_path = tmp_path / "result.json"

    with pytest.raises(oe.OnnxExportError) as exc:
        oe.run_worker_process(
            [sys.executable, "-c",
             "import sys;open(sys.argv[1],'w').write('{not json')",
             str(result_path)],
            cwd=str(tmp_path), timeout_seconds=60, result_path=result_path)

    assert exc.value.code == "MODEL_EXPORT_FAILED"


# ── 子进程入口（worker）──────────────────────────────────────────────


def _worker_inputs(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "yolov8n.pt"
    source.write_bytes(_PAYLOAD)
    return workdir, source


def test_the_worker_always_uses_the_frozen_export_arguments(tmp_path):
    workdir, source = _worker_inputs(tmp_path)
    calls = []

    def fake_export(target, precision, **kwargs):
        calls.append((target, precision, kwargs))
        return _produced(target.parent, "yolov8n.onnx")

    result = worker.run_export(workdir, source, "fp32", exporter=fake_export,
                               inspector=lambda path: {"float32"})

    assert result["ok"] is True
    assert calls[0][0] == source
    assert (calls[0][1], calls[0][2]) == ("fp32", {
        "imgsz": 640, "opset": 17, "dynamic": False, "simplify": False,
        "nms": False, "half": False})
    assert result["name"] == "yolov8n.onnx"
    assert result["size_bytes"] == len(_ONNX_BYTES)


def test_the_worker_requests_half_precision_only_for_fp16(tmp_path):
    workdir, source = _worker_inputs(tmp_path)
    calls = []

    def fake_export(target, precision, **kwargs):
        calls.append(kwargs)
        return _produced(target.parent, "yolov8n.fp16.onnx")

    result = worker.run_export(workdir, source, "fp16", exporter=fake_export,
                               inspector=lambda path: {"float16"})

    assert calls[0]["half"] is True
    assert result == {"ok": True, "precision": "fp16",
                      "name": "yolov8n.fp16.onnx",
                      "size_bytes": len(_ONNX_BYTES),
                      "weights_dtype": "float16"}


def test_a_failing_exporter_yields_a_stable_code_without_the_exception(tmp_path):
    workdir, source = _worker_inputs(tmp_path)

    def broken(target, precision, **kwargs):
        raise RuntimeError("boom at " + str(tmp_path / "secret.pt"))

    result = worker.run_export(workdir, source, "fp32", exporter=broken,
                               inspector=lambda path: {"float32"})

    assert result["ok"] is False
    assert result["error_code"] == "MODEL_EXPORT_FAILED"
    assert "boom" not in result["message"]
    assert str(tmp_path) not in result["message"]


def test_an_export_that_produces_no_file_is_rejected(tmp_path):
    workdir, source = _worker_inputs(tmp_path)

    result = worker.run_export(workdir, source, "fp32",
                               exporter=lambda target, precision, **kwargs:
                               workdir / "missing.onnx",
                               inspector=lambda path: {"float32"})

    assert (result["ok"], result["error_code"]) == (False,
                                                    "MODEL_EXPORT_INVALID_OUTPUT")


def test_an_export_that_fails_the_onnx_check_is_rejected(tmp_path):
    workdir, source = _worker_inputs(tmp_path)
    calls = []

    def broken_inspector(path):
        calls.append(path)
        raise ValueError("checker said no")

    result = worker.run_export(workdir, source, "fp32",
                               exporter=lambda target, precision, **kwargs:
                               _produced(target.parent, "yolov8n.onnx"),
                               inspector=broken_inspector)

    assert calls == [workdir / "yolov8n.onnx"]
    assert (result["ok"], result["error_code"]) == (False,
                                                    "MODEL_EXPORT_INVALID_OUTPUT")
    assert "checker" not in result["message"]


def test_fp16_that_silently_produced_fp32_weights_is_rejected(tmp_path):
    """上游只给 warning 时绝不能当成成功。"""
    workdir, source = _worker_inputs(tmp_path)

    result = worker.run_export(workdir, source, "fp16",
                               exporter=lambda target, precision, **kwargs:
                               _produced(target.parent, "yolov8n.fp16.onnx"),
                               inspector=lambda path: {"float32"})

    assert (result["ok"], result["error_code"]) == (
        False, "MODEL_EXPORT_PRECISION_UNSUPPORTED")
    assert result["message"]


def test_fp16_that_fails_the_structural_check_reports_the_precision(tmp_path):
    """半精度文件未通过校验时，原因必须是“本环境产不出半精度”，不是泛化失败。"""
    workdir, source = _worker_inputs(tmp_path)

    def broken_inspector(path):
        raise ValueError("graph is not topologically sorted")

    result = worker.run_export(workdir, source, "fp16",
                               exporter=lambda target, precision, **kwargs:
                               _produced(target.parent, "yolov8n.fp16.onnx"),
                               inspector=broken_inspector)

    assert (result["ok"], result["error_code"]) == (
        False, "MODEL_EXPORT_PRECISION_UNSUPPORTED")
    assert "topologically" not in result["message"]


def test_fp32_never_ships_a_half_precision_file(tmp_path):
    workdir, source = _worker_inputs(tmp_path)

    result = worker.run_export(workdir, source, "fp32",
                               exporter=lambda target, precision, **kwargs:
                               _produced(target.parent, "yolov8n.onnx"),
                               inspector=lambda path: {"float16"})

    assert (result["ok"], result["error_code"]) == (False,
                                                    "MODEL_EXPORT_INVALID_OUTPUT")


@pytest.mark.parametrize("precision", ["int8", "", None, "FP32"])
def test_the_worker_rejects_an_unknown_precision(tmp_path, precision):
    workdir, source = _worker_inputs(tmp_path)

    result = worker.run_export(workdir, source, precision,
                               exporter=lambda *a, **k: _produced(workdir),
                               inspector=lambda path: {"float32"})

    assert (result["ok"], result["error_code"]) == (
        False, "MODEL_EXPORT_INVALID_PRECISION")


def test_the_worker_rejects_a_source_missing_from_its_working_directory(tmp_path):
    workdir, _ = _worker_inputs(tmp_path)

    result = worker.run_export(workdir, workdir / "gone.pt", "fp32",
                               exporter=lambda *a, **k: _produced(workdir),
                               inspector=lambda path: {"float32"})

    assert (result["ok"], result["error_code"]) == (False, "MODEL_EXPORT_FAILED")


def test_the_worker_main_writes_failures_as_data_and_exits_zero(tmp_path):
    workdir, _ = _worker_inputs(tmp_path)
    result_path = workdir / "result.json"

    code = worker.main([
        "--source", str(workdir / "gone.pt"), "--workdir", str(workdir),
        "--precision", "fp32", "--result", str(result_path)])

    assert code == 0
    body = json.loads(result_path.read_text(encoding="utf-8"))
    assert body["ok"] is False
    assert body["error_code"] == "MODEL_EXPORT_FAILED"
    assert str(tmp_path) not in body["message"]


def test_the_worker_main_rejects_a_source_outside_its_working_directory(tmp_path):
    workdir, _ = _worker_inputs(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(_PAYLOAD)
    result_path = workdir / "result.json"

    code = worker.main([
        "--source", str(outside), "--workdir", str(workdir),
        "--precision", "fp32", "--result", str(result_path)])

    assert code == 0
    body = json.loads(result_path.read_text(encoding="utf-8"))
    assert (body["ok"], body["error_code"]) == (False, "MODEL_EXPORT_FAILED")


# ── 真实子进程冒烟（真实 worker 模块 + 伪造权重）─────────────────────


def test_importing_the_child_module_does_not_import_ultralytics():
    """子进程入口必须保持轻量：只有真正加载权重时才引入 Ultralytics。"""
    probe = (
        "import sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "import auto_tune.modules.model_store.onnx_export_worker;"
        "print('ultralytics' in sys.modules)"
    )
    outcome = subprocess.run([sys.executable, "-c", probe, str(_PROJECT_ROOT)],
                             capture_output=True, text=True, timeout=300)

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.stdout.strip().endswith("False")


def test_the_real_worker_module_fails_safely_on_a_broken_checkpoint(tmp_path):
    store = _store(tmp_path)
    record = _managed(store, name="not-a-checkpoint.pt", data=b"not a checkpoint")

    with pytest.raises(oe.OnnxExportError) as exc:
        _export(oe.OnnxExportService(store, timeout_seconds=600), record.model_id)

    assert exc.value.code == "MODEL_EXPORT_FAILED"
    assert str(tmp_path) not in exc.value.message
    assert "Traceback" not in exc.value.message
    assert _entries(_weights_dir(tmp_path)) == ["not-a-checkpoint.pt"]


def _produced(workdir, name):
    path = Path(workdir) / name
    path.write_bytes(_ONNX_BYTES)
    return path
