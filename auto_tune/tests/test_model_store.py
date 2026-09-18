"""F1.1-A Task 2: the controlled local weight store.

The store is the single place that lists, imports, resolves and re-verifies the
``.pt`` files a training may start from. It never deserialises a weight, never
touches the network and never scans training artifacts — a ``.pt`` is opaque
bytes plus a file identity (name, size, SHA-256), and a client only ever sees a
``model_id``.
"""

import hashlib
import io
import os
import pickle
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from auto_tune.modules.model_store import ModelStore, ModelStoreError
from auto_tune.modules.model_store import service as service_mod

_PAYLOAD = b"weights"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _store(tmp_path, **overrides):
    kwargs = {
        "root": tmp_path / "models" / "weights",
        "legacy_roots": [tmp_path],
        "max_upload_bytes": 1024,
        "max_models": 100,
    }
    kwargs.update(overrides)
    return ModelStore(**kwargs)


def _import(store, name="yolov8n.pt", data=_PAYLOAD):
    return store.import_stream(name, io.BytesIO(data))


def _names(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir())


def _make_dir_link(target: Path, link: Path) -> None:
    """Create a directory symlink/junction, or skip when the host forbids it."""
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            return
    pytest.skip("cannot create a directory link on this host")


# ── 公开接口与安全投影 ─────────────────────────────────────────────


def test_import_stream_saves_and_projects_a_managed_record(tmp_path):
    store = _store(tmp_path)
    record = _import(store)

    assert record.model_id == "sha256:" + _sha(_PAYLOAD)
    assert store.resolve(record.model_id) == record.path
    assert record.path.read_bytes() == _PAYLOAD
    assert record.path.parent == tmp_path / "models" / "weights"
    # 精确投影：只给界面需要的安全字段，绝不带服务器路径
    assert record.public_dict() == {
        "model_id": record.model_id,
        "name": "yolov8n.pt",
        "size_bytes": 7,
        "sha256": _sha(_PAYLOAD),
        "origin": "managed",
        "available": True,
    }
    assert "path" not in record.public_dict()
    assert str(tmp_path) not in repr(record.public_dict())


def test_stable_error_codes_are_exposed():
    for code in ("MODEL_UPLOAD_INVALID_NAME", "MODEL_UPLOAD_INVALID_TYPE",
                 "MODEL_UPLOAD_EMPTY", "MODEL_UPLOAD_TOO_LARGE",
                 "MODEL_NAME_CONFLICT", "MODEL_STORE_UNAVAILABLE",
                 "MODEL_NOT_FOUND", "MODEL_CHANGED", "MODEL_PATH_UNSAFE",
                 "MODEL_PATH_FORBIDDEN"):
        exc = ModelStoreError(code, "x")
        assert exc.code == code and exc.status_code >= 400


# ── 幂等与冲突 ─────────────────────────────────────────────────────


def test_same_name_same_hash_is_idempotent(tmp_path):
    store = _store(tmp_path)
    first = _import(store)
    second = _import(store)

    assert second.model_id == first.model_id
    assert second.public_dict() == first.public_dict()
    assert _names(tmp_path / "models" / "weights") == ["yolov8n.pt"]


def test_same_name_different_content_is_a_stable_conflict(tmp_path):
    store = _store(tmp_path)
    first = _import(store)

    with pytest.raises(ModelStoreError) as exc:
        _import(store, data=b"different")
    assert exc.value.code == "MODEL_NAME_CONFLICT"
    assert exc.value.status_code == 409

    # 绝不静默覆盖：原文件原样保留，也没有残留临时文件
    root = tmp_path / "models" / "weights"
    assert _names(root) == ["yolov8n.pt"]
    assert (root / "yolov8n.pt").read_bytes() == first.path.read_bytes()


def test_concurrent_same_name_uploads_with_the_same_content_are_idempotent(tmp_path):
    store = _store(tmp_path)
    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker():
        barrier.wait()
        try:
            results.append(_import(store))
        except Exception as exc:      # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len({r.model_id for r in results}) == 1
    assert _names(tmp_path / "models" / "weights") == ["yolov8n.pt"]


def test_concurrent_same_name_uploads_with_different_content_never_overwrite(tmp_path):
    store = _store(tmp_path)
    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker(data):
        barrier.wait()
        try:
            results.append(_import(store, data=data))
        except ModelStoreError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(data,))
               for data in (b"alpha", b"beta")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 恰好一个赢家，另一个得到稳定冲突码；磁盘上只有赢家的内容
    assert len(results) == 1
    assert [e.code for e in errors] == ["MODEL_NAME_CONFLICT"]
    root = tmp_path / "models" / "weights"
    assert _names(root) == ["yolov8n.pt"]
    assert (root / "yolov8n.pt").read_bytes() == results[0].path.read_bytes()


# ── 上传校验 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("filename", [
    "", ".", "..", "..\\escape.pt", "../escape.pt", "/abs/escape.pt",
    "C:\\abs\\escape.pt", "dir/evil.pt", "dir\\evil.pt", "bad\x00name.pt",
    "bad\nname.pt", " bad.pt ",
])
def test_illegal_filenames_are_rejected(tmp_path, filename):
    store = _store(tmp_path)
    with pytest.raises(ModelStoreError) as exc:
        _import(store, name=filename)
    assert exc.value.code == "MODEL_UPLOAD_INVALID_NAME"
    assert exc.value.status_code == 400
    assert not (tmp_path / "models" / "weights").exists()


@pytest.mark.parametrize("filename", ["yolov8n.ptx", "yolov8n.bin", "yolov8n",
                                      "yolov8n.PT.bak"])
def test_non_pt_extensions_are_rejected(tmp_path, filename):
    store = _store(tmp_path)
    with pytest.raises(ModelStoreError) as exc:
        _import(store, name=filename)
    assert exc.value.code == "MODEL_UPLOAD_INVALID_TYPE"
    assert not (tmp_path / "models" / "weights").exists()


def test_empty_upload_is_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ModelStoreError) as exc:
        _import(store, data=b"")
    assert exc.value.code == "MODEL_UPLOAD_EMPTY"
    root = tmp_path / "models" / "weights"
    assert _names(root) == [] if root.exists() else True


def test_oversized_upload_is_rejected_and_leaves_nothing_behind(tmp_path):
    store = _store(tmp_path, max_upload_bytes=8)
    with pytest.raises(ModelStoreError) as exc:
        _import(store, data=b"x" * 64)
    assert exc.value.code == "MODEL_UPLOAD_TOO_LARGE"
    assert exc.value.status_code == 413
    root = tmp_path / "models" / "weights"
    # 超限立即停止读取并删除临时文件：没有半文件、没有临时文件
    assert _names(root) == [] if root.exists() else True
    assert list(root.glob("*.tmp")) == [] if root.exists() else True


def test_write_failure_leaves_no_partial_or_temporary_file(tmp_path, monkeypatch):
    store = _store(tmp_path)
    root = tmp_path / "models" / "weights"

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service_mod.os, "fsync", boom)
    with pytest.raises(ModelStoreError) as exc:
        _import(store)
    assert exc.value.code == "MODEL_STORE_UNAVAILABLE"
    assert exc.value.status_code == 503
    assert _names(root) == [] if root.exists() else True
    assert list(root.glob("*.tmp")) == [] if root.exists() else True


def test_link_replacement_of_the_target_is_rejected(tmp_path):
    store = _store(tmp_path)
    root = tmp_path / "models" / "weights"
    root.mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    outside = tmp_path / "elsewhere" / "outside.pt"
    outside.write_bytes(_PAYLOAD)
    try:
        os.symlink(str(outside), str(root / "linked.pt"))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this host")

    assert store.list_models() == []
    with pytest.raises(ModelStoreError) as exc:
        _import(store, name="linked.pt", data=b"other")
    assert exc.value.code == "MODEL_PATH_UNSAFE"
    # 链接本身没有被跟随，也没有被写穿
    assert (root / "linked.pt").is_symlink()
    assert root.joinpath(".import-x.tmp").exists() is False
    assert _names(root) == ["linked.pt"]


def test_linked_store_root_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked-root"
    _make_dir_link(real, linked)

    store = ModelStore(linked, legacy_roots=[], max_upload_bytes=1024,
                       max_models=10)
    with pytest.raises(ModelStoreError) as exc:
        _import(store)
    assert exc.value.code == "MODEL_PATH_UNSAFE"


def test_store_root_that_is_a_file_is_unavailable(tmp_path):
    root = tmp_path / "weights"
    root.write_bytes(b"not a directory")
    store = ModelStore(root, legacy_roots=[], max_upload_bytes=1024,
                       max_models=10)
    with pytest.raises(ModelStoreError) as exc:
        _import(store)
    assert exc.value.code == "MODEL_STORE_UNAVAILABLE"
    assert store.list_models() == []


# ── 解析：提交边界上的重新核对 ─────────────────────────────────────


def test_resolve_refuses_a_file_replaced_after_listing(tmp_path):
    store = _store(tmp_path)
    record = _import(store)
    record.path.write_bytes(b"tampered")

    with pytest.raises(ModelStoreError) as exc:
        store.resolve(record.model_id)
    assert exc.value.code == "MODEL_CHANGED"
    assert exc.value.status_code == 409


def test_resolve_reports_not_found_after_the_file_is_deleted(tmp_path):
    store = _store(tmp_path)
    record = _import(store)
    record.path.unlink()

    with pytest.raises(ModelStoreError) as exc:
        store.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"
    assert exc.value.status_code == 404


def test_a_fresh_service_cannot_resolve_an_id_it_never_listed(tmp_path):
    first = _store(tmp_path)
    record = _import(first)

    # 服务重启后页面必须先重新获取列表，客户端不能凭旧页面 ID 绕过当前事实
    restarted = _store(tmp_path)
    with pytest.raises(ModelStoreError) as exc:
        restarted.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"

    listed = {row.model_id for row in restarted.list_models()}
    assert record.model_id in listed
    assert restarted.resolve(record.model_id) == record.path


def test_resolve_rejects_a_malformed_model_id(tmp_path):
    store = _store(tmp_path)
    for bad in ("", "sha256:", "yolov8n.pt", "sha256:" + "z" * 64,
                "sha256:" + "a" * 63, "../../etc/passwd"):
        with pytest.raises(ModelStoreError) as exc:
            store.resolve(bad)
        assert exc.value.code == "MODEL_NOT_FOUND"


# ── 列举：受控根、遗留兼容、绝不扫描训练产物 ───────────────────────


def test_list_includes_the_project_root_legacy_files_read_only(tmp_path):
    (tmp_path / "yolov8s.pt").write_bytes(b"legacy")
    store = _store(tmp_path)

    rows = store.list_models()
    assert [row.name for row in rows] == ["yolov8s.pt"]
    row = rows[0]
    assert row.origin == "legacy" and row.available is True
    assert row.public_dict()["origin"] == "legacy"
    # 遗留根只读：绝不在那里创建、覆盖或删除任何东西
    assert _names(tmp_path) == ["yolov8s.pt"]


def test_list_never_walks_into_training_artifacts_or_subdirectories(tmp_path):
    detect = tmp_path / "detect"
    (detect / "train1" / "weights").mkdir(parents=True)
    (detect / "train1" / "weights" / "best.pt").write_bytes(b"artifact")
    (detect / "train1" / "weights" / "last.pt").write_bytes(b"artifact")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.pt").write_bytes(b"nested")
    (tmp_path / "yolov8n.pt").write_bytes(b"legacy")

    store = _store(tmp_path)
    assert [row.name for row in store.list_models()] == ["yolov8n.pt"]


def test_list_merges_managed_and_legacy_and_sorts_stably(tmp_path):
    legacy = tmp_path / "b.pt"
    legacy.write_bytes(b"legacy-b")
    store = _store(tmp_path)
    for name in ("c.pt", "A.pt"):
        _import(store, name=name, data=name.encode())

    rows = store.list_models()
    assert [row.name for row in rows] == ["A.pt", "b.pt", "c.pt"]
    assert {row.origin for row in rows} == {"managed", "legacy"}
    # 同名时先受控、后遗留，顺序稳定
    _import(store, name="b.pt", data=b"managed-b")
    same_name = [row for row in store.list_models() if row.name == "b.pt"]
    assert [row.origin for row in same_name] == ["managed", "legacy"]


def test_list_is_bounded_by_max_models(tmp_path):
    store = _store(tmp_path, max_models=2)
    for name in ("a.pt", "b.pt", "c.pt", "d.pt"):
        _import(store, name=name, data=name.encode())

    rows = store.list_models()
    assert [row.name for row in rows] == ["a.pt", "b.pt"]


def test_list_marks_unreadable_entries_unavailable_without_hiding_them(
        tmp_path, monkeypatch):
    store = _store(tmp_path)
    _import(store)

    def boom(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(service_mod, "_hash_file", boom)
    rows = store.list_models()
    assert [row.name for row in rows] == ["yolov8n.pt"]
    assert rows[0].available is False


def test_list_of_a_missing_root_is_an_empty_controlled_scan(tmp_path):
    store = _store(tmp_path)
    assert store.list_models() == []
    assert not (tmp_path / "models").exists()


# ── 同一内容（同一 model_id）只投影一次，来源必须确定 ───────────────


def test_identical_managed_and_legacy_file_with_the_same_name_keeps_managed(tmp_path):
    """受控目录与项目根各有一个同名同内容的 .pt：列表只能出现一项。"""
    store = _store(tmp_path)
    managed = _import(store)
    (tmp_path / "yolov8n.pt").write_bytes(_PAYLOAD)

    rows = store.list_models()
    assert len(rows) == 1
    row = rows[0]
    assert row.model_id == managed.model_id
    assert row.origin == "managed"
    assert row.path == managed.path
    assert store.resolve(row.model_id) == managed.path
    # 重复文件本身没有被删除或改写
    assert _names(tmp_path / "models" / "weights") == ["yolov8n.pt"]
    assert (tmp_path / "yolov8n.pt").read_bytes() == _PAYLOAD


def test_identical_managed_and_legacy_file_with_a_different_name_keeps_managed(tmp_path):
    """不同名但同内容：同一 model_id 仍只投影一次，且稳定选择 managed。"""
    (tmp_path / "a-legacy.pt").write_bytes(_PAYLOAD)
    store = _store(tmp_path)
    managed = _import(store, name="z-managed.pt")

    rows = store.list_models()
    assert [row.model_id for row in rows] == [managed.model_id]
    assert rows[0].name == "z-managed.pt"
    assert rows[0].origin == "managed"
    assert store.resolve(rows[0].model_id) == managed.path


def test_duplicate_legacy_files_with_identical_content_keep_the_first_sorted_file(tmp_path):
    """同一来源的重复内容按稳定排序保留第一条，且与遍历顺序无关。"""
    (tmp_path / "b.pt").write_bytes(_PAYLOAD)
    (tmp_path / "a.pt").write_bytes(_PAYLOAD)
    store = _store(tmp_path)

    first = store.list_models()
    assert [row.name for row in first] == ["a.pt"]
    assert first[0].origin == "legacy"
    assert store.resolve(first[0].model_id) == first[0].path

    # 重复调用结果完全一致，且重复文件仍然存在（本轮只消除身份歧义）
    second = store.list_models()
    assert [row.public_dict() for row in second] == [row.public_dict() for row in first]
    assert store.resolve(second[0].model_id) == first[0].path
    assert _names(tmp_path) == ["a.pt", "b.pt"]


def test_every_listed_row_resolves_to_the_listed_path(tmp_path):
    """列表投影与 _known 解析必须一致：后遍历的重复记录不得覆盖首选记录。"""
    store = _store(tmp_path)
    managed = _import(store)
    (tmp_path / "yolov8n.pt").write_bytes(_PAYLOAD)
    (tmp_path / "extra.pt").write_bytes(b"extra")

    rows = store.list_models()
    assert {row.model_id for row in rows} == {managed.model_id,
                                              "sha256:" + _sha(b"extra")}
    for row in rows:
        assert row.available is True
        assert store.resolve(row.model_id) == row.path


# ── 列表刷新必须整体收敛 _known，绝不累积已经不在公开列表里的旧 ID ──────
#
# 旧实现只做 ``self._known[row.model_id] = row``，于是被 max_models 截断、被删除、
# 变为不可读或不再公开的旧 ID 在刷新后仍能解析，与“_known 与公开列表逐条一致”矛盾。


def test_a_row_pushed_out_by_max_models_is_no_longer_resolvable(tmp_path):
    """max_models=1：先上传 b.pt，再上传排序更前的 a.pt，旧 ID 必须失效。"""
    store = _store(tmp_path, max_models=1)
    stale = _import(store, name="b.pt", data=b"bee")
    assert store.resolve(stale.model_id) == stale.path

    fresh = _import(store, name="a.pt", data=b"ay")
    rows = store.list_models()
    assert [row.name for row in rows] == ["a.pt"]
    assert store.resolve(fresh.model_id) == fresh.path

    with pytest.raises(ModelStoreError) as exc:
        store.resolve(stale.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"
    assert exc.value.status_code == 404


def test_a_deleted_file_is_no_longer_resolvable_after_a_refresh(tmp_path):
    """文件删除后刷新列表，旧 ID 必须从 _known 移除。"""
    store = _store(tmp_path)
    record = _import(store)
    assert [row.model_id for row in store.list_models()] == [record.model_id]

    record.path.unlink()
    assert store.list_models() == []
    with pytest.raises(ModelStoreError) as exc:
        store.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_an_unreadable_file_is_no_longer_resolvable_after_a_refresh(
        tmp_path, monkeypatch):
    """文件变为不可读后刷新列表：仍可见但不可用，旧 ID 不得再解析。"""
    store = _store(tmp_path)
    record = _import(store)
    assert store.resolve(record.model_id) == record.path

    def boom(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(service_mod, "_hash_file", boom)
    rows = store.list_models()
    assert [row.name for row in rows] == ["yolov8n.pt"]
    assert rows[0].available is False

    with pytest.raises(ModelStoreError) as exc:
        store.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_a_file_replaced_by_a_link_is_no_longer_resolvable_after_a_refresh(tmp_path):
    """文件被换成链接/reparse 后刷新列表，旧 ID 不得再解析。"""
    store = _store(tmp_path)
    record = _import(store)
    assert store.resolve(record.model_id) == record.path

    # 链接目标放在遗留根的子目录里：遗留扫描是非递归的，不会把目标本身当成权重
    (tmp_path / "elsewhere").mkdir()
    outside = tmp_path / "elsewhere" / "outside.pt"
    outside.write_bytes(_PAYLOAD)
    record.path.unlink()
    try:
        os.symlink(str(outside), str(record.path))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this host")

    assert store.list_models() == []
    with pytest.raises(ModelStoreError) as exc:
        store.resolve(record.model_id)
    assert exc.value.code == "MODEL_NOT_FOUND"


def test_an_emptied_library_converges_to_no_resolvable_ids(tmp_path):
    """根目录从有记录变为空后刷新，旧 ID 不再可解析，且列表确实为空。"""
    store = _store(tmp_path)
    records = [_import(store, name=name, data=name.encode())
               for name in ("a.pt", "b.pt")]
    assert [row.model_id for row in store.list_models()] == \
        [row.model_id for row in records]

    for record in records:
        record.path.unlink()
    assert store.list_models() == []
    for record in records:
        with pytest.raises(ModelStoreError) as exc:
            store.resolve(record.model_id)
        assert exc.value.code == "MODEL_NOT_FOUND"


def test_public_list_and_resolvable_ids_are_exactly_the_same_set(tmp_path):
    """收敛后：当前公开列表的每条都能解析，被挤出列表的 ID 一律不可解析。"""
    store = _store(tmp_path, max_models=2)
    for name in ("a.pt", "b.pt", "c.pt"):
        _import(store, name=name, data=name.encode())

    rows = store.list_models()
    assert [row.name for row in rows] == ["a.pt", "b.pt"]
    for row in rows:
        assert store.resolve(row.model_id) == row.path

    stale = "sha256:" + _sha(b"c")
    with pytest.raises(ModelStoreError) as exc:
        store.resolve(stale)
    assert exc.value.code == "MODEL_NOT_FOUND"

    # 再次刷新（无变化）结果稳定，且仍然是同一组可解析 ID
    assert [row.model_id for row in store.list_models()] == \
        [row.model_id for row in rows]
    for row in rows:
        assert store.resolve(row.model_id) == row.path


# ── 默认配置按 basename 唯一解析：绝不隐式下载 ─────────────────────


def test_configured_name_resolves_from_the_controlled_store(tmp_path):
    store = _store(tmp_path)
    record = _import(store)
    assert store.resolve_configured_name("yolov8n.pt") == record.path
    # 受控来源优先于遗留来源
    (tmp_path / "yolov8n.pt").write_bytes(b"legacy")
    assert store.resolve_configured_name("yolov8n.pt") == record.path


def test_configured_name_falls_back_to_a_legacy_file(tmp_path):
    legacy = tmp_path / "yolov8s.pt"
    legacy.write_bytes(b"legacy")
    store = _store(tmp_path)
    assert store.resolve_configured_name("yolov8s.pt") == legacy


@pytest.mark.parametrize("configured", ["yolov8x.pt", "", None, "sub/dir.pt",
                                        "C:\\models\\yolov8n.pt"])
def test_configured_name_never_downloads_an_unknown_weight(tmp_path, configured):
    store = _store(tmp_path)
    with pytest.raises(ModelStoreError) as exc:
        store.resolve_configured_name(configured)
    assert exc.value.code == "MODEL_NOT_FOUND"
    assert exc.value.status_code == 404
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("configured", [
    r"C:\models\yolov8n.pt",
    r"\\server\share\yolov8n.pt",
    "/srv/models/yolov8n.pt",
    "models/weights/yolov8n.pt",
    r"models\weights\yolov8n.pt",
    "../yolov8n.pt",
    " yolov8n.pt ",
    "yolov8n.pt\n",
    "yolov8n.pt\x00",
    ".",
    "..",
    "C:yolov8n.pt",
])
def test_configured_name_rejects_anything_but_a_plain_basename(tmp_path, configured):
    """受控库中真实存在 yolov8n.pt 时，路径形态的配置值仍必须被拒绝。

    旧实现用 ``Path(configured).name`` 把完整路径静默截成 basename，于是配置里
    的 ``C:\\...\\models\\weights\\yolov8n.pt`` 会被当成受控库中的 ``yolov8n.pt``
    接受。这里的目标文件真实存在，因此能真正证明该静默截断被移除。
    """
    store = _store(tmp_path)
    record = _import(store)

    with pytest.raises(ModelStoreError) as exc:
        store.resolve_configured_name(configured)
    assert exc.value.code == "MODEL_NOT_FOUND"
    assert exc.value.status_code == 404
    # 不回显原始配置路径；被拒绝的输入绝不解析成受控库中的同 basename 文件
    assert str(tmp_path) not in exc.value.message
    assert configured != record.name


def test_configured_name_accepts_only_the_exact_basename(tmp_path):
    """同一受控库中，纯 basename 仍然可解析，路径形态一律拒绝。"""
    store = _store(tmp_path)
    record = _import(store)

    assert store.resolve_configured_name("yolov8n.pt") == record.path
    with pytest.raises(ModelStoreError) as exc:
        store.resolve_configured_name(
            str(tmp_path / "models" / "weights" / "yolov8n.pt"))
    assert exc.value.code == "MODEL_NOT_FOUND"


# ── 不反序列化、不联网 ─────────────────────────────────────────────


def test_store_never_deserialises_or_reaches_the_network(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("weights are opaque bytes")

    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch, "load", boom, raising=False)

    store = _store(tmp_path)
    record = _import(store, data=b"\x80\x04cbuiltins\neval\n.")
    assert store.resolve(record.model_id) == record.path
    assert [row.model_id for row in store.list_models()] == [record.model_id]


def test_module_has_no_deserialisation_or_network_dependency():
    """源码级契约：受控权重库不 import 反序列化或网络模块。"""
    import ast

    tree = ast.parse(Path(service_mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"torch", "pickle", "ultralytics", "requests",
                           "urllib", "http", "socket", "subprocess", "shutil"}
