"""H1.2 执行契约与原子存储测试 — 合成数据，无训练/LLM/网络。

覆盖 ExecutionConfig 严格输入、ExecutionRecord/Attempt 契约、execution.json
原子读写、跨 study/损坏/超大/重复键拒绝、revision 单调、根/祖先/锁/JSON 链接
以及同/跨进程执行锁 BUSY。存储层只处理已存在合法 study，不创建第二份 study。
"""

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_tune.modules.hpo import storage as study_storage
from auto_tune.modules.hpo.execution_models import (
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionRecord,
    ExecutionRoots,
    utc_now_iso,
)
from auto_tune.modules.hpo.execution_storage import (
    MAX_EXECUTION_JSON_BYTES,
    ExecutionStore,
)
from auto_tune.modules.hpo.models import (
    EnvironmentSnapshot,
    HpoError,
    ModelBinding,
    SnapshotBinding,
    StudyConfig,
    StudyRecord,
    TrialRecord,
)
from auto_tune.modules.hpo.storage import StudyStore

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── ExecutionConfig 严格输入 ──────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"batch": True}, {"batch": -1}, {"batch": 0}, {"batch": 257},
    {"imgsz": 65}, {"imgsz": 31}, {"imgsz": 33}, {"imgsz": 0},
    {"device": "0,1"}, {"device": "00"}, {"device": ""}, {"device": "cpu0"},
    {"device": "1.0"}, {"device": "64"}, {"device": "-1"},
    {"timeout_seconds": "60"}, {"timeout_seconds": 0}, {"timeout_seconds": 86401},
    {"timeout_seconds": True}, {"imgsz": "640"},
])
def test_reject_invalid_execution_config(kwargs):
    with pytest.raises(ValidationError):
        ExecutionConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"batch": 1}, {"imgsz": 32}, {"imgsz": 64}, {"device": "cpu"},
    {"device": "0"}, {"device": "1"}, {"device": "63"}, {"timeout_seconds": 1},
    {"timeout_seconds": 86400},
])
def test_accept_valid_execution_config(kwargs):
    cfg = ExecutionConfig(**kwargs)
    assert cfg.device


def test_mutated_config_revalidated_after_assignment():
    cfg = ExecutionConfig()
    with pytest.raises(ValidationError):
        ExecutionConfig(**{**cfg.model_dump(), "batch": 999})


# ── 记录构建 helper ──────────────────────────────────────────────

def _roots(tmp_path):
    return ExecutionRoots(
        storage_root=str(tmp_path / "hpo"),
        output_root=str(tmp_path / "out"),
        log_root=str(tmp_path / "log"),
    )


def _env():
    return ExecutionEnvironment(
        sys_executable=str(Path(sys.executable)),
        python_version="3.10.18",
        torch_version="2.5.1",
        cuda_version="",
        ultralytics_version="8.3.253",
        optuna_version="4.5.0",
        numpy_version="2.2.6",
    )


def _cfg(**overrides):
    data = dict(ExecutionConfig().model_dump())
    data.update(overrides)
    return ExecutionConfig(**data)


def _pending_study_record(study_id):
    return StudyRecord(
        study_id=study_id,
        config=StudyConfig(),
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        revision=0,
        snapshot_binding=SnapshotBinding(
            snapshot_id="a" * 64, manifest_digest="b" * 64,
            snapshot_path=str(Path("C:/snap/s1")),
            data_yaml_path=str(Path("C:/snap/s1/data.yaml"))),
        model_binding=ModelBinding(model_path=str(Path("C:/models/y.pt")),
                                   model_bytes=123, model_sha256="c" * 64),
        environment=EnvironmentSnapshot(
            python_version="3.10.18", optuna_version="4.5.0",
            numpy_version="2.2.6", ultralytics_version="8.3.253"),
    )


def _publish_study(tmp_path, study_id="hpo_" + "a" * 32):
    root = tmp_path / "hpo"
    store = StudyStore(root)
    record = _pending_study_record(study_id)
    (root / study_id).mkdir(parents=True, exist_ok=True)
    with store.locked(study_id):
        store.write(record)
    return root


def _make_record(study_id="hpo_" + "a" * 32, revision=0, status="READY",
                 attempts=None, tmp_path=None):
    return ExecutionRecord(
        schema_version="hpo-execution-v1",
        study_id=study_id,
        revision=revision,
        created_at="2026-09-08T00:00:00.000000+00:00",
        updated_at="2026-09-08T00:00:00.000000+00:00",
        config=_cfg(),
        roots=_roots(tmp_path) if tmp_path is not None else ExecutionRoots(
            storage_root="C:/hpo", output_root="C:/out", log_root="C:/log"),
        environment=_env(),
        status=status,
        stop_reason=None,
        attempts=list(attempts or []),
    )


def _publish(tmp_path, record):
    root = tmp_path / "hpo"
    store = ExecutionStore(root)
    with store.locked(record.study_id):
        store.write(record)


def _read(tmp_path, study_id):
    root = tmp_path / "hpo"
    store = ExecutionStore(root)
    with store.locked(study_id):
        return store.read(study_id)


# ── 正常往返 / revision ──────────────────────────────────────────

def test_round_trip_preserves_record_and_revision_bumps(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    record = _make_record(study_id=study_id, revision=0, tmp_path=tmp_path)
    _publish(tmp_path, record)
    assert _read(tmp_path, study_id).revision == 0

    store = ExecutionStore(tmp_path / "hpo")
    with store.locked(study_id):
        current = store.read(study_id)
        assert current.model_dump(mode="json") == record.model_dump(mode="json")
        store.write(current.model_copy(update={"revision": 1}))
    assert _read(tmp_path, study_id).revision == 1


def test_first_publish_revision_zero_and_reject_first_nonzero(tmp_path):
    study_id = "hpo_" + "b" * 32
    _publish_study(tmp_path, study_id)
    store = ExecutionStore(tmp_path / "hpo")
    with pytest.raises(HpoError) as err:
        with store.locked(study_id):
            store.write(_make_record(study_id=study_id, revision=3, tmp_path=tmp_path))
    assert err.value.code == "HPO_PERSISTENCE_ERROR"


def test_write_requires_study_exists(tmp_path):
    store = ExecutionStore(tmp_path / "hpo")
    with pytest.raises(HpoError) as err:
        with store.locked("hpo_" + "c" * 32):
            store.write(_make_record(study_id="hpo_" + "c" * 32, tmp_path=tmp_path))
    assert err.value.code == "HPO_NOT_FOUND"


# ── 损坏 / 重复键 / schema / 超大 / 跨 study ─────────────────────

def test_corrupt_json_is_corrupt_execution(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    target = tmp_path / "hpo" / study_id / "execution.json"
    before = target.read_bytes()
    target.write_text("{ nope !!!", encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert target.read_bytes() == b"{ nope !!!"
    assert target.read_bytes() != before


def test_duplicate_json_keys_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    target = tmp_path / "hpo" / study_id / "execution.json"
    payload = f'{{"study_id": "{study_id}", "study_id": "{study_id}"}}'
    target.write_text(payload, encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_nonfinite_json_constant_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    target = tmp_path / "hpo" / study_id / "execution.json"
    target.write_text('{"status": NaN}', encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_unknown_schema_version_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    target = tmp_path / "hpo" / study_id / "execution.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["schema_version"] = "hpo-execution-v2"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_execution_json_over_size_limit_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    target = tmp_path / "hpo" / study_id / "execution.json"
    with open(target, "wb") as fh:
        fh.write(b'{"pad":"' + b"x" * (MAX_EXECUTION_JSON_BYTES + 1) + b'"}')
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_cross_study_copy_rejected(tmp_path):
    sid_a = "hpo_" + "a" * 32
    sid_b = "hpo_" + "b" * 32
    _publish_study(tmp_path, sid_a)
    _publish_study(tmp_path, sid_b)
    _publish(tmp_path, _make_record(study_id=sid_a, tmp_path=tmp_path))
    # 把 A 的 execution.json 拷进 B 的目录：内容身份仍为 A。
    target_b = tmp_path / "hpo" / sid_b / "execution.json"
    target_b.write_bytes((tmp_path / "hpo" / sid_a / "execution.json").read_bytes())
    with pytest.raises(HpoError) as err:
        _read(tmp_path, sid_b)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_write_does_not_overwrite_corrupt_existing_facts(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    valid = _read(tmp_path, study_id)
    target = tmp_path / "hpo" / study_id / "execution.json"
    target.write_text("{broken", encoding="utf-8")
    store = ExecutionStore(tmp_path / "hpo")
    with pytest.raises(HpoError) as err:
        with store.locked(study_id):
            store.write(valid.model_copy(update={"revision": 1}))
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert target.read_text(encoding="utf-8") == "{broken"


def test_revision_rollback_and_jump_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    store = ExecutionStore(tmp_path / "hpo")
    with store.locked(study_id):
        current = store.read(study_id)
    # 回退
    with pytest.raises(HpoError) as err:
        with store.locked(study_id):
            store.write(current.model_copy(update={"revision": 0}))
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    # 跳跃
    with pytest.raises(HpoError) as err:
        with store.locked(study_id):
            store.write(current.model_copy(update={"revision": 5}))
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


# ── 写盘故障 ─────────────────────────────────────────────────────

def test_replace_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, revision=0, tmp_path=tmp_path))
    store = ExecutionStore(tmp_path / "hpo")
    target = tmp_path / "hpo" / study_id / "execution.json"
    before = target.read_bytes()

    def _fail_replace(src, dst):
        raise OSError("injected replace failure")

    with monkeypatch.context() as m:
        m.setattr(study_storage.os, "replace", _fail_replace)
        with pytest.raises(HpoError) as err:
            with store.locked(study_id):
                current = store.read(study_id)
                store.write(current.model_copy(update={"revision": 1}))
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert target.read_bytes() == before
    temps = [p for p in target.parent.iterdir()
             if p.name.startswith(".execution.json.") and p.name.endswith(".tmp")]
    assert temps == []
    assert _read(tmp_path, study_id).revision == 0


def test_fsync_failure_preserves_old_file(tmp_path, monkeypatch):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, revision=0, tmp_path=tmp_path))
    store = ExecutionStore(tmp_path / "hpo")
    target = tmp_path / "hpo" / study_id / "execution.json"
    before = target.read_bytes()

    def _fail_fsync(fd):
        raise OSError("injected fsync failure")

    with monkeypatch.context() as m:
        m.setattr(study_storage.os, "fsync", _fail_fsync)
        with pytest.raises(HpoError) as err:
            with store.locked(study_id):
                current = store.read(study_id)
                store.write(current.model_copy(update={"revision": 1}))
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert target.read_bytes() == before
    assert _read(tmp_path, study_id).revision == 0


# ── 链接拒绝 ─────────────────────────────────────────────────────

def test_storage_root_symlink_rejected(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    root = tmp_path / "root_link"
    os.symlink(destination, root, target_is_directory=True)
    store = ExecutionStore(root)
    with pytest.raises(HpoError) as err:
        with store.runner_locked():
            pass
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert not list(destination.iterdir())


def test_root_ancestor_link_rejected(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    linked_parent = tmp_path / "linked_parent"
    os.symlink(destination, linked_parent, target_is_directory=True)
    store = ExecutionStore(linked_parent / "hpo")
    with pytest.raises(HpoError) as err:
        with store.runner_locked():
            pass
    assert err.value.code == "HPO_CORRUPT_EXECUTION"


def test_runner_lock_file_link_rejected(tmp_path):
    root = tmp_path / "hpo"
    root.mkdir()
    target = tmp_path / "other"
    target.write_bytes(b"")
    os.symlink(target, root / ".hpo-runner.lock")
    store = ExecutionStore(root)
    with pytest.raises(HpoError) as err:
        with store.runner_locked():
            pass
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert target.read_bytes() == b""


def test_execution_json_link_rejected(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    target = tmp_path / "hpo" / study_id / "execution.json"
    other = tmp_path / "other"
    other.write_bytes(b"{}")
    os.symlink(other, target)
    with pytest.raises(HpoError) as err:
        _read(tmp_path, study_id)
    assert err.value.code == "HPO_CORRUPT_EXECUTION"
    assert other.read_bytes() == b"{}"


# ── 锁（进程内 / 跨进程）─────────────────────────────────────────

def test_same_process_execution_locked_busy(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    store1 = ExecutionStore(tmp_path / "hpo")
    store2 = ExecutionStore(tmp_path / "hpo")
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def hold():
        try:
            with store1.locked(study_id):
                acquired.set()
                release.wait(5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(5)
    with pytest.raises(HpoError) as err:
        with store2.locked(study_id):
            pass
    assert err.value.code == "HPO_STUDY_BUSY"
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    with store2.locked(study_id):
        restored = store2.read(study_id)
    assert restored.study_id == study_id


def test_runner_lock_busy_across_processes(tmp_path):
    root = tmp_path / "hpo"
    root.mkdir()
    script = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from pathlib import Path\n"
        "from auto_tune.modules.hpo.execution_storage import ExecutionStore\n"
        "s = ExecutionStore(Path(sys.argv[1]))\n"
        "with s.runner_locked():\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(5)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(REPO_ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "LOCKED"
        store = ExecutionStore(root)
        with pytest.raises(HpoError) as err:
            with store.runner_locked():
                pass
        assert err.value.code == "HPO_EXECUTION_BUSY"
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    with ExecutionStore(root).runner_locked():
        pass


def test_execution_lock_busy_across_processes(tmp_path):
    study_id = "hpo_" + "a" * 32
    _publish_study(tmp_path, study_id)
    _publish(tmp_path, _make_record(study_id=study_id, tmp_path=tmp_path))
    root = tmp_path / "hpo"
    script = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "from pathlib import Path\n"
        "from auto_tune.modules.hpo.execution_storage import ExecutionStore\n"
        "s = ExecutionStore(Path(sys.argv[1]))\n"
        "with s.locked(sys.argv[2]):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(5)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(root), study_id, str(REPO_ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "LOCKED"
        store = ExecutionStore(root)
        with pytest.raises(HpoError) as err:
            with store.locked(study_id):
                pass
        assert err.value.code == "HPO_STUDY_BUSY"
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    store = ExecutionStore(root)
    with store.locked(study_id):
        restored = store.read(study_id)
    assert restored.study_id == study_id
