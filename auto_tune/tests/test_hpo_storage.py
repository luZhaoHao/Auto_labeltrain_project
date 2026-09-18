"""H1.1 原子 JSON 存储与锁测试 — 合成数据，无真实训练/LLM/网络。"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from auto_tune.modules.hpo import storage as storage_module
from auto_tune.modules.hpo.models import (
    EnvironmentSnapshot,
    HpoError,
    ModelBinding,
    SnapshotBinding,
    StudyConfig,
    StudyRecord,
    TrialRecord,
)
from auto_tune.modules.hpo.storage import (
    MAX_STUDY_JSON_BYTES,
    StudyStore,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot_binding():
    return SnapshotBinding(snapshot_id="a" * 64, manifest_digest="b" * 64,
                           snapshot_path=str(Path("C:/snap/s1")),
                           data_yaml_path=str(Path("C:/snap/s1/data.yaml")))


def _model_binding():
    return ModelBinding(model_path=str(Path("C:/models/y.pt")), model_bytes=123,
                        model_sha256="c" * 64)


def _env():
    return EnvironmentSnapshot(python_version="3.10.18", optuna_version="4.5.0",
                               numpy_version="2.2.6", ultralytics_version="8.3.253")


def _sampled_sgd():
    return {"optimizer": "SGD", "lr0": 1e-4, "lrf": 0.05, "momentum_sgd": 0.9,
            "weight_decay": 0.0005, "warmup_epochs": 2}


def _distributions():
    raw = {
        "optimizer": {"name": "CategoricalDistribution",
                      "attributes": {"choices": ["SGD", "AdamW"]}},
        "lr0": {"name": "FloatDistribution", "attributes": {"low": 1e-5,
                                                            "high": 0.005,
                                                            "log": True}},
        "lrf": {"name": "FloatDistribution", "attributes": {"low": 0.01,
                                                            "high": 0.1,
                                                            "log": True}},
        "momentum_sgd": {"name": "FloatDistribution",
                         "attributes": {"low": 0.8, "high": 0.98, "log": False}},
        "weight_decay": {"name": "FloatDistribution",
                         "attributes": {"low": 0.0, "high": 0.001, "log": False}},
        "warmup_epochs": {"name": "IntDistribution",
                          "attributes": {"low": 0, "high": 5, "step": 1, "log": False}},
    }
    return {key: json.dumps(value) for key, value in raw.items()}


def _candidate():
    return {"optimizer": "SGD", "lr0": 1e-4, "lrf": 0.05, "momentum": 0.9,
            "weight_decay": 0.0005, "warmup_epochs": 2}


def make_record(study_id="hpo_" + "a" * 32, revision=0, trials=None):
    return StudyRecord(
        study_id=study_id,
        config=StudyConfig(),
        created_at="2026-09-07T00:00:00.000000+00:00",
        updated_at="2026-09-07T00:00:00.000000+00:00",
        revision=revision,
        snapshot_binding=_snapshot_binding(),
        model_binding=_model_binding(),
        environment=_env(),
        trials=list(trials or []),
    )


def _pending_trial(study_id, number, request_id="0" * 32):
    return TrialRecord(
        number=number,
        trial_id=f"{study_id}_t{number:04d}",
        request_id=request_id,
        state="PENDING",
        sampled_params=_sampled_sgd(),
        distributions=_distributions(),
        candidate_params=_candidate(),
        created_at="2026-09-07T00:00:00.000000+00:00",
    )


def _publish(root, record):
    store = StudyStore(root)
    (root / record.study_id).mkdir(parents=True, exist_ok=True)
    with store.locked(record.study_id):
        store.write(record)


def _read(root, study_id):
    store = StudyStore(root)
    with store.locked(study_id):
        return store.read(study_id)


# ── 正常往返 ───────────────────────────────────────────────────────

def test_round_trip_preserves_record(tmp_path):
    root = tmp_path / "hpo"
    record = make_record(revision=2)
    _publish(root, record)
    restored = _read(root, record.study_id)
    assert restored.study_id == record.study_id
    assert restored.revision == 2
    assert restored.model_dump(mode="json") == record.model_dump(mode="json")


def test_round_trip_preserves_trials(tmp_path):
    root = tmp_path / "hpo"
    sid = "hpo_" + "b" * 32
    record = make_record(study_id=sid, revision=1,
                         trials=[_pending_trial(sid, 0)])
    _publish(root, record)
    restored = _read(root, sid)
    assert len(restored.trials) == 1
    assert restored.trials[0].number == 0
    assert restored.trials[0].trial_id == f"{sid}_t0000"


def test_read_returns_fresh_copy(tmp_path):
    root = tmp_path / "hpo"
    record = make_record(revision=3)
    _publish(root, record)
    first = _read(root, record.study_id)
    first.revision = 999
    second = _read(root, record.study_id)
    assert second.revision == 3


# ── 不存在 / 空目录 / 路径穿越 ─────────────────────────────────────

def test_unknown_study_is_not_found(tmp_path):
    root = tmp_path / "hpo"
    with pytest.raises(HpoError) as err:
        _read(root, "hpo_" + "f" * 32)
    assert err.value.code == "HPO_NOT_FOUND"


def test_empty_unpublished_dir_is_not_a_study(tmp_path):
    root = tmp_path / "hpo"
    sid = "hpo_" + "e" * 32
    (root / sid).mkdir(parents=True)
    with pytest.raises(HpoError) as err:
        _read(root, sid)
    assert err.value.code == "HPO_NOT_FOUND"


def test_invalid_study_id_cannot_escape(tmp_path):
    root = tmp_path / "hpo"
    store = StudyStore(root)
    for bad in ["../../evil", "hpo_nothex", "hpo_" + "g" * 31, ""]:
        with pytest.raises(HpoError) as err:
            store.study_dir(bad)
        assert err.value.code == "HPO_NOT_FOUND"


def test_write_requires_lock(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    (root / record.study_id).mkdir(parents=True, exist_ok=True)
    store = StudyStore(root)
    with pytest.raises(RuntimeError):
        store.write(record)


def test_read_requires_lock(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    store = StudyStore(root)
    with pytest.raises(RuntimeError):
        store.read(record.study_id)


# ── 损坏 / 重复键 / 未知 schema / 超大文件 ─────────────────────────

def test_corrupt_json_is_corrupt_study(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    target = root / record.study_id / "study.json"
    before = target.read_bytes()
    target.write_text("{ not json !!!", encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(root, record.study_id)
    assert err.value.code == "HPO_CORRUPT_STUDY"
    # 不覆盖损坏文件
    assert target.read_bytes() == b"{ not json !!!"
    assert target.read_bytes() != before


def test_duplicate_json_keys_rejected(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    target = root / record.study_id / "study.json"
    sid = record.study_id
    payload = f'{{"study_id": "{sid}", "study_id": "{sid}"}}'
    target.write_text(payload, encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(root, record.study_id)
    assert err.value.code == "HPO_CORRUPT_STUDY"


def test_nonfinite_json_constants_rejected(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    target = root / record.study_id / "study.json"
    target.write_text('{"config": NaN}', encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(root, record.study_id)
    assert err.value.code == "HPO_CORRUPT_STUDY"


def test_unknown_schema_version_rejected(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    target = root / record.study_id / "study.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["config"]["schema_version"] = "hpo-study-v2"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HpoError) as err:
        _read(root, record.study_id)
    assert err.value.code == "HPO_CORRUPT_STUDY"


def test_study_json_over_size_limit_rejected(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    target = root / record.study_id / "study.json"
    with open(target, "wb") as fh:
        fh.write(b'{"pad":"' + b"x" * (MAX_STUDY_JSON_BYTES + 1) + b'"}')
    with pytest.raises(HpoError) as err:
        _read(root, record.study_id)
    assert err.value.code == "HPO_CORRUPT_STUDY"


# ── 写盘故障注入 ───────────────────────────────────────────────────

def test_fsync_failure_preserves_old_file(tmp_path, monkeypatch):
    root = tmp_path / "hpo"
    record = make_record(revision=1)
    _publish(root, record)
    store = StudyStore(root)
    target = root / record.study_id / "study.json"
    before_bytes = target.read_bytes()

    def _fail_fsync(fd):
        raise OSError("injected fsync failure")

    with monkeypatch.context() as m:
        m.setattr(storage_module.os, "fsync", _fail_fsync)
        with pytest.raises(HpoError) as err:
            with store.locked(record.study_id):
                store.write(record.model_copy(update={"revision": 2}))
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert target.read_bytes() == before_bytes
    assert _read(root, record.study_id).revision == 1


def test_replace_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    root = tmp_path / "hpo"
    record = make_record(revision=1)
    _publish(root, record)
    store = StudyStore(root)
    study_dir = root / record.study_id
    target = study_dir / "study.json"
    before_bytes = target.read_bytes()

    def _fail_replace(src, dst):
        raise OSError("injected replace failure")

    with monkeypatch.context() as m:
        m.setattr(storage_module.os, "replace", _fail_replace)
        with pytest.raises(HpoError) as err:
            with store.locked(record.study_id):
                store.write(record.model_copy(update={"revision": 7}))
    assert err.value.code == "HPO_PERSISTENCE_ERROR"
    assert target.read_bytes() == before_bytes
    temps = [p for p in study_dir.iterdir()
             if p.name.startswith(".study.json.") and p.name.endswith(".tmp")]
    assert temps == []
    assert _read(root, record.study_id).revision == 1


def test_successful_write_after_failure_does_not_resurrect(tmp_path, monkeypatch):
    root = tmp_path / "hpo"
    record = make_record(revision=1)
    _publish(root, record)
    store = StudyStore(root)

    def _fail_replace(src, dst):
        raise OSError("injected replace failure")

    with monkeypatch.context() as m:
        m.setattr(storage_module.os, "replace", _fail_replace)
        with pytest.raises(HpoError):
            with store.locked(record.study_id):
                store.write(record.model_copy(update={"revision": 2}))

    # 后续成功写盘从磁盘旧态继续；失败变更不复活。
    with store.locked(record.study_id):
        current = store.read(record.study_id)
        store.write(current.model_copy(update={"revision": 3}))
    assert _read(root, record.study_id).revision == 3


# ── 锁竞争（进程内）───────────────────────────────────────────────

def test_same_process_two_stores_busy(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    sid = record.study_id
    store1 = StudyStore(root)
    store2 = StudyStore(root)
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def hold():
        try:
            with store1.locked(sid):
                acquired.set()
                release.wait(5)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(5)
    with pytest.raises(HpoError) as err:
        with store2.locked(sid):
            pass
    assert err.value.code == "HPO_STUDY_BUSY"
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    with store2.locked(sid):
        restored = store2.read(sid)
    assert restored.study_id == sid


# ── 锁竞争（跨进程）───────────────────────────────────────────────

def test_lock_busy_across_processes(tmp_path):
    root = tmp_path / "hpo"
    record = make_record()
    _publish(root, record)
    sid = record.study_id
    store = StudyStore(root)

    script = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "from pathlib import Path\n"
        "from auto_tune.modules.hpo.storage import StudyStore\n"
        "s = StudyStore(Path(sys.argv[1]))\n"
        "with s.locked(sys.argv[2]):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(5)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(root), sid, str(REPO_ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "LOCKED"
    with pytest.raises(HpoError) as err:
        with store.locked(sid):
            pass
    assert err.value.code == "HPO_STUDY_BUSY"
    proc.terminate()
    proc.wait(timeout=15)
    # 子进程退出后 OS 释放文件锁，本进程可取得锁。
    with store.locked(sid):
        restored = store.read(sid)
    assert restored.study_id == sid
