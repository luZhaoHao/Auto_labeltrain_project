"""Task 3: Service projection, legacy JSON import and stable status (Studio S2 Core)."""

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from auto_tune.modules.local_index.models import (
    ExperimentQuery,
    LocalIndexConfig,
    LocalIndexConfigError,
    LocalIndexPersistenceError,
)
from auto_tune.modules.local_index.service import (
    MAX_LEGACY_FILE_BYTES,
    LocalIndexService,
    _CHUNK_SIZE,
    _LegacyFileTooLarge,
    _read_bounded,
)


def _cfg(tmp_path, **kw):
    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 5000),
    )


@pytest.fixture
def service(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    return svc


def _snapshot_payload(**kw):
    payload = {
        "source_dataset_path": "C:/data/source_ds",
        "data_yaml_path": "C:/log/dataset_snapshots/abc/data.yaml",
        "snapshot_id": "snap123",
        "snapshot_manifest_digest": "digest123",
        "snapshot_valid": True,
    }
    payload.update(kw)
    return payload


# ── Step 1-2: dataset projection ──


def test_index_dataset_snapshot_projection(service):
    payload = _snapshot_payload()
    record = service.index_dataset(payload)
    expected_id = hashlib.sha256(b"snapshot:snap123").hexdigest()
    assert record.dataset_id == expected_id
    assert record.canonical_path == os.path.normcase(os.path.normpath("C:/data/source_ds"))
    assert record.snapshot_id == "snap123"
    assert record.snapshot_digest == "digest123"
    assert record.validation_status == "valid"
    listed = service.list_datasets()
    assert len(listed) == 1
    assert listed[0]["dataset_id"] == expected_id


def test_index_dataset_path_projection_without_snapshot(service):
    payload = {"dataset_path": "C:/data/flat_ds"}
    record = service.index_dataset(payload)
    canonical = os.path.normcase(os.path.normpath("C:/data/flat_ds"))
    expected_id = hashlib.sha256(f"path:{canonical}".encode("utf-8")).hexdigest()
    assert record.dataset_id == expected_id
    assert record.snapshot_id is None
    assert record.validation_status == "unverified"


def test_index_dataset_rejects_relative_path(service):
    with pytest.raises(LocalIndexConfigError):
        service.index_dataset({"dataset_path": "relative/ds"})


def test_index_dataset_same_snapshot_idempotent(service):
    service.index_dataset(_snapshot_payload())
    service.index_dataset(_snapshot_payload())
    assert len(service.list_datasets()) == 1


# ── Step 3-4: experiment projection and runtime_run_id ──


def test_index_experiment_runtime_run_id(service):
    record = {
        "run_id": "tuning:s1:autotune_1",
        "run_name": "autotune_1",
        "source": "tuning",
        "status": "completed",
        "analysis_status": "completed",
        "params": {"model": "yolov8n.pt", "epochs": 100},
        "metrics": {"mAP50": 0.6},
        "finished_at": "2026-08-01T00:00:00Z",
    }
    service.index_experiment(record, runtime_run_id="tuning:uuid-123")
    got = service.get_experiment("tuning:uuid-123")
    assert got is not None
    assert got["run_id"] == "tuning:uuid-123"
    assert got["params"]["_legacy_record_run_id"] == "tuning:s1:autotune_1"
    assert service.get_experiment("tuning:s1:autotune_1") is None


def test_index_experiment_multi_writes_same_session(service):
    base = {
        "run_name": "autotune_1", "source": "tuning",
        "params": {}, "metrics": {},
    }
    service.index_experiment({**base, "run_id": "tuning:s1:autotune_1", "status": "running"},
                             runtime_run_id="tuning:uuid-1")
    service.index_experiment({**base, "run_id": "tuning:s1:autotune_1", "status": "completed"},
                             runtime_run_id="tuning:uuid-1")
    rows = service.list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


@pytest.mark.parametrize("raw,expected", [
    ("done", "completed"),
    ("error", "failed"),
    ("aborted", "cancelled"),
    ("completed", "completed"),
])
def test_index_experiment_status_mapping(service, raw, expected):
    record = {
        "run_id": "manual:1", "source": "manual", "status": raw,
        "params": {}, "metrics": {},
    }
    service.index_experiment(record)
    got = service.get_experiment("manual:1")
    assert got["status"] == expected


def test_index_experiment_dataset_association(service):
    service.index_dataset({
        "source_dataset_path": "C:/data/ds",
        "data_yaml_path": "C:/log/snapshots/x/data.yaml",
        "snapshot_id": "snapx",
        "snapshot_valid": True,
    })
    record = {
        "run_id": "manual:1", "source": "manual", "status": "completed",
        "params": {"model": "yolov8n.pt", "data": "C:/log/snapshots/x/data.yaml"},
        "metrics": {},
    }
    service.index_experiment(record)
    got = service.get_experiment("manual:1")
    assert got["dataset_id"] is not None
    assert got["params"]["model"] == "yolov8n.pt"


def test_index_experiment_dataset_association_normalized_path(service, tmp_path, monkeypatch):
    # Production latest_dataset.json stores a relative data.yaml path; the
    # experiment params.data is absolute. Both must resolve to the same key.
    monkeypatch.chdir(tmp_path)
    rel_yaml = os.path.join("log", "snapshots", "x", "data.yaml")
    service.index_dataset({
        "source_dataset_path": str(tmp_path / "ds"),
        "data_yaml_path": rel_yaml,
        "snapshot_id": "snapx",
        "snapshot_valid": True,
    })
    record = {
        "run_id": "manual:1", "source": "manual", "status": "completed",
        "params": {"model": "yolov8n.pt", "data": os.path.abspath(rel_yaml)},
        "metrics": {},
    }
    service.index_experiment(record)
    got = service.get_experiment("manual:1")
    assert got["dataset_id"] is not None


def test_index_experiment_no_dataset_matching_keeps_none(service):
    record = {
        "run_id": "manual:1", "source": "manual", "status": "completed",
        "params": {"data": "C:/no/such/dataset.yaml"}, "metrics": {},
    }
    service.index_experiment(record)
    got = service.get_experiment("manual:1")
    assert got["dataset_id"] is None


def test_index_experiment_unknown_source_marked(service):
    record = {
        "run_id": "manual:1", "source": "weird-source", "status": "completed",
        "params": {}, "metrics": {},
    }
    service.index_experiment(record)
    got = service.get_experiment("manual:1")
    assert got["source"] == "unknown"


# ── Step 5-6: legacy import ──


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_import_experiment_history_v1(service, tmp_path):
    f = _write(tmp_path / "experiment_history.json", {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:train1", "run_name": "train1", "source": "manual",
             "status": "completed", "analysis_status": "completed",
             "params": {"model": "yolov8n.pt"}, "metrics": {"mAP50": 0.5},
             "finished_at": "2026-08-01T00:00:00Z"},
            {"run_id": "tuning:s1:at1", "run_name": "at1", "source": "tuning",
             "status": "completed", "analysis_status": "completed",
             "params": {"model": "yolov8n.pt"}, "metrics": {"mAP50": 0.6},
             "finished_at": "2026-08-02T00:00:00Z"},
        ],
    })
    summary = service.import_legacy_files([f])
    assert summary.imported == 1
    assert summary.skipped == 0
    assert summary.failed == 0
    rows = service.list_experiments(ExperimentQuery(limit=100))
    assert {r["run_id"] for r in rows} == {"manual:train1", "tuning:s1:at1"}


def test_import_tuning_history_list(service, tmp_path):
    f = _write(tmp_path / "tuning_history.json", [
        {"train_name": "autotune_1", "result_mAP50": 0.5, "timestamp": "2026-08-01T00:00:00Z"},
        {"train_name": "autotune_2", "error": "用户取消", "timestamp": "2026-08-02T00:00:00Z"},
    ])
    summary = service.import_legacy_files([f])
    assert summary.imported == 1
    rows = service.list_experiments(ExperimentQuery(limit=100))
    tuning = [r for r in rows if r["run_id"].startswith("legacy-tuning:")]
    assert len(tuning) == 2
    by_name = {r["run_name"]: r for r in tuning}
    assert by_name["autotune_1"]["status"] == "completed"
    assert by_name["autotune_1"]["metrics"]["mAP50"] == 0.5
    assert by_name["autotune_2"]["status"] == "cancelled"


def test_import_idempotent_same_content_skips(service, tmp_path):
    f = _write(tmp_path / "experiment_history.json", {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:a", "run_name": "a", "source": "manual",
             "status": "completed", "params": {}, "metrics": {}},
        ],
    })
    first = service.import_legacy_files([f])
    second = service.import_legacy_files([f])
    assert first.imported == 1
    assert second.skipped == 1
    assert second.imported == 0
    assert len(service.list_experiments(ExperimentQuery(limit=100))) == 1


def test_import_updated_content_upserts_no_duplicate(service, tmp_path):
    f = _write(tmp_path / "experiment_history.json", {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:a", "run_name": "a", "source": "manual",
             "status": "completed", "params": {}, "metrics": {"mAP50": 0.1}},
        ],
    })
    service.import_legacy_files([f])
    _write(f, {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:a", "run_name": "a", "source": "manual",
             "status": "completed", "params": {}, "metrics": {"mAP50": 0.9}},
            {"run_id": "manual:b", "run_name": "b", "source": "manual",
             "status": "completed", "params": {}, "metrics": {}},
        ],
    })
    summary = service.import_legacy_files([f])
    assert summary.imported == 1
    assert summary.skipped == 0
    rows = service.list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 2
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["manual:a"]["metrics"]["mAP50"] == 0.9


def test_import_invalid_json_isolated(service, tmp_path):
    good = _write(tmp_path / "experiment_history.json", {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:a", "run_name": "a", "source": "manual",
             "status": "completed", "params": {}, "metrics": {}},
        ],
    })
    bad = _write(tmp_path / "broken.json", b"{not valid json")
    summary = service.import_legacy_files([good, bad])
    assert summary.imported == 1
    assert summary.failed == 1
    assert summary.failures[0].error_code == "LEGACY_IMPORT_INVALID_JSON"
    assert str(tmp_path) not in summary.failures[0].message
    assert "broken.json" in summary.failures[0].message
    assert len(service.list_experiments(ExperimentQuery(limit=100))) == 1


def test_import_invalid_schema(service, tmp_path):
    f = _write(tmp_path / "experiment_history.json", {"foo": "bar"})
    summary = service.import_legacy_files([f])
    assert summary.failed == 1
    assert summary.failures[0].error_code == "LEGACY_IMPORT_INVALID_SCHEMA"


def test_import_too_large(service, tmp_path):
    f = _write(tmp_path / "huge.json", b"x" * (MAX_LEGACY_FILE_BYTES + 1))
    summary = service.import_legacy_files([f])
    assert summary.failed == 1
    assert summary.failures[0].error_code == "LEGACY_IMPORT_TOO_LARGE"
    assert service.list_experiments(ExperimentQuery(limit=100)) == []


# ── S2 Core 返修：旧 JSON 导入改为流式有界读取（1 MiB 块，16 MiB 上限）──


def test_read_bounded_reads_in_at_most_1mib_chunks(tmp_path, monkeypatch):
    path = tmp_path / "chunked.json"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 123))
    requested: list[int] = []
    real_open = open

    def recording_open(p, mode="r", *args, **kwargs):
        fh = real_open(p, mode, *args, **kwargs)
        real_read = fh.read

        def read(n=-1):
            requested.append(n)
            return real_read(n)

        fh.read = read  # type: ignore[method-assign]
        return fh

    monkeypatch.setattr("builtins.open", recording_open)
    data, digest = _read_bounded(path)
    assert len(data) == 2 * 1024 * 1024 + 123
    assert requested
    assert max(requested) <= _CHUNK_SIZE
    assert all(n == _CHUNK_SIZE for n in requested)
    assert digest == hashlib.sha256(data).hexdigest()


def test_read_bounded_stops_when_stream_exceeds_limit_after_getsize(tmp_path, monkeypatch):
    path = tmp_path / "sneaky.json"
    path.write_bytes(b"x" * (MAX_LEGACY_FILE_BYTES + 1))
    monkeypatch.setattr(
        "auto_tune.modules.local_index.service.os.path.getsize", lambda p: 1024
    )
    with pytest.raises(_LegacyFileTooLarge):
        _read_bounded(path)


def test_read_bounded_valid_file_digest_and_content(tmp_path):
    f = tmp_path / "ok.json"
    payload = json.dumps({"experiments": []}).encode("utf-8")
    f.write_bytes(payload)
    data, digest = _read_bounded(f)
    assert data == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_import_stream_exceeding_declared_size_produces_no_records(service, tmp_path, monkeypatch):
    f = tmp_path / "sneaky.json"
    f.write_bytes(b"x" * (MAX_LEGACY_FILE_BYTES + 1))
    monkeypatch.setattr(
        "auto_tune.modules.local_index.service.os.path.getsize", lambda p: 1024
    )
    summary = service.import_legacy_files([f])
    assert summary.imported == 0
    assert summary.failed == 1
    assert summary.failures[0].error_code == "LEGACY_IMPORT_TOO_LARGE"
    assert str(tmp_path) not in summary.failures[0].message
    assert service.list_experiments(ExperimentQuery(limit=100)) == []


def test_import_preserves_source_file_bytes_mtime_sha(service, tmp_path):
    f = _write(tmp_path / "experiment_history.json", {
        "schema_version": "1.0",
        "experiments": [
            {"run_id": "manual:a", "run_name": "a", "source": "manual",
             "status": "completed", "params": {}, "metrics": {}},
        ],
    })
    before_bytes = f.read_bytes()
    before_stat = os.stat(f)
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    service.import_legacy_files([f])
    after_bytes = f.read_bytes()
    after_stat = os.stat(f)
    after_sha = hashlib.sha256(after_bytes).hexdigest()
    assert after_bytes == before_bytes
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_sha == before_sha


# ── Step 7: status ──


def test_status_available_after_initialize(service):
    service.index_dataset(_snapshot_payload())
    st = service.status()
    assert st["available"] is True
    assert st["schema_version"] == 2
    assert st["dataset_count"] == 1
    assert st["experiment_count"] == 0
    assert st["error_code"] is None


def test_status_corrupt_database_unavailable(tmp_path):
    db = tmp_path / "auto_tune.db"
    db.write_bytes(b"\x00\x01\x02 not sqlite " * 8)
    svc = LocalIndexService(_cfg(tmp_path))
    st = svc.status()
    assert st["available"] is False
    assert st["error_code"] == "LOCAL_INDEX_CORRUPT"
    assert db.read_bytes().startswith(b"\x00\x01\x02")
