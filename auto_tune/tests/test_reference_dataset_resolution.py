"""Bugfix P2 Task 1-2: reference dataset resolution models, errors and SQLite queries.

Task 1 freezes the resolution model, the stable error contract and the
parameterized ``run_name`` query boundary on the local index. Task 2 adds the
strict resolution service. Both are developed TDD here.
"""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.stderr.write(f"[env] {sys.executable}\n")

from auto_tune.modules.local_index.database import connect_database, initialize_database
from auto_tune.modules.local_index.models import (
    DatasetRecord,
    ExperimentRecord,
    LocalIndexConfig,
    LocalIndexCorruptError,
)
from auto_tune.modules.local_index.repository import LocalIndexRepository
from auto_tune.modules.local_index.service import LocalIndexService


def _cfg(tmp_path):
    return LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
        backup_max_files=3,
        busy_timeout_ms=200,
    )


def _dataset(dataset_id="d1", snapshot_id=None, data_yaml_path=None, **kw):
    return DatasetRecord(
        dataset_id=dataset_id,
        display_name=kw.get("display_name", f"Dataset {dataset_id}"),
        canonical_path=kw.get("canonical_path", f"C:/data/ds/{dataset_id}"),
        data_yaml_path=data_yaml_path,
        snapshot_id=snapshot_id,
        snapshot_digest=kw.get("snapshot_digest"),
        validation_status=kw.get("validation_status", "valid"),
        created_at=kw.get("created_at", "2026-08-01T00:00:00Z"),
        updated_at=kw.get("updated_at", "2026-08-01T00:00:00Z"),
        last_used_at=kw.get("last_used_at"),
    )


def _experiment(run_id="manual:1", run_name=None, dataset_id=None, **kw):
    return ExperimentRecord(
        run_id=run_id,
        source=kw.get("source", "manual"),
        run_name=run_name or run_id.rsplit(":", 1)[-1],
        dataset_id=dataset_id,
        status=kw.get("status", "completed"),
        phase=kw.get("phase", "terminal"),
        model_name=kw.get("model_name", "yolov8n.pt"),
        task_type=kw.get("task_type", "detect"),
        started_at=kw.get("started_at", "2026-08-01T00:00:00Z"),
        finished_at=kw.get("finished_at", "2026-08-01T01:00:00Z"),
        params=kw.get("params", {}),
        metrics=kw.get("metrics", {}),
        analysis_status=kw.get("analysis_status", "completed"),
        error=kw.get("error"),
        updated_at=kw.get("updated_at", "2026-08-01T01:00:00Z"),
    )


def _seed_repo(conn, datasets, experiments):
    repo = LocalIndexRepository(conn)
    for d in datasets:
        repo.upsert_dataset(d)
    for e in experiments:
        repo.upsert_experiment(e)
    return repo


# ── Task 1: models ──


def test_resolution_model_is_frozen_with_required_fields():
    from auto_tune.modules.reference_dataset.models import ReferenceDatasetResolution

    res = ReferenceDatasetResolution(
        reference_run="train52",
        dataset_id="ds-a",
        snapshot_id="a" * 64,
        data_yaml_path=Path("log/dataset_snapshots/a/data.yaml"),
        resolution_source="sqlite",
    )
    assert res.reference_run == "train52"
    assert res.dataset_id == "ds-a"
    assert res.snapshot_id == "a" * 64
    assert res.data_yaml_path == Path("log/dataset_snapshots/a/data.yaml")
    assert res.resolution_source == "sqlite"
    assert res.dataset_display_name is None
    assert res.index_warning is None
    with pytest.raises(Exception):
        res.dataset_id = "mutated"  # frozen


def test_resolution_error_subclasses_carry_stable_codes_and_status():
    from auto_tune.modules.reference_dataset import models as m

    assert m.ReferenceRunInvalidError.error_code == "REFERENCE_RUN_INVALID"
    assert m.ReferenceRunInvalidError.status_code == 400
    assert m.ReferenceDatasetUnresolvedError.error_code == "REFERENCE_DATASET_UNRESOLVED"
    assert m.ReferenceDatasetUnresolvedError.status_code == 400
    assert m.ReferenceDatasetAmbiguousError.error_code == "REFERENCE_DATASET_AMBIGUOUS"
    assert m.ReferenceDatasetAmbiguousError.status_code == 409
    assert m.ReferenceSnapshotInvalidError.error_code == "REFERENCE_SNAPSHOT_INVALID"
    assert m.ReferenceSnapshotInvalidError.status_code == 400
    assert m.LocalIndexUnavailableError.error_code == "LOCAL_INDEX_UNAVAILABLE"
    assert m.LocalIndexUnavailableError.status_code == 503
    for cls in (
        m.ReferenceRunInvalidError,
        m.ReferenceDatasetUnresolvedError,
        m.ReferenceDatasetAmbiguousError,
        m.ReferenceSnapshotInvalidError,
        m.LocalIndexUnavailableError,
    ):
        assert issubclass(cls, m.ReferenceDatasetError)
        exc = cls("some message")
        assert str(exc) == "some message"
        assert exc.message == "some message"


# ── Task 1: parameterized run_name query (repository) ──


@pytest.fixture
def seeded(tmp_path):
    cfg = _cfg(tmp_path)
    initialize_database(cfg)
    conn = connect_database(cfg)
    yield cfg, conn
    conn.close()


def test_repo_list_experiments_by_run_name_returns_matching_rows(seeded):
    cfg, conn = seeded
    _seed_repo(conn, [
        _dataset("ds-a", snapshot_id="a" * 64, data_yaml_path="log/dataset_snapshots/a/data.yaml"),
        _dataset("ds-b", snapshot_id="b" * 64, data_yaml_path="log/dataset_snapshots/b/data.yaml"),
    ], [
        _experiment("tuning:1", run_name="train52", dataset_id="ds-a"),
        _experiment("tuning:2", run_name="train52", dataset_id="ds-a"),
        _experiment("manual:3", run_name="train53", dataset_id="ds-b"),
    ])
    repo = LocalIndexRepository(conn)

    rows = repo.list_experiments_by_run_name("train52")
    assert [r["run_id"] for r in rows] == ["tuning:1", "tuning:2"]
    assert all(r["dataset_id"] == "ds-a" for r in rows)

    assert repo.list_experiments_by_run_name("train53")[0]["dataset_id"] == "ds-b"
    assert repo.list_experiments_by_run_name("missing") == []


def test_service_find_reference_experiments_returns_projection(seeded):
    cfg, conn = seeded
    _seed_repo(conn, [
        _dataset("ds-a", snapshot_id="a" * 64),
    ], [
        _experiment("tuning:1", run_name="train52", dataset_id="ds-a"),
    ])
    conn.close()

    svc = LocalIndexService(cfg)
    rows = svc.find_reference_experiments("train52")
    assert len(rows) == 1
    assert rows[0]["run_name"] == "train52"
    assert rows[0]["dataset_id"] == "ds-a"
    assert rows[0]["run_id"] == "tuning:1"


def test_reference_query_never_uses_string_concat(seeded):
    """A run_name with SQL metacharacters must be treated as a literal value."""
    cfg, conn = seeded
    _seed_repo(conn, [], [
        _experiment("manual:1", run_name="train52"),
    ])
    repo = LocalIndexRepository(conn)

    rows = repo.list_experiments_by_run_name("train52' OR '1'='1")
    assert rows == []
    rows = repo.list_experiments_by_run_name("train52")
    assert len(rows) == 1


def test_reference_query_empty_dataset_id_not_hidden(seeded):
    """Empty/None dataset_id associations must surface honestly, not as success."""
    cfg, conn = seeded
    _seed_repo(conn, [], [
        _experiment("manual:1", run_name="train52", dataset_id=None),
    ])
    repo = LocalIndexRepository(conn)
    rows = repo.list_experiments_by_run_name("train52")
    assert len(rows) == 1
    assert rows[0]["dataset_id"] is None


def test_reference_query_storage_error_not_disguised(seeded):
    """A corrupt database must raise the stable index error, never fake empty."""
    cfg, conn = seeded
    _seed_repo(conn, [], [])
    conn.close()
    Path(cfg.database_path).write_bytes(b"not a real sqlite database")

    from auto_tune.modules.local_index import LocalIndexError

    svc = LocalIndexService(cfg)
    with pytest.raises(LocalIndexError):
        svc.find_reference_experiments("train52")


# ── Task 2: strict resolution service ──


def _make_source(tmp_path, name="source", marker=None):
    source = tmp_path / name
    source.mkdir(exist_ok=True)
    tag = marker if marker is not None else name
    for i in range(4):
        (source / f"img{i}.jpg").write_bytes(f"image-{tag}-{i}".encode())
        (source / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (source / "data.yaml").write_text("names:\n  0: defect\nnc: 1\n", encoding="utf-8")
    return source


def _make_snapshot(tmp_path, name="source", marker=None):
    from auto_tune.modules.dataset_snapshot import create_dataset_snapshot

    source = _make_source(tmp_path, name, marker=marker)
    snap = create_dataset_snapshot(source, tmp_path / "dataset_snapshots", 0.2, 42, {0: "defect"})
    return source, snap


def _make_reference(tmp_path, snapshot, name="train52", data=None, with_results=True):
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True)
    data_yaml = data if data is not None else str(snapshot.data_yaml_path)
    (ref / "args.yaml").write_text(
        f"model: yolov8n.pt\ndata: {data_yaml}\nlr0: 0.01\nbatch: 16\nepochs: 100\n",
        encoding="utf-8",
    )
    if with_results:
        (ref / "results.csv").write_text("epoch, metrics/mAP50(B)\n0, 0.05\n", encoding="utf-8")
    return detect


def _resolve(tmp_path, detect, log_dir=None, service=None, run="train52"):
    from auto_tune.modules.reference_dataset import resolve_reference_dataset

    return resolve_reference_dataset(
        run, detect, log_dir or tmp_path, local_index_service=service
    )


def _seed_index(tmp_path, snapshot, dataset_id=None, run_name="train52", extra_dataset_id=None):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    svc.index_dataset({
        "source_dataset_path": str(snapshot.source_root),
        "data_yaml_path": str(snapshot.data_yaml_path),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_valid": True,
        "display_name": "defect set",
        "dataset_id": dataset_id,
    })
    if extra_dataset_id:
        svc.index_dataset({
            "source_dataset_path": str(snapshot.source_root),
            "data_yaml_path": str(snapshot.data_yaml_path),
            "snapshot_id": "b" * 64,
            "snapshot_valid": True,
            "display_name": "other set",
            "dataset_id": extra_dataset_id,
        })
    if run_name:
        svc.index_experiment({
            "run_id": "tuning:seed",
            "run_name": run_name,
            "source": "tuning",
            "status": "completed",
            "params": {"data": str(snapshot.data_yaml_path)},
            "metrics": {"mAP50": 0.5},
        }, runtime_run_id=f"tuning:{uuid4()}")
    return svc


def test_resolve_sqlite_unique_association_success(tmp_path):
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    svc = _seed_index(tmp_path, snap)

    res = _resolve(tmp_path, detect, service=svc)

    assert res.resolution_source == "sqlite"
    assert res.snapshot_id == snap.snapshot_id
    assert res.reference_run == "train52"
    assert res.data_yaml_path == snap.data_yaml_path
    assert res.dataset_display_name == "defect set"


def test_resolve_sqlite_ambiguous_multiple_ids(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceDatasetAmbiguousError

    _, snap_a = _make_snapshot(tmp_path, "source_a", marker="A")
    _, snap_b = _make_snapshot(tmp_path, "source_b", marker="B")
    detect = _make_reference(tmp_path, snap_a)
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    for snap in (snap_a, snap_b):
        svc.index_dataset({
            "source_dataset_path": str(snap.source_root),
            "data_yaml_path": str(snap.data_yaml_path),
            "snapshot_id": snap.snapshot_id,
            "snapshot_valid": True,
            "display_name": f"set {snap.snapshot_id[:8]}",
        })
    for i, snap in enumerate((snap_a, snap_b)):
        svc.index_experiment({
            "run_id": f"tuning:{i}",
            "run_name": "train52",
            "source": "tuning",
            "status": "completed",
            "params": {"data": str(snap.data_yaml_path)},
            "metrics": {},
        }, runtime_run_id=f"tuning:{uuid4()}")

    with pytest.raises(ReferenceDatasetAmbiguousError) as ei:
        _resolve(tmp_path, detect, service=svc)
    assert ei.value.error_code == "REFERENCE_DATASET_AMBIGUOUS"
    assert ei.value.status_code == 409


def test_resolve_sqlite_missing_relations_falls_back_to_args(tmp_path):
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    svc = _seed_index(tmp_path, snap, run_name=None)  # no experiment row

    res = _resolve(tmp_path, detect, service=svc)

    assert res.resolution_source == "reference_args"
    assert res.snapshot_id == snap.snapshot_id


def test_resolve_sqlite_and_args_same_snapshot(tmp_path):
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    svc = _seed_index(tmp_path, snap)

    res = _resolve(tmp_path, detect, service=svc)

    assert res.resolution_source == "sqlite"
    assert res.snapshot_id == snap.snapshot_id


def test_resolve_sqlite_and_args_conflict(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceDatasetAmbiguousError

    _, snap_a = _make_snapshot(tmp_path, "source_a", marker="A")
    _, snap_b = _make_snapshot(tmp_path, "source_b", marker="B")
    detect = _make_reference(tmp_path, snap_a, data=str(snap_b.data_yaml_path))
    svc = _seed_index(tmp_path, snap_a)

    with pytest.raises(ReferenceDatasetAmbiguousError) as ei:
        _resolve(tmp_path, detect, service=svc)
    assert ei.value.error_code == "REFERENCE_DATASET_AMBIGUOUS"


def test_resolve_args_fallback_valid_without_index(tmp_path):
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)

    res = _resolve(tmp_path, detect, service=None)

    assert res.resolution_source == "reference_args"
    assert res.snapshot_id == snap.snapshot_id
    assert res.data_yaml_path == snap.data_yaml_path


def test_resolve_args_fallback_corrupt_manifest(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    Path(snap.manifest_path).write_text("{broken", encoding="utf-8")

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"


def test_resolve_args_fallback_digest_mismatch(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    manifest = json.loads(Path(snap.manifest_path).read_text(encoding="utf-8"))
    manifest["train_count"] += 1
    Path(snap.manifest_path).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"


def test_resolve_args_fallback_external_path(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    external = tmp_path / "external" / "data.yaml"
    external.parent.mkdir(parents=True)
    external.write_text("path: .\n", encoding="utf-8")
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap, data=str(external))

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"


def test_resolve_args_fallback_escape_path(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    _, snap = _make_snapshot(tmp_path)
    escape = tmp_path / "dataset_snapshots" / ".." / "escape" / "data.yaml"
    detect = _make_reference(tmp_path, snap, data=str(escape))

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"


def test_resolve_args_fallback_missing_data(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceDatasetUnresolvedError

    _, snap = _make_snapshot(tmp_path)
    detect = tmp_path / "detect"
    ref = detect / "train52"
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")

    with pytest.raises(ReferenceDatasetUnresolvedError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_DATASET_UNRESOLVED"


def test_resolve_reference_run_invalid(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceRunInvalidError

    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)

    with pytest.raises(ReferenceRunInvalidError) as ei:
        _resolve(tmp_path, detect, service=None, run="missing")
    assert ei.value.error_code == "REFERENCE_RUN_INVALID"

    with pytest.raises(ReferenceRunInvalidError):
        _resolve(tmp_path, detect, service=None, run="../../etc")


def test_resolve_reference_args_yaml_missing(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceRunInvalidError

    _, snap = _make_snapshot(tmp_path)
    detect = tmp_path / "detect"
    (detect / "train52").mkdir(parents=True)

    with pytest.raises(ReferenceRunInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_RUN_INVALID"


def test_resolve_ignores_latest_dataset(tmp_path):
    """A global latest_dataset pointing elsewhere must never be used."""
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    (tmp_path / "latest_dataset.json").write_text(json.dumps({
        "snapshot_id": "b" * 64,
        "data_yaml_path": str(tmp_path / "dataset_snapshots" / ("b" * 64) / "data.yaml"),
    }), encoding="utf-8")

    res = _resolve(tmp_path, detect, service=None)

    assert res.snapshot_id == snap.snapshot_id
    assert res.snapshot_id != "b" * 64
    assert "latest" not in res.snapshot_id


def test_resolve_never_returns_latest_when_reference_gone(tmp_path):
    """Deleting the reference snapshot must fail, not silently use latest."""
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    _, snap_a = _make_snapshot(tmp_path, "source_a")
    _, snap_b = _make_snapshot(tmp_path, "source_b")
    detect = _make_reference(tmp_path, snap_a)
    (tmp_path / "latest_dataset.json").write_text(json.dumps({
        "snapshot_id": snap_b.snapshot_id,
        "data_yaml_path": str(snap_b.data_yaml_path),
    }), encoding="utf-8")

    import shutil
    shutil.rmtree(snap_a.snapshot_path)

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"


def test_resolve_sqlite_corrupt_db_falls_back_to_args(tmp_path):
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    svc = _seed_index(tmp_path, snap)
    Path(_cfg(tmp_path).database_path).write_bytes(b"not a real sqlite database")

    res = _resolve(tmp_path, detect, service=svc)

    assert res.resolution_source == "reference_args"
    assert res.snapshot_id == snap.snapshot_id
    assert res.index_warning is not None


def test_resolve_corrupt_db_and_invalid_args_raises_local_index_unavailable(tmp_path):
    from auto_tune.modules.reference_dataset import LocalIndexUnavailableError

    _, snap = _make_snapshot(tmp_path)
    detect = tmp_path / "detect"
    ref = detect / "train52"
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")
    svc = _seed_index(tmp_path, snap)
    Path(_cfg(tmp_path).database_path).write_bytes(b"not a real sqlite database")

    with pytest.raises(LocalIndexUnavailableError) as ei:
        _resolve(tmp_path, detect, service=svc)
    assert ei.value.error_code == "LOCAL_INDEX_UNAVAILABLE"
    assert ei.value.status_code == 503


def test_resolve_error_messages_contain_no_paths(tmp_path):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    Path(snap.manifest_path).write_text("{broken", encoding="utf-8")

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    msg = str(ei.value)
    assert str(tmp_path) not in msg
    assert "dataset_snapshots" not in msg
    assert "C:\\" not in msg
    assert "Traceback" not in msg


def test_resolve_reparse_translates_to_snapshot_invalid(tmp_path, monkeypatch):
    from auto_tune.modules.reference_dataset import ReferenceSnapshotInvalidError

    def boom(*args, **kwargs):
        from auto_tune.modules.dataset_snapshot import SnapshotValidationError

        raise SnapshotValidationError("reparse point not allowed")

    # Patch after materializing the snapshot so snapshot creation itself is
    # unaffected; only the resolver's reparse re-check must be rejected.
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    monkeypatch.setattr("auto_tune.modules.dataset_snapshot.service._reject_reparse_up_to", boom)

    with pytest.raises(ReferenceSnapshotInvalidError) as ei:
        _resolve(tmp_path, detect, service=None)
    assert ei.value.error_code == "REFERENCE_SNAPSHOT_INVALID"
