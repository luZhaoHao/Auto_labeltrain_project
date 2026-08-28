"""Task 2: dataset/experiment/artifact repository (Studio S2 Core)."""

import os
from pathlib import Path

import pytest

from auto_tune.modules.local_index.database import connect_database, initialize_database
from auto_tune.modules.local_index.models import (
    ArtifactRecord,
    DatasetRecord,
    ExperimentQuery,
    ExperimentRecord,
    LocalIndexCorruptError,
    LocalIndexPersistenceError,
)
from auto_tune.modules.local_index.repository import LocalIndexRepository


def _cfg(tmp_path, **kw):
    from auto_tune.modules.local_index.models import LocalIndexConfig

    return LocalIndexConfig(
        database_path=tmp_path / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
        backup_max_files=kw.get("backup_max_files", 3),
        busy_timeout_ms=kw.get("busy_timeout_ms", 5000),
    )


@pytest.fixture
def repo(tmp_path):
    cfg = _cfg(tmp_path, busy_timeout_ms=200)
    initialize_database(cfg)
    conn = connect_database(cfg)
    yield LocalIndexRepository(conn)
    conn.close()


def _dataset(dataset_id="d1", canonical_path=None, **kw):
    return DatasetRecord(
        dataset_id=dataset_id,
        display_name=kw.get("display_name", f"Dataset {dataset_id}"),
        canonical_path=canonical_path or f"C:/data/ds/{dataset_id}",
        data_yaml_path=kw.get("data_yaml_path"),
        snapshot_id=kw.get("snapshot_id"),
        snapshot_digest=kw.get("snapshot_digest"),
        validation_status=kw.get("validation_status", "valid"),
        created_at=kw.get("created_at", "2026-08-01T00:00:00Z"),
        updated_at=kw.get("updated_at", "2026-08-01T00:00:00Z"),
        last_used_at=kw.get("last_used_at"),
    )


def _experiment(run_id="manual:1", source="manual", **kw):
    return ExperimentRecord(
        run_id=run_id,
        source=source,
        run_name=kw.get("run_name", run_id.rsplit(":", 1)[-1]),
        dataset_id=kw.get("dataset_id"),
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


# ── Step 1: dataset idempotency and conflicts ──


def test_upsert_dataset_updates_same_id(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1", display_name="Old"))
    repo.upsert_dataset(_dataset(dataset_id="d1", display_name="New"))
    got = repo.get_dataset("d1")
    assert got is not None
    assert got.display_name == "New"
    assert len(repo.list_datasets()) == 1


def test_upsert_dataset_same_canonical_path_rejected(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1", canonical_path="C:/data/x"))
    with pytest.raises(LocalIndexPersistenceError):
        repo.upsert_dataset(_dataset(dataset_id="d2", canonical_path="C:/data/x"))


def test_snapshot_id_cannot_bind_two_datasets(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1", snapshot_id="snap1"))
    with pytest.raises(LocalIndexPersistenceError):
        repo.upsert_dataset(_dataset(dataset_id="d2", snapshot_id="snap1"))


def test_last_used_at_allowed_none(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1", last_used_at=None))
    got = repo.get_dataset("d1")
    assert got.last_used_at is None


def test_list_datasets_sorted_by_last_used_desc(repo):
    repo.upsert_dataset(_dataset(dataset_id="old", last_used_at="2026-08-01T00:00:00Z"))
    repo.upsert_dataset(_dataset(dataset_id="new", last_used_at="2026-08-05T00:00:00Z"))
    names = [d.dataset_id for d in repo.list_datasets()]
    assert names == ["new", "old"]


def test_find_dataset_by_data_yaml(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1", data_yaml_path="C:/data/ds/data.yaml"))
    found = repo.find_dataset_by_data_yaml("C:/data/ds/data.yaml")
    assert found is not None and found.dataset_id == "d1"


# ── Step 4: experiment/artifact idempotency and filtering ──


def test_upsert_experiment_replaces_same_runtime_run_id(repo):
    repo.upsert_experiment(_experiment(run_id="manual:uuid1", status="running"))
    repo.upsert_experiment(_experiment(run_id="manual:uuid1", status="completed"))
    rows = repo.list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_upsert_experiment_keeps_both_sources(repo):
    repo.upsert_experiment(_experiment(run_id="manual:1", source="manual"))
    repo.upsert_experiment(_experiment(run_id="tuning:uuid:autotune_1", source="tuning"))
    rows = repo.list_experiments(ExperimentQuery(limit=100))
    assert {r["source"] for r in rows} == {"manual", "tuning"}


def test_list_experiments_filters_by_dataset_source_status(repo):
    repo.upsert_dataset(_dataset(dataset_id="d1"))
    repo.upsert_dataset(_dataset(dataset_id="d2"))
    repo.upsert_experiment(_experiment(
        run_id="manual:1", source="manual", dataset_id="d1", status="completed"))
    repo.upsert_experiment(_experiment(
        run_id="manual:2", source="manual", dataset_id="d1", status="failed"))
    repo.upsert_experiment(_experiment(
        run_id="tuning:u:at1", source="tuning", dataset_id="d2", status="completed"))

    assert len(repo.list_experiments(ExperimentQuery(dataset_id="d1"))) == 2
    assert len(repo.list_experiments(ExperimentQuery(source="tuning"))) == 1
    assert len(repo.list_experiments(ExperimentQuery(status="failed"))) == 1
    assert len(repo.list_experiments(
        ExperimentQuery(dataset_id="d1", source="manual", status="completed"))) == 1


def test_list_experiments_orders_by_time_desc(repo):
    repo.upsert_experiment(_experiment(
        run_id="manual:early", finished_at="2026-08-01T00:00:00Z"))
    repo.upsert_experiment(_experiment(
        run_id="manual:late", finished_at="2026-08-05T00:00:00Z"))
    rows = repo.list_experiments(ExperimentQuery(limit=100))
    assert [r["run_id"] for r in rows] == ["manual:late", "manual:early"]


@pytest.mark.parametrize("kwargs", [
    {"limit": 0},
    {"limit": 501},
    {"limit": -1},
    {"limit": True},
    {"source": "bogus"},
    {"status": "bogus"},
])
def test_invalid_query_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        ExperimentQuery(**kwargs)


def test_artifacts_replay_no_duplicate(repo, tmp_path):
    report = tmp_path / "log" / "train1_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    artifacts = (ArtifactRecord(run_id="manual:1", kind="report", path=str(report)),)
    repo.upsert_experiment(_experiment(run_id="manual:1"), artifacts=artifacts)
    repo.upsert_experiment(_experiment(run_id="manual:1"), artifacts=artifacts)
    got = repo.get_experiment("manual:1")
    reports = [a for a in got["artifacts"] if a["kind"] == "report"]
    assert len(reports) == 1


def test_directory_artifact_reported_exists(repo, tmp_path):
    run_dir = tmp_path / "detect" / "train1"
    run_dir.mkdir(parents=True)
    artifacts = (ArtifactRecord(run_id="manual:1", kind="run_dir", path=str(run_dir)),)
    repo.upsert_experiment(_experiment(run_id="manual:1"), artifacts=artifacts)
    got = repo.get_experiment("manual:1")
    assert got["artifacts"][0]["exists_state"] == "exists"


def test_missing_artifact_reported_missing_record_kept(repo, tmp_path):
    missing = str(tmp_path / "log" / "does_not_exist.json")
    artifacts = (ArtifactRecord(run_id="manual:1", kind="report", path=missing),)
    repo.upsert_experiment(_experiment(run_id="manual:1"), artifacts=artifacts)
    got = repo.get_experiment("manual:1")
    assert got is not None
    assert got["artifacts"][0]["exists_state"] == "missing"
    assert os.path.exists(missing) is False


def test_experiment_decodes_params_metrics_error(repo):
    repo.upsert_experiment(_experiment(
        run_id="manual:1",
        params={"epochs": 100, "data": "C:/data/x.yaml"},
        metrics={"mAP50": 0.5},
        error={"error_type": "boom"},
    ))
    got = repo.get_experiment("manual:1")
    assert got["params"]["epochs"] == 100
    assert got["metrics"]["mAP50"] == 0.5
    assert got["error"]["error_type"] == "boom"


def test_invalid_experiment_json_raises_corrupt(repo):
    repo.upsert_experiment(_experiment(run_id="manual:1"))
    conn = repo._conn
    conn.execute("UPDATE experiments SET params_json='{not json' WHERE run_id='manual:1'")
    conn.commit()
    with pytest.raises(LocalIndexCorruptError):
        repo.get_experiment("manual:1")


# ── Step 6: database lock / concurrent writes ──


def test_write_lock_busy_timeout_then_succeeds(tmp_path):
    cfg = _cfg(tmp_path, busy_timeout_ms=200)
    initialize_database(cfg)
    conn1 = connect_database(cfg)
    conn2 = connect_database(cfg)
    try:
        repo1 = LocalIndexRepository(conn1)
        repo2 = LocalIndexRepository(conn2)
        conn1.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(LocalIndexPersistenceError):
                repo2.upsert_dataset(_dataset(dataset_id="d2"))
        finally:
            conn1.rollback()
        # Lock released: the same write now succeeds.
        repo2.upsert_dataset(_dataset(dataset_id="d2"))
        assert repo2.get_dataset("d2") is not None
    finally:
        conn1.close()
        conn2.close()
