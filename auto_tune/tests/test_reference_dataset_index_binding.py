"""Bugfix P2 review fix: explicit dataset_id must never be silently replaced.

When ``LocalIndexService.index_experiment`` receives an explicit ``dataset_id``
(the frozen P2 resolution identity), a missing dataset row must fail with a
stable ``LocalIndexPersistenceError`` instead of falling back to a params.data
path match that could silently associate the experiment with another dataset.
"""

import json
import sys
from pathlib import Path

import pytest

sys.stderr.write(f"[env] {sys.executable}\n")

from auto_tune.modules.local_index.models import LocalIndexConfig, LocalIndexPersistenceError
from auto_tune.modules.local_index.service import LocalIndexService
from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run


def _cfg(tmp_path):
    return LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
        busy_timeout_ms=200,
    )


def _register_dataset(svc, dataset_id, data_yaml_path, snapshot_id):
    svc.index_dataset({
        "source_dataset_path": f"C:/data/src/{dataset_id}",
        "data_yaml_path": data_yaml_path,
        "snapshot_id": snapshot_id,
        "snapshot_valid": True,
        "display_name": f"set {dataset_id}",
        "dataset_id": dataset_id,
    })


def _make_run(tmp_path, name, data_path):
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True)
    (run_dir / "args.yaml").write_text(f"model: yolov8n.pt\ndata: {data_path}\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(
        "epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n0, 0.2, 0.1\n", encoding="utf-8"
    )
    return str(run_dir)


# ── 1. explicit dataset_id missing must fail, never write the path match ──


def test_explicit_missing_dataset_id_raises_and_writes_nothing(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    path_b = str(tmp_path / "snapshots" / "b" / "data.yaml")
    _register_dataset(svc, "ds-b", path_b, "b" * 64)

    with pytest.raises(LocalIndexPersistenceError) as ei:
        svc.index_experiment({
            "run_id": "tuning:x", "run_name": "train52", "source": "tuning",
            "status": "completed", "params": {"data": path_b}, "metrics": {},
        }, runtime_run_id="tuning:uuid-1", dataset_id="ds-a")
    assert ei.value.error_code == "LOCAL_INDEX_UNAVAILABLE"
    # No experiment record was written with a wrong association.
    assert svc.get_experiment("tuning:uuid-1") is None
    rows = svc.find_reference_experiments("train52")
    assert rows == []


# ── 2. explicit dataset_id present wins over a matching data path ──


def test_explicit_present_dataset_id_wins_over_path_match(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    path_a = str(tmp_path / "snapshots" / "a" / "data.yaml")
    path_b = str(tmp_path / "snapshots" / "b" / "data.yaml")
    _register_dataset(svc, "ds-a", path_a, "a" * 64)
    _register_dataset(svc, "ds-b", path_b, "b" * 64)

    svc.index_experiment({
        "run_id": "tuning:x", "run_name": "train52", "source": "tuning",
        "status": "completed", "params": {"data": path_b}, "metrics": {},
    }, runtime_run_id="tuning:uuid-2", dataset_id="ds-a")

    row = svc.get_experiment("tuning:uuid-2")
    assert row is not None
    assert row["dataset_id"] == "ds-a"
    assert row["dataset_id"] != "ds-b"


# ── 3. without explicit dataset_id, path matching stays compatible ──


def test_no_explicit_dataset_id_uses_path_match(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    path_b = str(tmp_path / "snapshots" / "b" / "data.yaml")
    _register_dataset(svc, "ds-b", path_b, "b" * 64)

    svc.index_experiment({
        "run_id": "tuning:x", "run_name": "train52", "source": "tuning",
        "status": "completed", "params": {"data": path_b}, "metrics": {},
    }, runtime_run_id="tuning:uuid-3")

    row = svc.get_experiment("tuning:uuid-3")
    assert row is not None
    assert row["dataset_id"] == "ds-b"


# ── 4. finalizer keeps training fact + JSON history and reports index_error ──


def test_finalizer_explicit_missing_dataset_id_keeps_facts_and_index_error(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    path_b = str(tmp_path / "snapshots" / "b" / "data.yaml")
    _register_dataset(svc, "ds-b", path_b, "b" * 64)

    run_dir = _make_run(tmp_path, "autotune_1", path_b)
    result = finalize_training_run(
        run_dir, "autotune_1", "tuning", {"train_analyzer": {}}, log_dir=str(log_dir),
        session_id="sess-1",
        runtime_run_id="tuning:uuid-4", local_index_service=svc, dataset_id="ds-a",
    )

    # Training completed fact is unchanged.
    assert result["status"] == "completed"
    # Index failure is reported independently, never as a training failure.
    assert result["index_error"] is not None
    assert result["index_error"]["error_type"] == "local_index_persistence_error"
    assert result["index_error"]["stage"] == "index"
    # JSON history was still written with the params.data fact.
    history = json.loads((log_dir / "experiment_history.json").read_text("utf-8"))
    record = next(r for r in history["experiments"] if r["run_name"] == "autotune_1")
    assert Path(record["params"]["data"]) == Path(path_b)
    # No experiment was written with a wrong dataset association.
    assert svc.get_experiment("tuning:uuid-4") is None
    assert svc.find_reference_experiments("autotune_1") == []


# ── 5. error message must not leak paths / SQL / native exceptions ──


def test_error_message_has_no_paths_or_sql(tmp_path):
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    path_b = str(tmp_path / "snapshots" / "b" / "data.yaml")
    _register_dataset(svc, "ds-b", path_b, "b" * 64)

    with pytest.raises(LocalIndexPersistenceError) as ei:
        svc.index_experiment({
            "run_id": "tuning:x", "run_name": "train52", "source": "tuning",
            "status": "completed", "params": {"data": path_b}, "metrics": {},
        }, runtime_run_id="tuning:uuid-5", dataset_id="ds-a")
    msg = str(ei.value)
    assert str(tmp_path) not in msg
    assert "snapshots" not in msg
    assert "SELECT" not in msg
    assert "INSERT" not in msg
    assert "Traceback" not in msg
    assert "traceback" not in msg
