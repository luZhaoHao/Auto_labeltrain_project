"""Tests for the unified experiment history store."""

import json

import pytest

from auto_tune.modules.train_analyzer.experiment_history import (
    ExperimentHistoryError,
    ExperimentHistoryStore,
    make_run_id,
)


def test_make_run_id_is_stable():
    assert make_run_id("manual", "train39") == "manual:train39"
    assert make_run_id("tuning", "autotune_1", "s1") == "tuning:s1:autotune_1"


def test_make_run_id_rejects_tuning_without_session():
    with pytest.raises(ValueError):
        make_run_id("tuning", "autotune_1")


def test_upsert_creates_versioned_history(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"))
    record = {"run_id": "manual:train39", "run_name": "train39", "source": "manual"}
    store.upsert(record)
    saved = json.loads((tmp_path / "experiment_history.json").read_text("utf-8"))
    assert saved["schema_version"] == "1.0"
    assert saved["experiments"] == [record]


def test_upsert_replaces_same_run_id(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "history.json"))
    store.upsert({"run_id": "manual:train1", "source": "manual", "status": "running"})
    store.upsert({"run_id": "manual:train1", "source": "manual", "status": "completed"})
    assert len(store.list_experiments(False)) == 1
    assert store.list_experiments(False)[0]["status"] == "completed"


def test_atomic_failure_keeps_old_history(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    store = ExperimentHistoryStore(str(path))
    store.upsert({"run_id": "manual:train1", "source": "manual"})
    old = path.read_text("utf-8")
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.experiment_history.atomic_write_json",
        lambda *a: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        store.upsert({"run_id": "manual:train2", "source": "manual"})
    assert path.read_text("utf-8") == old


def test_upsert_sorts_finished_at_with_missing_last(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "history.json"))
    store.upsert({"run_id": "manual:a", "run_name": "a", "source": "manual", "finished_at": "2026-08-01T00:00:00Z"})
    store.upsert({"run_id": "manual:no_ts", "run_name": "no_ts", "source": "manual"})
    store.upsert({"run_id": "manual:b", "run_name": "b", "source": "manual", "finished_at": "2026-08-05T00:00:00Z"})
    names = [e["run_name"] for e in store.list_experiments(False)]
    assert names == ["b", "a", "no_ts"]


def test_legacy_tuning_included_once_and_legacy_file_unchanged(tmp_path):
    legacy = tmp_path / "tuning_history.json"
    legacy_data = [
        {"iteration": 1, "train_name": "autotune_1", "timestamp": "2026-08-01T00:00:00Z"},
        {"iteration": 2, "timestamp": "2026-08-02T00:00:00Z"},
    ]
    legacy.write_text(json.dumps(legacy_data), encoding="utf-8")
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"), str(legacy))

    experiments = store.list_experiments(True)
    assert len(experiments) == 2
    run_ids = [e["run_id"] for e in experiments]
    assert "legacy-tuning:autotune_1" in run_ids
    assert any(rid.startswith("legacy-tuning:1:") for rid in run_ids)
    assert store.list_experiments(False) == []
    assert legacy.read_text("utf-8") == json.dumps(legacy_data)


def test_new_record_wins_over_legacy_with_same_run_name(tmp_path):
    legacy = tmp_path / "tuning_history.json"
    legacy.write_text(json.dumps([
        {"iteration": 1, "train_name": "autotune_1", "timestamp": "2026-08-01T00:00:00Z"},
    ]), encoding="utf-8")
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"), str(legacy))
    store.upsert({
        "run_id": "tuning:s1:autotune_1",
        "run_name": "autotune_1",
        "source": "tuning",
        "status": "completed",
    })
    experiments = store.list_experiments(True)
    run_ids = [e["run_id"] for e in experiments]
    assert run_ids == ["tuning:s1:autotune_1"]


def test_load_raises_on_corrupt_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{invalid json", encoding="utf-8")
    store = ExperimentHistoryStore(str(path))
    with pytest.raises(ExperimentHistoryError):
        store.load()


def test_load_raises_on_invalid_schema(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    store = ExperimentHistoryStore(str(path))
    with pytest.raises(ExperimentHistoryError):
        store.load()


def test_upsert_does_not_overwrite_corrupt_file(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{invalid", encoding="utf-8")
    store = ExperimentHistoryStore(str(path))
    with pytest.raises(ExperimentHistoryError):
        store.upsert({"run_id": "manual:x", "run_name": "x", "source": "manual"})
    assert path.read_text("utf-8") == "{invalid"


def test_sort_handles_local_utc_and_missing(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "history.json"))
    store.upsert({"run_id": "manual:local", "run_name": "local", "source": "manual", "finished_at": "2026-08-01T12:00:00"})
    store.upsert({"run_id": "manual:utc", "run_name": "utc", "source": "manual", "finished_at": "2026-08-01T12:00:00Z"})
    store.upsert({"run_id": "manual:none", "run_name": "none", "source": "manual"})
    store.upsert({"run_id": "manual:utc_late", "run_name": "utc_late", "source": "manual", "finished_at": "2026-08-02T00:00:00Z"})

    names = [e["run_name"] for e in store.list_experiments(False)]

    assert names == ["utc_late", "utc", "local", "none"]


def test_legacy_status_mapping_fixture(tmp_path):
    legacy = tmp_path / "tuning_history.json"
    legacy.write_text(json.dumps([
        {"iteration": 1, "train_name": "a", "timestamp": "2026-08-01T00:00:00Z", "result_mAP50": 0.5},
        {"iteration": 2, "train_name": "b", "timestamp": "2026-08-01T00:00:00Z", "error": "决策失败: x"},
        {"iteration": 3, "train_name": "c", "timestamp": "2026-08-01T00:00:00Z", "error": "用户取消训练"},
        {"iteration": 4, "timestamp": "2026-08-01T00:00:00Z"},
    ]), encoding="utf-8")
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"), str(legacy))

    by_name = {e.get("run_name"): e for e in store.list_experiments(True)}

    assert by_name["a"]["status"] == "completed"
    assert by_name["a"]["analysis_status"] == "completed"
    assert by_name["a"]["metrics"]["mAP50"] == 0.5
    assert by_name["b"]["status"] == "failed"
    assert by_name["c"]["status"] == "cancelled"
    unknown = [e for e in store.list_experiments(True) if not e.get("run_name")]
    assert unknown and unknown[0]["status"] == "unknown"
