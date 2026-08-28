"""Task 4: integration of the local index with finalizer, loop and split API."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.agent_engine.loop import run_tuning_loop
from auto_tune.modules.local_index import LocalIndexService
from auto_tune.modules.local_index.models import (
    ExperimentQuery,
    LocalIndexConfig,
    LocalIndexPersistenceError,
)
from auto_tune.modules.run_state.service import new_run_state
from auto_tune.modules.train_analyzer.experiment_history import ExperimentHistoryStore
from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run
from auto_tune.ui import app as app_mod
from auto_tune.ui.components.experiment_panel import get_experiment_history_view


def _cfg(tmp_path):
    return LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
        backup_max_files=3,
        busy_timeout_ms=5000,
    )


def _make_run(tmp_path, name, csv_content="epoch,metrics/mAP50(B)\n1,0.2\n"):
    run_dir = tmp_path / "detect" / name
    run_dir.mkdir(parents=True)
    (run_dir / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(csv_content, encoding="utf-8")
    return str(run_dir)


# ── Step 1-2: finalizer indexes under the S1.5 runtime run_id ──


def test_finalize_manual_indexes_under_runtime_run_id(tmp_path):
    run_dir = _make_run(tmp_path, "train1")
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()

    result = finalize_training_run(
        run_dir, "train1", "manual", {}, log_dir=str(tmp_path / "log"),
        runtime_run_id="manual:uuid-1", local_index_service=svc,
    )

    assert result["status"] == "completed"
    assert result["index_error"] is None
    got = svc.get_experiment("manual:uuid-1")
    assert got is not None
    assert got["run_id"] == "manual:uuid-1"
    assert got["params"]["_legacy_record_run_id"] == "manual:train1"


def test_finalize_multi_iteration_same_runtime_run_id_single_row(tmp_path):
    run_dir1 = _make_run(tmp_path, "autotune_1")
    run_dir2 = _make_run(tmp_path, "autotune_2")
    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    log_dir = str(tmp_path / "log")

    finalize_training_run(
        run_dir1, "autotune_1", "tuning", {}, log_dir=log_dir,
        session_id="s1", runtime_run_id="tuning:uuid-1", local_index_service=svc,
    )
    finalize_training_run(
        run_dir2, "autotune_2", "tuning", {}, log_dir=log_dir,
        session_id="s1", runtime_run_id="tuning:uuid-1", local_index_service=svc,
    )

    rows = svc.list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "tuning:uuid-1"
    # JSON history keeps the existing per-run record semantics.
    history = json.loads((tmp_path / "log" / "experiment_history.json").read_text("utf-8"))
    assert len(history["experiments"]) == 2


# ── Step 3: _manual_finalize_cb passes the S1.5 UUID identity ──


def test_manual_finalize_cb_passes_s1_5_run_id(tmp_path, monkeypatch):
    captured = {}

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        captured["runtime_run_id"] = kw.get("runtime_run_id")
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name, "source": "manual",
            "status": "completed", "analysis_status": "skipped", "metrics": {},
            "artifacts": {"report_path": None}, "error": None,
            "analysis_error": None, "history_error": None,
        }

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    state = new_run_state("manual", run_name="train9")
    controller = _FakeController(state)
    app_mod._running_training.clear()
    try:
        app_mod._manual_finalize_cb(controller, 0)
    finally:
        app_mod._running_training.clear()

    assert captured["runtime_run_id"] == state.run_id
    assert state.run_id.startswith("manual:")
    # S1.5 UUID identity, never the legacy manual:train_name scheme.
    assert captured["runtime_run_id"] != "manual:train9"


class _FakeController:
    def __init__(self, state):
        self.run_id = state.run_id
        self.train_dir = "/tmp/detect/train9"
        self.train_name = "train9"
        self.started_iso = "2026-08-01T00:00:00Z"
        self._stop_applied = False


# ── Step 4: run_tuning_loop forwards the S1.5 tuning identity ──


def test_tuning_loop_forwards_runtime_run_id(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    detect_dir = tmp_path / "detect"
    ref = detect_dir / "train38"
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text("model: yolov8n.pt\ndata: x.yaml\nlr0: 0.01\nbatch: 16\n", encoding="utf-8")
    (ref / "results.csv").write_text("epoch,metrics/mAP50(B)\n0,0.05\n", encoding="utf-8")

    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      session_id=None, audit_path=None, started_at=None, finished_at=None,
                      tuning_context=None, **kw):
        calls.append(kw.get("runtime_run_id"))
        return {
            "run_id": f"tuning:{session_id}:{run_name}", "run_name": run_name,
            "source": "tuning", "status": "completed", "analysis_status": "completed",
            "metrics": {"mAP50": 0.06}, "epochs": {"configured": 100, "completed": 3, "best": 2},
            "artifacts": {"report_path": str(tmp_path / "x_report.json")},
            "analysis_error": None, "history_error": None, "index_error": None, "error": None,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception",
                        lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: {"diagnosis": "ok", "action": "keep",
                                         "hyperparameter_changes": {}, "training_overrides": {}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_yolo_command",
                        lambda *a, **k: ["yolo", "train"])

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training",
                        lambda *a, **k: FakeProc())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.finalize_training_run", fake_finalize)

    run_tuning_loop(
        {"probe": {"max_retries": 1}}, reference_run="train38",
        log_dir=str(tmp_path), auto_analyze=True, runtime_run_id="tuning:uuid-9",
    )

    assert calls == ["tuning:uuid-9"]


# ── Step 6-7: split API registers the dataset in the index ──


def _make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    for i in range(4):
        (source / f"img{i}.jpg").write_bytes(f"image-{i}".encode())
        (source / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (source / "data.yaml").write_text("names:\n  0: defect\nnc: 1\n", encoding="utf-8")
    return source


def _use_tmp_log(monkeypatch, tmp_path):
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    monkeypatch.setattr(app_mod, "LATEST_DATASET_PATH", log_dir / "latest_dataset.json")
    monkeypatch.setattr(app_mod, "DATASET_SNAPSHOT_ROOT", log_dir / "dataset_snapshots")
    return log_dir


def test_split_endpoint_indexes_dataset(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "latest_dataset.json").write_text(
        json.dumps({"dataset_path": str(source), "split": False}), encoding="utf-8")

    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == 200
    assert "index_warning" not in resp.json()

    svc = app_mod._local_index_service()
    datasets = svc.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["snapshot_id"] == resp.json()["snapshot_id"]
    assert datasets[0]["validation_status"] == "valid"


def test_history_view_falls_back_on_native_storage_error(tmp_path):
    """A native OSError at the storage boundary surfaces as a visible
    index_warning and the history page honestly falls back to JSON records."""
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a dir")
    svc = LocalIndexService(LocalIndexConfig(
        database_path=blocker / "auto_tune.db",
        backup_dir=tmp_path / "db_backups",
    ))
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"))
    store.upsert({
        "run_id": "manual:hist1", "run_name": "hist1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"}, "metrics": {"mAP50": 0.5},
        "finished_at": "2026-08-01T00:00:00Z",
    })

    view = get_experiment_history_view(log_dir=str(tmp_path), service=svc)

    assert view["source"] == "json_fallback"
    assert view["index_warning"] is not None
    assert view["index_warning"]["error_code"] == "LOCAL_INDEX_UNAVAILABLE"
    assert any(e["run_id"] == "manual:hist1" for e in view["experiments"])


def test_split_index_failure_keeps_snapshot_success(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    log_dir = _use_tmp_log(monkeypatch, tmp_path)
    (log_dir / "latest_dataset.json").write_text(
        json.dumps({"dataset_path": str(source), "split": False}), encoding="utf-8")

    real_builder = app_mod._local_index_service

    def boom_builder():
        svc = real_builder()

        def boom(payload):
            raise LocalIndexPersistenceError("database locked")

        svc.index_dataset = boom  # type: ignore[method-assign]
        return svc

    monkeypatch.setattr(app_mod, "_local_index_service", boom_builder)
    client = TestClient(app_mod.app)
    resp = client.post("/api/dataset/split", json={"val_ratio": 0.2, "seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["index_warning"]["error_code"] == "LOCAL_INDEX_PERSIST_FAILED"
    assert data["index_warning"]["message"] == "数据集已创建，但本地索引更新失败"
