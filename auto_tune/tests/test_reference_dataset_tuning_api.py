"""Bugfix P2 Task 3: /tuning/start freezes the resolved reference dataset.

The auto-tuning entry must resolve the reference run's own dataset snapshot
before any controller or YOLO subprocess is created, pass the frozen resolution
into the loop, and never consult the global ``latest_dataset``.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.ui import app as app_mod


def _redirect_log(monkeypatch, tmp_path):
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
    snap_root = tmp_path / "log" / "dataset_snapshots"
    snap = create_dataset_snapshot(source, snap_root, 0.2, 42, {0: "defect"})
    return source, snap


def _make_reference(tmp_path, snapshot, name="train52", data=None):
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True)
    data_yaml = data if data is not None else str(snapshot.data_yaml_path)
    (ref / "args.yaml").write_text(
        f"model: yolov8n.pt\ndata: {data_yaml}\nlr0: 0.01\nbatch: 16\nepochs: 100\n",
        encoding="utf-8",
    )
    (ref / "results.csv").write_text("epoch, metrics/mAP50(B)\n0, 0.05\n", encoding="utf-8")
    return detect


def _sse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _mock_loop(monkeypatch, capture):
    def fake_run_tuning_loop(config, **kwargs):
        capture["config"] = config
        capture["reference_run"] = kwargs.get("reference_run")
        capture["reference_dataset"] = kwargs.get("reference_dataset")
        capture["skip_execute"] = kwargs.get("skip_execute")
        return {
            "final_result": None,
            "best_iteration": None,
            "best_train_name": None,
            "best_metrics": None,
            "iterations": [],
            "eval_mode": "comprehensive",
            "error": None,
        }

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.run_tuning_loop", fake_run_tuning_loop)


@pytest.fixture
def _clean(app_mod=app_mod):
    app_mod._running_training.clear()
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()
    yield
    app_mod._running_training.clear()
    app_mod._tuning_loop_active = False
    app_mod._tuning_cancel_event.clear()


def test_tuning_start_resolves_reference_not_latest(tmp_path, monkeypatch, _clean):
    """Reference snapshot A wins over a global latest_dataset pointing at B."""
    log_dir = _redirect_log(monkeypatch, tmp_path)
    _, snap_a = _make_snapshot(tmp_path, "source_a", marker="A")
    _, snap_b = _make_snapshot(tmp_path, "source_b", marker="B")
    detect = _make_reference(tmp_path, snap_a)
    (log_dir / "latest_dataset.json").write_text(json.dumps({
        "snapshot_id": snap_b.snapshot_id,
        "data_yaml_path": str(snap_b.data_yaml_path),
        "snapshot_path": str(snap_b.snapshot_path),
    }), encoding="utf-8")

    capture = {}
    _mock_loop(monkeypatch, capture)
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "dry_run", "max_retries": 1,
    })
    assert resp.status_code == 200
    res = capture["reference_dataset"]
    assert res is not None
    assert res.snapshot_id == snap_a.snapshot_id
    assert res.snapshot_id != snap_b.snapshot_id
    assert res.resolution_source == "reference_args"
    assert os.path.normcase(str(res.data_yaml_path)) == os.path.normcase(str(snap_a.data_yaml_path))

    ref_events = [e for e in _sse_events(resp.text) if e.get("event") == "reference_dataset"]
    assert ref_events
    assert ref_events[0]["executable"] is True
    assert ref_events[0]["reference_dataset"]["snapshot_short_id"] == snap_a.snapshot_id[:8]
    assert "dataset_snapshots" not in json.dumps(ref_events[0])


def test_tuning_start_does_not_inject_config_data_yaml(tmp_path, monkeypatch, _clean):
    """The old config.training.data_yaml injection from latest_dataset is gone."""
    _redirect_log(monkeypatch, tmp_path)
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)

    capture = {}
    _mock_loop(monkeypatch, capture)
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "dry_run", "max_retries": 1,
    })
    assert resp.status_code == 200
    # The old snapshot-gate made a copied config with training.data_yaml injected;
    # P2 passes the frozen resolution instead, so the shared config is untouched.
    assert capture["config"] is app_mod.APP_CONFIG


def test_tuning_start_resolution_error_blocks_before_controller(tmp_path, monkeypatch, _clean):
    """Unresolvable reference returns a stable 4xx before any controller exists."""
    log_dir = _redirect_log(monkeypatch, tmp_path)
    _, snap = _make_snapshot(tmp_path)
    external = tmp_path / "external" / "data.yaml"
    external.parent.mkdir(parents=True)
    external.write_text("path: .\n", encoding="utf-8")
    detect = _make_reference(tmp_path, snap, data=str(external))
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "train", "max_retries": 1,
    })
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "REFERENCE_SNAPSHOT_INVALID"
    assert not (log_dir / "tuning_running.json").exists()
    assert str(tmp_path) not in resp.text
    assert "dataset_snapshots" not in resp.text
    assert "Traceback" not in resp.text
    assert "C:\\" not in resp.text


def test_tuning_start_ambiguous_returns_409(tmp_path, monkeypatch, _clean):
    """Same-name reference associated with two datasets returns a stable 409."""
    _redirect_log(monkeypatch, tmp_path)
    _, snap_a = _make_snapshot(tmp_path, "source_a", marker="A")
    _, snap_b = _make_snapshot(tmp_path, "source_b", marker="B")
    detect = _make_reference(tmp_path, snap_a)

    from auto_tune.modules.local_index.models import LocalIndexConfig
    from auto_tune.modules.local_index.service import LocalIndexService
    from uuid import uuid4

    svc = LocalIndexService(LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
    ))
    svc.initialize()
    for snap in (snap_a, snap_b):
        svc.index_dataset({
            "source_dataset_path": str(snap.source_root),
            "data_yaml_path": str(snap.data_yaml_path),
            "snapshot_id": snap.snapshot_id,
            "snapshot_valid": True,
            "display_name": f"set {snap.snapshot_id[:8]}",
        })
    for snap in (snap_a, snap_b):
        svc.index_experiment({
            "run_id": f"tuning:{uuid4()}", "run_name": "train52", "source": "tuning",
            "status": "completed", "params": {"data": str(snap.data_yaml_path)}, "metrics": {},
        }, runtime_run_id=f"tuning:{uuid4()}")

    # The API uses its own _local_index_service() against the redirected log dir,
    # which is the same database path seeded above.
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "train", "max_retries": 1,
    })
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "REFERENCE_DATASET_AMBIGUOUS"


def test_tuning_start_dry_run_unresolved_is_not_executable(tmp_path, monkeypatch, _clean):
    """Dry-run with an unresolved reference must report a non-executable plan."""
    log_dir = _redirect_log(monkeypatch, tmp_path)
    _, snap = _make_snapshot(tmp_path)
    external = tmp_path / "external" / "data.yaml"
    external.parent.mkdir(parents=True)
    external.write_text("path: .\n", encoding="utf-8")
    detect = _make_reference(tmp_path, snap, data=str(external))
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    capture = {}
    _mock_loop(monkeypatch, capture)
    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "dry_run", "max_retries": 1,
    })
    assert resp.status_code == 200
    assert capture["reference_dataset"] is None
    ref_events = [e for e in _sse_events(resp.text) if e.get("event") == "reference_dataset"]
    assert ref_events
    assert ref_events[0]["executable"] is False
    assert ref_events[0]["error_code"] == "REFERENCE_SNAPSHOT_INVALID"


def test_tuning_start_no_reference_real_training_fails(tmp_path, monkeypatch, _clean):
    """Real training without a unique reference must fail before launching."""
    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(tmp_path / "detect"))

    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": None, "mode": "train", "max_retries": 1,
    })
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "REFERENCE_RUN_INVALID"


def test_tuning_start_auto_detect_unique_reference_resolves(tmp_path, monkeypatch, _clean):
    """Auto-detected reference from the Module B report goes through resolution."""
    _redirect_log(monkeypatch, tmp_path)
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)
    # A Module B report for train52 makes it the unique auto-detect reference.
    (tmp_path / "log" / "train52_report.json").write_text(json.dumps({
        "module": "train_analyzer",
        "runs": {"train52": {"name": "train52", "args": {}, "results": {}}},
        "summary": {"best_mAP50": 0.05},
        "total_runs": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    capture = {}
    _mock_loop(monkeypatch, capture)
    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": None, "mode": "dry_run", "max_retries": 1,
    })
    assert resp.status_code == 200
    assert capture["reference_run"] == "train52"
    assert capture["reference_dataset"] is not None
    assert capture["reference_dataset"].snapshot_id == snap.snapshot_id


def test_tuning_start_sqlite_source_resolution(tmp_path, monkeypatch, _clean):
    """SQLite-indexed reference resolves with source='sqlite'."""
    log_dir = _redirect_log(monkeypatch, tmp_path)
    _, snap = _make_snapshot(tmp_path)
    detect = _make_reference(tmp_path, snap)

    from auto_tune.modules.local_index.models import LocalIndexConfig
    from auto_tune.modules.local_index.service import LocalIndexService

    svc = LocalIndexService(LocalIndexConfig(
        database_path=tmp_path / "log" / "auto_tune.db",
        backup_dir=tmp_path / "log" / "db_backups",
    ))
    svc.initialize()
    svc.index_dataset({
        "source_dataset_path": str(snap.source_root),
        "data_yaml_path": str(snap.data_yaml_path),
        "snapshot_id": snap.snapshot_id,
        "snapshot_valid": True,
        "display_name": "defect set",
    })
    svc.index_experiment({
        "run_id": "tuning:seed", "run_name": "train52", "source": "tuning",
        "status": "completed", "params": {"data": str(snap.data_yaml_path)}, "metrics": {},
    }, runtime_run_id="tuning:seed-1")
    monkeypatch.setattr(app_mod, "find_detect_dir", lambda: str(detect))

    capture = {}
    _mock_loop(monkeypatch, capture)
    client = TestClient(app_mod.app)
    resp = client.post("/tuning/start", json={
        "reference_run": "train52", "mode": "dry_run", "max_retries": 1,
    })
    assert resp.status_code == 200
    assert capture["reference_dataset"].resolution_source == "sqlite"
    assert capture["reference_dataset"].snapshot_id == snap.snapshot_id
    assert capture["reference_dataset"].dataset_display_name == "defect set"
