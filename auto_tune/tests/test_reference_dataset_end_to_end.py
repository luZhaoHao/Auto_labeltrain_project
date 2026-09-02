"""Bugfix P2 Task 4: training facts, audit, JSON history and SQLite all bind to
the resolved reference dataset identity — never the global latest_dataset."""

import json
import os
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine.loop import run_tuning_loop
from auto_tune.modules.local_index import LocalIndexService
from auto_tune.modules.local_index.models import LocalIndexConfig, LocalIndexPersistenceError
from auto_tune.modules.reference_dataset import resolve_reference_dataset


def _cfg(tmp_path):
    log_dir = tmp_path / "log"
    return LocalIndexConfig(
        database_path=log_dir / "auto_tune.db",
        backup_dir=log_dir / "db_backups",
        busy_timeout_ms=200,
    )


def _make_snapshot(tmp_path, marker):
    from auto_tune.modules.dataset_snapshot import create_dataset_snapshot

    source = tmp_path / f"source_{marker}"
    source.mkdir(exist_ok=True)
    for i in range(4):
        (source / f"img{i}.jpg").write_bytes(f"image-{marker}-{i}".encode())
        (source / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snap = create_dataset_snapshot(source, tmp_path / "log" / "dataset_snapshots", 0.2, 42, {0: "defect"})
    return snap


def _make_reference(tmp_path, snapshot, name="train52"):
    detect = tmp_path / "detect"
    ref = detect / name
    ref.mkdir(parents=True)
    (ref / "args.yaml").write_text(
        f"model: yolov8n.pt\ndata: {snapshot.data_yaml_path}\nlr0: 0.01\nbatch: 16\nepochs: 1\n",
        encoding="utf-8",
    )
    (ref / "results.csv").write_text("epoch, metrics/mAP50(B)\n0, 0.05\n", encoding="utf-8")
    return detect


def _valid_decision():
    return {
        "diagnosis": "ok", "action": "apply changes",
        "hyperparameter_changes": {"lr0": 0.001}, "training_overrides": {},
        "raw_response": "{}", "error": None,
    }


def _valid_fact_package():
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:test",
        "task": "detect",
        "reference_run": "train52",
        "sources": {
            "dataset_report": "dataset_report_1.json",
            "training_report": "train52_report.json",
            "metrics": "results.csv",
            "params": "args.yaml",
        },
        "facts": [{"fact_id": "training.params.lr0", "value": 0.01, "source": "params"}],
    }


def _loop_mocks(monkeypatch, detect_dir, launch_side_effect):
    from auto_tune.modules.agent_engine.probe_monitor import ProbeDecision

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.find_detect_dir", lambda: str(detect_dir))
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_perception",
                        lambda **k: {"dataset": {"total_images": 10}})
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.build_tuning_fact_package",
                        lambda *a, **k: _valid_fact_package())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.decide_hyperparameters",
                        lambda *a, **k: _valid_decision())
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.validate_training_preflight",
                        lambda *a, **k: [])

    captured = {"command": None}

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

    def fake_launch(train_name, args_path, merged_params, command=None):
        captured["command"] = list(command)
        out_dir = Path(args_path).parent
        (out_dir / "results.csv").write_text(
            "epoch, metrics/mAP50(B), metrics/mAP50-95(B), metrics/precision(B), metrics/recall(B)\n"
            "0, 0.06, 0.02, 0.01, 0.60\n", encoding="utf-8"
        )
        return FakeProc()

    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.launch_training", fake_launch)
    monkeypatch.setattr("auto_tune.modules.agent_engine.loop.monitor_training",
                        lambda *a, **k: ProbeDecision(ProbeDecision.CONTINUE, "ok"))
    return captured


def test_end_to_end_all_facts_bound_to_reference_snapshot(tmp_path, monkeypatch):
    """Reference snapshot A + global latest B: command/args/audit/history/SQLite all A."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    snap_a = _make_snapshot(tmp_path, "A")
    snap_b = _make_snapshot(tmp_path, "B")
    detect = _make_reference(tmp_path, snap_a)

    # global latest_dataset points at B — must never leak into any fact.
    (log_dir / "latest_dataset.json").write_text(json.dumps({
        "snapshot_id": snap_b.snapshot_id,
        "data_yaml_path": str(snap_b.data_yaml_path),
        "snapshot_path": str(snap_b.snapshot_path),
    }), encoding="utf-8")

    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    resolution = resolve_reference_dataset("train52", detect, log_dir, local_index_service=svc)
    assert resolution.snapshot_id == snap_a.snapshot_id
    assert resolution.resolution_source == "reference_args"

    captured = _loop_mocks(monkeypatch, detect, None)

    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train52",
        log_dir=str(log_dir),
        auto_analyze=True,
        runtime_run_id="tuning:uuid-e2e",
        local_index_service=svc,
        reference_dataset=resolution,
    )
    assert result["error"] is None
    assert result["iterations"]

    train_name = result["iterations"][0]["train_name"]
    run_dir = Path(detect) / train_name

    # 1. command carries data=A
    assert any(str(a) == f"data={os.path.normcase(str(resolution.data_yaml_path))}"
               or str(a).endswith(f"data={os.path.normcase(str(resolution.data_yaml_path))}")
               for a in captured["command"])
    # 2. new args.yaml data = A
    written = json.loads(json.dumps(yaml_safe_load(run_dir / "args.yaml")))
    assert os.path.normcase(str(written["data"])) == os.path.normcase(str(resolution.data_yaml_path))

    # 3. audit: full identity at session level + execution data = A
    audit = json.loads((log_dir / f"tuning_audit_{result['session_id']}.json").read_text("utf-8"))
    assert audit["reference_dataset"]["dataset_id"] == resolution.dataset_id
    assert audit["reference_dataset"]["snapshot_id"] == snap_a.snapshot_id
    assert audit["reference_dataset"]["resolution_source"] == "reference_args"
    assert audit["iterations"][0]["baseline"]["reference_run"] == "train52"
    exec_data = audit["iterations"][0]["execution"]["actual_params"]["data"]
    assert os.path.normcase(str(exec_data)) == os.path.normcase(str(resolution.data_yaml_path))

    # 4. JSON history params.data = A
    history = json.loads((log_dir / "experiment_history.json").read_text("utf-8"))
    record = next(r for r in history["experiments"] if r["run_name"] == train_name)
    assert os.path.normcase(str(record["params"]["data"])) == os.path.normcase(str(resolution.data_yaml_path))

    # 5. SQLite experiment dataset_id = resolution.dataset_id
    row = svc.get_experiment("tuning:uuid-e2e")
    assert row is not None
    assert row["dataset_id"] == resolution.dataset_id
    assert row["dataset_id"] != _dataset_id_of(snap_b.snapshot_id)


def _dataset_id_of(snapshot_id):
    import hashlib
    return hashlib.sha256(f"snapshot:{snapshot_id}".encode("utf-8")).hexdigest()


def yaml_safe_load(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_end_to_end_index_failure_keeps_training_facts(tmp_path, monkeypatch):
    """A failed index write must not change the training fact or JSON history."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    snap_a = _make_snapshot(tmp_path, "A")
    detect = _make_reference(tmp_path, snap_a)

    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    resolution = resolve_reference_dataset("train52", detect, log_dir, local_index_service=svc)

    def boom_index_experiment(record, runtime_run_id=None, **kw):
        raise LocalIndexPersistenceError("database locked")

    svc.index_experiment = boom_index_experiment  # type: ignore[method-assign]

    _loop_mocks(monkeypatch, detect, None)
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train52",
        log_dir=str(log_dir),
        auto_analyze=True,
        runtime_run_id="tuning:uuid-index",
        local_index_service=svc,
        reference_dataset=resolution,
    )
    assert result["error"] is None

    train_name = result["iterations"][0]["train_name"]
    run_dir = Path(detect) / train_name
    written = yaml_safe_load(run_dir / "args.yaml")
    assert os.path.normcase(str(written["data"])) == os.path.normcase(str(resolution.data_yaml_path))

    # JSON history still bound to A despite the index failure.
    history = json.loads((log_dir / "experiment_history.json").read_text("utf-8"))
    record = next(r for r in history["experiments"] if r["run_name"] == train_name)
    assert os.path.normcase(str(record["params"]["data"])) == os.path.normcase(str(resolution.data_yaml_path))

    # The finalizer reports an independent index_error, never a training failure.
    audit = json.loads((log_dir / f"tuning_audit_{result['session_id']}.json").read_text("utf-8"))
    assert audit["status"] == "completed"


def test_end_to_end_loop_result_public_projection_only(tmp_path, monkeypatch):
    """tuning_result reference_dataset projection must expose short IDs only."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True)
    snap_a = _make_snapshot(tmp_path, "A")
    detect = _make_reference(tmp_path, snap_a)

    svc = LocalIndexService(_cfg(tmp_path))
    svc.initialize()
    resolution = resolve_reference_dataset("train52", detect, log_dir, local_index_service=svc)

    _loop_mocks(monkeypatch, detect, None)
    result = run_tuning_loop(
        {"probe": {"max_retries": 1}, "train_analyzer": {}},
        reference_run="train52",
        log_dir=str(log_dir),
        auto_analyze=True,
        runtime_run_id="tuning:uuid-proj",
        local_index_service=svc,
        reference_dataset=resolution,
    )
    proj = result["reference_dataset"]
    assert proj is not None
    assert proj["snapshot_short_id"] == snap_a.snapshot_id[:8]
    assert proj["dataset_display_name"] is not None
    assert proj["resolution_source"] == "reference_args"
    # no absolute path leaks into the public projection
    assert str(snap_a.data_yaml_path) not in json.dumps(proj)
    assert "dataset_snapshots" not in json.dumps(proj)
