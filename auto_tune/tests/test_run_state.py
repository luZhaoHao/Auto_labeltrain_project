"""Tests for the unified run-state domain module (Studio S1.5 Task 1).

Covers schema/identity creation, atomic persistence, legacy-format
compatibility, corrupt-file projection, and the public response shape.
"""

import json
import os
from dataclasses import replace

import pytest

from auto_tune.modules.run_state.models import (
    RunStatePersistenceError,
    RunStateValidationError,
)
from auto_tune.modules.run_state.service import (
    new_run_state,
    project_public_state,
    read_run_state,
    update_run_state,
    write_run_state,
)


def test_new_manual_run_has_unique_versioned_identity():
    first = new_run_state("manual", run_name="train1")
    second = new_run_state("manual", run_name="train1")
    assert first.schema_version == "1.0"
    assert first.run_id.startswith("manual:")
    assert first.run_id != second.run_id
    assert first.status == "starting"
    assert first.phase == "preparing"


def test_new_tuning_run_has_unique_namespace():
    first = new_run_state("tuning", run_name="autotune_1")
    second = new_run_state("tuning", run_name="autotune_1")
    assert first.run_id.startswith("tuning:")
    assert first.run_id != second.run_id
    assert first.run_id.split(":", 1)[0] == second.run_id.split(":", 1)[0] == "tuning"


def test_new_run_state_rejects_unknown_kind():
    with pytest.raises(RunStateValidationError):
        new_run_state("bogus")


def test_atomic_failure_preserves_last_valid_state(tmp_path, monkeypatch):
    path = tmp_path / "training_running.json"
    original = new_run_state("manual", run_name="train1")
    write_run_state(path, original)
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(RunStatePersistenceError):
        write_run_state(path, replace(original, phase="training"))
    assert read_run_state(path).phase == "preparing"


def test_write_creates_schema_1_0_file(tmp_path):
    path = tmp_path / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    write_run_state(path, state)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["run_id"].startswith("manual:")
    assert data["status"] == "starting"
    assert data["phase"] == "preparing"


def test_update_run_state_advances_seq_and_terminal(tmp_path):
    path = tmp_path / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    write_run_state(path, state)
    state = update_run_state(
        path, state, status="running", phase="training",
        event_type="training_log", event_message="Epoch 1/10",
    )
    assert state.last_event.seq == 0
    assert state.last_event.message == "Epoch 1/10"
    state = update_run_state(
        path, state, event_type="training_log", event_message="Epoch 2/10",
    )
    assert state.last_event.seq == 1
    state = update_run_state(path, state, status="completed", phase="terminal")
    assert state.status == "completed"
    assert state.phase == "terminal"
    assert read_run_state(path).status == "completed"


@pytest.mark.parametrize("kwargs", [
    {"status": "bogus"},
    {"phase": "bogus"},
])
def test_invalid_status_phase_raise(tmp_path, kwargs):
    path = tmp_path / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    write_run_state(path, state)
    with pytest.raises(RunStateValidationError):
        update_run_state(path, state, **kwargs)


def test_roundtrip_new_schema(tmp_path):
    path = tmp_path / "training_running.json"
    state = new_run_state("tuning", run_name="autotune_1")
    state = update_run_state(
        path, state, status="running", phase="training",
        pid=1234, process_create_token="windows-filetime:133999999999999999",
        event_type="execute", event_message="启动训练",
    )
    loaded = read_run_state(path)
    assert loaded.run_id == state.run_id
    assert loaded.run_kind == "tuning"
    assert loaded.pid == 1234
    assert loaded.process_create_token == "windows-filetime:133999999999999999"
    assert loaded.last_event.seq == 0


def test_legacy_manual_running_projects_unknown(tmp_path):
    path = tmp_path / "training_running.json"
    path.write_text(
        json.dumps({"train_name": "train1", "status": "running", "start_time": 123}),
        encoding="utf-8",
    )
    state = read_run_state(path, run_kind="manual")
    assert state.status == "unknown"
    assert state.terminal_reason == "legacy_identity_unverifiable"
    assert state.run_kind == "manual"
    assert state.run_name == "train1"
    public = project_public_state(state)
    assert public["running"] is False
    assert public["status"] == "unknown"
    # Original bytes are never rewritten.
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "train_name": "train1", "status": "running", "start_time": 123,
    }


def test_legacy_tuning_running_projects_unknown(tmp_path):
    path = tmp_path / "tuning_running.json"
    path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    state = read_run_state(path, run_kind="tuning")
    assert state.status == "unknown"
    assert state.terminal_reason == "legacy_identity_unverifiable"
    assert state.run_kind == "tuning"


@pytest.mark.parametrize("legacy_status,expected", [
    ("completed", "completed"),
    ("failed", "failed"),
    ("aborted", "cancelled"),
])
def test_legacy_terminal_status_mapping(tmp_path, legacy_status, expected):
    path = tmp_path / "training_running.json"
    path.write_text(
        json.dumps({"train_name": "train1", "status": legacy_status}),
        encoding="utf-8",
    )
    state = read_run_state(path, run_kind="manual")
    assert state.status == expected
    assert state.phase == "terminal"


def test_corrupt_file_projects_unknown_and_preserves_bytes(tmp_path):
    path = tmp_path / "training_running.json"
    path.write_bytes(b"{not valid json!!")
    state = read_run_state(path, run_kind="manual")
    assert state.status == "unknown"
    assert state.terminal_reason == "state_corrupt"
    assert path.read_bytes() == b"{not valid json!!"


def test_unrecognized_legacy_value_projects_unknown(tmp_path):
    path = tmp_path / "training_running.json"
    path.write_text(json.dumps({"train_name": "x", "status": "weird"}), encoding="utf-8")
    state = read_run_state(path, run_kind="manual")
    assert state.status == "unknown"


def test_missing_file_returns_none(tmp_path):
    assert read_run_state(tmp_path / "training_running.json") is None


def test_project_public_state_no_record_stable_shape():
    public = project_public_state(None)
    assert public["running"] is False
    assert public["run_id"] is None
    assert public["status"] == "unknown"
    assert public["phase"] == "terminal"
    for key in ("running", "run_id", "run_kind", "status", "phase",
                "last_event", "terminal_reason", "run_name", "updated_at"):
        assert key in public


def test_project_public_state_running_shape(tmp_path):
    path = tmp_path / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    state = update_run_state(
        path, state, status="running", phase="training",
        event_type="training_log", event_message="Epoch 1/10",
    )
    public = project_public_state(state)
    assert public["running"] is True
    assert public["run_id"] == state.run_id
    assert public["run_kind"] == "manual"
    assert public["status"] == "running"
    assert public["phase"] == "training"
    assert public["last_event"]["seq"] == 0
    assert public["terminal_reason"] is None
    assert public["run_name"] == "train1"


def test_legacy_and_corrupt_projection_has_stable_updated_at(tmp_path):
    """Legacy/corrupt projections must not mint a fresh updated_at on every
    read (otherwise an unknown state could permanently shadow a real run in the
    UI merge)."""
    legacy = tmp_path / "training_running.json"
    legacy.write_text(json.dumps({"train_name": "t1", "status": "running"}), encoding="utf-8")
    first = read_run_state(legacy, run_kind="manual")
    second = read_run_state(legacy, run_kind="manual")
    assert first.updated_at == ""
    assert second.updated_at == ""
    assert first.updated_at == second.updated_at

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"{bad json")
    c1 = read_run_state(corrupt, run_kind="manual")
    c2 = read_run_state(corrupt, run_kind="manual")
    assert c1.updated_at == ""
    assert c1.updated_at == c2.updated_at


def test_unknown_schema_version_projects_unknown_with_stable_reason(tmp_path):
    path = tmp_path / "training_running.json"
    path.write_text(
        json.dumps({"schema_version": "9.9", "run_id": "manual:abc", "status": "running"}),
        encoding="utf-8",
    )
    state = read_run_state(path, run_kind="manual")
    assert state.status == "unknown"
    assert state.phase == "terminal"
    assert state.terminal_reason == "unknown_schema_version"
    assert state.run_id == "manual:abc"
    assert state.updated_at == ""
    # the file is never rewritten
    assert json.loads(path.read_text("utf-8"))["schema_version"] == "9.9"
