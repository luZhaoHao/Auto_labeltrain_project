"""Tests for durable audit records and atomic JSON persistence."""

import json
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine.audit import (
    TuningAuditSession,
    atomic_write_json,
    redact_sensitive,
)


def test_redact_sensitive_recurses_without_mutating_input():
    source = {
        "llm": {
            "api_key": "secret-key",
            "headers": {"Authorization": "Bearer abc"},
        },
        "items": [
            {"access_token": "token-value", "model": "deepseek"},
            ("plain", {"PASSWORD": "p@ss"}),
        ],
    }

    redacted = redact_sensitive(source)

    assert redacted["llm"]["api_key"] == "***REDACTED***"
    assert redacted["llm"]["headers"]["Authorization"] == "***REDACTED***"
    assert redacted["items"][0]["access_token"] == "***REDACTED***"
    assert redacted["items"][0]["model"] == "deepseek"
    assert redacted["items"][1][1]["PASSWORD"] == "***REDACTED***"
    assert source["llm"]["api_key"] == "secret-key"


def test_redact_sensitive_masks_partial_case_insensitive_key_names():
    assert redact_sensitive({"DeepSeekApiKey": "x"}) == {
        "DeepSeekApiKey": "***REDACTED***"
    }
    assert redact_sensitive({"client_secret_value": "x"}) == {
        "client_secret_value": "***REDACTED***"
    }


def test_atomic_write_json_replaces_existing_file(tmp_path):
    target = tmp_path / "audit.json"
    target.write_text('{"old": true}', encoding="utf-8")

    atomic_write_json(target, {"new": True, "text": "中文"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "new": True,
        "text": "中文",
    }
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []


def test_atomic_write_json_preserves_old_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "audit.json"
    target.write_text('{"stable": true}', encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("auto_tune.modules.agent_engine.audit.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"stable": False})

    assert json.loads(target.read_text(encoding="utf-8")) == {"stable": True}
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []


def test_audit_session_persists_iteration_lifecycle(tmp_path):
    session = TuningAuditSession("session-1", str(tmp_path), "train38", 2)
    session.flush()
    session.start_iteration(1)
    session.update_iteration(
        1,
        baseline={"reference_run": "train38", "params": {"lr0": 0.01}, "metrics": {}},
        decision={"raw_response": "{}", "action": "keep_params"},
    )
    session.complete_iteration(1)
    session.finalize("completed")

    saved = json.loads(Path(session.path).read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1.0"
    assert saved["session_id"] == "session-1"
    assert saved["status"] == "completed"
    assert saved["finished_at"].endswith("Z")
    assert saved["iterations"][0]["iteration"] == 1
    assert saved["iterations"][0]["status"] == "completed"
    assert saved["iterations"][0]["baseline"]["params"] == {"lr0": 0.01}


def test_audit_session_records_structured_failure(tmp_path):
    session = TuningAuditSession("session-2", str(tmp_path), None, 1)
    session.start_iteration(1)
    session.fail_iteration(
        1,
        stage="decision",
        error_type="decision_schema_error",
        message="invalid JSON",
        fatal=True,
    )
    session.finalize("failed", session.to_dict()["iterations"][0]["error"])

    saved = json.loads(Path(session.path).read_text(encoding="utf-8"))
    error = saved["iterations"][0]["error"]
    assert error["stage"] == "decision"
    assert error["error_type"] == "decision_schema_error"
    assert error["fatal"] is True
    assert saved["error"] == error


def test_audit_session_rejects_duplicate_iteration(tmp_path):
    session = TuningAuditSession("session-3", str(tmp_path), None, 1)
    session.start_iteration(1)
    with pytest.raises(ValueError):
        session.start_iteration(1)


def test_audit_session_rejects_unknown_update_field(tmp_path):
    session = TuningAuditSession("session-4", str(tmp_path), None, 1)
    session.start_iteration(1)
    with pytest.raises(KeyError):
        session.update_iteration(1, nonexistent_field=1)


def test_audit_session_rejects_invalid_final_status(tmp_path):
    session = TuningAuditSession("session-5", str(tmp_path), None, 1)
    with pytest.raises(ValueError):
        session.finalize("unknown")


def test_fail_iteration_persists_to_disk_immediately(tmp_path):
    """A probe ABORT/RETRY failure must survive a crash before finalize."""
    session = TuningAuditSession("session-6", str(tmp_path), None, 1)
    session.start_iteration(1)
    session.fail_iteration(
        1,
        stage="probe",
        error_type="probe_retry",
        message="mAP50 too low",
        fatal=False,
    )

    saved = json.loads(Path(session.path).read_text(encoding="utf-8"))
    assert saved["iterations"][0]["status"] == "failed"
    assert saved["iterations"][0]["error"]["stage"] == "probe"
    assert saved["iterations"][0]["error"]["error_type"] == "probe_retry"
    assert saved["iterations"][0]["error"]["fatal"] is False


def test_new_iteration_execution_command_defaults_to_array(tmp_path):
    """Approved schema requires execution.command to be a JSON array, not null."""
    session = TuningAuditSession("session-7", str(tmp_path), None, 1)
    session.start_iteration(1)
    iteration = session.to_dict()["iterations"][0]
    assert iteration["execution"]["command"] == []


def test_fail_iteration_error_includes_timestamp(tmp_path):
    session = TuningAuditSession("session-8", str(tmp_path), None, 1)
    session.start_iteration(1)
    session.fail_iteration(
        1, stage="decision", error_type="decision_schema_error", message="bad", fatal=True
    )
    error = session.to_dict()["iterations"][0]["error"]
    assert error["timestamp"].endswith("Z")


def test_start_iteration_returns_deep_copy_of_new_record(tmp_path):
    session = TuningAuditSession("session-9", str(tmp_path), None, 1)
    record = session.start_iteration(1)
    assert record["iteration"] == 1
    record["status"] = "tampered"
    assert session.to_dict()["iterations"][0]["status"] == "running"


def test_fail_iteration_write_error_propagates(tmp_path, monkeypatch):
    """A failed audit write must propagate, never be silently swallowed."""
    import auto_tune.modules.agent_engine.audit as audit_module

    session = TuningAuditSession("session-10", str(tmp_path), None, 1)
    session.start_iteration(1)

    def boom(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(audit_module, "atomic_write_json", boom)

    with pytest.raises(OSError, match="disk full"):
        session.fail_iteration(
            1, stage="probe", error_type="probe_retry", message="x", fatal=False
        )
