"""Tests for process identity capture/compare and conservative reconciliation.

The current-process tests execute for real (0 skipped on Windows/Linux); the
missing / reused-PID / unverifiable branches are covered via monkeypatch.
"""

import os

import pytest

from auto_tune.modules.run_state.models import (
    ProcessIdentity,
    RunState,
    SCHEMA_VERSION,
)
from auto_tune.modules.run_state import process_identity as pi
from auto_tune.modules.run_state.process_identity import (
    IdentityMatch,
    capture_process_identity,
    compare_process_identity,
    reconcile_persisted_state,
)


def _running_state(**overrides):
    base = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "manual:test",
        "run_kind": "manual",
        "status": "running",
        "phase": "training",
        "started_at": "2026-08-25T00:00:00Z",
        "updated_at": "2026-08-25T00:01:00Z",
        "pid": 1234,
        "process_create_token": "windows-filetime:133999999999999999",
        "last_event": None,
        "run_name": "train1",
        "terminal_reason": None,
    }
    base.update(overrides)
    return RunState(**base)


# ── capture / compare ──


def test_capture_current_process_identity_stable():
    first = capture_process_identity(os.getpid())
    second = capture_process_identity(os.getpid())
    assert first is not None
    assert first.pid == os.getpid()
    assert first.process_create_token
    assert first.process_create_token == second.process_create_token


def test_compare_missing(monkeypatch):
    monkeypatch.setattr(pi, "capture_process_identity", lambda pid: None)
    monkeypatch.setattr(pi, "_process_exists", lambda pid: False)
    assert compare_process_identity(ProcessIdentity(99999, "token")) == IdentityMatch.MISSING


def test_compare_unverifiable_when_process_exists_but_identity_unreadable(monkeypatch):
    monkeypatch.setattr(pi, "capture_process_identity", lambda pid: None)
    monkeypatch.setattr(pi, "_process_exists", lambda pid: True)
    assert compare_process_identity(ProcessIdentity(99999, "token")) == IdentityMatch.UNVERIFIABLE


def test_compare_mismatch_on_pid_reuse(monkeypatch):
    monkeypatch.setattr(
        pi, "capture_process_identity",
        lambda pid: ProcessIdentity(pid, "windows-filetime:different"),
    )
    assert compare_process_identity(ProcessIdentity(1234, "windows-filetime:original")) == IdentityMatch.MISMATCH


def test_compare_match(monkeypatch):
    token = "windows-filetime:133999999999999999"
    monkeypatch.setattr(
        pi, "capture_process_identity",
        lambda pid: ProcessIdentity(pid, token),
    )
    assert compare_process_identity(ProcessIdentity(1234, token)) == IdentityMatch.MATCH


def test_compare_none_is_unverifiable():
    assert compare_process_identity(None) == IdentityMatch.UNVERIFIABLE


# ── reconcile ──


def test_reconcile_controller_owned_keeps_running():
    state = _running_state()
    assert reconcile_persisted_state(state, controller_owned=True) is state


def test_reconcile_terminal_states_untouched():
    state = _running_state(status="completed", phase="terminal")
    assert reconcile_persisted_state(state, controller_owned=False) is state


def test_reconcile_controller_lost_when_identity_matches(monkeypatch):
    monkeypatch.setattr(pi, "compare_process_identity", lambda expected: IdentityMatch.MATCH)
    state = _running_state()
    result = reconcile_persisted_state(state, controller_owned=False)
    assert result.status == "interrupted"
    assert result.phase == "terminal"
    assert result.terminal_reason == "controller_lost"


def test_reconcile_process_missing(monkeypatch):
    monkeypatch.setattr(pi, "compare_process_identity", lambda expected: IdentityMatch.MISSING)
    result = reconcile_persisted_state(_running_state(), controller_owned=False)
    assert result.status == "interrupted"
    assert result.terminal_reason == "process_missing"


def test_reconcile_pid_reused(monkeypatch):
    monkeypatch.setattr(pi, "compare_process_identity", lambda expected: IdentityMatch.MISMATCH)
    result = reconcile_persisted_state(_running_state(), controller_owned=False)
    assert result.status == "interrupted"
    assert result.terminal_reason == "pid_reused"


def test_reconcile_unverifiable(monkeypatch):
    monkeypatch.setattr(pi, "compare_process_identity", lambda expected: IdentityMatch.UNVERIFIABLE)
    result = reconcile_persisted_state(_running_state(), controller_owned=False)
    assert result.status == "unknown"
    assert result.terminal_reason == "process_identity_unverifiable"


def test_reconcile_no_pid_controller_lost():
    state = _running_state(pid=None, process_create_token=None)
    result = reconcile_persisted_state(state, controller_owned=False)
    assert result.status == "interrupted"
    assert result.terminal_reason == "controller_lost"


def test_reconcile_none_returns_none():
    assert reconcile_persisted_state(None, controller_owned=False) is None


def test_windows_ctypes_signatures_are_64bit_safe():
    """OpenProcess/GetProcessTimes/CloseHandle must declare argtypes/restype
    so HANDLEs are not truncated to 32 bits on Win64."""
    from ctypes import wintypes

    from auto_tune.modules.run_state.process_identity import _windows_kernel32

    kernel32, _ = _windows_kernel32()
    assert kernel32.OpenProcess.argtypes is not None
    assert kernel32.OpenProcess.restype == wintypes.HANDLE
    assert kernel32.GetProcessTimes.argtypes is not None
    assert kernel32.GetProcessTimes.restype == wintypes.BOOL
    assert kernel32.CloseHandle.argtypes is not None
    assert kernel32.CloseHandle.restype == wintypes.BOOL
