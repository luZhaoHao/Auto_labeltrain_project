"""Process identity verification and conservative run-state reconciliation.

A process is only ever identified by ``(pid, process_create_token)`` where
the token is derived from the process creation time (Windows FILETIME via
``GetProcessTimes``, Linux starttime via ``/proc/<pid>/stat`` field 22).
Merely checking that a PID exists is never enough: an OS-reused PID must not
be mistaken for the original training process.
"""

from __future__ import annotations

import os
from dataclasses import replace
from enum import Enum

from .models import ProcessIdentity, RunState, utc_now_iso


class IdentityMatch(Enum):
    MATCH = "match"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


def _process_exists(pid: int) -> bool:
    """Best-effort liveness probe: True when the PID is present, even without
    permission to read its identity."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _windows_kernel32():
    """Return ``(kernel32, _FILETIME)`` with 64-bit-safe ctypes signatures.

    Explicit ``argtypes``/``restype`` prevent Win64 handle truncation: without
    them a HANDLE returned by ``OpenProcess`` is treated as a 32-bit int, which
    corrupts the token for high PIDs.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32, _FILETIME


def _windows_identity(pid: int) -> ProcessIdentity | None:
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32, _FILETIME = _windows_kernel32()
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None

    try:
        creation = _FILETIME()
        exit_time = _FILETIME()
        kernel_time = _FILETIME()
        user_time = _FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not ok:
            return None
        ft = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return ProcessIdentity(int(pid), f"windows-filetime:{ft}")
    finally:
        kernel32.CloseHandle(handle)


def _linux_identity(pid: int) -> ProcessIdentity | None:
    try:
        with open(f"/proc/{int(pid)}/stat", "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except OSError:
        return None
    rparen = data.rfind(")")
    if rparen < 0:
        return None
    fields = data[rparen + 1:].split()
    if len(fields) <= 19:
        return None
    starttime = fields[19]  # field 22 (starttime), 1-indexed after the comm field
    return ProcessIdentity(int(pid), f"linux-starttime:{starttime}")


def capture_process_identity(pid: int) -> ProcessIdentity | None:
    """Capture the creation-identity token for a PID, or ``None`` when the
    platform is unsupported, permission is missing, or the process is gone."""
    if os.name == "nt":
        return _windows_identity(pid)
    return _linux_identity(pid)


def compare_process_identity(expected: ProcessIdentity | None) -> IdentityMatch:
    """Compare an expected identity against the live process facts."""
    if expected is None:
        return IdentityMatch.UNVERIFIABLE
    current = capture_process_identity(expected.pid)
    if current is None:
        if _process_exists(expected.pid):
            return IdentityMatch.UNVERIFIABLE
        return IdentityMatch.MISSING
    if current.process_create_token == expected.process_create_token:
        return IdentityMatch.MATCH
    return IdentityMatch.MISMATCH


def _terminal(state: RunState, status: str, reason: str) -> RunState:
    return replace(
        state,
        status=status,
        phase="terminal",
        terminal_reason=reason,
        updated_at=utc_now_iso(),
    )


def reconcile_persisted_state(
    state: RunState | None,
    controller_owned: bool,
    *,
    identity_match: IdentityMatch | None = None,
) -> RunState | None:
    """Conservatively reconcile a persisted run state.

    - In-memory controller still owns the process (``controller_owned=True``):
      keep ``running`` — the caller must have verified the process is alive.
    - Controller lost (e.g. server restart): the service can no longer consume
      output or control the process, so a matching identity only proves the
      process still exists → ``interrupted``. A missing PID → ``process_missing``;
      a reused PID → ``pid_reused``; unverifiable → ``unknown``.
    - Terminal states are returned unchanged.

    ``identity_match`` lets a caller that has *already* observed the process
    identity pass that single observation in, so one gate decision never reads
    the PID twice (a PID reused between two reads would otherwise be able to
    change the resulting terminal reason). When omitted the identity is
    captured here, keeping every existing call site compatible.
    """
    if state is None:
        return None
    if state.status != "running":
        return state
    if controller_owned:
        return state
    if state.pid is None or state.process_create_token is None:
        # No process identity to verify (e.g. dry-run without a training PID);
        # after the controller is gone the run can only be interrupted.
        return _terminal(state, "interrupted", "controller_lost")
    match = identity_match
    if match is None:
        match = compare_process_identity(
            ProcessIdentity(state.pid, state.process_create_token)
        )
    if match is IdentityMatch.MATCH:
        return _terminal(state, "interrupted", "controller_lost")
    if match is IdentityMatch.MISSING:
        return _terminal(state, "interrupted", "process_missing")
    if match is IdentityMatch.MISMATCH:
        return _terminal(state, "interrupted", "pid_reused")
    return _terminal(state, "unknown", "process_identity_unverifiable")
