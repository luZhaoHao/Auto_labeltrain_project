"""Run-state persistence, legacy compatibility, and public projection.

Normal training and auto-tuning both write through this module so page
refresh / browser reconnect / server restart can only state facts that are
proven by the persisted record and (for running) by process identity checks.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from .models import (
    LastEvent,
    RunState,
    RunStatePersistenceError,
    RunStateValidationError,
    RUN_KINDS_CREATE,
    RUN_PHASES,
    RUN_STATUSES,
    SCHEMA_VERSION,
    utc_now_iso,
)

_UNSET = object()


def new_run_state(run_kind: str, run_name: str | None = None) -> RunState:
    """Create a fresh Schema 1.0 run with a unique ``<run_kind>:<uuid4>`` id."""
    if run_kind not in RUN_KINDS_CREATE:
        raise RunStateValidationError(f"invalid run_kind: {run_kind!r}")
    now = utc_now_iso()
    return RunState(
        schema_version=SCHEMA_VERSION,
        run_id=f"{run_kind}:{uuid.uuid4()}",
        run_kind=run_kind,
        status="starting",
        phase="preparing",
        started_at=now,
        updated_at=now,
        pid=None,
        process_create_token=None,
        last_event=None,
        run_name=run_name,
        terminal_reason=None,
    )


def _state_to_dict(state: RunState) -> dict:
    last_event = None
    if state.last_event is not None:
        last_event = {
            "seq": state.last_event.seq,
            "type": state.last_event.type,
            "at": state.last_event.at,
            "message": state.last_event.message,
        }
    return {
        "schema_version": state.schema_version,
        "run_id": state.run_id,
        "run_kind": state.run_kind,
        "status": state.status,
        "phase": state.phase,
        "started_at": state.started_at,
        "updated_at": state.updated_at,
        "pid": state.pid,
        "process_create_token": state.process_create_token,
        "last_event": last_event,
        "run_name": state.run_name,
        "terminal_reason": state.terminal_reason,
    }


def write_run_state(path: str | os.PathLike, state: RunState) -> None:
    """Atomically persist a run state: same-dir temp, flush, fsync, os.replace.

    On failure the previous valid file is left intact and a
    ``RunStatePersistenceError`` is raised.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(_state_to_dict(state), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
    except OSError as exc:
        raise RunStatePersistenceError(
            f"failed to write run state {target}: {exc}"
        ) from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def update_run_state(
    path: str | os.PathLike,
    state: RunState,
    *,
    status: object = _UNSET,
    phase: object = _UNSET,
    pid: object = _UNSET,
    process_create_token: object = _UNSET,
    run_name: object = _UNSET,
    terminal_reason: object = _UNSET,
    event_type: str | None = None,
    event_message: str | None = None,
    event_at: str | None = None,
) -> RunState:
    """Return and persist an updated run state.

    ``event_type``/``event_message`` advance ``last_event.seq`` by one; all
    other fields are only changed when explicitly provided. The updated state
    is atomically written before being returned.
    """
    updates: dict = {"updated_at": utc_now_iso()}
    if status is not _UNSET:
        updates["status"] = status
    if phase is not _UNSET:
        updates["phase"] = phase
    if pid is not _UNSET:
        updates["pid"] = pid
    if process_create_token is not _UNSET:
        updates["process_create_token"] = process_create_token
    if run_name is not _UNSET:
        updates["run_name"] = run_name
    if terminal_reason is not _UNSET:
        updates["terminal_reason"] = terminal_reason
    if event_type is not None or event_message is not None:
        seq = 0 if state.last_event is None else state.last_event.seq + 1
        prev_type = state.last_event.type if state.last_event else ""
        updates["last_event"] = LastEvent(
            seq=seq,
            type=event_type if event_type is not None else prev_type,
            at=event_at or utc_now_iso(),
            message=event_message,
        )
    new_state = replace(state, **updates)
    write_run_state(path, new_state)
    return new_state


def _corrupt_state(run_kind: str | None) -> RunState:
    return RunState(
        schema_version=SCHEMA_VERSION,
        run_id=None,
        run_kind=run_kind if run_kind in ("manual", "tuning") else "unknown",
        status="unknown",
        phase="terminal",
        started_at="",
        updated_at="",
        pid=None,
        process_create_token=None,
        last_event=None,
        run_name=None,
        terminal_reason="state_corrupt",
    )


def _unknown_state(
    run_kind: str | None, terminal_reason: str, run_id: str | None = None
) -> RunState:
    """Conservative unknown projection with a stable, empty timestamp.

    ``updated_at`` is intentionally empty (never the current time) so an
    unknown/legacy projection can never shadow a real run in the UI merge.
    """
    return RunState(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        run_kind=run_kind if run_kind in ("manual", "tuning") else "unknown",
        status="unknown",
        phase="terminal",
        started_at="",
        updated_at="",
        pid=None,
        process_create_token=None,
        last_event=None,
        run_name=None,
        terminal_reason=terminal_reason,
    )


def _parse_new(data: dict) -> RunState:
    run_id = data["run_id"]
    run_kind = data.get("run_kind") or str(run_id).split(":", 1)[0]
    status = data.get("status", "unknown")
    phase = data.get("phase", "terminal")
    if status not in RUN_STATUSES or phase not in RUN_PHASES:
        raise RunStateValidationError(f"invalid status/phase in record: {run_id}")
    last_event = None
    le = data.get("last_event")
    if isinstance(le, dict) and "seq" in le:
        last_event = LastEvent(
            seq=int(le["seq"]),
            type=str(le.get("type", "")),
            at=str(le.get("at", "")),
            message=le.get("message"),
        )
    pid_raw = data.get("pid")
    return RunState(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        run_id=str(run_id),
        run_kind=run_kind,
        status=status,
        phase=phase,
        started_at=str(data.get("started_at", "")),
        updated_at=str(data.get("updated_at", "")),
        pid=int(pid_raw) if pid_raw is not None else None,
        process_create_token=data.get("process_create_token"),
        last_event=last_event,
        run_name=data.get("run_name"),
        terminal_reason=data.get("terminal_reason"),
    )


def _legacy_state(
    run_kind: str,
    run_name: str | None,
    status: str,
    terminal_reason: str | None,
) -> RunState:
    return RunState(
        schema_version=SCHEMA_VERSION,
        run_id=None,
        run_kind=run_kind if run_kind in ("manual", "tuning") else "unknown",
        status=status,
        phase="terminal",
        started_at="",
        updated_at="",
        pid=None,
        process_create_token=None,
        last_event=None,
        run_name=run_name,
        terminal_reason=terminal_reason,
    )


_LEGACY_TERMINAL_MAP = {
    "completed": "completed",
    "failed": "failed",
    "aborted": "cancelled",
    "cancelled": "cancelled",
}


def _normalize_legacy(data: dict, run_kind: str | None) -> RunState:
    """Project an old pre-1.0 status record into a conservative RunState.

    Old records carry no run_id / process identity, so a legacy ``running``
    can never be claimed as running: it becomes ``unknown`` with
    ``terminal_reason=legacy_identity_unverifiable``. Recognized terminal
    values keep their meaning; anything else becomes ``unknown``.
    """
    status = data.get("status")
    run_name = data.get("train_name") or data.get("run_name")
    kind = run_kind if run_kind in ("manual", "tuning") else "unknown"
    if status == "running":
        return _legacy_state(kind, run_name, "unknown", "legacy_identity_unverifiable")
    if isinstance(status, str) and status in _LEGACY_TERMINAL_MAP:
        return _legacy_state(kind, run_name, _LEGACY_TERMINAL_MAP[status], None)
    return _legacy_state(kind, run_name, "unknown", "state_corrupt")


def read_run_state(
    path: str | os.PathLike, run_kind: str | None = None
) -> RunState | None:
    """Read a run-state file without rewriting it.

    Returns ``None`` when no file exists. Legacy and corrupt files project to
    conservative ``unknown`` states and are never silently overwritten.
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return _corrupt_state(run_kind)
    if not isinstance(data, dict):
        return _corrupt_state(run_kind)
    if "run_id" in data:
        schema_version = data.get("schema_version")
        if schema_version is not None and str(schema_version) != SCHEMA_VERSION:
            # A record with an identity we cannot interpret is conservatively
            # unknown; a stable terminal_reason keeps it debuggable.
            return _unknown_state(run_kind, "unknown_schema_version", run_id=str(data.get("run_id")))
        try:
            return _parse_new(data)
        except (RunStateValidationError, KeyError, TypeError, ValueError):
            return _corrupt_state(run_kind)
    return _normalize_legacy(data, run_kind)


def project_public_state(state: RunState | None) -> dict:
    """Project a run state (or no-record) into the stable public API shape."""
    if state is None:
        return {
            "running": False,
            "run_id": None,
            "run_kind": None,
            "status": "unknown",
            "phase": "terminal",
            "last_event": None,
            "terminal_reason": None,
            "run_name": None,
            "updated_at": None,
        }
    last_event = None
    if state.last_event is not None:
        last_event = {
            "seq": state.last_event.seq,
            "type": state.last_event.type,
            "at": state.last_event.at,
            "message": state.last_event.message,
        }
    return {
        "running": state.status == "running",
        "run_id": state.run_id,
        "run_kind": state.run_kind,
        "status": state.status,
        "phase": state.phase,
        "last_event": last_event,
        "terminal_reason": state.terminal_reason,
        "run_name": state.run_name,
        "updated_at": state.updated_at,
    }


def with_status_phase(
    state: RunState,
    *,
    status: str | None = None,
    phase: str | None = None,
    pid: int | None = None,
    process_create_token: str | None = None,
    terminal_reason: str | None = None,
) -> RunState:
    """Return a state with status/phase/identity fields replaced (no write)."""
    updates: dict = {"updated_at": utc_now_iso()}
    if status is not None:
        updates["status"] = status
    if phase is not None:
        updates["phase"] = phase
    if pid is not None:
        updates["pid"] = pid
    if process_create_token is not None:
        updates["process_create_token"] = process_create_token
    if terminal_reason is not None:
        updates["terminal_reason"] = terminal_reason
    return replace(state, **updates)


def with_last_event(
    state: RunState,
    *,
    event_type: str,
    message: str | None,
    seq: int,
    at: str | None = None,
) -> RunState:
    """Return a state whose last_event uses the given broker-assigned ``seq``.

    This keeps ``RunState.last_event.seq`` exactly equal to the seq of the
    last event published to the event bus.
    """
    return replace(
        state,
        last_event=LastEvent(seq=seq, type=event_type, at=at or utc_now_iso(), message=message),
        updated_at=utc_now_iso(),
    )


def with_terminal(
    state: RunState,
    *,
    status: str,
    event_type: str,
    message: str | None,
    seq: int,
    terminal_reason: str | None = None,
) -> RunState:
    """Return a terminal state whose last_event uses the given broker seq."""
    return replace(
        state,
        status=status,
        phase="terminal",
        terminal_reason=terminal_reason,
        last_event=LastEvent(seq=seq, type=event_type, at=utc_now_iso(), message=message),
        updated_at=utc_now_iso(),
    )
