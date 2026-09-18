"""Persisted-state training-slot gate (H1.3 重启保护).

The in-memory ``RunManager`` reservation is authoritative while the service
process is alive. After a restart the registry is empty, but that never means
"no training is active": a leftover ``starting``/``running`` run-state record
may still have a live (orphaned) training process. Before starting any new
real training, callers evaluate these persisted facts:

- no identity or the identity still matches / cannot be verified → BLOCK new
  training (the old process may still be running; we never adopt or kill it);
- the process is provably gone (missing / reused PID) → reconcile the record to
  a terminal ``interrupted`` state and allow a new run.

HPO execution is not a run-state record; its liveness is checked through the
persisted ``execution.json`` audit (a ``RUNNING`` attempt whose subprocess PID
still matches also blocks new training).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .models import ProcessIdentity, RunState, utc_now_iso
from .process_identity import (
    IdentityMatch,
    compare_process_identity,
    reconcile_persisted_state,
)

# Fields of a raw execution.json we inspect without importing the strict model:
# a leftover RUNNING attempt carries the subprocess identity.
_EXECUTION_FILE = "execution.json"


@dataclass(frozen=True)
class SlotDecision:
    """Result of evaluating one persisted run-state record.

    ``blocked`` True means a new real training must not start. When proceeding,
    ``persist`` optionally carries the reconciled terminal state that the caller
    should write back so the record stays truthful.
    """

    blocked: bool
    reason: str | None = None
    persist: RunState | None = None


def evaluate_persisted_run_state(state: RunState | None) -> SlotDecision:
    """Decide whether a persisted manual/tuning record blocks a new training."""
    if state is None:
        return SlotDecision(blocked=False)
    if state.status not in ("starting", "running"):
        # Terminal and unknown projections never hold a live slot by themselves.
        return SlotDecision(blocked=False)
    if state.pid is None or state.process_create_token is None:
        # A starting/running record with no process identity cannot be proven
        # gone; conservatively block (matching the spec's UNVERIFIABLE rule).
        return SlotDecision(
            blocked=True,
            reason="unverifiable",
        )
    # Observe the process identity exactly once. Blocking, the terminal reason
    # and the reconciled state must all come from this same observation: a PID
    # that is gone on the first read but reused on a second read would otherwise
    # flip MISSING into MISMATCH and change the persisted terminal reason.
    match = compare_process_identity(
        ProcessIdentity(state.pid, state.process_create_token)
    )
    if match in (IdentityMatch.MATCH, IdentityMatch.UNVERIFIABLE):
        return SlotDecision(
            blocked=True,
            reason="controller_lost" if match is IdentityMatch.MATCH else "process_identity_unverifiable",
        )
    # MISSING / MISMATCH: the training process is provably gone. Reconcile the
    # orphaned record to a terminal state and allow a new run. Foreign PIDs are
    # never killed; MISMATCH is only a reused-PID fact.
    reconciled = reconcile_persisted_state(
        state, controller_owned=False, identity_match=match)
    if reconciled is not None and reconciled != state:
        return SlotDecision(blocked=False, persist=reconciled)
    return SlotDecision(blocked=False)


def _blocking_attempt(record: dict) -> bool:
    """Return True when a RUNNING execution record may still hold a live subprocess."""
    status = record.get("status")
    if status != "RUNNING":
        return False
    attempts = record.get("attempts") or []
    if not attempts:
        # RUNNING with no attempt is a transient gap before the first claim.
        return False
    last = attempts[-1]
    phase = last.get("phase")
    if phase not in ("LAUNCH_INTENT", "RUNNING"):
        return False
    pid = last.get("pid")
    token = last.get("process_create_token")
    if pid is None or token is None:
        return True  # launched but identity not captured yet → cannot rule out
    match = compare_process_identity(ProcessIdentity(pid, token))
    return match in (IdentityMatch.MATCH, IdentityMatch.UNVERIFIABLE)


def hpo_storage_has_live_process(storage_root: str | os.PathLike) -> bool:
    """Scan the HPO storage root for a leftover RUNNING subprocess.

    A best-effort, bounded scan used only by the new-training gate: if any
    execution.json is unreadable/corrupt we conservatively treat it as blocking
    (we cannot prove no HPO training is active). Studies already at a terminal
    execution status never block.
    """
    root = Path(storage_root)
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("hpo_"):
            continue
        target = child / _EXECUTION_FILE
        if not target.is_file():
            # A study dir without an execution record has never started training.
            continue
        try:
            with open(target, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return True  # corrupt audit → cannot rule out an active HPO run
        if not isinstance(data, dict):
            return True
        if _blocking_attempt(data):
            return True
    return False
