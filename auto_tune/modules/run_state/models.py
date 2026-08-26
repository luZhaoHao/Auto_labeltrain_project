"""Run-state schema and value objects (Studio S1.5).

Normal training and auto-tuning share one versioned, atomically persisted
run-state contract. Field names and semantics are a compatibility boundary;
do not rename them without a migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

RUN_KINDS = {"manual", "tuning", "unknown"}
RUN_KINDS_CREATE = {"manual", "tuning"}
RUN_STATUSES = {
    "starting",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
}
RUN_PHASES = {
    "preparing",
    "launching",
    "training",
    "analyzing",
    "finalizing",
    "stopping",
    "terminal",
}


class RunStateError(Exception):
    """Base error for run-state domain failures."""


class RunStateValidationError(RunStateError):
    """Raised when a run state violates schema constraints."""


class RunStatePersistenceError(RunStateError):
    """Raised when an atomic run-state write fails."""


@dataclass(frozen=True)
class LastEvent:
    """Latest structured event; ``seq`` strictly increases within one run."""

    seq: int
    type: str
    at: str
    message: str | None = None


@dataclass(frozen=True)
class ProcessIdentity:
    """Process identity bound to a run: ``(pid, process_create_token)``."""

    pid: int
    process_create_token: str


@dataclass(frozen=True)
class RunState:
    """Versioned, persisted run-state record shared by manual/tuning runs."""

    schema_version: str
    run_id: str | None
    run_kind: str
    status: str
    phase: str
    started_at: str
    updated_at: str
    pid: int | None
    process_create_token: str | None
    last_event: LastEvent | None
    run_name: str | None
    terminal_reason: str | None

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise RunStateValidationError(f"invalid status: {self.status!r}")
        if self.phase not in RUN_PHASES:
            raise RunStateValidationError(f"invalid phase: {self.phase!r}")
        if self.run_kind not in RUN_KINDS:
            raise RunStateValidationError(f"invalid run_kind: {self.run_kind!r}")
        if self.pid is not None and (not isinstance(self.pid, int) or self.pid <= 0):
            raise RunStateValidationError(f"invalid pid: {self.pid!r}")
        if self.last_event is not None:
            if not isinstance(self.last_event.seq, int) or self.last_event.seq < 0:
                raise RunStateValidationError(
                    f"last_event.seq must be a non-negative integer, got {self.last_event.seq!r}"
                )
