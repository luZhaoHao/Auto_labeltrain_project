"""In-memory registry of active run controllers (single-worker boundary).

RunControllers live only while the service process is alive. After a restart
the registry is empty and persisted running records reconcile to
``interrupted`` — the service never re-adopts a process.

Finished controllers are briefly kept in a bounded terminal retention so a
client that disconnected just before completion can still replay the run's
buffered events (including the real terminal and finalizer results) via
``after_seq``. Retention is bounded by count and TTL; evicted runs fall back
to the persisted terminal state.

Ordinary training, LLM auto-tuning, HPO execution and a fixed-config
verification run all share one process-wide training slot (H1.3). ``reserve``
hands out a single opaque reservation token; the owning controller releases it
only after it truly finishes. ``release`` refuses to release a token owned by
someone else, so a stale/disconnected caller can never free a live slot.
"""

from __future__ import annotations

import threading
import time
import uuid

RETAIN_MAX = 20
RETAIN_TTL_SECONDS = 30 * 60

# Real-training slot kinds that share the single active-training reservation.
# ``dry_run`` preview never reserves a slot.
RESERVABLE_KINDS = frozenset({"manual", "tuning", "hpo"})


class TrainingBusyError(Exception):
    """A real-training slot is already occupied; routes map this to 409."""


class TrainingGateError(Exception):
    """Invalid reservation usage (unknown kind, wrong-token release)."""


class _RetainedRun:
    __slots__ = ("controller", "retained_at")

    def __init__(self, controller):
        self.controller = controller
        self.retained_at = time.time()


class RunManager:
    def __init__(self, retain_max: int = RETAIN_MAX, retain_ttl: float = RETAIN_TTL_SECONDS):
        self._controllers: dict = {}
        self._retained: dict = {}
        self._retain_max = retain_max
        self._retain_ttl = retain_ttl
        self._lock = threading.Lock()
        self._reservation: dict | None = None  # {run_kind, run_id, token}

    def register(self, controller) -> None:
        with self._lock:
            self._controllers[controller.run_id] = controller

    def get(self, run_id: str):
        with self._lock:
            c = self._controllers.get(run_id)
            if c is not None:
                return c
            r = self._retained.get(run_id)
            if r is None:
                return None
            if time.time() - r.retained_at > self._retain_ttl:
                del self._retained[run_id]
                return None
            return r.controller

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._controllers.pop(run_id, None)
            self._retained.pop(run_id, None)

    def retain(self, run_id: str, controller) -> None:
        """Move a finished controller into bounded terminal retention.

        The broker stays alive with all buffered events, so a briefly
        disconnected client can replay missed events after completion. Oldest /
        expired retained runs are pruned so retention never grows unbounded.
        """
        with self._lock:
            self._controllers.pop(run_id, None)
            self._retained[run_id] = _RetainedRun(controller)
            self._prune_locked()

    def _prune_locked(self) -> None:
        now = time.time()
        for rid in [
            rid for rid, r in self._retained.items()
            if now - r.retained_at > self._retain_ttl
        ]:
            del self._retained[rid]
        while len(self._retained) > self._retain_max:
            oldest = min(self._retained, key=lambda k: self._retained[k].retained_at)
            del self._retained[oldest]

    def prune(self) -> None:
        """Evict expired retained runs (bounded memory without new traffic)."""
        with self._lock:
            self._prune_locked()

    def retained_count(self) -> int:
        with self._lock:
            return len(self._retained)

    def active_manual(self):
        with self._lock:
            for c in self._controllers.values():
                if c.run_kind == "manual" and c.is_active():
                    return c
        return None

    def active_tuning(self):
        with self._lock:
            for c in self._controllers.values():
                if c.run_kind == "tuning" and c.is_active():
                    return c
        return None

    def _controller_for_kind(self, run_kind: str):
        """Return the active controller of ``run_kind`` (explicit kind only)."""
        if run_kind == "manual":
            return self.active_manual()
        if run_kind == "tuning":
            return self.active_tuning()
        if run_kind == "hpo":
            return self.active_hpo()
        raise TrainingGateError(f"unknown run kind for active lookup: {run_kind!r}")

    def active_for_kind(self, run_kind: str):
        """Explicit-kind active lookup; unknown kinds must never map to tuning."""
        return self._controller_for_kind(run_kind)

    def active_hpo(self):
        with self._lock:
            for c in self._controllers.values():
                if c.run_kind == "hpo" and c.is_active():
                    return c
        return None

    def active_train(self):
        """The single active real-training controller across manual/tuning/hpo."""
        with self._lock:
            for c in self._controllers.values():
                if c.run_kind in RESERVABLE_KINDS and c.is_active():
                    return c
        return None

    def _has_active_locked(self) -> bool:
        for c in self._controllers.values():
            if c.run_kind in RESERVABLE_KINDS and c.is_active():
                return True
        return False

    # ── single training-slot reservation (H1.3) ────────────────────

    def reserve(self, run_kind: str, run_id: str) -> str:
        """Atomically occupy the single training slot and return an opaque token.

        Fails with ``TrainingBusyError`` when another reservation or an active
        real-training controller is present. The caller must pass the returned
        token to the owning controller and release it only on true completion.
        """
        if run_kind not in RESERVABLE_KINDS:
            raise TrainingGateError(f"kind {run_kind!r} cannot reserve a training slot")
        with self._lock:
            if self._reservation is not None:
                raise TrainingBusyError(
                    f"training slot already reserved by {self._reservation['run_kind']}:"
                    f"{self._reservation['run_id']}")
            if self._has_active_locked():
                raise TrainingBusyError("an active training controller already occupies the slot")
            token = uuid.uuid4().hex
            self._reservation = {"run_kind": run_kind, "run_id": run_id, "token": token}
            return token

    def release(self, token: str) -> bool:
        """Release the reservation only when ``token`` is the owning token.

        A foreign token is refused (raises ``TrainingGateError``) and never
        releases someone else's live slot. Returns True when a reservation was
        actually released.
        """
        with self._lock:
            if self._reservation is None:
                return False
            if self._reservation["token"] != token:
                raise TrainingGateError("reservation token mismatch; refusing foreign release")
            self._reservation = None
            return True

    def reservation_owner(self) -> tuple | None:
        """Public read of the current reservation ``(run_kind, run_id)`` or None."""
        with self._lock:
            if self._reservation is None:
                return None
            return (self._reservation["run_kind"], self._reservation["run_id"])

    def snapshot(self) -> list:
        with self._lock:
            return list(self._controllers.values())


_RUN_MANAGER = RunManager()
