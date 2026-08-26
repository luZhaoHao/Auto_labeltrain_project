"""In-memory registry of active run controllers (single-worker boundary).

RunControllers live only while the service process is alive. After a restart
the registry is empty and persisted running records reconcile to
``interrupted`` — the service never re-adopts a process.

Finished controllers are briefly kept in a bounded terminal retention so a
client that disconnected just before completion can still replay the run's
buffered events (including the real terminal and finalizer results) via
``after_seq``. Retention is bounded by count and TTL; evicted runs fall back
to the persisted terminal state.
"""

from __future__ import annotations

import threading
import time

RETAIN_MAX = 20
RETAIN_TTL_SECONDS = 30 * 60


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

    def active_for_kind(self, run_kind: str):
        return self.active_manual() if run_kind == "manual" else self.active_tuning()

    def snapshot(self) -> list:
        with self._lock:
            return list(self._controllers.values())


_RUN_MANAGER = RunManager()
