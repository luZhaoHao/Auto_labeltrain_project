"""Background controller for an HPO execution (H1.3 Studio 接入).

The HpoRunner is the execution/persistence authority; this controller only runs
the blocking ``runner.run``/``runner.resume`` inside one daemon worker thread so
FastAPI event loops are never blocked, forwards a ``stop_event`` and releases
the shared training-slot reservation when the worker truly finishes. HTTP
requests only start/stop/query this controller; a disconnected browser never
cancels it.
"""

from __future__ import annotations

import re
import threading
import time

from auto_tune.modules.hpo import HpoError

# Transient short-transaction conflicts: a concurrent read-only request may hold
# the study/execution lock for the duration of one read. They are absorbed by a
# bounded internal retry instead of being treated as a terminal worker failure.
BUSY_RETRY_CODES = frozenset({"HPO_STUDY_BUSY", "HPO_EXECUTION_BUSY"})

# The retry budget is bounded by *time*, not by a fixed attempt count: a fixed
# count can be consumed in microseconds by a burst of overlapping reads, which
# would re-introduce the very failure this retries around. The last backoff
# value repeats until the window closes.
BUSY_RETRY_WINDOW_SECONDS = 10.0
BUSY_RETRY_BACKOFF = (0.05, 0.1, 0.2, 0.4, 0.8)

_ERROR_CODE_RE = re.compile(r"^HPO_[A-Z0-9_]{1,64}$")


class HpoController:
    """Owns exactly one HpoRunner worker thread for one study."""

    run_kind = "hpo"

    def __init__(self, *, study_id: str, runner, manager, reservation_token=None):
        self.study_id = study_id
        self.runner = runner
        self.manager = manager
        self.reservation_token = reservation_token
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._done = False
        self._started = False
        self.error_code: str | None = None
        # Number of transient-conflict retries this worker needed (observability
        # for support and for deterministic contention tests).
        self.busy_retries = 0

    # ── identity ──────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        # The manager keys active controllers by run_id; the study is the only
        # identity the UI needs for HPO.
        return self.study_id

    # ── lifecycle / thread-safety ─────────────────────────────────

    def is_active(self) -> bool:
        with self._lock:
            return not self._done

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def _mark_done(self) -> None:
        with self._lock:
            self._done = True

    def start(self, resume: bool = False) -> bool:
        """Spawn the single worker thread for this study.

        Returns True when a worker was started by this call; False when this
        controller already started one (duplicate start/resume is a no-op and
        never creates a second worker).
        """
        with self._lock:
            if self._started:
                return False
            self._started = True
            self._done = False
        thread = threading.Thread(
            target=self._run,
            kwargs={"resume": resume},
            name=f"hpo-runner-{self.study_id}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return True

    def wait_done(self, timeout: float) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def request_stop(self) -> bool:
        """Ask the runner to stop at the next safe point (trial boundary).

        Only meaningful while this controller still owns an active worker;
        returns False once the worker already finished. Does not claim the
        process has exited — that fact is converged by the runner.
        """
        if self.is_done():
            return False
        self._stop_event.set()
        return True

    # ── worker ────────────────────────────────────────────────────

    def _run(self, resume: bool) -> None:
        backoff = BUSY_RETRY_BACKOFF
        window = BUSY_RETRY_WINDOW_SECONDS
        deadline = time.monotonic() + window
        attempt = 0
        try:
            while True:
                try:
                    if attempt == 0 and not resume:
                        self.runner.run(self.study_id, stop_event=self._stop_event)
                    else:
                        # Every retry re-enters through the idempotent recovery
                        # entry point. ``run()`` must never be replayed: the
                        # busy error may have happened after a launch intent was
                        # published, and only the H1.2 recovery matrix knows how
                        # to reconcile that without a second launch.
                        self.runner.resume(self.study_id, stop_event=self._stop_event)
                    return
                except Exception as exc:  # noqa: BLE001 - terminal boundary
                    if (isinstance(exc, HpoError)
                            and exc.code in BUSY_RETRY_CODES
                            and not self._stop_event.is_set()):
                        delay = backoff[min(attempt, len(backoff) - 1)]
                        if time.monotonic() + delay <= deadline:
                            self.busy_retries += 1
                            attempt += 1
                            if self._stop_event.wait(delay):
                                # The user stopped during backoff: nothing was
                                # launched by this worker, so this is not a
                                # failure and must not be reported as one.
                                return
                            continue
                    self._record_error(exc)
                    return
        finally:
            self._mark_done()
            self._finish()

    def _record_error(self, exc: Exception) -> None:
        """Keep only a stable, displayable error code.

        The raw exception text is deliberately dropped rather than stored: it can
        embed absolute paths, commands, PIDs or credentials, and the status
        projection must never be able to echo it.
        """
        if isinstance(exc, HpoError) and isinstance(exc.code, str) \
                and _ERROR_CODE_RE.match(exc.code):
            self.error_code = exc.code
        else:
            self.error_code = "HPO_EXECUTION_ERROR"

    def _finish(self) -> None:
        """Publish the terminal fact, then free the training slot.

        A failed worker must stay readable after it ends: unregistering it would
        immediately erase the only record of why it stopped and let the page
        fall back to "READY with no error". Failures therefore move into the
        manager's bounded terminal retention (the same mechanism the manual and
        tuning controllers use for their finished runs), while successful runs
        are removed as before.
        """
        try:
            if self.error_code is None:
                self.manager.unregister(self.run_id)
            else:
                self.manager.retain(self.run_id, self)
        except Exception:
            pass
        token = self.reservation_token
        if token is not None:
            try:
                self.manager.release(token)
            except Exception:
                pass
            finally:
                self.reservation_token = None
