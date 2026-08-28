"""Background controller for auto-tuning runs.

The tuning loop runs in a daemon thread owned by this controller. The
controller bridges the loop's ``on_progress`` / ``on_state`` callbacks into
the ``EventBroker`` (unique, strictly increasing ``event_seq``), keeps the
``RunState`` current, and persists the terminal state after the thread ends —
independent of any SSE client. SSE is only a subscriber.
"""

from __future__ import annotations

import threading

from .events import EventBroker
from .models import RunStatePersistenceError
from .service import (
    with_last_event,
    with_status_phase,
    with_terminal,
    write_run_state,
)

DETAIL_PERSIST_THROTTLE = 10


class TuningRunController:
    run_kind = "tuning"

    def __init__(
        self,
        *,
        run_state,
        state_file,
        broker,
        manager,
        loop_runner,
    ):
        self.run_id = run_state.run_id
        self.state_file = state_file
        self.broker = broker
        self.manager = manager
        self.loop_runner = loop_runner
        self.cancel_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._done = False
        self._run_state = run_state

    # ── lifecycle / thread-safety ──

    @property
    def run_state(self):
        with self._lock:
            return self._run_state

    @run_state.setter
    def run_state(self, value):
        with self._lock:
            self._run_state = value

    def is_active(self) -> bool:
        with self._lock:
            return not self._done

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def _mark_done(self) -> None:
        with self._lock:
            self._done = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="tuning-run-controller", daemon=True)
        self._thread.start()

    def wait_done(self, timeout: float) -> bool:
        """Synchronously wait for the thread to finish. Returns True if done."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    # ── state / event helpers ──

    def _persist(self) -> None:
        try:
            write_run_state(self.state_file, self.run_state)
        except RunStatePersistenceError:
            pass

    def _publish(self, event, event_type=None, persist=False):
        stamped = dict(event)
        stamped.setdefault("run_id", self.run_id)
        stamped.setdefault("phase", self.run_state.phase)
        stamped = self.broker.publish(stamped)
        ev_type = event_type or stamped.get("event") or "run_event"
        self.run_state = with_last_event(
            self.run_state,
            event_type=ev_type,
            message=stamped.get("message"),
            seq=stamped["event_seq"],
        )
        if persist:
            self._persist()
        return stamped

    # ── stop ──

    def request_stop(self) -> bool:
        """Request cancellation and mark ``stopping``; tuning can always be
        asked to stop, so this returns True unless already terminal."""
        with self._lock:
            if self._done:
                return False
            # Mutate the backing field directly: the run_state property setter
            # re-acquires the same (non-reentrant) lock, which would deadlock.
            self._run_state = with_status_phase(self._run_state, phase="stopping")
        self.cancel_event.set()
        self._persist()
        return True

    # ── background thread ──

    def _run(self) -> None:
        last_iteration = [0]
        detail_since_persist = 0

        def on_state(phase, event_type, message, process_identity=None):
            nonlocal detail_since_persist
            if process_identity is not None:
                self.run_state = with_status_phase(
                    self.run_state,
                    status="running",
                    phase=phase,
                    pid=process_identity.pid,
                    process_create_token=process_identity.process_create_token,
                )
            else:
                self.run_state = with_status_phase(self.run_state, status="running", phase=phase)
            self._persist()
            self._publish({
                "status": "running", "phase": phase,
                "message": message, "event_type": event_type,
                "run_id": self.run_id,
            }, event_type=event_type, persist=False)
            detail_since_persist = 0

        def on_progress(iteration, message, step=None, params=None, **kwargs):
            nonlocal detail_since_persist
            if iteration != last_iteration[0]:
                last_iteration[0] = iteration
                self._publish({
                    "status": "running", "level": "info",
                    "message": f"--- Iter {iteration} ---",
                    "iteration": iteration, "step": "iteration_start",
                    "run_id": self.run_id, "phase": self.run_state.phase,
                })
            data = {
                "status": "running",
                "message": message,
                "level": kwargs.get("level", "info"),
                "iteration": iteration,
                "run_id": self.run_id,
                "phase": self.run_state.phase,
            }
            if step:
                data["step"] = step
            if params:
                data["params"] = params
            for key in ("event", "log_kind", "detail", "epoch", "total_epochs", "train_name"):
                if key in kwargs and kwargs[key] is not None:
                    data[key] = kwargs[key]
            self._publish(data)
            if data.get("log_kind") in ("epoch", "validation", "lifecycle", "warning", "error") or data.get("event") == "log_persistence_error":
                self._persist()
                detail_since_persist = 0
            else:
                detail_since_persist += 1
                if detail_since_persist >= DETAIL_PERSIST_THROTTLE:
                    self._persist()
                    detail_since_persist = 0

        result = None
        error = None
        try:
            result = self.loop_runner(
                on_progress=on_progress,
                on_state=on_state,
                cancel_event=self.cancel_event,
            )
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                if error:
                    if "用户取消" in str(error):
                        terminal_status = "cancelled"
                        terminal_msg = "训练已取消"
                    else:
                        terminal_status = "failed"
                        terminal_msg = f"调优失败: {error}"
                elif result is None:
                    terminal_status = "failed"
                    terminal_msg = "调优返回空结果"
                else:
                    result_error = result.get("error")
                    if result_error and "用户取消" in str(result_error):
                        terminal_status = "cancelled"
                        terminal_msg = "训练已取消"
                    else:
                        terminal_status = "completed" if not result_error else "failed"
                        terminal_msg = result_error or "调优完成"
                projected = {}
                if isinstance(result, dict):
                    # Redacted projection: only relative names, metrics and
                    # stable statuses reach the page — never an absolute path.
                    projected = {
                        "best_iteration": result.get("best_iteration"),
                        "best_train_name": result.get("best_train_name"),
                        "best_metrics": result.get("best_metrics"),
                        "best_score": result.get("best_score"),
                        "eval_mode": result.get("eval_mode"),
                        "final_summary_status": result.get("final_summary_status"),
                        "llm_summary_status": result.get("llm_summary_status"),
                        "summary_persistence_status": result.get("summary_persistence_status"),
                        # Public-safe projection only (short ids / names).
                        "reference_dataset": result.get("reference_dataset"),
                    }
                stamped = self.broker.publish({
                    "status": terminal_status, "phase": "terminal",
                    "message": terminal_msg,
                    "run_id": self.run_id,
                    "event_type": "tuning_terminal",
                    "result": projected,
                })
                self.run_state = with_terminal(
                    self.run_state,
                    status=terminal_status,
                    event_type="tuning_terminal",
                    message=terminal_msg,
                    seq=stamped["event_seq"],
                )
                self._persist()
            except Exception:
                pass
            finally:
                self._mark_done()
                try:
                    self.manager.retain(self.run_id, self)
                except Exception:
                    pass
