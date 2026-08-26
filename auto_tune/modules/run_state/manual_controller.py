"""Background controller for ordinary (manual) training runs.

The controller owns the training subprocess, reads stdout/stderr, writes
``training.log``, updates the ``RunState``, publishes structured events to the
``EventBroker``, waits for the process to end, runs the finalizer, and writes
the terminal state — all independent of any SSE client. SSE is only a
subscriber; disconnecting it never cancels the controller.
"""

from __future__ import annotations

import asyncio
import datetime

from .events import EventBroker
from .models import RunStatePersistenceError
from .service import (
    with_last_event,
    with_status_phase,
    with_terminal,
    write_run_state,
)
from auto_tune.modules.agent_engine.training_log import process_training_output_line
from auto_tune.modules.run_state.process_identity import capture_process_identity

DETAIL_PERSIST_THROTTLE = 10


class ManualRunController:
    run_kind = "manual"

    def __init__(
        self,
        *,
        run_state,
        state_file,
        cmd,
        params,
        train_name,
        train_dir,
        data_yaml,
        model,
        epochs,
        log_path,
        finalize_cb,
        broker,
        manager,
    ):
        self.run_id = run_state.run_id
        self.run_state = run_state
        self.state_file = state_file
        self.cmd = cmd
        self.params = params
        self.train_name = train_name
        self.train_dir = train_dir
        self.data_yaml = data_yaml
        self.model = model
        self.epochs = epochs
        self.log_path = log_path
        self.finalize_cb = finalize_cb
        self.broker = broker
        self.manager = manager
        self.started_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        self._task = None
        self._proc = None
        self._lock = __import__("threading").Lock()
        self._done = False
        self._stop_requested = False
        self._stop_applied = False

    # ── lifecycle / thread-safety ──

    def is_active(self) -> bool:
        with self._lock:
            return not self._done

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def _mark_done(self) -> None:
        with self._lock:
            self._done = True

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def wait_done(self, timeout: float) -> None:
        """Block until the controller finishes (or raise asyncio.TimeoutError)."""
        if self._task is None:
            return
        await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)

    # ── state / event helpers ──

    def _persist(self, state=None) -> None:
        try:
            write_run_state(self.state_file, state or self.run_state)
        except RunStatePersistenceError:
            pass

    def _publish(self, event, event_type=None, persist=False):
        """Publish an event (unique seq) and keep last_event consistent.

        Every published event is stamped with ``run_id``/``phase`` so that
        SSE consumers never see a field-less event.
        """
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

    # ── stop (stopping phase, then cancelled after confirmed end) ──

    def _finish_cancelled_before_start(self) -> None:
        """Publish and persist a cancelled terminal when no process was created."""
        stamped = self.broker.publish({
            "status": "cancelled", "level": "info",
            "message": "训练已取消（未启动训练进程）",
            "run_id": self.run_id, "phase": "terminal", "event": "lifecycle",
        })
        self.run_state = with_terminal(
            self.run_state,
            status="cancelled",
            event_type="lifecycle",
            message=stamped["message"],
            seq=stamped["event_seq"],
        )
        self._persist()

    def request_stop(self) -> bool:
        """Request a stop: mark ``stopping`` and terminate the subprocess.

        ``starting``/``preparing`` runs also accept a stop. While the
        subprocess is not yet created this returns True and the background
        task honors the request by either skipping process creation or
        terminating the fresh process immediately after it is created. Returns
        True when stopping was initiated (or the process already ended); False
        only when a live process genuinely cannot be terminated (caller must
        not report a fake success).
        """
        if self.is_done():
            return False
        self._stop_requested = True
        self.run_state = with_status_phase(self.run_state, phase="stopping")
        self._persist()
        proc = self._proc
        if proc is None:
            # Stop arrived before the subprocess exists; the background task
            # honors it (skip creation or terminate right after creation).
            return True
        if proc.returncode is not None:
            return True
        try:
            proc.terminate()
            self._stop_applied = True
            return True
        except ProcessLookupError:
            return True
        except Exception:
            try:
                proc.kill()
                self._stop_applied = True
                return True
            except Exception:
                return False

    # ── main background task ──

    async def _run(self) -> None:
        proc = None
        try:
            self._publish({
                "status": "running", "level": "info",
                "message": f"启动训练: {self.train_name}",
                "train_name": self.train_name,
                "run_id": self.run_id, "phase": self.run_state.phase,
            })
            self._publish({
                "status": "running", "level": "info",
                "message": f"数据集: {self.data_yaml}",
                "run_id": self.run_id, "phase": self.run_state.phase,
            })
            self._publish({
                "status": "running", "level": "info",
                "message": (
                    f"模型: {self.model}  |  轮次: {self.epochs}  |  "
                    f"batch: {self.params.get('batch')}  |  imgsz: {self.params.get('imgsz')}"
                ),
                "run_id": self.run_id, "phase": self.run_state.phase,
            })

            if self._stop_requested:
                # Stop arrived while starting: honor it without ever creating a
                # subprocess. Nothing trained, so no finalizer runs.
                self._stop_applied = True
                self._finish_cancelled_before_start()
                return

            proc = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 128,
            )
            self._proc = proc

            if self._stop_requested:
                # Stop raced with process creation (already unavoidable):
                # terminate the fresh process and let the read loop below
                # observe EOF and the confirmed exit.
                try:
                    proc.terminate()
                    self._stop_applied = True
                except ProcessLookupError:
                    pass
                except Exception:
                    try:
                        proc.kill()
                        self._stop_applied = True
                    except Exception:
                        pass

            identity = capture_process_identity(proc.pid)
            self.run_state = with_status_phase(
                self.run_state,
                status="running",
                phase="training",
                pid=proc.pid,
                process_create_token=identity.process_create_token if identity else None,
            )
            self._publish({
                "status": "running", "level": "info",
                "message": f"训练进程已启动 (PID {proc.pid})",
                "run_id": self.run_id, "phase": self.run_state.phase,
                "event": "lifecycle",
            }, event_type="lifecycle", persist=True)

            warned_once = set()
            epoch_keys = set()
            detail_since_persist = 0
            while True:
                line_b = await proc.stdout.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                payload = process_training_output_line(
                    line, self.train_name, self.log_path, warned_once
                )
                if payload is None:
                    continue
                if payload.get("event") == "training_log" and payload.get("log_kind") in ("epoch", "validation"):
                    key = (payload.get("log_kind"), payload.get("epoch"), payload.get("message"))
                    if key in epoch_keys:
                        payload["message"] = None
                    else:
                        epoch_keys.add(key)
                self._publish(payload)
                if payload.get("log_kind") in ("epoch", "validation", "lifecycle", "warning", "error") or payload.get("event") == "log_persistence_error":
                    self._persist()
                    detail_since_persist = 0
                else:
                    detail_since_persist += 1
                    if detail_since_persist >= DETAIL_PERSIST_THROTTLE:
                        self._persist()
                        detail_since_persist = 0

            await proc.wait()
            returncode = proc.returncode

            if self._stop_applied:
                terminal_status = "cancelled"
            elif returncode == 0:
                # A stop request that never actually terminated the process does
                # not change the process's own outcome: natural completion stays
                # completed.
                terminal_status = "completed"
            else:
                # Stop never took effect and the process exited non-zero on its
                # own: that is a failure, not a cancellation.
                terminal_status = "failed"

            stamped = self.broker.publish({
                "status": terminal_status, "level": "info",
                "message": f"训练进程退出 (exit code {returncode})",
                "run_id": self.run_id, "phase": "terminal", "event": "lifecycle",
            })
            self.run_state = with_terminal(
                self.run_state,
                status=terminal_status,
                event_type="lifecycle",
                message=stamped["message"],
                seq=stamped["event_seq"],
            )
            self._persist()

            if self.finalize_cb is not None:
                try:
                    final_event = self.finalize_cb(self, returncode)
                    if final_event:
                        self.broker.publish(final_event)
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                self.run_state = with_status_phase(self.run_state, status="failed", phase="terminal")
                stamped = self.broker.publish({
                    "status": "error", "level": "error",
                    "message": f"异常: {exc}",
                    "run_id": self.run_id, "phase": "terminal",
                })
                self.run_state = with_terminal(
                    self.run_state, status="failed", event_type="error",
                    message=stamped["message"], seq=stamped["event_seq"],
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
