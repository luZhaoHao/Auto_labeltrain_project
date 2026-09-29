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
import subprocess

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

# The bound on one read from the child's merged pipe: the same 128 KiB the
# asyncio subprocess call used (``limit=1024 * 128``), now applied to the pipe's
# own ``readline``. A line longer than this is consumed as several bounded
# pieces instead of being read into memory whole.
STDOUT_LINE_LIMIT = 1024 * 128

# How long the child gets to exit after ``terminate`` before it is killed, and
# how long the kill itself is waited for. Both are bounded so a child that
# ignores signals can never hold the training slot open forever.
TERMINATE_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 5.0

# How often a bounded wait re-checks the child's exit code.
EXIT_POLL_INTERVAL = 0.05


class _TrainingStdout:
    """The child's merged output pipe, read off the event loop."""

    def __init__(self, pipe):
        self._pipe = pipe

    async def readline(self) -> bytes:
        return await asyncio.to_thread(self._pipe.readline, STDOUT_LINE_LIMIT)

    @property
    def closed(self) -> bool:
        return self._pipe.closed

    def close(self) -> None:
        """Release the read end; a second call is a no-op."""
        if not self._pipe.closed:
            self._pipe.close()


class TrainingProcess:
    """A started training child with the surface the controller awaits.

    ``asyncio``'s own subprocess support is unavailable on the Windows selector
    loop — the only loop that can serve this product's sockets on Windows — so
    the child is a plain ``subprocess.Popen`` and its pipe is drained in worker
    threads. Only what the controller needs is exposed; the handle itself stays
    private so a blocking call cannot leak onto the loop.
    """

    def __init__(self, popen):
        self._popen = popen
        self.pid = popen.pid
        self.stdout = _TrainingStdout(popen.stdout)

    @property
    def returncode(self):
        """The live exit code, or ``None`` while the child still runs."""
        return self._popen.poll()

    @property
    def closed(self) -> bool:
        """Whether the child's output pipe has been released."""
        return self.stdout.closed

    def close(self) -> None:
        """Release the child's output pipe. Idempotent, so every exit path can
        call it without having to know whether another one already did."""
        self.stdout.close()

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()

    async def wait(self) -> int:
        return await asyncio.to_thread(self._popen.wait)

    async def wait_bounded(self, timeout: float) -> bool:
        """Wait up to ``timeout`` for the child to end; True when it has.

        Polls the exit code instead of parking a worker thread on ``wait``, so a
        child that outlives its grace period leaves only a stopped poll behind.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if self.returncode is not None:
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(EXIT_POLL_INTERVAL)


async def spawn_training_process(*cmd: str) -> TrainingProcess:
    """Start one training child: merged stderr, unparsed bytes, real exit code.

    The signature mirrors ``asyncio.create_subprocess_exec`` — an already
    validated command array, never a shell string — so the call site stays the
    same shape as the API it replaces. ``Popen`` itself blocks, so it runs in a
    worker thread: creating a slow child must not freeze the web event loop.
    """
    popen = await asyncio.to_thread(
        subprocess.Popen,
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return TrainingProcess(popen)


async def _await_uninterrupted(coro, cancelled: list):
    """Wait for ``coro`` to finish even when this task is cancelled again.

    Child creation and cleanup both run as their own task, precisely so that a
    repeated ``CancelledError`` cannot abandon them half-done: each one is
    awaited through ``shield`` and every cancellation is recorded in
    ``cancelled`` while the *same* task is awaited again — the work is never
    restarted and never dropped. Only the caller re-raises the cancellation,
    once the task has actually finished.
    """
    task = asyncio.ensure_future(coro)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.append(True)
            if task.done():
                return task.result()


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
        reservation_token=None,
    ):
        self.run_id = run_state.run_id
        self.reservation_token = reservation_token
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

            proc = await self._spawn_child()

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
            # The controller task is being cancelled, but the child must not be
            # abandoned with it. The training slot is released only after this.
            await self._stop_child()
            raise
        except Exception as exc:
            await self._stop_child()
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
            # The read end is released on every exit path, after the child has
            # been waited for, and before the training slot is handed back.
            self._close_child_pipe()
            self._mark_done()
            try:
                self.manager.retain(self.run_id, self)
            except Exception:
                pass
            self._release_reservation()

    async def _spawn_child(self) -> TrainingProcess:
        """Create the child so an outer cancellation cannot orphan it.

        ``Popen`` blocks in a worker thread that the event loop cannot stop once
        it is running, so the spawn is tracked as its own task: cancelling the
        controller while the child is being created waits for that task to hand
        the child back instead of returning with ``self._proc`` still ``None``
        while a brand-new process is born behind it, owned by nobody.
        """
        cancelled: list = []
        proc = await _await_uninterrupted(
            spawn_training_process(*self.cmd), cancelled)
        # Adopt the child before anything else can look at the controller, so
        # every path from here on reaps it instead of dropping it.
        self._proc = proc
        if cancelled:
            raise asyncio.CancelledError
        return proc

    async def _stop_child(self) -> None:
        """Leave no live child behind on the paths that skip the read loop.

        The controller task can be cancelled (or hit an unexpected error) at any
        await point, so the child may still be running with its pipe open. The
        reap runs as one uninterruptible cleanup: a cancellation arriving while
        it is under way only delays this coroutine, it never skips the kill and
        never reaches the ``finally`` block with the child still alive.
        """
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        cancelled: list = []
        await _await_uninterrupted(self._reap_child(proc), cancelled)

    async def _reap_child(self, proc) -> None:
        """Terminate one child, then kill it and confirm it really exited.

        ``terminate`` gets a bounded grace period; a child that ignores it is
        killed, and the kill is only accepted once the child has actually
        exited — a ``False`` from the bounded wait is never ignored, because the
        training slot must not be handed back while a process may still train.
        """
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        except Exception:
            pass
        wait_bounded = getattr(proc, "wait_bounded", None)
        if wait_bounded is None:
            # A stand-in process (tests) ends as soon as it is told to.
            await proc.wait()
            return
        if await wait_bounded(TERMINATE_GRACE_SECONDS):
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        except Exception:
            pass
        while not await wait_bounded(KILL_GRACE_SECONDS):
            # ``kill`` cannot be ignored by a real process, so a child still
            # there after it is killed again rather than left behind alive.
            try:
                proc.kill()
            except ProcessLookupError:
                return
            except Exception:
                pass

    def _close_child_pipe(self) -> None:
        """Release the child's read end; idempotent and safe on every path."""
        proc = self._proc
        if proc is None:
            return
        close = getattr(proc, "close", None)
        if close is None:
            return
        try:
            close()
        except OSError:
            pass

    def _release_reservation(self) -> None:
        """Free the training slot only after the controller truly finished.

        A foreign/expired token must never release someone else's slot, so any
        mismatch is swallowed here (the slot belongs to a different owner).
        """
        token = self.reservation_token
        if token is None:
            return
        try:
            self.manager.release(token)
        except Exception:
            pass
        finally:
            self.reservation_token = None
