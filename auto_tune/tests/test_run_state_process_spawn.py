"""The training child must start and stream on the Windows selector loop.

The Windows web start selects the selector loop, because the Proactor loop fails
every incoming connection with WinError 10014 on some machines. That same loop
cannot create subprocesses through ``asyncio``'s own API, so the controller must
start a ``subprocess.Popen`` child instead and consume its pipe without blocking
the event loop.

These tests run a *real*, short-lived child process on a *real* selector loop:
they do not stand in a fake subprocess for the behaviour the product depends on.
The first test pins the premise (the loop really cannot do asyncio subprocesses),
the rest pin what the controller needs from its own way of starting one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

# A real child that never touches the product's files, datasets or network.
_CHILD_TIMEOUT = 30.0


def _child(code: str) -> list[str]:
    """A real, side-effect-free Python child process, unbuffered."""
    return [sys.executable, "-u", "-c", textwrap.dedent(code)]


def _child_is_gone(proc) -> bool:
    """Whether the started child no longer exists.

    On POSIX the PID is gone once the child has been reaped. On Windows an
    exited process object lingers for as long as the parent keeps the ``Popen``
    handle open — that is the standard library's own bookkeeping, not a running
    child — so the observable fact there is the child's known exit code.
    """
    if proc.returncode is None:
        return False
    if os.name == "nt":
        return True
    from auto_tune.modules.run_state.process_identity import capture_process_identity

    return capture_process_identity(proc.pid) is None


@pytest.fixture
def selector_loop():
    """A loop built exactly the way the web start builds one.

    The policy is really set and ``asyncio.new_event_loop()`` really builds the
    loop from it — this is the call chain uvicorn follows with ``loop="none"``.
    The process-wide policy is restored afterwards so no other test inherits it.
    """
    original = asyncio.get_event_loop_policy()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    try:
        assert not isinstance(loop, getattr(asyncio, "ProactorEventLoop", ())), (
            "the fixture must build the loop the Windows web start uses")
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop_policy(original)


def _run(selector_loop, coro):
    return selector_loop.run_until_complete(coro)


# ── the premise, kept as evidence ──


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only the Windows selector loop refuses asyncio subprocesses; on "
           "POSIX the selector loop is the default and supports them",
)
def test_the_selector_loop_cannot_create_an_asyncio_subprocess(selector_loop):
    """Why the controller may not use ``asyncio.create_subprocess_exec``.

    On this loop that call raises ``NotImplementedError``: a web start would
    succeed and every training start would then die at launch time.
    """
    async def scenario():
        with pytest.raises(NotImplementedError):
            await asyncio.create_subprocess_exec(
                *_child("pass"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

    _run(selector_loop, scenario())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the POSIX contrast to the Windows premise: the same loop does "
           "support asyncio subprocesses there",
)
def test_the_same_loop_supports_asyncio_subprocesses_on_posix(selector_loop):
    """The premise above is Windows-specific, so the delivery must not depend
    on it: on POSIX the identical selector loop creates asyncio subprocesses,
    and the ``Popen`` adapter the Windows start needs works on that loop too."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        native = await asyncio.create_subprocess_exec(
            *_child("print('native')"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        native_line = await asyncio.wait_for(native.stdout.readline(),
                                             timeout=_CHILD_TIMEOUT)
        await asyncio.wait_for(native.wait(), timeout=_CHILD_TIMEOUT)

        adapted = await spawn_training_process(*_child("print('adapted')"))
        try:
            adapted_line = await asyncio.wait_for(adapted.stdout.readline(),
                                                  timeout=_CHILD_TIMEOUT)
            await asyncio.wait_for(adapted.wait(), timeout=_CHILD_TIMEOUT)
        finally:
            if adapted.returncode is None:
                adapted.kill()
                await adapted.wait()
        return native_line, adapted_line

    native_line, adapted_line = _run(selector_loop, scenario())

    assert native_line.strip() == b"native"
    assert adapted_line.strip() == b"adapted"


# ── what the controller needs instead ──


def test_a_real_child_streams_stdout_and_reports_its_exit_code(selector_loop):
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("""
            print('epoch 1')
            print('epoch 2')
        """))
        try:
            lines = []
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
                if not line:
                    break
                lines.append(line.decode("utf-8", errors="replace").rstrip())
            return lines, await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    lines, returncode = _run(selector_loop, scenario())

    assert lines == ["epoch 1", "epoch 2"]
    assert returncode == 0


def test_a_failing_child_reports_a_nonzero_exit_code(selector_loop):
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("raise SystemExit(3)"))
        try:
            while await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT):
                pass
            return await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    assert _run(selector_loop, scenario()) == 3


def test_a_non_utf8_byte_sequence_does_not_break_the_stream(selector_loop):
    """The child's output is decoded with ``errors='replace'``, as before."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child(r"""
            import sys
            sys.stdout.buffer.write(b'epoch \xff\xfe done\n')
            sys.stdout.buffer.flush()
        """))
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return line
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    raw = _run(selector_loop, scenario())

    assert raw.decode("utf-8", errors="replace").rstrip() == "epoch �� done"


def test_the_merged_stderr_streams_through_the_same_pipe(selector_loop):
    """stdout and stderr stay merged, so the log order is the child's own."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("""
            import sys
            print('out first')
            print('err second', file=sys.stderr)
        """))
        try:
            lines = []
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
                if not line:
                    break
                lines.append(line.decode("utf-8", errors="replace").rstrip())
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return lines
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    assert _run(selector_loop, scenario()) == ["out first", "err second"]


def test_a_long_running_child_can_be_stopped(selector_loop):
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("""
            import time
            print('started')
            time.sleep(120)
        """))
        try:
            first = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            assert first.strip() == b"started"
            assert proc.returncode is None, "a live child must not look finished"
            proc.terminate()
            return await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    returncode = _run(selector_loop, scenario())

    assert returncode is not None and returncode != 0


def test_the_event_loop_keeps_running_other_coroutines_while_the_child_works(selector_loop):
    """Reading the child must never block the web event loop."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("""
            import time
            print('a')
            time.sleep(0.5)
            print('b')
        """))
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            before = ticks
            second = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            after = ticks
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return before, after, second
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    before, after, second = _run(selector_loop, scenario())

    assert second.strip() == b"b"
    assert after - before >= 3, (
        "the loop stopped running other coroutines while the child was working")


# ── the controller itself, on a real child and a real selector loop ──


def _controller(tmp_path, cmd):
    from auto_tune.modules.run_state.events import EventBroker
    from auto_tune.modules.run_state.manager import RunManager
    from auto_tune.modules.run_state.manual_controller import ManualRunController
    from auto_tune.modules.run_state.service import new_run_state

    run_state = new_run_state("manual", run_name="train1")
    state_file = os.path.join(str(tmp_path), "training_running.json")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    controller = ManualRunController(
        run_state=run_state,
        state_file=state_file,
        cmd=cmd,
        params={"batch": 1, "imgsz": 64},
        train_name="train1",
        train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt",
        epochs=1,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None,
        broker=broker,
        manager=manager,
    )
    manager.register(controller)
    return controller, state_file


def test_the_controller_completes_a_real_child_on_the_selector_loop(tmp_path, selector_loop):
    from auto_tune.modules.run_state.service import read_run_state

    controller, state_file = _controller(tmp_path, _child("""
        print('  1/1  1.20G  1.234  0.456  0.789')
        print('all 10 50 0.123 0.456 0 0.111')
    """))

    _run(selector_loop, controller._run())

    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "completed"
    assert state.phase == "terminal"
    assert state.pid is not None, "the child's identity must be recorded"
    assert state.process_create_token, "the process identity token must be recorded"
    assert controller.is_done() is True


def test_the_controller_stops_a_real_long_running_child_on_the_selector_loop(
    tmp_path, selector_loop
):
    """A stop that really terminates the child ends the run ``cancelled``."""
    from auto_tune.modules.run_state.service import read_run_state

    controller, state_file = _controller(tmp_path, _child("""
        import time
        print('started')
        time.sleep(120)
    """))

    async def scenario():
        task = asyncio.ensure_future(controller._run())
        for _ in range(600):
            if controller._proc is not None:
                break
            await asyncio.sleep(0.05)
        assert controller._proc is not None, "the child was never started"
        # The child really is alive and really is a process we own.
        assert controller._proc.returncode is None
        assert controller.request_stop() is True
        await asyncio.wait_for(task, timeout=_CHILD_TIMEOUT)

    _run(selector_loop, scenario())

    state = read_run_state(state_file, run_kind="manual")
    assert state is not None
    assert state.status == "cancelled"
    assert state.phase == "terminal"
    assert controller.is_done() is True


# ── starting the child must not block the loop either ──


def test_a_slow_child_creation_still_lets_the_event_loop_work(selector_loop, monkeypatch):
    """``Popen`` itself is a blocking call chain, so it runs off the loop too.

    A child that takes a while to be created (a large interpreter start, a slow
    filesystem) must not freeze the SSE stream and every other request while it
    happens.
    """
    import time

    import auto_tune.modules.run_state.manual_controller as mc

    real_popen = subprocess.Popen

    def slow_popen(*args, **kwargs):
        time.sleep(0.4)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(mc, "subprocess", types.SimpleNamespace(
        Popen=slow_popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            proc = await asyncio.wait_for(
                mc.spawn_training_process(*_child("print('ok')")),
                timeout=_CHILD_TIMEOUT)
            during_creation = ticks
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return during_creation, line
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    during_creation, line = _run(selector_loop, scenario())

    assert line.strip() == b"ok", "the slow-created child still streams its output"
    assert during_creation >= 10, (
        "the loop stopped running other coroutines while the child was created")


# ── the read boundary the previous stream enforced ──


def test_the_read_limit_is_the_boundary_the_previous_stream_used():
    """128 KiB, exactly the ``limit`` the asyncio subprocess call used before."""
    from auto_tune.modules.run_state.manual_controller import STDOUT_LINE_LIMIT

    assert STDOUT_LINE_LIMIT == 1024 * 128


def test_a_short_line_still_arrives_whole_and_alone(selector_loop):
    """The bound must not split the ordinary output YOLO writes."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child(r"""
            import sys
            sys.stdout.buffer.write(b'epoch 1\n')
            sys.stdout.buffer.flush()
        """))
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT)
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return line
        finally:
            proc.close()
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    line = _run(selector_loop, scenario())

    assert line == b"epoch 1\n"


def test_a_line_longer_than_the_limit_is_consumed_in_bounded_chunks(selector_loop):
    """A single line of hundreds of KiB must never be read into memory at once.

    The child writes one newline-less line far past the bound and then a normal
    line: every piece handed to the controller is bounded, the bytes are all
    seen exactly once, and the short line after it is still one whole line.
    """
    from auto_tune.modules.run_state.manual_controller import (
        STDOUT_LINE_LIMIT,
        spawn_training_process,
    )

    long_length = STDOUT_LINE_LIMIT * 3 + 17

    async def scenario():
        proc = await spawn_training_process(*_child(f"""
            import sys
            sys.stdout.buffer.write(b'x' * {long_length} + b'\\n')
            sys.stdout.buffer.write(b'after\\n')
            sys.stdout.buffer.flush()
        """))
        try:
            chunks = []
            while True:
                chunk = await asyncio.wait_for(proc.stdout.readline(),
                                               timeout=_CHILD_TIMEOUT)
                if not chunk:
                    break
                chunks.append(chunk)
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            return chunks
        finally:
            proc.close()
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    chunks = _run(selector_loop, scenario())

    assert all(len(chunk) <= STDOUT_LINE_LIMIT for chunk in chunks), (
        "a piece larger than the read bound must never be handed over")
    assert b"".join(chunks) == b"x" * long_length + b"\nafter\n"
    assert len(chunks) >= 4, "the long line must really have been split"
    assert chunks[-1] == b"after\n", "the short line after the long one is unchanged"


# ── releasing the child's resources ──


def test_the_process_pipe_can_be_closed_twice(selector_loop):
    """``close`` is idempotent: every exit path calls it, and a path may run
    after another one already did."""
    from auto_tune.modules.run_state.manual_controller import spawn_training_process

    async def scenario():
        proc = await spawn_training_process(*_child("print('done')"))
        try:
            while await asyncio.wait_for(proc.stdout.readline(), timeout=_CHILD_TIMEOUT):
                pass
            await asyncio.wait_for(proc.wait(), timeout=_CHILD_TIMEOUT)
            assert proc.closed is False, "a live pipe is not closed yet"
            proc.close()
            assert proc.closed is True, "the pipe is released"
            proc.close()
            return proc.closed
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    assert _run(selector_loop, scenario()) is True


def test_the_controller_closes_the_pipe_of_a_completed_child(tmp_path, selector_loop):
    """A finished run owns no pipe any more."""
    controller, _ = _controller(tmp_path, _child("print('epoch 1')"))

    _run(selector_loop, controller._run())

    assert controller._proc is not None
    assert controller._proc.returncode == 0
    assert controller._proc.closed is True


def test_the_controller_closes_the_pipe_of_a_failed_child(tmp_path, selector_loop):
    controller, _ = _controller(tmp_path, _child("raise SystemExit(3)"))

    _run(selector_loop, controller._run())

    assert controller._proc.returncode == 3
    assert controller._proc.closed is True


def test_the_controller_closes_the_pipe_of_a_stopped_child(tmp_path, selector_loop):
    controller, _ = _controller(tmp_path, _child("""
        import time
        print('started')
        time.sleep(120)
    """))

    async def scenario():
        task = asyncio.ensure_future(controller._run())
        for _ in range(600):
            if controller._proc is not None:
                break
            await asyncio.sleep(0.05)
        assert controller.request_stop() is True
        await asyncio.wait_for(task, timeout=_CHILD_TIMEOUT)

    _run(selector_loop, scenario())

    assert controller._proc.closed is True


def test_a_cancelled_controller_leaves_no_child_and_releases_the_slot_last(
    tmp_path, selector_loop
):
    """A cancelled controller is not allowed to abandon a live child.

    The task is cancelled while the child runs; the child must be gone, its pipe
    closed, and the shared training slot must only be released once both are
    true — otherwise a new run could start alongside a process still training.
    """
    import auto_tune.modules.run_state.manual_controller as mc
    from auto_tune.modules.run_state.events import EventBroker
    from auto_tune.modules.run_state.manager import RunManager
    from auto_tune.modules.run_state.service import new_run_state

    run_state = new_run_state("manual", run_name="train1")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    token = manager.reserve("manual", run_state.run_id)

    observations = []
    real_release = manager.release

    def recording_release(tok):
        proc = controller._proc
        observations.append(("release",
                             proc is None or proc.returncode is not None,
                             proc is None or proc.closed))
        return real_release(tok)

    manager.release = recording_release

    controller = mc.ManualRunController(
        run_state=run_state,
        state_file=os.path.join(str(tmp_path), "training_running.json"),
        cmd=_child("""
            import time
            print('started')
            time.sleep(120)
        """),
        params={"batch": 1, "imgsz": 64},
        train_name="train1",
        train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt",
        epochs=1,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None,
        broker=broker,
        manager=manager,
        reservation_token=token,
    )
    manager.register(controller)

    async def scenario():
        task = asyncio.ensure_future(controller._run())
        for _ in range(600):
            if controller._proc is not None:
                break
            await asyncio.sleep(0.05)
        assert controller._proc is not None, "the child was never started"
        assert controller._proc.returncode is None, "the child must be alive"
        await asyncio.sleep(0.1)  # let the read loop reach its first read
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return controller._proc

    proc = _run(selector_loop, scenario())

    assert _child_is_gone(proc), "the child must not survive the cancellation"
    assert proc.closed is True, "the pipe must be closed"
    assert manager.reservation_owner() is None, "the slot was released"
    assert observations == [("release", True, True)], (
        "the slot is released last, after the child exited with its pipe closed")


def _reserved_controller(tmp_path, cmd, release_observations=None):
    """A controller that really owns the single training slot.

    ``release_observations`` (when given) records, for every slot release, the
    child's exit and pipe facts *as seen at release time* — the ordering the
    cleanup has to guarantee.
    """
    from auto_tune.modules.run_state.events import EventBroker
    from auto_tune.modules.run_state.manager import RunManager
    from auto_tune.modules.run_state.manual_controller import ManualRunController
    from auto_tune.modules.run_state.service import new_run_state

    run_state = new_run_state("manual", run_name="train1")
    broker = EventBroker(run_state.run_id)
    manager = RunManager()
    token = manager.reserve("manual", run_state.run_id)

    controller = ManualRunController(
        run_state=run_state,
        state_file=os.path.join(str(tmp_path), "training_running.json"),
        cmd=cmd,
        params={"batch": 1, "imgsz": 64},
        train_name="train1",
        train_dir=str(tmp_path),
        data_yaml=os.path.join(str(tmp_path), "data.yaml"),
        model="yolov8n.pt",
        epochs=1,
        log_path=os.path.join(str(tmp_path), "training.log"),
        finalize_cb=None,
        broker=broker,
        manager=manager,
        reservation_token=token,
    )
    manager.register(controller)

    if release_observations is not None:
        real_release = manager.release

        def recording_release(tok):
            proc = controller._proc
            release_observations.append((
                "release",
                proc is None or proc.returncode is not None,
                proc is None or proc.closed,
            ))
            return real_release(tok)

        manager.release = recording_release

    return controller, manager


def test_a_cancellation_during_child_creation_still_reaps_the_child(
    tmp_path, selector_loop, monkeypatch
):
    """``Popen`` runs in a worker thread the event loop cannot stop.

    Cancelling the controller while the child is being created must therefore
    wait for that thread to hand the child back, adopt it and reap it — not
    return with ``self._proc`` still ``None`` while a brand-new process is born
    behind it, owned by nobody.
    """
    import auto_tune.modules.run_state.manual_controller as mc

    observations = []
    controller, manager = _reserved_controller(
        tmp_path,
        _child("""
            import time
            print('started')
            time.sleep(120)
        """),
        release_observations=observations,
    )

    real_popen = subprocess.Popen
    creating = threading.Event()
    creation_finished = threading.Event()
    created = []

    def slow_popen(*args, **kwargs):
        creating.set()
        try:
            time.sleep(0.6)
            popen = real_popen(*args, **kwargs)
            created.append(popen)
            return popen
        finally:
            creation_finished.set()

    monkeypatch.setattr(mc, "subprocess", types.SimpleNamespace(
        Popen=slow_popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))

    async def scenario():
        task = asyncio.ensure_future(controller._run())
        for _ in range(600):
            if creating.is_set():
                break
            await asyncio.sleep(0.01)
        assert creating.is_set(), "the child was never being created"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return controller._proc

    try:
        proc = _run(selector_loop, scenario())
    finally:
        # The creation thread cannot be stopped, so wait for it to hand the
        # child over: a regression here leaves it running, and it must never
        # outlive the test.
        creation_finished.wait(10)
        for popen in created:
            if popen.poll() is None:
                popen.kill()
                popen.wait()

    assert proc is not None, "the child created during the cancellation was dropped"
    assert _child_is_gone(proc), "the child must not survive the cancellation"
    assert proc.closed is True, "the pipe must be closed"
    assert manager.reservation_owner() is None, "the slot was released"
    assert observations == [("release", True, True)], (
        "the slot is released last, after the child exited with its pipe closed")


def test_a_repeated_cancellation_cannot_skip_the_kill(
    tmp_path, selector_loop, monkeypatch
):
    """A second cancel during the terminate grace must not jump to the finally.

    The child ignores ``terminate`` and only exits when it is killed. The
    controller is cancelled inside the read loop and cancelled twice more while
    the child is still inside its terminate grace: the later cancellations are
    held back, cleanup runs exactly once, and the slot is released only after
    the child really exited with its pipe closed.
    """
    import auto_tune.modules.run_state.manual_controller as mc

    observations = []
    controller, manager = _reserved_controller(
        tmp_path,
        _child("""
            import time
            print('started')
            time.sleep(120)
        """),
        release_observations=observations,
    )

    events = []
    real_spawn = mc.spawn_training_process

    async def spawn_that_ignores_terminate(*cmd):
        proc = await real_spawn(*cmd)
        original_kill = proc.kill
        original_close = proc.close

        def terminate_ignored():
            events.append("terminate")

        def kill_recorded():
            events.append("kill")
            original_kill()

        def close_recorded():
            events.append("close")
            original_close()

        proc.terminate = terminate_ignored
        proc.kill = kill_recorded
        proc.close = close_recorded
        return proc

    monkeypatch.setattr(mc, "spawn_training_process", spawn_that_ignores_terminate)
    monkeypatch.setattr(mc, "TERMINATE_GRACE_SECONDS", 1.0)
    monkeypatch.setattr(mc, "KILL_GRACE_SECONDS", 1.0)

    async def scenario():
        task = asyncio.ensure_future(controller._run())
        for _ in range(600):
            if controller._proc is not None:
                break
            await asyncio.sleep(0.05)
        assert controller._proc is not None, "the child was never started"
        await asyncio.sleep(0.1)  # let the read loop reach its first read
        task.cancel()
        for _ in range(600):
            if events:
                break
            await asyncio.sleep(0.01)
        assert events, "cleanup never started"
        await asyncio.sleep(0.1)
        task.cancel()  # still inside the terminate grace
        await asyncio.sleep(0.1)
        task.cancel()  # and again, on the same cleanup
        with contextlib.suppress(asyncio.CancelledError):
            await task
        proc = controller._proc
        survived = proc is not None and not _child_is_gone(proc)
        if proc is not None and proc.returncode is None:
            # A regression leaves the child alive: never let it outlive the test.
            proc.kill()
            await proc.wait()
        return proc, survived

    proc, survived = _run(selector_loop, scenario())

    assert survived is False, "no orphan child may outlive the cancellations"
    assert events.count("terminate") == 1, "cleanup ran more than once"
    assert events.count("kill") == 1, "cleanup ran more than once"
    assert events.count("close") == 1
    assert _child_is_gone(proc), "the child must not survive the cancellations"
    assert proc.closed is True, "the pipe must be closed"
    assert manager.reservation_owner() is None, "the slot was released"
    assert observations == [("release", True, True)], (
        "the slot is released last, after the child exited with its pipe closed")


def test_running_repeatedly_leaves_no_child_or_pipe_behind(tmp_path, selector_loop):
    """Three sequential runs accumulate nothing: no live child, no open pipe."""
    completed = []

    async def run_once():
        controller, _ = _controller(tmp_path, _child("print('epoch 1')"))
        await controller._run()
        completed.append(controller._proc)

    for _ in range(3):
        _run(selector_loop, run_once())

    assert len(completed) == 3
    for proc in completed:
        assert proc.returncode == 0, "no child is still running"
        assert proc.closed is True, "no pipe is left open"
        assert _child_is_gone(proc)
