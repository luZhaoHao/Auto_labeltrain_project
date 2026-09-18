"""H1.3 Task 1: HpoController single-worker semantics + reservation release.

Uses a fake runner that blocks until allowed to finish — never starts a real
YOLO subprocess or network call. Verifies: reservation held across the whole
worker lifetime, stop request does not release early, release happens only on
true completion, duplicate start/resume never creates a second worker.
"""

import threading

import pytest

from auto_tune.modules.hpo import HpoError
from auto_tune.modules.run_state.manager import RunManager
from auto_tune.ui.hpo_controller import HpoController


class _BlockingRunner:
    """Runner that records calls and blocks until the test lets it finish."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started = threading.Event()
        self.allow_finish = threading.Event()
        self.run_calls = 0
        self.resume_calls = 0
        self.last_stop_event = None

    def _record(self, attr):
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def _wait_until_released(self):
        self.started.set()
        self.allow_finish.wait(10)

    def run(self, study_id, *, stop_event=None):
        self._record("run_calls")
        self.last_stop_event = stop_event
        self._wait_until_released()
        return None

    def resume(self, study_id, *, stop_event=None):
        self._record("resume_calls")
        self.last_stop_event = stop_event
        self._wait_until_released()
        return None


class _RaisingRunner:
    def run(self, study_id, *, stop_event=None):
        raise HpoError("HPO_CORRUPT_EXECUTION", "injected failure")


def _study(prefix="hpo"):
    return f"{prefix}_" + "0" * 32


def test_reservation_held_whole_worker_and_released_on_done():
    manager = RunManager()
    runner = _BlockingRunner()
    token = manager.reserve("hpo", _study())
    controller = HpoController(
        study_id=_study(), runner=runner, manager=manager,
        reservation_token=token)
    manager.register(controller)
    try:
        assert controller.start() is True
        assert runner.started.wait(5) is True
        assert controller.is_active() is True
        assert manager.active_hpo() is controller
        assert manager.active_for_kind("hpo") is controller
        assert manager.reservation_owner() is not None

        # stop request does not release while the worker is still converging
        assert controller.request_stop() is True
        assert manager.reservation_owner() is not None
        assert manager.active_hpo() is controller

        runner.allow_finish.set()
        assert controller.wait_done(10) is True
        assert controller.is_done() is True
        assert manager.reservation_owner() is None
        assert manager.active_hpo() is None
    finally:
        runner.allow_finish.set()
        manager.unregister(controller.run_id)


def test_duplicate_start_and_start_resume_no_second_worker():
    manager = RunManager()
    runner = _BlockingRunner()
    controller = HpoController(study_id=_study("hpo"), runner=runner,
                               manager=manager)
    manager.register(controller)
    try:
        assert controller.start() is True
        assert runner.started.wait(5) is True
        # duplicate start and a racing resume must not spawn a second worker
        assert controller.start() is False
        assert controller.start(resume=True) is False
        runner.allow_finish.set()
        assert controller.wait_done(10) is True
        assert runner.run_calls == 1
        assert runner.resume_calls == 0
    finally:
        runner.allow_finish.set()
        manager.unregister(controller.run_id)


def test_resume_path_calls_runner_resume():
    manager = RunManager()
    runner = _BlockingRunner()
    controller = HpoController(study_id=_study("hpo"), runner=runner,
                               manager=manager)
    manager.register(controller)
    try:
        assert controller.start(resume=True) is True
        assert runner.started.wait(5) is True
        runner.allow_finish.set()
        assert controller.wait_done(10) is True
        assert runner.resume_calls == 1
        assert runner.run_calls == 0
    finally:
        runner.allow_finish.set()
        manager.unregister(controller.run_id)


def test_error_capture_and_release_on_runner_failure():
    manager = RunManager()
    runner = _RaisingRunner()
    token = manager.reserve("hpo", _study("hpo"))
    controller = HpoController(study_id=_study("hpo"), runner=runner,
                               manager=manager, reservation_token=token)
    manager.register(controller)
    try:
        controller.start()
        assert controller.wait_done(10) is True
        assert controller.is_done() is True
        assert controller.error_code == "HPO_CORRUPT_EXECUTION"
        assert manager.reservation_owner() is None
        assert manager.active_hpo() is None
    finally:
        manager.unregister(controller.run_id)


def test_request_stop_after_done_returns_false():
    manager = RunManager()
    runner = _BlockingRunner()
    controller = HpoController(study_id=_study("hpo"), runner=runner,
                               manager=manager)
    manager.register(controller)
    try:
        controller.start()
        runner.allow_finish.set()
        assert controller.wait_done(10) is True
        assert controller.request_stop() is False
    finally:
        runner.allow_finish.set()
        manager.unregister(controller.run_id)
