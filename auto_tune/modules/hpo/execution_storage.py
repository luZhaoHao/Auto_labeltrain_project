"""H1.2 执行审计的原子 JSON 持久化与锁。

``storage_root/<study_id>/execution.json``（hpo-execution-v1）是执行与产物审计
唯一事实；study.json 仍是采样与 trial 结果事实。写入采用同目录临时文件 +
flush/fsync + ``os.replace``；每次读改写必须在 :meth:`ExecutionStore.locked`
短事务内。run/resume 全程持有根级 :meth:`ExecutionStore.runner_locked` 非阻塞
全局执行锁（``storage_root/.hpo-runner.lock``），防止同一受控根下多个 HPO
同时训练，忙时零启动（HPO_EXECUTION_BUSY）。
"""

import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .execution_models import ExecutionRecord, validate_execution
from .models import HpoError, STUDY_ID_RE
from .storage import (
    _acquire_os_lock,
    _acquire_process_lock,
    _is_reparse_point,
    _LockBusy,
    _no_duplicate_pairs,
    _proc_lock,
    _reject_nonfinite_constant,
    _release_os_lock,
    reject_link_chain,
)

EXECUTION_FILE_NAME = "execution.json"
EXECUTION_LOCK_FILE_NAME = ".execution.lock"
RUNNER_LOCK_FILE_NAME = ".hpo-runner.lock"
MAX_EXECUTION_JSON_BYTES = 5 * 1024 * 1024

_PROCESS_RUNNER_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_RUNNER_LOCKS_GUARD = threading.Lock()


def _proc_runner_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(os.path.normpath(str(path))))
    with _PROCESS_RUNNER_LOCKS_GUARD:
        lock = _PROCESS_RUNNER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_RUNNER_LOCKS[key] = lock
        return lock


def _loads_execution(text: str) -> dict:
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicate_pairs,
                         parse_constant=_reject_nonfinite_constant)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HpoError("HPO_CORRUPT_EXECUTION",
                       f"execution.json is not valid strict JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise HpoError("HPO_CORRUPT_EXECUTION", "execution.json root must be an object")
    return obj


class ExecutionStore:
    """单一 study 执行审计的原子 JSON 存储与锁。"""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._held = threading.local()

    # ── 路径与存在性 ───────────────────────────────────────────────

    def study_dir(self, study_id: str) -> Path:
        if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
            raise HpoError("HPO_NOT_FOUND", f"unknown study id {study_id!r}")
        return self._root / study_id

    def _check_study_legit(self, study_id: str) -> Path:
        """校验 study 目录已存在且非 reparse/symlink；返回目录路径。"""
        study_dir = self.study_dir(study_id)
        reject_link_chain(study_dir, code="HPO_CORRUPT_EXECUTION")
        if not study_dir.is_dir():
            raise HpoError("HPO_NOT_FOUND", f"study {study_id} not found")
        if _is_reparse_point(study_dir):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"study directory is a reparse point: {study_dir}")
        current = self._root
        try:
            rel = study_dir.relative_to(self._root)
        except ValueError as exc:
            raise HpoError("HPO_NOT_FOUND",
                           f"study {study_id} escapes storage root") from exc
        for part in rel.parts:
            current = current / part
            if _is_reparse_point(current):
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               f"reparse point in study path: {current}")
        return study_dir

    # ── 根级全局执行锁（run/resume 全程持有）──────────────────────

    @contextmanager
    def runner_locked(self) -> Iterator[None]:
        reject_link_chain(self._root, code="HPO_CORRUPT_EXECUTION")
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           "cannot create HPO storage root") from exc
        lock_path = self._root / RUNNER_LOCK_FILE_NAME
        reject_link_chain(lock_path, code="HPO_CORRUPT_EXECUTION")
        proc_lock = _proc_runner_lock(lock_path)
        if not proc_lock.acquire(blocking=False):
            raise HpoError("HPO_EXECUTION_BUSY",
                           "another HPO execution holds the global runner lock")
        fd: int | None = None
        try:
            try:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            except OSError as exc:
                raise HpoError("HPO_PERSISTENCE_ERROR",
                               "cannot open runner lock") from exc
            try:
                _acquire_os_lock(fd)
            except _LockBusy as exc:
                os.close(fd)
                fd = None
                raise HpoError("HPO_EXECUTION_BUSY",
                               "runner lock held by another process") from exc
            self._held.runner = True
            yield
        finally:
            self._held.runner = False
            if fd is not None:
                _release_os_lock(fd)
                os.close(fd)
            proc_lock.release()

    # ── study 内短事务锁 ──────────────────────────────────────────

    @contextmanager
    def locked(self, study_id: str) -> Iterator[None]:
        study_dir = self._check_study_legit(study_id)
        lock_path = study_dir / EXECUTION_LOCK_FILE_NAME
        reject_link_chain(lock_path, code="HPO_CORRUPT_EXECUTION")
        proc_lock = _proc_lock(lock_path)
        _acquire_process_lock(proc_lock, "HPO_STUDY_BUSY",
                              f"study {study_id} is busy")
        fd: int | None = None
        try:
            try:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            except OSError as exc:
                raise HpoError("HPO_PERSISTENCE_ERROR",
                               "cannot open execution lock") from exc
            try:
                _acquire_os_lock(fd)
            except _LockBusy as exc:
                os.close(fd)
                fd = None
                raise HpoError("HPO_STUDY_BUSY",
                               f"study {study_id} execution is locked by another process") from exc
            self._held.study_id = study_id
            yield
        finally:
            self._held.study_id = None
            if fd is not None:
                _release_os_lock(fd)
                os.close(fd)
            proc_lock.release()

    def _require_lock(self, study_id: str) -> None:
        if getattr(self._held, "study_id", None) != study_id:
            raise RuntimeError(
                "ExecutionStore read/write must be called inside locked(study_id)")

    # ── 读 ────────────────────────────────────────────────────────

    def read(self, study_id: str) -> ExecutionRecord:
        self._require_lock(study_id)
        study_dir = self._check_study_legit(study_id)
        target = study_dir / EXECUTION_FILE_NAME
        if not target.is_file():
            raise HpoError("HPO_NOT_FOUND",
                           f"study {study_id} has no execution record")
        if _is_reparse_point(target):
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"execution.json is a reparse point: {target}")
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"cannot stat execution.json: {exc}") from exc
        if size > MAX_EXECUTION_JSON_BYTES:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"execution.json exceeds {MAX_EXECUTION_JSON_BYTES} bytes")
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           f"execution.json unreadable: {exc}") from exc
        data = _loads_execution(text)
        return validate_execution(data, expected_study_id=study_id)

    # ── 写 ────────────────────────────────────────────────────────

    def write(self, record: ExecutionRecord) -> None:
        self._require_lock(record.study_id)
        record = validate_execution(record, expected_study_id=record.study_id)
        study_dir = self._check_study_legit(record.study_id)
        target = study_dir / EXECUTION_FILE_NAME
        reject_link_chain(target, code="HPO_CORRUPT_EXECUTION")
        if target.exists():
            # Existing corrupt facts must never be replaced with a fresh record.
            previous = self.read(record.study_id)
            if record.revision != previous.revision + 1:
                raise HpoError("HPO_CORRUPT_EXECUTION",
                               "execution revision must advance by exactly one")
        elif record.revision != 0:
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           "execution.json missing while advancing a revision")
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False,
                             allow_nan=False, sort_keys=True).encode("utf-8")
        if len(payload) > MAX_EXECUTION_JSON_BYTES:
            raise HpoError("HPO_CORRUPT_EXECUTION",
                           "execution record exceeds maximum JSON size")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=study_dir, prefix=f".{EXECUTION_FILE_NAME}.",
                suffix=".tmp", delete=False,
            ) as out:
                temp_path = Path(out.name)
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_path, target)
            temp_path = None
        except OSError as exc:
            raise HpoError("HPO_PERSISTENCE_ERROR",
                           f"failed to write execution for {record.study_id}: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
