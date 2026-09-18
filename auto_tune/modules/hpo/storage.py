"""H1.1 原子 JSON 持久化与同 study 排他锁。

JSON 是 HPO 唯一持久化事实（``storage_root/<study_id>/study.json``）。写入使用
同目录临时文件 + flush/fsync + ``os.replace``；读改写必须在
:meth:`StudyStore.locked` 上下文内进行，进程内使用共享 ``threading.Lock``，
跨进程使用 OS 排他文件锁（Windows ``msvcrt`` / POSIX ``fcntl``）。忙时立即
返回 ``HPO_STUDY_BUSY``，不使用手工过期时间强占锁。
"""

import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import HpoError, STUDY_ID_RE, StudyRecord
from .validation import validate_record

STUDY_FILE_NAME = "study.json"
LOCK_FILE_NAME = ".study.lock"
MAX_STUDY_JSON_BYTES = 5 * 1024 * 1024

_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()

# Opt-in, thread-scoped "the HPO execution session owns this thread" marker.
# Read-only callers keep the historical behaviour (fail fast with the busy code,
# so a client gets an explicit, retryable "temporarily busy"). The execution
# session instead waits out a short holder: otherwise a concurrent read could
# defeat the writer, which is exactly the Studio defect this guards against.
_WRITER_WAIT = threading.local()


class _LockBusy(Exception):
    pass


@contextmanager
def writer_waits_for_lock(seconds: float):
    """Let this thread's storage transactions wait out a short lock holder."""
    previous = getattr(_WRITER_WAIT, "seconds", None)
    _WRITER_WAIT.seconds = float(seconds)
    try:
        yield
    finally:
        _WRITER_WAIT.seconds = previous


def _writer_wait_seconds() -> float:
    return float(getattr(_WRITER_WAIT, "seconds", 0.0) or 0.0)


def _acquire_process_lock(proc_lock: threading.Lock,
                          busy_code: str, busy_message: str) -> None:
    """Take the in-process lock, honouring the writer-wait window.

    ``wait <= 0`` keeps the original non-blocking behaviour exactly: no lock is
    ever held longer, no verification is skipped, nothing is made lockless —
    only the *writer* is allowed to wait for a reader's short transaction.
    """
    wait = _writer_wait_seconds()
    acquired = proc_lock.acquire(timeout=wait) if wait > 0 \
        else proc_lock.acquire(blocking=False)
    if not acquired:
        raise HpoError(busy_code, busy_message)


def _is_reparse_point(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def reject_link_chain(path: Path, *, code='HPO_CORRUPT_STUDY') -> None:
    """Check original path components with lstat, before any mkdir or open.

    Do not resolve symlinks first: that would discard the evidence being checked.
    Missing components are allowed for a new controlled directory.
    """
    absolute = Path(os.path.abspath(path))
    if '..' in Path(path).parts:
        raise HpoError(code, 'parent traversal is not allowed in HPO paths')
    for current in (*reversed(absolute.parents), absolute):
        try:
            st = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HpoError(code, 'HPO path cannot be inspected') from exc
        linked = stat.S_ISLNK(st.st_mode) or (
            os.name == 'nt' and bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT))
        if linked:
            raise HpoError(code, 'HPO paths must not contain links or reparse points')


def _proc_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(os.path.normpath(str(path))))
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _acquire_os_lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\x00")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise _LockBusy() from exc
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise _LockBusy() from exc


def _release_os_lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _no_duplicate_pairs(pairs):
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str):
    raise ValueError(f"non-finite JSON constant: {token!r}")


def _loads_strict(text: str) -> dict:
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicate_pairs,
                         parse_constant=_reject_nonfinite_constant)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HpoError("HPO_CORRUPT_STUDY",
                       f"study.json is not valid strict JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise HpoError("HPO_CORRUPT_STUDY", "study.json root must be an object")
    return obj


class StudyStore:
    """单一 study 的原子 JSON 存储与锁。"""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._held = threading.local()

    # ── 路径与存在性 ────────────────────────────────────────────────

    def study_dir(self, study_id: str) -> Path:
        if not isinstance(study_id, str) or not STUDY_ID_RE.fullmatch(study_id):
            raise HpoError("HPO_NOT_FOUND", f"unknown study id {study_id!r}")
        return self._root / study_id

    def _check_dir_legit(self, study_id: str) -> Path:
        """校验 study 目录存在且非 reparse/symlink；返回目录路径。"""
        study_dir = self.study_dir(study_id)
        reject_link_chain(study_dir)
        if not study_dir.is_dir():
            raise HpoError("HPO_NOT_FOUND", f"study {study_id} not found")
        if _is_reparse_point(study_dir):
            raise HpoError("HPO_CORRUPT_STUDY",
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
                raise HpoError("HPO_CORRUPT_STUDY",
                               f"reparse point in study path: {current}")
        return study_dir

    # ── 锁 ──────────────────────────────────────────────────────────

    @contextmanager
    def locked(self, study_id: str) -> Iterator[None]:
        study_dir = self._check_dir_legit(study_id)
        lock_path = study_dir / LOCK_FILE_NAME
        reject_link_chain(lock_path)
        proc_lock = _proc_lock(lock_path)
        _acquire_process_lock(proc_lock, "HPO_STUDY_BUSY",
                              f"study {study_id} is busy")
        fd: int | None = None
        try:
            try:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            except OSError as exc:
                raise HpoError('HPO_PERSISTENCE_ERROR', 'cannot open study lock') from exc
            try:
                _acquire_os_lock(fd)
            except _LockBusy as exc:
                os.close(fd)
                fd = None
                raise HpoError("HPO_STUDY_BUSY",
                               f"study {study_id} is locked by another process") from exc
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
                "StudyStore read/write must be called inside locked(study_id)")

    # ── 读 ──────────────────────────────────────────────────────────

    def read(self, study_id: str) -> StudyRecord:
        self._require_lock(study_id)
        study_dir = self._check_dir_legit(study_id)
        target = study_dir / STUDY_FILE_NAME
        if not target.is_file():
            raise HpoError("HPO_NOT_FOUND", f"study {study_id} has no published record")
        if _is_reparse_point(target):
            raise HpoError("HPO_CORRUPT_STUDY",
                           f"study.json is a reparse point: {target}")
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise HpoError("HPO_CORRUPT_STUDY",
                           f"cannot stat study.json: {exc}") from exc
        if size > MAX_STUDY_JSON_BYTES:
            raise HpoError("HPO_CORRUPT_STUDY",
                           f"study.json exceeds {MAX_STUDY_JSON_BYTES} bytes")
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HpoError("HPO_CORRUPT_STUDY",
                           f"study.json unreadable: {exc}") from exc
        data = _loads_strict(text)
        return validate_record(data, expected_study_id=study_id)

    # ── 写 ──────────────────────────────────────────────────────────

    def write(self, record: StudyRecord) -> None:
        self._require_lock(record.study_id)
        record = validate_record(record, expected_study_id=record.study_id)
        study_dir = self._check_dir_legit(record.study_id)
        target = study_dir / STUDY_FILE_NAME
        reject_link_chain(target)
        if target.exists():
            # Existing corrupt facts must never be replaced with a fresh record.
            self.read(record.study_id)
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False,
                             allow_nan=False, sort_keys=True).encode("utf-8")
        if len(payload) > MAX_STUDY_JSON_BYTES:
            raise HpoError('HPO_CORRUPT_STUDY', 'study exceeds maximum JSON size')
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=study_dir, prefix=f".{STUDY_FILE_NAME}.",
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
                           f"failed to write study {record.study_id}: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
