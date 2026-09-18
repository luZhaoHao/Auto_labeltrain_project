"""F1.1-A controlled local weight store.

Single responsibility: list, import, resolve and re-verify the ``.pt`` files a
training may start from. A weight is opaque bytes — this module never
deserialises one (no ``torch.load``/``pickle``), never touches the network and
never downloads an Ultralytics checkpoint. Clients only ever see a
``model_id`` (``sha256:<64 hex>``); the server re-derives the real path and
re-checks the file identity at every training-submit boundary.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, Sequence

__all__ = ["ModelRecord", "ModelStore", "ModelStoreError"]

_CHUNK_SIZE = 1024 * 1024
_MODEL_ID_PREFIX = "sha256:"
_EXTENSION = ".pt"
_MAX_FILE_NAME = 255

MODEL_UPLOAD_INVALID_NAME = "MODEL_UPLOAD_INVALID_NAME"
MODEL_UPLOAD_INVALID_TYPE = "MODEL_UPLOAD_INVALID_TYPE"
MODEL_UPLOAD_EMPTY = "MODEL_UPLOAD_EMPTY"
MODEL_UPLOAD_TOO_LARGE = "MODEL_UPLOAD_TOO_LARGE"
MODEL_NAME_CONFLICT = "MODEL_NAME_CONFLICT"
MODEL_STORE_UNAVAILABLE = "MODEL_STORE_UNAVAILABLE"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
MODEL_CHANGED = "MODEL_CHANGED"
MODEL_PATH_UNSAFE = "MODEL_PATH_UNSAFE"
MODEL_PATH_FORBIDDEN = "MODEL_PATH_FORBIDDEN"

_STATUS_CODES = {
    MODEL_UPLOAD_INVALID_NAME: 400,
    MODEL_UPLOAD_INVALID_TYPE: 400,
    MODEL_UPLOAD_EMPTY: 400,
    MODEL_UPLOAD_TOO_LARGE: 413,
    MODEL_NAME_CONFLICT: 409,
    MODEL_STORE_UNAVAILABLE: 503,
    MODEL_NOT_FOUND: 404,
    MODEL_CHANGED: 409,
    MODEL_PATH_UNSAFE: 400,
    MODEL_PATH_FORBIDDEN: 422,
}


class ModelStoreError(Exception):
    """A stable, client-safe failure. ``message`` never contains a path."""

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = (_STATUS_CODES.get(code, 400)
                            if status_code is None else status_code)


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    name: str
    size_bytes: int
    sha256: str
    origin: Literal["managed", "legacy"]
    path: Path = field(repr=False, compare=False)
    available: bool = True
    # True only when this record was produced by a fresh import (i.e. the call
    # actually wrote the file); an idempotent re-upload reports False. Never
    # part of the public projection.
    created_now: bool = field(default=False, repr=False, compare=False)

    def public_dict(self) -> dict:
        """The only shape a client ever receives: no path, no exception text."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "origin": self.origin,
            "available": self.available,
        }


_PATH_SEPARATORS = ("/", "\\")


def _configured_basename(configured) -> str | None:
    """Return ``configured`` only when it already is a plain basename.

    Absolute paths, drive paths, UNC paths, relative paths containing a
    separator, ``.``/``..``, control characters and surrounding whitespace are
    all rejected. Both Windows and POSIX separators are checked explicitly so
    the verdict never depends on the host's ``Path.name`` behaviour — silently
    truncating a configured path into a different file is exactly what let a
    full path match a same-named controlled weight.
    """
    if not isinstance(configured, str):
        return None
    if not configured or configured != configured.strip():
        return None
    if len(configured) > _MAX_FILE_NAME:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in configured):
        return None
    if configured in {".", ".."}:
        return None
    if any(sep in configured for sep in _PATH_SEPARATORS):
        return None
    if ":" in configured:
        # 驱动器路径（C:yolov8n.pt）与 NTFS 备用数据流（yolov8n.pt:stream）
        return None
    return configured


def _identity_rank(row: ModelRecord) -> tuple:
    """同一内容的唯一胜出顺序：managed 优先，其次稳定排序的首条。"""
    return (row.origin != "managed", row.name.lower(), str(row.path).lower())


def _dedupe_identity(rows: list[ModelRecord]) -> list[ModelRecord]:
    """Keep exactly one record per ``model_id``.

    ``model_id`` is the content hash, so two files with identical bytes share an
    identity even across origins. The winner is chosen by :func:`_identity_rank`
    (managed over legacy, then name/path order) instead of by iteration order, so
    a later legacy record can never displace the managed one. Unavailable rows
    carry no identity and are never hidden.
    """
    winners: dict[str, ModelRecord] = {}
    for row in rows:
        if not row.model_id:
            continue
        current = winners.get(row.model_id)
        if current is None or _identity_rank(row) < _identity_rank(current):
            winners[row.model_id] = row
    seen: set[str] = set()
    result: list[ModelRecord] = []
    for row in rows:
        if not row.model_id:
            result.append(row)
            continue
        if row.model_id in seen:
            continue
        seen.add(row.model_id)
        result.append(winners[row.model_id])
    return result


def _is_reparse_point(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_plain_file(path: Path) -> bool:
    """A regular, non-linked file — checked with ``lstat`` so links are caught."""
    try:
        st = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if os.name == "nt" and bool(st.st_file_attributes
                                & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        return False
    return stat.S_ISREG(st.st_mode)


class ModelStore:
    """Controlled weight library over one managed root plus legacy roots."""

    def __init__(self, root: Path, *, legacy_roots: Sequence[Path] = (),
                 max_upload_bytes: int = 2_147_483_648,
                 max_models: int = 100):
        self._root = Path(root)
        self._legacy_roots = tuple(Path(row) for row in legacy_roots)
        self._max_upload_bytes = int(max_upload_bytes)
        self._max_models = int(max_models)
        # Populated by list_models()/import_stream() only; there is deliberately
        # no on-disk index — a restarted service must be re-listed before any
        # previously seen model_id becomes resolvable again.
        self._known: dict[str, ModelRecord] = {}
        self._lock = threading.Lock()

    # ── 列举 ────────────────────────────────────────────────────────

    def list_models(self) -> list[ModelRecord]:
        """Non-recursive scan of the managed root and the legacy roots.

        Training artifacts (``detect/trainN/weights/*.pt``) are never scanned:
        they stay reachable through the training history, not as initial
        weights. Unreadable entries stay visible but are marked unavailable.
        """
        rows: list[ModelRecord] = []
        for origin, base in self._scan_roots():
            if not base.is_dir() or _is_reparse_point(base):
                continue
            try:
                children = sorted(base.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            for path in children:
                if path.suffix.lower() != _EXTENSION or not _is_plain_file(path):
                    continue
                rows.append(self._describe(path, origin))
                if len(rows) >= max(self._max_models * 4, self._max_models):
                    break

        # 同一内容（同一 model_id）只投影一次：先按来源优先级选出该身份的唯一
        # 记录，再按展示顺序排序。顺序必须与本次扫描顺序无关。
        rows = _dedupe_identity(rows)
        # 受控来源优先：同名时 managed 排在 legacy 前面，顺序完全确定
        rows.sort(key=lambda row: (row.name.lower(), row.origin != "managed",
                                   row.sha256, str(row.path).lower()))
        rows = rows[:self._max_models]
        with self._lock:
            # 整体替换而不是累加：被 max_models 截断、已删除、不可读或不再公开
            # 的旧 ID 必须在刷新后失效，_known 与本次公开列表逐条一致。
            self._known = {
                row.model_id: row
                for row in rows
                if row.available and row.model_id
            }
        return rows

    def _scan_roots(self):
        yield "managed", self._root
        for base in self._legacy_roots:
            yield "legacy", base
        return

    def _describe(self, path: Path, origin: str) -> ModelRecord:
        name = path.name
        try:
            size = path.stat().st_size
            digest = _hash_file(path)
        except OSError:
            return ModelRecord(
                model_id="", name=name, size_bytes=0, sha256="",
                origin=origin, path=path, available=False)
        return ModelRecord(
            model_id=_MODEL_ID_PREFIX + digest,
            name=name,
            size_bytes=size,
            sha256=digest,
            origin=origin,
            path=Path(os.path.abspath(str(path))),
            available=True,
        )

    # ── 上传 ────────────────────────────────────────────────────────

    def import_stream(self, filename: str, stream: BinaryIO) -> ModelRecord:
        """Stream one operator-supplied ``.pt`` into the managed root.

        The bytes are opaque: they are hashed while being written to a unique
        same-directory temporary file, then committed with an atomic
        ``os.link`` that refuses to overwrite an existing name. Any failure
        removes this call's temporary file — never a half file, never a stale
        temp name.
        """
        name = self._validated_name(filename)
        root = self._ready_root()
        target = root / name
        if _is_reparse_point(target):
            raise ModelStoreError(MODEL_PATH_UNSAFE, "受控权重库中的目标不是普通文件。")

        temp = root / (".import-%s.tmp" % uuid.uuid4().hex)
        digest = hashlib.sha256()
        size = 0
        try:
            with open(temp, "xb") as handle:
                while True:
                    chunk = stream.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self._max_upload_bytes:
                        raise ModelStoreError(
                            MODEL_UPLOAD_TOO_LARGE,
                            "权重文件超过允许的大小上限，已停止上传。")
                    digest.update(chunk)
                    handle.write(chunk)
                if size == 0:
                    raise ModelStoreError(MODEL_UPLOAD_EMPTY, "权重文件为空。")
                handle.flush()
                os.fsync(handle.fileno())
            record = self._commit(temp, target, name, size, digest.hexdigest())
        except ModelStoreError:
            self._discard(temp)
            raise
        except OSError as exc:
            self._discard(temp)
            raise ModelStoreError(
                MODEL_STORE_UNAVAILABLE,
                "受控权重库写入失败，本次上传未生效。") from exc
        self._discard(temp)
        with self._lock:
            self._known[record.model_id] = record
        return record

    def _commit(self, temp: Path, target: Path, name: str, size: int,
                sha256: str) -> ModelRecord:
        try:
            # 原子创建最终名称；目标已存在时该调用失败，绝不覆盖
            os.link(temp, target)
        except FileExistsError:
            existing = self._existing_identity(target)
            if existing is not None and existing[1] == sha256:
                return ModelRecord(_MODEL_ID_PREFIX + sha256, name, existing[0],
                                   sha256, "managed", target, True)
            raise ModelStoreError(
                MODEL_NAME_CONFLICT,
                "受控权重库中已存在同名但内容不同的权重，请改名后重试。")
        return ModelRecord(_MODEL_ID_PREFIX + sha256, name, size, sha256,
                           "managed", target, True, True)

    def _existing_identity(self, target: Path):
        if not _is_plain_file(target):
            return None
        try:
            return target.stat().st_size, _hash_file(target)
        except OSError:
            return None

    def _discard(self, temp: Path) -> None:
        try:
            temp.unlink()
        except OSError:
            pass

    def _validated_name(self, filename: str) -> str:
        raw = filename if isinstance(filename, str) else ""
        if not raw or len(raw) > _MAX_FILE_NAME:
            raise ModelStoreError(MODEL_UPLOAD_INVALID_NAME, "权重文件名不合法。")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
            raise ModelStoreError(MODEL_UPLOAD_INVALID_NAME, "权重文件名不合法。")
        if raw != raw.strip():
            raise ModelStoreError(MODEL_UPLOAD_INVALID_NAME, "权重文件名不合法。")
        name = Path(raw).name
        if name != raw or not name or name in {".", ".."}:
            raise ModelStoreError(MODEL_UPLOAD_INVALID_NAME, "权重文件名不合法。")
        if Path(name).suffix.lower() != _EXTENSION:
            raise ModelStoreError(MODEL_UPLOAD_INVALID_TYPE,
                                  "只允许上传 .pt 权重文件。")
        return name

    def _ready_root(self) -> Path:
        if _is_reparse_point(self._root):
            raise ModelStoreError(MODEL_PATH_UNSAFE, "受控权重库路径不是普通目录。")
        if self._root.exists() and not self._root.is_dir():
            raise ModelStoreError(MODEL_STORE_UNAVAILABLE, "受控权重库不可用。")
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelStoreError(MODEL_STORE_UNAVAILABLE,
                                  "受控权重库不可用。") from exc
        if not _is_plain_directory(self._root):
            raise ModelStoreError(MODEL_PATH_UNSAFE, "受控权重库路径不是普通目录。")
        return self._root

    # ── 解析 ────────────────────────────────────────────────────────

    def resolve(self, model_id: str) -> Path:
        """Re-verify a previously listed id against the current file facts."""
        if not isinstance(model_id, str) or not model_id.startswith(_MODEL_ID_PREFIX):
            raise ModelStoreError(MODEL_NOT_FOUND, "未找到该受控权重，请重新选择。")
        digest = model_id[len(_MODEL_ID_PREFIX):]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ModelStoreError(MODEL_NOT_FOUND, "未找到该受控权重，请重新选择。")
        with self._lock:
            record = self._known.get(model_id)
        if record is None:
            # 从未列举/导入，或文件已被删除：都要求调用方按当前事实重新获取列表
            raise ModelStoreError(MODEL_NOT_FOUND, "未找到该受控权重，请重新选择。")
        if not _is_plain_file(record.path):
            raise ModelStoreError(MODEL_NOT_FOUND, "受控权重文件已不存在，请重新选择。")
        try:
            if _hash_file(record.path) != record.sha256:
                raise ModelStoreError(MODEL_CHANGED,
                                      "受控权重文件已被替换，请重新选择。")
        except OSError as exc:
            raise ModelStoreError(MODEL_NOT_FOUND,
                                  "受控权重文件已不存在，请重新选择。") from exc
        return record.path

    def resolve_configured_record(self, configured: str | None) -> ModelRecord:
        """Resolve a configured basename to its controlled-library record.

        The single configuration-name resolution entry: direct training (via
        :meth:`resolve_configured_name`) and the HPO defaults both go through
        it, so the path-legality rule and the managed-over-legacy precedence
        exist exactly once.

        Only a plain basename such as ``yolov8n.pt`` is accepted; any path form
        is ``MODEL_NOT_FOUND`` instead of being truncated to its last segment.

        Never falls back to the network: an unknown name is ``MODEL_NOT_FOUND``
        so Ultralytics can never implicitly download a checkpoint.
        """
        name = _configured_basename(configured)
        if name is None or not name.lower().endswith(_EXTENSION):
            raise ModelStoreError(MODEL_NOT_FOUND,
                                  "配置中的初始权重在受控权重库中不存在。")
        rows = [row for row in self.list_models()
                if row.name == name and row.available]
        for origin in ("managed", "legacy"):
            for row in rows:
                if row.origin == origin:
                    return row
        raise ModelStoreError(MODEL_NOT_FOUND,
                              "配置中的初始权重在受控权重库中不存在，请先上传。")

    def resolve_configured_name(self, configured: str | None) -> Path:
        """Return the path of :meth:`resolve_configured_record`."""
        return self.resolve_configured_record(configured).path


def _is_plain_directory(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return False
    if os.name == "nt" and bool(st.st_file_attributes
                                & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        return False
    return stat.S_ISDIR(st.st_mode)
