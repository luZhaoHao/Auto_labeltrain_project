"""Credential store and resolver for Studio API keys.

Priority per purpose: environment variable, then the writable persistent store
of the current platform, then missing. The persistent store is the Windows
Credential Manager on Windows and a JSON credential file on Linux (the
container has no OS credential store, so a key typed into the page has to
survive a container restart). Windows never falls back to the file.

Resolved values are cached in-process for at most five minutes and never enter
config, YAML, URL, response, history, audit or logs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

CredentialPurpose = Literal["text", "vision"]
CredentialSource = Literal[
    "environment", "windows_credential_manager", "file_credential_store", "missing"
]

_PURPOSES: dict[str, tuple[str, str]] = {
    "text": ("AUTO_TUNE_TEXT_API_KEY", "AutoTuneStudio/text/deepseek"),
    "vision": ("AUTO_TUNE_VISION_API_KEY", "AutoTuneStudio/vision/qwen"),
}

_CACHE_TTL_SECONDS = 300.0

# The user-facing line for a text-service credential gap. It keeps the stable
# ``credential_missing`` code every caller greps for, and adds what the code
# cannot say: which service is unavailable and where the key is saved.
TEXT_CREDENTIAL_MISSING_MESSAGE = (
    "DeepSeek API error: credential_missing"
    "（尚未保存 DeepSeek API Key，文本诊断与大模型调参不可用。"
    "请在网页“AI 服务配置”中保存 DeepSeek API Key。）"
)

# Project-defined placeholders must never be treated as real credentials.
_PLACEHOLDER_VALUES = {"YOUR_DEEPSEEK_API_KEY", "YOUR_QWEN_API_KEY"}

# The file backend: an explicit path, a bounded read and bounded values.
CREDENTIALS_PATH_ENV = "AUTO_TUNE_CREDENTIALS_PATH"
CONTAINER_CREDENTIALS_PATH = Path("/data/secrets/credentials.json")
_MAX_FILE_BYTES = 64 * 1024
_MAX_CREDENTIAL_LENGTH = 512

# Fixed, safe messages: a failure never echoes a value, an absolute path or an
# exception detail.
_FILE_UNREADABLE = "凭据文件无法读取或格式非法，请检查或删除该文件后重试。"
_FILE_TOO_LARGE = "凭据文件过大，已拒绝读取。"
_DIRECTORY_UNWRITABLE = "凭据存储目录不可写，无法保存凭据。"
_WRITE_FAILED = "凭据保存失败，原有凭据未被修改。"

# Function (not bool) so tests can monkeypatch platform detection.
def _is_windows() -> bool:
    return sys.platform == "win32"

# purpose -> (resolved_value, monotonic_expiry)
_cache: dict[str, tuple[str, float]] = {}
_last_tested_at: dict[str, str] = {}
_last_test_result: dict[str, str] = {}

# One lock for the file backend: the two purposes share one file, so a
# read-modify-write is a single critical section.
_store_lock = threading.RLock()


class CredentialError(Exception):
    """Safe credential failure that never embeds secret material."""


class UnsupportedPlatformError(CredentialError):
    """Persistent OS credential store is unavailable on this platform."""


@dataclass(frozen=True)
class CredentialStatus:
    purpose: CredentialPurpose
    configured: bool
    source: CredentialSource
    writable: bool
    last_tested_at: str | None = None
    last_test_result: str | None = None


def _validate_purpose(purpose: object) -> str:
    try:
        valid = purpose in _PURPOSES
    except TypeError:
        # Unhashable values (e.g. a list) are never valid purposes.
        valid = False
    if not valid:
        raise ValueError(f"invalid credential purpose: {purpose!r}")
    return str(purpose)


def _is_usable(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.strip() != ""
        and value not in _PLACEHOLDER_VALUES
    )


def _is_valid_length(value: str) -> bool:
    return len(value) <= _MAX_CREDENTIAL_LENGTH


# ---------------------------------------------------------------------------
# File backend (Linux / container). Windows keeps the Credential Manager.
# ---------------------------------------------------------------------------


def credentials_file_path() -> Path:
    """The credential file the file backend reads and writes.

    ``AUTO_TUNE_CREDENTIALS_PATH`` selects it explicitly; a blank value is
    treated as unset so a mistyped variable can never break the settings page.
    """
    raw = os.environ.get(CREDENTIALS_PATH_ENV)
    if raw is None:
        return CONTAINER_CREDENTIALS_PATH
    value = raw.strip()
    if not value:
        return CONTAINER_CREDENTIALS_PATH
    return Path(os.path.expanduser(os.path.expandvars(value)))


def _directory_writable(directory: Path) -> bool:
    """Whether the nearest existing directory can accept the credential file."""
    probe = directory
    while True:
        if probe.exists():
            return probe.is_dir() and os.access(probe, os.W_OK)
        parent = probe.parent
        if parent == probe:
            return False
        probe = parent


def _read_file_store() -> dict[str, str]:
    """The stored purposes, or an empty mapping when nothing is stored yet.

    An unreadable, oversized or malformed file is a *failure*, never an empty
    store: silently treating it as empty would let the next save overwrite
    content this process did not understand.
    """
    path = credentials_file_path()
    try:
        if not path.exists():
            return {}
        if not path.is_file():
            raise CredentialError(_FILE_UNREADABLE)
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise CredentialError(_FILE_TOO_LARGE)
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CredentialError(_FILE_UNREADABLE) from exc

    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CredentialError(_FILE_UNREADABLE) from exc
    if not isinstance(data, dict):
        raise CredentialError(_FILE_UNREADABLE)

    stored: dict[str, str] = {}
    for purpose in _PURPOSES:
        if purpose not in data:
            continue
        value = data[purpose]
        if not _is_usable(value) or not _is_valid_length(value):
            raise CredentialError(_FILE_UNREADABLE)
        stored[purpose] = value
    if set(data) - set(_PURPOSES):
        # An unknown key means this is not the file this module wrote.
        raise CredentialError(_FILE_UNREADABLE)
    return stored


def _restrict_file_permissions(path: Path) -> None:
    """Best effort ``0600``: a mount that refuses chmod must not lose the key."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _write_file_store(values: dict[str, str]) -> None:
    """Replace the credential file atomically: same directory, then replace."""
    path = credentials_file_path()
    directory = path.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CredentialError(_DIRECTORY_UNWRITABLE) from exc
    if not directory.is_dir():
        raise CredentialError(_DIRECTORY_UNWRITABLE)

    staging: Path | None = None
    try:
        handle_fd, staging_name = tempfile.mkstemp(
            prefix=f"{path.name}.", suffix=".tmp", dir=directory
        )
        staging = Path(staging_name)
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(values, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        _restrict_file_permissions(staging)
        os.replace(staging, path)
        staging = None
    except OSError as exc:
        raise CredentialError(_WRITE_FAILED) from exc
    finally:
        if staging is not None:
            try:
                staging.unlink()
            except OSError:
                pass


def _remove_file_store() -> None:
    try:
        credentials_file_path().unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CredentialError(_WRITE_FAILED) from exc


# ---------------------------------------------------------------------------
# Windows Credential Manager backend (ctypes, no third-party dependency).
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    # CRED_TYPE_GENERIC; CRED_PERSIST_LOCAL_MACHINE keeps the credential across
    # reboots without enterprise roaming (spec forbids cross-user sharing).
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    # ERROR_NOT_FOUND returned by CredReadW/CredDeleteW when no entry exists.
    _ERROR_NOT_FOUND = 1168

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.wintypes.DWORD),
            ("dwHighDateTime", ctypes.wintypes.DWORD),
        ]

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.wintypes.DWORD),
            ("Type", ctypes.wintypes.DWORD),
            ("TargetName", ctypes.wintypes.LPWSTR),
            ("Comment", ctypes.wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", ctypes.wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", ctypes.wintypes.DWORD),
            ("AttributeCount", ctypes.wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", ctypes.wintypes.LPWSTR),
            ("UserName", ctypes.wintypes.LPWSTR),
        ]

    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    _CredReadW = _advapi32.CredReadW
    _CredReadW.argtypes = [
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
    ]
    _CredReadW.restype = ctypes.wintypes.BOOL

    _CredWriteW = _advapi32.CredWriteW
    _CredWriteW.argtypes = [
        ctypes.POINTER(_CREDENTIALW),
        ctypes.wintypes.DWORD,
    ]
    _CredWriteW.restype = ctypes.wintypes.BOOL

    _CredDeleteW = _advapi32.CredDeleteW
    _CredDeleteW.argtypes = [
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.DWORD,
    ]
    _CredDeleteW.restype = ctypes.wintypes.BOOL

    _CredFree = _advapi32.CredFree
    _CredFree.argtypes = [ctypes.c_void_p]
    _CredFree.restype = None


def _read_windows_credential(target: str) -> str | None:
    if not _is_windows():
        return None
    pcred = ctypes.POINTER(_CREDENTIALW)()
    ok = _CredReadW(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pcred))
    if not ok:
        error = ctypes.get_last_error()
        if error == _ERROR_NOT_FOUND:
            return None
        raise CredentialError(f"windows credential read failed (error code {error})")
    try:
        size = pcred.contents.CredentialBlobSize
        raw = ctypes.string_at(pcred.contents.CredentialBlob, size)
        return raw.decode("utf-16-le", errors="replace")
    finally:
        _CredFree(pcred)


def _write_windows_credential(target: str, value: str) -> None:
    if not _is_windows():
        raise UnsupportedPlatformError(
            "current platform does not support a native credential store"
        )
    blob = value.encode("utf-16-le")
    blob_buffer = ctypes.create_string_buffer(blob)
    cred = _CREDENTIALW()
    cred.Type = _CRED_TYPE_GENERIC
    cred.TargetName = target
    cred.CredentialBlobSize = len(blob)
    cred.CredentialBlob = ctypes.cast(
        blob_buffer, ctypes.POINTER(ctypes.c_ubyte)
    )
    cred.Persist = _CRED_PERSIST_LOCAL_MACHINE
    if not _CredWriteW(ctypes.byref(cred), 0):
        error = ctypes.get_last_error()
        raise CredentialError(f"windows credential write failed (error code {error})")


def _delete_windows_credential(target: str) -> None:
    if not _is_windows():
        raise UnsupportedPlatformError(
            "current platform does not support a native credential store"
        )
    if not _CredDeleteW(target, _CRED_TYPE_GENERIC, 0):
        error = ctypes.get_last_error()
        # Deleting a non-existent target is idempotent success.
        if error != _ERROR_NOT_FOUND:
            raise CredentialError(f"windows credential delete failed (error code {error})")


# ---------------------------------------------------------------------------
# Public contract.
# ---------------------------------------------------------------------------


def supports_os_credential_store() -> bool:
    return _is_windows()


def supports_file_credential_store() -> bool:
    """Linux and every other non-Windows platform persist to the credential file."""
    return not _is_windows()


_INVALIDATE_ALL = object()


def invalidate_credential_cache(purpose: object = _INVALIDATE_ALL) -> None:
    if purpose is _INVALIDATE_ALL:
        _cache.clear()
        return
    _validate_purpose(purpose)
    _cache.pop(purpose, None)


def resolve_credential(purpose: CredentialPurpose) -> str | None:
    _validate_purpose(purpose)
    env_var, target = _PURPOSES[purpose]
    # The environment is the highest-priority source and is re-read on every
    # resolution, so adding/changing/removing it takes effect immediately and
    # never reuses a cache populated from another source.
    env_value = os.environ.get(env_var)
    if _is_usable(env_value):
        return env_value

    now = time.monotonic()
    cached = _cache.get(purpose)
    if cached is not None and cached[1] > now:
        return cached[0]

    if _is_windows():
        value = _read_windows_credential(target)
    else:
        try:
            value = _read_file_store().get(purpose)
        except CredentialError:
            # A broken credential file must never stop the Studio: it resolves
            # as "no credential", and the settings page reports why.
            _cache.pop(purpose, None)
            return None

    if _is_usable(value):
        _cache[purpose] = (value, now + _CACHE_TTL_SECONDS)
        return value
    _cache.pop(purpose, None)
    return None


def store_credential(purpose: CredentialPurpose, value: str) -> None:
    _validate_purpose(purpose)
    env_var, target = _PURPOSES[purpose]
    if _is_usable(os.environ.get(env_var)):
        raise CredentialError(
            "credential is managed by the environment and cannot be modified"
        )
    if not _is_usable(value) or not _is_valid_length(value):
        raise CredentialError("credential value is empty, a placeholder, or too long")

    if _is_windows():
        _write_windows_credential(target, value)
    else:
        if not _directory_writable(credentials_file_path().parent):
            raise CredentialError(_DIRECTORY_UNWRITABLE)
        with _store_lock:
            # Read-modify-write as one critical section: the two purposes share
            # the file, so an unlocked write could drop the other one.
            current = _read_file_store()
            current[purpose] = value
            _write_file_store(current)

    _cache.pop(purpose, None)
    _last_test_result.pop(purpose, None)
    _last_tested_at.pop(purpose, None)


def delete_credential(purpose: CredentialPurpose) -> None:
    _validate_purpose(purpose)
    env_var, target = _PURPOSES[purpose]
    if _is_usable(os.environ.get(env_var)):
        raise CredentialError(
            "credential is managed by the environment and cannot be deleted"
        )

    if _is_windows():
        _delete_windows_credential(target)
    else:
        with _store_lock:
            current = _read_file_store()
            if purpose in current:
                del current[purpose]
                if current:
                    if not _directory_writable(credentials_file_path().parent):
                        raise CredentialError(_DIRECTORY_UNWRITABLE)
                    _write_file_store(current)
                else:
                    # One rule for the last purpose: nothing stored, no file left.
                    _remove_file_store()

    # Even a delete of something absent must drop any cached value.
    _cache.pop(purpose, None)
    _last_test_result.pop(purpose, None)
    _last_tested_at.pop(purpose, None)


def get_credential_status(purpose: CredentialPurpose) -> CredentialStatus:
    _validate_purpose(purpose)
    env_var, target = _PURPOSES[purpose]
    if _is_usable(os.environ.get(env_var)):
        source: CredentialSource = "environment"
        configured = True
        writable = False
    elif _is_windows():
        try:
            value = _read_windows_credential(target)
        except CredentialError:
            source = "missing"
            configured = False
            writable = True
        else:
            if _is_usable(value):
                source = "windows_credential_manager"
                configured = True
                writable = True
            else:
                source = "missing"
                configured = False
                writable = True
    else:
        directory_ok = _directory_writable(credentials_file_path().parent)
        try:
            stored = _read_file_store()
        except CredentialError:
            # Present but unusable: the page must not offer a save that would
            # overwrite a file this process does not understand.
            source = "missing"
            configured = False
            writable = False
        else:
            if _is_usable(stored.get(purpose)):
                source = "file_credential_store"
                configured = True
                writable = True
            else:
                source = "missing"
                configured = False
                writable = directory_ok

    return CredentialStatus(
        purpose=purpose,
        configured=configured,
        source=source,
        writable=writable,
        last_tested_at=_last_tested_at.get(purpose),
        last_test_result=_last_test_result.get(purpose),
    )


def set_last_test_result(purpose: CredentialPurpose, result: str) -> None:
    _validate_purpose(purpose)
    _last_test_result[purpose] = result
    _last_tested_at[purpose] = datetime.now(timezone.utc).isoformat()


def clear_last_test_result(purpose: CredentialPurpose) -> None:
    _validate_purpose(purpose)
    _last_test_result.pop(purpose, None)
    _last_tested_at.pop(purpose, None)


def known_credentials() -> tuple[str, ...]:
    """Currently-known resolved values (environment, cache, persisted store).

    The file store is read here as well: a redaction pass must cover a key that
    has not been resolved yet in this process.
    """
    secrets: set[str] = set()
    for purpose, (env_var, _target) in _PURPOSES.items():
        env_value = os.environ.get(env_var)
        if _is_usable(env_value):
            secrets.add(env_value)
        cached = _cache.get(purpose)
        if cached is not None:
            secrets.add(cached[0])
    if not _is_windows():
        try:
            secrets.update(_read_file_store().values())
        except CredentialError:
            pass
    return tuple(secrets)
