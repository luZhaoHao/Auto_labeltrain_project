"""Credential store and resolver for Studio API keys.

Priority per purpose: environment variable, then the current Windows user's
Credential Manager, then missing.  Resolved values are cached in-process for at
most five minutes and never enter config, YAML, URL, response, history, audit
or logs.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

CredentialPurpose = Literal["text", "vision"]
CredentialSource = Literal["environment", "windows_credential_manager", "missing"]

_PURPOSES: dict[str, tuple[str, str]] = {
    "text": ("AUTO_TUNE_TEXT_API_KEY", "AutoTuneStudio/text/deepseek"),
    "vision": ("AUTO_TUNE_VISION_API_KEY", "AutoTuneStudio/vision/qwen"),
}

_CACHE_TTL_SECONDS = 300.0

# Project-defined placeholders must never be treated as real credentials.
_PLACEHOLDER_VALUES = {"YOUR_DEEPSEEK_API_KEY", "YOUR_QWEN_API_KEY"}

# Function (not bool) so tests can monkeypatch platform detection.
def _is_windows() -> bool:
    return sys.platform == "win32"

# purpose -> (resolved_value, monotonic_expiry)
_cache: dict[str, tuple[str, float]] = {}
_last_tested_at: dict[str, str] = {}
_last_test_result: dict[str, str] = {}


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
        and value != ""
        and value not in _PLACEHOLDER_VALUES
    )


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

    if not _is_windows():
        _cache.pop(purpose, None)
        return None

    value = _read_windows_credential(target)
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
    if not _is_usable(value):
        raise CredentialError("credential value is empty or a placeholder")
    _write_windows_credential(target, value)
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
    _delete_windows_credential(target)
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
        source = "missing"
        configured = False
        writable = False

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
    """Currently-known resolved values (environment + in-process cache)."""
    secrets: set[str] = set()
    for purpose, (env_var, _target) in _PURPOSES.items():
        env_value = os.environ.get(env_var)
        if _is_usable(env_value):
            secrets.add(env_value)
        cached = _cache.get(purpose)
        if cached is not None:
            secrets.add(cached[0])
    return tuple(secrets)
