"""Domain models for the directory-input safety module (Studio S1.4).

The error hierarchy maps one-to-one onto the S1.4 error contract; the web
layer only serializes ``error_code`` / ``status_code`` and never re-implements
the safety rules.
"""

from dataclasses import dataclass
from pathlib import Path


class InputSafetyError(Exception):
    """Base class for all directory-input safety errors."""

    error_code = "INPUT_ERROR"
    status_code = 400


class InputPolicyInvalidError(InputSafetyError):
    error_code = "INPUT_POLICY_INVALID"
    status_code = 500


class InputPathNotAllowedError(InputSafetyError):
    error_code = "INPUT_PATH_NOT_ALLOWED"
    status_code = 403


class InputLinkNotAllowedError(InputSafetyError):
    error_code = "INPUT_LINK_NOT_ALLOWED"
    status_code = 400


class InputPermissionDeniedError(InputSafetyError):
    error_code = "INPUT_PERMISSION_DENIED"
    status_code = 403


class InputMemberLimitExceededError(InputSafetyError):
    error_code = "INPUT_MEMBER_LIMIT_EXCEEDED"
    status_code = 413


class InputSizeLimitExceededError(InputSafetyError):
    error_code = "INPUT_SIZE_LIMIT_EXCEEDED"
    status_code = 413


class InputChangedDuringScanError(InputSafetyError):
    error_code = "INPUT_CHANGED_DURING_SCAN"
    status_code = 409


@dataclass(frozen=True)
class InputSafetyPolicy:
    """Frozen, validated policy that governs every directory input."""

    max_directory_members: int = 200000
    max_directory_bytes: int = 536870912000
    allowed_roots: tuple[Path, ...] = ()
    allow_unc_paths: bool = False


@dataclass(frozen=True)
class DirectoryScanResult:
    """Bounded scan facts. Only the root path, member count, and byte total are kept."""

    root: Path
    member_count: int
    total_bytes: int
