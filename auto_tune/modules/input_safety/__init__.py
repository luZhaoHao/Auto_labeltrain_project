"""Directory-input safety (Studio S1.4): policy parsing, path validation,
bounded scanning, and safe subdirectory enumeration.

Only the public interface is exported; the web layer must not copy the safety
rules.
"""

from .models import (
    DirectoryScanResult,
    InputChangedDuringScanError,
    InputLinkNotAllowedError,
    InputMemberLimitExceededError,
    InputPathNotAllowedError,
    InputPermissionDeniedError,
    InputPolicyInvalidError,
    InputSafetyError,
    InputSafetyPolicy,
    InputSizeLimitExceededError,
)
from .service import (
    list_safe_subdirectories,
    load_input_safety_policy,
    scan_directory_bounded,
    validate_directory_path,
)

__all__ = [
    "DirectoryScanResult",
    "InputChangedDuringScanError",
    "InputLinkNotAllowedError",
    "InputMemberLimitExceededError",
    "InputPathNotAllowedError",
    "InputPermissionDeniedError",
    "InputPolicyInvalidError",
    "InputSafetyError",
    "InputSafetyPolicy",
    "InputSizeLimitExceededError",
    "list_safe_subdirectories",
    "load_input_safety_policy",
    "scan_directory_bounded",
    "validate_directory_path",
]
