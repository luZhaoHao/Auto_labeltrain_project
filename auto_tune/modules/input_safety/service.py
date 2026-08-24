"""Policy parsing, path validation, and bounded scanning (Studio S1.4).

Task 1 milestone: strict ``input_safety`` configuration parsing. The three
public operations are stubbed and filled in by Tasks 2/3.
"""

import os
import stat
from pathlib import Path
from typing import Any, Mapping

from .models import (
    DirectoryScanResult,
    InputChangedDuringScanError,
    InputLinkNotAllowedError,
    InputMemberLimitExceededError,
    InputPathNotAllowedError,
    InputPermissionDeniedError,
    InputPolicyInvalidError,
    InputSafetyPolicy,
    InputSizeLimitExceededError,
)

DEFAULT_MAX_DIRECTORY_MEMBERS = 200000
DEFAULT_MAX_DIRECTORY_BYTES = 536870912000
MEMBER_LIMIT_MAX = 1000000
BYTE_LIMIT_MAX = 10995116277760

_ALLOWED_POLICY_FIELDS = frozenset({
    "max_directory_members",
    "max_directory_bytes",
    "allowed_roots",
    "allow_unc_paths",
})


def _require_integer(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise InputPolicyInvalidError(f"{field} 必须是整数")
    if value < low or value > high:
        raise InputPolicyInvalidError(f"{field} 必须在 {low}..{high} 范围内")
    return value


def _parse_allowed_roots(raw: Any) -> tuple[Path, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InputPolicyInvalidError("allowed_roots 必须是绝对目录列表")
    roots: list[Path] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise InputPolicyInvalidError("allowed_roots 必须是绝对目录字符串")
        path = Path(item)
        if not path.is_absolute():
            raise InputPolicyInvalidError(f"allowed_roots 必须使用绝对路径: {item}")
        roots.append(path)
    return tuple(roots)


def load_input_safety_policy(config: Mapping[str, Any]) -> InputSafetyPolicy:
    """Parse the ``input_safety`` section with strict typing and bounds.

    A missing section yields the spec defaults. A present but invalid value is
    a configuration error; it is never silently coerced to an unbounded policy.
    """
    section = (config or {}).get("input_safety")
    if section is None:
        return InputSafetyPolicy()
    if not isinstance(section, dict):
        raise InputPolicyInvalidError("input_safety 配置必须是映射")
    unknown = set(section) - _ALLOWED_POLICY_FIELDS
    if unknown:
        raise InputPolicyInvalidError(
            f"未知的 input_safety 字段: {', '.join(sorted(unknown))}"
        )
    members = _require_integer(
        section.get("max_directory_members", DEFAULT_MAX_DIRECTORY_MEMBERS),
        "max_directory_members",
        1,
        MEMBER_LIMIT_MAX,
    )
    byte_limit = _require_integer(
        section.get("max_directory_bytes", DEFAULT_MAX_DIRECTORY_BYTES),
        "max_directory_bytes",
        1,
        BYTE_LIMIT_MAX,
    )
    allowed_roots = _parse_allowed_roots(section.get("allowed_roots"))
    allow_unc = section.get("allow_unc_paths", False)
    if not isinstance(allow_unc, bool):
        raise InputPolicyInvalidError("allow_unc_paths 必须是布尔值")
    return InputSafetyPolicy(
        max_directory_members=members,
        max_directory_bytes=byte_limit,
        allowed_roots=allowed_roots,
        allow_unc_paths=allow_unc,
    )


def _is_reparse_point(path: Path) -> bool:
    """Return True when ``path`` is a symlink, junction, or other reparse point."""
    try:
        st = path.lstat()
    except OSError:
        return False
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def _reject_reparse_components(path: Path) -> None:
    """Reject when any path component from the drive root down to ``path`` is a link.

    ``path`` must already be normalized without following links so a symlinked
    leaf is rejected instead of being resolved to its target.
    """
    parts: list[Path] = []
    current = path
    while True:
        parts.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for component in reversed(parts):
        if _is_reparse_point(component):
            raise InputLinkNotAllowedError(f"符号链接或 junction 不允许: {component}")


def _norm_parts(path: Path) -> tuple[str, ...]:
    """Case-normalized path components; never a raw string prefix."""
    return tuple(os.path.normcase(part) for part in Path(path).parts)


def _within_root(target: Path, root: Path) -> bool:
    """True when ``target`` is ``root`` itself or lies inside it."""
    target_parts = _norm_parts(target)
    root_parts = _norm_parts(root)
    return len(target_parts) >= len(root_parts) and target_parts[: len(root_parts)] == root_parts


def validate_directory_path(path: str | Path, policy: InputSafetyPolicy) -> Path:
    """Validate and normalize a directory input path.

    Rejects relative paths, NUL bytes, device namespaces, UNC (unless enabled),
    non-existent paths, regular files, symlinks/reparse points, and paths outside
    ``policy.allowed_roots``. Returns the resolved absolute directory path.
    """
    raw = os.fspath(path) if isinstance(path, Path) else str(path)
    if not raw or not raw.strip():
        raise InputPathNotAllowedError("路径不能为空")
    if "\x00" in raw:
        raise InputPathNotAllowedError("路径包含 NUL 字符")
    if raw.startswith("\\\\?\\") or raw.startswith("\\\\.\\"):
        raise InputPathNotAllowedError("设备命名空间路径不允许")
    if not os.path.isabs(raw):
        raise InputPathNotAllowedError("路径必须是绝对路径")
    if raw.startswith("\\\\") and not policy.allow_unc_paths:
        raise InputPathNotAllowedError("UNC 路径不允许")

    normalized = Path(os.path.normpath(raw))
    _reject_reparse_components(normalized)
    resolved = normalized.resolve(strict=False)

    if policy.allowed_roots:
        if not any(
            _within_root(resolved, Path(root).resolve(strict=False))
            for root in policy.allowed_roots
        ):
            raise InputPathNotAllowedError("路径不在允许根目录内")

    try:
        st = os.stat(resolved)
    except PermissionError as exc:
        raise InputPermissionDeniedError("目录访问被拒绝") from exc
    except OSError as exc:
        raise InputPathNotAllowedError("目录不存在或无法访问") from exc
    if not stat.S_ISDIR(st.st_mode):
        raise InputPathNotAllowedError("路径不是目录")
    return resolved


def _entry_stat(entry):
    """stat an entry without following links; a vanished/mutated member conflicts."""
    try:
        return entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise InputChangedDuringScanError("成员在扫描期间消失或变化") from exc


def _stat_is_reparse(st) -> bool:
    if os.name == "nt":
        return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def _iter_entries_sorted(root: Path) -> list:
    """Return ``root``'s direct entries in a deterministic order.

    Permission errors surface as such; they are never disguised as an empty
    directory. Other OSErrors (directory vanished) are scan conflicts.
    """
    try:
        with os.scandir(root) as it:
            return sorted(it, key=lambda e: e.name)
    except PermissionError as exc:
        raise InputPermissionDeniedError(f"目录访问被拒绝: {root}") from exc
    except OSError as exc:
        raise InputChangedDuringScanError(f"目录在扫描期间变化: {root}") from exc


def scan_directory_bounded(path: str | Path, policy: InputSafetyPolicy) -> DirectoryScanResult:
    """Iteratively scan a directory with hard member/byte limits and early stop.

    Only the root path, member count, and byte total are retained. Every entry
    is counted immediately; ordinary files accumulate no-follow stat sizes; any
    limit breach raises before deeper directories are entered.
    """
    root = validate_directory_path(path, policy)
    member_count = 0
    total_bytes = 0
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        for entry in _iter_entries_sorted(current):
            member_count += 1
            if member_count > policy.max_directory_members:
                raise InputMemberLimitExceededError(
                    f"目录成员数量超过上限 {policy.max_directory_members}: {root}"
                )
            st = _entry_stat(entry)
            if _stat_is_reparse(st):
                raise InputLinkNotAllowedError(f"符号链接或 junction 不允许: {entry.path}")
            if stat.S_ISDIR(st.st_mode):
                stack.append(Path(entry.path))
            else:
                total_bytes += st.st_size
                if total_bytes > policy.max_directory_bytes:
                    raise InputSizeLimitExceededError(
                        f"目录总容量超过上限 {policy.max_directory_bytes}: {root}"
                    )
    return DirectoryScanResult(
        root=root,
        member_count=member_count,
        total_bytes=total_bytes,
    )


def list_safe_subdirectories(path: str | Path, policy: InputSafetyPolicy) -> tuple[Path, ...]:
    """Safely enumerate the direct subdirectories of ``path``.

    Permission and link errors are raised, never disguised as an empty listing.
    """
    root = validate_directory_path(path, policy)
    subs: list[Path] = []
    for entry in _iter_entries_sorted(root):
        st = _entry_stat(entry)
        if _stat_is_reparse(st):
            raise InputLinkNotAllowedError(f"符号链接或 junction 不允许: {entry.path}")
        if stat.S_ISDIR(st.st_mode):
            subs.append(Path(entry.path))
    return tuple(subs)
