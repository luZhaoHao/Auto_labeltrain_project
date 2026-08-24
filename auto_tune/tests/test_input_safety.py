"""Tests for the unified directory-input safety module (Studio S1.4)."""

import os
import stat
from pathlib import Path

import pytest

from auto_tune.modules.input_safety import (
    InputChangedDuringScanError,
    InputLinkNotAllowedError,
    InputMemberLimitExceededError,
    InputPathNotAllowedError,
    InputPermissionDeniedError,
    InputPolicyInvalidError,
    InputSafetyError,
    InputSizeLimitExceededError,
    InputSafetyPolicy,
    load_input_safety_policy,
    list_safe_subdirectories,
    scan_directory_bounded,
    validate_directory_path,
)

DEFAULT_MEMBERS = 200000
DEFAULT_BYTES = 536870912000
MEMBER_LIMIT_MAX = 1000000
BYTE_LIMIT_MAX = 10995116277760


def _policy(**overrides):
    base = {
        "max_directory_members": DEFAULT_MEMBERS,
        "max_directory_bytes": DEFAULT_BYTES,
        "allowed_roots": (),
        "allow_unc_paths": False,
    }
    base.update(overrides)
    return InputSafetyPolicy(**base)


# ── Task 1: strict configuration parsing ──


def test_load_policy_defaults_when_section_missing():
    policy = load_input_safety_policy({})
    assert policy.max_directory_members == DEFAULT_MEMBERS
    assert policy.max_directory_bytes == DEFAULT_BYTES
    assert policy.allowed_roots == ()
    assert policy.allow_unc_paths is False


def test_load_policy_accepts_valid_fields():
    policy = load_input_safety_policy({
        "input_safety": {
            "max_directory_members": 100,
            "max_directory_bytes": 1024,
            "allowed_roots": ["C:/data", "D:/datasets"],
            "allow_unc_paths": True,
        }
    })
    assert policy.max_directory_members == 100
    assert policy.max_directory_bytes == 1024
    assert policy.allow_unc_paths is True
    assert len(policy.allowed_roots) == 2
    assert all(p.is_absolute() for p in policy.allowed_roots)


def test_load_policy_rejects_bool_as_integer():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_members": True}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_bytes": False}})


def test_load_policy_rejects_zero_and_negative():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_members": 0}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_bytes": 0}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_members": -5}})


def test_load_policy_rejects_above_upper_bound():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_members": MEMBER_LIMIT_MAX + 1}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_bytes": BYTE_LIMIT_MAX + 1}})


def test_load_policy_rejects_relative_allowed_root():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"allowed_roots": ["relative/dir"]}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"allowed_roots": ["", "C:/data"]}})


def test_load_policy_rejects_non_bool_allow_unc():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"allow_unc_paths": "yes"}})


def test_load_policy_rejects_unknown_fields():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"max_directory_member": 5}})
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": {"allow_unc": True}})


def test_load_policy_rejects_non_mapping_section():
    with pytest.raises(InputPolicyInvalidError):
        load_input_safety_policy({"input_safety": [1, 2, 3]})


def test_error_codes_match_spec_contract():
    cases = [
        (InputPolicyInvalidError("x"), "INPUT_POLICY_INVALID", 500),
        (InputPathNotAllowedError("x"), "INPUT_PATH_NOT_ALLOWED", 403),
        (InputLinkNotAllowedError("x"), "INPUT_LINK_NOT_ALLOWED", 400),
        (InputPermissionDeniedError("x"), "INPUT_PERMISSION_DENIED", 403),
        (InputMemberLimitExceededError("x"), "INPUT_MEMBER_LIMIT_EXCEEDED", 413),
        (InputSizeLimitExceededError("x"), "INPUT_SIZE_LIMIT_EXCEEDED", 413),
        (InputChangedDuringScanError("x"), "INPUT_CHANGED_DURING_SCAN", 409),
    ]
    for error, code, status in cases:
        assert error.error_code == code
        assert error.status_code == status
        assert isinstance(error, InputSafetyError)


# ── Task 2: path validation & allowed roots ──


def _try_symlink_dir(target, link):
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")


def test_validate_rejects_empty_path():
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("", _policy())
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("   ", _policy())


def test_validate_rejects_relative_path():
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("relative/dir", _policy())


def test_validate_rejects_nul_byte():
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("C:/data\x00evil", _policy())


def test_validate_rejects_device_namespace():
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("\\\\?\\C:\\data", _policy())


def test_validate_rejects_unc_by_default():
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path("\\\\server\\share\\dir", _policy())


def test_validate_rejects_missing_path(tmp_path):
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path(str(tmp_path / "does-not-exist"), _policy())


def test_validate_rejects_regular_file(tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path(str(f), _policy())


def test_validate_accepts_existing_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    result = validate_directory_path(str(sub), _policy())
    assert result == sub.resolve()
    assert result.is_dir()


def test_validate_enforces_allowed_roots(tmp_path):
    root = tmp_path / "root"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    policy = _policy(allowed_roots=(root.resolve(),))
    assert validate_directory_path(str(nested), policy) == nested.resolve()
    with pytest.raises(InputPathNotAllowedError):
        validate_directory_path(str(outside), policy)


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive semantics are Windows-specific")
def test_validate_allowed_root_case_insensitive(tmp_path):
    root = tmp_path / "RootDir"
    sub = root / "Sub"
    sub.mkdir(parents=True)
    policy = _policy(allowed_roots=(root.resolve(),))
    mixed = str(sub).replace("RootDir", "rootdir").replace("Sub", "sub")
    result = validate_directory_path(mixed, policy)
    assert result.is_dir()
    assert result == sub.resolve()


def test_validate_rejects_symlink_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _try_symlink_dir(str(real), str(link))
    with pytest.raises(InputLinkNotAllowedError):
        validate_directory_path(str(link), _policy())


@pytest.mark.skipif(os.name != "nt", reason="junction is Windows-specific")
def test_validate_rejects_junction(tmp_path):
    import subprocess

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "junction"
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(real)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip("junction creation not permitted in this environment")
    with pytest.raises(InputLinkNotAllowedError):
        validate_directory_path(str(link), _policy())


def test_validate_reports_permission_denied(tmp_path, monkeypatch):
    import auto_tune.modules.input_safety.service as svc

    secret = tmp_path / "secret"
    secret.mkdir()
    real_stat = os.stat

    def deny_stat(p, *args, **kwargs):
        if os.path.normcase(os.fspath(p)) == os.path.normcase(str(secret.resolve())):
            raise PermissionError(5, "Access is denied")
        return real_stat(p, *args, **kwargs)

    monkeypatch.setattr(svc.os, "stat", deny_stat)
    with pytest.raises(InputPermissionDeniedError):
        validate_directory_path(str(secret), _policy())


# ── Task 3: bounded scan & safe subdirectory enumeration ──


def _make_tree(root, files):
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_scan_counts_members_and_bytes(tmp_path):
    d = tmp_path / "data"
    _make_tree(d, ["a.jpg", "sub/b.jpg", "sub/c.png"])
    result = scan_directory_bounded(d, _policy())
    assert result.root == d.resolve()
    assert result.member_count == 4  # a.jpg + sub dir + b.jpg + c.png
    assert result.total_bytes == 3


def test_scan_exact_member_limit_passes(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(3):
        (d / f"f{i}.txt").write_bytes(b"x")
    policy = _policy(max_directory_members=3)
    result = scan_directory_bounded(d, policy)
    assert result.member_count == 3
    assert result.total_bytes == 3


def test_scan_exceeds_member_limit_by_one(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(4):
        (d / f"f{i}.txt").write_bytes(b"x")
    policy = _policy(max_directory_members=3)
    with pytest.raises(InputMemberLimitExceededError):
        scan_directory_bounded(d, policy)


def test_scan_exact_byte_limit_passes(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_bytes(b"xx")
    (d / "b.txt").write_bytes(b"xx")
    policy = _policy(max_directory_bytes=4)
    result = scan_directory_bounded(d, policy)
    assert result.total_bytes == 4


def test_scan_exceeds_byte_limit(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_bytes(b"xxx")
    (d / "b.txt").write_bytes(b"xxx")
    policy = _policy(max_directory_bytes=5)
    with pytest.raises(InputSizeLimitExceededError):
        scan_directory_bounded(d, policy)


def test_scan_nested_bytes_accumulate(tmp_path):
    d = tmp_path / "data"
    _make_tree(d, ["top/a.txt", "mid/deep/b.txt", "top/c.txt"])
    policy = _policy(max_directory_bytes=3)
    result = scan_directory_bounded(d, policy)
    assert result.total_bytes == 3
    assert result.member_count == 6


def test_scan_rejects_symlink_inside(tmp_path):
    d = tmp_path / "data"
    real = tmp_path / "real"
    real.mkdir()
    d.mkdir()
    (d / "ok.txt").write_bytes(b"x")
    link = d / "evil_link"
    _try_symlink_dir(str(real), str(link))
    with pytest.raises(InputLinkNotAllowedError):
        scan_directory_bounded(d, _policy())


def test_scan_reports_changed_when_entry_stat_fails(tmp_path, monkeypatch):
    import auto_tune.modules.input_safety.service as svc

    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_bytes(b"x")

    def fail_stat(_entry):
        raise InputChangedDuringScanError("member vanished during scan")

    monkeypatch.setattr(svc, "_entry_stat", fail_stat)
    with pytest.raises(InputChangedDuringScanError):
        scan_directory_bounded(d, _policy())


def test_scan_reports_permission_on_nested_dir(tmp_path, monkeypatch):
    import auto_tune.modules.input_safety.service as svc

    d = tmp_path / "data"
    (d / "sub").mkdir(parents=True)
    real_scandir = os.scandir

    def denied_scandir(path):
        if os.path.basename(os.fspath(path)) == "sub":
            raise PermissionError(5, "Access is denied")
        return real_scandir(path)

    monkeypatch.setattr(svc.os, "scandir", denied_scandir)
    with pytest.raises(InputPermissionDeniedError):
        scan_directory_bounded(d, _policy())


def test_scan_member_limit_does_not_enter_trap_dir(tmp_path, monkeypatch):
    import auto_tune.modules.input_safety.service as svc

    d = tmp_path / "data"
    d.mkdir()
    for i in range(2):
        (d / f"f{i}.txt").write_bytes(b"x")
    (d / "trap").mkdir()
    policy = _policy(max_directory_members=2)
    real_scandir = os.scandir
    visited = []

    def spy_scandir(path):
        visited.append(Path(os.fspath(path)))
        return real_scandir(path)

    monkeypatch.setattr(svc.os, "scandir", spy_scandir)
    with pytest.raises(InputMemberLimitExceededError):
        scan_directory_bounded(d, policy)
    assert not any(p.name == "trap" for p in visited)


def test_list_safe_subdirectories_returns_directories_only(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "sub1").mkdir()
    (d / "sub2").mkdir()
    (d / "file.txt").write_bytes(b"x")
    result = list_safe_subdirectories(d, _policy())
    assert len(result) == 2
    assert {p.name for p in result} == {"sub1", "sub2"}
    assert all(p.is_dir() for p in result)


def test_list_safe_subdirectories_rejects_permission(tmp_path, monkeypatch):
    import auto_tune.modules.input_safety.service as svc

    d = tmp_path / "data"
    d.mkdir()

    def denied(_):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(svc.os, "scandir", denied)
    with pytest.raises(InputPermissionDeniedError):
        list_safe_subdirectories(d, _policy())


def test_list_safe_subdirectories_rejects_link(tmp_path):
    d = tmp_path / "data"
    real = tmp_path / "real"
    real.mkdir()
    d.mkdir()
    link = d / "lnk"
    _try_symlink_dir(str(real), str(link))
    with pytest.raises(InputLinkNotAllowedError):
        list_safe_subdirectories(d, _policy())
