"""F1.2-D 返修：the Windows delivery installs with no network at all.

The package carries a private Python (the pinned Miniconda), the CUDA PyTorch
wheels and every ordinary wheel inside ``offline\\``, described by a
deterministic ``offline-lock.json``. The installer verifies that bundle against
the pinned manifest and the lock before it touches a single file, installs the
runtime with ``pip --no-index --find-links``, and asks the private interpreter to
prove it is really the locked CUDA runtime.

The tests drive the shipping PowerShell module through
``ps/windows_delivery_harness.ps1``; the pip runner is the only boundary that is
replaced, and a global download trap records any attempt to reach the network.

These tests need ``powershell.exe``; the whole module is skipped otherwise.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent / "ps" / "windows_delivery_harness.ps1"
MODULE = REPO_ROOT / "windows" / "lib" / "AutoTuneDelivery.psm1"
WINDOWS = REPO_ROOT / "windows"

_POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(_POWERSHELL is None,
                                reason="Windows PowerShell is not available")

_CACHE: dict[str, dict] = {}


@pytest.fixture(scope="module")
def work_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("f12d-offline")


def _run(scenario: str, work_root: Path, **extra) -> dict:
    key = f"{scenario}:{sorted(map(str, extra.items()))}:{work_root}"
    if key in _CACHE:
        return _CACHE[key]
    command = [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(HARNESS), "-Scenario", scenario,
               "-ModulePath", str(MODULE), "-WorkRoot", str(work_root)]
    for name, value in extra.items():
        command += [f"-{name}", str(value)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    lines = [line for line in (result.stdout or "").splitlines()
             if line.startswith("##RESULT## ")]
    assert lines, f"the harness produced no result: {result.stdout!r} {result.stderr!r}"
    payload = json.loads(lines[-1][len("##RESULT## "):])
    if not payload.get("ok"):
        pytest.fail(f"{scenario} could not run: {payload.get('error_code')}: "
                    f"{payload.get('error_message')}")
    _CACHE[key] = payload["facts"]
    return payload["facts"]


# ── the offline bundle the installer verifies ───────────────────────────────


def test_the_offline_bundle_lives_in_the_promised_directories(work_root):
    facts = _run("offline-bundle", work_root, Variant="valid")

    assert facts["minicondaInside"] is True, "the Miniconda installer sits in offline\\miniconda"
    assert facts["wheelhouseInside"] is True
    assert facts["lockInside"] is True, "the lock is offline\\offline-lock.json"


def test_the_bundle_pins_exactly_three_core_files(work_root):
    facts = _run("offline-bundle", work_root, Variant="valid")

    assert len(facts["coreNames"]) == 3
    assert sorted(facts["corePurposes"]) == ["cuda-torch", "cuda-torchvision",
                                             "private-python-runtime"]
    for name in ("Miniconda3-test-Windows-x86_64.exe",
                 "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl",
                 "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl"):
        assert name in facts["coreNames"]
    assert all(size > 0 for size in facts["coreSizes"])
    for path in facts["corePaths"]:
        assert not re.match(r"^[A-Za-z]:", path), f"the expectation is not relative: {path}"
        assert path.startswith(("miniconda", "wheelhouse")), path
        assert ".." not in path, path


def test_a_complete_bundle_is_accepted(work_root):
    facts = _run("offline-bundle", work_root, Variant="valid")

    assert facts["testedOk"] is True, facts["message"]
    assert facts["accepted"] is True, facts["code"]
    assert facts["offenders"] == []


def test_the_lock_records_exactly_the_files_the_bundle_carries(work_root):
    """The positive control for the exact-file-set check: a complete bundle has
    nothing extra and nothing missing, so the two sides must agree exactly."""
    facts = _run("offline-bundle", work_root, Variant="valid")

    assert facts["lockedFiles"] == facts["onDiskFiles"], facts["offenders"]
    assert facts["onDiskFiles"], "the fixture must ship files"


def test_the_lock_records_no_absolute_path(work_root):
    facts = _run("offline-bundle", work_root, Variant="valid")

    assert facts["lockHasAbsolutePath"] is False, "the lock must travel between machines"
    lock = json.loads(facts["lockText"])
    assert lock["schema_version"] == "1.0"
    assert lock["requirements_sha256"], "the dependency lock identity is recorded"
    assert {entry["purpose"] for entry in lock["files"]} == {
        "private-python-runtime", "cuda-torch", "cuda-torchvision", "runtime-dependency"}
    for entry in lock["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["size"] > 0


@pytest.mark.parametrize("variant,name", [
    ("miniconda-missing", "Miniconda3-test-Windows-x86_64.exe"),
    ("torch-missing", "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"),
    ("torchvision-missing", "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl"),
    ("no-bundle", "offline"),
    ("no-lock", "offline-lock.json"),
])
def test_a_missing_bundle_file_is_reported_as_missing(work_root, variant, name):
    facts = _run("offline-bundle", work_root, Variant=variant)

    assert facts["accepted"] is False
    assert facts["code"] == "OFFLINE_BUNDLE_MISSING"
    assert name in facts["message"], "the operator must be told which file to re-fetch"


@pytest.mark.parametrize("variant", ["core-hash-mismatch", "torch-hash-mismatch",
                                     "requirements-changed"])
def test_a_tampered_core_file_is_reported_as_a_hash_mismatch(work_root, variant):
    facts = _run("offline-bundle", work_root, Variant=variant)

    assert facts["accepted"] is False
    assert facts["code"] == "OFFLINE_BUNDLE_HASH_MISMATCH"
    assert facts["message"]


@pytest.mark.parametrize("variant", ["wheel-missing", "wheel-hash-mismatch",
                                     "unlocked-wheel", "cpu-torch", "linux-wheel",
                                     "wrong-abi", "sdist"])
def test_an_incomplete_or_foreign_wheelhouse_is_refused(work_root, variant):
    facts = _run("offline-bundle", work_root, Variant=variant)

    assert facts["accepted"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE", (
        f"{variant} was not refused as an incomplete wheelhouse")
    assert facts["message"]


def test_a_cpu_torch_wheel_never_passes_verification(work_root):
    """The product is GPU-only: a CPU build must not slip in beside the CUDA one."""
    facts = _run("offline-bundle", work_root, Variant="cpu-torch")

    assert facts["accepted"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    assert any("torch-2.5.1-cp310-cp310-win_amd64.whl" in str(offender)
               for offender in facts["offenders"]), facts["offenders"]


@pytest.mark.parametrize("variant,needle", [
    ("miniconda-extra-file", "extra-cuda.dll"),
    ("miniconda-nested-extra-file", "extra-nested.dll"),
    ("wheelhouse-nested-extra-file", "extra-nested-1.0.0-py3-none-any.whl"),
    ("offline-root-extra-file", "build-notes.txt"),
])
def test_a_file_the_lock_does_not_name_is_refused_at_any_depth(work_root, variant, needle):
    """The exact file set is part of the offline promise: the whole ``offline\\``
    tree is enumerated, so an extra file beside Miniconda, nested inside it,
    nested inside the wheelhouse or sitting at the bundle root cannot travel
    unnoticed."""
    facts = _run("offline-bundle", work_root, Variant=variant)

    assert facts["accepted"] is False, f"{variant} was accepted"
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    assert any(needle in str(offender) for offender in facts["offenders"]), facts["offenders"]
    assert facts["offendersHaveAbsolutePath"] is False, "a local path must not be echoed"
    assert facts["messageHasAbsolutePath"] is False
    assert "不在离线清单内" in facts["message"]


@pytest.mark.parametrize("variant", ["lock-duplicate-path", "lock-traversal-path",
                                     "lock-absolute-path"])
def test_a_lock_path_that_escapes_or_repeats_is_a_broken_lock(work_root, variant):
    """A relative path that leaves the bundle — or that is recorded twice — is a
    malformed lock, not a missing wheel, and it is refused as such."""
    facts = _run("offline-bundle", work_root, Variant=variant)

    assert facts["accepted"] is False, f"{variant} was accepted"
    assert facts["code"] == "OFFLINE_BUNDLE_HASH_MISMATCH"
    assert facts["message"]
    assert facts["messageHasAbsolutePath"] is False, "a local path must not be echoed"
    assert facts["offendersHaveAbsolutePath"] is False


def test_a_recorded_size_that_does_not_match_is_refused(work_root):
    """The lock promises a size as well as a hash; both are checked."""
    facts = _run("offline-bundle", work_root, Variant="lock-size-mismatch")

    assert facts["accepted"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    assert any("大小与离线清单不一致" in str(offender) for offender in facts["offenders"]), \
        facts["offenders"]


def test_a_wheel_for_another_platform_or_abi_is_named(work_root):
    linux = _run("offline-bundle", work_root, Variant="linux-wheel")
    abi = _run("offline-bundle", work_root, Variant="wrong-abi")

    assert any("manylinux" in str(offender) for offender in linux["offenders"])
    assert any("cp39" in str(offender) for offender in abi["offenders"])


def test_an_abi3_wheel_does_not_make_a_complete_bundle_incomplete(work_root):
    """The build machine and the installer must agree on abi3.

    ``prepare_offline_bundle`` downloads and accepts a wheel whose ``abi3`` tag
    names an older CPython (``opencv``/``psutil`` ship ``cp37-abi3``), so the
    finished bundle — verified by the same rule — must not be reported as an
    incomplete wheelhouse. This is the whole ``Assert-OfflineBundle`` path that
    ``build_zip.ps1`` runs before it writes a single file."""
    facts = _run("offline-bundle", work_root, Variant="abi3-wheel")

    assert facts["testedOk"] is True, facts["offenders"]
    assert facts["accepted"] is True, f"{facts['code']}: {facts['message']}"
    assert facts["code"] is None
    assert facts["offenders"] == []


def _wheel_rules(work_root) -> dict[str, dict]:
    facts = _run("wheel-name-rules", work_root)
    return {row["name"]: row for row in facts["results"]}


@pytest.mark.parametrize("name", [
    "fastapi-0.139.2-py3-none-any.whl",          # pure Python
    "pytz-2025.2-py2.py3-none-any.whl",          # the dual Python 2/3 tag many pins use
    "humanfriendly-10.0-py2.py3-none-any.whl",
    "contourpy-1.3.2-cp310-cp310-win_amd64.whl",  # this interpreter's ABI
    "greenlet-3.5.5-cp310-cp310-win_amd64.whl",
    "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl",
    "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl",
])
def test_a_wheel_the_private_runtime_can_install_is_accepted(work_root, name):
    row = _wheel_rules(work_root)[name]

    assert row["ok"] is True, f"{name} was refused: {row['reason']}"


@pytest.mark.parametrize("name", [
    "opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl",  # really shipped as cp37-abi3
    "psutil-7.0.0-cp37-abi3-win_amd64.whl",             # really shipped as cp37-abi3
    "MarkupSafe-3.0.2-cp39-abi3-win_amd64.whl",
    "cryptography-44.0.1-cp310-abi3-win_amd64.whl",
])
def test_an_abi3_wheel_the_runtime_is_at_least_as_new_as_is_accepted(work_root, name):
    """An ``abi3`` tag names the *oldest* CPython that can load the wheel.

    ``opencv`` and ``psutil`` ship ``cp37-abi3`` Windows wheels: they install
    into the private CPython 3.10 just as well as a ``py3-none-any`` wheel does,
    so the build machine and the installer must both take them."""
    row = _wheel_rules(work_root)[name]

    assert row["ok"] is True, f"{name} was refused: {row['reason']}"


@pytest.mark.parametrize("name,needle", [
    ("torch-2.5.1-cp310-cp310-win_amd64.whl", "CUDA"),
    ("torchvision-0.20.1-cp310-cp310-win_amd64.whl", "CUDA"),
    ("fastapi-0.139.2-py3-none-manylinux1_x86_64.whl", "Windows"),
    ("fastapi-0.139.2-py3-none-macosx_11_0_arm64.whl", "Windows"),
    ("uvicorn-0.51.0-cp39-cp39-win_amd64.whl", "cp310"),
    ("uvicorn-0.51.0-cp311-cp311-win_amd64.whl", "cp310"),
    ("uvicorn-0.51.0-cp37-cp37-win_amd64.whl", "cp310"),
    ("uvicorn-0.51.0-cp311-abi3-win_amd64.whl", "3.10"),
    ("numpy-2.2.6-cp310-cp310-win32.whl", "Windows"),
    ("uvicorn-0.51.0.tar.gz", "sdist"),
    ("fastapi-0.139.2.zip", "wheel"),
])
def test_a_wheel_the_private_runtime_cannot_install_is_refused(work_root, name, needle):
    row = _wheel_rules(work_root)[name]

    assert row["ok"] is False, f"{name} was accepted"
    assert needle in row["reason"], row["reason"]


def test_the_abi3_rule_does_not_loosen_the_other_abi_tags(work_root):
    """Only an ``abi3`` tag is forward compatible.

    ``cp37-cp37`` names one exact ABI and ``cp311-abi3`` demands a newer
    interpreter than the private one; neither may be admitted just because
    ``cp37-abi3`` now is."""
    rules = _wheel_rules(work_root)

    assert rules["uvicorn-0.51.0-cp37-cp37-win_amd64.whl"]["ok"] is False
    assert rules["uvicorn-0.51.0-cp311-abi3-win_amd64.whl"]["ok"] is False


# ── the offline installation ────────────────────────────────────────────────


def test_the_installation_needs_no_network(work_root):
    facts = _run("offline-install", work_root)

    assert facts["ok"] is True, f"{facts['errorCode']}: {facts['message']}"
    assert facts["status"] == "complete"
    assert facts["fetchCalls"] == 0, "nothing may go through the network"
    assert facts["networkish"] == [], "no command may reach an index or a URL"


def test_every_dependency_command_is_offline(work_root):
    facts = _run("offline-install", work_root)

    assert facts["pipCommands"], "the runtime dependencies are installed with pip"
    assert facts["badPip"] == [], "every pip command must use --no-index and --find-links"
    for command in facts["pipCommands"]:
        assert "http" not in command.lower()
        assert "download.pytorch.org" not in command


def test_the_cuda_wheels_are_installed_from_the_package(work_root):
    facts = _run("offline-install", work_root)

    assert facts["torchInstallArgs"], "torch and torchvision are installed explicitly"
    assert facts["torchOutsideWheelhouse"] == [], (
        "the CUDA wheels must come from the bundle's wheelhouse, never an index")


def test_the_private_python_is_the_packaged_miniconda(work_root):
    facts = _run("offline-install", work_root)

    assert facts["interpreterPrivate"] is True, facts["interpreter"]
    assert facts["interpreterInside"] is True
    assert facts["interpreterExists"] is True
    assert facts["installerInside"] is True, (
        "Miniconda is installed straight into the private runtime directory")
    arguments = " ".join(facts["installerArgs"])
    for flag in ("/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0", "/S"):
        assert flag in arguments, flag


def test_no_conda_environment_is_created(work_root):
    """The private interpreter is the Miniconda installation itself, not a second
    environment built over the network."""
    facts = _run("offline-install", work_root)

    assert facts["networkish"] == []
    for command in facts["pipCommands"]:
        assert "create" not in command.split()


def test_the_offline_precheck_proves_the_locked_cuda_runtime(work_root):
    facts = _run("offline-install", work_root)

    assert facts["offlineCheckArgs"], "the private interpreter is asked to prove itself"
    for command in facts["offlineCheckArgs"]:
        assert "--require-offline-runtime" in command
        assert "--expect-python" in command
        assert facts["interpreter"] in command


def test_the_runtime_is_stamped_once_it_verified(work_root):
    facts = _run("offline-install", work_root)

    assert facts["stampPresent"] is True
    assert facts["stampKey"], "the runtime identity is recorded"


def test_re_running_the_installer_keeps_the_runtime(work_root):
    facts = _run("offline-install", work_root)

    assert facts["secondOk"] is True
    assert facts["secondMode"] == "already-installed"
    assert facts["secondRuns"] == 0, "a complete installation is not rebuilt"
    assert facts["secondPip"] == []
    assert facts["secondFetch"] == 0
    assert facts["interpreterUnchanged"] is True


@pytest.mark.parametrize("variant,code", [
    ("miniconda-missing", "OFFLINE_BUNDLE_MISSING"),
    ("torch-hash-mismatch", "OFFLINE_BUNDLE_HASH_MISMATCH"),
    ("wheel-missing", "OFFLINE_WHEELHOUSE_INCOMPLETE"),
    ("cpu-torch", "OFFLINE_WHEELHOUSE_INCOMPLETE"),
    ("no-lock", "OFFLINE_BUNDLE_MISSING"),
])
def test_a_broken_bundle_stops_a_fresh_installation(work_root, variant, code):
    facts = _run("offline-install-broken", work_root, Variant=variant)

    assert facts["ok"] is False
    assert facts["errorCode"] == code
    assert facts["message"]
    assert facts["runCount"] == 0, "no runtime work may start"
    assert facts["pipCommands"] == []
    assert facts["fetchCalls"] == 0
    assert facts["networkish"] == []


def test_a_broken_bundle_never_produces_a_complete_installation(work_root):
    facts = _run("offline-install-broken", work_root, Variant="torch-hash-mismatch")

    assert facts["stateExists"] is False, "no state file is written at all"
    assert facts["appCopied"] is False
    assert facts["interpreterExists"] is False
    assert facts["stampPresent"] is False
    assert facts["leftovers"] == []


@pytest.mark.parametrize("variant", ["torch-hash-mismatch", "wheel-missing"])
def test_a_broken_bundle_leaves_a_complete_installation_untouched(work_root, variant):
    facts = _run("offline-install-broken-existing", work_root, Variant=variant)

    assert facts["ok"] is False
    assert facts["errorCode"] in ("OFFLINE_BUNDLE_HASH_MISMATCH",
                                  "OFFLINE_WHEELHOUSE_INCOMPLETE")
    assert facts["runCount"] == 0
    assert facts["fetchCalls"] == 0
    assert facts["leftovers"] == []


def test_the_installed_product_survives_a_refused_reinstall(work_root):
    facts = _run("offline-install-broken-existing", work_root, Variant="torch-hash-mismatch")

    assert facts["stateUnchanged"] is True
    assert facts["stateStatus"] == "complete"
    assert facts["stateVersion"] == "1.0.0"
    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["stampUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["weightsKept"] is True


def test_a_runtime_that_fails_the_offline_precheck_is_not_a_success(work_root):
    facts = _run("offline-install-runtime-mismatch", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "OFFLINE_RUNTIME_INSTALL_FAILED"
    assert facts["message"]
    assert facts["stateStatus"] != "complete"
    assert facts["stampPresent"] is False, "an unverified runtime is never stamped as ready"
    assert facts["offlineChecks"] == 1
    assert facts["fetchCalls"] == 0


# ── where the product may be installed ──────────────────────────────────────


def test_the_recommended_directory_is_not_on_the_system_drive(work_root):
    facts = _run("install-root-layout", work_root, Variant="recommended")

    assert facts["recommended"], "a destination must be proposed"
    assert facts["isAbsolute"] is True
    assert facts["notSystemDrive"] is True, facts["recommended"]
    assert facts["leaf"] == "AutoTuneStudio"


def test_an_explicit_directory_is_used_as_given(work_root, tmp_path):
    target = tmp_path / "AutoTuneStudio"
    facts = _run("install-root-layout", work_root, Variant="accept", InstallRoot=target)

    assert facts["accepted"] is True
    assert facts["root"] == str(target)
    assert facts["sameAsAsked"] is True
    assert facts["dataExists"] is True
    assert facts["separateData"] is True
    assert facts["interpreterInside"] is True
    assert facts["createdCount"] > 0


def test_a_directory_on_another_drive_is_accepted(work_root):
    """The operator is expected to pick a roomy local disk, not the system one.

    The candidate is only resolved, never created: the point is that the
    delivery accepts the destination, and a test must not write to the
    operator's own disk.
    """
    other = next((f"{letter}:\\AutoTuneStudio" for letter in "DEFG"
                  if Path(f"{letter}:\\").exists()), "")
    if not other:
        pytest.skip("this machine has no second fixed drive")
    facts = _run("install-root-layout", work_root, Variant="inspect", InstallRoot=other)

    assert facts["accepted"] is True
    assert facts["root"] == other
    assert facts["separateData"] is True


def test_chinese_space_and_long_directories_are_accepted(work_root, tmp_path):
    target = tmp_path / "自动 调优 安装 目录" / ("很长的目录名" * 6) / "AutoTuneStudio"
    facts = _run("install-root-layout", work_root, Variant="accept", InstallRoot=target)

    assert facts["accepted"] is True
    assert facts["root"] == str(target)
    assert facts["dataExists"] is True


def test_a_short_path_names_the_same_directory(work_root):
    facts = _run("install-root-layout", work_root, Variant="accept-short")

    if not facts["shortAvailable"]:
        pytest.skip("this volume does not hand out 8.3 short names")
    assert facts["isLongForm"] is True, (
        "the file system's own spelling of the directory is the one installation uses")


@pytest.mark.parametrize("candidate", [
    r"AutoTuneStudio",
    r"..\AutoTuneStudio",
    r"\\server\share\AutoTuneStudio",
    "C:\\",
    r"C:\Windows\AutoTuneStudio",
    r"C:\Windows\System32\AutoTuneStudio",
    r"C:\Program Files\AutoTuneStudio",
    r"C:\Program Files (x86)\AutoTuneStudio",
    r"C:\ProgramData\AutoTuneStudio",
])
def test_a_dangerous_destination_is_refused(work_root, candidate):
    facts = _run("install-root-layout", work_root, Variant="reject", InstallRoot=candidate)

    assert facts["accepted"] is False, f"{candidate} was accepted"
    assert facts["code"] == "INSTALL_ROOT_INVALID"
    assert facts["message"]


@pytest.mark.parametrize("variable", ["USERPROFILE", "LOCALAPPDATA", "APPDATA", "PUBLIC",
                                      "SystemRoot", "ProgramFiles"])
def test_a_user_or_system_root_is_never_an_installation(work_root, variable):
    import os

    candidate = os.environ.get(variable)
    if not candidate:
        pytest.skip(f"{variable} is not set on this machine")
    facts = _run("install-root-layout", work_root, Variant="reject", InstallRoot=candidate)

    assert facts["accepted"] is False, f"{variable} root was accepted"
    assert facts["code"] == "INSTALL_ROOT_INVALID"


def test_the_installed_entry_points_always_name_the_same_directory(work_root, held_port):
    """A chosen destination must survive: start, upgrade and uninstall read the
    installation's own record instead of asking again."""
    facts = _run("install-root-binding", work_root, Port=held_port)

    assert facts["root"].endswith("chosen-root\\AutoTuneStudio")
    assert facts["locationExists"] is True, "the installation records where it lives"
    assert facts["recordedRoot"] == facts["root"]
    assert facts["entryResolved"] is True
    assert facts["recordResolved"] is True
    assert facts["nothingResolved"] is True, "with no record there is no guess"
    assert facts["reachedStartGate"] is True, (
        f"the installed entry point did not reach its own installation: {facts}")
    assert facts["wrongRootRejected"] is True, (
        "the entry point must not fall back to a second directory")


@pytest.fixture(scope="module")
def held_port():
    """A loopback port that stays occupied while the installed entry point runs.

    The installed start.bat is executed for real, so the product's own start
    gates answer PORT_IN_USE — which is what proves the entry point reached the
    installation it belongs to."""
    import socket

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
    except OSError:  # pragma: no cover - depends on the machine
        listener.close()
        pytest.skip("no loopback port could be reserved")
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()


# ── preparing the bundle on the build machine ───────────────────────────────


def test_preparation_produces_a_deterministic_bundle(work_root):
    facts = _run("offline-prepare", work_root, Variant="ok")

    assert facts["ok"] is True, f"{facts['code']}: {facts['message']}"
    assert facts["lockExists"] is True
    assert facts["deterministic"] is True, "the same inputs must give the same lock"
    assert facts["lockHasAbsolutePath"] is False
    assert facts["requirementsSha"] == facts["expectedRequirementsSha"]
    assert sorted(facts["lockPurposes"]) == ["cuda-torch", "cuda-torchvision",
                                             "private-python-runtime", "runtime-dependency",
                                             "runtime-dependency"]
    assert facts["minicondaCopied"] is True
    assert facts["minicondaHash"] == facts["minicondaSourceHash"]


def test_preparation_takes_the_core_files_from_the_local_folder(work_root):
    """The 2.4 GB CUDA wheel is a local input, never downloaded again."""
    facts = _run("offline-prepare", work_root, Variant="ok")

    assert facts["torchDownloads"] == [], "nothing may ask for the torch wheel again"
    for command in facts["pipArgs"]:
        assert "--find-links" in command
        assert "torch-" not in command, "the wheel itself is not a download target"


def test_preparation_only_accepts_windows_cpython_binary_wheels(work_root):
    facts = _run("offline-prepare", work_root, Variant="ok")

    assert facts["pipArgs"], "the ordinary dependencies are resolved with pip"
    command = facts["pipArgs"][0]
    for flag in ("--only-binary=:all:", "--platform win_amd64", "--python-version 3.10",
                 "--implementation cp", "--abi cp310"):
        assert flag in command, flag
    assert facts["pipFile"].endswith("python.exe")


def test_preparation_lists_the_packages_it_could_not_get(work_root):
    facts = _run("offline-prepare", work_root, Variant="missing-wheel")

    assert facts["ok"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    detail = facts["detail"]
    assert detail, "the failure must name the package, not just fail"
    missing = detail["Missing"]
    assert [entry["Name"] for entry in missing] == ["uvicorn"]
    assert missing[0]["Version"] == "0.51.0"
    assert missing[0]["Url"].startswith("https://pypi.org/project/uvicorn/")
    assert any("uvicorn-0.51.0" in wheel for wheel in missing[0]["Wheels"])


def test_a_failed_download_keeps_the_verified_files_and_says_how_to_retry(work_root):
    facts = _run("offline-prepare", work_root, Variant="transient-download-failure")

    assert facts["ok"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    assert facts["minicondaCopied"] is True, "the verified files are kept"
    assert facts["minicondaHash"] == facts["minicondaSourceHash"]
    assert "重试" in facts["message"]


def test_preparation_names_the_transitive_dependency_pip_could_not_get(work_root):
    """pip can fail on a package the lock never pins directly.

    Every wheel the requirements file names is already in the wheelhouse, so the
    direct-pin report is empty. Without reading pip's own "no distribution"
    lines the operator would be handed an exit code and nothing to fetch."""
    facts = _run("offline-prepare", work_root, Variant="transitive-missing-wheel")

    assert facts["ok"] is False
    assert facts["code"] == "OFFLINE_WHEELHOUSE_INCOMPLETE"
    assert facts["directMissing"] == [], "the fixture must resolve every pinned wheel"
    assert facts["missingFromPip"] == ["numpy>=1.23.5", "contourpy==1.3.2"]
    for spec in ("numpy>=1.23.5", "contourpy==1.3.2"):
        assert spec in facts["message"], facts["message"]


def test_a_failed_preparation_never_echoes_pips_own_output(work_root):
    """pip's output names local build directories and index credentials; only the
    package specifier may survive into the report."""
    facts = _run("offline-prepare", work_root, Variant="transitive-missing-wheel")

    assert facts["leakedSecrets"] == [], (
        "the report carried a credential or a local path from pip's output")
    assert facts["messageHasAbsolutePath"] is False
    assert facts["minicondaCopied"] is True, "the verified files are kept"
    assert "重试" in facts["message"]


def test_the_pip_report_stays_bounded_and_keeps_only_package_names(work_root):
    """A long failing list must not become a report, and a line that merely looks
    like a requirement must still be refused when it carries a path."""
    facts = _run("offline-prepare", work_root, Variant="transitive-missing-many")

    assert facts["ok"] is False
    assert facts["missingFromPip"] == ["numpy==1.0.0", "contourpy==1.0.0", "pillow==1.0.0",
                                       "scipy==1.0.0", "pandas==1.0.0"], \
        "the report is capped and carries specifiers only"
    assert facts["leakedSecrets"] == []
    assert facts["messageHasAbsolutePath"] is False


def test_a_pip_failure_with_no_named_package_is_still_reported_honestly(work_root):
    """The transient failure names no package at all: the report must say so
    rather than invent one."""
    facts = _run("offline-prepare", work_root, Variant="transient-download-failure")

    assert facts["missingFromPip"] == []
    assert facts["leakedSecrets"] == []


def test_preparation_refuses_a_missing_or_tampered_core_file(work_root):
    missing = _run("offline-prepare", work_root, Variant="miniconda-missing")
    tampered = _run("offline-prepare", work_root, Variant="source-hash-mismatch")

    assert missing["ok"] is False
    assert missing["code"] == "OFFLINE_BUNDLE_MISSING"
    assert tampered["ok"] is False
    assert tampered["code"] == "OFFLINE_BUNDLE_HASH_MISMATCH"


# ── the scripts themselves ──────────────────────────────────────────────────


def test_the_preparation_script_is_a_thin_entry_point():
    text = (WINDOWS / "prepare_offline_bundle.ps1").read_text(encoding="utf-8")

    assert "Prepare-OfflineBundle" in text
    assert "-DependencySource" in text
    assert "$PSScriptRoot" in text
    assert (WINDOWS / "prepare_offline_bundle.ps1").read_bytes().startswith(b"\xef\xbb\xbf")


def _code_lines(path: Path) -> str:
    """The file without its comment lines: a comment may explain a rule, only a
    call can break it."""
    return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                     if not line.strip().startswith("#"))


def test_no_delivery_script_can_fetch_anything():
    """The installation is offline: no download client may exist in the delivery.

    Two URLs may remain. The loopback health check, and the PyPI *project page*
    the build machine prints in its missing-package report for a human to open —
    nothing fetches either of them, and no remote host is reachable by code.
    """
    clients = ("System.Net.WebClient", "HttpClient", "DownloadFile",
               "Invoke-RestMethod", "DownloadString", "SecurityProtocol",
               "Start-BitsTransfer")
    for path in sorted(WINDOWS.glob("*.ps1")) + [MODULE]:
        code = _code_lines(path)
        for needle in clients:
            assert needle not in code, f"{path.name} can reach the network: {needle}"
        for url in re.findall(r"https?://[^\s'\"]+", code):
            assert ("127.0.0.1" in url or url.startswith("https://pypi.org/project/")), (
                f"{path.name} mentions a fetchable URL: {url}")


def test_the_dependency_download_is_the_only_network_step():
    """The one step that needs a network is the build machine's preparation, and
    the command it runs is built in the tested module with binary-only,
    Windows, CPython 3.10 arguments."""
    script = _code_lines(WINDOWS / "prepare_offline_bundle.ps1")

    assert "Prepare-OfflineBundle" in script
    assert "'pip'" not in script and "'download'" not in script, (
        "the download arguments live in the tested module")
    module = MODULE.read_text(encoding="utf-8")
    for flag in ("--only-binary=:all:", "'--platform'", "'win_amd64'", "'--python-version'",
                 "'--implementation'", "'--abi'"):
        assert flag in module, flag
    # No index may be configured anywhere in the delivery: the pinned wheel index
    # is provenance recorded in the manifest (and checked against the trusted-host
    # list), never an argument a script hands to pip.
    code = _code_lines(MODULE)
    assert "index-url" not in code, "the delivery configures a package index"
    assert "--find-links" in code and "--no-index" in code


def test_the_installer_never_calls_conda_create():
    text = MODULE.read_text(encoding="utf-8")

    assert "conda.exe" not in text, "the private interpreter is the packaged Miniconda"
    assert "'create'" not in text
    assert '"create"' not in text
