"""F1.2-D: static contracts for the Windows delivery files.

The scripts are read as text and, where a syntax check is possible, parsed by
the real PowerShell parser. Nothing here installs anything, touches the network,
the registry or the operator's data; the behaviour suites drive the same files
with injected boundaries.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWS = REPO_ROOT / "windows"

BATS = ("install.bat", "start.bat", "upgrade.bat", "uninstall.bat")
SCRIPTS = ("install.ps1", "start.ps1", "upgrade.ps1", "uninstall.ps1", "build_zip.ps1",
           "prepare_offline_bundle.ps1")
MODULE = WINDOWS / "lib" / "AutoTuneDelivery.psm1"
MANIFEST = WINDOWS / "package-manifest.json"
LOCK = WINDOWS / "requirements-windows.lock.txt"
DOCKER_LOCK = REPO_ROOT / "docker" / "requirements-runtime.txt"
HARNESS = Path(__file__).resolve().parent / "ps" / "windows_delivery_harness.ps1"

_POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

_PIN_PATTERN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;]+)$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![\w-])[A-Za-z]:[\\/]")

SYSTEM_MODIFYING = (
    "reg add",
    "reg delete",
    "Set-ItemProperty",
    "New-ItemProperty",
    "HKCU:",
    "HKLM:",
    "[Environment]::SetEnvironmentVariable",
    "setx ",
    "schtasks",
    "AddToPath=1",
    "RegisterPython=1",
)

# A credential is a *value*: the redaction patterns themselves may name the
# fields, so only an assignment of something key-shaped counts.
_CREDENTIAL_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|password|secret|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}")
_API_KEY_LITERAL = re.compile(r"sk-[A-Za-z0-9]{16,}")

CPU_FALLBACK_SWITCHES = (
    "-SkipGpu",
    "--no-gpu",
    "--allow-cpu",
    "AUTO_TUNE_ALLOW_CPU",
    "AUTO_TUNE_DEVICE_OVERRIDE",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bat_files() -> list[Path]:
    return [WINDOWS / name for name in BATS]


def _script_files() -> list[Path]:
    return [WINDOWS / name for name in SCRIPTS] + [MODULE]


def _pinned_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw_line in _text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_PATTERN.match(line)
        assert match, f"entry is not exactly pinned: {line!r}"
        packages[match.group(1).lower()] = match.group(2)
    return packages


# ── inventory and syntax ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", BATS + SCRIPTS)
def test_every_delivery_script_exists(name):
    assert (WINDOWS / name).is_file(), f"windows/{name} is missing"


def test_the_delivery_has_one_shared_module():
    assert MODULE.is_file(), "windows/lib/AutoTuneDelivery.psm1 is missing"


@pytest.mark.parametrize("path", [WINDOWS / name for name in SCRIPTS] + [MODULE])
def test_powershell_files_carry_a_utf8_bom(path):
    """Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.

    The operator-facing messages are Chinese, so without the BOM every script
    would fail to parse on a machine that has not opted into UTF-8."""
    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), f"{path.name} has no UTF-8 BOM"


def test_every_powershell_file_parses():
    """A syntax check by the real parser, not a string comparison."""
    if _POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")
    files = _script_files() + [HARNESS]
    arguments = ", ".join(f"'{path}'" for path in files)
    command = (
        "$failed = 0; "
        f"foreach ($f in @({arguments})) {{ "
        "$errors = $null; "
        "$tokens = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$tokens, [ref]$errors); "
        "foreach ($e in $errors) { Write-Output \"$f::$($e.Message)\"; $failed = 1 } }; "
        "exit $failed"
    )
    result = subprocess.run([_POWERSHELL, "-NoProfile", "-NonInteractive",
                             "-ExecutionPolicy", "Bypass", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=180)
    assert result.returncode == 0, f"PowerShell parse errors:\n{result.stdout}{result.stderr}"


# ── the double-click entry points ───────────────────────────────────────────


@pytest.mark.parametrize("name", BATS)
def test_batch_entry_points_locate_their_script_from_their_own_location(name):
    text = _text(WINDOWS / name)

    assert "%~dp0" in text, "a double-clicked .bat must not depend on the current directory"
    assert "@echo off" in text.lower()
    for line in text.lower().splitlines():
        if line.strip().startswith("cd "):
            assert "%~dp0" in line, f"{name} changes directory by assumption: {line!r}"


@pytest.mark.parametrize("name", BATS)
def test_batch_entry_points_delegate_to_powershell(name):
    text = _text(WINDOWS / name)
    script = name.replace(".bat", ".ps1")

    assert script in text
    assert "%~dp0" + script in text
    assert "powershell.exe" in text.lower()
    assert "pwsh" not in text.lower(), "PowerShell 7 must not be required"


PROMPT_FORBIDDING_FLAGS = ("-noninteractive",)


def _powershell_arguments(name: str) -> list[str]:
    """The switches a .bat hands to powershell.exe, up to the ``-File`` part."""
    for raw_line in _text(WINDOWS / name).splitlines():
        line = raw_line.strip()
        if "powershell.exe" not in line.lower():
            continue
        arguments: list[str] = []
        for token in line.split()[1:]:
            if token.lower() == "-file":
                break
            arguments.append(token)
        return arguments
    raise AssertionError(f"windows/{name} never calls powershell.exe")


def test_the_install_entry_point_does_not_forbid_the_prompt():
    """install.ps1 asks for the install directory on a first installation.

    PowerShell refuses ``Read-Host`` in a non-interactive host, so the install
    entry point is the one .bat that must not pass that switch: a double-clicked
    install.bat would otherwise fail on the very first question instead of
    showing it."""
    arguments = [argument.lower() for argument in _powershell_arguments("install.bat")]

    for flag in PROMPT_FORBIDDING_FLAGS:
        assert flag not in arguments, (
            f"install.bat launches PowerShell with {flag}, which turns the "
            "install-directory prompt into a failure")


@pytest.mark.parametrize("name", ["start.bat", "upgrade.bat", "uninstall.bat"])
def test_every_other_entry_point_stays_non_interactive(name):
    """Only the install entry point needs a console: the others never ask, so
    they must not be switched to an interactive host by accident."""
    arguments = [argument.lower() for argument in _powershell_arguments(name)]

    assert "-noninteractive" in arguments, (
        f"{name} never prompts, so it must stay non-interactive")


def test_only_the_install_script_ever_reads_from_the_operator():
    """One script asks, one batch entry point allows it, and the rest never wait."""
    readers = sorted(path.name for path in sorted(WINDOWS.glob("*.ps1"))
                     if any(not line.strip().startswith("#") and "Read-Host" in line
                            for line in _text(path).splitlines()))

    assert readers == ["install.ps1"], (
        f"asking the operator is install.ps1's job alone: {readers}")
    assert "Get-InstallPromptPlan" in _text(WINDOWS / "install.ps1"), (
        "the prompt decision is shared with the behaviour suite, not re-derived here")


@pytest.mark.parametrize("name", BATS)
def test_batch_entry_points_forward_arguments_and_propagate_the_exit_code(name):
    text = _text(WINDOWS / name)

    # install/start/upgrade forward their arguments verbatim; uninstall.bat
    # translates the switch spelling each PowerShell script expects.
    assert "%*" in text or "%~1" in text, "arguments must reach the PowerShell layer"
    assert "exit /b" in text.lower(), "the exit code must reach the caller"
    assert "AUTO_TUNE_NO_PAUSE" in text, "an automated run must be able to skip the pause"


@pytest.mark.parametrize("name", BATS)
def test_batch_entry_points_are_ascii(name):
    """cmd.exe decodes a .bat in the console code page; ASCII cannot go wrong.

    The Chinese messages are printed by the PowerShell layer, which is UTF-8
    with a BOM and therefore unambiguous."""
    text = (WINDOWS / name).read_bytes()

    assert text.decode("ascii")
    assert not text.startswith(b"\xef\xbb\xbf"), "a BOM confuses cmd.exe"


@pytest.mark.parametrize("name", BATS)
def test_batch_entry_points_never_call_a_bare_interpreter(name):
    text = _text(WINDOWS / name)

    for forbidden in ("python", "conda", "pip", "py -"):
        assert not re.search(rf"(?mi)^\s*{re.escape(forbidden)}\b", text), forbidden


# ── the PowerShell layer ────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [WINDOWS / name for name in SCRIPTS])
def test_scripts_resolve_their_own_location(path):
    text = _text(path)

    assert "$PSScriptRoot" in text
    assert "Get-Location" not in text
    assert "Split-Path $MyInvocation" not in text


def test_scripts_never_modify_the_system():
    for path in _script_files():
        text = _text(path)
        for forbidden in SYSTEM_MODIFYING:
            assert forbidden not in text, f"{path.name} modifies the system: {forbidden}"


@pytest.mark.parametrize("name", [f"{stem}.ps1" for stem in ("install", "start", "upgrade")])
def test_scripts_never_offer_a_cpu_fallback(name):
    text = _text(WINDOWS / name)
    for forbidden in CPU_FALLBACK_SWITCHES:
        assert forbidden not in text, f"{name} offers a CPU fallback: {forbidden}"


def test_the_private_runtime_installation_stays_out_of_path_and_the_registry():
    text = _text(MODULE)

    assert "RegisterPython=0" in text
    assert "AddToPath=0" in text


def test_start_runs_the_shared_gpu_preflight_before_the_studio():
    """The order is asserted where the commands are actually built.

    start.ps1 delegates to Start-Studio, so the two commands live in one place.
    That the preflight really runs first, with the GPU required, is asserted
    behaviourally in test_windows_delivery_lifecycle.py."""
    module = _text(MODULE)

    assert "'auto_tune.delivery.preflight'" in module
    assert "'-m', 'auto_tune.main'" in module
    assert module.index("'auto_tune.delivery.preflight'") < module.index("'auto_tune.main'")
    assert "-Arguments @('--require-gpu')" in module

    start = _text(WINDOWS / "start.ps1")
    assert "Start-Studio" in start, "the launcher must not re-implement the start gates"


def test_the_delivery_reuses_the_shared_runtime_and_preflight_modules():
    """One GPU rule for both deliveries: the Windows layer must not re-implement it.

    The installer may check that an NVIDIA *driver* is present, but whether the
    delivered runtime can really see a CUDA device is decided by the shared
    preflight, so the PowerShell layer never imports torch to answer it.
    """
    text = _text(MODULE)

    assert "auto_tune.delivery.preflight" in text
    assert "cuda.is_available" not in text
    assert "import torch" not in text
    assert "PREFLIGHT_FAILED" in text, "the preflight exit code is the GPU gate"


def test_the_module_exposes_the_stable_error_codes():
    text = _text(MODULE)

    for code in (
        "LOCALAPPDATA_MISSING",
        "DISK_SPACE_INSUFFICIENT",
        "NVIDIA_DRIVER_MISSING",
        "PACKAGE_HASH_MISMATCH",
        "LAUNCHER_INSTALL_FAILED",
        "INSTALL_STATE_INCOMPLETE",
        "PORT_INVALID",
        "PORT_IN_USE",
        "PREFLIGHT_FAILED",
        "UNSAFE_REMOVAL_TARGET",
        "CONFIRMATION_REQUIRED",
        "HEALTH_CHECK_FAILED",
        # a recorded process that is alive but no longer answers /healthz
        "STALE_INSTANCE_DETECTED",
        "STALE_INSTANCE_CLEARED",
        "STALE_INSTANCE_UNSTOPPABLE",
        # the offline installation
        "INSTALL_ROOT_INVALID",
        "OFFLINE_BUNDLE_MISSING",
        "OFFLINE_BUNDLE_HASH_MISMATCH",
        "OFFLINE_WHEELHOUSE_INCOMPLETE",
        "OFFLINE_RUNTIME_INSTALL_FAILED",
    ):
        assert code in text, f"the module does not define {code}"


def test_the_delivery_no_longer_knows_how_to_download():
    """The installation is offline; a download path would be dead code that a
    future change could quietly reactivate."""
    text = _text(MODULE)

    for gone in ("DOWNLOAD_FAILED", "DOWNLOAD_HASH_MISMATCH", "New-DownloadRequest",
                 "Get-VerifiedFile", "Invoke-HttpsDownload", "New-DownloadReport"):
        assert gone not in text, f"{gone} survived the offline rework"


def _code_text(path: Path) -> str:
    """The file without its comment lines.

    A comment may legitimately explain *why* the cmdlet is not used; only a call
    would make the delivery depend on it."""
    kept = []
    for line in _text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.lower().startswith("rem"):
            continue
        kept.append(line)
    return "\n".join(kept)


def test_the_delivery_never_calls_the_get_file_hash_cmdlet():
    """A machine with PowerShell 7 installed can break Get-FileHash for 5.1.

    The cmdlet comes from Microsoft.PowerShell.Utility, which is loaded from
    PSModulePath; when PowerShell 7's copy is picked up by a 5.1 host the
    cmdlet is simply gone. The delivery must therefore hash with the CLR,
    which is always present, rather than ask the shell for it.
    """
    for path in _delivery_files():
        assert "Get-FileHash" not in _code_text(path), f"{path.name} depends on Get-FileHash"


def test_the_delivery_hashes_files_with_the_cryptography_api():
    text = _text(MODULE)

    assert "System.Security.Cryptography.SHA256" in text
    assert "Get-FileSha256" in text, "one shared hash helper is exported"


def test_the_harness_hides_the_get_file_hash_cmdlet():
    """The behaviour suites must run on a machine where the cmdlet is missing."""
    text = _text(HARNESS)

    assert "function global:Get-FileHash" in text


def test_the_harness_only_ever_mentions_the_cmdlet_to_hide_it():
    """A test driver that still used the cmdlet would fail for the wrong reason."""
    code = _code_text(HARNESS)

    assert "System.Security.Cryptography.SHA256" in code
    assert "-Algorithm" not in code, "the harness still calls Get-FileHash"


# ── the permanent entry points ──────────────────────────────────────────────


def test_the_installer_deploys_the_permanent_entry_points():
    text = _text(MODULE)

    for name in ("start.bat", "start.ps1", "uninstall.bat", "uninstall.ps1"):
        assert name in text, f"the installer does not deploy {name}"
    assert "Auto Tune Studio.lnk" in text, "the desktop shortcut is part of the install"
    assert "WScript.Shell" in text, "the shortcut is a real .lnk, not a stub"


def test_the_uninstall_entry_point_forwards_its_arguments():
    """uninstall.bat translates the switches *and* passes anything else on."""
    text = _text(WINDOWS / "uninstall.bat")

    assert "%*" in text or "%~1" in text
    assert "ARGS" in text.upper(), "extra arguments must reach the PowerShell layer"


def test_start_and_uninstall_batch_files_are_present_in_the_package():
    manifest = _manifest()

    includes = manifest["package"]["scripts_include"]
    assert "*.bat" in includes
    assert any(pattern.startswith("lib") for pattern in includes)


def test_the_service_stays_on_the_loopback_interface():
    assert "127.0.0.1" in _text(MODULE)


def test_the_delivery_never_writes_a_credential_into_a_file():
    """The scripts may name the *fields* they redact, but never a real value."""
    for path in _delivery_files():
        text = _text(path)
        assert not _CREDENTIAL_VALUE.search(text), f"{path.name} contains a credential"
        assert not _API_KEY_LITERAL.search(text), f"{path.name} contains an API key"


def test_the_delivery_refuses_to_ship_the_persisted_credential_file():
    """The container's key file is operator data: the payload whitelist must drop
    it whatever the manifest says, exactly like the other local-state names."""
    text = _text(MODULE)

    assert "PayloadDeniedNames" in text
    assert "credentials.json" in text, "the credential file name is not denied"


def _delivery_files() -> list[Path]:
    return (sorted(WINDOWS.glob("*.bat")) + sorted(WINDOWS.glob("*.ps1"))
            + [MODULE, MANIFEST, LOCK])


# ── the dependency lock ─────────────────────────────────────────────────────


def test_the_windows_lock_pins_every_runtime_package_exactly():
    pinned = _pinned_packages(LOCK)

    for name in ("fastapi", "uvicorn", "jinja2", "pyyaml", "numpy",
                 "opencv-python", "scikit-learn", "ultralytics", "optuna",
                 "onnx", "onnxruntime"):
        assert name in pinned, f"{name} is imported in production but not pinned"

    text = _text(LOCK)
    for forbidden in (">=", "<=", "~=", "!=", ">", "<"):
        assert forbidden not in text, f"the lock uses a loose constraint {forbidden!r}"


def test_the_windows_lock_agrees_with_the_accepted_container_runtime():
    """One verified runtime, two deliveries: no second version of a shared package."""
    windows = _pinned_packages(LOCK)
    container = _pinned_packages(DOCKER_LOCK)

    assert set(windows) == set(container), (
        "the Windows lock must not add or drop runtime packages silently: "
        f"windows-only={sorted(set(windows) - set(container))} "
        f"container-only={sorted(set(container) - set(windows))}")
    assert windows == container, "a shared package has two versions"


def test_the_windows_lock_keeps_the_cuda_torch_and_unused_extras_out():
    pinned = _pinned_packages(LOCK)

    assert "torch" not in pinned, "torch comes from the pinned CUDA 12.1 index"
    assert "torchvision" not in pinned
    for name in ("onnxruntime-gpu", "onnxslim", "pytest", "pyreadline3", "gradio", "httpx"):
        assert name not in pinned, f"{name} is not part of the delivered runtime"


def test_the_windows_lock_contains_no_url_path_or_credential():
    text = _text(LOCK)

    for forbidden in ("git+", "git@", "github.com", "file://", "http://", "https://",
                      "\\\\", "api_key", "apikey", "token", "password", "secret"):
        assert forbidden.lower() not in text.lower(), f"the lock leaks {forbidden!r}"
    assert not _WINDOWS_ABSOLUTE_PATH.search(text)


# ── the package manifest ────────────────────────────────────────────────────


def _manifest() -> dict:
    return json.loads(_text(MANIFEST))


def test_the_manifest_declares_the_product_and_the_python():
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["product"] == "auto-tune-studio"
    assert manifest["python_version"] == "3.10"
    assert manifest["version"]


def test_the_manifest_carries_the_repaired_delivery_version():
    """The candidate package replaces ``AutoTuneStudio-Setup-0.2.0.zip``: a later
    version is what tells the installer this is not the same build again."""
    assert _manifest()["version"] == "0.2.1"


def test_the_shipped_package_name_follows_the_manifest_version():
    """``build_zip.ps1`` names the archive from the manifest when no explicit
    ``-Version`` is given (the behavioural proof is in test_windows_delivery_zip.py)."""
    text = _text(WINDOWS / "build_zip.ps1")
    manifest = _manifest()

    assert "AutoTuneStudio-Setup-{0}.zip" in text
    assert text.index("$manifest = Get-PackageManifest") < \
        text.index("AutoTuneStudio-Setup-{0}.zip")
    assert f"AutoTuneStudio-Setup-{manifest['version']}.zip" == \
        "AutoTuneStudio-Setup-0.2.1.zip"


def test_the_manifest_pins_the_private_runtime_download():
    installer = _manifest()["runtime"]["conda_installer"]

    assert installer["url"].startswith("https://")
    assert "example.invalid" not in installer["url"]
    assert installer["file_name"].endswith(".exe")
    assert re.fullmatch(r"[0-9a-f]{64}", installer["sha256"]), "a real SHA-256 is required"
    assert int(installer["size"]) > 0


def test_the_manifest_pins_the_cuda_torch_build():
    torch = _manifest()["runtime"]["torch"]

    assert torch["packages"] == ["torch==2.5.1", "torchvision==0.20.1"]
    assert torch["index_url"] == "https://download.pytorch.org/whl/cu121"


def test_the_manifest_pins_the_offline_wheels_in_the_package():
    """The CUDA wheels travel inside the ZIP: the installer has no index to ask."""
    wheels = {wheel["purpose"]: wheel for wheel in _manifest()["runtime"]["torch"]["wheels"]}

    assert set(wheels) == {"cuda-torch", "cuda-torchvision"}
    assert wheels["cuda-torch"]["file_name"] == "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
    assert wheels["cuda-torch"]["version"] == "2.5.1+cu121"
    assert wheels["cuda-torch"]["size"] == 2449372784
    assert wheels["cuda-torch"]["sha256"] == \
        "9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f"
    assert wheels["cuda-torchvision"]["file_name"] == \
        "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl"
    assert wheels["cuda-torchvision"]["sha256"] == \
        "4cb1e44c1f7a4992f6d38a15633a1e694c093f1c52f3a036b1a719968031507a"
    for wheel in wheels.values():
        assert wheel["file_name"].endswith(".whl")
        assert re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
        assert int(wheel["size"]) > 0


def test_the_manifest_declares_the_offline_bundle_layout():
    offline = _manifest()["runtime"]["offline"]

    assert offline == {
        "directory": "offline",
        "miniconda_directory": "miniconda",
        "wheelhouse_directory": "wheelhouse",
        "lock_file": "offline-lock.json",
    }


def test_the_manifest_keeps_the_miniconda_installer_pinned_for_the_bundle():
    installer = _manifest()["runtime"]["conda_installer"]

    assert installer["file_name"] == "Miniconda3-py310_25.1.1-2-Windows-x86_64.exe"
    assert installer["size"] == 90623048


def test_the_package_ships_no_build_time_script():
    """The two build tools stay in the repository: they must not be handed to an
    operator who only unzips and installs."""
    excluded = _manifest()["package"]["scripts_exclude"]

    assert "build_zip.ps1" in excluded
    assert "prepare_offline_bundle.ps1" in excluded


def test_the_manifest_points_at_the_locked_requirements():
    runtime = _manifest()["runtime"]

    assert runtime["pip_requirements"] == "requirements-windows.lock.txt"
    assert runtime["pip_requirements_sha256"] == _sha256(LOCK)


def test_the_manifest_limits_the_payload_to_the_sanitized_program():
    payload = _manifest()["payload"]

    assert payload["include"] == ["auto_tune/"]
    # test fixtures, the operator-only verification script, the evaluation
    # harness (used by that script only), internal notes and the stale
    # dependency list are not part of the product
    for excluded in ("auto_tune/tests/", "auto_tune/scripts/", "auto_tune/evaluation/",
                     "auto_tune/docs/", "auto_tune/requirements.txt", "**/*.md"):
        assert excluded in payload["exclude"]


def test_the_manifest_contains_no_credential_or_local_path():
    text = _text(MANIFEST)

    for forbidden in ("api_key", "apikey", "token", "password", "secret", "D:\\", "E:\\"):
        assert forbidden.lower() not in text.lower(), f"the manifest leaks {forbidden!r}"


def test_the_manifest_requires_a_disk_budget():
    budget = _manifest()["install"]["required_free_bytes"]

    assert int(budget) >= 8 * 1024 ** 3, "the CUDA runtime needs a realistic budget"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
