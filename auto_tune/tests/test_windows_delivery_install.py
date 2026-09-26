"""F1.2-D: behaviour of the Windows installer, executed for real.

``auto_tune/tests/ps/windows_delivery_harness.ps1`` imports the shipping
PowerShell module and replaces only the boundaries the operator's machine
provides: the HTTP downloader, the process runner and the probes (disk, driver,
port, PID, health). Everything asserted below — the layout, the cache, the
SHA-256 gate, the bounded retries, the staging, the state file and the safety
checks — is the code the operator runs, minus the network and the hardware.

These tests need ``powershell.exe``; the whole module is skipped otherwise.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent / "ps" / "windows_delivery_harness.ps1"
MODULE = REPO_ROOT / "windows" / "lib" / "AutoTuneDelivery.psm1"

_POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(_POWERSHELL is None,
                                reason="Windows PowerShell is not available")

_CACHE: dict[str, dict] = {}


@pytest.fixture(scope="module")
def work_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("f12d-install")


@pytest.fixture(scope="module")
def held_port():
    """A port that is really occupied while the harness runs.

    The installed launcher is executed for real, so the product's own start
    gates must answer PORT_IN_USE — no interpreter and no service are needed to
    observe that the entry point reached the real logic."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()


def _run(scenario: str, work_root: Path, **extra) -> dict:
    """Execute one harness scenario and return its facts (cached per scenario)."""
    key = f"{scenario}:{sorted(extra.items())}:{work_root}"
    if key in _CACHE:
        return _CACHE[key]
    command = [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(HARNESS), "-Scenario", scenario,
               "-ModulePath", str(MODULE), "-WorkRoot", str(work_root)]
    for name, value in extra.items():
        command += [f"-{name}", str(value)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
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


# ── layout: program, runtime, cache, logs and data are separate ─────────────


def test_layout_keeps_program_runtime_cache_and_user_data_apart(work_root):
    facts = _run("layout-default", work_root)

    assert facts["underRoot"] is True
    assert facts["separateData"] is True
    assert facts["root"].endswith("AutoTuneStudio")
    for name in ("app", "runtime", "data", "cache", "logs"):
        assert facts[name].startswith(facts["root"])
    assert facts["interpreter"].startswith(facts["runtime"])
    assert facts["interpreter"].endswith("python.exe")
    assert facts["stateFile"] == str(Path(facts["root"]) / "install-state.json")


def test_layout_rejects_a_missing_or_blank_localappdata(work_root):
    facts = _run("layout-blank-localappdata", work_root)

    assert facts["rejected"] is True
    assert facts["code"] == "LOCALAPPDATA_MISSING"


def test_layout_supports_chinese_and_space_paths(work_root):
    facts = _run("layout-unicode-space", work_root)

    assert facts["unicodeKept"] is True
    assert facts["dataExists"] is True
    assert facts["stateWritable"] is True
    assert facts["createdCount"] > 0


# ── the private runtime, never the developer's python ───────────────────────


def test_install_runs_the_studio_preflight_with_the_private_interpreter(work_root):
    facts = _run("install-fresh", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["interpreterInside"] is True
    assert facts["interpreterUsed"] == facts["interpreter"]
    assert facts["preflightCwd"] == str(Path(facts["interpreter"]).parents[2] / "data")


def test_install_asks_the_shared_preflight_for_the_gpu(work_root):
    """No CPU fallback: the installer proves the delivered runtime sees the GPU."""
    facts = _run("install-fresh", work_root)

    assert "--require-gpu" in facts["preflightArgs"]


def test_install_exports_the_delivered_data_layout_to_the_preflight(work_root):
    facts = _run("install-fresh", work_root)
    env = facts["preflightEnv"]
    data = str(Path(facts["interpreter"]).parents[2] / "data")

    assert env["AUTO_TUNE_APP_ROOT"] == data
    assert env["AUTO_TUNE_CONFIG_PATH"] == str(Path(data) / "config" / "config.yaml")
    assert env["AUTO_TUNE_DATASETS_DIR"] == str(Path(data) / "datasets")
    assert env["PYTHONPATH"] == str(Path(facts["interpreter"]).parents[2] / "app")


def test_install_never_calls_a_bare_python_or_conda(work_root):
    """Every external program is absolute and lives in the installation."""
    facts = _run("install-fresh", work_root)

    assert facts["allCommandsAbsolute"] is True
    assert facts["outsideInstallCalls"] == 0, "no program outside the installation is run"
    assert facts["devInterpreterCalls"] == 0, "the developer's python is never used"


def test_install_creates_the_persistent_data_directories(work_root):
    facts = _run("install-fresh", work_root)

    assert set(facts["dataDirs"]) == {"log", "detect", "runs", "datasets"}
    assert facts["weightsDir"] is True


# ── payload and configuration ───────────────────────────────────────────────


def test_install_copies_the_sanitized_program_files(work_root):
    facts = _run("install-fresh", work_root)

    assert facts["appCopied"] is True
    assert facts["payloadTrimmed"] is True, "tests are not part of the product"
    assert facts["noRealConfigCopied"] is True, "a real config.yaml is never installed"
    assert facts["noWeightCopied"] is True, "weights never travel inside the program"
    assert facts["noCacheCopied"] is True, "python caches are not installed"
    assert facts["copiedFiles"] == 3


def test_install_bootstraps_the_configuration_and_then_leaves_it_alone(work_root):
    facts = _run("install-fresh", work_root)

    assert facts["configCreated"] is True
    assert facts["configKept"] is True, "the operator's configuration is never overwritten"


def test_install_preserves_data_from_an_earlier_installation(work_root):
    facts = _run("install-preserves-existing-data", work_root)

    assert facts["ok"] is True
    assert facts["preserved"] is True
    assert len(facts["files"]) == 3


# ── state file ──────────────────────────────────────────────────────────────


def test_install_writes_a_versioned_state_only_when_it_finished(work_root):
    facts = _run("install-fresh", work_root)

    assert facts["status"] == "complete"
    assert facts["schemaVersion"] == "1.0"
    assert facts["product"] == "auto-tune-studio"
    assert facts["stateRuntimeLock"]
    assert facts["configPathInState"].endswith("config.yaml")


def test_re_running_install_reuses_the_verified_runtime(work_root):
    """Nothing is downloaded at all: the reuse that matters is the private
    runtime, which is built once and then adopted by every later run."""
    facts = _run("install-fresh", work_root)

    assert facts["secondRunOk"] is True, facts["secondRunError"]
    assert facts["stateAfterSecond"] == "complete"
    assert facts["downloadCalls"] == 0, "the installation never downloads"
    assert facts["runtimeInstallCalls"] == 1, "the private runtime is not rebuilt"


def test_install_resumes_after_a_failed_attempt(work_root):
    """A half-finished installation must never be a dead end."""
    facts = _run("install-resume-after-failure", work_root)

    assert facts["firstFailed"] is True
    assert facts["partialStateStatus"] != "complete"
    assert facts["resumedOk"] is True, facts["resumedError"]
    assert facts["finalStatus"] == "complete"
    assert facts["configKept"] is True


def test_install_resumes_when_the_runtime_stopped_halfway(work_root):
    """The interpreter exists but the dependencies do not.

    Regression: a pip warning used to abort the dependency step (PowerShell 5.1
    turns a native command's stderr into a terminating error under the default
    preference). The environment must then be completed in place, not trusted as
    ready and not thrown away."""
    facts = _run("install-resumes-partial-runtime", work_root)

    assert facts["firstOk"] is False
    assert facts["interpreterCreated"] is True
    assert facts["stampAbsent"] is True, "an unfinished runtime is not stamped as ready"
    assert facts["resumedOk"] is True, facts["resumedError"]
    assert facts["finalStatus"] == "complete"
    assert facts["refreshOnly"] is True, "the second run finishes the dependencies in place"
    assert facts["installerCalls"] == 1


# ── the two delivery checks a first installation runs, and when ─────────────


def test_a_first_installation_checks_the_private_runtime_before_it_installs_anything(work_root):
    """Codex's real finding, as a regression.

    The runtime has just been built from the offline bundle, but the program
    payload, the configuration and the persistent directories do not exist yet.
    The offline precheck is a *runtime* check and must not need any of them: it
    asks the private interpreter about itself, nothing else."""
    facts = _run("install-runtime-check-order", work_root)

    assert facts["ok"] is True, f"{facts['errorCode']}: {facts['message']}"
    assert facts["runtimeCheckRan"] is True, (
        "the private runtime is asked to prove itself")
    assert "--require-offline-runtime" in facts["runtimeCheckArgs"]
    assert "--expect-python" in facts["runtimeCheckArgs"]
    assert facts["runtimeCheckFile"] == facts["interpreter"], (
        "the check runs the private interpreter, never a developer one")
    assert facts["runtimeCheckConfig"] is False, (
        "a first installation has no configuration yet; the runtime check must not need one")
    assert facts["runtimeCheckPayload"] is False, (
        "the program payload is not installed yet when the runtime is verified")


def test_the_runtime_check_needs_neither_mounts_nor_the_gpu_flag(work_root):
    """Asking for the GPU would make it the formal start-up preflight, whose
    full directory and configuration rules the installer cannot satisfy yet."""
    facts = _run("install-runtime-check-order", work_root)

    assert "--require-gpu" not in facts["runtimeCheckArgs"]
    assert "--require-mounts" not in facts["runtimeCheckArgs"]
    assert "--bootstrap-config" not in facts["runtimeCheckArgs"]


def test_the_runtime_check_is_handed_no_delivery_layout(work_root):
    """The installer gives it the private interpreter and the program it is
    importing, nothing else — so the check has to be able to run without a
    delivery layout, and the CLI must not fall back to a packaged one."""
    facts = _run("install-runtime-check-order", work_root)

    assert facts["runtimeCheckEnv"]["PYTHONPATH"]
    leaked = [key for key in facts["runtimeCheckEnv"] if key.startswith("AUTO_TUNE_")]
    assert leaked == [], f"a runtime check with no installation asked for {leaked}"


def test_the_installation_still_runs_the_gpu_preflight_once_it_installed(work_root):
    facts = _run("install-runtime-check-order", work_root)

    assert facts["startupCheckRan"] is True
    assert "--require-gpu" in facts["startupCheckArgs"]
    assert facts["startupCheckPython"] == facts["interpreter"]
    assert facts["startupCheckConfig"] is True, (
        "the formal preflight runs after the configuration is in place")
    assert facts["startupCheckPayload"] is True, (
        "the formal preflight runs after the program payload is installed")


def test_the_runtime_check_is_not_the_formal_startup_preflight(work_root):
    """Two distinct checks with different rules: the private runtime answers for
    itself first, and the formal ``--require-gpu`` preflight comes after the
    installation is in place — once, at the end."""
    facts = _run("install-runtime-check-order", work_root)

    assert facts["status"] == "complete"
    assert facts["order"].count("preflight") == 1
    assert facts["order"].count("run") > 1, (
        "the runtime check is one of the recorded commands, not the preflight")
    assert facts["fetchCalls"] == 0, "both checks stay offline"


# ── the permanent entry points the installation leaves behind ───────────────


def test_install_leaves_a_permanent_start_and_uninstall_entry(work_root):
    """The operator must not have to keep the extracted ZIP around."""
    facts = _run("install-deploys-launcher", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    for key in ("startBatExists", "startScriptExists", "uninstallBatExists",
                "uninstallScriptExists", "moduleExists"):
        assert facts[key] is True, key
    assert facts["insideInstallRoot"] is True
    assert facts["startBat"].endswith("start.bat")
    assert facts["uninstallBat"].endswith("uninstall.bat")
    assert Path(facts["modulePath"]).parent.name == "lib"


def test_the_installed_entry_points_are_the_verified_package_files(work_root):
    facts = _run("install-deploys-launcher", work_root)

    assert facts["startBatMatchesPackage"] is True
    assert facts["moduleMatchesPackage"] is True


def test_install_creates_a_desktop_shortcut_to_the_installed_launcher(work_root):
    facts = _run("install-deploys-launcher", work_root)

    assert facts["shortcutCreated"] is True
    assert facts["shortcutExists"] is True
    assert facts["shortcut"].endswith("Auto Tune Studio.lnk")
    assert facts["shortcutOnFakeDesktop"] is True, "the test must never touch the real desktop"
    assert facts["shortcutTarget"] == facts["startBat"]


def test_the_installed_launcher_still_works_after_the_package_is_deleted(work_root, held_port):
    facts = _run("launcher-survives-package-removal", work_root, Port=held_port)

    assert facts["packageGone"] is True, "the extracted delivery folder is gone"
    assert facts["missingEntry"] is False, "start.bat must not need the extracted package"
    assert facts["reachedStartGate"] is True, (
        f"the installed launcher did not reach the product start gates: {facts['output']!r}")


def test_the_installed_launcher_reports_a_non_zero_exit_code_on_failure(work_root, held_port):
    facts = _run("launcher-survives-package-removal", work_root, Port=held_port)

    assert facts["exitCode"] == 1
    assert "PORT_IN_USE" in facts["output"]
    assert facts["dataKept"] is True


# ── preconditions ───────────────────────────────────────────────────────────


def test_install_stops_when_the_disk_is_too_small(work_root):
    facts = _run("install-insufficient-disk", work_root)

    assert facts["rejected"] is True
    assert facts["code"] == "DISK_SPACE_INSUFFICIENT"


def test_a_first_installation_stops_before_building_anything_when_the_disk_is_full(work_root):
    """1 GiB free where 12 GiB is required: nothing is downloaded or built."""
    facts = _run("install-fresh-insufficient-disk", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "DISK_SPACE_INSUFFICIENT"
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["appCopied"] is False, "the program files are not put in place"
    assert facts["leftovers"] == [], "no staging directory is left behind"


def test_an_incomplete_installation_is_never_reported_as_a_repair(work_root):
    """A state file that says "complete" over a missing runtime is not usable.

    Repairing the entry points of an installation that could not start would
    only hide the broken installation, so this needs the full installation
    budget and fails honestly when there is none."""
    facts = _run("install-incomplete-insufficient-disk", work_root)

    assert facts["firstOk"] is True
    assert facts["ok"] is False
    assert facts["errorCode"] == "DISK_SPACE_INSUFFICIENT"
    assert facts["mode"] not in ("already-installed", "launcher-repair")
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["interpreterRestored"] is False


def test_install_restores_the_entry_points_without_the_fresh_installation_budget(work_root):
    """Codex's real finding.

    An installation of the same version that is complete except for the
    permanent entry points must be repaired on a machine that has 1 GiB free
    and could never host a fresh 12 GiB installation."""
    facts = _run("install-repairs-launcher-with-scarce-disk", work_root)

    assert facts["launchersBefore"] is False, "the fixture starts without the permanent entry points"
    assert facts["ok"] is True, facts["errorCode"]
    assert facts["mode"] == "launcher-repair"
    assert facts["launchersAfter"] is True
    assert facts["startBatMatchesPackage"] is True
    assert facts["moduleMatchesPackage"] is True
    assert facts["shortcutExists"] is True
    assert facts["shortcutTarget"].endswith("start.bat")


def test_the_entry_point_repair_touches_nothing_but_the_entry_points(work_root):
    facts = _run("install-repairs-launcher-with-scarce-disk", work_root)

    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["appUnchanged"] is True
    assert facts["interpreterUnchanged"] is True, "the runtime is not rebuilt"
    assert facts["stateUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["cacheKept"] is True
    assert facts["leftovers"] == []


def test_re_running_install_on_a_complete_installation_needs_no_disk_at_all(work_root):
    facts = _run("install-already-installed-with-scarce-disk", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["mode"] == "already-installed"
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["stateUnchanged"] is True


# ── a failed run never damages what is already installed ────────────────────


def test_a_failed_installation_leaves_a_complete_installation_byte_for_byte(work_root):
    """The state file is the promise that an installation can be started.

    A run that fails before it builds anything — here because there is no room
    for a second runtime — must not downgrade it to ``incomplete``."""
    facts = _run("install-failure-keeps-complete-state", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "DISK_SPACE_INSUFFICIENT"
    assert facts["stateUnchanged"] is True, "the state file came out as it went in"
    assert facts["stateStatus"] == "complete"
    assert facts["stateVersion"] == "1.0.0", "the installed version is still the installed one"


def test_a_failed_installation_leaves_the_program_the_runtime_and_the_data_alone(work_root):
    facts = _run("install-failure-keeps-complete-state", work_root)

    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0


def test_a_failure_before_anything_was_built_writes_no_state(work_root):
    """``incomplete`` is a promise to finish a build: without a build there is
    nothing to promise, so no state file appears at all."""
    assert _run("install-fresh-insufficient-disk", work_root)["stateExists"] is False
    assert _run("install-no-nvidia-driver", work_root)["stateExists"] is False


def test_the_partial_state_is_written_only_once_a_build_really_started(work_root):
    facts = _run("install-resumes-partial-runtime", work_root)

    assert facts["interpreterCreated"] is True, "this run really created a partial runtime"
    assert facts["stampAbsent"] is True, "an unfinished runtime is not stamped as ready"
    assert facts["partialStateStatus"] == "incomplete", "the next run must know to finish it"


def test_a_failed_entry_point_repair_puts_the_entry_points_back(work_root):
    """One launcher file cannot be replaced: all of them must be as they were.

    The repair swaps several files, so it is only complete when every one of
    them is in place; a failure halfway through must be undone."""
    facts = _run("install-repair-launcher-failure-keeps-everything", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "LAUNCHER_INSTALL_FAILED"
    assert facts["launchersUnchanged"] is True, "the swapped-in entry point was restored"
    assert facts["launchersPresent"] is True
    assert facts["leftovers"] == [], "no staged or backup copy survives"


def test_a_failed_entry_point_repair_changes_nothing_that_was_installed(work_root):
    facts = _run("install-repair-launcher-failure-keeps-everything", work_root)

    assert facts["stateUnchanged"] is True
    assert facts["stateStatus"] == "complete"
    assert facts["appUnchanged"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0


def test_an_unwritable_desktop_is_reported_without_failing_the_installation(work_root):
    facts = _run("install-repair-shortcut-failure-keeps-everything", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["shortcutCreated"] is False
    assert facts["shortcutWarned"] is True, "the operator is told the shortcut is missing"
    assert facts["launchersPresent"] is True
    assert facts["leftovers"] == []


def test_an_unwritable_desktop_leaves_the_installation_untouched(work_root):
    facts = _run("install-repair-shortcut-failure-keeps-everything", work_root)

    assert facts["stateUnchanged"] is True
    assert facts["stateStatus"] == "complete"
    assert facts["appUnchanged"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0


# ── short (8.3) paths name the same tree ────────────────────────────────────


def test_a_package_reached_through_a_short_path_installs_the_same_files(work_root):
    """Codex's second finding: the package root arrives as ``...\\ADMINI~1.DES\\``.

    The file system reports long paths for everything below it, so a relative
    path computed from the length of the short spelling cuts in the wrong
    place and silently drops every file."""
    facts = _run("install-from-short-package-path", work_root)
    if not facts["shortAvailable"]:
        pytest.skip("this volume does not hand out 8.3 short names")

    assert facts["shortIsShorter"] is True, "the test must really use a short path"
    assert facts["lockPayloadFiles"] == 3, "every payload file is registered in the lock"
    assert facts["ok"] is True, facts["errorCode"]
    assert facts["copiedFiles"] == 3
    assert facts["runtimeInstallCalls"] == 1


def test_a_package_reached_through_a_short_path_installs_a_working_installation(work_root):
    facts = _run("install-from-short-package-path", work_root)
    if not facts["shortAvailable"]:
        pytest.skip("this volume does not hand out 8.3 short names")

    assert facts["appCopied"] is True
    assert facts["mainMarker"] is True, "the payload copied is the payload shipped"
    assert facts["launchersAfter"] is True
    assert facts["shortcutExists"] is True
    assert facts["stateStatus"] == "complete"


def test_install_stops_before_downloading_when_no_nvidia_driver_is_present(work_root):
    facts = _run("install-no-nvidia-driver", work_root)

    assert facts["rejected"] is True
    assert facts["code"] == "NVIDIA_DRIVER_MISSING"
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["appCopied"] is False


# ── the first installation asks for a directory, or does not ────────────────


def test_a_first_installation_may_ask_the_operator_for_a_directory(work_root):
    """A double-clicked install.bat owns a console, so the prompt can be shown.

    The decision lives in the shared module, so this is the same code
    install.ps1 runs; install.bat only has to launch PowerShell in a mode that
    still permits Read-Host (asserted below)."""
    facts = _run("install-destination-prompt", work_root, Variant="console")

    assert facts["ask"] is True, "a console installation must be able to ask"
    assert facts["reason"] == "console"
    assert facts["waitsForInput"] is True
    assert facts["destination"] == ""


def test_the_prompt_never_waits_for_a_human_when_it_is_not_a_console(work_root):
    facts = _run("install-destination-prompt", work_root, Variant="input-redirected")

    assert facts["ask"] is False, "redirected input has no operator to answer"
    assert facts["reason"] == "input-redirected"
    assert facts["usesRecommended"] is True


def test_the_accept_recommended_switch_answers_instead_of_the_operator(work_root):
    facts = _run("install-destination-prompt", work_root, Variant="accept-recommended")

    assert facts["ask"] is False
    assert facts["reason"] == "accept-recommended"
    assert facts["usesRecommended"] is True


def test_the_non_interactive_variable_answers_instead_of_the_operator(work_root):
    facts = _run("install-destination-prompt", work_root, Variant="noninteractive")

    assert facts["ask"] is False
    assert facts["reason"] == "noninteractive"
    assert facts["usesRecommended"] is True


def test_an_explicit_directory_is_used_without_asking(work_root, tmp_path):
    target = tmp_path / "AutoTuneStudio"
    facts = _run("install-destination-prompt", work_root, Variant="requested",
                 InstallRoot=target)

    assert facts["ask"] is False
    assert facts["reason"] == "requested"
    assert facts["destination"] == str(target)


def test_a_second_run_reuses_the_recorded_directory_without_asking(work_root):
    facts = _run("install-destination-prompt", work_root, Variant="recorded")

    assert facts["ask"] is False, "the destination is chosen once, on the first installation"
    assert facts["reason"] == "recorded"
    assert facts["destination"] == facts["recordedRoot"]


def test_the_answer_the_prompt_would_not_give_is_a_real_directory(work_root):
    """What a non-interactive run installs into instead of asking."""
    facts = _run("install-destination-prompt", work_root, Variant="console")
    recommended = Path(facts["recommended"])

    assert recommended.is_absolute()
    assert recommended.name == "AutoTuneStudio"


def _entry_arguments(name: str) -> list[str]:
    """The switches a .bat hands to powershell.exe, without the script path."""
    for raw_line in (REPO_ROOT / "windows" / name).read_text(encoding="utf-8").splitlines():
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


def _read_host_probe(tmp_path: Path, arguments: list[str]) -> int:
    """Run a real Read-Host under the given switches; the exit code is the answer.

    This is the failure install.ps1 hits: the script sets
    ``$ErrorActionPreference = 'Stop'``, so a Read-Host its host refuses becomes a
    terminating error and the whole installation stops."""
    probe = tmp_path / "read-host-probe.ps1"
    probe.write_text("﻿$ErrorActionPreference = 'Stop'\ntry { [void](Read-Host 'dir') }\n"
                     "catch { exit 3 }\nexit 0\n", encoding="utf-8")
    result = subprocess.run(
        [_POWERSHELL, *arguments, "-File", str(probe)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120)
    return result.returncode


def test_the_install_entry_point_starts_powershell_able_to_prompt(tmp_path):
    arguments = _entry_arguments("install.bat")

    assert not any(a.lower() == "-noninteractive" for a in arguments), (
        "install.bat must not forbid Read-Host: install.ps1 asks for the install "
        "directory on a first installation, and PowerShell refuses Read-Host in "
        f"non-interactive mode (flags: {arguments})")
    assert _read_host_probe(tmp_path, arguments) == 0, (
        "the switches install.bat uses must allow Read-Host")


def test_the_probe_would_catch_a_read_host_forbidding_entry_point(tmp_path):
    """The negative control for the check above, so it cannot pass by accident."""
    arguments = _entry_arguments("install.bat") + ["-NonInteractive"]

    assert _read_host_probe(tmp_path, arguments) != 0


def test_a_non_interactive_installation_run_never_waits_for_input():
    """The real entry point, executed for real, with a destination it must refuse.

    A refused destination reaches the installer's own error path with no install
    work: it proves cmd.exe → install.bat → install.ps1 finishes without ever
    waiting for a human. The destination has to stay one ``Assert-InstallRootAllowed``
    refuses — an accepted one would install for real into the test machine."""
    import os

    environment = dict(os.environ)
    environment["AUTO_TUNE_NONINTERACTIVE"] = "1"
    environment["AUTO_TUNE_NO_PAUSE"] = "1"
    # An artifact of this test host, not of the product: it makes cmd.exe refuse
    # to run a program named by a bare relative path, which is exactly how a
    # double-clicked .bat is reached.
    environment.pop("NoDefaultCurrentDirectoryInExePath", None)
    # The absolute path is how Explorer hands a double-clicked file to cmd.exe.
    result = subprocess.run(
        [os.environ.get("ComSpec", "cmd.exe"), "/c", str(REPO_ROOT / "windows" / "install.bat"),
         "-InstallRoot", "AutoTuneStudio"],
        env=environment, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)

    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 1, output
    assert "INSTALL_ROOT_INVALID" in output, output


# ── the offline bundle: the only source of a runtime ────────────────────────


def test_the_installation_builds_the_runtime_from_the_package(work_root):
    """No downloader is left: the pinned Miniconda installer travels in the ZIP."""
    facts = _run("offline-install", work_root)

    assert facts["fetchCalls"] == 0, "nothing may go through the network"
    assert facts["networkish"] == []
    assert facts["interpreterPrivate"] is True


def test_a_complete_installation_is_not_rebuilt_by_a_second_run(work_root):
    facts = _run("offline-install", work_root)

    assert facts["secondMode"] == "already-installed"
    assert facts["secondRuns"] == 0
    assert facts["interpreterUnchanged"] is True


# ── logs ────────────────────────────────────────────────────────────────────


def test_install_log_redacts_secrets_paths_and_stack_traces(work_root):
    facts = _run("install-log-redaction", work_root)

    assert facts["failed"] is True
    assert facts["logExists"] is True
    assert facts["forbidden"] == [], "the log leaked a secret, path or stack trace"
    assert facts["hasRedaction"] is True
    assert facts["mentionsCode"] is True
