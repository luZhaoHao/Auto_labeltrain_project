"""F1.2-D: start, upgrade and uninstall behaviour of the Windows delivery.

Same harness as the installer suite: the shipping PowerShell module is imported
and only the process runner, the probes and the browser opener are replaced. The
upgrade and uninstall suites therefore exercise the real staging, rollback and
safety logic in a controlled temporary tree, never the operator's data.

These tests need ``powershell.exe``; the whole module is skipped otherwise.
"""

from __future__ import annotations

import json
import shutil
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
    return tmp_path_factory.mktemp("f12d-lifecycle")


def _run(scenario: str, work_root: Path, **extra) -> dict:
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


# ── start ───────────────────────────────────────────────────────────────────


def test_start_refuses_an_incomplete_installation(work_root):
    facts = _run("start-incomplete-state", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "INSTALL_STATE_INCOMPLETE"
    assert facts["launches"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["browserOpens"] == 0


def test_start_runs_the_gpu_preflight_before_the_studio(work_root):
    facts = _run("start-happy-path", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["order"][0] == "preflight"
    assert "launch-service" in facts["order"]
    assert facts["order"].index("preflight") < facts["order"].index("launch-service")
    assert "--require-gpu" in facts["preflightArgs"]


def test_start_uses_the_private_interpreter_and_the_data_directory(work_root):
    facts = _run("start-happy-path", work_root)
    app_root = Path(facts["launchFile"]).parents[2]

    assert facts["launchFile"] == facts["preflightPython"]
    assert str(app_root).endswith(str(Path("AutoTuneStudio")))
    assert facts["launchCwd"] == str(app_root / "data")
    assert facts["launchArgs"][-2:] == ["-m", "auto_tune.main"]


def test_start_opens_the_browser_only_after_healthz_answers(work_root):
    facts = _run("start-happy-path", work_root)

    assert facts["order"].index("healthz") < facts["order"].index("browser")
    assert facts["browserOpens"] == 1
    assert facts["browserUrl"].startswith("http://127.0.0.1:8000")
    assert facts["instanceExists"] is True


def test_start_never_launches_the_service_without_a_gpu(work_root):
    facts = _run("start-no-gpu", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "PREFLIGHT_FAILED"
    assert facts["launches"] == 0
    assert facts["healthProbes"] == 0
    assert facts["browserOpens"] == 0
    assert "GPU" in facts["message"]


@pytest.mark.parametrize("value", ["abc", "0", "70000", "-1", "65536", "80 80"])
def test_start_rejects_an_invalid_port_before_anything_else(work_root, value):
    facts = _run("start-port-invalid", work_root, EnvPort=value)

    assert facts["ok"] is False
    assert facts["errorCode"] == "PORT_INVALID"
    assert facts["launches"] == 0
    assert facts["preflightCalls"] == 0


def test_start_reports_an_occupied_port_instead_of_starting_a_second_server(work_root):
    facts = _run("start-port-in-use", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "PORT_IN_USE"
    assert facts["launches"] == 0
    assert facts["preflightCalls"] == 0


def test_starting_again_only_reopens_the_browser(work_root):
    facts = _run("start-second-instance", work_root)

    assert facts["ok"] is True
    assert facts["alreadyRunning"] is True
    assert facts["launches"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["browserOpens"] == 1


def test_port_comes_from_the_state_file_and_the_environment_wins(work_root):
    facts = _run("start-port-from-state", work_root)

    assert facts["fromState"] == 8123
    assert facts["fromEnv"] == 9000


# ── start: a service that never answers is not a success ────────────────────


def test_a_health_check_timeout_is_reported_as_a_failure(work_root):
    """The process starting is not the same as the product working."""
    facts = _run("start-health-timeout", work_root)

    assert facts["launches"] == 1, "the service was started exactly once"
    assert facts["healthProbes"] > 0, "the health endpoint was really polled"
    assert facts["ok"] is False
    assert facts["errorCode"] == "HEALTH_CHECK_FAILED"
    assert facts["healthy"] is False


def test_a_health_check_timeout_terminates_the_process_it_just_started(work_root):
    facts = _run("start-health-timeout", work_root)

    assert facts["processAliveAfter"] is False, "the unhealthy service must not survive"


def test_a_health_check_timeout_leaves_no_instance_record(work_root):
    facts = _run("start-health-timeout", work_root)

    assert facts["instanceExists"] is False, "studio.json must not describe a dead service"


def test_a_health_check_timeout_logs_one_fixed_redacted_line(work_root):
    facts = _run("start-health-timeout", work_root)

    assert facts["logMentionsCode"] is True
    assert facts["logMentionsVolatile"] is False, "the log must not carry a pid or a log path"


def test_a_health_check_timeout_never_opens_a_browser(work_root):
    facts = _run("start-health-timeout", work_root)

    assert facts["browserOpens"] == 0


def test_a_health_check_failure_does_not_touch_a_healthy_existing_instance(work_root):
    facts = _run("start-health-timeout-existing-instance", work_root)

    assert facts["ok"] is True
    assert facts["alreadyRunning"] is True
    assert facts["launches"] == 0
    assert facts["existingAlive"] is True, "the instance running before the launch is left alone"
    assert facts["instanceKept"] is True
    assert facts["browserOpens"] == 1


# ── upgrade ─────────────────────────────────────────────────────────────────


def test_upgrade_replaces_the_program_and_keeps_every_user_file(work_root):
    facts = _run("upgrade-preserves-data", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["versionBefore"] == "1.0.0"
    assert facts["versionAfter"] == "1.0.1"
    assert facts["appMarkerChanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["historyKept"] is True
    assert facts["cacheKept"] is True
    assert facts["stateHistory"] == 1


def test_upgrade_leaves_the_runtime_alone_when_the_lock_did_not_change(work_root):
    facts = _run("upgrade-preserves-data", work_root)

    assert facts["runtimeUntouched"] is True
    assert facts["runtimeInstallCalls"] == 1, "only the first install built the runtime"


def test_upgrade_updates_the_runtime_when_the_dependency_lock_changes(work_root):
    facts = _run("upgrade-runtime-lock-change", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["runtimeUpdated"] is True
    assert facts["runtimeInstalls"] == 1


def test_a_dependency_lock_change_is_refused_before_anything_is_replaced(work_root):
    """A changed lock builds a second runtime: it needs the full budget.

    The refusal must come before the download, the staged runtime and the swap,
    so a machine with 1 GiB free is told so instead of being left half
    upgraded."""
    facts = _run("upgrade-lock-change-insufficient-disk", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "DISK_SPACE_INSUFFICIENT"
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["preflightCalls"] == 0
    assert facts["leftovers"] == [], "no staging or rollback directory is created"


def test_a_refused_dependency_upgrade_leaves_the_installed_version_untouched(work_root):
    facts = _run("upgrade-lock-change-insufficient-disk", work_root)

    assert facts["stateVersion"] == "1.0.0"
    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True, "the previous version is still the installed one"
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["stampUnchanged"] is True
    assert facts["stateUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True


def test_a_program_only_upgrade_runs_without_the_fresh_installation_budget(work_root):
    """An unchanged dependency lock means no runtime is built, so the disk
    budget for a fresh installation must not be required."""
    facts = _run("upgrade-code-only-insufficient-disk", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["version"] == "1.0.1"
    assert facts["appMarker"] is True
    assert facts["runtimeUpdated"] is False
    assert facts["downloadCalls"] == 0
    assert facts["runtimeInstallCalls"] == 0
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["leftovers"] == []


def test_upgrade_refuses_a_package_that_fails_its_integrity_check(work_root):
    facts = _run("upgrade-verify-failure", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] in ("PACKAGE_HASH_MISMATCH", "UPGRADE_VERIFY_FAILED")
    assert facts["appUnchanged"] is True, "the previous version is still the installed one"
    assert facts["appRunnable"] is True
    assert facts["stateVersion"] == "1.0.0"


def test_upgrade_refreshes_the_installed_launcher(work_root):
    facts = _run("upgrade-updates-launcher", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["startBatRestored"] is True, "a stale start.bat is replaced"
    assert facts["uninstallRestored"] is True
    assert facts["modulePresent"] is True
    assert facts["version"] == "1.0.1"


def test_upgrade_refreshes_the_launcher_without_touching_user_data(work_root):
    facts = _run("upgrade-updates-launcher", work_root)

    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["shortcutExists"] is True
    assert facts["shortcutTarget"].endswith("start.bat")


# ── upgrade: a changed dependency lock is built beside the live runtime ─────


def test_upgrade_builds_a_new_runtime_off_to_the_side(work_root):
    """The staged runtime is preflighted while the live one is still intact."""
    facts = _run("upgrade-runtime-staged-success", work_root)

    assert facts["ok"] is True, facts["errorCode"]
    assert facts["stagedPreflightOutsideLiveRuntime"] is True, (
        "the new runtime is preflighted at its staging path")
    assert facts["stagedPreflightLiveInterpreterHash"] == facts["interpreterHashBefore"], (
        "the runtime in use must not be modified in place")
    assert ".staging-" in facts["stagedPreflightEnvPythonPath"].replace("/", "\\"), (
        "the preflight must import the staged program, not the installed one")


def test_upgrade_switches_to_the_new_runtime_only_after_it_verified(work_root):
    facts = _run("upgrade-runtime-staged-success", work_root)

    assert facts["runtimeUpdated"] is True
    assert facts["version"] == "1.0.1"
    assert facts["interpreterPath"].endswith("runtime\\py310\\python.exe")
    assert facts["interpreterPresent"] is True
    assert facts["finalPreflightPython"] == facts["interpreterPath"], (
        "the installed program is verified with the runtime it will use")


def test_upgrade_records_the_new_runtime_identity(work_root):
    facts = _run("upgrade-runtime-staged-success", work_root)

    assert facts["runtimeKeyAfter"] != facts["runtimeKeyBefore"]
    assert facts["stampHashAfter"] != facts["stampHashBefore"]
    assert facts["stateHashAfter"] != facts["stateHashBefore"]


# ── upgrade: the offline bundle is part of the runtime identity ─────────────


def test_a_new_offline_bundle_with_the_same_requirements_is_a_new_runtime(work_root):
    """The requirements file is the *pinned versions*; the bundle is the bytes.

    A package whose wheels were legitimately replaced keeps the same
    requirements and still asks for a different runtime, so the identity has to
    cover the verified offline lock — not only the requirements hash."""
    facts = _run("upgrade-offline-wheel-refresh", work_root)

    assert facts["requirementsUnchanged"] is True, "the fixture must not change the pins"
    assert facts["offlineLockChanged"] is True, "the bundle really changed"
    assert facts["ok"] is True, facts["errorCode"]
    assert facts["runtimeUpdated"] is True
    assert facts["runtimeInstalls"] == 1
    assert facts["runtimeKeyAfter"] != facts["runtimeKeyBefore"]
    assert facts["version"] == "1.0.1"
    assert facts["appMarker"] is True


def test_the_new_runtime_for_the_bundle_is_built_beside_the_live_one(work_root):
    facts = _run("upgrade-offline-wheel-refresh", work_root)

    assert facts["stagedOutsideLiveRuntime"] is True, (
        "the replacement runtime must be staged, never built over the live one")
    assert facts["stagedPreflightLiveInterpreterHash"] == facts["interpreterBefore"], (
        "the runtime in use is untouched while its replacement is built")
    assert facts["leftovers"] == []
    assert facts["interpreterUnchanged"] is False, "the new runtime is the installed one now"
    assert facts["interpreterPresent"] is True
    assert facts["preflightCount"] == 2, (
        "the staged pair is preflighted before the swap and the installed pair after it")
    assert facts["fetchCalls"] == 0


def test_a_failed_bundle_refresh_keeps_the_previous_version_working(work_root):
    facts = _run("upgrade-offline-wheel-refresh", work_root, Variant="fail")

    assert facts["ok"] is False
    assert facts["errorCode"] == "OFFLINE_RUNTIME_INSTALL_FAILED"
    assert facts["runtimeUpdated"] is False
    assert facts["stateVersion"] == "1.0.0"
    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["stampUnchanged"] is True
    assert facts["stateUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["leftovers"] == []


def test_a_failed_bundle_refresh_never_touches_the_live_runtime(work_root):
    facts = _run("upgrade-offline-wheel-refresh", work_root, Variant="fail")

    assert facts["stagedOutsideLiveRuntime"] is True
    assert facts["runtimeInstalls"] == 1, "only the staged build was attempted"
    assert facts["interpreterUnchanged"] is True


# ── upgrade: a stamp from an older delivery ─────────────────────────────────


@pytest.mark.parametrize("variant", ["legacy-key", "missing-key", "unknown-schema"])
def test_an_unrecognised_runtime_stamp_is_rebuilt_instead_of_raising(work_root, variant):
    """An older stamp has no offline-bundle identity — or no key at all.

    It is simply not a match: the dependencies are installed into the existing
    interpreter again and a current stamp is written. No missing field may reach
    the operator as an unhandled error."""
    facts = _run("runtime-stamp-legacy", work_root, Variant=variant)

    assert facts["ok"] is True, f"{facts['errorCode']}: {facts['errorMessage']}"
    assert facts["errorCode"] is None
    assert facts["stampParses"] is True
    assert facts["transferMode"] == "dependency-update"
    assert facts["runtimeInstalls"] == 1
    assert facts["refreshOnlyCount"] == 1, "the interpreter is reused, only the deps are re-installed"
    assert facts["interpreterPresent"] is True
    assert facts["keyChanged"] is True
    assert facts["stampHasOfflineIdentity"] is True, (
        "the stamp records the identity an upgrade will compare against")


def test_upgrade_cleans_up_its_staging_and_old_runtime_directories(work_root):
    facts = _run("upgrade-runtime-staged-success", work_root)

    assert facts["leftovers"] == []
    assert facts["oldRuntimeLeftover"] == []
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["appMarker"] is True


def test_a_failed_dependency_install_changes_nothing_that_was_installed(work_root):
    facts = _run("upgrade-runtime-build-failure", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "OFFLINE_RUNTIME_INSTALL_FAILED"
    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True


def test_a_failed_dependency_install_keeps_the_runtime_identity_and_state(work_root):
    facts = _run("upgrade-runtime-build-failure", work_root)

    assert facts["stampUnchanged"] is True, "the old runtime identity must not move"
    assert facts["stateUnchanged"] is True
    assert facts["stateVersion"] == "1.0.0"
    assert facts["leftovers"] == [], "the staged runtime and app are cleaned up"
    assert facts["dataKept"] is True


def test_an_offline_precheck_failure_rolls_the_staged_upgrade_back(work_root):
    """A staged runtime the private interpreter cannot certify is not switched in.

    The staged runtime is built beside the live one and preflighted while the
    installed pair is still untouched; a mismatch has to put everything back.
    """
    facts = _run("upgrade-offline-precheck-failure", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "OFFLINE_RUNTIME_INSTALL_FAILED"
    assert facts["offlineChecks"] == 1, "the staged interpreter really was asked"
    assert facts["pipCommands"], "the offline pip steps ran before the check"
    assert all("--no-index" in command for command in facts["pipCommands"])
    assert facts["fetchCalls"] == 0


def test_a_rolled_back_offline_precheck_leaves_the_installed_version_intact(work_root):
    facts = _run("upgrade-offline-precheck-failure", work_root)

    assert facts["stateVersion"] == "1.0.0"
    assert facts["stateUnchanged"] is True
    assert facts["appUnchanged"] is True
    assert facts["appRunnable"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True
    assert facts["stampUnchanged"] is True
    assert facts["configKept"] is True
    assert facts["sqliteKept"] is True
    assert facts["leftovers"] == []


def test_a_failed_runtime_preflight_changes_nothing_that_was_installed(work_root):
    facts = _run("upgrade-runtime-preflight-failure", work_root)

    assert facts["ok"] is False
    assert facts["errorCode"] == "PREFLIGHT_FAILED"
    assert facts["appUnchanged"] is True
    assert facts["interpreterUnchanged"] is True
    assert facts["interpreterPresent"] is True


def test_a_failed_runtime_preflight_keeps_the_runtime_identity_and_state(work_root):
    facts = _run("upgrade-runtime-preflight-failure", work_root)

    assert facts["stampUnchanged"] is True
    assert facts["stateUnchanged"] is True
    assert facts["stateVersion"] == "1.0.0"
    assert facts["leftovers"] == []
    assert facts["configKept"] is True


# ── uninstall ───────────────────────────────────────────────────────────────


def test_uninstall_removes_the_program_and_keeps_the_data(work_root):
    facts = _run("uninstall-keeps-data", work_root)

    assert facts["ok"] is True
    assert facts["appRemoved"] is True
    assert facts["runtimeRemoved"] is True
    assert facts["dataKept"] is True
    assert facts["dataFiles"] > 0
    assert facts["configKept"] is True
    assert facts["stateRemoved"] is True
    assert facts["noteMentionsData"] is True
    assert facts["message"]


def test_uninstall_deletes_data_only_after_explicit_confirmation(work_root):
    facts = _run("uninstall-remove-data-requires-confirm", work_root)

    assert facts["firstOk"] is False
    assert facts["firstError"] == "CONFIRMATION_REQUIRED"
    assert facts["dataAfterBlock"] is True
    assert facts["secondOk"] is True
    assert facts["dataRemoved"] is True
    assert facts["dataAfterConfirm"] is False


def test_uninstall_refuses_dangerous_removal_targets(work_root):
    facts = _run("uninstall-safe-targets", work_root)
    results = dict(row.split(":", 1) for row in facts["results"])

    for name in ("empty", "drive-root", "user-profile", "localappdata-root",
                 "install-root", "outside-install", "repo-root"):
        assert results[name] == "UNSAFE_REMOVAL_TARGET", f"{name} was accepted"
    assert facts["dataAccepted"] is True, "the controlled data directory is removable"


def test_uninstall_removes_the_launcher_and_the_desktop_shortcut(work_root):
    facts = _run("uninstall-removes-launcher-and-shortcut", work_root)

    assert facts["ok"] is True
    assert facts["startBatRemoved"] is True
    assert facts["startScriptRemoved"] is True
    assert facts["libRemoved"] is True
    assert facts["shortcutRemoved"] is True


def test_uninstall_keeps_the_data_the_cache_and_a_way_to_run_again(work_root):
    facts = _run("uninstall-removes-launcher-and-shortcut", work_root)

    assert facts["dataKept"] is True
    assert facts["configKept"] is True
    assert facts["cacheKept"] is True
    assert facts["noteExists"] is True


def test_uninstall_never_deletes_the_entry_point_it_is_running_from(work_root):
    """cmd.exe re-reads a running .bat: deleting it inline is a half uninstall."""
    facts = _run("uninstall-removes-launcher-and-shortcut", work_root)

    assert facts["uninstallBatKept"] is True
    assert facts["uninstallScriptKept"] is True
    assert facts["deferredCount"] == 1
    paths = [str(path).replace("/", "\\").lower() for path in facts["deferredPaths"]]
    assert any(path.endswith("uninstall.bat") for path in paths)
    assert any(path.endswith("uninstall.ps1") for path in paths)
    assert len(paths) == 2


def test_the_installed_uninstall_entry_works_without_the_package(work_root):
    """The whole removal runs from the install root, with the ZIP deleted."""
    facts = _run("uninstall-entry-runs-from-install-root", work_root)

    assert facts["packageGone"] is True
    assert facts["exitCode"] == 0, facts["output"]
    assert facts["appRemoved"] is True
    assert facts["runtimeRemoved"] is True
    assert facts["startBatRemoved"] is True
    assert facts["libRemoved"] is True


def test_the_installed_uninstall_entry_deletes_itself_after_it_finished(work_root):
    facts = _run("uninstall-entry-runs-from-install-root", work_root)

    assert facts["uninstallBatRemoved"] is True, "the launcher must not survive its own uninstall"
    assert facts["uninstallScriptRemoved"] is True


def test_the_installed_uninstall_entry_keeps_the_user_data_and_the_cache(work_root):
    facts = _run("uninstall-entry-runs-from-install-root", work_root)

    assert facts["dataKept"] is True
    assert facts["configKept"] is True
    assert facts["cacheKept"] is True
    assert facts["noteExists"] is True
    assert facts["unrelatedShortcutKept"] is True, (
        "a shortcut that does not point at this installation is left alone")
