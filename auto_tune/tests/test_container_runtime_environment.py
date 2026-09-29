"""F1.2-D follow-up: the runtime environment the container hands over.

The container start drops to ``studio`` (10001:10001) but the environment it
inherits still carries ``HOME=/root`` from the base image. Ultralytics resolves
its settings directory through ``YOLO_CONFIG_DIR`` and only then through
``$HOME/.config``, so the first real 1-epoch training started through
``POST /api/training/start`` in a real container died with

    PermissionError: [Errno 13] Permission denied: '/root/.config/Ultralytics'

``docker/entrypoint.sh`` is the one place that fixes the runtime environment
before the hand-over. Both values are *assigned* rather than defaulted, so a
preset ``HOME`` or ``YOLO_CONFIG_DIR`` cannot reach a business process, and the
directory is derived from the configuration path the start-up whitelist already
governs, so nothing outside the declared container paths can be named. The
tests below read the shipped script and — where a POSIX shell is available — run
it with a stub ``python``, so the environment the initialisation module really
receives is observed instead of inferred. On Windows that shell has to be Git
for Windows' own bash: a bare ``bash`` on ``PATH`` is the WSL launcher there,
which filters the caller's environment and cannot read a Windows path argument,
so it would neither run this script nor report anything about it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"

# The unprivileged account's home, created by the Dockerfile's
# ``useradd --create-home``. It is where the drop lands and where nothing but
# the Studio writes.
STUDIO_HOME = "/home/studio"
# The declared configuration directory: the parent of the default configuration
# path, and the directory the Studio's own state (including Ultralytics
# settings) is written into.
CONTROLLED_CONFIG_DIR = "/data/config"

# An unconditional POSIX assignment. The ``${NAME:-...}`` form is deliberately
# not matched by a caller of :func:`_fixed_assignment`: that form lets the
# inherited value win, which is exactly what must not happen here.
_ASSIGNMENT = re.compile(
    r'^export (?P<name>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>[^"]*)"[ \t]*$',
    re.MULTILINE)


def _entrypoint_text() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def _fixed_assignment(name: str) -> str | None:
    """The value the entrypoint always exports for ``name``.

    ``None`` when the variable is not assigned unconditionally — either absent,
    or only given an environment-overridable default.
    """
    assigned = None
    for match in _ASSIGNMENT.finditer(_entrypoint_text()):
        if match.group("name") == name:
            assigned = match.group("value")
    return assigned


def _falls_back_to_the_environment(name: str) -> bool:
    """Whether the script ever lets an inherited value for ``name`` win."""
    return f"${{{name}:-" in _entrypoint_text()


# ── the fixed runtime environment the shell declares ─────────────────────────


def test_the_entrypoint_fixes_home_for_the_dropped_studio_account():
    """``HOME`` must be the unprivileged account's home, not the image default.

    Ultralytics writes its settings on the first training start, falling back to
    ``$HOME/.config``: inheriting ``/root`` while running as ``studio`` is what
    turned that first start into a ``PermissionError``.
    """
    assert _fixed_assignment("HOME") == STUDIO_HOME
    assert not _falls_back_to_the_environment("HOME"), (
        "an inherited HOME must not be able to win over the fixed one")


def test_the_entrypoint_fixes_yolo_config_dir_inside_the_controlled_directory():
    assert _fixed_assignment("YOLO_CONFIG_DIR") == "${AUTO_TUNE_CONFIG_DIR}"
    assert not _falls_back_to_the_environment("YOLO_CONFIG_DIR"), (
        "a preset YOLO_CONFIG_DIR must not be able to point outside the "
        "controlled configuration directory")


def test_the_controlled_configuration_directory_follows_the_controlled_config_path():
    """The directory cannot be widened from the environment: it is the parent of
    the configuration path the start-up whitelist already refuses outside the
    declared container layout."""
    declared = _fixed_assignment("AUTO_TUNE_CONFIG_DIR")

    assert declared, "the entrypoint must declare the controlled config directory"
    assert declared.startswith("${AUTO_TUNE_CONFIG_PATH"), declared
    assert not _falls_back_to_the_environment("AUTO_TUNE_CONFIG_DIR"), (
        "the controlled directory must not be settable from the environment")


# ── the environment the hand-over really receives ────────────────────────────


def _is_wsl_launcher(path: Path) -> bool:
    """Whether ``path`` is the Windows launcher that redirects into WSL.

    ``C:\\Windows\\System32\\bash.exe`` and the ``WindowsApps`` shim start a Linux
    distribution instead of running a shell here: its environment is filtered
    through ``WSLENV`` (so a caller's variables never arrive) and a Windows path
    argument is not resolvable. Neither is a POSIX shell for this suite.
    """
    parts = [part.lower() for part in path.parts]
    return "system32" in parts or "windowsapps" in parts


def _git_bash() -> Path | None:
    """Git for Windows' own bash, or ``None`` when it is not installed.

    ``shutil.which("bash")`` is deliberately not consulted on Windows: it answers
    with whatever comes first on ``PATH``, and on a machine with WSL installed
    that is the WSL launcher. Git for Windows ships its bash inside its own
    installation, so the interpreter this repository's shell scripts are meant to
    run under is derived from the git that is actually installed, and every
    candidate has to exist on disk.
    """
    candidates: list[Path] = []

    git = shutil.which("git")
    if git:
        # <root>\mingw64\bin\git.exe, <root>\cmd\git.exe and <root>\bin\git.exe
        # are all real layouts, so the installation root is whichever ancestor
        # actually carries a bash.
        for ancestor in Path(git).resolve().parents:
            candidates.append(ancestor / "bin" / "bash.exe")
            candidates.append(ancestor / "usr" / "bin" / "bash.exe")

    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if root:
            candidates.append(Path(root) / "Git" / "bin" / "bash.exe")
            candidates.append(Path(root) / "Git" / "usr" / "bin" / "bash.exe")

    for candidate in candidates:
        if candidate.name.lower() != "bash.exe" or _is_wsl_launcher(candidate):
            continue
        if candidate.is_file():
            return candidate
    return None


def _posix_shell() -> Path | None:
    """The shell the shipped entrypoint is executed with, or ``None``.

    Off Windows any ``bash`` on ``PATH`` is a real POSIX shell. On Windows only
    Git for Windows' bash is accepted, so a WSL alias can never be mistaken for
    the interpreter the delivery script runs under.
    """
    if os.name == "nt":
        return _git_bash()
    found = shutil.which("bash")
    return Path(found) if found else None


@pytest.fixture(scope="module")
def posix_shell() -> Path:
    """The shell the shipped entrypoint is really interpreted by.

    Without one the entrypoint cannot be run, and the contract is then proven by
    the script tests alone rather than by a shell that would misread it.
    """
    shell = _posix_shell()
    if shell is None:
        pytest.skip("Git for Windows' bash was not found: the entrypoint cannot "
                    "be run as a POSIX shell")
    return shell


def _shell_path(path) -> str:
    """A path in the form the selected shell resolves.

    Git Bash names the drives ``/c``, ``/d``, ...: handing it ``E:\\a\\b`` either
    loses the separators or is read as a literal file name — the same class of
    failure these tests exist to catch. Everywhere else the path is already
    POSIX.
    """
    text = os.fspath(path)
    if os.name != "nt":
        return text
    drive, rest = os.path.splitdrive(os.path.abspath(text))
    if not drive:
        return text.replace("\\", "/")
    return "/" + drive.rstrip(":").lower() + rest.replace("\\", "/")


def _hand_over(tmp_path: Path, shell: Path, preset: dict | None = None) -> dict:
    """Run the shipped entrypoint with a stub ``python`` and read what it saw.

    The shell hands over with ``exec python -m ...``; the stub stands in for
    that interpreter and reports the variables the training subprocess reads, so
    a missing or leaked value shows up as an observed fact. ``APP_ROOT`` is the
    filesystem root — the entrypoint changes into it before the hand-over, and
    it exists on every host the suite runs on.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_bytes(
        b"#!/bin/sh\n"
        b'printf "HOME=%s\\n" "${HOME-}"\n'
        b'printf "AUTO_TUNE_CONFIG_DIR=%s\\n" "${AUTO_TUNE_CONFIG_DIR-}"\n'
        b'printf "YOLO_CONFIG_DIR=%s\\n" "${YOLO_CONFIG_DIR-}"\n'
    )
    stub.chmod(0o755)

    environment = dict(os.environ)
    environment["APP_ROOT"] = "/"
    # The shell reads its own search path, so the stub directory and every
    # inherited entry are spelled the way that shell resolves them; the stub
    # comes first, so the hand-over reaches it instead of a system interpreter.
    environment["PATH"] = ":".join(_shell_path(entry) for entry in (
        os.fspath(stub_dir), *environment.get("PATH", "").split(os.pathsep)
    ) if str(entry).strip())
    # The operator's own values, applied last so they are the ones a leak would
    # carry: the entrypoint has to overwrite them, not defer to them.
    environment.update(preset or {})

    completed = subprocess.run([os.fspath(shell), _shell_path(ENTRYPOINT)],
                               env=environment, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")

    observed: dict = {}
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        name, separator, value = line.partition("=")
        if separator:
            observed[name] = value
    return observed


def test_the_hand_over_receives_the_studio_home(tmp_path, posix_shell):
    observed = _hand_over(tmp_path, posix_shell, preset={"HOME": "/root"})

    assert observed.get("HOME") == STUDIO_HOME, observed


def test_the_hand_over_receives_the_controlled_config_directory(tmp_path, posix_shell):
    observed = _hand_over(tmp_path, posix_shell,
                          preset={"AUTO_TUNE_CONFIG_DIR": "/etc"})

    assert observed.get("AUTO_TUNE_CONFIG_DIR") == CONTROLLED_CONFIG_DIR, observed


def test_a_preset_yolo_config_dir_cannot_bypass_the_controlled_directory(
        tmp_path, posix_shell):
    observed = _hand_over(tmp_path, posix_shell, preset={
        "AUTO_TUNE_CONFIG_DIR": "/etc",
        "YOLO_CONFIG_DIR": "/etc",
    })

    assert observed.get("YOLO_CONFIG_DIR") == CONTROLLED_CONFIG_DIR, observed
