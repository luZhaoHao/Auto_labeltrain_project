"""F1.2-D: the Windows delivery package the operator unzips.

``build_zip.ps1`` is executed for real over a repository that carries the real
delivery scripts, the real program tree and a complete offline bundle (with
stand-in files for the three downloads, so no test has to move 2.4 GB), and that
deliberately also contains every kind of file the package must never ship. The
assertions are about the archive's contents — including the integrity lock that
covers the offline bundle and the installer verifies before it copies anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWS = REPO_ROOT / "windows"
BUILD = WINDOWS / "build_zip.ps1"
MODULE = WINDOWS / "lib" / "AutoTuneDelivery.psm1"
HARNESS = Path(__file__).resolve().parent / "ps" / "windows_delivery_harness.ps1"

_POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(_POWERSHELL is None,
                                reason="Windows PowerShell is not available")

VERSION = "1.0.0"

# What the offline bundle must contribute to the archive.
OFFLINE_ENTRIES = (
    "offline/miniconda/Miniconda3-test-Windows-x86_64.exe",
    "offline/wheelhouse/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl",
    "offline/wheelhouse/torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl",
    "offline/wheelhouse/fastapi-0.139.2-py3-none-any.whl",
    "offline/wheelhouse/uvicorn-0.51.0-py3-none-any.whl",
    "offline/offline-lock.json",
)

# Everything a delivery ZIP must never contain, as it would otherwise appear
# in the archive: (relative repo path, kind).
FORBIDDEN_SENTINELS = {
    ".git/config": "git",
    ".git/HEAD": "git",
    ".env": "credential file",
    ".env.local": "credential file",
    "auto_tune/config.yaml": "real configuration",
    "auto_tune/tests/test_hpo_api.py": "tests",
    "auto_tune/scripts/verify_hpo_execution.py": "verification entry",
    "auto_tune/__pycache__/main.cpython-310.pyc": "python cache",
    "log/auto_tune.db": "sqlite index",
    "log/db_backups/auto_tune.db.bak": "sqlite backup",
    "log/tuning_audit_session.json": "audit run file",
    "detect/train63/weights/best.pt": "training artifact",
    "models/weights/uploaded.pt": "weight",
    "models/weights/exported.onnx": "model",
    "runs/detect/train/args.yaml": "training run",
    "docker-data/config/config.yaml": "container runtime data",
    "docker-data/secrets/credentials.json": "container credential file",
    "dataset_demo/part_1.jpg": "dataset",
    "secrets.json": "credential file",
    "credentials.json": "credential file",
    ".pytest_cache/v/cache/nodeids": "test cache",
    "build_output/previous.zip": "previous build output",
    "Dockerfile": "docker delivery",
    "compose.yaml": "docker delivery",
    "docker/entrypoint.sh": "docker delivery",
    "requirements.txt": "unverified dependency list",
    "environment.yml": "development environment",
}

# The sanitized program, which must be there.
REQUIRED_ENTRIES = (
    "install.bat",
    "start.bat",
    "upgrade.bat",
    "uninstall.bat",
    "install.ps1",
    "start.ps1",
    "upgrade.ps1",
    "uninstall.ps1",
    "lib/AutoTuneDelivery.psm1",
    "package-manifest.json",
    "package-manifest.lock.json",
    "requirements-windows.lock.txt",
    "payload/auto_tune/main.py",
    "payload/auto_tune/config.template.yaml",
    "payload/auto_tune/delivery/preflight.py",
)


def _run_harness(scenario: str, work_root: Path, **extra) -> dict:
    """Execute one harness scenario (the harness owns the synthetic bundle)."""
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
    assert lines, f"the harness produced no result: {result.stdout!r}"
    payload = json.loads(lines[-1][len("##RESULT## "):])
    assert payload.get("ok"), f"{scenario} could not run: {payload}"
    return payload["facts"]


def _fixture_repo(root: Path) -> Path:
    """A repository that contains everything, including what must not ship.

    The program tree is the *real* one, so the sanitizing rules are exercised on
    the files the product actually has; the delivery scripts and the offline
    bundle come from the harness, which builds them with the shipping generators
    (the three downloaded files are stand-ins of the same name).
    """
    shutil.copytree(REPO_ROOT / "auto_tune", root / "auto_tune",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative, _kind in FORBIDDEN_SENTINELS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sentinel")
    _run_harness("zip-fixture-source", root, TargetRoot=root)
    return root


def _build(repo_root: Path, output_dir: Path,
           version: str | None = VERSION) -> subprocess.CompletedProcess:
    command = [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", str(BUILD), "-RepoRoot", str(repo_root),
               "-OutputDir", str(output_dir)]
    if version is not None:
        command += ["-Version", version]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)


def _package(output_dir: Path, version: str = VERSION) -> Path:
    return output_dir / f"AutoTuneStudio-Setup-{version}.zip"


@pytest.fixture(scope="module")
def fixture_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("f12d-repo")
    output = tmp_path_factory.mktemp("f12d-out")
    _fixture_repo(root)
    result = _build(root, output)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    assert _package(output).is_file()
    with zipfile.ZipFile(_package(output)) as archive:
        names = archive.namelist()
        lock = json.loads(archive.read("package-manifest.lock.json").decode("utf-8"))
        payload_hashes = {
            entry["path"]: hashlib.sha256(archive.read(entry["path"])).hexdigest()
            for entry in lock["files"] if entry["path"].startswith("payload/")
        }
    return {"root": root, "output": output, "names": names, "lock": lock,
            "payload_hashes": payload_hashes, "result": result}


# ── what the package must contain ───────────────────────────────────────────


@pytest.mark.parametrize("entry", REQUIRED_ENTRIES)
def test_the_package_contains_the_delivery_and_the_sanitized_program(fixture_build, entry):
    assert entry in fixture_build["names"], f"{entry} is missing from the ZIP"


def test_install_bat_sits_at_the_top_of_the_archive(fixture_build):
    """The operator unzips and double-clicks the first thing they see."""
    assert "install.bat" in fixture_build["names"]
    assert not [name for name in fixture_build["names"] if name.lower().endswith("install.bat")
                and "/" in name]


def test_the_archive_flattens_the_scripts_to_its_root(fixture_build):
    assert not [name for name in fixture_build["names"] if name.startswith("windows/")]


def test_the_shipped_configuration_is_the_sanitized_template(fixture_build):
    assert "payload/auto_tune/config.yaml" not in fixture_build["names"]
    assert "payload/auto_tune/config.template.yaml" in fixture_build["names"]


# ── the offline bundle travels with the package ─────────────────────────────


@pytest.mark.parametrize("entry", OFFLINE_ENTRIES)
def test_the_package_carries_the_complete_offline_bundle(fixture_build, entry):
    assert entry in fixture_build["names"], f"{entry} is missing from the ZIP"


def test_the_offline_bundle_lock_is_inside_the_archive(fixture_build, tmp_path):
    with zipfile.ZipFile(_package(fixture_build["output"])) as archive:
        lock = json.loads(archive.read("offline/offline-lock.json").decode("utf-8"))

    assert lock["schema_version"] == "1.0"
    assert lock["requirements_sha256"]
    purposes = {entry["purpose"] for entry in lock["files"]}
    assert purposes == {"private-python-runtime", "cuda-torch", "cuda-torchvision",
                        "runtime-dependency"}
    for entry in lock["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["size"] > 0
        assert ":" not in entry["path"], "the bundle lock never names a local path"


def test_the_package_lock_covers_every_offline_file(fixture_build):
    recorded = {entry["path"] for entry in fixture_build["lock"]["files"]}

    for entry in OFFLINE_ENTRIES:
        assert entry in recorded, f"{entry} is not verified by the installer"
    sizes = {entry["path"]: entry["size"] for entry in fixture_build["lock"]["files"]}
    assert sizes["offline/miniconda/Miniconda3-test-Windows-x86_64.exe"] == 4096


def test_the_package_never_carries_the_local_dependency_folder(fixture_build, tmp_path):
    """``依赖\\`` is a build input on one machine, not part of the delivery."""
    root = fixture_build["root"]

    assert (root / "依赖").is_dir(), "the fixture must really contain it"
    for name in fixture_build["names"]:
        assert not name.startswith("依赖/"), name
        assert "依赖" not in name, name
        assert name != "offline_cache"
        assert not name.startswith("offline_cache/"), name


def test_no_python_or_wheel_file_ships_outside_the_offline_bundle(fixture_build):
    for name in fixture_build["names"]:
        if name.endswith((".whl", ".exe")):
            assert name.startswith("offline/"), name
        assert Path(name).suffix.lower() not in (".pyc", ".log", ".tar.gz")


# ── what the package must never contain ─────────────────────────────────────


@pytest.mark.parametrize("relative", sorted(FORBIDDEN_SENTINELS))
def test_the_package_excludes_operator_data_and_build_noise(fixture_build, relative):
    kind = FORBIDDEN_SENTINELS[relative]
    names = set(fixture_build["names"])
    basename = Path(relative).name

    assert relative not in names
    assert f"payload/{relative}" not in names
    assert not [name for name in names if name.endswith("/" + relative)], kind
    assert not [name for name in names if Path(name).name == basename], kind


def test_the_package_never_contains_the_builder(fixture_build):
    assert "build_zip.ps1" not in fixture_build["names"]


def test_the_package_contains_no_weight_or_model_extension(fixture_build):
    for name in fixture_build["names"]:
        assert Path(name).suffix.lower() not in (".pt", ".pth", ".onnx", ".engine", ".db")


# ── the integrity lock the installer verifies ───────────────────────────────


def test_the_lock_records_a_hash_for_every_payload_file(fixture_build):
    lock = fixture_build["lock"]

    assert lock["schema_version"] == "1.0"
    assert lock["version"] == VERSION
    assert lock["files"], "the lock must record the payload"
    for entry in lock["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"]
        # 0 is a real size: the program tree has empty package markers.
        assert entry["size"] >= 0
    assert any(entry["size"] > 0 for entry in lock["files"])
    assert any(entry["path"].startswith("offline/") for entry in lock["files"]), (
        "the offline bundle is part of what the installer verifies")


def test_the_recorded_hashes_match_the_files_inside_the_archive(fixture_build):
    recorded = {entry["path"]: entry["sha256"] for entry in fixture_build["lock"]["files"]}
    for path, digest in fixture_build["payload_hashes"].items():
        assert recorded[path] == digest


def test_the_lock_covers_the_manifest_and_the_dependency_lock(fixture_build):
    recorded = {entry["path"] for entry in fixture_build["lock"]["files"]}

    assert "package-manifest.json" in recorded
    assert "requirements-windows.lock.txt" in recorded


def test_the_lock_covers_the_entry_points_that_are_installed_permanently(fixture_build):
    """The launcher is copied into the installation, so it is verified first."""
    recorded = {entry["path"] for entry in fixture_build["lock"]["files"]}

    for name in ("start.bat", "start.ps1", "uninstall.bat", "uninstall.ps1",
                 "lib/AutoTuneDelivery.psm1"):
        assert name in recorded, f"{name} is installed but not verified"


# ── repeatability and build hygiene ─────────────────────────────────────────


def test_building_twice_produces_the_same_package(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    output = tmp_path / "out"
    output.mkdir()

    first = _build(root, output)
    assert first.returncode == 0, (first.stdout or "") + (first.stderr or "")
    with zipfile.ZipFile(_package(output)) as archive:
        names_before = sorted(archive.namelist())

    second = _build(root, output)
    assert second.returncode == 0, (second.stdout or "") + (second.stderr or "")
    with zipfile.ZipFile(_package(output)) as archive:
        names_after = sorted(archive.namelist())

    assert names_before == names_after
    assert not [name for name in names_after if name.startswith("staging")]


def test_the_package_name_is_the_packaged_version(tmp_path):
    """The candidate package name is the manifest version, built for real.

    The version is the only thing that tells the operator (and the installer)
    which candidate they unzipped, so the archive name must come from the
    manifest rather than from a hand-typed argument."""
    root = _fixture_repo(tmp_path / "repo")
    manifest_path = root / "windows" / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "0.2.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    result = _build(root, output, version=None)

    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    assert (output / "AutoTuneStudio-Setup-0.2.1.zip").is_file()
    assert sorted(path.name for path in output.iterdir()) == [
        "AutoTuneStudio-Setup-0.2.1.zip"]


def test_an_output_directory_inside_the_repository_never_enters_the_zip(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    output = root / "build_output"

    first = _build(root, output)
    assert first.returncode == 0, (first.stdout or "") + (first.stderr or "")
    second = _build(root, output)
    assert second.returncode == 0, (second.stdout or "") + (second.stderr or "")

    with zipfile.ZipFile(_package(output)) as archive:
        names = archive.namelist()

    assert not [name for name in names if name.startswith("build_output")]
    assert not [name for name in names if name.endswith(".zip")]
    assert not [name for name in names if name == _package(output).name]


def test_the_builder_writes_nothing_outside_its_output_directory(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    output = tmp_path / "out"
    output.mkdir()

    assert _build(root, output).returncode == 0

    leftovers = sorted(path.name for path in output.iterdir())
    assert leftovers == [_package(output).name]
    assert not [path for path in root.parent.iterdir() if path.name.startswith("staging")]


def _short_path(path: Path) -> str:
    """The real 8.3 spelling of *path* (``...\\ADMINI~1.DES\\...``).

    Asked of the file system, not of a string: the point of the test is that
    the builder copes with the spelling Windows itself uses. Returns '' when
    the volume hands out no short names."""
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
         "$fso = New-Object -ComObject Scripting.FileSystemObject;"
         "$p = (Get-Item -LiteralPath $env:F12D_SHORT_PATH -Force).FullName;"
         "$s = [string]$fso.GetFolder($p).ShortPath;"
         "if ($s) { [Console]::Out.Write($s) }"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        env={**os.environ, "F12D_SHORT_PATH": str(path)})
    short = (result.stdout or "").strip()
    if not short or "~" not in short or not Path(short).exists():
        return ""
    if short.lower().rstrip("\\") == str(path).lower().rstrip("\\"):
        return ""
    return short


def test_the_package_builds_from_a_short_path_repository(tmp_path):
    """Codex's second finding, at build time.

    ``Get-ChildItem`` reports long paths for everything below a short root, so
    a relative path computed from the length of the short spelling cuts in the
    wrong place: the scripts and the whole payload would be silently dropped."""
    root = _fixture_repo(tmp_path / "repo")
    output = tmp_path / "out"
    output.mkdir()
    long_result = _build(root, output)
    assert long_result.returncode == 0, (long_result.stdout or "") + (long_result.stderr or "")

    short_root = _short_path(root)
    if not short_root:
        pytest.skip("this volume does not hand out 8.3 short names")
    short_output = tmp_path / "out-short"
    short_output.mkdir()
    short_result = _build(Path(short_root), short_output)
    assert short_result.returncode == 0, (short_result.stdout or "") + (short_result.stderr or "")

    with zipfile.ZipFile(_package(output)) as archive:
        expected_names = sorted(archive.namelist())
        expected_count = len([name for name in expected_names if name.startswith("payload/")])
    with zipfile.ZipFile(_package(short_output)) as archive:
        actual_names = sorted(archive.namelist())

    assert actual_names == expected_names, "the short path built a different package"
    assert expected_count > 0
    assert len([name for name in actual_names if name.startswith("payload/")]) == expected_count


def test_the_builder_refuses_a_package_without_a_valid_manifest(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    (root / "windows" / "package-manifest.json").write_text("{ not json",
                                                            encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    result = _build(root, output)

    assert result.returncode != 0, "a package without a valid manifest must not be built"
    assert not _package(output).exists()
    assert "PACKAGE_MANIFEST_INVALID" in (result.stdout or "") + (result.stderr or "")


# ── the real program tree ───────────────────────────────────────────────────


def test_the_package_built_from_the_real_program_tree_is_sanitized(fixture_build):
    """The fixture carries the real ``auto_tune\\``: what it drops is what the
    product really has, not a hand-written stub."""
    names = fixture_build["names"]
    forbidden_suffixes = (".pt", ".pth", ".onnx", ".engine", ".db", ".db-wal", ".db-shm",
                          ".zip", ".pyc", ".log")
    forbidden_prefixes = ("log/", "detect/", "runs/", "models/", "docker-data/",
                          "payload/log/", "payload/detect/", "payload/runs/",
                          "payload/models/", "payload/docker-data/", ".git/")

    for name in names:
        assert not name.endswith(forbidden_suffixes), name
        assert not name.startswith(forbidden_prefixes), name
        assert "config.yaml" not in name, name
        assert "__pycache__" not in name, name
        assert ".pytest_cache" not in name, name
        assert not name.startswith("build_output"), name
        assert name != "build_zip.ps1"
        assert name != "prepare_offline_bundle.ps1", "a build tool must not ship"


def test_the_real_program_tree_ships_the_sanitized_template_and_not_a_config(fixture_build):
    names = set(fixture_build["names"])

    assert "payload/auto_tune/config.template.yaml" in names
    assert "payload/auto_tune/config.yaml" not in names
    assert (fixture_build["root"] / "auto_tune" / "config.yaml").is_file(), (
        "the fixture must really contain a real configuration to exclude")


def test_the_real_program_tree_hides_the_search_and_verification_scripts(fixture_build):
    names = fixture_build["names"]

    assert not [name for name in names if "/tests/" in name]
    assert not [name for name in names if "auto_tune/scripts/" in name]


def test_the_real_program_tree_ships_no_internal_notes_or_stale_dependency_list(fixture_build):
    """Only what the product imports at run time: internal notes, the evaluation
    harness and the stale requirements list stay in the repository."""
    names = fixture_build["names"]

    assert not [name for name in names if name.endswith(".md")]
    assert not [name for name in names if "auto_tune/evaluation/" in name]
    assert not [name for name in names if "auto_tune/docs/" in name]
    assert "payload/auto_tune/requirements.txt" not in names


def test_the_package_ships_the_dependency_lock_byte_for_byte(fixture_build, tmp_path):
    with zipfile.ZipFile(_package(fixture_build["output"])) as archive:
        archive.extract("requirements-windows.lock.txt", tmp_path)
    shipped = (tmp_path / "requirements-windows.lock.txt").read_bytes()

    source = fixture_build["root"] / "windows" / "requirements-windows.lock.txt"
    assert shipped == source.read_bytes()


# ── an incomplete bundle must stop the build ────────────────────────────────


def test_the_builder_refuses_to_build_without_a_verified_offline_bundle(tmp_path):
    """No bundle, no package: the failure is the same one the installer would
    report, and it happens before anything is written."""
    root = _fixture_repo(tmp_path / "repo")
    shutil.rmtree(root / "offline_cache")
    output = tmp_path / "out"
    output.mkdir()

    result = _build(root, output)

    assert result.returncode != 0
    assert "OFFLINE_BUNDLE_MISSING" in (result.stdout or "") + (result.stderr or "")
    assert not _package(output).exists(), "no ZIP may be produced"
    assert not [name for name in os.listdir(output) if name.startswith(".staging")]


def test_the_builder_refuses_a_tampered_offline_bundle(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    wheel = (root / "offline_cache" / "wheelhouse"
             / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl")
    wheel.write_bytes(b"not the pinned wheel")
    output = tmp_path / "out"
    output.mkdir()

    result = _build(root, output)

    assert result.returncode != 0
    assert "OFFLINE_BUNDLE_HASH_MISMATCH" in (result.stdout or "") + (result.stderr or "")
    assert not _package(output).exists()


def test_the_builder_refuses_an_incomplete_wheelhouse(tmp_path):
    root = _fixture_repo(tmp_path / "repo")
    (root / "offline_cache" / "wheelhouse" / "uvicorn-0.51.0-py3-none-any.whl").unlink()
    output = tmp_path / "out"
    output.mkdir()

    result = _build(root, output)

    assert result.returncode != 0
    assert "OFFLINE_WHEELHOUSE_INCOMPLETE" in (result.stdout or "") + (result.stderr or "")
    assert not _package(output).exists()
