"""F1.2-B: static contracts for the Docker delivery files.

These tests read the delivery files as text. They never run Docker, so they
pass on a machine where the Linux Engine is unavailable, and they never
assert on a real configuration, credential or dataset.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_REQUIREMENTS = REPO_ROOT / "docker" / "requirements-runtime.txt"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
COMPOSE = REPO_ROOT / "compose.yaml"

_PIN_PATTERN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;]+)$")
# A Windows drive path ("D:\x" / "C:/x") that is not a volume separator such
# as the "log:/opt/..." mount syntax.
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![\w-])[A-Za-z]:[\\/]")


def _pinned_packages(path: Path) -> dict[str, str]:
    """Return {lowercased package name: pinned version} from a requirements file."""
    packages: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_PATTERN.match(line)
        assert match, f"entry is not exactly pinned: {line!r}"
        packages[match.group(1).lower()] = match.group(2)
    return packages


# ── Task 3: controlled Docker runtime dependencies ──────────────────────────


def test_runtime_requirements_pins_every_package_production_imports():
    pinned = _pinned_packages(RUNTIME_REQUIREMENTS)

    for name in (
        "fastapi",
        "uvicorn",
        "jinja2",
        "python-multipart",
        "starlette",
        "pydantic",
        "pyyaml",
        "requests",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "ultralytics",
        "optuna",
    ):
        assert name in pinned, f"{name} is imported in production but not pinned"


def test_runtime_requirements_pins_the_accepted_ultralytics_version():
    assert _pinned_packages(RUNTIME_REQUIREMENTS)["ultralytics"] == "8.3.253"


def test_runtime_requirements_keeps_torch_out_of_the_pypi_install():
    """The Dockerfile installs the CUDA 12.1 wheels first; pip must not override."""
    pinned = _pinned_packages(RUNTIME_REQUIREMENTS)
    assert "torch" not in pinned
    assert "torchvision" not in pinned


@pytest.mark.parametrize(
    "name",
    ["pytest", "pluggy", "iniconfig", "exceptiongroup", "tomli", "pyreadline3"],
)
def test_runtime_requirements_excludes_test_only_and_windows_only_packages(name):
    assert name not in _pinned_packages(RUNTIME_REQUIREMENTS)


def test_runtime_requirements_contains_no_url_path_or_credential():
    text = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")

    for forbidden in (
        "git+",
        "git@",
        "github.com",
        "file://",
        "http://",
        "https://",
        "\\\\",
        "/opt/",
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
    ):
        assert forbidden.lower() not in text.lower(), f"runtime file leaks {forbidden!r}"
    assert not _WINDOWS_ABSOLUTE_PATH.search(text)


def test_runtime_requirements_pins_the_accepted_onnx_export_stack():
    """F1.2-C: the container must serve the F1.2-A trusted `.pt` → FP32 `.onnx`
    path, and the versions are the ones the accepted environment uses."""
    pinned = _pinned_packages(RUNTIME_REQUIREMENTS)

    assert pinned["onnx"] == "1.17.0"
    assert pinned["protobuf"] == "7.35.1"
    assert pinned["onnxruntime"] == "1.22.0"


@pytest.mark.parametrize("name", ["onnxruntime-gpu", "onnxslim"])
def test_runtime_requirements_keeps_unused_onnx_extras_out(name):
    """The export path is frozen at simplify=False on CPU: the CUDA execution
    provider wheel and the simplifier are not part of the delivery."""
    assert name not in _pinned_packages(RUNTIME_REQUIREMENTS)


def test_runtime_requirements_pins_exact_versions_without_ranges():
    text = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    for forbidden in (">=", "<=", "~=", "!=", ">", "<"):
        assert forbidden not in text, f"runtime file uses a loose constraint {forbidden!r}"


# ── Task 4: single-image Docker delivery ────────────────────────────────────


def _dockerfile_lines() -> list[str]:
    return [
        line.strip()
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerfile_builds_one_pinned_image_from_an_accepted_base():
    lines = _dockerfile_lines()
    from_lines = [line for line in lines if line.upper().startswith("FROM ")]

    assert len(from_lines) == 1, "exactly one final image, no multi-stage build"
    base = from_lines[0].split()[1]
    assert base.startswith("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04")
    assert ":latest" not in base
    assert ":" in base, "the base image must be pinned to an explicit tag"


def test_dockerfile_sets_reproducible_python_runtime_flags():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "PYTHONUNBUFFERED=1" in text


def test_dockerfile_uses_the_controlled_workdir():
    assert "WORKDIR /opt/auto-tune" in DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_runs_the_container_as_a_non_root_user():
    user_lines = [
        line for line in _dockerfile_lines() if line.upper().startswith("USER ")
    ]
    assert user_lines, "the image must drop root privileges"
    user = user_lines[-1].split()[1]
    assert user not in ("root", "0"), "the runtime user must not be root"


def test_dockerfile_installs_pinned_cuda_torch_wheels_before_the_runtime_file():
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "torch==2.5.1" in text
    assert "torchvision==0.20.1" in text
    assert "download.pytorch.org/whl/cu121" in text
    assert text.index("torch==2.5.1") < text.index("-r docker/requirements-runtime.txt")


def test_dockerfile_installs_the_controlled_runtime_requirements():
    assert "docker/requirements-runtime.txt" in DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_declares_a_health_check_on_the_operational_probe():
    health_lines = [
        line
        for line in _dockerfile_lines()
        if line.upper().startswith("HEALTHCHECK")
    ]
    assert len(health_lines) == 1
    assert "/healthz" in health_lines[0]


def test_dockerfile_entrypoint_is_the_controlled_script():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["/opt/auto-tune/docker/entrypoint.sh"]' in text
    assert "chmod +x" in text


def test_dockerfile_never_copies_or_creates_a_real_configuration():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "auto_tune/config.yaml" not in text
    for forbidden in ("api_key", "apikey", "token", "password", "secret"):
        assert forbidden.lower() not in text.lower()


_REQUIRED_DOCKERIGNORE_ENTRIES = (
    ".git",
    ".claude",
    ".codex",
    "auto_tune/config.yaml",
    "dataset",
    "detect",
    "runs",
    "log",
    "models/weights",
    "*.pt",
    "*.onnx",
    "*.zip",
    "__pycache__",
    "render",
    "venv",
)


@pytest.mark.parametrize("entry", _REQUIRED_DOCKERIGNORE_ENTRIES)
def test_dockerignore_excludes_local_state_and_build_context_noise(entry):
    patterns = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any(entry in pattern for pattern in patterns), f".dockerignore misses {entry!r}"


def test_dockerignore_keeps_the_sanitized_template_available_to_the_build():
    patterns = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    excluded = {pattern for pattern in patterns if not pattern.startswith("!")}
    assert "auto_tune/config.yaml" in excluded
    assert "auto_tune/config.template.yaml" not in excluded
    assert "docker/requirements-runtime.txt" not in excluded


def _excluded_dockerignore_patterns() -> set[str]:
    return {
        line.strip().rstrip("/").lstrip("/")
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("!")
    }


def test_dockerignore_excludes_the_compose_runtime_directory():
    """compose.yaml keeps the runtime config, SQLite index, logs and weights
    under ./docker-data, so the build context must prune that whole tree."""
    assert "docker-data" in _excluded_dockerignore_patterns()


def test_dockerignore_excludes_the_runtime_files_that_already_exist():
    """Codex review 2026-09-21: the runtime files present on disk must not be
    sent to the builder even though the Dockerfile does not COPY them today."""
    patterns = _excluded_dockerignore_patterns()
    for runtime_file in ("docker-data/config/config.yaml", "docker-data/log/auto_tune.db"):
        first_segment = runtime_file.split("/", 1)[0]
        assert first_segment in patterns, f"{runtime_file} would enter the build context"


@pytest.mark.parametrize("entry", [".env", ".env.*"])
def test_dockerignore_excludes_local_env_credential_files(entry):
    assert entry in _excluded_dockerignore_patterns()


def _entrypoint_preflight_command() -> str:
    """The logical preflight command, including shell line continuations."""
    lines = ENTRYPOINT.read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines)
                  if "auto_tune.delivery.preflight" in line), None)
    assert start is not None, "the entrypoint must run the shared delivery preflight"
    command: list[str] = []
    for line in lines[start:]:
        command.append(line.rstrip("\\").strip())
        if not line.rstrip().endswith("\\"):
            break
    return " ".join(command)


def test_entrypoint_runs_the_shared_delivery_preflight_before_the_studio():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert text.index("auto_tune.delivery.preflight") < text.index("-m auto_tune.main")


def test_entrypoint_delegates_directory_and_configuration_initialisation():
    """Directory creation, write permission and the configuration bootstrap
    live in ``auto_tune/delivery/preflight.py`` so the Windows delivery can
    reuse the same rules; the entrypoint only asks for the container subset."""
    command = _entrypoint_preflight_command()

    assert "--require-mounts" in command
    assert "--bootstrap-config" in command


def test_entrypoint_requires_the_gpu_instead_of_falling_back_to_cpu():
    assert "--require-gpu" in _entrypoint_preflight_command()


def test_entrypoint_stops_the_start_when_the_preflight_fails():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "set -e" in text or "set -euo pipefail" in text


def test_entrypoint_exports_the_controlled_configuration_path():
    assert "AUTO_TUNE_CONFIG_PATH" in ENTRYPOINT.read_text(encoding="utf-8")


def test_entrypoint_execs_the_studio_so_signals_reach_the_application():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "exec python" in text
    assert "-m auto_tune.main" in text


def test_entrypoint_never_prints_configuration_or_credential_contents():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    for forbidden in ("cat ", "api_key", "token", "password", "secret", "set -x"):
        assert forbidden not in text, f"entrypoint must not print {forbidden!r}"


def _compose() -> dict:
    import yaml

    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_defines_exactly_one_service_with_one_image():
    services = _compose()["services"]
    assert list(services) == ["studio"]
    assert services["studio"]["image"].startswith("auto-tune")
    assert services["studio"]["image"].endswith(":latest") is False


def test_compose_publishes_the_container_port_on_loopback_only():
    import yaml

    rendered = COMPOSE.read_text(encoding="utf-8")
    ports = _compose()["services"]["studio"]["ports"]
    assert len(ports) == 1
    port = str(ports[0])
    assert port.startswith("127.0.0.1:")
    assert port.endswith(":8000")
    assert "AUTO_TUNE_PORT" in port
    assert yaml.safe_load("a: " + port)["a"] == port


def test_compose_binds_the_container_to_the_container_interface():
    environment = _compose()["services"]["studio"]["environment"]
    assert environment["AUTO_TUNE_HOST"] == "0.0.0.0"
    assert str(environment["AUTO_TUNE_PORT"]) == "8000"


def test_compose_mounts_the_six_controlled_host_directories():
    volumes = _compose()["services"]["studio"]["volumes"]
    targets = {str(volume).split(":")[-1] for volume in volumes}

    assert targets == {
        "/data/config",
        "/opt/auto-tune/log",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/runs",
        "/opt/auto-tune/models/weights",
        "/data/datasets",
    }


# ── F1.2-E: the shared-memory budget the training DataLoaders need ──────────

_GIB = 1024**3

_SHM_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": _GIB,
    "gb": _GIB,
}


def _shm_size_bytes(value: object) -> int:
    """Parse a Compose ``shm_size`` byte value into an integer byte count."""
    if isinstance(value, bool):
        raise AssertionError(f"shm_size must be a byte value, not a boolean: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]*)\s*", value)
        if match and match.group(2).lower() in _SHM_BYTE_UNITS:
            return int(match.group(1)) * _SHM_BYTE_UNITS[match.group(2).lower()]
    raise AssertionError(f"shm_size is not a byte value: {value!r}")


def _require_shm_size_at_least_1gib(value: object) -> int:
    size = _shm_size_bytes(value)
    assert size >= _GIB, f"shm_size {value!r} is below the 1 GiB floor ({size} bytes)"
    return size


@pytest.mark.parametrize(
    "value", [None, 0, "0", "0b", "", "64m", "64mb", "128m", "1m", "1023mb", "1gib?"]
)
def test_the_shared_memory_floor_rejects_missing_or_insufficient_values(value):
    """The floor must not be satisfiable by a missing value or a budget below
    1 GiB, and it must not silently accept an unparseable value either."""
    with pytest.raises(AssertionError):
        _require_shm_size_at_least_1gib(value)


@pytest.mark.parametrize(
    "value", ["1g", "1gb", "2g", "1024mb", "1073741824", _GIB, 2 * _GIB]
)
def test_the_shared_memory_floor_accepts_values_at_or_above_one_gib(value):
    assert _require_shm_size_at_least_1gib(value) >= _GIB


def test_compose_declares_the_shared_memory_the_training_dataloaders_need():
    """F1.2-E: the Engine's default /dev/shm is 64 MiB, which makes the
    multi-worker training DataLoader die with a bus error (exit code 1), so the
    studio service must request at least 1 GiB. This asserts the parsed Compose
    semantics, not a comment."""
    service = _compose()["services"]["studio"]

    assert "shm_size" in service, "the studio service must declare shm_size"
    assert _require_shm_size_at_least_1gib(service["shm_size"]) >= _GIB


def test_compose_requests_the_gpu_for_the_normal_start():
    """F1.2-C: the formal delivery starts with a GPU reservation. The container
    is never meant to run training on the CPU, so this is not an opt-in."""
    deploy = _compose()["services"]["studio"]["deploy"]

    reservations = deploy["resources"]["reservations"]["devices"]
    assert reservations == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    ]


def test_compose_mount_targets_match_the_shared_delivery_layout():
    """One layout, defined once: the compose mounts must be exactly the
    persistent directories the start-up preflight validates."""
    from auto_tune.delivery.preflight import (
        CONTAINER_APP_ROOT,
        CONTAINER_DATASETS_DIR,
        resolve_paths,
    )

    paths = resolve_paths(app_root=CONTAINER_APP_ROOT,
                          config_path=Path("/data/config/config.yaml"),
                          datasets_dir=CONTAINER_DATASETS_DIR)
    # ``as_posix`` keeps the container paths comparable on any host platform.
    expected = {path.as_posix() for path in paths.persistent_dirs.values()}
    volumes = _compose()["services"]["studio"]["volumes"]
    mounted = {str(volume).split(":")[-1] for volume in volumes}

    assert mounted == expected


def test_compose_keeps_the_gpu_language_out_of_a_cpu_fallback():
    """No CPU-only mode is delivered: the compose file must not advertise one."""
    text = COMPOSE.read_text(encoding="utf-8").lower()
    assert "cpu-only" not in text
    assert "no gpu" not in text


def test_compose_contains_no_credential_or_local_absolute_path():
    text = COMPOSE.read_text(encoding="utf-8")
    for forbidden in ("api_key", "apikey", "token", "password", "secret"):
        assert forbidden.lower() not in text.lower()
    assert not _WINDOWS_ABSOLUTE_PATH.search(text)
