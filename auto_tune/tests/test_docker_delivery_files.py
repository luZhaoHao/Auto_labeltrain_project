"""F1.2-B: static contracts for the Docker delivery files.

These tests read the delivery files as text. They never run Docker, so they
pass on a machine where the Linux Engine is unavailable, and they never
assert on a real configuration, credential or dataset.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_REQUIREMENTS = REPO_ROOT / "docker" / "requirements-runtime.txt"
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
CONTAINER_ENTRYPOINT = REPO_ROOT / "auto_tune" / "delivery" / "container_entrypoint.py"
COMPOSE = REPO_ROOT / "compose.yaml"
SOURCE_REQUIREMENTS = REPO_ROOT / "requirements.txt"
SOURCE_ENVIRONMENT = REPO_ROOT / "environment.yml"

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


# ── Source installation: one accepted dependency contract ───────────────────


def test_source_requirements_match_the_accepted_runtime_without_torch():
    """A source install must not silently lose HPO/ONNX or drift from delivery."""
    assert _pinned_packages(SOURCE_REQUIREMENTS) == _pinned_packages(RUNTIME_REQUIREMENTS)


def test_conda_environment_builds_the_same_runtime_with_cuda_121_pytorch():
    """The one-file Conda path must be complete and must not resolve CPU torch."""
    data = yaml.safe_load(SOURCE_ENVIRONMENT.read_text(encoding="utf-8"))
    dependencies = data["dependencies"]
    pip_section = next(item["pip"] for item in dependencies if isinstance(item, dict))
    pip_pins = {}
    for line in pip_section:
        match = _PIN_PATTERN.match(line)
        assert match, f"Conda pip entry is not exactly pinned: {line!r}"
        pip_pins[match.group(1).lower()] = match.group(2)

    assert "python=3.10" in dependencies
    assert "pytorch=2.5.1" in dependencies
    assert "torchvision=0.20.1" in dependencies
    assert "pytorch-cuda=12.1" in dependencies
    assert pip_pins == _pinned_packages(RUNTIME_REQUIREMENTS)


# ── Task 4: single-image Docker delivery ────────────────────────────────────


def _dockerfile_lines() -> list[str]:
    return [
        line.strip()
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _dockerfile_instructions() -> list[str]:
    """Dockerfile instructions with shell line continuations joined.

    A ``RUN`` that spans several lines is one instruction; the credential
    directory must be created and owned by it before the image drops to the
    runtime user, and that is a property of the whole instruction.
    """
    instructions: list[str] = []
    pending = ""
    for line in _dockerfile_lines():
        pending = f"{pending} {line}".strip() if pending else line
        if pending.endswith("\\"):
            pending = pending[:-1].strip()
            continue
        instructions.append(pending)
        pending = ""
    if pending:
        instructions.append(pending)
    return instructions


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


def test_dockerfile_keeps_the_entrypoint_privileged_for_the_first_start():
    """The first start must be able to hand a root-owned bind directory to the
    runtime user, so the image cannot drop to ``studio`` before the entrypoint
    runs. The drop itself happens inside the entrypoint and is asserted in
    test_container_entrypoint.py; what must not happen here is the image taking
    those privileges away first."""
    user_lines = [
        line for line in _dockerfile_lines() if line.upper().startswith("USER ")
    ]
    assert user_lines, "the image must state which user starts the container"
    assert all(line.split()[1] in ("root", "0") for line in user_lines), (
        "a `USER studio` would remove the privileges the entrypoint needs before "
        "it has initialised the mounted directories")


def test_dockerfile_creates_the_runtime_user_as_10001_10001():
    """The drop target is verified numerically, so the account and its group must
    exist with exactly those ids instead of whatever useradd would pick."""
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "groupadd --gid 10001 studio" in text
    assert "useradd" in text
    assert "--uid 10001" in text
    assert "--gid 10001" in text


def test_dockerfile_never_starts_business_code_as_root():
    """The image starts as root for one directory-initialisation step only: the
    Studio is exec'd by the entrypoint, after it has dropped to the runtime user.
    There is no CMD for anything else to run."""
    instructions = _dockerfile_instructions()

    assert not [line for line in instructions if line.upper().startswith("CMD ")], \
        "the Studio is started by the entrypoint, never straight from the image"
    user_index = next(index for index, line in enumerate(instructions)
                      if line.upper().startswith("USER "))
    entry_index = next(index for index, line in enumerate(instructions)
                       if line.upper().startswith("ENTRYPOINT "))
    assert instructions[user_index].split()[1] in ("root", "0")
    assert entry_index == user_index + 1, (
        "the entrypoint must start with the privileges the initialisation needs")


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
    for forbidden in ("api_key", "apikey", "token", "password"):
        assert forbidden.lower() not in text.lower(), f"Dockerfile leaks {forbidden!r}"
    # ``secrets`` is the controlled credential *directory* the named volume is
    # mounted at (see test_dockerfile_creates_the_persisted_credential_directory).
    # It is a path, never a credential: no other secret-shaped text may appear.
    for line in text.splitlines():
        if "secret" in line.lower():
            assert "/data/secrets" in line, f"Dockerfile leaks a credential: {line!r}"


def test_dockerfile_creates_the_persisted_credential_directory_owned_by_the_runtime_user():
    """The mount point of the credential named volume must exist in the image,
    owned by the runtime user, before the image drops privileges.

    A fresh named volume is seeded from the image directory: if the directory
    only existed as root (or did not exist), Docker would leave the mount point
    root-owned, the start-up preflight would refuse to run and the operator
    could not save an API key.
    """
    instructions = _dockerfile_instructions()
    user_index = next(
        index for index, line in enumerate(instructions)
        if line.upper().startswith("USER ")
    )

    created = [
        line for line in instructions[:user_index]
        if "mkdir -p" in line and "/data/secrets" in line
        and "chown" in line and "studio:studio" in line
    ]
    assert created, (
        "one privileged instruction must create /data/secrets and own it studio:studio")
    assert "chown -R studio:studio" in created[0], (
        "the recursive ownership must cover /data, the parent of /data/secrets")


def test_the_image_ships_no_credential_content_into_the_named_volume():
    """The volume is seeded from /data/secrets: that directory must stay free of
    credential content, and no instruction may copy a credential file into it."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "credentials.json" not in text
    copied = [
        line for line in _dockerfile_instructions()
        if line.upper().startswith(("COPY ", "ADD ")) and "/data/secrets" in line
    ]
    assert copied == [], f"the image must not carry credential content: {copied}"


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


@pytest.mark.parametrize("entry", ["credentials.json", "credentials.json.*"])
def test_dockerignore_excludes_the_persisted_credential_file(entry):
    """The key typed into the page lives in ``secrets/`` outside the image; a
    stray copy in the build context must never be sent to the builder."""
    assert entry in _excluded_dockerignore_patterns()


def _entrypoint_handover_command() -> str:
    """The single command the shell entrypoint execs (line continuations joined).

    The shell no longer builds the preflight command line: the privileged
    initialisation, the privilege drop and the preflight moved into one module so
    they can be tested without a container. The shell only hands over.
    """
    lines = ENTRYPOINT.read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.strip().startswith("exec ")), None)
    assert start is not None, "the entrypoint must exec its handover instead of forking"
    command: list[str] = []
    for line in lines[start:]:
        command.append(line.rstrip("\\").strip())
        if not line.rstrip().endswith("\\"):
            break
    return " ".join(command)


def test_entrypoint_hands_over_to_the_container_initialisation_module():
    assert _entrypoint_handover_command() == \
        "exec python -m auto_tune.delivery.container_entrypoint"


def test_entrypoint_runs_the_shared_delivery_preflight_before_the_studio():
    """Both steps live in the container module, and the preflight comes first."""
    text = CONTAINER_ENTRYPOINT.read_text(encoding="utf-8")

    assert text.index("auto_tune.delivery.preflight") < text.index("auto_tune.main")


def test_entrypoint_delegates_directory_and_configuration_initialisation():
    """Directory creation, write permission and the configuration bootstrap
    live in ``auto_tune/delivery/preflight.py`` so the Windows delivery can
    reuse the same rules; the container entrypoint only asks for the container
    subset, and it asks for it after dropping privileges."""
    from auto_tune.delivery.container_entrypoint import build_preflight_argv

    command = " ".join(build_preflight_argv())

    assert "--require-mounts" in command
    assert "--bootstrap-config" in command


def test_entrypoint_requires_the_gpu_instead_of_falling_back_to_cpu():
    from auto_tune.delivery.container_entrypoint import build_preflight_argv

    assert "--require-gpu" in build_preflight_argv()


def test_entrypoint_stops_the_start_when_the_preflight_fails():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "set -e" in text or "set -euo pipefail" in text
    assert "CONTAINER_PREFLIGHT_FAILED" in CONTAINER_ENTRYPOINT.read_text(encoding="utf-8")


def test_entrypoint_exports_the_controlled_configuration_path():
    assert "AUTO_TUNE_CONFIG_PATH" in ENTRYPOINT.read_text(encoding="utf-8")


def test_entrypoint_exports_the_persisted_credential_path():
    """The container default must be documented in one place, and it must match
    the path compose mounts, so a bare `docker run` behaves like compose."""
    text = ENTRYPOINT.read_text(encoding="utf-8")

    assert "AUTO_TUNE_CREDENTIALS_PATH" in text
    assert "/data/secrets/credentials.json" in text


def test_entrypoint_execs_the_studio_so_signals_reach_the_application():
    """The shell replaces itself, and the module replaces *that* process with the
    Studio, so the container's PID 1 is the Python process that receives SIGTERM
    and nothing sits in between to swallow it."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    module = CONTAINER_ENTRYPOINT.read_text(encoding="utf-8")

    assert "exec python" in text
    assert "os.execv" in module
    assert "Popen" not in module
    assert "-m auto_tune.main" not in text, (
        "the Studio is exec'd by the module, not by the shell")


def test_entrypoint_never_prints_configuration_or_credential_contents():
    """The entrypoint exports the credential *path* only.

    The directory name ``secrets`` is part of the documented path, so the check
    is about printing: no file dump, no shell tracing, no credential variable
    echoed and no key-shaped or credential-field text anywhere.
    """
    text = ENTRYPOINT.read_text(encoding="utf-8")

    for forbidden in ("cat ", "set -x", "api_key", "apikey", "token", "password", "sk-"):
        assert forbidden not in text.lower(), f"entrypoint must not print {forbidden!r}"
    assert 'echo "$AUTO_TUNE_CREDENTIALS_PATH' not in text
    assert "echo '${AUTO_TUNE_CREDENTIALS_PATH" not in text


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


def test_compose_mounts_the_seven_controlled_directories():
    volumes = _compose()["services"]["studio"]["volumes"]
    targets = {str(volume).split(":")[-1] for volume in volumes}

    assert targets == {
        "/data/config",
        "/opt/auto-tune/log",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/runs",
        "/opt/auto-tune/models/weights",
        "/data/datasets",
        "/data/secrets",
    }


def test_compose_persists_the_credential_directory_in_a_named_volume():
    """The credential directory must not be a host bind mount.

    On Linux a bind mount whose host directory does not exist yet is created by
    Docker as root:root, which the unprivileged runtime user cannot write — a
    first start would be refused. A named volume is seeded once from the image
    directory, which the Dockerfile creates owned by the runtime user.
    """
    compose = _compose()
    volumes = [str(volume) for volume in compose["services"]["studio"]["volumes"]]

    assert "secrets:/data/secrets" in volumes
    assert "secrets" in (compose.get("volumes") or {}), (
        "the named volume must be declared so compose creates and keeps it")
    assert not any(volume.endswith("/secrets:/data/secrets") for volume in volumes), (
        "the credential directory is still a host bind mount")


def test_the_credential_volume_is_kept_by_down_and_only_removed_explicitly():
    """``docker compose down`` keeps a declared named volume; only ``down -v``
    (or ``docker volume rm``) removes it. An ``external`` volume would not be
    removed by ``down -v``, and a ``local`` driver bound to a host path would
    bring the root-owned bind directory straight back."""
    definition = (_compose().get("volumes") or {}).get("secrets") or {}

    assert definition.get("external") in (None, False), (
        "an external volume is not removed by `down -v`")
    device = (definition.get("driver_opts") or {}).get("device")
    assert device is None, f"the volume must not be backed by a host path: {device!r}"


def test_only_the_credential_directory_left_the_host_bind_mounts():
    """The six operator-visible directories keep their host bind mounts; exactly
    one mount is not a bind."""
    volumes = [str(volume) for volume in _compose()["services"]["studio"]["volumes"]]
    mounts = [volume for volume in volumes if not volume.startswith("${")]
    binds = [volume for volume in volumes if volume.startswith("${")]

    assert mounts == ["secrets:/data/secrets"]
    assert sorted(volume.split(":")[-1] for volume in binds) == [
        "/data/config",
        "/data/datasets",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/log",
        "/opt/auto-tune/models/weights",
        "/opt/auto-tune/runs",
    ]


def test_compose_points_the_container_at_the_persisted_credential_file():
    service = _compose()["services"]["studio"]

    assert service["environment"]["AUTO_TUNE_CREDENTIALS_PATH"] == \
        "/data/secrets/credentials.json"
    targets = {str(volume).split(":")[-1] for volume in service["volumes"]}
    assert "/data/secrets" in targets, "the file's directory is the mounted one"


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
                          datasets_dir=CONTAINER_DATASETS_DIR,
                          credentials_path=Path("/data/secrets/credentials.json"))
    # ``as_posix`` keeps the container paths comparable on any host platform.
    expected = {path.as_posix() for path in paths.persistent_dirs.values()}
    volumes = _compose()["services"]["studio"]["volumes"]
    mounted = {str(volume).split(":")[-1] for volume in volumes}

    assert mounted == expected
    assert expected == {
        "/data/config",
        "/data/datasets",
        "/opt/auto-tune/log",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/runs",
        "/opt/auto-tune/models/weights",
        "/data/secrets",
    }


def test_compose_keeps_everything_the_first_start_contract_needs():
    """The first-start permission fix is a change to *how* the container
    initialises, not to what it is given: the service keeps its GPU reservation,
    its loopback port, its shared-memory budget, all six host bind mounts and the
    credential named volume."""
    service = _compose()["services"]["studio"]
    volumes = [str(volume) for volume in service["volumes"]]
    mounts = [volume for volume in volumes if not volume.startswith("${")]

    assert service["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    ]
    assert str(service["ports"][0]) == "127.0.0.1:${AUTO_TUNE_PORT:-8000}:8000"
    assert _require_shm_size_at_least_1gib(service["shm_size"]) >= _GIB
    assert sorted(volume.split(":")[-1] for volume in volumes) == [
        "/data/config",
        "/data/datasets",
        "/data/secrets",
        "/opt/auto-tune/detect",
        "/opt/auto-tune/log",
        "/opt/auto-tune/models/weights",
        "/opt/auto-tune/runs",
    ]
    assert mounts == ["secrets:/data/secrets"]
    assert "secrets" in (_compose().get("volumes") or {})


def test_compose_keeps_the_gpu_language_out_of_a_cpu_fallback():
    """No CPU-only mode is delivered: the compose file must not advertise one."""
    text = COMPOSE.read_text(encoding="utf-8").lower()
    assert "cpu-only" not in text
    assert "no gpu" not in text


# ── the controlled input browse roots the folder pickers start from ─────────


def _declared_browse_roots() -> list[str]:
    environment = _compose()["services"]["studio"]["environment"]
    declared = environment["AUTO_TUNE_INPUT_ALLOWED_ROOTS"]
    return str(declared).split(";")


def test_compose_declares_the_input_browse_roots():
    """Without them the folder pickers list nothing at all in the container:
    the configuration ships no allowed root, and the container has no drives."""
    assert _declared_browse_roots() == ["/data/datasets", "/opt/auto-tune/detect"]


def test_the_browse_roots_are_the_dataset_share_and_the_detect_directory():
    """The roots come from the shared delivery layout, not from a second
    hand-written copy of the container paths. ``detect`` is where
    ``find_detect_dir()`` really writes every training run; the retired ``runs``
    mount is deliberately *not* offered as well, which would only hide the
    mismatch between the browser and the product."""
    from auto_tune.delivery.preflight import CONTAINER_APP_ROOT, CONTAINER_DATASETS_DIR

    assert _declared_browse_roots() == [
        CONTAINER_DATASETS_DIR.as_posix(),
        (CONTAINER_APP_ROOT / "detect").as_posix(),
    ]
    from auto_tune.delivery.runtime import CONTAINER_INPUT_ROOTS

    assert CONTAINER_INPUT_ROOTS == tuple(_declared_browse_roots())


def test_the_retired_runs_directory_is_not_offered_as_a_second_root():
    """A browse root that no product flow writes into would mask the real path
    the training analysis picker has to reach."""
    from auto_tune.delivery.preflight import CONTAINER_APP_ROOT

    assert (CONTAINER_APP_ROOT / "runs").as_posix() not in _declared_browse_roots()


def test_every_browse_root_is_a_mounted_directory():
    """A root that is not one of the mounts would let the picker walk into the
    container's own filesystem; widening this must require a new mount."""
    service = _compose()["services"]["studio"]
    mounted = {str(volume).split(":")[-1] for volume in service["volumes"]}

    assert _declared_browse_roots(), "the container must declare its roots"
    for root in _declared_browse_roots():
        assert root in mounted, f"{root} is browsable but not mounted"


@pytest.mark.parametrize("wide", ["/", "/data", "/opt", "/opt/auto-tune"])
def test_compose_never_offers_a_container_wide_root(wide):
    assert wide not in _declared_browse_roots()


def test_the_declared_start_roots_pass_the_runtimes_own_validation(monkeypatch):
    """The value the container ships is the value the resolver accepts."""
    from auto_tune.delivery import runtime

    declared = ";".join(_declared_browse_roots())
    monkeypatch.setenv(runtime.INPUT_ALLOWED_ROOTS_ENV, declared)

    roots = runtime.resolve_input_allowed_roots()

    assert [root.as_posix() for root in roots] == _declared_browse_roots()


def test_entrypoint_defaults_the_input_browse_roots_for_a_bare_container():
    """``docker run`` without compose must offer the same two directories."""
    text = ENTRYPOINT.read_text(encoding="utf-8")

    assert "AUTO_TUNE_INPUT_ALLOWED_ROOTS" in text
    assert "/data/datasets;/opt/auto-tune/detect" in text


def test_the_shipped_defaults_are_exactly_the_whitelist_the_code_accepts(monkeypatch):
    """Compose and the entrypoint cannot drift from the resolver: the value they
    ship is the value the resolver accepts, character for character."""
    from auto_tune.delivery import runtime

    declared = ";".join(runtime.CONTAINER_INPUT_ROOTS)
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert _compose()["services"]["studio"]["environment"][
        "AUTO_TUNE_INPUT_ALLOWED_ROOTS"] == declared
    assert f'${{AUTO_TUNE_INPUT_ALLOWED_ROOTS:-{declared}}}' in entrypoint

    monkeypatch.setenv(runtime.INPUT_ALLOWED_ROOTS_ENV, declared)

    roots = runtime.resolve_input_allowed_roots()

    assert [root.as_posix() for root in roots] == declared.split(";")


def test_the_desktop_delivery_never_sets_the_container_browse_roots():
    """The Windows start lists the machine's drives: setting the variable there
    would silently replace that with two container paths."""
    windows = REPO_ROOT / "windows"
    for path in sorted(windows.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".ps1", ".bat", ".psm1", ".json",
                                                     ".txt"):
            assert "AUTO_TUNE_INPUT_ALLOWED_ROOTS" not in path.read_text(
                encoding="utf-8", errors="replace"), path.name


def test_compose_contains_no_credential_or_local_absolute_path():
    """Compose names *where* a credential is persisted, never a credential."""
    text = COMPOSE.read_text(encoding="utf-8")
    for forbidden in ("api_key", "apikey", "token", "password", "sk-", "your_"):
        assert forbidden.lower() not in text.lower(), f"compose leaks {forbidden!r}"
    assert not _WINDOWS_ABSOLUTE_PATH.search(text)
    # the only credential-related text is the persisted location
    assert "/data/secrets" in text
