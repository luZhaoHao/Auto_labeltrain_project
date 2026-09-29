"""Delivery start-up preflight shared by the container and the desktop build.

The Studio writes its configuration, logs, SQLite index, training directories
and controlled weights into directories that the *operator* provides. This
module is the one place that decides whether those directories are usable
before the server starts, so a missing mount, an unwritable directory, a dataset
share the runtime user cannot read, a missing runtime component or an
unavailable GPU is reported as one stable code with a fixed message instead of
surfacing later as a traceback or — worse — being silently absorbed by the
container's writable layer.

Nothing here touches training, HPO, tuning, snapshot or persistence semantics.
Every check takes an injectable probe so the rules can be exercised without a
real container or a real GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .runtime import PACKAGE_CONFIG_PATH, resolve_config_path

__all__ = [
    "CONTAINER_APP_ROOT",
    "CONTAINER_DATASETS_DIR",
    "CUDA_VERSION",
    "DELIVERY_CONFIG_MISSING",
    "DELIVERY_DATASETS_UNREADABLE",
    "DELIVERY_DEPENDENCY_MISSING",
    "DELIVERY_DIR_NOT_MOUNTED",
    "DELIVERY_DIR_NOT_WRITABLE",
    "DELIVERY_GPU_UNAVAILABLE",
    "DELIVERY_RUNTIME_MISMATCH",
    "DELIVERY_TEMPLATE_MISSING",
    "DeliveryError",
    "DeliveryPaths",
    "OFFLINE_MODULES",
    "PYTHON_VERSION",
    "REQUIRED_MODULES",
    "TORCHVISION_VERSION",
    "TORCH_VERSION",
    "bootstrap_config",
    "check_dependencies",
    "check_mounts",
    "check_offline_runtime",
    "check_writable",
    "ensure_directories",
    "gpu_available",
    "main",
    "offline_runtime_details",
    "require_gpu",
    "resolve_paths",
    "run",
]

APP_ROOT_ENV = "AUTO_TUNE_APP_ROOT"
DATASETS_DIR_ENV = "AUTO_TUNE_DATASETS_DIR"
TEMPLATE_PATH_ENV = "AUTO_TUNE_TEMPLATE_PATH"
CREDENTIALS_PATH_ENV = "AUTO_TUNE_CREDENTIALS_PATH"

# The container delivery mounts these exactly (see compose.yaml); the desktop
# delivery points the same names at its own installation directory.
CONTAINER_APP_ROOT = Path("/opt/auto-tune")
CONTAINER_DATASETS_DIR = Path("/data/datasets")

# Relative to the application root, in the order the operator sees them.
_PERSISTENT_SUBDIRECTORIES = ("log", "detect", "runs", "models/weights")

DELIVERY_DIR_NOT_MOUNTED = "DELIVERY_DIR_NOT_MOUNTED"
DELIVERY_DIR_NOT_WRITABLE = "DELIVERY_DIR_NOT_WRITABLE"
DELIVERY_DATASETS_UNREADABLE = "DELIVERY_DATASETS_UNREADABLE"
DELIVERY_CONFIG_MISSING = "DELIVERY_CONFIG_MISSING"
DELIVERY_TEMPLATE_MISSING = "DELIVERY_TEMPLATE_MISSING"
DELIVERY_DEPENDENCY_MISSING = "DELIVERY_DEPENDENCY_MISSING"
DELIVERY_GPU_UNAVAILABLE = "DELIVERY_GPU_UNAVAILABLE"
DELIVERY_RUNTIME_MISMATCH = "DELIVERY_RUNTIME_MISMATCH"

# What the Windows offline bundle pins. The installer builds the private runtime
# from that bundle and then has the interpreter answer for itself, so "no CPU
# fallback" is a verified fact instead of a promise. A test asserts these agree
# with the wheels windows/package-manifest.json ships.
PYTHON_VERSION = "3.10"
TORCH_VERSION = "2.5.1+cu121"
TORCHVISION_VERSION = "0.20.1+cu121"
CUDA_VERSION = "12.1"

# What the Studio itself imports at run time. torch and torchvision are checked
# by version above rather than by existence.
OFFLINE_MODULES = ("fastapi", "ultralytics", "optuna", "onnx", "onnxruntime")

# Which importable component provides which product capability. A missing entry
# is a broken delivery, not a user error, so the message stays short.
REQUIRED_MODULES = {
    "training": "torch",
    "ultralytics": "ultralytics",
    "onnx-export": "onnx",
    "onnx-runtime": "onnxruntime",
}

# Shipping probes. Tests replace these attributes rather than monkeypatching
# the standard library for the whole process.
mount_probe = os.path.ismount


class DeliveryError(Exception):
    """A start-up precondition failed. ``message`` never contains a secret."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DeliveryPaths:
    """Where the delivery reads its configuration and writes its state."""

    app_root: Path
    config_path: Path
    template_path: Path
    datasets_dir: Path
    # Only set when the platform persists API keys to a file (the container).
    credentials_path: Path | None = None

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    @property
    def credentials_dir(self) -> Path | None:
        """The directory the credential file lives in, when that backend is used.

        The *file* is allowed to be absent — a first start has no key yet — but
        its directory must be a real mount, otherwise a key typed into the page
        would be written into the container's writable layer and lost on the
        next recreation.
        """
        if self.credentials_path is None:
            return None
        return self.credentials_path.parent

    @property
    def persistent_dirs(self) -> dict[str, Path]:
        directories = {
            "config": self.config_dir,
            "datasets": self.datasets_dir,
            "log": self.app_root / "log",
            "detect": self.app_root / "detect",
            "runs": self.app_root / "runs",
            "models/weights": self.app_root / "models" / "weights",
        }
        if self.credentials_dir is not None:
            directories["secrets"] = self.credentials_dir
        return directories


def _from_env(name: str, default: Path) -> Path:
    """Environment overrides are resolved; the default is already absolute."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise ValueError(f"{name} must not be blank")
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _optional_env_path(name: str) -> Path | None:
    """A path that may legitimately be unset, with a blank value meaning unset.

    An absent variable means "this platform has no credential file": the desktop
    keeps the OS credential store, and nothing about its layout changes.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def resolve_paths(app_root: Path | None = None,
                  config_path: Path | None = None,
                  datasets_dir: Path | None = None,
                  template_path: Path | None = None,
                  credentials_path: Path | None = None) -> DeliveryPaths:
    """Resolve the delivery layout.

    Explicit arguments are used verbatim; anything left unset comes from the
    controlled environment, and finally the packaged defaults. The credential
    path is the one entry with no default: unset means the platform does not
    keep a credential file at all.
    """
    package_dir = PACKAGE_CONFIG_PATH.parent
    return DeliveryPaths(
        app_root=app_root if app_root is not None
        else _from_env(APP_ROOT_ENV, package_dir),
        config_path=config_path if config_path is not None
        else resolve_config_path(PACKAGE_CONFIG_PATH),
        datasets_dir=datasets_dir if datasets_dir is not None
        else _from_env(DATASETS_DIR_ENV, CONTAINER_DATASETS_DIR),
        template_path=template_path if template_path is not None
        else _from_env(TEMPLATE_PATH_ENV, package_dir / "config.template.yaml"),
        credentials_path=credentials_path if credentials_path is not None
        else _optional_env_path(CREDENTIALS_PATH_ENV),
    )


def check_mounts(paths: DeliveryPaths, probe=None) -> None:
    """Every persistent directory must be a mount point.

    A directory that exists only inside the container's writable layer looks
    healthy until the container is recreated, which is exactly when the
    operator's configuration, history and weights disappear. Fail at start-up
    instead.
    """
    probe = mount_probe if probe is None else probe  # type: ignore[assignment]
    unmounted = [
        f"{name} ({directory})"
        for name, directory in paths.persistent_dirs.items()
        if not probe(directory)
    ]
    if unmounted:
        raise DeliveryError(
            DELIVERY_DIR_NOT_MOUNTED,
            "以下持久化目录没有挂载，运行数据会在容器重建时丢失："
            + "、".join(unmounted)
            + "。请在 compose 或 docker run 中挂载这些目录。",
        )


def ensure_directories(paths: DeliveryPaths) -> list[Path]:
    """Create the persistent directories and return the ones created now."""
    created: list[Path] = []
    for name, directory in paths.persistent_dirs.items():
        if directory.is_dir():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DeliveryError(
                DELIVERY_DIR_NOT_WRITABLE,
                f"目录 {name} ({directory}) 无法创建，请检查挂载与权限。",
            ) from exc
        if not directory.is_dir():
            raise DeliveryError(
                DELIVERY_DIR_NOT_WRITABLE,
                f"目录 {name} ({directory}) 不是目录，请检查挂载。",
            )
        created.append(directory)
    return created


def check_writable(paths: DeliveryPaths, probe=None, dataset_probe=None) -> None:
    """The runtime user must be able to use every persistent directory.

    Every directory the Studio writes into must be writable. The dataset share is
    the one exception: it is an *input*. The Studio reads it (and copies it into
    its own snapshot) and writes nothing into it, so a dataset folder the
    operator mounted read-only is a correct setup, not a broken one — it only has
    to be readable and traversable.
    """
    if probe is None:
        def probe(path, mode):  # noqa: ARG001 - os.access needs the mode
            return os.access(path, os.W_OK)
    if dataset_probe is None:
        def dataset_probe(path):
            return os.access(path, os.R_OK | os.X_OK)

    # The exemption is decided by the path itself, not by the layout's label for
    # it, so renaming a key can never silently turn the dataset share back into a
    # directory the start demands to be writable.
    unusable = [
        f"{name} ({directory})"
        for name, directory in paths.persistent_dirs.items()
        if directory != paths.datasets_dir and not probe(directory, os.W_OK)
    ]
    if unusable:
        raise DeliveryError(
            DELIVERY_DIR_NOT_WRITABLE,
            "以下目录当前运行用户不可写，请检查挂载来源的权限："
            + "、".join(unusable),
        )

    if not dataset_probe(paths.datasets_dir):
        raise DeliveryError(
            DELIVERY_DATASETS_UNREADABLE,
            f"数据集目录 datasets ({paths.datasets_dir}) 当前运行用户不可读或不可遍历；"
            "数据集只要求可读，不需要可写，请检查挂载来源的读取权限。",
        )


def bootstrap_config(paths: DeliveryPaths, *, create: bool) -> bool:
    """Create the configuration from the sanitized template when it is absent.

    Returns whether the configuration was created. An existing configuration is
    never overwritten: the operator's file is the only source of truth.
    """
    if paths.config_path.is_file():
        return False
    if not create:
        raise DeliveryError(
            DELIVERY_CONFIG_MISSING,
            f"配置文件 {paths.config_path} 不存在；容器首次启动时由脱敏模板生成。",
        )
    if not paths.template_path.is_file():
        raise DeliveryError(
            DELIVERY_TEMPLATE_MISSING,
            f"脱敏配置模板 {paths.template_path} 不存在，无法生成配置。",
        )
    try:
        shutil.copyfile(paths.template_path, paths.config_path)
    except OSError as exc:
        raise DeliveryError(
            DELIVERY_DIR_NOT_WRITABLE,
            f"配置目录 {paths.config_dir} 无法写入，请检查该目录的挂载与权限。",
        ) from exc
    return True


def check_dependencies(finder=None) -> None:
    """Every controlled runtime component must be importable."""
    if finder is None:
        def finder(name):
            return importlib.util.find_spec(name)

    for capability, module in REQUIRED_MODULES.items():
        if finder(module) is None:
            raise DeliveryError(
                DELIVERY_DEPENDENCY_MISSING,
                f"缺少运行组件：{module}（{capability}）。"
                "请使用受控镜像，不要在容器内手动安装依赖。",
            )


def gpu_available() -> bool:
    """Whether a CUDA device is visible to the delivery's PyTorch build."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - a broken torch is "no GPU"
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def require_gpu(probe=None) -> None:
    """The formal Docker start-up always requests the GPU; there is no CPU mode."""
    probe = gpu_available if probe is None else probe
    if probe():
        return
    raise DeliveryError(
        DELIVERY_GPU_UNAVAILABLE,
        "未检测到可用的 NVIDIA GPU。正式 Docker 交付在启动时申请 GPU，"
        "不提供 CPU 训练回退；请确认宿主机 NVIDIA 驱动与容器 GPU 预留。",
    )


def offline_runtime_details() -> dict:
    """What the running interpreter can say about itself.

    Only facts that decide whether the private runtime is the locked one; every
    failure to read one (a broken torch, an unimportable component) is reported
    as a missing fact rather than raised, so the answer never carries a
    traceback into the operator's log.
    """
    facts: dict = {
        "executable": sys.executable or "",
        "python_version": platform.python_version(),
        "torch_version": "",
        "torchvision_version": "",
        "cuda_version": "",
        "cuda_available": False,
        "missing_modules": [],
    }
    try:
        import torch  # noqa: PLC0415 - the delivery's own component
    except Exception:  # noqa: BLE001 - a broken torch is a failed check, not a crash
        pass
    else:
        facts["torch_version"] = str(getattr(torch, "__version__", "") or "")
        facts["cuda_version"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        try:
            facts["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            facts["cuda_available"] = False
    try:
        import torchvision  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        pass
    else:
        facts["torchvision_version"] = str(getattr(torchvision, "__version__", "") or "")

    missing: list[str] = []
    for module in OFFLINE_MODULES:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except Exception:  # noqa: BLE001 - an unimportable package is a missing one
            missing.append(module)
    facts["missing_modules"] = missing
    return facts


def _same_executable(left: str, right: str) -> bool:
    """Two spellings of the same interpreter (case, separators, ``..``)."""
    def normalise(value: str) -> str:
        return os.path.normcase(os.path.abspath(value)).rstrip("\\/")

    return normalise(left) == normalise(right)


def check_offline_runtime(*, expect_python: str | None = None, details=None) -> None:
    """The private runtime must be the runtime the package pins.

    Interpreter, Python version, the CUDA PyTorch and Torchvision builds, the
    CUDA version they were compiled against, a visible GPU and the components
    the Studio imports. A CPU torch, another CUDA build or another interpreter
    is refused — the message never repeats a foreign path.
    """
    facts = offline_runtime_details() if details is None else details()

    executable = str(facts.get("executable") or "").strip()
    if not executable:
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            "无法确认私有解释器：当前运行的 Python 没有可用的解释器路径。",
        )
    if expect_python and not _same_executable(executable, expect_python):
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            "当前解释器不是交付安装的私有解释器，运行环境不可信。",
        )

    python_version = str(facts.get("python_version") or "")
    if not python_version.startswith(f"{PYTHON_VERSION}."):
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            f"私有运行环境不是 Python {PYTHON_VERSION}（当前 {python_version or '未知'}）。",
        )

    torch_version = str(facts.get("torch_version") or "")
    if torch_version != TORCH_VERSION:
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            f"torch 不是交付锁定的 {TORCH_VERSION}（当前 {torch_version or '未安装'}），"
            "不接受 CPU 版或其它来源的 torch。",
        )

    torchvision_version = str(facts.get("torchvision_version") or "")
    if torchvision_version != TORCHVISION_VERSION:
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            f"torchvision 不是交付锁定的 {TORCHVISION_VERSION}（当前 {torchvision_version or '未安装'}）。",
        )

    cuda_version = str(facts.get("cuda_version") or "")
    if cuda_version != CUDA_VERSION:
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            f"torch 不是 CUDA {CUDA_VERSION} 构建（当前 {cuda_version or '无'}）。",
        )

    if not facts.get("cuda_available"):
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            "私有运行环境看不到可用的 NVIDIA GPU。本产品不提供 CPU 回退，请检查显卡驱动。",
        )

    missing = [str(name) for name in (facts.get("missing_modules") or [])]
    if missing:
        raise DeliveryError(
            DELIVERY_RUNTIME_MISMATCH,
            "私有运行环境缺少交付锁定的运行组件：" + "、".join(missing) + "。",
        )


def _report(paths: DeliveryPaths, created: list[Path], *, bootstrap: bool) -> None:
    names = "、".join(paths.persistent_dirs)
    print(f"[delivery] 持久化目录就绪：{names}", flush=True)
    if created:
        print(f"[delivery] 新建目录 {len(created)} 个", flush=True)
    if bootstrap:
        print(f"[delivery] 配置文件：{paths.config_path}", flush=True)
    components = "、".join(sorted(set(REQUIRED_MODULES.values())))
    print(f"[delivery] 运行组件就绪：{components}", flush=True)


def run(argv=None) -> int:
    """Container start-up preflight. Returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m auto_tune.delivery.preflight",
        description="Auto-Tune Studio 交付启动预检（不修改任何训练语义）",
    )
    parser.add_argument("--require-mounts", action="store_true",
                        help="每个持久化目录都必须是挂载点，防止数据写入镜像层")
    parser.add_argument("--require-gpu", action="store_true",
                        help="必须有可用的 NVIDIA GPU，不做 CPU 回退")
    parser.add_argument("--require-offline-runtime", action="store_true",
                        help="私有运行环境必须是交付锁定的 CUDA PyTorch 运行环境（Windows 离线安装）")
    parser.add_argument("--expect-python", default="",
                        help="私有解释器的期望路径，与 --require-offline-runtime 一起使用")
    parser.add_argument("--bootstrap-config", action="store_true",
                        help="配置缺失时由脱敏模板生成（绝不覆盖已有配置）")
    args = parser.parse_args(argv)

    try:
        # The Windows installer asks whether the private runtime it has just
        # built is the locked CUDA one — at a point where the program payload,
        # the configuration and the persistent directories do not exist yet. It
        # is a question to the interpreter about itself, so this one request is
        # answered without resolving, creating or requiring the delivery layout.
        # Any other flag makes it an ordinary start-up preflight, whose full
        # directory and configuration rules stay exactly as they were.
        runtime_only = args.require_offline_runtime and not (
            args.require_mounts or args.require_gpu or args.bootstrap_config)
        if runtime_only:
            check_offline_runtime(expect_python=args.expect_python or None)
            print("[delivery] 私有运行环境是交付锁定的 CUDA PyTorch 运行环境。", flush=True)
            return 0

        paths = resolve_paths()
        if args.require_mounts:
            check_mounts(paths)
        created = ensure_directories(paths)
        check_writable(paths)
        bootstrap_config(paths, create=args.bootstrap_config)
        check_dependencies()
        if args.require_offline_runtime:
            check_offline_runtime(expect_python=args.expect_python or None)
        if args.require_gpu:
            require_gpu()
    except DeliveryError as exc:
        print(f"[delivery] ERROR {exc.code}: {exc.message}", file=sys.stderr,
              flush=True)
        return 1
    except ValueError as exc:
        print(f"[delivery] ERROR DELIVERY_CONFIG_INVALID: {exc}", file=sys.stderr,
              flush=True)
        return 1

    _report(paths, created, bootstrap=args.bootstrap_config)
    return 0


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
