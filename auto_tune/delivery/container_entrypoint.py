"""Container start-up: hand the mounted directories to ``studio``, then drop.

The Studio runs as the unprivileged ``studio`` account (UID/GID 10001), while a
bind-mount host directory that does not exist yet is created by the container
engine owned by ``root``. A first ``docker compose up -d`` would therefore hand
the application a directory it cannot write and the delivery preflight would
refuse to start — a start the operator is entitled to expect to work without
running any preparation script of their own.

This module is that one privileged step, and the only code in the product that
ever runs as root. It does five things, in this order:

1. it verifies this really is the container start-up: Linux, root, and a
   ``studio`` account that is actually 10001:10001, so a wrongly built image
   stops here instead of failing later as a puzzling permission error;
2. it verifies the declared browse roots are exactly the directory set the
   product accepts, so a value the picker would refuse stops the start instead
   of surfacing as a failed request after the operator clicks;
3. it resolves the delivery layout and refuses it unless it is exactly the
   declared persistent container paths — an environment variable can never make
   the privileged step touch a directory the product did not declare;
4. for each of those directories it creates a missing one (owned
   ``studio:studio``, mode 0755) and hands an existing one to ``studio`` — but
   only when the container engine left it owned by ``root``, which is what a
   *missing* bind directory looks like. A directory owned by any other account
   belongs to that account: it is never chowned or chmodded, and the start is
   refused with a clear reason. The dataset share is an input and is only ever
   checked for read and traverse permission. Every change is made to the
   directory inode alone: nothing is done recursively, nothing is done through a
   symlink, and no file inside those directories is read, moved, overwritten or
   deleted;
5. it drops to 10001:10001 with the standard library and verifies that the
   identity really changed;
6. it runs the shared delivery preflight *after* the drop and then ``exec``s the
   Studio, so the container's PID 1 is the Python process that receives SIGTERM.

Every failure is a stable code and a stopped start: there is no path that keeps
running as root, and no path that widens the permissions of an operator's data.
No configuration value, credential or host path is ever printed.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from .preflight import (
    CONTAINER_APP_ROOT,
    CONTAINER_DATASETS_DIR,
    DeliveryError,
    DeliveryPaths,
    resolve_paths,
)
from .runtime import CONTAINER_INPUT_ROOTS, resolve_input_allowed_roots

__all__ = [
    "ALLOWED_DIRECTORIES",
    "CONTAINER_DATASETS_UNREADABLE",
    "CONTAINER_DIR_NOT_CREATABLE",
    "CONTAINER_DIR_NOT_WRITABLE",
    "CONTAINER_DIR_PERMISSION_INSUFFICIENT",
    "CONTAINER_ENVIRONMENT_UNSUPPORTED",
    "CONTAINER_INPUT_ROOTS_INVALID",
    "CONTAINER_PATH_NOT_ALLOWED",
    "CONTAINER_PERMISSION_CHANGE_FAILED",
    "CONTAINER_PREFLIGHT_FAILED",
    "CONTAINER_PRIVILEGE_DROP_FAILED",
    "CONTAINER_ROOT_REQUIRED",
    "CONTAINER_TARGET_NOT_DIRECTORY",
    "CONTAINER_TARGET_SYMLINK",
    "CONTAINER_TARGET_UNAVAILABLE",
    "DIRECTORY_MODE",
    "ROOT_UID",
    "STUDIO_ACCOUNT",
    "STUDIO_GID",
    "STUDIO_UID",
    "build_preflight_argv",
    "build_studio_argv",
    "container_targets",
    "drop_privileges",
    "exec_studio",
    "initialise_directories",
    "main",
    "require_studio_identity",
    "run",
    "run_preflight",
    "studio_account",
    "verify_environment",
    "verify_input_roots",
]

# The account the Studio runs as. The Dockerfile creates it with exactly this
# UID/GID, and the drop below verifies it really is this pair.
STUDIO_ACCOUNT = "studio"
STUDIO_UID = 10001
STUDIO_GID = 10001

# The account the container engine leaves behind when it creates a *missing*
# bind directory. Only a directory owned by this account is handed over; any
# other owner is a host user's own directory and is reported instead.
ROOT_UID = 0

# The only mode this module ever sets, and only on a directory it created itself:
# owner-writable, group and others read/traverse, never a world-writable mode.
DIRECTORY_MODE = 0o755

# The persistent container paths this module may touch, and nothing else. They are
# the compose mount targets the shared preflight validates (see preflight.py);
# the root step is bound to this list rather than to whatever the environment
# happens to say.
ALLOWED_DIRECTORIES = (
    Path("/data/config"),
    Path("/data/secrets"),
    CONTAINER_DATASETS_DIR,
    CONTAINER_APP_ROOT / "log",
    CONTAINER_APP_ROOT / "detect",
    CONTAINER_APP_ROOT / "runs",
    CONTAINER_APP_ROOT / "models" / "weights",
)

CONTAINER_ENVIRONMENT_UNSUPPORTED = "CONTAINER_ENVIRONMENT_UNSUPPORTED"
CONTAINER_ROOT_REQUIRED = "CONTAINER_ROOT_REQUIRED"
CONTAINER_INPUT_ROOTS_INVALID = "CONTAINER_INPUT_ROOTS_INVALID"
CONTAINER_PATH_NOT_ALLOWED = "CONTAINER_PATH_NOT_ALLOWED"
CONTAINER_TARGET_UNAVAILABLE = "CONTAINER_TARGET_UNAVAILABLE"
CONTAINER_TARGET_SYMLINK = "CONTAINER_TARGET_SYMLINK"
CONTAINER_TARGET_NOT_DIRECTORY = "CONTAINER_TARGET_NOT_DIRECTORY"
CONTAINER_DIR_NOT_CREATABLE = "CONTAINER_DIR_NOT_CREATABLE"
CONTAINER_DIR_NOT_WRITABLE = "CONTAINER_DIR_NOT_WRITABLE"
CONTAINER_DIR_PERMISSION_INSUFFICIENT = "CONTAINER_DIR_PERMISSION_INSUFFICIENT"
CONTAINER_DATASETS_UNREADABLE = "CONTAINER_DATASETS_UNREADABLE"
CONTAINER_PERMISSION_CHANGE_FAILED = "CONTAINER_PERMISSION_CHANGE_FAILED"
CONTAINER_PRIVILEGE_DROP_FAILED = "CONTAINER_PRIVILEGE_DROP_FAILED"
CONTAINER_PREFLIGHT_FAILED = "CONTAINER_PREFLIGHT_FAILED"


def _os_attr(name: str, override: "Callable | None") -> "Callable | None":
    """The injected callable when there is one, else the platform's own.

    The lookups stay lazy because ``os.setuid`` does not exist on Windows: the
    module must import and its rules must be testable without a Linux container.
    """
    if override is not None:
        return override
    return getattr(os, name, None)


def _normalise(path) -> str:
    """Compare paths without following symlinks or normalising case."""
    return os.path.normpath(os.fspath(path))


def studio_account() -> tuple[int, int] | None:
    """The ``studio`` account's real ``(uid, gid)``, or ``None`` when unusable."""
    try:
        import pwd  # noqa: PLC0415 - Linux-only, and only for the container start-up
    except ImportError:
        return None
    try:
        entry = pwd.getpwnam(STUDIO_ACCOUNT)
    except KeyError:
        return None
    return entry.pw_uid, entry.pw_gid


def verify_environment(*, platform_name=None, geteuid=None, account=None) -> None:
    """The container start-up context, verified before anything is modified.

    A start-up outside the container (a developer's desktop, a wrongly built
    image, an operator override that removed the privileges this step needs) is
    refused instead of half-done.
    """
    name = sys.platform if platform_name is None else platform_name
    if not str(name).startswith("linux"):
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            "容器启动初始化只能在 Linux 容器内运行。",
        )

    getter = _os_attr("geteuid", geteuid)
    if getter is None:
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            "当前平台无法确认运行身份，容器启动初始化已停止。",
        )
    if getter() != 0:
        raise DeliveryError(
            CONTAINER_ROOT_REQUIRED,
            "容器启动初始化需要 root 完成一次性目录授权（随后立即降权），"
            "请勿在 compose 或 docker run 中用 user 覆盖启动用户。",
        )

    lookup = studio_account if account is None else account
    identity = lookup()
    if identity is None:
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            f"镜像中不存在运行用户 {STUDIO_ACCOUNT}，容器启动初始化已停止。",
        )
    if tuple(identity) != (STUDIO_UID, STUDIO_GID):
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            f"运行用户 {STUDIO_ACCOUNT} 不是交付声明的 "
            f"{STUDIO_UID}:{STUDIO_GID}，容器启动初始化已停止。",
        )


def verify_input_roots(*, resolver=None) -> None:
    """Refuse to start unless the browse roots are exactly the declared pair.

    The roots decide which directories an operator may point the product at, and
    the product itself refuses any directory that is not one of the declared
    ones. A value it would refuse has to stop the start here rather than surface
    later as a failed browse request; an absent value is a misconfiguration too,
    because both the compose file and this image's entrypoint always declare
    them, and it would leave the operator with a picker that lists nothing.

    The refusal never repeats the value: it is an environment value, and this
    module never prints one.
    """
    resolve = resolve_input_allowed_roots if resolver is None else resolver
    try:
        roots = resolve()
    except ValueError as exc:
        raise DeliveryError(
            CONTAINER_INPUT_ROOTS_INVALID,
            "容器浏览根目录不是交付声明的目录集合，已停止启动；"
            "请使用交付镜像声明的默认值。",
        ) from exc
    if roots is None:
        raise DeliveryError(
            CONTAINER_INPUT_ROOTS_INVALID,
            "容器未声明浏览根目录，已停止启动；"
            "请使用交付镜像声明的默认值。",
        )
    # The accepted set is checked here as well, not only inside the resolver:
    # the start-up refuses to hand a picker that can reach only one of the two
    # declared directories, whichever way the value was produced.
    accepted = {Path(root).as_posix() for root in roots}
    if accepted != set(CONTAINER_INPUT_ROOTS):
        raise DeliveryError(
            CONTAINER_INPUT_ROOTS_INVALID,
            "容器浏览根目录缺少交付声明的目录，已停止启动；"
            "请使用交付镜像声明的默认值。",
        )


def container_targets(layout: DeliveryPaths | None = None) -> tuple[Path, ...]:
    """The declared persistent directories, validated against the whitelist.

    The layout comes from the same controlled environment the preflight uses, so
    the privileged step and the checks that follow always talk about the same
    directories. Anything outside :data:`ALLOWED_DIRECTORIES` is refused rather
    than silently initialised.
    """
    resolved = resolve_paths() if layout is None else layout
    declared: list[Path] = [resolved.config_dir]
    if resolved.credentials_dir is not None:
        declared.append(resolved.credentials_dir)
    declared.extend((
        resolved.datasets_dir,
        resolved.app_root / "log",
        resolved.app_root / "detect",
        resolved.app_root / "runs",
        resolved.app_root / "models" / "weights",
    ))

    allowed = {_normalise(path) for path in ALLOWED_DIRECTORIES}
    seen: set[str] = set()
    for directory in declared:
        key = _normalise(directory)
        if key not in allowed or key in seen:
            # Two names for one directory would make the dataset share's
            # read-only contract apply to a directory the Studio writes into.
            raise DeliveryError(
                CONTAINER_PATH_NOT_ALLOWED,
                f"持久化目录 {directory} 不在交付声明的容器路径内；"
                "启动初始化只处理交付声明的目录，已停止启动。",
            )
        seen.add(key)
    return tuple(declared)


def initialise_directories(
    targets: Sequence[Path],
    *,
    datasets_directory: Path | None = None,
    uid: int = STUDIO_UID,
    gid: int = STUDIO_GID,
    probes=None,
) -> list[Path]:
    """Create the missing directories and hand the engine's own ones to ``studio``.

    ``probes`` injects the filesystem calls (``lstat``, ``mkdir``, ``chmod``,
    ``chown``) so these rules are exercised without a Linux container. Only the
    directory inode named in ``targets`` is ever touched: an entry inside one of
    those directories is never passed to a permission call.

    ``datasets_directory`` is the one target with a different contract: it is an
    input, so it is only ever checked for read and traverse permission. It is
    never chowned and never chmodded, whoever owns it.
    """
    if probes is None:
        probes = _FilesystemProbes()

    created: list[Path] = []
    for directory in targets:
        path = os.fspath(directory)
        dataset_share = (datasets_directory is not None
                         and _normalise(directory) == _normalise(datasets_directory))
        try:
            info = probes.lstat(path)
        except FileNotFoundError:
            _create_directory(directory, uid, gid, probes=probes,
                              hand_over=not dataset_share)
            created.append(directory)
            if dataset_share:
                _require_readable_datasets(directory, probes.lstat(path), uid, gid)
            continue
        except OSError as exc:
            raise DeliveryError(
                CONTAINER_TARGET_UNAVAILABLE,
                f"持久化目录 {directory} 无法读取，请检查挂载。",
            ) from exc

        if stat.S_ISLNK(info.st_mode):
            raise DeliveryError(
                CONTAINER_TARGET_SYMLINK,
                f"持久化目录 {directory} 是符号链接；不会透过链接修改权限，已停止启动。",
            )
        if not stat.S_ISDIR(info.st_mode):
            raise DeliveryError(
                CONTAINER_TARGET_NOT_DIRECTORY,
                f"持久化目录 {directory} 不是目录，无法挂载为持久化目录，已停止启动。",
            )

        # The dataset share is an input: readable and traversable is the whole
        # contract, and neither its ownership nor its mode is ever changed —
        # not by the privileged step, whatever its current owner is.
        if dataset_share:
            _require_readable_datasets(directory, info, uid, gid)
            continue

        info = _hand_directory_to_studio(directory, info, uid, gid, probes=probes)
        if not _directory_is_usable(info, uid, gid):
            raise DeliveryError(
                CONTAINER_DIR_NOT_WRITABLE,
                f"持久化目录 {directory} 交给运行用户后仍不可写或不可进入，"
                "请在宿主机上调整该目录本身的权限（其内容不会被修改）。",
            )

    return created


def _create_directory(directory: Path, uid: int, gid: int, *, probes,
                      hand_over: bool = True) -> None:
    """Create an absent directory, owned by the runtime user.

    The directory is empty by definition, so owning its own inode recursively
    changes nothing: there is no content to preserve or to widen. The dataset
    share is created without either call (`hand_over=False`): its contract is
    read and traverse, and its ownership is not this step's business.
    """
    path = os.fspath(directory)
    try:
        probes.mkdir(path, DIRECTORY_MODE)
    except OSError as exc:
        raise DeliveryError(
            CONTAINER_DIR_NOT_CREATABLE,
            f"持久化目录 {directory} 无法创建，请检查挂载与权限。",
        ) from exc
    if not hand_over:
        return
    try:
        probes.chmod(path, DIRECTORY_MODE)
        probes.chown(path, uid, gid)
    except OSError as exc:
        raise DeliveryError(
            CONTAINER_PERMISSION_CHANGE_FAILED,
            f"新建的持久化目录 {directory} 无法交给运行用户，已停止启动。",
        ) from exc


def _hand_directory_to_studio(directory: Path, info, uid: int, gid: int, *, probes):
    """Hand a directory the container engine created to ``studio``, inode only.

    Three cases, and only one of them changes anything:

    * the runtime user can already use it — nothing happens at all;
    * it is owned by ``root``, which is what ``docker compose up`` leaves behind
      when it creates a *missing* bind directory: its own inode is chowned so the
      Studio can use it;
    * it belongs to any other account — that is a host user's directory, so it is
      neither chowned nor chmodded, and the start is refused with a clear reason
      rather than taking the directory away from its owner.

    What is inside the directory is never touched in any of the three cases.
    """
    if _directory_is_usable(info, uid, gid):
        return info

    if info.st_uid != ROOT_UID:
        raise DeliveryError(
            CONTAINER_DIR_PERMISSION_INSUFFICIENT,
            f"持久化目录 {directory} 属于宿主机上的其它用户，运行用户无法使用；"
            "该目录及其中的文件都不会被改动。"
            "请在宿主机上授予运行用户写入权限后重试。",
        )

    path = os.fspath(directory)
    try:
        probes.chown(path, uid, gid)
    except OSError as exc:
        raise DeliveryError(
            CONTAINER_PERMISSION_CHANGE_FAILED,
            f"持久化目录 {directory} 无法交给运行用户；"
            "不会递归修改其中的文件，已停止启动。",
        ) from exc
    try:
        return probes.lstat(path)
    except OSError as exc:
        raise DeliveryError(
            CONTAINER_TARGET_UNAVAILABLE,
            f"持久化目录 {directory} 在授权后无法读取，已停止启动。",
        ) from exc


def _require_readable_datasets(directory: Path, info, uid: int, gid: int) -> None:
    """The dataset share is an input: it must be readable, and stays untouched."""
    if _permission_for(info, uid, gid,
                       stat.S_IRUSR | stat.S_IXUSR,
                       stat.S_IRGRP | stat.S_IXGRP,
                       stat.S_IROTH | stat.S_IXOTH):
        return
    raise DeliveryError(
        CONTAINER_DATASETS_UNREADABLE,
        f"数据集目录 {directory} 对运行用户不可读或不可遍历；"
        "数据集内容不会被递归改权，请在宿主机上授予其读取权限。",
    )


def _permission_for(info, uid: int, gid: int, owner, group, other) -> bool:
    """Whether an account's own permission bits include every requested bit.

    As root, ``os.access`` answers for root rather than for ``studio``, so the
    account's effective access is decided from the mode the file actually has.
    """
    if info.st_uid == uid:
        return (info.st_mode & owner) == owner
    if info.st_gid == gid:
        return (info.st_mode & group) == group
    return (info.st_mode & other) == other


def _directory_is_usable(info, uid: int, gid: int) -> bool:
    """A directory is usable with write *and* search permission, both needed.

    Write alone creates nothing that can be found again, and search alone cannot
    create anything, so a directory missing either is handed over as if it were
    unwritable and reported if that does not fix it.
    """
    return _permission_for(info, uid, gid,
                           stat.S_IWUSR | stat.S_IXUSR,
                           stat.S_IWGRP | stat.S_IXGRP,
                           stat.S_IWOTH | stat.S_IXOTH)


class _FilesystemProbes:
    """The real filesystem, behind the injectable interface the rules use."""

    def lstat(self, path: str):
        return os.lstat(path)

    def mkdir(self, path: str, mode: int) -> None:
        os.mkdir(path, mode)

    def chmod(self, path: str, mode: int) -> None:
        os.chmod(path, mode)

    def chown(self, path: str, uid: int, gid: int) -> None:
        os.chown(path, uid, gid)


def drop_privileges(
    *,
    uid: int = STUDIO_UID,
    gid: int = STUDIO_GID,
    setgroups=None,
    setgid=None,
    setuid=None,
    getuid=None,
    getgid=None,
) -> None:
    """Drop to the runtime account with the standard library, then verify it.

    Supplementary groups go first, while the process is still privileged enough
    to drop them. The identity is read back afterwards: a ``setuid`` that quietly
    did nothing would otherwise turn into business code running as root.
    """
    handlers = {
        "setgroups": _os_attr("setgroups", setgroups),
        "setgid": _os_attr("setgid", setgid),
        "setuid": _os_attr("setuid", setuid),
        "getuid": _os_attr("getuid", getuid),
        "getgid": _os_attr("getgid", getgid),
    }
    if any(handler is None for handler in handlers.values()):
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            "当前平台无法完成降权，已停止启动。",
        )

    try:
        handlers["setgroups"]([])
        handlers["setgid"](gid)
        handlers["setuid"](uid)
    except OSError as exc:
        raise DeliveryError(
            CONTAINER_PRIVILEGE_DROP_FAILED,
            f"无法降权到运行用户 {uid}:{gid}，已停止启动。",
        ) from exc

    if handlers["getuid"]() != uid or handlers["getgid"]() != gid:
        raise DeliveryError(
            CONTAINER_PRIVILEGE_DROP_FAILED,
            f"降权后的实际身份不是 {uid}:{gid}，已停止启动。",
        )


def require_studio_identity(*, geteuid=None) -> None:
    """Refuse to hand over to any business code from a privileged process."""
    getter = _os_attr("geteuid", geteuid)
    if getter is None:
        raise DeliveryError(
            CONTAINER_ENVIRONMENT_UNSUPPORTED,
            "当前平台无法确认运行身份，已停止启动。",
        )
    if getter() != STUDIO_UID:
        raise DeliveryError(
            CONTAINER_PRIVILEGE_DROP_FAILED,
            f"业务进程必须以运行用户 {STUDIO_UID} 启动，拒绝以 root 运行，已停止启动。",
        )


def build_preflight_argv() -> list[str]:
    """The delivery preflight the container start must pass, as a fixed array."""
    return [
        sys.executable,
        "-m",
        "auto_tune.delivery.preflight",
        "--require-mounts",
        "--require-gpu",
        "--bootstrap-config",
    ]


def run_preflight(*, runner=None, geteuid=None) -> None:
    """Run the shared preflight as the unprivileged user, before the Studio."""
    require_studio_identity(geteuid=geteuid)
    run_command = subprocess.run if runner is None else runner
    completed = run_command(build_preflight_argv(), check=False)
    if completed.returncode != 0:
        raise DeliveryError(
            CONTAINER_PREFLIGHT_FAILED,
            "交付启动预检未通过，已停止启动；不会以 root 继续运行。",
        )


def build_studio_argv() -> list[str]:
    """The Studio command, as a fixed array: no shell and no interpolated text."""
    return [sys.executable, "-m", "auto_tune.main"]


def exec_studio(*, execv=None, geteuid=None) -> None:
    """Replace this process with the Studio.

    ``execv`` keeps the process identity, so PID 1 stays a Python process and
    ``docker stop``/SIGTERM reaches the Studio's own signal handling instead of
    being swallowed by a shell parent.
    """
    require_studio_identity(geteuid=geteuid)
    replace = os.execv if execv is None else execv
    replace(sys.executable, build_studio_argv())


def run(argv=None) -> int:
    """The privileged initialisation, up to and including the preflight."""
    parser = argparse.ArgumentParser(
        prog="python -m auto_tune.delivery.container_entrypoint",
        description="Auto-Tune Studio 容器启动初始化（目录授权 → 降权 → 交付预检）",
    )
    # The container layout is fixed and comes from the controlled environment;
    # nothing here is taken from the command line or from user text.
    parser.parse_args(argv)

    try:
        verify_environment()
        verify_input_roots()
        layout = resolve_paths()
        targets = container_targets(layout)
        initialise_directories(targets, datasets_directory=layout.datasets_dir)
        drop_privileges()
        run_preflight()
    except DeliveryError as exc:
        print(f"[entrypoint] ERROR {exc.code}: {exc.message}",
              file=sys.stderr, flush=True)
        return 1
    except ValueError as exc:
        print(f"[entrypoint] ERROR DELIVERY_CONFIG_INVALID: {exc}",
              file=sys.stderr, flush=True)
        return 1

    print(f"[entrypoint] 持久化目录已就绪（{len(targets)} 个）；"
          f"运行用户 {STUDIO_UID}:{STUDIO_GID} 已确认。", flush=True)
    return 0


def main(argv=None) -> int:
    status = run(argv)
    if status != 0:
        return status
    try:
        exec_studio()
    except DeliveryError as exc:
        print(f"[entrypoint] ERROR {exc.code}: {exc.message}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
