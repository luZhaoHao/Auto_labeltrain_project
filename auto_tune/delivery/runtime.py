"""Resolve where the Studio listens, what it reads and what it may browse.

Windows desktop and the Linux container run the same FastAPI Studio. Only the
bind address, the port, the configuration file location and the directories the
folder pickers may offer differ, and all four are controlled exclusively through
documented environment variables. With no variables set the desktop defaults
(``127.0.0.1:8000``, the packaged ``auto_tune/config.yaml`` and the drives the
machine actually has) are preserved exactly.

Invalid environment values raise ``ValueError`` without echoing the raw value,
so a mistyped secret can never reach logs or the UI.

The module also owns the one platform decision the web server has to make
before its event loop exists: which asyncio loop the Windows socket layer can
serve connections on.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import re
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

HOST_ENV = "AUTO_TUNE_HOST"
PORT_ENV = "AUTO_TUNE_PORT"
CONFIG_PATH_ENV = "AUTO_TUNE_CONFIG_PATH"
INPUT_ALLOWED_ROOTS_ENV = "AUTO_TUNE_INPUT_ALLOWED_ROOTS"

PORT_MIN = 1
PORT_MAX = 65535

PACKAGE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# The container directories the folder pickers may be rooted at, and nothing
# else. This is a whitelist, not a list of wide directories to refuse: the
# dataset share is the input the product only ever reads, and ``detect`` is the
# directory ``find_detect_dir()`` really writes every training run into (logs,
# weights, the SQLite index and the retired ``runs`` mount stay unreachable).
CONTAINER_INPUT_ROOTS = ("/data/datasets", "/opt/auto-tune/detect")

_PORT_PATTERN = re.compile(r"[0-9]+")


def resolve_server_bind(
    default_host: str = DEFAULT_HOST,
    default_port: int = DEFAULT_PORT,
) -> tuple[str, int]:
    """Return the ``(host, port)`` the server should bind to.

    Environment variables take precedence over the caller's defaults so a
    container can override anything without a code change.
    """
    raw_host = os.environ.get(HOST_ENV)
    if raw_host is None:
        host = default_host
    else:
        host = raw_host.strip()
        if not host:
            raise ValueError(f"{HOST_ENV} must not be blank")

    raw_port = os.environ.get(PORT_ENV)
    if raw_port is None:
        port = default_port
    else:
        if _PORT_PATTERN.fullmatch(raw_port) is None:
            raise ValueError(_port_error())
        port = int(raw_port)
        if not PORT_MIN <= port <= PORT_MAX:
            raise ValueError(_port_error())

    return host, port


def resolve_config_path(default: Path) -> Path:
    """Return the configuration file path the product should read.

    Without :data:`CONFIG_PATH_ENV` the caller's default is returned unchanged,
    which keeps the packaged ``auto_tune/config.yaml`` location stable.
    """
    raw = os.environ.get(CONFIG_PATH_ENV)
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        raise ValueError(f"{CONFIG_PATH_ENV} must not be blank")

    expanded = os.path.expanduser(os.path.expandvars(value))
    return Path(expanded).resolve()


def _port_error() -> str:
    return f"{PORT_ENV} must be an integer between {PORT_MIN} and {PORT_MAX}"


def resolve_input_allowed_roots() -> tuple[Path, ...] | None:
    """The directories the folder pickers may browse, when the container set them.

    ``None`` means the variable is unset and the configuration file alone decides,
    which is the desktop behaviour. When it is set it *replaces* the configured
    roots: a container ships no browse root of its own, and the mounts it can see
    are the only directories an operator may point the product at.

    The value is a ``;``-separated list of *container* directories, and every
    entry is normalised and then compared against the declared set
    (:data:`CONTAINER_INPUT_ROOTS`): an entry that is not one of them — ``/etc``,
    ``/data/secrets``, the retired ``/opt/auto-tune/runs`` or anything else — is
    refused, so the accepted set can only ever be as wide as the delivery
    declared it. The normalised entries must then cover *all* of the declared
    roots: declaring only the dataset share (or only ``detect``, or one of them
    twice) leaves a picker that can never reach the other directory every
    training run is written into, so it is a misconfiguration rather than a
    narrower but working delivery. An empty list is refused for the same reason.

    The input order is free — the returned tuple always follows
    :data:`CONTAINER_INPUT_ROOTS`, so the picker's fixed order does not depend on
    how the container happened to spell the value. Invalid values raise
    ``ValueError`` without echoing the value, like the rest of this module.
    """
    raw = os.environ.get(INPUT_ALLOWED_ROOTS_ENV)
    if raw is None:
        return None
    if "\x00" in raw:
        raise ValueError(_input_roots_error())

    accepted: set[str] = set()
    for entry in raw.split(";"):
        value = entry.strip()
        if not value:
            continue
        accepted.add(_container_root(value))
    if accepted != set(CONTAINER_INPUT_ROOTS):
        raise ValueError(_input_roots_error())
    return tuple(Path(root) for root in CONTAINER_INPUT_ROOTS)


def _container_root(value: str) -> str:
    """One declared root, normalised, or ``ValueError`` if it is not declared."""
    if not value.startswith("/"):
        raise ValueError(_input_roots_error())
    # ``normpath`` collapses ``.``/``..``/duplicate separators, but POSIX keeps
    # exactly two leading slashes, so ``//data`` has to be folded onto ``/data``
    # for the comparison below to hold.
    normalised = posixpath.normpath(value)
    if normalised.startswith("//"):
        normalised = "/" + normalised.lstrip("/")
    if normalised not in CONTAINER_INPUT_ROOTS:
        raise ValueError(_input_roots_error())
    return normalised


def _input_roots_error() -> str:
    return (
        f"{INPUT_ALLOWED_ROOTS_ENV} must be exactly the declared container "
        f"directories ({';'.join(CONTAINER_INPUT_ROOTS)}), separated by ';'"
    )


def configure_platform_event_loop() -> bool:
    """Select the asyncio event loop the platform's socket layer can serve on.

    Windows defaults to the Proactor loop, which on some machines fails every
    incoming connection with ``WinError 10014`` ("an address of a wrong kind
    was supplied") even though the port is bound successfully. The selector
    loop serves those sockets instead, so it is selected explicitly.

    Must be called before anything creates or acquires the loop. It is a no-op
    everywhere but Windows, and returns whether a policy was selected so the
    decision is observable without inspecting process-global state.
    """
    if sys.platform != "win32":
        return False
    policy_class = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_class is None:
        return False
    asyncio.set_event_loop_policy(policy_class())
    return True
