"""Resolve where the Studio listens and which configuration file it reads.

Windows desktop and the Linux container run the same FastAPI Studio. Only the
bind address, the port and the configuration file location may differ, and all
three are controlled exclusively through documented environment variables.
With no variables set the desktop defaults (``127.0.0.1:8000`` and the packaged
``auto_tune/config.yaml``) are preserved exactly.

Invalid environment values raise ``ValueError`` without echoing the raw value,
so a mistyped secret can never reach logs or the UI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

HOST_ENV = "AUTO_TUNE_HOST"
PORT_ENV = "AUTO_TUNE_PORT"
CONFIG_PATH_ENV = "AUTO_TUNE_CONFIG_PATH"

PORT_MIN = 1
PORT_MAX = 65535

PACKAGE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

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
