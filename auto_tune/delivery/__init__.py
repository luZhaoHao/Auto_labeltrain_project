"""Delivery boundary shared by the Windows desktop and Linux container builds.

Locates the server bind address and the configuration file (``runtime``), and
validates the start-up preconditions of the delivery — persistent directories,
their mount contract, write permission, the configuration bootstrap, the
controlled runtime components and the required GPU (``preflight``). The Linux
container's one privileged step — handing the mounted directories to the
unprivileged runtime account, dropping to it, and only then running the preflight
and the Studio (``container_entrypoint``) — lives here too. No training, HPO,
tuning, snapshot or persistence semantics live here.
"""
