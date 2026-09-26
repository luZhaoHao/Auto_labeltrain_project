"""Delivery boundary shared by the Windows desktop and Linux container builds.

Locates the server bind address and the configuration file (``runtime``), and
validates the start-up preconditions of the delivery — persistent directories,
their mount contract, write permission, the configuration bootstrap, the
controlled runtime components and the required GPU (``preflight``). No
training, HPO, tuning, snapshot or persistence semantics live here.
"""
