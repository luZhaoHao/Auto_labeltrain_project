"""Reference dataset resolution domain (Bugfix P2).

Deterministically binds a selected reference training run to the immutable
dataset snapshot it actually trained on, and refuses to fall back to the global
``latest_dataset``. Full identity is frozen before the tuning loop starts and
the same identity flows into commands, args.yaml, audit, JSON history and the
SQLite experiment association.
"""

from .models import (
    LocalIndexUnavailableError,
    ReferenceDatasetAmbiguousError,
    ReferenceDatasetError,
    ReferenceDatasetResolution,
    ReferenceDatasetUnresolvedError,
    ReferenceRunInvalidError,
    ReferenceSnapshotInvalidError,
)
from .service import resolve_reference_dataset

__all__ = [
    "LocalIndexUnavailableError",
    "ReferenceDatasetAmbiguousError",
    "ReferenceDatasetError",
    "ReferenceDatasetResolution",
    "ReferenceDatasetUnresolvedError",
    "ReferenceRunInvalidError",
    "ReferenceSnapshotInvalidError",
    "resolve_reference_dataset",
]
