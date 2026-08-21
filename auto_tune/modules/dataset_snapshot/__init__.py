"""Immutable dataset snapshot domain (Studio S1.2)."""

from .models import (
    DatasetSnapshot,
    SnapshotConflictError,
    SnapshotError,
    SnapshotInsufficientSpaceError,
    SnapshotIOError,
    SnapshotSample,
    SnapshotValidationError,
)
from .service import (
    SourceSample,
    SnapshotPlan,
    build_snapshot_plan,
    create_dataset_snapshot,
    discover_samples,
    required_snapshot_bytes,
    sha256_file,
    validate_dataset_snapshot,
)

__all__ = [
    "DatasetSnapshot",
    "SnapshotSample",
    "SnapshotError",
    "SnapshotValidationError",
    "SnapshotConflictError",
    "SnapshotInsufficientSpaceError",
    "SnapshotIOError",
    "SourceSample",
    "SnapshotPlan",
    "build_snapshot_plan",
    "create_dataset_snapshot",
    "discover_samples",
    "required_snapshot_bytes",
    "sha256_file",
    "validate_dataset_snapshot",
]
