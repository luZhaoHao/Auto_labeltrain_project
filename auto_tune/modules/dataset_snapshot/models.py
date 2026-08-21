"""Immutable dataset snapshot domain models and error contract (Studio S1.2)."""

from dataclasses import dataclass
from pathlib import Path


class SnapshotError(Exception):
    """Base error for dataset snapshot operations.

    Subclasses carry a stable ``error_code`` and HTTP ``status_code`` so the
    FastAPI layer can map failures without inspecting messages.
    """

    error_code = "SNAPSHOT_ERROR"
    status_code = 500

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


class SnapshotValidationError(SnapshotError):
    """Invalid parameters, paths, sample counts, or label formats."""

    error_code = "SNAPSHOT_VALIDATION_FAILED"
    status_code = 400


class SnapshotConflictError(SnapshotError):
    """Source changed during creation or a concurrent creation conflict."""

    error_code = "SNAPSHOT_CONFLICT"
    status_code = 409


class SnapshotInsufficientSpaceError(SnapshotError):
    """Not enough free disk space to materialize the snapshot."""

    error_code = "SNAPSHOT_INSUFFICIENT_SPACE"
    status_code = 507


class SnapshotIOError(SnapshotError):
    """Read, copy, write, or atomic-publish failure."""

    error_code = "SNAPSHOT_IO_FAILED"
    status_code = 500


@dataclass(frozen=True)
class SnapshotSample:
    """One image (and optional label) as recorded in the manifest.

    All path fields are POSIX-style relative strings; ``label_*`` fields are
    ``None`` for images that carry no label file.
    """

    source_image: str
    source_label: str | None
    snapshot_image: str
    snapshot_label: str | None
    split: str
    is_background: bool
    image_size: int
    label_size: int | None
    image_sha256: str
    label_sha256: str | None


@dataclass(frozen=True)
class DatasetSnapshot:
    """Immutable result of a dataset snapshot creation or validation."""

    schema_version: str
    snapshot_id: str
    snapshot_path: Path
    manifest_path: Path
    data_yaml_path: Path
    source_root: Path
    source_layout: str
    seed: int
    val_ratio: float
    train_count: int
    val_count: int
    background_count: int
    total_bytes: int
    manifest_digest: str
    reused: bool
    samples: tuple[SnapshotSample, ...] = ()
