"""Frozen resolution model and stable error contract (Bugfix P2).

``ReferenceDatasetResolution`` is the single frozen identity that the tuning
entry resolves and passes through the whole loop. Error classes carry stable
``error_code`` and ``status_code`` values so the API layer maps failures without
inspecting messages; messages never contain absolute paths, SQL, tracebacks or
native exception text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResolutionSource = Literal["sqlite", "reference_args"]


@dataclass(frozen=True)
class ReferenceDatasetResolution:
    """Deterministically resolved dataset identity for a reference training run.

    ``data_yaml_path`` is the single verified snapshot ``data.yaml`` the loop
    must train against. ``index_warning`` carries a stable, non-fatal backfill
    diagnostic when the SQLite index could not be updated.
    """

    reference_run: str
    dataset_id: str
    snapshot_id: str
    data_yaml_path: Path
    resolution_source: ResolutionSource
    dataset_display_name: str | None = None
    index_warning: str | None = None


class ReferenceDatasetError(Exception):
    """Base error for reference dataset resolution.

    Subclasses carry a stable ``error_code`` and HTTP ``status_code`` so the
    FastAPI layer can map failures without inspecting messages.
    """

    error_code = "REFERENCE_DATASET_ERROR"
    status_code = 400

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


class ReferenceRunInvalidError(ReferenceDatasetError):
    """The reference run directory or its required fact files are invalid."""

    error_code = "REFERENCE_RUN_INVALID"
    status_code = 400


class ReferenceDatasetUnresolvedError(ReferenceDatasetError):
    """No verifiable dataset association exists for the reference run."""

    error_code = "REFERENCE_DATASET_UNRESOLVED"
    status_code = 400


class ReferenceDatasetAmbiguousError(ReferenceDatasetError):
    """Same-name reference runs associate multiple different datasets."""

    error_code = "REFERENCE_DATASET_AMBIGUOUS"
    status_code = 409


class ReferenceSnapshotInvalidError(ReferenceDatasetError):
    """The snapshot or manifest is missing, corrupt, or escapes its root."""

    error_code = "REFERENCE_SNAPSHOT_INVALID"
    status_code = 400


class LocalIndexUnavailableError(ReferenceDatasetError):
    """Database is unavailable and the safe args.yaml fallback also failed."""

    error_code = "LOCAL_INDEX_UNAVAILABLE"
    status_code = 503
