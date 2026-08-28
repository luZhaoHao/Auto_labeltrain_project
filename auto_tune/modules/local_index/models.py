"""Domain types and stable errors for the local SQLite stable index (Studio S2 Core).

SQLite is only a rebuildable query projection: JSON audit files, run-state files
and analysis reports remain the source of truth. Domain values are frozen so a
mistaken mutation cannot silently desynchronize the projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Domain whitelists shared by the service projection and the repository query.
EXPERIMENT_SOURCES = frozenset({"manual", "tuning"})
EXPERIMENT_STATUSES = frozenset({
    "starting",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "unknown",
})
EXPERIMENT_SORT_FIELDS = frozenset({
    "finished_at",
    "started_at",
    "updated_at",
    "name",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
})
EXPERIMENT_ORDERS = frozenset({"asc", "desc"})
MAX_QUERY_SEARCH_LENGTH = 200


@dataclass(frozen=True)
class LocalIndexConfig:
    """Resolved, validated local-index configuration."""

    database_path: Path
    backup_dir: Path
    backup_max_files: int = 3
    busy_timeout_ms: int = 5000


@dataclass(frozen=True)
class DatasetRecord:
    """Indexed projection of a dataset (snapshot or plain directory)."""

    dataset_id: str
    display_name: str
    canonical_path: str
    data_yaml_path: str | None = None
    snapshot_id: str | None = None
    snapshot_digest: str | None = None
    validation_status: str = "unverified"
    created_at: str | None = None
    updated_at: str | None = None
    last_used_at: str | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    """A report/product path reference belonging to one experiment."""

    run_id: str
    kind: str
    path: str
    exists_state: str = "unknown"
    created_at: str | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    """Indexed projection of one training/tuning run (S1.5 run_id is the key)."""

    run_id: str
    source: str
    run_name: str | None = None
    dataset_id: str | None = None
    status: str = "unknown"
    phase: str | None = None
    model_name: str | None = None
    task_type: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    analysis_status: str | None = None
    error: dict[str, Any] | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ExperimentQuery:
    """Stable, bounded query for the experiment list/pagination.

    Invalid filters, sort fields or ordering raise ``ValueError``; the API maps
    that to a 400 response. Sort/order come from backend whitelists only, so a
    client can never inject raw SQL.
    """

    dataset_id: str | None = None
    source: str | None = None
    status: str | None = None
    search: str | None = None
    sort: str = "finished_at"
    order: str = "desc"
    limit: int = 25
    offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or not (1 <= self.limit <= 100):
            raise ValueError("limit must be an integer in [1,100]")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if self.source is not None and self.source not in EXPERIMENT_SOURCES:
            raise ValueError("source must be 'manual' or 'tuning'")
        if self.status is not None and self.status not in EXPERIMENT_STATUSES:
            raise ValueError("status is not allowed")
        if self.sort not in EXPERIMENT_SORT_FIELDS:
            raise ValueError("sort field is not allowed")
        if self.order not in EXPERIMENT_ORDERS:
            raise ValueError("order must be 'asc' or 'desc'")
        if self.search is not None:
            if isinstance(self.search, bool) or not isinstance(self.search, str) or not self.search.strip():
                raise ValueError("search must be a non-empty string")
            if len(self.search) > MAX_QUERY_SEARCH_LENGTH:
                raise ValueError("search is too long")


@dataclass(frozen=True)
class ImportFailure:
    """One failed legacy-import file; message never leaks a full local path."""

    source_path: str
    error_code: str
    message: str


@dataclass(frozen=True)
class ImportSummary:
    """Bounded, idempotent result of scanning legacy JSON history files."""

    source_files: int
    imported: int
    skipped: int
    failed: int
    failures: tuple[ImportFailure, ...] = field(default_factory=tuple)


class LocalIndexError(Exception):
    """Base error for the local index domain."""

    error_code = "LOCAL_INDEX_ERROR"
    status_code = 500


class LocalIndexConfigError(LocalIndexError):
    """Invalid local-index configuration."""

    error_code = "LOCAL_INDEX_CONFIG_INVALID"
    status_code = 500


class LocalIndexCorruptError(LocalIndexError):
    """The database file exists but failed integrity / JSON decoding."""

    error_code = "LOCAL_INDEX_CORRUPT"
    status_code = 503


class LocalIndexMigrationError(LocalIndexError):
    """A schema migration failed and was rolled back."""

    error_code = "LOCAL_INDEX_MIGRATION_FAILED"
    status_code = 503


class LocalIndexPersistenceError(LocalIndexError):
    """A write/lock/query failure; the database is currently unavailable."""

    error_code = "LOCAL_INDEX_UNAVAILABLE"
    status_code = 503


class LocalIndexRebuildInProgress(LocalIndexError):
    """A reconciliation rebuild is already running (single-instance gate)."""

    error_code = "LOCAL_INDEX_REBUILD_IN_PROGRESS"
    status_code = 409


class LocalIndexReconcileError(LocalIndexError):
    """A bounded reconciliation operation failed without touching the facts."""

    error_code = "LOCAL_INDEX_RECONCILE_FAILED"
    status_code = 503


class LocalIndexCompareError(LocalIndexError):
    """Invalid comparison input (count/baseline/existence); never touches facts."""

    error_code = "LOCAL_INDEX_COMPARE_INVALID"
    status_code = 400


class LocalIndexNotFoundError(LocalIndexError):
    """A requested record does not exist in the projection."""

    error_code = "NOT_FOUND"
    status_code = 404


@dataclass(frozen=True)
class AuditIssue:
    """One bounded reconciliation finding; never carries a full local path."""

    code: str
    subject: str
    detail: str


@dataclass(frozen=True)
class AuditResult:
    """Read-only audit of the projection against the controlled fact files."""

    schema_version: int = 1
    timestamp: str = ""
    error_code: str | None = None
    scanned: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    issues: tuple[AuditIssue, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RebuildResult:
    """Outcome of an atomic index rebuild (backup -> temp -> publish).

    ``snapshot_unresolved`` counts experiments whose snapshot dataset identity
    could not be restored (missing/corrupt/oversized manifest); ``snapshot_issues``
    carries the bounded manifest scan status counts (valid/missing/corrupt/
    too_large/unreadable) plus an ``exceeded`` flag when the bounded scan
    capacity was hit.
    """

    schema_version: int = 1
    timestamp: str = ""
    error_code: str | None = None
    backup_created: bool = False
    datasets: int = 0
    experiments: int = 0
    artifacts: int = 0
    dataset_unresolved: int = 0
    snapshot_unresolved: int = 0
    snapshot_issues: dict = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of a bounded startup backfill over recent JSON facts."""

    schema_version: int = 1
    timestamp: str = ""
    error_code: str | None = None
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failures: int = 0
