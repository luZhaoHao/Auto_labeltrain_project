"""Studio S2 Core: local SQLite stable index for datasets and experiments.

SQLite is a rebuildable query projection over the JSON/audit fact files. It
never replaces them and never stores dataset binaries, weights, logs or secrets.
"""

from .database import (
    check_database_integrity,
    connect_database,
    create_backup,
    initialize_database,
    load_local_index_config,
    transaction,
)
from .compare import COMPARE_MAX_RUNS, COMPARE_MIN_RUNS, compare_experiments, is_noise_param
from .reconciliation import (
    MAX_BACKFILL_RECORDS,
    MAX_ISSUES,
    audit_index,
    backfill_startup,
    rebuild_index,
    rebuild_lock_path,
)
from .repository import LocalIndexRepository
from .service import LocalIndexService
from .models import (
    ArtifactRecord,
    AuditIssue,
    AuditResult,
    BackfillResult,
    DatasetRecord,
    ExperimentQuery,
    ExperimentRecord,
    ImportFailure,
    ImportSummary,
    LocalIndexCompareError,
    LocalIndexConfig,
    LocalIndexConfigError,
    LocalIndexCorruptError,
    LocalIndexError,
    LocalIndexMigrationError,
    LocalIndexNotFoundError,
    LocalIndexPersistenceError,
    LocalIndexRebuildInProgress,
    LocalIndexReconcileError,
    RebuildResult,
)

__all__ = [
    "ArtifactRecord",
    "AuditIssue",
    "AuditResult",
    "BackfillResult",
    "COMPARE_MAX_RUNS",
    "COMPARE_MIN_RUNS",
    "DatasetRecord",
    "ExperimentQuery",
    "ExperimentRecord",
    "ImportFailure",
    "ImportSummary",
    "LocalIndexCompareError",
    "LocalIndexConfig",
    "LocalIndexConfigError",
    "LocalIndexCorruptError",
    "LocalIndexError",
    "LocalIndexMigrationError",
    "LocalIndexNotFoundError",
    "LocalIndexPersistenceError",
    "LocalIndexRebuildInProgress",
    "LocalIndexReconcileError",
    "LocalIndexRepository",
    "LocalIndexService",
    "MAX_BACKFILL_RECORDS",
    "MAX_ISSUES",
    "RebuildResult",
    "audit_index",
    "backfill_startup",
    "check_database_integrity",
    "compare_experiments",
    "connect_database",
    "create_backup",
    "initialize_database",
    "is_noise_param",
    "load_local_index_config",
    "rebuild_index",
    "rebuild_lock_path",
    "transaction",
]
