"""Shared display vocabulary for experiment-management UI and future reports.

Bugfix P4: field keys and enum values map once to stable translation keys;
the actual zh/en text stays in ``auto_tune.ui.i18n`` and is resolved through a
caller-supplied translator. The web UI and any future P5 report generator call
the same public interface below.
"""

from .experiment_labels import (
    BOOLEAN_LABEL_KEYS,
    ENUM_LABEL_KEYS,
    FIELD_LABEL_KEYS,
    build_experiment_labels,
    experiment_boolean_label,
    experiment_enum_label,
    experiment_field_label,
)
from .experiment_views import (
    ArtifactIdentityMismatchError,
    ArtifactTooLargeError,
    ArtifactUnavailableError,
    AuditInvalidError,
    AuditNotAvailableError,
    ExperimentNotFoundError,
    ExperimentViewError,
    ReportInvalidError,
    ReportNotAvailableError,
    build_audit_view,
    build_report_view,
)

__all__ = [
    "ArtifactIdentityMismatchError",
    "ArtifactTooLargeError",
    "ArtifactUnavailableError",
    "AuditInvalidError",
    "AuditNotAvailableError",
    "BOOLEAN_LABEL_KEYS",
    "ENUM_LABEL_KEYS",
    "ExperimentNotFoundError",
    "ExperimentViewError",
    "FIELD_LABEL_KEYS",
    "ReportInvalidError",
    "ReportNotAvailableError",
    "build_audit_view",
    "build_experiment_labels",
    "build_report_view",
    "experiment_boolean_label",
    "experiment_enum_label",
    "experiment_field_label",
]
