"""Shared display vocabulary for experiment-management UI and future reports.

Maps internal field keys and enum values once to stable English translation
keys, then localizes through a caller-supplied translator (the same callable
that the UI template uses). The module is deliberately pure: it has no imports
and no dependency on any web framework, storage engine or the browser, so the
web UI and any future P5 report generator reuse the exact same labels.

Only user-visible field labels and stable enum labels are translated. Real
values (run_id, dataset_id, snapshot_id, run_name, model names, numbers, metric
names and stable error codes) are never dictionary keys and pass through the
fallback unchanged. ``None`` renders as the stable dash ``—``.
"""

# Internal field key -> stable English translation key. The actual zh/en text
# lives in auto_tune/ui/i18n.py; only the mapping is defined here, once.
FIELD_LABEL_KEYS: dict[str, str] = {
    "run_id": "Run ID",
    "run_name": "Run name",
    "dataset_id": "Dataset ID",
    "source": "Source",
    "status": "Run status",
    "phase": "Run phase",
    "model_name": "Model",
    "task_type": "Task type",
    "analysis_status": "Analysis status",
    "validation_status": "Validation status",
    "started_at": "Started at",
    "finished_at": "Finished at",
    "updated_at": "Updated at",
    "last_used_at": "Last used",
    "display_name": "Dataset name",
    "snapshot_id": "Snapshot ID",
    "kind": "Artifact type",
    "name": "Name",
    "exists_state": "Artifact availability",
    "diagnosis": "Diagnosis",
    "action": "Action",
    "guardrails_valid": "Guardrails valid",
    "probe_verdict": "Probe verdict",
    "audit": "Audit record",
    "valid": "Valid",
    "warnings": "Warnings",
    "clamped": "Clamped",
    "error_code": "Error code",
    "scanned": "Scanned",
    "issues": "Issues",
    "epochs": "Epochs",
    "dataset": "Dataset",
    "params": "Parameters",
    "metrics": "Metrics",
    "artifacts": "Artifacts",
    # Bugfix P5: report/audit read-only view fields.
    "duration": "Duration",
    "severity": "Severity",
    "issue_type": "Issue type",
    "description": "Description",
    "content_origin": "Content origin",
    "reference_run": "Reference run",
    "session_id": "Session ID",
    "total_iterations": "Total iterations",
    "best_iteration": "Best iteration",
    "termination_reason": "Termination reason",
    "metric_delta": "Metric changes",
    "suggested_parameters": "Suggested parameters",
    "guarded_parameters": "Guarded parameters",
    "executed_parameters": "Executed parameters",
    # Q1.2: semantic validation summary (status / reason / failed parameter).
    "semantic_validation": "Semantic validation",
    "semantic_reason": "Semantic reason",
    "semantic_parameter": "Semantic parameter",
}

# Stable enum value -> stable English translation key. Covers the real values
# used by the local-index domain (sources, S1.5 run statuses, run phases,
# analysis status, task type, dataset validation and artifact availability).
ENUM_LABEL_KEYS: dict[str, str] = {
    # experiment source
    "manual": "Manual training",
    "tuning": "Auto tuning",
    # run status (S1.5 seven states)
    "starting": "Starting",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "interrupted": "Interrupted",
    "unknown": "Unknown",
    # analysis status
    "pending": "Pending",
    "skipped": "Skipped",
    # run phase
    "preparing": "Preparing",
    "launching": "Launching",
    "training": "Training",
    "analyzing": "Analyzing",
    "finalizing": "Finalizing",
    "stopping": "Stopping",
    "terminal": "Terminal",
    # task type
    "detect": "Object detection",
    "classify": "Image classification",
    # dataset validation status
    "valid": "Valid",
    "invalid": "Invalid",
    "unverified": "Unverified",
    # artifact availability
    "exists": "Available",
    "missing": "Missing",
    "unregistered": "Unregistered",
    "unavailable": "Unavailable",
    # artifact kind
    "report": "Analysis report",
    "audit": "Audit record",
    "run_dir": "Training directory",
    "results_csv": "Training metrics",
    "args_yaml": "Training arguments",
    "best_pt": "Best weights",
    "last_pt": "Last weights",
    "manifest": "Artifact manifest",
    # index source
    "sqlite": "SQLite index",
    "json_fallback": "JSON fallback",
    # Bugfix P5: report/audit view enum values (analysis origin and final facts).
    "stored": "Stored",
    "generated": "Generated",
    "ok": "OK",
    # Bugfix P5 返修: training-issue type and severity enums.
    "overfitting": "Overfitting",
    "plateau": "Plateau",
    "unstable_training": "Unstable training",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

# Stable display booleans: Python True/False -> stable English translation key.
BOOLEAN_LABEL_KEYS: dict[str, str] = {
    "true": "Yes",
    "false": "No",
}


def experiment_field_label(key, translator) -> str:
    """Localized label for an internal field key.

    Unknown keys are returned unchanged (real field names, model names, run
    names, error codes etc. are never dictionary keys). ``None`` renders as the
    stable dash ``—``.
    """
    if key is None:
        return "—"
    stable = FIELD_LABEL_KEYS.get(key)
    if stable is None:
        return key
    return translator(stable)


def experiment_enum_label(value, translator):
    """Localized label for a stable enum value.

    Unknown values are returned unchanged. ``None`` renders as the stable dash
    ``—``.
    """
    if value is None:
        return "—"
    stable = ENUM_LABEL_KEYS.get(value)
    if stable is None:
        return value
    return translator(stable)


def experiment_boolean_label(value, translator):
    """Localized label for an explicit display boolean fact.

    Only real ``True``/``False``/``None`` are formatted (Yes/No/dash); any other
    value passes through unchanged so technical parameter tables never get
    rewritten.
    """
    if value is None:
        return "—"
    if value is True:
        return translator(BOOLEAN_LABEL_KEYS["true"])
    if value is False:
        return translator(BOOLEAN_LABEL_KEYS["false"])
    return value


def build_experiment_labels(translator) -> dict:
    """Pre-computed display vocabulary for one language.

    Returns ``{"fields": {...}, "enums": {...}, "booleans": {...}}`` where every
    value is already localized through ``translator``. JSON-serializable so the
    template can inject it once via ``tojson`` and the frontend helpers can read
    it.
    """
    return {
        "fields": {
            key: translator(stable) for key, stable in FIELD_LABEL_KEYS.items()
        },
        "enums": {
            value: translator(stable) for value, stable in ENUM_LABEL_KEYS.items()
        },
        "booleans": {
            key: translator(stable) for key, stable in BOOLEAN_LABEL_KEYS.items()
        },
    }
