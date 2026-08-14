"""Unified experiment history display helpers (read-only adapter).

Persistence is owned by ExperimentHistoryStore and the training finalizer;
this adapter only reads and merges legacy tuning entries for display.
"""

import os
from pathlib import Path

from auto_tune.modules.train_analyzer.experiment_history import (
    ExperimentHistoryError,
    ExperimentHistoryStore,
)


def get_experiment_history(log_dir: str = "log") -> list:
    """Load unified experiment history, newest first, merging legacy tuning.

    A corrupt history file is surfaced as an empty list so the UI never crashes;
    the store itself still raises ExperimentHistoryError to protect the file.
    """
    store = ExperimentHistoryStore(
        str(Path(log_dir) / "experiment_history.json"),
        legacy_tuning_path=str(Path(log_dir) / "tuning_history.json"),
    )
    try:
        experiments = store.list_experiments(include_legacy=True)
    except ExperimentHistoryError:
        return []
    # Derive a traversal-safe audit filename for the template's audit entry link.
    for exp in experiments:
        audit_path = exp.get("audit_path")
        if audit_path:
            exp["audit_filename"] = os.path.basename(audit_path)
    return experiments
