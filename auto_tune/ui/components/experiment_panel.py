"""Unified experiment history display helpers (read-only adapter).

Persistence is owned by ExperimentHistoryStore and the training finalizer;
this adapter only reads and merges legacy tuning entries for display. Since
Studio S2 Core, the UI prefers the local SQLite index and honestly falls back
to the JSON adapter when the index is unavailable.
"""

from pathlib import Path

from auto_tune.modules.local_index import (
    ExperimentQuery,
    LocalIndexError,
)
from auto_tune.modules.local_index.projection import project_outward
from auto_tune.modules.train_analyzer.experiment_history import (
    ExperimentHistoryError,
    ExperimentHistoryStore,
)


def _redact_display_paths(exp: dict) -> dict:
    """Redact full business paths in a history projection for display.

    The UI may show only basenames plus artifact kind/status. The unified
    outward projection is applied recursively so nested paths under
    ``params``/``artifacts``/``decision``/``tuning`` can never leak either;
    ``audit_filename`` is kept as a basename convenience for the template.
    """
    redacted = project_outward(exp)
    audit_path = redacted.get("audit_path")
    if audit_path:
        redacted["audit_filename"] = audit_path
    return redacted


def get_experiment_history(log_dir: str = "log") -> list:
    """Load unified experiment history, newest first, merging legacy tuning.

    A corrupt history file is surfaced as an empty list so the UI never crashes;
    the store itself still raises ExperimentHistoryError to protect the file.
    Full business paths are redacted to basenames before display.
    """
    store = ExperimentHistoryStore(
        str(Path(log_dir) / "experiment_history.json"),
        legacy_tuning_path=str(Path(log_dir) / "tuning_history.json"),
    )
    try:
        experiments = store.list_experiments(include_legacy=True)
    except ExperimentHistoryError:
        return []
    return [_redact_display_paths(exp) for exp in experiments]


def _project_sqlite_experiments(items: list[dict]) -> list[dict]:
    """Map repository dicts to the template shape (UI-friendly keys)."""
    result = []
    for exp in items:
        exp = dict(exp)
        params = exp.get("params") or {}
        for meta_key, target in (
            ("_epochs", "epochs"),
            ("_tuning", "tuning"),
            ("_decision", "decision"),
            ("_probe_decision", "probe_decision"),
            ("_audit_path", "audit_path"),
        ):
            if target not in exp and params.get(meta_key) is not None:
                exp[target] = params.get(meta_key)
        artifacts = exp.pop("artifacts", None) or []
        report_path = next((a["path"] for a in artifacts if a["kind"] == "report"), None)
        run_dir = next((a["path"] for a in artifacts if a["kind"] == "run_dir"), None)
        exp["artifacts"] = {"report_path": report_path, "run_dir": run_dir}
        result.append(_redact_display_paths(exp))
    return result


def get_experiment_history_view(log_dir: str = "log", service=None) -> dict:
    """Return experiments for the history page: SQLite first, honest JSON fallback.

    A missing database is initialized on the fly; a corrupt or locked database
    falls back to the JSON adapter with a visible ``index_warning``. The warning
    never hides JSON-fallback records and never looks like an empty history.
    """
    warning = None
    if service is not None:
        try:
            service.initialize()
            # Server renders only the first page (matching the JS page size);
            # search/sort/filter/pagination are served by GET /api/experiments.
            experiments = service.list_experiments(ExperimentQuery(limit=25))
            return {
                "experiments": _project_sqlite_experiments(experiments),
                "source": "sqlite",
                "index_warning": None,
            }
        except (LocalIndexError, ValueError) as exc:
            warning = {
                "error_code": getattr(exc, "error_code", "LOCAL_INDEX_ERROR"),
                "message": "本地索引不可用，已回退 JSON 历史",
            }
    experiments = get_experiment_history(log_dir)
    return {
        "experiments": experiments,
        "source": "json_fallback",
        "index_warning": warning,
    }
