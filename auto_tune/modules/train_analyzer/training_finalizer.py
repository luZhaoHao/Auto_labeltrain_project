"""Shared training finalizer: Module B analysis, KPI extraction and history.

Both manual training and auto-tuning funnel their post-training work through
``finalize_training_run()`` so a single code path produces the report, KPIs and
unified experiment history. Training status and analysis status are kept
separate: an analysis failure never overwrites a successful training fact.
"""

from __future__ import annotations

import os

import yaml

from .analyzer import analyze_training_results
from .experiment_history import ExperimentHistoryStore, make_run_id
from auto_tune.modules.agent_engine.audit import atomic_write_json, utc_now_iso

_METRIC_MAP = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


def _error(stage: str, error_type: str, message: str) -> dict:
    return {"stage": stage, "error_type": error_type, "message": message, "timestamp": utc_now_iso()}


def _extract_metrics(final_metrics: dict) -> dict:
    metrics = {}
    for source_key, target in _METRIC_MAP.items():
        value = final_metrics.get(source_key)
        if value is not None:
            metrics[target] = value
    return metrics


def _load_params(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, "args.yaml"), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _analyze_completed_run(run_dir: str, run_name: str, config: dict, log_dir: str, record: dict) -> None:
    """Run Module B analysis and fill metrics/analysis status on the record."""
    detect_dir = os.path.dirname(run_dir)
    report = analyze_training_results(
        detect_dir, config, run_name=run_name, enable_llm=False, enable_vision=False
    )
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, f"{run_name}_report.json")
    atomic_write_json(report_path, report)
    record["artifacts"]["report_path"] = report_path

    if report.get("error"):
        record["analysis_status"] = "failed"
        record["analysis_error"] = _error("analysis", "analysis_failed", report["error"])
        return

    run_data = report.get("runs", {}).get(run_name, {})
    results = run_data.get("results", {}) or {}
    if "error" in results:
        record["analysis_status"] = "failed"
        record["analysis_error"] = _error("analysis", "analysis_failed", results["error"])
        return

    record["metrics"] = _extract_metrics(results.get("final_metrics", {}) or {})
    record["epochs"] = {
        "configured": record["params"].get("epochs"),
        "completed": results.get("total_epochs"),
        "best": results.get("best_epoch"),
    }
    if not record["metrics"]:
        record["analysis_status"] = "failed"
        record["analysis_error"] = _error("analysis", "analysis_failed", "no metric columns in results.csv")
        return
    record["analysis_status"] = "completed"


def finalize_training_run(
    run_dir: str,
    run_name: str,
    source: str,
    config: dict,
    log_dir: str = "log",
    training_status: str = "completed",
    session_id: str | None = None,
    audit_path: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    training_error: dict | None = None,
    tuning_context: dict | None = None,
) -> dict:
    """Finalize a training run: analyze, extract KPIs and persist unified history.

    ``tuning_context`` carries structured auto-tuning facts (decision diagnosis,
    action, parameter changes, guardrail outcomes). It is only persisted for
    tuning runs; manual runs pass ``None`` and never gain tuning fields.
    """
    params = _load_params(run_dir)
    record: dict = {
        "run_id": make_run_id(source, run_name, session_id),
        "run_name": run_name,
        "source": source,
        "status": training_status,
        "analysis_status": "skipped",
        "started_at": started_at,
        "finished_at": finished_at,
        "params": params,
        "metrics": {},
        "epochs": {
            "configured": params.get("epochs"),
            "completed": None,
            "best": None,
        },
        "artifacts": {"report_path": None, "run_dir": run_dir},
        "audit_path": audit_path,
        "analysis_error": None,
        "history_error": None,
        "error": training_error,
    }
    if source == "tuning" and tuning_context:
        record["tuning"] = tuning_context

    if training_status == "completed":
        record["analysis_status"] = "pending"
        try:
            _analyze_completed_run(run_dir, run_name, config, log_dir, record)
        except Exception as exc:
            record["analysis_status"] = "failed"
            record["analysis_error"] = _error("analysis", "analysis_failed", str(exc))

    try:
        store = ExperimentHistoryStore(os.path.join(log_dir, "experiment_history.json"))
        store.upsert(record)
    except Exception as exc:
        record["history_error"] = _error("history", "history_persistence_error", str(exc))

    return record
