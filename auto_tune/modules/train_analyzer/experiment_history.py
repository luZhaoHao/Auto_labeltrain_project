"""Unified experiment history store for manual and tuning training runs.

The store is atomic, idempotent (same run_id replaces the prior record), and
versioned. A corrupt or schema-invalid history file raises
ExperimentHistoryError so callers never silently overwrite a damaged store.
The legacy tuning_history.json is read-only for UI display and is never mutated.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
from typing import Any

from auto_tune.modules.agent_engine.audit import atomic_write_json

EMPTY_HISTORY = {"schema_version": "1.0", "experiments": []}

_ASIA_SHANGHAI = datetime.timezone(datetime.timedelta(hours=8))


class ExperimentHistoryError(Exception):
    """Raised when the history file exists but is corrupt or schema-invalid."""


def make_run_id(source: str, run_name: str, session_id: str | None = None) -> str:
    """Build a stable, unique run id for a training record."""
    if source == "manual":
        return f"manual:{run_name}"
    if source == "tuning" and session_id:
        return f"tuning:{session_id}:{run_name}"
    raise ValueError("tuning source requires session_id")


def _parse_sort_time(value: Any):
    """Parse finished_at into (valid, tz-aware datetime); naive times assume Asia/Shanghai."""
    if not value:
        return False, None
    text = str(value).strip()
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.datetime.fromisoformat(text)
        except ValueError:
            return False, None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ASIA_SHANGHAI)
    return True, dt


def _sort_key(record: dict):
    valid, dt = _parse_sort_time(record.get("finished_at"))
    fallback = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    return valid, (dt if dt is not None else fallback)


def _legacy_status(item: dict) -> str:
    error = item.get("error")
    if error:
        if "取消" in str(error):
            return "cancelled"
        return "failed"
    if item.get("train_name") or item.get("result_mAP50") is not None:
        return "completed"
    probe = item.get("probe_decision") or {}
    if probe.get("verdict") == "continue":
        return "completed"
    return "unknown"


def _legacy_analysis_status(item: dict) -> str:
    if item.get("result_mAP50") is not None or item.get("result_mAP50_95") is not None:
        return "completed"
    return "unknown"


def _legacy_metrics(item: dict) -> dict:
    mapping = {
        "result_mAP50": "mAP50",
        "result_mAP50_95": "mAP50_95",
        "result_precision": "precision",
        "result_recall": "recall",
    }
    metrics: dict[str, float] = {}
    for source, target in mapping.items():
        value = item.get(source)
        if value is not None:
            metrics[target] = value
    return metrics


def _legacy_to_experiments(legacy_path: str) -> list[dict]:
    """Map legacy tuning_history.json entries to display-only experiment records."""
    if not os.path.isfile(legacy_path):
        return []
    try:
        with open(legacy_path, encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    records: list[dict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        run_name = item.get("train_name")
        timestamp = item.get("timestamp") or item.get("finished_at") or ""
        if run_name:
            run_id = f"legacy-tuning:{run_name}"
        else:
            run_id = f"legacy-tuning:{index}:{timestamp}"
        records.append({
            "run_id": run_id,
            "source": "tuning",
            "run_name": run_name,
            "status": _legacy_status(item),
            "analysis_status": _legacy_analysis_status(item),
            "metrics": _legacy_metrics(item),
            "finished_at": timestamp,
            "decision": item.get("decision"),
            "probe_decision": item.get("probe_decision"),
            "_legacy": True,
        })
    return records


class ExperimentHistoryStore:
    """Atomic, idempotent, versioned store for unified training history."""

    def __init__(self, path: str, legacy_tuning_path: str | None = None) -> None:
        self.path = path
        self.legacy_tuning_path = legacy_tuning_path

    def load(self) -> dict:
        if not os.path.isfile(self.path):
            return copy.deepcopy(EMPTY_HISTORY)
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ExperimentHistoryError(f"corrupt experiment history: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
            raise ExperimentHistoryError("invalid experiment history schema")
        return data

    def list_experiments(self, include_legacy: bool = True) -> list[dict]:
        data = self.load()
        experiments = list(data.get("experiments", []))
        if include_legacy and self.legacy_tuning_path:
            legacy_items = _legacy_to_experiments(self.legacy_tuning_path)
            new_run_names = {e.get("run_name") for e in experiments if e.get("run_name")}
            legacy_items = [li for li in legacy_items if li.get("run_name") not in new_run_names]
            experiments = experiments + legacy_items
        experiments.sort(key=_sort_key, reverse=True)
        return experiments

    def upsert(self, record: dict) -> dict:
        if not record.get("run_id") or not record.get("source"):
            raise ValueError("record must contain run_id and source")
        data = self.load()
        experiments = [e for e in data["experiments"] if e.get("run_id") != record.get("run_id")]
        experiments.append(record)
        experiments.sort(key=_sort_key, reverse=True)
        data["experiments"] = experiments
        atomic_write_json(self.path, data)
        return copy.deepcopy(record)
