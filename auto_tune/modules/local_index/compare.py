"""Basic experiment comparison (Studio S2.4): 2-5 runs, one baseline.

Comparison is a pure fact projection: it never creates labels, baselines or
release markers, never stores management state, keeps missing metrics as null
(never 0), treats invalid/non-finite metric values as unavailable, marks
incomparable mixes with stable warnings, and returns only factual summaries
(highest metric / shortest duration / most complete artifacts). No "best model"
or auto-training recommendation is ever produced.
"""

from __future__ import annotations

import datetime
import json
import math

from .projection import is_path_key

COMPARE_MIN_RUNS = 2
COMPARE_MAX_RUNS = 5

_METRIC_KEYS = ("mAP50", "mAP50_95", "precision", "recall")

# Run-noise parameters folded out of the comparison (name/project/save_dir and
# underscore-prefixed internal metadata such as _epochs/_tuning/_decision).
# Path-typed parameters (``params.data`` etc.) are excluded via the shared
# ``is_path_key`` predicate, and nested path-typed keys are removed recursively.
_NOISE_PARAM_KEYS = frozenset({"name", "project", "save_dir", "project_name"})

# task_type -> the comparability metric whose availability confirms the metric
# scope. Empty/unknown/unrecognized tasks and tasks outside this matrix are not
# comparable.
_UNKNOWN_TASK_TYPES = frozenset({
    "", "unknown", "unrecognized", "unrecognised", "none", "null", "na",
    "n/a", "undefined", "other",
})
_SUPPORTED_TASK_METRICS: dict[str, str] = {"detect": "mAP50"}


def is_noise_param(key: str) -> bool:
    return str(key).startswith("_") or key in _NOISE_PARAM_KEYS or is_path_key(key)


def _is_finite_number(value) -> bool:
    """True only for finite int/float values (never bool/NaN/Inf/str/None)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _normalize_task_type(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in _UNKNOWN_TASK_TYPES:
        return None
    return text


def _metric_value(experiment: dict, key: str):
    value = (experiment.get("metrics") or {}).get(key)
    return value if _is_finite_number(value) else None


def _metric_invalid_count(experiments: list[dict]) -> int:
    """Count present-but-invalid metric values across every metric key."""
    invalid = 0
    for experiment in experiments:
        metrics = experiment.get("metrics") or {}
        for key in _METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            if not _is_finite_number(value):
                invalid += 1
    return invalid


def _duration_seconds(experiment: dict) -> int | None:
    start = experiment.get("started_at")
    finish = experiment.get("finished_at")
    if not start or not finish:
        return None
    try:
        start_dt = datetime.datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        finish_dt = datetime.datetime.fromisoformat(str(finish).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((finish_dt - start_dt).total_seconds()))


def _clean_value(value):
    """Recursively drop path-typed keys so a nested business path can never
    surface in a parameter diff (common or differences)."""
    if isinstance(value, dict):
        return {
            k: _clean_value(v) for k, v in value.items()
            if not is_path_key(k)
        }
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    return value


def _clean_params(experiment: dict) -> dict:
    raw = experiment.get("params") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: _clean_value(value)
        for key, value in raw.items()
        if not is_noise_param(key)
    }


def _param_diff(experiments: list[dict]) -> dict:
    cleaned = [_clean_params(e) for e in experiments]
    all_keys: set = set()
    for params in cleaned:
        all_keys |= set(params)
    common: dict = {}
    for key in sorted(all_keys):
        encoded = {
            json.dumps(params.get(key), ensure_ascii=False, default=str)
            for params in cleaned
        }
        if len(encoded) == 1:
            common[key] = cleaned[0].get(key)
    differences: dict = {}
    for run_id, params in zip((e["run_id"] for e in experiments), cleaned):
        differences[run_id] = {k: v for k, v in params.items() if k not in common}
    return {"common": common, "differences": differences}


def _is_comparable(experiments: list[dict]) -> tuple[bool, list[str]]:
    comparable = True
    warnings: list[str] = []
    task_types = {_normalize_task_type(e.get("task_type")) for e in experiments}
    if None in task_types:
        comparable = False
        warnings.append("experiment task type is empty, unknown or unrecognized")
    elif len(task_types) > 1:
        comparable = False
        warnings.append("experiments use different task types")
    else:
        task = next(iter(task_types))
        metric = _SUPPORTED_TASK_METRICS.get(task)
        if metric is None:
            comparable = False
            warnings.append(f"task type '{task}' is not supported for comparison")
        else:
            if _metric_invalid_count(experiments) > 0:
                comparable = False
                warnings.append("experiment metric values are not finite numbers")
            else:
                present = sum(
                    1 for e in experiments
                    if _metric_value(e, metric) is not None
                )
                if present < 2:
                    comparable = False
                    warnings.append(
                        f"fewer than two experiments share a comparable {metric} metric"
                    )
                elif present < len(experiments):
                    comparable = False
                    warnings.append("experiments have inconsistent metric availability")
    dataset_ids = {e.get("dataset_id") for e in experiments}
    if None in dataset_ids:
        comparable = False
        warnings.append("experiment dataset is unknown")
    elif len(dataset_ids) > 1:
        comparable = False
        warnings.append("experiments use different datasets")
    return comparable, warnings


def _artifact_score(experiment: dict) -> tuple[int, int]:
    rows = experiment.get("artifacts") or []
    exists = sum(1 for a in rows if a.get("exists_state") == "exists")
    return exists, len(rows)


def _factual_summary(experiments: list[dict]) -> dict | None:
    """Only factual highlights; never a "best model" conclusion."""
    with_map = [
        e for e in experiments
        if e.get("status") == "completed"
        and _metric_value(e, "mAP50") is not None
    ]
    with_duration = [(e, _duration_seconds(e)) for e in experiments]
    with_duration = [x for x in with_duration if x[1] is not None]
    return {
        "highest_mAP50": (
            max(with_map, key=lambda e: _metric_value(e, "mAP50"))["run_id"]
            if with_map else None
        ),
        "shortest_duration": (
            min(with_duration, key=lambda x: x[1])[0]["run_id"] if with_duration else None
        ),
        "most_complete_artifacts": (
            max(experiments, key=_artifact_score)["run_id"] if experiments else None
        ),
    }


def _identity(experiment: dict) -> dict:
    return {
        "run_id": experiment.get("run_id"),
        "run_name": experiment.get("run_name"),
        "source": experiment.get("source"),
        "status": experiment.get("status"),
        "model_name": experiment.get("model_name"),
        "task_type": experiment.get("task_type"),
        "dataset_id": experiment.get("dataset_id"),
        "started_at": experiment.get("started_at"),
        "finished_at": experiment.get("finished_at"),
    }


def _tuning_facts(experiment: dict) -> dict | None:
    if experiment.get("source") != "tuning":
        return None
    params = experiment.get("params") or {}
    tuning = params.get("_tuning")
    tuning = tuning if isinstance(tuning, dict) else {}
    decision = params.get("_decision")
    if not isinstance(decision, dict):
        decision = tuning.get("decision") if isinstance(tuning.get("decision"), dict) else {}
    guardrails = tuning.get("guardrails")
    return {
        "diagnosis": decision.get("diagnosis"),
        "action": decision.get("action"),
        "guardrails_valid": guardrails.get("valid") if isinstance(guardrails, dict) else None,
        "has_audit": bool(params.get("_audit_path")),
    }


def compare_experiments(experiments: list[dict], baseline_run_id: str) -> dict:
    """Project a comparison from fact rows; does not mutate any state."""
    run_ids = [e["run_id"] for e in experiments]
    comparable, warnings = _is_comparable(experiments)
    metrics: dict = {}
    relative: dict = {}
    baseline = next(e for e in experiments if e["run_id"] == baseline_run_id)
    for metric_key in _METRIC_KEYS:
        metrics[metric_key] = {
            e["run_id"]: _metric_value(e, metric_key) for e in experiments
        }
        base_value = _metric_value(baseline, metric_key)
        rel: dict = {}
        for e in experiments:
            value = _metric_value(e, metric_key)
            if value is None or base_value is None or base_value == 0:
                rel[e["run_id"]] = None
            else:
                rel[e["run_id"]] = (value - base_value) / base_value
        relative[metric_key] = rel

    training = {
        e["run_id"]: {
            "duration_seconds": _duration_seconds(e),
            "epochs": (e.get("params") or {}).get("_epochs") or {},
            "status": e.get("status"),
            "analysis_status": e.get("analysis_status"),
        }
        for e in experiments
    }
    artifacts = {
        e["run_id"]: {
            "exists": _artifact_score(e)[0],
            "total": _artifact_score(e)[1],
            "kinds": {a.get("kind"): a.get("exists_state") for a in (e.get("artifacts") or [])},
        }
        for e in experiments
    }
    return {
        "run_ids": run_ids,
        "baseline_run_id": baseline_run_id,
        "comparable": comparable,
        "warnings": warnings,
        "identity": [_identity(e) for e in experiments],
        "parameters": _param_diff(experiments),
        "metrics": metrics,
        "relative": relative,
        "training": training,
        "tuning": {e["run_id"]: _tuning_facts(e) for e in experiments},
        "artifacts": artifacts,
        "summary": _factual_summary(experiments) if comparable else None,
    }
