"""Detail projection: experiment detail, dataset association, artifact manifest.

The detail endpoint returns facts (identity/source/status/times/model/task/
params/metrics/tuning/error) plus a controlled artifact manifest. The manifest
never echoes full business paths: it returns ``kind``/``name``/``status`` with
honest ``exists``/``missing``/``unregistered``/``unavailable`` states, so the UI
never builds a path from client input and the detail API never reads arbitrary
local paths.
"""

from __future__ import annotations

import os

# (kind, 展示名, 相对受控 run_dir 的路径)。标准 YOLO Detect 运行把 results.csv /
# args.yaml 放在 run_dir 根，权重放在 run_dir/weights/ 下；权重不能按根目录文件判断。
_ARTIFACT_DERIVED_FILES = (
    ("results_csv", "results.csv", "results.csv"),
    ("args_yaml", "args.yaml", "args.yaml"),
    ("best_pt", "best.pt", os.path.join("weights", "best.pt")),
    ("last_pt", "last.pt", os.path.join("weights", "last.pt")),
)

_STORED_KINDS = (
    ("report", "report.json"),
    ("audit", "audit.json"),
    ("run_dir", "run_dir"),
)


def _sanitize_value(value):
    """Redact path-like values to their basename (never the full path).

    Both absolute paths and relative paths that contain separators are reduced
    to their basename, so a business path can never leak through the detail
    projection (e.g. ``params.data``, ``report_path``, ``run_dir``).
    """
    if isinstance(value, str) and (os.path.isabs(value) or "/" in value or "\\" in value):
        return os.path.basename(value) or "…"
    return value


def _sanitize_params(params: dict) -> dict:
    return {str(key): _sanitize_value(value) for key, value in (params or {}).items()}


def _path_state(path) -> str:
    """Honest availability state for a controlled registered path."""
    if not path:
        return "unregistered"
    try:
        return "exists" if os.path.exists(path) else "missing"
    except OSError:
        return "unavailable"


def _manifest_state(dataset: dict | None) -> str:
    """Snapshot manifest availability derived from the registered data.yaml."""
    if not dataset:
        return "unregistered"
    data_yaml = dataset.get("data_yaml_path")
    if not data_yaml or os.path.basename(data_yaml) != "data.yaml" or not dataset.get("snapshot_id"):
        return "unregistered"
    manifest_path = os.path.join(os.path.dirname(data_yaml), "manifest.json")
    return _path_state(manifest_path)


def _safe_name(path, fallback: str) -> str | None:
    if not path:
        return None
    return os.path.basename(path)


def build_artifact_manifest(experiment: dict, dataset: dict | None) -> list[dict]:
    """Build the bounded artifact manifest for one experiment.

    ``experiment`` must be a repository projection (with ``artifacts`` as a list
    of ``{kind, path, exists_state, created_at}``). Only registered paths are
    consulted; the response carries kind/name/status and never a full path.
    """
    stored = {}
    for artifact in experiment.get("artifacts") or []:
        stored[artifact.get("kind")] = artifact
    run_dir = stored.get("run_dir", {}).get("path")
    manifest: list[dict] = []
    for kind, fallback in _STORED_KINDS:
        path = stored.get(kind, {}).get("path")
        manifest.append({
            "kind": kind,
            "name": _safe_name(path, fallback),
            "status": _path_state(path),
        })
    for kind, name, relpath in _ARTIFACT_DERIVED_FILES:
        if run_dir:
            manifest.append({
                "kind": kind,
                "name": name,
                "status": _path_state(os.path.join(run_dir, relpath)),
            })
        else:
            manifest.append({"kind": kind, "name": name, "status": "unregistered"})
    manifest.append({
        "kind": "manifest",
        "name": "manifest.json",
        "status": _manifest_state(dataset),
    })
    return manifest


def _dataset_summary(dataset: dict | None) -> dict | None:
    if not dataset:
        return None
    return {
        "dataset_id": dataset.get("dataset_id"),
        "display_name": dataset.get("display_name"),
        "snapshot_id": dataset.get("snapshot_id"),
        "validation_status": dataset.get("validation_status"),
        "last_used_at": dataset.get("last_used_at"),
        "data_yaml_name": os.path.basename(dataset["data_yaml_path"])
        if dataset.get("data_yaml_path") else None,
    }


def _tuning_summary(experiment: dict) -> dict | None:
    if experiment.get("source") != "tuning":
        return None
    params = experiment.get("params") or {}
    tuning = params.get("_tuning")
    tuning = tuning if isinstance(tuning, dict) else {}
    decision = params.get("_decision")
    if decision is None:
        decision = tuning.get("decision")
    audit_path = params.get("_audit_path")
    return {
        "decision": decision,
        "probe_decision": params.get("_probe_decision"),
        "guardrails": tuning.get("guardrails"),
        "audit_filename": os.path.basename(str(audit_path)) if audit_path else None,
    }


def build_experiment_detail(experiment: dict, dataset: dict | None) -> dict:
    """Project one experiment into the extended, sanitized detail shape."""
    params = experiment.get("params") or {}
    return {
        "run_id": experiment.get("run_id"),
        "source": experiment.get("source"),
        "run_name": experiment.get("run_name"),
        "status": experiment.get("status"),
        "phase": experiment.get("phase"),
        "model_name": experiment.get("model_name"),
        "task_type": experiment.get("task_type"),
        "started_at": experiment.get("started_at"),
        "finished_at": experiment.get("finished_at"),
        "updated_at": experiment.get("updated_at"),
        "analysis_status": experiment.get("analysis_status"),
        "params": _sanitize_params(params),
        "metrics": experiment.get("metrics") or {},
        "error": experiment.get("error"),
        "epochs": params.get("_epochs"),
        "dataset": _dataset_summary(dataset),
        "tuning": _tuning_summary(experiment),
        "artifacts": build_artifact_manifest(experiment, dataset),
    }


def _best_candidates(experiments: list[dict], dataset_id: str) -> list[dict]:
    return [
        exp for exp in experiments
        if exp.get("dataset_id") == dataset_id
        and exp.get("status") == "completed"
        and (exp.get("metrics") or {}).get("mAP50") is not None
    ]


def compute_best_experiment(experiments: list[dict], dataset_id: str) -> dict | None:
    """Best completed experiment by mAP50 within one dataset (same task scope).

    Only completed runs that carry ``mAP50`` compete. A single best is only
    determinable when all candidates share one ``task_type``; when tasks are
    mixed the single best is undefined and ``None`` is returned. Per-task bests
    are available via ``group_best_by_task``.
    """
    candidates = _best_candidates(experiments, dataset_id)
    if not candidates:
        return None
    tasks = {exp.get("task_type") or "unknown" for exp in candidates}
    if len(tasks) > 1:
        return None
    best = max(candidates, key=lambda exp: float((exp.get("metrics") or {}).get("mAP50")))
    return {
        "metric": "mAP50",
        "value": best["metrics"]["mAP50"],
        "run_id": best["run_id"],
        "run_name": best.get("run_name"),
        "task_type": best.get("task_type"),
    }


def group_best_by_task(experiments: list[dict], dataset_id: str) -> dict:
    """Per-task best (mAP50) facts for a dataset, keyed by task type.

    The dataset best fact is grouped by task and metric scope so a Detect run
    and a Classify run never produce one mixed "best experiment" conclusion.
    """
    groups: dict[str, list[dict]] = {}
    for exp in _best_candidates(experiments, dataset_id):
        groups.setdefault(exp.get("task_type") or "unknown", []).append(exp)
    result: dict = {}
    for task, group in groups.items():
        best = max(group, key=lambda e: float((e.get("metrics") or {}).get("mAP50")))
        result[task] = {
            "metric": "mAP50",
            "value": best["metrics"]["mAP50"],
            "run_id": best["run_id"],
            "run_name": best.get("run_name"),
        }
    return result


def _recent_projection(experiments: list[dict]) -> list[dict]:
    return [{
        "run_id": exp.get("run_id"),
        "run_name": exp.get("run_name"),
        "source": exp.get("source"),
        "status": exp.get("status"),
        "finished_at": exp.get("finished_at"),
        "model_name": exp.get("model_name"),
        "task_type": exp.get("task_type"),
        "mAP50": (exp.get("metrics") or {}).get("mAP50"),
    } for exp in experiments]


def build_dataset_experiments(
    dataset: dict, experiment_count: int, recent_experiments: list[dict], best: dict | None,
    best_by_task: dict | None = None,
) -> dict:
    """Project the dataset association summary (no full business paths).

    ``best`` is the single determinable best (None when mixed tasks make it
    undefined); ``best_by_task`` carries the per-task/metric-scope grouping.
    """
    return {
        "dataset_id": dataset.get("dataset_id"),
        "display_name": dataset.get("display_name"),
        "snapshot_id": dataset.get("snapshot_id"),
        "validation_status": dataset.get("validation_status"),
        "last_used_at": dataset.get("last_used_at"),
        "experiment_count": experiment_count,
        "best": best,
        "best_by_task": best_by_task or {},
        "recent_experiments": _recent_projection(recent_experiments),
    }
