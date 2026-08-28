"""Bugfix P5: bounded, read-only, run_id-bound report/audit view projections.

Pure Python (no web framework, no storage engine, no network). The web UI
consumes these view models; any future P5 report generator reuses the same
builders.

Only *registered* artifacts (bound to a run_id in the local index) are read:
paths are never client-supplied, never pattern-matched by name, and must stay
inside the controlled artifact roots. Reads are chunked and hard-capped (1 MiB chunks,
16 MiB max, stops when the stream grows past the cap mid-read). The JSON root
must be an object; a minimal schema check then binds the content to the
experiment (report run_name match / audit session reconciliation). Any identity
conflict or unsafe path is a stable error, never guessed content.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
MAX_AUDIT_ITERATIONS = 50

# Only these core parameters are projected into the report view.
REPORT_PARAMETER_KEYS = ("model", "epochs", "batch", "imgsz", "optimizer", "lr0", "device")

_METRIC_KEYS = ("mAP50", "mAP50_95", "precision", "recall")
_METRIC_MAP = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


class ExperimentViewError(Exception):
    """Base error for the P5 report/audit view projection."""

    error_code = "EXPERIMENT_VIEW_ERROR"
    status_code = 500


class ExperimentNotFoundError(ExperimentViewError):
    error_code = "EXPERIMENT_NOT_FOUND"
    status_code = 404


class ReportNotAvailableError(ExperimentViewError):
    error_code = "REPORT_NOT_AVAILABLE"
    status_code = 404


class AuditNotAvailableError(ExperimentViewError):
    error_code = "AUDIT_NOT_AVAILABLE"
    status_code = 404


class ReportInvalidError(ExperimentViewError):
    error_code = "REPORT_INVALID"
    status_code = 409


class AuditInvalidError(ExperimentViewError):
    error_code = "AUDIT_INVALID"
    status_code = 409


class ArtifactIdentityMismatchError(ExperimentViewError):
    error_code = "ARTIFACT_IDENTITY_MISMATCH"
    status_code = 409


class ArtifactTooLargeError(ExperimentViewError):
    error_code = "ARTIFACT_TOO_LARGE"
    status_code = 413


class ArtifactUnavailableError(ExperimentViewError):
    error_code = "ARTIFACT_UNAVAILABLE"
    status_code = 503


class _InvalidArtifact(Exception):
    """A readable artifact whose content is not a valid UTF-8 JSON object."""


# ── bounded, safe reads ──


def _is_reparse_point(path: str) -> bool:
    """Return True when ``path`` is a symlink, junction, or other reparse point."""
    try:
        st = Path(path).lstat()
    except OSError:
        return False
    if os.name == "nt":
        import stat

        return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    import stat

    return stat.S_ISLNK(st.st_mode)


def _reject_reparse(path: str) -> None:
    """Reject when any component from the drive root down to ``path`` is a link."""
    parts: list[Path] = []
    current = Path(path)
    while True:
        parts.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for component in reversed(parts):
        if _is_reparse_point(str(component)):
            raise ArtifactUnavailableError("artifact path contains a link")


def _norm_parts(path: str) -> tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in Path(path).parts)


def _within_root(target: str, root: str) -> bool:
    target_parts = _norm_parts(target)
    root_parts = _norm_parts(root)
    return len(target_parts) >= len(root_parts) and target_parts[: len(root_parts)] == root_parts


def _validate_artifact_path(path: object, artifact_roots) -> str:
    """Validate a registered artifact path and return its resolved absolute form.

    Rejects missing/empty paths, NUL bytes, suspicious basenames, reparse points
    and any path outside the controlled artifact roots. The file must exist and
    be a regular file.
    """
    if not isinstance(path, str) or not path:
        raise ArtifactUnavailableError("artifact path is missing")
    if "\x00" in path:
        raise ArtifactUnavailableError("artifact path contains NUL")
    base = os.path.basename(path)
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        raise ArtifactUnavailableError("artifact basename is invalid")
    abs_path = os.path.abspath(path)
    _reject_reparse(abs_path)
    real = os.path.realpath(abs_path)
    roots = [os.path.realpath(os.path.abspath(r)) for r in (artifact_roots or ())]
    if not roots or not any(_within_root(real, root) for root in roots):
        raise ArtifactUnavailableError("artifact outside controlled roots")
    if not os.path.isfile(real):
        raise ArtifactUnavailableError("artifact is not an existing regular file")
    return real


def _read_bounded_json(path: str) -> dict:
    """Chunked, hard-capped UTF-8 JSON read returning an object root.

    The declared size is only a fast-path guard: the stream is actually read in
    1 MiB chunks and abandoned as soon as the accumulated bytes exceed the cap,
    so a file that grows during the read is still bounded. Parse failures raise
    ``_InvalidArtifact``; I/O failures raise ``ArtifactUnavailableError``.
    """
    try:
        declared = os.path.getsize(path)
    except OSError as exc:
        raise ArtifactUnavailableError("artifact size check failed") from exc
    if declared > MAX_ARTIFACT_BYTES:
        raise ArtifactTooLargeError("artifact exceeds size limit")
    chunks: list[bytes] = []
    total = 0
    try:
        with open(path, "rb") as fh:
            while True:
                read_len = min(_CHUNK_SIZE, MAX_ARTIFACT_BYTES - total + 1)
                chunk = fh.read(read_len)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARTIFACT_BYTES:
                    raise ArtifactTooLargeError("artifact grew past size limit during read")
                chunks.append(chunk)
    except ArtifactTooLargeError:
        raise
    except OSError as exc:
        raise ArtifactUnavailableError("artifact unreadable") from exc
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InvalidArtifact("not valid UTF-8") from exc
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise _InvalidArtifact("invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise _InvalidArtifact("JSON root is not an object")
    return parsed


# ── shared projection helpers ──


def _sanitize_value(value: Any) -> Any:
    """Reduce path-like strings to their basename (never the full path)."""
    if isinstance(value, str) and (os.path.isabs(value) or "/" in value or "\\" in value):
        return os.path.basename(value) or "…"
    return value


def _sanitize_params(params: Any) -> dict:
    """Sanitize a parameter mapping; a non-object value is a malformed audit."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise AuditInvalidError("audit parameter field is not an object")
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[str(key)] = _sanitize_value(value)
    return out


def _as_param_object(value: Any, field: str) -> dict:
    """Return ``value`` as a plain dict, or reject a malformed audit field.

    A ``None`` (absent) field is empty; anything that is not a JSON object
    (list/string/number) is a malformed audit and fails stably instead of
    raising a native ``ValueError``/``TypeError`` on ``dict(...)``/``.items()``.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AuditInvalidError(f"audit field {field} is not an object")
    return value


_CREDENTIAL_ASSIGN_RE = re.compile(
    r"(?i)(api[\s_\-]*key|apikey|authorization|token|secret|password|credential)"
    r"\s*[=:]\s*\S+"
)
_WIN_PATH_RE = re.compile(r"(?i)[a-z]:[\\/][^\s\"'<>]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\\\s]+(?:\\[^\s]+)*")
_POSIX_PATH_RE = re.compile(r"(?i)(?:^|[\s\"'(])(/[^\s\"'<>]+)")


def _safe_warning_text(value: Any) -> str:
    """Outbound-safe projection of one guardrail warning.

    Traceback text is replaced by a fixed marker; absolute paths and credential
    assignments are redacted so a malicious/corrupt audit can never leak a local
    path, a credential or a traceback through the warnings column.
    """
    if value is None:
        return ""
    text = str(value)
    if "traceback" in text.lower():
        return "…"
    text = _CREDENTIAL_ASSIGN_RE.sub(lambda m: m.group(1) + "=REDACTED", text)
    text = _WIN_PATH_RE.sub("<path>", text)
    text = _UNC_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub(lambda m: m.group(1) + "<path>", text)
    return text


def _as_warnings(value: Any) -> list[str]:
    """Require ``guardrails.warnings`` to be an array of safely projected texts.

    A string must never be split into single characters by ``list(string)``; a
    non-array value is a malformed audit.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AuditInvalidError("audit guardrails.warnings is not an array")
    return [_safe_warning_text(item) for item in value]


def _select_artifact(experiment: dict, kind: str) -> list[dict]:
    artifacts = experiment.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [
        a for a in artifacts
        if isinstance(a, dict) and a.get("kind") == kind and a.get("path")
    ]


def _parse_iso(text: Any) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def _project_dataset(dataset: Any) -> dict | None:
    if not isinstance(dataset, dict):
        return None
    return {
        "dataset_id": dataset.get("dataset_id"),
        "display_name": dataset.get("display_name"),
        "snapshot_id": dataset.get("snapshot_id"),
        "validation_status": dataset.get("validation_status"),
    }


def _project_artifacts(experiment: dict) -> list[dict]:
    out: list[dict] = []
    for artifact in experiment.get("artifacts") or ():
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        exists_state = artifact.get("exists_state")
        if exists_state is None and isinstance(path, str) and path:
            exists_state = "exists" if os.path.exists(path) else "missing"
        out.append({
            "kind": artifact.get("kind"),
            "name": os.path.basename(path) if isinstance(path, str) and path else None,
            "exists_state": exists_state or "unavailable",
        })
    return out


# ── report view ──


def _report_identity_matches(report: dict, run_name: str) -> bool:
    runs = report.get("runs")
    if isinstance(runs, dict) and run_name in runs:
        return True
    if isinstance(runs, dict):
        for entry in runs.values():
            if isinstance(entry, dict) and entry.get("name") == run_name:
                return True
    summary = report.get("summary")
    if isinstance(summary, dict) and summary.get("best_overall_run") == run_name:
        return True
    comparison = report.get("comparison")
    if isinstance(comparison, dict) and comparison.get("best_run") == run_name:
        return True
    return False


def _report_run_data(report: dict, run_name: str) -> dict:
    runs = report.get("runs")
    if not isinstance(runs, dict):
        return {}
    entry = runs.get(run_name)
    if not isinstance(entry, dict):
        return {}
    return entry


def _project_metrics(experiment: dict, results: dict) -> dict:
    stored = experiment.get("metrics")
    if isinstance(stored, dict) and stored:
        return {k: stored[k] for k in _METRIC_KEYS if k in stored}
    final = results.get("final_metrics")
    if isinstance(final, dict):
        mapped: dict[str, Any] = {}
        for source, target in _METRIC_MAP.items():
            if final.get(source) is not None:
                mapped[target] = final[source]
        return mapped
    return {}


def _project_epochs(experiment: dict, args: dict, results: dict) -> dict:
    params = experiment.get("params")
    stored = params.get("_epochs") if isinstance(params, dict) else None
    if isinstance(stored, dict):
        return {
            "configured": stored.get("configured"),
            "completed": stored.get("completed"),
            "best": stored.get("best"),
        }
    return {
        "configured": args.get("epochs"),
        "completed": results.get("total_epochs"),
        "best": results.get("best_epoch"),
    }


def _project_parameters(args: dict, experiment: dict) -> dict:
    source = args if args else (experiment.get("params") or {})
    out: dict[str, Any] = {}
    for key in REPORT_PARAMETER_KEYS:
        if key in source and source[key] is not None:
            out[key] = _sanitize_value(source[key])
    return out


def _project_issues(run_data: dict) -> list[dict]:
    issues = run_data.get("issues")
    if not isinstance(issues, list):
        return []
    out: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        # Production schema uses type/detail; legacy reports use issue/description.
        issue_type = issue.get("type")
        if issue_type is None:
            issue_type = issue.get("issue")
        description = issue.get("detail")
        if description is None:
            description = issue.get("description")
        out.append({
            "issue": issue_type,
            "severity": issue.get("severity"),
            "description": description,
        })
    return out


def _project_ai_analysis(report: dict, run_name: str) -> dict | None:
    llm = report.get("llm_analysis")
    if not isinstance(llm, dict) or not llm:
        return None
    if run_name in llm and isinstance(llm[run_name], dict):
        llm = llm[run_name]
    return {
        "diagnosis": llm.get("diagnosis"),
        "action": llm.get("action"),
        "content_origin": "stored",
    }


def _project_vision_analysis(report: dict, run_name: str) -> dict | None:
    vision = report.get("vision_analysis")
    if not isinstance(vision, dict) or not vision:
        return None
    if run_name in vision and isinstance(vision[run_name], dict):
        vision = vision[run_name]
    if not vision:
        return None
    if vision.get("error"):
        return {"error": vision["error"]}
    cm = vision.get("confusion_matrix_analysis")
    ec = vision.get("error_crop_analysis")
    out: dict[str, Any] = {}
    if isinstance(cm, dict) and cm.get("analysis"):
        out["confusion_matrix"] = {"analysis": cm["analysis"]}
    if isinstance(ec, dict) and ec.get("analysis"):
        out["error_crop"] = {"analysis": ec["analysis"]}
    return out or None


def _project_timing(experiment: dict) -> dict:
    started = experiment.get("started_at")
    finished = experiment.get("finished_at")
    duration = None
    start_dt = _parse_iso(started)
    finish_dt = _parse_iso(finished)
    if start_dt is not None and finish_dt is not None:
        duration = round((finish_dt - start_dt).total_seconds(), 6)
    return {
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": duration,
    }


def build_report_view(experiment: dict, dataset: dict | None = None,
                      artifact_roots: tuple = ()) -> dict:
    """Build the minimal report display model for one run_id-bound experiment.

    The report artifact must be registered against this run_id and its content
    must reference the experiment's run_name (identity check). No LLM, vision,
    name-pattern lookup or metric recomputation ever happens here.
    """
    candidates = _select_artifact(experiment, "report")
    if not candidates:
        raise ReportNotAvailableError("no report artifact registered for this run")
    run_name = experiment.get("run_name")
    if not run_name:
        raise ArtifactIdentityMismatchError("experiment run_name is missing")

    last_error: ExperimentViewError | None = None
    for artifact in candidates:
        try:
            real = _validate_artifact_path(artifact.get("path"), artifact_roots)
            report = _read_bounded_json(real)
        except ArtifactUnavailableError as exc:
            last_error = exc
            continue
        except ArtifactTooLargeError:
            raise
        except _InvalidArtifact:
            last_error = ReportInvalidError("report content is invalid")
            continue
        if not _report_identity_matches(report, run_name):
            last_error = ArtifactIdentityMismatchError("report run_name does not match the experiment")
            continue
        return _project_report(experiment, dataset, report)

    if isinstance(last_error, ArtifactIdentityMismatchError):
        raise last_error
    if isinstance(last_error, ReportInvalidError):
        raise last_error
    raise ArtifactUnavailableError("no usable report artifact")


def _project_report(experiment: dict, dataset: dict | None, report: dict) -> dict:
    run_name = experiment.get("run_name")
    run_data = _report_run_data(report, run_name)
    results = run_data.get("results") if isinstance(run_data.get("results"), dict) else {}
    args = run_data.get("args") if isinstance(run_data.get("args"), dict) else {}

    return {
        "run_id": experiment.get("run_id"),
        "run_name": run_name,
        "source": experiment.get("source"),
        "status": experiment.get("status"),
        "task_type": experiment.get("task_type"),
        "model_name": experiment.get("model_name"),
        "dataset": _project_dataset(dataset),
        "timing": _project_timing(experiment),
        "metrics": _project_metrics(experiment, results),
        "epochs": _project_epochs(experiment, args, results),
        "parameters": _project_parameters(args, experiment),
        "issues": _project_issues(run_data),
        "ai_analysis": _project_ai_analysis(report, run_name),
        "vision_analysis": _project_vision_analysis(report, run_name),
        "artifacts": _project_artifacts(experiment),
        "analysis_status": experiment.get("analysis_status"),
    }


# ── audit view ──


def _audit_identity_matches(audit: dict, path: str) -> bool:
    session_id = audit.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return False
    base = os.path.basename(path)
    return base == f"tuning_audit_{session_id}.json"


def _termination_reason(audit: dict) -> Any:
    """Termination fact for the session overview.

    Only the stable ``error_type``/``error_code`` is shown, never the raw
    ``error.message`` (which can carry absolute paths, credentials or a
    traceback). With no stable code, a fixed safe copy is used instead.
    """
    error = audit.get("error")
    if isinstance(error, dict):
        code = error.get("error_code") or error.get("error_type")
        if isinstance(code, str) and code:
            return code
        return "termination_error"
    return None


def _project_reference_dataset(audit: dict, dataset: dict | None) -> dict | None:
    ref = audit.get("reference_dataset")
    if isinstance(ref, dict):
        return {
            "dataset_id": ref.get("dataset_id"),
            "display_name": ref.get("dataset_display_name") or ref.get("display_name"),
            "snapshot_id": ref.get("snapshot_id"),
            "resolution_source": ref.get("resolution_source") or "sqlite",
        }
    if isinstance(dataset, dict):
        return {
            "dataset_id": dataset.get("dataset_id"),
            "display_name": dataset.get("display_name"),
            "snapshot_id": dataset.get("snapshot_id"),
            "resolution_source": "sqlite",
        }
    return None


def _project_iteration(it: dict) -> dict:
    decision = it.get("decision") if isinstance(it.get("decision"), dict) else {}
    guard = it.get("guardrails") if isinstance(it.get("guardrails"), dict) else {}
    execution = it.get("execution") if isinstance(it.get("execution"), dict) else {}
    result = it.get("result") if isinstance(it.get("result"), dict) else {}
    error = it.get("error") if isinstance(it.get("error"), dict) else None

    # Every object-typed audit field is validated: a list/string/number where a
    # JSON object is expected is a malformed audit (AUDIT_INVALID), never a
    # native ValueError/TypeError from dict(...) or .items().
    suggested = dict(_as_param_object(
        decision.get("hyperparameter_changes"), "hyperparameter_changes"))
    suggested.update(_as_param_object(
        decision.get("training_overrides"), "training_overrides"))
    guarded = _sanitize_params(_as_param_object(
        guard.get("sanitized_changes"), "sanitized_changes"))
    executed = _sanitize_params(_as_param_object(
        execution.get("actual_params"), "actual_params"))
    clamped = _sanitize_params(_as_param_object(guard.get("clamped"), "clamped"))
    metrics = _sanitize_params(_as_param_object(
        result.get("after_metrics"), "after_metrics"))
    metric_delta = _sanitize_params(_as_param_object(
        result.get("metric_delta"), "metric_delta"))

    return {
        "iteration": it.get("iteration"),
        "status": it.get("status"),
        "diagnosis": decision.get("diagnosis"),
        "action": decision.get("action"),
        "suggested_parameters": suggested,
        "guarded_parameters": guarded,
        "executed_parameters": executed,
        "guardrails": {
            "valid": guard.get("valid"),
            "warnings": _as_warnings(guard.get("warnings")),
            "clamped": clamped,
        },
        "metrics": metrics,
        "metric_delta": metric_delta,
        "run_name": execution.get("train_name"),
        "error_code": (error or {}).get("error_type"),
    }


def _project_best_result(audit: dict) -> dict | None:
    final = audit.get("final_summary")
    if not isinstance(final, dict) or final.get("best_iteration") is None:
        return None
    best_metrics = final.get("best_metrics") if isinstance(final.get("best_metrics"), dict) else {}
    return {
        "iteration": final.get("best_iteration"),
        "run_name": final.get("best_train_name"),
        "metrics": {k: best_metrics.get(k) for k in _METRIC_KEYS},
    }


def _project_final_summary(audit: dict) -> dict | None:
    status = audit.get("final_summary_status")
    if status is None:
        return None
    # The audit fact file does not store the LLM closing-summary status; it is
    # honestly null so the UI shows the dash instead of guessing.
    return {"status": status, "llm_summary_status": None}


def build_audit_view(experiment: dict, dataset: dict | None = None,
                     artifact_roots: tuple = ()) -> dict:
    """Build the minimal tuning-audit display model for one run_id.

    Only tuning-source experiments carry audits. The audit artifact must be
    registered against this run_id and its session_id must reconcile with the
    artifact filename. Iterations are ordered by their real ``iteration`` field
    and capped at ``MAX_AUDIT_ITERATIONS`` with ``truncated`` set honestly.
    """
    if experiment.get("source") != "tuning":
        raise AuditNotAvailableError("manual training has no audit record")
    candidates = _select_artifact(experiment, "audit")
    if not candidates:
        raise AuditNotAvailableError("no audit artifact registered for this run")

    last_error: ExperimentViewError | None = None
    for artifact in candidates:
        path = artifact.get("path")
        try:
            real = _validate_artifact_path(path, artifact_roots)
            audit = _read_bounded_json(real)
        except ArtifactUnavailableError as exc:
            last_error = exc
            continue
        except ArtifactTooLargeError:
            raise
        except _InvalidArtifact:
            last_error = AuditInvalidError("audit content is invalid")
            continue
        if not _audit_identity_matches(audit, real):
            last_error = ArtifactIdentityMismatchError("audit session does not reconcile with the artifact")
            continue
        return _project_audit(experiment, dataset, audit)

    if isinstance(last_error, ArtifactIdentityMismatchError):
        raise last_error
    if isinstance(last_error, AuditInvalidError):
        raise last_error
    raise ArtifactUnavailableError("no usable audit artifact")


def _project_audit(experiment: dict, dataset: dict | None, audit: dict) -> dict:
    iterations_raw = audit.get("iterations")
    iterations = (
        [it for it in iterations_raw if isinstance(it, dict)]
        if isinstance(iterations_raw, list) else []
    )

    def _iter_key(it: dict):
        value = it.get("iteration")
        return value if isinstance(value, int) and not isinstance(value, bool) else float("inf")

    iterations.sort(key=_iter_key)
    total = len(iterations_raw) if isinstance(iterations_raw, list) else len(iterations)
    truncated = len(iterations) > MAX_AUDIT_ITERATIONS
    shown = iterations[:MAX_AUDIT_ITERATIONS] if truncated else iterations

    final = audit.get("final_summary")
    best_iteration = final.get("best_iteration") if isinstance(final, dict) else None

    return {
        "run_id": experiment.get("run_id"),
        "session": {
            "session_id": audit.get("session_id"),
            "reference_run": audit.get("reference_run"),
            "status": audit.get("status"),
            "termination_reason": _termination_reason(audit),
            "total_iterations": total,
            "best_iteration": best_iteration,
        },
        "reference_dataset": _project_reference_dataset(audit, dataset),
        "iterations": [_project_iteration(it) for it in shown],
        "truncated": truncated,
        "best_result": _project_best_result(audit),
        "final_summary": _project_final_summary(audit),
    }
