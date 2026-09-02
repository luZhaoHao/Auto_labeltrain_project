"""Auto-Tuning Loop orchestrator — Perception → Decision → Guardrails → Execute → Monitor.

Orchestrates the full closed-loop hyperparameter optimization cycle:
1. Perception: aggregate Module A + Module B outputs
2. Decision: LLM suggests hyperparameter changes
3. Guardrails: validate and clamp changes
4. Execute: launch training with new params
5. Probe: monitor early epochs, decide to continue/abort/retry
"""

from __future__ import annotations

import copy
import json
import os
import time
import logging
import datetime
import threading
from typing import Any

from .perception import (
    build_perception,
    find_module_b_report,
    perception_blocking_error,
)
from . import final_summary as final_summary_mod
from .decision_agent import decide_hyperparameters
from .decision_facts import FactPackageError, build_tuning_fact_package
from .guardrails import sanitize_tuning_parameters, merge_params
from .executor import (
    find_detect_dir, read_args_yaml, prepare_training, launch_training, TrainingProcess,
    build_yolo_command, validate_training_preflight, write_training_config,
)
from .probe_monitor import monitor_training, ProbeDecision
from .audit import TuningAuditSession, atomic_write_json
from auto_tune.modules.train_analyzer.results_parser import parse_results_csv
from auto_tune.modules.train_analyzer.training_finalizer import finalize_training_run
from auto_tune.modules.agent_engine.training_log import (
    append_training_log,
    classify_training_line,
)
from auto_tune.modules.run_state.process_identity import capture_process_identity

logger = logging.getLogger(__name__)


def _forward_output_line(log_path, iteration, on_progress, line):
    """Classify/persist/forward one raw output line into the unified log contract."""
    event = classify_training_line(line)
    try:
        append_training_log(log_path, event.raw)
    except OSError:
        return
    if on_progress:
        on_progress(
            iteration,
            event.summary,
            step="execute",
            event="training_log",
            log_kind=event.kind,
            level=event.level,
            detail=event.raw,
            epoch=event.epoch,
            total_epochs=event.total_epochs,
        )


def _emit_state(on_state, phase, event_type, message, process_identity=None):
    """Report a run-state fact through the optional callback; never raises.

    The loop only reports facts; the UI layer owns the status file.
    """
    if on_state is None:
        return
    try:
        on_state(phase, event_type, message, process_identity)
    except Exception:
        logger.exception("on_state callback failed")


def _start_output_forwarder(train_proc, log_path, iteration, on_progress):
    """Drain the training subprocess output into training.log + SSE in background.

    Single consumer of ``train_proc.drain_output`` (constraint 10): the loop's
    probe/wait paths never read the output log themselves. The daemon thread
    exits once the subprocess ends, after one final drain so no tail line is lost.
    """
    def forward(line):
        _forward_output_line(log_path, iteration, on_progress, line)

    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                train_proc.drain_output(on_line=forward)
            except Exception:
                pass
            try:
                done = train_proc.proc is None or train_proc.proc.poll() is not None
            except Exception:
                done = True
            if done:
                break
            stop.wait(0.05)
        try:
            train_proc.drain_output(on_line=forward)  # final flush after process end
        except Exception:
            pass

    thread = threading.Thread(target=loop, name="training-output-forwarder", daemon=True)
    thread.start()
    return thread


def sanitize_and_merge_tuning_params(
    base_args: dict,
    hyperparameter_changes: dict,
    training_overrides: dict,
    dataset_info: dict | None = None,
) -> tuple[dict | None, Any]:
    """Return merged executable params, or ``None`` when validation rejects them."""
    guard_result = sanitize_tuning_parameters(
        hyperparameter_changes,
        training_overrides,
        dataset_info,
    )
    if not guard_result.valid:
        return None, guard_result
    return merge_params(base_args, guard_result.params), guard_result


def _reference_dataset_public(reference_dataset) -> dict | None:
    """Public-safe projection of the frozen resolution (no absolute paths)."""
    if reference_dataset is None:
        return None
    snapshot_id = getattr(reference_dataset, "snapshot_id", "") or ""
    return {
        "reference_run": getattr(reference_dataset, "reference_run", None),
        "dataset_display_name": getattr(reference_dataset, "dataset_display_name", None),
        "snapshot_short_id": snapshot_id[:8] if snapshot_id else None,
        "resolution_source": getattr(reference_dataset, "resolution_source", None),
    }


def _failure(stage: str, error_type: str, message: str, fatal: bool = True) -> dict:
    """Structured failure dict used by both the audit record and the loop result."""
    return {
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "fatal": fatal,
    }


CONTRACT_FAILURE_TYPES = {
    "DECISION_SCHEMA_INVALID": "decision_schema_invalid",
    "DECISION_FACT_PACKAGE_MISMATCH": "decision_fact_package_mismatch",
    "DECISION_EVIDENCE_MISSING": "decision_evidence_missing",
    "DECISION_EVIDENCE_UNKNOWN": "decision_evidence_unknown",
}

SEMANTIC_FAILURE_TYPES = {
    "DECISION_SEMANTIC_UNSUPPORTED": "decision_semantic_error",
    "DECISION_SEMANTIC_DIRECTION_CONFLICT": "decision_semantic_error",
    "DECISION_SEMANTIC_CHANGE_TOO_LARGE": "decision_semantic_error",
    "DECISION_SEMANTIC_CURRENT_VALUE_MISSING": "decision_semantic_error",
    "DECISION_SEMANTIC_EVIDENCE_CONFLICT": "decision_semantic_error",
}

SEMANTIC_USER_MESSAGE = (
    "LLM 建议引用了真实证据，但证据与参数修改之间缺少受支持的语义关系。"
    "本轮自动调优已停止，未启动训练。"
)


def _classify_decision_error(err_msg: str) -> str:
    """Classify a non-contract decision error using the legacy provider rules."""
    lower = err_msg.lower()
    if ("deepseek api error" in lower or "request" in lower
            or "timeout" in lower or "connection" in lower):
        return "decision_api_error"
    return "decision_schema_error"


def _metric_delta(before: dict, after: dict) -> dict:
    """Compute after-before deltas for metrics present in both; zero is kept."""
    delta: dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if before_value is None or after_value is None:
            continue
        delta[key] = round(float(after_value) - float(before_value), 10)
    return delta


def _read_reference_before_metrics(reference_run: str | None, detect_dir: str) -> tuple[dict, dict]:
    """Read reference-run metrics from its results.csv as authoritative before facts.

    Uses the final-epoch metrics so before and after share the same scope.
    Missing metrics stay missing; real zeros are kept. Returns (metrics, source)
    where source records provenance and any structured parse error.
    """
    source: dict = {"type": "results_csv", "path": None, "epoch_scope": "final", "error": None}
    if not reference_run:
        source["error"] = "no_reference_run"
        return {}, source
    csv_path = os.path.join(detect_dir, reference_run, "results.csv")
    source["path"] = csv_path
    if not os.path.isfile(csv_path):
        source["error"] = "results_csv_missing"
        return {}, source
    try:
        results = parse_results_csv(csv_path)
    except Exception as exc:
        source["error"] = f"results_csv_parse_error: {exc}"
        return {}, source
    if "error" in results:
        source["error"] = results["error"]
        return {}, source
    final = results.get("final_metrics", {}) or {}
    mapping = {
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    }
    metrics: dict[str, float] = {}
    for source_key, target in mapping.items():
        value = final.get(source_key)
        if value is not None:
            metrics[target] = value
    if not metrics:
        source["error"] = "no_valid_metric_columns"
    return metrics, source


def _extract_after_metrics(iter_result: TuningResult) -> dict:
    """Collect result metrics from an iteration result, keeping zero values."""
    after = {
        "mAP50": iter_result.result_mAP50,
        "mAP50_95": iter_result.result_mAP50_95,
        "precision": iter_result.result_precision,
        "recall": iter_result.result_recall,
    }
    return {k: v for k, v in after.items() if v is not None}


def _persist_iteration_failure(
    audit: TuningAuditSession,
    iteration: int,
    stage: str,
    error_type: str,
    message: str,
    fatal: bool = True,
) -> dict:
    """Record a failure in the audit session, translating persistence errors to
    audit_persistence_error without recursing into fail_iteration again."""
    try:
        audit.fail_iteration(iteration, stage, error_type, message, fatal=fatal)
        return _failure(stage, error_type, message, fatal=fatal)
    except Exception as exc:
        return _failure("audit", "audit_persistence_error",
                        f"{message}; 审计写入失败: {exc}")


def _abort_tuning(
    tuning_result: dict,
    iter_result: TuningResult,
    audit: TuningAuditSession,
    iteration: int,
    failure: dict,
    history: TuningHistory,
    log_dir: str,
    on_progress: callable = None,
) -> None:
    """Finalize a fatal failure into the loop result and audit (best effort).

    When the failure is already an audit_persistence_error, no further audit
    write is attempted (the audit file cannot be updated); the error is still
    reported in the return value. Best-iteration data is never overwritten.
    """
    # Persist the final summary for rounds that already completed before this
    # fatal failure, without changing the failure fact.
    _best_effort_final_summary(tuning_result, audit)
    iter_result.error = failure["message"]
    history.add_attempt(iter_result.to_dict())
    tuning_result["iterations"].append(iter_result.to_dict())
    if on_progress:
        on_progress(iteration, failure["message"])
    tuning_result["failure"] = failure
    tuning_result["error"] = iter_result.error
    if failure["error_type"] != "audit_persistence_error":
        try:
            audit.finalize("failed", failure)
        except Exception as exc:
            tuning_result["failure"] = _failure(
                "audit", "audit_persistence_error",
                f"{failure['message']}; 最终审计写入失败: {exc}",
            )
            tuning_result["error"] = tuning_result["failure"]["message"]
    try:
        history.to_json(os.path.join(log_dir, "tuning_history.json"))
    except Exception:
        pass


class TuningHistory:
    """Records the history of auto-tuning attempts."""

    def __init__(self):
        self.attempts: list[dict] = []

    def add_attempt(self, attempt: dict):
        self.attempts.append(attempt)

    def get_previous_changes(self) -> list[dict]:
        """Return previous attempts formatted for LLM context."""
        feedback = []
        for attempt in self.attempts:
            decision = attempt.get("decision", {}) or {}
            before = attempt.get("before_metrics", {}) or {}
            after = {
                "mAP50": attempt.get("result_mAP50"),
                "mAP50_95": attempt.get("result_mAP50_95"),
                "precision": attempt.get("result_precision"),
                "recall": attempt.get("result_recall"),
            }
            after = {k: v for k, v in after.items() if v is not None}
            delta = {}
            for key, value in after.items():
                if before.get(key) is not None:
                    delta[key] = round(value - before[key], 10)
            probe = attempt.get("probe_decision", {}) or {}
            feedback.append({
                "changes": decision.get("hyperparameter_changes", {}),
                "before_metrics": before,
                "after_metrics": after,
                "metric_delta": delta,
                "probe_verdict": probe.get("verdict"),
                "status": "failed" if attempt.get("error") else "completed",
                "diagnosis": decision.get("diagnosis", ""),
            })
        return feedback

    def to_dict(self) -> list[dict]:
        return self.attempts

    def to_json(self, path: str):
        atomic_write_json(path, self.attempts)

    @classmethod
    def from_json(cls, path: str) -> "TuningHistory":
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                h = cls()
                h.attempts = data
                return h
        return cls()


class TuningResult:
    """Result of a single tuning loop iteration."""

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.perception: dict = {}
        self.decision: dict = {}
        self.guard_result: Any = None
        self.merged_params: dict = {}
        self.train_name: str | None = None
        self.probe_decision: ProbeDecision | None = None
        self.error: str | None = None
        # Metrics from Module B analysis
        self.result_mAP50: float | None = None
        self.result_mAP50_95: float | None = None
        self.result_precision: float | None = None
        self.result_recall: float | None = None
        self.result_best_epoch: int | None = None
        self.result_analysis_status: str | None = None

    def get_composite_score(self, mode: str = "comprehensive") -> float:
        """Compute composite score based on evaluation mode.

        - quick (双指标): mAP50×0.6 + mAP50-95×0.4
        - comprehensive (四维): mAP50×0.35 + mAP50-95×0.25 + precision×0.20 + recall×0.20
        """
        if mode == "quick":
            if self.result_mAP50 is not None and self.result_mAP50_95 is not None:
                return self.result_mAP50 * 0.6 + self.result_mAP50_95 * 0.4
            if self.result_mAP50 is not None:
                return self.result_mAP50
            return 0.0
        # comprehensive (default)
        scores = [self.result_mAP50, self.result_mAP50_95, self.result_precision, self.result_recall]
        weights = [0.35, 0.25, 0.20, 0.20]
        has_all = all(s is not None for s in scores)
        if has_all:
            return sum(s * w for s, w in zip(scores, weights))  # type: ignore
        # Fallback to quick
        return self.get_composite_score(mode="quick")

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "perception": {
                "dataset_total_images": self.perception.get("dataset", {}).get("total_images"),
                "training_best_mAP50": self.perception.get("training", {}).get("best_mAP50"),
            },
            "decision": {
                "diagnosis": self.decision.get("diagnosis", ""),
                "action": self.decision.get("action", ""),
                "hyperparameter_changes": self.decision.get("hyperparameter_changes", {}),
                "training_overrides": self.decision.get("training_overrides", {}),
            },
            "guard_results": {
                "valid": getattr(self.guard_result, "valid", True),
                "warnings": getattr(self.guard_result, "warnings", []),
                "errors": getattr(self.guard_result, "errors", []),
                "clamped": getattr(self.guard_result, "clamped", {}),
            },
            "merged_params": {k: v for k, v in self.merged_params.items()
                              if not k.startswith("_")},
            "train_name": self.train_name,
            "probe_decision": {
                "verdict": self.probe_decision.verdict if self.probe_decision else None,
                "reason": self.probe_decision.reason if self.probe_decision else "",
                "suggestion": self.probe_decision.suggestion if self.probe_decision else "",
            },
            "result_mAP50": self.result_mAP50,
            "result_mAP50_95": self.result_mAP50_95,
            "result_precision": self.result_precision,
            "result_recall": self.result_recall,
            "result_best_epoch": self.result_best_epoch,
            "result_analysis_status": self.result_analysis_status,
            "error": self.error,
        }


def _compute_best(tuning_result: dict, eval_mode: str = "comprehensive"):
    """Compute best iteration from all completed iterations in tuning_result."""
    successful = final_summary_mod.successful_iterations(tuning_result)
    if not successful:
        return

    best_it = max(successful, key=lambda it: final_summary_mod.composite_score(it, eval_mode))
    tuning_result["best_iteration"] = best_it.get("iteration")
    tuning_result["best_train_name"] = best_it.get("train_name")
    tuning_result["best_metrics"] = {
        "mAP50": best_it.get("result_mAP50"),
        "mAP50_95": best_it.get("result_mAP50_95"),
        "precision": best_it.get("result_precision"),
        "recall": best_it.get("result_recall"),
    }
    tuning_result["best_score"] = final_summary_mod.composite_score(best_it, eval_mode)
    if not tuning_result.get("final_result"):
        tuning_result["final_result"] = {
            "train_name": best_it.get("train_name"),
            "iteration": best_it.get("iteration"),
            "decision": (best_it.get("decision") or {}).get("diagnosis"),
            "changes": (best_it.get("decision") or {}).get("hyperparameter_changes", {}),
        }


def _session_short_id(tuning_session_id: str) -> str:
    """Deterministic short session identity derived from the tuning session.

    Never regenerated per round; the same value is reused for the whole session
    so auto-loop run names stay short and stable.
    """
    import hashlib
    return hashlib.sha1(str(tuning_session_id).encode("utf-8")).hexdigest()[:8]


def _make_train_name(session_short_id: str, iteration: int) -> str:
    """Fixed-width, path-safe auto-loop run name. The reference run lives in the
    audit field, never inside the directory name, so names cannot grow or nest."""
    return f"autotune_{session_short_id}_iter{int(iteration):02d}"


def _finalize_final_summary(
    tuning_result: dict,
    audit: TuningAuditSession,
    config: dict,
    detect_dir: str,
    eval_mode: str,
    session_id: str,
    reference_run: str | None,
) -> None:
    """Build the deterministic final summary, call the LLM at most once, and
    atomically write tuning_final_summary.txt into the best tuning run dir.

    Every failure inside is mapped to a stable status; the training fact and the
    audit terminal state are never changed by a summary failure. This function
    never raises: any unexpected error is mapped to ``failed`` so the tuning
    interface can never crash because of final-summary generation.
    """
    try:
        summary = final_summary_mod.build_deterministic_summary(
            tuning_result, eval_mode, session_id, reference_run,
        )
    except Exception:
        logger.exception("final summary build failed")
        summary = None
        _mark_final_summary_failed(tuning_result, audit)
        return

    if summary is None:
        tuning_result["final_summary"] = None
        tuning_result["final_summary_status"] = "skipped"
        tuning_result["final_summary_path"] = None
        tuning_result["llm_summary_status"] = final_summary_mod.LLM_SUMMARY_SKIPPED
        tuning_result["summary_persistence_status"] = final_summary_mod.SUMMARY_PERSISTENCE_SKIPPED
        try:
            audit.set_final_summary(status="skipped", summary=None)
        except Exception:
            pass
        return

    try:
        llm_result = final_summary_mod.call_final_summary_llm(summary, config)
        tuning_result["llm_summary_status"] = llm_result["status"]
        if llm_result.get("error_code"):
            tuning_result["llm_summary_error_code"] = llm_result["error_code"]

        text = final_summary_mod.render_final_summary_text(summary, llm_result.get("text"))
        best_train_dir = None
        if summary.get("best_train_name"):
            best_train_dir = os.path.join(detect_dir, summary["best_train_name"])
        if best_train_dir:
            write_result = final_summary_mod.write_final_summary_txt(summary, best_train_dir, text)
        else:
            write_result = {
                "status": final_summary_mod.SUMMARY_PERSISTENCE_SKIPPED,
                "path": None,
                "error_code": None,
            }
        tuning_result["summary_persistence_status"] = write_result["status"]
        tuning_result["final_summary_path"] = write_result.get("path")
        tuning_result["final_summary"] = summary
        # The deliverable is only "generated" when the TXT actually landed; a
        # failed persistence marks the whole final-summary status failed.
        final_summary_status = (
            "generated"
            if write_result["status"] != final_summary_mod.SUMMARY_PERSISTENCE_FAILED
            else "failed"
        )
        tuning_result["final_summary_status"] = final_summary_status

        try:
            audit.set_final_summary(status=final_summary_status, summary=summary)
        except Exception:
            pass
    except Exception:
        logger.exception("final summary persistence failed")
        _mark_final_summary_failed(tuning_result, audit)


def _mark_final_summary_failed(tuning_result: dict, audit: TuningAuditSession) -> None:
    """Map any unexpected final-summary failure to a stable, non-fatal state."""
    tuning_result["final_summary"] = None
    tuning_result["final_summary_status"] = "failed"
    tuning_result["final_summary_path"] = None
    tuning_result["llm_summary_status"] = final_summary_mod.LLM_SUMMARY_FAILED
    tuning_result["summary_persistence_status"] = final_summary_mod.SUMMARY_PERSISTENCE_FAILED
    try:
        audit.set_final_summary(status="failed", summary=None)
    except Exception:
        pass


def _best_effort_final_summary(tuning_result: dict, audit: TuningAuditSession) -> None:
    """Persist the final summary for rounds that completed before an early exit
    (fatal failure / cancellation).

    Best-effort and non-fatal: it never changes the exit fact and never raises.
    When no measurable round succeeded it does nothing, leaving the statuses at
    their ``skipped`` initial value.
    """
    if not final_summary_mod.successful_iterations(tuning_result):
        return
    eval_mode = tuning_result.get("eval_mode", "comprehensive")
    _compute_best(tuning_result, eval_mode)
    try:
        _finalize_final_summary(
            tuning_result,
            audit,
            tuning_result.get("_config") or {},
            tuning_result.get("detect_dir"),
            eval_mode,
            tuning_result.get("session_id"),
            tuning_result.get("reference_run"),
        )
    except Exception:
        logger.exception("best-effort final summary failed")


def run_tuning_loop(
    config: dict,
    reference_run: str | None = None,
    max_retries: int | None = None,
    log_dir: str = "log",
    skip_execute: bool = False,
    auto_analyze: bool = False,
    auto_loop: bool = False,
    on_progress: callable = None,
    cancel_event = None,
    keep_params: bool = False,
    eval_mode: str = "comprehensive",
    on_state: callable = None,
    runtime_run_id: str | None = None,
    local_index_service=None,
    reference_dataset=None,
) -> dict:
    """Run the complete auto-tuning loop.

    This is the main entry point for Module C.

    Args:
        config: full config dict (from config.yaml).
        reference_run: specific training to use as reference (e.g., "train8").
                       If None, uses the latest from Module B report.
        max_retries: max tuning attempts (default from config).
        log_dir: directory for log output.
        skip_execute: if True, stop at merged_params (for testing).
        on_progress: optional callback(iteration, message) for UI updates.
        reference_dataset: frozen ReferenceDatasetResolution bound to
            reference_run; when provided its data_yaml_path is the single
            training dataset for every iteration (never latest_dataset).

    Returns:
        Dict with final results including all iteration details.
    """
    logger.info(f"[AutoTune] run_tuning_loop called: skip_execute={skip_execute}, auto_analyze={auto_analyze}, auto_loop={auto_loop}, max_retries={max_retries}, reference_run={reference_run}")

    if max_retries is None:
        max_retries = config.get("probe", {}).get("max_retries", 3)

    if reference_run is None:
        # Auto-detect from latest Module B report
        report = find_module_b_report(log_dir=log_dir)
        if report and report.get("runs"):
            reference_run = list(report["runs"].keys())[0]

    history = TuningHistory()
    detect_dir = find_detect_dir()
    tuning_session_id = str(int(time.time() * 1000))  # unique per tuning session

    # Internal audit identity: the frozen resolution's full dataset_id,
    # snapshot_id and source are stored for reconciliation. Public projections
    # only show names / short ids.
    reference_dataset_audit = None
    if reference_dataset is not None:
        reference_dataset_audit = {
            "reference_run": getattr(reference_dataset, "reference_run", None),
            "dataset_id": getattr(reference_dataset, "dataset_id", None),
            "snapshot_id": getattr(reference_dataset, "snapshot_id", None),
            "data_yaml_path": str(getattr(reference_dataset, "data_yaml_path", "")),
            "resolution_source": getattr(reference_dataset, "resolution_source", None),
            "dataset_display_name": getattr(reference_dataset, "dataset_display_name", None),
        }

    audit = TuningAuditSession(
        tuning_session_id, log_dir, reference_run, max_retries,
        reference_dataset=reference_dataset_audit,
    )
    try:
        audit.flush()
    except Exception as exc:
        failure = _failure("audit", "audit_persistence_error", f"审计初始化失败: {exc}")
        tuning_result = {
            "module": "agent_engine",
            "version": "1.0",
            "detect_dir": detect_dir,
            "reference_run": reference_run,
            "max_retries": max_retries,
            "iterations": [],
            "final_result": None,
            "error": f"审计持久化失败: {exc}",
            "eval_mode": eval_mode,
            "session_id": tuning_session_id,
            "audit_path": audit.path,
            "failure": failure,
        }
        return tuning_result

    tuning_result = {
        "module": "agent_engine",
        "version": "1.0",
        "detect_dir": detect_dir,
        "reference_run": reference_run,
        "max_retries": max_retries,
        "iterations": [],
        "final_result": None,
        "error": None,
        "eval_mode": eval_mode,
        "session_id": tuning_session_id,
        "audit_path": audit.path,
        "failure": None,
        "final_summary": None,
        "final_summary_status": "skipped",
        "final_summary_path": None,
        "llm_summary_status": "skipped",
        "summary_persistence_status": "skipped",
        # Public-safe projection of the frozen reference dataset resolution.
        # Absolute data_yaml_path is never included here.
        "reference_dataset": _reference_dataset_public(reference_dataset),
        # Internal only: never serialized to clients; used by best-effort
        # finalization on early-exit paths.
        "_config": config,
    }

    _emit_state(on_state, "preparing", "tuning_start", "调优会话开始")

    for iteration in range(1, max_retries + 1):
        # Check cancellation before each iteration
        if cancel_event and cancel_event.is_set():
            logger.info("[AutoTune] Cancelled before iteration %d", iteration)
            _best_effort_final_summary(tuning_result, audit)
            tuning_result["error"] = "用户取消"
            audit.finalize("cancelled")
            history.to_json(os.path.join(log_dir, "tuning_history.json"))
            return tuning_result

        iter_result = TuningResult(iteration)
        audit.start_iteration(iteration)
        if on_progress:
            on_progress(iteration, f"迭代 {iteration}/{max_retries} 开始")
        _emit_state(on_state, "analyzing", "iteration_start", f"迭代 {iteration}/{max_retries} 开始")

        try:
            # ── Step 1: Perception ──
            if on_progress:
                on_progress(iteration, "感知层：聚合数据集和训练分析数据", step="perception")
            perception = build_perception(log_dir=log_dir, reference_run=reference_run)
            iter_result.perception = perception

            # Persist a redacted perception summary before the gate so a blocked
            # round still records the facts it did have.
            ds_src = {}
            perception_status = "provided"
            if isinstance(perception.get("sources"), dict):
                ds_src = perception["sources"].get("dataset_report") or {}
                perception_status = ds_src.get("status", "unknown")
            tr_src = {}
            if isinstance(perception.get("sources"), dict):
                tr_src = perception["sources"].get("training_report") or {}
            try:
                audit.update_iteration(iteration, perception={
                    "status": perception_status,
                    "dataset_report_basename": ds_src.get("basename"),
                    "dataset_total_images": perception.get("dataset", {}).get("total_images"),
                    "training_report_basename": tr_src.get("basename"),
                    "reference_run": reference_run,
                    "training_best_mAP50": perception.get("training", {}).get("best_mAP50"),
                })
            except Exception as exc:
                failure = _persist_iteration_failure(
                    audit, iteration, "audit", "audit_persistence_error",
                    f"审计写入失败: {exc}",
                )
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            # Perception availability gate: never call the LLM / start YOLO on
            # unusable facts, and never fall back to keep_params.
            blocking_code, blocking_msg = perception_blocking_error(perception)
            if blocking_code:
                failure = _persist_iteration_failure(
                    audit, iteration, "perception", blocking_code, blocking_msg,
                )
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            # ── Step 2: Freeze the FactPackage before any LLM call ──
            ref_dir = os.path.join(detect_dir, reference_run) if reference_run else None
            base_args = read_args_yaml(ref_dir) if (ref_dir and os.path.isdir(ref_dir)) else {}
            before_metrics, metrics_source = _read_reference_before_metrics(reference_run, detect_dir)
            try:
                fact_package = build_tuning_fact_package(
                    perception, reference_run, base_args, before_metrics, metrics_source,
                )
            except FactPackageError as exc:
                failure = _persist_iteration_failure(
                    audit, iteration, "facts", "fact_package_invalid", str(exc.detail),
                )
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result
            try:
                audit.update_iteration(iteration, fact_package=fact_package)
            except Exception as exc:
                failure = _persist_iteration_failure(
                    audit, iteration, "audit", "audit_persistence_error", f"审计写入失败: {exc}",
                )
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            # ── Step 3: Decision ──
            if keep_params:
                if on_progress:
                    on_progress(iteration, "决策层：保持原有参数，跳过 LLM 调参", step="decision", params={"_keep_params": True, "reason": "AI 判断无需调整超参数，使用原有参数继续训练"})
                decision = {
                    "schema_version": "1.0",
                    "fact_package_id": fact_package["fact_package_id"],
                    "diagnosis": "按原有参数训练，不做超参数调整",
                    "action": "keep_params",
                    "hyperparameter_changes": {},
                    "training_overrides": {},
                    "evidence_ids": {},
                    "validation": {
                        "valid": True, "error_code": None, "error_detail": None,
                        "retried": False, "referenced_fact_ids": [],
                    },
                    "semantic_validation": {
                        "valid": True, "error_code": None, "reason_code": None,
                        "retried": False, "parameters": [],
                    },
                }
            else:
                if on_progress:
                    on_progress(iteration, "决策层：LLM 分析并输出调参建议", step="decision")
                decision_attempts: list[dict] = []

                def _record_decision_attempt(attempt: dict) -> None:
                    # 每次模型响应的 Q1.1/Q1.2 校验结果在决定继续前立即落盘。
                    # 绝不在已交给审计对象的列表上原地修改：先构造独立候选值
                    # （deepcopy 隔离嵌套对象），只有写盘成功后才把本地
                    # "已成功持久化"列表更新为候选值；写盘失败时本地状态仍只
                    # 含此前成功落盘的 attempt，异常向上传播由下方 try/except
                    # 映射为 audit_persistence_error fail-closed 终态。
                    nonlocal decision_attempts
                    candidate_attempts = [*decision_attempts, copy.deepcopy(attempt)]
                    audit.update_iteration(iteration, decision_attempts=candidate_attempts)
                    decision_attempts = candidate_attempts

                try:
                    decision = decide_hyperparameters(
                        fact_package, config, on_attempt=_record_decision_attempt,
                    )
                except Exception as exc:
                    failure = _persist_iteration_failure(
                        audit, iteration, "audit", "audit_persistence_error",
                        f"审计写入失败: {exc}",
                    )
                    _abort_tuning(tuning_result, iter_result, audit, iteration, failure,
                                  history, log_dir, on_progress)
                    return tuning_result
                combined_params = {}
                if decision.get("hyperparameter_changes"):
                    combined_params.update(decision["hyperparameter_changes"])
                if decision.get("training_overrides"):
                    combined_params["_overrides"] = decision["training_overrides"]
                if on_progress:
                    if combined_params:
                        on_progress(iteration, f"参数变更: {json.dumps(combined_params, ensure_ascii=False)}",
                                    step="decision", params=combined_params)
                    else:
                        on_progress(iteration, "决策层：AI 判断当前超参数无需调整，沿用原有参数",
                                    step="decision", params={"_keep_params": True, "reason": "AI 判断无需调整超参数"})
            iter_result.decision = decision

            decision_validation = decision.get("validation") or {
                "valid": decision.get("error") is None,
                "error_code": decision.get("error") if decision.get("error") in CONTRACT_FAILURE_TYPES else None,
                "error_detail": None,
                "retried": bool(decision.get("retried", False)),
                "referenced_fact_ids": [],
            }
            try:
                audit.update_iteration(iteration, decision={
                    "raw_response": decision.get("raw_response"),
                    "diagnosis": decision.get("diagnosis"),
                    "action": decision.get("action"),
                    "hyperparameter_changes": decision.get("hyperparameter_changes", {}),
                    "training_overrides": decision.get("training_overrides", {}),
                    "schema_version": decision.get("schema_version"),
                    "fact_package_id": decision.get("fact_package_id"),
                    "evidence_ids": decision.get("evidence_ids", {}),
                }, decision_validation=decision_validation,
                   semantic_validation=decision.get("semantic_validation"))
            except Exception as exc:
                failure = _persist_iteration_failure(
                    audit, iteration, "audit", "audit_persistence_error", f"审计写入失败: {exc}",
                )
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            if decision.get("error"):
                err_msg = str(decision["error"])
                if err_msg in SEMANTIC_FAILURE_TYPES:
                    fail_message = SEMANTIC_USER_MESSAGE
                else:
                    fail_message = err_msg
                iter_result.error = f"决策失败: {fail_message}"
                error_type = (CONTRACT_FAILURE_TYPES.get(err_msg)
                              or SEMANTIC_FAILURE_TYPES.get(err_msg)
                              or _classify_decision_error(err_msg))
                failure = _persist_iteration_failure(audit, iteration, "decision", error_type, fail_message)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            # ── Step 4: Guardrails ──
            if on_progress:
                on_progress(iteration, "安全护栏：校验和约束参数", step="guardrails")
            changes = decision.get("hyperparameter_changes", {})
            overrides = decision.get("training_overrides", {})
            dataset_info = perception.get("dataset", {})

            merged, guard_result = sanitize_and_merge_tuning_params(
                base_args, changes, overrides, dataset_info
            )
            iter_result.guard_result = guard_result
            audit.update_iteration(iteration, guardrails={
                "valid": guard_result.valid,
                "warnings": list(guard_result.warnings),
                "errors": list(guard_result.errors),
                "clamped": dict(getattr(guard_result, "clamped", {})),
                "sanitized_changes": dict(getattr(guard_result, "params", {})),
            })
            if not guard_result.valid:
                err_msg = f"护栏拦截: {'; '.join(guard_result.errors)}"
                failure = _persist_iteration_failure(audit, iteration, "guardrails", "guardrail_rejected", err_msg)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            iter_result.merged_params = merged

            # Store old batch for guardrail scaling check
            merged["_old_batch"] = base_args.get("batch", 16)

            # Write real baseline facts bound to the same reference_run:
            # reference params (or merged params when no reference run exists)
            # plus before metrics read from the reference results.csv (already
            # read once for the fact package; reused here unchanged).
            # Internal underscore-prefixed fields are excluded.
            if reference_run:
                base_params = {k: v for k, v in base_args.items() if not str(k).startswith("_")}
            else:
                base_params = {k: v for k, v in merged.items() if not str(k).startswith("_")}
            audit.update_iteration(iteration, baseline={
                "reference_run": reference_run,
                "params": base_params,
                "metrics": before_metrics,
                "metrics_source": metrics_source,
            })

            for warning in guard_result.warnings:
                logger.warning(f"[Guardrail] {warning}")

            if skip_execute:
                iter_result.train_name = "dry_run"
                iter_result.probe_decision = ProbeDecision(ProbeDecision.CONTINUE, "跳过执行")
                tuning_result["iterations"].append(iter_result.to_dict())
                tuning_result["final_result"] = {
                    "train_name": "dry_run",
                    "iteration": iteration,
                    "decision": decision.get("diagnosis"),
                    "changes": changes,
                    "guard_warnings": guard_result.warnings,
                }
                if on_progress:
                    on_progress(iteration, "跳过执行（dry-run 模式）")
                _emit_state(on_state, "finalizing", "dry_run", "dry-run 计划已生成")
                audit.complete_iteration(iteration)
                audit.finalize("completed")
                history.to_json(os.path.join(log_dir, "tuning_history.json"))
                return tuning_result

            # ── Step 4: Execute ──
            # Bind the frozen reference dataset identity (when resolved) so
            # auto-tuning always trains against the reference run's own verified
            # snapshot and never consults the global latest_dataset. When no
            # resolution was frozen (dry-run plan only) the reference args.yaml
            # data is left untouched and the plan stays not-executable.
            if reference_dataset is not None:
                merged["data"] = str(reference_dataset.data_yaml_path)
            if on_progress:
                on_progress(iteration, f"执行层：启动训练 {merged.get('model', 'yolov8')}", step="execute")
            session_short_id = _session_short_id(tuning_session_id)
            train_name = _make_train_name(session_short_id, iteration)
            output_dir = os.path.join(detect_dir, train_name)

            # ── Preflight before creating the output directory ──
            reference_dir = os.path.join(detect_dir, reference_run) if reference_run else None
            preflight_errors = validate_training_preflight(reference_run, reference_dir, merged)
            if preflight_errors:
                err_msg = f"训练预检失败: {'; '.join(preflight_errors)}"
                failure = _persist_iteration_failure(audit, iteration, "preflight", "preflight_error", err_msg)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            os.makedirs(output_dir, exist_ok=False)  # fresh directory — must not exist

            # Write merged config
            args_path = write_training_config(base_args, merged, output_dir)
            iter_result.train_name = train_name
            _emit_state(on_state, "training", "launching", f"启动训练 {train_name}")

            # Build the exact command once, audit it, then launch it.
            try:
                command = build_yolo_command(train_name, args_path, merged)
            except Exception as exc:
                err_msg = f"命令构造失败: {exc}"
                failure = _persist_iteration_failure(audit, iteration, "execute", "command_build_error", err_msg)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            actual_params = {k: v for k, v in merged.items() if not k.startswith("_")}
            try:
                audit.update_iteration(iteration, execution={
                    "actual_params": actual_params,
                    "args_yaml_path": args_path,
                    "command": command,
                    "train_name": train_name,
                })
            except Exception as exc:
                err_msg = f"审计写入失败: {exc}"
                failure = _failure("audit", "audit_persistence_error", err_msg)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result

            # Capture started_at before launch so duration is never zero.
            training_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            # Launch training
            try:
                proc = launch_training(train_name, args_path, merged, command=command)
            except Exception as exc:
                err_msg = f"训练启动失败: {exc}"
                failure = _persist_iteration_failure(audit, iteration, "execute", "training_launch_error", err_msg)
                _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                return tuning_result
            train_proc = TrainingProcess(train_name, output_dir, proc)
            training_start_time = time.time()
            _train_pid = train_proc.pid
            _proc_identity = capture_process_identity(_train_pid) if _train_pid is not None else None
            _emit_state(
                on_state, "training", "executing",
                f"训练进程已启动 (PID {_train_pid})",
                process_identity=_proc_identity,
            )

            # S1.1: forward subprocess output into unified training.log + SSE.
            _train_log_path = os.path.join(output_dir, "training.log")
            with open(_train_log_path, "w", encoding="utf-8") as _lf:
                _lf.write("")
            _output_forwarder = _start_output_forwarder(
                train_proc, _train_log_path, iteration, on_progress
            )

            # ── Step 5: Probe Monitor ──
            if on_progress:
                on_progress(iteration, "探针监控：监测前 N 个 epoch 的训练趋势", step="probe")

            def epoch_cb(epoch, metrics):
                if on_progress:
                    mAP = metrics.get("metrics/mAP50(B)", "?")
                    on_progress(iteration, f"Epoch {epoch}: mAP50={mAP}")

            probe_decision = monitor_training(
                train_proc, config, on_epoch_callback=epoch_cb,
                process_start_time=training_start_time,
                cancel_event=cancel_event,
            )
            iter_result.probe_decision = probe_decision

            if probe_decision.verdict == ProbeDecision.ABORT:
                iter_result.error = f"训练中止: {probe_decision.reason}"
                failure = _persist_iteration_failure(audit, iteration, "probe", "probe_abort", iter_result.error, fatal=False)
                if failure["error_type"] == "audit_persistence_error":
                    _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                    return tuning_result
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                _output_forwarder.join(timeout=2)
                if on_progress:
                    on_progress(iteration, f"训练中止: {probe_decision.reason}")
                continue
            elif probe_decision.verdict == ProbeDecision.RETRY:
                iter_result.error = f"需要重试: {probe_decision.reason}"
                failure = _persist_iteration_failure(audit, iteration, "probe", "probe_retry", iter_result.error, fatal=False)
                if failure["error_type"] == "audit_persistence_error":
                    _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                    return tuning_result
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                _output_forwarder.join(timeout=2)
                if on_progress:
                    on_progress(iteration, f"需要重试: {probe_decision.reason}")
                continue

            # ── Probe passed: mark pipeline complete (green) ──
            if on_progress:
                on_progress(iteration, "✅ 探针通过，继续训练", step="complete", level="success")
            logger.info(f"[AutoTune] Probe passed. auto_analyze={auto_analyze}, auto_loop={auto_loop}")
            if auto_analyze:
                # ── Wait for full training to complete ──
                if on_progress:
                    on_progress(iteration, f"探针通过，等待完整训练完成...")

                # Start from where probe left off so we report every epoch from here
                last_reported_epoch = train_proc.current_epoch
                while True:
                    # Check cancellation
                    if cancel_event and cancel_event.is_set():
                        train_proc.terminate()
                        logger.info("[AutoTune] Cancelled during training wait")
                        _best_effort_final_summary(tuning_result, audit)
                        iter_result.error = "用户取消训练"
                        history.add_attempt(iter_result.to_dict())
                        tuning_result["iterations"].append(iter_result.to_dict())
                        tuning_result["error"] = "用户取消"
                        failure = _persist_iteration_failure(audit, iteration, "execute", "user_cancelled", iter_result.error)
                        if failure["error_type"] == "audit_persistence_error":
                            tuning_result["failure"] = failure
                            tuning_result["error"] = f"审计持久化失败: {failure['message']}"
                            try:
                                history.to_json(os.path.join(log_dir, "tuning_history.json"))
                            except Exception:
                                pass
                            return tuning_result
                        try:
                            audit.finalize("cancelled", failure)
                        except Exception as exc:
                            tuning_result["failure"] = _failure(
                                "audit", "audit_persistence_error",
                                f"{iter_result.error}; 最终审计写入失败: {exc}",
                            )
                            tuning_result["error"] = tuning_result["failure"]["message"]
                            try:
                                history.to_json(os.path.join(log_dir, "tuning_history.json"))
                            except Exception:
                                pass
                            return tuning_result
                        tuning_result["failure"] = failure
                        history.to_json(os.path.join(log_dir, "tuning_history.json"))
                        return tuning_result

                    status = train_proc.poll()
                    if status in ("completed", "failed"):
                        break
                    # Report progress every epoch so SSE stays alive and
                    # the user can see training is still progressing.
                    metrics = train_proc.read_results_csv()
                    if metrics:
                        epoch = int(metrics.get("epoch", 0))
                        if epoch > last_reported_epoch:
                            last_reported_epoch = epoch
                            mAP = metrics.get("metrics/mAP50(B)", "?")
                            if on_progress:
                                on_progress(iteration, f"训练进度: Epoch {epoch}/{merged.get('epochs', '?')}, mAP50={mAP}")
                            _emit_state(on_state, "training", "progress",
                                        f"训练进度: Epoch {epoch}/{merged.get('epochs', '?')}, mAP50={mAP}")
                    time.sleep(10)

                _output_forwarder.join(timeout=2)

                if status == "failed":
                    iter_result.error = "训练进程异常退出"
                    history.add_attempt(iter_result.to_dict())
                    tuning_result["iterations"].append(iter_result.to_dict())
                    if on_progress:
                        on_progress(iteration, f"训练进程异常退出")
                    if auto_loop:
                        failure = _persist_iteration_failure(audit, iteration, "execute", "training_failed", iter_result.error)
                        if failure["error_type"] == "audit_persistence_error":
                            _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                            return tuning_result
                        continue
                    failure = _persist_iteration_failure(audit, iteration, "execute", "training_failed", iter_result.error)
                    _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
                    return tuning_result

                # ── Run shared finalizer: Module B analysis + KPI + unified history ──
                if on_progress:
                    on_progress(iteration, "训练完成，开始 Module B 分析...")
                _emit_state(on_state, "analyzing", "analyzing", "训练完成，开始 Module B 分析...")
                finalizer_result = finalize_training_run(
                    output_dir,
                    train_name,
                    "tuning",
                    config,
                    log_dir=log_dir,
                    training_status="completed",
                    session_id=tuning_session_id,
                    audit_path=audit.path,
                    started_at=training_start_iso,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    tuning_context={
                        "decision": {
                            "diagnosis": decision.get("diagnosis"),
                            "action": decision.get("action"),
                            "hyperparameter_changes": dict(decision.get("hyperparameter_changes", {}) or {}),
                            "training_overrides": dict(decision.get("training_overrides", {}) or {}),
                        },
                        "guardrails": {
                            "valid": guard_result.valid,
                            "warnings": list(guard_result.warnings),
                            "errors": list(guard_result.errors),
                            "clamped": dict(getattr(guard_result, "clamped", {}) or {}),
                        },
                    },
                    runtime_run_id=runtime_run_id,
                    local_index_service=local_index_service,
                    dataset_id=reference_dataset.dataset_id if reference_dataset is not None else None,
                )
                finalizer_metrics = finalizer_result.get("metrics", {})
                iter_result.result_mAP50 = finalizer_metrics.get("mAP50")
                iter_result.result_mAP50_95 = finalizer_metrics.get("mAP50_95")
                iter_result.result_precision = finalizer_metrics.get("precision")
                iter_result.result_recall = finalizer_metrics.get("recall")
                iter_result.result_analysis_status = finalizer_result.get("analysis_status")
                _epochs_info = finalizer_result.get("epochs") or {}
                iter_result.result_best_epoch = _epochs_info.get("best") if isinstance(_epochs_info, dict) else None

                analysis_record = None
                if finalizer_result.get("analysis_status") != "completed":
                    analysis_record = finalizer_result.get("analysis_error")
                    logger.warning("Module B analysis failed for %s: %s", train_name,
                                   (analysis_record or {}).get("message"))
                    if on_progress:
                        on_progress(iteration, f"Module B 分析失败: {(analysis_record or {}).get('message', 'unknown')}")
                elif on_progress:
                    cs = iter_result.get_composite_score(eval_mode)
                    on_progress(iteration, f"Module B 分析完成: mAP50={iter_result.result_mAP50}, mAP50-95={iter_result.result_mAP50_95}, Precision={iter_result.result_precision}, Recall={iter_result.result_recall}, 综合分={cs:.4f}")

                if finalizer_result.get("history_error") and on_progress:
                    on_progress(iteration, f"历史写入警告: {finalizer_result['history_error'].get('message')}")

                # ── Record result facts (only metrics actually present) ──
                after_metrics = _extract_after_metrics(iter_result)
                audit.update_iteration(iteration, result={
                    "before_metrics": before_metrics,
                    "after_metrics": after_metrics,
                    "metric_delta": _metric_delta(before_metrics, after_metrics),
                    "probe": {
                        "verdict": probe_decision.verdict,
                        "reason": probe_decision.reason,
                        "suggestion": probe_decision.suggestion,
                    },
                    "analysis": analysis_record,
                })

                # ── Auto-loop: continue to next iteration ──
                if auto_loop:
                    audit.complete_iteration(iteration)
                    history.add_attempt(iter_result.to_dict())
                    tuning_result["iterations"].append(iter_result.to_dict())
                    reference_run = train_name
                    if on_progress:
                        on_progress(iteration, f"自动循环 → 下一轮 (新参考: {train_name})")
                    continue

                # ── auto_analyze only: return success ──
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                _analysis_status = finalizer_result.get("analysis_status")
                tuning_result["final_result"] = {
                    "train_name": train_name,
                    "iteration": iteration,
                    "decision": decision.get("diagnosis"),
                    "changes": changes,
                    "guard_warnings": guard_result.warnings,
                    "module_b_analyzed": _analysis_status == "completed",
                    "analysis_status": _analysis_status,
                    "analysis_error": finalizer_result.get("analysis_error"),
                }
                if on_progress:
                    if _analysis_status == "completed":
                        on_progress(iteration, f"✅ 训练 {train_name} 完成，Module B 分析已生成")
                    else:
                        on_progress(iteration, f"⚠️ 训练 {train_name} 完成，Module B 分析失败")
                # ── Compute best iteration from all completed iterations ──
                _emit_state(on_state, "finalizing", "finalizing", "调优收尾")
                _compute_best(tuning_result, eval_mode)
                _finalize_final_summary(
                    tuning_result, audit, config, detect_dir, eval_mode,
                    tuning_session_id, tuning_result.get("reference_run"),
                )
                audit.complete_iteration(iteration)
                audit.finalize("completed")
                history.to_json(os.path.join(log_dir, "tuning_history.json"))
                return tuning_result

            # ── No auto-analyze: training continues in background ──
            audit.update_iteration(iteration, result={
                "before_metrics": before_metrics,
                "after_metrics": {},
                "metric_delta": _metric_delta(before_metrics, {}),
                "probe": {
                    "verdict": probe_decision.verdict,
                    "reason": probe_decision.reason,
                    "suggestion": probe_decision.suggestion,
                },
                "analysis": None,
            })
            history.add_attempt(iter_result.to_dict())
            tuning_result["iterations"].append(iter_result.to_dict())
            tuning_result["final_result"] = {
                "train_name": train_name,
                "iteration": iteration,
                "decision": decision.get("diagnosis"),
                "changes": changes,
                "guard_warnings": guard_result.warnings,
            }
            if on_progress:
                on_progress(iteration, f"✅ 训练 {train_name} 已通过探针期，继续进行完整训练")

            # ── Compute best iteration from all completed iterations ──
            _emit_state(on_state, "finalizing", "finalizing", "调优收尾")
            _compute_best(tuning_result, eval_mode)
            _finalize_final_summary(
                tuning_result, audit, config, detect_dir, eval_mode,
                tuning_session_id, tuning_result.get("reference_run"),
            )
            audit.complete_iteration(iteration)
            audit.finalize("completed")
            # Save history
            history.to_json(os.path.join(log_dir, "tuning_history.json"))
            return tuning_result

        except Exception as e:
            iter_result.error = f"异常: {str(e)}"
            failure = _persist_iteration_failure(
                audit, iteration, "loop", "iteration_exception", iter_result.error, fatal=True
            )
            _abort_tuning(tuning_result, iter_result, audit, iteration, failure, history, log_dir, on_progress)
            logger.exception(f"Tuning iteration {iteration} failed")
            return tuning_result

    # All retries exhausted — check if we have any successful iterations
    _compute_best(tuning_result, eval_mode)
    _finalize_final_summary(
        tuning_result, audit, config, detect_dir, eval_mode,
        tuning_session_id, tuning_result.get("reference_run"),
    )
    if tuning_result.get("best_iteration") is not None:
        audit.finalize("completed")
        if on_progress:
            best_m = tuning_result.get("best_metrics", {})
            on_progress(
                tuning_result["best_iteration"],
                f"🏆 最佳迭代: 第 {tuning_result['best_iteration']} 次 ({tuning_result['best_train_name']}) — "
                f"mAP50={best_m.get('mAP50')}, "
                f"mAP50-95={best_m.get('mAP50_95')}, "
                f"Precision={best_m.get('precision')}, "
                f"Recall={best_m.get('recall')}",
            )
    else:
        tuning_result["error"] = f"已达最大重试次数 ({max_retries})，所有尝试均未成功"
        failure = _failure("loop", "retries_exhausted", tuning_result["error"])
        audit.finalize("failed", failure)
        tuning_result["failure"] = failure

    # Save history
    history.to_json(os.path.join(log_dir, "tuning_history.json"))
    return tuning_result
