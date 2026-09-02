"""Durable audit records for Module C tuning sessions."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_tune.modules.security.credentials import known_credentials
from auto_tune.modules.security.redaction import REDACTED, redact_sensitive

AUDIT_SCHEMA_VERSION = "1.2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: str | os.PathLike, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                redact_sensitive(payload, known_secrets=known_credentials()),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _new_iteration(iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "status": "running",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "baseline": {
            "reference_run": None,
            "params": {},
            "metrics": {},
        },
        "decision": {
            "raw_response": None,
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "schema_version": None,
            "fact_package_id": None,
            "evidence_ids": {},
        },
        "fact_package": None,
        "decision_validation": {
            "valid": None,
            "error_code": None,
            "error_detail": None,
            "retried": False,
            "referenced_fact_ids": [],
        },
        "semantic_validation": {
            "valid": None,
            "error_code": None,
            "reason_code": None,
            "retried": False,
            "parameters": [],
        },
        "decision_attempts": [],
        "guardrails": {
            "valid": None,
            "warnings": [],
            "errors": [],
            "clamped": {},
            "sanitized_changes": {},
        },
        "perception": {
            "status": None,
            "dataset_report_basename": None,
            "dataset_total_images": None,
            "training_report_basename": None,
            "reference_run": None,
            "training_best_mAP50": None,
        },
        "execution": {
            "actual_params": {},
            "args_yaml_path": None,
            "command": [],
            "train_name": None,
        },
        "result": {
            "before_metrics": {},
            "after_metrics": {},
            "metric_delta": {},
            "probe": {"verdict": None, "reason": None, "suggestion": None},
            "analysis": None,
        },
        "error": None,
    }


class TuningAuditSession:
    """Immutable-by-copy, atomically persisted audit session for a tuning run."""

    def __init__(
        self,
        session_id: str,
        log_dir: str,
        reference_run: str | None,
        max_retries: int | None,
        reference_dataset: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.path = str(Path(log_dir) / f"tuning_audit_{session_id}.json")
        self.data: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": session_id,
            "status": "running",
            "started_at": utc_now_iso(),
            "finished_at": None,
            "reference_run": reference_run,
            "reference_dataset": reference_dataset,
            "max_retries": max_retries,
            "iterations": [],
            "error": None,
            "final_summary": None,
            "final_summary_status": None,
        }

    def _get_iteration(self, iteration: int) -> dict[str, Any]:
        for record in self.data["iterations"]:
            if record["iteration"] == iteration:
                return record
        raise KeyError(f"audit iteration {iteration} not found")

    def start_iteration(self, iteration: int) -> dict[str, Any]:
        for record in self.data["iterations"]:
            if record["iteration"] == iteration:
                raise ValueError(f"audit iteration {iteration} already exists")
        new_iter = _new_iteration(iteration)
        self.data["iterations"].append(new_iter)
        return copy.deepcopy(new_iter)

    def update_iteration(self, iteration: int, **fields: object) -> None:
        record = self._get_iteration(iteration)
        for key in fields:
            if key not in record:
                raise KeyError(f"Unknown audit iteration field: {key}")
        # Memory atomicity: apply the fields on deep copies (never on objects a
        # caller still mutates), persist, and only commit the in-memory state
        # when the flush succeeds. On a failed flush the record is rolled back
        # to its pre-call state and the original write exception re-raised, so a
        # later successful flush can never resurrect a modification that never
        # reached disk.
        snapshot = {key: copy.deepcopy(record[key]) for key in fields}
        for key, value in fields.items():
            record[key] = copy.deepcopy(value)
        try:
            self.flush()
        except Exception:
            for key, value in snapshot.items():
                record[key] = value
            raise

    def fail_iteration(
        self,
        iteration: int,
        stage: str,
        error_type: str,
        message: str,
        fatal: bool = True,
    ) -> None:
        record = self._get_iteration(iteration)
        record["status"] = "failed"
        record["finished_at"] = utc_now_iso()
        record["error"] = {
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "fatal": fatal,
            "timestamp": utc_now_iso(),
        }
        # Persist immediately so a crash after ABORT/RETRY does not lose the
        # failure fact. Write failures must propagate; callers translate them
        # into audit_persistence_error (never silently degraded).
        self.flush()

    def complete_iteration(self, iteration: int) -> None:
        record = self._get_iteration(iteration)
        record["status"] = "completed"
        record["finished_at"] = utc_now_iso()

    def finalize(self, status: str, error: dict | None = None) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid final status: {status}")
        self.data["status"] = status
        if error is not None:
            self.data["error"] = error
        self.data["finished_at"] = utc_now_iso()
        self.flush()

    def set_final_summary(
        self, status: str, summary: dict | None = None, error_code: str | None = None
    ) -> None:
        """Record the deterministic final summary facts and its status.

        The summary is already redacted by construction (relative names and
        metric/param facts only); the audit's sensitive-field redaction applies
        on top before persistence.
        """
        self.data["final_summary"] = summary
        self.data["final_summary_status"] = status
        if error_code is not None:
            self.data["final_summary_error_code"] = error_code
        self.flush()

    def flush(self) -> None:
        atomic_write_json(self.path, self.data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)
