"""Durable audit records for Module C tuning sessions."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "1.0"
REDACTED = "***REDACTED***"
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_sensitive(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value


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
                redact_sensitive(payload),
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
        },
        "guardrails": {
            "valid": None,
            "warnings": [],
            "errors": [],
            "clamped": {},
            "sanitized_changes": {},
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
            "max_retries": max_retries,
            "iterations": [],
            "error": None,
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
        for key, value in fields.items():
            if key not in record:
                raise KeyError(f"Unknown audit iteration field: {key}")
            record[key] = value
        self.flush()

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

    def flush(self) -> None:
        atomic_write_json(self.path, self.data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)
