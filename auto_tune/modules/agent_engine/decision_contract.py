"""Q1.1 — TuningDecision v1: the auto-tuning LLM response contract.

``parse_tuning_decision_response`` extracts the JSON object and enforces the
fixed root schema (seven fields, ``schema_version == 1.0``, valid action,
parameter buckets, tunable names, change-count limits). It never interprets
parameter ranges or semantic combinations — those remain the job of
``parameter_registry`` / ``guardrails``.

``validate_decision_evidence`` binds the response to the frozen fact package:
the package id must match the current run, evidence keys must exactly cover the
changed parameters, and every referenced fact must exist in the package.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .parameter_registry import get_tunable_parameter_names

DECISION_SCHEMA_VERSION = "1.0"
ROOT_FIELDS = frozenset({
    "schema_version", "fact_package_id", "diagnosis", "action",
    "hyperparameter_changes", "training_overrides", "evidence_ids",
})
VALID_ACTIONS = frozenset({"adjust", "keep_params"})


class DecisionContractError(ValueError):
    def __init__(self, error_code: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail


def _extract_json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _schema_error(detail: str):
    raise DecisionContractError("DECISION_SCHEMA_INVALID", detail)


def parse_tuning_decision_response(text: str) -> dict:
    """Parse and structurally validate one TuningDecision v1 response.

    Returns a normalized decision dict. Any structural failure raises
    ``DecisionContractError("DECISION_SCHEMA_INVALID", ...)`` whose detail names
    fields only and never embeds raw model text.
    """
    if not isinstance(text, str):
        _schema_error("response must be text")
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        _schema_error("response is not a JSON object")

    if set(parsed) != ROOT_FIELDS:
        extra = sorted(set(parsed) - ROOT_FIELDS)
        missing = sorted(ROOT_FIELDS - set(parsed))
        _schema_error(f"unexpected root fields extra={extra} missing={missing}")

    schema_version = parsed.get("schema_version")
    if schema_version != DECISION_SCHEMA_VERSION:
        _schema_error("schema_version must be 1.0")

    diagnosis = parsed.get("diagnosis")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        _schema_error("diagnosis must be a non-empty string")

    action = parsed.get("action")
    if not isinstance(action, str) or action not in VALID_ACTIONS:
        _schema_error("action must be adjust or keep_params")

    changes = parsed.get("hyperparameter_changes")
    overrides = parsed.get("training_overrides")
    evidence = parsed.get("evidence_ids")
    if not isinstance(changes, dict):
        _schema_error("hyperparameter_changes must be an object")
    if not isinstance(overrides, dict):
        _schema_error("training_overrides must be an object")
    if not isinstance(evidence, dict):
        _schema_error("evidence_ids must be an object")

    overlap = set(changes) & set(overrides)
    if overlap:
        _schema_error(f"parameters must not appear in both buckets: {sorted(overlap)}")

    combined = {**changes, **overrides}
    unknown = sorted(set(combined) - get_tunable_parameter_names())
    if unknown:
        _schema_error(f"unknown parameter(s): {', '.join(unknown)}")

    if action == "adjust":
        if not 1 <= len(combined) <= 3:
            _schema_error("adjust requires 1-3 changed parameters")
    else:  # keep_params
        if combined:
            _schema_error("keep_params requires empty parameter objects")
        if evidence:
            _schema_error("keep_params requires empty evidence_ids")

    return {
        "schema_version": schema_version,
        "fact_package_id": parsed["fact_package_id"],
        "diagnosis": diagnosis.strip(),
        "action": action,
        "hyperparameter_changes": dict(changes),
        "training_overrides": dict(overrides),
        "evidence_ids": dict(evidence),
    }


def validate_decision_evidence(decision: dict, fact_package: dict) -> dict:
    """Bind ``decision`` to ``fact_package`` and verify evidence references.

    Returns the (unchanged) decision dict on success. Raises
    ``DecisionContractError`` with a stable error code otherwise.
    """
    if decision["fact_package_id"] != fact_package["fact_package_id"]:
        raise DecisionContractError(
            "DECISION_FACT_PACKAGE_MISMATCH", "fact_package_id does not match current facts"
        )

    changed = set(decision["hyperparameter_changes"]) | set(decision["training_overrides"])
    evidence = decision["evidence_ids"]
    if set(evidence) != changed:
        raise DecisionContractError(
            "DECISION_EVIDENCE_MISSING", "evidence keys must exactly match changed parameters"
        )

    known = {item["fact_id"] for item in fact_package["facts"]}
    for parameter, ids in evidence.items():
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            raise DecisionContractError(
                "DECISION_EVIDENCE_MISSING", f"invalid evidence for {parameter}"
            )
        if len(ids) != len(set(ids)):
            raise DecisionContractError(
                "DECISION_EVIDENCE_MISSING", f"duplicate evidence for {parameter}"
            )
        unknown = sorted(set(ids) - known)
        if unknown:
            raise DecisionContractError(
                "DECISION_EVIDENCE_UNKNOWN", f"unknown evidence for {parameter}: {', '.join(unknown)}"
            )
    return decision
