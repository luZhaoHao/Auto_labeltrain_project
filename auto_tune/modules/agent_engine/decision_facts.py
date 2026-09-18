"""Q1.1 — FactPackage v1: deterministic, source-bound facts for auto-tuning.

Only this module decides which facts enter the frozen package consumed by the
LLM decision contract. Facts are collected from the allowlisted perception
fields, the reference-run ``args.yaml`` parameters (registry-whitelisted) and
the reference ``results.csv`` metrics. Missing values, non-finite numbers,
unregistered parameters and absolute paths never become facts.

The package identity is the SHA-256 of the canonical payload (stable key order,
compact separators, ``fact_package_id`` excluded from its own hash).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

from .parameter_registry import (
    FACT_PARAMETER_SPECS,
    PARAMETER_REGISTRY,
    ParameterSpec,
)

# 参考运行 args.yaml 中可进入事实包的参数规格：可调参数 + 只读事实参数（model）。
# model 只有「事实可引用」资格，没有「可修改」资格。
_FACT_SPECS: dict[str, ParameterSpec] = {**PARAMETER_REGISTRY, **FACT_PARAMETER_SPECS}

FACT_PACKAGE_SCHEMA_VERSION = "1.0"

# Deterministic enum boundaries for issue / curve facts. Unknown values fail
# closed so arbitrary strings can never become legal, LLM-referenceable facts.
DATASET_ISSUE_TYPES = frozenset({
    "low_label_coverage", "tiny_bbox_high_ratio", "center_spatial_bias", "high_blur_ratio",
})
LONG_TAIL_PREFIX = "long_tail_class_"
TRAINING_ISSUE_TYPES = frozenset({
    "parse_error", "nan_loss", "overfitting", "underfitting", "plateau",
    "low_final_map", "unstable_training", "early_stop_too_soon",
})
# 只包含训练报告真会提供的曲线。TrainAnalyzer 仅并入 analyze_loss_curves 的结果，
# mAP50 指标趋势（analyze_metric_curves）从未进入报告，因此这里不声明它：声明了
# 就等于允许一个永远缺失的事实存在，并让依赖它的语义规则成为死规则。
CURVE_FIELDS = frozenset({"val_box_loss", "val_cls_loss"})
LOSS_TRENDS = frozenset({"plateaued", "descending", "rising"})

# String parameters (currently only ``model``) are reduced to a safe basename.
_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_MAX_MODEL_BASENAME = 255


class FactPackageError(ValueError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.error_code = "FACT_PACKAGE_INVALID"
        self.detail = detail


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _append(facts: list[dict], fact_id: str, value: Any, source: str) -> None:
    if value is None or value == "":
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    facts.append({"fact_id": fact_id, "value": value, "source": source})


def _canonical_payload(package: dict) -> bytes:
    payload = {k: v for k, v in package.items() if k != "fact_package_id"}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _package_id(package: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical_payload(package)).hexdigest()


def _normalize_parameter_fact(name: str, value: Any, spec: ParameterSpec) -> Any | None:
    """Normalize one registry parameter into a fact value, or None when illegal.

    Values are normalized by ``ParameterSpec.kind`` so optimizer / model /
    cos_lr (choice / string / bool) can enter the fact package in a safe,
    canonical form. Unknown or malformed values never create a fact; Guardrails
    keeps the final responsibility for executable parameter legality.
    """
    kind = spec.kind
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if kind == "int":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if kind == "choice":
        if not isinstance(value, str) or value not in spec.choices:
            return None
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            return None
        return value
    if kind == "string":
        if not isinstance(value, str) or not value:
            return None
        base = value.replace("\\", "/").rsplit("/", 1)[-1]
        if not base or base in (".", "..") or len(base) > _MAX_MODEL_BASENAME:
            return None
        if not _SAFE_BASENAME_RE.fullmatch(base):
            return None
        return base
    return None


def _build_dataset_issue_facts(dataset: dict, facts: list[dict], source: str) -> None:
    """Collect dataset issue facts from a deterministic enum.

    Long-tail classes collapse to the single fixed fact ``dataset.issue.long_tail_class``
    (never the raw class name). Repeated issues produce one fact. Unknown or
    malformed issue values fail closed with a fixed, safe detail.
    """
    added: set[str] = set()
    for issue in dataset.get("key_issues") or ():
        if not isinstance(issue, str) or not issue:
            raise FactPackageError("unknown dataset issue type")
        if issue.startswith(LONG_TAIL_PREFIX):
            if "long_tail_class" in added:
                continue
            added.add("long_tail_class")
            _append(facts, "dataset.issue.long_tail_class", True, source)
            continue
        if issue not in DATASET_ISSUE_TYPES:
            raise FactPackageError("unknown dataset issue type")
        if issue in added:
            continue
        added.add(issue)
        _append(facts, f"dataset.issue.{issue}", True, source)


def _build_dataset_facts(dataset: dict) -> list[dict]:
    facts: list[dict] = []
    source = "dataset_report"
    for key in ("total_images", "total_annotations", "label_rate", "quality_score"):
        _append(facts, f"dataset.{key}", dataset.get(key), source)
    cb = dataset.get("class_balance")
    if isinstance(cb, dict):
        _append(facts, "dataset.class_balance.is_balanced", cb.get("is_balanced"), source)
    ba = dataset.get("bbox_analysis")
    if isinstance(ba, dict):
        for key in ("tiny_bbox_ratio", "small_bbox_ratio", "avg_relative_area"):
            _append(facts, f"dataset.bbox_analysis.{key}", ba.get(key), source)
    iq = dataset.get("image_quality")
    if isinstance(iq, dict):
        for key in ("blur_ratio", "overexposure_ratio", "underexposure_ratio"):
            _append(facts, f"dataset.image_quality.{key}", iq.get(key), source)
    _build_dataset_issue_facts(dataset, facts, source)
    return facts


def _build_reference_run_facts(perception: dict, reference_run: str) -> list[dict]:
    facts: list[dict] = []
    per_run = perception.get("training", {}).get("per_run", {}) or {}
    ref = per_run.get(reference_run)
    if not isinstance(ref, dict):
        return facts
    for issue in ref.get("issues") or ():
        if not isinstance(issue, dict):
            raise FactPackageError("unknown training issue type")
        issue_type = issue.get("type")
        if not isinstance(issue_type, str) or issue_type not in TRAINING_ISSUE_TYPES:
            raise FactPackageError("unknown training issue type")
        _append(facts, f"training.issue.{issue_type}", True, "training_report")
    curves = ref.get("curve_trends") or {}
    if not isinstance(curves, dict):
        raise FactPackageError("unknown curve field")
    for name, trend in curves.items():
        if name not in CURVE_FIELDS:
            raise FactPackageError("unknown curve field")
        # An empty trend means the source report had no curve data: the fact is
        # omitted (spec §6.2 missing-value rule), never fabricated. Only a
        # non-empty, unknown trend fails closed.
        if isinstance(trend, str) and trend == "":
            continue
        if not isinstance(trend, str):
            raise FactPackageError("unknown curve trend")
        if trend not in LOSS_TRENDS:
            raise FactPackageError("unknown curve trend")
        _append(facts, f"training.curve.{name}", trend, "training_report")
    return facts


def build_tuning_fact_package(
    perception: dict,
    reference_run: str,
    base_args: dict,
    before_metrics: dict,
    metrics_source: dict,
) -> dict:
    """Return a normalized, stably hashed FactPackage v1.

    Raises ``FactPackageError`` (FACT_PACKAGE_INVALID) when identity, sources,
    metrics source or the resulting fact list cannot form a legal package.
    """
    if not isinstance(reference_run, str) or not reference_run:
        raise FactPackageError("reference_run is empty")
    training = perception.get("training") or {}
    if training.get("reference_run") != reference_run:
        raise FactPackageError("reference_run does not match perception")

    sources = perception.get("sources") or {}
    ds_source = sources.get("dataset_report") or {}
    tr_source = sources.get("training_report") or {}
    if ds_source.get("status") != "available" or not ds_source.get("basename"):
        raise FactPackageError("dataset report source is unavailable")
    if tr_source.get("status") != "available" or not tr_source.get("basename"):
        raise FactPackageError("training report source is unavailable")
    if metrics_source.get("error") is not None:
        raise FactPackageError("metrics source has an error")
    if not metrics_source.get("path"):
        raise FactPackageError("metrics source path is empty")

    facts: list[dict] = []
    facts.extend(_build_dataset_facts(perception.get("dataset") or {}))

    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        value = before_metrics.get(key)
        if _finite(value):
            _append(facts, f"training.metrics.{key}", value, "metrics")

    for key, value in (base_args or {}).items():
        spec = _FACT_SPECS.get(key)
        if spec is None:
            continue
        normalized = _normalize_parameter_fact(key, value, spec)
        if normalized is None:
            continue
        _append(facts, f"training.params.{key}", normalized, "params")

    facts.extend(_build_reference_run_facts(perception, reference_run))

    if not facts:
        raise FactPackageError("no facts can be built for this reference run")

    facts.sort(key=lambda f: f["fact_id"])
    seen: set[str] = set()
    for fact in facts:
        if fact["fact_id"] in seen:
            raise FactPackageError(f"duplicate fact_id: {fact['fact_id']}")
        seen.add(fact["fact_id"])

    package = {
        "schema_version": FACT_PACKAGE_SCHEMA_VERSION,
        "fact_package_id": "",
        "task": "detect",
        "reference_run": reference_run,
        "sources": {
            "dataset_report": ds_source["basename"],
            "training_report": tr_source["basename"],
            "metrics": os.path.basename(str(metrics_source.get("path") or "results.csv")),
            "params": "args.yaml",
        },
        "facts": facts,
    }
    package["fact_package_id"] = _package_id(package)
    return package
