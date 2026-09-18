"""Perception layer — aggregate Module A (dataset) and Module B (training) outputs.

Reads JSON reports from the log/ directory, validates minimal schema, and
combines them into a structured perception dict that the Decision Agent uses to
recommend hyperparameter changes.

Availability contract
---------------------
Module C must never disguise an unknown fact as a numeric zero. Every source is
tracked in ``perception["sources"]`` with a stable status:

- ``available``      a schema-valid report was selected;
- ``unavailable``    no candidate file existed;
- ``corrupt``        candidates existed but none parsed;
- ``schema_invalid`` candidates parsed but none satisfied the minimal schema.

Module A candidates are restricted to ``dataset_report.json`` /
``dataset_report_*.json`` — the broad ``*_dataset.json`` / ``dataset_*.json``
rules that matched ``latest_dataset.json`` (a snapshot pointer) are gone.
"""

import glob
import json
import math
import os

# Module A report files must start with this exact prefix. ``latest_dataset.json``
# and snapshot manifests never match, so they can never be misread as analysis.
MODULE_A_PATTERNS = ("dataset_report.json", "dataset_report_*.json")
MODULE_B_GLOB = "*_report.json"

PERCEPTION_DATASET_REPORT_UNAVAILABLE = "PERCEPTION_DATASET_REPORT_UNAVAILABLE"
PERCEPTION_DATASET_REPORT_INVALID = "PERCEPTION_DATASET_REPORT_INVALID"
PERCEPTION_TRAINING_REPORT_UNAVAILABLE = "PERCEPTION_TRAINING_REPORT_UNAVAILABLE"
PERCEPTION_TRAINING_REPORT_INVALID = "PERCEPTION_TRAINING_REPORT_INVALID"


def _is_nonneg_int(value) -> bool:
    """True for a non-negative integer (never bool)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return value.is_integer() and value >= 0
    return False


def _finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_module_a_report(data) -> tuple[bool, str]:
    """Minimal Module A schema: root object, module marker, counts, summary and
    the two analysis sections the decision prompt relies on."""
    if not isinstance(data, dict):
        return False, "root_not_object"
    if data.get("module") != "dataset_analyzer":
        return False, "module_mismatch"
    if not _is_nonneg_int(data.get("total_images")):
        return False, "total_images_invalid"
    if not _is_nonneg_int(data.get("total_annotations")):
        return False, "total_annotations_invalid"
    if not isinstance(data.get("summary"), dict):
        return False, "summary_missing"
    if not isinstance(data.get("class_balance"), dict):
        return False, "class_balance_missing"
    if not isinstance(data.get("bbox_analysis"), dict):
        return False, "bbox_analysis_missing"
    return True, None


def validate_module_b_report(data) -> tuple[bool, str]:
    """Minimal Module B schema: root object, module marker, runs and summary."""
    if not isinstance(data, dict):
        return False, "root_not_object"
    if data.get("module") != "train_analyzer":
        return False, "module_mismatch"
    runs = data.get("runs")
    if not isinstance(runs, dict) or not runs:
        return False, "runs_missing_or_empty"
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return False, "summary_missing"
    best_mAP50 = summary.get("best_mAP50")
    if best_mAP50 is not None and not _finite_number(best_mAP50):
        return False, "best_mAP50_invalid"
    return True, None


def _candidate_sort_key(path: str):
    """Deterministic ordering: mtime descending, basename descending."""
    return (os.path.getmtime(path), os.path.basename(path))


def _module_a_candidates(log_dir: str) -> list[str]:
    candidates = []
    for pat in MODULE_A_PATTERNS:
        candidates.extend(glob.glob(os.path.join(log_dir, pat)))
    return candidates


def _load_json_candidate(path: str):
    """Return the parsed dict, or None when the file is unreadable/corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data


def select_module_a_report(log_dir: str = "log") -> tuple[dict | None, dict]:
    """Select the newest schema-valid Module A report.

    Corrupt/schema-invalid candidates are skipped and the next candidate is
    checked. Returns ``(report, source)`` where ``source`` carries the stable
    status, the chosen basename and a stable error code when unavailable.
    """
    candidates = _module_a_candidates(log_dir)
    candidates.sort(key=_candidate_sort_key, reverse=True)
    if not candidates:
        return None, {
            "status": "unavailable",
            "basename": None,
            "error_code": PERCEPTION_DATASET_REPORT_UNAVAILABLE,
        }
    saw_corrupt = False
    saw_invalid = False
    for path in candidates:
        data = _load_json_candidate(path)
        if data is None:
            saw_corrupt = True
            continue
        ok, _ = validate_module_a_report(data)
        if ok:
            return data, {
                "status": "available",
                "basename": os.path.basename(path),
                "error_code": None,
            }
        saw_invalid = True
    if saw_corrupt and not saw_invalid:
        status = "corrupt"
    elif saw_invalid:
        status = "schema_invalid"
    else:
        status = "unavailable"
    return None, {
        "status": status,
        "basename": os.path.basename(candidates[0]),
        "error_code": PERCEPTION_DATASET_REPORT_INVALID,
    }


def _report_contains_run(data: dict, run_name: str) -> bool:
    """True when a Module B report has a run matching ``run_name``.

    Only the report's own run names are trusted — never a caller-supplied
    path component. This prevents joining an untrusted run_name onto an
    arbitrary absolute path.
    """
    runs = data.get("runs") or {}
    if run_name in runs:
        return True
    for rn, rd in runs.items():
        if isinstance(rd, dict) and rd.get("name") == run_name:
            return True
    return False


def _module_b_candidates(log_dir: str) -> list[str]:
    candidates = glob.glob(os.path.join(log_dir, MODULE_B_GLOB))
    return [p for p in candidates if not os.path.basename(p).startswith("all_")]


def select_module_b_report(log_dir: str, reference_run: str | None) -> tuple[dict | None, dict]:
    """Select the Module B report associated with ``reference_run``.

    The exact ``{reference_run}_report.json`` wins when it is schema-valid and
    contains the run. Otherwise a bounded scan over ``*_report.json`` verifies
    each candidate's schema AND run association before it may be selected; a
    mismatched or invalid candidate never overrides a valid one. With no
    reference_run the newest schema-valid report is returned.
    """
    if reference_run:
        exact = os.path.join(log_dir, f"{reference_run}_report.json")
        exact_exists = os.path.isfile(exact)
        exact_corrupt = False
        exact_invalid = False
        exact_mismatch = False
        if exact_exists:
            exact_data = _load_json_candidate(exact)
            if exact_data is None:
                exact_corrupt = True
            else:
                ok, _ = validate_module_b_report(exact_data)
                if not ok:
                    exact_invalid = True
                elif not _report_contains_run(exact_data, reference_run):
                    exact_mismatch = True
                else:
                    return exact_data, {
                        "status": "available",
                        "basename": f"{reference_run}_report.json",
                        "error_code": None,
                    }

        candidates = _module_b_candidates(log_dir)
        candidates.sort(key=_candidate_sort_key, reverse=True)
        saw_corrupt = False
        saw_invalid = False
        matched = False
        for path in candidates:
            if os.path.normpath(path) == os.path.normpath(exact):
                continue
            data = _load_json_candidate(path)
            if data is None:
                saw_corrupt = True
                continue
            ok, _ = validate_module_b_report(data)
            if not ok:
                saw_invalid = True
                continue
            if not _report_contains_run(data, reference_run):
                continue  # schema-valid but not this reference: no match
            matched = True
            return data, {
                "status": "available",
                "basename": os.path.basename(path),
                "error_code": None,
            }
        # A schema-valid report for another run is a "no match" for this
        # reference: report unavailable, not schema-invalid.
        saw_corrupt = saw_corrupt or exact_corrupt
        saw_invalid = saw_invalid or exact_invalid
        if saw_corrupt and not saw_invalid and not exact_mismatch:
            status = "corrupt"
            error_code = PERCEPTION_TRAINING_REPORT_INVALID
        elif saw_invalid and not exact_mismatch:
            status = "schema_invalid"
            error_code = PERCEPTION_TRAINING_REPORT_INVALID
        else:
            status = "unavailable"
            error_code = PERCEPTION_TRAINING_REPORT_UNAVAILABLE
        return None, {
            "status": status,
            "basename": f"{reference_run}_report.json",
            "error_code": error_code,
        }

    candidates = _module_b_candidates(log_dir)
    candidates.sort(key=_candidate_sort_key, reverse=True)
    for path in candidates:
        data = _load_json_candidate(path)
        if data is None:
            continue
        ok, _ = validate_module_b_report(data)
        if ok:
            return data, {
                "status": "available",
                "basename": os.path.basename(path),
                "error_code": None,
            }
    return None, {
        "status": "unavailable",
        "basename": None,
        "error_code": PERCEPTION_TRAINING_REPORT_UNAVAILABLE,
    }


def find_module_a_report(log_dir: str = "log") -> dict | None:
    """Backward-compatible: return the newest valid Module A report or None."""
    report, _ = select_module_a_report(log_dir)
    return report


def find_module_b_report(run_name: str | None = None, log_dir: str = "log") -> dict | None:
    """Backward-compatible: return the Module B report for run_name or None."""
    report, _ = select_module_b_report(log_dir, run_name)
    return report


def _explicit_source(data, kind: str) -> dict:
    """Source provenance for a caller-supplied report dict."""
    if data is None:
        if kind == "dataset":
            return {"status": "unavailable", "basename": None,
                    "error_code": PERCEPTION_DATASET_REPORT_UNAVAILABLE}
        return {"status": "unavailable", "basename": None,
                "error_code": PERCEPTION_TRAINING_REPORT_UNAVAILABLE}
    return {"status": "available", "basename": None, "error_code": None}


def _build_dataset_section(ds: dict | None) -> dict:
    if not ds:
        return {}
    return {
        "total_images": ds.get("total_images"),
        "total_annotations": ds.get("total_annotations"),
        "label_rate": ds.get("label_coverage", {}).get("label_rate"),
        "class_balance": ds.get("class_balance", {}),
        "image_quality": ds.get("image_quality", {}),
        "bbox_analysis": {
            "tiny_bbox_ratio": ds.get("bbox_analysis", {}).get("tiny_bbox_ratio"),
            "small_bbox_ratio": ds.get("bbox_analysis", {}).get("small_bbox_ratio"),
            "medium_bbox_ratio": ds.get("bbox_analysis", {}).get("medium_bbox_ratio"),
            "large_bbox_ratio": ds.get("bbox_analysis", {}).get("large_bbox_ratio"),
            "avg_relative_area": ds.get("bbox_analysis", {}).get("avg_relative_area"),
        },
        "spatial_bias": ds.get("spatial_bias", {}),
        "class_distribution": ds.get("class_distribution", {}),
        "key_issues": ds.get("summary", {}).get("key_issues", []),
        "quality_score": ds.get("summary", {}).get("dataset_quality_score"),
    }


def _build_training_section(tr: dict | None, reference_run: str | None) -> dict:
    if not tr:
        return {}
    runs = tr.get("runs", {})
    per_run = {}
    for rn, rd in runs.items():
        args = rd.get("args", {})
        results = rd.get("results", {})
        final = results.get("final_metrics", {})
        issues = rd.get("issues", [])
        curves = rd.get("curve_analysis", {})
        per_run[rn] = {
            "model": args.get("model", ""),
            "epochs": args.get("epochs"),
            "batch": args.get("batch"),
            "imgsz": args.get("imgsz"),
            "optimizer": args.get("optimizer", "auto"),
            "lr0": args.get("lr0", 0.01),
            "lrf": args.get("lrf", 0.01),
            "box": args.get("box", 7.5),
            "cls": args.get("cls", 0.5),
            "dfl": args.get("dfl", 1.5),
            "mosaic": args.get("mosaic", 1.0),
            "mixup": args.get("mixup", 0.0),
            "copy_paste": args.get("copy_paste", 0.0),
            "degrees": args.get("degrees", 0.0),
            "weight_decay": args.get("weight_decay", 0.0005),
            "dropout": args.get("dropout", 0.0),
            "patience": args.get("patience"),
            "mAP50": final.get("metrics/mAP50(B)"),
            "mAP50_95": final.get("metrics/mAP50-95(B)"),
            "precision": final.get("metrics/precision(B)"),
            "recall": final.get("metrics/recall(B)"),
            "issues": [
                {"type": i.get("type"), "severity": i.get("severity")}
                for i in (issues or [])
            ],
            # 只投影训练报告真实提供的损失曲线。TrainAnalyzer 不产出 mAP50
            # 指标趋势，投影它只会造出一个永远为空、且事实层会直接拒绝的键。
            "curve_trends": {
                "val_box_loss": curves.get("val_box", {}).get("trend", ""),
                "val_cls_loss": curves.get("val_cls", {}).get("trend", ""),
            },
        }

    ref_metrics = None
    if reference_run and reference_run in per_run:
        ref_metrics = per_run[reference_run]

    training_info = {
        "reference_run": reference_run,
        "total_runs": tr.get("total_runs", len(runs)),
        "best_run": tr.get("summary", {}).get("best_overall_run", ""),
        "best_mAP50": tr.get("summary", {}).get("best_mAP50"),
        "average_mAP50": tr.get("summary", {}).get("average_mAP50"),
        "runs_with_issues": tr.get("summary", {}).get("runs_with_issues", 0),
        "common_issues": tr.get("summary", {}).get("common_issues", []),
        "per_run": per_run,
        "llm_analysis": {
            rn: v.get("llm_diagnosis", "")
            for rn, v in tr.get("llm_analysis", {}).items()
            if isinstance(v, dict)
        },
    }
    if ref_metrics is not None:
        training_info["reference_metrics"] = {
            k: ref_metrics.get(k) for k in
            ("mAP50", "mAP50_95", "precision", "recall")
        }
    return training_info


def build_perception(
    dataset_report: dict | None = None,
    training_report: dict | None = None,
    log_dir: str = "log",
    reference_run: str | None = None,
) -> dict:
    """Build aggregated perception data from Module A and Module B reports.

    Args:
        dataset_report: pre-loaded Module A report, or None to auto-detect.
        training_report: pre-loaded Module B report, or None to auto-detect.
        log_dir: log directory path for auto-detection.
        reference_run: training run the Module B report must be bound to.

    Returns:
        Aggregated perception dict with sections: dataset, training, project
        and a ``sources`` availability contract.
    """
    if dataset_report is None:
        dataset_report, ds_source = select_module_a_report(log_dir)
    else:
        ds_source = _explicit_source(dataset_report, "dataset")
    if training_report is None:
        training_report, tr_source = select_module_b_report(log_dir, reference_run)
    else:
        tr_source = _explicit_source(training_report, "training")

    perception: dict = {
        "dataset": _build_dataset_section(dataset_report),
        "training": _build_training_section(training_report, reference_run),
        "project": {},
        "sources": {
            "dataset_report": ds_source,
            "training_report": tr_source,
        },
    }
    if training_report:
        perception["project"] = training_report.get("project", {})
    return perception


def perception_blocking_error(perception: dict) -> tuple[str | None, str | None]:
    """Return ``(error_code, message)`` when perception facts are unusable.

    The gate only activates on the ``sources`` contract produced by the real
    perception layer. Callers that inject a hand-built perception without
    ``sources`` opt out (used by unit tests of downstream stages).
    """
    sources = perception.get("sources")
    if not isinstance(sources, dict):
        return None, None
    ds = sources.get("dataset_report") or {}
    tr = sources.get("training_report") or {}
    if ds.get("status") != "available":
        code = ds.get("error_code") or PERCEPTION_DATASET_REPORT_UNAVAILABLE
        return code, f"数据集分析报告不可用 ({ds.get('status')})"
    if tr.get("status") != "available":
        code = tr.get("error_code") or PERCEPTION_TRAINING_REPORT_UNAVAILABLE
        return code, f"训练分析报告不可用 ({tr.get('status')})"
    return None, None


def summarize_perception(perception: dict) -> str:
    """Produce a concise text summary of the perception data for LLM prompt building.

    Args:
        perception: dict from build_perception().

    Returns:
        Human-readable markdown summary.
    """
    lines = []
    ds = perception.get("dataset", {})
    tr = perception.get("training", {})
    proj = perception.get("project", {})

    if proj.get("name"):
        lines.append(f"### 项目：{proj['name']}")
    if proj.get("description"):
        lines.append(f"描述：{proj['description']}")
    lines.append("")

    # Dataset summary
    lines.append("### 数据集分析")
    if ds.get("key_issues"):
        lines.append(f"- 关键问题：{', '.join(ds['key_issues'])}")
    lines.append(f"- 图片总数：{ds.get('total_images', '?')}")
    lines.append(f"- 标注总数：{ds.get('total_annotations', '?')}")
    lines.append(f"- 标注率：{ds.get('label_rate', '?')}")
    lines.append(f"- 质量评分：{ds.get('quality_score', '?')}")
    if ds.get("bbox_analysis"):
        b = ds["bbox_analysis"]
        lines.append(f"- Tiny框占比：{b.get('tiny_bbox_ratio', '?')}")
        lines.append(f"- 平均相对面积：{b.get('avg_relative_area', '?')}")
    if ds.get("image_quality"):
        iq = ds["image_quality"]
        lines.append(f"- 模糊率：{iq.get('blur_ratio', '?')}")
        lines.append(f"- 过曝率：{iq.get('overexposure_ratio', '?')}")
        lines.append(f"- 欠曝率：{iq.get('underexposure_ratio', '?')}")
    lines.append("")

    # Training summary
    lines.append("### 训练分析")
    lines.append(f"- 参考训练：{tr.get('reference_run', '?')}")
    lines.append(f"- 分析轮次：{tr.get('total_runs', '?')}")
    lines.append(f"- 最佳训练：{tr.get('best_run', '?')}")
    lines.append(f"- 最佳 mAP50：{tr.get('best_mAP50', '?')}")
    lines.append(f"- 平均 mAP50：{tr.get('average_mAP50', '?')}")

    per_run = tr.get("per_run", {})
    if per_run:
        lines.append("\n#### 参数现状")
        for rn, rd in per_run.items():
            lines.append(f"\n**{rn}** (mAP50={rd.get('mAP50', '?')}, "
                         f"mAP50-95={rd.get('mAP50_95', '?')}, "
                         f"Precision={rd.get('precision', '?')}, "
                         f"Recall={rd.get('recall', '?')})")
            lines.append(f"- 模型：{rd.get('model', '?')}, 优化器：{rd.get('optimizer', '?')}")
            lines.append(f"- lr0={rd.get('lr0')}, lrf={rd.get('lrf')}")
            lines.append(f"- box={rd.get('box')}, cls={rd.get('cls')}, dfl={rd.get('dfl')}")
            lines.append(f"- mosaic={rd.get('mosaic')}, mixup={rd.get('mixup')}, degrees={rd.get('degrees')}")
            lines.append(f"- weight_decay={rd.get('weight_decay')}, dropout={rd.get('dropout')}")
            if rd.get("issues"):
                issues_str = "; ".join(f"[{i['severity']}] {i['type']}" for i in rd["issues"])
                lines.append(f"- 检测问题：{issues_str}")

    return "\n".join(lines)
