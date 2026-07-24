"""Perception layer — aggregate Module A (dataset) and Module B (training) outputs.

Reads JSON reports from the log/ directory and combines them into a structured
perception dict that the Decision Agent uses to recommend hyperparameter changes.
"""

import json
import os
import glob


def find_module_a_report(log_dir: str = "log") -> dict | None:
    """Find the most recent Module A (dataset analyzer) JSON report."""
    # Module A reports are named dataset_report.json or {prefix}_dataset.json
    patterns = ["dataset_report.json", "*_dataset.json", "dataset_*.json"]
    candidates = []
    for pat in patterns:
        for path in glob.glob(os.path.join(log_dir, pat)):
            candidates.append(path)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    try:
        with open(latest, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_module_b_report(run_name: str | None = None, log_dir: str = "log") -> dict | None:
    """Find Module B (training analyzer) JSON report.

    Args:
        run_name: specific run like "train8". If None, finds latest.
        log_dir: path to log directory.

    Returns:
        Parsed JSON report or None.
    """
    if run_name:
        path = os.path.join(log_dir, f"{run_name}_report.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    # Find latest *_report.json (excluding all_)
    candidates = []
    for path in glob.glob(os.path.join(log_dir, "*_report.json")):
        basename = os.path.basename(path)
        if basename.startswith("all_"):
            continue
        candidates.append(path)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def build_perception(
    dataset_report: dict | None = None,
    training_report: dict | None = None,
    log_dir: str = "log",
) -> dict:
    """Build aggregated perception data from Module A and Module B reports.

    Args:
        dataset_report: pre-loaded Module A report, or None to auto-detect.
        training_report: pre-loaded Module B report, or None to auto-detect.
        log_dir: log directory path for auto-detection.

    Returns:
        Aggregated perception dict with sections: dataset, training, project.
    """
    if dataset_report is None:
        dataset_report = find_module_a_report(log_dir)
    if training_report is None:
        training_report = find_module_b_report(log_dir)

    perception: dict = {
        "dataset": {},
        "training": {},
        "project": {},
    }

    # ── Module A: dataset analysis ──
    if dataset_report:
        ds = dataset_report
        dataset_info = {
            "total_images": ds.get("total_images", 0),
            "total_annotations": ds.get("total_annotations", 0),
            "label_rate": ds.get("label_coverage", {}).get("label_rate", 0),
            "class_balance": ds.get("class_balance", {}),
            "image_quality": ds.get("image_quality", {}),
            "bbox_analysis": {
                "tiny_bbox_ratio": ds.get("bbox_analysis", {}).get("tiny_bbox_ratio", 0),
                "small_bbox_ratio": ds.get("bbox_analysis", {}).get("small_bbox_ratio", 0),
                "medium_bbox_ratio": ds.get("bbox_analysis", {}).get("medium_bbox_ratio", 0),
                "large_bbox_ratio": ds.get("bbox_analysis", {}).get("large_bbox_ratio", 0),
                "avg_relative_area": ds.get("bbox_analysis", {}).get("avg_relative_area", 0),
            },
            "spatial_bias": ds.get("spatial_bias", {}),
            "class_distribution": ds.get("class_distribution", {}),
            "key_issues": ds.get("summary", {}).get("key_issues", []),
            "quality_score": ds.get("summary", {}).get("dataset_quality_score", 0),
        }
        perception["dataset"] = dataset_info

    # ── Module B: training analysis ──
    if training_report:
        tr = training_report
        training_info = {
            "total_runs": tr.get("total_runs", 0),
            "best_run": tr.get("summary", {}).get("best_overall_run", ""),
            "best_mAP50": tr.get("summary", {}).get("best_mAP50", 0),
            "average_mAP50": tr.get("summary", {}).get("average_mAP50", 0),
            "runs_with_issues": tr.get("summary", {}).get("runs_with_issues", 0),
            "common_issues": tr.get("summary", {}).get("common_issues", []),
        }

        # Extract per-run details (focus on the best or first run)
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
                "epochs": args.get("epochs", 0),
                "batch": args.get("batch", 16),
                "imgsz": args.get("imgsz", 640),
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
                "patience": args.get("patience", 50),
                "mAP50": final.get("metrics/mAP50(B)", 0),
                "mAP50_95": final.get("metrics/mAP50-95(B)", 0),
                "precision": final.get("metrics/precision(B)", 0),
                "recall": final.get("metrics/recall(B)", 0),
                "issues": [
                    {"type": i.get("type"), "severity": i.get("severity")}
                    for i in (issues or [])
                ],
                "curve_trends": {
                    "val_box_loss": curves.get("val_box", {}).get("trend", ""),
                    "val_cls_loss": curves.get("val_cls", {}).get("trend", ""),
                    "mAP50": curves.get("mAP50", {}).get("trend", ""),
                },
            }

        training_info["per_run"] = per_run
        training_info["llm_analysis"] = {
            rn: v.get("llm_diagnosis", "")
            for rn, v in tr.get("llm_analysis", {}).items()
            if isinstance(v, dict)
        }
        perception["training"] = training_info

        # Project context from training report
        perception["project"] = tr.get("project", {})

    return perception


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
    lines.append(f"- 分析轮次：{tr.get('total_runs', '?')}")
    lines.append(f"- 最佳训练：{tr.get('best_run', '?')}")
    lines.append(f"- 最佳 mAP50：{tr.get('best_mAP50', '?')}")
    lines.append(f"- 平均 mAP50：{tr.get('average_mAP50', '?')}")

    per_run = tr.get("per_run", {})
    if per_run:
        lines.append("\n#### 参数现状")
        for rn, rd in per_run.items():
            lines.append(f"\n**{rn}** (mAP50={rd.get('mAP50', '?')})")
            lines.append(f"- 模型：{rd.get('model', '?')}, 优化器：{rd.get('optimizer', '?')}")
            lines.append(f"- lr0={rd.get('lr0')}, lrf={rd.get('lrf')}")
            lines.append(f"- box={rd.get('box')}, cls={rd.get('cls')}, dfl={rd.get('dfl')}")
            lines.append(f"- mosaic={rd.get('mosaic')}, mixup={rd.get('mixup')}, degrees={rd.get('degrees')}")
            lines.append(f"- weight_decay={rd.get('weight_decay')}, dropout={rd.get('dropout')}")
            if rd.get("issues"):
                issues_str = "; ".join(f"[{i['severity']}] {i['type']}" for i in rd["issues"])
                lines.append(f"- 检测问题：{issues_str}")

    return "\n".join(lines)
