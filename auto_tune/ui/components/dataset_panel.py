"""Module A dataset analysis display helpers."""
import json
import os
from pathlib import Path


def get_dataset_report(log_dir: str = "log") -> dict | None:
    """Load the most recent Module A dataset analysis report."""
    log_path = Path(log_dir)
    # Must match dataset_report_*.json, NOT latest_dataset.json
    ds_reports = sorted(log_path.glob("dataset_report_*.json"), key=os.path.getmtime)
    if not ds_reports:
        return None
    with open(ds_reports[-1], encoding="utf-8") as f:
        return json.load(f)


def format_dataset_summary(report: dict) -> dict:
    """Extract a UI-friendly summary from the dataset report."""
    if not report or "error" in report:
        return {"error": report.get("error", "No data")}

    return {
        "total_images": report.get("total_images", 0),
        "total_annotations": report.get("total_annotations", 0),
        "label_coverage": report.get("label_coverage", {}),
        "image_quality": report.get("image_quality", {}),
        "bbox_analysis": report.get("bbox_analysis", {}),
        "class_balance": report.get("class_balance", {}),
        "class_distribution": report.get("class_distribution", {}),
        "spatial_bias": report.get("spatial_bias", {}),
        "summary": report.get("summary", {}),
    }
