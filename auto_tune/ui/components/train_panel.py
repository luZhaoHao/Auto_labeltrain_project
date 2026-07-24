"""Module B training analysis display helpers."""
import json
import os
from pathlib import Path


def get_training_report(log_dir: str = "log") -> dict | None:
    """Load the most recent Module B training analysis report."""
    log_path = Path(log_dir)
    candidates = list(log_path.glob("*_report.json"))
    # Exclude all_ prefix and dataset prefix
    reports = [p for p in candidates if not p.name.startswith("all_") and "dataset" not in p.name.lower()]
    if not reports:
        return None
    latest = max(reports, key=os.path.getmtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def get_project_info(log_dir: str = "log") -> dict:
    """Extract project info from the training report."""
    report = get_training_report(log_dir)
    if report:
        return report.get("project", {})
    return {}
