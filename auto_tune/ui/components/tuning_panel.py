"""Module C auto-tuning display helpers."""
import json
import os
from pathlib import Path


def get_tuning_history(log_dir: str = "log") -> list:
    """Load auto-tuning history."""
    path = Path(log_dir) / "tuning_history.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_tuning_status() -> str:
    """Check if a tuning process is currently running."""
    status_file = Path("log") / "tuning_running.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("status", "idle")
    return "idle"
