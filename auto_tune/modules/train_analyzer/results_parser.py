"""Parse YOLO training results: args.yaml + results.csv."""

import os
import csv
import yaml


def parse_args(args_path: str) -> dict:
    """Read and normalize training args.yaml."""
    if not os.path.exists(args_path):
        return {"error": f"args.yaml not found: {args_path}"}
    with open(args_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return args if args else {}


def parse_results_csv(csv_path: str) -> dict:
    """Read results.csv and return structured metrics.

    Returns:
        dict with keys:
            columns: dict mapping column_name -> list of values
            final_metrics: dict of last-epoch metric values
            best_epoch: epoch (1-indexed) with highest mAP50, or None
    """
    if not os.path.exists(csv_path):
        return {"error": f"results.csv not found: {csv_path}"}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # Strip whitespace from fieldnames
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
        rows = []
        for row in reader:
            # Strip whitespace from values, skip empty rows
            clean_row = {k.strip(): v.strip() if v else "" for k, v in row.items()}
            if not clean_row.get("epoch", ""):
                continue
            rows.append(clean_row)

    if not rows:
        return {"columns": {}, "final_metrics": {}, "best_epoch": None}

    columns = {}
    fieldnames = list(rows[0].keys())
    for key in fieldnames:
        values = []
        for r in rows:
            val_str = r.get(key, "").strip()
            if val_str == "" or val_str.lower() == "nan":
                values.append(None)
            else:
                try:
                    values.append(float(val_str))
                except (ValueError, TypeError):
                    values.append(None)
        columns[key] = values

    # Last-epoch metrics
    last = {k: v[-1] if v else None for k, v in columns.items()}
    final_metrics = {k: v for k, v in last.items() if k != "epoch"}

    # Best epoch by mAP50
    best_epoch = None
    map_col = "metrics/mAP50(B)"
    if map_col in columns:
        valid = [(i, v) for i, v in enumerate(columns[map_col]) if v is not None]
        if valid:
            best_idx = max(valid, key=lambda x: x[1])[0]
            best_epoch = int(columns["epoch"][best_idx])

    return {
        "columns": columns,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "total_epochs": len(rows),
    }


def load_training_run(run_dir: str) -> dict:
    """Load and combine training run data from a single run directory.

    Returns:
        dict with "name", "args", "results" keys.
    """
    name = os.path.basename(run_dir.rstrip("/\\"))
    args = parse_args(os.path.join(run_dir, "args.yaml"))
    results = parse_results_csv(os.path.join(run_dir, "results.csv"))
    return {"name": name, "args": args, "results": results}


def find_all_runs(detect_dir: str) -> list[str]:
    """List all training run subdirectories under detect_dir.

    Returns sorted list of directory paths whose names start with 'train'.
    """
    if not os.path.isdir(detect_dir):
        return []
    dirs = []
    for entry in sorted(os.listdir(detect_dir)):
        full = os.path.join(detect_dir, entry)
        if os.path.isdir(full) and entry.startswith("train"):
            dirs.append(full)
    return dirs


def find_latest_run(detect_dir: str) -> str | None:
    """Find the most recently modified training run directory.

    Returns the full path of the newest 'train*' subdirectory, or None if none found.
    """
    runs = find_all_runs(detect_dir)
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)
