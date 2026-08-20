"""Executor — modify YOLO training config and launch/manage training processes.

Supports:
- Reading and writing args.yaml
- Launching YOLO training as a subprocess
- Finding the next available training name
- Tracking running/finished/aborted training state
"""

import os
import re
import shutil
import subprocess
import sys
import time
import yaml
from typing import Any
from pathlib import Path


def resolve_yolo_executable() -> str:
    """Resolve YOLO from PATH or the active Python environment."""
    found = shutil.which("yolo")
    if found:
        return found
    candidates = (
        Path(sys.prefix) / "Scripts" / "yolo.exe",
        Path(sys.prefix) / "Scripts" / "yolo",
        Path(sys.prefix) / "bin" / "yolo",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"YOLO executable not found in PATH or Python environment: {sys.prefix}"
    )


def find_detect_dir(base_dir: str = ".") -> str:
    """Find the detect/ directory containing training runs.

    Searches:
    1. {base_dir}/detect
    2. {base_dir}/runs/detect
    3. YOLO default: runs/detect (relative to cwd)

    Returns:
        Absolute path to the detect directory.
    """
    candidates = [
        os.path.join(base_dir, "detect"),
        os.path.join(base_dir, "runs", "detect"),
        os.path.join(os.getcwd(), "runs", "detect"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    # Fallback: create detect/ in base_dir
    fallback = os.path.abspath(os.path.join(base_dir, "detect"))
    os.makedirs(fallback, exist_ok=True)
    return fallback


def get_next_train_name(detect_dir: str) -> str:
    """Get the next available training name (e.g., train_autotune_1).

    Args:
        detect_dir: path to detect/ directory.

    Returns:
        Name like "train_autotune_1"
    """
    os.makedirs(detect_dir, exist_ok=True)
    existing = set()
    for d in os.listdir(detect_dir):
        if os.path.isdir(os.path.join(detect_dir, d)):
            existing.add(d)

    # Find next autotune number
    max_num = 0
    pattern = re.compile(r"^train_autotune_(\d+)$")
    for name in existing:
        m = pattern.match(name)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num

    return f"train_autotune_{max_num + 1}"


def read_args_yaml(train_dir: str) -> dict:
    """Read args.yaml from a training directory.

    Args:
        train_dir: path to training directory (e.g., detect/train86).

    Returns:
        Dict of training args.
    """
    path = os.path.join(train_dir, "args.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"args.yaml not found in {train_dir}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_training_config(base_args: dict, merged_params: dict, output_dir: str) -> str:
    """Write a new args.yaml with merged hyperparameters.

    Args:
        base_args: original args from a successful training.
        merged_params: validated param changes merged with base_args.
        output_dir: target training directory to write to.

    Returns:
        Path to the written args.yaml.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "args.yaml")

    # Remove save_dir, project, name if present — YOLO computes them from CLI args.
    # Stale values (especially relative paths) can cause YOLO to save to the wrong path.
    for key in ("save_dir", "project", "name"):
        merged_params.pop(key, None)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(merged_params, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return out_path


def prepare_training(
    detect_dir: str,
    reference_run: str,
    merged_params: dict,
) -> tuple[str, str]:
    """Prepare a new training directory with merged hyperparameters.

    Args:
        detect_dir: path to detect/ directory.
        reference_run: name of reference run (e.g., "train86") to copy base args from.
        merged_params: validated hyperparameter changes merged with base args.

    Returns:
        (train_name, args_yaml_path)
    """
    # Get reference args
    ref_dir = os.path.join(detect_dir, reference_run)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference run not found: {ref_dir}")

    # Get next training name
    train_name = get_next_train_name(detect_dir)
    output_dir = os.path.join(detect_dir, train_name)

    # Write config
    args_path = write_training_config(None, merged_params, output_dir)

    return train_name, args_path


def validate_training_preflight(
    reference_run: str | None,
    reference_dir: str | None,
    merged_params: dict,
) -> list[str]:
    """Validate training prerequisites deterministically.

    Returns a list of human-readable errors; empty list means valid.
    Never raises for validation failures.
    """
    errors: list[str] = []

    if reference_run:
        if not reference_dir or not os.path.isdir(reference_dir):
            errors.append(f"Reference directory missing: {reference_dir}")
        else:
            args_path = os.path.join(reference_dir, "args.yaml")
            if not (os.path.isfile(args_path) and os.access(args_path, os.R_OK)):
                errors.append(f"Reference args.yaml missing or unreadable: {args_path}")

    model = merged_params.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append("Model must be a non-empty string")
    else:
        is_local = os.path.isabs(model) or ("/" in model) or ("\\" in model)
        if is_local and not os.path.exists(model):
            errors.append(f"Model file not found: {model}")

    data = merged_params.get("data")
    if not isinstance(data, str) or not data.strip():
        errors.append("Data YAML must be a non-empty string")
    elif not (data.endswith(".yaml") or data.endswith(".yml")):
        errors.append(f"Data path must be a .yaml/.yml file: {data}")
    elif not (os.path.isfile(data) and os.access(data, os.R_OK)):
        errors.append(f"Data YAML missing or unreadable: {data}")

    try:
        resolve_yolo_executable()
    except Exception as exc:
        errors.append(f"YOLO executable unavailable: {exc}")

    return errors


def build_yolo_command(train_name: str, args_path: str, merged_params: dict) -> list[str]:
    """Build the YOLO training command.

    Args:
        train_name: training name (e.g., "train_autotune_1").
        args_path: path to the new args.yaml.
        merged_params: full merged training params.

    Returns:
        Command list for subprocess.
    """
    cmd = [resolve_yolo_executable(), "train"]

    # Map merged params to CLI args
    param_map = {
        "model": "model",
        "data": "data",
        "epochs": "epochs",
        "patience": "patience",
        "batch": "batch",
        "imgsz": "imgsz",
        "device": "device",
        "workers": "workers",
        "optimizer": "optimizer",
        "lr0": "lr0",
        "lrf": "lrf",
        "momentum": "momentum",
        "weight_decay": "weight_decay",
        "warmup_epochs": "warmup_epochs",
        "box": "box",
        "cls": "cls",
        "dfl": "dfl",
        "degrees": "degrees",
        "translate": "translate",
        "scale": "scale",
        "shear": "shear",
        "perspective": "perspective",
        "flipud": "flipud",
        "fliplr": "fliplr",
        "mosaic": "mosaic",
        "mixup": "mixup",
        "copy_paste": "copy_paste",
        "hsv_h": "hsv_h",
        "hsv_s": "hsv_s",
        "hsv_v": "hsv_v",
        "dropout": "dropout",
        "cos_lr": "cos_lr",
        "close_mosaic": "close_mosaic",
        "label_smoothing": "label_smoothing",
        "freeze": "freeze",
        "multi_scale": "multi_scale",
        "rect": "rect",
        "resume": "resume",
        "seed": "seed",
        "deterministic": "deterministic",
        "single_cls": "single_cls",
        "fraction": "fraction",
        "save": "save",
        "save_period": "save_period",
        "val": "val",
        "plots": "plots",
    }

    for yaml_key, cli_flag in param_map.items():
        if yaml_key in merged_params:
            val = merged_params[yaml_key]
            if val is None or (isinstance(val, float) and val != val):  # skip None and NaN
                continue
            if isinstance(val, list):
                cmd.append(f"{cli_flag}={val[0]}")  # single value for imgsz
            else:
                cmd.append(f"{cli_flag}={val}")

    # Force name and project
    # args_path = {detect_dir}/{train_name}/args.yaml
    # We want YOLO to save to {detect_dir}/{train_name}/ so TrainingProcess can find results.csv
    train_dir = os.path.dirname(args_path)       # {detect_dir}/{train_name}
    detect_parent = os.path.dirname(train_dir)    # {detect_dir}
    cmd.append(f"project={detect_parent or 'detect'}")
    cmd.append(f"name={os.path.basename(train_dir)}")
    cmd.append("exist_ok=True")

    return cmd


def launch_training(
    train_name: str,
    args_path: str,
    merged_params: dict,
    command: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a YOLO training subprocess.

    Args:
        train_name: training name.
        args_path: path to config.
        merged_params: merged training params.
        command: exact prebuilt command to launch; when None, build it here.

    Returns:
        Popen process handle.
    """
    cmd = list(command) if command is not None else build_yolo_command(
        train_name, args_path, merged_params
    )
    print(f"[Executor] Launching: {' '.join(cmd)}")

    # Redirect stdout/stderr to a log file to prevent pipe buffer deadlock
    # while still preserving output for debugging.
    train_dir = os.path.dirname(args_path)
    log_path = os.path.join(train_dir, "yolo_train.log")
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        log_file.write(f"# Command: {' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        # Close the parent's handle; the child holds its own inherited handle,
        # so the log file stays valid for the training process lifetime.
        log_file.close()
    return proc


class TrainingProcess:
    """Wrapper around a running YOLO training process with status tracking."""

    def __init__(self, train_name: str, train_dir: str, proc: subprocess.Popen):
        self.train_name = train_name
        self.train_dir = train_dir
        self.proc = proc
        self.status = "running"
        self.start_time = time.time()
        self.current_epoch = 0
        self.latest_metrics: dict[str, float] = {}
        self._output_offset = 0
        self._output_tail = ""

    @property
    def elapsed(self) -> float:
        """Seconds since start."""
        return time.time() - self.start_time

    def poll(self) -> str:
        """Check process status. Returns 'running', 'completed', or 'failed'."""
        if self.status != "running":
            return self.status

        ret = self.proc.poll()
        if ret is None:
            return "running"
        self.status = "completed" if ret == 0 else "failed"
        return self.status

    def terminate(self):
        """Stop the training process."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.status = "aborted"

    def drain_output(self, on_line=None) -> list[str]:
        """Non-blockingly consume newly available decoded output lines once.

        Reads the subprocess output log (``yolo_train.log``) incrementally from
        the last consumed offset. A trailing partial line is carried over so no
        tail line is lost at process end. ``on_line`` (optional) is invoked for
        each decoded line; the full list of consumed lines is also returned.
        """
        lines: list[str] = []
        log_path = os.path.join(self.train_dir, "yolo_train.log")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                f.seek(self._output_offset)
                data = f.read()
                self._output_offset = f.tell()
        except OSError:
            return lines

        if not data:
            return lines

        text = self._output_tail + data
        terminated = text.endswith("\n") or text.endswith("\r")
        pieces = text.splitlines()
        if pieces:
            complete = pieces if terminated else pieces[:-1]
            self._output_tail = "" if terminated else pieces[-1]
        else:
            complete = []
            self._output_tail = text if not terminated else ""
        for ln in complete:
            if on_line is not None:
                on_line(ln)
        lines.extend(complete)
        return lines

    def read_results_csv(self) -> dict | None:
        """Read the latest results.csv and extract metrics."""
        csv_path = os.path.join(self.train_dir, "results.csv")
        if not os.path.exists(csv_path):
            return None
        try:
            import csv
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None
            latest = rows[-1]
            metrics = {}
            for key, val in latest.items():
                key = key.strip()
                try:
                    metrics[key] = float(val)
                except (ValueError, TypeError):
                    continue
            self.current_epoch = int(metrics.get("epoch", 0))
            self.latest_metrics = metrics
            return metrics
        except Exception:
            return None
