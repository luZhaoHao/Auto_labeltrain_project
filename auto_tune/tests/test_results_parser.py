"""Tests for results_parser.py."""

import os
import yaml
import pytest
from auto_tune.modules.train_analyzer.results_parser import (
    parse_args, parse_results_csv, load_training_run, find_all_runs, find_latest_run,
)


SAMPLE_CSV = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
    "1,1.5,3.0,2.0,0.1,0.2,0.05,0.01,1.6,3.1,2.1\n"
    "2,1.3,2.8,1.9,0.15,0.3,0.08,0.02,1.4,2.9,2.0\n"
    "3,1.2,2.5,1.8,0.2,0.35,0.12,0.03,1.3,2.7,1.9\n"
    "4,1.1,2.3,1.7,0.25,0.4,0.15,0.04,1.2,2.5,1.8\n"
    "5,1.0,2.0,1.6,0.3,0.45,0.2,0.05,1.1,2.3,1.7\n"
)


def test_parse_args(tmp_path):
    args_file = tmp_path / "args.yaml"
    args_data = {"model": "yolov8s.yaml", "epochs": 100, "imgsz": [640, 640]}
    with open(args_file, "w") as f:
        yaml.dump(args_data, f)

    result = parse_args(str(args_file))
    assert result["model"] == "yolov8s.yaml"
    assert result["epochs"] == 100


def test_parse_args_not_found(tmp_path):
    result = parse_args(str(tmp_path / "nonexistent.yaml"))
    assert "error" in result


def test_parse_results_csv(tmp_path):
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(SAMPLE_CSV, encoding="utf-8")

    result = parse_results_csv(str(csv_file))
    assert result["total_epochs"] == 5
    assert result["best_epoch"] == 5  # highest mAP50 at epoch 5
    assert result["final_metrics"]["metrics/mAP50(B)"] == 0.2
    assert len(result["columns"]["epoch"]) == 5


def test_parse_results_csv_with_nan(tmp_path):
    csv_file = tmp_path / "results.csv"
    csv_data = (
        "epoch,train/box_loss,metrics/mAP50(B)\n"
        "1,1.5,0.1\n"
        "2,nan,0.2\n"
        "3,1.2,nan\n"
    )
    csv_file.write_text(csv_data, encoding="utf-8")
    result = parse_results_csv(str(csv_file))
    assert result["total_epochs"] == 3
    assert result["columns"]["train/box_loss"][1] is None
    assert result["columns"]["metrics/mAP50(B)"][2] is None
    assert result["best_epoch"] == 2  # epoch 2 has mAP50=0.2


def test_parse_results_csv_not_found(tmp_path):
    result = parse_results_csv(str(tmp_path / "nonexistent.csv"))
    assert "error" in result


def test_load_training_run(tmp_path):
    args_file = tmp_path / "args.yaml"
    yaml.dump({"model": "yolov8s.yaml", "epochs": 100}, open(args_file, "w"))
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(SAMPLE_CSV, encoding="utf-8")

    result = load_training_run(str(tmp_path))
    assert result["name"] == os.path.basename(str(tmp_path))
    assert result["args"]["model"] == "yolov8s.yaml"
    assert result["results"]["total_epochs"] == 5


def test_find_all_runs(tmp_path):
    for name in ["train", "train2", "train10", "not_a_run"]:
        (tmp_path / name).mkdir()
    (tmp_path / "some_file.txt").write_text("hello")

    runs = find_all_runs(str(tmp_path))
    run_names = [os.path.basename(r) for r in runs]
    assert "train" in run_names
    assert "train2" in run_names
    assert "train10" in run_names
    assert "not_a_run" not in run_names
    assert len(run_names) == 3


def test_find_all_runs_not_found(tmp_path):
    runs = find_all_runs(str(tmp_path / "nonexistent"))
    assert runs == []


def test_find_latest_run(tmp_path):
    (tmp_path / "train_old").mkdir()
    import time
    time.sleep(0.1)
    (tmp_path / "train_new").mkdir()

    latest = find_latest_run(str(tmp_path))
    assert latest is not None
    assert os.path.basename(latest) == "train_new"


def test_find_latest_run_empty(tmp_path):
    latest = find_latest_run(str(tmp_path))
    assert latest is None
