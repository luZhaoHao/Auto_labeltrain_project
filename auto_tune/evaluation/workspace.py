"""把离线案例物化成真实的「报告 + 参考运行目录」工作区。

案例只给源材料，事实包一律由生产代码生成：评估运行器先用真实
``build_perception`` 选择报告，再用真实 ``build_tuning_fact_package``
冻结事实包。这样案例集测到的是生产链路，而不是评估脚本自己的拼接。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import yaml

from .cases import EvalCase, get_metric_columns

_CSV_HEADER = (
    "epoch", "time",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
)


@dataclass(frozen=True)
class CaseWorkspace:
    root: str
    log_dir: str
    detect_dir: str
    reference_run: str


def _curve_rows(case: EvalCase, epochs: int) -> list[dict]:
    """生成确定性收敛曲线：末轮指标严格等于案例给定的最终指标。

    缺失指标（``None``）写成空单元格，由 ``parse_results_csv`` 还原为 None，
    以此表达「缺失」而不是零。
    """
    columns = get_metric_columns()
    final = case.final_metrics
    rows: list[dict] = []
    for epoch in range(1, epochs + 1):
        progress = epoch / epochs
        row = {name: "" for name in _CSV_HEADER}
        row["epoch"] = str(epoch)
        row["time"] = f"{3.0 + epoch * 0.1:.5f}"
        row["train/box_loss"] = f"{2.2 - 1.2 * progress:.5f}"
        row["train/cls_loss"] = f"{4.0 - 2.4 * progress:.5f}"
        row["train/dfl_loss"] = f"{2.0 - 1.0 * progress:.5f}"
        row["val/box_loss"] = f"{2.3 - 1.1 * progress:.5f}"
        row["val/cls_loss"] = f"{4.2 - 2.3 * progress:.5f}"
        row["val/dfl_loss"] = f"{2.1 - 0.95 * progress:.5f}"
        row["lr/pg0"] = f"{case.args.get('lr0', 0.01):.6f}"
        row["lr/pg1"] = row["lr/pg0"]
        row["lr/pg2"] = row["lr/pg0"]
        for key, column in columns.items():
            value = final.get(key)
            row[column] = "" if value is None else repr(float(value))
        rows.append(row)
    return rows


def _write_results_csv(path: str, case: EvalCase) -> None:
    epochs = int(case.args.get("epochs", 100))
    rows = _curve_rows(case, epochs)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_HEADER))
        writer.writeheader()
        writer.writerows(rows)


def materialize(case: EvalCase, root: str) -> CaseWorkspace:
    """在 ``root`` 下写出该案例的 log/ 与 detect/ 工作区。"""
    log_dir = os.path.join(root, "log")
    detect_dir = os.path.join(root, "detect")
    run_dir = os.path.join(detect_dir, case.reference_run)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(log_dir, f"dataset_report_ds_{case.case_id}.json"),
              "w", encoding="utf-8") as handle:
        json.dump(case.dataset_report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(log_dir, f"{case.reference_run}_report.json"),
              "w", encoding="utf-8") as handle:
        json.dump(case.training_report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "args.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(case.args, handle, allow_unicode=True, sort_keys=True)
    _write_results_csv(os.path.join(run_dir, "results.csv"), case)

    return CaseWorkspace(
        root=root, log_dir=log_dir, detect_dir=detect_dir,
        reference_run=case.reference_run,
    )
