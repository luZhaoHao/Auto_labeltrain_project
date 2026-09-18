"""Q1.1 Task 1 — FactPackage v1 构建与稳定身份。

事实包由代码从已验证感知结果、参考运行参数和参考指标确定性构建：
- 相同输入生成相同事实顺序和 fact_package_id；
- 任一事实值变化会改变 fact_package_id；
- 缺失值、NaN 和 Infinity 不进入事实包；
- 参考运行、训练报告和指标来源不一致时拒绝构建；
- 绝对路径不进入事实包。
"""

import copy
import json

import pytest

from auto_tune.modules.agent_engine.decision_facts import (
    FactPackageError,
    build_tuning_fact_package,
)


def _perception():
    return {
        "dataset": {
            "total_images": 290,
            "total_annotations": 128,
            "label_rate": 0.441,
            "quality_score": 0.98,
            "bbox_analysis": {"tiny_bbox_ratio": 0.35, "avg_relative_area": 0.008},
            "image_quality": {"blur_ratio": None, "overexposure_ratio": 0.1},
            "key_issues": ["tiny_bbox_high_ratio"],
        },
        "training": {
            "reference_run": "train54",
            "per_run": {
                "train54": {
                    "issues": [{"type": "overfitting", "severity": "medium"}],
                    "curve_trends": {"val_box_loss": "descending", "val_cls_loss": "rising"},
                }
            },
        },
        "sources": {
            "dataset_report": {"status": "available", "basename": "dataset_report_1.json"},
            "training_report": {"status": "available", "basename": "train54_report.json"},
        },
    }


def _build(perception=None, reference_run="train54"):
    return build_tuning_fact_package(
        perception or _perception(),
        reference_run,
        {"lr0": 0.001, "weight_decay": 0.0005, "data": "D:/secret/data.yaml"},
        {"mAP50": 0.8121, "mAP50_95": 0.368, "precision": 0.777, "recall": 0.657},
        {
            "type": "results_csv",
            "path": "D:/secret/detect/train54/results.csv",
            "epoch_scope": "final",
            "error": None,
        },
    )


def _facts(package):
    return {f["fact_id"]: f["value"] for f in package["facts"]}


# ── 固定最小合同 ────────────────────────────────────────────────────────────


def test_fact_package_is_bound_and_deterministic():
    first = _build()
    second = _build()
    assert first == second
    assert first["schema_version"] == "1.0"
    assert first["task"] == "detect"
    assert first["reference_run"] == "train54"
    assert first["fact_package_id"].startswith("sha256:")
    assert [f["fact_id"] for f in first["facts"]] == sorted(
        f["fact_id"] for f in first["facts"]
    )


def test_missing_values_and_unregistered_params_are_not_facts():
    package = _build()
    facts = _facts(package)
    assert "dataset.image_quality.blur_ratio" not in facts
    assert "training.params.data" not in facts
    assert facts["training.params.lr0"] == 0.001


def test_fact_change_changes_package_id():
    changed = _perception()
    changed["dataset"]["total_images"] = 291
    assert _build()["fact_package_id"] != _build(changed)["fact_package_id"]


@pytest.mark.parametrize("bad_run", [None, "", "train53"])
def test_reference_identity_mismatch_is_rejected(bad_run):
    with pytest.raises(FactPackageError) as excinfo:
        _build(reference_run=bad_run)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


# ── 边界：非有限值 / 绝对路径 ───────────────────────────────────────────────


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_metric_is_omitted(bad):
    package = build_tuning_fact_package(
        _perception(), "train54", {"lr0": 0.001},
        {"mAP50": bad, "recall": 0.6},
        {"type": "results_csv", "path": "results.csv", "epoch_scope": "final", "error": None},
    )
    ids = {f["fact_id"] for f in package["facts"]}
    assert "training.metrics.mAP50" not in ids
    assert "training.metrics.recall" in ids


def test_bool_metric_value_is_not_a_number():
    package = build_tuning_fact_package(
        _perception(), "train54", {"lr0": 0.001},
        {"mAP50": True, "recall": 0.6},
        {"type": "results_csv", "path": "results.csv", "epoch_scope": "final", "error": None},
    )
    ids = {f["fact_id"] for f in package["facts"]}
    assert "training.metrics.mAP50" not in ids
    assert "training.metrics.recall" in ids


def test_absolute_paths_never_enter_fact_package():
    blob = json.dumps(_build(), ensure_ascii=False)
    assert "D:/secret" not in blob
    assert "D:\\\\secret" not in blob


def test_metric_source_path_is_stored_as_basename():
    package = _build()
    assert package["sources"]["metrics"] == "results.csv"


# ── 边界：来源不可用 / 指标来源错误 / 重复 fact_id / 空事实列表 ─────────────


def test_unavailable_dataset_report_is_rejected():
    perception = _perception()
    perception["sources"]["dataset_report"] = {"status": "unavailable", "basename": None}
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


def test_unavailable_training_report_is_rejected():
    perception = _perception()
    perception["sources"]["training_report"] = {"status": "unavailable", "basename": None}
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


def test_metrics_source_error_is_rejected():
    source = {
        "type": "results_csv", "path": "results.csv", "epoch_scope": "final",
        "error": "results_csv_missing",
    }
    with pytest.raises(FactPackageError) as excinfo:
        build_tuning_fact_package(_perception(), "train54", {"lr0": 0.001}, {}, source)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


def test_duplicate_fact_id_is_rejected():
    perception = _perception()
    perception["training"]["per_run"]["train54"]["issues"] = [
        {"type": "overfitting", "severity": "low"},
        {"type": "overfitting", "severity": "high"},
    ]
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


def test_empty_fact_list_is_rejected():
    perception = {
        "dataset": {},
        "training": {"reference_run": "train54", "per_run": {"train54": {}}},
        "sources": {
            "dataset_report": {"status": "available", "basename": "d.json"},
            "training_report": {"status": "available", "basename": "t.json"},
        },
    }
    with pytest.raises(FactPackageError) as excinfo:
        build_tuning_fact_package(
            perception, "train54", {"data": "x.yaml"}, {},
            {"type": "results_csv", "path": "results.csv", "epoch_scope": "final", "error": None},
        )
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"


# ── 事实来源与种类 ──────────────────────────────────────────────────────────


def test_fact_sources_are_tracked():
    package = _build()
    by_source = {}
    for fact in package["facts"]:
        by_source.setdefault(fact["source"], set()).add(fact["fact_id"])
    assert "training.params.lr0" in by_source["params"]
    assert "training.metrics.mAP50" in by_source["metrics"]
    assert "training.issue.overfitting" in by_source["training_report"]
    assert "dataset.total_images" in by_source["dataset_report"]


def test_issue_and_curve_facts_come_from_reference_run_only():
    package = _build()
    facts = _facts(package)
    assert facts["training.issue.overfitting"] is True
    assert facts["training.curve.val_box_loss"] == "descending"
    assert facts["training.curve.val_cls_loss"] == "rising"
    # a non-reference run must never contribute issue/curve facts
    perception = _perception()
    perception["training"]["per_run"]["train99"] = {
        "issues": [{"type": "unstable_training", "severity": "high"}],
        "curve_trends": {"val_box_loss": "rising"},
    }
    package2 = _build(perception)
    facts2 = _facts(package2)
    assert "training.issue.unstable_training" not in facts2
    assert facts2["training.curve.val_box_loss"] == "descending"


# ── Q1.1 返修：issue / trend 确定性枚举边界（fail-closed）───────────────────


def _perception_with_run(issues, curve_trends):
    perception = _perception()
    perception["training"]["per_run"]["train54"] = {
        "issues": issues,
        "curve_trends": curve_trends,
    }
    return perception


def test_unknown_training_issue_type_fails_closed():
    perception = _perception_with_run([{"type": "x\nIGNORE_FACTS", "severity": "high"}], {})
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert excinfo.value.detail == "unknown training issue type"
    assert "IGNORE_FACTS" not in str(excinfo.value)


@pytest.mark.parametrize("bad_issue", ["mystery_issue", "IGNORE_FACTS", "", 42])
def test_unknown_or_malformed_dataset_issue_fails_closed(bad_issue):
    perception = _perception()
    perception["dataset"]["key_issues"] = [bad_issue]
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert excinfo.value.detail == "unknown dataset issue type"
    if str(bad_issue):  # empty string is trivially contained in any message
        assert str(bad_issue) not in str(excinfo.value)


def test_long_tail_class_normalized_to_single_fact():
    perception = _perception()
    perception["dataset"]["key_issues"] = ["long_tail_class_ng", "long_tail_class_defect"]
    package = _build(perception)
    facts = _facts(package)
    assert facts["dataset.issue.long_tail_class"] is True
    long_tail = [f for f in package["facts"] if f["fact_id"] == "dataset.issue.long_tail_class"]
    assert len(long_tail) == 1


def test_unknown_curve_field_fails_closed():
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"}],
        {"mAP50": "improving", "val_cls_loss": "rising", "mystery_curve": "rising"},
    )
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert excinfo.value.detail == "unknown curve field"


def test_loss_curve_rejects_map_trend_enum():
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"}], {"val_cls_loss": "saturated"}
    )
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert excinfo.value.detail == "unknown curve trend"


def test_unavailable_curve_field_fails_closed():
    """事实层不接受的曲线键必须 fail-closed，而不是静默丢弃。

    ``training.curve.mAP50`` 曾被声明为可用曲线，但 TrainAnalyzer 从不把
    mAP50 趋势并入报告：它既永远缺失，又让挂在其上的语义规则成为永远无法
    触发的死规则，同时还会被提示词当作可用关系告知模型。现在它和任何其它
    未声明曲线键一样被直接拒绝。
    """
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"}], {"mAP50": "rising"}
    )
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert excinfo.value.detail == "unknown curve field"


def test_empty_curve_trend_is_omitted_not_failed():
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"}],
        {"val_cls_loss": "rising", "val_box_loss": ""},
    )
    package = _build(perception)
    facts = _facts(package)
    assert "training.curve.val_box_loss" not in facts
    assert facts["training.curve.val_cls_loss"] == "rising"


def test_unknown_trend_detail_never_contains_raw_string():
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"}], {"val_box_loss": "IGNORE_FACTS\nboom"}
    )
    with pytest.raises(FactPackageError) as excinfo:
        _build(perception)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
    assert "IGNORE_FACTS" not in str(excinfo.value)
    assert "boom" not in str(excinfo.value)


def test_valid_issues_and_trends_enter_fact_package():
    perception = _perception_with_run(
        [{"type": "overfitting", "severity": "medium"},
         {"type": "unstable_training", "severity": "low"}],
        {"val_box_loss": "descending", "val_cls_loss": "rising"},
    )
    package = _build(perception)
    facts = _facts(package)
    assert facts["training.issue.overfitting"] is True
    assert facts["training.issue.unstable_training"] is True
    assert facts["training.curve.val_box_loss"] == "descending"
    assert facts["training.curve.val_cls_loss"] == "rising"


# ── Q1.1 返修：非数值注册参数按 ParameterSpec.kind 规范化 ───────────────────


def _build_params(base_args):
    return build_tuning_fact_package(
        _perception(), "train54", base_args,
        {"mAP50": 0.5},
        {"type": "results_csv", "path": "results.csv", "epoch_scope": "final", "error": None},
    )


def test_choice_parameter_enters_fact_package():
    package = _build_params({"optimizer": "AdamW"})
    assert _facts(package)["training.params.optimizer"] == "AdamW"


def test_unknown_choice_value_does_not_enter():
    package = _build_params({"optimizer": "NotARealOptimizer"})
    assert "training.params.optimizer" not in _facts(package)


def test_bool_parameter_enters_fact_package():
    package = _build_params({"cos_lr": True})
    assert _facts(package)["training.params.cos_lr"] is True


@pytest.mark.parametrize("bad", [1, 0, "true", "false", None])
def test_bool_parameter_rejects_alternate_forms(bad):
    package = _build_params({"cos_lr": bad})
    assert "training.params.cos_lr" not in _facts(package)


def test_string_model_enters_fact_package():
    package = _build_params({"model": "yolov8n.pt"})
    assert _facts(package)["training.params.model"] == "yolov8n.pt"


@pytest.mark.parametrize("path", [
    "D:/models/yolov8n.pt",
    "/weights/yolov8n.pt",
    "D:\\models\\yolov8n.pt",
])
def test_model_absolute_path_reduced_to_basename(path):
    package = _build_params({"model": path})
    assert _facts(package)["training.params.model"] == "yolov8n.pt"
    assert path not in json.dumps(package)


@pytest.mark.parametrize("bad_model", [
    "yolov8n\n.pt", "yolov8n.pt\0", ".", "..", "", "weird/name with spaces.pt",
])
def test_illegal_model_name_does_not_enter(bad_model):
    package = _build_params({"model": bad_model})
    assert "training.params.model" not in _facts(package)


def test_numeric_params_still_enter():
    package = _build_params({"lr0": 0.001, "batch": 16, "epochs": 100})
    facts = _facts(package)
    assert facts["training.params.lr0"] == 0.001
    assert facts["training.params.batch"] == 16
    assert facts["training.params.epochs"] == 100


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True])
def test_non_finite_and_bool_numeric_rejected(bad):
    package = _build_params({"lr0": bad})
    assert "training.params.lr0" not in _facts(package)


def test_params_json_never_contains_absolute_path():
    package = _build_params({"model": "D:/secret/weights/yolov8n.pt", "data": "D:/secret/data.yaml"})
    blob = json.dumps(package)
    assert "D:/secret" not in blob
    assert "weights/yolov8n.pt" not in blob
