"""Bugfix P4: shared display vocabulary for experiment UI and future P5 reports.

The presentation module maps internal field keys and enum values to stable
translation keys, then localizes through the caller-supplied translator. It is
pure Python (no FastAPI/Jinja/SQLite/browser), so the web UI and any future
report generator consume the exact same labels.
"""

import inspect
import json

from auto_tune.modules.presentation import (
    BOOLEAN_LABEL_KEYS,
    ENUM_LABEL_KEYS,
    FIELD_LABEL_KEYS,
    build_experiment_labels,
    experiment_boolean_label,
    experiment_enum_label,
    experiment_field_label,
)
from auto_tune.ui.i18n import make_translator

_ZH = make_translator("zh")
_EN = make_translator("en")


# ── zh field labels ──


def test_zh_field_labels():
    assert experiment_field_label("run_id", _ZH) == "运行 ID"
    assert experiment_field_label("source", _ZH) == "来源"
    assert experiment_field_label("run_name", _ZH) == "训练名称"
    assert experiment_field_label("status", _ZH) == "运行状态"
    assert experiment_field_label("phase", _ZH) == "运行阶段"
    assert experiment_field_label("model_name", _ZH) == "模型"
    assert experiment_field_label("task_type", _ZH) == "任务类型"
    assert experiment_field_label("analysis_status", _ZH) == "分析状态"
    assert experiment_field_label("validation_status", _ZH) == "校验状态"
    assert experiment_field_label("started_at", _ZH) == "开始时间"
    assert experiment_field_label("finished_at", _ZH) == "完成时间"
    assert experiment_field_label("updated_at", _ZH) == "更新时间"
    assert experiment_field_label("kind", _ZH) == "产物类型"
    assert experiment_field_label("exists_state", _ZH) == "产物可用性"


# ── en field labels ──


def test_en_field_labels():
    assert experiment_field_label("run_id", _EN) == "Run ID"
    assert experiment_field_label("source", _EN) == "Source"
    assert experiment_field_label("run_name", _EN) == "Run name"
    assert experiment_field_label("status", _EN) == "Run status"
    assert experiment_field_label("phase", _EN) == "Run phase"
    assert experiment_field_label("model_name", _EN) == "Model"
    assert experiment_field_label("task_type", _EN) == "Task type"
    assert experiment_field_label("analysis_status", _EN) == "Analysis status"
    assert experiment_field_label("validation_status", _EN) == "Validation status"
    assert experiment_field_label("started_at", _EN) == "Started at"
    assert experiment_field_label("finished_at", _EN) == "Finished at"
    assert experiment_field_label("updated_at", _EN) == "Updated at"
    assert experiment_field_label("kind", _EN) == "Artifact type"
    assert experiment_field_label("exists_state", _EN) == "Artifact availability"


# ── zh enums ──


def test_zh_enums():
    assert experiment_enum_label("manual", _ZH) == "普通训练"
    assert experiment_enum_label("tuning", _ZH) == "自动调优"
    assert experiment_enum_label("completed", _ZH) == "已完成"
    assert experiment_enum_label("running", _ZH) == "运行中"
    assert experiment_enum_label("failed", _ZH) == "失败"
    assert experiment_enum_label("cancelled", _ZH) == "已取消"
    assert experiment_enum_label("interrupted", _ZH) == "已中断"
    assert experiment_enum_label("unknown", _ZH) == "未知"
    assert experiment_enum_label("pending", _ZH) == "等待中"
    assert experiment_enum_label("detect", _ZH) == "目标检测"
    assert experiment_enum_label("classify", _ZH) == "图像分类"
    assert experiment_enum_label("exists", _ZH) == "可用"
    assert experiment_enum_label("missing", _ZH) == "缺失"
    assert experiment_enum_label("unverified", _ZH) == "未验证"
    assert experiment_enum_label("valid", _ZH) == "有效"
    assert experiment_enum_label("invalid", _ZH) == "无效"
    assert experiment_enum_label("sqlite", _ZH) == "SQLite 索引"
    assert experiment_enum_label("json_fallback", _ZH) == "JSON 回退"


# ── en enums ──


def test_en_enums():
    assert experiment_enum_label("manual", _EN) == "Manual training"
    assert experiment_enum_label("tuning", _EN) == "Auto tuning"
    assert experiment_enum_label("completed", _EN) == "Completed"
    assert experiment_enum_label("running", _EN) == "Running"
    assert experiment_enum_label("failed", _EN) == "Failed"
    assert experiment_enum_label("cancelled", _EN) == "Cancelled"
    assert experiment_enum_label("interrupted", _EN) == "Interrupted"
    assert experiment_enum_label("unknown", _EN) == "Unknown"
    assert experiment_enum_label("pending", _EN) == "Pending"
    assert experiment_enum_label("detect", _EN) == "Object detection"
    assert experiment_enum_label("classify", _EN) == "Image classification"
    assert experiment_enum_label("exists", _EN) == "Available"
    assert experiment_enum_label("missing", _EN) == "Missing"
    assert experiment_enum_label("unverified", _EN) == "Unverified"
    assert experiment_enum_label("valid", _EN) == "Valid"
    assert experiment_enum_label("invalid", _EN) == "Invalid"
    assert experiment_enum_label("sqlite", _EN) == "SQLite index"
    assert experiment_enum_label("json_fallback", _EN) == "JSON fallback"


# ── real values are never translated ──


def test_unknown_field_returns_original():
    assert experiment_field_label("no_such_field", _ZH) == "no_such_field"
    assert experiment_field_label("no_such_field", _EN) == "no_such_field"


def test_unknown_enum_returns_original():
    assert experiment_enum_label("no_such_enum", _ZH) == "no_such_enum"
    assert experiment_enum_label("no_such_enum", _EN) == "no_such_enum"


def test_id_values_are_not_translated():
    # run_id / dataset_id / snapshot_id real values are never dictionary keys.
    assert experiment_enum_label("manual:1234", _ZH) == "manual:1234"
    assert experiment_enum_label("tuning:abc:train1", _ZH) == "tuning:abc:train1"
    assert experiment_field_label("manual:1234", _ZH) == "manual:1234"
    assert experiment_field_label("ds_abc123", _ZH) == "ds_abc123"


def test_model_names_run_names_and_numbers_are_not_translated():
    assert experiment_enum_label("yolov8n.pt", _ZH) == "yolov8n.pt"
    assert experiment_enum_label("train1", _ZH) == "train1"
    assert experiment_field_label("train1", _ZH) == "train1"
    assert experiment_enum_label(0.5, _ZH) == 0.5
    assert experiment_field_label(100, _ZH) == 100


def test_error_codes_are_not_translated():
    assert experiment_enum_label("LOCAL_INDEX_CORRUPT", _ZH) == "LOCAL_INDEX_CORRUPT"
    assert experiment_field_label("LOCAL_INDEX_CORRUPT", _ZH) == "LOCAL_INDEX_CORRUPT"
    assert experiment_enum_label("NOT_FOUND", _ZH) == "NOT_FOUND"


def test_none_displays_dash():
    assert experiment_field_label(None, _ZH) == "—"
    assert experiment_enum_label(None, _ZH) == "—"
    assert experiment_field_label(None, _EN) == "—"
    assert experiment_enum_label(None, _EN) == "—"


# ── build_experiment_labels stable structure ──


def test_build_experiment_labels_returns_stable_structure():
    labels = build_experiment_labels(_ZH)
    assert set(labels.keys()) == {"fields", "enums", "booleans"}
    assert isinstance(labels["fields"], dict)
    assert isinstance(labels["enums"], dict)
    assert isinstance(labels["booleans"], dict)
    # Every mapping resolves to a non-empty localized string.
    for key, label in labels["fields"].items():
        assert key and label
    for key, label in labels["enums"].items():
        assert key and label
    for key, label in labels["booleans"].items():
        assert key and label
    # Deterministic and JSON-serializable (the template injects it via tojson).
    assert build_experiment_labels(_ZH) == labels
    json.dumps(labels, ensure_ascii=False)


def test_build_experiment_labels_zh_and_en():
    zh = build_experiment_labels(_ZH)
    en = build_experiment_labels(_EN)
    assert zh["fields"]["run_id"] == "运行 ID"
    assert en["fields"]["run_id"] == "Run ID"
    assert zh["fields"]["run_name"] == "训练名称"
    assert en["fields"]["run_name"] == "Run name"
    assert zh["fields"]["status"] == "运行状态"
    assert en["fields"]["status"] == "Run status"
    assert zh["fields"]["analysis_status"] == "分析状态"
    assert en["fields"]["analysis_status"] == "Analysis status"
    assert zh["enums"]["manual"] == "普通训练"
    assert en["enums"]["manual"] == "Manual training"
    assert zh["enums"]["tuning"] == "自动调优"
    assert en["enums"]["tuning"] == "Auto tuning"
    assert zh["booleans"]["true"] == "是"
    assert en["booleans"]["true"] == "Yes"
    assert zh["booleans"]["false"] == "否"
    assert en["booleans"]["false"] == "No"
    # Same key sets in both languages.
    assert set(zh["fields"]) == set(en["fields"])
    assert set(zh["enums"]) == set(en["enums"])
    assert set(zh["booleans"]) == set(en["booleans"])


def test_build_experiment_labels_matches_function_interface():
    # UI consumes the pre-built dict; a simulated P5 report consumer uses the
    # function interface. Both must agree for every key.
    zh = build_experiment_labels(_ZH)
    en = build_experiment_labels(_EN)
    for key in FIELD_LABEL_KEYS:
        assert zh["fields"][key] == experiment_field_label(key, _ZH)
        assert en["fields"][key] == experiment_field_label(key, _EN)
    for value in ENUM_LABEL_KEYS:
        assert zh["enums"][value] == experiment_enum_label(value, _ZH)
        assert en["enums"][value] == experiment_enum_label(value, _EN)
    for key, _ in (("true", True), ("false", False)):
        assert zh["booleans"][key] == experiment_boolean_label(True if key == "true" else False, _ZH)
        assert en["booleans"][key] == experiment_boolean_label(True if key == "true" else False, _EN)


# ── shared interface: UI consumer and future P5 report consumer agree ──


def test_ui_consumer_and_p5_report_consumer_get_same_text():
    # UI path: the template receives build_experiment_labels(...) output and the
    # JS helpers read from it.
    ui_labels = build_experiment_labels(_ZH)
    # P5 report path: a report generator would call the same public functions.
    p5_fields = {key: experiment_field_label(key, _ZH) for key in FIELD_LABEL_KEYS}
    p5_enums = {value: experiment_enum_label(value, _ZH) for value in ENUM_LABEL_KEYS}
    p5_booleans = {"true": experiment_boolean_label(True, _ZH), "false": experiment_boolean_label(False, _ZH)}
    assert ui_labels["fields"] == p5_fields
    assert ui_labels["enums"] == p5_enums
    assert ui_labels["booleans"] == p5_booleans


# ── artifact kind enum labels ──


def test_artifact_kind_enum_zh():
    assert experiment_enum_label("report", _ZH) == "分析报告"
    assert experiment_enum_label("audit", _ZH) == "审计记录"
    assert experiment_enum_label("run_dir", _ZH) == "训练目录"
    assert experiment_enum_label("results_csv", _ZH) == "训练指标"
    assert experiment_enum_label("args_yaml", _ZH) == "训练参数"
    assert experiment_enum_label("best_pt", _ZH) == "最佳权重"
    assert experiment_enum_label("last_pt", _ZH) == "最终权重"
    assert experiment_enum_label("manifest", _ZH) == "产物清单"


def test_artifact_kind_enum_en():
    assert experiment_enum_label("report", _EN) == "Analysis report"
    assert experiment_enum_label("audit", _EN) == "Audit record"
    assert experiment_enum_label("run_dir", _EN) == "Training directory"
    assert experiment_enum_label("results_csv", _EN) == "Training metrics"
    assert experiment_enum_label("args_yaml", _EN) == "Training arguments"
    assert experiment_enum_label("best_pt", _EN) == "Best weights"
    assert experiment_enum_label("last_pt", _EN) == "Last weights"
    assert experiment_enum_label("manifest", _EN) == "Artifact manifest"


def test_unknown_artifact_kind_falls_back_original():
    assert experiment_enum_label("some_future_kind", _ZH) == "some_future_kind"
    assert experiment_enum_label("some_future_kind", _EN) == "some_future_kind"


# ── P5 返修：训练问题 type / severity 枚举标签 ──


def test_issue_type_enum_labels_zh():
    assert experiment_enum_label("overfitting", _ZH) == "过拟合"
    assert experiment_enum_label("plateau", _ZH) == "平台期"
    assert experiment_enum_label("unstable_training", _ZH) == "训练不稳定"


def test_issue_type_enum_labels_en():
    assert experiment_enum_label("overfitting", _EN) == "Overfitting"
    assert experiment_enum_label("plateau", _EN) == "Plateau"
    assert experiment_enum_label("unstable_training", _EN) == "Unstable training"


def test_severity_enum_labels_zh():
    assert experiment_enum_label("low", _ZH) == "低"
    assert experiment_enum_label("medium", _ZH) == "中"
    assert experiment_enum_label("high", _ZH) == "高"


def test_severity_enum_labels_en():
    assert experiment_enum_label("low", _EN) == "Low"
    assert experiment_enum_label("medium", _EN) == "Medium"
    assert experiment_enum_label("high", _EN) == "High"


def test_unknown_issue_type_falls_back_original():
    assert experiment_enum_label("some_future_issue_type", _ZH) == "some_future_issue_type"
    assert experiment_enum_label("some_future_issue_type", _EN) == "some_future_issue_type"


# ── display boolean labels ──


def test_boolean_labels_zh():
    assert experiment_boolean_label(True, _ZH) == "是"
    assert experiment_boolean_label(False, _ZH) == "否"
    assert experiment_boolean_label(None, _ZH) == "—"


def test_boolean_labels_en():
    assert experiment_boolean_label(True, _EN) == "Yes"
    assert experiment_boolean_label(False, _EN) == "No"
    assert experiment_boolean_label(None, _EN) == "—"


def test_boolean_label_other_values_unchanged():
    # Only real booleans/None are formatted; technical values pass through.
    assert experiment_boolean_label(1, _ZH) == 1
    assert experiment_boolean_label(0, _ZH) == 0
    assert experiment_boolean_label("true", _ZH) == "true"
    assert experiment_boolean_label(0.5, _ZH) == 0.5


def test_boolean_label_keys_are_stable():
    assert BOOLEAN_LABEL_KEYS == {"true": "Yes", "false": "No"}


# ── no framework dependency ──


def test_presentation_module_has_no_framework_dependency():
    module_file = inspect.getsourcefile(experiment_field_label)
    assert module_file
    with open(module_file, encoding="utf-8") as fh:
        source = fh.read()
    lowered = source.lower()
    # The module must not depend on FastAPI, Jinja or SQLite: it stays a pure
    # shared vocabulary so both the web UI and future report generators can
    # reuse it without pulling in any web framework.
    for forbidden in ("fastapi", "jinja2", "sqlite3"):
        assert forbidden not in lowered
    # Strongest guarantee: the module is pure — it has no imports at all, so it
    # can never import a framework, the UI layer or the translation dictionary.
    assert "import " not in source


def test_presentation_module_works_with_plain_callable_translator():
    # A report generator may pass its own translator; the module must not rely
    # on make_translator or i18n at all.
    fake = lambda text: f"[{text}]"
    assert experiment_field_label("run_id", fake) == "[Run ID]"
    assert experiment_enum_label("completed", fake) == "[Completed]"
    labels = build_experiment_labels(fake)
    assert labels["fields"]["run_id"] == "[Run ID]"
    assert labels["enums"]["completed"] == "[Completed]"
