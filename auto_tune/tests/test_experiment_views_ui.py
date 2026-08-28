"""Bugfix P5: bilingual report/audit read-only modals wired by run_id.

The history detail buttons open in-page modals that fetch the narrow
report-view / audit-view APIs. Everything is built with textContent (never
unescaped innerHTML), zh/en fixed copy differs, real IDs/names/metrics are
preserved, parameters never render as [object Object], long text wraps, missing
values show the stable dash, and a report failure never blocks the audit button
(or vice versa).
"""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from auto_tune.modules.presentation import (
    build_experiment_labels,
    experiment_enum_label,
    experiment_field_label,
)
from auto_tune.ui import app as app_mod
from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import make_translator

_ZH = make_translator("zh")
_EN = make_translator("en")


def _render(active_page="history", lang="zh", **ctx):
    translator = make_translator(lang)
    defaults = {
        "_": translator,
        "current_lang": lang,
        "experiment_labels": build_experiment_labels(translator),
        "active_page": active_page,
        "experiment_history": [],
        "experiment_history_source": "sqlite",
        "experiment_index_warning": None,
        "dataset_index": [],
        "tuning_history": [],
        "dataset": None,
        "training": None,
        "project": {},
        "latest_suggestion": None,
        "current_args": None,
        "dataset_analyzer_config": {},
        "training_config": {},
        "llm_analysis": None,
        "vision_analysis": None,
        "latest_dataset": None,
        "ai_config": {},
        "csrf_token": "",
    }
    defaults.update(ctx)
    return _jinja_env.get_template("single_page.html").render(**defaults)


def _js_function(html, name, end_marker):
    start = html.index("function " + name)
    end = html.index(end_marker, start)
    return html[start:end]


def _tuning_exp():
    return {
        "run_id": "tuning:u1", "run_name": "autotune_x_iter01", "source": "tuning",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.5}, "epochs": {"completed": 10},
        "finished_at": "2026-08-01T00:00:00Z",
        "audit_filename": "tuning_audit_sess1.json",
        "artifacts": {"report_path": "autotune_x_iter01_report.json", "run_dir": "detect/x"},
    }


def _manual_exp():
    return {
        "run_id": "manual:train1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.5}, "epochs": {"completed": 10},
        "finished_at": "2026-08-01T00:00:00Z",
        "artifacts": {"report_path": "train1_report.json", "run_dir": "detect/train1"},
    }


# ── 33/34. 按钮使用 run_id 且只打开对应 modal ──


def test_history_buttons_use_run_id_and_open_modals_zh():
    html = _render(lang="zh", experiment_history=[_tuning_exp()])
    assert 'data-open-audit' in html
    assert 'data-open-report' in html
    assert 'data-run-id="tuning:u1"' in html
    assert 'id="reportViewModal"' in html
    assert 'id="auditViewModal"' in html
    assert "openReportView(" in html
    assert "openAuditView(" in html


def test_history_buttons_no_raw_json_links():
    html = _render(lang="zh", experiment_history=[_tuning_exp()])
    history_table = html.split('id="historyBody"', 1)[1].split("</tbody>", 1)[0]
    assert 'href="/api/audit/' not in history_table
    assert "report-by-name" not in history_table


# ── 35. 不跳转裸 JSON ──


def test_audit_modal_open_fetches_narrow_api():
    html = _render(lang="zh")
    block = _js_function(html, "openAuditView", "function _renderAuditView")
    assert "/audit-view" in block
    assert "encodeURIComponent(runId)" in block
    assert "report-by-name" not in block


def test_report_modal_open_fetches_narrow_api():
    html = _render(lang="zh")
    block = _js_function(html, "openReportView", "function _renderReportView")
    assert "/report-view" in block
    assert "encodeURIComponent(runId)" in block
    assert "report-by-name" not in block


# ── 38/39/40. 中英文标题、字段、状态正确且不同 ──


def _injected_labels(html):
    import json
    marker = "window._EXPERIMENT_LABELS = "
    start = html.index(marker) + len(marker)
    depth = 0
    for i in range(start, len(html)):
        ch = html[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:i + 1])
    raise AssertionError("window._EXPERIMENT_LABELS not closed")


def test_zh_section_titles():
    html = _render(lang="zh")
    for title in ("训练分析报告", "自动调优审计", "训练概况", "核心指标", "核心参数",
                  "训练问题", "AI 分析", "视觉分析", "产物完整性", "会话概况",
                  "逐轮审计", "最佳结果", "终局总结状态"):
        assert title in html
    # Field labels injected through the shared vocabulary (escaped JSON) cover
    # the iteration-audit column headers.
    labels = _injected_labels(html)
    assert labels["fields"]["suggested_parameters"] == "建议参数"
    assert labels["fields"]["guarded_parameters"] == "护栏后参数"
    assert labels["fields"]["executed_parameters"] == "实际执行参数"
    assert labels["fields"]["metric_delta"] == "指标变化"


def test_en_section_titles():
    html = _render(lang="en")
    for title in ("Training analysis report", "Auto tuning audit", "Training overview",
                  "Core metrics", "Core parameters", "Training issues", "AI analysis",
                  "Vision analysis", "Artifact integrity", "Session overview",
                  "Iteration audit", "Suggested parameters", "Guarded parameters",
                  "Executed parameters", "Metric changes", "Best result",
                  "Final summary status"):
        assert title in html


def test_zh_en_fixed_copy_really_differs():
    zh = _render(lang="zh")
    en = _render(lang="en")
    assert "训练分析报告" in zh and "训练分析报告" not in en
    assert "Training analysis report" in en
    assert "自动调优审计" in zh and "自动调优审计" not in en
    assert "Auto tuning audit" in en
    assert "训练概况" in zh and "Training overview" in en
    assert "逐轮审计" in zh and "Iteration audit" in en


def test_zh_error_and_empty_copy():
    html = _render(lang="zh")
    for text in ("暂无训练分析报告", "报告文件不可用", "报告格式无效",
                 "报告与实验身份不一致", "本地索引暂不可用", "暂无审计记录",
                 "审计文件不可用", "审计格式无效", "审计与实验身份不一致",
                 "审计内容过大", "实验不存在", "无可用视觉分析", "历史原始分析"):
        assert text in html


def test_en_error_and_empty_copy():
    html = _render(lang="en")
    for text in ("No training analysis report.", "Report file unavailable.",
                 "Invalid report format.", "Report identity mismatch.",
                 "Local index temporarily unavailable.", "No audit record.",
                 "Audit file unavailable.", "Invalid audit format.",
                 "Audit identity mismatch.", "Audit content too large.",
                 "Experiment not found.", "No vision analysis available.",
                 "Original stored analysis"):
        assert text in html


# ── 36/37. 报告/审计按多个表格展示 ──


def test_report_render_builds_multiple_tables():
    html = _render(lang="zh")
    block = _js_function(html, "_renderReportView", "function closeReportView")
    assert block.count("_viewSection(") >= 5
    assert "createElement('table')" in _js_function(html, "_viewSection", "function _viewTh")


def test_audit_render_builds_session_and_iteration_tables():
    html = _render(lang="zh")
    block = _js_function(html, "_renderAuditView", "function closeAuditView")
    assert block.count("_viewSection(") >= 4
    assert "_viewColumnTable(" in block


# ── P5 返修：训练问题 issue/severity 走共享枚举翻译，detail 原文保留 ──


def test_report_issues_table_localizes_issue_and_severity():
    html = _render(lang="zh")
    block = _js_function(html, "_renderReportView", "function closeReportView")
    assert "_experimentEnumLabel(iss.issue)" in block
    assert "_experimentEnumLabel(iss.severity)" in block
    # The detail text stays original (no enum/localization transform).
    assert "iss.description" in block
    assert "_experimentEnumLabel(iss.description)" not in block


# ── 41. ID、训练名、模型名、指标保持原样 ──


def test_render_uses_textcontent_preserving_real_values():
    html = _render(lang="zh")
    block = _js_function(html, "_renderReportView", "function closeReportView")
    assert "d.run_name" in block
    assert "d.model_name" in block
    assert "d.metrics" in block
    row = _js_function(html, "_viewRow", "function _viewValue")
    assert "textContent" in row


# ── 42. 参数对象不会显示 [object Object] ──


def test_param_object_format_never_object_object():
    html = _render(lang="zh")
    fmt = _js_function(html, "_formatParamObject", "function _viewColumnTable")
    val = _js_function(html, "_viewValue", "function _formatParamObject")
    for block in (fmt, val):
        assert "[object Object]" not in block
        assert "String(" in block


# ── 43. 长文本换行 ──


def test_long_text_cells_wrap():
    html = _render(lang="zh")
    row = _js_function(html, "_viewRow", "function _viewValue")
    col = _js_function(html, "_viewColumnTable", "function _viewLoading")
    for block in (row, col):
        assert "wordBreak" in block


# ── 44. 缺失值显示 — ──


def test_missing_values_render_dash():
    html = _render(lang="zh")
    val = _js_function(html, "_viewValue", "function _formatParamObject")
    assert "'—'" in val


# ── 45/46. 外部内容使用 textContent，无未经转义 innerHTML ──


def test_report_audit_render_never_uses_innerhtml():
    html = _render(lang="zh")
    report_block = _js_function(html, "_renderReportView", "function closeReportView")
    audit_block = _js_function(html, "_renderAuditView", "function closeAuditView")
    for block in (report_block, audit_block):
        assert "innerHTML" not in block
    row = _js_function(html, "_viewRow", "function _viewValue")
    col = _js_function(html, "_viewColumnTable", "function _viewLoading")
    for block in (row, col):
        assert "innerHTML" not in block


# ── 47. 恶意诊断/run_name/参数值不能执行脚本 ──


def test_malicious_run_name_escaped_in_buttons():
    exp = _tuning_exp()
    exp["run_id"] = 'tuning:x"><script>alert(1)</script>'
    html = _render(lang="zh", experiment_history=[exp])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_injected_p5_copy_never_contains_user_values():
    html = _render(lang="zh")
    labels = _injected_labels(html)
    for section in ("fields", "enums", "booleans"):
        for value in labels[section].values():
            assert "<script>" not in value


# ── 48. 报告故障不影响审计按钮，审计故障不影响报告按钮 ──


def test_manual_report_button_present_without_audit():
    html = _render(lang="zh", experiment_history=[_manual_exp()])
    history_table = html.split('id="historyBody"', 1)[1].split("</tbody>", 1)[0]
    assert 'data-open-report' in history_table
    assert 'data-open-audit' not in history_table


def test_tuning_audit_button_present_with_report():
    html = _render(lang="zh", experiment_history=[_tuning_exp()])
    assert 'data-open-report' in html
    assert 'data-open-audit' in html


# ── 49. P3/P4 行为不回归 ──


def test_p4_shared_labels_still_injected():
    zh = _render(lang="zh")
    labels_start = zh.index("window._EXPERIMENT_LABELS = ") + len("window._EXPERIMENT_LABELS = ")
    import json
    labels = json.loads(zh[labels_start:].split(";\n", 1)[0])
    assert labels["fields"]["run_id"] == "运行 ID"
    assert labels["enums"]["tuning"] == "自动调优"
    assert experiment_field_label("run_name", _ZH) == "训练名称"
    assert experiment_enum_label("manual", _ZH) == "普通训练"


def test_p3_recent_training_not_regressed():
    html = _render(active_page="agent_suggestion", lang="zh")
    start = html.index("function loadRecentTraining")
    block = html[start:html.index("// ── Training folder analyze ──", start)]
    assert "item.run_dir || ''" in block
    assert "innerHTML" not in block


# ── 50. template_xss 保持通过 ──


def test_ai_config_xss_still_escaped():
    malicious = {
        "text": {
            "purpose": "text", "enabled": True,
            "provider": 'x" onfocus="alert(1)',
            "model": "<script>alert(1)</script>",
            "endpoint": "https://example.invalid/v1",
            "allow_private_endpoint": False,
            "default_endpoint": "https://default.invalid/v1",
            "migration_required": False,
        },
        "vision": {
            "purpose": "vision", "enabled": True,
            "provider": 'x" onfocus="alert(4)',
            "model": "<script>alert(4)</script>",
            "endpoint": "https://example.invalid/v1",
            "allow_private_endpoint": False,
            "default_endpoint": "https://default.invalid/v1",
            "migration_required": False,
        },
    }
    html = _render(lang="zh", ai_config=malicious)
    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(4)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# ── modal 结构：标题/关闭按钮/加载态/空态/错误态/滚动 ──


def test_modal_structure_has_close_loading_and_scroll():
    html = _render(lang="zh")
    report_modal = html[html.index('<div id="reportViewModal"'):html.index('<div id="auditViewModal"')]
    audit_modal = html[html.index('<div id="auditViewModal"'):]
    for modal in (report_modal, audit_modal):
        assert "closeReportView" in modal or "closeAuditView" in modal
        assert "overflow-y" in modal or "overflowY" in modal
    loading = _js_function(html, "_viewLoading", "function _viewEmpty")
    assert "Loading" in loading or "加载中" in loading
    report_open = _js_function(html, "openReportView", "function _renderReportView")
    audit_open = _js_function(html, "openAuditView", "function _renderAuditView")
    for block in (report_open, audit_open):
        assert "_viewLoading(" in block
        assert "_viewError(" in block


def test_audit_view_api_route_registered(tmp_path, monkeypatch):
    # The narrow route is reachable without touching the real log dir.
    import os as real_os
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    client = TestClient(app_mod.app)
    resp = client.get("/api/experiments/does-not-exist/audit-view")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "EXPERIMENT_NOT_FOUND"
