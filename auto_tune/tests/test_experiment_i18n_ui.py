"""Bugfix P4: bilingual wiring of the SQLite experiment-management UI.

Renders the real single_page.html through the app's Jinja2 environment with the
shared presentation labels injected (matching production ``_render``), and pins
that the history list, detail modal, compare modal, dataset association and
diagnostics/maintenance copy consume the same shared vocabulary that future P5
report generators will reuse. API JSON contracts must stay untouched.
"""

import json
import re

from fastapi.testclient import TestClient

from auto_tune.modules.presentation import (
    build_experiment_labels,
    experiment_boolean_label,
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


def _extract_experiment_labels(html):
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
    raise AssertionError("window._EXPERIMENT_LABELS not closed in rendered HTML")


def _js_function(html, name, end_marker):
    start = html.index("function " + name)
    end = html.index(end_marker, start)
    return html[start:end]


# ── injected shared labels (UI consumes build_experiment_labels output) ──


def test_ui_injects_shared_zh_labels_from_build_experiment_labels():
    html = _render(lang="zh")
    labels = _extract_experiment_labels(html)
    assert labels == build_experiment_labels(make_translator("zh"))
    assert labels["fields"]["run_id"] == "运行 ID"
    assert labels["fields"]["source"] == "来源"
    assert labels["enums"]["manual"] == "普通训练"
    assert labels["enums"]["completed"] == "已完成"


def test_ui_injects_shared_en_labels_from_build_experiment_labels():
    html = _render(lang="en")
    labels = _extract_experiment_labels(html)
    assert labels == build_experiment_labels(make_translator("en"))
    assert labels["fields"]["run_id"] == "Run ID"
    assert labels["fields"]["source"] == "Source"
    assert labels["enums"]["manual"] == "Manual training"
    assert labels["enums"]["completed"] == "Completed"


# ── helpers fall back on unknown values (no blank / no JS error) ──


def test_experiment_label_helpers_fall_back_to_original():
    html = _render(lang="zh")
    assert "function _experimentFieldLabel(key)" in html
    assert "function _experimentEnumLabel(value)" in html
    helper_src = html[html.index("function _experimentFieldLabel(key)"):]
    # The helper reads the injected map and falls back to the original value.
    assert "hasOwnProperty" in helper_src
    assert "String(key)" in helper_src
    assert "String(value)" in helper_src
    assert "'—'" in helper_src


def test_unknown_backend_field_or_enum_is_not_a_translation_key():
    # Unknown values are not part of the injected map, so the helper falls back
    # to the raw value; no blank, no undefined error.
    labels = _extract_experiment_labels(_render(lang="zh"))
    assert "some_future_field" not in labels["fields"]
    assert "some_future_status" not in labels["enums"]


# ── history list (zh / en) ──


def test_history_zh_list_shows_zh_fields_and_status():
    exp = {
        "run_id": "manual:1", "run_name": "train1", "source": "tuning",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt", "data": "C:/data/ds1/data.yaml"},
        "metrics": {"mAP50": 0.5},
        "epochs": {"completed": 10},
        "finished_at": "2026-08-01T00:00:00Z",
    }
    html = _render(lang="zh", experiment_history=[exp])
    assert "自动调优" in html
    assert "已完成" in html
    assert "train1" in html


def test_history_en_list_shows_en_fields_and_status():
    exp = {
        "run_id": "manual:1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.5},
        "epochs": {"completed": 10},
        "finished_at": "2026-08-01T00:00:00Z",
    }
    html = _render(lang="en", experiment_history=[exp])
    assert "Manual training" in html
    assert "Completed" in html


# ── detail modal ──


def test_detail_modal_zh_uses_shared_field_and_enum_helpers():
    html = _render(lang="zh")
    block = _js_function(
        html, "_renderExperimentDetail", "function showExperimentDetail"
    )
    # Row labels come from the shared field helper, never raw field names.
    assert "_experimentFieldLabel(" in block
    assert "_experimentEnumLabel(" in block
    # Block titles are translated.
    assert "运行身份" in block
    assert "数据集关联" in block
    assert "调优事实" in block
    assert "训练产物" in block


def test_detail_modal_en_shows_natural_english_labels():
    html = _render(lang="en")
    labels = _extract_experiment_labels(html)
    assert labels["fields"]["run_id"] == "Run ID"
    assert labels["fields"]["source"] == "Source"
    assert labels["fields"]["status"] == "Run status"
    # The detail modal helper renders those labels, not the raw field keys.
    block = _js_function(
        html, "_renderExperimentDetail", "function showExperimentDetail"
    )
    assert "Run Identity" in block


def test_source_status_task_analysis_exists_enum_translation_zh():
    labels = _extract_experiment_labels(_render(lang="zh"))
    assert labels["enums"]["manual"] == "普通训练"
    assert labels["enums"]["tuning"] == "自动调优"
    assert labels["enums"]["completed"] == "已完成"
    assert labels["enums"]["detect"] == "目标检测"
    assert labels["enums"]["exists"] == "可用"
    assert labels["enums"]["missing"] == "缺失"


def test_real_id_values_are_preserved():
    # run_id / dataset_id real values appear verbatim in rendered history rows.
    exp = {
        "run_id": "tuning:abc:run1", "run_name": "run1", "source": "tuning",
        "status": "completed", "analysis_status": "completed",
        "dataset_id": "ds_sha256_1", "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.5}, "epochs": {}, "finished_at": "2026-08-01T00:00:00Z",
    }
    html = _render(lang="zh", experiment_history=[exp])
    assert "tuning:abc:run1" in html
    assert 'data-dataset-id="ds_sha256_1"' in html


# ── filter controls: translated text, raw option values ──


def _history_exp():
    return {
        "run_id": "manual:1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt"},
        "metrics": {"mAP50": 0.5}, "epochs": {}, "finished_at": "2026-08-01T00:00:00Z",
    }


def test_filter_controls_translated_but_option_values_raw():
    html = _render(lang="zh", experiment_history=[_history_exp()])
    assert '<option value="manual">' in html
    assert '<option value="tuning">' in html
    assert '<option value="completed">' in html
    assert '<option value="failed">' in html
    assert "全部来源" in html  # zh: "All Sources"
    assert "普通训练" in html
    assert "自动调优" in html


def test_filter_controls_en_option_values_still_raw():
    html = _render(lang="en", experiment_history=[_history_exp()])
    assert '<option value="manual">' in html
    assert '<option value="tuning">' in html
    assert '<option value="completed">' in html


# ── compare modal fixed copy ──


def test_compare_modal_fixed_copy_zh():
    html = _render(lang="zh")
    assert "比较实验" in html
    assert "不可比较" in html
    assert "基线" in html
    assert "相同参数" in html
    assert "事实摘要" in html
    assert "mAP50 最高" in html
    assert "耗时最短" in html
    assert "产物最完整" in html


def test_compare_modal_fixed_copy_en():
    html = _render(lang="en")
    assert "Compare Experiments" in html
    assert "Not comparable" in html
    assert "Baseline" in html
    assert "Common parameters" in html
    assert "Factual summary" in html
    assert "Highest mAP50" in html
    assert "Shortest duration" in html
    assert "Most complete artifacts" in html


def test_compare_modal_uses_shared_field_labels():
    html = _render(lang="zh")
    block = _js_function(html, "_renderCompare", "function openCompareModal")
    assert "_experimentFieldLabel(" in block
    assert "_experimentEnumLabel(" in block


# ── diagnostics / audit / rebuild / import fixed copy ──


def test_diagnostics_and_maintenance_fixed_copy_zh():
    html = _render(lang="zh")
    assert "数据库" in html
    assert "实验数" in html
    assert "数据集数" in html
    assert "产物数" in html
    assert "备份数" in html
    assert "错误码" in html
    assert "审计失败" in html
    assert "重建失败" in html
    assert "导入失败" in html
    assert "请求错误" in html


def test_diagnostics_and_maintenance_fixed_copy_en():
    html = _render(lang="en")
    assert "Database" in html
    assert "Experiments count" in html
    assert "Error code" in html
    assert "Audit failed" in html
    assert "Rebuild failed" in html
    assert "Import failed" in html


# ── pagination copy ──


def test_pagination_copy_zh_and_en():
    zh = _render(lang="zh")
    assert "{{ _(\"Page\") }}" not in zh  # must not leak the jinja call
    en = _render(lang="en")
    assert "Page " in en


# ── textContent / XSS safety ──


def test_detail_modal_renders_via_textcontent_not_innerhtml():
    html = _render(lang="zh")
    block = _js_function(
        html, "_renderExperimentDetail", "function showExperimentDetail"
    )
    assert "textContent" in block
    assert "innerHTML" not in block


def test_compare_modal_renders_via_textcontent_not_innerhtml():
    html = _render(lang="zh")
    block = _js_function(html, "_renderCompare", "function openCompareModal")
    assert "textContent" in block
    assert "innerHTML" not in block


def test_malicious_run_name_and_model_are_escaped():
    exp = {
        "run_id": "manual:1", "run_name": "<script>alert(1)</script>",
        "source": "manual", "status": "completed", "analysis_status": "completed",
        "params": {"model": '" onmouseover="alert(2)'},
        "metrics": {"mAP50": 0.5}, "epochs": {}, "finished_at": "2026-08-01T00:00:00Z",
    }
    html = _render(lang="zh", experiment_history=[exp])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert ' onmouseover="alert(2)' not in html


def test_injected_labels_never_contain_user_values():
    # The injected vocabulary is a fixed structure; user run_name/model values
    # can never appear inside it.
    html = _render(lang="zh")
    labels = _extract_experiment_labels(html)
    for value in labels["fields"].values():
        assert "<script>" not in value
    for value in labels["enums"].values():
        assert "<script>" not in value


# ── P3 recent training does not regress ──


def test_p3_recent_training_not_regressed():
    html = _render(active_page="agent_suggestion", lang="zh")
    block_start = html.index("function loadRecentTraining")
    block = html[block_start:html.index("// ── Training folder analyze ──", block_start)]
    assert "input.value = item.run_dir || ''" in block
    assert "innerHTML" not in block
    assert "普通训练" in block
    assert "自动调优" in block


# ── API JSON contracts unchanged ──


def _make_fake_service():
    class FakeService:
        def query_experiments(self, query):
            return {
                "items": [{
                    "run_id": "manual:1", "run_name": "train1", "source": "manual",
                    "status": "completed", "analysis_status": "completed",
                    "params": {"model": "yolov8n.pt"},
                    "metrics": {"mAP50": 0.5}, "epochs": {},
                    "finished_at": "2026-08-01T00:00:00Z",
                }],
                "total": 1, "limit": query.limit, "offset": query.offset,
            }

        def get_experiment_detail(self, run_id):
            return {
                "run_id": run_id, "source": "manual", "run_name": "train1",
                "status": "completed", "phase": "terminal", "model_name": "yolov8n.pt",
                "task_type": "detect", "started_at": None, "finished_at": None,
                "updated_at": None, "analysis_status": "completed",
                "params": {"model": "yolov8n.pt"}, "metrics": {"mAP50": 0.5},
                "error": None, "epochs": {"configured": 100, "completed": 10, "best": 9},
                "dataset": None, "tuning": None, "artifacts": [],
            }

        def diagnostics(self):
            return {
                "available": True, "schema_version": 1,
                "database_size_bytes": 100, "wal_size_bytes": 0,
                "dataset_count": 0, "experiment_count": 1, "artifact_count": 0,
                "backup_count": 0, "quick_check": "ok", "journal_mode": "wal",
                "recent_import": None, "recent_events": [],
                "recent_error_codes": [], "error_code": None,
            }

        def compare_experiments(self, run_ids, baseline_run_id):
            return {
                "run_ids": run_ids, "baseline_run_id": baseline_run_id,
                "comparable": True, "warnings": [],
                "identity": [{
                    "run_id": rid, "run_name": "run", "source": "manual",
                    "status": "completed", "model_name": "yolov8n.pt",
                    "task_type": "detect", "dataset_id": None,
                    "started_at": None, "finished_at": None,
                } for rid in run_ids],
                "parameters": {"common": {}, "differences": {}},
                "metrics": {"mAP50": {rid: 0.5 for rid in run_ids}, "mAP50_95": {}, "precision": {}, "recall": {}},
                "relative": {"mAP50": {rid: 0.0 for rid in run_ids}, "mAP50_95": {}, "precision": {}, "recall": {}},
                "training": {rid: {"duration_seconds": 60, "epochs": {}, "status": "completed", "analysis_status": "completed"} for rid in run_ids},
                "tuning": {rid: None for rid in run_ids},
                "artifacts": {rid: {"exists": 0, "total": 0, "kinds": {}} for rid in run_ids},
                "summary": {"highest_mAP50": run_ids[0], "shortest_duration": run_ids[0], "most_complete_artifacts": run_ids[0]},
            }

    return FakeService()


def test_api_experiments_contract_unchanged(monkeypatch):
    fake = _make_fake_service()
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: fake)
    client = TestClient(app_mod.app)
    resp = client.get("/api/experiments?limit=25&offset=0&sort=finished_at&order=desc")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    item = body["items"][0]
    assert item["run_id"] == "manual:1"
    assert item["source"] == "manual"
    assert item["status"] == "completed"


def test_api_experiment_detail_contract_unchanged(monkeypatch):
    fake = _make_fake_service()
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: fake)
    client = TestClient(app_mod.app)
    resp = client.get("/api/experiments/manual:1")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "run_id", "source", "run_name", "status", "phase", "model_name",
        "task_type", "started_at", "finished_at", "updated_at",
        "analysis_status", "params", "metrics", "error", "epochs",
        "dataset", "tuning", "artifacts",
    ):
        assert key in body
    assert body["run_id"] == "manual:1"


def test_api_local_index_diagnostics_contract_unchanged(monkeypatch):
    fake = _make_fake_service()
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: fake)
    client = TestClient(app_mod.app)
    resp = client.get("/api/local-index/diagnostics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["quick_check"] == "ok"
    assert "experiment_count" in body
    assert "recent_error_codes" in body


def test_api_experiments_compare_contract_unchanged(monkeypatch):
    fake = _make_fake_service()
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: fake)
    client = TestClient(app_mod.app)
    resp = client.post(
        "/api/experiments/compare",
        headers={
            "X-CSRF-Token": app_mod._CSRF_TOKEN,
            "Origin": "http://testserver",
            "Content-Type": "application/json",
        },
        json={"run_ids": ["manual:1", "manual:2"], "baseline_run_id": "manual:1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "run_ids", "baseline_run_id", "comparable", "warnings", "identity",
        "parameters", "metrics", "relative", "training", "tuning", "artifacts",
        "summary",
    }
    assert body["baseline_run_id"] == "manual:1"


# ── language switch uses the matching labels ──


def test_language_switch_produces_matching_labels():
    zh = _extract_experiment_labels(_render(lang="zh"))
    en = _extract_experiment_labels(_render(lang="en"))
    assert set(zh["fields"]) == set(en["fields"])
    assert set(zh["enums"]) == set(en["enums"])
    assert set(zh["booleans"]) == set(en["booleans"])
    assert zh["fields"]["run_id"] == "运行 ID"
    assert en["fields"]["run_id"] == "Run ID"


# ── 返修 1: unified field terminology across history list / detail / P5 ──


def test_run_name_zh_consistent_across_history_detail_p5():
    exp = _history_exp()
    html = _render(lang="zh", experiment_history=[exp])
    labels = _extract_experiment_labels(html)
    assert labels["fields"]["run_name"] == "训练名称"
    # History list header renders the shared field label.
    assert "训练名称" in html
    # Detail modal helper reads the same injected label.
    block = _js_function(html, "_renderExperimentDetail", "function showExperimentDetail")
    assert "_experimentFieldLabel(kv[0])" in block
    # A simulated P5 consumer returns the identical Chinese.
    assert experiment_field_label("run_name", make_translator("zh")) == "训练名称"
    assert labels["fields"]["run_name"] == experiment_field_label("run_name", make_translator("zh"))
    # The old capitalised duplicate spelling is gone from the whole page.
    assert "运行名称" not in html


def test_status_zh_consistent_across_history_detail_p5():
    exp = _history_exp()
    html = _render(lang="zh", experiment_history=[exp])
    labels = _extract_experiment_labels(html)
    assert labels["fields"]["status"] == "运行状态"
    assert "运行状态" in html
    assert experiment_field_label("status", make_translator("zh")) == "运行状态"
    assert labels["fields"]["status"] == experiment_field_label("status", make_translator("zh"))
    # The old "Training Status" spelling is gone from the whole page.
    assert "训练状态" not in html


def test_english_terms_consistent_across_history_detail_p5():
    exp = _history_exp()
    en = _render(lang="en", experiment_history=[exp])
    zh = _render(lang="zh", experiment_history=[exp])
    en_labels = _extract_experiment_labels(en)
    zh_labels = _extract_experiment_labels(zh)
    for key in ("run_id", "run_name", "status", "analysis_status", "source"):
        assert en_labels["fields"][key] == experiment_field_label(key, make_translator("en"))
        assert zh_labels["fields"][key] == experiment_field_label(key, make_translator("zh"))
    assert en_labels["fields"]["run_name"] == "Run name"
    assert en_labels["fields"]["status"] == "Run status"
    assert en_labels["fields"]["analysis_status"] == "Analysis status"
    assert "Run name" in en
    assert "Run status" in en


# ── 返修 2: artifact kind enum labels ──


def test_artifact_kind_cell_uses_shared_enum_in_detail_modal():
    html = _render(lang="zh")
    block = _js_function(html, "_renderExperimentDetail", "function showExperimentDetail")
    assert "_experimentEnumLabel(a.kind)" in block
    assert "td1.textContent = a.kind;" not in block
    # The injected enums carry every real artifact kind.
    labels = _extract_experiment_labels(html)
    for kind in ("report", "audit", "run_dir", "results_csv", "args_yaml", "best_pt", "last_pt", "manifest"):
        assert kind in labels["enums"]
    assert labels["enums"]["best_pt"] == "最佳权重"
    assert labels["enums"]["manifest"] == "产物清单"


def test_artifact_api_kind_values_preserved(monkeypatch):
    fake = _make_fake_service()
    detail = fake.get_experiment_detail("manual:1")
    detail["artifacts"] = [
        {"kind": "report", "name": "report.json", "status": "exists"},
        {"kind": "run_dir", "name": "train1", "status": "exists"},
        {"kind": "best_pt", "name": "best.pt", "status": "missing"},
        {"kind": "some_future_kind", "name": "x", "status": "unverified"},
    ]
    fake.get_experiment_detail = lambda run_id: detail
    monkeypatch.setattr(app_mod, "_local_index_service", lambda: fake)
    client = TestClient(app_mod.app)
    resp = client.get("/api/experiments/manual:1")
    assert resp.status_code == 200
    kinds = [a["kind"] for a in resp.json()["artifacts"]]
    assert kinds == ["report", "run_dir", "best_pt", "some_future_kind"]


def test_unknown_artifact_kind_falls_back_raw_in_shared_module():
    assert experiment_enum_label("some_future_kind", make_translator("zh")) == "some_future_kind"


# ── 返修 3: display booleans for explicit facts ──


def test_guardrails_boolean_uses_shared_boolean_interface():
    html = _render(lang="zh")
    labels = _extract_experiment_labels(html)
    assert labels["booleans"]["true"] == "是"
    assert labels["booleans"]["false"] == "否"
    block = _js_function(html, "_renderExperimentDetail", "function showExperimentDetail")
    assert "_experimentBooleanLabel(" in block
    # The old enum path for guardrails.valid is gone.
    assert "_experimentEnumLabel(d.tuning.guardrails.valid)" not in block
    # History detail row also formats guardrails.valid through the boolean helper.
    start = html.index("function _historyDetailRow")
    end = html.index("function _historyRow", start)
    hblock = html[start:end]
    assert "_experimentBooleanLabel(guard.valid)" in hblock


def test_boolean_zh_yes_no_dash():
    assert experiment_boolean_label(True, make_translator("zh")) == "是"
    assert experiment_boolean_label(False, make_translator("zh")) == "否"
    assert experiment_boolean_label(None, make_translator("zh")) == "—"


def test_boolean_en_yes_no_dash():
    assert experiment_boolean_label(True, make_translator("en")) == "Yes"
    assert experiment_boolean_label(False, make_translator("en")) == "No"
    assert experiment_boolean_label(None, make_translator("en")) == "—"


def test_params_table_keeps_technical_values_raw():
    html = _render(lang="zh")
    block = _js_function(html, "_renderExperimentDetail", "function showExperimentDetail")
    # The params block still renders raw technical values, never the boolean display.
    assert "_detailRow(k, d.params[k])" in block


# ── 返修 4: unified source terms (no Auto-Tuning / Manual Training) ──


def test_source_term_unified_no_hyphenated_form():
    exp = _history_exp()
    zh = _render(lang="zh", experiment_history=[exp])
    en = _render(lang="en", experiment_history=[exp])
    # Filter options render the shared source forms.
    assert 'value="tuning">自动调优<' in zh
    assert 'value="tuning">Auto tuning<' in en
    assert 'value="manual">普通训练<' in zh
    assert 'value="manual">Manual training<' in en
    # No hyphenated / capitalised source-term Jinja references remain anywhere.
    assert "_(\"Auto-Tuning\")" not in zh and "_(\"Auto-Tuning\")" not in en
    assert "_(\"Manual Training\")" not in zh and "_(\"Manual Training\")" not in en
    # The shared module agrees with what the page renders.
    assert experiment_enum_label("tuning", make_translator("en")) == "Auto tuning"
    assert experiment_enum_label("manual", make_translator("en")) == "Manual training"
