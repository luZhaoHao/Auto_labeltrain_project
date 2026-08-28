"""Task 5: minimal UI contract for the local index (Studio S2 Core)."""

from auto_tune.modules.presentation import build_experiment_labels
from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import make_translator

_ZH = make_translator("zh")
_ZH_LABELS = build_experiment_labels(_ZH)

_EXPERIMENTS = [
    {
        "run_id": "manual:1", "run_name": "train1", "source": "manual",
        "status": "completed", "analysis_status": "completed",
        "params": {"model": "yolov8n.pt", "data": "C:/data/ds1/data.yaml"},
        "metrics": {"mAP50": 0.5, "mAP50_95": 0.3, "precision": 0.4, "recall": 0.6},
        "epochs": {"configured": 100, "completed": 10, "best": 9},
        "finished_at": "2026-08-01T00:00:00Z",
        "artifacts": {"report_path": "C:/log/train1_report.json"},
        "dataset_id": "ds1id",
    },
]

_DATASETS = [
    {
        "dataset_id": "ds1id", "display_name": "ds1",
        "canonical_path": "C:/data/ds1", "validation_status": "valid",
        "last_used_at": "2026-08-05T00:00:00Z",
    },
]


def _render(active_page="history", **ctx):
    defaults = {
        "_": _ZH,
        "current_lang": "zh",
        "experiment_labels": _ZH_LABELS,
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


def test_history_page_dataset_filter_and_index_status():
    html = _render(
        active_page="history",
        experiment_history=_EXPERIMENTS,
        experiment_history_source="sqlite",
        dataset_index=_DATASETS,
    )
    assert 'id="historyDatasetFilter"' in html
    assert 'id="historySourceFilter"' in html
    assert 'id="historyStatusFilter"' in html
    assert "SQLite 索引" in html  # zh: "SQLite index"
    assert '<option value="ds1id">ds1</option>' in html
    assert 'data-dataset-id="ds1id"' in html
    # No unimplemented management actions.
    assert "重命名" not in html
    assert "批量删除" not in html
    assert "自动归档" not in html
    assert "导出 Excel" not in html


def test_history_page_json_fallback_shows_records_and_warning():
    html = _render(
        active_page="history",
        experiment_history=_EXPERIMENTS,
        experiment_history_source="json_fallback",
        experiment_index_warning={"error_code": "LOCAL_INDEX_CORRUPT"},
    )
    assert "JSON 回退" in html
    assert "LOCAL_INDEX_CORRUPT" in html
    assert "索引不可用" in html
    # JSON-fallback records are not hidden.
    assert "train1" in html


def test_history_page_import_button_present():
    html = _render(active_page="history")
    assert 'onclick="importLegacyHistory()"' in html
    assert "导入旧记录" in html  # zh: "Import legacy history"


def test_dataset_page_index_summary():
    html = _render(
        active_page="dataset",
        dataset_index=_DATASETS + [
            {
                "dataset_id": "ds2id", "display_name": "ds2",
                "canonical_path": "C:/data/ds2", "validation_status": "invalid",
                "last_used_at": "2026-08-06T00:00:00Z",
            },
        ],
    )
    assert "本地索引" in html  # zh: "Local index"
    assert "ds1" in html and "ds2" in html
    assert "valid" in html and "invalid" in html
    assert "2026-08-06" in html
    assert "重命名" not in html
    assert "批量删除" not in html


def test_dataset_index_values_are_html_escaped():
    datasets = [{
        "dataset_id": "x", "display_name": "<script>alert(1)</script>",
        "canonical_path": "C:/data/x",
        "validation_status": '" onmouseover="alert(2)',
        "last_used_at": None,
    }]
    html = _render(active_page="dataset", dataset_index=datasets)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'onmouseover="alert(2)' not in html
    assert "&#34; onmouseover=&#34;alert(2)" in html


def test_missing_artifact_label_translatable():
    assert _ZH("Missing artifact") == "产物缺失"
    assert make_translator("en")("Missing artifact") == "Missing artifact"


def test_history_page_detail_button_and_modal():
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert "showExperimentDetail('manual:1')" in html
    assert 'id="expDetailModal"' in html
    assert 'id="expDetailBody"' in html
    assert "打开详情" in html  # zh: "Open Detail"
    # The detail modal renders via textContent and never innerHTML.
    assert "innerHTML" not in html.split("function _renderExperimentDetail")[1][:2000]


def test_dataset_page_association_card():
    html = _render(active_page="dataset", dataset_index=_DATASETS)
    assert "datasetAssociationBody" in html
    assert "数据集关联" in html  # zh: "Dataset Association"
    assert "_DATASET_IDS" in html
    assert '"ds1id"' in html


def test_history_page_search_sort_pagination_controls():
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert 'id="historySearchInput"' in html
    assert 'id="historySortSelect"' in html
    assert 'id="historyOrderSelect"' in html
    assert 'id="historyPrevBtn"' in html
    assert 'id="historyNextBtn"' in html
    assert 'id="historyPageInfo"' in html
    assert "historyPrevPage" in html
    assert "historyNextPage" in html
    assert 'data-map50="0.5"' in html
    assert 'data-name="train1"' in html


def test_history_page_diagnostics_section():
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert 'id="historyDiagnostics"' in html
    assert "loadHistoryDiagnostics" in html


def test_history_page_server_side_query_controls():
    """返修 4: history search/sort/filter/pagination must drive GET
    /api/experiments (server-side), never a render-all-then-browser-paginate."""
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert "loadHistoryPage" in html
    assert "_historyQuery" in html
    assert "fetch('/api/experiments?' + _historyQuery())" in html
    assert "URLSearchParams" in html
    assert "offset" in html
    assert "_HISTORY_DEGRADED" in html
    assert 'id="historyDegradedBanner"' in html
    assert "搜索/排序/分页仅限已加载记录" in html
    # The legacy client-side-only pagination entry points are gone.
    assert "historyPrevPage()" in html
    assert "historyNextPage()" in html


def test_history_page_maintenance_controls():
    """返修 8: maintenance UI exposes a read-only audit run and a
    double-confirmed rebuild (POST + CSRF), with result areas."""
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert "runIndexAudit()" in html
    assert "confirmIndexRebuild()" in html
    assert 'id="historyAuditResult"' in html
    assert 'id="historyRebuildResult"' in html
    assert 'id="rebuildBtn"' in html
    assert 'id="historyMaintenance"' in html
    assert "确认重建？再次点击确认" in html
    # Rebuild result surfaces the honest snapshot recovery counts and warning.
    assert "snapshot_unresolved" in html
    assert "快照清单警告" in html
    assert "snapshot_issues" in html
    # No unimplemented management actions were added.
    assert "重命名" not in html
    assert "批量删除" not in html
    assert "自动归档" not in html


def test_history_page_compare_controls():
    html = _render(active_page="history", experiment_history=_EXPERIMENTS)
    assert "openCompareModal" in html
    assert "closeCompareModal" in html
    assert 'id="compareModal"' in html
    assert 'id="compareBody"' in html
    assert 'class="compare-check"' in html
    assert "updateCompareSelection" in html
    # Comparison must not create permanent labels/baseline markers.
    assert "compare-check" in html
    assert "baseline_run_id" in html


def test_history_page_no_invalid_open_action_for_missing_product():
    """Missing products must not generate a dead open action."""
    exp = dict(_EXPERIMENTS[0])
    exp["artifacts"] = {"report_path": None}
    html = _render(active_page="history", experiment_history=[exp])
    # The server-rendered history detail row must not link to a report that is
    # absent (the literal href for this run is not emitted).
    assert 'report-by-name?name=train1' not in html
    assert "audit" not in html.split("exp_details_")[1][:800]


def test_history_page_never_leaks_full_business_paths():
    """Server-rendered history rows must not contain full data/report/run/audit
    paths — only basenames may be shown (返修 6)."""
    exp = dict(_EXPERIMENTS[0])
    exp["source"] = "tuning"
    exp["params"] = {
        "model": "yolov8n.pt",
        "data": "C:/data/ds1/data.yaml",
        "epochs": 100,
    }
    exp["artifacts"] = {
        "report_path": "C:/log/train1_report.json",
        "run_dir": "D:/detect/train1",
    }
    exp["audit_path"] = "C:/log/tuning_audit_s1.json"
    exp["audit_filename"] = "tuning_audit_s1.json"
    exp["run_name"] = "train1"

    html = _render(active_page="history", experiment_history=[exp])

    assert "C:/data/ds1/data.yaml" not in html
    assert "C:/log/train1_report.json" not in html
    assert "D:/detect/train1" not in html
    assert "C:/log/tuning_audit_s1.json" not in html
    # The audit entry is a run_id-bound modal button (Bugfix P5); the full
    # audit path never leaks and no raw-JSON audit link is emitted.
    assert 'data-open-audit' in html
    assert 'data-run-id="manual:1"' in html
    assert 'href="/api/audit/' not in html
