"""Tests for the unified training-completion contract consumed by the UI."""

import json

from auto_tune.modules.train_analyzer.experiment_history import ExperimentHistoryStore
from auto_tune.ui.app import _finalize_and_build_event
from auto_tune.ui.components.experiment_panel import get_experiment_history


def _fake_result(status="completed", analysis_status="completed", metrics=None):
    return {
        "run_id": "manual:train1",
        "run_name": "train1",
        "source": "manual",
        "status": status,
        "analysis_status": analysis_status,
        "metrics": metrics or {},
        "artifacts": {"report_path": None},
        "error": None,
        "analysis_error": None,
        "history_error": None,
    }


def test_completion_event_calls_finalizer_once_with_manual_completed(tmp_path, monkeypatch):
    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, started_at=None, finished_at=None, **kw):
        calls.append({
            "run_name": run_name,
            "source": source,
            "training_status": training_status,
        })
        return _fake_result()

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    event = _finalize_and_build_event(
        0, "/tmp/detect/train1", "train1", {"train_analyzer": {}},
        str(tmp_path), "2026-08-01T00:00:00Z",
    )

    assert calls == [{"run_name": "train1", "source": "manual", "training_status": "completed"}]
    assert event["status"] == "done"
    assert event["level"] == "success"
    assert event["result"]["status"] == "completed"


def test_failure_event_calls_finalizer_with_manual_failed(tmp_path, monkeypatch):
    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, started_at=None, finished_at=None, training_error=None, **kw):
        calls.append({"run_name": run_name, "training_status": training_status, "training_error": training_error})
        result = _fake_result(status="failed", analysis_status="skipped")
        result["error"] = training_error
        return result

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    event = _finalize_and_build_event(
        1, "/tmp/detect/train2", "train2", {"train_analyzer": {}},
        str(tmp_path), None,
    )

    assert calls[0]["training_status"] == "failed"
    assert calls[0]["training_error"]["error_type"] == "training_process_failed"
    assert "退出码 1" in calls[0]["training_error"]["message"]
    assert event["status"] == "error"
    assert event["result"]["error"]["error_type"] == "training_process_failed"


def test_cancelled_event_records_user_cancelled(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    app_mod._running_training["status"] = "aborted"
    calls = []

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, started_at=None, finished_at=None, training_error=None, **kw):
        calls.append(training_error)
        result = _fake_result(status="failed", analysis_status="skipped")
        result["error"] = training_error
        return result

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    try:
        event = _finalize_and_build_event(
            1, "/tmp/detect/train_c", "train_c", {"train_analyzer": {}},
            str(tmp_path), None,
        )
        assert calls[0]["error_type"] == "user_cancelled"
        assert event["result"]["error"]["error_type"] == "user_cancelled"
    finally:
        app_mod._running_training.clear()


def test_partial_success_emits_warning_not_error(tmp_path, monkeypatch):
    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status, started_at=None, finished_at=None, **kw):
        return _fake_result(status="completed", analysis_status="failed")

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    event = _finalize_and_build_event(
        0, "/tmp/detect/train3", "train3", {"train_analyzer": {}},
        str(tmp_path), None,
    )

    assert event["status"] == "done"
    assert event["level"] == "warning"
    assert "分析失败" in event["message"]
    assert event["result"]["analysis_status"] == "failed"


def test_completion_event_integration_uses_real_finalizer(tmp_path):
    run_dir = tmp_path / "detect" / "train4"
    run_dir.mkdir(parents=True)
    (run_dir / "args.yaml").write_text("model: yolov8n.pt\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.25,0.50,0.40,0.20\n",
        encoding="utf-8",
    )

    event = _finalize_and_build_event(
        0, str(run_dir), "train4", {"train_analyzer": {}}, str(tmp_path / "log"), None,
    )

    assert event["status"] == "done"
    assert event["result"]["metrics"]["mAP50"] == 0.4
    assert event["result"]["metrics"]["precision"] == 0.25
    assert event["result"]["artifacts"]["report_path"] is not None


def test_get_experiment_history_empty_when_files_absent(tmp_path):
    assert get_experiment_history(str(tmp_path)) == []


def test_get_experiment_history_sorts_newest_first(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"))
    store.upsert({"run_id": "manual:a", "run_name": "a", "source": "manual", "finished_at": "2026-08-01T00:00:00Z"})
    store.upsert({"run_id": "manual:b", "run_name": "b", "source": "manual", "finished_at": "2026-08-05T00:00:00Z"})

    experiments = get_experiment_history(str(tmp_path))

    assert [e["run_name"] for e in experiments] == ["b", "a"]


def test_get_experiment_history_merges_legacy_tuning(tmp_path):
    (tmp_path / "tuning_history.json").write_text(
        json.dumps([{"iteration": 1, "train_name": "autotune_1", "timestamp": "2026-08-01T00:00:00Z"}]),
        encoding="utf-8",
    )

    experiments = get_experiment_history(str(tmp_path))

    assert len(experiments) == 1
    assert experiments[0]["run_id"] == "legacy-tuning:autotune_1"
    assert experiments[0]["source"] == "tuning"


def _render_history_page(experiments):
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="history",
        experiment_history=experiments,
        tuning_history=[],
        dataset=None,
        training=None,
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )


def test_history_renders_sources_metrics_and_analysis_status():
    manual = {
        "run_id": "manual:train39",
        "run_name": "train39",
        "source": "manual",
        "status": "completed",
        "analysis_status": "completed",
        "metrics": {"mAP50": 0.4, "mAP50_95": 0.2, "precision": 0.3, "recall": 0.5},
        "epochs": {"configured": 100, "completed": 3, "best": 2},
        "finished_at": "2026-08-01T00:00:00Z",
        "artifacts": {"report_path": "/tmp/r.json", "run_dir": "/tmp/detect/train39"},
        "params": {"model": "yolov8n.pt", "epochs": 100, "batch": 16},
    }
    tuning = {
        "run_id": "tuning:s1:autotune_1",
        "run_name": "autotune_1",
        "source": "tuning",
        "status": "completed",
        "analysis_status": "failed",
        "metrics": {"mAP50": 0.5},
        "finished_at": "2026-08-02T00:00:00Z",
        "decision": {"diagnosis": "学习率偏高", "hyperparameter_changes": {"lr0": 0.001}},
        "artifacts": {"report_path": "/tmp/a.json", "run_dir": "/tmp/detect/autotune_1"},
    }

    html = _render_history_page([tuning, manual])

    assert "普通训练" in html
    assert "自动调优" in html
    assert "mAP50" in html
    assert "分析失败" in html
    assert "学习率偏高" in html
    # Four KPIs render with values
    assert "0.4000" in html  # manual mAP50
    assert "0.3000" in html  # manual precision
    assert "0.5000" in html  # manual recall
    # Param keys AND values render in details
    assert "yolov8n.pt" in html
    assert "batch" in html
    assert "16" in html
    # best epoch shown
    assert "best 2" in html
    # Export link points to unified history endpoint
    assert "/api/experiments/history" in html


def test_api_experiments_history_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    records = [{"run_id": "manual:train39", "run_name": "train39", "source": "manual"}]
    monkeypatch.setattr(app_mod, "get_experiment_history", lambda *a, **k: records)

    client = TestClient(app_mod.app)
    resp = client.get("/api/experiments/history")

    assert resp.status_code == 200
    assert resp.json() == records


def _tuning_record(**overrides):
    record = {
        "run_id": "tuning:s1:autotune_1",
        "run_name": "autotune_1",
        "source": "tuning",
        "status": "completed",
        "analysis_status": "completed",
        "metrics": {"mAP50": 0.4},
        "finished_at": "2026-08-01T00:00:00Z",
        "params": {"model": "yolov8n.pt", "data": "/data/ds"},
        "tuning": {
            "decision": {
                "diagnosis": "学习率偏高，建议调低",
                "action": "adjust",
                "hyperparameter_changes": {"lr0": 0.001},
                "training_overrides": {"workers": 2},
            },
            "guardrails": {
                "valid": True,
                "warnings": ["lr0 从 0.005 约束到 0.001"],
                "errors": [],
                "clamped": {"lr0": 0.001},
            },
        },
        "audit_filename": "tuning_audit_s1.json",
        "audit_path": "e:/proj/log/tuning_audit_s1.json",
        "artifacts": {"report_path": "/tmp/a.json", "run_dir": "/tmp/detect/autotune_1"},
    }
    record.update(overrides)
    return record




def test_history_renders_new_tuning_schema_details():
    html = _render_history_page([_tuning_record()])

    # Structured decision fields render
    assert "学习率偏高，建议调低" in html
    assert "<td>lr0</td>" in html
    assert "<td>0.001</td>" in html
    # Guardrail outcomes render (Passed badge + warning)
    assert "通过" in html
    assert "lr0 从 0.005 约束到 0.001" in html
    # Audit entry link renders from audit_filename
    assert "/api/audit/tuning_audit_s1.json" in html
    assert "查看审计记录" in html


def test_history_keep_params_shows_keep_original():
    record = _tuning_record(
        tuning={
            "decision": {
                "diagnosis": "保持原参数",
                "action": "keep_params",
                "hyperparameter_changes": {},
                "training_overrides": {},
            },
            "guardrails": {"valid": True, "warnings": [], "errors": [], "clamped": {}},
        }
    )

    html = _render_history_page([record])

    assert "保持原参数" in html
    assert "保持原参数训练" in html


def test_history_manual_record_hides_tuning_sections():
    manual = {
        "run_id": "manual:train39",
        "run_name": "train39",
        "source": "manual",
        "status": "completed",
        "analysis_status": "completed",
        "metrics": {"mAP50": 0.4},
        "finished_at": "2026-08-01T00:00:00Z",
        "params": {"model": "yolov8n.pt"},
        "artifacts": {"report_path": "/tmp/r.json", "run_dir": "/tmp/detect/train39"},
    }

    html = _render_history_page([manual])

    assert "AI 诊断" not in html
    assert "保持原参数训练" not in html
    assert "查看审计记录" not in html
    assert "/api/audit/" not in html


def test_history_detail_colspan_matches_header_columns():
    html = _render_history_page([_tuning_record(), {
        "run_id": "manual:train39",
        "run_name": "train39",
        "source": "manual",
        "status": "completed",
        "analysis_status": "completed",
        "metrics": {"mAP50": 0.4},
        "finished_at": "2026-08-01T00:00:00Z",
        "params": {"model": "yolov8n.pt"},
    }])
    table = html.split('id="historySourceFilter"', 1)[1]

    # Header row (first <tr> after the filter) has exactly the expected columns
    header_row = table.split("<tr>", 1)[1].split("</tr>", 1)[0]
    header_cols = header_row.count("<th")
    assert header_cols == 12

    # Every history-details detail row spans the header column count
    detail_rows = table.count('class="history-details"')
    assert detail_rows == 2
    assert table.count('colspan="12"') == detail_rows


def test_api_audit_route_returns_record_and_blocks_traversal(tmp_path, monkeypatch):
    import os as real_os
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod
    from auto_tune.modules.agent_engine.audit import atomic_write_json

    log_dir = tmp_path / "log"
    log_dir.mkdir()
    atomic_write_json(
        str(log_dir / "tuning_audit_s1.json"),
        {"schema_version": "1.0", "terminal_status": "completed"},
    )

    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)

    client = TestClient(app_mod.app)

    ok = client.get("/api/audit/tuning_audit_s1.json")
    assert ok.status_code == 200
    assert ok.json()["terminal_status"] == "completed"

    prefix_blocked = client.get("/api/audit/other.json")
    assert prefix_blocked.status_code == 400

    missing = client.get("/api/audit/tuning_audit_missing.json")
    assert missing.status_code == 404

    # URL normalization collapses ../ before routing; either way the request is blocked
    traversal = client.get("/api/audit/%2E%2E%2Fsecret.json")
    assert traversal.status_code in (400, 404)

    # Defense in depth: the route itself rejects a decoded traversal path
    import asyncio
    blocked = asyncio.run(app_mod.api_audit_record("../secret.json"))
    assert blocked.status_code == 400
