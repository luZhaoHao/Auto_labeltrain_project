"""Tests for the unified training-completion contract consumed by the UI."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from auto_tune.modules.train_analyzer.experiment_history import ExperimentHistoryStore
from auto_tune.modules.run_state.service import new_run_state, update_run_state
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
    from auto_tune.modules.presentation import build_experiment_labels
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        experiment_labels=build_experiment_labels(translator),
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
    assert "失败" in html  # tuning analysis_status=failed via the shared enum
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
    # Audit entry renders as a run_id-bound modal button (Bugfix P5), not a
    # raw-JSON link.
    assert 'data-open-audit' in html
    assert "查看审计结果" in html
    assert 'data-run-id="tuning:s1:autotune_1"' in html


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

    # Manual records never render an audit action in the server-rendered rows.
    assert 'href="/api/audit/' not in html
    # The JS-driven history renderer only emits the tuning sections (AI
    # diagnosis / guardrails / audit) for tuning-source records.
    assert "exp.source === 'tuning'" in html


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
    assert header_cols == 13

    # Every history-details detail row spans the header column count
    detail_rows = table.count('class="history-details"')
    assert detail_rows == 2
    assert table.count('colspan="13"') == detail_rows


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


# ── S1.1 structured training-log SSE (Task 2) ──


def test_process_training_output_line_persists_and_builds_payload(tmp_path):
    from auto_tune.ui.app import _process_training_output_line

    log_path = tmp_path / "training.log"
    payload = _process_training_output_line(
        "  1/100  1.20G  1.234  0.456  0.789", "train87", log_path, set()
    )

    assert payload["event"] == "training_log"
    assert payload["log_kind"] == "epoch"
    assert payload["message"].startswith("Epoch 1/100:")
    assert payload["detail"] == "  1/100  1.20G  1.234  0.456  0.789"
    assert log_path.read_text("utf-8") == "  1/100  1.20G  1.234  0.456  0.789\n"


def test_process_training_output_line_detail_writes_log_only(tmp_path):
    from auto_tune.ui.app import _process_training_output_line

    log_path = tmp_path / "training.log"
    payload = _process_training_output_line(
        "1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]", "train87", log_path, set()
    )

    assert payload["log_kind"] == "detail"
    assert payload["message"] is None
    assert "5/10" in payload["detail"]
    assert log_path.read_text("utf-8") == "1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]\n"


def test_process_training_output_line_persistence_warn_once(tmp_path, monkeypatch):
    from auto_tune.ui import app as app_mod

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(app_mod, "append_training_log", boom)
    warned = set()
    log_path = tmp_path / "training.log"

    first = app_mod._process_training_output_line("epoch line", "train87", log_path, warned)
    second = app_mod._process_training_output_line("another line", "train87", log_path, warned)

    assert first["event"] == "log_persistence_error"
    assert first["level"] == "warning"
    assert "训练继续" in first["message"]
    assert second is None


def test_training_sse_batching_bounds_chunk_size():
    from auto_tune.ui.app import _SseBatch

    # Simulate the streaming loop: add one payload then flush when due.
    batcher = _SseBatch(max_batch=20, flush_interval=999)
    chunk_sizes = []
    for i in range(50):
        batcher.add({
            "status": "running",
            "event": "training_log",
            "log_kind": "detail",
            "message": None,
            "detail": str(i),
        })
        if batcher.should_flush():
            chunk_sizes.append(batcher.take().count("data: "))

    assert chunk_sizes == [20, 20]
    assert batcher.pending == 10
    chunk_sizes.append(batcher.take().count("data: "))
    assert chunk_sizes == [20, 20, 10]
    assert sum(chunk_sizes) == 50
    assert all(size <= 20 for size in chunk_sizes)


def test_training_sse_batch_flushes_by_time_window():
    from auto_tune.ui.app import _SseBatch

    batcher = _SseBatch(max_batch=1000, flush_interval=1.0)
    batcher.add({"status": "running", "message": "m"})
    batcher._last_flush -= 2.0  # simulate elapsed > interval
    assert batcher.should_flush() is True
    assert batcher.take().count("data: ") == 1
    assert batcher.pending == 0


# ── S1.1 frontend log-layering contract (Task 3) ──


def test_s11_template_log_layering_contract():
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    training = {
        "summary": {
            "total_runs_analyzed": 0,
            "best_mAP50": None,
            "best_overall_run": None,
            "average_mAP50": None,
            "runs_with_issues": 0,
        },
        "runs": {},
        "suggestion": None,
    }
    html = _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="training_monitor",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training=training,
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )

    assert 'id="monitorLog"' in html
    assert 'id="monitorFullLog"' in html
    assert 'id="toggleMonitorFullLog"' in html
    assert 'id="tuningLog"' in html
    assert 'id="tuningFullLog"' in html
    assert 'function consumeSseChunk' in html
    assert 'function appendBounded' in html
    assert 'function appendTrainingLogLine' in html
    assert 'var MAX_DEFAULT_LOG_LINES = 500' in html
    assert 'var MAX_FULL_LOG_LINES = 2000' in html
    assert 'DocumentFragment' in html
    assert 'textContent' in html
    # S1.1 constraint 1/2: training-log rendering must not use innerHTML
    assert "monitorLog.innerHTML" not in html
    assert "tuningLog.innerHTML" not in html


def test_s11_log_layering_i18n_keys_present():
    from auto_tune.ui.i18n import TRANSLATIONS

    zh = TRANSLATIONS["zh"]
    en = TRANSLATIONS["en"]
    for key in ("完整日志", "展开完整日志", "收起完整日志", "日志保存失败"):
        assert key in zh
        assert zh[key]
    assert en["完整日志"] == "Full Log"
    assert en["日志保存失败"] == "Log save failed"


# ── Ordinary-training running status: terminal states must be reported ──


def _training_running_log_dir(monkeypatch, tmp_path):
    """Redirect os.path.join("log", ...) to a temp dir, returning that dir."""
    import os as real_os

    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    real_join = real_os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "log":
            return str(log_dir / parts[1])
        return real_join(*parts)

    monkeypatch.setattr(real_os.path, "join", fake_join)
    return log_dir


class _FakeProc:
    def __init__(self, ret):
        self._ret = ret

    def poll(self):
        return self._ret


@pytest.mark.parametrize("expected", ["completed", "failed"])
def test_training_running_terminal_from_finished_proc(tmp_path, monkeypatch, expected):
    """A terminal persisted state is reported and never reported as running."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _training_running_log_dir(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status=expected, phase="terminal")

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()

        assert data["running"] is False
        assert data["status"] == expected
        assert data["run_id"] == state.run_id
    finally:
        app_mod._running_training.clear()


def test_training_running_aborted_reported(tmp_path, monkeypatch):
    """The cancelled terminal state is reported as cancelled."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _training_running_log_dir(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    state = new_run_state("manual", run_name="train1")
    update_run_state(state_file, state, status="cancelled", phase="terminal")

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()
        assert data["running"] is False
        assert data["status"] == "cancelled"
        assert data["phase"] == "terminal"
    finally:
        app_mod._running_training.clear()


def test_training_running_idle_when_nothing_pending(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _training_running_log_dir(monkeypatch, tmp_path)
    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()
        assert data["running"] is False
        assert data["status"] == "unknown"
        assert data["run_id"] is None
    finally:
        app_mod._running_training.clear()


def test_training_running_true_while_proc_alive(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod
    from auto_tune.modules.run_state.events import EventBroker
    import auto_tune.modules.run_state.manual_controller as mc

    log_dir = _training_running_log_dir(monkeypatch, tmp_path)
    state_file = log_dir / "training_running.json"
    run_state = new_run_state("manual", run_name="train9")
    run_state = update_run_state(
        state_file, run_state, status="running", phase="training", pid=1234,
    )
    broker = EventBroker(run_state.run_id)
    controller = mc.ManualRunController(
        run_state=run_state, state_file=str(state_file), cmd=["yolo"],
        params={}, train_name="train9", train_dir=str(tmp_path),
        data_yaml=str(tmp_path / "data.yaml"), model="yolov8n.pt", epochs=1,
        log_path=str(tmp_path / "training.log"), finalize_cb=None,
        broker=broker, manager=app_mod._RUN_MANAGER,
    )
    app_mod._RUN_MANAGER.register(controller)
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()
        assert data["running"] is True
        assert data["status"] == "running"
        assert data["run_id"] == run_state.run_id
    finally:
        app_mod._RUN_MANAGER.unregister(controller.run_id)


def test_legacy_running_file_not_reported_as_running(tmp_path, monkeypatch):
    """A legacy 'running' file without verifiable identity must downgrade to
    unknown — it can never be reported as still running."""
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _training_running_log_dir(monkeypatch, tmp_path)
    (log_dir / "training_running.json").write_text(
        json.dumps({"train_name": "trainX", "status": "running"}), encoding="utf-8")
    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        data = client.get("/api/training/running").json()
        assert data["running"] is False
        assert data["status"] == "unknown"
        assert data["terminal_reason"] == "legacy_identity_unverifiable"
        # The legacy file itself is never rewritten.
        assert json.loads((log_dir / "training_running.json").read_text("utf-8")) == {
            "train_name": "trainX", "status": "running"}
    finally:
        app_mod._running_training.clear()


def test_training_start_failed_writes_failed_terminal(tmp_path, monkeypatch):
    """Simulate ordinary training exiting non-zero: the terminal state is
    persisted as failed/terminal (never left 'running', never deleted)."""
    import asyncio
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    log_dir = _training_running_log_dir(monkeypatch, tmp_path)

    class FakeStdout:
        async def readline(self):
            return b""

    class FakeProc:
        returncode = 1
        pid = 99999
        stdout = FakeStdout()

        async def wait(self):
            return 1

    async def fake_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.find_detect_dir",
        lambda: str(tmp_path / "detect"),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo",
    )

    def fake_finalize(run_dir, run_name, source, config, log_dir, training_status,
                      started_at=None, finished_at=None, training_error=None, **kw):
        return {
            "run_id": f"manual:{run_name}", "run_name": run_name, "source": "manual",
            "status": "failed", "analysis_status": "skipped", "metrics": {},
            "artifacts": {"report_path": None}, "error": training_error,
            "analysis_error": None, "history_error": None,
        }

    monkeypatch.setattr("auto_tune.ui.app.finalize_training_run", fake_finalize)

    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/api/training/start", json={
            "data_yaml": str(tmp_path / "data.yaml"), "model": "yolov8n.pt", "epochs": 1,
        })
        assert resp.status_code == 200
        # The terminal state is persisted as failed/terminal (never deleted).
        persisted = json.loads((log_dir / "training_running.json").read_text("utf-8"))
        assert persisted["status"] == "failed"
        assert persisted["phase"] == "terminal"
        api = client.get("/api/training/running").json()
        assert api["status"] == "failed"
        assert api["running"] is False
    finally:
        app_mod._running_training.clear()


def test_training_monitor_stop_button_hidden_after_terminal_state():
    """On page refresh the stop button must not appear for a terminal state:
    it starts hidden and the page-load handler only shows it when running."""
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    html = _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="training_monitor",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training={
            "summary": {"total_runs_analyzed": 0, "best_mAP50": None,
                        "best_overall_run": None, "average_mAP50": None, "runs_with_issues": 0},
            "runs": {},
            "suggestion": None,
        },
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )

    # Stop button is hidden by default...
    assert 'id="stopTrainingBtn"' in html
    assert 'style="display:none;" id="stopTrainingBtn"' in html
    # ...and the unified renderer only reveals it when state.running is true.
    assert "state.running === true" in html


# ── Studio S1.2: immutable dataset snapshot UI ──


def _render_dataset_page(latest_dataset):
    from auto_tune.ui.app import _jinja_env
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="dataset",
        experiment_history=[],
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
        latest_dataset=latest_dataset,
    )


def _latest_with_snapshot(valid=True, **overrides):
    data = {
        "dataset_path": "e:/data/source",
        "split": True,
        "snapshot_id": "a" * 64,
        "snapshot_path": "e:/log/dataset_snapshots/" + "a" * 64,
        "manifest_path": "e:/log/dataset_snapshots/" + "a" * 64 + "/manifest.json",
        "data_yaml_path": "e:/log/dataset_snapshots/" + "a" * 64 + "/data.yaml",
        "train_count": 80,
        "val_count": 20,
        "background_count": 3,
        "snapshot_valid": valid,
    }
    data.update(overrides)
    return data


def test_snapshot_ui_create_button_seed_and_hints():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    assert "创建训练快照" in html
    assert 'id="splitRatio"' in html
    assert 'id="splitSeed"' in html
    assert 'value="42"' in html
    assert "原始数据不会被移动或修改" in html
    assert "额外占用磁盘空间" in html
    # ratio input strictly avoids 0/1
    assert 'min="0.05"' in html
    assert 'max="0.5"' in html


def test_snapshot_ui_status_shows_valid_snapshot():
    html = _render_dataset_page(_latest_with_snapshot(valid=True))
    assert "快照有效" in html
    assert "80" in html
    assert "20" in html
    assert "背景" in html


def test_snapshot_ui_corrupted_snapshot_not_ready():
    html = _render_dataset_page(_latest_with_snapshot(valid=False))
    assert "快照已损坏" in html
    assert "数据集已就绪" not in html
    assert "无法用于训练" in html


def test_snapshot_ui_old_latest_not_ready():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    assert "数据集已就绪" not in html
    assert "未创建快照" in html


def test_snapshot_ui_training_modal_uses_snapshot_yaml_when_valid():
    html = _render_dataset_page(_latest_with_snapshot(valid=True))
    assert 'value="e:/log/dataset_snapshots/' + "a" * 64 + '/data.yaml"' in html


def test_snapshot_ui_corrupted_modal_not_autofilled():
    html = _render_dataset_page(_latest_with_snapshot(valid=False))
    assert 'value="e:/log/dataset_snapshots/' + "a" * 64 + '/data.yaml"' not in html


def test_snapshot_ui_js_sends_ratio_and_seed_and_disables_button():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    assert "val_ratio: ratioNum" in html
    assert "seed: seedNum" in html
    assert "btn.disabled = true" in html
    assert "btn.disabled = false" in html
    assert "正在校验并复制" in html


def test_snapshot_ui_error_text_uses_textcontent():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    # splitDataset must render server errors through textContent (safe text), not innerHTML.
    assert "statusEl.textContent = '快照创建失败: '" in html


# ── Studio S1.4: directory-input safety feedback in the UI ──


def test_s14_folder_analyze_errors_restore_buttons_and_use_textcontent():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    # Both dataset and training folder analysis restore the button on success and
    # failure, and render server errors through textContent (never innerHTML).
    assert "btn.disabled = false" in html
    assert "statusEl.textContent = '错误: ' + (result.error || 'Unknown')" in html
    assert "if (result.error_code)" in html
    assert "if (resp.error_code)" in html
    assert "statusEl.textContent = '{{ _(\"Error\") }}: ' + err.message" not in html
    assert "statusEl.innerHTML" not in html


def test_s14_browse_error_surfaces_stable_error_not_empty_dir():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    # browse-folder must surface stable server errors (data.error + error_code)
    # instead of silently rendering an empty directory listing.
    assert "if (data.error)" in html
    assert "error_code" in html
    assert ".textContent" in html


def test_s14_page_has_no_zip_upload_control():
    html = _render_dataset_page({"dataset_path": "e:/data/source", "split": False})
    # The S1.4 page must not reintroduce ZIP/JSON upload widgets.
    assert 'type="file"' not in html
    assert "dropZone" not in html
    assert "uploadDataset" not in html
    assert "uploadTraining" not in html
    assert "splitStatus.innerHTML" not in html


# ── P1b 返修：训练监控指标投影（首屏 + 终态事件）────────────────────
#
# 这里执行的是**渲染后的真实模板 + 真实前端脚本**（tests/js/minidom.js 在 Node 中
# 按页面顺序执行内联脚本与 /static/*.js），不是源码字符串断言，也不是 JS 逻辑的
# Python 改写。缺失的指标体系显示“—”，真实 0 显示 0/0.0000/0.000。

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_HARNESS = Path(__file__).resolve().parent / "js" / "minidom.js"
_NODE = shutil.which("node")

monitor_node = pytest.mark.skipif(_NODE is None, reason="node is not available")


def _render_monitor_page(training, experiment_history=()):
    from auto_tune.modules.presentation import build_experiment_labels
    from auto_tune.ui.app import _jinja_env, _monitor_latest_projection
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        experiment_labels=build_experiment_labels(translator),
        active_page="training_monitor",
        experiment_history=list(experiment_history),
        tuning_history=[],
        dataset=None,
        training=training,
        monitor_latest=_monitor_latest_projection(training, list(experiment_history)),
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )


def _run_monitor_scenario(scenario, training, tmp_path, history=()):
    """Execute one monitor scenario against the rendered page (real scripts)."""
    page = tmp_path / "rendered_monitor.html"
    page.write_text(_render_monitor_page(training, history), encoding="utf-8")
    result = subprocess.run(
        [_NODE, str(_HARNESS), scenario, str(_UI_DIR), str(page)],
        capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload.get("error")
    return payload


def _training_runs(**runs):
    """A training report whose runs are keyed by run_name (Module B shape)."""
    return {
        "summary": {"best_overall_run": list(runs)[0] if runs else None},
        "runs": {
            name: {
                "name": name,
                "args": {"epochs": spec.get("epochs")},
                "results": {"final_metrics": spec.get("final_metrics", {})},
            }
            for name, spec in runs.items()
        },
    }


def _history_record(run_name, status="completed"):
    return {
        "run_id": "manual:" + run_name,
        "run_name": run_name,
        "source": "manual",
        "status": status,
        "metrics": {},
        "artifacts": {},
    }


@monitor_node
def test_monitor_first_paint_shows_the_latest_valid_run(tmp_path):
    """刷新后四个卡片展示最近有效训练事实（按服务端既有顺序，不靠字典遍历）。"""
    # dict 插入顺序故意与“最近”相反：报告里 train2 在前，历史顺序里 train9 最新
    training = _training_runs(
        train2={"epochs": 30, "final_metrics": {"metrics/mAP50(B)": 0.31,
                                               "metrics/mAP50-95(B)": 0.11,
                                               "metrics/precision(B)": 0.4,
                                               "metrics/recall(B)": 0.5}},
        train9={"epochs": 7, "final_metrics": {"metrics/mAP50(B)": 0.00037,
                                               "metrics/mAP50-95(B)": 0.00007,
                                               "metrics/precision(B)": 0.00066,
                                               "metrics/recall(B)": 0.06452}},
    )
    history = [_history_record("train9"), _history_record("train2")]
    facts = _run_monitor_scenario("monitor_first_paint", training, tmp_path, history)

    assert facts["cards"]["epochs"] == "7"
    assert facts["cards"]["map50"] == "0.0004"
    assert facts["cards"]["map5095"] == "0.0001"
    assert facts["cards"]["pr"] == "0.001 / 0.065"


@monitor_node
def test_monitor_first_paint_is_honest_without_history(tmp_path):
    """空历史时四个卡片显示“—”，绝不猜测。"""
    facts = _run_monitor_scenario("monitor_first_paint", _training_runs(), tmp_path)
    cards = facts["cards"]
    assert cards["epochs"] == "—"
    assert cards["map50"] == "—"
    assert cards["map5095"] == "—"
    assert cards["pr"] == "—"


@monitor_node
def test_monitor_terminal_event_projects_metrics_to_the_cards(tmp_path):
    """重连流终态事件必须投影 epochs 与四项指标，且不受他 run/重复事件影响。"""
    facts = _run_monitor_scenario("monitor_terminal_metrics", _training_runs(), tmp_path)

    after = facts["afterTerminal"]
    assert after["epochs"] == "2"
    assert after["map50"] == "0.0004"
    assert after["map5095"] == "0.0001"
    assert after["pr"] == "0.001 / 0.065"
    assert facts["logText"]  # 日志仍然渲染，未被指标投影破坏


@monitor_node
def test_monitor_epochs_card_follows_the_finalized_result_contract(tmp_path):
    """终态 result.epochs 是结构化对象：优先 configured，逐级回退，缺失诚实显示“—”。"""
    facts = _run_monitor_scenario("monitor_epochs_contract", _training_runs(), tmp_path)

    # 结构化对象存在时以“已配置”为准，而不是 completed/best/params
    assert facts["configuredPreferred"] == "2"
    # configured 缺失 → completed；再缺 → params.epochs
    assert facts["completedFallback"] == "5"
    assert facts["paramsFallback"] == "9"
    # 真实 0 是有效数值，不得因真值判断显示“—”
    assert facts["realZero"] == "0"
    # 整个 epochs 结构缺失 → 诚实显示“—”，绝不补造 0
    assert facts["missingStructure"] == "—"
    # 旧标量事件仍兼容
    assert facts["legacyScalar"] == "3"
    # 候选值全部非法 → “—”
    assert facts["allIllegal"] == "—"


@monitor_node
def test_monitor_projection_handles_real_zeros_and_missing_values(tmp_path):
    """真实 0 显示数值，缺失项才显示“—”，并按位数格式化。"""
    facts = _run_monitor_scenario("monitor_result_projection", _training_runs(), tmp_path)

    assert facts["zeros"]["epochs"] == "0"
    assert facts["zeros"]["map50"] == "0.0000"
    assert facts["zeros"]["map5095"] == "0.0000"
    assert facts["zeros"]["pr"] == "0.000 / 0.000"
    assert facts["zeros"]["reportVisible"] is True

    assert facts["partial"]["epochs"] == "3"
    assert facts["partial"]["map50"] == "0.5000"
    assert facts["partial"]["map5095"] == "—"
    assert facts["partial"]["pr"] == "—"
    assert facts["partial"]["reportVisible"] is False

    assert facts["rounded"]["map50"] == "0.1235"
    assert facts["rounded"]["map5095"] == "0.9877"
    assert facts["rounded"]["pr"] == "0.123 / 0.988"


@monitor_node
def test_monitor_original_stream_and_reconnect_stream_agree(tmp_path):
    """普通训练原始流与重连流必须产生一致的 DOM 结果。"""
    facts = _run_monitor_scenario("monitor_stream_parity", _training_runs(), tmp_path)
    assert facts["reconnect"] == facts["legacy"]
    assert facts["reconnect"]["epochs"] == "2"
    assert facts["reconnect"]["map50"] == "0.1234"
    assert facts["reconnect"]["pr"] == "0.891 / 0.234"


# ── 第四轮：HPO 正式训练复用共用监控屏幕（同一组唯一 ID） ────────────


def _render_intelligent_page(training, experiment_history=()):
    """Rendered Intelligent-Analysis page (page 2 active) with the monitor projection."""
    from auto_tune.modules.presentation import build_experiment_labels
    from auto_tune.ui.app import _jinja_env, _monitor_latest_projection
    from auto_tune.ui.i18n import make_translator

    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        experiment_labels=build_experiment_labels(translator),
        active_page="agent_suggestion",
        experiment_history=list(experiment_history),
        tuning_history=[],
        dataset=None,
        training=training,
        monitor_latest=_monitor_latest_projection(training, list(experiment_history)),
        project={},
        latest_suggestion=None,
        current_args=None,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=None,
        vision_analysis=None,
        latest_dataset=None,
    )


def _run_intelligent_scenario(scenario, training, tmp_path, history=()):
    page = tmp_path / "rendered_intelligent.html"
    page.write_text(_render_intelligent_page(training, history), encoding="utf-8")
    result = subprocess.run(
        [_NODE, str(_HARNESS), scenario, str(_UI_DIR), str(page)],
        capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload.get("error")
    return payload


@monitor_node
def test_hpo_formal_training_reuses_the_shared_monitor_screen(tmp_path):
    """202 后自动订阅同一 runtime run；共用监控节点在 HPO 模式下就位于正式训练旁。"""
    facts = _run_intelligent_scenario("formal_monitor_reuse",
                                      _training_runs(), tmp_path)

    # 监控块被移动到正式训练监控宿主，且指标 ID 仍然唯一
    assert facts["hostAfter"] == "hpoFormalMonitorHost"
    assert facts["monitorInsideFormalHost"] is True
    assert facts["monitorIdCounts"] == [1, 1, 1, 1, 1]
    # 只按 202 返回的完整 runtime 身份订阅，绝不使用 JSON 历史 ID
    assert facts["subscribed"] is True
    # 终态事件把同一份投影写进共用卡片
    cards = facts["cards"]
    assert cards["epochs"] == "2"
    assert cards["map50"] == "0.0004"
    assert cards["map5095"] == "0.0001"
    assert cards["pr"] == "0.001 / 0.065"
    assert "已接受" in facts["formalStatus"]


@monitor_node
def test_hpo_formal_monitor_belongs_to_the_current_study(tmp_path):
    """进入 HPO / 切换研究必须收敛监控所有权，刷新恢复只认权威 runtime 投影。"""
    training = _training_runs(train9={"epochs": 7, "final_metrics": {
        "metrics/mAP50(B)": 0.31, "metrics/mAP50-95(B)": 0.11,
        "metrics/precision(B)": 0.4, "metrics/recall(B)": 0.5}})
    history = [_history_record("train9")]
    facts = _run_intelligent_scenario("hpo_monitor_ownership", training, tmp_path,
                                      history)

    # 首屏：全局最近有效训练（另一条 run）已投影到公共监控
    assert facts["firstPaint"]["cards"]["epochs"] == "7"
    # 进入 HPO：立即收敛，绝不保留全局最近运行
    assert facts["hpoEntry"]["cards"]["epochs"] == "—"
    assert facts["hpoEntry"]["cards"]["map50"] == "—"
    assert "train9" not in (facts["hpoEntry"]["log"] or "")
    # study B 没有关联正式训练：保持空状态
    assert facts["studyB"]["cards"]["map50"] == "—"
    assert "train5" not in (facts["studyB"]["log"] or "")

    # study C：只按权威投影里的合法 runtime UUID 订阅监控
    assert facts["subscribed"] is True
    # 刷新恢复：该 run 的 epochs 与四项指标来自其自身权威事实，终态事件不伪造
    cards = facts["studyC"]["cards"]
    assert cards["epochs"] == "2"
    assert cards["map50"] == "0.7100"
    assert cards["map5095"] == "0.4400"
    assert cards["pr"] == "0.620 / 0.550"
    assert "epoch 2/2" in facts["studyC"]["log"]
    # 其它 run 的终态事件不得污染当前卡片
    assert "train9" not in facts["studyC"]["log"]

    # 从 C 切回 B：立即清空并收敛，C 的订阅被断开
    assert facts["backToB"]["cards"]["map50"] == "—"
    assert facts["backToB"]["pendingRunStreams"] == 0
    assert facts["monitorRunId"] is None

    # 离开 HPO：其它三模式恢复进入前的公共监控投影
    assert facts["leftHpo"]["cards"]["epochs"] == "7"
    assert facts["leftHpo"]["log"] == facts["firstPaint"]["log"]
    # 仍然只有一组监控身份
    assert facts["monitorIdCounts"] == [1, 1, 1, 1, 1]


@monitor_node
def test_hpo_monitor_keeps_ordinary_training_layout_intact(tmp_path):
    """非 HPO 模式下共用监控仍留在训练监控页宿主里（其它三模式不回归）。"""
    page = tmp_path / "rendered_monitor.html"
    page.write_text(_render_monitor_page(_training_runs()), encoding="utf-8")
    result = subprocess.run(
        [_NODE, str(_HARNESS), "monitor_first_paint", str(_UI_DIR), str(page)],
        capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert "error" not in payload, payload.get("error")
    # 首屏投影不受监控块搬家影响
    assert payload["cards"]["epochs"] == "—"


# ── 第二轮返修 Task 1/2：公共监控的完整状态所有权 ─────────────────────


@monitor_node
def test_hpo_takeover_clears_every_monitor_field_of_the_previous_run(tmp_path):
    """进入 HPO 且未绑定正式运行时，指标/徽章/提示/按钮/链接/两套日志全部收敛。"""
    facts = _run_intelligent_scenario("hpo_monitor_full_state",
                                      _training_runs(), tmp_path)
    before = facts["beforeHpo"]
    # 进入前：运行中的普通训练 A 的完整投影
    assert before["badge"] == {"text": "运行中", "cls": "badge badge-yellow"}
    assert before["stopDisplay"] == "inline-flex"
    assert before["cards"]["epochs"] == "7"
    assert before["cards"]["reportVisible"] is True
    assert "epoch 1/2" in before["log"]
    assert "A full detail line" in before["fullLog"]

    for name in ("hpoEntry", "emptyStudy"):
        state = facts[name]
        # 四项指标显示“—”
        assert state["cards"]["epochs"] == "—", name
        assert state["cards"]["map50"] == "—", name
        assert state["cards"]["map5095"] == "—", name
        assert state["cards"]["pr"] == "—", name
        # 徽章不得显示上一运行的状态，而是明确的“未选择”
        assert state["badge"]["text"] == "未选择", name
        assert "badge-green" not in state["badge"]["cls"], name
        assert "badge-yellow" not in state["badge"]["cls"], name
        assert "badge-red" not in state["badge"]["cls"], name
        # 提示清空隐藏、停止按钮隐藏、报告链接隐藏
        assert state["note"] == {"text": "", "display": "none"}, name
        assert state["stopDisplay"] == "none", name
        assert state["cards"]["reportVisible"] is False, name
        # 默认日志只有未选择提示，完整日志被清空
        assert "尚未选择正式训练" in state["log"], name
        assert "epoch 1/2" not in state["log"], name
        assert state["fullLog"] == "", name
        assert "A full detail line" not in state["fullLog"], name

    # 上一订阅真正断开：没有遗留的 run 流在途
    assert facts["emptyStudy"]["pendingRunStreams"] == 0
    # DOM 中仍只有一组公共监控身份
    assert facts["monitorIdCounts"] == [1, 1, 1, 1, 1, 1]


@monitor_node
def test_hpo_formal_run_owns_status_log_and_metrics(tmp_path):
    """绑定正式运行 B1 后，状态/提示/停止按钮/日志都只属于 B1。"""
    facts = _run_intelligent_scenario("hpo_monitor_full_state",
                                      _training_runs(), tmp_path)
    bound = facts["b1Bound"]
    # 徽章来自该 formal run 的权威状态（运行中），不是上一运行也不是未选择
    assert bound["badge"] == {"text": "运行中", "cls": "badge badge-yellow"}
    assert bound["stopDisplay"] == "inline-flex"
    # 绑定即切换监控目标：上一运行的日志不得残留
    assert "epoch 1/2" not in bound["log"]
    assert "A full detail line" not in bound["fullLog"]
    # B1 自己的日志只来自 B1 的流
    assert "B1 epoch 1/2" in facts["b1Log"]
    assert "epoch 1/2" not in facts["b1Log"].replace("B1 epoch 1/2", "")

    # B1 的终态：徽章显示 B1 自己的“失败”，绝不被全局最近运行 A（运行中）覆盖
    terminal = facts["b1Terminal"]
    assert terminal["badge"] == {"text": "失败", "cls": "badge badge-red"}
    assert terminal["stopDisplay"] == "none"
    assert facts["b1Terminal"]["log"] == facts["b1Log"]


@monitor_node
def test_late_terminal_event_of_a_superseded_subscription_changes_nothing(tmp_path):
    """切换研究后，上一订阅的迟到终态无权更新指标、状态或日志。"""
    facts = _run_intelligent_scenario("hpo_monitor_full_state",
                                      _training_runs(), tmp_path)
    switched = facts["switched"]
    # 切换研究后是“未选择”的空状态
    assert switched["badge"]["text"] == "未选择"
    for state in (facts["afterLateB1"], facts["afterStaleEnd"]):
        assert state["badge"] == switched["badge"]
        assert state["cards"] == switched["cards"]
        assert state["note"] == switched["note"]
        assert state["stopDisplay"] == switched["stopDisplay"]
        assert state["log"] == switched["log"]
        assert state["fullLog"] == switched["fullLog"]
        assert "B1 迟到终态" not in state["log"]
        assert state["cards"]["map50"] == "—"


@monitor_node
def test_leaving_hpo_restores_the_full_monitor_state(tmp_path):
    """离开 HPO 完整恢复进入前的 A：徽章 class、提示、按钮、链接与两套日志。"""
    facts = _run_intelligent_scenario("hpo_monitor_full_state",
                                      _training_runs(), tmp_path)
    before, after = facts["beforeHpo"], facts["leftHpo"]
    assert after["badge"] == before["badge"]
    assert after["note"] == before["note"]
    assert after["stopDisplay"] == before["stopDisplay"]
    assert after["cards"] == before["cards"]
    assert after["log"] == before["log"]
    assert after["fullLog"] == before["fullLog"]
    # 恢复的是 A 的事实，不是 HPO 空状态
    assert after["cards"]["epochs"] == "7"
    assert after["badge"]["text"] == "运行中"
    assert "epoch 1/2" in after["log"]


# ── 第三轮 P1：同一 runtime_run_id 的状态与结果必须每次收敛 ─────────────

# 与 minidom.js 的 hpo_formal_monitor_* 场景保持一致的三条运行身份
_FORMAL_RUN_A = "manual:aaaaaaaa-1111-4111-8111-111111111111"
_FORMAL_RUN_B = "manual:bbbbbbbb-2222-4222-8222-222222222222"
_FORMAL_RUN_C = "manual:cccccccc-3333-4333-8333-333333333333"


@monitor_node
def test_accepted_formal_submission_converges_the_monitor_to_the_new_run(tmp_path):
    """202 之后（首个流事件之前）监控必须收敛为该 run 的已接受/运行中事实。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_convergence",
                                      _training_runs(), tmp_path)
    before, accepted = facts["beforeSubmit"], facts["accepted"]
    # 提交前：上一正式运行 A1 的完整终态显示在公共监控上
    assert before["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    assert before["cards"]["map50"] == "0.7000"
    assert before["cards"]["epochs"] == "2"
    assert "A1 epoch 1/2" in before["log"]

    # 202 之后：身份已是新 run 的完整 UUID，徽章不再是上一 run 的“已完成”
    assert accepted["monitorRunId"] == _FORMAL_RUN_B
    assert accepted["badge"]["text"] == "运行中"
    assert "badge-green" not in accepted["badge"]["cls"]
    # 停止按钮可见（该 run 处于活跃阶段）
    assert accepted["stopDisplay"] == "inline-flex"
    # 上一 run 的指标、报告链接与两套日志全部清除
    assert accepted["cards"]["epochs"] == "—"
    assert accepted["cards"]["map50"] == "—"
    assert accepted["cards"]["map5095"] == "—"
    assert accepted["cards"]["pr"] == "—"
    assert accepted["cards"]["reportVisible"] is False
    assert "A1 epoch 1/2" not in accepted["log"]
    assert "A1 epoch 1/2" not in accepted["fullLog"]
    # 新的 run 只被订阅一次
    assert accepted["streamsB"] == 1
    assert accepted["streamsA"] == 1


@monitor_node
def test_same_runtime_run_refreshes_status_and_result_without_resubscribing(tmp_path):
    """同一 runtime_run_id 由 running → completed：刷新权威状态与结果，不重订阅。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_convergence",
                                      _training_runs(), tmp_path)
    assert "B epoch 1/5" in facts["bLogBeforePoll"]

    running = facts["bRunning"]
    assert running["badge"] == {"text": "运行中", "cls": "badge badge-yellow"}
    assert running["stopDisplay"] == "inline-flex"
    # 同一身份的刷新不清空该 run 已有的日志，也不重新订阅
    assert "B epoch 1/5" in running["log"]
    assert running["streamsB"] == 1

    completed = facts["bCompleted"]
    assert completed["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    assert completed["stopDisplay"] == "none"
    # 最终指标来自该 run 的权威结果投影
    assert completed["cards"]["epochs"] == "5"
    assert completed["cards"]["map50"] == "0.6600"
    assert completed["cards"]["map5095"] == "0.3300"
    assert completed["cards"]["pr"] == "0.510 / 0.420"
    assert "B epoch 1/5" in completed["log"]
    assert completed["streamsB"] == 1


@monitor_node
def test_repeated_identical_polling_is_idempotent(tmp_path):
    """完全重复的 /formal-runs 轮询不得重复订阅、重置日志或制造重复投影。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_convergence",
                                      _training_runs(), tmp_path)
    completed, repeated = facts["bCompleted"], facts["repeated"]
    # 重复轮询后的事实仍是该 run 自己的权威终态与结果
    assert repeated["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    assert repeated["cards"]["epochs"] == "5"
    assert repeated["cards"]["map50"] == "0.6600"
    # 与第一次收敛后的状态完全一致：没有重复订阅、没有重置日志
    assert repeated["badge"] == completed["badge"]
    assert repeated["cards"] == completed["cards"]
    assert repeated["log"] == completed["log"] == "B epoch 1/5"
    assert repeated["fullLog"] == completed["fullLog"]
    assert repeated["stopDisplay"] == completed["stopDisplay"]
    assert repeated["streamsB"] == 1


@monitor_node
def test_other_run_events_cannot_change_the_current_formal_monitor(tmp_path):
    """上一订阅的迟到事件不得更新任何监控字段。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_convergence",
                                      _training_runs(), tmp_path)
    for key in ("afterLateA",):
        state = facts[key]
        assert state["badge"] == facts["bCompleted"]["badge"], key
        assert state["cards"] == facts["bCompleted"]["cards"], key
        assert state["log"] == facts["bCompleted"]["log"], key
        assert "A1 迟到终态" not in state["log"], key
        assert state["cards"]["map50"] == "0.6600", key


@monitor_node
def test_switching_to_another_runtime_clears_and_resubscribes_once(tmp_path):
    """切换到另一 runtime：清空上一 run、使其迟到事件失效，且只订阅新 run。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_convergence",
                                      _training_runs(), tmp_path)
    switched, after_late = facts["switchedToC"], facts["afterLateB"]
    assert switched["monitorRunId"] == _FORMAL_RUN_C
    assert switched["badge"]["text"] == "运行中"
    # 上一正式 run 的事实被清空
    assert switched["cards"]["map50"] == "—"
    assert switched["cards"]["epochs"] == "—"
    assert "B epoch 1/5" not in switched["log"]
    # 新 runtime 只订阅一次，旧 runtime 不再新增订阅
    assert switched["streamsC"] == 1
    assert switched["streamsB"] == 1
    # B 的迟到终态不改变任何监控字段
    assert after_late["badge"] == switched["badge"]
    assert after_late["cards"] == switched["cards"]
    assert after_late["log"] == switched["log"]
    assert after_late["monitorRunId"] == switched["monitorRunId"]


@monitor_node
def test_running_to_failed_converges_without_resubscribing(tmp_path):
    """同一 runtime 由 running → failed：徽章收敛为失败并隐藏停止按钮。"""
    facts = _run_intelligent_scenario("hpo_formal_monitor_failure",
                                      _training_runs(), tmp_path)
    assert facts["running"]["badge"] == {"text": "运行中", "cls": "badge badge-yellow"}
    failed = facts["failed"]
    assert failed["badge"] == {"text": "失败", "cls": "badge badge-red"}
    assert failed["stopDisplay"] == "none"
    assert failed["streams"] == 1
    assert "B epoch 1/5" in failed["log"]


@monitor_node
def test_each_hpo_session_captures_a_fresh_baseline(tmp_path):
    """第二次进出 HPO 必须恢复“新的 B”，而不是第一次的旧快照 A。"""
    facts = _run_intelligent_scenario("hpo_monitor_full_state",
                                      _training_runs(), tmp_path)
    before2, after2 = facts["beforeHpo2"], facts["leftHpo2"]
    # 第二次进入前的事实是 B（与第一次的 A 完全不同）
    assert before2["cards"]["epochs"] == "30"
    assert before2["cards"]["map50"] == "0.5000"
    assert before2["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    assert "B epoch 1/30" in before2["log"]
    assert "epoch 1/2" not in before2["log"]
    # HPO 内连续切换两个研究后离开：恢复的仍是 B
    assert after2 == before2
    assert after2["cards"]["epochs"] == "30"
    assert after2["badge"]["text"] == "已完成"
    assert "B epoch 1/30" in after2["log"]
    assert "A full detail line" not in after2["fullLog"]


# ── 第四轮 P1：202 后必须自动启动轮询（不依赖用户手动刷新）─────────────
#
# 这些场景只推进 fake clock 并应答实现自己发出的请求；它们从不调用
# hpoRefreshFormalRuns()/hpoRefreshRound()，因此断言的是“自动轮询”本身，而不是
# 手动刷新后的投影结果。


@monitor_node
def test_accepted_formal_training_starts_the_only_poll_timer(tmp_path):
    """已完成研究本来没有定时器：202 之后必须自动建立唯一的 2 秒轮询。"""
    facts = _run_intelligent_scenario("hpo_formal_polling_after_accept",
                                      _training_runs(), tmp_path)
    before, accepted = facts["beforeSubmit"], facts["accepted"]
    # 提交前：已完成研究 + 已完成的正式运行 A1，没有任何轮询定时器
    assert before["timers"] == 0
    assert before["monitorRunId"] == _FORMAL_RUN_A
    assert before["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    # 202 之后（首个流事件之前）：自动产生唯一轮询，且只订阅该 run 一次
    assert accepted["timers"] == 1
    assert accepted["monitorRunId"] == _FORMAL_RUN_B
    assert accepted["badge"] == {"text": "运行中", "cls": "badge badge-yellow"}
    assert accepted["stopDisplay"] == "inline-flex"
    assert accepted["watchFormal"] is True
    assert facts["subscriptions"] == 1
    assert facts["streamOpen"] is True


@monitor_node
def test_the_poll_timer_converges_the_linked_list_then_stops(tmp_path):
    """只推进时钟即可自动收敛关联列表/结果按钮/最终模型，并在终态后自动停止。"""
    facts = _run_intelligent_scenario("hpo_formal_polling_after_accept",
                                      _training_runs(), tmp_path)
    # 终态轮询不清空该 run 已有的日志，也不重复订阅
    assert "B epoch 1/5" in facts["logBeforePoll"]

    converged = facts["converged"]
    assert converged["timers"] == 0          # watchFormal=false 后自动停止，不空转
    assert converged["badge"] == {"text": "已完成", "cls": "badge badge-green"}
    assert converged["stopDisplay"] == "none"
    assert "已完成" in converged["list"]
    assert "train2" in converged["list"]
    buttons = {b["label"]: b["disabled"] for b in converged["buttons"]}
    assert buttons["查看结果"] is False       # 权威实验身份可用 → 入口启用
    assert "下载最终 best.pt" in buttons      # best_pt_available=true → 出现

    # 公共监控保持终态与最终指标（来自该 run 的权威事实）
    assert converged["cards"]["epochs"] == "5"
    assert converged["cards"]["map50"] == "0.6600"
    assert converged["cards"]["map5095"] == "0.3300"
    assert converged["cards"]["pr"] == "0.510 / 0.420"
    assert "B epoch 1/5" in converged["log"]
    assert facts["subscriptions"] == 1

    # 终态后继续推进时钟：没有新请求，也没有新定时器
    assert facts["afterExtraTick"]["timers"] == 0
    assert facts["afterExtraTick"]["formal"] == facts["afterTerminal"]["formal"]
    assert facts["afterExtraTick"]["total"] == facts["afterTerminal"]["total"]


@monitor_node
def test_an_interval_tick_during_the_accepted_refresh_is_coalesced(tmp_path):
    """202 后的首次刷新在途时，tick 走既有合并机制：不并发，也不丢最后一次刷新。"""
    facts = _run_intelligent_scenario("hpo_formal_polling_coalesces_inflight_tick",
                                      _training_runs(), tmp_path)
    # 搜索仍在进行 → 提交前已有唯一轮询
    assert facts["beforeSubmit"]["timers"] == 1
    inflight, after_tick = facts["inflight"], facts["afterTick"]
    assert inflight["pending"] > 0
    # tick 落在在途刷新上：不得并发发出第二次请求
    assert after_tick["pending"] == inflight["pending"]
    assert after_tick["formal"] == inflight["formal"]

    # 被合并的那一次随后补跑（最后一次刷新不丢），且始终只有一个定时器
    assert facts["formalAfterSettle"] == inflight["formal"] + 2
    coalesced = facts["coalesced"]
    assert coalesced["timers"] == 1
    assert coalesced["monitorRunId"] == _FORMAL_RUN_B
    assert "运行中" in coalesced["list"]


@monitor_node
def test_switching_studies_stops_the_old_timer_and_drops_its_response(tmp_path):
    """训练期间切换研究：旧定时器与在途响应失效，新研究不继续旧轮询。"""
    facts = _run_intelligent_scenario("hpo_formal_polling_switch_study",
                                      _training_runs(), tmp_path)
    assert facts["training"]["timers"] == 1
    assert facts["training"]["monitorRunId"] == _FORMAL_RUN_B
    assert facts["inflightFormal"] is True

    # 切换后立即停止旧定时器、断开旧订阅并清空监控
    after_switch = facts["afterSwitch"]
    assert after_switch["timers"] == 0
    assert after_switch["monitorRunId"] is None
    assert after_switch["pendingRunStreams"] == 0
    assert after_switch["badge"]["text"] == "未选择"

    # 旧研究关联结果的迟到响应无权更新任何字段
    assert facts["afterLateA"] == after_switch
    switched = facts["switched"]
    assert switched["timers"] == 0
    assert switched["watchFormal"] is False
    assert switched["monitorRunId"] is None
    assert switched["list"] == "暂无由该研究发起的正式训练。"

    # 再推进时钟：不复活旧轮询，也不产生任何新请求
    assert facts["afterFinalTick"]["timers"] == 0
    assert facts["afterFinalTick"]["total"] == facts["beforeFinalTick"]
