"""S1.5 Task 5: UI template contract for the six run states and SSE identity.

Renders the real single_page.html and asserts the renderer, the refresh hook,
the stop-button rule, safe textContent rendering, and SSE run-id/seq filtering.
"""

from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import TRANSLATIONS, make_translator


def _render_monitor():
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
    return _jinja_env.get_template("single_page.html").render(
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


def _render_zh():
    translator = make_translator("zh")
    return _jinja_env.get_template("single_page.html").render(
        _=translator,
        current_lang="zh",
        active_page="dashboard",
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
        latest_dataset=None,
    )


# ── Step 1: six-state template contract ──


def test_s15_six_state_ui_contract():
    html = _render_monitor()
    # renderer + refresh hook present
    assert "function renderRunState" in html
    assert "function refreshRunState" in html
    assert "function _sseInCurrentRun" in html
    # stop button is shown only when running
    assert "state.running === true" in html
    # the six states are rendered via textContent
    assert "badge.textContent" in html
    assert "note.textContent" in html
    assert "状态无法确认" in html
    assert "运行控制已中断，无法确认或继续原进程" in html
    # refresh consults both unified status APIs and picks the newest
    assert "'/api/training/running'" in html
    assert "'/api/tuning/status'" in html
    assert "function _latestUpdatedAt" in html


def test_s15_running_state_wins_over_newer_terminal():
    """An active running state must win over a historical terminal state in the
    UI merge, even when the terminal record has a newer updated_at."""
    html = _render_monitor()
    # running checks come before updated_at comparison
    assert "a.running === true" in html
    assert "b.running === true" in html
    ta_idx = html.index("var ta = a && a.updated_at")
    run_a_idx = html.index("a.running === true")
    assert run_a_idx < ta_idx


def test_s15_state_i18n_keys_present():
    zh = TRANSLATIONS["zh"]
    en = TRANSLATIONS["en"]
    for key in ("Running", "Completed", "Failed", "Cancelled", "Interrupted", "Unknown"):
        assert key in zh
        assert zh[key]
    assert zh["Interrupted"] == "已中断"
    assert zh["Unknown"] == "未知"
    assert "运行控制已中断，无法确认或继续原进程" in zh
    assert "状态无法确认" in zh
    assert en["状态无法确认"] == "Status cannot be confirmed"
    assert "resumed" in en["运行控制已中断，无法确认或继续原进程"]


def test_s15_state_badge_mapping_covers_all_statuses():
    html = _render_monitor()
    for status in ("running", "completed", "failed", "cancelled", "interrupted", "unknown"):
        assert "'" + status + "'" in html


def test_s15_interrupted_never_claims_resumed():
    html = _render_monitor()
    # The interrupted copy must state the process cannot be confirmed/resumed.
    assert "无法确认或继续原进程" in html
    # No promise of resumption anywhere in the monitor script.
    assert "恢复成功" not in html
    assert "继续运行" not in html


def test_s15_status_messages_use_textcontent():
    html = _render_monitor()
    # All server status text flows through textContent; no innerHTML on the
    # tuning status element (XSS-safe per S1.3 boundary).
    assert "tuningStatus.textContent" in html
    assert "tuningStatus.innerHTML" not in html


# ── Step 3: stale SSE / duplicate event filtering ──


def test_s15_sse_run_id_filtering_contract():
    html = _render_monitor()
    # A different run_id (stale connection) must be dropped.
    assert "data.run_id !== _activeRunId" in html
    # Duplicate/out-of-order event_seq must be dropped.
    assert "data.event_seq <= _lastEventSeq" in html
    # A new run resets the sequence.
    assert "_activeRunId = null" in html
    assert "_lastEventSeq = -1" in html
    # Terminal events hide the stop button and refresh the state API.
    assert "stopBtn.style.display = 'none'" in html
    assert "refreshRunState()" in html


def test_s15_switch_to_monitor_refreshes_unified_state():
    html = _render_zh()
    # switchPage(3) must now call refreshRunState instead of guessing per API.
    assert "if (idx === 3) {" in html
    assert "refreshRunState();" in html
