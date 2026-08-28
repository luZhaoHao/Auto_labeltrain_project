"""Bugfix P3: recent training runs shortcut (UI contract).

Renders the real single_page.html and pins the intelligent-analysis "recent
training" section: it only fills trainingPathInput on click (never auto-analyzes),
degrades honestly on empty/API failure without blocking the existing controls,
and renders every external string through textContent (never innerHTML).
"""

from auto_tune.ui.app import _jinja_env
from auto_tune.ui.i18n import make_translator

_ZH = make_translator("zh")
_EN = make_translator("en")


def _render(active_page="agent_suggestion", **ctx):
    from auto_tune.modules.presentation import build_experiment_labels

    defaults = {
        "_": _ZH,
        "current_lang": "zh",
        "experiment_labels": build_experiment_labels(_ZH),
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


def _recent_js(html):
    """Slice the Bugfix P3 recent-training JavaScript block out of the page."""
    start = html.index("function loadRecentTraining")
    end = html.index("// ── Training folder analyze ──", start)
    return html[start:end]


# ── structure ──


def test_recent_training_section_is_after_path_input():
    html = _render()
    assert 'id="trainingPathInput"' in html
    assert 'id="recentTrainingSection"' in html
    assert 'id="recentTrainingList"' in html
    assert html.index('id="trainingPathInput"') < html.index('id="recentTrainingSection"')


def test_recent_js_present_and_fetches_bounded_limit():
    html = _render()
    block = _recent_js(html)
    assert "loadRecentTraining" in block
    assert "renderRecentTrainingItem" in block
    assert "/api/training/recent-runs?limit=4" in block


def test_recent_section_loaded_when_entering_intelligent_analysis():
    html = _render()
    assert "switchPage(2)" in html  # nav already exists
    # The switchPage hook must load recent training on entry to page 2.
    assert "if (idx === 2) loadRecentTraining()" in html
    # DOMContentLoaded also loads it when page 2 is the initially active page.
    assert "loadRecentTraining();" in html


# ── click behavior: fill only, never auto-analyze ──


def test_recent_click_only_fills_training_path_input():
    html = _render()
    block = _recent_js(html)
    assert "trainingPathInput" in block
    assert "input.value = item.run_dir || ''" in block
    assert "analyzeTrainingFolder" not in block


def test_recent_item_rendered_via_textcontent():
    html = _render()
    block = _recent_js(html)
    assert "textContent" in block
    assert "innerHTML" not in block


def test_recent_map50_missing_uses_dash():
    html = _render()
    block = _recent_js(html)
    assert "_formatRecentMap" in block
    # Missing map50 must render as the true dash, never as 0 or empty.
    assert "isNaN(n)) return '—'" in block
    assert "toFixed(4)" in block


# ── degradation ──


def test_recent_empty_message_present():
    html = _render()
    assert "暂无可用的最近训练" in html  # zh: "No recent training results available"


def test_recent_unavailable_message_present():
    html = _render()
    assert "最近训练暂不可用" in html  # zh: "Recent training is temporarily unavailable"


def test_recent_failure_does_not_disable_existing_controls():
    html = _render()
    # The analyze and browse buttons keep working regardless of the recent block.
    assert 'id="trainingAnalyzeBtn"' in html
    block = _recent_js(html)
    assert "disabled" not in block
    assert "browseFolder" not in block
    assert "analyzeTrainingFolder" not in block


# ── i18n ──


def test_recent_i18n_keys_zh():
    assert _ZH("Recent training") == "最近训练"
    assert _ZH("No recent training results available") == "暂无可用的最近训练"
    assert _ZH("Recent training is temporarily unavailable") == "最近训练暂不可用"
    assert _ZH("Manual training") == "普通训练"
    assert _ZH("Auto tuning") == "自动调优"
    assert _ZH("Completed at") == "完成时间"
    assert _ZH("mAP50") == "mAP50"


def test_recent_i18n_keys_en():
    assert _EN("Recent training") == "Recent training"
    assert _EN("No recent training results available") == "No recent training results available"
    assert _EN("Recent training is temporarily unavailable") == "Recent training is temporarily unavailable"
    assert _EN("Manual training") == "Manual training"
    assert _EN("Auto tuning") == "Auto tuning"
    assert _EN("Completed at") == "Completed at"
    assert _EN("mAP50") == "mAP50"


def test_recent_source_labels_zh_and_en():
    assert _ZH("Manual training") != _ZH("Auto tuning")
    assert _EN("Manual training") == "Manual training"
    assert _EN("Auto tuning") == "Auto tuning"
