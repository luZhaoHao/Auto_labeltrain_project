"""H1.3 Task 3: training-mode selection + HPO configuration UI.

Structural assertions on the real templates + HTTP behaviour of /tuning/start.
These are NOT a browser pass — Codex still performs the real browser walkthrough.
No real training, no network LLM.

F1.1-C Task 1: the page offers the four approved selectable modes
(``dry_run``, ``keep_params``, ``hpo``, ``full``) in the frozen order, and the
backend ``/tuning/start`` contract keeps accepting all of them.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from auto_tune.ui.i18n import make_translator

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
_TEMPLATES = _UI_DIR / "templates"
_MODE_ORDER = ["dry_run", "keep_params", "hpo", "full"]


def _raw(name: str) -> str:
    return (_TEMPLATES / name).read_text(encoding="utf-8")


def _mode_options(html: str) -> list[str]:
    block = html.split('name="mode"', 1)[1].split("</select>", 1)[0]
    return re.findall(r'<option value="([^"]+)"', block)


# ── four selectable modes: values / names / order identical in both pages ──


@pytest.mark.parametrize("template", ["single_page.html", "agent_suggestion.html"])
def test_page_mode_options_order_and_values(template):
    assert _mode_options(_raw(template)) == _MODE_ORDER


@pytest.mark.parametrize("template", ["single_page.html", "agent_suggestion.html"])
def test_page_mode_options_share_translation_keys(template):
    html = _raw(template)
    for key in ("Dry-Run (generate plan only)", "Train with Original Parameters",
                "HPO Algorithm Tuning", "LLM Parameter Tuning"):
        assert key in html
    # 干运行是真实可选选项：在 DOM 里，不是靠 CSS/属性藏起来的
    assert 'value="dry_run"' in html


def test_mode_labels_localized():
    zh = make_translator("zh")
    assert zh("Dry-Run (generate plan only)") == "干运行（仅生成计划）"
    assert zh("Train with Original Parameters") == "按原来参数训练"
    assert zh("HPO Algorithm Tuning") == "HPO 算法调参"
    assert zh("LLM Parameter Tuning") == "大模型调参"
    en = make_translator("en")
    assert en("HPO Algorithm Tuning") == "HPO Algorithm Tuning"
    assert en("LLM Parameter Tuning") == "LLM Parameter Tuning"


# ── HPO panel markup: independent of any LLM suggestion ────────────


def test_hpo_panel_is_independent_of_latest_suggestion():
    html = _raw("single_page.html")
    form = html.index('<form id="tuningForm">')
    panel = html.index('id="hpoPanel"')
    assert form < panel
    # no suggestion-gated conditional wraps the HPO panel
    between = html[form:panel]
    assert "latest_suggestion" not in between


def test_hpo_config_defaults_and_ranges():
    """Ranges and defaults of the HPO draft; the snapshot/weights inputs are
    selectors since the 2026-09-14 rework (no hand-typed ids/paths required)."""
    html = _raw("single_page.html")
    assert 'id="hpoSnapshotSelect"' in html                  # snapshot selector
    assert 'id="hpoSelectBtn"' in html
    assert 'id="hpoModelSelect"' in html                     # local .pt selector
    # 第四轮：主界面不再提供可编辑的物理路径输入
    assert 'id="hpoModelPath"' not in html
    assert 'value="10" min="1" max="100"' in html            # budget
    assert 'value="30" min="1" max="1000"' in html           # epochs
    assert 'value="42" min="0" max="2147483647"' in html     # seed
    assert 'value="16" min="1" max="256"' in html            # batch
    assert 'value="640" min="32" max="2048" step="32"' in html  # imgsz
    assert 'value="3600" min="1" max="86400"' in html        # timeout
    # 设备是明确选择控件；模板不预置 cpu，选项由服务端探测结果填充
    device_block = html.split('id="hpoDevice"', 1)[0].rsplit("<", 1)[1]
    assert device_block.startswith("select")
    assert 'value="cpu"' not in html.split('id="hpoDevice"', 1)[1].split("</select>", 1)[0]
    sampler_block = html.split('id="hpoSampler"', 1)[1].split("</select>", 1)[0]
    assert re.findall(r'<option value="([^"]+)"', sampler_block) == ["tpe", "random"]
    # 新建研究只允许全面/快速两种评价模式（旧版单指标只能被读取与展示）
    mode_block = html.split('id="hpoEvaluationMode"', 1)[1].split("</select>", 1)[0]
    assert re.findall(r'<option value="([^"]+)"', mode_block) == [
        "comprehensive", "quick"]


def test_hpo_controls_and_actions_present():
    """HPO actions/regions exist. ``Create & Prepare Study`` was replaced by the
    single ``Create & Start Tuning`` primary action in the 2026-09-14 rework."""
    html = _raw("single_page.html")
    for control in ("hpoCreateAndStartBtn", "hpoStartBtn", "hpoStopBtn",
                    "hpoResumeBtn", "hpoRefreshBtn", "hpoVerifyBtn",
                    "hpoOpenResultFolder", "hpoDownloadBest", "hpoHistoryToggle",
                    "hpoStatusText", "hpoHistoryList", "hpoProgressBar",
                    "hpoPanel", "hpoProgress", "hpoBestArea", "hpoHistorySection",
                    "hpoFormalMonitorHost", "sharedMonitorBlock"):
        assert 'id="' + control + '"' in html
    # 完整 Trial 表与搜索阶段 last.pt 入口已从主界面移除
    assert 'id="hpoTrialsBody"' not in html
    assert "last.pt" not in html
    assert '/static/hpo.js' in html


# ── LLM-only controls hidden for HPO ───────────────────────────────


def test_llm_only_markers_present():
    html = _raw("single_page.html")
    assert "suggestion-card llm-only" in html
    assert html.count('class="llm-only"') >= 3  # ref run / max retries / eval mode


def test_hpo_js_toggles_llm_only():
    script = (_UI_DIR / "static" / "hpo.js").read_text(encoding="utf-8")
    assert "llm-only" in script
    assert "textContent" in script
    assert ".innerHTML" not in script  # user text never injected as HTML


def test_tuning_form_early_returns_for_hpo_before_llm_fetch():
    html = _raw("single_page.html")
    guard = html.index("formData.get('mode') === 'hpo'")
    fetch_tuning = html.index("fetch('/tuning/start'")
    assert guard < fetch_tuning
    snippet = html[guard:guard + 400]
    assert "hpoHandleSubmit" in snippet
    assert "return;" in snippet


def test_agent_suggestion_also_early_returns():
    html = _raw("agent_suggestion.html")
    guard = html.index("=== 'hpo'")
    fetch_tuning = html.index("fetch('/tuning/start'")
    assert guard < fetch_tuning
    assert "hpoHandleSubmit" in html[guard:guard + 400]


# ── /tuning/start rejects unknown + hpo; no LLM / real training ────


def _redirect_log(monkeypatch, tmp_path):
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


@pytest.mark.parametrize("mode", ["hpo", "bogus", ""])
def test_tuning_start_rejects_unknown_modes_before_llm(tmp_path, monkeypatch, mode):
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("unknown mode must not reach the LLM tuning loop")

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception", boom)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)
    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json={
            "reference_run": None, "max_retries": 1, "mode": mode,
            "auto_analyze": False, "auto_loop": False,
        })
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "INVALID_MODE"
        assert app_mod._RUN_MANAGER.active_tuning() is None
    finally:
        app_mod._running_training.clear()


def test_tuning_start_accepts_dry_run(tmp_path, monkeypatch):
    """页面恢复四模式后，后端 dry-run 语义仍原样保留。

    页面渲染 ``dry_run`` 选项不等于后端可以省掉 dry-run 的业务分支：它必须只生成
    计划、不执行训练，也不占训练槽位。
    """
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    _redirect_log(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_read_latest_dataset", lambda: None)
    captured = {}

    def fake_run_tuning_loop(config, **kwargs):
        captured.update(kwargs)
        return {"error": None, "iterations": []}

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.run_tuning_loop", fake_run_tuning_loop)
    app_mod._running_training.clear()
    try:
        client = TestClient(app_mod.app)
        resp = client.post("/tuning/start", json={
            "reference_run": None, "max_retries": 1, "mode": "dry_run",
            "auto_analyze": False, "auto_loop": False,
        })
        assert resp.status_code == 200, resp.text
        assert "INVALID_MODE" not in resp.text
        # 真正走到了原有 dry-run 分支：只生成计划，不执行训练，也不占训练槽位
        assert captured, "dry_run must still reach the tuning loop"
        assert captured["skip_execute"] is True
        assert app_mod._RUN_MANAGER.reservation_owner() is None
    finally:
        app_mod._running_training.clear()


# ── static script route + JS syntax ────────────────────────────────


def test_static_hpo_js_route():
    from fastapi.testclient import TestClient
    from auto_tune.ui import app as app_mod

    client = TestClient(app_mod.app)
    resp = client.get("/static/hpo.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers.get("content-type", "")
    assert "hpoHandleSubmit" in resp.text


def test_hpo_js_syntax_with_node_if_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = _UI_DIR / "static" / "hpo.js"
    result = subprocess.run([node, "--check", str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
