"""Bugfix P1 — unified structured decision contract.

The intelligent analysis (analyze-folder), the Decision Agent, auto-tuning and
the UI must all share the same strict structured decision contract:

  - diagnosis (non-empty string)
  - action (non-empty string)
  - hyperparameter_changes (object)
  - training_overrides (object)
  - action != keep_params  => combined params non-empty
  - action == keep_params  => empty params allowed, UI shows an explicit reason
  - <= 3 combined params per iteration in normal mode
  - Rationale is sourced from `action`, never from a non-existent llm_rationale
  - structured failures keep the plain diagnosis, persist a stable/redacted
    suggestion error, and never masquerade as keep_params
  - one controlled "fix the JSON format" retry (still through the strict parser)
"""

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tune.modules.agent_engine import decision_agent
from auto_tune.modules.agent_engine.decision_agent import parse_decision_response
from auto_tune.ui import app as app_mod
from auto_tune.ui.i18n import make_translator

SAMPLE_CSV = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
    "1,1.5,3.0,2.0,0.1,0.2,0.05,0.01,1.6,3.1,2.1\n"
)

VALID_DECISION = {
    "diagnosis": "学习率偏高导致收敛不稳",
    "action": "降低学习率并增大 warmup",
    "hyperparameter_changes": {"lr0": 0.002},
    "training_overrides": {},
}

VALID_JSON = json.dumps(VALID_DECISION, ensure_ascii=False)


def _config():
    return {"llm": {"model": "deepseek-v4-flash", "endpoint": "https://api.deepseek.com/v1/chat/completions"}}


# ── core strict parser contract ─────────────────────────────────────────────


def test_markdown_json_code_block_parses():
    raw = "Here is the answer:\n```json\n" + VALID_JSON + "\n```\n"
    result = parse_decision_response(raw)
    assert result["error"] is None
    assert result["hyperparameter_changes"] == {"lr0": 0.002}


def test_empty_diagnosis_rejected():
    raw = json.dumps({
        "diagnosis": "", "action": "a",
        "hyperparameter_changes": {"lr0": 0.002}, "training_overrides": {},
    })
    assert "diagnosis" in parse_decision_response(raw)["error"]


def test_empty_action_rejected():
    raw = json.dumps({
        "diagnosis": "d", "action": "  ",
        "hyperparameter_changes": {"lr0": 0.002}, "training_overrides": {},
    })
    assert "action" in parse_decision_response(raw)["error"]


def test_unknown_parameter_rejected():
    raw = json.dumps({
        "diagnosis": "d", "action": "a",
        "hyperparameter_changes": {"not_a_param": 1}, "training_overrides": {},
    })
    assert "Unknown parameter" in parse_decision_response(raw)["error"]


def test_over_three_changes_rejected():
    raw = json.dumps({
        "diagnosis": "d", "action": "a",
        "hyperparameter_changes": {"lr0": 0.002, "box": 8.0, "cls": 0.7, "mosaic": 0.5},
        "training_overrides": {},
    })
    assert "最多修改 3 个" in parse_decision_response(raw)["error"]


def test_non_keep_params_empty_rejected():
    raw = json.dumps({
        "diagnosis": "d", "action": "change",
        "hyperparameter_changes": {}, "training_overrides": {},
    })
    assert "keep_params" in parse_decision_response(raw)["error"]


def test_keep_params_empty_accepted():
    raw = json.dumps({
        "diagnosis": "指标稳定", "action": "keep_params",
        "hyperparameter_changes": {}, "training_overrides": {},
    })
    result = parse_decision_response(raw)
    assert result["error"] is None
    assert result["action"] == "keep_params"


def test_training_overrides_count_towards_the_three_parameter_limit():
    raw = json.dumps({
        "diagnosis": "d", "action": "a",
        "hyperparameter_changes": {"lr0": 0.002, "box": 8.0},
        "training_overrides": {"epochs": 200, "imgsz": 640},
    })
    assert "最多修改 3 个" in parse_decision_response(raw)["error"]


# ── controlled single JSON-fix retry ────────────────────────────────────────


def test_parse_failure_retry_fix_succeeds(monkeypatch):
    calls = []
    valid = VALID_JSON

    def fake_call(prompt, config, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "not valid json at all {{"
        return valid

    monkeypatch.setattr(decision_agent, "call_decision_llm", fake_call)
    result = decision_agent.generate_suggestion("summary", None, _config())

    assert result["error"] is None
    assert result["retried"] is True
    assert result["hyperparameter_changes"] == {"lr0": 0.002}
    assert len(calls) == 2
    # The retry prompt explicitly asks to fix the JSON format.
    assert "JSON" in calls[1]


def _tuning_fact_package():
    return {
        "schema_version": "1.0",
        "fact_package_id": "sha256:retry",
        "task": "detect",
        "reference_run": "train38",
        "sources": {},
        "facts": [
            {"fact_id": "training.params.lr0", "value": 0.01, "source": "params"},
            {"fact_id": "training.issue.plateau", "value": True, "source": "training_report"},
        ],
    }


def _tuning_json():
    return json.dumps({
        "schema_version": "1.0",
        "fact_package_id": "sha256:retry",
        "diagnosis": "mAP 停滞",
        "action": "adjust",
        "hyperparameter_changes": {"lr0": 0.006},
        "training_overrides": {},
        "evidence_ids": {"lr0": ["training.issue.plateau"]},
    }, ensure_ascii=False)


def test_decide_hyperparameters_also_retries_once(monkeypatch):
    calls = []

    def fake_call(prompt, config, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "```json\n{oops}"
        return _tuning_json()

    monkeypatch.setattr(decision_agent, "call_decision_llm", fake_call)
    result = decision_agent.decide_hyperparameters(_tuning_fact_package(), _config())
    assert result["error"] is None
    assert result["retried"] is True
    assert len(calls) == 2
    assert result["evidence_ids"] == {"lr0": ["training.issue.plateau"]}


def test_retry_both_fail_stable_error_no_raw_leak(monkeypatch):
    leaky = "secret-key-sk1234567890 at C:\\Users\\evil\\path not json"

    def fake_call(prompt, config, **kwargs):
        return leaky

    monkeypatch.setattr(decision_agent, "call_decision_llm", fake_call)
    result = decision_agent.generate_suggestion("summary", None, _config())

    assert result["error"] == "Failed to parse JSON from LLM response"
    assert result["retried"] is True
    assert "secret-key" not in result["error"]
    assert "C:\\Users" not in result["error"]
    assert "raw_response" not in result  # suggestion never carries the raw model response


def test_retry_schema_failure_after_fix_is_stable(monkeypatch):
    # First attempt parses but fails validation (missing diagnosis); the retry
    # also fails the strict check. The final error is stable and honest.
    bad = json.dumps({"hyperparameter_changes": {"lr0": 0.002}})

    def fake_call(prompt, config, **kwargs):
        return bad

    monkeypatch.setattr(decision_agent, "call_decision_llm", fake_call)
    result = decision_agent.generate_suggestion("summary", None, _config())

    assert result["error"]
    assert result["retried"] is True
    assert result["hyperparameter_changes"] == {}
    assert result["action"] is None
    assert "raw_response" not in result


# ── _get_latest_suggestion: rationale = action, honest failure / keep_params ─


def test_latest_suggestion_rationale_comes_from_action():
    history = [{"decision": {
        "diagnosis": "学习率偏高",
        "action": "降低学习率",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {},
    }}]
    result = app_mod._get_latest_suggestion(history, None)
    assert result["diagnosis"] == "学习率偏高"
    assert result["rationale"] == "降低学习率"
    assert result["action"] == "降低学习率"
    assert result["error"] is None


def test_latest_suggestion_keep_params_surfaces_action():
    history = [{"decision": {
        "diagnosis": "指标稳定，暂不调整",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
    }}]
    result = app_mod._get_latest_suggestion(history, None)
    assert result is not None
    assert result["action"] == "keep_params"
    assert result["error"] is None
    assert result["hyperparameter_changes"] == {}


def test_latest_suggestion_structured_error_is_surfaced():
    training = {"suggestion": {"error": "Failed to parse JSON from LLM response"}}
    result = app_mod._get_latest_suggestion([], training)
    assert result is not None
    assert result["error"] == "Failed to parse JSON from LLM response"


def test_latest_suggestion_tuning_history_error_is_surfaced():
    history = [{"decision": {"error": "Failed to parse JSON from LLM response"}}]
    result = app_mod._get_latest_suggestion(history, None)
    assert result is not None
    assert result["error"]


def test_latest_suggestion_plain_diagnosis_is_not_a_suggestion():
    # A plain LLM diagnosis without a structured decision must not be presented
    # as an executable suggestion (avoids the "looks analyzed but cannot tune"
    # false-complete state).
    training = {"llm_analysis": {"train1": {"llm_diagnosis": "普通诊断"}}}
    assert app_mod._get_latest_suggestion([], training) is None


# ── template rendering: success / keep_params / structured failure ─────────


def _render_suggestion_page(latest_suggestion, current_args=None, lang="zh",
                            training=None, llm_analysis=None):
    from auto_tune.ui.app import _jinja_env

    translator = make_translator(lang)
    if training is None:
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
        current_lang=lang,
        active_page="agent_suggestion",
        experiment_history=[],
        tuning_history=[],
        dataset=None,
        training=training,
        project={},
        latest_suggestion=latest_suggestion,
        current_args=current_args,
        dataset_analyzer_config={},
        training_config={},
        llm_analysis=llm_analysis,
        vision_analysis=None,
        latest_dataset=None,
    )


def test_page_shows_diagnosis_rationale_current_and_suggested_values():
    latest = {
        "diagnosis": "学习率偏高导致收敛不稳",
        "rationale": "降低学习率并增大 warmup",
        "action": "降低学习率并增大 warmup",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {},
        "error": None,
    }
    html = _render_suggestion_page(latest, current_args={"lr0": 0.01, "batch": 16})
    assert "学习率偏高导致收敛不稳" in html       # Diagnosis
    assert "降低学习率并增大 warmup" in html      # Rationale (from action)
    assert "0.01" in html                        # current value
    assert "0.002" in html                       # suggested value


def test_page_shows_training_overrides_when_only_overrides():
    latest = {
        "diagnosis": "训练轮数偏少",
        "rationale": "增加训练轮数并提高耐心",
        "action": "增加训练轮数并提高耐心",
        "hyperparameter_changes": {},
        "training_overrides": {"epochs": 100, "patience": 20},
        "error": None,
    }
    html = _render_suggestion_page(latest, current_args={"epochs": 300, "patience": 50, "batch": 16})
    # both override params + suggested + current values are shown
    assert "epochs" in html
    assert "100" in html
    assert "300" in html
    assert "patience" in html
    assert "20" in html
    assert "50" in html
    # training_overrides must not be treated as "no suggestion"
    assert "暂无超参数修改建议" not in html
    # combined count is 2 (only training_overrides)
    assert "智能体建议 2 项修改" in html
    # each row carries a suggestion-type label
    assert "训练覆盖" in html


def test_page_shows_both_sections_with_total_count():
    latest = {
        "diagnosis": "学习率偏高且轮数不足",
        "rationale": "降低学习率并增加轮数",
        "action": "降低学习率并增加轮数",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {"epochs": 200},
        "error": None,
    }
    html = _render_suggestion_page(latest, current_args={"lr0": 0.01, "epochs": 300})
    # both sections render
    assert "lr0" in html
    assert "0.002" in html
    assert "epochs" in html
    assert "200" in html
    # total count = 1 + 1
    assert "智能体建议 2 项修改" in html
    # type labels for both kinds
    assert "超参数调整" in html
    assert "训练覆盖" in html


def test_page_overlap_deduplicates_with_override_priority():
    latest = {
        "diagnosis": "调整训练配置",
        "rationale": "以训练覆盖为准",
        "action": "以训练覆盖为准",
        "hyperparameter_changes": {"epochs": 100, "lr0": 0.002},
        "training_overrides": {"epochs": 200},
        "error": None,
    }
    html = _render_suggestion_page(latest, current_args={"epochs": 300, "lr0": 0.01})
    # the training override wins on overlap; the hyperparameter value is hidden
    assert 'value="200" name="epochs"' in html
    assert 'value="100" name="epochs"' not in html
    # the non-overlapping hyperparameter change still shows
    assert "lr0" in html
    assert 'value="0.002" name="lr0"' in html
    # deduped total count is 2 (epochs once + lr0)
    assert "智能体建议 2 项修改" in html


def test_page_training_overrides_english_no_hardcoded_labels():
    latest = {
        "diagnosis": "short epochs",
        "rationale": "raise epochs",
        "action": "raise epochs",
        "hyperparameter_changes": {},
        "training_overrides": {"epochs": 150},
        "error": None,
    }
    html = _render_suggestion_page(latest, current_args={"epochs": 300}, lang="en")
    assert "Training override" in html
    assert "智能体建议" not in html


def test_merge_suggestion_changes_override_priority():
    from auto_tune.ui.app import _merge_suggestion_changes

    merged = _merge_suggestion_changes([{"epochs": 100, "lr0": 0.002}, {"epochs": 200}])
    assert merged == [
        ["epochs", 200, "training"],
        ["lr0", 0.002, "hyperparameter"],
    ]


def test_merge_suggestion_changes_none_sections_handled():
    from auto_tune.ui.app import _merge_suggestion_changes

    assert _merge_suggestion_changes([None, {"epochs": 200}]) == [["epochs", 200, "training"]]
    assert _merge_suggestion_changes([{"lr0": 0.002}, None]) == [["lr0", 0.002, "hyperparameter"]]
    assert _merge_suggestion_changes([None, None]) == []


def test_page_keep_params_shows_explicit_reason():
    latest = {
        "diagnosis": "指标稳定，暂不调整",
        "rationale": "keep_params",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
        "error": None,
    }
    html = _render_suggestion_page(latest)
    assert "指标稳定，暂不调整" in html
    assert "保持原参数" in html
    assert "暂无超参数修改建议" not in html


def test_page_structured_failure_shows_generation_failed_not_no_suggestions():
    latest = {
        "diagnosis": "",
        "rationale": "",
        "action": "",
        "hyperparameter_changes": {},
        "training_overrides": {},
        "error": "Failed to parse JSON from LLM response",
    }
    html = _render_suggestion_page(latest)
    assert "调参建议生成失败" in html
    assert "暂无超参数修改建议" not in html
    assert "Failed to parse JSON from LLM response" in html


def test_page_structured_failure_english_label():
    latest = {
        "diagnosis": "", "rationale": "", "action": "",
        "hyperparameter_changes": {}, "training_overrides": {},
        "error": "Failed to parse JSON from LLM response",
    }
    html = _render_suggestion_page(latest, lang="en")
    assert "Suggestion generation failed" in html
    assert "No hyperparameter changes suggested yet." not in html


# ── i18n: both languages carry stable labels, no hardcoded Chinese in template ─


def test_error_label_translated_in_zh_and_en():
    assert make_translator("zh")("Suggestion generation failed") == "调参建议生成失败"
    assert make_translator("en")("Suggestion generation failed") == "Suggestion generation failed"
    assert make_translator("zh")("Rationale") == "调参理由"


# ── analyze-folder: strict parser instead of _extract_json ──────────────────


def _client():
    return TestClient(app_mod.app)


def _enable_llm(monkeypatch):
    monkeypatch.setitem(app_mod.APP_CONFIG, "llm", {
        "enabled": True,
        "model": "deepseek-v4-flash",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
    })
    monkeypatch.setitem(app_mod.APP_CONFIG, "vision", {"enabled": False})


def _mock_training_analysis(monkeypatch):
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.results_parser.load_training_run",
        lambda d: {"name": "train1", "results": {"total_epochs": 2, "best_epoch": 1}, "args": {}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.analyze_loss_curves",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.analyze_metric_curves",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.curve_analysis.detect_early_stopping",
        lambda r, c: {},
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.issue_detector.detect_issues", lambda r, c: []
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.run_comparator.compare_runs", lambda r, c: {}
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.run_comparator.summarize_runs",
        lambda r, c: {
            "best_mAP50": None, "average_mAP50": None,
            "runs_with_issues": 0, "common_issues": [],
        },
    )
    monkeypatch.setattr(
        "auto_tune.modules.train_analyzer.llm_analyzer.analyze_with_llm",
        lambda report, config: {
            "train1": {"llm_diagnosis": "普通诊断文本", "model_used": "m", "error": None},
        },
    )


def _train_dir(tmp_path):
    t = tmp_path / "train"
    t.mkdir()
    (t / "results.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    (t / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return t


def _latest_report(tmp_path):
    log_dir = tmp_path / "log"
    reports = sorted(log_dir.glob("*_report.json"), key=lambda p: p.stat().st_mtime)
    assert reports, f"no report saved in {log_dir}"
    with open(reports[-1], encoding="utf-8") as f:
        return json.load(f)


def test_analyze_folder_uses_strict_parser_not_extract_json(tmp_path, monkeypatch):
    """_extract_json accepts {hyperparameter_changes:...} without diagnosis; the
    strict parser must reject it as a structured failure."""
    _train_dir(tmp_path)
    _enable_llm(monkeypatch)
    _mock_training_analysis(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda prompt, config, **kwargs: json.dumps({"hyperparameter_changes": {"lr0": 0.002}}),
    )

    resp = _client().post("/api/training/analyze-folder", json={"path": str(tmp_path / "train")})
    assert resp.status_code == 200

    report = _latest_report(tmp_path)
    suggestion = report.get("suggestion") or {}
    assert suggestion.get("error")
    assert not suggestion.get("hyperparameter_changes")


def test_analyze_folder_plain_diagnosis_kept_structured_failure_no_executable(tmp_path, monkeypatch):
    """Plain diagnosis is preserved; a structured failure yields no executable
    parameters (never masquerades as keep_params)."""
    _train_dir(tmp_path)
    _enable_llm(monkeypatch)
    _mock_training_analysis(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        decision_agent, "call_decision_llm",
        lambda prompt, config, **kwargs: json.dumps({"hyperparameter_changes": {"lr0": 0.002}}),
    )

    resp = _client().post("/api/training/analyze-folder", json={"path": str(tmp_path / "train")})
    assert resp.status_code == 200

    report = _latest_report(tmp_path)
    # plain diagnosis preserved
    llm = report.get("llm_analysis") or {}
    assert llm.get("train1", {}).get("llm_diagnosis") == "普通诊断文本"
    # structured failure: no executable params
    suggestion = report.get("suggestion") or {}
    assert suggestion.get("error")
    assert suggestion.get("action") is None
    assert suggestion.get("hyperparameter_changes") == {}


def test_analyze_folder_valid_suggestion_written(tmp_path, monkeypatch):
    _train_dir(tmp_path)
    _enable_llm(monkeypatch)
    _mock_training_analysis(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda prompt, config, **k: VALID_JSON)

    resp = _client().post("/api/training/analyze-folder", json={"path": str(tmp_path / "train")})
    assert resp.status_code == 200

    report = _latest_report(tmp_path)
    suggestion = report.get("suggestion") or {}
    assert suggestion.get("error") is None
    assert suggestion.get("diagnosis") == "学习率偏高导致收敛不稳"
    assert suggestion.get("action") == "降低学习率并增大 warmup"
    assert suggestion.get("hyperparameter_changes") == {"lr0": 0.002}


# ── both analyze entries share the same strict decision path ────────────────


class _FakeUpload:
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("results.csv", SAMPLE_CSV)
        zf.writestr("args.yaml", "epochs: 1\n")
    return buf.getvalue()


def test_legacy_zip_and_folder_analyze_reject_bad_schema_consistently(tmp_path, monkeypatch):
    """Both training-analyze entries must reject a missing-diagnosis decision as
    a structured failure (shared strict parser), never a partial suggestion."""
    _enable_llm(monkeypatch)
    _mock_training_analysis(monkeypatch)
    monkeypatch.chdir(tmp_path)
    bad = json.dumps({"hyperparameter_changes": {"lr0": 0.002}})
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda prompt, config, **k: bad)

    # folder entry
    _train_dir(tmp_path)
    resp = _client().post("/api/training/analyze-folder", json={"path": str(tmp_path / "train")})
    assert resp.status_code == 200
    folder_report = _latest_report(tmp_path)
    folder_sug = folder_report.get("suggestion") or {}

    # legacy ZIP entry
    asyncio.run(app_mod._analyze_train_zip(_FakeUpload("train.zip", _zip_bytes())))
    zip_report = _latest_report(tmp_path)
    zip_sug = zip_report.get("suggestion") or {}

    assert folder_sug.get("error")
    assert zip_sug.get("error")
    assert not folder_sug.get("hyperparameter_changes")
    assert not zip_sug.get("hyperparameter_changes")


# ── auto-tuning: decision schema error aborts before training launch ────────


def test_loop_schema_error_aborts_before_launch(tmp_path, monkeypatch):
    from auto_tune.modules.agent_engine.loop import run_tuning_loop

    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_tuning_fact_package",
        lambda *a, **k: _tuning_fact_package(),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: {
            "diagnosis": None, "action": None,
            "hyperparameter_changes": {}, "training_overrides": {},
            "raw_response": None, "error": "Failed to parse JSON from LLM response",
            "retried": True,
        },
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
        skip_execute=False,
    )

    assert launched == []
    assert result["failure"]["error_type"] == "decision_schema_error"
    assert result["failure"]["stage"] == "decision"


# ── F1.1: current training report must not be crossed with unrelated history ─

TRAIN64_TRAINING = {
    "runs": {"train64": {"name": "train64"}},
    "summary": {"best_overall_run": "train64"},
    "suggestion": {
        "diagnosis": "train64 诊断",
        "action": "调整 train64",
        "hyperparameter_changes": {
            "weight_decay": 0.0015,
            "mixup": 0.1,
            "lr0": 0.0005,
        },
        "training_overrides": {},
        "error": None,
    },
}

UNRELATED_KEEP_PARAMS_HISTORY = [{
    "train_name": "autotune_9ff8be0f_iter01",
    "decision": {
        "diagnosis": "按原有参数训练，不做超参数调整",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
    },
}]


def test_latest_suggestion_prefers_current_report_over_unrelated_history():
    result = app_mod._get_latest_suggestion(
        UNRELATED_KEEP_PARAMS_HISTORY, TRAIN64_TRAINING)

    assert result is not None
    assert result["diagnosis"] == "train64 诊断"
    assert result["action"] == "调整 train64"
    assert result["hyperparameter_changes"] == {
        "weight_decay": 0.0015, "mixup": 0.1, "lr0": 0.0005,
    }
    assert result["action"] != "keep_params"


def test_latest_suggestion_history_fallback_requires_run_match():
    decision = {
        "diagnosis": "同运行诊断",
        "action": "降低学习率",
        "hyperparameter_changes": {"lr0": 0.002},
        "training_overrides": {},
    }
    # Unrelated history must not fill in for a report without its own suggestion.
    assert app_mod._get_latest_suggestion(
        UNRELATED_KEEP_PARAMS_HISTORY,
        {"runs": {"train64": {}}, "suggestion": None},
    ) is None
    # A history entry bound to the same run is a valid fallback.
    matched = [{"train_name": "autotune_same", "decision": decision}]
    fallback = app_mod._get_latest_suggestion(
        matched, {"runs": {"autotune_same": {}}, "suggestion": None})
    assert fallback is not None
    assert fallback["action"] == decision["action"]
    assert fallback["hyperparameter_changes"] == {"lr0": 0.002}
    # Without a training report the legacy "latest history" behaviour stays.
    assert app_mod._get_latest_suggestion(
        UNRELATED_KEEP_PARAMS_HISTORY, None) is not None


def test_latest_suggestion_scans_history_backwards_for_matching_run():
    stale = {
        "diagnosis": "较早的同运行建议",
        "action": "提高 imgsz",
        "hyperparameter_changes": {"imgsz": 960},
        "training_overrides": {},
    }
    newer = {
        "diagnosis": "其他运行的建议",
        "action": "降低学习率",
        "hyperparameter_changes": {"lr0": 0.001},
        "training_overrides": {},
    }
    history = [
        {"train_name": "autotune_same", "decision": stale},
        {"train_name": "autotune_other", "decision": newer},
    ]
    result = app_mod._get_latest_suggestion(
        history, {"runs": {"autotune_same": {}}, "suggestion": None})
    assert result is not None
    assert result["action"] == "提高 imgsz"
    assert result["hyperparameter_changes"] == {"imgsz": 960}


def test_latest_suggestion_report_failure_wins_over_history():
    training = {
        "runs": {"train64": {}},
        "summary": {"best_overall_run": "train64"},
        "suggestion": {"error": "Suggestion generation failed"},
    }
    result = app_mod._get_latest_suggestion(
        UNRELATED_KEEP_PARAMS_HISTORY, training)
    assert result is not None
    assert result["error"] == "Suggestion generation failed"
    assert result["action"] == ""


def test_latest_suggestion_report_keep_params_wins_over_history():
    training = {
        "runs": {"train64": {}},
        "summary": {"best_overall_run": "train64"},
        "suggestion": {
            "diagnosis": "指标稳定",
            "action": "keep_params",
            "hyperparameter_changes": {},
            "training_overrides": {},
            "error": None,
        },
    }
    history = [{
        "train_name": "autotune_other",
        "decision": {
            "diagnosis": "其他运行",
            "action": "降低学习率",
            "hyperparameter_changes": {"lr0": 0.002},
            "training_overrides": {},
        },
    }]
    result = app_mod._get_latest_suggestion(history, training)
    assert result is not None
    assert result["action"] == "keep_params"
    assert result["hyperparameter_changes"] == {}


# ── F1.1: blank LLM analysis must not leave a title-only card ──────────────

TRAIN64_STRUCTURED_DIAGNOSIS = "验证损失上升而训练损失下降，出现中度过拟合。"


def _train64_training_with_llm(llm_diagnosis="", suggestion_diagnosis=TRAIN64_STRUCTURED_DIAGNOSIS):
    return {
        "runs": {"train64": {"name": "train64"}},
        "summary": {"best_overall_run": "train64"},
        "llm_analysis": {
            "train64": {
                "llm_diagnosis": llm_diagnosis,
                "model_used": "deepseek-v4-flash",
                "error": None,
            },
        },
        "suggestion": {
            "diagnosis": suggestion_diagnosis,
            "action": "调整正则化和学习率",
            "hyperparameter_changes": {"weight_decay": 0.0015},
            "training_overrides": {},
            "error": None,
        },
    }


def test_llm_analysis_falls_back_to_same_report_diagnosis():
    training = _train64_training_with_llm()
    display = app_mod._get_llm_analysis_display(training)
    assert display is not None
    assert display["train64"]["llm_diagnosis"] == TRAIN64_STRUCTURED_DIAGNOSIS
    assert display["train64"]["model_used"] == "deepseek-v4-flash"
    # the source report must not be mutated in place
    assert training["llm_analysis"]["train64"]["llm_diagnosis"] == ""


def test_llm_analysis_does_not_cross_runs():
    training = {
        "runs": {"train64": {"name": "train64"}, "train65": {"name": "train65"}},
        "summary": {"best_overall_run": "train64"},
        "llm_analysis": {
            "train65": {"llm_diagnosis": "", "model_used": "m", "error": None},
        },
        "suggestion": {
            "diagnosis": TRAIN64_STRUCTURED_DIAGNOSIS,
            "action": "调整正则化和学习率",
            "hyperparameter_changes": {"weight_decay": 0.0015},
            "training_overrides": {},
            "error": None,
        },
    }
    assert app_mod._get_llm_analysis_display(training) is None

    # An unresolvable multi-run identity must not guess either.
    ambiguous = dict(training)
    ambiguous["summary"] = {"best_overall_run": "train99"}
    assert app_mod._get_llm_analysis_display(ambiguous) is None


def test_llm_analysis_blank_or_missing_returns_none():
    training = _train64_training_with_llm(suggestion_diagnosis="")
    assert app_mod._get_llm_analysis_display(training) is None
    assert app_mod._get_llm_analysis_display({"runs": {}, "llm_analysis": {}}) is None
    assert app_mod._get_llm_analysis_display(None) is None


def test_llm_analysis_whitespace_treated_as_empty():
    training = _train64_training_with_llm(llm_diagnosis="   \n  ")
    display = app_mod._get_llm_analysis_display(training)
    assert display is not None
    assert display["train64"]["llm_diagnosis"] == TRAIN64_STRUCTURED_DIAGNOSIS


def test_llm_analysis_keeps_existing_diagnosis():
    training = _train64_training_with_llm(llm_diagnosis="普通诊断文本")
    display = app_mod._get_llm_analysis_display(training)
    assert display["train64"]["llm_diagnosis"] == "普通诊断文本"


def test_llm_analysis_error_not_masked_by_suggestion():
    training = _train64_training_with_llm()
    training["llm_analysis"]["train64"]["error"] = "LLM request failed"
    display = app_mod._get_llm_analysis_display(training)
    assert display is not None
    assert display["train64"]["error"] == "LLM request failed"
    assert display["train64"]["llm_diagnosis"] == ""
    assert TRAIN64_STRUCTURED_DIAGNOSIS not in (display["train64"]["llm_diagnosis"] or "")


def test_selected_report_page_shows_bound_diagnosis_and_hides_unrelated_keep_params():
    training = _train64_training_with_llm()
    display = app_mod._get_llm_analysis_display(training)
    latest = app_mod._get_latest_suggestion(UNRELATED_KEEP_PARAMS_HISTORY, training)
    html = _render_suggestion_page(
        latest,
        current_args={"weight_decay": 0.0005, "lr0": 0.01},
        training=training,
        llm_analysis=display,
    )
    assert "train64" in html
    assert "deepseek-v4-flash" in html
    assert TRAIN64_STRUCTURED_DIAGNOSIS in html
    assert "weight_decay" in html
    assert "按原有参数训练" not in html


def test_blank_analysis_card_not_rendered_when_nothing_to_show():
    training = _train64_training_with_llm(suggestion_diagnosis="")
    display = app_mod._get_llm_analysis_display(training)
    latest = app_mod._get_latest_suggestion([], training)
    html = _render_suggestion_page(
        latest, training=training, llm_analysis=display)
    # The card (and its title span) must be gone; the label string also lives in
    # an unrelated client-side template, so the card element is the discriminator.
    assert 'id="llmAnalysisCard"' not in html
    assert '<span class="card-title" style="border:none; margin:0; padding:0;">' \
        + make_translator("zh")("LLM Analysis Report") not in html


def test_llm_analysis_fallback_text_is_html_escaped():
    malicious = "<script>alert(1)</script>"
    training = _train64_training_with_llm(suggestion_diagnosis=malicious)
    display = app_mod._get_llm_analysis_display(training)
    latest = app_mod._get_latest_suggestion([], training)
    html = _render_suggestion_page(
        latest, training=training, llm_analysis=display)
    assert malicious not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
