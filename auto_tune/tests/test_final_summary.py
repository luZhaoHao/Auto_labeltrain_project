"""Bugfix P1.1 — deterministic final tuning summary + LLM closing + TXT write.

After all tuning iterations complete, a deterministic final summary must be
built (best round via the evaluation-mode score), a single optional LLM call may
add a closing explanation, and ``tuning_final_summary.txt`` must be written
atomically into the best tuning run directory. LLM failure never changes the
completed training fact and never blocks the deterministic TXT.
"""

import json
import os
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine.final_summary import (
    LLM_SUMMARY_FAILED,
    LLM_SUMMARY_OK,
    build_deterministic_summary,
    composite_score,
    render_final_summary_text,
    write_final_summary_txt,
)


def _iteration(iteration, map50, map50_95=None, precision=None, recall=None,
               train_name=None, decision=None, guardrails=None, merged=None,
               error=None, analysis_status="completed"):
    return {
        "iteration": iteration,
        "timestamp": "2026-08-01T00:00:00Z",
        "perception": {},
        "decision": decision or {"diagnosis": "d", "action": "a",
                                 "hyperparameter_changes": {}, "training_overrides": {}},
        "guard_results": guardrails or {"valid": True, "warnings": [], "errors": [],
                                        "clamped": {}, "sanitized_changes": {}},
        "merged_params": merged or {"epochs": 30, "patience": 20, "lr0": 0.01},
        "train_name": train_name or f"autotune_s1_iter{iteration:02d}",
        "probe_decision": {"verdict": "continue", "reason": "", "suggestion": ""},
        "result_mAP50": map50,
        "result_mAP50_95": map50_95,
        "result_precision": precision,
        "result_recall": recall,
        "result_best_epoch": 5,
        "result_analysis_status": analysis_status,
        "error": error,
    }


def _tuning_result(iterations, eval_mode="comprehensive", **top):
    result = {
        "module": "agent_engine",
        "version": "1.0",
        "reference_run": "train53",
        "eval_mode": eval_mode,
        "session_id": "s1",
        "iterations": iterations,
    }
    result.update(top)
    return result


def test_composite_score_quick_and_comprehensive():
    it = {"result_mAP50": 0.5, "result_mAP50_95": 0.3, "result_precision": 0.4, "result_recall": 0.6}
    assert composite_score(it, "quick") == pytest.approx(0.5 * 0.6 + 0.3 * 0.4)
    assert composite_score(it, "comprehensive") == pytest.approx(
        0.5 * 0.35 + 0.3 * 0.25 + 0.4 * 0.20 + 0.6 * 0.20
    )


def test_build_summary_single_round(tmp_path):
    summary = build_deterministic_summary(
        _tuning_result([_iteration(1, 0.5, 0.3, 0.4, 0.6)]),
        "comprehensive", "s1", "train53",
    )
    assert summary["best_iteration"] == 1
    assert summary["best_train_name"] == "autotune_s1_iter01"
    assert summary["total_iterations"] == 1
    assert summary["evaluation_mode"] == "comprehensive"
    assert summary["best_metrics"]["mAP50"] == 0.5
    assert summary["best_weights_path"] == "weights/best.pt"
    assert summary["generated_at"]


def test_build_summary_three_rounds_comprehensive_best(tmp_path):
    iters = [
        _iteration(1, 0.40, 0.20, 0.30, 0.50),
        _iteration(2, 0.55, 0.30, 0.45, 0.60),
        _iteration(3, 0.50, 0.25, 0.40, 0.55),
    ]
    summary = build_deterministic_summary(
        _tuning_result(iters), "comprehensive", "s1", "train53",
    )
    assert summary["total_iterations"] == 3
    assert summary["best_iteration"] == 2
    assert summary["best_train_name"] == "autotune_s1_iter02"
    scores = [composite_score(it, "comprehensive") for it in iters]
    assert summary["best_score"] == pytest.approx(max(scores))


def test_build_summary_quick_mode_uses_quick_score(tmp_path):
    iters = [
        _iteration(1, 0.40, 0.20, 0.30, 0.50),
        _iteration(2, 0.55, 0.30, 0.45, 0.60),
    ]
    summary = build_deterministic_summary(_tuning_result(iters, eval_mode="quick"), "quick", "s1", "train53")
    quick_scores = [composite_score(it, "quick") for it in iters]
    assert summary["best_iteration"] == 2
    assert summary["best_score"] == pytest.approx(max(quick_scores))


def test_build_summary_analysis_failed_round_listed_not_best(tmp_path):
    failed = _iteration(1, None, None, None, None, analysis_status="failed", error=None)
    ok = _iteration(2, 0.5, 0.3, 0.4, 0.6)
    summary = build_deterministic_summary(
        _tuning_result([failed, ok]), "comprehensive", "s1", "train53",
    )
    assert summary["total_iterations"] == 2
    rounds = {r["iteration"]: r for r in summary["iterations"]}
    assert rounds[1]["analysis_status"] == "failed"
    assert summary["best_iteration"] == 2


def test_build_summary_none_when_no_successful_rounds():
    summary = build_deterministic_summary(
        _tuning_result([_iteration(1, None, None, None, None, error="训练中止")]),
        "comprehensive", "s1", "train53",
    )
    assert summary is None


def test_build_summary_none_when_no_measurable_round():
    """A round that trained but produced no metrics has no best round."""
    summary = build_deterministic_summary(
        _tuning_result([_iteration(1, None, None, None, None, analysis_status="failed")]),
        "comprehensive", "s1", "train53",
    )
    assert summary is None


def test_render_text_contains_all_rounds_and_best(tmp_path):
    summary = build_deterministic_summary(
        _tuning_result([
            _iteration(1, 0.40, 0.20, 0.30, 0.50),
            _iteration(2, 0.55, 0.30, 0.45, 0.60),
        ]),
        "comprehensive", "s1", "train53",
    )
    text = render_final_summary_text(summary)
    assert "autotune_s1_iter01" in text
    assert "autotune_s1_iter02" in text
    assert "最佳轮次" in text
    assert "第 2 轮" in text or "iter02" in text
    assert "weights/best.pt" in text


def test_render_text_includes_llm_closing_when_provided(tmp_path):
    summary = build_deterministic_summary(
        _tuning_result([_iteration(1, 0.5, 0.3, 0.4, 0.6)]), "comprehensive", "s1", "train53",
    )
    text = render_final_summary_text(summary, llm_text="第二轮的 lr0 调整明显有效。")
    assert "AI 总结" in text
    assert "第二轮的 lr0 调整明显有效。" in text


def test_write_txt_atomic_success(tmp_path):
    best_dir = tmp_path / "detect" / "autotune_s1_iter01"
    best_dir.mkdir(parents=True)
    result = write_final_summary_txt(
        {"best_train_name": "autotune_s1_iter01"}, str(best_dir), "确定性内容",
    )
    assert result["status"] == "generated"
    assert Path(result["path"]).name == "tuning_final_summary.txt"
    assert (best_dir / "tuning_final_summary.txt").read_text(encoding="utf-8") == "确定性内容"
    # no temp files left behind
    assert [p.name for p in best_dir.iterdir()] == ["tuning_final_summary.txt"]


def test_write_txt_failure_is_non_fatal(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file", encoding="utf-8")
    result = write_final_summary_txt({"best_train_name": "x"}, str(blocker), "text")
    assert result["status"] == "failed"
    assert result["error_code"]


def test_summary_does_not_leak_absolute_path_or_secrets(tmp_path):
    """The deterministic summary only carries relative names and basenames."""
    it = _iteration(1, 0.5, 0.3, 0.4, 0.6,
                    merged={"epochs": 30, "data": "C:/secret/dataset.yaml"})
    # merged_params may carry an absolute data path internally, but the
    # rendered text must not expose a full command or dataset path.
    summary = build_deterministic_summary(
        _tuning_result([it]), "comprehensive", "s1", "train53",
    )
    text = render_final_summary_text(summary)
    assert "C:/secret" not in text
    assert "api_key" not in text.lower()


def test_build_summary_llm_failure_contract():
    """Direct module contract: LLM failure maps to a stable status."""
    from auto_tune.modules.agent_engine import final_summary as fs

    # simulate a provider error; the module maps it to a stable code
    class _Boom:
        def __init__(self, message):
            self.message = message

        def __str__(self):
            return self.message

    result = fs._llm_failure_result(_Boom("DeepSeek API error: credential_missing"))
    assert result["status"] == LLM_SUMMARY_FAILED
    assert result["error_code"] == "LLM_SUMMARY_CREDENTIAL_MISSING"
    assert result["text"] is None


# ── P1.1-min: LLM closing-summary honest failure / empty / disabled ─────────


def test_call_final_summary_llm_skipped_when_disabled():
    from auto_tune.modules.agent_engine import final_summary as fs

    result = fs.call_final_summary_llm({"x": 1}, {"llm": {"enabled": False}})
    assert result["status"] == "skipped"
    assert result["error_code"] is None
    assert result["text"] is None


def test_call_final_summary_llm_network_error_stable_code(monkeypatch):
    from auto_tune.modules.agent_engine import final_summary as fs
    from auto_tune.modules.agent_engine import decision_agent

    def boom(prompt, config):
        raise RuntimeError("DeepSeek API error: network_failed")

    monkeypatch.setattr(decision_agent, "call_decision_llm", boom)
    result = fs.call_final_summary_llm({"x": 1}, {"llm": {"enabled": True}})
    assert result["status"] == LLM_SUMMARY_FAILED
    assert result["error_code"] == "LLM_SUMMARY_NETWORK_FAILED"
    assert result["text"] is None


def test_call_final_summary_llm_empty_response_is_failed(monkeypatch):
    """An empty/whitespace LLM response is honestly failed, not 'ok'."""
    from auto_tune.modules.agent_engine import final_summary as fs
    from auto_tune.modules.agent_engine import decision_agent

    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda p, c: "   \n  ")
    result = fs.call_final_summary_llm({"x": 1}, {"llm": {"enabled": True}})
    assert result["status"] == LLM_SUMMARY_FAILED
    assert result["error_code"] == "LLM_SUMMARY_EMPTY_RESPONSE"
    assert result["text"] is None


def test_call_final_summary_llm_ok_with_text(monkeypatch):
    from auto_tune.modules.agent_engine import final_summary as fs
    from auto_tune.modules.agent_engine import decision_agent

    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda p, c: "本轮 lr0 调整有效")
    result = fs.call_final_summary_llm({"x": 1}, {"llm": {"enabled": True}})
    assert result["status"] == LLM_SUMMARY_OK
    assert result["text"] == "本轮 lr0 调整有效"


# ── P1.1-min: atomic write failure boundaries ───────────────────────────────


def test_write_txt_target_is_directory_fails_and_preserves_dir(tmp_path):
    best_dir = tmp_path / "detect" / "autotune_s1_iter01"
    best_dir.mkdir(parents=True)
    target_dir = best_dir / "tuning_final_summary.txt"
    target_dir.mkdir()  # target path is a directory → os.replace fails

    result = write_final_summary_txt(
        {"best_train_name": "autotune_s1_iter01"}, str(best_dir), "内容",
    )
    assert result["status"] == "failed"
    assert result["error_code"] == "FINAL_SUMMARY_WRITE_FAILED"
    assert target_dir.is_dir()  # the pre-existing directory is not destroyed
    assert not list(best_dir.glob("*.tmp"))
    assert not list(best_dir.glob(".tuning_final_summary.*"))


def test_write_txt_dir_unwritable(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file, not a directory", encoding="utf-8")
    result = write_final_summary_txt({"best_train_name": "x"}, str(blocker), "text")
    assert result["status"] == "failed"
    assert result["error_code"] == "FINAL_SUMMARY_WRITE_FAILED"


def test_write_txt_replace_failure_preserves_old_file(tmp_path):
    """A failed os.replace must never destroy a pre-existing TXT file."""
    best_dir = tmp_path / "detect" / "autotune_s1_iter01"
    best_dir.mkdir(parents=True)
    target = best_dir / "tuning_final_summary.txt"
    target.write_text("OLD CONTENT", encoding="utf-8")
    os.chmod(target, 0o444)  # read-only → os.replace fails on Windows
    try:
        result = write_final_summary_txt(
            {"best_train_name": "autotune_s1_iter01"}, str(best_dir), "NEW CONTENT",
        )
        assert result["status"] == "failed"
        assert result["error_code"] == "FINAL_SUMMARY_WRITE_FAILED"
        # old content preserved; no temp residue
        assert target.read_text(encoding="utf-8") == "OLD CONTENT"
        assert not list(best_dir.glob(".tuning_final_summary.*"))
        assert not list(best_dir.glob("*.tmp"))
    finally:
        os.chmod(target, 0o666)


def test_write_txt_repeated_calls_idempotent(tmp_path):
    """Repeated finalization overwrites the same file atomically: no numbered
    copies, no .tmp residue, no appended duplicate content."""
    best_dir = tmp_path / "detect" / "autotune_s1_iter01"
    best_dir.mkdir(parents=True)
    text = "\n".join([
        "# 自动调优终局总结",
        "## 第 1 轮 (autotune_s1_iter01)",
        "## 最佳轮次",
    ])
    r1 = write_final_summary_txt({"best_train_name": "autotune_s1_iter01"}, str(best_dir), text)
    r2 = write_final_summary_txt({"best_train_name": "autotune_s1_iter01"}, str(best_dir), text)

    assert r1["status"] == "generated"
    assert r2["status"] == "generated"
    assert r1["path"] == r2["path"]
    files = [p.name for p in best_dir.iterdir()]
    assert files == ["tuning_final_summary.txt"]
    content = (best_dir / "tuning_final_summary.txt").read_text(encoding="utf-8")
    assert content.count("## 第 1 轮") == 1  # not appended twice
    assert content == text
