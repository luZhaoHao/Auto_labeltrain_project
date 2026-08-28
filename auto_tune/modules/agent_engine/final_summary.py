"""Deterministic final tuning summary + optional LLM closing explanation.

After all tuning iterations complete, this module builds the deterministic
final facts (session, reference run, per-round params/metrics, best round),
renders a plain-text report, optionally calls the LLM once for a closing
explanation, and atomically writes ``tuning_final_summary.txt`` into the best
tuning run directory.

Failure boundaries
------------------
- LLM/API/network failures never change the ``completed`` training fact.
- The LLM is called at most once and is never retried.
- The deterministic TXT is always attempted, with or without the LLM text.
- Only relative names and metric/param facts enter the LLM prompt and the TXT:
  never API keys, full commands, dataset absolute paths, full training logs,
  weight contents, or audit file contents.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone

LLM_SUMMARY_SKIPPED = "skipped"
LLM_SUMMARY_OK = "ok"
LLM_SUMMARY_FAILED = "failed"

SUMMARY_PERSISTENCE_GENERATED = "generated"
SUMMARY_PERSISTENCE_FAILED = "failed"
SUMMARY_PERSISTENCE_SKIPPED = "skipped"

LLM_SUMMARY_EMPTY_RESPONSE = "LLM_SUMMARY_EMPTY_RESPONSE"

FINAL_SUMMARY_FILENAME = "tuning_final_summary.txt"

# Stable LLM closing-summary error codes (never raw provider text/tracebacks).
_LLM_ERROR_CODE_MAP = (
    ("credential_missing", "LLM_SUMMARY_CREDENTIAL_MISSING"),
    ("endpoint_rejected", "LLM_SUMMARY_ENDPOINT_REJECTED"),
    ("network_failed", "LLM_SUMMARY_NETWORK_FAILED"),
    ("timeout", "LLM_SUMMARY_TIMEOUT"),
    ("incompatible_response", "LLM_SUMMARY_INCOMPATIBLE_RESPONSE"),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def composite_score(iteration: dict, eval_mode: str = "comprehensive") -> float:
    """Deterministic composite score matching the loop's best-iteration formula.

    - quick: mAP50 × 0.6 + mAP50-95 × 0.4
    - comprehensive: mAP50 × 0.35 + mAP50-95 × 0.25 + precision × 0.20 + recall × 0.20
    """
    m1 = iteration.get("result_mAP50") or 0
    m2 = iteration.get("result_mAP50_95") or 0
    if eval_mode == "quick":
        return m1 * 0.6 + m2 * 0.4
    p = iteration.get("result_precision") or 0
    r = iteration.get("result_recall") or 0
    return m1 * 0.35 + m2 * 0.25 + p * 0.20 + r * 0.20


def successful_iterations(tuning_result: dict) -> list[dict]:
    """Iterations that actually trained and produced a final mAP50."""
    return [
        it for it in tuning_result.get("iterations", [])
        if not it.get("error") and it.get("train_name") and it["train_name"] != "dry_run"
        and it.get("result_mAP50") is not None
    ]


def _all_completed_iterations(tuning_result: dict) -> list[dict]:
    """Iterations that actually trained (error-free), with or without metrics."""
    return [
        it for it in tuning_result.get("iterations", [])
        if not it.get("error") and it.get("train_name") and it["train_name"] != "dry_run"
    ]


def _safe_param_value(value):
    """Reduce path-like values to their basename so absolute dataset/model
    paths never leak into the final summary text or the LLM prompt."""
    if isinstance(value, str):
        if os.sep in value or "/" in value or "\\" in value:
            return os.path.basename(value.replace("\\", "/"))
    return value


def _round_facts(it: dict, eval_mode: str) -> dict:
    decision = it.get("decision") or {}
    guard = it.get("guard_results") or {}
    suggested = dict(decision.get("hyperparameter_changes") or {})
    suggested.update(decision.get("training_overrides") or {})
    return {
        "iteration": it.get("iteration"),
        "train_name": it.get("train_name"),
        "diagnosis": decision.get("diagnosis"),
        "action": decision.get("action"),
        "suggested_params": {k: _safe_param_value(v) for k, v in suggested.items()},
        "guardrail_params": {k: _safe_param_value(v)
                             for k, v in (guard.get("sanitized_changes") or {}).items()},
        "executed_params": {k: _safe_param_value(v)
                            for k, v in (it.get("merged_params") or {}).items()
                            if not str(k).startswith("_")},
        "mAP50": it.get("result_mAP50"),
        "mAP50_95": it.get("result_mAP50_95"),
        "precision": it.get("result_precision"),
        "recall": it.get("result_recall"),
        "composite_score": composite_score(it, eval_mode),
        "analysis_status": it.get("result_analysis_status"),
    }


def build_deterministic_summary(
    tuning_result: dict,
    eval_mode: str = "comprehensive",
    session_id: str | None = None,
    reference_run: str | None = None,
    generated_at: str | None = None,
) -> dict | None:
    """Build the deterministic final facts.

    Returns None when no round trained or when no round produced a measurable
    result (no best round can be chosen).
    """
    rounds = [_round_facts(it, eval_mode) for it in _all_completed_iterations(tuning_result)]
    if not rounds:
        return None
    successful = successful_iterations(tuning_result)
    if not successful:
        return None
    best = max(successful, key=lambda it: composite_score(it, eval_mode))
    return {
        "tuning_session_id": session_id,
        "reference_run": reference_run,
        "evaluation_mode": eval_mode,
        "total_iterations": len(rounds),
        "iterations": rounds,
        "best_iteration": best.get("iteration") if best else None,
        "best_train_name": best.get("train_name") if best else None,
        "best_metrics": {
            "mAP50": best.get("result_mAP50") if best else None,
            "mAP50_95": best.get("result_mAP50_95") if best else None,
            "precision": best.get("result_precision") if best else None,
            "recall": best.get("result_recall") if best else None,
        },
        "best_score": composite_score(best, eval_mode) if best else None,
        "best_weights_path": "weights/best.pt",
        "generated_at": generated_at or utc_now_iso(),
    }


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_final_summary_text(summary: dict, llm_text: str | None = None) -> str:
    """Render the deterministic facts (and optional LLM text) to plain text."""
    lines = []
    lines.append("# 自动调优终局总结")
    lines.append(f"- 调优会话: {summary.get('tuning_session_id', '—')}")
    lines.append(f"- 参考训练: {summary.get('reference_run', '—')}")
    lines.append(f"- 评估模式: {summary.get('evaluation_mode', '—')}")
    lines.append(f"- 训练轮次: {summary.get('total_iterations', 0)}")
    lines.append("")

    for r in summary.get("iterations", []):
        lines.append(f"## 第 {r.get('iteration')} 轮 ({r.get('train_name')})")
        lines.append(f"- 诊断: {r.get('diagnosis') or '—'}")
        lines.append(f"- 动作: {r.get('action') or '—'}")
        lines.append(f"- 建议参数: {json_safe(r.get('suggested_params'))}")
        lines.append(f"- 护栏后参数: {json_safe(r.get('guardrail_params'))}")
        lines.append(f"- 实际执行参数: {json_safe(r.get('executed_params'))}")
        lines.append(
            f"- 指标: mAP50={_fmt(r.get('mAP50'))}, mAP50-95={_fmt(r.get('mAP50_95'))}, "
            f"Precision={_fmt(r.get('precision'))}, Recall={_fmt(r.get('recall'))}"
        )
        lines.append(f"- 综合评分: {_fmt(r.get('composite_score'))}")
        lines.append(f"- Module B 分析: {r.get('analysis_status') or '—'}")
        lines.append("")

    lines.append("## 最佳轮次")
    lines.append(f"- 最佳轮次: 第 {summary.get('best_iteration')} 轮")
    lines.append(f"- 最佳训练: {summary.get('best_train_name')}")
    best_metrics = summary.get("best_metrics") or {}
    lines.append(
        f"- 最佳指标: mAP50={_fmt(best_metrics.get('mAP50'))}, "
        f"mAP50-95={_fmt(best_metrics.get('mAP50_95'))}, "
        f"Precision={_fmt(best_metrics.get('precision'))}, Recall={_fmt(best_metrics.get('recall'))}"
    )
    lines.append(f"- 综合评分: {_fmt(summary.get('best_score'))}")
    lines.append(f"- 最佳权重: {summary.get('best_weights_path', 'weights/best.pt')}")
    lines.append(f"- 生成时间: {summary.get('generated_at')}")

    if llm_text:
        lines.append("")
        lines.append("## AI 总结")
        lines.append(llm_text)

    return "\n".join(lines)


def json_safe(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _llm_failure_result(exc: Exception) -> dict:
    message = str(exc)
    lower = message.lower()
    code = "LLM_SUMMARY_CALL_FAILED"
    for needle, stable in _LLM_ERROR_CODE_MAP:
        if needle in lower:
            code = stable
            break
    return {"status": LLM_SUMMARY_FAILED, "error_code": code, "text": None}


def build_llm_summary_prompt(summary: dict) -> str:
    """Build the closing-explanation prompt from deterministic facts only.

    Only relative run names, metrics and parameter summaries are included —
    never API keys, full commands, absolute dataset paths, log tails, weights,
    or audit contents.
    """
    lines = []
    lines.append("你是 YOLOv8 超参数调优专家。以下是刚完成的一次自动调优终局事实，请给出简要的调优总结。")
    lines.append("")
    lines.append(f"- 评估模式: {summary.get('evaluation_mode')}")
    lines.append(f"- 参考训练: {summary.get('reference_run')}")
    lines.append(f"- 训练轮次: {summary.get('total_iterations')}")
    lines.append("")
    lines.append("各轮次事实：")
    for r in summary.get("iterations", []):
        lines.append(
            f"- 轮次{r.get('iteration')} ({r.get('train_name')}): "
            f"诊断[{r.get('diagnosis') or '—'}] 动作[{r.get('action') or '—'}] "
            f"建议参数[{json_safe(r.get('suggested_params'))}] "
            f"mAP50={_fmt(r.get('mAP50'))} mAP50-95={_fmt(r.get('mAP50_95'))} "
            f"Precision={_fmt(r.get('precision'))} Recall={_fmt(r.get('recall'))} "
            f"综合分={_fmt(r.get('composite_score'))}"
        )
    best = summary.get("best_iteration")
    lines.append("")
    lines.append(f"最佳轮次: 第 {best} 轮 ({summary.get('best_train_name')})，"
                 f"最佳综合分={_fmt(summary.get('best_score'))}，"
                 f"最佳指标={json_safe(summary.get('best_metrics'))}，"
                 f"权重={summary.get('best_weights_path')}")
    lines.append("")
    lines.append("请用中文回答，说明：")
    lines.append("1. 为什么选中该最佳轮次；")
    lines.append("2. 各轮指标变化趋势；")
    lines.append("3. 哪些参数调整有效或无效；")
    lines.append("4. 推荐保留的模型与参数；")
    lines.append("5. 后续训练建议。")
    lines.append("")
    lines.append("不要输出 JSON，直接给出简洁的文本总结，300 字以内。")
    return "\n".join(lines)


def call_final_summary_llm(summary: dict, config: dict) -> dict:
    """One optional LLM call producing the closing explanation.

    Called at most once and never retried. A single API/parse failure maps to a
    stable error code; it never changes the training completion fact.
    """
    if not (config or {}).get("llm", {}).get("enabled", False):
        return {"status": LLM_SUMMARY_SKIPPED, "error_code": None, "text": None}
    try:
        from .decision_agent import call_decision_llm
        raw = call_decision_llm(build_llm_summary_prompt(summary), config)
    except Exception as exc:
        return _llm_failure_result(exc)
    if not raw or not raw.strip():
        # An empty closing explanation is not usable content; mark it failed so
        # the deterministic TXT still stands on its own.
        return {"status": LLM_SUMMARY_FAILED, "error_code": LLM_SUMMARY_EMPTY_RESPONSE, "text": None}
    return {"status": LLM_SUMMARY_OK, "error_code": None, "text": raw}


def write_final_summary_txt(
    summary: dict,
    best_train_dir: str,
    text: str,
) -> dict:
    """Atomically write ``tuning_final_summary.txt`` into the best run dir.

    Same-directory temp file, flush + fsync, then os.replace. A write failure
    returns a stable non-fatal result and never deletes any existing file.
    """
    target = os.path.join(best_train_dir, FINAL_SUMMARY_FILENAME)
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tuning_final_summary.", suffix=".tmp", dir=best_train_dir, text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return {"status": SUMMARY_PERSISTENCE_GENERATED, "path": target, "error_code": None}
    except OSError:
        return {
            "status": SUMMARY_PERSISTENCE_FAILED,
            "path": target,
            "error_code": "FINAL_SUMMARY_WRITE_FAILED",
        }
