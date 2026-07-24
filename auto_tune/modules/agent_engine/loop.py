"""Auto-Tuning Loop orchestrator — Perception → Decision → Guardrails → Execute → Monitor.

Orchestrates the full closed-loop hyperparameter optimization cycle:
1. Perception: aggregate Module A + Module B outputs
2. Decision: LLM suggests hyperparameter changes
3. Guardrails: validate and clamp changes
4. Execute: launch training with new params
5. Probe: monitor early epochs, decide to continue/abort/retry
"""

import json
import os
import time
import logging
from typing import Any

from .perception import build_perception, find_module_b_report
from .decision_agent import decide_hyperparameters
from .guardrails import validate_and_clamp, merge_params
from .executor import find_detect_dir, read_args_yaml, prepare_training, launch_training, TrainingProcess
from .probe_monitor import monitor_training, ProbeDecision

logger = logging.getLogger(__name__)


class TuningHistory:
    """Records the history of auto-tuning attempts."""

    def __init__(self):
        self.attempts: list[dict] = []

    def add_attempt(self, attempt: dict):
        self.attempts.append(attempt)

    def get_previous_changes(self) -> list[dict]:
        """Return previous attempts formatted for LLM context."""
        return [
            {
                "changes": a.get("hyperparameter_changes", {}),
                "result": a.get("result", "unknown"),
                "diagnosis": a.get("diagnosis", ""),
            }
            for a in self.attempts
        ]

    def to_dict(self) -> list[dict]:
        return self.attempts

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.attempts, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TuningHistory":
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                h = cls()
                h.attempts = data
                return h
        return cls()


class TuningResult:
    """Result of a single tuning loop iteration."""

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.perception: dict = {}
        self.decision: dict = {}
        self.guard_result: Any = None
        self.merged_params: dict = {}
        self.train_name: str | None = None
        self.probe_decision: ProbeDecision | None = None
        self.error: str | None = None
        # Metrics from Module B analysis
        self.result_mAP50: float | None = None
        self.result_mAP50_95: float | None = None
        self.result_precision: float | None = None
        self.result_recall: float | None = None
        self.result_best_epoch: int | None = None

    def get_composite_score(self, mode: str = "comprehensive") -> float:
        """Compute composite score based on evaluation mode.

        - quick (双指标): mAP50×0.6 + mAP50-95×0.4
        - comprehensive (四维): mAP50×0.35 + mAP50-95×0.25 + precision×0.20 + recall×0.20
        """
        if mode == "quick":
            if self.result_mAP50 is not None and self.result_mAP50_95 is not None:
                return self.result_mAP50 * 0.6 + self.result_mAP50_95 * 0.4
            if self.result_mAP50 is not None:
                return self.result_mAP50
            return 0.0
        # comprehensive (default)
        scores = [self.result_mAP50, self.result_mAP50_95, self.result_precision, self.result_recall]
        weights = [0.35, 0.25, 0.20, 0.20]
        has_all = all(s is not None for s in scores)
        if has_all:
            return sum(s * w for s, w in zip(scores, weights))  # type: ignore
        # Fallback to quick
        return self.get_composite_score(mode="quick")

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "perception": {
                "dataset_total_images": self.perception.get("dataset", {}).get("total_images", 0),
                "training_best_mAP50": self.perception.get("training", {}).get("best_mAP50", 0),
            },
            "decision": {
                "diagnosis": self.decision.get("diagnosis", ""),
                "action": self.decision.get("action", ""),
                "hyperparameter_changes": self.decision.get("hyperparameter_changes", {}),
                "training_overrides": self.decision.get("training_overrides", {}),
            },
            "guard_results": {
                "valid": getattr(self.guard_result, "valid", True),
                "warnings": getattr(self.guard_result, "warnings", []),
                "errors": getattr(self.guard_result, "errors", []),
                "clamped": getattr(self.guard_result, "clamped", {}),
            },
            "merged_params": {k: v for k, v in self.merged_params.items()
                              if not k.startswith("_")},
            "train_name": self.train_name,
            "probe_decision": {
                "verdict": self.probe_decision.verdict if self.probe_decision else None,
                "reason": self.probe_decision.reason if self.probe_decision else "",
                "suggestion": self.probe_decision.suggestion if self.probe_decision else "",
            },
            "result_mAP50": self.result_mAP50,
            "result_mAP50_95": self.result_mAP50_95,
            "result_precision": self.result_precision,
            "result_recall": self.result_recall,
            "result_best_epoch": self.result_best_epoch,
            "error": self.error,
        }


def _compute_best(tuning_result: dict, eval_mode: str = "comprehensive"):
    """Compute best iteration from all completed iterations in tuning_result."""
    successful = [
        it for it in tuning_result.get("iterations", [])
        if not it.get("error") and it.get("train_name") and it["train_name"] != "dry_run"
        and it.get("result_mAP50") is not None
    ]
    if not successful:
        return

    def _composite_score(it: dict) -> float:
        if eval_mode == "quick":
            m1 = it.get("result_mAP50") or 0
            m2 = it.get("result_mAP50_95") or 0
            return m1 * 0.6 + m2 * 0.4
        m1 = it.get("result_mAP50") or 0
        m2 = it.get("result_mAP50_95") or 0
        p = it.get("result_precision") or 0
        r = it.get("result_recall") or 0
        return m1 * 0.35 + m2 * 0.25 + p * 0.20 + r * 0.20

    best_it = max(successful, key=_composite_score)
    tuning_result["best_iteration"] = best_it.get("iteration")
    tuning_result["best_train_name"] = best_it.get("train_name")
    tuning_result["best_metrics"] = {
        "mAP50": best_it.get("result_mAP50"),
        "mAP50_95": best_it.get("result_mAP50_95"),
        "precision": best_it.get("result_precision"),
        "recall": best_it.get("result_recall"),
    }
    if not tuning_result.get("final_result"):
        tuning_result["final_result"] = {
            "train_name": best_it.get("train_name"),
            "iteration": best_it.get("iteration"),
            "decision": (best_it.get("decision") or {}).get("diagnosis"),
            "changes": (best_it.get("decision") or {}).get("hyperparameter_changes", {}),
        }


def run_tuning_loop(
    config: dict,
    reference_run: str | None = None,
    max_retries: int | None = None,
    log_dir: str = "log",
    skip_execute: bool = False,
    auto_analyze: bool = False,
    auto_loop: bool = False,
    on_progress: callable = None,
    cancel_event = None,
    keep_params: bool = False,
    eval_mode: str = "comprehensive",
) -> dict:
    """Run the complete auto-tuning loop.

    This is the main entry point for Module C.

    Args:
        config: full config dict (from config.yaml).
        reference_run: specific training to use as reference (e.g., "train8").
                       If None, uses the latest from Module B report.
        max_retries: max tuning attempts (default from config).
        log_dir: directory for log output.
        skip_execute: if True, stop at merged_params (for testing).
        on_progress: optional callback(iteration, message) for UI updates.

    Returns:
        Dict with final results including all iteration details.
    """
    logger.info(f"[AutoTune] run_tuning_loop called: skip_execute={skip_execute}, auto_analyze={auto_analyze}, auto_loop={auto_loop}, max_retries={max_retries}, reference_run={reference_run}")

    if max_retries is None:
        max_retries = config.get("probe", {}).get("max_retries", 3)

    if reference_run is None:
        # Auto-detect from latest Module B report
        report = find_module_b_report(log_dir=log_dir)
        if report and report.get("runs"):
            reference_run = list(report["runs"].keys())[0]

    history = TuningHistory()
    detect_dir = find_detect_dir()
    tuning_session_id = str(int(time.time() * 1000))  # unique per tuning session

    tuning_result = {
        "module": "agent_engine",
        "version": "1.0",
        "detect_dir": detect_dir,
        "reference_run": reference_run,
        "max_retries": max_retries,
        "iterations": [],
        "final_result": None,
        "error": None,
        "eval_mode": eval_mode,
    }

    for iteration in range(1, max_retries + 1):
        # Check cancellation before each iteration
        if cancel_event and cancel_event.is_set():
            logger.info("[AutoTune] Cancelled before iteration %d", iteration)
            tuning_result["error"] = "用户取消"
            history.to_json(os.path.join(log_dir, "tuning_history.json"))
            return tuning_result

        iter_result = TuningResult(iteration)
        if on_progress:
            on_progress(iteration, f"迭代 {iteration}/{max_retries} 开始")

        try:
            # ── Step 1: Perception ──
            if on_progress:
                on_progress(iteration, "感知层：聚合数据集和训练分析数据", step="perception")
            perception = build_perception(log_dir=log_dir)
            iter_result.perception = perception

            # ── Step 2: Decision ──
            if keep_params:
                if on_progress:
                    on_progress(iteration, "决策层：保持原有参数，跳过 LLM 调参", step="decision", params={"_keep_params": True, "reason": "AI 判断无需调整超参数，使用原有参数继续训练"})
                decision = {
                    "diagnosis": "按原有参数训练，不做超参数调整",
                    "action": "keep_params",
                    "hyperparameter_changes": {},
                    "training_overrides": {},
                }
            else:
                if on_progress:
                    on_progress(iteration, "决策层：LLM 分析并输出调参建议", step="decision")
                prev_changes = history.get_previous_changes()
                decision = decide_hyperparameters(perception, config, prev_changes)
                combined_params = {}
                if decision.get("hyperparameter_changes"):
                    combined_params.update(decision["hyperparameter_changes"])
                if decision.get("training_overrides"):
                    combined_params["_overrides"] = decision["training_overrides"]
                if on_progress:
                    if combined_params:
                        on_progress(iteration, f"参数变更: {json.dumps(combined_params, ensure_ascii=False)}",
                                    step="decision", params=combined_params)
                    else:
                        on_progress(iteration, "决策层：AI 判断当前超参数无需调整，沿用原有参数",
                                    step="decision", params={"_keep_params": True, "reason": "AI 判断无需调整超参数"})
            iter_result.decision = decision

            if decision.get("error"):
                iter_result.error = f"决策失败: {decision['error']}"
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                if on_progress:
                    on_progress(iteration, f"决策失败: {decision['error']}")
                continue

            # ── Step 3: Guardrails ──
            if on_progress:
                on_progress(iteration, "安全护栏：校验和约束参数", step="guardrails")
            changes = decision.get("hyperparameter_changes", {})
            overrides = decision.get("training_overrides", {})
            dataset_info = perception.get("dataset", {})

            guard_result = validate_and_clamp(changes, dataset_info)
            iter_result.guard_result = guard_result

            if not guard_result.valid:
                iter_result.error = f"护栏拦截: {'; '.join(guard_result.errors)}"
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                if on_progress:
                    on_progress(iteration, f"护栏拦截: {iter_result.error}")
                continue

            # ── Merge params ──
            if reference_run:
                ref_dir = os.path.join(detect_dir, reference_run)
                if os.path.isdir(ref_dir):
                    base_args = read_args_yaml(ref_dir)
                else:
                    base_args = {}
            else:
                base_args = {}

            all_changes = dict(changes)
            all_changes.update(overrides)
            merged = merge_params(base_args, all_changes)
            iter_result.merged_params = merged

            # Store old batch for guardrail scaling check
            merged["_old_batch"] = base_args.get("batch", 16)

            # Re-validate with merged params
            guard_result2 = validate_and_clamp(merged, dataset_info)
            if guard_result2.warnings:
                for w in guard_result2.warnings:
                    logger.warning(f"[Guardrail] {w}")

            if skip_execute:
                iter_result.train_name = "dry_run"
                iter_result.probe_decision = ProbeDecision(ProbeDecision.CONTINUE, "跳过执行")
                tuning_result["iterations"].append(iter_result.to_dict())
                tuning_result["final_result"] = {
                    "train_name": "dry_run",
                    "iteration": iteration,
                    "decision": decision.get("diagnosis"),
                    "changes": changes,
                    "guard_warnings": guard_result.warnings,
                }
                if on_progress:
                    on_progress(iteration, "跳过执行（dry-run 模式）")
                continue

            # ── Step 4: Execute ──
            if on_progress:
                on_progress(iteration, f"执行层：启动训练 {merged.get('model', 'yolov8')}", step="execute")
            train_name = f"autotune_{iteration}_{reference_run or 'latest'}_{tuning_session_id}"
            output_dir = os.path.join(detect_dir, train_name)
            os.makedirs(output_dir, exist_ok=False)  # fresh directory — must not exist

            # Write merged config
            from .executor import write_training_config
            args_path = write_training_config(base_args, merged, output_dir)
            iter_result.train_name = train_name

            # Launch training
            proc = launch_training(train_name, args_path, merged)
            train_proc = TrainingProcess(train_name, output_dir, proc)
            training_start_time = time.time()

            # ── Step 5: Probe Monitor ──
            if on_progress:
                on_progress(iteration, "探针监控：监测前 N 个 epoch 的训练趋势", step="probe")

            def epoch_cb(epoch, metrics):
                if on_progress:
                    mAP = metrics.get("metrics/mAP50(B)", "?")
                    on_progress(iteration, f"Epoch {epoch}: mAP50={mAP}")

            probe_decision = monitor_training(
                train_proc, config, on_epoch_callback=epoch_cb,
                process_start_time=training_start_time,
                cancel_event=cancel_event,
            )
            iter_result.probe_decision = probe_decision

            if probe_decision.verdict == ProbeDecision.ABORT:
                iter_result.error = f"训练中止: {probe_decision.reason}"
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                if on_progress:
                    on_progress(iteration, f"训练中止: {probe_decision.reason}")
                continue
            elif probe_decision.verdict == ProbeDecision.RETRY:
                iter_result.error = f"需要重试: {probe_decision.reason}"
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                if on_progress:
                    on_progress(iteration, f"需要重试: {probe_decision.reason}")
                continue

            # ── Probe passed: mark pipeline complete (green) ──
            if on_progress:
                on_progress(iteration, "✅ 探针通过，继续训练", step="complete", level="success")
            logger.info(f"[AutoTune] Probe passed. auto_analyze={auto_analyze}, auto_loop={auto_loop}")
            if auto_analyze:
                # ── Wait for full training to complete ──
                if on_progress:
                    on_progress(iteration, f"探针通过，等待完整训练完成...")

                # Start from where probe left off so we report every epoch from here
                last_reported_epoch = train_proc.current_epoch
                while True:
                    # Check cancellation
                    if cancel_event and cancel_event.is_set():
                        train_proc.terminate()
                        logger.info("[AutoTune] Cancelled during training wait")
                        iter_result.error = "用户取消训练"
                        history.add_attempt(iter_result.to_dict())
                        tuning_result["iterations"].append(iter_result.to_dict())
                        tuning_result["error"] = "用户取消"
                        history.to_json(os.path.join(log_dir, "tuning_history.json"))
                        return tuning_result

                    status = train_proc.poll()
                    if status in ("completed", "failed"):
                        break
                    # Report progress every epoch so SSE stays alive and
                    # the user can see training is still progressing.
                    metrics = train_proc.read_results_csv()
                    if metrics:
                        epoch = int(metrics.get("epoch", 0))
                        if epoch > last_reported_epoch:
                            last_reported_epoch = epoch
                            mAP = metrics.get("metrics/mAP50(B)", "?")
                            if on_progress:
                                on_progress(iteration, f"训练进度: Epoch {epoch}/{merged.get('epochs', '?')}, mAP50={mAP}")
                    time.sleep(10)

                if status == "failed":
                    iter_result.error = "训练进程异常退出"
                    history.add_attempt(iter_result.to_dict())
                    tuning_result["iterations"].append(iter_result.to_dict())
                    if on_progress:
                        on_progress(iteration, f"训练进程异常退出")
                    if auto_loop:
                        continue
                    tuning_result["error"] = iter_result.error
                    history.to_json(os.path.join(log_dir, "tuning_history.json"))
                    return tuning_result

                # ── Run Module B analysis ──
                if on_progress:
                    on_progress(iteration, "训练完成，开始 Module B 分析...")
                try:
                    from auto_tune.modules.train_analyzer.results_parser import load_training_run
                    from auto_tune.modules.train_analyzer.curve_analysis import (
                        analyze_loss_curves, analyze_metric_curves, detect_early_stopping
                    )
                    from auto_tune.modules.train_analyzer.issue_detector import detect_issues
                    from auto_tune.modules.train_analyzer.run_comparator import compare_runs, summarize_runs

                    run_data = load_training_run(output_dir)
                    run_data["name"] = train_name

                    ta_config = config.get("train_analyzer", {})
                    curve_analysis = analyze_loss_curves(run_data["results"], ta_config)
                    metric_analysis = analyze_metric_curves(run_data["results"], ta_config)
                    early_stop = detect_early_stopping(run_data, ta_config)
                    curve_analysis["early_stopping"] = early_stop
                    issues = detect_issues(run_data, ta_config)

                    run_data["curve_analysis"] = curve_analysis
                    run_data["metric_analysis"] = metric_analysis
                    run_data["issues"] = issues

                    report = {
                        "module": "train_analyzer",
                        "version": "1.0",
                        "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "detect_dir": output_dir,
                        "project": config.get("project", {}),
                        "total_runs": 1,
                        "runs": {train_name: run_data},
                        "comparison": compare_runs([run_data], ta_config),
                        "summary": summarize_runs([run_data], ta_config),
                    }

                    # Extract metrics from Module B results
                    final_metrics = run_data.get("results", {}).get("final_metrics", {})
                    iter_result.result_mAP50 = final_metrics.get("metrics/mAP50(B)")
                    iter_result.result_mAP50_95 = final_metrics.get("metrics/mAP50-95(B)")
                    iter_result.result_precision = final_metrics.get("metrics/precision(B)")
                    iter_result.result_recall = final_metrics.get("metrics/recall(B)")
                    iter_result.result_best_epoch = run_data.get("results", {}).get("best_epoch")

                    report_path = os.path.join(log_dir, f"{train_name}_report.json")
                    os.makedirs(log_dir, exist_ok=True)
                    with open(report_path, "w", encoding="utf-8") as f:
                        json.dump(report, f, ensure_ascii=False, indent=2)

                    if on_progress:
                        cs = iter_result.get_composite_score(eval_mode)
                        on_progress(iteration, f"Module B 分析完成: mAP50={iter_result.result_mAP50}, mAP50-95={iter_result.result_mAP50_95}, Precision={iter_result.result_precision}, Recall={iter_result.result_recall}, 综合分={cs:.4f}")
                except Exception as e:
                    logger.exception(f"Module B analysis failed for {train_name}")
                    if on_progress:
                        on_progress(iteration, f"Module B 分析失败: {e}")

                # ── Auto-loop: continue to next iteration ──
                if auto_loop:
                    history.add_attempt(iter_result.to_dict())
                    tuning_result["iterations"].append(iter_result.to_dict())
                    reference_run = train_name
                    if on_progress:
                        on_progress(iteration, f"自动循环 → 下一轮 (新参考: {train_name})")
                    continue

                # ── auto_analyze only: return success ──
                history.add_attempt(iter_result.to_dict())
                tuning_result["iterations"].append(iter_result.to_dict())
                tuning_result["final_result"] = {
                    "train_name": train_name,
                    "iteration": iteration,
                    "decision": decision.get("diagnosis"),
                    "changes": changes,
                    "guard_warnings": guard_result.warnings,
                    "module_b_analyzed": True,
                }
                if on_progress:
                    on_progress(iteration, f"✅ 训练 {train_name} 完成，Module B 分析已生成")
                # ── Compute best iteration from all completed iterations ──
                _compute_best(tuning_result, eval_mode)
                history.to_json(os.path.join(log_dir, "tuning_history.json"))
                return tuning_result

            # ── No auto-analyze: training continues in background ──
            history.add_attempt(iter_result.to_dict())
            tuning_result["iterations"].append(iter_result.to_dict())
            tuning_result["final_result"] = {
                "train_name": train_name,
                "iteration": iteration,
                "decision": decision.get("diagnosis"),
                "changes": changes,
                "guard_warnings": guard_result.warnings,
            }
            if on_progress:
                on_progress(iteration, f"✅ 训练 {train_name} 已通过探针期，继续进行完整训练")

            # ── Compute best iteration from all completed iterations ──
            _compute_best(tuning_result, eval_mode)
            # Save history
            history.to_json(os.path.join(log_dir, "tuning_history.json"))
            return tuning_result

        except Exception as e:
            iter_result.error = f"异常: {str(e)}"
            tuning_result["iterations"].append(iter_result.to_dict())
            logger.exception(f"Tuning iteration {iteration} failed")
            if on_progress:
                on_progress(iteration, f"❌ 异常: {str(e)}")

    # All retries exhausted — check if we have any successful iterations
    _compute_best(tuning_result, eval_mode)
    if tuning_result.get("best_iteration") is not None and on_progress:
        best_m = tuning_result.get("best_metrics", {})
        on_progress(
            tuning_result["best_iteration"],
            f"🏆 最佳迭代: 第 {tuning_result['best_iteration']} 次 ({tuning_result['best_train_name']}) — "
            f"mAP50={best_m.get('mAP50')}, "
            f"mAP50-95={best_m.get('mAP50_95')}, "
            f"Precision={best_m.get('precision')}, "
            f"Recall={best_m.get('recall')}",
        )
    else:
        tuning_result["error"] = f"已达最大重试次数 ({max_retries})，所有尝试均未成功"

    # Save history
    history.to_json(os.path.join(log_dir, "tuning_history.json"))
    return tuning_result
