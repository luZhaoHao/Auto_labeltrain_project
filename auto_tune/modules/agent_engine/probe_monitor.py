"""Probe Monitor — early-training monitoring and go/no-go decision.

Monitors the first N epochs of training, parses results.csv trends,
and decides whether to continue, abort, or retry with different params.
"""

import time
import logging
from typing import Callable
from .executor import TrainingProcess

logger = logging.getLogger(__name__)


class ProbeDecision:
    """Result of a probe monitoring cycle."""

    CONTINUE = "continue"
    ABORT = "abort"
    RETRY = "retry"

    def __init__(self, verdict: str, reason: str = "", suggestion: str = ""):
        self.verdict = verdict
        self.reason = reason
        self.suggestion = suggestion

    def __repr__(self):
        return f"ProbeDecision({self.verdict}: {self.reason})"


def check_early_metrics(
    metrics_history: list[dict],
    current_epoch: int,
    config: dict,
) -> ProbeDecision:
    """Analyze early training metrics and decide on course of action.

    Args:
        metrics_history: list of per-epoch metric dicts from results.csv.
        current_epoch: current epoch number.
        config: probe config from config.yaml (probe section).

    Returns:
        ProbeDecision with verdict.
    """
    if len(metrics_history) < 3:
        # Need at least 3 data points for trend analysis
        return ProbeDecision(ProbeDecision.CONTINUE, "初期数据不足，继续监控")

    probe_cfg = config.get("probe", {})
    probe_epochs = probe_cfg.get("probe_epochs", 10)
    auto_continue_threshold = probe_cfg.get("auto_continue_threshold_mAP50", 0.05)

    # Check if we've reached the probe epoch limit
    if current_epoch >= probe_epochs:
        return _decide_at_probe_end(metrics_history, auto_continue_threshold)

    # ── Early abort conditions ──

    # 1. Loss explosion (NaN or inf)
    for m in metrics_history[-3:]:
        for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
            val = m.get(key)
            if val is not None and (val != val or val > 1e5):  # NaN check via self-comparison
                return ProbeDecision(
                    ProbeDecision.ABORT,
                    f"Loss 异常: {key}={val}，梯度可能爆炸",
                    "降低 lr0 或使用 AdamW 优化器"
                )

    # 2. Check for convergence to zero loss (possible dead model)
    recent_box_losses = [
        m.get("train/box_loss", 1) for m in metrics_history[-3:]
    ]
    if all(v == 0 for v in recent_box_losses):
        return ProbeDecision(
            ProbeDecision.ABORT,
            "box_loss 持续为 0，模型可能未正常训练",
            "检查数据路径和模型加载是否正确"
        )

    # 3. Stagnant loss (no improvement in 5 epochs)
    if len(metrics_history) >= 6:
        first_loss = metrics_history[-6].get("train/box_loss", 0)
        last_loss = metrics_history[-1].get("train/box_loss", 1)
        if first_loss != 0 and abs(last_loss - first_loss) / first_loss < 0.01:
            return ProbeDecision(
                ProbeDecision.ABORT,
                "box_loss 停滞超过 5 个 epoch，模型未有效学习",
                "增大 lr0 或检查数据标注"
            )

    return ProbeDecision(ProbeDecision.CONTINUE, f"Epoch {current_epoch}/{probe_epochs} 正常")


def _decide_at_probe_end(
    metrics_history: list[dict],
    auto_continue_threshold: float,
) -> ProbeDecision:
    """Decision at the end of probe phase (N epochs completed).

    Args:
        metrics_history: full metric history.
        auto_continue_threshold: minimum mAP50 to auto-continue.

    Returns:
        ProbeDecision.
    """
    latest = metrics_history[-1] if metrics_history else {}
    mAP50 = latest.get("metrics/mAP50(B)", 0)

    if mAP50 >= auto_continue_threshold:
        return ProbeDecision(
            ProbeDecision.CONTINUE,
            f"探针期结束: mAP50={mAP50:.4f} >= {auto_continue_threshold}，进入完整训练"
        )

    # Check if metrics are improving
    if len(metrics_history) >= 4:
        first_map = metrics_history[-4].get("metrics/mAP50(B)", 0)
        last_map = metrics_history[-1].get("metrics/mAP50(B)", 0)
        if last_map > first_map:
            return ProbeDecision(
                ProbeDecision.CONTINUE,
                f"mAP50 仍在上升 ({first_map:.4f} -> {last_map:.4f})，继续训练"
            )

    return ProbeDecision(
        ProbeDecision.RETRY,
        f"探针期结束: mAP50={mAP50:.4f} < {auto_continue_threshold} 且无上升趋势",
        "建议调整学习率或数据增强参数后重试"
    )


def monitor_training(
    train_proc: TrainingProcess,
    config: dict,
    probe_epochs: int | None = None,
    poll_interval: float = 10.0,
    on_epoch_callback: Callable | None = None,
    process_start_time: float | None = None,
    cancel_event=None,
) -> ProbeDecision:
    """Monitor training during probe phase and return final decision.

    Args:
        train_proc: running TrainingProcess.
        config: full config dict.
        probe_epochs: number of epochs for probe (default from config).
        poll_interval: seconds between polls.
        on_epoch_callback: optional callback(epoch, metrics) for UI updates.

    Returns:
        ProbeDecision.
    """
    if probe_epochs is None:
        probe_cfg = config.get("probe", {})
        probe_epochs = probe_cfg.get("probe_epochs", 10)

    metrics_history: list[dict] = []
    last_epoch = -1

    # ── Detect stale results.csv from previous training ──
    # If results.csv already exists and hasn't been modified since the current
    # training process started, its data is stale and must be ignored.
    import os as _os
    csv_path = _os.path.join(train_proc.train_dir, "results.csv")
    stale_threshold = process_start_time or (train_proc.start_time if process_start_time is None else 0)
    if _os.path.exists(csv_path) and _os.path.getmtime(csv_path) < stale_threshold:
        logger.info(f"[Probe] Ignoring stale results.csv (mtime={_os.path.getmtime(csv_path):.0f} < start={stale_threshold:.0f})")
        # Read the last epoch so we know what's "old" and wait for newer data
        stale_metrics = train_proc.read_results_csv()
        if stale_metrics:
            last_epoch = int(stale_metrics.get("epoch", -1))
            logger.info(f"[Probe] Stale results.csv has epoch {last_epoch}, waiting for fresh data...")

    while True:
        # Check cancellation
        if cancel_event and cancel_event.is_set():
            train_proc.terminate()
            return ProbeDecision(
                ProbeDecision.ABORT,
                "用户取消训练",
                ""
            )

        status = train_proc.poll()
        metrics = train_proc.read_results_csv()

        if metrics:
            epoch = int(metrics.get("epoch", 0))
            if epoch > last_epoch:
                metrics_history.append(metrics)
                last_epoch = epoch
                if on_epoch_callback:
                    on_epoch_callback(epoch, metrics)
                logger.info(f"Epoch {epoch}: mAP50={metrics.get('metrics/mAP50(B)', '?')}")

        # Check if we need to make a decision
        if last_epoch >= probe_epochs or status in ("completed", "failed"):
            if status == "failed" and len(metrics_history) == 0:
                return ProbeDecision(
                    ProbeDecision.ABORT,
                    "训练进程启动后立即失败，未产生任何 epoch 数据",
                    "检查 YOLO 环境、GPU 可用性、或参数配置"
                )
            decision = check_early_metrics(metrics_history, last_epoch, config)
            if status == "failed" and decision.verdict == ProbeDecision.CONTINUE:
                decision = ProbeDecision(
                    ProbeDecision.ABORT,
                    f"训练进程在 epoch {last_epoch} 异常退出",
                    "检查 YOLO 日志排查崩溃原因"
                )
            if decision.verdict == ProbeDecision.CONTINUE and status == "completed":
                pass  # Training finished
            return decision

        # Check early abort conditions (before reaching probe_epochs)
        if last_epoch >= 3:
            decision = check_early_metrics(metrics_history, last_epoch, config)
            if decision.verdict != ProbeDecision.CONTINUE:
                train_proc.terminate()
                logger.warning(f"Training {train_proc.train_name}: {decision}")
                return decision

        time.sleep(poll_interval)

    # Default: continue
    return ProbeDecision(ProbeDecision.CONTINUE, "探针监控结束")
