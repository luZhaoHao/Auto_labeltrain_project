"""H1.2 可复现验收入口：调用公开 HpoService/HpoRunner 跑一次真实 HPO。

只在显式执行时启动真实训练；参数齐全且本地快照/权重存在才运行。失败非零退出，
不下载权重/数据、不依赖 LLM 凭据，报告不含凭据。普通自动化测试用 fake 适配器
验证 CLI→公共服务映射，不在此启动真实 YOLO。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from auto_tune.modules.hpo import (
    ExecutionConfig,
    HpoError,
    HpoRunner,
    HpoService,
    StudyConfig,
    rank_trials,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_hpo_execution",
        description="Run one fixed-budget HPO execution (real training).",
    )
    parser.add_argument("--snapshot-dir", required=True,
                        help="published dataset snapshot directory")
    parser.add_argument("--model-path", required=True,
                        help="local YOLOv8 Detect .pt weight file")
    parser.add_argument("--storage-root", default=str(Path("runs") / "hpo"))
    parser.add_argument("--output-root", default=str(Path("runs") / "hpo-out"))
    parser.add_argument("--log-root", default=str(Path("log")))
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser


def _summary(study, ranked, storage_root: str) -> str:
    terminal = [t for t in study.trials if t.state != "PENDING"]
    best = ""
    if ranked:
        top = ranked[0]
        best = (f"best trial={top.number} value={top.result.value} "
                f"epoch={top.result.evidence.epoch}")
    else:
        best = "no valid objective result"
    return (
        f"study_id={study.study_id}\n"
        f"terminal_trials={len(terminal)}/{study.config.budget}\n"
        f"sampler={study.config.sampler} seed={study.config.seed} "
        f"epochs={study.config.epochs}\n"
        f"{best}\n"
        f"execution_audit={storage_root}/{study.study_id}/execution.json"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = Path(args.snapshot_dir)
    model = Path(args.model_path)
    if not snapshot.is_dir():
        print(f"snapshot directory not found: {snapshot}", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"model file not found: {model}", file=sys.stderr)
        return 2
    try:
        config = StudyConfig(sampler=args.sampler, budget=args.budget,
                             epochs=args.epochs, seed=args.seed)
        service = HpoService(Path(args.storage_root))
        study = service.create_study(config, snapshot_dir=snapshot,
                                     model_path=model)
        runner = HpoRunner(Path(args.storage_root), Path(args.output_root),
                           Path(args.log_root))
        runner.prepare(study.study_id, ExecutionConfig(
            batch=args.batch, imgsz=args.imgsz, device=args.device,
            timeout_seconds=args.timeout_seconds))
        record = runner.run(study.study_id)
        final = service.load_study(study.study_id)
        ranked = rank_trials(final)
        failures = _failure_report(record, final, ranked)
        if failures is not None:
            print(failures, file=sys.stderr)
            return 1
        print(_summary(final, ranked, args.storage_root))
        return 0
    except HpoError as exc:
        print(f"HPO error {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1


def _failure_report(record, final, ranked):
    """预算流程 COMPLETED 不等于验收成功：需全部真实验收 trial 均 SUCCESS。"""
    if record.status != "COMPLETED":
        return (f"execution did not complete (status={record.status}, "
                f"stop_reason={record.stop_reason})")
    if not final.trials:
        return "budget ran no trials"
    states = {}
    for trial in final.trials:
        states[trial.state] = states.get(trial.state, 0) + 1
    if any(t.state != "SUCCESS" for t in final.trials):
        return (f"not all trials succeeded: {states}; "
                f"terminal={len(final.trials)}/{final.config.budget}")
    if not ranked:
        return "no valid objective result despite success trials"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
