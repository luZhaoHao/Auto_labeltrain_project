"""F1.1-B 受控评估入口 —— 用固定离线案例集回放 LLM 调优决策质量。

用法::

    # 只验证案例集与事实包（不调用任何 LLM）
    python auto_tune/scripts/evaluate_tuning_decisions.py --dry-run

    # 单案例冒烟（1 次真实调用链）
    python auto_tune/scripts/evaluate_tuning_decisions.py --case overfitting --runs 1 --label smoke

    # 修改前基线：8 案例 × 2 次
    python auto_tune/scripts/evaluate_tuning_decisions.py --label baseline --runs 2

    # 只重算指标（不调用 LLM）
    python auto_tune/scripts/evaluate_tuning_decisions.py --report log/eval_f11b/baseline

    # 修改前/后对比
    python auto_tune/scripts/evaluate_tuning_decisions.py --compare \
        log/eval_f11b/baseline log/eval_f11b/rework

退出码：0 全部成功；1 存在失败（provider/contract/semantic 任一）；2 参数或环境错误。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from auto_tune.evaluation import metrics as metrics_mod  # noqa: E402
from auto_tune.evaluation.cases import get_case, get_cases  # noqa: E402
from auto_tune.evaluation.runner import (  # noqa: E402
    OUTCOME_PROVIDER_ERROR,
    OUTCOME_VALID,
    prepare_case,
    run_case,
    write_record,
)

DEFAULT_OUT_ROOT = os.path.join("log", "eval_f11b")

_GATE_LABELS = (
    ("illegal_launch_count", "非法建议启动训练次数"),
    ("unknown_param_pass_count", "未知参数通过次数"),
    ("wrong_evidence_pass_count", "错误事实引用通过次数"),
    ("direction_reversed_pass_count", "方向相反的建议通过次数"),
    ("forced_change_on_keep_case_count", "健康/证据不足场景被强迫改参次数"),
)

# 「格式合法」只说明建议通过了契约与语义校验（逐参数 + 逐引用事实），
# 不说明建议正确。质量指标必须单独看。
_FORMAT_LABELS = (
    ("first_response_valid_rate", "首次响应合法率"),
    ("final_valid_rate", "一次纠错后最终合法率"),
    ("evidence_param_match_rate", "事实与参数匹配率（结构）"),
    ("direction_correct_rate", "参数方向正确率（主因参数）"),
    ("keep_params_correct_rate", "keep_params 判断正确率"),
)

_QUALITY_LABELS = (
    ("main_cause_hit_rate", "主因命中率"),
    ("counter_param_rate", "改用反向/不对症参数率（越低越好）"),
    ("ineligible_evidence_rate", "引用无规则事实作证据率（越低越好）"),
    ("off_scope_evidence_rate", "引用合法但非本案例期望证据率（越低越好）"),
    ("incoherent_combo_rate", "组合自相矛盾率（越低越好）"),
    ("amplitude_within_preference_rate", "幅度落在稳健区间率"),
)

_RATE_LABELS = _FORMAT_LABELS  # 兼容 compare 输出


def _load_config(path: str | None) -> dict:
    import yaml
    config_path = path or os.path.join(_PROJECT_ROOT, "auto_tune", "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _select_cases(case_ids: str | None, tier: str | None = None):
    if not case_ids:
        return get_cases(tier=tier)
    return tuple(get_case(case_id.strip()) for case_id in case_ids.split(",") if case_id.strip())


def _summary_of(record_paths: list[str]) -> dict:
    evaluations = []
    results = []
    for path in record_paths:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        results.append(record)
        evaluations.append(metrics_mod.evaluate_run(record, get_case(record["case_id"])))
    summary = metrics_mod.compute_summary(evaluations)
    summary["cases"] = metrics_mod.case_table(evaluations)
    summary["failure_breakdown"] = metrics_mod.failure_breakdown(evaluations)
    return summary, results


def _print_summary(summary: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"runs={summary['runs']}  retried={summary['retried_runs']}  "
          f"would_start_training={summary['would_start_training_count']}")
    print("-- 硬性安全门槛（必须为 0） --")
    for key, label in _GATE_LABELS:
        value = summary[key]
        print(f"  {'OK ' if value == 0 else 'FAIL'} {label}: {value}")
    print("-- 格式合法性 --")
    for key, label in _FORMAT_LABELS:
        value = summary[key]
        shown = "—" if value is None else f"{value:.2%}"
        print(f"  {label}: {shown}")
    print(f"-- 建议质量（仅统计合法通过的 {summary.get('accepted_runs', 0)} 次） --")
    for key, label in _QUALITY_LABELS:
        value = summary.get(key)
        shown = "—" if value is None else f"{value:.2%}"
        print(f"  {label}: {shown}")
    score = summary.get("advice_quality_score")
    if score is not None:
        print(f"  建议质量综合分（客观合成，非人工矩阵）: {score:.2%}")
    breakdown = summary.get("failure_breakdown") or {}
    if breakdown:
        print("-- 失败类别（稳定错误码） --")
        for code, count in breakdown.items():
            print(f"  {code}: {count}")


def _print_case_table(summary: dict) -> None:
    print("\n-- 逐案例 --")
    header = f"  {'case':32s} {'runs':>4s} {'first':>7s} {'final':>7s} {'dir':>7s} {'keepOK':>7s} {'legal':>6s}"
    print(header)
    for row in summary["cases"]:
        def fmt(value):
            return "—" if value is None else f"{value:.0%}"
        print(f"  {row['case_id']:32s} {row['runs']:>4d} "
              f"{fmt(row['first_response_valid_rate']):>7s} "
              f"{fmt(row['final_valid_rate']):>7s} "
              f"{fmt(row['direction_correct_rate']):>7s} "
              f"{fmt(row['keep_params_correct_rate']):>7s} "
              f"{row['would_start_training_count']:>6d}")


def _print_records(results: list[dict]) -> None:
    print("\n-- 逐次回放 --")
    for record in results:
        decision = record.get("decision") or {}
        changes = {**decision.get("hyperparameter_changes", {}),
                   **decision.get("training_overrides", {})}
        print(f"  [{record['case_id']}#{record['run_index']}] {record['outcome']}"
              f" err={record['error_code']} retried={decision.get('retried')}")
        print(f"      action={decision.get('action')} changes={changes}")
        print(f"      evidence={decision.get('evidence_ids')}")


def _write_assessment(out_dir: str, summary: dict) -> str:
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _compare(before_dir: str, after_dir: str) -> int:
    before, _ = _summary_of(sorted(glob.glob(os.path.join(before_dir, "*__run*.json"))))
    after, _ = _summary_of(sorted(glob.glob(os.path.join(after_dir, "*__run*.json"))))
    _print_summary(before, f"修改前 {before_dir}")
    _print_summary(after, f"修改后 {after_dir}")
    print("\n=== 对比（修改后 - 修改前） ===")
    for group, labels in (("格式合法性", _FORMAT_LABELS), ("建议质量", _QUALITY_LABELS)):
        print(f"-- {group} --")
        for key, label in labels:
            before_value, after_value = before.get(key), after.get(key)
            if before_value is None or after_value is None:
                print(f"  {label}: —")
                continue
            delta = (after_value - before_value) * 100
            print(f"  {label}: {before_value:.2%} → {after_value:.2%}  ({delta:+.2f}pp)")
    before_score, after_score = before.get("advice_quality_score"), after.get("advice_quality_score")
    if before_score is not None and after_score is not None:
        print(f"  建议质量综合分: {before_score:.2%} → {after_score:.2%}  "
              f"({(after_score - before_score) * 100:+.2f}pp)")
    print("\n-- 硬性门槛 --")
    failed = []
    for key, label in _GATE_LABELS:
        value = after[key]
        print(f"  {'OK ' if value == 0 else 'FAIL'} {label}: {value}")
        if value != 0:
            failed.append(label)
    if failed:
        print(f"\n结论：硬性门槛未通过（{', '.join(failed)}）")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1.1-B 调优决策质量受控评估")
    parser.add_argument("--case", help="逗号分隔的案例 id；缺省按 --tier 选择")
    parser.add_argument("--tier", choices=("core", "hard"),
                        help="core=单主因回归案例；hard=多合法动作并存的困难案例")
    parser.add_argument("--runs", type=int, default=2, help="每个案例重复次数（默认 2）")
    parser.add_argument("--label", default="baseline", help="产物子目录名")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="产物根目录")
    parser.add_argument("--config", help="config.yaml 路径")
    parser.add_argument("--dry-run", action="store_true", help="只构建事实包，不调用 LLM")
    parser.add_argument("--report", help="只重算指标，输入既有产物目录")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="对比两个产物目录")
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(args.compare[0], args.compare[1])

    if args.report:
        paths = sorted(glob.glob(os.path.join(args.report, "*__run*.json")))
        if not paths:
            print(f"错误：{args.report} 下没有回放记录", file=sys.stderr)
            return 2
        summary, results = _summary_of(paths)
        _print_summary(summary, args.report)
        _print_case_table(summary)
        _print_records(results)
        return 0

    if args.runs < 1:
        print("错误：--runs 必须 >= 1", file=sys.stderr)
        return 2

    cases = _select_cases(args.case, args.tier)
    out_dir = os.path.join(args.out_root, args.label)

    if args.dry_run:
        print(f"干运行：{len(cases)} 个案例（不调用 LLM）")
        for case in cases:
            prepared = prepare_case(case, os.path.join(out_dir, "workspaces", case.case_id))
            if prepared.blocking_code:
                print(f"  FAIL {case.case_id}: perception blocked {prepared.blocking_code}")
                continue
            if prepared.fact_package is None:
                print(f"  FAIL {case.case_id}: fact package invalid ({prepared.fact_error})")
                continue
            package = prepared.fact_package
            facts = {f["fact_id"]: f["value"] for f in package["facts"]}
            issues = sorted(fid for fid in facts
                            if fid.startswith(("training.issue.", "dataset.issue.")))
            print(f"  OK   {case.case_id:32s} facts={len(facts):3d} "
                  f"before={prepared.before_metrics}")
            print(f"       issues={issues}")
        return 0

    config = _load_config(args.config)
    print(f"模型: {config.get('llm', {}).get('model')}  案例: {len(cases)}  重复: {args.runs}")
    print(f"产物目录: {out_dir}")

    exit_code = 0
    record_paths: list[str] = []
    for case in cases:
        for run_index in range(1, args.runs + 1):
            root = os.path.join(out_dir, "workspaces", f"{case.case_id}__run{run_index}")
            record = run_case(case, config, run_index, root)
            record_paths.append(write_record(record, out_dir, run_index))
            decision = record.get("decision") or {}
            marker = "OK  " if record["outcome"] == OUTCOME_VALID else "FAIL"
            print(f"  {marker} {case.case_id} #{run_index} "
                  f"action={decision.get('action')} err={record['error_code']}")
            if record["outcome"] not in (OUTCOME_VALID,):
                exit_code = 1
            if record["outcome"] == OUTCOME_PROVIDER_ERROR:
                exit_code = max(exit_code, 1)

    summary, results = _summary_of(record_paths)
    _print_summary(summary, args.label)
    _print_case_table(summary)
    _print_records(results)
    print(f"\n产物: {_write_assessment(out_dir, summary)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
