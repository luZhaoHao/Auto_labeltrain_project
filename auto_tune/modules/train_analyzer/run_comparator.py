"""Compare multiple YOLO training runs."""

from .curve_analysis import analyze_metric_curves


def compare_runs(runs: list[dict], config: dict) -> dict:
    """Compare all training runs and generate ranking.

    Args:
        runs: list of dicts from load_training_run().
        config: train_analyzer config dict.

    Returns:
        dict with:
            ranked_by: metric name used for ranking
            top_runs: list of top K run summaries
            best_run: name of the best run
            summary_table: dict mapping run_name -> key metrics
    """
    compare_metric = config.get("compare_metric", "metrics/mAP50(B)")
    top_k = config.get("top_k_runs", 5)

    summaries = {}
    for run in runs:
        name = run.get("name", "unknown")
        results = run.get("results", {})
        final = results.get("final_metrics", {})
        best_epoch = results.get("best_epoch")
        args = run.get("args", {})
        if not isinstance(args, dict):
            args = {}

        imgsz = args.get("imgsz", "?")
        if isinstance(imgsz, list):
            imgsz = f"{imgsz[0]}x{imgsz[1]}"

        map50 = final.get("metrics/mAP50(B)")
        map95 = final.get("metrics/mAP50-95(B)")
        precision = final.get("metrics/precision(B)")
        recall = final.get("metrics/recall(B)")

        summaries[name] = {
            "epochs": results.get("total_epochs", 0),
            "best_epoch": best_epoch,
            "final_mAP50": round(map50, 4) if map50 is not None else None,
            "final_mAP50-95": round(map95, 4) if map95 is not None else None,
            "final_precision": round(precision, 4) if precision is not None else None,
            "final_recall": round(recall, 4) if recall is not None else None,
            "imgsz": imgsz,
            "model": args.get("model", "?"),
            "optimizer": args.get("optimizer", "auto"),
            "cos_lr": args.get("cos_lr", False),
        }

    # Sort by the compare metric
    def _sort_key(item):
        name, data = item
        val = data.get("final_mAP50" if "mAP50" in compare_metric else "final_mAP50")
        return val if val is not None else -1

    sorted_runs = sorted(summaries.items(), key=_sort_key, reverse=True)
    top_runs = [{"name": name, **data} for name, data in sorted_runs[:top_k]]

    best_run_name = sorted_runs[0][0] if sorted_runs else None

    return {
        "ranked_by": compare_metric,
        "top_runs": top_runs,
        "best_run": best_run_name,
        "summary_table": summaries,
        "total_runs": len(runs),
    }


def summarize_runs(runs: list[dict], config: dict) -> dict:
    """Generate an executive summary across all runs.

    Returns:
        dict with high-level stats about the training experiments.
    """
    comp = compare_runs(runs, config)
    summaries = comp["summary_table"]

    if not summaries:
        return {"total_runs_analyzed": 0, "message": "No runs to summarize."}

    map_values = [s["final_mAP50"] for s in summaries.values() if s["final_mAP50"] is not None]
    best_map = max(map_values) if map_values else None
    avg_map = sum(map_values) / len(map_values) if map_values else None
    best_run_name = max(summaries, key=lambda n: summaries[n]["final_mAP50"] or -1)

    # Count issues across runs
    runs_with_issues = 0
    all_issues = []
    for run in runs:
        issues = run.get("_issues", [])
        if issues:
            runs_with_issues += 1
            for iss in issues:
                all_issues.append(iss.get("type", "unknown"))

    from collections import Counter
    common_issues = [{"type": t, "count": c} for t, c in Counter(all_issues).most_common()] if all_issues else []

    return {
        "total_runs_analyzed": len(summaries),
        "runs_with_issues": runs_with_issues,
        "common_issues": common_issues,
        "best_overall_run": best_run_name,
        "best_mAP50": round(best_map, 4) if best_map is not None else None,
        "average_mAP50": round(avg_map, 4) if avg_map is not None else None,
        "metric_range": {
            "mAP50": [round(min(map_values), 4), round(max(map_values), 4)] if map_values else None,
        },
    }
