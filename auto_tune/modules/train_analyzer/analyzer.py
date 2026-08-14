"""Training result analyzer orchestrator — runs full analysis, produces JSON report.

Three stages:
  Stage 1: Python-based parsing, curve analysis, issue detection (0 token cost)
  Stage 2: Text LLM diagnosis via DeepSeek (~200 tokens)
  Stage 3: Multimodal vision consultation via Qwen-VL (confusion matrix + error crops)
"""

import os
import datetime

from .results_parser import find_all_runs, find_latest_run, load_training_run
from .curve_analysis import analyze_loss_curves, analyze_metric_curves, detect_early_stopping
from .issue_detector import detect_issues
from .run_comparator import compare_runs, summarize_runs


def analyze_training_results(detect_dir: str, config: dict,
                             run_name: str | None = None,
                             enable_llm: bool = True,
                             enable_vision: bool = True) -> dict:
    """Run full training results analysis pipeline.

    Args:
        detect_dir: path to detect/ folder containing train subdirectories.
        config: dict with train_analyzer thresholds (from config.yaml).
        run_name: which run to analyze.
            None -> auto-detect the latest train folder.
            "__all__" -> analyze ALL train folders.
            "train10" -> analyze the specified folder only.

    Returns:
        JSON-serializable dict with per-run analysis, comparison, and summary.
    """
    # Project context — extracted upfront so it flows into every return path
    project_info = config.get("project", {}) if isinstance(config, dict) else {}
    analysis_config = config.get("train_analyzer", config) if isinstance(config, dict) else {}

    if not os.path.isdir(detect_dir):
        return {
            "module": "train_analyzer",
            "version": "1.0",
            "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "detect_dir": detect_dir,
            "project": project_info,
            "error": f"Directory not found: {detect_dir}",
        }

    if run_name is None:
        # Auto-detect latest run
        latest = find_latest_run(detect_dir)
        if latest is None:
            return {
                "module": "train_analyzer",
                "version": "1.0",
                "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "detect_dir": detect_dir,
                "project": project_info,
                "total_runs": 0,
                "error": "No training run directories (train*) found.",
            }
        run_dirs = [latest]
    elif run_name == "__all__":
        run_dirs = find_all_runs(detect_dir)
    else:
        # Specific run name
        candidate = os.path.join(detect_dir, run_name)
        if os.path.isdir(candidate):
            run_dirs = [candidate]
        else:
            return {
                "module": "train_analyzer",
                "version": "1.0",
                "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "detect_dir": detect_dir,
                "project": project_info,
                "total_runs": 1,
                "error": f"Run directory not found: {candidate}",
            }

    if not run_dirs:
        return {
            "module": "train_analyzer",
            "version": "1.0",
            "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "detect_dir": detect_dir,
            "project": project_info,
            "total_runs": 0,
            "error": "No training run directories (train*) found.",
        }

    # Parse each run
    runs = []
    run_results = {}
    for run_dir in run_dirs:
        run_data = load_training_run(run_dir)

        # Skip runs with parse errors
        if "error" in run_data.get("results", {}):
            run_data["_skip"] = True
            runs.append(run_data)
            run_results[run_data["name"]] = run_data
            continue

        # Curve analysis
        curve_analysis = analyze_loss_curves(run_data["results"], analysis_config)
        metric_analysis = analyze_metric_curves(run_data["results"], analysis_config)

        # Early stopping analysis
        early_stop_input = dict(run_data["results"])
        early_stop_input["args"] = run_data.get("args", {})
        early_stop = detect_early_stopping(early_stop_input, analysis_config)
        curve_analysis["early_stopping"] = early_stop

        # Issue detection
        issues = detect_issues(run_data, analysis_config)
        run_data["_issues"] = issues

        run_data["curve_analysis"] = curve_analysis
        run_data["metric_analysis"] = metric_analysis
        run_data["issues"] = issues

        runs.append(run_data)
        run_results[run_data["name"]] = run_data

    # Cross-run comparison
    comparison = compare_runs(runs, analysis_config)
    summary = summarize_runs(runs, analysis_config)

    # Build clean output (remove internal keys)
    output_runs = {}
    for name, rd in run_results.items():
        clean = {
            "name": rd.get("name"),
            "args": rd.get("args", {}),
            "results": rd.get("results", {}),
        }
        if not rd.get("_skip"):
            clean["curve_analysis"] = rd.get("curve_analysis")
            clean["metric_analysis"] = rd.get("metric_analysis")
            clean["issues"] = rd.get("issues")
        output_runs[name] = clean

    report = {
        "module": "train_analyzer",
        "version": "1.0",
        "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "detect_dir": detect_dir,
        "project": project_info,
        "total_runs": len(run_dirs),
        "runs": output_runs,
        "comparison": comparison,
        "summary": summary,
    }

    # === Stage 2: Text LLM diagnosis ===
    if enable_llm and config.get("llm", {}).get("enabled", False):
        from .llm_analyzer import analyze_with_llm
        report["llm_analysis"] = analyze_with_llm(report, config)

    # === Stage 3: Vision consultation ===
    if enable_vision and config.get("vision", {}).get("enabled", False):
        from .vision_analyzer import multimodal_consult
        vision_results = {}
        for name, rd in run_results.items():
            if rd.get("_skip"):
                continue
            run_dir = None
            for d in run_dirs:
                if os.path.basename(d) == name:
                    run_dir = d
                    break
            if run_dir:
                vision_results[name] = multimodal_consult(run_dir, config, project_info)
        report["vision_analysis"] = vision_results

    return report
