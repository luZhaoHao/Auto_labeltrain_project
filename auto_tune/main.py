"""YOLOv8 Auto-Tuning Agent — Unified Entry Point.

Usage:
    python -m auto_tune.main              # Start web UI
    python -m auto_tune.main --dry-run    # Run dry-run tuning test
    python -m auto_tune.main --train      # Run full tuning loop (no UI)
"""

import sys
import yaml
import os

# Ensure project root is on sys.path so `auto_tune` package is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = set(sys.argv[1:])

    if "--dry-run" in args:
        from auto_tune.modules.agent_engine.loop import run_tuning_loop
        config = load_config()
        print("=" * 60)
        print("Auto-Tuning Dry-Run")
        print("=" * 60)
        print("Reference run: auto-detect")
        print("Mode: dry-run (no actual training)")
        print()
        result = run_tuning_loop(config, skip_execute=True)
        iterations = result.get("iterations", [])
        print(f"\n完成 {len(iterations)} 次迭代")
        for it in iterations:
            print(f"\n--- 迭代 {it['iteration']} ---")
            diag = it.get("decision", {}).get("diagnosis", "无诊断")
            print(f"诊断: {diag}")
            changes = it.get("decision", {}).get("hyperparameter_changes", {})
            if changes:
                print(f"参数修改: {changes}")
            if it.get("guard_results", {}).get("warnings"):
                for w in it["guard_results"]["warnings"]:
                    print(f"  [WARN] {w}")
            if it.get("error"):
                print(f"  [ERROR] {it['error']}")
        return

    if "--train" in args:
        from auto_tune.modules.agent_engine.loop import run_tuning_loop
        config = load_config()
        print("Starting full tuning loop...")
        result = run_tuning_loop(config, skip_execute=False)
        final = result.get("final_result")
        if final:
            print(f"Training {final['train_name']} started")
        else:
            print(f"Tuning failed: {result.get('error', 'Unknown')}")
        return

    # Default: start web UI
    from auto_tune.ui.app import start_server
    config = load_config()
    host = "127.0.0.1"
    port = 8000
    print("=" * 60)
    print("  Auto-Tune Dashboard")
    print("=" * 60)
    print(f"  Module A: Dataset Analyzer    [OK]")
    print(f"  Module B: Training Analyzer   [OK]")
    print(f"  Module C: Auto-Tuning Agent   [OK]")
    print("=" * 60)
    start_server(host=host, port=port)


if __name__ == "__main__":
    main()
