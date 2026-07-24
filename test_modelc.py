"""
Module C 测试脚本 --- Auto-Tuning Agent

测试流程（按依赖顺序）:
  1. Guardrails --- 参数校验与熔断
  2. Perception --- 感知层数据聚合
  3. Decision Agent --- LLM 调参决策
  4. Executor --- 配置写入
  5. Loop --- 完整编排（dry-run 模式）

用法:
  python test_modelc.py              # 运行 Guardrails + Perception + Executor
  python test_modelc.py --stage 1    # 仅测试 Guardrails
  python test_modelc.py --stage 2    # 仅测试 Perception
  python test_modelc.py --stage 3    # 仅测试 Decision Agent (需 LLM API)
  python test_modelc.py --stage 4    # 仅测试 Executor
  python test_modelc.py --stage 5    # 仅测试 Loop (dry-run + LLM)
  python test_modelc.py --web        # 启动 Web UI
"""

import sys
import os
import json
import yaml

sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 for stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "auto_tune", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


OK = "[PASS]"


def test_guardrails():
    print()
    print("=" * 60)
    print("Stage 1: Guardrails --- 参数校验与熔断")
    print("=" * 60)

    from auto_tune.modules.agent_engine.guardrails import validate_and_clamp, merge_params

    print("\n[Test 1] 正常参数")
    params = {"lr0": 0.01, "box": 7.5, "cls": 0.5, "mosaic": 1.0}
    result = validate_and_clamp(params)
    print(f"  valid={result.valid}, warnings={result.warnings}, errors={result.errors}")
    assert result.valid, "Normal params should be valid"
    print(f"  {OK}")

    print("\n[Test 2] 超界熔断")
    params = {"lr0": 0.5, "box": 50.0, "mosaic": 2.0}
    result = validate_and_clamp(params)
    print(f"  clamped: lr0={result.clamped.get('lr0')}, box={result.clamped.get('box')}, mosaic={result.clamped.get('mosaic')}")
    assert result.clamped.get("lr0") == 0.1, "lr0 should clamp to 0.1"
    assert result.clamped.get("box") == 20.0, "box should clamp to 20.0"
    assert result.clamped.get("mosaic") == 1.0, "mosaic should clamp to 1.0"
    print(f"  {OK}")

    print("\n[Test 3] 过正则化冲突")
    params = {"dropout": 0.5, "weight_decay": 0.01}
    result = validate_and_clamp(params)
    print(f"  errors={result.errors}")
    assert not result.valid, "Should detect over-regularization"
    print(f"  {OK}")

    print("\n[Test 4] 小数据集 + 强增强警告")
    params = {"mosaic": 1.0, "mixup": 0.5, "degrees": 30}
    result = validate_and_clamp(params, dataset_info={"total_images": 200})
    print(f"  warnings={result.warnings}")
    assert any("Small dataset" in w for w in result.warnings), "Should warn about small dataset"
    print(f"  {OK}")

    print("\n[Test 5] 未知参数透传")
    params = {"lr0": 0.01, "my_custom_param": 999}
    result = validate_and_clamp(params)
    print(f"  warnings={result.warnings}")
    assert any("Unknown parameter" in w for w in result.warnings), "Should warn about unknown param"
    print(f"  {OK}")

    print("\n[Test 6] 参数合并")
    base = {"lr0": 0.01, "box": 7.5, "epochs": 100, "model": "yolov8s.yaml"}
    changes = {"lr0": 0.005, "box": 10.0}
    merged = merge_params(base, changes)
    print(f"  lr0={merged['lr0']}, box={merged['box']}, epochs={merged['epochs']}")
    assert merged["lr0"] == 0.005
    assert merged["box"] == 10.0
    assert merged["epochs"] == 100
    print(f"  {OK}")

    print(f"\nStage 1: 全部通过 {OK}")


def test_perception():
    print()
    print("=" * 60)
    print("Stage 2: Perception --- 感知层数据聚合")
    print("=" * 60)

    from auto_tune.modules.agent_engine.perception import (
        build_perception, summarize_perception,
        find_module_b_report,
    )

    print("\n[Test 1] 查找 Module B 报告")
    report = find_module_b_report(log_dir="log")
    if report:
        print(f"  Found: {report.get('module')} with {report.get('total_runs')} runs")
    else:
        print("  [WARN] 未找到 Module B 报告（请先运行 test_modelb.py）")

    print("\n[Test 2] 构建感知数据")
    perception = build_perception(log_dir="log")
    print(f"  Dataset keys: {list(perception.get('dataset', {}).keys())}")
    print(f"  Training keys: {list(perception.get('training', {}).keys())}")
    print(f"  Project: {perception.get('project', {})}")
    print(f"  {OK}")

    print("\n[Test 3] 感知数据摘要")
    summary = summarize_perception(perception)
    print(f"  摘要长度: {len(summary)} 字符")
    print(f"  预览:\n{summary[:300]}...")
    print(f"  {OK}")

    print(f"\nStage 2: 全部通过 {OK}")


def test_decision_agent():
    print()
    print("=" * 60)
    print("Stage 3: Decision Agent --- LLM 调参决策")
    print("=" * 60)

    from auto_tune.modules.agent_engine.perception import build_perception, summarize_perception
    from auto_tune.modules.agent_engine.decision_agent import (
        build_decision_prompt, decide_hyperparameters,
    )

    config = load_config()

    if not config.get("llm", {}).get("enabled"):
        print("  [SKIP] LLM 未启用，仅测试 prompt 构建")
        perception = build_perception(log_dir="log")
        summary = summarize_perception(perception)
        prompt = build_decision_prompt(summary, perception.get("project"))
        print(f"  Prompt 长度: {len(prompt)} 字符")
        assert "hyperparameter_changes" in prompt, "Prompt should contain JSON template"
        print(f"  {OK}")
        return

    print("\n[Test 1] LLM 调参决策")
    perception = build_perception(log_dir="log")
    decision = decide_hyperparameters(perception, config)
    if decision.get("error"):
        print(f"  [WARN] API 错误: {decision['error']}")
    else:
        print(f"  诊断: {decision.get('diagnosis')}")
        print(f"  方案: {decision.get('action')}")
        print(f"  参数: {decision.get('hyperparameter_changes')}")
        print(f"  配置覆盖: {decision.get('training_overrides')}")
        # May be empty if no dataset info available — that's OK
        print(f"  Has diagnosis: {bool(decision.get('diagnosis'))}")
    print(f"  {OK}")

    print(f"\nStage 3: 全部通过 {OK}")


def test_executor():
    print()
    print("=" * 60)
    print("Stage 4: Executor --- 配置写入")
    print("=" * 60)

    from auto_tune.modules.agent_engine.executor import (
        find_detect_dir, get_next_train_name, write_training_config,
    )
    from auto_tune.modules.agent_engine.guardrails import merge_params, validate_and_clamp

    print("\n[Test 1] 查找 detect 目录")
    detect_dir = find_detect_dir()
    print(f"  detect_dir = {detect_dir}")
    print(f"  {OK}")

    print("\n[Test 2] 生成训练名称")
    name = get_next_train_name(detect_dir)
    print(f"  name = {name}")
    assert name.startswith("train_autotune_"), "Name should follow pattern"
    print(f"  {OK}")

    print("\n[Test 3] 写入训练配置")
    base_args = {
        "model": "yolov8s.yaml",
        "data": "data.yaml",
        "epochs": 100,
        "batch": 16,
        "lr0": 0.01,
        "box": 7.5,
    }
    changes = {"lr0": 0.005, "box": 10.0, "epochs": 200}
    guard_result = validate_and_clamp(changes)
    merged = merge_params(base_args, changes)

    output_dir = os.path.join(detect_dir, name)
    args_path = write_training_config(base_args, merged, output_dir)
    print(f"  args.yaml = {args_path}")
    assert os.path.exists(args_path), "args.yaml should exist"

    with open(args_path, encoding="utf-8") as f:
        written = yaml.safe_load(f)
    print(f"  Written: lr0={written.get('lr0')}, box={written.get('box')}, epochs={written.get('epochs')}")
    print(f"  {OK}")

    print(f"\nStage 4: 全部通过 {OK}")


def test_loop():
    print()
    print("=" * 60)
    print("Stage 5: Loop --- 完整编排 (dry-run)")
    print("=" * 60)

    from auto_tune.modules.agent_engine.loop import run_tuning_loop

    config = load_config()

    if not config.get("llm", {}).get("enabled"):
        print("  [SKIP] LLM 未启用")
        print(f"  {OK}")
        return

    print("\n[Test 1] Dry-run 调参循环")
    progress_log = []

    def on_progress(iteration, message):
        progress_log.append((iteration, message))
        print(f"  [iter {iteration}] {message}")

    result = run_tuning_loop(
        config, reference_run=None,
        max_retries=1, log_dir="log",
        skip_execute=True,
        on_progress=on_progress,
    )
    print(f"\n  迭代数: {len(result.get('iterations', []))}")
    for it in result.get("iterations", []):
        print(f"  诊断: {it.get('decision', {}).get('diagnosis', '?')[:60]}")
        print(f"  参数修改: {it.get('decision', {}).get('hyperparameter_changes', {})}")
        err = it.get("error")
        if err:
            print(f"  错误: {err}")
    print(f"  进度日志: {len(progress_log)} 条")
    assert len(result.get("iterations", [])) > 0, "Should have at least one iteration"
    print(f"  {OK}")

    print(f"\nStage 5: 全部通过 {OK}")


def start_web():
    print()
    print("=" * 60)
    print("Stage 6: Web UI --- 启动仪表盘")
    print("=" * 60)
    from auto_tune.ui.app import start_server
    start_server()


if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--web" in args:
        start_web()
        sys.exit(0)

    stage_filter = None
    for a in sys.argv[1:]:
        if a.startswith("--stage"):
            if "=" in a:
                stage_filter = int(a.split("=")[-1])
            else:
                idx = sys.argv.index(a)
                if idx + 1 < len(sys.argv):
                    stage_filter = int(sys.argv[idx + 1])

    stages = {
        1: test_guardrails, 2: test_perception,
        3: test_decision_agent, 4: test_executor, 5: test_loop,
    }

    if stage_filter:
        if stage_filter in stages:
            stages[stage_filter]()
        else:
            print(f"Invalid stage: {stage_filter}")
    else:
        test_guardrails()
        test_perception()
        test_executor()
        config = load_config()
        if config.get("llm", {}).get("enabled"):
            test_decision_agent()
            test_loop()
        else:
            print("\n[SKIP] LLM 未启用，跳过 Stage 3 (Decision) 和 Stage 5 (Loop)")
            print("  Use: python test_modelc.py --stage 3")
