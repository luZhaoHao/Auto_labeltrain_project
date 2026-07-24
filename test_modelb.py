import json
import os
import yaml
from auto_tune.modules.train_analyzer.analyzer import analyze_training_results

config = yaml.safe_load(open("auto_tune/config.yaml", encoding="utf-8"))

# ============================================================
# 项目信息 — 填在这里会覆盖 config.yaml 中的 project 字段
# 留空则使用 config.yaml 里的值
# ============================================================
project = {
    "name": "请在此填写项目名称",                    # 项目名称
    "description": "请在此填写项目描述",             # 项目描述，如 "PCB 焊点缺陷检测"
    "detection_target": "缺陷",        # 检测目标，如 "焊点缺陷（虚焊、连锡、少锡）"
    "data_type": "工业相机拍摄RGB图片",               # 数据类型，如 "工业 X 光图像 / 灰度 BMP"
    "extra": {},                   # 预留扩展字段
}
# 合并到 config（若有内容则覆盖 config.yaml 的同名字段）
if any(v for v in project.values() if v):
    config["project"] = project

# ============================================================
# 运行模式选择
# ============================================================

# 模式1：分析最新一次训练（默认）
#report = analyze_training_results("detect", config)

# 模式2：指定某次训练
report = analyze_training_results("detect", config, run_name="train8")

# 模式3：分析所有训练
#report = analyze_training_results("detect", config, run_name="__all__")

# 仅 Stage 1（不调用 LLM，免费）
#report = analyze_training_results("detect", config, enable_llm=False, enable_vision=False)

# === 输出到文件 ===

os.makedirs("log", exist_ok=True)

# 用训练名做前缀
run_names = list(report.get("runs", {}).keys())
prefix = run_names[0] if len(run_names) == 1 else "all"

# 1. 人类可读报告 (Markdown)
summary = report.get("summary", {})
lines = []
lines.append("# 训练结果分析报告")
lines.append("")

# ── 项目信息 ──
project = report.get("project", {})
if project and any(v for v in [project.get(k) for k in ("name", "description", "detection_target", "data_type")] if v):
    lines.append("## 项目信息")
    for key, label in [("name", "**项目名称**"), ("description", "**项目描述**"),
                        ("detection_target", "**检测目标**"), ("data_type", "**数据类型**")]:
        val = project.get(key)
        if val:
            lines.append(f"- {label}: {val}")
    lines.append("")

# ── 总体摘要 ──
if summary:
    lines.append("## 总体摘要")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| 分析训练数 | {summary.get('total_runs_analyzed', 'N/A')} |")
    lines.append(f"| 最佳训练 | {summary.get('best_overall_run', 'N/A')} |")
    lines.append(f"| 最佳 mAP50 | {summary.get('best_mAP50', 'N/A')} |")
    lines.append(f"| 平均 mAP50 | {summary.get('average_mAP50', 'N/A')} |")
    lines.append(f"| 有问题的训练数 | {summary.get('runs_with_issues', 'N/A')} |")
    lines.append("")

issues = summary.get("common_issues", [])
if issues:
    lines.append("### 常见问题")
    for iss in issues:
        lines.append(f"- **{iss['type']}**: {iss['count']} 次")
    lines.append("")

if "error" in report:
    lines.append(f"> ⚠ **错误**: {report['error']}")
    lines.append("")

human_path = f"log/{prefix}_report.md"
with open(human_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

    # llm_diagnosis — 按每个训练追加
    for rn in run_names:
        llm = report.get("llm_analysis", {}).get(rn, {})
        if llm.get("llm_diagnosis"):
            f.write(f"\n---\n## LLM 诊断 — {rn}\n\n")
            f.write(llm["llm_diagnosis"] + "\n")
        if llm.get("error"):
            f.write(f"\n> LLM 诊断错误: {llm['error']}\n")

    # vision_analysis — 混淆矩阵 + 错检裁剪
    for rn in run_names:
        vision = report.get("vision_analysis", {}).get(rn, {})
        cm = vision.get("confusion_matrix_analysis", {})
        if cm.get("analysis"):
            f.write(f"\n---\n## 混淆矩阵分析 — {rn}\n\n")
            f.write(cm["analysis"] + "\n")
        if cm.get("error"):
            f.write(f"\n> 混淆矩阵分析错误: {cm['error']}\n")

        ec = vision.get("error_crop_analysis", {})
        if ec.get("analysis"):
            f.write(f"\n---\n## 错检区域分析 — {rn}\n\n")
            f.write(ec["analysis"] + "\n")
        if ec.get("error"):
            f.write(f"\n> 错检区域分析错误: {ec['error']}\n")

print(f"[保存] {human_path}")

# 2. 给 LLM 用的 JSON 输出
json_path = f"log/{prefix}_report.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"[保存] {json_path}")