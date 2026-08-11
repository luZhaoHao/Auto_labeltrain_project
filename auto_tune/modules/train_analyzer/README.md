# 训练结果分析器 (Training Result Analyzer)

三阶段分析流水线：Python 指标分析 → LLM 文本诊断 → 多模态视觉会诊。

## 快速开始

```python
import yaml
from modules.train_analyzer.analyzer import analyze_training_results

config = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# 模式1：分析最新一次训练（默认）
report = analyze_training_results("detect", config)

# 模式2：指定某次训练
report = analyze_training_results("detect", config, run_name="train10")

# 模式3：分析所有训练
report = analyze_training_results("detect", config, run_name="__all__")

# 仅 Stage 1（不调用 LLM，免费）
report = analyze_training_results("detect", config, enable_llm=False, enable_vision=False)
```

## 三阶段说明

### Stage 1：Python 血检（免费）
解析 `results.csv` + `args.yaml`，提取指标、检测问题。
- 输入：`detect/train*/results.csv`，`detect/train*/args.yaml`
- 输出：结构化 dict（指标、曲线趋势、问题列表）
- 检测的问题类型：`overfitting`, `underfitting`, `plateau`, `nan_loss`, `low_final_map`, `early_stop_too_soon`

### Stage 2：LLM 文本诊断
将 Stage 1 的结构化数据发给 DeepSeek，返回自然语言诊断。
- 模型：`deepseek-v4-flash`
- 费用：约 200 tokens/次
- 输出：`report["llm_analysis"][run_name]["llm_diagnosis"]`

### Stage 3：多模态视觉会诊
将混淆矩阵图和错检区域图发给 Qwen-VL，返回视觉分析。
- 模型：`qwen3-vl-flash`
- 自动生成错检裁剪图（比较 `val_batch_labels` vs `val_batch_pred` 的像素差异）
- 输出：`report["vision_analysis"][run_name]["confusion_matrix_analysis"]["analysis"]`
- 输出：`report["vision_analysis"][run_name]["error_crop_analysis"]["analysis"]`

## 配置 (config.yaml)

```yaml
llm:
  provider: deepseek
  api_key: sk-xxx
  model: deepseek-v4-flash
  endpoint: https://api.deepseek.com/v1/chat/completions
  temperature: 0.3
  max_tokens: 2000
  enabled: true

vision:
  provider: qwen
  api_key: sk-xxx
  model: qwen3-vl-flash
  endpoint: https://xxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions
  temperature: 0.3
  max_tokens: 500
  enabled: true
  error_crops_enabled: true

train_analyzer:
  plateau_epochs: 10
  overfit_threshold: 0.15
  val_loss_ma_window: 5
  stale_threshold: 15
  convergence_gap: 0.1
  min_acceptable_map: 0.5
  top_k_runs: 5
  compare_metric: metrics/mAP50(B)
```

## 输出格式

```python
{
  "module": "train_analyzer",
  "version": "1.0",
  "analysis_timestamp": "2026-07-20T...Z",
  "detect_dir": "detect",
  "total_runs": 1,          # 实际分析的训练数

  # Stage 1 输出
  "runs": {
    "train10": {
      "args": {"model": "yolov8s.yaml", "imgsz": [544, 544], ...},
      "results": {
        "columns": {"epoch": [...], "metrics/mAP50(B): [...], ...},
        "final_metrics": {"metrics/mAP50(B)": 0.995, ...},
        "best_epoch": 89,
        "total_epochs": 100
      },
      "curve_analysis": {
        "val_box": {"trend": "descending", "slope": -0.008},
        "val_cls": {"trend": "rising", "slope": 0.003},
        "mAP50": {"trend": "improving", ...},
        "early_stopping": {"stopped_early": false}
      },
      "issues": [
        {"type": "overfitting", "severity": "warning", "detail": "val_cls_loss rising (slope=0.003)"}
      ]
    }
  },

  # Stage 2 输出
  "llm_analysis": {
    "train10": {
      "llm_diagnosis": "## 1. 质量评估...（中文自然语言）",
      "model_used": "deepseek-v4-flash",
      "error": null
    }
  },

  # Stage 3 输出
  "vision_analysis": {
    "train10": {
      "confusion_matrix_analysis": {
        "analysis": "从混淆矩阵看...（中文自然语言）",
        "error": null
      },
      "error_crop_analysis": {
        "analysis": "错检区域分析...（中文自然语言）",
        "error": null
      }
    }
  },

  # 对比汇总（多训练时有效）
  "comparison": {
    "ranked_by": "metrics/mAP50(B)",
    "top_runs": [{"name": "train10", "final_mAP50": 0.995, ...}],
    "best_run": "train10",
    "total_runs": 1
  },
  "summary": {
    "total_runs_analyzed": 1,
    "best_overall_run": "train10",
    "best_mAP50": 0.995,
    "average_mAP50": 0.995,
    "runs_with_issues": 1,
    "common_issues": [{"type": "overfitting", "count": 1}],
    "recommendation": "best model from train10 at epoch 89"
  }
}
```

## 各文件说明

| 文件 | 作用 |
|---|---|
| `analyzer.py` | 总调度，三阶段编排 |
| `results_parser.py` | 解析 results.csv + args.yaml |
| `curve_analysis.py` | 损失/指标曲线趋势分析 |
| `issue_detector.py` | 训练问题检测 |
| `run_comparator.py` | 多训练结果对比汇总 |
| `llm_analyzer.py` | Stage 2：调用 DeepSeek |
| `vision_analyzer.py` | Stage 3：调用 Qwen-VL |
| `crop_utils.py` | 生成错检裁剪图 |
