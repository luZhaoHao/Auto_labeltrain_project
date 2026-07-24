# 🔄 YOLOv8 Auto-Tuning Agent

> 一个三模块闭环的 YOLOv8 自动调优系统 — **数据集分析 → 训练诊断 → 超参自动调整**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.x-green?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-orange)](#)
[![Qwen-VL](https://img.shields.io/badge/Vision-Qwen_VL-purple)](#)

---

## 📋 项目概述

本项目是一个 **自动化 YOLOv8 训练优化工具**，通过三个模块形成闭环：

```
用户上传数据集 → Module A 分析 → Module B 训练诊断 → Module C 自动调优
                                                      ↕
                                              反复迭代直到收敛
```

| 模块 | 名称 | 功能 |
|------|------|------|
| **Module A** | 数据集分析器 | 分析模糊度、曝光、SNR、bbox 几何、类别平衡、特征聚类 |
| **Module B** | 训练诊断器 | 三阶段诊断：Python 指标 → DeepSeek LLM 文本诊断 → Qwen-VL 视觉分析 |
| **Module C** | 智能调优引擎 | 感知 → LLM 决策 → 护栏规则 → 执行训练 → 探针监控 → 自动循环 |

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 创建 conda 环境
conda create -n auto_tune python=3.10 -y
conda activate auto_tune

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API 密钥
cp auto_tune/config.template.yaml auto_tune/config.yaml
# 编辑 config.yaml，填入 DeepSeek 和 Qwen-VL 的 API key
```

### 启动 Web UI

```bash
# 方式一：直接启动
python -m auto_tune.main

# 方式二：通过辅助脚本
python start_server.py

# 方式三：（Windows）后台启动
start_server.bat
```

打开浏览器访问 `http://127.0.0.1:8000`。

### 命令行模式

```bash
# 调优模拟（不实际训练）
python -m auto_tune.main --dry-run

# 完整自动调优
python -m auto_tune.main --train
```

---

## 🧩 架构详解

### Module A — 数据集分析

| 组件 | 功能 |
|------|------|
| `image_quality.py` | 模糊度、曝光度、信噪比检测 |
| `bbox_geometry.py` | bbox 大小、宽高比、重叠度、空间偏差 |
| `class_stats.py` | 类别分布、平衡度分析 |
| `feature_cluster.py` | 特征提取 + DBScan 聚类 |

### Module B — 训练诊断（三阶段）

```
Stage 1: Python 指标分析
  results.csv 解析 → 损失/指标趋势 → 过拟合/欠拟合检测 → 多轮对比

Stage 2: DeepSeek LLM 文本诊断
  将 Stage 1 的指标报告送入 DeepSeek → 自然语言诊断 + 改进建议

Stage 3: Qwen-VL 视觉分析
  confusion_matrix 和 error crops → 视觉模型分析 → 错误模式识别
```

### Module C — 自动调优引擎

```
Perception → Decision (LLM) → Guardrails → Execute → Probe Monitor → 循环
   ↓            ↓               ↓            ↓           ↓
 聚合 A+B 报告  生成超参调整  验证/钳制调整  启动训练    前 N 轮监控
                 建议          (防止破坏性参数)             继续/重试/中止
```

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest auto_tune/tests -v

# Module B 诊断测试
python test_modelb.py

# Module C 组件测试
python test_modelc.py --stage 1   # 护栏规则测试
python test_modelc.py --stage 3   # 决策代理测试（需 LLM API）
python test_modelc.py --stage 5   # 完整循环模拟
```

---

## 📁 项目结构

```
auto_tune/
├── main.py                        # 统一入口（UI / dry-run / train）
├── config.yaml                    # 全局配置（LLM 密钥、阈值、参数）
├── modules/
│   ├── dataset_analyzer/          # Module A
│   ├── train_analyzer/            # Module B
│   └── agent_engine/              # Module C
├── ui/                            # Web UI（FastAPI + SSE + SPA）
├── tests/                         # pytest 测试套件
├── utils/                         # 工具函数
└── docs/                          # 文档
```

---

## 📊 评分机制

- **综合评分（四维）**: `mAP50 × 0.35 + mAP50-95 × 0.25 + precision × 0.20 + recall × 0.20`
- **快速评分（双指标）**: `mAP50 × 0.6 + mAP50-95 × 0.4`
- **数据集质量**: 模糊度(0.15) + 欠曝(0.15) + 过曝(0.15) + 类别均衡(0.25) + bbox质量(可配置)

---

## ⚙️ 配置

主要配置位于 `config.yaml`：

| 配置项 | 说明 |
|--------|------|
| `llm` | DeepSeek API 配置（端點、模型、温度） |
| `vision` | Qwen-VL 配置（混淆矩阵 + 错误裁剪分析） |
| `guardrails` | 参数验证模式（strict / lenient） |
| `probe` | 探针监控：前 N 轮检测、自动继续阈值、最大重试次数 |
| `training` | 默认 YOLO 参数（epochs, batch, imgsz, workers） |

---

## 🛣️ 路线图

- [x] Module A: 数据集质量分析
- [x] Module B: 三阶段训练诊断
- [x] Module C: 自动调优循环
- [x] Web UI + SSE 实时流
- [x] 中英双语界面
- [ ] 更多预训练模型支持
- [ ] 分布式训练支持
- [ ] 实验对比仪表盘

---

## 📄 许可证

[MIT](LICENSE)

---

## 🤝 贡献

欢迎 Issue 和 Pull Request！

---

> **提示**: `auto_tune/config.yaml` 包含 API 密钥，已被 git 排除。请使用 `config.template.yaml` 作为模板配置你自己的密钥。
