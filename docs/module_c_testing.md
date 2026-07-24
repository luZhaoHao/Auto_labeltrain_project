# Module C 自动调优测试指南

## 前提条件

| 项目 | 状态 |
|------|------|
| conda 环境 `auto_tune` | ✅ 已配置 |
| YOLOv8 (ultralytics) | ✅ 8.3.253 |
| CUDA 可用 | ✅ 有 GPU |
| 已有数据集分析报告 | 需确认（`log/dataset_report_*.json`） |
| 已有训练分析报告 | 需确认（`log/*_report.json`） |

---

## 测试方式一：Web UI 完整流程（推荐）

### 1. 启动服务

```powershell
conda run -n auto_tune python -m auto_tune.main
```

访问 http://127.0.0.1:8000

### 2. 确保数据就绪

Module C 需要 Module A（数据集分析）和 Module B（训练分析）的输出才能工作。

**如果已有报告文件**（如 `log/dataset_report_*.json` 和 `log/train8_report.json`），直接进入第 3 步。

**如果没有**，按以下顺序操作：

**a) 上传数据集分析**（在"数据集报告"页面）
- 准备一个 YOLO 格式的数据集 ZIP（包含 images/、labels/、data.yaml）
- 拖拽上传 → 点击"上传并分析"
- 等待分析完成

**b) 上传训练结果**（在"智能分析"页面）
- 准备一个 YOLO train 目录 ZIP（至少包含 results.csv + args.yaml）
- 点击文件上传 → 点击"上传并分析训练"
- 等待分析完成，页面显示训练诊断结果

### 3. 执行调优

在"智能分析"页面，会看到"自动调优控制"面板：

| 字段 | 说明 | 建议值 |
|------|------|--------|
| 参考运行 | 选择要优化的基线训练 | 自动检测（默认） |
| 最大重试次数 | 最多尝试几次参数调整 | 默认 3 |
| 模式 | Dry-Run 或 Full Training | 先选 Dry-Run |
| 探查模式 | 先跑 10 轮再决定是否继续 | ✅ 勾选 |
| 自动循环 | 调优完成后自动重新训练 | 可选 |

**步骤：**

1. 选择模式为 **Dry-Run（干运行）**—— 只生成调优计划，不会实际训练
2. 点击"确认并开始训练"
3. 观察右侧控制台日志输出，会实时显示：
   ```
   Perception layer aggregating data...
   Decision agent analyzing...
   Guardrail validation...
   Iteration complete: diagnosis + suggested params
   ```
4. 确认 LLM 给出了合理的超参修改建议
5. 如果满意，切换模式为 **Full Training**，再次点击开始
6. 真实训练启动后，可以在"训练监控"页面查看实时指标

### 4. 查看结果

- **智能分析页面**：调优历史表格，显示每次迭代的诊断、参数修改、状态
- **训练监控页面**：实时损失曲线、mAP 曲线、训练日志
- **历史记录页面**：所有训练和调优的历史汇总

---

## 测试方式二：命令行 Dry-Run（快速验证）

不需要启动 Web UI，直接在终端运行：

```powershell
conda run -n auto_tune python -m auto_tune.main --dry-run
```

流程：
1. 自动从 `log/` 读取最新的 Module A 和 Module B 报告
2. 调用 DeepSeek LLM 分析并生成超参修改建议
3. 通过 Guardrails 验证参数合法性
4. **不执行实际训练**
5. 输出诊断结果和建议参数到控制台

预期输出示例：
```
============================================================
Auto-Tuning Dry-Run
============================================================
Reference run: auto-detect
Mode: dry-run (no actual training)

--- 迭代 1 ---
诊断: 模型存在轻微过拟合，建议降低 epochs 并增加 patience
参数修改: {'lr0': 0.008, 'patience': 25}
```

---

## 测试方式三：命令行完整训练

```powershell
conda run -n auto_tune python -m auto_tune.main --train
```

注意：
- 会实际调用 `yolo train` 启动训练
- 需要 YOLO 数据集路径配置正确（训练依赖 detect 目录下的 dataset 配置）
- 训练过程可能耗时较长（取决于 epochs 设置）

---

## 各阶段验证清单

### 阶段 1：感知层（Perception）

| 检查项 | 通过标准 |
|--------|----------|
| 找到数据集报告 | `dataset_report_*.json` 存在且能解析 |
| 找到训练报告 | `*_report.json` 存在（排除 all_ / dataset_ 前缀） |
| 提取项目背景 | 包含 description / detection_target / data_type |

### 阶段 2：决策层（Decision）

| 检查项 | 通过标准 |
|--------|----------|
| LLM API 调用成功 | DeepSeek 返回有效 JSON |
| 返回含 diagnosis | 诊断文本非空 |
| 返回含 hyperparameter_changes | 至少 1 项参数修改 |
| JSON 格式正确 | `action`, `hyperparameter_changes`, `training_overrides` 字段完整 |

### 阶段 3：防护栏（Guardrails）

| 检查项 | 通过标准 |
|--------|----------|
| 参数值在合法范围内 | lr0 ∈ [1e-5, 0.1], batch ∈ [1, 256] 等 |
| 无冲突参数 | 如 AdamW + lr > 0.005 会告警 |
| Integer 参数取整 | epochs / batch 为整数 |

### 阶段 4：执行层（Execute）— 仅 Full Training 模式

| 检查项 | 通过标准 |
|--------|----------|
| 生成训练配置 | `detect/autotune_N_*/args.yaml` 写入成功 |
| 启动 YOLO 子进程 | subprocess.Popen 成功 |
| 训练目录创建 | `detect/autotune_N_*/` 含 weights/、results.csv |

### 阶段 5：探查监控（Probe Monitor）— 仅 Full Training 模式

| 检查项 | 通过标准 |
|--------|----------|
| 读取 results.csv | 每 10 秒轮询，epoch 数据递增 |
| mAP50 计算 | 首轮可能为 0，后续应上升 |
| 正常退出决策 | mAP50 >= 0.05 → CONTINUE |
| 异常处理 | NaN/Inf loss → ABORT |

---

## 预期耗时参考

| 操作 | 耗时 |
|------|------|
| Dry-Run（1 次迭代） | ~10-30 秒（取决于 LLM API 响应） |
| Full Training （10 epoch probe） | ~5-30 分钟（取决于数据集大小和 GPU） |
| Full Training （完整 100 epoch） | ~30 分钟 - 数小时 |

---

## 常见问题

**Q: 提示"No hyperparameter changes suggested yet"**
- 原因：Module B 训练分析报告未生成
- 解决：先上传训练结果进行 Module B 分析

**Q: Dry-Run 成功但 Full Training 失败**
- 检查 detect/ 目录下是否有可用的数据集和 data.yaml
- 检查 YOLO 训练是否能在该环境下独立运行：`yolo train data=path/to/data.yaml epochs=1`

**Q: SSE 连接失败 / 控制台无输出**
- 检查浏览器控制台网络请求
- 确认 /tuning/start 返回 200
- 重启服务器后重试

**Q: LLM API 调用失败**
- 检查 config.yaml 中的 api_key 是否有效
- 检查 endpoint URL 是否正确
- DeepSeek API 可能需要科学上网
