# LLM 调优语义规则说明（Q1.2）

> 本文是 Q1.2 运行时规则的人类可读说明。`auto_tune/modules/agent_engine/semantic_rules.py` 是唯一运行时事实来源；两者不一致时以代码为准，并必须同步修正文档与测试。

## 1. 结论与执行位置

LLM 只能依据同一版本化事实包提出建议，系统按以下固定顺序执行：

`FactPackage → Q1.1 契约校验 → Q1.2 语义校验 → Guardrails → 命令构造 → 训练`

- `keep_params` 不修改参数，可直接通过语义层。
- `adjust_params` 的每个修改都必须存在当前值，并至少有一条已登记事实支持。
- 同一参数同时存在相反方向证据时拒绝执行。
- 方向正确但幅度越界、建议值未变化或参数未登记时均拒绝执行。
- 首次校验失败只允许基于同一 `fact_package` / `fact_package_id` 纠错一次，不增加第三次模型调用。
- 每次模型响应完成 Q1.1/Q1.2 校验后，必须先将该 attempt 写入审计；写入失败立即 fail-closed，不继续模型调用、Guardrails、命令构造或训练。
- 顶层 `decision`、`decision_validation`、`semantic_validation` 保留最终一次结果；`decision_attempts` 保存逐次校验记录，不进入只读 UI/API 投影。

## 2. 已登记事实—参数规则

| 事实 | 参数 | 允许方向 | 最大幅度或附加条件 |
|---|---|---|---|
| `training.issue.overfitting` | `weight_decay` | 增加 | 最多当前值 4 倍；当前值为 0 时上限 0.001 |
| `training.issue.overfitting` | `epochs` | 减少 | 不得低于当前值的 50% |
| `training.curve.val_box_loss`（值为 `rising`） | `weight_decay` | 增加 | 最多当前值 4 倍；当前值为 0 时上限 0.001 |
| `training.curve.val_box_loss`（值为 `rising`） | `epochs` | 减少 | 不得低于当前值的 50% |
| `training.curve.val_cls_loss`（值为 `rising`） | `weight_decay` | 增加 | 最多当前值 4 倍；当前值为 0 时上限 0.001 |
| `training.curve.val_cls_loss`（值为 `rising`） | `epochs` | 减少 | 不得低于当前值的 50% |
| `training.issue.underfitting` | `weight_decay` | 减少 | 不得低于当前值的 25%；当前值必须大于 0 |
| `training.issue.underfitting` | `epochs` | 增加 | 最多当前值 2 倍 |
| `training.issue.plateau` | `lr0` | 减少 | 建议值为当前值的 25%–80% |
| `training.issue.plateau` | `cos_lr` | 开启 | 仅允许 `false → true` |
| `training.curve.mAP50`（值为 `saturated`） | `lr0` | 减少 | 建议值为当前值的 25%–80% |
| `training.curve.mAP50`（值为 `saturated`） | `cos_lr` | 开启 | 仅允许 `false → true` |
| `training.issue.unstable_training` | `lr0` | 减少 | 建议值为当前值的 25%–80% |
| `training.issue.unstable_training` | `warmup_epochs` | 增加 | 最多增加 3 |
| `training.issue.nan_loss` | `lr0` | 减少 | 建议值为当前值的 25%–80% |
| `training.issue.nan_loss` | `warmup_epochs` | 增加 | 最多增加 3 |
| `training.issue.early_stop_too_soon` | `patience` | 增加 | 最多当前值 2 倍；当前值为 0 时上限 20 |
| `dataset.issue.tiny_bbox_high_ratio` | `imgsz` | 增加 | 最多当前值 2 倍 |
| `dataset.issue.tiny_bbox_high_ratio` | `box` | 增加 | 最多当前值 2 倍 |
| `dataset.issue.long_tail_class` | `cls` | 增加 | 最多当前值 2 倍 |
| `dataset.issue.center_spatial_bias` | `translate` | 增加 | 最多增加 0.2 |

未登记的参数—事实关系一律视为不受支持，包括但不限于模型、优化器、批大小以及未列入本表的数据增强参数。它们不能仅凭模型自由解释进入训练。

## 3. 语义错误码

| 错误码 | 含义 |
|---|---|
| `DECISION_SEMANTIC_UNSUPPORTED` | 没有已登记证据支持该参数修改，或建议值等于当前值 |
| `DECISION_SEMANTIC_DIRECTION_CONFLICT` | 修改方向与支持证据规定的方向不一致 |
| `DECISION_SEMANTIC_CHANGE_TOO_LARGE` | 修改幅度超过规则允许范围 |
| `DECISION_SEMANTIC_CURRENT_VALUE_MISSING` | 事实包中缺少该参数的当前值 |
| `DECISION_SEMANTIC_EVIDENCE_CONFLICT` | 同一参数存在相反方向的有效证据 |

## 4. 审计与故障关闭

每条 `decision_attempts` 记录包含完整 `TuningDecision`（Q1.1 失败时为 `null`）、`decision_validation` 和 `semantic_validation`（未到达 Q1.2 时为 `null`），但不保存原始模型响应、凭据或本地路径。

`audit.update_iteration()` 对一次多字段更新执行“字段校验 → 深拷贝快照 → 应用候选值 → 原子 flush → 成功提交 / 失败回滚并重抛”。回调只在写盘成功后推进本地已持久化 attempt 列表，因此后一次写盘失败不会通过共享可变引用或后续 flush 重新进入审计文件。

## 5. 规则维护要求

新增或修改规则时必须同时完成：

1. 修改运行时规则表，并为支持关系、方向、边界值、越界值和冲突证据补齐单元测试。
2. 验证初始提示和纠错提示只描述规则，不由系统提供或猜测具体替代值。
3. 验证每次响应先审计后继续，任意 attempt 写入失败均保持零放行。
4. 运行 Q1.1/Q1.2 聚合测试、完整测试套件以及必要的最小 CUDA 短训练。
5. 同步更新本文、交接记录、路线图、实施计划和研发执行版 DOCX。

## 6. 2026-09-02 验收基线

- 定向审计与循环测试：71 passed。
- Q1.1/Q1.2 聚合测试：183 passed。
- 完整套件：1484 passed，2 个 sklearn PCA warning，0 skipped。
- 合法语义决策：完成一次最小 Detect CUDA 1 epoch 短训练，实际参数与审计一致。
- 非法语义决策：无支持关系、错误方向、幅度过大三类场景均完成唯一一次纠错后失败；共六次模型响应，命令构造 0、训练启动 0。
- Studio 页面：历史记录可读取，语义校验列可见，旧审计显示“未执行语义校验”。

当前状态：代码与自动化/短训练/UI 验收已由 Codex 通过，等待艾卡线下体验确认后再进行 Git 提交与推送。
