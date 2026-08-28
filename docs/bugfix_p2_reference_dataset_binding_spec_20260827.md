# Bugfix P2：参考训练与数据集快照绑定规格（2026-08-27）

## 1. 结论与优先级

P2 是进入后续体验功能前必须完成的训练正确性修复。当前自动调优虽然以 `reference_run` 读取参考运行的参数、指标和 Module B 报告，但非 dry-run 启动时仍会用全局 `log/latest_dataset.json` 覆盖参考运行 `args.yaml` 中的 `data`。因此，历史运行可能在另一个数据集快照上被重新训练，形成“参考指标属于数据集 A，调优训练实际使用数据集 B”的错误比较。

P2 完成前，不使用与当前 `latest_dataset` 不同快照的历史运行执行正式调优；至少完成 P2 编码、Codex 独立验收和艾卡人工确认后，才进入 GitHub 提交与推送检查。

## 2. 已确认的真实证据

2026-08-27 的三轮自动调优以 `train52` 为参考运行：

- `train52/args.yaml` 记录的原快照 ID：`00915b4ce7998041848b614be2fbfb10a4ef8497512c3dd4650de48886a1d8be`；
- 当时全局 `latest_dataset` 快照 ID：`ea4bd249f66ea5cea156e37bbe647adc86e623cebaab1c9bd5e1bf71d521a7d7`；
- 三轮 `autotune_bfb10de5_iter01`、`iter02`、`iter03` 的实际 `args.yaml` 和审计命令均使用 `ea4bd249.../data.yaml`；
- SQLite 最终实验记录也关联到 `ea4bd249...`，而不是 `train52` 的原快照。

该会话可用于证明 P1 的终局 TXT 稳定保存，但不能作为 `train52` 原数据集上的严格调优效果证据。

## 3. P2 目标

当用户选择 `reference_run=train52` 等历史训练时，系统必须确定性解析并验证该参考运行对应的数据集快照，然后将同一个快照用于：

1. 参考参数与参考指标；
2. Module B 感知报告；
3. Guardrails 数据集事实；
4. 每轮实际训练命令和 `args.yaml`；
5. 审计记录、JSON 历史和 SQLite 实验关联；
6. 启动前 UI 确认信息。

任何环节无法确认数据集身份时必须拒绝启动，不能静默使用 `latest_dataset`。

## 4. 数据源优先级与解析规则

### 4.1 第一优先级：SQLite 实验关联

按参考运行名查询实验记录，并取得 `dataset_id`：

```text
reference_run
  -> experiments.run_name
  -> experiments.dataset_id
  -> datasets.snapshot_id / data_yaml_path
```

规则：

- 所有同名有效记录指向同一个非空 `dataset_id` 时可以继续；
- 同名记录对应多个不同 `dataset_id` 时返回歧义错误；
- 数据集记录不存在、快照 ID 缺失或路径缺失时进入安全回退；
- SQLite 是可重建索引，查询失败不能直接信任错误记录，也不能改用 `latest_dataset`。

### 4.2 第二优先级：参考运行 `args.yaml` 安全回退

SQLite 不可用或旧实验未建立关联时，可以读取：

```text
detect/<reference_run>/args.yaml -> data
```

只有同时满足以下条件才允许回退：

- `data` 指向项目受控的 `log/dataset_snapshots/<snapshot_id>/data.yaml`；
- 路径规范化后仍位于快照根目录内；
- 不包含符号链接、junction 或 reparse point 逃逸；
- 对应 manifest 通过 S1.2 现有严格校验；
- manifest 与 `data.yaml` 的快照身份一致。

验证成功后可以对 SQLite 索引执行幂等补录，但补录失败不得改变已经确认的快照事实。

### 4.3 禁止的回退

以下行为全部禁止：

- 用 `latest_dataset` 替代参考运行的数据集；
- 用 Module B 全局最新报告推测数据集；
- 使用未经快照验证的外部 `data.yaml`；
- 仅按路径字符串相似判断两个数据集相同；
- 数据库损坏或歧义时继续启动训练；
- 为修复关联而复制、移动或改写原始数据集。

## 5. 建议接口边界

新增单一服务入口，例如：

```python
resolve_reference_dataset(
    reference_run: str,
    detect_dir: Path,
    log_dir: Path,
    local_index_service: LocalIndexService | None,
) -> ReferenceDatasetResolution
```

返回值至少包含：

- `reference_run`；
- `dataset_id`；
- `snapshot_id`；
- 已验证的 `data_yaml_path`；
- `resolution_source`：`sqlite` 或 `reference_args`；
- `index_warning`：可选的非致命补录错误。

该服务只负责解析和验证，不启动训练、不修改共享 `APP_CONFIG`、不读取全局最新数据集作为替代来源。

## 6. 稳定错误码

建议使用以下稳定错误码并映射为启动前 400/409/503 响应：

- `REFERENCE_DATASET_UNRESOLVED`：没有可验证的数据集关联；
- `REFERENCE_DATASET_AMBIGUOUS`：同名参考运行关联多个不同数据集；
- `REFERENCE_SNAPSHOT_INVALID`：快照或 manifest 损坏、缺失或越界；
- `REFERENCE_RUN_INVALID`：参考运行目录或必要事实文件无效；
- `LOCAL_INDEX_UNAVAILABLE`：仅当数据库故障且安全回退也无法完成时作为附加诊断。

错误响应不得包含本机绝对路径、SQL、Traceback 或原生异常正文。

## 7. UI 与审计要求

启动自动调优前，页面至少显示：

```text
参考训练：train52
关联数据集：<显示名称或 dataset_id 短标识>
快照：<snapshot_id 短标识>
解析来源：SQLite / 参考运行快照
```

实际审计必须保存完整内部身份，但公开 SSE/API 只返回脱敏后的显示字段。每轮 `execution.actual_params.data`、命令、`args.yaml`、最终 SQLite `dataset_id` 必须与解析结果一致。

## 8. 兼容边界

- 不升级 SQLite Schema，除非实现审查证明现有 `datasets` 和 `experiments.dataset_id` 无法表达该关联；
- 保持旧 JSON、旧审计和旧 `args.yaml` 可读；
- 不改变 P1 的终局 TXT 保存契约；
- 不改变 S1.2 快照生成和校验算法；
- 不新增依赖；
- dry-run 可以返回解析计划，但不得把未解析数据集表示为可执行；
- 无 `reference_run` 的自动检测必须先得到唯一参考运行，再执行同一解析流程。

## 9. 测试与验收

### 9.1 自动化测试

至少覆盖：

1. SQLite 精确关联成功；
2. SQLite 缺失时通过合法参考快照回退；
3. SQLite 与 `args.yaml` 指向同一快照；
4. SQLite 与 `args.yaml` 冲突时保守拒绝或按明确可信来源处理并记录冲突，不能静默继续；
5. 同名运行多个不同 `dataset_id` 时拒绝；
6. 空 `dataset_id`、缺失数据集记录和数据库损坏；
7. manifest 损坏、digest 不一致、路径逃逸和 reparse point；
8. `latest_dataset` 与参考快照不同时仍使用参考快照；
9. 实际命令、`args.yaml`、审计、历史和 SQLite 关联一致；
10. 错误出站脱敏；
11. P1 TXT、S1.5 状态和 S2 查询回归不受影响。

### 9.2 真实验收

不再执行三轮 300 epoch。使用以下方式验收：

- 只读比较 `train52` 原快照和解析结果；
- 使用最小合法数据集进行 1–3 epoch 冒烟，或在启动 YOLO 前截获并核对唯一命令；
- 确认新运行 `args.yaml` 的 `data` 为 `00915b4c.../data.yaml`，而不是当前 `latest_dataset`；
- 确认 SQLite、审计和 JSON 历史关联同一 `dataset_id`；
- Chromium 检查启动前的数据集/快照提示与错误状态。

## 10. 发布门

P2 实现后先由 Claude Code 提供改动与测试报告，不执行 Git 操作。Codex完成代码审查、自动化测试和最小真实验收后，进入艾卡人工确认。至少完成该确认后，才整理待提交源码、测试和当前文档，并检查 `.gitignore`、大文件、数据集、权重、日志、审计和凭据，再决定是否提交与推送 GitHub。

## 11. 后续顺序

- P3：智能分析接入 SQLite，在训练路径输入框下展示最近 4 个训练结果并可点击回填；
- P4：数据库和实验详情字段的中英文动态翻译；
- P5：检查报告完善，形成可下载的训练总结报告；
- 后续专项：AI 输出可信度，统一确定性事实、提示词约束、视觉矩阵语义、最佳/最终轮次口径和机器校验。
