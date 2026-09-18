# Bugfix P2 参考训练与数据集快照绑定实施计划

> **Claude Code 执行要求：** 严格按本计划逐项 TDD。不得修改项目文档，不得执行任何 Git 操作；完成后停止并提交交付报告，等待 Codex 独立验收。

**目标：** 自动调优始终使用所选参考训练实际绑定的数据集快照，任何无法确认、歧义或冲突均在启动训练前失败，绝不静默使用 `latest_dataset`。

**架构：** 在 `dataset_snapshot` 与 `local_index` 之上新增单一的参考数据集解析服务。UI 启动端点先解析并冻结 `ReferenceDatasetResolution`，再将同一身份传入调优循环、训练命令、审计、历史和 SQLite；循环内部不再读取全局最新数据集。

**技术栈：** Python 3.10、FastAPI、SQLite、PyYAML、pytest；只使用现有依赖。

**规格：** `docs/bugfix_p2_reference_dataset_binding_spec_20260827.md`

## 全局约束

- 使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，测试首行记录 `sys.executable` 与 Python 版本。
- 不修改 SQLite Schema，不改变 S1.2 快照校验算法、P1 TXT、S1.5 状态/SSE 和旧 JSON 可读性。
- 不读取 `latest_dataset` 作为参考训练数据集的候选或回退来源。
- 不新增依赖、不删除文件、不写入真实数据集、权重、训练产物或真实配置。
- API/SSE 错误与成功投影不得泄漏绝对路径、SQL、原生异常或 Traceback。
- 本批结束不执行 `git add`、`git commit`、`git push`、PR 或合并。

---

### Task 1：冻结解析模型、错误和 SQLite 查询边界

**文件：**

- 新增：`auto_tune/modules/reference_dataset/__init__.py`
- 新增：`auto_tune/modules/reference_dataset/models.py`
- 修改：`auto_tune/modules/local_index/repository.py`
- 修改：`auto_tune/modules/local_index/service.py`
- 新增测试：`auto_tune/tests/test_reference_dataset_resolution.py`

**接口：**

```python
@dataclass(frozen=True)
class ReferenceDatasetResolution:
    reference_run: str
    dataset_id: str
    snapshot_id: str
    data_yaml_path: Path
    resolution_source: Literal["sqlite", "reference_args"]
    dataset_display_name: str | None = None
    index_warning: str | None = None

class ReferenceDatasetError(Exception):
    error_code: str
```

仓储增加参数化查询：

```python
LocalIndexRepository.list_experiments_by_run_name(run_name: str) -> list[dict]
LocalIndexService.find_reference_experiments(run_name: str) -> list[dict]
```

- [ ] 先写失败测试：唯一非空 `dataset_id` 成功；多个不同 ID 返回歧义；空关联、数据集记录缺失和存储异常不被伪装成成功。
- [ ] 运行新增测试，保存 RED 证据。
- [ ] 最小实现模型、稳定错误类和参数化查询；不得通过字符串拼接 SQL。
- [ ] 运行新增测试至 GREEN。

### Task 2：实现参考快照解析与严格验证

**文件：**

- 新增：`auto_tune/modules/reference_dataset/service.py`
- 修改：`auto_tune/modules/reference_dataset/__init__.py`
- 修改测试：`auto_tune/tests/test_reference_dataset_resolution.py`

**核心入口：**

```python
def resolve_reference_dataset(
    reference_run: str,
    detect_dir: Path,
    log_dir: Path,
    local_index_service: LocalIndexService | None,
) -> ReferenceDatasetResolution:
    ...
```

**确定性规则：**

1. 校验 `reference_run` 为单个安全目录名，且 `detect/<reference_run>` 存在。
2. SQLite 唯一关联成功后，验证其 dataset、snapshot、`data.yaml` 与 manifest。
3. SQLite 无关联或不可用时，读取参考目录 `args.yaml` 的 `data`；只接受位于 `log/dataset_snapshots/<snapshot_id>/data.yaml` 的受控路径。
4. 复用 S1.2 现有 manifest、digest、路径边界及 reparse-point 校验，不复制一套宽松校验。
5. SQLite 与合法 `args.yaml` 同时存在但身份冲突时返回 `REFERENCE_DATASET_AMBIGUOUS` 或 `REFERENCE_SNAPSHOT_INVALID`，不得选一方静默继续。
6. 回退成功后可幂等补录索引；补录失败只写稳定 `index_warning`，不得改变已验证快照。

- [ ] 写失败测试：合法 SQLite、合法 args 回退、双方一致、双方冲突、同名歧义、manifest 损坏、digest 不一致、越界、symlink/junction/reparse point、数据库损坏。
- [ ] 增加判别性测试：人为设置不同的 `latest_dataset`，解析结果仍必须是参考快照；删除参考关联后必须失败，不能返回 latest。
- [ ] 运行测试并保存 RED 证据。
- [ ] 实现最小解析服务和稳定错误码。
- [ ] 运行测试至 GREEN，并确认异常消息不含 `tmp_path`、绝对路径或原生异常正文。

### Task 3：在调优启动前冻结解析结果

**文件：**

- 修改：`auto_tune/ui/app.py`
- 修改：`auto_tune/modules/agent_engine/loop.py`
- 新增测试：`auto_tune/tests/test_reference_dataset_tuning_api.py`
- 修改测试：`auto_tune/tests/test_tuning_loop.py`

**接口流向：**

```text
POST /tuning/start
  -> 确定唯一 reference_run
  -> resolve_reference_dataset(...)
  -> run_tuning_loop(..., reference_dataset=resolution)
  -> training.data_yaml = resolution.data_yaml_path
```

- [ ] 写失败测试，证明当前 `start_tuning` 会被 `latest_dataset` 覆盖。
- [ ] 测试所有解析错误均在创建 controller/YOLO 子进程前返回稳定 400/409/503；响应只含错误码和脱敏显示信息。
- [ ] 删除该入口中 `_read_latest_dataset()` 和 `_resolve_validated_snapshot_data_yaml(latest)` 对调优训练数据的覆盖行为。
- [ ] `run_tuning_loop` 接收冻结的 `ReferenceDatasetResolution`；dry-run 可以展示解析计划，但未解析状态不得标记为可执行。
- [ ] 无 `reference_run` 时必须先按既有规则确定唯一参考运行，再调用同一解析服务；没有唯一值时失败。
- [ ] 运行 API 与 loop 定向测试至 GREEN。

### Task 4：统一训练事实、审计、历史与 SQLite 关联

**文件：**

- 修改：`auto_tune/modules/agent_engine/loop.py`
- 修改：`auto_tune/modules/agent_engine/audit.py`
- 必要时修改：`auto_tune/modules/train_analyzer/training_finalizer.py`
- 新增测试：`auto_tune/tests/test_reference_dataset_end_to_end.py`
- 回归：`auto_tune/tests/test_local_index_integration.py`

- [ ] 写端到端失败测试：参考快照 A、全局 latest B；决策后的训练命令、`execution.actual_params.data`、新运行 `args.yaml`、审计、JSON 历史和 SQLite `dataset_id` 必须全部为 A。
- [ ] 测试 Guardrails 和 Module B 使用相同 `reference_run`/snapshot 身份，不允许在循环中重新解析 latest。
- [ ] 只在内部审计保存完整 `dataset_id`、`snapshot_id` 和解析来源；公开投影只显示名称/短 ID。
- [ ] 索引写失败仍不得改写已完成训练事实，但必须产生稳定 `index_warning/index_error`。
- [ ] 实现最小透传和对账，运行端到端及 local-index 回归至 GREEN。

### Task 5：启动确认 UI 与出站脱敏

**文件：**

- 修改：`auto_tune/ui/templates/single_page.html`
- 修改：`auto_tune/ui/i18n.py`
- 修改：`auto_tune/ui/app.py`
- 新增测试：`auto_tune/tests/test_reference_dataset_ui.py`

启动前或启动响应必须向用户展示：参考训练、关联数据集、快照短标识、解析来源。中英文文案必须通过现有 i18n；不得输出内部绝对路径。

- [ ] 先写失败测试覆盖 SQLite 来源、args 回退、歧义、损坏和脱敏。
- [ ] 前端错误分支使用 `textContent` 或既有安全渲染，不新增不可信 `innerHTML`。
- [ ] 解析失败时停止按钮、运行 badge 和 SSE 状态不得伪装成 running。
- [ ] 运行 UI、模板 XSS、S1.5 SSE 回归至 GREEN。

### Task 6：整批回归与最小真实验收准备

**自动化命令：**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_reference_dataset_resolution.py auto_tune\tests\test_reference_dataset_tuning_api.py auto_tune\tests\test_reference_dataset_end_to_end.py auto_tune\tests\test_reference_dataset_ui.py -v -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_local_index_integration.py auto_tune\tests\test_run_state_tuning_api.py auto_tune\tests\test_run_manager_reconnect.py auto_tune\tests\test_final_summary.py auto_tune\tests\test_loop_bugfix_p1.py auto_tune\tests\test_template_xss.py -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

- [ ] 报告三组命令的完整结果、首次 RED 原因、修复过程和新增测试数量。
- [ ] 只读核对 `train52`：解析结果应为 `00915b4c...`，而不是当前 latest 的 `ea4bd249...`。
- [ ] 不运行三轮 300 epoch；由 Codex 后续使用命令截获或最小 1–3 epoch 完成真实验收。
- [ ] 检查无服务器、YOLO 子进程、临时锁、测试 DB、真实 log/detect 产物残留。
- [ ] 输出实际修改文件、接口偏离、Windows/Linux 分支、遗留风险以及“未改文档、未新增依赖、未执行 Git”的明确确认，然后停止。

## 交付门

Claude Code 的完整测试通过不等于验收通过。Codex 将独立审查以下事实：

1. 调优入口已完全移除 `latest_dataset` 静默覆盖；
2. `train52` 的解析快照为 `00915b4c...`；
3. 命令、args、审计、历史与 SQLite 身份一致；
4. P1 TXT、S1.5 重连和 S2 查询无回归；
5. 艾卡人工确认前不进行 Git 提交或推送。
