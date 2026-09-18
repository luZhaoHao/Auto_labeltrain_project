# F1.1-A Codex 第二轮复审返修提示词

> 当前版本管理口径：本子批次结果保留在本地，不单独提交或推送；待 F1.1 整体验收后统一处理 GitHub 上传。

你正在处理 F1.1-A 的第二轮 Codex 独立复审问题。第一轮返修的四项核心实现已经通过独立测试，本轮只修复下列 3 个遗漏，不重做已经通过的权重库、mtime、LLM 注册表、页面布局、上传界面或 HPO 进度功能。

## 一、开始前必须完整阅读

按顺序完整阅读：

1. `AGENTS.md`
2. `docs/development_handoff_20260814.md`
3. `docs/f1_1_a_experience_model_store_spec_20260917.md`
4. `docs/superpowers/plans/2026-09-17-f1-1-a-experience-model-store.md`
5. 当前 F1.1-A 实现与对应测试

重点检查：

- `auto_tune/modules/agent_engine/decision_agent.py`
- `auto_tune/modules/agent_engine/decision_contract.py`
- `auto_tune/modules/agent_engine/decision_facts.py`
- `auto_tune/modules/agent_engine/parameter_registry.py`
- `auto_tune/modules/model_store/service.py`
- `auto_tune/ui/app.py`
- 对应 pytest 文件

## 二、严格边界

- 只修改业务代码和对应测试。
- 不得修改任何项目文档，包括本提示词、规格、计划、README、路线图、交接记录和 CLAUDE.md。
- 不新增依赖，不删除文件，不提交 Git，不推送 GitHub。
- 工作区有前序 F1.1-A 改动和更早脏文件；不得回退、覆盖、清理或顺手格式化无关内容。
- 所有 Python、pytest 和项目脚本必须使用：

```text
D:\Program Files\anaconda3\envs\auto_tune\python.exe
```

- 每项必须先增加准确复现问题的测试并运行得到真实 RED，再做最小实现转 GREEN。
- 不得削弱或删除既有测试；第一轮已经通过的 4 项规则必须继续保留：
  1. managed/legacy 同哈希时 managed 优先且身份唯一；
  2. `model` 不属于 LLM 可调参数，但参考模型事实和最终执行参数继续保留；
  3. 新 HPO 研究冻结 `model_mtime_ns`，旧记录兼容；
  4. `ModelStore.resolve_configured_name()` 只接受纯 basename。
- 不运行真实训练，不调用网络 LLM，不下载或反序列化模型。
- 不声称 F1.1-A、浏览器或 GPU 验收通过；完成后等待 Codex 第三轮独立复验。

## 三、本轮只修复以下 3 项

### Task 1 — P1：LLM 提示词仍诱导“换更大模型”

#### 已复现事实

虽然 `model` 已从 `PARAMETER_REGISTRY` 和允许参数列表移除，但旧决策提示词的欠拟合规则仍包含：

```text
规则8：欠拟合（所有指标偏低）
动作：换更大模型, imgsz 提升, lr0 适当提高, 增加 epochs
```

独立探针结果：

```text
contains_change_larger_model=True
```

这会诱导 LLM 按提示输出 `model`，随后结构契约又以未知参数拒绝，导致整轮调优失败。不能把它留到 F1.1-B。

#### 冻结要求

- 删除所有要求、建议或暗示 LLM 更换模型/权重的调优动作。
- 欠拟合规则只能建议当前允许且有语义规则支撑的参数，例如 `imgsz`、`box`、`lr0`、`epochs`；具体方向必须与现有 semantic rules 一致。
- 两套仍在使用的决策提示词都要明确：模型/初始权重属于参考运行的不可变条件，禁止写入 `hyperparameter_changes` 或 `training_overrides`。
- 参考模型仍可作为只读事实 `training.params.model` 进入真实 FactPackage；不得为了让提示词测试“看不到 model”而删除该事实。
- `model` 继续不在 `get_tunable_parameter_names()` 中。
- LLM 违规返回 `model` 时继续在结构契约边界以 `DECISION_SCHEMA_INVALID` 拒绝，零训练启动。
- 不修改 `/tuning/start` 的 `LLM_MODEL_OVERRIDE_FORBIDDEN`。

#### 必须先写的 RED 测试

至少覆盖：

1. `build_decision_prompt()` 不再包含“换更大模型”“更换模型”“切换权重”等正向调优动作。
2. 欠拟合规则只列出允许参数，并明确模型/初始权重不可修改。
3. 构造包含真实 `training.params.model` 事实的 FactPackage，验证结构化提示词：
   - 可以展示当前参考模型这一只读事实；
   - 允许参数列表不含 `model`；
   - 明确禁止在两个修改对象中输出 `model`。
4. 旧提示词与结构化提示词均不存在任何“建议换模型”的语义。
5. `parse_tuning_decision_response()` 对 model 修改仍在结构契约层拒绝。
6. `sanitize_and_merge_tuning_params()` 仍满足：

```python
merged["model"] == reference_args["model"]
```

不要使用仅搜索英文 token 的弱测试；必须覆盖中文“换更大模型”等真实文案。

### Task 2 — P1：HPO 默认绑定仍绕过纯 basename 校验

#### 已复现事实

`ModelStore.resolve_configured_name()` 已正确拒绝完整路径，但 `auto_tune/ui/app.py::_model_row_by_name()` 仍执行：

```python
candidate = Path(name.strip()).name
```

因此 HPO defaults 维护了第二套、更宽松的配置解析。独立探针在目标文件真实存在时得到：

```text
{
  'model_id': 'sha256:...',
  'name': 'yolov8n.pt',
  'source': 'project.model',
  'available': True,
  'reason_code': 'MODEL_BOUND'
}
```

也就是完整绝对路径仍被静默截成 basename，并被 HPO 认作合法默认绑定。

#### 冻结要求

- HPO 默认绑定与直接训练必须复用 `ModelStore` 的同一套配置名称校验。
- `app.py` 不得继续用 `Path(...).name`、`os.path.basename()` 或类似方式维护第二套“截断后匹配”逻辑。
- `project.model` 或 `training.model` 只有纯 basename（例如 `yolov8n.pt`）才能形成 `MODEL_BOUND`。
- Windows 绝对路径、POSIX 绝对路径、UNC、驱动器相对路径、含 `/` 或 `\` 的相对路径、首尾空白、控制字符均必须投影为：

```text
available=False
reason_code=MODEL_CONFIG_INVALID
model_id=None
```

- 即使受控库中真实存在相同 basename 的文件，也不得把非法配置路径静默改绑过去。
- 合法 basename 继续保持 managed 优先、legacy 兼容、不泄露路径、不联网下载。
- 不改变 HPO 创建接口只接受 `model_id` 的规则。

#### 实现约束

优先在 `ModelStore` 提供一个返回 `ModelRecord` 的统一配置解析入口，再让：

- `resolve_configured_name()` 返回该记录的 path；
- HPO defaults 使用同一记录的 `model_id`/`name`。

或者让 `app.py` 先调用现有 `resolve_configured_name()`，再从同一轮安全列表中精确获取对应记录。无论采用哪种方式，都不得复制路径合法性规则。

#### 必须先写的 RED 测试

在权重文件真实存在的前提下，至少覆盖 `_default_model_binding()` 或真实 `/api/hpo/defaults`：

```text
C:\models\yolov8n.pt
\\server\share\yolov8n.pt
/srv/models/yolov8n.pt
models/weights/yolov8n.pt
models\weights\yolov8n.pt
C:yolov8n.pt
../yolov8n.pt
 yolov8n.pt
```

全部必须得到 `MODEL_CONFIG_INVALID`，不能只测试底层 `ModelStore`。同时覆盖合法 managed basename、合法 legacy basename和缺失 basename。

### Task 3 — P2：`_known` 未按最新公开列表整体收敛

#### 已复现事实

`list_models()` 当前只执行：

```python
for row in rows:
    if row.available:
        self._known[row.model_id] = row
```

不会移除被 `max_models` 截断、已删除、变为不可读或不再出现在最新公开列表中的旧 ID。

独立探针：

```text
max_models=1
first=b.pt
新增排序更前的 a.pt
second=a.pt
stale_resolve=b.pt
```

即最新公开列表只显示 `a.pt`，但旧 `b.pt` 仍能通过 ID 解析。实现注释所称“`_known` 与公开列表逐条一致”并不成立。

#### 冻结要求

- 每次 `list_models()` 成功形成最终公开列表后，`_known` 必须在锁内整体替换为本轮所有 `available=True` 且有合法 `model_id` 的记录。
- 被 `max_models` 截断、已删除、不可读、链接化或不再公开的旧 ID，在刷新列表后必须返回 `MODEL_NOT_FOUND`。
- 当前公开列表中的每条记录仍必须满足 `resolve(row.model_id) == row.path`。
- `import_stream()` 成功后仍应立即登记刚上传的记录；后续列表刷新再按当前公开事实整体收敛。
- 不删除磁盘文件，不改变哈希 ID、managed 优先级或上传幂等/并发规则。
- 列表刷新与上传并发时不得产生字典部分更新；所有 `_known` 变更都必须在既有锁内完成。

#### 必须先写的 RED 测试

至少覆盖：

1. `max_models=1`：旧 ID 被新排序记录挤出后，刷新列表，再 resolve 旧 ID → `MODEL_NOT_FOUND`。
2. 文件删除后调用 `list_models()`，旧 ID 从 `_known` 移除。
3. 文件变成链接/reparse 或不可读后刷新列表，旧 ID 不再可解析。
4. 根目录从有记录变为空后刷新，旧 ID 不再可解析。
5. 当前公开列表所有 ID 仍逐条解析到对应 path。
6. managed/legacy 同哈希唯一身份、managed 优先、并发上传和同名冲突测试保持通过。

## 四、执行顺序与 RED→GREEN

严格按 Task 1 → Task 2 → Task 3 执行：

1. 先增加本 Task 测试。
2. 使用未修改的业务代码运行并取得真实 RED。
3. 记录失败数和关键断言。
4. 做最小实现。
5. 用相同聚焦命令转 GREEN。
6. 运行相关模块完整测试后再进入下一 Task。

建议聚焦命令：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_agent.py auto_tune\tests\test_decision_contract.py auto_tune\tests\test_tuning_loop.py -k "model or prompt or inherits" -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_api.py -k "default and model" -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py -k "known or stale or max_models or identical" -q -p no:cacheprovider
```

若实际测试名不同，可调整 `-k`，但必须在交付报告中给出真实命令和完整结果。

## 五、最终回归要求

三项 GREEN 后，至少运行：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py auto_tune\tests\test_model_store_api.py auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_lifecycle.py auto_tune\tests\test_hpo_formal_training.py auto_tune\tests\test_hpo_artifacts.py auto_tune\tests\test_hpo_studio_concurrency.py -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_decision_contract.py auto_tune\tests\test_decision_agent.py auto_tune\tests\test_decision_facts.py auto_tune\tests\test_decision_semantics.py auto_tune\tests\test_semantic_rules.py auto_tune\tests\test_guardrails.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_run_manager_reconnect.py auto_tune\tests\test_run_state_training_api.py auto_tune\tests\test_training_gate.py -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pip check

node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
```

同时检查：

- 本轮实际修改文件无行尾空白；
- 无 `<<<<<<<`、`=======`、`>>>>>>>`；
- `git diff --check` 如果只命中既有无关文件，必须列出文件和证据，不得擅自修改；
- 本提示词、规格、计划及其他项目文档保持不变；
- 不得把自动化测试冒充浏览器、GPU 或真实 LLM 验收。

## 六、最终交付报告

报告必须包含：

1. 三项根因和最终规则；
2. 实际修改文件及职责；
3. 每项真实 RED→GREEN 命令、失败断言和最终结果；
4. 中文“换更大模型”已从调优建议移除的证据；
5. 含真实 `training.params.model` 事实时，提示词仍明确禁止修改模型的证据；
6. HPO defaults 对真实存在目标文件的完整路径反例结果；
7. `_known` 截断、删除、链接/不可读及空列表收敛证据；
8. 定向、全量、pip check、node check 的精确结果；
9. 偏离项；
10. 剩余风险；
11. 完整 `git status --short`，区分本轮、F1.1-A 前序和更早脏工作区；
12. 明确声明未新增依赖、未删除项目文件、未修改项目文档、未真实训练、未调用网络 LLM、未提交、未推送。

完成后停止，等待 Codex 第三轮独立复验。
