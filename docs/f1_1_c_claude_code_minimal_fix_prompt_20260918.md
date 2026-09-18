# F1.1-C Claude Code 最小修复提示词

> 将本文件全文交给 Claude Code 执行。当前批次只关闭 Codex 审查确认的两项 P1 和三处确定的代码/配置整理问题。不要扩大范围，不提交或推送 GitHub。

你正在处理 Auto-Tune 项目 F1.1-C 稳定冻结审查后的最小修复批次。艾卡要求以软件稳定性为优先，不进行大改、重构或顺手清理。Codex 已完成只读审查和独立全量测试，审查记录为 `docs/f1_1_c_codex_review_20260918.md`；当前基线测试为 `2732 passed, 2 warnings`。

## 一、开始前必须阅读

按顺序完整阅读：

1. `AGENTS.md`
2. `docs/development_handoff_20260814.md`
3. `docs/f1_1_a_experience_model_store_spec_20260917.md`
4. `docs/superpowers/plans/2026-09-17-f1-1-a-experience-model-store.md`
5. `docs/f1_1_c_codex_review_20260918.md`
6. 本批涉及的现有实现和测试

重点文件：

- `auto_tune/ui/templates/single_page.html`
- `auto_tune/ui/templates/agent_suggestion.html`
- `auto_tune/tests/test_hpo_ui_behaviour.py`
- `auto_tune/tests/test_hpo_ui_rework.py`
- `auto_tune/tests/test_hpo_ui_lifecycle.py`
- `auto_tune/tests/js/minidom.js`
- `auto_tune/modules/agent_engine/loop.py`，仅用于理解现有返回契约，不应修改
- `auto_tune/tests/test_tuning_loop.py`，仅用于理解后端已覆盖的基线语义
- `_final_test.py`
- `test_modelb.py`
- `auto_tune/config.template.yaml`
- `auto_tune/main.py`

## 二、严格边界

- 只修改本提示词列出的业务代码、配置模板和对应测试。
- 不修改任何项目文档，包括本提示词、审查记录、README、路线图、交接记录、实施计划、DOCX 和 CLAUDE.md。
- 不处理 README 破图、当前删除的 `img/` 文件或 `PROJECT_REVIEW.md`；不恢复、不删除、不移动这些文件。这部分由 Codex 在最终发布整理时处理。
- 不修改 `.gitignore`，不删除 `.tmp_h1_render/`、`.tmp_h1_*.ps1`、`papper/` 或其他临时、参考文件；由 Codex 在 F1.1 统一提交前处理。
- 不拆分或重写 `single_page.html`、`app.py`、HPO 前端状态机或训练循环。
- 不改变训练、HPO、LLM、评分、审计、持久化、运行身份或 API 协议。
- 不新增依赖，不升级依赖，不删除项目文件，不格式化无关文件。
- 工作区包含 F1.1 前序和文档改动；不得 reset、checkout、clean、stash、覆盖或回退无关内容。
- 不运行真实训练，不调用网络 LLM，不下载模型，不处理数据集或权重。
- 不提交 Git，不推送 GitHub，不声称 F1.1 已验收或冻结；完成后等待 Codex 独立复验。
- 所有 Python 和 pytest 命令必须使用：

```text
D:\Program Files\anaconda3\envs\auto_tune\python.exe
```

- 开始测试前先报告：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -c "import sys; print(sys.executable); print(sys.version)"
```

## 三、Task 1：恢复已批准的 `dry_run` 前端模式

### 已确认根因

F1.1-A 已批准规格要求页面提供以下四种完整模式值：

```text
dry_run
keep_params
hpo
full
```

当前 `single_page.html` 和 `agent_suggestion.html` 主动把页面改成三种模式；`test_hpo_ui_behaviour.py`、`test_hpo_ui_rework.py` 和相关 minidom 场景还将“页面不得存在 dry_run”固化成测试。后端 `/tuning/start` 仍支持 `dry_run`，不需要改后端。

### 冻结要求

- 在两个仍可能渲染的模式选择器中恢复 `<option value="dry_run">`。
- 模式顺序固定为：`dry_run`、`keep_params`、`hpo`、`full`。
- `dry_run`、`keep_params`、`full` 继续复用现有 `startTuningBtn`；`hpo` 继续只使用 `hpoCreateAndStartBtn`。
- `FormData(tuningForm).get("mode")` 切换四种模式时必须分别得到完整值。
- `dry_run` 提交 `/tuning/start`，不得启动正式训练；继续使用现有后端兼容语义。
- 恢复干运行的可见说明和主操作区域，但不要重新设计页面布局。
- 删除“页面只提供三种模式”“干运行已移除”等与批准规格冲突的注释和反向断言。
- 不修改 `/tuning/start` 后端实现，不改变其他三种模式行为。

### RED→GREEN 要求

先把现有反向测试改回批准规格，并在未恢复模板前得到真实 RED。至少覆盖：

1. 两个模板均包含四个选项且顺序正确；
2. 不通过 CSS 隐藏 `dry_run`；
3. 真实执行 `FormData(tuningForm)` 后四种值完整；
4. LLM 路由请求包含 `dry_run`、`keep_params`、`full`；
5. HPO 仍走独立按钮和既有创建流程；
6. 后端继续接受 `dry_run`。

不得通过弱化断言、删除行为测试或只检查字符串来获得 GREEN；至少保留一个 minidom 行为测试。

## 四、Task 2：正确展示“原参考基线仍为总体最佳”

### 已确认根因

后端返回契约已经明确：

- `best_iteration`、`best_train_name`、`best_metrics` 表示成功调优轮次中的最佳结果；
- `kept_reference_baseline === true` 表示没有调优轮次严格超过会话开始时的原参考训练；
- 此时总体最佳仍是 `baseline_run`，并可从 `reference_baseline.run_name`、`reference_baseline.score` 读取原参考信息；
- 消费端不得把 `best_train_name` 显示成总体最佳。

当前 `single_page.html` 完成态没有检查 `kept_reference_baseline`，无条件显示“Best Iteration”，并把查看、保存按钮绑定到 `best_train_name`。

### 冻结要求

当 `result.kept_reference_baseline === true`：

- 主结果卡明确显示“总体最佳：原参考训练”或等价、不会误导的中英文文案；
- 总体最佳运行名优先使用 `result.reference_baseline.run_name`，缺失时回退 `result.baseline_run`；
- 总体评分使用 `result.reference_baseline.score`，缺失时可用 `result.baseline_score`；缺少分数时显示 `-`，不得伪造 `0`；
- “查看完整分析”按钮的 `data-train-name` 和实际调用目标绑定到原参考运行；
- `saveReportBtn` 的 `data-train-name` 绑定到原参考运行；
- `best_iteration`、`best_train_name`、`best_metrics` 可以作为“本次调优轮次中最佳”次级信息展示，不能再使用总体最佳标题；
- 对运行名和动态文本继续使用现有安全编码/渲染方式，不引入未转义 HTML。

当 `result.kept_reference_baseline !== true`：

- 保持当前调优最佳轮次的展示、评分计算、查看和保存行为；
- 不改变完成态、停止态或失败态的其他逻辑。

### RED→GREEN 要求

先增加真实执行完成态渲染逻辑的前端/minidom 测试，并在未改模板前取得真实 RED。至少覆盖：

1. `kept_reference_baseline=true` 时主卡文案指向原参考训练；
2. 查看按钮和保存按钮的 `data-train-name` 均等于原参考运行名；
3. 原参考分数为 `0.90` 时显示真实分数；分数为 `null` 时显示 `-`；
4. 调优轮次最佳如果展示，必须标为次级比较结果；
5. `kept_reference_baseline=false` 时仍绑定 `best_train_name`；
6. 不改变后端 `loop.py` 现有字段及 TXT 汇总逻辑。

不要只用源码字符串搜索代替行为测试。

## 五、Task 3：跨机器解释器与公开样例脱敏

这是确定性的发布整理，只做最小改动，不需要增加与实现同义的单元测试。

### 3.1 `_final_test.py`

- 将写死的本机路径：

```python
conda_python = r"D:\Program Files\anaconda3\envs\auto_tune\python.exe"
```

恢复为：

```python
# Use the current Python interpreter (must be run in the auto_tune conda env)
conda_python = sys.executable
```

- 不改变该脚本的 SSE 检查流程。
- 不实际启动该脚本；本批不需要启动服务器。

### 3.2 `test_modelb.py`

- 将具体产线、检测对象和缺陷类型等业务内容恢复为通用占位文案。
- 保留字段结构和原有使用说明，不改变脚本执行逻辑。
- 不读取、打印或提交真实 `auto_tune/config.yaml` 内容。

建议使用：

```python
"name": "请在此填写项目名称"
"description": "请在此填写项目描述"
```

其余示例也不得包含客户、产线、真实数据集或项目专用信息。

### 3.3 `auto_tune/config.template.yaml`

- 将 `project.name`、`project.description`、`project.detection_target`、`project.data_type` 改成通用、可公开的工业缺陷检测占位示例。
- 保留本次已经正确更新的 `llm.model: deepseek-flash`。
- 保留 `model_store` 配置及其注释。
- 不修改、不读取、不提交真实 `auto_tune/config.yaml`。
- 不改 API endpoint、credential reference 或其他无关配置。

## 六、Task 4：清理一处确定的尾随空格

- 只清理 `auto_tune/main.py:75` 当前空白行上的尾随空格。
- 不格式化整个文件，不改相邻逻辑。
- 如果实际行号因工作区变化而移动，以 `git diff --check` 命中的同一处空白为准。

## 七、执行顺序

严格按以下顺序：

1. Task 1：修改测试并取得 RED；最小修改模板转 GREEN。
2. Task 2：增加行为测试并取得 RED；最小修改完成态结果卡转 GREEN。
3. Task 3：恢复跨机器解释器用法并脱敏公开样例。
4. Task 4：只清理已确认的尾随空格。
5. 运行定向测试、静态检查和完整回归。
6. 检查最终 diff，确认没有夹带文档、README、图片、Git 排除规则或其他无关改动。

## 八、最低验证要求

根据实际测试位置调整文件组合，但必须报告真实命令和结果。

### 8.1 Task 1 定向测试

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_lifecycle.py auto_tune\tests\test_hpo_api.py -k "dry_run or mode or routing" -q -p no:cacheprovider
```

### 8.2 Task 2 定向测试

将新行为测试放在现有最接近的 UI 测试文件中，然后运行该文件。若放入 `test_hpo_ui_lifecycle.py`，命令示例：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_lifecycle.py -k "baseline or best" -q -p no:cacheprovider
```

同时确认现有后端契约未回归：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py -k "reference_baseline or kept_reference" -q -p no:cacheprovider
```

### 8.3 静态检查

```powershell
node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
git diff --check
```

如果 `git diff --check` 命中本批之外的既有内容，只记录文件、行号和证据，不得顺手修改。

### 8.4 完整回归

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

预期不得少于当前基线 `2732 passed`；测试总数因新增回归测试应增加。两条既有 sklearn PCA warning 可以保留，但不得新增 warning 或失败。

不需要运行真实训练、浏览器人工验收、GPU 验证或网络 LLM 调用。

## 九、完成前自检

- 四种模式的页面选项、FormData 值、按钮路由和后端兼容语义一致。
- `kept_reference_baseline=true` 时，主卡、查看和保存均指向原参考运行。
- `kept_reference_baseline=false` 时，现有调优最佳行为不变。
- `loop.py`、评分规则、审计和持久化协议没有变化。
- `_final_test.py` 使用 `sys.executable`。
- `test_modelb.py` 和 `config.template.yaml` 不含真实业务信息。
- `deepseek-flash` 和 `model_store` 配置保持不变。
- 只清理指定尾随空格，没有格式化无关文件。
- 没有修改任何 Markdown、DOCX、README、图片、`.gitignore` 或临时参考文件。
- 没有新增依赖、删除文件、真实训练、网络调用、提交或推送。

## 十、最终交付报告

完成后停止并向艾卡/Codex 提供：

1. 两项 P1 的根因和最终行为；
2. 实际修改文件及每个文件的职责；
3. Task 1、Task 2 的真实 RED 命令、关键失败断言和 GREEN 结果；
4. `kept_reference_baseline=true/false` 两种前端结果的测试证据；
5. 跨机器解释器恢复和两个公开样例脱敏结果；
6. 定向测试、后端契约测试、Node 检查、`git diff --check`、完整回归的精确结果；
7. 完整 `git status --short`，区分本批修改、F1.1 前序改动和文档/历史脏工作区；
8. 偏离计划之处和剩余风险；
9. 明确声明：未修改项目文档、README、图片和 `.gitignore`，未新增依赖，未删除文件，未真实训练，未调用网络 LLM，未提交，未推送。

完成后不要继续扩展范围，等待 Codex 独立审查和验收。
