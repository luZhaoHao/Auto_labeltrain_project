# F1.1-B 智能分析报告运行绑定与空白展示修复 Implementation Plan

**状态：已于 2026-09-18 完成实现和验收。** 定向测试 `114 passed`，完整自动化 `2732 passed, 2 warnings`；艾卡使用 3–4 个真实训练结果复验通过。最终证据见 [F1.1-B 验收记录](../../f1_1_b_codex_review_20260918.md)。本计划不再作为待执行任务清单；该结果暂不单独提交或推送，待 F1.1 整体验收后统一处理 GitHub 上传。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复智能训练页面把当前训练报告与全局调优历史混用、以及普通 LLM 诊断为空时出现空白分析卡的问题，确保选择 `train64` 后所有建议和诊断都来自 `train64` 对应报告。

**Architecture:** 保持训练、决策、语义校验和审计协议不变，只在 Studio 的只读展示投影层建立明确的来源优先级与运行身份匹配。当前训练报告中的结构化 `suggestion` 是该报告页面的首选来源；只有不存在报告级建议且历史记录与报告运行名称匹配时，才允许使用调优历史。LLM 分析正文为空时，从同一报告的结构化建议 `diagnosis` 兜底，不跨报告补值。

**Tech Stack:** Python 3.10、FastAPI、Jinja2、pytest、现有 `auto_tune` Conda 环境。

**Spec:** 本计划中的“已复现事实、冻结行为和验收标准”即本次边界明确的修复规格；不新建或修改其他项目规格文档。

## Global Constraints

- 完整阅读 `AGENTS.md`、`docs/development_handoff_20260814.md` 和本计划后再修改代码。
- 只修改业务代码和对应测试；不得修改本计划、README、路线图、交接记录、配置模板或其他项目文档。
- 不新增依赖，不删除文件，不清理、回退、覆盖或格式化工作区中的既有改动。
- 不修改 LLM 提示词、参数注册表、语义规则、Guardrails、多轮停止策略、训练执行器、审计写入格式或历史持久化格式。
- 不把 `keep_params` 改造成其他动作，也不伪造参数建议；本次只修复读取来源和展示。
- 所有 Python 和 pytest 命令必须使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。开始前记录 `sys.executable` 和 Python 版本。
- 严格执行 RED→GREEN：先新增能够复现真实问题的测试并取得失败证据，再做最小实现。
- 不运行真实训练，不调用网络 LLM，不下载或反序列化模型，不提交 Git，不推送 GitHub。
- 当前工作区是脏工作区；最终报告必须区分本轮修改与既有修改。

---

## 已复现事实

2026-09-18 对 `train64` 的最新分析报告为 `log/train_1789700830_report.json`：

```json
{
  "llm_analysis": {
    "train64": {
      "llm_diagnosis": "",
      "model_used": "deepseek-v4-flash",
      "error": null
    }
  },
  "suggestion": {
    "diagnosis": "验证损失上升而训练损失下降，出现中度过拟合，mAP50已饱和，验证损失上升趋势提示训练不稳定。",
    "action": "增加权重衰减和mixup抑制过拟合，并将学习率减半以缓解震荡和平台。",
    "hyperparameter_changes": {
      "weight_decay": 0.0015,
      "mixup": 0.1,
      "lr0": 0.0005
    },
    "training_overrides": {},
    "error": null,
    "retried": true
  }
}
```

与此同时，`log/tuning_history.json` 的最后一条记录属于另一条调优运行：

```json
{
  "decision": {
    "diagnosis": "按原有参数训练，不做超参数调整",
    "action": "keep_params",
    "hyperparameter_changes": {},
    "training_overrides": {}
  },
  "train_name": "autotune_9ff8be0f_iter01"
}
```

当前 `auto_tune/ui/app.py::_get_latest_suggestion()` 无条件优先读取全局调优历史最后一条，再读取当前训练报告。当前 `auto_tune/ui/templates/single_page.html` 的“大模型分析报告”标题从 `training.suggestion` 读取“3 项修改”，正文却只遍历 `llm_analysis[*].llm_diagnosis`；空字符串导致卡片只有标题而没有正文。

因此页面同时出现：

- 顶部：`train64` 的“智能体建议 3 项修改”，正文空白；
- 下方：其他运行的 `keep_params`，显示“按原有参数训练 / 保持原参数训练”。

审计记录没有损坏 `train64` 的建议，模型也没有失去推荐能力。根因是展示层跨运行混用数据以及空字符串未兜底。

## 冻结行为

1. 当前页面存在训练报告且该报告包含结构化 `suggestion` 时，必须优先展示该建议，包括成功、`keep_params` 和结构化失败三种状态。
2. 报告级建议存在时，任何不属于该报告运行的全局调优历史都不得覆盖它。
3. 当前报告没有结构化建议时，只有历史条目的 `train_name` 与当前报告 `runs` 中的运行名称一致，才允许作为回退来源。
4. 当前没有训练报告时，保留现有“显示最新调优历史建议”的兼容行为。
5. `llm_analysis` 中 `llm_diagnosis` 为 `""`、纯空白或缺失，而同一训练报告的 `suggestion.diagnosis` 非空时，大模型分析卡显示后者。
6. 不得从另一份报告、另一条历史记录或全局最后记录中补诊断。
7. 普通 LLM 分析和结构化建议都没有可显示正文时，不显示一个只有标题的空白分析卡。
8. 结构化建议的参数表继续同时展示 `hyperparameter_changes` 与 `training_overrides`，保持现有去重和覆盖优先级。

---

### Task 1: 建立当前报告优先且按运行匹配的建议投影

**Files:**

- Modify: `auto_tune/ui/app.py`（`_get_latest_suggestion` 附近）
- Test: `auto_tune/tests/test_decision_unified.py`

**Interfaces:**

- 保留对外函数名：`_get_latest_suggestion(tuning_history, training) -> dict | None`。
- 返回结构继续包含：`diagnosis`、`rationale`、`action`、`hyperparameter_changes`、`training_overrides`、`error`。
- 可以新增私有小函数消除报告建议与历史 decision 的重复投影，但不得改变持久化 JSON Schema。

- [ ] **Step 1: 为真实串线场景写失败测试**

在 `auto_tune/tests/test_decision_unified.py` 增加测试，构造：

```python
training = {
    "runs": {"train64": {"name": "train64"}},
    "summary": {"best_overall_run": "train64"},
    "suggestion": {
        "diagnosis": "train64 诊断",
        "action": "调整 train64",
        "hyperparameter_changes": {
            "weight_decay": 0.0015,
            "mixup": 0.1,
            "lr0": 0.0005,
        },
        "training_overrides": {},
        "error": None,
    },
}
history = [{
    "train_name": "autotune_9ff8be0f_iter01",
    "decision": {
        "diagnosis": "按原有参数训练，不做超参数调整",
        "action": "keep_params",
        "hyperparameter_changes": {},
        "training_overrides": {},
    },
}]
```

断言 `_get_latest_suggestion(history, training)` 返回 `train64` 的诊断、动作和 3 项修改，不能返回 `keep_params`。

- [ ] **Step 2: 为历史回退的运行绑定写失败测试**

至少覆盖：

```python
# 不相关历史不得覆盖当前无 suggestion 的 train64 报告
training = {"runs": {"train64": {}}, "suggestion": None}
unrelated = [{"train_name": "autotune_other", "decision": valid_decision}]
assert _get_latest_suggestion(unrelated, training) is None

# 同一运行允许回退
training = {"runs": {"autotune_same": {}}, "suggestion": None}
matched = [{"train_name": "autotune_same", "decision": valid_decision}]
assert _get_latest_suggestion(matched, training)["action"] == valid_decision["action"]

# 没有训练报告时保留最新历史兼容行为
assert _get_latest_suggestion(history, None) is not None
```

另加一例：历史列表中最后一条不匹配、较早一条匹配时，应从后向前找到最近的匹配条目，不能只检查最后一条。

- [ ] **Step 3: 为报告级失败和 keep_params 优先级写失败测试**

分别构造当前训练报告：

- `suggestion={"error": "Suggestion generation failed"}`，同时历史有成功建议；必须返回当前报告的失败。
- 当前报告 `suggestion.action == "keep_params"`，同时历史有参数修改；必须返回当前报告的 `keep_params`。

这样可以证明“当前报告优先”适用于所有结构化终态，而不只适用于成功修改。

- [ ] **Step 4: 使用未修改业务代码取得 RED**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_unified.py -k "latest_suggestion" -v -p no:cacheprovider
```

Expected: 新增的“当前报告优先”和“不相关历史不回退”断言失败；记录准确失败数和关键差异。

- [ ] **Step 5: 实现最小来源选择规则**

在 `auto_tune/ui/app.py` 中按以下顺序实现：

```text
1. 检查 training.suggestion：
   - 有 error，返回当前报告的结构化失败；
   - 有 action、hyperparameter_changes 或 training_overrides，返回当前报告建议。
2. 若 training 存在：
   - 收集 training.runs 的运行名；
   - 从 tuning_history 末尾向前查找 train_name 属于该集合的最近记录；
   - 没有匹配就返回 None。
3. 若 training 不存在：
   - 保留使用全局最新历史记录的兼容逻辑。
4. 对选中的历史 decision 使用既有成功、keep_params、error 投影语义。
```

不要通过时间戳猜测关联，不按文件名包含关系模糊匹配，不把 `summary.best_overall_run` 与不相干的 `train_name` 强行对应。

- [ ] **Step 6: 运行相同测试转 GREEN**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_unified.py -k "latest_suggestion" -v -p no:cacheprovider
```

Expected: 所有 `latest_suggestion` 测试通过。

---

### Task 2: 消除大模型分析报告空白卡片

**Files:**

- Modify: `auto_tune/ui/app.py`（`_common_context` 附近，可新增只读展示投影函数）
- Modify: `auto_tune/ui/templates/single_page.html`（`llmAnalysisCard`）
- Test: `auto_tune/tests/test_decision_unified.py`

**Interfaces:**

- 建议新增私有纯函数：`_get_llm_analysis_display(training) -> dict | None`。
- 返回值保持模板现有映射形状：`{run_name: {"llm_diagnosis": str, "model_used": str | None, "error": str | None}}`。
- `_common_context()` 的 `llm_analysis` 改为该展示投影；原始报告对象和 `/api/training` 返回内容不得被原地修改。

- [ ] **Step 1: 为 `train64` 空诊断场景写失败测试**

构造：

```python
training = {
    "runs": {"train64": {"name": "train64"}},
    "summary": {"best_overall_run": "train64"},
    "llm_analysis": {
        "train64": {
            "llm_diagnosis": "",
            "model_used": "deepseek-v4-flash",
            "error": None,
        },
    },
    "suggestion": {
        "diagnosis": "验证损失上升而训练损失下降，出现中度过拟合。",
        "action": "调整正则化和学习率",
        "hyperparameter_changes": {"weight_decay": 0.0015},
        "training_overrides": {},
        "error": None,
    },
}
```

断言展示投影中 `train64.llm_diagnosis` 等于结构化建议诊断，并保留 `model_used`。

- [ ] **Step 2: 为不跨运行兜底和空卡隐藏写失败测试**

至少覆盖：

1. `llm_analysis` 属于 `train65`、结构化建议对应当前 `train64` 时，不得把 `train64` 的诊断写进 `train65`。
2. 普通诊断为空且结构化建议也没有非空 `diagnosis` 时，展示投影返回 `None` 或不包含可渲染项。
3. `llm_diagnosis` 只有空格时按空值处理。
4. 原有非空普通诊断优先保留，不被结构化建议覆盖。
5. 原有 `error` 继续显示，不能被建议诊断掩盖。

- [ ] **Step 3: 为最终 HTML 写失败测试**

扩展测试渲染帮助函数，使其可以传入完整 `training` 和展示投影，然后断言：

- 页面包含 `train64`、`deepseek-v4-flash` 和结构化诊断正文；
- 页面仍显示结构化参数建议；
- 不出现只有“大模型分析报告”标题而没有诊断、错误或建议正文的空卡；
- 不出现不相关历史中的“按原有参数训练”。

测试不得只检查函数返回值，必须至少有一条覆盖 `single_page.html` 的真实渲染。

- [ ] **Step 4: 使用未修改业务代码取得 RED**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_unified.py -k "llm_analysis or blank or selected_report" -v -p no:cacheprovider
```

Expected: 空诊断兜底或空卡隐藏的新增断言失败；记录准确失败原因。

- [ ] **Step 5: 实现同报告诊断展示投影**

实现要求：

```text
- 对原始 llm_analysis 做复制投影，不原地修改 training。
- 非空 llm_diagnosis 原样保留。
- error 原样保留并优先作为错误展示。
- 只有当普通诊断为空、suggestion.diagnosis 非空，而且能够确定为同一报告运行时，才使用 suggestion.diagnosis 兜底。
- 单运行报告直接绑定该运行；多运行报告优先使用 summary.best_overall_run，且该名称必须真实存在于 runs。
- 无法确定运行身份时不猜测，不跨运行兜底。
- 最终没有诊断或错误的条目不作为可渲染内容；若全部为空，模板不渲染 llmAnalysisCard。
```

模板继续自动转义所有 LLM 文本，不得添加 `|safe`。

- [ ] **Step 6: 运行相同测试转 GREEN**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_unified.py -k "llm_analysis or blank or selected_report" -v -p no:cacheprovider
```

Expected: 新增展示投影和模板渲染测试全部通过。

---

### Task 3: 相关回归与静态检查

**Files:**

- Modify: none（仅验证 Task 1–2 的改动）
- Test: `auto_tune/tests/test_decision_unified.py`
- Test: `auto_tune/tests/test_ui_training_results.py`
- Test: `auto_tune/tests/test_template_xss.py`
- Test: `auto_tune/tests/test_hpo_ui.py`

**Interfaces:**

- 验证建议投影、模板转义、训练结果页和 HPO 页面边界不回归。

- [ ] **Step 1: 运行定向测试**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_unified.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_template_xss.py auto_tune\tests\test_hpo_ui.py -q -p no:cacheprovider
```

Expected: 全部通过。

- [ ] **Step 2: 运行与调优历史、审计和循环相关的回归**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py auto_tune\tests\test_loop_bugfix_p1.py auto_tune\tests\test_run_state_training_api.py -q -p no:cacheprovider
```

Expected: 全部通过，证明本次展示修复没有改变训练和审计语义。

- [ ] **Step 3: 运行完整自动化回归**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: 完整套件通过；如存在既有 warning，准确报告数量和内容，不得把 warning 写成失败或忽略。

- [ ] **Step 4: 运行环境和改动检查**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -c "import sys; print(sys.executable); print(sys.version)"
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pip check
git diff --check -- auto_tune/ui/app.py auto_tune/ui/templates/single_page.html auto_tune/tests/test_decision_unified.py
git status --short
```

Expected:

- 解释器路径指向 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`；
- `pip check` 无依赖冲突；
- 本轮文件无空白错误或冲突标记；
- `git status --short` 中本轮仅包含计划允许的业务代码和测试文件，其他项目既有改动原样保留。

## 人工复验场景（由 Codex 后续执行，Claude Code 不得声称完成）

1. 在智能训练页选择并分析 `train64`。
2. 顶部“大模型分析报告”显示 `train64` 的非空诊断正文，不再只剩标题。
3. 参数建议区域显示 `weight_decay`、`mixup`、`lr0` 三项建议。
4. 页面不得显示来自 `train67`、`train70` 或 `autotune_9ff8be0f_iter01` 的 `keep_params` 结论。
5. 切换到真实 `keep_params` 运行时，仍应明确显示“保持原参数”，不能伪造参数变化。
6. 打开无结构化建议的旧报告时，只显示同运行可验证的内容；不从全局最后一条历史补值。

## Claude Code 最终交付报告

完成代码与测试后停止，不提交、不推送。报告必须包含：

1. 两项根因和最终来源优先级；
2. 实际修改文件及每个文件的职责；
3. 每项新增 RED 测试的命令、失败断言和 GREEN 结果；
4. `train64` 与不相关 `keep_params` 历史不再串线的自动化证据；
5. 空 `llm_diagnosis` 使用同报告结构化诊断兜底的证据；
6. XSS 自动转义仍有效的证据；
7. 定向回归、审计/循环回归、完整回归和 `pip check` 的精确结果；
8. 偏离本计划之处及理由；
9. 剩余风险；
10. 完整 `git status --short`，明确区分本轮改动、F1.1-A 前序改动和更早脏工作区；
11. 明确声明未新增依赖、未删除文件、未修改项目文档、未运行真实训练、未调用网络 LLM、未提交、未推送。
