# F1.1-C Claude Code 第一轮返修提示词

你正在处理 F1.1-C 最小修复的 Codex 独立复审问题。第一轮实现的两项原始 P1 业务语义已经基本正确，Codex 独立完整回归为 `2735 passed, 2 warnings`，Node 语法检查通过；但新增的总体最佳结果分支存在 1 项 P1 HTML 转义遗漏，因此当前尚未验收。

本轮只处理下面 3 项，不重做已经通过的四模式路由、总体最佳判断、按钮绑定、配置脱敏或其他 F1.1 功能。

## 一、开始前读取

1. `AGENTS.md`
2. `docs/f1_1_c_claude_code_minimal_fix_prompt_20260918.md`
3. `docs/f1_1_c_codex_review_20260918.md`
4. 当前 `single_page.html`、`test_hpo_ui_behaviour.py` 和 `tests/js/minidom.js` 实现

## 二、严格边界

- 只修改：
  - `auto_tune/ui/templates/single_page.html`
  - `auto_tune/tests/test_hpo_ui_behaviour.py`
  - `auto_tune/tests/js/minidom.js`
  - `auto_tune/main.py`，仅在确认能恢复为 HEAD 零 diff 时删除无意义空白行
- 不修改 `agent_suggestion.html`、后端、评分、循环、审计、持久化、配置模板或其他测试。
- 不修改任何项目文档、README、图片、DOCX、`.gitignore`、临时文件或参考材料。
- 不新增依赖，不删除文件，不格式化无关内容。
- 不提交、不推送、不真实训练、不调用网络 LLM。
- Python 和 pytest 只使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。

## 三、Task 1：P1——转义总体最佳结果卡中的动态运行名

### 已确认问题

`auto_tune/ui/templates/single_page.html` 当前把以下动态值直接拼入 `bestContent.innerHTML`：

- `overallName`，来源为 `reference_baseline.run_name`、`baseline_run` 或 `best_train_name`；
- `result.best_train_name`。

后端运行名校验拒绝路径分隔符、冒号、方括号和空字符，但允许 `<`、`>`、单引号和双引号。恶意或被篡改的参考运行名可以闭合 `<span>` 并注入新标签或事件属性。按钮的 `data-train-name` 已经使用 `encodeURIComponent`/`setAttribute`，绑定语义正确；问题只在正文 HTML 插值。

### 修复要求

- 在所有进入 `bestContent.innerHTML` 的动态运行名插值点进行 HTML 转义。
- 优先复用页面已有 `_escapeHtml()`，或用 `createElement` + `textContent` 构造动态节点；不要新增第二套不一致的转义规则。
- `overallName` 的原始值继续用于查看/保存目标；只对显示文本转义，不能把转义后的实体写回 `data-train-name`。
- 同时保护 `result.best_train_name` 的主分支和次级“本次调优轮次中最佳”分支。
- 不改变评分、标题、总体最佳判断、回退顺序、按钮目标和现有正常名称显示。

### 必须先写的 RED 测试

扩展现有 `reference_baseline_best` 行为场景，至少使用包含标签和事件属性的恶意名称，例如：

```text
train54</span><img data-xss="1" src="x" onerror="alert(1)">
```

测试必须证明：

1. 结果卡将完整恶意名称显示为普通文本；
2. 卡片内没有新增 `IMG` 等注入节点；
3. 没有 `onerror` 等事件属性进入 DOM；
4. 查看和保存目标仍保留原始运行名语义；
5. 正常 `train54`、`train60` 场景继续通过；
6. `kept_reference_baseline=false` 的 `best_train_name` 显示也经过同样保护。

不要只做源码字符串搜索；必须执行真实完成态渲染逻辑并检查生成 DOM。

## 四、Task 2：使 minidom 能真实验证转义

当前 `Element` 只实现了 `set innerHTML`，没有 getter；而页面 `_escapeHtml()` 的实现是先写 `textContent` 再读取 `innerHTML`。因此直接复用 `_escapeHtml()` 后，测试 harness 会返回 `undefined`，不能模拟真实浏览器。

最小补齐以下能力：

- 为 `Element.innerHTML` 增加 getter，至少正确序列化纯文本节点中的 `& < > " '` 为 HTML 实体，满足页面 `_escapeHtml()` 的真实使用方式；
- 若恶意名称行为测试需要读取动态 `data-*`，让 `dataset` 与 `getAttribute('data-*')` 保持一致；不要实现与本任务无关的完整浏览器 DOM；
- 保留现有 `set innerHTML`、模板解析、FormData、按钮提交和所有既有场景行为。

先运行新增测试得到真实 RED，再修改模板和 harness 转 GREEN。

## 五、Task 3：确认 `dry_run` 默认并消除无意义空行 diff

- **确认当前默认继续是 `dry_run`，不要给 `keep_params` 添加 `selected`。**
- 理由：批准规格要求四值与路由，F1.1-C 提示词冻结顺序为 `dry_run, keep_params, hpo, full`，仓库 HEAD 的原四模式顺序及默认值也是 `dry_run`。没有书面契约要求默认 `keep_params`。
- 在现有模式行为测试中增加初始 `select.value == 'dry_run'` 断言，冻结该决定，避免再次产生歧义。
- `auto_tune/main.py` 当前相对 HEAD 只新增一个空白行。删除该空白行，使该文件恢复为 HEAD 零 diff；不要改任何打印逻辑。

## 六、验证要求

先运行新增恶意名称测试取得 RED，再做最小修复。完成后至少运行：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_behaviour.py -k "baseline or overall or mode" -q -p no:cacheprovider

node --check auto_tune/tests/js/minidom.js

git diff --check

& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

完整回归不得少于当前 `2735 passed`，新增测试后总数应增加；不得新增 warning 或失败。

## 七、交付报告

完成后停止，并报告：

1. 恶意运行名测试的真实 RED 断言和 GREEN 结果；
2. 动态显示文本如何转义、按钮如何保留原始目标；
3. minidom 只增加了哪些必要能力；
4. `dry_run` 初始默认断言；
5. `auto_tune/main.py` 是否恢复为零 diff；
6. 定向、Node、`git diff --check` 和完整回归结果；
7. 本轮实际修改文件、偏离项和剩余风险；
8. 明确声明未改后端/协议/文档、未新增依赖、未删除文件、未提交、未推送。

完成后等待 Codex 再次独立复验。
