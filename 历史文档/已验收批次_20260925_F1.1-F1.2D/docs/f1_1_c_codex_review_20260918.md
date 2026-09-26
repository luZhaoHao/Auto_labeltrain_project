# F1.1-C Codex 稳定冻结审查记录

日期：2026-09-18
审查结论：**F1.1-C 代码修复已通过 Codex 独立验收，原 2 项 P1 业务语义回归及后续发现的动态文本转义 P1 均已关闭；未发现新的 P0/P1/P2。F1.1 整体冻结仍等待发布范围整理，不进行结构重写。**
操作边界：本次只读审查并更新项目文档；未修改业务代码，未提交或推送 GitHub。

## 1. 结论依据

- 艾卡已连续检查 3–4 个真实训练结果，F1.1-B 的初步作用和性能达到当前业务条件。
- Codex 使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 独立运行完整测试：`2732 passed, 2 warnings in 313.78s`。两条警告均来自 sklearn PCA 在全零方差样本上的既有运行时警告。
- `node --check auto_tune/ui/static/hpo.js` 通过。
- `git diff --check` 未发现冲突标记，只发现 `auto_tune/main.py:75` 一处尾随空格及换行格式提示。
- 敏感信息候选扫描未发现真实 API Key、Token、密码或其他凭据；命中项均为测试中的伪造进程令牌或脱敏测试输入。
- 已跟踪文件中没有超过 10 MB 的文件；未跟踪目录 `.tmp_h1_render/` 约 83.6 MB，`papper/` 约 5.5 MB，均不应进入版本提交。

完整回归通过证明当前主路径没有系统性崩溃，但不能覆盖下列前端语义错误和发布范围问题，因此本次不能直接给出 F1.1 整体冻结结论。

## 2. P0

未发现 P0。

## 3. P1 业务语义回归

### P1-1 前端移除了已批准的干运行模式

证据：

- 已批准规格 `docs/f1_1_a_experience_model_store_spec_20260917.md:92` 要求表单分别提交 `dry_run`、`keep_params`、`hpo`、`full`，第 100 行还要求干运行主按钮位于可见配置区末尾。
- 实施计划 `docs/superpowers/plans/2026-09-17-f1-1-a-experience-model-store.md:57-70` 和第 100 行同样明确保留四种模式。
- 当前 `auto_tune/ui/templates/single_page.html:894-899` 和 `auto_tune/ui/templates/agent_suggestion.html:144` 将页面改成三种模式并移除 `dry_run`。
- `auto_tune/tests/test_hpo_ui_rework.py:1560-1607` 被同步改成断言页面中不得存在 `dry_run`，使测试固化了与批准规格相反的行为。
- 后端 `/tuning/start` 仍保留 `dry_run` 兼容语义，说明不需要改动训练协议。

影响：用户失去“只生成训练计划、不启动训练”的低风险检查入口，且页面行为与已批准规格、后端能力不一致。

最小修复：只恢复两个模板中的 `dry_run` 入口及原按钮复用逻辑，并把对应 UI 测试恢复为四模式断言；不改后端、不重写状态机。

### P1-2 原参考训练仍优于所有调优轮次时，页面把较差轮次显示为“最佳结果”

证据：

- `auto_tune/modules/agent_engine/loop.py:506-513` 已用 `kept_reference_baseline` 表示没有调优轮次严格超过原参考训练，并返回 `baseline_run`。
- `auto_tune/tests/test_tuning_loop.py:1581-1588` 已覆盖“保留原参考基线”的后端与最终 TXT 汇总语义。
- 当前 `auto_tune/ui/templates/single_page.html:4654-4687` 在完成时无条件把 `best_iteration`、`best_train_name` 渲染为“最佳结果”，并将“查看完整分析”和“保存报告”绑定到该调优轮次；代码没有检查 `kept_reference_baseline`。

影响：后端虽然正确保留原参考训练，页面仍可能引导用户查看、保存或采用性能更差的调优结果，直接影响业务决策。

最小修复：当 `kept_reference_baseline === true` 时，将 `baseline_run` 显示为总体保留结果并绑定查看/保存目标；`best_train_name` 只作为“调优轮次中的最佳结果”进行次级展示。增加一个前端回归测试覆盖该状态。不要改变评分、循环或持久化协议。

## 4. 发布整理阻断项

以下问题不要求重构业务逻辑，但在 F1.1 统一提交前必须处理：

1. **README 破图和意外删除待确认**：`README.md:40-48`、`README_EN.md:40-48` 仍引用 `img/主界面.png`、`img/输入训练结果.png`、`img/视觉大模型分析.png`，这些文件及另外 5 张图片当前处于删除状态；`PROJECT_REVIEW.md` 也处于删除状态。应由艾卡确认删除意图，并恢复图片或同步更新 README，不能按当前状态直接提交。
2. **本机绝对路径**：`_final_test.py:4` 把解释器写死为 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，应恢复为 `sys.executable`，同时保留“必须在 auto_tune Conda 环境运行”的文档约束。
3. **业务样例未脱敏**：`test_modelb.py:13-14` 和 `auto_tune/config.template.yaml:2-4` 含具体产线、检测对象和缺陷类型。公开仓库只允许脱敏配置模板，应改为通用占位示例。
4. **临时和参考产物未排除**：`.tmp_h1_render/`、`.tmp_h1_*.ps1`、`papper/` 目前均为未跟踪且未忽略内容。统一提交前应加入排除清单；本次审查不删除这些文件。

## 5. P2 记录

- `auto_tune/main.py:75` 存在一处尾随空格，统一提交前可随最小修复清理。
- 模型库上传文件名对 Windows 冒号/ADS 和父级 junction 的专项测试仍可补强；当前根目录、目标链接、大小、哈希和原子提交已有防护，未找到可证明的逃逸路径，不作为 F1.1 阻断项。
- HPO 状态变更接口目前主要依赖本机回环地址和前端携带令牌，服务端访问边界可在 F1.2 API 交付时统一收紧，不在 F1.1-C 扩大修改面。
- 模型库容量配置对负数等异常值的校验可在后续配置治理中补强，当前默认配置不受影响。

## 6. F1.1 统一提交白名单

通过最小修复和重新验收后，可进入提交候选范围：

- `auto_tune/` 下经审查的业务源码、前端模板和静态脚本；
- 与 F1.1 功能直接对应的自动化测试；
- 脱敏后的 `auto_tune/config.template.yaml`；
- 当前权威 Markdown、三份研发 DOCX，以及按归档规则确认后的历史文档；
- 必要且引用有效的 README 图片。

## 7. 必须排除清单

- 真实 `auto_tune/config.yaml`、API Key、Token、密码和其他凭据；
- 数据集、标注、上传缓存、模型权重、训练结果、日志、审计运行文件、数据库备份、虚拟环境和构建产物；
- `.tmp_h1_render/`、`.tmp_h1_*.ps1`、临时截图和其他本机渲染产物；
- `papper/` 下的论文 PDF，以及不属于产品交付的个人参考材料；
- `test_modelb.py` 中的真实业务样例和任何客户、产线或数据集专用信息；
- 本机专用绝对路径。

## 8. 建议的最小后续批次

建议由 Claude Code 只实施以下内容，Codex 再独立验收：

1. 恢复前端 `dry_run` 入口和四模式测试；
2. 修复 `kept_reference_baseline` 对应的总体最佳展示、查看和保存目标，并补前端回归测试；
3. 整理发布阻断项：恢复或更新 README 图片引用、恢复跨机器解释器用法、脱敏模板和样例、清理尾随空格；
4. 不拆分 `app.py` 或 `single_page.html`，不改训练/HPO/LLM 协议，不新增依赖，不删除文件；
5. 定向测试和完整 `2732+` 回归全部通过后，再由艾卡决定 F1.1 整体验收与冻结。

在以上 P1 和发布阻断项关闭前，不提交、不推送、不进入 F1.2。

Claude Code 最小修复提示词已于 2026-09-18 编写，见 [f1_1_c_claude_code_minimal_fix_prompt_20260918.md](f1_1_c_claude_code_minimal_fix_prompt_20260918.md)。提示词只授权处理两项 P1、跨机器解释器、公开样例脱敏和一处尾随空格；README、图片删除、临时产物及 Git 排除规则继续由 Codex 在最终发布整理阶段处理。

### 第一轮最小修复复审

Claude Code 第一轮修复后，Codex 独立完整回归为 `2735 passed, 2 warnings in 350.44s`，两个 Node 语法检查通过，`git diff --check` 无错误。原两项 P1 的核心业务语义已按计划恢复，配置模板和样例已脱敏，`_final_test.py` 已恢复 `sys.executable`。

本轮仍发现 1 项 P1：`single_page.html` 将 `reference_baseline.run_name`、`baseline_run` 和 `best_train_name` 直接插入 `bestContent.innerHTML`，后端运行名校验允许 HTML 元字符，存在持久化页面注入风险，也违反最小修复提示词的动态文本转义要求。当前仍不验收；第一轮返修提示词见 [f1_1_c_claude_code_followup_prompt_20260918.md](f1_1_c_claude_code_followup_prompt_20260918.md)。

默认模式决定：保持 `dry_run`。四模式顺序、仓库 HEAD 和最小修复提示词均以 `dry_run` 为首项，且没有权威书面条款要求默认 `keep_params`；返修只补测试固化该决定。

### 第二轮返修复审与代码验收

Claude Code 完成动态运行名转义和对应行为测试后，Codex 于 2026-09-18 独立复验：

- 恶意运行名、总体最佳分支和默认模式定向测试：`16 passed, 67 deselected`；
- 完整自动化：`2738 passed, 2 warnings in 335.32s`；
- 两条 warning 仍为 sklearn PCA 在全零方差测试样本上的既有警告，无新增 warning；
- `node --check auto_tune/tests/js/minidom.js` 与 `node --check auto_tune/ui/static/hpo.js` 通过；
- `git diff --check` 无空白错误，`auto_tune/main.py` 和 `_final_test.py` 相对 HEAD 均无净 diff；
- 独立代码复核未发现新的 P0/P1/P2。

最终行为：

1. 页面恢复 `dry_run`、`keep_params`、`hpo`、`full` 四种模式，默认保持 `dry_run`，三条 LLM/普通入口与 HPO 独立入口路由正确；
2. 原参考训练仍为总体最佳时，主卡、查看和保存均绑定原参考运行，调优轮次最佳只作为次级信息；
3. 原参考和调优运行名在结果卡正文中统一 HTML 转义，按钮继续保留原始运行名语义；恶意名称不会形成注入节点或事件属性；
4. 公开配置模板和样例已脱敏，`deepseek-flash` 与受控模型库配置保持不变。

因此，**F1.1-C 业务代码与对应测试验收通过**。F1.1 尚不标记为整体冻结：README 破图与删除文件意图、临时/参考产物排除、最终提交白名单和敏感信息检查仍需由 Codex 在统一版本整理阶段关闭；按艾卡要求，在 F1.1 整体验收前不提交、不推送 GitHub。
