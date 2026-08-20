# Auto LabelTrain 开发交接记录（2026-08-14）

## 1. 当前结论

当前稳定版本为 `v0.2`；A0 审计闭环、统一训练收尾以及 Studio S1.1 训练日志分层均已完成独立验收并进入版本发布。下次开发不得重新实现或破坏这些兼容边界；应从本文件列出的剩余事项中选择一个小批次，先形成规格和测试边界，再交由 Claude Code 实现业务代码，最后由 Codex 独立复验和发布。

## 2. GitHub 与版本状态

- 仓库：`https://github.com/luZhaoHao/Auto_labeltrain_project`
- 可见性：Public，已由艾卡于 2026-08-14 明确批准。
- 默认分支：`main`
- `v0`：初始版本快照。
- `v0.1`：当前稳定版本，Release 名称为“审计闭环与统一训练历史”。
- `v0.1` 合并提交：`d81720c2585e0c30da67a4b0f141f6d83259ec2f`
- 发布 PR：`#1`，已合并。
- 发布分支 `agent/v0.1-unified-training` 暂时保留，未执行删除。

## 3. 已完成并验收的范围

### 3.1 A0 调优审计闭环

- 每个调优会话生成独立原子化审计记录。
- 记录 LLM 原文、候选参数、护栏结果、实际参数、精确命令、before/after/delta 和探针结论。
- 敏感字段递归脱敏。
- 决策失败、护栏拒绝、训练预检失败或审计持久化失败均为 fatal，不启动 YOLO 训练。
- 命令只构造一次，审计命令与实际执行命令一致。
- 参考参数和参考指标绑定同一个 `reference_run`；指标直接读取该运行的 `results.csv` 最终轮。
- 所有会话和迭代返回路径具有明确终态，不残留 `running`。

### 3.2 普通训练与自动调优统一收尾

- 两种训练来源共用 `training_finalizer.py` 和统一 KPI 口径。
- 统一历史使用稳定 `run_id` 幂等更新和原子 JSON 写入。
- 记录训练状态、分析状态、四项 KPI、configured/completed/best epoch、参数、产物和结构化错误。
- 分析失败不会覆盖已经成功的训练事实。
- 调优历史额外保存 AI 诊断、参数变化、安全护栏和审计入口；普通训练不显示调优专属内容。
- `/api/experiments/history` 提供统一历史导出。
- `/api/audit/{filename}` 提供审计查看，并进行 basename 与文件名前缀校验以防路径遍历。
- 历史页已补齐 Precision、Recall、参数键值、来源/状态筛选和正确的详情列宽。

### 3.3 文档与发布

- 根目录提供面向操作人员的中文 `README.md` 和英文 `README_EN.md`。
- README 使用现有的主界面、训练结果诊断和视觉大模型分析三张截图。
- 已排除含旧逻辑、旧界面或 `[object Object]` 缺陷的调优截图。
- `.gitignore` 已排除数据集、训练输出、日志、审计运行文件、模型权重、真实配置、密钥和临时 API 测试脚本。
- `start_server.bat` 已移除本机绝对 Python 路径，改为从脚本目录使用当前环境的 `python`。

## 4. 验证基线

当前完整自动化测试基线：

```powershell
python -m pytest auto_tune/tests -q -p no:cacheprovider
```

最近结果（2026-08-20，S1.1 验收）：`205 passed, 2 warnings, 0 skipped`。

两条 warning 来自 sklearn PCA 在测试常量数据上的除零数值警告，不是本批回归。下次改动后必须至少运行相关测试；准备发布时必须重新运行完整套件，不能只引用本次结果。

已执行过一次真实 1-epoch 调优验证，训练时长、epoch、CSV、Module B 报告、统一历史和审计 KPI 一致。后续除非改动训练执行、收尾或指标解析，不必每批重复真实训练；涉及这些路径时继续使用最小合法数据集和短 epoch。

## 5. 下次开发建议顺序

下次开始时先由艾卡确认一个批次，不要同时展开多个方向。建议优先级如下：

1. Studio S1.2：数据集划分改为不可变快照，不移动原始文件。
2. Studio S1.3：API 密钥迁移到环境变量，并制定已暴露密钥轮换步骤。
3. Studio S1.4：补充目录选择、上传总量、成员数量和解压后容量限制。
4. 完成上述可靠性事项后，按 2026-08-20 研发执行版进入 SQLite、多数据集与三项任务闭环；YOLO11/YOLO26 当前暂缓，不作为近期完成标准。

### 5.1 Studio S1.1 已验收能力（2026-08-20）

- 普通训练与自动调优共用训练日志清理、分类、UTF-8 落盘和结构化事件契约。
- 默认面板仅显示 epoch、验证、生命周期、警告和错误摘要；完整日志可折叠查看，错误堆栈不丢失。
- SSE 使用有界批量发送；前端禁止训练输出 `innerHTML +=`，默认区最多 500 行、完整区最多 2000 行，热路径不扫描 DOM。
- 训练输出文件句柄显式关闭；同一训练输出只有一个消费者。
- 普通训练 completed/failed/aborted 终态均清理运行状态文件；失败后刷新不再误报仍在训练。
- 验收证据：S1.1 定向测试 94 passed；完整套件 205 passed、2 条既有 PCA warning、0 skipped；真实 Chromium 失败路径、折叠保留和刷新终态检查通过。

模型结构解析与模型结构自调整仍属于后续阶段；一期只允许大模型在白名单和 Guardrails 内调整超参数，不应提前扩大 LLM 动作空间。

## 6. 下次启动检查清单

1. 先读取本文件、`docs/roadmap_20260814.md` 和 `docs/implementation_plan_20260814.md`。
2. 检查 GitHub `main` 和最新 Release 是否仍为预期版本，确认没有未审查的远程提交。
3. 检查当前本地文件；本地项目目录不一定是 Git 工作区，不得假设存在 `.git`。
4. 选择一个小批次，Codex 先给出设计规格与验收标准，艾卡批准后再安排 Claude Code 编码。
5. Claude Code 只负责业务代码和对应测试，不修改 README、路线图、规格、实施计划或发布说明。
6. Codex 独立审查实际差异，运行相关测试；涉及 UI 时启动服务做真实浏览器检查；涉及训练链路时按风险决定是否进行短 epoch 冒烟。
7. 发布前重新检查 `.gitignore`、待提交文件清单、真实 API Key、本机绝对路径和大文件。
8. 使用基于 `main` 的独立 `agent/<description>` 分支和 PR；未经艾卡批准不强推、不重写历史、不删除远程分支。

## 7. 已知注意事项

- 当前为单用户、单机、JSON 文件存储和本地 subprocess 训练架构；不要把它描述成已经支持多用户或训练队列。
- 统一历史的 JSON Schema 与 A0 审计格式已经形成兼容边界。后续增加字段应保持旧记录可读，不能静默覆盖损坏历史。
- `tuning_history.json` 继续服务于 LLM 反馈；`experiment_history.json` 服务于统一 UI 历史，两者职责不同，不应直接合并。
- before 指标必须来自与 `baseline.params` 相同的 `reference_run`，不得重新使用 Module B 全局最佳摘要代替。
- 训练成功、分析失败属于部分成功，UI 和 API 不得将训练状态改成 failed。
- 审计持久化失败属于 fatal，任何替代实现都不得绕过该失败策略继续启动训练。
- 缺失指标显示为 `—`，真实零值显示为 `0.0000`，不能使用真值判断混淆二者。
- 正式数据接入采用目录选择；ZIP 只作为遗留兼容路径，不再作为主流程。
- GitHub 仓库为公开仓库，任何本地数据、标签、权重、训练产物、日志、审计运行文件和真实凭据都不得提交。
- 本机 Git Credential Manager 曾提示 TLS 证书校验被禁用。后续再次推送前应检查并修复本机 Git/GCM TLS 配置；不要把关闭证书校验作为长期方案。
- 磁盘空间有限，不复制包含数据集和权重的完整项目副本；发布时只使用轻量源码临时克隆，并在获得明确授权后清理临时目录。

## 8. 文档真源

- 当前交接状态：`docs/development_handoff_20260814.md`
- 产品与阶段路线：`docs/roadmap_20260814.md`
- 后续任务顺序和验收：`docs/implementation_plan_20260814.md`
- 操作说明：`docs/操作说明_操作工手册.md`
- 技术 Leader 评估：`docs/Auto-Tune后续研发路线与实施评估_技术Leader版_20260814.docx`
- Linux 迁移准备：`auto_tune/docs/linux-migration-plan.md`

为避免修改由 Claude Code 维护的 `CLAUDE.md`，暂时保留它直接引用的 `docs/方案审查_20260814.md`、`docs/后续研发计划_20260814.md` 和 `docs/superpowers/plans/2026-08-14-hyperparameter-tuning-reliability.md`。这三份文件仅作为兼容归档，不是当前状态真源。下次向 Claude Code 安排业务开发任务时，应同时要求它自行更新 `CLAUDE.md` 的版本状态与文档入口；Codex 随后审核其内容和链接，审核通过后再删除这三份兼容归档。

如上述文件发生冲突，以本交接记录中的已完成事实和 GitHub 最新 Release 为当前状态依据；对未来范围，以艾卡在新任务中的最新批准为准。
