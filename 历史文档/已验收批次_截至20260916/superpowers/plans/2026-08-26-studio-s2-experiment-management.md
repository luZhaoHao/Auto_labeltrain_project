# Studio S2.1–S2.4 Experiment Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 S2.0 SQLite 稳定索引上一次性完成恢复、详情、查询诊断和基础实验比较闭环。

**Architecture:** 保持 storage/repository/service/API/UI 分层。事实文件只读，所有恢复先构建并校验临时索引再原子发布；查询与比较只消费稳定 Service 投影，不让 UI 或业务代码直接拼 SQL。

**Tech Stack:** Python 3.10 标准库 sqlite3/hashlib/json/pathlib，FastAPI/Jinja2，现有 JavaScript，pytest；不得新增依赖。

**Spec:** `docs/superpowers/specs/2026-08-26-studio-s2-experiment-management-design.md`

## Global Constraints

- 只使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 运行 Python 和 pytest，并在报告首行记录解释器。
- Claude Code 不修改任何项目文档，不执行 git add/commit/push，不删除文件，不新增依赖。
- SQLite 仍是可重建投影；任何索引故障不得改变训练、S1.5、JSON、报告或快照事实。
- 不开发标签、备注、收藏、永久管理标识、归档、删除、自动清理、Excel 或 AI 最佳实验。
- 不创建 worktree；测试临时数据使用 tmp_path；禁止触碰正式数据集和既有无关工作区改动。

---

### Checkpoint 1: S2.1 对账、原子重建与启动补录

**Files:** 扩展 `auto_tune/modules/local_index/` 中 models/database/repository/service，必要时新增单一职责的 reconciliation 模块；接入 `auto_tune/ui/app.py`；新增对应测试文件。

**Produces:** 稳定 `AuditResult`/`RebuildResult`；只读 audit；备份后临时库重建、quick_check 和原子发布；有界启动补录；rebuild 单实例门禁及尾部补录。

- [ ] 先写失败测试：缺失/多余/状态/指标/外键/产物差异，损坏事实隔离，问题摘要有界且不泄漏路径。
- [ ] 写失败测试：重建成功、备份失败、临时库构建失败、quick_check 失败、os.replace 失败、并发重建和原库字节保持。
- [ ] 写失败测试：启动补录只处理最近有界记录，不删除、不把未知终态猜成完成，训练并发产生的尾部记录最终可补齐。
- [ ] 最小实现领域模型、Repository/Service 和 API；所有原生异常转换为稳定 LocalIndexError。
- [ ] 运行 S2.0 数据库/Repository/import/integration/API 回归与本检查点测试，记录 RED→GREEN。

### Checkpoint 2: S2.2 详情、数据集关联与产物可用性

**Files:** 扩展 local_index 查询投影、`auto_tune/ui/components/experiment_panel.py`、`auto_tune/ui/app.py`、`single_page.html`、`i18n.py` 和对应测试。

**Produces:** 统一实验详情、受控产物 manifest、数据集关联实验摘要；文件/目录入口只接受已登记路径。

- [ ] 先写失败测试：普通训练/调优详情、数据集快照、决策/护栏、错误和旧记录缺失字段诚实降级。
- [ ] 写失败测试：report/audit/results.csv/args.yaml/run_dir/best.pt/last.pt/manifest 的 exists/missing/unregistered/unavailable。
- [ ] 写失败测试：数据集关联数量、最近使用和同任务同口径最佳指标；不同任务指标不得混排。
- [ ] 实现 API/UI，所有不可信文本转义或 textContent，缺失产物不生成失效打开动作。
- [ ] 回归 template_xss、S1.1 性能、S1.5 UI、训练历史和 S2.0 API/UI。

### Checkpoint 3: S2.3 搜索、排序、分页与诊断

**Files:** 扩展 models/repository/service/API/UI；必要时新增 diagnostics 模块和独立测试。

**Produces:** `limit+offset` 稳定分页；白名单排序；组合筛选；诊断、quick_check、checkpoint 和手动备份接口。

- [ ] 先写失败测试：search、dataset/source/status、sort/order、total、limit/offset 边界和确定性排序。
- [ ] 写注入测试：LIKE `%`/`_` 转义、恶意排序字段拒绝、全部 SQL 参数化；limit 默认 25、范围 1–100。
- [ ] 写诊断测试：Schema/DB/WAL/计数/备份/最近操作；损坏/锁定/权限失败使用稳定错误且 JSON 回退仍可用。
- [ ] 对 checkpoint、backup、audit、rebuild 写 CSRF/origin 测试；禁止清空/删除接口。
- [ ] 实现分页 UI 和诊断区，验证大列表不一次构造/渲染全部记录。

### Checkpoint 4: S2.4 基础实验比较与统一验收

**Files:** 新增聚焦的 compare 领域模块和测试；扩展 API、模板、i18n；避免把比较计算塞入 app.py。

**Produces:** `compare_experiments(run_ids, baseline_run_id)` 的稳定事实投影和 UI。

- [ ] 先写失败测试：2–5 个唯一 run_id、baseline 必须存在、缺失实验、顺序稳定。
- [ ] 写参数差异测试：折叠相同项，忽略 name/project/save_dir/下划线元数据，保留实际业务参数差异。
- [ ] 写指标测试：绝对/相对变化、基线为零、缺失值、失败/中断、不同数据集快照/任务/口径 `comparable=false`。
- [ ] 写调优三层事实和产物完整性测试；只输出最高指标/最短耗时等事实，不输出“最佳模型”。
- [ ] 实现历史页勾选 2–5 条、临时基线和比较页面；不新增永久标签或 Schema 管理字段。
- [ ] 运行全部 S2 定向测试、S1.5/安全/UI 回归和完整 `auto_tune/tests -q -p no:cacheprovider`。
- [ ] 使用最小合法快照执行 1 epoch 普通训练和 keep_params 最小非 dry-run 调优；核对 S1.5、JSON、SQLite、报告、results.csv。
- [ ] Chromium 验证对账/重建、详情、关联、搜索分页、诊断、比较、JSON 回退、缺失产物和控制台错误。

## 交付报告

Claude Code 完成四个检查点后一次性交付：实际文件清单、每个检查点 RED→GREEN、定向/完整测试、Schema 迁移与备份、扫描/分页上限、真实训练、UI 证据、偏离计划、遗留风险、敏感和大文件检查，并明确未修改文档、未提交、未推送。
