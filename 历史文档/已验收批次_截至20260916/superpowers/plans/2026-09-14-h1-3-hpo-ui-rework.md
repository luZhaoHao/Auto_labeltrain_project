# H1.3 HPO界面返修与正式训练 Implementation Plan

> For agentic workers: 按 executing-plans 逐任务执行；本项目由 Claude Code 执行业务实现及测试，不提交、不推送、不改文档，不建立工作树。项目规则优先于技能默认提交建议。

**Goal:** 保持四模式布局一致，完成HPO选择、运行、结果、最佳参数正式训练及模型取得闭环。

**Architecture:** HpoService/HpoRunner继续为搜索权威，新增只读安全投影与正式训练提交接口。复用ManualRunController和原子门禁；前端按study身份隔离异步响应，正式训练使用JSON启动与既有运行查询，不消费SSE为JSON。

**Tech Stack:** 现有Python/FastAPI/Pydantic/Optuna/Ultralytics/HTML/JavaScript，无新增依赖。

**Spec:** ../specs/2026-09-14-h1-3-hpo-ui-rework-design.md

## Global Constraints

- 仅YOLOv8 Detect；不改搜索空间、恢复矩阵、排名口径和其他三模式业务。
- Python绝对路径 D:\Program Files\anaconda3\envs\auto_tune\python.exe；先记录sys.executable/version。
- 不删除文件、创建工作树、复制数据、真实训练、调用网络LLM或提交推送；遇范围冲突停止报告。
- UI所有外部文本用textContent；不能将完整底层异常作为安全字段错误。
- 保留当前未提交修改；以下文件清单是允许范围，不要求无差别修改。

## 文件职责

- ui/templates/single_page.html：共同布局及HPO内容区，不大规模重构整页。
- ui/static/hpo.js：草稿/选择/请求代号、运行状态、最佳及正式训练关联。
- ui/hpo_api.py：安全输入、快照/研究投影及产物路由；不能成为第二训练执行器。
- ui/hpo_reuse.py：保留resolve_hpo_verification；新增resolve_hpo_formal_training(service, runner, study_id, trial_id, training_config)，输出VerifiedHpoConfig并重新校验候选。
- ui/hpo_training.py（新增）：FormalTrainingConfig、普通训练提交/关联投影与受控产物解析，避免持续扩大app.py。
- ui/app.py：应用绑定、复用普通训练创建/收尾、保持旧接口兼容。
- ui/i18n.py：HPO中文字段；modules/hpo/search_space.py仅允许只读摘要函数，算法不变。
- tests/test_hpo_ui_rework.py、test_hpo_formal_training.py、test_hpo_artifacts.py（新增）：本轮反例；既有HPO API/UI/生命周期/复用测试追加回归。

## Task 1：安全投影与选择输入

接口：现有GET study/list/best-config扩展只读字段；新增受控快照选择GET，禁止latest自动替代；新增安全字段错误投影，限定field/reason_code/message。

- [ ] 写反例：latest未登记但受控旧快照可选；损坏快照可见不可选；非法输入能定位imgsz字段且无异常原文；history第二页可查询。
- [ ] 单独运行新增测试，记录真实失败。
- [ ] 实现投影/快照入口/字段白名单错误；复用已有本地选择器，必要最小.pt选择接口不扫描任意根。
- [ ] 运行test_hpo_api.py及新增测试；读接口不改study/execution，以前后字节比较证明。

## Task 2：最佳正式训练提交

接口：POST /api/hpo/studies/{id}/train-best，严格trial_id和FormalTrainingConfig(epochs,batch,imgsz,device)，202返回run_id/train_name/source；旧固定验证SSE保留。

- [ ] 写参数化失败反例：bool/字符串数字/未知字段/NaN/越界/错trial/非SUCCESS/未COMPLETED/绑定漂移/候选不适配新epochs；断言零进程、零新目录。
- [ ] 写成功反例：新epochs2/imgsz96生效；六参数、seed、初始权重与快照来自服务端，不接受客户端搜索参数。
- [ ] 运行test_hpo_formal_training.py确认red；实现严格模型与resolve_hpo_formal_training。
- [ ] 抽取最小普通训练提交公共逻辑，复用ManualRunController、broker/finalizer/门禁，新增JSON路由；旧接口仍返回SSE。不得把控制器生命周期绑定HTTP。
- [ ] 写来源metadata及关联持久化查询；重建服务后仍可查、多个运行都保留；校验占用竞争、初始化失败和断开不中止。
- [ ] 运行新测试、test_hpo_best_reuse.py、test_training_gate.py、test_training_finalizer.py、test_hpo_studio_concurrency.py。如测试文件名不同，先rg核对再报告实际命令，不略过对应层。

## Task 3：受控权重与关联结果

接口：Trial产物仅best.pt/last.pt；正式产物由受控身份解析。关联查询按study显示普通run事实及安全结果入口，不向study/execution写入。

- [ ] 写test_hpo_artifacts.py：合法文件下载、文件不存在、错study/trial、路径穿越、链接/重解析点、恶意run_relpath、正式来源不匹配、请求任意文件拒绝。
- [ ] 确认red；实现受控路径解析和FileResponse，服务端重验证身份与根，不信任元数据中的任意路径。
- [ ] 查询正在运行/失败/停止/完成，缺失或坏metadata明确不可用，最终指标与对应正式run一致。
- [ ] 运行新测试及test_hpo_formal_training.py、test_hpo_best_reuse.py，比较HPO事实前后不变。

## Task 4：共同布局与选择一致性

- [ ] 写test_hpo_ui_rework.py行为反例：模式固定位置；HPO无LLM建议；草稿与冻结详情区分；公共提交不启动旧选中研究；切换清空best；旧detail/best响应不覆盖新选择；单timer绑定新ID；刷新不自动训练。
- [ ] 运行red；修改模板和hpo.js，采用selection generation、请求身份检查、串行轮询和独立操作pending；不要只写字符串存在性测试来宣称异步行为正确。
- [ ] “创建并开始”串行create/start，未知创建结果不重发；READY失败可显式启动此任务。保留原API分离。
- [ ] 历史翻页与高亮、冻结完整摘要、中文状态/字段、字段级错误、停止pending、断网最后可信状态与错误独立区域。
- [ ] 运行test_hpo_ui.py、test_hpo_ui_lifecycle.py、新UI测试及node --check auto_tune/ui/static/hpo.js。

## Task 5：最佳结果与正式训练交互

- [ ] 写反例：自动最佳显示无LLM理由/无虚构提升；同分提示；正式提交前确认四字段；六参数不可自由覆盖；启动202立即显示关联ID；重载后保留关联；按对应run显示最终指标/权重；409来源失效不提示训练占用。
- [ ] 确认red；接入最佳卡片、受控Trial权重、正式确认表单、JSON提交和关联结果；固定验证仅次要入口且明确同条件，不改变旧接口。
- [ ] 运行全部HPO相关测试、原三模式邻近测试、完整pytest auto_tune/tests -q -p no:cacheprovider、pip check、node语法和git diff --check。已有失败单独说明，不擅自修改无关内容。
- [ ] 交付实际文件、各测试命令/精确结果、red→green证据、偏离及剩余风险；不宣称浏览器或真实YOLO已验证。

## Task 6：Codex独立验收与艾卡体验（Claude不执行）

- [ ] 独立代码审查、参数/产物安全反例、全量回归。
- [ ] 浏览器四模式位置/风格，研究切换乱序、输入选择/错误、创建/停止/恢复、翻页、Trial权重与正式训练完整操作。
- [ ] TPE/Random budget2/epochs1/batch1/imgsz64；正式epochs2/imgsz96，核对命令、args、CSV、metadata、权重、关联重载、原HPO事实不变。
- [ ] 技术验证通过交艾卡体验，艾卡通过后最终验收；Codex再回写文档、经批准清理临时提示词、分批提交与上传。

工作量预估：Claude实现与自测约3–5人日，Codex独立验收约1–2人日，艾卡体验与必要返修另计；不作为交付承诺。
