# H1.3 智能训练 HPO 接入规格

日期：2026-09-09。方向已由艾卡确认；本规格落实当前批次，不代表编码或验收完成。

## 1 目标与职责

在现有智能训练页面接入已验收 H1.1/H1.2，形成配置、启动、查看、停止、恢复、历史和最佳参数复用闭环。不重设计整站。

Claude Code：实现计划任务 1–5，补齐对应自动化测试并运行回归，提交交付报告后停止。普通 pytest 不启动真实 YOLO、网络 LLM 或浏览器。

Codex：独立代码审查、复跑自动化、必要真实短训练、浏览器走查、返修复验、文档回写及 GitHub 提交检查。艾卡：浏览器实际体验与最终交互确认。第六项为 Codex/艾卡验收，不能算作 Claude 已完成任务。

H1.2 最后独立基线：1837 passed、2 条既有 PCA 警告；累计 21 项独立反例；TPE/Random 各两次真实短训练通过。此数字仅作基线，交付及复验必须记录本次实际结果。

## 2 当前代码与架构

- 主页面：auto_tune/ui/templates/single_page.html；兼容页面：agent_suggestion.html。
- 现有 /tuning/start 用 mode= dry_run / keep_params / full；不能把未知 mode 当 full。现有 /api/training/start 为普通训练入口。
- RunManager 位于 modules/run_state/manager.py；现有 register 和 active_manual/active_tuning 不能直接提供跨策略原子占用。
- HpoService.create_study/load_study、HpoRunner.prepare/run/resume/status 与 rank_trials 继续作为事实与执行权威。网页不直接 ask/tell，不自行构造训练命令或解释恢复矩阵。
- 增加独立 hpo_api.py 与 hpo_controller.py；app.py 仅挂载路由、绑定已有输入解析及训练入口门禁。HPO 使用后台线程运行同步 runner，HTTP 请求只投递和查询，不阻塞事件循环。
- HPO 界面使用约 2 秒一次的只读轮询，离开页面停止轮询，不停止训练；不强求复用现有 LLM SSE 协议。普通训练和 LLM 原 SSE 行为保持兼容。

## 3 界面与输入

四项依次为：干运行（仅生成计划）[dry_run]、按原来参数训练 [keep_params]、HPO 算法调参 [hpo]、大模型调参 [full]。保留旧 full 值以兼容已有调用。/tuning/start 只接受原三种合法值，hpo 走独立路由；其他值 422，不能落入 LLM 分支。

HPO 模式隐藏 LLM 专属 probe、auto_loop、max_retries、综合评分和建议编辑控件，不把它们发送到 HPO。干运行保持原计划预览语义，不生成 HPO trial；本批不新增 HPO 干运行子模式。

HPO 展示当前明确选择的数据快照和本地初始权重，不要求已有 LLM 建议或参考训练。使用现有正式目录选择与快照发布流程；请求带明确 snapshot_id，服务端从受控快照目录查找并验证，不能用 latest_dataset 替换缺失/失效 ID。权重沿用已有本地输入安全策略与 HpoService 哈希验证，不自动下载。切换参考训练不得静默切换 HPO 数据。启动后输入冻结；改输入意味着创建新 study。

| 配置 | 默认值 | 范围与语义 |
|---|---|---|
| sampler | tpe | tpe / random，显示 TPE / 随机搜索 |
| budget | 10 | 整数 1–100，包含失败槽，不是保证成功次数 |
| epochs | 30 | 整数 1–1000，每次试验轮数 |
| seed | 42 | 整数 0–2147483647 |
| batch | 16 | 整数 1–256 |
| imgsz | 640 | 整数 32–2048 且为 32 倍数 |
| device | cpu | 字符串 cpu 或单 GPU 编号 0–63，沿用 ExecutionConfig 默认值；用户可明确选 GPU，实际可用性由预检判断 |
| timeout_seconds | 3600 | 整数 1–86400，单次试验超时 |

UI 可预填既有合法配置，但必须显示实际提交值。JSON 数字必须是数字，不接受 bool、字符串数值、未知字段、非有限值。search_space、目标指标及固定参数沿用 H1.2，不提供自由命令、输出根、executable 等字段。

## 4 HTTP 契约

新增 APIRouter，前缀 /api/hpo。沿用应用现有访问边界，不新增远程发布能力。错误格式统一为 error_code、error、next_action，均为可展示的脱敏中文说明。输入错误 422，未知对象 404，占用/冲突/无法恢复 409，持久化及损坏错误 500；保留底层稳定 HPO 错误码，不转为成功。

| 方法与地址 | 请求 | 行为 |
|---|---|---|
| POST /studies | snapshot_id、model_path、study_config、execution_config | 验证明确输入，create_study + prepare；201 返回 study_id。仅准备，不训练 |
| POST /studies/{study_id}/start | 空 JSON 对象 | 原子占用后后台 runner.run；202。相同 study 已在本控制器执行返回 200 与当前状态，不重复启动 |
| GET /studies/{study_id} | 无 | 只读 study/execution 投影，返回状态、计数、试验与排名；不能启动或收尾 |
| POST /studies/{study_id}/stop | 空 JSON 对象 | 向持有本任务的控制器设置 stop_event；202 停止请求已提交。终态重复停止 200；丢失控制权 409，不猜进程或杀错任务 |
| POST /studies/{study_id}/resume | 空 JSON 对象 | 原子占用后 runner.resume；不改配置、不增预算、不清 BLOCKED、不重试旧 trial |
| GET /studies | offset=0、limit=20（1–100） | 列举受控根下合法 study，按创建时间倒序、ID 次序打破同分；坏记录显示不可读取及错误码，不静默消失 |
| GET /studies/{study_id}/best-config | 无 | 仅从重验证后的 rank_trials 首项返回六参数、固定条件、冻结数据/模型绑定及 source study/trial；无 SUCCESS 返回 409 HPO_NO_SUCCESS。只读、不启动 |

三根由应用现有项目/日志路径解析后在服务初始化时冻结：storage_root 为既有 log 根下 hpo/studies；output_root 为既有 Detect 输出根；log_root 为既有日志根。请求不能更改它们。根链与 study ID 校验沿用 H1.2。不得扫描任意用户路径。

创建和启动分离：双击由前端 pending 控制；网络结果不明时不得自动重新 POST 创建。可从历史找到已创建 READY study 后启动。同一 study 的 start/resume 在服务器端原子排重；空白/未知请求字段也必须拒绝。

状态响应至少含 study_id、execution_status、execution_revision、budget、claimed_count、terminal_count、success_count、current_trial_number、control_active、can_stop、can_resume、error_code、next_action、trials、ranking。计数从事实计算，终态槽不等于成功槽；trial 展示编号从 1 起并保留内部 number，不改变底层排序。

跨文件并发读取若处于短事务冲突，返回明确的暂忙响应并由前端稍后重试，不把中间快照伪造为损坏或空结果；不可无限重试隐藏真实损坏。读取一致性问题需要最小核心调整时，保留校验并添加竞争反例。

## 5 占用、停止与恢复

Studio 仍为单服务 worker。增加统一进程内原子 reservation：在任何创建训练目录、启动控制器、LLM 调参调用或子进程副作用前取得；同时来的普通训练、keep_params、full、HPO start/resume/验证训练只能一个取得。dry_run 不持有真实训练槽。已有跨进程 HPO 根锁继续保留，不宣称支持多 worker 或跨独立 CLI 的统一调度。

reservation 从预检到控制器最终收尾覆盖完整生命周期，不能在 SSE/HTTP 断开、停止请求返回或 controller 抛错时无条件释放。初始化失败且确定无进程可释放；进程状态不明/仍活跃需保留阻断。RunManager.active_for_kind 必须显式支持 hpo，未知 kind 不再错误映射 tuning。

服务重启后内存空闲不代表可训练：在新启动前核对已有持久化非终态记录与原进程身份。MATCH 仍占用；UNVERIFIABLE 或无法排除活动进程则阻断新训练；只读页面不领养原进程。MISSING/MISMATCH 按已有身份规则处理且不杀外来进程。HPO 由用户明确 resume 调用已有恢复矩阵；BLOCKED 没有强制继续入口。遗留无 command_executable 的旧 H1.2 attempt 显示不可读取，保留原文件，不能自动补齐或覆盖。

运行态显示准备中、运行中、正在停止、已暂停/已中断、预算已执行完、需要处理。COMPLETED 只表示预算耗尽；无成功时明确“没有可用结果”，不得显示调参提升。stop 返回请求接收不能立刻显示已停止；等待 runner 事实收敛。恢复条件无法事先确定时显示“检查并恢复”，返回原稳定拒绝原因及下一步，不承诺必定恢复。

## 6 历史与最佳配置复用

智能训练内展示 HPO 历史入口及每个 study 的试验表，链接既有单次训练详情。study/execution/产物仍为事实，不能依赖 SQLite 才能查看 HPO；本批不扩数据库 schema，不重复写训练历史。仅 SUCCESS 参与 ranking，使用 H1.2 的 mAP50-95(B) 最佳 epoch 口径与确定性同分规则。

“复用最佳配置”载入只读来源与可核对的训练表单，不自动启动。固定配置验证走普通训练控制器和统一门禁，使用绑定初始权重而非 winner 的 best.pt；保留同一快照、六搜索参数和 H1.2 固定条件。请求仅传来源 study/trial，服务端重新验证并从权威记录构造全部有效参数，不能信任客户端 candidate 字典。若用户改 epochs 等条件，明确标为新的对照条件；首版验证入口不允许修改这些条件，避免新增契约。记录 source study/trial 到普通训练附加元数据，不改 HPO trial 数或排名、不调用 LLM、不覆盖已有模型。

普通训练入口需增加互斥 source_hpo 对象（study_id、trial_id），与普通自由参数混用时 422。来源不是当前 rank 第一、失败或不存在时 409。输出使用新运行目录；args 六参数与固定条件实际生效纳入自动化和真实验收。

用户输入、失败消息与模型文件名按文本渲染，不拼接 innerHTML；完整堆栈、凭据及任意文件下载不可暴露。

## 7 验收与边界

Claude 自测：API 契约、无 LLM 调用、原子占用竞争、停止、刷新重连、受控重启与身份矩阵、损坏旧记录、排名、复用配置实际传递、原三模式回归和全量 pytest。真实进程仅使用受控本地短进程进行身份测试，不启动 YOLO。

Codex 独立：复跑并针对竞态、断线、恢复和参数污染审查；浏览器完成四选项、模式切换、输入错误、HPO 启动/停止/重连/历史/复用走查。用最小合法本地数据与权重，各 TPE/Random budget=2、epochs=1、batch=1、imgsz=64 做正常闭环，再进行一次最佳固定配置验证；停止/恢复另用必要最小场景验证，不把自动化替身当真实训练证据。短训练不证明算法优越或 LLM 效果提升。

艾卡确认页面符合习惯；核心无阻断问题且 Codex 明确验收通过后，Codex 回写文档并按批准上传指定 GitHub 仓库。H1.1/H1.2/H1.3 按批次可解释提交，不夹带无关修改、真实配置、数据、权重及日志，不擅自打正式产品 Release。

本批不做 F1 全面体验打磨及 LLM 效果专项，不新增依赖、不进入 Research/Cloud/Docker/API 平台交付。所有 Python 使用 D:\Program Files\anaconda3\envs\auto_tune\python.exe。文件不得删除；实现确需超出范围或改变已验收协议时，Claude 报告具体原因供 Codex 评审，不静默放宽规则。
