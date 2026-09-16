# H1.2 Detect HPO 训练执行与恢复闭环规格

2026-09-09 H1.2 独立验收通过。全量 1837 passed、2 条既有 PCA 警告（97.89 秒），累计 21 项独立反例通过；TPE/Random 各两次真实短训练与命令、args、CSV、历史、finished_at 及冻结 executable 核对通过。详见 docs/h1_2_codex_review_20260908.md 的最终验收结论。

兼容边界：ExecutionAttempt 必须含 command_executable；已含 attempt 但缺字段的旧预验收 execution-v1 记录拒读，不补齐、不覆盖。旧产物保留追溯，后续运行使用新 study ID。此为未发布阶段格式收紧，不承诺同时篡改多个字段的防伪能力。

验收后契约修订（2026-09-09）：ExecutionAttempt.command_executable 为必填、非空、最长 1024 的字符串，首次创建 attempt 时从已构造命令复制冻结，与 command[0] 交叉一致；历史命令重建采用此字段，不依赖当前 resolver，新启动再比对当前 executable。缺字段的旧 attempt 读入返回 HPO_CORRUPT_EXECUTION，schema 标识仍为 hpo-execution-v1，不自动迁移。build_yolo_command 增加可选 executable 参数，省略时沿用原解析行为。启动前同字节校验 args 摘要和完整计划参数语义，LAUNCH_INTENT 前预验、临近启动再次复验；HpoError 保留稳定错误码并阻断预算。RunState 新增可选 finished_at，HPO 终态按结果事实写入。以下初版字段清单、伪代码如有遗漏，以本修订及最终验收记录为准。

本批已完成并通过独立验收；以下初版计划保留用于实现追溯，验收修订与格式边界以本文开头及最终验收记录为准，不作为新一轮编码指令。

日期：2026-09-08。作者与验收：Codex；计划实现：Claude Code；方向确认：艾卡。

状态：艾卡已于 2026-09-08 确认本方案，交 Claude Code 按配套提示词实施；本次仅完成交接，尚无 H1.2 实现或训练验收结果。前置为 H1.1 返修验收通过，完整套件 1672 passed、2 条既有 PCA 警告，证据见 `docs/h1_1_codex_review_20260907.md`。

## 1 范围与交付

在 H1.1 HpoService 之上增加可直接由 Python 调用的顺序执行器：执行固定预算、停止、恢复、读取真实指标与确定性排名。一个 trial 从同一绑定的初始模型开始，不接续上一 trial 的权重。失败不重试原 trial，不自动降低 batch，不扩预算。一次 resume 是继续尚未消耗的预算，不是 YOLO resume checkpoint。

本批不修改 UI、HTTP 路由、前端脚本、LLM 调优控制器、配置凭据或数据库 schema，不接 Optuna RDB，不新增依赖。H1.3 沿用智能训练页面，模式依次为干运行（仅生成计划）、按原来参数训练、HPO 算法调参、大模型调参。四个页面选项包含一个计划预览模式，真实训练策略仍为三种；HPO 不经过 LLM。H1.2 不承诺与尚未接入的 Studio 入口共用并发门禁，H1.3 必须完成该接线。

## 2 现有代码依据与复用边界

- `hpo/service.py`：create_study/ask/tell/load_study；维持 hpo-study-v1、detect-hpo-v1、rebuild-per-trial-v1、原预算/条件空间/严格输入与幂等语义。
- `agent_engine/executor.py`：validate_training_preflight、write_training_config、build_yolo_command、launch_training(command=精确命令)。HPO 不用需要 reference_run 的 prepare_training，也不扫描“最新训练”或“最新数据集”。允许仅给命令白名单追加 task、amp，以显式固定 detect 与禁用 AMP；既有调用行为保持兼容。
- `agent_engine/guardrails.py`：validate_and_clamp；HPO 必须拒绝任何实际参数变化，不能将 clamp 后结果当作原候选。
- `run_state/process_identity.py`：capture_process_identity/compare_process_identity 及 IdentityMatch；不得仅凭 PID 杀进程或判定恢复安全。
- `run_state/service.py`：new_run_state("tuning")、with_status_phase、with_terminal、write_run_state。trial 使用已有 tuning 类型，附加 HPO 身份放执行审计；不伪造新的 run_kind。
- `train_analyzer/training_finalizer.py`：finalize_training_run(source="tuning", session_id=study_id, runtime_run_id=trial 的运行身份)。tuning_context 标记 strategy="hpo"、study_id、trial_number；复用无 LLM 的分析和幂等历史 upsert，排名不用 finalizer 的 final_metrics。
- 不直接复用 ManualRunController 的生命周期：它的 _persist 会吞掉写盘失败，不满足 HPO 的审计门槛。不为本批重构普通训练控制器。

## 3 公共契约与固定训练条件

新增接口集中导出于 `hpo/__init__.py`，所有模型实例与字典在公共入口重新严格验证，extra=forbid；禁止 bool 充当数值、字符串/NumPy 隐式转换及 NaN/Inf。

```python
ExecutionConfig(batch=16, imgsz=640, device="cpu", timeout_seconds=3600)
HpoRunner(storage_root: Path, output_root: Path, log_root: Path)
runner.prepare(study_id: str, config: ExecutionConfig) -> ExecutionRecord
runner.run(study_id: str, *, stop_event: threading.Event | None = None) -> ExecutionRecord
runner.resume(study_id: str, *, stop_event: threading.Event | None = None) -> ExecutionRecord
runner.status(study_id: str) -> ExecutionRecord
rank_trials(record: StudyRecord) -> list[TrialRecord]
extract_objective(run_dir: Path, *, artifact_root: Path,
                  run_id: str, epochs: int) -> ResultInput
```

ExecutionConfig：batch 原生整数 1..256；imgsz 原生整数 32..2048 且为 32 倍数；device 仅 "cpu" 或无前导零的单个非负 GPU 索引字符串（0..63）；timeout_seconds 原生整数 1..86400。设备默认 CPU，调用者可显式选已有 GPU；禁止 auto batch、多 GPU、动态设备回退。缺失 GPU 返回预检错误，不偷偷改用 CPU。

固定值：task=detect、workers=0、resume=False、deterministic=True、patience=0、val=True、save=True、plots=False、amp=False；model/data/epochs/seed 来自 study 绑定及 config，其余六个搜索参数来自 candidate_params。其余 YOLO 默认值由固定环境版本约束，最终实际 args.yaml 完整记录，不从全局 config.yaml 或参考训练混入参数。workers=0 与单设备避免本批引入 DDP/多 worker 生命周期。

prepare 深拷贝并冻结配置、三个绝对根路径和运行环境（sys.executable、Python/torch/CUDA/ultralytics/optuna/numpy 版本）。相同配置重复 prepare 返回既有记录，不同配置返回 HPO_EXECUTION_CONFLICT。不修改 H1.1 study config。已有未绑定执行器的 PENDING trial 可在 prepare 中登记；已经有终态 trial 的 study 不允许首次接入执行器，返回 HPO_EXECUTION_CONFLICT，避免把外部合成结果混入真实排名。已绑定 study 的正常恢复不受此限制。

run 仅启动 READY 的执行记录；已有完成结果幂等返回；PAUSED/INTERRUPTED 须显式 resume。resume 对同一记录重复调用不得重复启动。status 只读取校验后的事实，不触发训练、恢复或停止。stop_event 是当前 Python 调用者的协作停止信号，本批不增加跨 HTTP 停止接口。

## 4 持久化与身份

在 `storage_root/<study_id>/execution.json` 保存 hpo-execution-v1。study.json 仍是采样与 trial 结果事实；execution.json 是执行与产物审计事实，不能修改 study.json 的 schema 或绕过 HpoService 写 trial。

ExecutionRecord 必含 schema_version、study_id、revision（初始 0，每次成功发布 +1）、created_at/updated_at、config、roots、environment、status、stop_reason、attempts。status 仅 READY/RUNNING/PAUSED/INTERRUPTED/COMPLETED/BLOCKED；BLOCKED 表示安全性或审计事实不足，不能简单 resume 强行解除。

每个 ExecutionAttempt 必含 trial_number、trial_id、request_id、run_id、phase、candidate_params、effective_params、command、run_relpath、args_sha256、actual_args、actual_args_sha256、metric_diagnostics、pid/process_create_token、started_at/finished_at、returncode、termination_reason、result（待提交的 ResultInput）、finalizer_record、error_code、error_message。尚未产生的字段显式 None；termination_reason 仅 None/user_stopped/timeout/audit_failure，终止前先持久化，恢复时不能因退出码 0 丢失停止原因；写失败则走持久化故障停止路径。metric_diagnostics 仅包含 total_rows/excluded_rows 两个非负整数且 excluded_rows<=total_rows。phase 仅 PREPARED/LAUNCH_INTENT/RUNNING/EXITED/RESULT_READY/TOLD/FINALIZED。按 trial_number 连续，身份均与 study 对齐；同一 trial 只有一次启动机会。错误消息有长度上限 2048，不记录环境凭据。

输出目录固定为 `output_root/<study_id>/<trial_id>/`；Evidence.artifact_relpath 相对冻结的 output_root，使用 POSIX 分隔符。命令中的 project/name 与该目录严格一致。已有陌生目录、已有未审计产物不覆盖。write_training_config 会修改字典，必须传深拷贝并记录最终实际传入命令的参数。

execution.json 采用与 H1.1 等价的原子 JSON：UTF-8、重复键/非有限拒绝、5 MiB 上限、flush/fsync/os.replace、失败不推进内存 revision，不用默认值重建损坏文件。读写均校验跨字段状态、身份、路径、预算与结果语义。原文件损坏不能被后续正常对象覆盖。

新增执行锁使用已验证的 OS 非阻塞锁与进程内共享锁模式；`storage_root/.hpo-runner.lock` 在 run/resume 全生命周期持有，防止同一受控根下多个 HPO 同时训练，BUSY 时零启动。执行记录读改写另用 study 内短事务锁。锁顺序固定为全局执行锁 → 执行记录锁 → H1.1 study 锁；不得反向获取，也不得持有 study 锁等待训练完成。所有根、父链、记录、锁、输出与 CSV 拒绝 symlink/reparse 和路径穿越。stop_event 的检查不依赖另一个线程取得这些锁。

## 5 从采样到收尾的提交顺序

1. run/resume 取得执行锁，重载两份事实、校验路径、环境和数据/模型绑定。新增 HpoService.validate_binding(study_id) 公开只读接口，复用现有 _check_environment/_check_binding，供幂等 PENDING 执行前也进行逐文件复验；不要通过重新 ask 幂等分支假设已复验。
2. 新 trial 的 request_id 固定为 uuid5(NAMESPACE_URL, f"hpo-execution-v1:{study_id}:{number}").hex。ask 原子发布候选；ask 后崩溃且尚无执行记录的尾部 PENDING 可用同一 request_id 收编，不另取样。
3. 合并严格固定参数，验证候选和护栏；guard.valid=False、guard.clamped 非空或 guard.params 与输入不同均零启动并将该 trial 登记 FAILED/invalid_params。完整预检通过后创建专属目录、写 args、构造唯一精确命令。任何配置/预检失败要保留具体 error_code，不能只保存“训练失败”。
4. 分配 new_run_state("tuning") 的 run_id，写 run_state.json 及 PREPARED 执行记录；写盘失败不能启动。记录命令、候选、effective_params、输入 args SHA256。
5. 写 LAUNCH_INTENT 并成功落盘，随后唯一一次调用 launch_training(command=记录中的命令)。返回后立即捕获 pid+创建 token，写 RUNNING；短命进程如已退出则直接记录 EXITED。活动进程身份无法捕获时终止并 wait，阻断本轮；不能带着不可追溯进程继续执行。
6. 用 poll + 单调时钟监控，周期上限 0.2 秒，输出由现有 executor 写 yolo_train.log，不使用可能阻塞的 readline。每轮先看已观察到的进程退出，再判 stop_event、timeout。停止/超时调用 terminate，等待最多 10 秒，再 kill 并等待最多 10 秒；未确认退出保持 BLOCKED，禁止下一个 trial。
7. 进程退出后持久化 EXITED 和退出码；只有返回码 0 且未实施停止/超时的 trial 才解析真实 results.csv 与 args.yaml。验证实际 args 的六搜索参数及固定条件与计划一致（路径规范化比较，imgsz 允许整数或同值单元素列表；实数用 rel_tol=1e-9、abs_tol=1e-12）；不一致 FAILED/invalid_params。
8. 持久化 RESULT_READY（含完整 ResultInput 与证据），再 tell；tell 成功后写 TOLD。两个文件之间故障通过同一 payload 重放 tell 收敛，不重训。终态冲突必须 BLOCKED，不能覆盖。
9. 对确实启动过的 trial 调用共享 finalizer，记录返回值；没有启动的 invalid_params 不伪造训练历史。finalizer 的分析/索引失败独立记录，不改合法 objective；history_error 或执行审计落盘失败阻断后续 trial。保存 FINALIZED 后才能继续 ask。最终按预算终态数写 COMPLETED；COMPLETED 表示预算流程结束，不保证存在成功 trial。

任何持久化失败停止取样；若持有本轮活动进程则尽力终止并确认退出。写不进去时只返回 HPO_PERSISTENCE_ERROR，不声称磁盘已写 BLOCKED，不尝试用后续 flush 发布失败的内存变更。恢复只以最后一次成功发布事实为准。

## 6 失败、停止与恢复矩阵

| 事实/事件 | trial 结果及执行器行为 |
|---|---|
| OOM（非零退出且日志含 CUDA out of memory / OutOfMemoryError） | FAILED/oom，下一预算 trial；不自动重试、不缩 batch |
| 其他非零退出、启动异常、无有效指标 | FAILED/training_failed；具体子错误写执行审计；可继续预算 |
| 超时并确认退出 | FAILED/timeout，继续下一预算 trial |
| 已收到停止且未取样 | PAUSED，无新 trial，不消耗预算 |
| 已取样，启动前或运行中实施停止并确认退出 | CANCELLED/user_stopped，PAUSED；预算已消耗 |
| 已观察到正常退出后才收到 stop_event | 完成已有结果登记与收尾，PAUSED，不再 ask |
| PREPARED，无 LAUNCH_INTENT | 校验绑定与命令/args 后可启动一次；stop_event 已置位则取消 |
| LAUNCH_INTENT，无可核验进程身份/无退出事实 | BLOCKED/HPO_RECOVERY_REQUIRED，零启动，不根据 CSV 或等待时间猜测成功/未启动 |
| RUNNING，PID+token MATCH 或 UNVERIFIABLE | BLOCKED/HPO_PROCESS_STILL_ACTIVE 或 HPO_RECOVERY_REQUIRED；不附着、不杀未知进程、不启动下一 trial |
| RUNNING，身份 MISSING 或 MISMATCH | 原训练身份已不存在，INTERRUPTED/process_interrupted；MISMATCH 不杀复用 PID；显式 resume 后继续剩余预算 |
| EXITED，已持久化退出码 | 从本 trial 产物解析，不再次启动 |
| RESULT_READY/TOLD | 幂等 tell / 重放 finalizer upsert，最终 FINALIZED；不得重训 |
| study 终态与执行结果不一致、文件损坏/跨 study 替换 | BLOCKED/HPO_CORRUPT_EXECUTION 或 HPO_RESULT_CONFLICT；零启动、保留原文件 |

LAUNCH_INTENT 是不可原子化的外部副作用窗口：本批优先保证不重复训练，不承诺此极窄窗口无人干预自动恢复。BLOCKED 无“强制继续”接口；resume 可重新核验阻断原因，只有新事实证明条件解除（例如已记录身份的进程现已消失、历史写盘恢复）才按矩阵继续；缺失启动身份不能仅因时间流逝解除。后续可由 Codex 核实运行事实，不能把猜测做成自动恢复。普通进程已确认消失的中断、写 tell 前后的故障必须自动幂等恢复。停止和恢复不复活 CANCELLED/INTERRUPTED trial，只消耗剩余名额。

## 7 指标与排名

仅读取已退出的对应 run_dir/results.csv，一次读字节并对同一字节计算 SHA256 后解析，大小上限 5 MiB。用 csv 标准库，不用 DataFrame 的宽松数值推断。表头去首尾空白，拒绝重复表头；必须存在 epoch 与 metrics/mAP50-95(B)。epoch 必须是十进制正整数、1..config.epochs、严格递增且不重复。非法结构直接判无可信指标。

目标列空值、无法转为有限数值、NaN/Inf、超出 0..1 的行不参与最大值计算，记录排除数量；全部无效则 FAILED/training_failed，子错误 HPO_INVALID_METRICS，不能返回 0 伪造成功。有效行最大 mAP50-95 为 objective，同值取最早 epoch。Ultralytics 8.3.253 本地 trainer.save_metrics 写 self.epoch+1，本批按 1-based epoch 接收，不自动猜测 0-based。

rank_trials 重新严格验证 StudyRecord，仅 SUCCESS 排序，key=(-value, trial.number)。空列表表示无优胜 trial；不调用 Optuna.best_trial、LLM 或使用 best.pt fitness 替代目标。排名为可重算视图，不建第二份排名事实。执行审计保留 metrics 来源与实际参数；通用历史 final_metrics 与 HPO 最佳 epoch 分数是不同口径，H1.3 展示时必须区分。

## 8 文件与错误边界

新增 `hpo/execution_models.py`、`execution_storage.py`、`execution_adapter.py`、`execution.py`、`metrics.py`、`ranking.py`，及一一对应的测试文件。修改仅限 `hpo/__init__.py`、`hpo/service.py`（公开只读绑定校验）、`agent_engine/executor.py`（task/amp 白名单）及必要已有测试。不改 H1.1 搜索协议或终态枚举；不将 H1.2 错误码塞入 trial.reason_code。

新增错误码：HPO_INVALID_EXECUTION_CONFIG、HPO_EXECUTION_CONFLICT、HPO_EXECUTION_BUSY、HPO_CORRUPT_EXECUTION、HPO_PREFLIGHT_FAILED、HPO_INVALID_METRICS、HPO_RECOVERY_REQUIRED、HPO_PROCESS_STILL_ACTIVE。复用 HPO_PERSISTENCE_ERROR、HPO_RESULT_CONFLICT 与 H1.1 绑定/环境错误。配置错误零发布，存储错误零后续启动，环境/绑定漂移阻止新启动，但允许将已经退出且有持久化结果的 trial 做幂等 tell/收尾；恢复新训练前再严格复验。

## 9 验收与估算

预计 H1.2 3–5 人日，包含实现、故障注入、Codex 独立验收及短训练；GPU 等待、环境问题另计。这是本批估算，不与原 H1 总估算 6–10 人日重复相加，也不意味着原整体工期已重新估算。

Claude Code 完成自动化测试与报告；Codex 独立复跑并做真实短训练。当前 Conda 解释器固定 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`（3.10.18），H1.1 五个已装依赖版本不变，pip check 前后留证。新增测试必须包含跨进程锁、真实 Windows 路径链接、模拟所有启动/提交故障窗口，不能靠 skip 过关。

真实验收：受控最小合法 Detect train/val 数据快照、已有本地 YOLOv8 Detect 权重；TPE 和 Random 各 budget=2、epochs=1、batch=1、imgsz=64、workers=0，已有 GPU 单卡或明确 CPU，最多 4 次成功短训练。另用可控本地假训练进程验证超时、停止与崩溃窗口，不故意制造真实显存 OOM。TPE startup 后路径由至少 6 trial 的合成自动化覆盖，不能把两个真实 trial 声称为 TPE 贝叶斯搜索质量证明。无 LLM 配置也能执行；禁止下载权重/数据或隐式安装。缺少合法本地输入则如实报告，不能以合成目标代替真实验收结论。

不提交、推送、发布或进入 H1.3。艾卡在 UI 修改前无需介入测试；Codex 更新验收与权威文档后再报告下一步。
