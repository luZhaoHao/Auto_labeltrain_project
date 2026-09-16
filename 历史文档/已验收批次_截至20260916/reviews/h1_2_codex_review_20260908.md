# H1.2 Codex 独立验收记录与返修要求

## 2026年9月9日最终独立验收 当前有效结论

**H1.2 验收通过。** R1–R7 及后续 R2a-1/R2a-2/R2a-3、R2b/R2c 的已列问题均已通过独立复验，无待返修阻断项。未提交、未推送；H1.3 尚未编码，下一步由 Codex 制定 Studio 接入规格与计划，按艾卡确认的四种训练选项推进。

### 本次独立证据

- 解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。
- 全量 `python -m pytest auto_tune/tests -q -p no:cacheprovider`：**1837 passed、2 条既有 sklearn PCA 警告，97.89 秒**；未跳过测试。
- 四轮累计独立反例：**21 passed，12.16 秒**。对应原探针、round2、round3、round4 测试文件，未用交付报告数字代替本次复跑。
- `python -m pip check`：No broken requirements found；未新增或升级依赖。相关已跟踪文件 diff 检查通过。
- TPE、Random 各 budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120，使用既有最小合法快照和本地 YOLOv8n 权重，**4 次真实训练均 SUCCESS/FINALIZED，两个 CLI 退出码均 0**。
- `log/h1_2_codex_review_20260909/verify_real_evidence.py` 逐项核对命令日志、command_executable、六搜索参数与固定条件、实际 args 与摘要、CSV 同字节 SHA256、最佳 epoch、tell、排名、历史及 finished_at，通过。四个 trial 均 AdamW、mAP50-95 为真实 0.0、最佳 epoch=1，同分 trial 0 优先。
- 旧记录验证：`python -m log.h1_2_codex_review_20260909.verify_legacy_readonly` 对前轮两个 study 分别调用 status/run/resume，共六次均 HPO_CORRUPT_EXECUTION，execution.json 字节不变、无启动。验证脚本最初按文件运行时遇到项目导入路径问题，改为从根目录以模块方式执行后通过；该导入错误不属于业务缺陷。

### 冻结身份和旧记录边界

ExecutionAttempt 的 `command_executable` 为必填字段，与 command 分开持久化。模型和跨记录校验要求它等于 command[0]；历史命令重建以该字段为依据，不解析当前 YOLO。新启动仍以当前环境解析结果进行严格比对。准确实现事实是：runner 在首次创建 attempt 时将已构造命令的 executable 复制到独立字段，并非再次独立调用 resolver；这满足本次限定的交叉字段一致性要求。它不构成签名校验，不承诺发现两个字段同时被一致篡改。

本次接受未发布 H1.2 预验收格式的兼容中断：已有 attempt 缺少 command_executable 的旧 execution-v1 记录拒读，不补齐、不迁移、不覆盖。保留旧 study/产物供追溯；后续验证使用新 study ID 和新目录，禁止在旧 study 上删除记录或强行重跑。尚无 attempt 的空记录不包含该必填字段，不能笼统称所有旧文件必然拒读。格式标识仍为 hpo-execution-v1；以上是未发布阶段的必填字段收紧，后续发布版本不得把此先例用于静默破坏已发布格式。

本次验收不包含 UI/API 接线、真实 SGD、长训练性能或真实训练中的全部故障注入；停止/超时/恢复边界依据自动化与既有受控进程测试，四次正常训练不替代这些故障证据。短训练结果不证明模型质量或 TPE 优于 Random。无新增依赖、无业务代码改动由 Codex 在本轮实施；仅独立验证和通过后的文档回写。

H1.2 初版编码及返修提示词已失效，归档后只用于追溯；不得据此启动 H1.3。以下为验收前各轮历史，旧“未通过”结论及数字不代表当前状态。

## 第四轮返修复验 历史记录

2026-09-08：**尚未验收通过，仅剩下述 executable 历史审计缺口；不提交、不推送、不进入 H1.3。** 本轮已独立确认 R2a-1 内容复验、R2a-2 校验错误阻断及 R6 无当前 YOLO 的历史收尾行为通过前三轮反例。完整运行结果为 **1830 passed、2 条既有 PCA 警告，91.87 秒**；累计 19 项原独立反例 **19 passed，8.10 秒**；pip check 无冲突。解释器 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。本轮 Codex 未修改业务代码或依赖。

本轮再次通过公开 CLI 执行 TPE、Random 各 budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120，四次真实训练均 SUCCESS/FINALIZED，两个 CLI 均退出 0。`log/h1_2_codex_review_20260908/round4/verify_real_evidence.py` 核对命令日志、六搜索参数与固定条件、实际 args 及摘要、CSV 字节哈希、最佳 epoch、tell、排名、历史和 finished_at 全部通过。四个 trial 均 AdamW、真实 mAP50-95=0.0、epoch=1，同分 trial 0 优先。此为正常链路证据，不证明调参质量；未新增 SGD 或真实故障中止训练。

### R2a-3 P2 历史 executable 使用被验证字段自身作为依据

位置：`auto_tune/modules/hpo/execution.py` 的 `_cross_validate`，传入 `executable=attempt.command[0]` 重建预期命令（约 694–697 行）。

独立反例：先正常完成 study，将 FINALIZED attempt 的 command[0] 单独替换成 `C:/unrelated/not-a-training-program.exe`，其他字段不变，status 未拒绝。因为预期 executable 直接取自被校验字段，无论替换成什么都与自身相等。第四轮独立探针 `round4/test_round4.py` 为 **1 failed、1 passed，3.28 秒**；配套正向边界证明同样的替换在真实 adapter 新启动前被 HPO_PREFLIGHT_FAILED 拦截，launch_training 零调用。

影响仅按已验证证据界定：历史记录可能把已完成训练归于不相关 executable；不是已经证实的错误程序启动漏洞。Claude 已披露这一限制，但此前提示词明确要求历史命令污染仍拒绝，艾卡没有批准削弱此要求，不能将其直接视为已接受权衡。

返修要求：为 executable 提供与 attempt.command 分离的冻结依据，历史命令校验只能使用该依据，不解析当前环境；新启动继续解析当前环境并与冻结依据比较。缺失依据的旧记录必须采用明确、可审计的兼容策略，不能从待验证的 command[0] 自动补齐后声称验证通过。此处要求交叉字段审计一致性，不扩展为签名系统或抵御所有文件同时被篡改。保留已通过的 args 双检、错误传播、预算、停止和恢复行为。若涉及 schema/兼容语义变化，先给出具体方案由 Codex 核对，再编码该部分；不能再次单方面降低历史校验要求。

使用现有 `历史文档/H1.2_20260909_已验收/h1_2_claude_code_repair_round2_20260908.md` 入口，内容已更新为本轮要求。前三轮结论与数字仅作历史证据。

## 第三轮返修复验 历史记录

2026-09-08：**仍未通过验收，不提交、不推送、不进入 H1.3。** R2b 固定参数对账与 R2c 预算耗尽检查已补齐，原两轮 16 项独立反例全部通过（8.09 秒）。本轮独立全量 **1818 passed、2 条既有 PCA 警告，92.76 秒**；pip check 无冲突。解释器为 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。未新增依赖或修改业务代码。

本轮重新执行 TPE、Random 各 budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120，四次真实训练全部 SUCCESS/FINALIZED，两个 CLI 退出码均 0。本轮 `round3/verify_real_evidence.py` 独立核对命令日志、实际 args/哈希、六搜索参数及固定条件、CSV 字节哈希、最佳 epoch、tell、排名、历史和 finished_at 全部通过。四个 trial 仍均为 AdamW、真实 mAP50-95=0.0、epoch=1，同分 trial 0 优先。正常链路通过不代表故障恢复通过，也不证明调参质量。本轮没有进行真实故障中止训练或 SGD 训练。

独立补充反例 `log/h1_2_codex_review_20260908/round3/test_round3.py` 最终 **3 failed，6.32 秒**。第一项使用完整合法的冻结 effective/命令，仅污染磁盘 args.epochs 并更新其摘要；失败不是夹具缺字段。测试启动函数均被 spy/fake 替换，污染命令没有真实执行。剩余问题属于 R2a 未完成及其引入的 R6 回退，下一轮按下列三项返修，第二轮记录保留为历史。

### R2a-1 P2 args 只验证摘要 未验证内容与计划一致

位置：execution_adapter.py 的 launch（约 224–260 行）。完整合法 attempt 的 effective/command 不变，将磁盘 args.yaml 的 epochs 改为 999 并提供对应正确 SHA256，launch 仍调用 launch_training。当前仅检查字节摘要和 command==build_yolo_command(effective)，没有解析 args 内容与权威 effective 对账。

影响是接受不真实的准备阶段参数审计事实；当前 executor 命令由 effective 构建，此反例不代表实际会按污染 args 的 999 epochs 训练。要求在启动副作用之前对已读取的同一份字节进行安全解析和完整语义复验，验证六搜索参数、固定项、输入绑定及类型；缺字段、非法数值/类型、矛盾字段均拒绝。保持摘要校验，不能以重新写文件消除污染，也不能把将来训练覆盖 args 当作验收通过。

### R2a-2 P1 启动校验损坏错误被吞掉并继续预算

位置：execution.py 的 _launch_new except Exception（约 320–337 行）。让 adapter.launch 抛 HPO_CORRUPT_EXECUTION（args hash mismatch），runner 未向外抛出该错误，反而登记普通 FAILED/training_failed 并进入下一预算槽；独立日志可见两次取样。

要求将验证失败与真正的进程创建失败分开。完整磁盘/命令预验应在发布 LAUNCH_INTENT 之前完成；临近启动仍需复验，复验如发现损坏须保留稳定原错误码、阻断后续取样和启动、保留证据。若已发布 LAUNCH_INTENT，必须明确“已知尚未启动”与“无法确认副作用”事实，不能把不确定结果自动认定为训练失败。不得笼统用 except Exception 把 HpoError 降级成普通训练失败。

### R6 回退 P1 历史收尾依赖当前 YOLO 可执行文件

位置：execution.py 的 _cross_validate 调用 _authoritative_command（约 671 行），后者经 build_yolo_command 调用动态 resolve_yolo_executable。

复现：第一次执行因 finalizer history_error 留在 TOLD；修复 history 后模拟当前 YOLO 无法解析，resume 在入口命令重建时抛 FileNotFoundError，原 attempt 仍 TOLD，无法先补齐 FINALIZED。PATH 指向另一 YOLO 时也会改变动态预期命令，按当前代码导致历史记录被当作损坏。这是已经批准的“环境漂移只阻止新训练、允许已发布结果幂等收尾”契约回退，不是可接受的环境稳定假设。

要求拆分纯历史事实校验与启动环境校验。历史命令参数仍必须与冻结事实一致，不得为允许恢复取消命令/参数完整性验证；但纯 status、RESULT_READY/TOLD 收尾不能解析或要求当前 YOLO 存在。新训练前再解析并核验当前 executable/环境，出现漂移时零新启动，错误稳定映射到 HPO 错误。若必须补充冻结可执行文件身份或调整 builder 接口，应明确兼容旧 execution-v1 记录的规则，不能静默迁移、重新猜测旧记录身份或直接放行。

下一轮继续使用 `历史文档/H1.2_20260909_已验收/h1_2_claude_code_repair_round2_20260908.md`（内容已更新为本轮后续返修，文件名保留以稳定引用）。R2b/R2c 不重新开发，R1/R3–R7 原有通过行为不得回退。全部真实产物与探针留在忽略的 log 目录。

## 第二轮返修复验 历史记录

2026-09-08：**仍未通过验收，不提交、不推送、不进入 H1.3。** 原 12 项独立反例现为 12 passed；本轮独立全量为 **1805 passed、2 条既有 PCA 警告，81.27 秒**，pip check 无冲突。解释器仍为 auto_tune Conda 环境 Python 3.10.18。本轮未修改业务代码或新增依赖。

本轮重新执行 TPE、Random 各 budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120，四次真实训练全部 SUCCESS/FINALIZED，两个 CLI 退出码均为 0。独立产物核对通过：命令日志、六搜索参数及固定条件、实际 args 与 SHA256、CSV 字节哈希及最佳 epoch、tell、排名、历史身份与收尾一致。四份 finished_at 均非空并与 attempt.finished_at 相等。四个 trial 均 AdamW，mAP50-95 为真实 0.0、最佳 epoch=1，同分 trial 0 优先；此结果不证明调参效果，也未覆盖真实 SGD 或故障中止训练。

本轮证据位于忽略目录 `log/h1_2_codex_review_20260908/round2/`。独立 `test_round2.py` 为 **4 failed，5.22 秒**，全部因预期拒绝却未抛错，不是环境或准备失败。`verify_real_evidence.py` 对本轮 real 目录核对通过。剩余三项均是原 R2 契约的遗漏，不能以正常短训练通过替代。

### R2a P1 命令形状与 args 哈希不能证明执行命令正确

位置：execution.py 的 _valid_command_shape/_cross_validate（约 68、659 行），execution_adapter.py 的 _looks_like_yolo_command/launch（约 87、229 行）。向已有命令追加 `lr0=999`，status 仍接受。另一个独立反例保持 args.yaml 内容及其正确 SHA256，传入 `yolo train lr0=999 epochs=999`，launch 仍调用了被 spy 替换的 launch_training。反例没有真正执行污染命令。

要求：从权威 study、ExecutionConfig、绑定输入、受控输出路径与已验证 candidate 重建预期 effective 和命令；逐项验证 executable、参数和值，包括 project/name/exist_ok。拒绝重复键、额外覆盖参数、无关可执行文件或子命令。完整验证必须先于 LAUNCH_INTENT/启动副作用；启动前再次核对磁盘 args 及其与计划的语义一致性。仅检查名字包含 yolo 和存在 train 不足以接受。损坏执行记录必须稳定拒绝，不得伪装普通训练失败后继续预算。

### R2b P1 固定条件没有与权威配置对账

位置：execution.py 的 _cross_validate（约 651 行）。仅修改 execution.json 中 effective_params.epochs 为 999，保留 study epochs=30 及合法 candidate，status 仍接受。当前只比较六个候选字段；即使实际训练产物与被污染的 effective 一致，也不能证明遵守了冻结配置。

要求：从原始绑定重建完整 effective，验证 epochs/seed、batch/imgsz/device、model/data 和所有 FIXED_PARAMS，不能把可编辑的 attempt.effective_params 自身当作权威。补充单字段与 effective/command 同时污染的反例，status/run/resume 均须拒绝且零启动。

### R2c P2 未消耗预算也能声称 COMPLETED

位置：execution.py 的 _cross_validate COMPLETED 分支（约 679 行）及 run/resume 的提前返回（约 153、166 行）。将刚 prepare 的空执行记录 status 改成 COMPLETED，study 尚无 trial、budget=2，status 仍接受；当前检查只排除现有 unfinished/PENDING，空集合绕过检查。

要求：COMPLETED 必须与 study 的预算耗尽事实一致，不能只用 all/any 检查现有记录。保留“预算完成可以包含失败”的公共语义和 CLI 全成功要求。若合法护栏拒绝路径没有 attempt，应按既有协议处理，不要机械要求每个 trial 都有 attempt，也不要为了让反例通过放宽完成条件。

### 下一轮交付范围

Claude Code 仅返修上述 R2a–R2c 与对应测试；保留已通过的 R1、R3–R7 行为。将本轮四项反例转为正式测试并增加完整参数矩阵，修正 FakeAdapter 中与 study epochs 不一致的 command 夹具，不能靠放宽生产校验保留旧夹具。真实进程清理“无法确认退出”的限制仍需如实保留；本轮四次正常完成不构成该故障路径的实测证据。下一轮提示词见 `历史文档/H1.2_20260909_已验收/h1_2_claude_code_repair_round2_20260908.md`。

以下为第一轮验收历史，1786 passed、12 failed 和 finished_at 为 null 均是返修前事实，不代表当前结果。

---

日期：2026-09-08。结论：**本轮验收未通过。真实短训练正常链路已验证，但恢复、审计和进程清理边界仍有阻断缺陷；不提交、不推送、不进入 H1.3。**

本轮仅审查、独立验证与文档回写，没有修改 H1.2 业务实现。Claude Code 的“六任务完成”是交付声明，不能替代验收；下列问题应返修后再复验。H1.1 已通过的结论不撤销。

## 独立验证证据

- 解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。
- 独立全量：`python -m pytest auto_tune/tests -q -p no:cacheprovider` → **1786 passed、2 条既有 sklearn PCA 警告，74.12 秒**。
- `python -m pip check` → No broken requirements found；未安装或升级依赖。
- 独立边界探针：`log/h1_2_codex_review_20260908/test_independent_review.py` → **12 failed，8.30 秒，无跳过**。探针使用受控 pytest 临时目录、假训练进程与真实 Windows 符号链接，没有额外真实 YOLO 训练或网络 LLM 调用。最终结果已排除测试准备阶段错误；每项失败对应下文的实际断言。
- 范围内 diff/文档检查与现有测试通过不能覆盖缺失边界。工作区仍有早于 H1.2 的无关修改，不能整体提交。

## 真实短训练结果与限制

使用已有合法最小 Detect 快照（5 train / 3 val）与本地 YOLOv8n 权重，既有单卡 GPU；未下载资源、未复制正式数据集、未修改训练环境。通过公开验收 CLI，TPE 与 Random 各执行 budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120，总计 **4 次真实训练，4 个 SUCCESS，4 个 FINALIZED，退出码均 0**。

| 采样器 | trial 0 / trial 1 | 最佳 epoch | 同分排名 |
|---|---|---|---|
| TPE | mAP50-95 0.0 / 0.0；均 AdamW | 均为 1 | trial 0 优先 |
| Random | mAP50-95 0.0 / 0.0；均 AdamW | 均为 1 | trial 0 优先 |

独立脚本 `log/h1_2_codex_review_20260908/verify_real_evidence.py` 直接读取真实产物，核对 command 与 yolo_train.log 首行、六搜索参数与固定条件、实际 args.yaml 及其 SHA256、CSV 同字节 SHA256、最佳指标/epoch、tell 结果、运行身份、相对产物路径、历史 study/trial 身份及 analysis/history 状态，全部一致。四份通用历史分析均 completed，无 history_error。

同时实查四份 run_state.json 的 finished_at 均为 null，见 R5。1 epoch 的 0.0 是 CSV 中真实有效零值，不是无效数据填零；此试验只证明正常执行链路，不证明模型质量或 TPE 优于 Random。两个 TPE trial 都在 startup 阶段，当前种子下两种采样器产生相同候选合理；没有验证真实 SGD 路径，不额外超出本次最多四次成功短训练授权。

运行产物、私人输入路径和探针保留在 Git 忽略的 log 目录，不提交。正式返修测试需从探针提炼到 auto_tune/tests 中，不能依赖日志目录中的测试代码。

## R1 P1 活动训练在写盘失败后未被清理

位置：`auto_tune/modules/hpo/execution.py`，_launch_new 的 RUNNING commit（约 308 行）以及 _monitor 的停止意图/状态写盘路径（约 345–358 行）。

复现：让 launch 返回仍在运行的假进程，在 RUNNING 记录发布时注入 HPO_PERSISTENCE_ERROR；run 抛异常，但 terminate/kill 从未调用，进程仍存活。当前仅在部分 run_state 写失败时做清理，未覆盖发布 RUNNING、termination_reason 和 stopping 状态失败。执行锁随异常释放，也不能代表进程已经结束。

返修：从拿到 Popen 开始为整个活动进程生命周期建立异常清理边界；任何写盘失败均停止取样，并对本次明确持有的进程执行 terminate→wait→kill→wait，核验退出。未确认退出不得报告成功或启动下一 trial。不要把失败内存记录当作已落盘状态，也不得因重放清理而误杀复用 PID。测试每个 launch 后故障点，既断言零后续 launch，也断言原活动进程确已退出，不能只用预先返回退出码的 FakeProc。

## R2 P1 执行语义与启动命令未完整复验

位置：`execution_models.py` 的 validate_execution（约 288 行）、`execution.py` 的 status/run/resume/_launch_new、`execution_adapter.py` 的 launch（约 201 行）。

复现：将已发布 PREPARED attempt 的 candidate.lr0 改为 999，或 command 换成无关命令，status 均接受；将仍有 PENDING trial/PREPARED attempt 的执行记录标为 COMPLETED，也被接受，run 可直接按完成返回。launch 对不匹配的 args_sha256 仍调用 launch_training；_launch_new 重建 prepared 时根本没有传入 args_sha256。缺失 args.yaml 还会被 launch 静默重建。

返修：读写和运行入口校验 execution 与 study 的同一身份、trial/request/运行身份、参数映射与固定条件、预算/顺序、phase/result/status 不变量；候选、effective_params、command、run_relpath 必须互相对应。COMPLETED 必须所有预算 trial 终态且对应收尾完整。准备后的命令/args/路径应在 launch 前完整验证；禁止接受无关 executable/CLI 参数，禁止缺失或不匹配 args 时无审计重建。已审计参数不允许通过“执行后发现错误”补救一次不合法启动。纯 shape 校验、字典能 JSON 序列化或只测候选生成都不能代替此边界。

注意 run_id/request_id 必须与对应事实关联；状态字段不能单独授予“完成”身份。补 RUNNING 无身份、缺结果/矛盾结果、跨 trial evidence、伪造终态及真实准备记录被改写的反例。现有 FakeAdapter 只提供 epochs/seed 的 effective_params 等不完整夹具，返修时应修正夹具，不能为夹具放松生产契约。

## R3 P1 冻结根路径与祖先链接边界遗漏

位置：`execution.py` 的 run/resume/status；`metrics.py` 的 read_objective（约 56–83 行），以及适配器读写产物/历史路径。

复现：prepare 已绑定 output/log 根后，重新构造同 storage_root、不同 output_root/log_root 的 HpoRunner，直接 run 可以执行完成，而不是 HPO_EXECUTION_CONFLICT。prepare 比较 roots，run/resume 却不比较。另将 artifact_root 指向真实 Windows 目录符号链接，通过其普通 trial 子目录读取 results.csv，extract_objective 接受链接目标并产出成功证据；只检查 run_dir 和 CSV 最末节点不够。

返修：所有公开运行/恢复入口对照冻结 roots，检查全部受控根和原始父链；metrics、args、run_state、日志、finalizer 的输出路径也采用同一链检查，禁止先 resolve 丢掉链接证据。run_relpath 必须精确绑定 study/trial，不能接受盘符、绝对路径或内部 ..。错误时零启动、零越界读写，不能因最后文件不是链接而放行。补变更根、根链接、祖先链接、锁/args/state/log 链接及跨 trial 目录反例。

## R4 P1 实际参数缺字段仍可登记 SUCCESS

位置：`execution_adapter.py` 的 _params_match（约 272–300 行）。

复现：args.yaml 为 `{}`，results.csv 有合法 mAP50-95=0.9，collect 返回 SUCCESS。原因是 expected_value 为 None 或 actual 缺少字段就 continue；model/data 的非字符串值也可能绕过路径比较。这会把缺失证据当作“参数已匹配”。

返修：六搜索参数与所有固定条件必须存在、类型合法并与计划相符。仅保留规格允许的 device/单元素 imgsz/路径规范化/有限数值容差，不能让缺字段、错误路径类型、bool 数值、NaN/Inf 通过。缺失/不一致返回 FAILED/invalid_params 并记录具体子错误，不产生成功指标；补逐字段删除及污染测试，而不是只测完整正常 args。

## R5 P1 RunState 终态与持久化门槛不一致

位置：`execution.py` 的 _write_terminal_state（约 386–396 行）、退出/停止/中断及收尾调用顺序。

复现一：终态 write_run_state 注入异常，异常被 except/pass 吞掉，run 继续第二次训练并返回 COMPLETED。复现二：进程收到停止后以退出码 0 结束，trial 是 CANCELLED，但 run_state.status 是 completed。实测四次正常训练 run_state.finished_at 全部 null，因为使用 with_status_phase 而非完整终态转换。

返修：以确定的训练停止/超时/中断/退出事实投影 RunState，使用兼容的终态 API 补齐 finished_at/terminal_reason 等字段；正常训练完成与后续指标/分析失败按已有语义区分，不能直接把所有返回码 0 标成功。终态写盘失败稳定返回 HPO_PERSISTENCE_ERROR，阻断预算；恢复要幂等补齐终态再继续。覆盖用户停止返回码 0、超时、启动前取消、恢复发现原进程消失、EXITED 后重放、终态写失败及成功终态时间。

## R6 P2 环境漂移阻断了已提交事实的幂等收尾

位置：`execution.py`，_run_session 在任何 phase 分派前调用 _check_execution_environment（约 151 行）。

复现：trial 已 tell，finalizer 历史写失败留下 TOLD；模拟 torch 版本改变，再 resume，直接拒绝，TOLD 永远无法补齐 finalizer。规格要求先完成已退出/已发布结果的幂等登记与收尾，在新启动前才核验环境漂移。

返修：按 phase 区分“发布已存在的事实”和“开始新的训练”。RESULT_READY/TOLD 必须可重放原 payload，不重训、不重新解释指标；新启动前严格验证全部冻结环境和绑定，漂移阻止启动。补 TOLD/RESULT_READY 漂移恢复测试，并核对 sys.executable、Python、CUDA 等已冻结字段，不仅四个包版本。

## R7 P2 全部 trial 失败时验收 CLI 仍返回成功退出码

位置：`auto_tune/scripts/verify_hpo_execution.py`，main 末尾无条件 return 0（约 91 行）。

复现：fake training 非零退出导致预算内全部 FAILED，脚本打印 no valid objective result 但退出码 0。自动化调用者会把“预算耗尽但没有任何有效结果”当作短训练验收成功。

返修：区分执行流程 COMPLETED 与验收成功；该 verify 脚本只有预算完整、预期真实验收 trial 均成功且有有效排名时返回 0，否则清晰输出数量/失败原因并返回非零。公共 HpoRunner 的 COMPLETED 仍按规格表示预算流程结束，不能为 CLI 改掉该语义。补全失败、部分失败、被暂停、异常和全成功的 CLI 退出码测试。

## 返修交接提示词

```text
请先阅读 docs/h1_2_codex_review_20260908.md、本批规格和六任务计划。
仅返修本记录 R1–R7 及对应测试，保持 H1.1 协议、H1.2 范围和现有依赖不变。
将 log/h1_2_codex_review_20260908/test_independent_review.py 中有效反例迁入正式测试，
补足每条问题列出的矩阵；正式测试不能导入 log 中的探针。
先逐项确认 red，再实现修复，最后运行 HPO 定向、相关回归与全量测试。
活动进程清理必须使用确实保持运行的受控进程/替身测试，不能用已退出的假进程代替。
普通 pytest 不启动真实 YOLO，不调用网络 LLM。
不要修改 UI、app.py、模板、LLM、配置、数据库、requirements 或项目文档；
不要提交推送，不进入 H1.3。保留全部无关工作区修改。
报告实际测试证据、改动清单、偏离与风险，等待 Codex 重新验收。
```

本提示词是可供艾卡安排的返修交接，不表示 Codex 已执行修复。真实短训练正常链路证据保留；返修后是否重跑真实训练由改动范围决定，涉及实际命令/参数/收尾时应重验，不能直接沿用本次结果判定整个批次通过。
