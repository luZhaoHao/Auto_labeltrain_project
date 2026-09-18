# H1.1 Detect HPO 搜索契约与持久化基础规格

2026-09-08 当前结论：H1.1 返修后验收通过，尚未提交或推送；H1.2 尚未启动。Claude Code 完成初版，Codex 按艾卡安排修复 R1–R4：恢复语义校验、study 身份绑定、路径父链与链接检查、可变模型入口复验。

验证证据：auto_tune Conda 环境，Python 3.10.18；全量 1672 passed、2 条既有 PCA 警告（59.97 秒），新增返修回归 35 passed，原独立复现 9 passed；pip check 无依赖冲突。解释器为 D:\Program Files\anaconda3\envs\auto_tune\python.exe。

批次边界：H1.1 仅基础模块，无真实训练、训练子进程、网络 LLM 或 UI/API 接入。修改 UI 前的自动化验证由 Codex 完成，艾卡暂不介入手工测试；H1.2 的必要短训练由 Codex 验证，H1.3 再安排界面体验。下一步先确认 H1.2 规格与实施计划，不复用 H1.1 提示词启动新批次。

完整验收记录：docs/h1_1_codex_review_20260907.md。以下早期实施指令和初验失败证据保留用于追溯；以本次状态为准。原 H1 6–10 人日为包含已完成 H1.1 的总估算，剩余工期尚未重新估算。

日期：2026-09-07。负责人：Codex；实现：Claude Code；批准与验收方向：艾卡。

本文按艾卡本次“补齐并确认 H1.1 详细规格、实施计划和依赖版本”的要求形成可执行交接。H1 三批顺序已确认；本文是 H1.1 唯一编码规格，不将路线摘要当作接口规格。H1.1 已实现并返修验收通过，本文继续作为后续批次依赖的基础契约。

## 1 目标与边界

交付可在 Python 内调用的 HPO 基础模块：创建 study、生成一个候选、登记外部结果、重载历史并继续采样。使用真实 Optuna TPE/RandomSampler，通过合成目标函数测试，无真实训练、UI、HTTP 路由、后台线程、LLM、训练启动或进程管理。

H1.2 负责执行器、真实产物提取、停止和运行恢复、确定性排名；H1.3 负责 Studio、历史展示及端到端验收。本批保存合法的成功值只为训练 TPE，不输出最佳 trial，不调用 Optuna best_trial 作为产品排名。

## 2 已核对的事实与技术选择

- 参数注册表：`auto_tune/modules/agent_engine/parameter_registry.py`。
- 实际护栏：`auto_tune/modules/agent_engine/guardrails.py`。其中 lrf 上限 0.1 比注册表 1.0 更严格，warmup_epochs 被列为整数。HPO 取兼容子集，不修改现有护栏。
- `validate_dataset_snapshot(snapshot_dir: Path, expected_root: Path | None = None, require_data_dirs: bool = True) -> DatasetSnapshot` 位于 `modules/dataset_snapshot/service.py`；使用默认严格验证，不自行伪造快照。
- `modules/run_state/service.py` 已有同目录临时文件、flush、fsync、os.replace 的写盘模式。复用模式，不拿 RunState schema 存 HPO 数据；本批不增加 run_kind。
- Ultralytics 8.2.0 官方源码及本机 8.3.253 均将 AdamW 的 momentum 映射为 beta1；auto 会覆盖 lr0/momentum。首版禁止 auto。

选择 JSON 原子保存业务事实，按历史重建 Optuna 内存 study。备选 Optuna RDBStorage 会产生与业务 JSON 的双写恢复问题；仅保存 pickle sampler 会引入不安全反序列化和版本耦合。H1 最多 100 trials，重建成本可控，无需第二套数据库。

## 3 依赖版本与验证限度

技术选型固定 `optuna==4.5.0`，MIT，支持 Python 3.10；采用公开 create_study/add_trial/ask/suggest API，不引入 integration、dashboard、Ray 或额外数据库服务。选用已发布、公开 API 满足本批的固定版本，不声明它是最新版。

2026-09-07 在 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`（Python 3.10.18）执行 pip dry-run 成功。锁定本次依赖解析结果：

| 包 | 版本 | 作用 |
|---|---|---|
| optuna | 4.5.0 | TPE 与随机采样 |
| alembic | 1.19.2 | Optuna 声明的数据库迁移依赖，本批不调用 |
| SQLAlchemy | 2.0.52 | Optuna 声明的存储依赖，本批不用 RDBStorage |
| colorlog | 6.12.0 | 日志格式 |
| greenlet | 3.5.5 | SQLAlchemy 间接依赖 |

其余需求由环境现有包满足；试算未要求升级/降级已有包。现有 numpy 2.2.6、torch 2.5.1+cu121、ultralytics 8.3.253、pydantic 2.13.4，与仓库部分旧锁定不同。这是既有环境事实，不能运行整个 requirements 的重新安装来“修正”。本批不变更这些版本，不宣称已通过运行时兼容测试。

实施时在 `auto_tune/requirements.txt` 仅追加上述五行版本约束。安装用五包精确版本命令，不用 `pip install -r`、`--upgrade` 或 `--no-deps`。安装前再次 dry-run；若新增包或替换现有包超出本表，报告变化等待艾卡确认。上述五包现已安装并独立核验通过。艾卡转发配套执行提示词即明确授权本表五包；不扩大到其他依赖。

## 4 输入契约

使用现有 Pydantic v2，所有模型 `extra='forbid'`，拒绝字符串转数字、bool 充当整数/实数、NaN/Infinity；标识、摘要、版本、状态使用枚举/正则校验。数值允许原生 int/float 转为 float，但必须先排除 bool，不接受 NumPy 标量隐式转换。JSON 加载必须拒绝重复键及非有限常量。

`StudyConfig` 字段：schema_version 固定 `hpo-study-v1`；task 固定 `detect`；model_family 固定 `yolov8`；sampler 为 `tpe` 或 `random`、默认 `tpe`；sampler_protocol 固定 `rebuild-per-trial-v1`；search_space_version 固定 `detect-hpo-v1`；objective 固定 `val_map50_95_best_epoch_v1`；direction 固定 `maximize`；budget 为严格整数 1..100、默认 10；seed 为严格整数 0..2147483647、默认 42；epochs 为严格整数 1..1000、默认 30。上述固定字段提供默认值但输入其他值必须拒绝。

创建方法额外传入 `snapshot_dir: Path` 和 `model_path: Path`。model_path 必须为现存普通本地 `.pt` 文件，拒绝符号链接/reparse point、目录和 URL，不下载、不反序列化权重。记录绝对路径、字节数及 SHA256；文件名前缀不作为模型类型证明，model_family 是调用方声明，本批不承诺权重内部类型验证，H1.2 预检负责实际 Detect 类型验证。快照通过现有服务验证并保存 snapshot_id、manifest_digest、snapshot_path、data_yaml_path。

`StudyRecord` 除 config 外包含：study_id（服务生成 `hpo_`+UUID hex，正则 `^hpo_[0-9a-f]{32}$`）、created_at/updated_at（UTC ISO8601）、revision（创建 0，每次成功变更加 1）、snapshot_binding、model_binding、environment（Python/Optuna/NumPy/Ultralytics 版本）、trials。绑定只读，不接受修改 config 的恢复接口。读取返回深拷贝，调用方不能修改服务内部状态。

## 5 搜索空间和映射

用户本批不能自定义范围。按下面顺序调用 suggest，顺序也是版本化协议的一部分。

| 次序 | Optuna 名称 | 分布 | 映射到训练候选 |
|---|---|---|---|
| 1 | optimizer | categorical `["SGD", "AdamW"]` | optimizer |
| 2 | lr0 | float 0.00001..0.005，log=True | lr0 |
| 3 | lrf | float 0.01..0.1，log=True | lrf |
| 4 SGD 分支 | momentum_sgd | float 0.8..0.98，log=False | momentum |
| 4 AdamW 分支 | beta1_adamw | float 0.85..0.95，log=False | momentum |
| 5 | weight_decay | float 0.0..0.001，log=False | weight_decay |
| 6 | warmup_epochs | int 0..min(5, epochs-1)，step=1 | warmup_epochs |

两个条件化名称避免把 SGD momentum 与 Adam beta1 当成同一个统计参数。`sampled_params` 保存 Optuna 原名及对应 distribution JSON；`candidate_params` 始终恰好为六个训练键，额外锁定训练 epochs 在 config 中。本批 candidate 不是可直接启动训练的完整配置；不得导出 shell 命令。映射后按上表和现有注册表/护栏交集严格校验，禁止静默 clamp 或重采样掩盖非法候选。H1.2 合并配置时必须以这六项覆盖基础配置并再次预检。

## 6 采样与可恢复性

只允许每个 study 一个 outstanding trial。采用 `sampler_protocol='rebuild-per-trial-v1'`：每次分配编号 n 时，重新创建内存 study，将历史 0..n-1 按编号加入，再创建 sampler，seed 为 `(config.seed+n) % 2147483648`。

TPE 参数固定 `n_startup_trials=5, n_ei_candidates=24, multivariate=False, constant_liar=False`，其他保持 4.5.0 默认；pruner 使用 NopPruner。RandomSampler 使用同样的派生 seed。每次创建一个新内存 study，不缓存 sampler RNG、不保存 pickle。历史 SUCCESS 以 COMPLETE/value 加入；FAILED/CANCELLED/INTERRUPTED 以 FAIL/value=None 加入；保留所有 trial 顺序和分布。用 `create_trial`/`add_trial`，不读写 Optuna 私有属性。

少于 5 个成功 trial 时 TPE 仍在启动阶段；测试至少登记 5 个成功值后再检验采样，不能用只有两个 trial 的用例宣称覆盖了 TPE 阶段。

恢复语义：返回已保存的 pending 候选，不分配新编号，不消耗第二份预算；相同依赖版本、配置、相同顺序历史和同样结果下，下一个候选相同。不保证与 Optuna 默认连续 RNG 序列相同，不保证不同版本一致，也不承诺 GPU 训练数值逐位一致。load 时 Optuna/NumPy 版本不匹配则可只读，拒绝新的候选分配；调用方变更快照/模型内容后拒绝继续采样。

## 7 trial 状态和预算

`TrialRecord`：number 从 0 连续；trial_id=`{study_id}_t{number:04d}`；request_id 为调用方 UUID hex；state；sampled_params；distributions；candidate_params；created_at；finished_at；result（可空）。

H1.1 状态仅 `PENDING → SUCCESS|FAILED|CANCELLED|INTERRUPTED`。PENDING 表示候选已持久化，**不代表进程在运行**；不提供 RUNNING、不推断进程死亡、不自动将 pending 改为 interrupted。H1.2 再定义运行身份绑定和进程核对。

预算按成功持久化的候选条数计数，所有终态均消耗预算；无自动补偿、无自动重试。FAILED 后如继续 ask，需要新 request_id 且消耗下一名额。配置非法或写盘失败未发布的候选不计数。无基线 trial、无额外验证 trial 隐性占位。预算用尽抛 `HPO_BUDGET_EXHAUSTED`。

同 request_id 重复 ask 返回同一记录（即使已终态）；不同 request_id 在已有 PENDING 时抛 `HPO_PENDING_TRIAL`。同终态、同结果重复 tell 为幂等且 revision 不变；冲突 tell 抛 `HPO_RESULT_CONFLICT`，不能覆写已有结果。

ResultInput 的 state 必填，value/evidence/reason_code 默认 None。成功 ResultInput：state SUCCESS；value 为有限 0..1 数；evidence 含 run_id（非空受限字符串）、artifact_relpath（POSIX 相对路径，不含反斜线、绝对路径、盘符、空段、`.`/`..`）、artifact_sha256（64 位小写 hex）、metric_key 固定 `metrics/mAP50-95(B)`、epoch 为 1..config.epochs 的整数；reason_code 必须空。SUCCESS 缺失证据或超范围值拒绝，trial 保持 PENDING，不能写为 0。run_id/artifact_relpath 等可变文本最长 256 字符；study.json 超过 5 MiB 返回 CORRUPT_STUDY，不无限读入内存。

失败结果 value/evidence 必须空，reason_code 从 `training_failed/timeout/oom/invalid_params/user_stopped/process_interrupted` 选取；FAILED 对应前四项，CANCELLED 仅 user_stopped，INTERRUPTED 仅 process_interrupted。本批校验证据契约，不读取真实训练产物；测试证据均明确为合成值，不伪称真实训练。

目标指标的 H1.2 解释固定为验证集 results.csv 中 mAP50-95(B) 有效有限值的最大值；同分取最早 epoch，不等同于 Ultralytics best.pt 的 fitness 选择或训练 final 指标。最终 trial 同分排名在 H1.2 按 trial.number 升序定义，不使用 LLM。

## 8 存储与并发边界

构造 `HpoService(storage_root: Path)`，调用方提供受控本地目录；未来默认 `runs/hpo`，本批只在测试 tmp_path 使用。文件仅 `storage_root/<study_id>/study.json` 和锁文件/临时文件。服务生成目录名，任何输入不能决定任意写入路径。父链及已有目标拒绝 reparse/symlink，拒绝路径逃逸；NAS、多进程服务、网络文件系统不在本批支持范围。

每次读改写持有同 study 的进程内线程锁及 OS 文件锁，使用标准库 Windows msvcrt/Linux fcntl 非阻塞排他锁，忙返回 HPO_STUDY_BUSY。OS 退出释放锁；锁文件可保留但文件存在本身不代表已锁，不使用手工过期时间强占锁。所有参与访问的 HPO 服务遵守此锁；不承诺抵御外部程序直接改写文件。

写盘顺序：在锁内重读并严格校验 → 构造新对象 → JSON 序列化 allow_nan=False → 同目录唯一临时文件 → flush/fsync → os.replace → 才返回新候选/终态。写失败丢弃内存新状态，下次重读磁盘；不得返回未保存候选，不得后续 flush 复活失败变更。不覆盖损坏文件，不创建空 study 假恢复；无自动迁移未知 schema。读取和结果校验失败不修改 revision。

创建时原子创建 UUID 目录；配置与输入验证失败不得留下已发布 study。仅允许清理由本次调用创建且未发布的临时文件；不得删除用户文件/成功 study。注入 fsync/replace 失败测试必须证实旧 JSON 可读且候选数未变。不能声称该机制保证磁盘硬件断电不丢最后一次事务。

## 9 公共接口与错误码

```python
class HpoService:
    def __init__(self, storage_root: Path): ...
    def create_study(self, config: StudyConfig, *, snapshot_dir: Path,
                     model_path: Path) -> StudyRecord: ...
    def load_study(self, study_id: str) -> StudyRecord: ...
    def ask(self, study_id: str, *, request_id: str) -> TrialRecord: ...
    def tell(self, study_id: str, trial_number: int,
             result: ResultInput) -> TrialRecord: ...
```

这些省略号仅表示规格中的接口声明，不是交付实现。服务无 LLM/config.yaml 依赖。错误 `HpoError(code, message)` 使用稳定 code：HPO_INVALID_CONFIG、HPO_BINDING_MISMATCH、HPO_NOT_FOUND、HPO_CORRUPT_STUDY、HPO_VERSION_MISMATCH、HPO_STUDY_BUSY、HPO_PENDING_TRIAL、HPO_BUDGET_EXHAUSTED、HPO_INVALID_RESULT、HPO_RESULT_CONFLICT、HPO_PERSISTENCE_ERROR。未知字段及 request_id 非法归 INVALID_CONFIG，结果字段归 INVALID_RESULT。错误信息不得包含凭据。

## 10 验收与交付

全部契约/边界、双采样器、TPE 启动后路径、JSON 重载、幂等、预算、结果冲突、文件损坏、版本/绑定变化、写盘故障和锁竞争可测；明确零训练命令、零训练子进程、零网络 LLM 调用。H1.1 不做真实训练是分批边界，不以跳过测试代替实现。完整现有套件回归，并记录实际解释器、版本和测试数字；既有历史 1484 passed 不是本次结果。

旧 S1/S2/Q1/P2 批次规格可能包含仍需追溯的已验收约束，不自动视为 H1 规格或删除。本批只读参考代码事实；新规格没有替代它们全部技术细节，归档前仍须核对替代关系与引用，不能为“清空 docs”批量归档。

## 11 官方依据

- [Optuna 4.5.0 包元数据](https://pypi.org/pypi/optuna/4.5.0/json)
- [TPE 4.5.0 公共参数](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [create_trial 与历史重建](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.trial.create_trial.html)
- [Ultralytics 8.2.0 optimizer 映射](https://github.com/ultralytics/ultralytics/blob/v8.2.0/ultralytics/engine/trainer.py)
