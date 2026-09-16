# H1.2 HPO 训练执行与恢复闭环实施计划

2026-09-09 H1.2 独立验收通过。全量 1837 passed、2 条既有 PCA 警告（97.89 秒），累计 21 项独立反例通过；TPE/Random 各两次真实短训练与命令、args、CSV、历史、finished_at 及冻结 executable 核对通过。详见 docs/h1_2_codex_review_20260908.md 的最终验收结论。

兼容边界：ExecutionAttempt 必须含 command_executable；已含 attempt 但缺字段的旧预验收 execution-v1 记录拒读，不补齐、不覆盖。旧产物保留追溯，后续运行使用新 study ID。此为未发布阶段格式收紧，不承诺同时篡改多个字段的防伪能力。

完成情况：本批六任务及所有验收返修已完成；初版任务中的待办形式仅用于追溯，不代表仍需执行。最终实现包含 command_executable 必填冻结字段、builder 可选 executable、启动前 args 语义双检、审计错误阻断、RunState.finished_at 与环境漂移下幂等收尾；完整修订见同名 H1.2 规格开头。最终基线为 1837 passed，当前没有 H1.2 待返修项。旧编码和返修提示词归档，不作为 H1.3 编码依据。

本批已完成并通过独立验收；以下初版计划保留用于实现追溯，验收修订与格式边界以本文开头及最终验收记录为准，不作为新一轮编码指令。

> **For agentic workers:** 按 superpowers:executing-plans 逐任务执行；项目协作规则优先：Claude Code 只写业务代码和测试，不修改文档、不提交推送、不自动开工作树或启动下一批。

**Goal:** 在已验收 H1.1 上完成可真实训练、可停止恢复、可审计和确定性排名的顺序 HPO 执行闭环。

**Architecture:** HpoService 继续独占 study/trial 事实；HpoRunner 组合独立的 execution.json 审计、严格训练适配器和 CSV 指标读取器。写启动意图后才创建进程，写结果意图后才 tell，恢复以已发布事实收敛。

**Tech Stack:** Python 3.10.18 / Pydantic / 已装 Optuna 4.5.0 / Ultralytics 8.3.253 / 标准库锁、CSV、subprocess；零新增依赖。

**Spec:** `docs/superpowers/specs/2026-09-08-h1-2-hpo-execution-design.md`。状态：艾卡已于 2026-09-08 确认，可按配套提示词交 Claude Code 实施；本计划不是完成报告。

## 全局约束与入口

- 阅读 AGENTS.md、当前交接、H1.1 规格与返修验收记录，再读本批规格和本计划。旧 H1.1 提示词已执行完毕，不再次照其范围开发。
- 只做 H1.2；不改 UI/app.py、模板、LLM 业务、config.yaml、requirements、数据库或 H1.1 搜索协议。
- 保留现有工作区全部无关修改。无需创建项目副本或工作树；不要对整个工作区 git add。
- 所有 Python/pytest 用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。运行前留证 sys.executable、Python 版本和 pip check。下面的 `$py` 必须在每个新 PowerShell 会话重新赋值。
- 预算计算、目标字段、错误码、严格输入、固定参数及故障矩阵逐字遵循规格；不能以新 schema 回避 H1.1 测试。
- 每任务先运行新反例确认 red，再写实现并确认 green。提交步骤由交接报告替代，Claude Code 不提交。

```powershell
$py='D:\Program Files\anaconda3\envs\auto_tune\python.exe'
& $py -c "import sys; print(sys.executable); print(sys.version)"
& $py -m pip check
git status --short
```

## 文件分工与依赖

| 任务 | 新增文件 | 允许修改 |
|---|---|---|
| 1 执行契约与存储 | hpo/execution_models.py、execution_storage.py；test_hpo_execution_storage.py | hpo/__init__.py |
| 2 目标指标与排名 | hpo/metrics.py、ranking.py；test_hpo_metrics.py、test_hpo_ranking.py | hpo/__init__.py |
| 3 严格训练适配 | hpo/execution_adapter.py；test_hpo_execution_adapter.py | hpo/service.py、agent_engine/executor.py、test_executor.py、test_hpo_service.py |
| 4 顺序执行与收尾 | hpo/execution.py；test_hpo_execution.py | hpo/__init__.py |
| 5 停止与恢复 | test_hpo_execution_recovery.py | 上述新增执行文件 |
| 6 验收入口与回归 | scripts/verify_hpo_execution.py；test_hpo_execution_smoke.py | 上述 H1.2 文件的验收修复 |

表中 hpo 前缀为 `auto_tune/modules/hpo/`；测试文件位于 `auto_tune/tests/`；验收脚本完整路径 `auto_tune/scripts/verify_hpo_execution.py`。业务模块不得依赖测试帮助函数。自动化测试使用 tmp_path 和受控假进程，普通 pytest 不启动真实 YOLO。

## 任务 1 执行契约与原子存储

**输入：** H1.1 StrictModel、StudyRecord、TrialRecord、ResultInput；既有锁与路径拒绝语义。

**输出：** ExecutionConfig/ExecutionAttempt/ExecutionRecord，字段与状态不变量见规格第 3–4 节；ExecutionStore(root: Path) 的 `locked(study_id)`、`runner_locked()` 上下文管理器、`read(study_id)->ExecutionRecord`、`write(record)->None`。read/write 仅允许在 locked 内；runner_locked 是根级全生命周期非阻塞锁。存储层只处理已存在合法 study，不创建第二份 study。

- [ ] 新建严格配置反例，包括赋值污染、model_copy、嵌套字段、未知字段和非法路径；示例测试必须先 red：

```python
import pytest
from pydantic import ValidationError
from auto_tune.modules.hpo.execution_models import ExecutionConfig

@pytest.mark.parametrize("kwargs", [
    {"batch": True}, {"batch": -1}, {"imgsz": 65},
    {"device": "0,1"}, {"device": "00"}, {"timeout_seconds": "60"},
])
def test_reject_invalid_execution_config(kwargs):
    with pytest.raises(ValidationError):
        ExecutionConfig(**kwargs)
```

- [ ] 运行 `& $py -m pytest auto_tune/tests/test_hpo_execution_storage.py -q -p no:cacheprovider`，确认因缺功能失败而非解释器错误。
- [ ] 按规格建立三种模型；运行入口先 model_dump(mode="python", warnings=False) 再 model_validate，不信任已实例化对象。
- [ ] 写存储测试：fsync/replace 失败后旧字节不变、损坏原记录拒绝覆盖、5 MiB、重复键、跨 study JSON、非法状态组合、revision 回退/跳跃拒绝、root/祖先/锁/JSON 链接、同进程和跨进程 BUSY。首次发布 revision=0，后续 write 必须 old+1。
- [ ] 实现原子写和锁，按以下顺序发布；所有序列化验证都在替换前，异常清理仅限本次创建的临时文件：

```python
# locked 内的提交顺序；validate_execution 是本任务新增的内部完整校验函数。
checked = validate_execution(record, expected_study_id=record.study_id)
payload = checked.model_dump_json().encode("utf-8")
if len(payload) > 5 * 1024 * 1024:
    raise HpoError("HPO_CORRUPT_EXECUTION", "execution record exceeds limit")
# 原文件存在时先严格 read，校验 revision == old.revision + 1。
# 同目录临时文件 write(payload) -> flush -> os.fsync -> os.replace。
# 完成 replace 后调用方才更新持有的内存记录。
```

- [ ] 重跑本任务及全部 H1.1 存储/契约/返修测试，记录结果和差异。

## 任务 2 真实指标与确定性排名

**输入：** 已退出进程的 run_dir/results.csv、冻结 output_root、运行身份及 epochs；严格 StudyRecord。

**输出：** `extract_objective(run_dir, *, artifact_root, run_id, epochs)->ResultInput`，非法文件抛 HPO_INVALID_METRICS；`rank_trials(record)->list[TrialRecord]` 返回深拷贝的 SUCCESS trial。提取器的排除行数量通过内部 `read_objective` 返回的诊断对象供适配器写审计，不修改 ResultInput schema。

- [ ] 写最佳 epoch 反例：

```python
from auto_tune.modules.hpo.metrics import extract_objective

def test_objective_uses_best_value_and_earliest_tie(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.2\n2,0.8\n3,0.8\n4,0.1\n",
        encoding="utf-8",
    )
    result = extract_objective(run_dir, artifact_root=tmp_path,
                               run_id="tuning:test", epochs=4)
    assert result.value == 0.8
    assert result.evidence.epoch == 2
    assert result.evidence.artifact_relpath == "run/results.csv"
```

- [ ] 运行本任务两份测试确认 red。
- [ ] 实现一次读字节/同字节 hash、表头和 epoch 结构检查、有效值筛选、最大值与最早 epoch；缺文件、全无效、重复表头/epoch、0-based/超范围 epoch、链接/越界、超限文件均稳定拒绝。
- [ ] 排名复用 hpo.validation.validate_record，按以下规则实现；不能排序后再校验：

```python
checked = validate_record(record, expected_study_id=record.study_id)
ordered = sorted(
    (trial for trial in checked.trials if trial.state == "SUCCESS"),
    key=lambda trial: (-trial.result.value, trial.number),
)
return [trial.model_copy(deep=True) for trial in ordered]
```

- [ ] 补 0.0 合法、混合无效行、全失败空排名、同分编号排序、污染模型拒绝及调用后原对象不变测试，重跑本任务与 H1.1 sampler 测试。

## 任务 3 严格训练适配器

**输入：** StudyRecord、TrialRecord、ExecutionConfig；三个冻结根目录。

**输出：** `ExecutionAdapter(output_root: Path, log_root: Path)`：`prepare(record, trial, config)->dict` 返回 candidate_params/effective_params/command/run_relpath/args_sha256；`launch(prepared: dict)->subprocess.Popen`；`collect(record, attempt)->ResultInput`；`finalize(record, attempt)->dict`。函数不自行 ask/tell 或重试；Prepared 数据写入 ExecutionAttempt 后，launch 使用该已发布 attempt 的数据重建同结构字典并复验，不使用未持久化缓存。监控/停止由任务 4–5 的 runner 持有 Popen。

- [ ] 写命令一致性测试：捕获 launch_training 的 command 参数，断言与准备、审计中的列表逐项相等；改变候选、args hash、输出路径或绑定文件后必须零 launch。
- [ ] 给 HpoService 增加公开只读 validate_binding，并写“幂等 PENDING 后模型/快照变化仍拒绝启动”的反例。
- [ ] 给现有 build_yolo_command 增加 task/amp 两个映射，先写精确断言：

```python
from auto_tune.modules.agent_engine.executor import build_yolo_command

def test_explicit_detect_and_amp_command(monkeypatch, tmp_path):
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
                        lambda: "yolo-test")
    cmd = build_yolo_command("trial", str(tmp_path / "trial" / "args.yaml"),
                             {"task": "detect", "amp": False})
    assert "task=detect" in cmd
    assert "amp=False" in cmd
```

- [ ] 用规格固定参数构造 effective_params；验证候选，调用 validate_and_clamp；若护栏改变参数则拒绝，不采纳结果。预检在 launch 前，显式检查单设备可用性与本地输入。禁止读取真实 config.yaml、调用模型 API 或自动下载缺少的资源。
- [ ] 用专属目录、write_training_config 深拷贝、build_yolo_command 和 launch_training(command=...) 组合适配；保存启动前 args hash，结束后保存实际 args 与 hash。输出和模型/数据路径全链拒绝链接。
- [ ] collect 验证实际六搜索参数/固定项，再调用 metrics；训练错误使用既有 ResultInput reason_code，细节放 attempt.error_code。finalize 传 source="tuning"、session_id=study_id、运行身份与 strategy="hpo"，不依赖 finalizer 的 final mAP 做排名。
- [ ] 测护栏 clamp、缺资源、GPU 不存在、参数漂移、错误目录、finalizer 的 analysis/history/index 三种独立失败；重跑本任务、test_executor.py、test_guardrails.py、test_training_finalizer.py、test_hpo_service.py。

## 任务 4 顺序执行和幂等收尾

**输入：** 前三任务与 HpoService；ExecutionAdapter；规格规定的同步接口。

**输出：** HpoRunner.prepare/run/status，逐 trial 完成 PREPARED→LAUNCH_INTENT→RUNNING→EXITED→RESULT_READY→TOLD→FINALIZED，全预算完成；执行故障时返回稳定 HpoError。可通过 monkeypatch 模块中的 ExecutionAdapter、time.monotonic 和进程方法注入测试，不为生产接口暴露“跳过持久化”开关。

- [ ] 在测试文件定义 tmp_path HpoService study fixture，复用已有测试创建合成快照的方式；fake adapter 的 launch 返回可 poll/wait/terminate/kill 的对象，记录启动次数与命令，不启动 YOLO。
- [ ] 写 budget=2、首个失败次个成功和全失败用例，要求 ask 数、launch 数、终态数均不超预算。固定请求身份必须用：

```python
from uuid import NAMESPACE_URL, uuid5

def execution_request_id(study_id, number):
    return uuid5(NAMESPACE_URL, f"hpo-execution-v1:{study_id}:{number}").hex
```

- [ ] 测 run 第二次返回相同完成事实且零新增 launch；run(PAUSED) 返回 HPO_EXECUTION_CONFLICT；prepare 不同配置、已有外部终态拒绝；status 不触发任何执行。
- [ ] 按规格第 5 节实现“写意图→执行→写结果→tell→finalize”，所有 copy 在成功落盘后才替换内存状态。run_state 写盘异常不能吞；每条训练使用固定 tuning run_id 并贯穿 Evidence 与 finalizer。
- [ ] 注入 ask 后、PREPARED、LAUNCH_INTENT、PID 写入、EXITED、RESULT_READY、tell、TOLD、finalizer/FINALIZED 写盘失败，断言零后续 launch。活进程发生写盘失败必须走停止确认路径。
- [ ] 测 finalizer.history_error 阻断预算，analysis_error/index_error 仅记录且不改合法结果；没有 launch 的 invalid_params 不调用 finalizer。重跑任务 1–4 测试。

## 任务 5 停止、超时与恢复

**输入：** 已发布 ExecutionRecord、StudyRecord、process_identity 工具和 stop_event。

**输出：** HpoRunner.resume，run 的停止/超时路径，规格第 6 节全矩阵。BLOCKED 不允许强制恢复；异常窗口不能猜测成功。

- [ ] 写参数化恢复决策测试，逐项覆盖规格表。使用 monkeypatch 的 IdentityMatch 四种结果、固定发布记录及 launch 计数；以下是必须满足的判定骨架：

```python
if attempt.phase == "LAUNCH_INTENT" and attempt.process_create_token is None:
    raise HpoError("HPO_RECOVERY_REQUIRED", "launch outcome is not provable")
if attempt.phase == "RUNNING":
    match = compare_process_identity(ProcessIdentity(attempt.pid,
                                                     attempt.process_create_token))
    if match is IdentityMatch.MATCH:
        raise HpoError("HPO_PROCESS_STILL_ACTIVE", "original process remains alive")
    if match is IdentityMatch.UNVERIFIABLE:
        raise HpoError("HPO_RECOVERY_REQUIRED", "process identity is unverifiable")
    # MISSING/MISMATCH：写 INTERRUPTED 结果并 tell；绝不 terminate 复用 PID。
```

- [ ] 运行 recovery 测试确认缺功能失败，记录 red。
- [ ] 实现同进程 stop_event 协作停止、单调超时和 terminate→wait→kill→wait；0.2 秒轮询不阻塞日志管道。已观察到退出优先收尾；真正实施停止后不能把退出码 0 改写为 SUCCESS。
- [ ] 测启动前停止零 launch、launch 竞态停止、超时、拒绝终止不报已结束、用户停止不再 ask、正常退出与 stop 同时到达的确定性顺序。
- [ ] 恢复 PREPARED/无执行记录尾部 PENDING 时校验原绑定和 request_id；恢复 RESULT_READY/TOLD 时先完成幂等提交，再在新启动前检查环境漂移，不能因为环境变化丢弃已经有事实的结果。
- [ ] 用可控本地短进程补跨进程根锁竞争及真实 token 核验（不启动训练、不 kill 任意 PID），并补同 request/result 重放、conflict 不覆盖、FAILED/CANCELLED/INTERRUPTED 不复活。
- [ ] 重跑所有 HPO 测试及 test_run_state_process.py；阻塞窗口的测试不得 skip，也不得将 BLOCKED 改为“成功恢复”。

## 任务 6 可复现验收入口及完整回归

**输出：** Python 命令行验收脚本，调用公开 HpoService/HpoRunner，不建立第二套执行逻辑。参数必须有 `--snapshot-dir`、`--model-path`、`--storage-root`、`--output-root`、`--log-root`、`--sampler {tpe,random}`、`--budget`、`--epochs`、`--batch`、`--imgsz`、`--device`；无参数不启动训练，缺本地路径失败，不下载。脚本只在显式执行时启动真实训练。

- [ ] 写脚本 --help 和非法参数零 launch 测试；普通 test_hpo_execution_smoke 使用 fake adapter 验证 CLI 到公共服务的参数映射。
- [ ] 实现 CLI：解析参数→create_study→prepare→run→rank_trials→输出 study_id、终态数量、最佳 trial/无有效结果及本地审计位置；异常非零退出。所有失败不伪装成功，报告不含凭据。
- [ ] 执行全量自动化测试与依赖检查，保存本次实际结果，禁止沿用旧 1672 数字：

```powershell
$py='D:\Program Files\anaconda3\envs\auto_tune\python.exe'
& $py -m pytest auto_tune/tests/test_hpo_execution_storage.py auto_tune/tests/test_hpo_metrics.py auto_tune/tests/test_hpo_ranking.py auto_tune/tests/test_hpo_execution_adapter.py auto_tune/tests/test_hpo_execution.py auto_tune/tests/test_hpo_execution_recovery.py auto_tune/tests/test_hpo_execution_smoke.py -q -p no:cacheprovider
& $py -m pytest auto_tune/tests -q -p no:cacheprovider
& $py -m pip check
git diff --check -- auto_tune/modules/hpo auto_tune/modules/agent_engine/executor.py auto_tune/tests auto_tune/scripts/verify_hpo_execution.py
```

- [ ] Claude Code 提交文字报告：新增/修改文件、解释器与依赖实查、red/green、每个测试命令和结果、偏离计划、未解决风险；不要提交 Git，不修改 docs，不进入 H1.3。
- [ ] Codex 独立审查并复跑自动化，然后使用已有合法本地输入运行脚本：两采样器各 budget=2/epochs=1/batch=1/imgsz=64，共最多 4 次成功短训练。受控产物放忽略目录，不复制正式数据集，不在文档记录私人输入路径。
- [ ] Codex 验证实际命令/args/CSV SHA256/最佳 epoch/排名/tell/历史一致，额外以可控假进程验证停止与恢复。真实输入缺失则报告未完成验收，不能拿合成测试冒充。

## 自审对应与交接

规格 3–4 对应任务 1；规格 7 对应任务 2；规格 2/3/5 的训练边界对应任务 3；提交顺序对应任务 4；恢复与故障对应任务 5；验收对应任务 6。无需艾卡在 H1.2 手工测试。

本批估算 3–5 人日：契约存储约 0.5–1、适配与指标约 0.5–1、顺序执行与恢复约 1–1.5、回归和独立验收约 1–1.5。依赖现有本地合法输入和可用 Conda 环境；不叠加计算原 H1 总估算。H1.3 页面改动与实际体验验收另行实施。
