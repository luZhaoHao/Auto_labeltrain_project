# H1.1 Detect HPO 基础实施计划

2026-09-08 当前结论：H1.1 返修后验收通过，尚未提交或推送；H1.2 尚未启动。Claude Code 完成初版，Codex 按艾卡安排修复 R1–R4：恢复语义校验、study 身份绑定、路径父链与链接检查、可变模型入口复验。

验证证据：auto_tune Conda 环境，Python 3.10.18；全量 1672 passed、2 条既有 PCA 警告（59.97 秒），新增返修回归 35 passed，原独立复现 9 passed；pip check 无依赖冲突。解释器为 D:\Program Files\anaconda3\envs\auto_tune\python.exe。

批次边界：H1.1 仅基础模块，无真实训练、训练子进程、网络 LLM 或 UI/API 接入。修改 UI 前的自动化验证由 Codex 完成，艾卡暂不介入手工测试；H1.2 的必要短训练由 Codex 验证，H1.3 再安排界面体验。下一步先确认 H1.2 规格与实施计划，不复用 H1.1 提示词启动新批次。

完整验收记录：docs/h1_1_codex_review_20260907.md。以下早期实施指令和初验失败证据保留用于追溯；以本次状态为准。原 H1 6–10 人日为包含已完成 H1.1 的总估算，剩余工期尚未重新估算。

> 执行者：Claude Code。按任务顺序实现，艾卡本次委托 Codex 完成设计和计划；不再让 Claude 自行另写设计。项目协作规则优先于通用技能的自动提交/并行代理建议：不提交、不推送、不跨入 H1.2。

**Goal:** 交付可持久化和重载的 Detect HPO 候选生成与结果登记模块。

**Architecture:** JSON 是 HPO 唯一持久化事实；每次候选按历史重建 Optuna 内存 study。输入契约、条件空间、存储和采样分离，服务组合它们，不依赖 UI/LLM/训练执行。

**Tech Stack:** Python 3.10 auto_tune Conda，现有 Pydantic v2，Optuna 4.5.0，标准库文件锁和 JSON。

**Spec:** `docs/superpowers/specs/2026-09-07-h1-1-hpo-foundation-design.md`，必须全文读取。

## 全局约束

- 只实现 H1.1；无训练启动、真实训练、UI/API、LLM、产品排名和并行搜索。
- 解释器固定 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。
- 不重装整个 requirements，不改变已有 Torch/Ultralytics/NumPy/Pydantic。
- 不修改现有护栏/运行状态/快照协议，已有工作区修改原样保留。
- 临时测试输入使用 pytest tmp_path；不使用真实数据、凭据或大模型权重。
- 仅文末列出的依赖及新增模块/测试属于修改范围。若确实需要修改其他业务文件，先报告接口证据和原因。
- 全部任务完成只提交交付报告给 Codex；每步不执行 git commit。

## 任务 0 基线与依赖

- [ ] 读取 AGENTS.md、当前交接及本规格/计划；执行 `git status --short`、`git diff --name-only` 记录既有变更。不要 checkout/reset/stash/清理。
- [ ] 检查解释器及环境（下面命令不读取业务配置）。

```powershell
$py = 'D:\Program Files\anaconda3\envs\auto_tune\python.exe'
& $py -c "import sys; print(sys.executable); print(sys.version)"
& $py -m pip check
& $py -m pip install --dry-run optuna==4.5.0 alembic==1.19.2 SQLAlchemy==2.0.52 colorlog==6.12.0 greenlet==3.5.5
```

- [ ] 艾卡转发配套提示词已授权这五包。若试算仍只新增这五包，执行下列命令，并在 `auto_tune/requirements.txt` 仅追加同样五个版本。若变更不同则暂停安装报告，不自行替换版本。

```powershell
& $py -m pip install optuna==4.5.0 alembic==1.19.2 SQLAlchemy==2.0.52 colorlog==6.12.0 greenlet==3.5.5
& $py -c "import sys,optuna; print(sys.executable); print(optuna.__version__)"
& $py -m pip check
```

安装前后 pip check 如有既有问题，逐条比较，不擅自升级全环境。新增冲突须解决后才能交付；既有问题记录限制。

## 任务 1 严格契约与条件搜索空间

**新增文件**：`auto_tune/modules/hpo/__init__.py`、`models.py`、`search_space.py`。

**测试**：`auto_tune/tests/test_hpo_contracts.py`、`test_hpo_search_space.py`。

**接口**：models 暴露 StudyConfig、StudyRecord、TrialRecord、ResultInput、HpoError；search_space 暴露 `suggest_candidate(trial, *, epochs: int) -> tuple[dict, dict]`，返回 `(sampled_params, candidate_params)`，另暴露 `validate_candidate(params: dict, *, epochs: int) -> dict`。

- [ ] 写失败测试，再运行单文件确认失败原因是缺失本批实现，不是环境配置错误。

```python
import pytest
from pydantic import ValidationError
from auto_tune.modules.hpo.models import StudyConfig, HpoError
from auto_tune.modules.hpo.search_space import validate_candidate

@pytest.mark.parametrize('patch', [
    {'budget': True}, {'budget': '10'}, {'budget': 0}, {'budget': 101},
    {'seed': -1}, {'epochs': 0}, {'task': 'classify'},
    {'sampler': 'other'}, {'search_space_version': 'unknown'}, {'extra': 1},
])
def test_reject_invalid_config(patch):
    with pytest.raises(ValidationError):
        StudyConfig(**patch)

def test_candidate_is_not_silently_clamped():
    p = dict(optimizer='SGD', lr0=0.001, lrf=0.5, momentum=0.9,
             weight_decay=0.0005, warmup_epochs=0)
    with pytest.raises(HpoError) as error:
        validate_candidate(p, epochs=1)
    assert error.value.code == 'HPO_INVALID_CONFIG'
    assert p['lrf'] == 0.5
```

- [ ] 实现规格第 4–5 节。使用 float 的 before validator 排除 bool/str；ResultInput 的跨字段约束由模型和 service 分层完成。默认值也必须验证。
- [ ] 采样代码按下列顺序实现，并保存 Optuna 分布，不额外采样另一 optimizer 分支。

```python
optimizer = trial.suggest_categorical('optimizer', ['SGD', 'AdamW'])
lr0 = trial.suggest_float('lr0', 1e-5, 0.005, log=True)
lrf = trial.suggest_float('lrf', 0.01, 0.1, log=True)
momentum_key = 'momentum_sgd' if optimizer == 'SGD' else 'beta1_adamw'
lo, hi = (0.8, 0.98) if optimizer == 'SGD' else (0.85, 0.95)
momentum = trial.suggest_float(momentum_key, lo, hi)
wd = trial.suggest_float('weight_decay', 0.0, 0.001)
warmup = trial.suggest_int('warmup_epochs', 0, min(5, epochs - 1))
```

- [ ] 用 Optuna FixedTrial 分别覆盖两种 optimizer；断言恰好六个候选键、条件原名映射、epochs=1 时 warmup=0；参数化 nan/inf/bool/未知键/越界/小数 warmup 全部拒绝。与现有 `validate_and_clamp` 对比候选不得发生改写或 clamp。
- [ ] 两个测试文件逐个通过后进入任务 2。

## 任务 2 原子 JSON 存储与锁

**新增文件**：`auto_tune/modules/hpo/storage.py`。

**测试**：`auto_tune/tests/test_hpo_storage.py`。

**接口**：`StudyStore(root: Path)`；`locked(study_id)` 上下文管理器；锁内 `read(study_id) -> StudyRecord`、`write(record: StudyRecord) -> None`。create 的目录生成由 service 负责，store 对 ID/路径仍校验。公开函数不允许无锁读改写；由 store 或 service 统一保证锁时序。

- [ ] 为损坏 JSON、重复 JSON 键、未知 schema、路径穿越、锁竞争及写盘故障写失败测试。测试记录 fixture 按规格构造合法对象，不用绕过 validation 的 model_construct。
- [ ] 采用现有 run_state 的原子写模式；OS 文件锁按平台使用 msvcrt/fcntl，进程内使用同路径 RLock；不要把“有锁文件”当成锁。

```python
# 写盘关键顺序，外部包裹同 study 排他锁。
payload = json.dumps(record.model_dump(mode='json'), ensure_ascii=False,
                     allow_nan=False, sort_keys=True).encode('utf-8')
with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
    temp_path = Path(out.name)
    out.write(payload)
    out.flush()
    os.fsync(out.fileno())
os.replace(temp_path, target)
```

实际实现必须有 finally 清理本次未发布临时文件及 OSError → HPO_PERSISTENCE_ERROR 映射；不得丢失原异常 cause。读取限制 JSON 大小为 5 MiB，超限返回 CORRUPT_STUDY（100 trials 的上限不需无限大文档）；同时限制 reason/run_id 等文本长度不超过 256 字符。数值分布及 trial ID 必须与 config/number 一致。

- [ ] 注入 os.fsync 与 os.replace 失败，确认旧文件字节不变、revision 不变；在模拟写入后续调用中不复活失败数据。
- [ ] 用两个服务实例以及两个测试子进程争用同一 study 锁：第二个立即得到 BUSY，持锁进程退出后可取得锁。测试子进程只用于锁验证，不能启动训练。
- [ ] 验证本批创建的空/未发布目录不会被当作已存在 study；未知 ID 为 NOT_FOUND。

## 任务 3 Optuna 历史重建与候选生成

**新增文件**：`auto_tune/modules/hpo/sampler.py`。

**测试**：`auto_tune/tests/test_hpo_sampler.py`。

**接口**：`sample_next(config: StudyConfig, history: list[TrialRecord]) -> tuple[dict, dict, dict]`，返回 sampled、candidate、distribution JSON。只接受无 PENDING 的连续终态历史；不写文件，不改传入对象。

- [ ] 先写两个 sampler 的可重现与条件分布测试，再实现下列协议。

```python
seed = (config.seed + len(history)) % 2147483648
sampler = (optuna.samplers.TPESampler(seed=seed, n_startup_trials=5,
            n_ei_candidates=24, multivariate=False, constant_liar=False)
           if config.sampler == 'tpe' else optuna.samplers.RandomSampler(seed=seed))
study = optuna.create_study(direction='maximize', sampler=sampler,
                           pruner=optuna.pruners.NopPruner())
for saved in history:
    distributions = {k: optuna.distributions.json_to_distribution(v)
                     for k, v in saved.distributions.items()}
    success = saved.state == 'SUCCESS'
    study.add_trial(optuna.trial.create_trial(
        params=saved.sampled_params, distributions=distributions,
        state=optuna.trial.TrialState.COMPLETE if success else optuna.trial.TrialState.FAIL,
        value=saved.result.value if success else None))
trial = study.ask()
sampled, candidate = suggest_candidate(trial, epochs=config.epochs)
distributions = {k: optuna.distributions.distribution_to_json(v)
                 for k, v in trial.distributions.items()}
```

- [ ] 固定配置和六个成功历史（混合 SGD/AdamW）重复调用应得到相同候选；加入失败历史后按总编号派生 seed，失败 value 不进入 TPE。不同进程重载输入也一致。
- [ ] 验证 TPE 在至少五个 SUCCESS 后运行仍合法；不能只 mock sampler 或只跑启动随机阶段。
- [ ] malformed distribution/历史错序/重复编号/PENDING 被拒绝，不修补或跳过历史。

## 任务 4 服务组合与恢复幂等

**新增文件**：`auto_tune/modules/hpo/service.py`。

**测试**：`auto_tune/tests/test_hpo_service.py`。

**接口**：完全采用规格第 9 节 HpoService；在 `hpo/__init__.py` 导出公共契约。已有 app、loop、executor 不导入本模块，本模块也不导入训练启动逻辑。

- [ ] 以下 fixture/test 作为最小端到端测试起点；另按后面的矩阵补足故障边界。

```python
import uuid
import pytest
from PIL import Image
from auto_tune.modules.dataset_snapshot.service import create_dataset_snapshot
from auto_tune.modules.hpo import HpoService, StudyConfig, ResultInput, HpoError

@pytest.fixture
def hpo_inputs(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for n in range(4):
        Image.new('RGB', (16, 16)).save(source / f'{n}.jpg')
        (source / f'{n}.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
    snapshot = create_dataset_snapshot(source, tmp_path / 'snapshots',
                                       val_ratio=0.5, seed=42, class_names={0: 'part'})
    model = tmp_path / 'fixture.pt'
    model.write_bytes(b'hpo-contract-test-not-a-real-model')
    return snapshot.snapshot_path, model

def test_pending_reload_and_budget(tmp_path, hpo_inputs):
    snapshot, model = hpo_inputs
    root = tmp_path / 'hpo'
    service = HpoService(root)
    study = service.create_study(StudyConfig(budget=1), snapshot_dir=snapshot, model_path=model)
    key = uuid.uuid4().hex
    trial = service.ask(study.study_id, request_id=key)
    restored = HpoService(root)
    assert restored.ask(study.study_id, request_id=key) == trial
    with pytest.raises(HpoError) as error:
        restored.ask(study.study_id, request_id=uuid.uuid4().hex)
    assert error.value.code == 'HPO_PENDING_TRIAL'
    restored.tell(study.study_id, trial.number,
                  ResultInput(state='FAILED', reason_code='oom'))
    assert restored.ask(study.study_id, request_id=key).state == 'FAILED'
    with pytest.raises(HpoError) as error:
        restored.ask(study.study_id, request_id=uuid.uuid4().hex)
    assert error.value.code == 'HPO_BUDGET_EXHAUSTED'
```

- [ ] 实现 create 的快照严格验证、模型普通文件/hash 绑定；ask 在持锁内重读、优先处理同 request_id、再检查 pending、预算、版本和绑定、生成并写盘后才返回；tell 校验状态和结果后原子更新，终态完全一致才幂等。
- [ ] ask 返回已存在 request_id 不重新采样、不重新校验模型内容（只读结果）；新 request_id 必须核对绑定。load 可只读环境版本不同的已有记录，不可读取未知 schema 或损坏数据。
- [ ] tell SUCCESS 的 epoch 与 config.epochs、value 范围及 evidence 交叉验证；错误结果保持 PENDING，FAILED 不能携带 value。终态重复调用不改 revision/time。
- [ ] 补全服务矩阵：

| 行为 | 必须断言 |
|---|---|
| 配置/路径非法 | 零 study 发布 |
| 快照 manifest 或模型被修改 | 新 ask 为 BINDING_MISMATCH，旧 JSON 不变 |
| 环境 Optuna/NumPy 版本变化 | load 可读，新 ask 为 VERSION_MISMATCH |
| ask/tell 写盘失败 | 未返回新记录，重载无失败变更 |
| tell 同值/冲突 | 同值幂等、不同值 RESULT_CONFLICT |
| SUCCESS 缺证据/NaN/越界 | INVALID_RESULT，仍 PENDING |
| FAILED/CANCELLED/INTERRUPTED | 原因匹配、value 空、预算消耗不恢复 |
| 两个 service 同时 ask | 只发布一候选，另一 BUSY/PENDING，不超预算 |
| 修改 load 返回对象 | 不影响磁盘与后续读取 |
| 第六个成功之后恢复继续 | TPE 重载路径仍可重复且无额外候选 |

## 任务 5 回归与交付

- [ ] 运行新增测试，逐文件，不并行。首次 red/green 记录保留在交付摘要，不伪造数字。

```powershell
$py = 'D:\Program Files\anaconda3\envs\auto_tune\python.exe'
$tests = @('test_hpo_contracts.py','test_hpo_search_space.py','test_hpo_storage.py',
           'test_hpo_sampler.py','test_hpo_service.py','test_guardrails.py',
           'test_dataset_snapshot.py','test_run_state_training_api.py','test_tuning_loop.py')
foreach ($test in $tests) {
    & $py -m pytest (Join-Path 'auto_tune/tests' $test) -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "Test failed: $test" }
}
& $py -m pytest auto_tune/tests -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw 'Full regression failed' }
git diff --check
git diff --stat
```

- [ ] 检查 HPO 模块没有导入 LLM、Popen/launch_training、UI 或 config.yaml。正常测试禁止网络请求；锁竞争的测试子进程与训练进程明确区分。
- [ ] 交付实际改动文件、测试命令和数字、安装版本、pip check 前后差异、偏离计划和遗留风险。结论为“实现完成，待 Codex 独立验收”。不运行真实训练，不修改项目文档，不提交推送，不自动进入 H1.2。

## 自查覆盖

规格第 3 节对应任务 0；第 4–5 节对应任务 1；第 8 节对应任务 2；第 6 节对应任务 3；第 7/9 节对应任务 4；第 10 节对应任务 5。文中测试代码是实施起点，不替代矩阵全部覆盖。计划已执行完成；安装版本与返修验收证据见本文顶部及独立验收记录。
