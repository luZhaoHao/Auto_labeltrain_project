# H1.1 Codex 独立验收记录与返修要求

2026-09-08 当前结论：H1.1 返修后验收通过，尚未提交或推送；H1.2 尚未启动。Claude Code 完成初版，Codex 按艾卡安排修复 R1–R4：恢复语义校验、study 身份绑定、路径父链与链接检查、可变模型入口复验。

验证证据：auto_tune Conda 环境，Python 3.10.18；全量 1672 passed、2 条既有 PCA 警告（59.97 秒），新增返修回归 35 passed，原独立复现 9 passed；pip check 无依赖冲突。解释器为 D:\Program Files\anaconda3\envs\auto_tune\python.exe。

批次边界：H1.1 仅基础模块，无真实训练、训练子进程、网络 LLM 或 UI/API 接入。修改 UI 前的自动化验证由 Codex 完成，艾卡暂不介入手工测试；H1.2 的必要短训练由 Codex 验证，H1.3 再安排界面体验。下一步先确认 H1.2 规格与实施计划，不复用 H1.1 提示词启动新批次。

完整验收记录：docs/h1_1_codex_review_20260907.md。以下早期实施指令和初验失败证据保留用于追溯；以本次状态为准。原 H1 6–10 人日为包含已完成 H1.1 的总估算，剩余工期尚未重新估算。

返修实现：新增 validation.py 统一 read/write/采样历史语义检查；存储写入前验证原记录，拒绝覆盖损坏事实；保持原子写入与稳定错误码。新增 test_hpo_recovery_validation.py 共 35 项。TDD：初批 30 项先见 28 failed、2 passed；扩展后 35 项曾见 1 failed、34 passed，修复后 35 passed。以下为 2026-09-07 初验历史，所列 R1–R4 均已关闭，原返修提示词不再执行。

日期：2026-09-07。结论：**本轮验收未通过，禁止提交推送，不进入 H1.2。**

Claude Code 已实现 H1.1 正常业务路径；Codex 独立复跑现有完整测试通过，但额外边界复现暴露四类阻断问题。此结论不是否定既有测试结果，而是现有覆盖不足以证明恢复数据的可信边界。

## 验证证据

- 解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。
- 依赖实查：optuna 4.5.0、alembic 1.19.2、SQLAlchemy 2.0.52、colorlog 6.12.0、greenlet 3.5.5，与批准版本一致。
- `python -m pip check`：No broken requirements found。
- `python -m pytest auto_tune/tests -q -p no:cacheprovider`：**1637 passed, 2 warnings，47.14 秒**。警告为 test_analyzer 两项既有 sklearn PCA warning。
- Codex 独立探针：`log/h1_1_codex_review_20260907/test_independent_review.py`，**9 failed，3.77 秒，无跳过**。这些是要求拒绝非法操作而当前实现未拒绝的失败，不是依赖或测试启动失败。
- 探针在受控 pytest 临时目录创建合成快照、非真实权重文件和目录链接；不启动训练、不调用 LLM。探针文件在 Git 忽略的 log 目录，不提交运行产物；返修者需将有效案例整理到正式测试中。
- requirements 实际仅追加批准的五行；现有业务入口无 HPO 集成，本批边界未扩至 H1.2。

## R1 持久化记录缺少完整语义校验 P1

位置：`models.py` 的 TrialRecord/StudyRecord validators，`storage.py` 的 read，`service.py` 的 ask 幂等返回分支。

复现：创建合法 pending trial，将 study.json 中 candidate_params.lr0 改为 999.0；load_study 接受，相同 request_id 的 ask 直接返回该候选。该分支不会走新候选 validate_candidate。

另四种损坏同样被 load 接受：distribution 字符串为 `{}`、SUCCESS 结果却只有 oom 原因、成功证据 epoch=999 超过配置 epochs、两个 trial 复用同一 request_id。不是要求防止外部篡改所有合法事实，而是规格明确要求损坏/非法记录必须拒绝。

返修：建立供读取/写入/历史重建共用的语义验证。验证六候选键及范围、sampled 到 candidate 的一致映射、条件分支、固定分布类型/范围/log/step/choices、状态与 result/reason/finished_at、epoch 上限、pending 数量与位置、request_id 唯一、trial 数不超预算。缺失/非法分布解析与 Optuna 历史导入失败均转换为稳定 HPO_CORRUPT_STUDY，不泄漏 KeyError/ValueError。不静默修复历史，失败时文件和 revision 不变。

至少新增上述六个复现（五种 load 损坏及一次幂等 ask 返回越界候选），再补规格中预算/pending/映射/分布不一致的反例。成功值属于 SUCCESS；失败/取消/中断不得携带成功结果。未知 schema 仍须 fail-closed。

## R2 文件内 study_id 未与请求目录绑定 P1

位置：`storage.py` read 中 StudyRecord.model_validate 后直接 return。

复现：创建 A/B 两个合法 study，将 B 的 study.json 内容复制至 A 的文件；load_study(A) 返回 B 的记录。当前内部 trial_id 校验只能证明 trial 跟随文件内 study_id，不能证明它是所请求的 A；锁仍持有 A，后续读写也可能用错身份。

返修：在 read 中验证 record.study_id == 请求 study_id，错误返回 HPO_CORRUPT_STUDY；验证发生在任何幂等返回、采样或结果处理之前。测试 A 中放 B 的空记录及包含 trial 的记录均拒绝，不得读写 B 或返回其候选。无需对历史无损数据添加新 schema。

## R3 存储根及父链链接检查遗漏 P1

位置：`storage.py` _check_dir_legit 的 current=self._root 后仅遍历相对 study 子路径；`service.py` create_study 在进入 store 校验前先 mkdir。

复现：storage_root 为指向另一临时目录的真实 Windows 目录符号链接；create_study 成功，study.json 被写入链接目标。独立试验实际创建链接成功，不是 mock，也没有跳过。

返修：创建目录前检查受控 root 及已有父链，拒绝 reparse/symlink；再次检查最终 study 目录和已有 study.json/锁文件。检查链接应使用 lstat 保留原始路径语义，不能先 resolve 抹掉链接。所有创建/读取/写入使用相同规则；不因拒绝非法路径而先留下已发布 study。补 root 链接、祖先链接及锁文件链接测试；输入模型父链按绑定边界同样核对。保持本地可信单机范围，不要求抵御外部恶意进程的任意实时替换竞态。

## R4 已实例化配置绕过入口复验 P2

位置：`models.py` StrictModel 配置与 `service.py` create_study/tell 对模型实例的直接信任。

复现：`config=StudyConfig(); config.budget=101; create_study(config, ...)` 返回成功并发布 JSON。Pydantic 模型未冻结/无赋值验证，服务 isinstance 分支直接采用该实例；因此字典输入会拒绝的配置，模型输入却可写入。

返修：在公共入口对模型实例也进行完整重新验证，可采用 dump 后统一 model_validate，并处理嵌套修改/model_copy(update=...)，或采用真正不可绕过的等效方案。不能仅调用默认不复验实例的 model_validate。create 的非法配置映射 HPO_INVALID_CONFIG 且零发布；tell 的非法结果映射 HPO_INVALID_RESULT 且状态不变。store.write 必须在提交前验证新记录语义，避免可变对象绕过。不必为了这个问题重写现有非 HPO 模型。

## 返修后的验收顺序

1. Claude Code 只修上述 H1.1 问题及对应测试；保持已批准搜索协议、依赖版本和分批范围。
2. 先运行独立探针确认红色基线，再把反例纳入正式 HPO 测试，完成修复后逐文件测试。
3. 复跑独立探针（全部通过、链接测试不可静默跳过）、正式 HPO 测试及完整现有套件，记录实际数字；不以原 1637 代替修复后结果。
4. 不修改项目文档，不提交推送，不进入 H1.2；向 Codex 返回文件清单、测试证据和偏离说明。
5. Codex 再次独立验收通过后更新批次完成状态与文档。

## 可直接交给 Claude Code 的返修提示词

```text
艾卡安排：按 docs/h1_1_codex_review_20260907.md 修复 H1.1 的 R1–R4。
先读取当前 H1.1 规格、实施计划与该审查记录，不扩大到 H1.2。
Codex 完整回归实跑为 1637 passed/2 warnings，但独立探针 9 failed，验收未通过。
请先运行 log/h1_1_codex_review_20260907/test_independent_review.py 复现，再将有效反例
整理到正式 HPO 测试；修复重载语义、目录身份绑定、root/父链链接检查和模型输入复验。
不要改搜索空间/采样协议、不要加依赖，不覆盖工作区已有修改。
仅用 D:\Program Files\anaconda3\envs\auto_tune\python.exe 运行代码与测试，逐文件测试，
完成后复跑独立探针与完整套件，报告实际结果。链接检查不可用时明确报告，不能用跳过冒充通过。
不启动真实训练、不改项目文档、不提交、不推送、不进入下一批，完成后交 Codex 再验收。
```
