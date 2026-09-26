# H1.2 Claude Code 执行提示词

2026-09-08 H1.2 第四轮独立复验：尚未通过。全量 1830 passed、2 条既有 PCA 警告（91.87 秒），原 19 项反例通过；补充反例 1 failed、1 passed。本轮四次真实短训练及产物核对通过。仅剩 R2a-3 executable 历史审计缺口，详见 docs/h1_2_codex_review_20260908.md。

当前仅按验收记录安排返修；以下初版实施说明保留为契约与追溯依据，不代表验收通过。

状态：艾卡已于 2026-09-08 确认 H1.2 方案，并要求按计划提供执行提示词。以下提示词可直接转交 Claude Code，只授权本批实现与自动化测试。本文件替代旧 H1.1 编码入口，不代表代码已完成。

```text
请在当前 Auto_labeltrain_project 工作区实现 H1.2 Detect HPO 训练执行与恢复闭环。

艾卡已确认本批规格和六任务计划，请直接按计划逐项实现并测试，不停留在方案建议或重复请求本批启动确认。遇到规格冲突、缺少必要输入或需要超范围修改时，先核对现有代码和文档，再明确报告具体阻碍。

必须先读取：
1. AGENTS.md
2. docs/development_handoff_20260814.md
3. docs/h1_1_codex_review_20260907.md
4. docs/superpowers/specs/2026-09-07-h1-1-hpo-foundation-design.md
5. docs/superpowers/specs/2026-09-08-h1-2-hpo-execution-design.md
6. docs/superpowers/plans/2026-09-08-h1-2-hpo-execution.md

以本批规格和六任务实施计划为唯一 H1.2 编码依据。H1.1 已返修验收通过，基线 1672 passed、2 条既有 PCA 警告；此数字不是你本次的测试结果。

实现公开 HpoRunner.prepare/run/resume/status；增加严格 execution.json 执行审计、独立非阻塞执行锁、训练适配、CSV 最佳 epoch 指标和确定性排名。通过 HpoService ask/tell 管理预算与结果，不改变原 schema、搜索协议、条件空间、终态或幂等语义。

重点：
- 启动意图成功落盘后才 launch，结果意图落盘后才 tell；命令、候选、实际 args、CSV 证据、运行身份一一绑定。
- 使用绑定快照和同一初始本地模型；每 trial 独立训练，不接上个 trial 权重，不依赖参考训练或全局最新数据。
- 护栏发生 clamp 就拒绝，不能改变已采样候选；不自动缩 batch、不补预算、不重试原 trial。
- 保守恢复：LAUNCH_INTENT 身份未知时阻断，PID+token 验证，不盲目重训或误杀 PID；正常中断、tell/finalizer 故障按矩阵幂等恢复。
- 持久化失败不得吞掉，不推进失败的内存状态，不启动下一 trial；停止/超时必须确认进程结束。
- 指标来自对应 results.csv 的有效最大 metrics/mAP50-95(B)，同分最早 epoch；trial 同分按 number；不能替换成 final 指标、best.pt fitness 或 LLM 决策。

范围：仅计划列出的 H1.2 模块、测试、验收脚本及必要 executor task/amp 白名单、HpoService 只读绑定校验接口。禁止改 UI、app.py、模板、LLM 控制器、真实配置、数据库、requirements 或其他已有业务。保留全部无关工作区修改，不创建项目副本/工作树。

不新增或升级依赖。所有 Python/pytest 使用：
D:\Program Files\anaconda3\envs\auto_tune\python.exe
先打印 sys.executable、Python 版本并 pip check。按任务先 red 后 green，运行计划的定向与全量测试，不能用 skip 规避进程锁/路径链接/恢复边界。自动化测试不启动真实 YOLO、不调用网络 LLM；提供显式运行的短训练验收脚本，真实训练由 Codex 独立验收时执行。

UI 在 H1.3：沿用智能训练页面，四个选项为干运行、按原来参数训练、HPO 算法调参、大模型调参。本批只保证后台可接入，不提前改页面。

完成后提供：改动文件、实际解释器与版本、每条测试命令和结果、首次失败证据、六任务完成情况、偏离计划与遗留风险。若规格不可实现或需要超范围修改，明确报告具体原因，不私自改协议。

不要修改任何项目文档，不要提交/推送 GitHub，不要启动 H1.3。等待 Codex 独立验收及文档回写。
```
