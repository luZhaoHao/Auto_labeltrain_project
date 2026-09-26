# F1.2-C Claude Code 最终小返修提示词（由艾卡转发）

Codex 已完成最后一轮针对性验收：CPU HPO 新建/启动/恢复禁令、GPU 创建、HPO 页面 GPU-only、LLM 两次 JSON 决策对应两条 `decision_attempts` 共 8 项独立测试全部通过。不要再改这些部分，不跑全量回归、GPU 训练、HPO trial、LLM 训练、ONNX、镜像重建或容器验证。

F1.2-C 尚余一个同契约缺口：实现报告主动披露，历史 **COMPLETED CPU HPO** 仍可通过 `/api/hpo/studies/{study_id}/train-best` 或 `/api/training/start` 的 `source_hpo` 路径发起 `device=cpu` 的正式训练。当前测试中仍明确断言 `device=cpu` 命令可生成（例如 `test_hpo_best_reuse.py`、`test_hpo_formal_training.py`）。艾卡的最终要求是产品**不提供任何 CPU 训练**，因此这不是可保留的 H1.3 语义。

只做以下最小返修：

1. 在两条 HPO 最佳配置正式训练入口的服务端权威边界上，读取并重建完 `verified.effective` 后、创建训练目录/状态/控制器/进程之前，调用共享的 GPU-only 校验。历史 CPU study 仍允许 GET、列表、排名和产物读取，但不得通过任何入口启动正式训练。
2. `/train-best` 与 `/api/training/start {source_hpo: ...}` 对历史 CPU 来源均返回 HTTP 422、稳定错误码 `HPO_GPU_REQUIRED` 和固定安全文案；零新训练目录、零状态文件、零命令构造、零控制器、零进程。
3. GPU 来源 `device="0"` 保持现有行为。若正式训练边界还需要验证该编号当前实际可见，应复用已注入的 GPU 探测，不引入第二套错误码或硬件探测逻辑。
4. 更新原先断言 `device=cpu` 可以启动的测试：保留旧 CPU 研究可读性测试，将启动断言改为 422 与零副作用；新增 GPU 来源仍可到达既有 accepted/controller seam 的正例。
5. 不删除旧记录，不迁移历史文件，不修改权威文档，不提交、不推送。

验证仅需：

- 先记录 RED：两条入口对历史 CPU source 原本能进入启动路径。
- GREEN：CPU 两入口均 422 `HPO_GPU_REQUIRED` 且零副作用；GPU 正例通过。
- 运行 `test_hpo_best_reuse.py`、`test_hpo_formal_training.py`、与本次门禁直接相关的 HPO API 测试即可；不再跑完整 3029 项（实现方上一轮完整回归及 Codex 针对性验收已经覆盖）。
- 运行 `git diff --check`，报告改动文件、测试数量及结果后停止，等待 Codex 做最后一次静态/定向复验。
