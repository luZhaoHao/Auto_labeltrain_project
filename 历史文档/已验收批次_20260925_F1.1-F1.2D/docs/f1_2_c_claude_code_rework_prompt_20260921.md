# F1.2-C Claude Code 返修提示词（由艾卡转发）

Codex 对 F1.2-C 做了独立复验。以下项目已经通过，不要重复长时间工作：当前镜像 `sha256:07727b3b25f6dfc594cfde632fa16d053bab0bd66411607129c6c1d42546e8af` 可见 RTX 3060；隔离 GPU 容器健康页和首页正常；真实直接训练 1 epoch 完成；HPO `budget=1 × epochs=1` 在 GPU 0 上完成且容器重启后仍为 `COMPLETED`；3015 项完整测试通过；ONNX 依赖版本与镜像体积符合批准范围。不要重新下载基础镜像、不要使用 `--no-cache`、不要重复 GPU 长训练、不要清理镜像/卷/容器或现有数据。

本轮只返修以下两个问题。继续使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，不提交、不推送、不修改权威文档。

## P1：Docker 正式交付仍允许创建 CPU HPO

艾卡已明确：正式 Docker 正常使用必带 GPU，不交付 CPU 训练。当前实现与此冲突：

- `auto_tune/ui/static/hpo.js::renderDeviceOptions()` 仍固定追加 `CPU`；页面行为测试也断言用户能选择 CPU。
- `auto_tune/modules/hpo/execution_models.py::ExecutionConfig` 默认仍为 `cpu`，校验允许 `cpu`。
- Codex 在正式 GPU 容器内向 `POST /api/hpo/studies` 提交 `execution_config.device="cpu"`，实际返回 HTTP 201 和 `READY`。Codex没有启动该 CPU study。

先新增失败测试，再做最小修复：

1. HPO 页面只列出服务端探测到的 GPU 编号，不显示 CPU；无 GPU 时创建按钮不可用并显示稳定、可操作的中文提示。
2. `build_hpo_defaults()` 不得把无 GPU 情况降级成 CPU；返回明确的 GPU 不可用事实/提示。
3. 服务端创建边界必须拒绝 `execution_config.device="cpu"`，返回稳定错误码和 HTTP 422；不能只靠前端隐藏。GPU 编号仍须按现有规则校验，并在执行前验证设备实际可用。
4. 直接训练和 LLM 调优已经固定/实测 `device=0`，不要改它们的业务语义。不要为了 Docker 约束破坏旧 HPO 历史记录的读取；如果持久化旧记录包含 `cpu`，允许只读展示，但禁止新建、启动、恢复任何 CPU HPO。为旧 CPU study 的 start/resume 拒绝补测试。
5. 更新所有仍断言 CPU 可选、CPU 可提交或 CPU 默认值的测试与 JS 测试场景，明确新契约是 GPU-only。

建议稳定码使用一个单一、明确的交付错误，例如 `HPO_GPU_REQUIRED`；错误文本不得包含路径、堆栈或底层异常。

## P1 调查项：LLM 响应数与 decision_attempts 不一致

实现报告称首轮受控桩看到 4 次响应（其中 2 次文字、2 次 decision），但产品审计只记录 1 个 `decision_attempt`；第二次复跑又无法复现。现有代码理论上会在每次 `json_mode=True` 决策响应校验后立即持久化 attempt，Codex 已复跑两项相关测试并通过，因此先调查、不要猜测性重构。

1. 用确定性桩为每次 `call_decision_llm` 记录：顺序号、`json_mode`、调用用途（决策/纠错/最终摘要/分析）、响应摘要哈希；不得记录密钥或完整敏感提示词。
2. 构造“第一次 decision 缺 `fact_package_id`、第二次纠错合法”的测试：必须恰好产生 2 个决策调用，审计必须恰好有 2 个 `decision_attempts`，第一个无效、第二个有效，顺序一致。
3. 若额外两次确认为 `json_mode=False` 的文字总结/分析调用，说明其来源，并证明它们不属于 `decision_attempts`，无需强行塞入决策审计；但应确保现有相应日志/审计边界清楚。
4. 若出现任何 `json_mode=True` 响应没有对应 attempt，先写可稳定复现的失败测试，再修复；审计写入失败必须继续 fail-closed，不能进入命令构造或训练。
5. 删除或回收测试钩子，不在正式产品输出额外提示词、模型原文或凭据。

## 复验要求

- 报告 P1 的 RED→GREEN：前端无 CPU、API `device=cpu` 返回 422、旧 CPU study 禁止 start/resume、GPU `device="0"` 仍可创建。
- 报告 LLM 调用分类和确定性测试结果；明确首轮“4 次响应/1 个 attempt”的根因或当前可证实边界。无法确定根因时不得写“已解决”，应保留为阻塞或说明还缺什么证据。
- 运行受影响测试及完整 `auto_tune/tests`。仅在依赖/代码发生影响镜像内容的变化时执行一次 `docker build --pull=false -t auto-tune:local .`；预计本轮不需要依赖下载。
- Docker 只需一次短验证：正式 Compose 仍申请 GPU、GPU HPO 最小请求可创建；CPU HPO 请求在启动任何训练前返回 422。不要再跑 1 epoch GPU 训练、HPO 完整 trial、LLM 真实训练或 ONNX 推理，Codex 已取得本轮独立证据。
- 给出改动文件、测试结果、错误码、LLM 调用分类证据、偏离和遗留风险；停止等待 Codex 再验收。
