# H1.2 第四轮复验后返修提示词

请仅处理 H1.2 第四轮独立复验的 R2a-3 executable 历史审计缺口，不进入 H1.3。文件名保留 round2 以稳定引用，以当前内容为准。

先读取 AGENTS.md、docs/development_handoff_20260814.md、docs/h1_2_codex_review_20260908.md 的“第四轮返修复验 当前有效结论”，以及 H1.2 详细规格和六任务计划。前三轮记录仅作历史追溯。

当前独立证据：1830 passed、2 条既有 PCA 警告；前三轮 19 项独立反例通过；本轮 TPE/Random 各两次真实 1 epoch 训练及产物核对通过。第四轮探针 1 failed、1 passed。R2a-1、R2a-2 和 R6 历史收尾行为已通过现有反例，不重新开发。

唯一剩余问题：_cross_validate 用 attempt.command[0] 重建预期 executable，属于待校验字段与自身比较。将 FINALIZED attempt 的 command[0] 换为无关程序，status 仍接受。新启动已有当前环境校验会拦截；本次修复历史审计一致性，不夸大成错误程序启动漏洞。

要求：
1. executable 的历史校验使用与 attempt.command 分离的冻结依据，不从待校验 command[0] 自行推导“正确答案”。新记录在初次准备阶段冻结该身份，后续不可随当前 PATH 漂移改写。
2. 历史 status 和 RESULT_READY/TOLD 收尾不解析当前 YOLO，但 executable 与尾部参数均须与冻结事实一致。新启动仍解析当前 executable/环境并进行严格比对。
3. 对缺少冻结依据的旧 execution-v1 记录给出明确兼容策略。不得从 command[0] 自动补齐字段后声称验证通过，不静默迁移、不覆盖旧审计事实。若改变 schema 或兼容语义，先提交具体方案供 Codex 核对，再实现该部分；不能自行将此缺口当作已获批准的权衡。
4. 本次只要求交叉字段审计一致性，不新增依赖、签名系统或通用防篡改服务。
5. 正式测试覆盖：单独替换 executable 被拒绝；尾部污染仍拒绝；合法历史在当前 resolver 不可用/路径变化时可按契约收尾；新启动漂移仍拒绝；旧记录兼容路径明确且不伪造验证通过。

独立反例参考 log/h1_2_codex_review_20260908/round4/test_round4.py。正式测试放入 auto_tune/tests，不 import 日志目录。先保留失败证据再修复，不放宽生产校验迎合夹具。

所有 Python/pytest 使用 D:\Program Files\anaconda3\envs\auto_tune\python.exe，先记录 sys.executable/Python 版本。运行本轮、全部 HPO、executor、RunState、finalizer 回归，再运行全量 auto_tune/tests -q -p no:cacheprovider、pip check、范围内 git diff --check。普通 pytest 不启动真实 YOLO，真实短训练由 Codex 独立验收。

仅修改必要业务代码和对应测试；不改项目文档、UI、LLM、真实配置或依赖声明。不删除文件，不提交或推送，保留无关工作区修改。已通过行为不得回退。交付报告列出改动、实际测试证据、兼容策略、偏离和风险，然后等待 Codex 复验。