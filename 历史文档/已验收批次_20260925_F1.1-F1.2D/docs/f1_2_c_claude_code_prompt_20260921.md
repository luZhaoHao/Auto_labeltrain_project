# F1.2-C Claude Code 提示词（由艾卡转发；先依赖决策，再实施）

你负责 F1.2-C「共享交付层与正式 Docker」的业务代码及对应测试。F1.2-A ONNX 通道、F1.2-B Docker 可行性已通过 Codex 独立验收；工作区含多个批次的未提交改动，全部保留，不重置、不提交、不推送。请先阅读 `AGENTS.md`、`docs/development_handoff_20260814.md`、`docs/implementation_plan_20260814.md` 第 5.3 节、`docs/f1_2_windows_docker_onnx_spec_20260918.md`、`docs/f1_2_b_codex_review_20260921.md` 和当前 `Dockerfile`、`compose.yaml`、`docker/entrypoint.sh`、`docker/requirements-runtime.txt`、`auto_tune/delivery/`。

## 第 0 步：依赖决策门槛（先做、先停）

Docker 运行依赖当前有意排除了 ONNX 相关包，不能把 A 批导出功能视为已交付。先只读核对 A 批导出代码实际 import、已验收的 `auto_tune` Conda 环境、容器当前依赖及 Ultralytics 导出调用；给艾卡与 Codex 一份最小依赖表：包名、精确版本、用途、来源、预计下载/镜像体积增量、版本冲突风险，以及哪些功能需要 `onnx`、`onnxruntime`、`onnxslim`。对不确定的依赖先做不改变镜像的验证。**在艾卡明确确认新增依赖及版本前，不修改依赖文件、不安装包、不重建镜像、不启动长训练。**不要把 `latest`、未固定版本或自动安装作为解决方案。

## 获批后的实现范围

1. 保持单个 `auto-tune` 镜像、一个完整 Studio 容器。**正式 Docker 的正常启动配置必须申请 NVIDIA GPU**，不把 GPU 做成用户额外启用的可选路径；不交付 CPU 训练，也不安排无 GPU 服务/UI 或 CPU 回退验收。GPU 不可用时应明确报错，不得悄悄改用 CPU。只补 C 批确需的 Compose/运行说明，不影响 Windows 无 Docker 的后续直接安装。
2. 收敛 Windows/Docker 共用的配置路径、目录初始化、权限/依赖预检及清晰错误提示。Docker 的配置、日志、SQLite、数据集、训练结果和权重必须留在挂载目录；失败时不得静默写入镜像层或把真实配置复制进镜像。
3. 将经艾卡确认的最小 ONNX 运行依赖纳入受控 Docker 依赖，保证容器内 A 批已验收的可信 `.pt` → FP32 `.onnx` 路径可用；不要扩展 A 批业务入口，不增加任意服务器路径输入。FP16 仅按已有条件显示/验证。
4. 针对配置与挂载路径、拒写目录、缺依赖、正式启动配置申请 GPU、GPU 不可用时明确报错、重启恢复及镜像敏感文件排除先写失败测试，再做最小实现。对 HPO 与 LLM 调优分别补最简流程测试，核对创建、参数/决策校验、实际执行、终态与产物关联；LLM 使用受控测试响应，不消耗真实外部模型额度。测试应尽量使用临时目录和桩，不能修改真实用户数据。除 GPU 必需这一交付约束外，不改训练业务语义；不增加第二镜像、公开调度 API、创建新依赖环境或修改权威项目文档。

## 自测与交付报告

- Python 与 pytest 必须使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`，先报告 `sys.executable` 和版本。先跑新增/受影响测试，最后跑完整 `auto_tune/tests`；记录命令、通过数、warning 和耗时。
- Docker 已由艾卡开启时，先检查 Engine、E 盘镜像空间、现有容器与端口；不得自行启动 Docker Desktop，不改镜像存储位置，不删除任何既有镜像、卷、容器或 `docker-data/`。构建使用 `docker build --pull=false -t auto-tune:local .`，不加 `--no-cache`，不反复重建；如遇网络/权限阻塞，停止并报告，不改宿主机代理或 mirror。
- 用来源可信、最小合法 YOLOv8 Detect 数据集和已在受控库内的可信小权重，在同一 GPU 镜像、隔离挂载及非默认端口下做最简 HPO 和 LLM 调优闭环。HPO 以最少合法 trial、每 trial 最少合法 epoch 验证；LLM 以受控、确定性的测试响应验证建议→校验→实际参数→最短合法 GPU 训练→终态，不调用付费外部 LLM。若已有受控流程允许复用一次 GPU 训练证据，则不重复启动等价训练；否则分别跑两条路线的最小合法任务。不运行 CPU 训练，也不单独启动无 GPU 容器。记录镜像 ID、Torch/CUDA/GPU、任务终态、日志/SQLite/权重/结果路径和重启前后核对值；不跑长训练。额外做一次容器内 ONNX FP32 导出、结构和最小推理检查，避免只凭导入成功宣称可用。
- 提供 RED→GREEN、新旧测试对比、构建结果、GPU/HPO/LLM/ONNX 实测、完整改动清单、资源占用、偏离计划和遗留风险。不要提交、推送或改动 README、路线图、规格、实施计划、交接记录及 DOCX。完成后停下，等待 Codex 独立验收。

## Codex 的省额度独立验收约定（供你准备证据）

请把每个关键结论压缩为可复核的命令与结果摘要，并保留必要日志路径；不要粘贴整份长日志。Codex 先审文件和失败测试，再用已构建镜像独立验证正式 Compose 的 GPU 接入、训练、HPO 与 LLM 最简流程，复核重启持久化及 ONNX 最小路径。不启动 CPU 或无 GPU 验收容器，不重复下载基础镜像、不做无缓存构建、不做长训练。仅在代码或镜像与自测不一致、或证据不足时追加针对性测试。该约定不降低独立验收门槛。
