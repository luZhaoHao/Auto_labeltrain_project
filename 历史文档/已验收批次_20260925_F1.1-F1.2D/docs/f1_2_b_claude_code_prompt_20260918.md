# F1.2 B Claude Code 实施提示词

请严格实施 F1.2-B Docker 可行性基线。只有 F1.2-A ONNX 导出已经完成 Codex 独立验收后才可使用本提示词；不得开始正式 Docker 收尾、Windows 交付或扩展 PT 转 ONNX。

本提示词由艾卡手动发送给 Claude Code。当前工作区已有 F1.2-A 和其他未提交改动，全部保留，不得重置、覆盖或混入 B 批报告。此前中断的尝试可能留下未跟踪的 `auto_tune/tests/test_delivery_runtime.py`；开始前先检查该文件与 `git status --short`，将其作为待核对的 RED 测试草稿，不能假定测试已经运行或通过。

开始前依次阅读：

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/development_handoff_20260814.md`
4. `docs/f1_2_windows_docker_onnx_spec_20260918.md`
5. `docs/superpowers/plans/2026-09-18-f1-2-b-docker-feasibility.md`

## 已批准范围

- 为现有完整 Studio Web UI 建立一个 Linux Docker 镜像和一个容器的可行性基线。
- 增加受控的启动地址、端口和配置路径解析，桌面默认值必须保持 `127.0.0.1:8000`。
- 增加最小 `/healthz` 运维探针。它不是用户 API，不进入页面导航，不暴露版本、路径、环境、配置或错误细节。
- 建立受控 Docker 运行依赖文件。
- 新增 `Dockerfile`、`.dockerignore`、`docker/entrypoint.sh` 和一个服务的 `compose.yaml`。
- 验证 CPU 容器启动、健康检查、首页、配置初始化、目录挂载和重启持久化。
- 编写并运行计划要求的自动化测试。

## 明确禁止

- 不新增 `/api/v1/jobs`、外部任务 API、Engine 模式、队列、权限或 CVAT 代码。
- 不增加任何 ONNX 功能或依赖。
- 不修改直接训练、HPO、LLM 调优、评分、审计、数据快照、运行身份或持久化语义。
- 不修改 README、路线图、交接记录、实施计划、DOCX、图片或历史文档。
- 不把真实 `auto_tune/config.yaml`、API 密钥、数据集、模型权重、日志、审计、训练结果或缓存放入镜像和 Git。
- 不使用未固定的 `latest` 镜像。
- 不新增第二个容器，不拆 Web/Worker。
- 不删除文件，不格式化无关代码，不提交，不推送。
- 不自行启动、重启、安装、升级或重置 Docker Desktop，不删除镜像或卷，不迁移 Docker 存储。本机 Docker 镜像存储挂载在 E 盘；需要 Docker 服务时先请艾卡开启并等待确认。

## 实施要求

严格逐任务执行 `docs/superpowers/plans/2026-09-18-f1-2-b-docker-feasibility.md`，使用真实 RED→GREEN 流程。所有宿主机 Python 与 pytest 命令必须使用：

```yaml
D:\Program Files\anaconda3\envs\auto_tune\python.exe
```

Docker 推荐基线为：

- `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Python 3.10
- `torch==2.5.1`、`torchvision==0.20.1` 的 CUDA 12.1 wheel
- 一个 `auto-tune` 镜像、一个完整 Studio 容器

若该基线与实际驱动或依赖冲突，先记录可复核证据并停止，不得自行换成 `latest` 或大幅改变范围。

当前已知 Docker CLI 存在，但最近一次检查时 Docker Desktop Linux Engine 未运行；后台启动曾因 `dockerInference` 本地套接字绑定冲突而失败。艾卡负责开启 Docker 服务，Codex 负责后续宿主机排查与独立验收。如果艾卡尚未确认服务已开启，先做代码和静态测试，不要尝试启动 Docker；若确认后 `docker info` 仍失败：

1. 完成可独立验证的代码、静态测试和 `docker compose config` 检查；
2. 明确标记 Docker build、运行、GPU 和持久化实测为阻塞；
3. 不得声称 F1.2-B 完成或通过；Claude Code 只需提供可复核的失败信息，不为宿主机故障改动业务逻辑。

## 最低验证

按计划运行：

- 新增 delivery/runtime 定向测试；
- Docker 文件静态契约测试；
- AI 配置、启动、上传安全、训练门禁、运行状态和 HPO 生命周期相关回归；
- Docker Engine 可用时运行镜像构建、CPU 启动、`/healthz`、首页、配置初始化和重启持久化；
- 完整 `auto_tune/tests` 回归；
- `git diff --check`；
- 最终 Git 文件清单与敏感内容检查。

本批不要求真实训练，也不要求 GPU 容器训练；这些由 Codex 在后续 F1.2-C 正式 Docker 验收中完成。F1.2-A 只验收 ONNX 导出。必须如实报告 GPU 验证条件和当前状态。

## 完成后只提交报告，不提交 Git

报告必须包含：

1. 根因和设计说明；
2. 修改文件及职责；
3. 每项 RED→GREEN 证据；
4. Docker build、CPU 启动、健康检查、UI 和持久化实测结果，或明确阻塞原因；
5. 所有测试命令、解释器和结果；
6. 偏离计划之处及理由；
7. 剩余风险，特别是 GPU、镜像体积、依赖和 Windows 路径问题；
8. `git status --short` 分类；
9. 明确声明未修改文档、未提交、未推送、未加入真实数据或凭据。

完成后停止，等待 Codex 独立审查。
