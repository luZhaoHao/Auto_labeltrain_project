# F1.2-B Codex 独立复审（2026-09-21）

结论（2026-09-21 返修后再验收）：**F1.2-B 验收通过；尚未提交或推送**。此前发现的 Docker 构建上下文排除缺口已补齐，并完成独立回归与镜像复验。本批结论仅覆盖 Docker 可行性基线，不代表 F1.2-C 的真实 CPU/GPU 训练已通过。

## 返修后独立再验收

- 实际文件核对：`.dockerignore` 新增 `.env`、`.env.*` 和 `docker-data` 排除规则；`test_docker_delivery_files.py` 新增 4 项对应断言。未将 Claude Code 的报告本身当成测试证据。
- 指定解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`，`sys.executable` 一致，Python 3.10.18。
- 独立复跑两组定向测试：`82 passed in 19.63s`（Docker 交付文件 51、交付运行时 31）。
- 独立复跑完整 `auto_tune/tests`：`2968 passed, 2 warnings in 428.50s`；两条仍为既有的 sklearn PCA warning。
- Docker Engine 29.6.2 运行中。再验收开始时，`auto-tune:local` 为 Claude Code 报告的 `sha256:bda5e1737c0f5f9df2601aece9f85666fa8bb82d574626f28f1d06d19dab30f7`，大小 5,455,055,707 字节。独立执行 `docker build --pull=false -t auto-tune:local .` 成功；BuildKit 显示上下文增量传输 38.41 kB，所有构建步骤命中缓存。构建后同一标签的镜像清单 ID 变为 **`sha256:fa60ee1fd947304d850312cc87a4d4c184ec92a8edf034ef0bcec3435b7897d9`**，大小不变，原 `bda5…` ID 已无法单独 inspect。此次未运行镜像清理命令；Docker 重建同名 tag 的这一副作用已记录。增量上下文数字本身不作为排除规则的唯一证据。
- 在独立重建前，用返修镜像启动一次性容器（宿主端口 `127.0.0.1:18003`）：`/healthz` 返回 HTTP 200 与精确 JSON `{"status":"ok","product":"auto-tune-studio"}`；首页返回 HTTP 200、190077 字节，并包含 `YOLOv8 Auto-Tuning Agent`。容器内 `/opt/auto-tune/docker-data`、真实 `config.yaml`、`.git`、`.env` 均不存在；脱敏配置模板存在，运行用户为 `studio`。本次临时容器已停止并自动移除，未触碰既有卷或 CVAT 容器。
- Claude Code 返修报告另记录了 40 MiB `docker-data` 哨兵文件的上下文排除实验、镜像全盘检查和配置模板逐字节对比。这些属于实现方提供的补充证据；Codex 未重复该哨兵实验，也不把 BuildKit 的增量上下文大小误读为完整上下文体积。
- 旧 `auto-tune-studio` 容器仍处于退出状态且指向旧镜像 ID `396f1bc7e179…`；本次未启动或重建它。后续需要正式 Compose 运行时应按当前镜像重建该容器，勿直接把旧容器状态视作新镜像验证结果。

## 原复审发现及处理

原结论为暂不验收，原因是 `.dockerignore` 未排除 `docker-data/` 及常见本地凭据文件。返修补齐规则及测试后，该阻塞项已解除。以下原始复审证据保留用于追溯，不再代表当前验收状态。

## 独立验证证据

- 指定解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`，Python 3.10.18。
- 定向及相关回归：`180 passed in 17.38s`。
- 完整回归：`2964 passed, 2 warnings in 386.52s`；两条 warning 均来自既有的 sklearn PCA 测试。
- `docker info` 显示 Linux Engine 29.6.2；`auto-tune:local` 镜像 ID 为 `sha256:396f1bc7e17953303a890ff9ee6e8608df68ee18a6a996e37df0d308b1b2d33d`，大小 5,455,054,366 字节，运行用户 `studio`。
- `docker compose config --services` 只有 `studio`；独立执行 `docker compose up -d --no-build` 后，容器为 `healthy`，六路挂载，宿主机只绑定 `127.0.0.1:8000`。`/healthz` 返回 HTTP 200 和精确 JSON；首页返回 HTTP 200、190077 字节。
- 独立执行 `docker compose restart` 后，既有持久化标记、SQLite 与配置文件 SHA-256 均保持不变。镜像中未发现 `/opt/auto-tune/auto_tune/config.yaml` 或 `.git`。`docker run --rm --gpus all` 中 PyTorch 2.5.1+cu121 报告 `cuda_available=True`，设备为 RTX 3060 Laptop GPU；本批未执行 GPU 训练。
- `git diff --check` 无差异错误。复验启动的容器已用 `docker compose stop` 停止；镜像、容器、挂载文件及卷未清理。

## 原阻塞项：Docker 构建上下文未排除运行时目录（已修复）

`.gitignore` 已排除 `/docker-data/`，但 `.dockerignore` 没有该规则。`compose.yaml` 默认将配置、日志、SQLite、训练目录和模型权重置于仓库下的 `docker-data/`；当前已存在 `docker-data/config/config.yaml` 和 `docker-data/log/auto_tune.db`。因此 Git 排除不能保证这些运行时文件不进入 Docker build context。当前 Dockerfile 只复制 `auto_tune/` 和受控交付文件，未发现它们进入最终镜像；但构建上下文可能把运行时文件发送给 builder，且以后扩展 COPY 时会扩大风险。这违反本批“不把真实配置、数据和训练产物送入构建上下文”的门槛。

当时提出的返修要求：先新增失败测试，明确断言 `.dockerignore` 排除 `docker-data/` 及常见 `.env` 凭据文件；再补最小规则，运行 RED→GREEN、完整回归、`docker build --pull=false`，确认构建上下文和最终镜像不包含运行时数据。实现方报告了 RED→GREEN 与哨兵实测；Codex 独立复核了实际规则、回归、构建和镜像运行。

## 后续批次风险，不作为 B 批返修范围

- 镜像尚无 ONNX 导出所需依赖；F1.2-C 正式 Docker 应先经艾卡确认固定版本与体积，再验证完整 Studio 导出能力。
- 本批仅验证 GPU 透传可见；CPU/GPU 最小真实训练与重启恢复属于 F1.2-C 退出门槛。
- Linux 宿主机拒绝写入的挂载权限分支尚未在真实受限挂载下实测。

本记录只复审 F1.2-B；工作区中 F1.2-A 及其他既有未提交改动保持原状，不得混入本批验收或提交。
