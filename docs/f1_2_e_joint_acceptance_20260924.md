# F1.2-E 联合验收记录

日期：2026-09-24
状态：代码与两种交付关键路径通过，手册截图和最终发布包待完成

## 1. 结论

F1.2-A 至 F1.2-D 的功能与交付实现已完成联合复验，未发现新的产品 P0/P1。Windows ZIP 与单 Docker GPU 镜像均能运行同一个完整 Auto-Tune Studio；正式训练不提供 CPU 回退。PT 转 ONNX、HPO 和 LLM 调优关键路径可用。

F1.2 尚未判定完整产品最终验收通过。剩余事项是：艾卡补充软件安装手册和软件操作手册的实际截图；Codex 完成截图脱敏、插入和 DOCX 版式复核；随后重新构建最终 Windows ZIP 和 Docker 镜像，生成 SHA-256 发布清单并执行最终快速验收。

## 2. 完整自动化

- 解释器：`D:\Program Files\anaconda3\envs\auto_tune\python.exe`
- Python：3.10.18
- 命令：`python -m pytest auto_tune\tests -q -p no:cacheprovider`
- 结果：**3249 passed, 2 warnings in 434.28s**
- 两条 warning 均为既有 sklearn PCA `explained_variance_ratio_` 运行时提示。
- `git diff --check`：通过，仅有既有 CRLF 提示。

## 3. Docker 联合验收

- Docker Engine：29.6.2，Linux/x86_64。
- 候选镜像：`auto-tune:local`。
- 镜像 ID：`sha256:92f2c9ef01091d9441c2d248894daead81dc3fb3ae45d181720268a5f991eafb`。
- 镜像大小：5,487,282,666 字节，约 5.49 GB。
- Compose：单服务、单镜像、单容器；声明 NVIDIA GPU 预留和六个持久化挂载。
- GPU 探针：CUDA 可用，设备数 1，NVIDIA GeForce RTX 3060 Laptop GPU；PyTorch 2.5.1+cu121，CUDA Runtime 12.1。
- 健康接口：HTTP 200，响应 `{"status":"ok","product":"auto-tune-studio"}`。
- 首页：HTTP 200，190,077 字节，标题 `YOLOv8 Auto-Tuning Agent`。
- 容器健康状态：`healthy`；非特权用户：`studio`；无 OOM、无运行错误。
- 持久化：容器重启后验收标记存在，配置 SHA-256 保持不变。
- 镜像内容：不存在 `.git`、`docker-data` 或真实 `auto_tune/config.yaml`；脱敏模板和入口脚本在位。

本轮自建验收容器和专用目录已清理，未删除现有镜像、卷或业务数据。本机仍有一个引用旧镜像 ID 的已停止 `auto-tune-studio` 容器，不应直接重新启动；应使用当前 Compose 按当前镜像标签重新创建。

## 4. Windows ZIP 联合验收

- 当前源码重新构建 `AutoTuneStudio-Setup-0.2.0.zip` 成功。
- ZIP 大小：549,743 字节；归档条目：131。
- 包含安装、启动、升级、卸载入口和完整性锁文件。
- 不包含真实 `config.yaml`、`.env`、SQLite、测试目录、`.pt` 或 `.onnx`。
- 私有运行环境：Python 3.10.21、PyTorch 2.5.1+cu121、ONNX 1.17.0、ONNX Runtime 1.22.0、Ultralytics 8.3.253。
- Windows GPU：NVIDIA GeForce RTX 3060 Laptop GPU 可用。
- 永久启动入口在正确隔离 `LOCALAPPDATA` 下能读取安装状态，并对非法端口返回稳定 `PORT_INVALID`，不启动服务。
- 安装、同版本入口修复、代码升级、依赖锁变化升级、失败回滚、卸载与数据保留已由 F1.2-D 定向测试和实际安装复验覆盖。

本节记录的是验收候选包。最终发布 ZIP 必须在手册收口和发布检查后重新构建并重新计算 SHA-256。

## 5. ONNX 联合验收

- 来源：已完成 5 epoch、`imgsz=640` 的真实 GPU 训练权重。
- FP32 ONNX 大小：12,238,452 字节。
- `onnx.checker.check_model`：通过。
- ONNX Runtime 最小推理：通过。
- 输入：`[1, 3, 640, 640]`。
- 输出：`[1, 5, 8400]`。
- 推理 Provider：`CPUExecutionProvider`，用于验证交付文件可加载和执行，不表示产品提供 CPU 训练。

## 6. HPO 与 LLM 双数据集验收

使用两套真实数据集：`dataset_cegai_914v2` 与 `dataset_cegai_buchon513`。

两套 HPO 均使用 GPU 0、`imgsz=640`、1 epoch 和 1 个候选，研究终态均为 `COMPLETED`，候选终态均为 `SUCCESS`，并生成 `results.csv`、`best.pt` 和 `last.pt`。

两套 LLM 调优均真实调用已配置的 DeepSeek 服务。每套模型响应一次，结构校验和确定性语义校验均通过，无纠错重试。模型建议降低 `lr0` 并启用 `cos_lr`；建议通过事实引用、语义规则和 Guardrails 后写入实际 `args.yaml`，并分别完成一次 GPU 训练。两套审计和迭代终态均为 `completed`，审计未发现凭据或模型响应原文泄露。

该验证证明流程可执行，不用于评价 1 epoch 模型的业务精度。

## 7. 未处置事项与退出门槛

当前无未处置产品 P0/P1。F1.2-E 剩余退出门槛：

1. 艾卡按清单提供软件安装手册和软件操作手册截图。
2. Codex 完成截图脱敏、编号、图注、插入及 DOCX 全页渲染检查。
3. 更新交接记录、路线图、实施计划、规格和三份研发 DOCX。
4. 检查 `.gitignore`、提交范围、敏感信息和许可证。
5. 重新构建最终 Windows ZIP 和 Docker 镜像，记录版本、大小、镜像 ID 和 SHA-256。
6. 对最终发布物执行健康页、GPU、Windows 启动和 ONNX 快速验收。
7. 经艾卡确认后，才提交并推送到指定公开仓库。

## 8. 2026-09-25 最终交付与手册复验

- 最终 Docker 镜像 ID：`sha256:118adba004766b5f7704fb7519e74da7446b323eed1295821d9a280198af8461`。
- Docker 镜像归档：`AutoTuneStudio-Docker-0.2.0-image.tar`，`5,487,472,128` 字节，SHA-256 `9852F736B6127C6333A834732D14170FC6FAE4F2352322397279077724B3A244`。
- Docker 源码包：`AutoTuneStudio-Docker-0.2.0-source.zip`，`531,875` 字节，SHA-256 `90B3D9190E4CA4C49F8CF65F2BF1C84F53BB3890439E6BBA3B1432A1F49561D3`。
- Windows 完全离线包：`AutoTuneStudio-Setup-0.2.0.zip`，`2,777,946,444` 字节，SHA-256 `12DEAB27C7D3A53BA9F8E4DA5DECFFAE3A14A83944EA978750E31FED855A72FB`。
- Docker 正式 Compose 已验证 GPU、1 GiB 共享内存、健康接口、普通训练、HPO、DeepSeek LLM 调优与重启后历史数据保持。
- Windows 最终 ZIP 已验证完全离线安装、私有运行环境、GPU 普通训练、HPO、DeepSeek LLM 调优、FP32 ONNX、同版本入口修复及默认卸载保留用户数据。
- 《软件安装手册》9 页、7 张实际安装/验收截图；《软件操作手册》15 页、21 张实际界面截图。两份 DOCX 均已使用 LibreOffice 转换并逐页检查，无截断、破图、密钥或明显版式缺陷。

至此，原第 7 节第 1、2、5、6 项已完成。尚未关闭的发布动作是最终完整回归、敏感信息与提交范围检查、清理分类，以及艾卡确认后的提交与推送。
