# Auto-Tune Studio 0.2.0 发布文件清单

日期：2026-09-25
适用版本：Auto-Tune Studio 0.2.0

## 1. 大文件交付物

以下文件保存在本地 `build_output/`，不提交 GitHub。接收后应先核对字节大小和 SHA-256，再安装或导入。

| 文件 | 用途 | 字节大小 | SHA-256 |
|---|---|---:|---|
| `AutoTuneStudio-Setup-0.2.0.zip` | Windows 完全离线 ZIP 安装包 | 2,777,946,444 | `12DEAB27C7D3A53BA9F8E4DA5DECFFAE3A14A83944EA978750E31FED855A72FB` |
| `AutoTuneStudio-Docker-0.2.0-image.tar` | 可直接 `docker load` 的 GPU 镜像归档 | 5,487,472,128 | `9852F736B6127C6333A834732D14170FC6FAE4F2352322397279077724B3A244` |
| `AutoTuneStudio-Docker-0.2.0-source.zip` | 供用户自行执行 `docker build` 的脱敏源码包 | 531,875 | `90B3D9190E4CA4C49F8CF65F2BF1C84F53BB3890439E6BBA3B1432A1F49561D3` |

配套校验文件：

- `AutoTuneStudio-Setup-0.2.0.sha256.txt`
- `AutoTuneStudio-Docker-0.2.0-SHA256SUMS.txt`

## 2. 用户文档

- `docs/Auto-Tune软件安装手册.docx`
- `docs/Auto-Tune软件操作手册.docx`

对应 Markdown 用作项目内可审查来源；DOCX 为面向用户的交付版本。操作视频可后续补充，不属于 0.2.0 验收要求。

## 3. 已验证边界

- Windows 安装包完全离线建立私有 Python 3.10 GPU 运行环境，不要求用户预装 Conda、Python、Docker、WSL 或 CUDA Toolkit。
- Docker 正式启动申请 NVIDIA GPU，不提供 CPU 训练降级。
- Windows 与 Docker 均已验证正常训练、HPO、DeepSeek LLM 调优及持久化关键路径。
- FP32 ONNX 已完成导出、结构检查和 ONNX Runtime 最小推理。
- 最终完整自动化：`3447 passed, 2 warnings`；两条 warning 为既有 sklearn PCA 空数据方差提示。

## 4. 安全与分发要求

- 不把安装包、镜像归档、依赖缓存、数据集、权重、训练结果、日志、SQLite、真实配置或凭据提交到 GitHub。
- API Key 由用户在 Studio 中保存到 Windows 凭据管理器或对应主机凭据存储；不得写入安装包、配置模板或命令行。
- Docker 数据目录和 Windows 用户数据在卸载、容器重建或镜像升级前应先备份。
