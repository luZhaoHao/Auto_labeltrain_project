# F1.2-B Docker 构建上下文最小返修提示词

请先读 `AGENTS.md`、`docs/f1_2_b_codex_review_20260921.md`、`docs/f1_2_b_claude_code_prompt_20260918.md`。只修复 Codex 复审指出的 Docker build context 排除缺口；不要进入 F1.2-C。

1. 保留所有已有未提交改动、`docker-data/`、镜像和卷；先记录 `git status --short`。不得重置、删除或清理。
2. 在 `auto_tune/tests/test_docker_delivery_files.py` 先新增测试，要求 `.dockerignore` 明确排除仓库根的 `docker-data/` 以及 `.env`、`.env.*` 等常见本地凭据文件。用指定 Conda 解释器运行新增测试并记录预期 RED；不要把现有静态测试通过当成此项已覆盖。
3. 只修改 `.dockerignore` 的必要规则使测试 GREEN。不得修改业务代码、ONNX 依赖、Docker 存储位置、README 或任何权威文档。
4. 使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 独立记录 `sys.executable` 与版本，运行新增/现有 Docker 静态测试和完整 `auto_tune/tests` 回归；运行 `git diff --check`。
5. Docker 服务若未运行，先请艾卡开启并等待确认，不自行启动。服务可用后用固定基础镜像和 `docker build --pull=false -t auto-tune:local .` 重建，记录发送的 build context 大小；复核镜像无真实配置、数据库、数据集、权重或凭据。不得使用 `--no-cache`、`--pull`、清理缓存或变更 mirror/代理。
6. 报告 RED→GREEN、完整测试结果、构建与镜像检查结果、修改文件清单及遗留风险；不提交、不推送。完成后停止，等待 Codex 再验收。
