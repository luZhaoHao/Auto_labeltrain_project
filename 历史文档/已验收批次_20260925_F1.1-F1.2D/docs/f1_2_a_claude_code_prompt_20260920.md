# F1.2-A Claude Code 实施提示词：上传 PT 后手动导出 ONNX

请只实施 F1.2-A；不要开始 Docker、Windows、Cloud API 或训练流程重构。先读 `AGENTS.md`、`docs/development_handoff_20260814.md`、`docs/f1_2_windows_docker_onnx_spec_20260918.md` 第 4、6 节，以及 `docs/superpowers/plans/2026-09-20-f1-2-a-onnx-export.md`，按该计划逐项做真实 RED→GREEN。

产品需求很小：沿用现有受控权重库上传区，操作员上传可信 YOLOv8 Detect `.pt`，选择这条已上传的 `model_id`，点击“导出 ONNX”，最后在该 `.pt` 同目录得到 `<stem>.onnx` 并能在网页下载。FP32 是默认和必须验收的路径；“高级选项”仅在实测支持时开放 FP16，标作“半精度”而非 INT8 量化。固定 `imgsz=640`、`opset=17`、`dynamic=False`、`simplify=False`、`nms=False`，页面不提供路径输入或这些参数的编辑。若目标已存在，提示冲突，不覆盖。兼容来源 `legacy` 不可导出。不要从训练结果、历史记录、任意服务器路径选取权重；不要自动导出或修改训练终态。

复用 `auto_tune/modules/model_store/service.py` 的来源身份与 SHA-256 复核、`auto_tune/ui/model_store_api.py` 的路由及 CSRF/origin 限制、`auto_tune/ui/templates/single_page.html` 的唯一上传区和 `auto_tune/ui/static/hpo.js` 的模型列表。新增尽量小的独立导出服务及定向测试；加载 `.pt` 只发生在用户确认可信来源并点击导出后的受控子进程里。子进程用于保护 Studio 主进程可用性，**并不是恶意权重的安全沙箱**。导出超时或异常只给稳定错误码，不回传服务器路径和堆栈；临时产物要清理，原权重与已有 ONNX 不受影响；重复点击和并发相同导出不能产生覆盖。导出后用 `onnx.checker` 验证，网页刷新后仍能查询和下载。

先用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 输出 `sys.executable`、Python 和现有 Ultralytics/ONNX/ONNX Runtime 版本。当前环境目录已有 `ultralytics 8.3.253`、`onnx 1.17.0`、`onnxruntime-gpu 1.22.0`，但这不是可运行验证。如解释器无法正常启动或导入，报告阻断原因；不得改用系统 Python。不要自行新增依赖；确实缺包时先列包名、固定版本、大小和用途，由艾卡确认后再安装。防止 Ultralytics 缺包时自动联网安装；真实导出必须用来源可信的小权重，不能通过下载未知权重取巧。

测试至少覆盖 managed/legacy、伪造和变化的 `model_id`、已存在 ONNX、重复提交、超时/失败、临时清理、无路径泄露、网页状态及下载。对真实 FP32 输出完成结构与固定输入最小 PyTorch/ONNX Runtime 推理验证（原始输出形状一致，并报告数值容差及结果）；FP16 若无法做真实验证就保持禁用，不得让上游的警告变成“成功”。运行定向回归、相关 UI 测试、`node --check`、`git diff --check`，最后运行完整 `auto_tune/tests`，只如实报告实测结果。

只修改业务代码和对应测试。不要修改 README、路线图、规格、实施计划、DOCX、图片或历史文档；不提交、不推送 GitHub。完成后给 Codex 汇报修改文件、RED/GREEN、真实导出和推理证据、全量回归结果、偏离计划和剩余风险，等待独立审查。
