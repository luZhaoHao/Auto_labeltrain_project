# 详细实施计划（2026-08-14）

> 配套路线文档：`docs/roadmap_20260814.md`。当前版本、测试基线和下次启动检查见 `docs/development_handoff_20260814.md`。每次只选择一个经艾卡批准的小批次执行。

> 2026-08-14 发布状态：A0 审计闭环和统一训练收尾已经验收并进入 `v0.1`；A0.2 第 3-6 项仍待后续分批实施。

---

## 阶段 A0：当前闭环可靠性加固（约 4-7 天）

**目标**：保证 Module B 产生可信事实，Module C 只执行经过验证的参数，并让建议值、护栏值、实际命令和训练结果可追溯。

### A0.1 已完成（2026-08-14）

- 新增统一超参数注册表；
- `hyperparameter_changes` 和 `training_overrides` 统一校验；
- Guardrails 清洗值成为 Loop 唯一合并值；
- 拦截未知参数、非法类型和 `optimizer=auto` 冲突；
- LLM JSON 增加结构、白名单、空修改和每轮最多 3 参数校验；
- 修复 `train_analyzer` 嵌套配置读取和 Early Stopping 数据层级；
- 调优历史反馈包含 before/after/delta；
- YOLO CLI 从当前 Python 环境可靠解析；
- ZIP 上传使用安全解压；
- 新增正式 Module C/安全测试，完整测试为 79 passed。

### A0.2 后续任务

1. ✅ **已完成（2026-08-14）**：独立原子化审计记录、训练预检、精确命令复用和 fatal 失败策略。
2. ✅ **已完成（2026-08-14）**：普通训练和自动调优共用收尾流程，自动解析结果、回填 KPI、生成 Module B 报告并进入统一历史。
3. 默认训练日志只显示 epoch 关键指标，完整日志折叠查看。
4. 数据集划分改为不可变快照，不再移动原始文件。
5. API 密钥迁移到环境变量并轮换已暴露密钥。
6. 增加上传总大小、成员数量和解压后容量限制。

**验收标准**：UI、报告、审计记录、`args.yaml` 和实际命令一致；非法 LLM 参数进入训练命令的数量为 0。

**最终验收证据（2026-08-14，`v0.1`）**：完整套件 `python -m pytest auto_tune/tests -q -p no:cacheprovider` → **160 passed, 2 warnings**；两条 warning 均为既有 sklearn PCA 数值警告。真实 1-epoch 调优验证确认训练时间、epoch、CSV、Module B 报告、统一历史和审计 KPI 一致。详细兼容边界见 `docs/development_handoff_20260814.md`。

---

## 阶段 A：本地基础（约 2-3 天）

### A1. yolo11/yolo26 + task/version 配置框架（0.5-1 天）

**目标**：新增任务类型（detect/segment/classify/obb）与 YOLO 版本（v8/v11/v26）配置维度，贯穿 config → 训练 → 分析 → UI。为后续任务扩展与架构实验铺底。

**关键步骤**：
1. **模型目录**：确认/放置模型文件（`yolo11n.pt`、`yolo26n.pt` 等），与 `yolov8n.pt` 同目录。
2. **config.yaml**：`training` 段新增 `task`（默认 detect）、`version`（默认 v8）；新增 `model_catalog` 描述各版本/任务可用的模型清单（供 UI 下拉）。
3. **executor.py**：`build_yolo_command` 透传 `task`；确认 `model=yolo11n.pt` 自动推断任务。`YOLO_INTERNAL_PARAMS`（guardrails.py）补充新模型名/参数。
4. **UI（single_page.html）**：模型选择下拉改为按版本分组展示 yolov8 / yolo11 / yolo26 各型号，加入 seg/obb/cls 模型。
5. **i18n.py**：补充版本/任务相关翻译。
6. **ultralytics 升级**：确认当前环境 ultralytics 版本支持 yolo11/yolo26（必要时升级并回归测试现有流程）。

**涉及文件**：`config.yaml`、`executor.py`、`guardrails.py`、`app.py`、`single_page.html`、`i18n.py`、`loop.py`（透传 task）。

**验收标准**：
- 用 `yolo11n.pt` 和 `yolo26n.pt` 各跑通一次训练（dry-run 或短 epoch）。
- UI 能选择不同版本模型并正确启动训练。
- 现有 yolov8 流程无回归（跑通既有 pytest）。

---

### A2. Linux 迁移（0.5-1 天）

**目标**：按既有文档将项目迁移到 Linux 服务器，支持局域网访问。

**关键步骤**（按 `auto_tune/docs/linux-migration-plan.md`）：
1. 新建 `environment-linux.yml`（移除 `pyreadline3`，按需配置 GPU PyTorch 源）。
2. 新建 5 个 `.sh` 脚本：`start_server.sh`、`start_app.sh`、`setup.sh`、`run_tests.sh`、`build_package.sh`。
3. `app.py` 文件夹浏览器（~L926-952）增加 `os.name == "posix"` 分支：空路径返回 `/` 和 `~` 作为可选项；根目录 `/` 处理 `parent = None`。
4. 清理测试脚本硬编码 Windows 路径/GBK：`_final_test.py`、`_test_vision_models.py`、`_test_qwen_keys.py`、`_test_llm_apis.py`、`_test_pipeline_full.py`。
5. `host` 改为 `0.0.0.0`；文档补充 Linux 启动命令。

**涉及文件**：`environment-linux.yml`（新）、5 个 `.sh`（新）、`app.py`、5 个测试脚本、`CLAUDE.md`。

**验收标准**：在 Linux 上 `setup.sh` 建环境成功、`start_server.sh` 启动、浏览器访问 `/` 正常、文件夹浏览器能浏览 Linux 目录。

---

### A3. 目录选择分析（0.5 天）

**目标**：智能分析页通过文件夹浏览器直接选择服务器目录，不再把 ZIP 上传作为正式流程。

**关键步骤**：
1. 复核文件夹浏览器的允许根目录、路径规范化和错误提示。
2. 数据集与训练结果分别绑定目录分析端点。
3. 对不存在、无权限或结构不合法的目录给出明确反馈。

**涉及文件**：`single_page.html`、`i18n.py`。

**验收标准**：选择合法目录后触发分析并展示结果；非法目录不能越过允许根目录。

---

## 阶段 B：在线平台 Phase 1（11-18 天，以多用户最终形态一次性落地）

> 架构原则：核心层（Module A/B/C + executor + SSE）复用，数据/接入层重构。本地 = 单用户模式（默认不登录），在线 = 开启多用户层。建议一套代码、配置开关切换。

### B1. SQLite 数据库 + 用户体系（2-3 天）
- **数据模型**：`users`（id/username/password_hash/role/created_at）、`datasets`（id/owner_id/name/path/status/created_at）、`training_jobs`（id/owner_id/dataset_id/status/stage/gpu_id/params/metrics/created_at）。
- 注册/登录/登出（JWT 或 session）、角色（普通用户/管理员）、密码哈希（bcrypt）。
- 认证依赖注入：FastAPI `Depends` 校验 token；`training_running.json`/`tuning_running.json` 迁移为 DB 行或按用户分目录。
- **涉及**：`models.py`/`db.py`（新）、`auth.py`（新）、`app.py` 路由改造。

### B2. 数据隔离（2-3 天）
- 文件路径按用户隔离：`log/{user}/`、`detect/{user}/`（或 `data/users/{user}/...`）。
- 改造 `dataset_panel.py`、`train_panel.py`、`tuning_panel.py` 与 `app.py` 全部路由：所有读写都带 owner 维度。
- 文件夹浏览器限制在用户空间内（防路径遍历）。
- **涉及**：`app.py`、三个 panel、`executor.py`（detect 目录定位）、`loop.py`。

### B3. 训练任务队列 + GPU 并发调度（3-5 天，难点）
- 任务状态机：排队 → 准备数据 → 训练中 → 完成/失败/取消。
- GPU 租约：NVIDIA 管理接口发现 GPU；单任务独占单卡（`CUDA_VISIBLE_DEVICES`）；多卡可并行多任务；无空闲 GPU 排队。数据库锁/原子租约避免双分配。
- 进程注册表：任务 ID → Popen，支持重启后对照 DB/进程/Checkpoint 修复状态。
- 后台 worker：`asyncio` 队列或独立线程/进程调度器，替代当前单次触发模式。
- **涉及**：`scheduler.py`（新）、`executor.py`、`loop.py`、`probe_monitor.py`。

### B4. 多数据集 + 历史记录 + SSE 恢复（2-3 天）
- 多数据集：数据集列表 CRUD、每用户多数据集管理、数据集与训练任务关联。
- 历史记录/历史分析：由 SQLite 驱动，训练任务列表 + 报告查询 + 结果对比。
- SSE 重连：基于 DB 任务状态，断线后重新连接正在运行的训练（EventSource 重连 + 恢复当前进度）。
- **涉及**：`app.py`、`single_page.html`、`train_panel.py`、`tuning_panel.py`。

### B5. 前端适配 + 安全加固（2-4 天）
- 登录页、会话状态、个人空间、队列/GPU 状态展示。
- 安全：认证授权（每接口校验）、上传文件校验与大小限制、路径遍历防护、SQL 注入防御（ORM 参数化）、CORS/CSRF。
- 部署：nginx 反向代理 + HTTPS（可选）。
- **涉及**：`single_page.html`、`app.py`、`i18n.py`。

**阶段 B 验收标准**：多用户注册登录、各自数据集/训练相互隔离、多人提交训练自动排队、GPU 单卡独占、训练中断后重连恢复、历史记录可追溯。

---

## 阶段 C：后置 / 穿插（约 15-25 天）

> 阶段 A/B 完成及 Codex 审核后再细化。以下为任务清单与优先级。

1. **优化 LLM 分析能力**（1-2 天，随时穿插）：prompt 改进、诊断规则增强。
2. **一键分析**（1-2 天）：流程编排，合并到在线平台 UX。
3. **任务类型扩展**：
   - 分割（2-3 天）：Module A 解析多边形标签 → bbox；Module B/C 确认 mAP 列兼容；加 seg 模型。
   - OBB（1-2 天）：Module A 解析 6 值标签含角度；加 obb 模型。
   - yolov5（2-3 天）：executor 命令构造 + 指标列映射（`mAP@0.5`/`obj_loss`）。
   - 分类（3-4 天）：目录结构/data.yaml/metrics/评分公式全链路改造。
4. **模型架构实验平台**（8-15 天，优先级最后）：模块库 → 结构生成器 → executor yaml 训练 → 决策层动作空间 → UI 消融实验记录。预设变体模板 + 有限组合。
5. **在线标注 Phase 2**（将来按需评估）：接 CVAT 路线 B 解耦集成（导出对接 1-2 天 / 模型回传 2-3 天 / token 鉴权 3-5 天 / 训练队列沿用 B3）。

---

## 依赖关系

```
A1（task/version 框架） → 后续所有任务扩展 + 架构实验
A2（Linux 迁移）        → 在线平台部署前提
A3（拖拽上传）          → 独立，随时可做
B1/B2（DB+用户+隔离）   → B3/B4 的前提
B3（训练队列）          → B4/B5 的前提
C 各项                 → 依赖阶段 A/B 完成
```
