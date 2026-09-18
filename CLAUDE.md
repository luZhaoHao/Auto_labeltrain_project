# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

请用中文回答用户。

## 协作分工与工作流程（重要）

本项目采用 **Claude Code + Codex 双工具协作**：

| 职责 | 负责方 |
|------|--------|
| 编写代码 | **Claude Code** |
| 方案定调、技术路线决策、代码审核 | **Codex** |
| 运行测试、验证结果 | **Codex** |
| 测试通过后将代码上传到 GitHub | **Codex** |

### 工作流程

1. **定调**：Codex 确定技术方案、路线与验收标准。规划文档见：
   - `docs/development_handoff_20260814.md`（当前版本、验收基线与下次开发入口）
   - `docs/roadmap_20260814.md`（路线图）
   - `docs/implementation_plan_20260814.md`（后续实施顺序与验收）
2. **编码**：Claude Code 按 Codex 定调的计划编写/修改代码。
3. **测试**：Codex 运行测试验证（完整 pytest + 必要的 1 epoch 冒烟训练）。
4. **发布**：测试通过后由 **Codex** 上传代码到 GitHub。

### 注意事项

- **Claude Code 不执行 GitHub 上传**，上传统一由 Codex 负责。
- **Claude Code 只编写业务代码和对应测试**；不修改 README、路线图、规格、实施计划或发布说明（这些由 Codex 维护，见 `docs/development_handoff_20260814.md` 启动检查清单）。
- **本文件 CLAUDE.md 由 Claude Code 维护**（Codex 不编写）。编码期间发现的文档/规范问题，在交付时口头提示即可。
- 测试结果以 **Codex 的验证为准**；Claude Code 完成编码后不得宣称"已验证通过"，须等 Codex 测试确认。
- **当前已发布稳定版本为 v0.2**；Studio S1.1–S1.5、S2.0–S2.4、体验修复 P1–P5、Q1、H1 与 **F1.1 产品优化与稳定版本冻结**均已完成独立验收。艾卡于 2026-09-18 确认 F1.1-A、F1.1-B、F1.1-C 整体验收通过；最终完整自动化为 **2738 passed / 2 warnings**，两条 warning 均为既有 sklearn PCA warning。F1.1 已恢复四模式、修正总体最佳展示与目标绑定、完成动态运行名转义、LLM 报告绑定与结构化输出约束，并保持已验收训练/HPO/LLM 协议不变。当前正式任务仍为 **YOLOv8 Detect**，下一入口为 **F1.2 Windows、单 Docker 镜像与 API 交付适配**；不设置 H1.4。软件操作手册已完成，软件安装手册待下一 Part 完成验收后编写。发布状态与下一批范围以交接记录和艾卡的新批准为准。每次只选择一个经艾卡批准的小批次，不要同时展开多个方向。
- 每次测试完后杀死测试用的服务器进程，让用户自己开启服务器自己测试。
- **测试运行内存约束**：不要一次并行跑多个测试文件；逐个文件运行以节省本机内存（完整套件 `pytest auto_tune/tests` 除外）。

YOLOv8 Auto-Tuning Agent — a three-module closed-loop system for automated YOLOv8 training optimization:

- **Module A (Dataset Analyzer)**: Analyzes dataset quality (blur, exposure, SNR, bbox geometry, class balance, clustering)
- **Module B (Train Analyzer)**: Three-stage training diagnosis pipeline (Python metrics → LLM text diagnosis → Qwen-VL vision analysis)
- **Module C (Agent Engine)**: Perception → Decision (LLM) → Guardrails → Execute → Probe Monitor auto-tuning loop

> **当前已发布版本 v0.2**。S1.1–S1.5、S2.0–S2.4、P1–P5、Q1、H1 与 F1.1 均已完成独立验收；F1.1 最终基线为 **2738 passed / 2 warnings**。艾卡于 2026-09-18 确认 F1.1 整体验收通过并冻结，下一入口为 **F1.2 Windows、单 Docker 镜像与版本化 API 交付适配**。Research R1 与 Cloud 暂不排期、不自动启动。已验证环境为 Windows、Python 3.10 和 Ultralytics YOLOv8 Detect；Segment、OBB、Pose、YOLO11/26 与 YOLOv5 Detect 不在当前计划。

## Commands

> **重要**: 每次测试完后，杀死测试用的服务器进程。让用户自己开启服务器自己测试。

> **测试环境（必须记录）**：所有 pytest 测试统一使用 `auto_tune` conda 环境，禁止使用系统 Python。
> - 交互式终端内：先 `conda activate auto_tune`，再运行 `python -m pytest ...`。
> - 非交互式 PowerShell 工具中 `conda activate` 不可用，必须使用环境绝对路径解释器：
>   `& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest ...`

```powershell
# Activate environment (interactive terminal)
conda activate auto_tune

# Start web UI
python -m auto_tune.main

# Start web UI (via helper script)
python start_server.py

# Start web from batch (background, logs to log\server_log.txt)
start_server.bat

# Dry-run tuning test (no actual training)
python -m auto_tune.main --dry-run

# Full tuning loop (no UI, CLI mode)
python -m auto_tune.main --train

# Full suite（正式基线见 docs/development_handoff_20260814.md）
# 最近官方验收基线：1484 passed / 2 PCA warnings / 0 skipped（Q1.1/Q1.2 验收后完整套件）
# H1.1 新增 modules/hpo 模块与 6 个测试文件（test_hpo_*.py），已通过 Codex 验收（2026-09-08）；H1.1 验收后完整套件本地 1672 passed / 2 PCA warnings / 0 skipped
# 最新完整基线：F1.1 验收冻结后 2738 passed / 2 warnings（此前本地运行：H1.3 后 2426、F1.1-A 页面返修后 2616、F1.1-B 后 2732）
# 内存约束：新批次定向测试请逐个文件运行，不要一次并行跑多个测试文件
python -m pytest auto_tune\tests -q -p no:cacheprovider

# Run all tests (verbose)
python -m pytest auto_tune\tests -v

# Run single test file
python -m pytest auto_tune\tests\test_train_analyzer.py -v

# Run single test function
python -m pytest auto_tune\tests\test_train_analyzer.py::test_analyze_training_results_all_runs -v

# Module B: analyze training results (CLI, edit test_modelb.py to select run_name)
python test_modelb.py

# Module C: component tests (stages 1-5)
python test_modelc.py                    # Run all
python test_modelc.py --stage 1          # Guardrails only
python test_modelc.py --stage 3          # Decision Agent (needs LLM API)
python test_modelc.py --stage 5          # Loop dry-run

# 首次配置（config.yaml 含本地路径，不入库）
Copy-Item auto_tune\config.template.yaml auto_tune\config.yaml

# 前端：语法检查 + 直接运行单个 JS 行为场景（需要 node 在 PATH）
node --check auto_tune/ui/static/hpo.js
node auto_tune/tests/js/minidom.js <scenario> auto_tune/ui

# H1.2 执行验收入口（只读校验，但**需要真实短训练**，仅 Codex 验收使用）
python auto_tune/scripts/verify_hpo_execution.py --snapshot-dir <snap> --model-path <pt.pt>
```

> 前端行为测试（`test_hpo_ui_behaviour.py`）在 Node 中执行真实 `hpo.js`：无 `node` 时整个模块 skip，`_run()` 结果按 scenario 名进程内缓存，因此实测耗时很短。只改 JS/模板后必须至少跑该文件 + `test_hpo_ui_rework.py`。

## Architecture

### Directory Layout

```
auto_tune/
├── main.py                          # Unified entry point (UI / dry-run / train)
├── config.yaml                      # Global config (thresholds, params; 不含真实 API 凭据)
├── modules/
│   ├── dataset_analyzer/            # Module A
│   │   ├── analyzer.py              #   Orchestrator
│   │   ├── image_quality.py         #   Blur, exposure, SNR analysis
│   │   ├── bbox_geometry.py         #   Size, aspect ratio, overlap, spatial bias
│   │   ├── class_stats.py           #   Distribution, balance
│   │   └── feature_cluster.py       #   Feature extraction + DBScan clustering
│   ├── train_analyzer/              # Module B (active — NOT training_diagnosis which is stale)
│   │   ├── analyzer.py              #   Three-stage orchestrator
│   │   ├── results_parser.py        #   Parse results.csv + args.yaml
│   │   ├── curve_analysis.py        #   Loss/metric trend analysis (uses np.polyfit)
│   │   ├── issue_detector.py        #   Overfitting, underfitting, plateau, etc.
│   │   ├── run_comparator.py        #   Multi-run comparison & summary
│   │   ├── llm_analyzer.py          #   Stage 2: DeepSeek text diagnosis
│   │   ├── vision_analyzer.py       #   Stage 3: Qwen-VL vision consultation
│   │   ├── crop_utils.py            #   Generate error crop images for vision
│   │   ├── experiment_history.py    #   Unified history store (atomic/idempotent; legacy tuning read-only)
│   │   └── training_finalizer.py    #   Shared finalize: Module B + KPI + history (manual & tuning)
│   ├── dataset_snapshot/            # S1.2 不可变数据集快照（复制发布/manifest/data.yaml 校验）
│   ├── security/                    # S1.3 凭据安全（凭据管理器/env）+ Endpoint 策略 + 脱敏
│   ├── input_safety/                # S1.4 目录输入预检（允许根/成员数/容量/链接）
│   ├── run_state/                   # S1.5 统一运行状态 + PID 身份校验 + EventBroker
│   ├── local_index/                 # S2 Core SQLite 查询索引（models/database/repository/service）
│   ├── reference_dataset/           # P2 参考训练与数据集快照绑定（resolve_reference_dataset）
│   ├── presentation/                # P4/P5 共享展示层（纯 Python，无框架依赖）
│   │   ├── experiment_labels.py     #   P4 字段/枚举/布尔标签 → 稳定翻译键（experiment_field_label 等）
│   │   └── experiment_views.py      #   P5 报告/审计只读 View Model 构建（有界读取 + run_id 身份 + 稳定错误码）
│   ├── hpo/                         # H1.1/H1.2 Detect HPO 搜索契约、持久化与训练执行（JSON 唯一事实源）
│   │   ├── models.py                #   严格契约（StudyConfig/StudyRecord/TrialRecord/ResultInput/HpoError）
│   │   ├── search_space.py          #   条件搜索空间与候选映射（SGD momentum_sgd / AdamW beta1_adamw → momentum）
│   │   ├── sampler.py               #   Optuna 历史重建采样（rebuild-per-trial-v1，不缓存 pickle）
│   │   ├── storage.py               #   原子 JSON + 排他锁（flush/fsync/replace + msvcrt/fcntl）
│   │   ├── validation.py            #   读/写/重建同一深度重校验（trial/history/record 不变量）
│   │   ├── execution.py             #   HpoRunner.prepare/run/resume/status（顺序执行、停止、恢复）
│   │   ├── execution_models.py      #   execution.json（hpo-execution-v1）契约，含冻结 command_executable
│   │   ├── execution_storage.py     #   执行审计原子落盘 + `.hpo-runner.lock` 全局执行锁
│   │   ├── execution_adapter.py     #   严格训练适配（FIXED_PARAMS 白名单，不解析客户端参数）
│   │   ├── metrics.py               #   目标指标提取（val mAP50-95 best epoch）
│   │   ├── ranking.py               #   确定性排名（rank_trials，不用 LLM 裁决）
│   │   └── service.py               #   HpoService（create/ask/tell/load；绑定/版本/预算/幂等）
│   └── agent_engine/                # Module C
│       ├── loop.py                  #   Main tuning loop orchestrator
│       ├── perception.py            #   Aggregate Module A + B reports
│       ├── decision_agent.py        #   LLM-driven hyperparameter suggestions
│       ├── parameter_registry.py    #   Unified hyperparameter registry (whitelist/types/bounds)
│       ├── guardrails.py            #   Validate & clamp hyperparameter changes
│       ├── executor.py              #   Preflight + launch/manage YOLO training subprocess
│       ├── audit.py                 #   TuningAuditSession: atomic audit JSON + redaction
│       ├── final_summary.py         #   P1 确定性终局总结 TXT（build_deterministic_summary/LLM 摘要）
│       └── probe_monitor.py         #   Early-epoch monitoring (continue/abort/retry)
├── ui/
│   ├── app.py                       # FastAPI server with SSE streaming（路由/应用绑定总汇）
│   ├── i18n.py                      # zh/en translations
│   ├── hpo_api.py                   # H1.3 HPO HTTP API 投影（create/start/query；只读校验，不重实现采样）
│   ├── hpo_controller.py            # H1.3 HPO 后台控制器（daemon 线程跑 runner.run/resume，释放共享槽位）
│   ├── hpo_reuse.py                 # H1.3 最佳配置复用（resolve_hpo_verification：固定条件验证）
│   ├── hpo_training.py              # H1.3 正式训练提交 + 受控产物 + 关联结果投影（复用普通训练控制器/门禁）
│   ├── static/
│   │   └── hpo.js                   # H1.3 HPO 前端（草稿/选择代号/轮询/最佳与正式训练关联）
│   ├── templates/
│   │   └── single_page.html         # SPA (all-in-one HTML+CSS+JS)
│   └── components/
│       ├── dataset_panel.py         # Module A display helpers
│       ├── train_panel.py           # Module B display helpers
│       ├── tuning_panel.py          # Module C display helpers
│       └── experiment_panel.py      # Unified history display adapter (adds audit_filename)
├── scripts/
│   └── verify_hpo_execution.py      # H1.2 执行验收入口（启动真实短训练；仅 Codex 验收用）
├── tests/                           # pytest test suite
│   ├── test_analyzer.py             # Module A integration tests
│   ├── test_bbox_geometry.py
│   ├── test_class_stats.py
│   ├── test_feature_cluster.py
│   ├── test_image_quality.py
│   ├── test_train_analyzer.py       # Module B integration tests
│   ├── test_results_parser.py
│   ├── test_curve_analysis.py
│   ├── test_issue_detector.py
│   ├── test_run_comparator.py
│   └── js/minidom.js                # 无依赖 DOM+fetch 驱动，供 test_hpo_ui_behaviour.py 执行真实 hpo.js
├── utils/
├── docs/                            # Documentation (operator manual, architecture, learning guide)
└── log/                             # Analysis reports + uploaded datasets (gitignored)

# Root-level test/integration scripts
├── test_modelb.py                   # Module B CLI — edit run_name and run directly
├── test_modelc.py                   # Module C staged tests (--stage 1..5)
├── _test_*.py                       # Ad-hoc scripts for API/model testing
├── start_server.py                  # Helper: python start_server.py
└── start_server.bat                 # Windows background startup → log\server_log.txt
```

### Data Flow

```
User uploads dataset → Module A analysis → dataset_report.json
                          ↓
User starts training → Module B analysis (3 stages) → train_*_report.json
                          ↓
Module C: Perception (reads reports) → Decision (LLM suggests changes)
  → Guardrails (validate) → Execute (launch training) → Probe Monitor (early epochs)
  → auto-analyze (Module B again) → auto-loop (repeat) or return
```

### Training Output Structure

Training runs are stored in `detect/train*/` (manual) or `detect/autotune_*/` (auto-tuning):
```
detect/train8/
├── args.yaml           # Training parameters
├── results.csv          # Per-epoch metrics
├── confusion_matrix_normalized.png
├── val_batch_labels/    # Ground truth labels visualization
└── val_batch_pred/      # Predictions visualization
```

### Reports (JSON)

All analysis reports go to `log/`:
- `dataset_report_ds_*.json` — Module A
- `train_*_report.json` — Module B
- `tuning_history.json` — Module C history (**LLM feedback only**; read-only for UI)
- `experiment_history.json` — **Unified history** (UI source; atomic/idempotent, schema v1.0)
- `tuning_audit_{session_id}.json` — Atomic per-session audit record (decision/guardrails/command/baseline/result, redacted)
- `training_running.json` / `tuning_running.json` — Runtime status flags
- `auto_tune.db` / `db_backups/` — S2 Core SQLite 可重建查询索引及备份（位于 `log/`）；JSON 审计/报告/运行状态仍为事实来源，数据库损坏不改变训练/调优真实终态

### Key Config ([config.yaml](auto_tune/config.yaml))

Configured sections: `project`, `llm` (DeepSeek), `vision` (Qwen-VL), `guardrails`, `probe`, `dataset_analyzer`, `train_analyzer`, `training`, `local_index`. API 凭据不写入 config.yaml（S1.3）：Windows 默认经 Windows Credential Manager（凭据引用 `AutoTuneStudio/text/deepseek`、`AutoTuneStudio/vision/qwen`），环境变量 `AUTO_TUNE_TEXT_API_KEY` / `AUTO_TUNE_VISION_API_KEY` 为只读最高优先级来源；config.yaml 仅保留 model/endpoint 等非敏感配置，由 `auto_tune/config.template.yaml` 复制而来且不入库。

### UI Pages (SPA)

Single-page app at [single_page.html](auto_tune/ui/templates/single_page.html) with 5 tabs:
1. **Projects** — Edit project info, view config
2. **Dataset Report** — Upload dataset, run Module A
3. **Intelligent Analysis** — Module B results + 四种训练模式（干运行 / 按原参数训练 / HPO 算法调参 / 大模型调参，H1.3 起；四种模式共用同一控制区与布局，由 `hpo.js` 驱动 HPO 部分）
4. **Training Monitor** — Real-time SSE training output
5. **History** — Past training/tuning iterations（S2 Core：SQLite 索引徽标 + 数据集筛选 + 旧历史导入；P3 最近训练快捷选择；P5 历史详情"查看报告/查看审计结果"改为页内中英文表格化只读 modal；索引损坏/不可用时诚实回退 JSON 且不隐藏记录）

### I18n

Built-in zh/en translation system at [i18n.py](auto_tune/ui/i18n.py). Templates use `{{ _("key") }}` syntax.

## Scoring Formulas

### Composite Score (Module C — loop.py TuningResult.get_composite_score)

**Quick mode** (双指标): `mAP50 × 0.6 + mAP50-95 × 0.4`

**Comprehensive** (四维, default): `mAP50 × 0.35 + mAP50-95 × 0.25 + precision × 0.20 + recall × 0.20`

If precision/recall are missing, falls back to quick mode.

### Dataset Quality Score (Module A)

Weighted from: blur_weight(0.15) + under_exposure_weight(0.15) + over_exposure_weight(0.15) + class_imbalance_weight(0.25) + coverage/bbox quality (configurable in [config.yaml](auto_tune/config.yaml)).

### Best Iteration Selection

The iteration with the highest composite score is tracked in `_compute_best()` in [loop.py](auto_tune/modules/agent_engine/loop.py).

## SSE Threading Pattern

The UI uses a thread-safe queue pattern for real-time training log delivery:

```python
# FastAPI async route + blocking YOLO training in a thread pool
msg_queue = queue.Queue()

def on_progress(iteration, message):
    msg_queue.put(json.dumps({"status": "running", "message": message}))

@app.get("/api/training/stream")
async def event_stream():
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, lambda: run_tuning_loop(config, on_progress=on_progress))
    
    while True:
        try:
            msg = msg_queue.get_nowait()
            yield f"data: {msg}\n\n"
        except queue.Empty:
            pass
        if future.done():
            break
        await asyncio.sleep(0.15)  # Non-blocking poll
```

Key points:
- `queue.Queue` (threading) bridges blocking YOLO subprocess → async SSE
- `asyncio.sleep(0.15)` polls instead of blocking on `queue.get(timeout=...)` to avoid deadlock
- `threading.Event` (`_tuning_cancel_event`) signals cancellation from web requests
- See [app.py](auto_tune/ui/app.py) for the production version

## Test Patterns

Tests use `tmp_path` (pytest fixture) to create synthetic directories, avoiding real YOLO/LLM dependencies:

- **Module A tests** ([test_analyzer.py](auto_tune/tests/test_analyzer.py)): Create synthetic images via numpy + cv2, write .txt label files, call `analyze_dataset()` to verify structure/counts
- **Module B tests** ([test_train_analyzer.py](auto_tune/tests/test_train_analyzer.py)): Create synthetic `results.csv` with epoch metric columns + `args.yaml`, call `analyze_training_results()` to verify parsing, run detection, comparisons
- **Module C / A0 闭环测试**（pytest）：`test_guardrails.py`、`test_decision_agent.py`、`test_executor.py`（预检/命令复用）、`test_audit.py`（原子审计/脱敏/fatal 策略）、`test_tuning_loop.py`（Loop 编排，mock finalizer）、`test_experiment_history.py`（统一历史/legacy 兼容/损坏保护）、`test_training_finalizer.py`（统一收尾/KPI/tuning_context）、`test_ui_training_results.py`（SSE 收尾事件/历史页模板/导出与审计路由）、`test_upload_security.py`
- **S1.2–S2 新增测试组**：`test_dataset_snapshot*.py`（快照物化/门禁）、`test_credentials.py`/`test_endpoint_policy.py`/`test_ai_settings_api.py`（凭据安全）、`test_input_safety*.py`（目录预检）、`test_run_state*.py`/`test_run_manager_reconnect.py`（运行状态与重连）、`test_local_index_*.py`（S2 Core：database/repository/import/integration/api/ui）
- **P 系列测试组**：`test_reference_dataset*.py`/`test_loop_bugfix_p2.py`（P2 快照绑定）、`test_recent_training_runs*.py`（P3）、`test_experiment_presentation.py`/`test_experiment_i18n_ui.py`（P4 展示词汇，含 `test_presentation_module_has_no_framework_dependency` 纯模块校验）、`test_experiment_report_view.py`/`test_experiment_audit_view.py`/`test_experiment_views_api.py`/`test_experiment_views_ui.py`（P5 报告/审计视图：畸形字段 409、安全投影、中英文 modal、textContent/XSS）
- **H1.3 Studio 接入测试组**：`test_hpo_api.py`（投影/快照选择/字段级错误）、`test_hpo_ui.py`/`test_hpo_ui_lifecycle.py`（页面与生命周期）、`test_hpo_ui_rework.py`（布局/选择代号/异步乱序）、`test_hpo_formal_training.py`（正式训练提交、来源 metadata、关联结果与身份解析）、`test_hpo_artifacts.py`（受控产物路径）、`test_hpo_best_reuse.py`/`test_hpo_studio_concurrency.py`（复用与并发门禁）
- **Module C CLI**：[test_modelc.py](test_modelc.py) at repo root is a standalone script (not pytest), with `--stage N` for component isolation

常见模式：mock `finalize_training_run` / `launch_training` / `monitor_training` 以隔离训练进程；模板测试直接渲染 `_jinja_env.get_template("single_page.html")` 断言 HTML 片段；真实调优验证复用 `log/tuning_audit_{session}.json` 产物。

前端行为测试不要只断言源码字符串：`tests/js/minidom.js` 解析**真实的** `single_page.html` 建立最小 DOM，用 `vm` 执行真实 `hpo.js`，`fetch` 只入队不发起 I/O，由 `api.respond/settle/drain` 决定响应**顺序**（这是乱序/迟到响应可测的原因）。新增行为：在 `SCENARIOS` 加一个 `async (api) => ({...facts})` 场景，再在 `test_hpo_ui_behaviour.py` 里断言事实（按钮 disabled、传入的 ID、state、计时器数），不是断言按钮文案存在。

## Config Sections ([config.yaml](auto_tune/config.yaml))

| Section | Purpose |
| ------- | ------- |
| `project` | Display metadata (name, description, detection_target) |
| `llm` | DeepSeek API config (endpoint, model, temperature) |
| `vision` | Qwen-VL API config for confusion matrix + error crop analysis |
| `guardrails` | Parameter validation mode (strict/lenient), custom rules dir |
| `probe` | Early-epoch monitoring: probe_epochs(10), auto_continue_threshold(0.05 mAP50), max_retries(3) |
| `dataset_analyzer` | Thresholds for blur, exposure, bbox size, DBSCAN, quality weights |
| `train_analyzer` | Plateau detection, overfit threshold, min_acceptable_map, stale_threshold |
| `training` | Default YOLO params (epochs, batch, imgsz, workers, model) |
| `local_index` | S2 Core SQLite 索引（database_path: log/auto_tune.db, backup_dir: log/db_backups, backup_max_files: 3, busy_timeout_ms: 5000） |

## Report JSON Structure

All reports stored in `log/`:

- **Module A**: `dataset_report_ds_{timestamp}.json` — image_quality, label_coverage, bbox_stats, class_distribution, cluster, quality_score
- **Module B**: `train_{name}_{timestamp}_report.json` — per-run metrics, issues, llm_analysis (per run), vision_analysis (per run), comparison, summary
- **Module C**: `tuning_history.json` — array of iterations, each with perception + decision + guard_results + merged_params + result metrics

## Key Contracts & Compatibility Boundaries

这些是当前版本已形成的兼容边界，改动时必须保持（详见 `docs/development_handoff_20260814.md`）：

- **两个历史职责分离**：`tuning_history.json` 服务 LLM 反馈；`experiment_history.json` 服务统一 UI 历史。不要直接合并二者。
- **before 指标必须来自与 `baseline.params` 相同的 `reference_run`**，直接读 `detect/<ref>/results.csv` 最终轮指标（`metrics/mAP50(B)`→`mAP50` 等映射）；不得用 Module B 全局最佳摘要替代。
- **训练成功 + 分析失败 = 部分成功**：UI 与 API 不得把训练状态改成 failed，只降级 analysis_status。
- **审计持久化失败 = fatal**：任何替代实现都不得绕过该失败策略继续启动训练（决策/护栏/预检/审计失败均不启动训练）。
- **缺失指标显示 `—`，真实零值显示 `0.0000`**：不能用真值判断混淆二者。
- **统一历史向前兼容**：后续加字段须保持旧记录可读；损坏历史抛 `ExperimentHistoryError` 且不静默覆盖。
- **命令只构造一次**：审计记录的 `execution.command` 与实际执行命令必须一致（`list[str]`）。
- **run_id 格式（JSON 历史）**：手动 `manual:{run_name}`；调优 `tuning:{session_id}:{run_name}`。
- **S1.5 运行身份（SQLite experiments 主键）**：`manual:<uuid4>` / `tuning:<uuid4>`；原 JSON run_id 保留在 `params._legacy_record_run_id`。
- **三种 run_id 不得混用**：JSON 历史 ID（`manual:{run_name}`）、S1.5 运行身份（`manual:<uuid4>`）、实验索引详情身份（SQLite 主键）是三件事。`hpo_source.json` 的 `runtime_run_id` 是一次**声明**：字段存在（即使值非法）即身份，只允许精确命中索引中同 `run_name` **且**同受控 `detect/trainN` 目录的那条记录，失配返回 `EXPERIMENT_NOT_INDEXED`，**绝不改绑**到同名实验；只有字段真正缺失/为空才允许按 run_name 回退，且必须唯一（0 条 `EXPERIMENT_NOT_INDEXED`／多条 `EXPERIMENT_AMBIGUOUS`）。解析全程只读。
- **监控入口与结果入口身份不同**：“查看监控”只接受 `RUNTIME_RUN_ID_RE` 命中的 `<kind>:<uuid4>`（交给 `/api/runs/{run_id}/stream`；metadata 缺失/非法时才回退到**活动**控制器的 `run_state.run_id`，且同样要求格式合法，绝不补造）；“查看结果”只接受实验索引详情身份。JSON 历史 ID 两者都不得使用，身份不可用只禁用对应入口并显示稳定原因码。
- **SQLite 只作可重建查询索引**：JSON 审计/报告/运行状态仍是事实来源；SQLite 故障不得改变训练/调优真实终态，损坏时诚实回退 JSON 且不隐藏记录。
- **统一历史 KPI**：`epochs={configured,completed,best}`；审计与历史 KPI 必须一致。
- **敏感字段递归脱敏**：api_key/apikey/authorization/token/secret/password 等不得进入审计/历史。
- **文本模型默认名统一为 `deepseek-flash`**（`config.template.yaml`、`decision_agent.call_decision_llm`、`llm_analyzer` 的 payload 默认与 `model_used`、`ui/app.py` 的 `_default_model_for("text")`）；显式配置的 `llm.model` 始终优先，config.yaml 本地值可不同但不得回写死名。
- **JSON 输出约束按路径区分**：`call_decision_llm(prompt, config, json_mode=False)` 只在**结构化 JSON 路径**传 `json_mode=True`（`generate_suggestion` 的首次与一次 JSON 纠错重试、TuningDecision v1 的首次与一次纠正重试），payload 才带 `response_format={"type":"json_object"}`；**终局摘要等纯文本调用必须保持默认 False**（该提示词明确要求"不要输出 JSON"）。两条 JSON 路径的提示词都必须含 `JSON` 字样（DeepSeek `json_object` 模式的前置要求）。新增 `call_decision_llm` 调用方时按其输出类型选择，不要无条件开启。

## P4/P5 展示层与只读视图契约

### 共享展示词汇（P4，`modules/presentation/experiment_labels.py`）

- 字段/枚举/布尔键只映射**一次**到稳定英文翻译键，通过调用方传入的翻译器（`make_translator`）本地化；zh 文本在 `ui/i18n.py`，`en` 为恒等可省略。
- `experiment_labels.py` 是**纯模块（无 import、无框架依赖）**——UI 与任何报告生成器复用同一接口；`build_experiment_labels(_)` 注入 `window._EXPERIMENT_LABELS`，前端经 `_experimentFieldLabel/_experimentEnumLabel/_experimentBooleanLabel` 读取。
- 新增展示标签：把字段键加进 `FIELD_LABEL_KEYS`（或枚举值加进 `ENUM_LABEL_KEYS`）+ 在 i18n zh 补译文；**不要另建平行词典**。真实值（run_id/dataset_id/snapshot_id/run_name/模型名/指标名/错误码）永不作翻译键，原样透传；`None` 显示 `—`。

### 报告/审计只读视图（P5，`modules/presentation/experiment_views.py`）

- 两个窄接口，仅接受 run_id 路径参数：`GET /api/experiments/{run_id}/report-view`、`GET /api/experiments/{run_id}/audit-view`（app.py 调 `LocalIndexService.get_report_view/get_audit_view`，再委托 view builder）。
- **run_id 是唯一身份**：只读该 run 已登记 `kind == report`/`audit` artifact；报告内容必须引用实验 `run_name`（`runs` 键 / run 条目 `name` / `summary.best_overall_run` / `comparison.best_run`）；审计 `session_id` 必须与文件名 `tuning_audit_{session_id}.json` 对账。身份冲突 → `ARTIFACT_IDENTITY_MISMATCH`。
- **有界、只读、安全**：1 MiB 分块 / 16 MiB 上限 / 读取期间增长同样受限；不用 glob、不接受客户端路径、拒绝 symlink/reparse、限制在受控产物根（`_artifact_root()` = fact-log 目录，**不要套用数据集 allowed_roots**）；只返回最小展示字段，参数路径值降级 basename，不返回命令/日志/凭据/traceback，不调用 LLM/Vision，不写文件。
- **稳定错误码**：`EXPERIMENT_NOT_FOUND→404`、`REPORT_NOT_AVAILABLE/AUDIT_NOT_AVAILABLE→404`、`REPORT_INVALID/AUDIT_INVALID/ARTIFACT_IDENTITY_MISMATCH→409`、`ARTIFACT_TOO_LARGE→413`、`ARTIFACT_UNAVAILABLE→503`、`LOCAL_INDEX_*→沿用现有映射`。
- **畸形审计容错**：7 个对象字段（hyperparameter_changes/training_overrides/sanitized_changes/actual_params/clamped/after_metrics/metric_delta）非对象时稳定返回 `AUDIT_INVALID/409`（不泄漏原生 ValueError/TypeError）；`termination_reason` 只展示稳定 `error_type/error_code`（无则固定安全文案，绝不返回原始 message）；`guardrails.warnings` 必须是数组并做安全投影（traceback→`…`、凭据赋值→`key=REDACTED`、绝对路径→`<path>`）。
- **训练问题字段映射**：生产报告 issues 用 `type/severity/detail`，投影为 `issue/severity/description`；旧格式 `issue/description` 回退。结构化类型（overfitting/plateau/unstable_training）与严重程度（low/medium/high）经共享枚举随语言切换，detail 原文保留不翻译。
- 历史 AI 诊断/视觉分析原文保持生成时语言，切换界面语言不重新调用模型；AI/视觉区标注"历史原始分析 / Original stored analysis"。

## Known Issues / Pending

- 训练日志已分层（S1.1）：默认仅显示 epoch/验证/生命周期/警告摘要，完整日志折叠查看，错误堆栈不丢失
- 数据集划分已改为 S1.2 不可变快照；Ultralytics 会在快照 `labels` 目录生成 `train.cache`/`val.cache`，manifest 图片与标签不变，后续评估迁移缓存到运行目录
- API 凭据已迁移到 Windows Credential Manager/环境变量（S1.3）；不得重新写回 YAML，Linux/Docker Secret 文件仍属后续跨平台小批次
- 上传/分析目录已接入 S1.4 `input_safety` 预检；ZIP 不作为主流程，安全提取未恢复
- Tuning status uses a JSON status file (`tuning_running.json`) written/cleaned by the tuning endpoint（与 S1.5 run_state 统一运行状态并存）
- S2 Core 只实现 SQLite 基础索引；数据集重命名、标签、批量操作、归档、高级搜索、统计图表、导入导出、复杂对比不开发，仅保留 Repository/Service/API 扩展边界
- 当前路线（2026-09-18 修订，见 `docs/roadmap_20260814.md`/`docs/implementation_plan_20260814.md`）：**Q1、H1 与 F1.1 均已完成验收**。F1.1-A 完成体验优化与受控权重库，F1.1-B 完成 LLM 效果专项，F1.1-C 完成稳定冻结审查与必要最小修复；最终完整自动化为 **2738 passed / 2 warnings**。F1.1 已关闭前端干运行入口、原参考总体最佳展示与目标绑定、动态运行名转义等 P1，未做结构重写，未改已验收协议。下一入口为 **F1.2 Windows、单 Docker 镜像与 API 交付适配**；不设置 H1.4。**R1 轻量 Research** 与 **Cloud（CL1–CL6）** 暂不排期、不自动启动。用户可在直接训练、HPO 和 LLM 三种独立策略中选择，不串联决策；YOLOv5 Detect、Segment、OBB、Pose、YOLO11/26 不在当前计划。
- **H1.1（已通过 Codex 验收，2026-09-08；本批不含真实训练/UI/API/LLM/产品排名）**：`auto_tune/modules/hpo/` 严格契约（`extra='forbid'`、拒绝 bool/str/NumPy/NaN/Inf、JSON 唯一事实源、版本化 study/trial）、条件搜索空间（SGD momentum_sgd / AdamW beta1_adamw 独立采样键 → 统一 momentum；lrf 上限 0.1；epochs=1 时 warmup=0）、Optuna 4.5.0 TPE/RandomSampler（每次按历史重建内存 study，不落 RDB/pickle）、原子 JSON + 锁（同目录临时文件 flush/fsync/os.replace；进程内共享 Lock + msvcrt/fcntl 非阻塞 OS 锁，忙返回 `HPO_STUDY_BUSY`）、候选写盘成功才返回、同 request_id 幂等、失败 trial 消耗预算不补偿、历史损坏/版本不匹配/输入绑定改变拒绝继续。验收后加固：新增 `validation.py`（validate_trial/validate_history/validate_record：读/写/重建同一深度重校验，含候选↔sampled 映射、条件分布 JSON 字符串键序/有限性/属性集合、state↔result 不变量、history 预算/去重/连续性）与 `storage.reject_link_chain`（storage 根/study/锁/study.json/model 路径拒绝 symlink 与 reparse、`..`；写前不覆盖已损坏 study.json），新增 `test_hpo_recovery_validation.py`。H1.1 测试共 6 个 `test_hpo_*.py`（188 tests）；依赖新增 optuna 4.5.0、alembic 1.19.2、SQLAlchemy 2.0.52、colorlog 6.12.0、greenlet 3.5.5（`requirements.txt` 仅追加五行）。实现与验收规格见 `历史文档/已验收批次_截至20260916/superpowers/specs/2026-09-07-h1-1-hpo-foundation-design.md`、`历史文档/已验收批次_截至20260916/superpowers/plans/2026-09-07-h1-1-hpo-foundation.md`
- **H1.2（已通过 Codex 验收，2026-09-09；全量本地 1837 passed / 2 PCA warnings / 0 skipped，97.89 秒）**：`modules/hpo/` 新增 execution_models/execution_storage/execution_adapter/execution/metrics/ranking；`HpoRunner.prepare/run/resume/status` + `rank_trials` + `extract_objective`；execution.json（hpo-execution-v1）原子审计 + `.hpo-runner.lock` 全局执行锁；executor.py 仅追加 task/amp 白名单；`ExecutionAttempt` 新增必填 `command_executable`（与 command 分离并等于 command[0]；历史命令重建以此为据、不解析当前 YOLO，新启动才解析当前环境比对；缺该字段的旧 execution-v1 记录拒读、不补齐/不迁移/不覆盖，属未发布格式收紧，后续发布版本不得以此先例静默破坏已发布格式）；新增 `auto_tune/scripts/verify_hpo_execution.py`（预算完整 + 真实验收 trial 全成功 + 有效排名才返回 0）。四轮返修（R1–R7、R2a-1/R2a-2/R2a-3、R2b/R2c）后独立反例累计 **21 passed**；**4 次真实短训练**（TPE/Random×budget=2、epochs=1、batch=1、imgsz=64、device=0、timeout_seconds=120）均 SUCCESS/FINALIZED、CLI 退出 0；**未提交、未推送**。验收不含 UI/API 接线、真实 SGD、长训练性能与真实训练中全部故障注入。规格/计划/验收见 `历史文档/已验收批次_截至20260916/superpowers/specs/2026-09-08-h1-2-hpo-execution-design.md`、`历史文档/已验收批次_截至20260916/superpowers/plans/2026-09-08-h1-2-hpo-execution.md`、`历史文档/已验收批次_截至20260916/reviews/h1_2_codex_review_20260908.md`
- **H1.3 Studio 接入（已通过验收，2026-09-16）**：`ui/hpo_training.py`（严格 `FormalTrainingConfig`/`TrainBestRequest`、`resolve_hpo_formal_training` 只采纳 epochs 且 batch/imgsz/device 必须与权威 execution 逐项相等、候选在新 epochs 下重校验不裁剪、`submit_formal_training` 复用 `ManualRunController`/共享门禁/原子写盘且失败零进程零新目录、受控产物与关联结果投影）、`ui/hpo_api.py`（字段级安全错误投影 `INVALID_HPO_FIELD`、快照/本地模型投影）、`ui/app.py`（第二个 `/api/hpo` router，service/runner/manager 以 accessor 传参）、`ui/hpo_controller.py`、`ui/hpo_reuse.py`、`ui/static/hpo.js` + `single_page.html`（四模式统一布局、HPO 专属元素 `hpo-mode`/`non-hpo-only`、选择代号丢弃乱序响应、202 后经 `refreshRound` 自动轮询、公共监控所有权守卫、设备选择器）、`ui/static/monitor.js` 的 `epochsValue`（终态 epoch KPI 是 `{configured,completed,best}` 结构，不是标量）。返修节点：09-14 主体 → 09-15 结构化 epochs 投影 → 09-15 最终浏览器阻断项 → 09-15 第四轮（202 后自动轮询）→ 09-16 第五轮（正式训练恢复 YOLO 静态绘图：`FORMAL_TRAINING_OVERRIDES = {"plots": True}` 只作用于 `/train-best`，搜索阶段仍 `plots=False`；同条件验证路径 `ui/hpo_reuse.py` 未改）。验收前本地完整套件 **2426 passed / 2 PCA warnings / 0 skipped**；**遗留的浏览器体验问题转 F1 第一轮产品打磨统一处理**
- 报告/审计只读视图（P5）不生成 PDF/DOCX/Excel，不新增下载系统，不写新报告文件；历史详情已无裸 JSON 链接，训练监控页/智能分析页既有的 `report-by-name` 用法（非历史详情）保留
- `auto_tune/modules/training_diagnosis/` is a stale empty directory — do not reference it

> 当前架构为单用户/单机/JSON 文件存储/local subprocess；不要把它描述成已支持多用户或训练队列。GitHub 仓库为公开仓库，任何本地数据/标签/权重/训练产物/日志/审计/真实凭据都不得提交。
