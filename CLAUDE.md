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
- **当前已发布稳定版本为 v0.2**；Studio S1.2 与 S1.3 已完成独立验收，发布状态与下一批范围以交接记录和艾卡的新批准为准。每次只选择一个经艾卡批准的小批次，不要同时展开多个方向。
- 每次测试完后杀死测试用的服务器进程，让用户自己开启服务器自己测试。

YOLOv8 Auto-Tuning Agent — a three-module closed-loop system for automated YOLOv8 training optimization:

- **Module A (Dataset Analyzer)**: Analyzes dataset quality (blur, exposure, SNR, bbox geometry, class balance, clustering)
- **Module B (Train Analyzer)**: Three-stage training diagnosis pipeline (Python metrics → LLM text diagnosis → Qwen-VL vision analysis)
- **Module C (Agent Engine)**: Perception → Decision (LLM) → Guardrails → Execute → Probe Monitor auto-tuning loop

> **当前已发布版本 v0.2**（含 Studio S1.1 训练日志分层）。S1.2 不可变数据集快照与 S1.3 API 凭据安全已验收，是否发布以交接记录为准。已验证环境：Windows、Python 3.10、Ultralytics YOLOv8 detection；Linux、YOLO11/26、其他视觉任务与模型结构自调整仍在后续计划中。

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

# Official full-suite baseline (S1.2+S1.3 acceptance: 396 passed, 2 sklearn PCA warnings)
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
```

## Architecture

### Directory Layout

```
auto_tune/
├── main.py                          # Unified entry point (UI / dry-run / train)
├── config.yaml                      # Global config (LLM keys, thresholds, params)
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
│   └── agent_engine/                # Module C
│       ├── loop.py                  #   Main tuning loop orchestrator
│       ├── perception.py            #   Aggregate Module A + B reports
│       ├── decision_agent.py        #   LLM-driven hyperparameter suggestions
│       ├── parameter_registry.py    #   Unified hyperparameter registry (whitelist/types/bounds)
│       ├── guardrails.py            #   Validate & clamp hyperparameter changes
│       ├── executor.py              #   Preflight + launch/manage YOLO training subprocess
│       ├── audit.py                 #   TuningAuditSession: atomic audit JSON + redaction
│       └── probe_monitor.py         #   Early-epoch monitoring (continue/abort/retry)
├── ui/
│   ├── app.py                       # FastAPI server with SSE streaming
│   ├── i18n.py                      # zh/en translations
│   ├── templates/
│   │   └── single_page.html         # SPA (all-in-one HTML+CSS+JS)
│   └── components/
│       ├── dataset_panel.py         # Module A display helpers
│       ├── train_panel.py           # Module B display helpers
│       ├── tuning_panel.py          # Module C display helpers
│       └── experiment_panel.py      # Unified history display adapter (adds audit_filename)
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
│   └── test_run_comparator.py
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

### Key Config ([config.yaml](auto_tune/config.yaml))

Configured sections: `project`, `llm` (DeepSeek), `vision` (Qwen-VL), `guardrails`, `probe`, `dataset_analyzer`, `train_analyzer`, `training`. API keys are kept in config.yaml (not env vars).

### UI Pages (SPA)

Single-page app at [single_page.html](auto_tune/ui/templates/single_page.html) with 5 tabs:
1. **Projects** — Edit project info, view config
2. **Dataset Report** — Upload dataset, run Module A
3. **Intelligent Analysis** — Module B results + auto-tuning controls
4. **Training Monitor** — Real-time SSE training output
5. **History** — Past tuning iterations

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
- **Module C CLI**：[test_modelc.py](test_modelc.py) at repo root is a standalone script (not pytest), with `--stage N` for component isolation

常见模式：mock `finalize_training_run` / `launch_training` / `monitor_training` 以隔离训练进程；模板测试直接渲染 `_jinja_env.get_template("single_page.html")` 断言 HTML 片段；真实调优验证复用 `log/tuning_audit_{session}.json` 产物。

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
- **run_id 格式**：手动 `manual:{run_name}`；调优 `tuning:{session_id}:{run_name}`。
- **统一历史 KPI**：`epochs={configured,completed,best}`；审计与历史 KPI 必须一致。
- **敏感字段递归脱敏**：api_key/apikey/authorization/token/secret/password 等不得进入审计/历史。

## Known Issues / Pending (A0.2 backlog)

- 训练日志默认仍含原始 YOLO stdout，前端过滤展示——计划改为默认仅显示 epoch 关键指标、完整日志折叠查看
- 数据集划分当前移动原始文件——计划改为不可变快照
- API 凭据已迁移到 Windows Credential Manager/环境变量；不得重新写回 YAML，Linux/Docker Secret 文件仍属后续跨平台小批次
- 上传暂无总量/成员数/解压后容量限制——计划补充（含目录选择校验）
- Tuning status uses a JSON status file (`tuning_running.json`) written/cleaned by the tuning endpoint
- `auto_tune/modules/training_diagnosis/` is a stale empty directory — do not reference it

> 当前架构为单用户/单机/JSON 文件存储/local subprocess；不要把它描述成已支持多用户或训练队列。GitHub 仓库为公开仓库，任何本地数据/标签/权重/训练产物/日志/审计/真实凭据都不得提交。
