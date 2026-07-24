# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

请用中文回答用户。

## Project Overview

YOLOv8 Auto-Tuning Agent — a three-module closed-loop system for automated YOLOv8 training optimization:

- **Module A (Dataset Analyzer)**: Analyzes dataset quality (blur, exposure, SNR, bbox geometry, class balance, clustering)
- **Module B (Train Analyzer)**: Three-stage training diagnosis pipeline (Python metrics → LLM text diagnosis → Qwen-VL vision analysis)
- **Module C (Agent Engine)**: Perception → Decision (LLM) → Guardrails → Execute → Probe Monitor auto-tuning loop

## Commands

> **重要**: 每次测试完后，杀死测试用的服务器进程。让用户自己开启服务器自己测试。

```powershell
# Activate environment
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

# Run all tests
python -m pytest auto_tune\tests -v

# Run single test
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
│   │   └── crop_utils.py            #   Generate error crop images for vision
│   └── agent_engine/                # Module C
│       ├── loop.py                  #   Main tuning loop orchestrator
│       ├── perception.py            #   Aggregate Module A + B reports
│       ├── decision_agent.py        #   LLM-driven hyperparameter suggestions
│       ├── guardrails.py            #   Validate & clamp hyperparameter changes
│       ├── executor.py              #   Launch/manage YOLO training subprocess
│       └── probe_monitor.py         #   Early-epoch monitoring (continue/abort/retry)
├── ui/
│   ├── app.py                       # FastAPI server with SSE streaming
│   ├── i18n.py                      # zh/en translations
│   ├── templates/
│   │   └── single_page.html         # SPA (all-in-one HTML+CSS+JS)
│   └── components/
│       ├── dataset_panel.py         # Module A display helpers
│       ├── train_panel.py           # Module B display helpers
│       └── tuning_panel.py          # Module C display helpers
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
- `tuning_history.json` — Module C history
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
- **Module C**: [test_modelc.py](test_modelc.py) at repo root is a standalone script (not pytest), with `--stage N` for component isolation

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

## Known Issues (from memory)

- Dataset split (train/val) is done during upload; the ratio is configured per-dataset
- SSE streaming for first-time training sends raw YOLO stdout — frontend filters it to show only epoch metrics
- Tuning status uses a JSON status file (`tuning_running.json`) written/cleaned by the tuning endpoint
- `auto_tune/modules/training_diagnosis/` is a stale empty directory — do not reference it
