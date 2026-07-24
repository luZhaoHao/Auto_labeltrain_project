# YOLOv8 Auto-Tuning Agent — Design Spec

## 1. Overview

A multi-modal agent system for industrial defect detection that automatically analyzes image datasets, diagnoses YOLOv8 training results, and autonomously modifies hyperparameters in a closed tuning loop. Built as a local web application (Gradio + FastAPI).

**Goal:** Minimize human involvement in YOLOv8 hyperparameter tuning for defect detection while maintaining expert-level diagnostic quality.

**Target hardware:** Local GPU, 8–32 GB VRAM.

---

## 2. Architecture

### 2.1 Three Independent Modules

```
┌─────────────────────┐     ┌─────────────────────┐
│  Module A           │     │  Module B           │
│  Dataset Analyzer   │     │  Training Diagnosis │
│  —————————————————  │     │  —————————————————  │
│  Traditional CV +   │     │  CSV Parser →       │
│  Statistical        │     │  Text LLM →         │
│  Analysis → JSON    │     │  MLLM (on demand)   │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └──────────┬────────────────┘
                      ▼
           ┌──────────────────────┐
           │  Module C            │
           │  Agent Decision &    │
           │  Execution Engine    │
           │  —————————————————   │
           │  LLM Decision →      │
           │  Guardrails →        │
           │  Subprocess YOLOv8   │
           └──────────────────────┘
```

**Key rule:** Module A and B are completely independent. They produce JSON reports consumed by Module C. No direct cross-coupling.

### 2.2 UI Layer

A separate rendering layer reads the same JSON outputs for human display. It never modifies the JSON.

- **Framework:** Gradio blocks inside FastAPI mount
- **Pages:** Projects → Dataset Report → Agent Suggestion → Training Monitor → History
- **Design system:** Data-Dense Dashboard style, Primary #1E40AF, Fira Sans + Fira Code

---

## 3. Module A — Dataset Analyzer

### 3.1 Input
- Image directory path
- Label directory path (YOLO format .txt)
- Data YAML (class names, paths)
- Config: threshold values, cluster parameters

### 3.2 Analysis Pipeline

| Step | Method | Output |
|------|--------|--------|
| Image quality | Laplacian variance (blur), grayscale histogram (exposure), SNR | Blur ratio, over/under-exposure ratio, noise level |
| Bounding box geometry | Width/height distribution, aspect ratio, area statistics | BBox size categories (small/medium/large%), aspect ratio range |
| Overlap analysis | IoU matrix between bboxes (intra-class & inter-class) | High-overlap ratio, cluster detection suggestions |
| Position heatmap | Normalized (x_center, y_center) distribution | Spatial bias score (center vs edge concentration) |
| Class distribution | Per-class instance count | Class balance ratio, long-tail categories |
| Feature clustering | ResNet18 embeddings → PCA → DBSCAN | Outlier sample count, silhouette score |

### 3.3 Output JSON

```json
{
  "module": "dataset_analyzer",
  "version": "1.0",
  "analysis_timestamp": "2026-07-20T12:00:00Z",
  "dataset_path": "/data/train",
  "total_images": 5000,
  "total_annotations": 15000,
  "class_distribution": {
    "crack": {"count": 8000, "ratio": 0.53},
    "scratch": {"count": 5000, "ratio": 0.33},
    "dent": {"count": 2000, "ratio": 0.13}
  },
  "class_balance": {
    "is_balanced": false,
    "long_tail_classes": ["dent"],
    "imbalance_ratio": 4.0
  },
  "image_quality": {
    "blur_ratio": 0.03,
    "overexposure_ratio": 0.01,
    "underexposure_ratio": 0.05,
    "low_snr_ratio": 0.02
  },
  "bbox_analysis": {
    "tiny_bbox_ratio": 0.25,
    "small_bbox_ratio": 0.40,
    "medium_bbox_ratio": 0.25,
    "large_bbox_ratio": 0.10,
    "aspect_ratio_range": [0.5, 2.0]
  },
  "spatial_bias": {
    "center_concentration_score": 0.7,
    "edge_distribution_ratio": 0.1
  },
  "overlap_analysis": {
    "high_iou_ratio": 0.15,
    "severe_overlap_classes": ["crack"]
  },
  "outlier_analysis": {
    "outlier_count": 50,
    "outlier_ratio": 0.01,
    "silhouette_score": 0.65
  },
  "summary": {
    "dataset_quality_score": 0.82,
    "key_issues": [
      "long_tail_class_dent",
      "tiny_bbox_high_ratio",
      "center_spatial_bias"
    ]
  }
}
```

---

## 4. Module B — Training Diagnosis

### 4.1 Input
- `results.csv` from YOLOv8 training
- `args.yaml` training configuration
- (Optional) `confusion_matrix_normalized.png`
- (Optional) Cropped misclassification patches (IoU < 0.3)

### 4.2 Three-Stage Diagnosis

**Stage 1 — Python Parser (0 tokens):**

| Extraction | Method |
|-----------|--------|
| Loss trends (last 20 epochs) | Parse CSV, compute slope |
| mAP50 / mAP50-95 curve | Parse CSV, detect plateau/decline |
| Overfitting flag | val_loss rising > 5 epochs while train_loss falling |
| Gradient explosion flag | NaN values or loss > 50x initial |
| Best epoch & early stop status | CSV argmin/max + patience check |
| Miss rate (漏检率) | Derived from confusion: FN / (TP + FN) |
| False alarm rate (误剔率) | Derived from confusion: FP / (FP + TP) |

Output: structured JSON with numeric indicators.

**Stage 2 — Text LLM (~200 tokens):**

Input: Stage 1 JSON + `args.yaml` as text. Output: diagnosis + recommended hyperparameter changes in structured JSON.

Triggers for Stage 3:
- Any class has mAP50 < 0.1
- Miss rate or false alarm rate > 30%
- Confusion pattern unclear from text data

**Stage 3 — MLLM Vision (conditional, ~500–1000 tokens):**

Input: `confusion_matrix_normalized.png` + cropped misclassification patches (224×224 crops from val_batch label/pred pairs where IoU < 0.3). Output: fine-grained visual diagnosis.

### 4.3 Key Metrics

| Metric | Definition | Formula |
|--------|-----------|---------|
| miss_rate (漏检率) | Proportion of undetected targets | FN / (TP + FN) |
| false_alarm_rate (误剔率) | Proportion of false detections | FP / (FP + TP) |

### 4.4 Output JSON

```json
{
  "module": "training_diagnosis",
  "version": "1.0",
  "training_summary": {
    "epochs_completed": 100,
    "best_epoch": 85,
    "best_mAP50": 0.82,
    "best_mAP50_95": 0.55,
    "miss_rate": 0.12,
    "false_alarm_rate": 0.08
  },
  "overfitting": {
    "detected": true,
    "start_epoch": 70,
    "val_box_loss_trend": "rising",
    "val_cls_loss_trend": "stable",
    "severity": "moderate"
  },
  "gradient_issues": {
    "nan_occurred": false,
    "loss_spikes": false
  },
  "class_performance": {
    "crack": {"mAP50": 0.85, "miss_rate": 0.08, "false_alarm_rate": 0.05},
    "scratch": {"mAP50": 0.78, "miss_rate": 0.15, "false_alarm_rate": 0.10},
    "dent": {"mAP50": 0.45, "miss_rate": 0.35, "false_alarm_rate": 0.20}
  },
  "diagnosis_stage": "text_llm",
  "summary": {
    "status": "overfitting_detected",
    "worst_class": "dent",
    "recommend_action": "retune"
  }
}
```

---

## 5. Module C — Agent Decision & Execution Engine

### 5.1 Architecture

```
┌─────────────────────────────────────────────┐
│              Decision Agent                  │
│  (LLM: Claude API / GPT-4o, fallback: tree) │
│  Input: Module A JSON + Module B JSON        │
│  Output: Structured hyperparameter changes   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│              Guardrails Layer                │
│  Plugin-based rule system                    │
│  Schema validation + boundary checks         │
│  Coupling conflict detection                │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│              Executor                        │
│  Modify args.yaml → Launch YOLOv8           │
│  subprocess → Monitor epochs →               │
│  Kill/Cycle on probe failure                │
└─────────────────────────────────────────────┘
```

### 5.2 LLM Decision Prompt

The LLM receives:
- Dataset analysis JSON (Module A)
- Training diagnosis JSON (Module B)
- Current hyperparameter config
- Task description + constraint rules (as system prompt)

Output format (enforced via structured output):

```json
{
  "diagnosis": "Model shows severe overfitting on dent class with high miss rate. Dataset has long-tail distribution and tiny bboxes.",
  "action": "Increase regularization, boost cls loss for rare class, enable mosaic for generalization",
  "hyperparameter_changes": {
    "lr0": 0.005,
    "box": 10.0,
    "cls": 2.0,
    "mosaic": 1.0,
    "weight_decay": 0.001
  },
  "reasoning": "Overfitting detected from epoch 70. dent class mAP50=0.45 indicates vanishing gradient for rare class. Increasing cls loss to 2.0 to focus on rare class. Enabling mosaic to augment tiny bboxes."
}
```

### 5.3 Guardrails System

Plugin-based architecture: each rule is an independent `.py` file, auto-discovered by a registry.

**Built-in rules:**

| Rule | Logic | Priority |
|------|-------|----------|
| lr0_boundary | Clamp lr0 to [1e-5, 0.1] | CRITICAL |
| lrf_boundary | Clamp lrf to [1e-6, 0.1] | CRITICAL |
| lr_batch_scale | lr0 ∝ batch_size scaling factor | HIGH |
| augmentation_dataset_size | Reduce aug strength when dataset < 1000 images | HIGH |
| dropout_weight_decay_conflict | Warn when both > 0.1 | HIGH |
| cls_balance | Boost cls when long-tail detected | MEDIUM |
| box_tiny_bbox | Boost box weight when tiny_bbox_ratio > 0.2 | MEDIUM |
| batch_auto_on_oom | Force batch=-1 when VRAM constrained | CRITICAL |
| mosaic_off_for_small_data | Disable mosaic when < 300 images | MEDIUM |

**Rule interface:**

```python
class GuardrailRule(ABC):
    @abstractmethod
    def check(self, changes: dict, context: dict) -> RuleResult: ...
```

**Behavior on violation:**
- CRITICAL: clamp/override the value, log warning
- HIGH: clamp value, include explanation in JSON
- MEDIUM: log warning, pass through

### 5.4 Decision Tree Fallback

When LLM is unavailable (no API key, network failure, local mode):

A rule-based decision tree using Module A and B JSON fields:
- If `overfitting.detected` → increase `weight_decay`, add `mosaic`
- If `bbox_analysis.tiny_bbox_ratio > 0.2` → increase `box`, increase `imgsz`
- If `class_balance.long_tail_classes` not empty → increase `cls` for rare classes
- If `gradient_issues.nan_occurred` → reduce `lr0`, increase `warmup_epochs`

### 5.5 Executor

Executes via subprocess (not Python SDK) for clean kill/abort:

1. Write updated `args.yaml` with new hyperparameters
2. Spawn `yolo train ...` as subprocess
3. Parse stdout for epoch logs
4. After N epochs (probe mode) or on completion, trigger Module B
5. On Module B abort signal → kill subprocess, retune, restart

**Probe mode config:**
```yaml
probe:
  enabled: true
  probe_epochs: 10
  auto_continue_conditions:
    max_box_loss_increase: 0.05
    min_mAP50: 0.01
  max_retries: 3
```

---

## 6. UI (Gradio + FastAPI)

### 6.1 Tech Stack
- **Backend:** FastAPI (Python) — serves API + mounts Gradio
- **Frontend:** Gradio Blocks (multi-page)
- **Design:** Data-Dense Dashboard style (#1E40AF primary, Fira Sans/Fira Code)

### 6.2 Pages

| Page | Content |
|------|---------|
| Projects | Project list, create/select/delete, training config |
| Dataset Report | Module A results: quality metrics, bbox distribution, class balance, feature cluster scatter |
| Agent Suggestion | Module C recommendation: table of changes, reasoning, editable before approve |
| Training Monitor | Real-time loss/mAP curves, console log, probe result banner |
| History | Past runs: expandable rows, compare mode (side-by-side), export |

### 6.3 Interaction Points

- Agent suggestion page: human-in-the-loop confirmation before execution
- Hyperparameter table: editable cells for manual override
- Probe mode toggle: enable/disable per project
- Compare mode: select 2+ history runs for side-by-side metric comparison

---

## 7. Configuration

### 7.1 Config File (`config.yaml`)

```yaml
llm:
  provider: claude  # claude | openai | ollama
  api_key_env: LLM_API_KEY
  model: claude-opus-4-7
  temperature: 0.3
  fallback_to_tree: true

guardrails:
  mode: strict  # strict | warn | off
  custom_rules_dir: guardrails/rules/

probe:
  enabled: true
  probe_epochs: 10
  auto_continue_threshold_mAP50: 0.05
  max_retries: 3

dataset_analyzer:
  blur_threshold: 100.0
  high_iou_threshold: 0.7
  dbscan_eps: 0.3
  dbscan_min_samples: 5

training:
  default_epochs: 100
  patience: 20
  imgsz: 640
```

---

## 8. Project Structure

```
auto_tune/
├── main.py                     # FastAPI + Gradio entry
├── config.yaml                 # Global configuration
├── modules/
│   ├── dataset_analyzer/       # Module A
│   │   ├── __init__.py
│   │   ├── image_quality.py    # Laplacian, histogram, SNR
│   │   ├── bbox_geometry.py    # BBox statistics, overlap, heatmap
│   │   ├── class_stats.py      # Class distribution
│   │   ├── feature_cluster.py  # ResNet18 + PCA + DBSCAN
│   │   └── analyzer.py         # Orchestrator
│   ├── training_diagnosis/     # Module B
│   │   ├── __init__.py
│   │   ├── csv_parser.py       # Stage 1: parse results.csv
│   │   ├── text_diagnosis.py   # Stage 2: text LLM
│   │   ├── vision_diagnosis.py # Stage 3: MLLM (conditional)
│   │   ├── metrics.py          # miss_rate, false_alarm_rate
│   │   └── diagnostician.py    # Orchestrator
│   └── agent_engine/           # Module C
│       ├── __init__.py
│       ├── decision_agent.py   # LLM prompt + structured output
│       ├── decision_tree.py    # Fallback decision tree
│       ├── executor.py         # Subprocess YOLOv8
│       ├── probe.py            # Probe mode logic
│       └── guardrails/
│           ├── registry.py     # Auto-discover rules
│           ├── base.py         # Rule ABC
│           ├── validator.py    # Schema validation
│           └── rules/          # Individual rule plugins
│               ├── lr_boundary.py
│               ├── lr_batch_scale.py
│               ├── aug_dataset_size.py
│               ├── dropout_weight_decay.py
│               ├── cls_balance.py
│               ├── box_tiny_bbox.py
│               ├── batch_auto.py
│               └── mosaic_small_data.py
├── ui/                         # Gradio UI
│   ├── app.py                  # Gradio app definition
│   ├── pages/
│   │   ├── projects.py
│   │   ├── dataset_report.py
│   │   ├── agent_suggestion.py
│   │   ├── training_monitor.py
│   │   └── history.py
│   ├── components/             # Reusable UI components
│   │   ├── kpi_card.py
│   │   ├── badge.py
│   │   ├── hyperparam_table.py
│   │   ├── console_log.py
│   │   └── charts.py
│   └── static/
│       └── style.css
├── utils/
│   ├── yolo_subprocess.py      # YOLO subprocess wrapper
│   ├── file_utils.py
│   └── json_utils.py
└── tests/
    ├── test_dataset_analyzer.py
    ├── test_training_diagnosis.py
    ├── test_guardrails.py
    └── test_decision_tree.py
```

---

## 9. Data Flow Summary

```
User selects dataset & config
          │
          ▼
Module A: Analyze dataset → dataset_report.json
          │
          ▼
Module B (when training finishes): analyze results → training_report.json
          │
          ▼
Module C: Read dataset_report.json + training_report.json
  → LLM decision → Guardrail validation → Apply changes
          │
          ▼
Executor: Write args.yaml → Launch YOLOv8 subprocess
          │
          ▼
[Probe mode:] After N epochs → Module B partial analysis
  → Decide: continue | abort & retune
          │
          ▼
Training complete → Module B full analysis → Log to History
          │
          ▼
Loop: if auto_tune enabled → Module C again
```

---

## 10. Edge Cases & Error Handling

| Scenario | Behavior |
|----------|----------|
| No GPU available | Auto-detect, warn user, fallback to CPU training (slow) |
| LLM API call fails | Use decision tree fallback, log error |
| Dataset has 0 annotated images | Module A returns error, training blocked |
| All classes perform well | Module C suggests no changes, log "optimal" |
| Probe detects no improvement for 3 retries | Abort loop, flag for human review |
| Guardrail CRITICAL violation | Override value, log with full context |
| YOLO subprocess crashes | Parse stderr, report error, offer retry |
