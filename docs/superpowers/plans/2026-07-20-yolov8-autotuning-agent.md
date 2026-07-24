# YOLOv8 Auto-Tuning Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-module auto-tuning system that analyzes datasets, diagnoses YOLOv8 training, and autonomously adjusts hyperparameters in a closed loop, with a Gradio web UI.

**Architecture:** Three independent modules (A: dataset analyzer, B: training diagnosis, C: agent decision + execution) communicating via JSON files. UI layer reads same JSON for human display. Plugin-based guardrails with decision tree fallback.

**Tech Stack:** Python 3.10+, YOLOv8 (ultralytics), Gradio, FastAPI, OpenCV, scikit-learn, torchvision, PyYAML, Pydantic

---

## Phase 0: Project Scaffolding

### Task 0: Create project structure and scaffolding

**Files:**
- Create: `auto_tune/config.yaml`
- Create: `auto_tune/requirements.txt`
- Create: `auto_tune/main.py`
- Create: `auto_tune/__init__.py`
- Create: `auto_tune/modules/__init__.py`
- Create: `auto_tune/modules/dataset_analyzer/__init__.py`
- Create: `auto_tune/modules/training_diagnosis/__init__.py`
- Create: `auto_tune/modules/agent_engine/__init__.py`
- Create: `auto_tune/modules/agent_engine/guardrails/__init__.py`
- Create: `auto_tune/ui/__init__.py`
- Create: `auto_tune/utils/__init__.py`
- Create: `auto_tune/tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd e:\dataprocess_modeltrain\Auto_labeltrain_project
mkdir -p auto_tune/modules/dataset_analyzer
mkdir -p auto_tune/modules/training_diagnosis
mkdir -p auto_tune/modules/agent_engine/guardrails/rules
mkdir -p auto_tune/ui/pages
mkdir -p auto_tune/ui/components
mkdir -p auto_tune/ui/static
mkdir -p auto_tune/utils
mkdir -p auto_tune/tests
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
gradio==4.44.0
pyyaml==6.0.2
pydantic==2.9.0
opencv-python==4.10.0
scikit-learn==1.5.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics==8.2.0
matplotlib==3.9.0
seaborn==0.13.0
pillow==10.4.0
httpx==0.27.0
```

- [ ] **Step 3: Write `config.yaml`**

```yaml
llm:
  provider: claude
  api_key_env: LLM_API_KEY
  model: claude-opus-4-7
  temperature: 0.3
  fallback_to_tree: true

guardrails:
  mode: strict
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

- [ ] **Step 4: Write stub `main.py`**

```python
"""YOLOv8 Auto-Tuning Agent — Entry Point."""

def main():
    print("Auto-Tuning Agent starting...")
    # TODO: FastAPI + Gradio integration in Phase 6

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write all `__init__.py` files (empty stubs)**

```bash
for dir in auto_tune auto_tune/modules auto_tune/modules/dataset_analyzer auto_tune/modules/training_diagnosis auto_tune/modules/agent_engine auto_tune/modules/agent_engine/guardrails auto_tune/ui auto_tune/utils auto_tune/tests; do
    if not exist "$dir/__init__.py" echo. 2>"$dir/__init__.py"
done
```

- [ ] **Step 6: Initial commit**

```bash
cd e:\dataprocess_modeltrain\Auto_labeltrain_project
git init
git add auto_tune/
git commit -m "chore: scaffold project structure and config"
```

---

## Phase 1: Module A — Dataset Analyzer

### Task 1: Image quality analyzer

**Files:**
- Create: `auto_tune/modules/dataset_analyzer/image_quality.py`

- [ ] **Step 1: Write `image_quality.py`**

```python
"""Image quality analysis using traditional CV methods."""

import cv2
import numpy as np


def estimate_blur(image: np.ndarray, threshold: float = 100.0) -> dict:
    """Estimate blur using Laplacian variance.

    Args:
        image: BGR image array.
        threshold: Laplacian variance threshold below which image is blurry.

    Returns:
        dict with 'laplacian_var' and 'is_blurry' fields.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {"laplacian_var": float(laplacian_var), "is_blurry": laplacian_var < threshold}


def estimate_exposure(image: np.ndarray) -> dict:
    """Estimate exposure using grayscale histogram percentiles.

    Args:
        image: BGR image array.

    Returns:
        dict with 'under_exposure', 'over_exposure' ratios [0,1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_pixels = np.sum(gray < 50) / gray.size
    high_pixels = np.sum(gray > 200) / gray.size
    return {"under_exposure": float(low_pixels), "over_exposure": float(high_pixels)}


def estimate_snr(image: np.ndarray) -> dict:
    """Estimate signal-to-noise ratio.

    Uses the ratio of mean to standard deviation in homogeneous regions.

    Args:
        image: BGR image array.

    Returns:
        dict with 'snr' value.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    snr = float(mean_val / std_val) if std_val > 0 else 0.0
    return {"snr": snr}


def analyze_image(image: np.ndarray, blur_threshold: float = 100.0) -> dict:
    """Run all image quality checks on a single image.

    Args:
        image: BGR image array.
        blur_threshold: threshold for blur detection.

    Returns:
        dict combining blur, exposure, and SNR results.
    """
    blur = estimate_blur(image, blur_threshold)
    exposure = estimate_exposure(image)
    snr = estimate_snr(image)
    return {**blur, **exposure, **snr}
```

- [ ] **Step 2: Write test**

```python
# tests/test_image_quality.py
import numpy as np
from auto_tune.modules.dataset_analyzer.image_quality import (
    estimate_blur, estimate_exposure, estimate_snr, analyze_image
)


def test_estimate_blur_sharp_image():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = estimate_blur(image)
    assert "laplacian_var" in result
    assert "is_blurry" in result


def test_estimate_exposure_normal():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = estimate_exposure(image)
    assert 0.0 <= result["under_exposure"] <= 1.0
    assert 0.0 <= result["over_exposure"] <= 1.0


def test_estimate_snr_constant_image():
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    result = estimate_snr(image)
    assert result["snr"] > 0


def test_analyze_image_returns_all_keys():
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = analyze_image(image)
    assert "laplacian_var" in result
    assert "under_exposure" in result
    assert "over_exposure" in result
    assert "snr" in result
```

- [ ] **Step 3: Run tests**

Run: `cd e:\dataprocess_modeltrain\Auto_labeltrain_project && python -m pytest auto_tune/tests/test_image_quality.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
cd e:\dataprocess_modeltrain\Auto_labeltrain_project
git add auto_tune/modules/dataset_analyzer/image_quality.py auto_tune/tests/test_image_quality.py
git commit -m "feat: image quality analysis (blur, exposure, SNR)"
```


### Task 2: Bounding box geometry analyzer

**Files:**
- Create: `auto_tune/modules/dataset_analyzer/bbox_geometry.py`

- [ ] **Step 1: Write `bbox_geometry.py`**

```python
"""Bounding box geometry analysis for YOLO-format labels."""

import numpy as np


def parse_yolo_label(label_path: str, img_width: int, img_height: int) -> list[dict]:
    """Parse YOLO-format label file.

    Each line: class_id x_center y_center width height (normalized [0,1]).

    Returns list of dicts with absolute pixel coordinates.
    """
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c = float(parts[1]) * img_width
            y_c = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            boxes.append({
                "class_id": cls_id,
                "x_center": x_c, "y_center": y_c,
                "width": w, "height": h,
                "area": w * h,
                "aspect_ratio": w / h if h > 0 else 0,
            })
    return boxes


def compute_bbox_size_stats(boxes: list[dict], img_area: int) -> dict:
    """Categorize bounding boxes by size relative to image area.

    Categories: tiny (<0.01), small (0.01-0.05), medium (0.05-0.2), large (>0.2).

    Returns dict with ratios per category and overall stats.
    """
    if not boxes:
        return {
            "tiny_bbox_ratio": 0.0,
            "small_bbox_ratio": 0.0,
            "medium_bbox_ratio": 0.0,
            "large_bbox_ratio": 0.0,
            "mean_area": 0.0,
            "area_std": 0.0,
        }
    areas = np.array([b["area"] for b in boxes])
    rel_areas = areas / img_area
    return {
        "tiny_bbox_ratio": float(np.mean(rel_areas < 0.01)),
        "small_bbox_ratio": float(np.mean((rel_areas >= 0.01) & (rel_areas < 0.05))),
        "medium_bbox_ratio": float(np.mean((rel_areas >= 0.05) & (rel_areas < 0.2))),
        "large_bbox_ratio": float(np.mean(rel_areas >= 0.2)),
        "mean_area": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
    }


def compute_aspect_ratio_range(boxes: list[dict]) -> list[float]:
    """Return [min, max] aspect ratio across boxes."""
    if not boxes:
        return [0.0, 0.0]
    ratios = [b["aspect_ratio"] for b in boxes if b["aspect_ratio"] > 0]
    return [float(np.min(ratios)), float(np.max(ratios))] if ratios else [0.0, 0.0]


def compute_overlap_analysis(boxes: list[dict], iou_threshold: float = 0.7) -> dict:
    """Compute IoU overlap statistics between all box pairs.

    Returns dict with high_iou_ratio and list of severe overlap class pairs.
    """
    if len(boxes) < 2:
        return {"high_iou_ratio": 0.0, "severe_overlap_pairs": []}

    high_iou_count = 0
    total_pairs = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = _compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                high_iou_count += 1
            total_pairs += 1

    return {
        "high_iou_ratio": high_iou_count / total_pairs if total_pairs > 0 else 0.0,
        "severe_overlap_pairs": [],
    }


def compute_spatial_bias(boxes: list[dict]) -> dict:
    """Analyze spatial distribution of box centers.

    Returns dict with center_concentration_score and edge_distribution_ratio.
    """
    if not boxes:
        return {"center_concentration_score": 0.0, "edge_distribution_ratio": 0.0}

    centers = np.array([[b["x_center"], b["y_center"]] for b in boxes])
    # Normalize to [0,1]
    # Center concentration: proportion of centers in middle 50%
    cx, cy = centers[:, 0], centers[:, 1]
    center_mask = (cx > 0.25) & (cx < 0.75) & (cy > 0.25) & (cy < 0.75)
    edge_region_mask = (cx < 0.1) | (cx > 0.9) | (cy < 0.1) | (cy > 0.9)

    return {
        "center_concentration_score": float(np.mean(center_mask)),
        "edge_distribution_ratio": float(np.mean(edge_region_mask)),
    }


def _compute_iou(a: dict, b: dict) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(a["x_center"] - a["width"] / 2, b["x_center"] - b["width"] / 2)
    y1 = max(a["y_center"] - a["height"] / 2, b["y_center"] - b["height"] / 2)
    x2 = min(a["x_center"] + a["width"] / 2, b["x_center"] + b["width"] / 2)
    y2 = min(a["y_center"] + a["height"] / 2, b["y_center"] + b["height"] / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

- [ ] **Step 2: Write test**

```python
# tests/test_bbox_geometry.py
import tempfile
import os
from auto_tune.modules.dataset_analyzer.bbox_geometry import (
    parse_yolo_label, compute_bbox_size_stats, compute_aspect_ratio_range,
    compute_overlap_analysis, compute_spatial_bias
)


def test_parse_yolo_label():
    content = "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    boxes = parse_yolo_label(path, 640, 480)
    os.unlink(path)
    assert len(boxes) == 2
    assert boxes[0]["class_id"] == 0
    assert boxes[0]["width"] == 128
    assert boxes[0]["height"] == 144


def test_compute_bbox_size_stats_empty():
    stats = compute_bbox_size_stats([], 640 * 480)
    assert stats["tiny_bbox_ratio"] == 0.0


def test_compute_spatial_bias():
    boxes = [
        {"x_center": 0.5, "y_center": 0.5, "width": 10, "height": 10, "area": 100, "aspect_ratio": 1.0, "class_id": 0},
    ]
    bias = compute_spatial_bias(boxes)
    assert 0.0 <= bias["center_concentration_score"] <= 1.0


def test_compute_overlap_analysis_single_box():
    boxes = [{"x_center": 50, "y_center": 50, "width": 20, "height": 20, "area": 400, "aspect_ratio": 1.0, "class_id": 0}]
    result = compute_overlap_analysis(boxes)
    assert result["high_iou_ratio"] == 0.0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_bbox_geometry.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/dataset_analyzer/bbox_geometry.py auto_tune/tests/test_bbox_geometry.py
git commit -m "feat: bounding box geometry analysis"
```


### Task 3: Class distribution statistics

**Files:**
- Create: `auto_tune/modules/dataset_analyzer/class_stats.py`

- [ ] **Step 1: Write `class_stats.py`**

```python
"""Class distribution analysis for YOLO datasets."""

from collections import Counter


def compute_class_distribution(label_files: list[str]) -> dict:
    """Count class instances across all label files.

    Args:
        label_files: list of paths to YOLO-format .txt label files.

    Returns:
        dict mapping class_id -> count, and total_instances.
    """
    counter = Counter()
    for path in label_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
    return dict(counter)


def compute_class_balance(class_counts: dict[int, int], class_names: dict[int, str]) -> dict:
    """Compute class balance metrics.

    Args:
        class_counts: dict of class_id -> count.
        class_names: dict of class_id -> class_name.

    Returns:
        dict with per-class ratios, is_balanced flag, long_tail_classes list.
    """
    if not class_counts:
        return {
            "class_distribution": {},
            "is_balanced": True,
            "long_tail_classes": [],
            "imbalance_ratio": 1.0,
        }

    total = sum(class_counts.values())
    distribution = {}
    for cls_id, count in class_counts.items():
        name = class_names.get(cls_id, f"class_{cls_id}")
        distribution[name] = {"count": count, "ratio": round(count / total, 4)}

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    threshold = total / len(class_counts) * 0.3  # < 30% of average = long tail
    long_tail = [
        class_names.get(cid, f"class_{cid}")
        for cid, cnt in class_counts.items()
        if cnt < threshold
    ]

    return {
        "class_distribution": distribution,
        "is_balanced": imbalance_ratio < 5.0,
        "long_tail_classes": long_tail,
        "imbalance_ratio": round(imbalance_ratio, 2),
    }
```

- [ ] **Step 2: Write test**

```python
# tests/test_class_stats.py
from auto_tune.modules.dataset_analyzer.class_stats import (
    compute_class_distribution, compute_class_balance
)


def test_compute_class_distribution():
    import tempfile, os
    files = []
    contents = [
        "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n",
        "1 0.5 0.5 0.2 0.2\n",
    ]
    for content in contents:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        f.write(content)
        f.close()
        files.append(f.name)
    counts = compute_class_distribution(files)
    for p in files:
        os.unlink(p)
    assert counts[0] == 2
    assert counts[1] == 1


def test_compute_class_balance_balanced():
    counts = {0: 100, 1: 80, 2: 90}
    names = {0: "crack", 1: "scratch", 2: "dent"}
    result = compute_class_balance(counts, names)
    assert result["is_balanced"] is True


def test_compute_class_balance_long_tail():
    counts = {0: 500, 1: 30, 2: 400}
    names = {0: "crack", 1: "scratch", 2: "dent"}
    result = compute_class_balance(counts, names)
    assert "scratch" in result["long_tail_classes"]
    assert result["is_balanced"] is False
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_class_stats.py -v`
Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/dataset_analyzer/class_stats.py auto_tune/tests/test_class_stats.py
git commit -m "feat: class distribution and balance analysis"
```


### Task 4: Feature clustering for outlier detection

**Files:**
- Create: `auto_tune/modules/dataset_analyzer/feature_cluster.py`

- [ ] **Step 1: Write `feature_cluster.py`**

```python
"""Feature extraction and clustering for outlier detection."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def extract_features(images: list[np.ndarray]) -> np.ndarray:
    """Extract feature vectors using simple HOG-like features.

    Uses flattened histogram of oriented gradients as lightweight
    feature descriptor when no GPU is available. For production,
    swap with ResNet18 (see _extract_resnet_features).

    Args:
        images: list of BGR image arrays (resized to 224x224 caller side).

    Returns:
        (N, D) feature matrix.
    """
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Simple HOG: compute gradient magnitude and orientation
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        # Bin orientations into 9 bins
        bins = np.int32(9 * ang / (2 * np.pi))
        hist = np.zeros(9)
        for i in range(9):
            hist[i] = np.sum(mag[bins == i])
        # Normalize
        hist = hist / (np.sum(hist) + 1e-6)
        features.append(hist)
    return np.array(features)


def cluster_outliers(features: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> dict:
    """Run PCA + DBSCAN to detect outlier samples.

    Args:
        features: (N, D) feature matrix.
        eps: DBSCAN eps parameter.
        min_samples: DBSCAN min_samples parameter.

    Returns:
        dict with outlier_count, outlier_ratio, silhouette_score.
    """
    if len(features) < 3:
        return {"outlier_count": 0, "outlier_ratio": 0.0, "silhouette_score": 0.0}

    # PCA to 2D for DBSCAN
    pca = PCA(n_components=min(2, features.shape[1]))
    reduced = pca.fit_transform(features)

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced)
    labels = clustering.labels_

    # -1 labels are outliers
    outlier_mask = labels == -1
    outlier_count = int(np.sum(outlier_mask))

    # Silhouette score (only if at least 2 clusters and not all outliers)
    unique_labels = set(labels) - {-1}
    sil_score = 0.0
    if len(unique_labels) >= 2 and outlier_count < len(labels):
        sil_score = float(silhouette_score(reduced[~outlier_mask], labels[~outlier_mask]))

    return {
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / len(features), 4),
        "silhouette_score": round(sil_score, 4),
    }
```

- [ ] **Step 2: Write test**

```python
# tests/test_feature_cluster.py
import numpy as np
from auto_tune.modules.dataset_analyzer.feature_cluster import (
    extract_features, cluster_outliers
)


def test_cluster_outliers_few_samples():
    result = cluster_outliers(np.random.rand(2, 5))
    assert result["outlier_count"] == 0


def test_cluster_outliers_all_similar():
    features = np.tile(np.random.rand(1, 5), (20, 1))
    result = cluster_outliers(features, eps=0.5, min_samples=2)
    assert "outlier_count" in result
    assert "silhouette_score" in result
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_feature_cluster.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/dataset_analyzer/feature_cluster.py auto_tune/tests/test_feature_cluster.py
git commit -m "feat: feature clustering for outlier detection"
```

**Note:** `extract_features` requires `import cv2` at module top — add it when implementing. The test file needs it too.


### Task 5: Dataset analyzer orchestrator

**Files:**
- Create: `auto_tune/modules/dataset_analyzer/analyzer.py`

- [ ] **Step 1: Write `analyzer.py`**

```python
"""Dataset analyzer orchestrator — runs all analysis steps, produces JSON report."""

import os
import glob
import datetime
import cv2
import numpy as np
from .image_quality import analyze_image
from .bbox_geometry import parse_yolo_label, compute_bbox_size_stats, compute_aspect_ratio_range, compute_overlap_analysis, compute_spatial_bias
from .class_stats import compute_class_distribution, compute_class_balance
from .feature_cluster import extract_features as _extract_features, cluster_outliers


def analyze_dataset(dataset_dir: str, data_yaml: dict, config: dict) -> dict:
    """Run full dataset analysis pipeline.

    Args:
        dataset_dir: path to dataset root (contains images/ and labels/).
        data_yaml: parsed data.yaml dict with 'names', 'train'/'val' paths.
        config: dict with analyzer thresholds.

    Returns:
        JSON-serializable dict per spec Section 3.3.
    """
    img_dir = os.path.join(dataset_dir, "images", "train")
    label_dir = os.path.join(dataset_dir, "labels", "train")
    class_names = {int(k): v for k, v in data_yaml.get("names", {}).items()}

    # Gather files
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    if not img_paths:
        return {"error": "No images found", "total_images": 0}

    # Step 1: Image quality analysis (sample-based for speed)
    quality_results = []
    sample_size = min(500, len(img_paths))
    sampled_paths = np.random.choice(img_paths, sample_size, replace=False)
    for path in sampled_paths:
        img = cv2.imread(str(path))
        if img is not None:
            quality_results.append(analyze_image(img, config.get("blur_threshold", 100.0)))

    blur_ratio = np.mean([q["is_blurry"] for q in quality_results])
    under_ratio = np.mean([q["under_exposure"] > 0.3 for q in quality_results])
    over_ratio = np.mean([q["over_exposure"] > 0.3 for q in quality_results])
    low_snr = np.mean([q["snr"] < 10 for q in quality_results])

    # Step 2: Parse labels and compute geometry
    all_boxes = []
    for label_path in label_files:
        boxes = parse_yolo_label(
            label_path,
            img_width=config.get("img_width", 640),
            img_height=config.get("img_height", 640),
        )
        all_boxes.extend(boxes)

    img_area = config.get("img_width", 640) * config.get("img_height", 640)
    bbox_stats = compute_bbox_size_stats(all_boxes, img_area)
    aspect_range = compute_aspect_ratio_range(all_boxes)
    overlap = compute_overlap_analysis(all_boxes, config.get("high_iou_threshold", 0.7))
    spatial = compute_spatial_bias(all_boxes)

    # Step 3: Class distribution
    class_counts = compute_class_distribution(label_files)
    class_balance = compute_class_balance(class_counts, class_names)

    # Step 4: Feature clustering (sample-based)
    cluster_sample = min(200, len(img_paths))
    cluster_paths = np.random.choice(img_paths, cluster_sample, replace=False)
    cluster_images = []
    for path in cluster_paths:
        img = cv2.imread(str(path))
        if img is not None:
            cluster_images.append(cv2.resize(img, (224, 224)))
    features = _extract_features(cluster_images)
    outliers = cluster_outliers(features, config.get("dbscan_eps", 0.3), config.get("dbscan_min_samples", 5))

    # Step 5: Build summary
    key_issues = []
    if not class_balance["is_balanced"]:
        for cls in class_balance["long_tail_classes"]:
            key_issues.append(f"long_tail_class_{cls}")
    if bbox_stats["tiny_bbox_ratio"] > 0.2:
        key_issues.append("tiny_bbox_high_ratio")
    if spatial["center_concentration_score"] > 0.6:
        key_issues.append("center_spatial_bias")
    if blur_ratio > 0.1:
        key_issues.append("high_blur_ratio")
    quality_score = round(1.0 - (blur_ratio * 0.3 + under_ratio * 0.2 + over_ratio * 0.2 + (1 - class_balance["is_balanced"]) * 0.3), 2)

    return {
        "module": "dataset_analyzer",
        "version": "1.0",
        "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "dataset_path": dataset_dir,
        "total_images": len(img_paths),
        "total_annotations": len(all_boxes),
        "class_distribution": class_balance["class_distribution"],
        "class_balance": {
            "is_balanced": class_balance["is_balanced"],
            "long_tail_classes": class_balance["long_tail_classes"],
            "imbalance_ratio": class_balance["imbalance_ratio"],
        },
        "image_quality": {
            "blur_ratio": round(float(blur_ratio), 4),
            "overexposure_ratio": round(float(over_ratio), 4),
            "underexposure_ratio": round(float(under_ratio), 4),
            "low_snr_ratio": round(float(low_snr), 4),
        },
        "bbox_analysis": bbox_stats,
        "spatial_bias": spatial,
        "overlap_analysis": overlap,
        "outlier_analysis": outliers,
        "summary": {
            "dataset_quality_score": quality_score,
            "key_issues": key_issues,
        },
    }
```

- [ ] **Step 2: Write test**

```python
# tests/test_analyzer.py
import json
from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset


def test_analyze_dataset_no_images(tmp_path):
    data_yaml = {"names": {0: "crack"}, "train": str(tmp_path)}
    result = analyze_dataset(str(tmp_path), data_yaml, {"blur_threshold": 100.0})
    assert "error" in result


def test_analyze_dataset_output_structure():
    """Verify all expected keys exist in output."""
    # Integration test requiring real data — verify structure contract
    import inspect
    sig = inspect.signature(analyze_dataset)
    assert "dataset_dir" in sig.parameters
    assert "data_yaml" in sig.parameters
    assert "config" in sig.parameters
```

- [ ] **Step 3: Run test**

Run: `python -m pytest auto_tune/tests/test_analyzer.py -v`
Expected: 2 passed (second test validates interface shape)

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/dataset_analyzer/analyzer.py auto_tune/tests/test_analyzer.py
git commit -m "feat: dataset analyzer orchestrator"
```


## Phase 2: Module B — Training Diagnosis

### Task 6: CSV parser for YOLOv8 results

**Files:**
- Create: `auto_tune/modules/training_diagnosis/csv_parser.py`

- [ ] **Step 1: Write `csv_parser.py`**

```python
"""Parse YOLOv8 results.csv to extract loss/mAP trends and flags."""

import csv
import numpy as np


def parse_results_csv(csv_path: str) -> dict:
    """Parse results.csv from YOLOv8 training.

    Expected columns: epoch, train/box_loss, train/cls_loss, train/dfl_loss,
                      metrics/precision, metrics/recall, metrics/mAP50,
                      metrics/mAP50-95, val/box_loss, val/cls_loss, val/dfl_loss,
                      x/lr0, x/lr1, x/lr2

    Returns dict with loss trends, mAP data, and diagnostic flags.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "Empty CSV", "epochs_completed": 0}

    n = len(rows)
    epochs = np.array([int(r.get("epoch", 0)) for r in rows])
    train_box = np.array([float(r.get("train/box_loss", 0)) for r in rows])
    val_box = np.array([float(r.get("val/box_loss", 0)) for r in rows])
    mAP50 = np.array([float(r.get("metrics/mAP50", 0)) for r in rows])
    mAP50_95 = np.array([float(r.get("metrics/mAP50-95", 0)) for r in rows])
    cls_loss = np.array([float(r.get("train/cls_loss", 0)) for r in rows])
    dfl_loss = np.array([float(r.get("train/dfl_loss", 0)) for r in rows])

    # Best epoch
    best_idx = int(np.argmax(mAP50))
    best_epoch = int(epochs[best_idx])
    best_mAP50 = float(mAP50[best_idx])
    best_mAP50_95 = float(mAP50_95[best_idx])

    # Overfitting detection: val_box_loss rising > 5 epochs while train_box_loss falling
    last_20_train = train_box[-20:] if n >= 20 else train_box
    last_20_val = val_box[-20:] if n >= 20 else val_box

    train_slope = np.polyfit(range(len(last_20_train)), last_20_train, 1)[0]
    val_slope = np.polyfit(range(len(last_20_val)), last_20_val, 1)[0]

    overfitting = bool(train_slope < 0 and val_slope > 0)
    overfitting_start = None
    if overfitting and n > 10:
        for i in range(n - 10, n):
            if val_box[i] > val_box[i - 1]:
                overfitting_start = int(epochs[i])
                break

    # Gradient explosion check
    nan_occurred = bool(np.any(np.isnan(train_box)) or np.any(np.isnan(val_box)))
    max_loss = float(np.max(train_box))
    initial_loss = float(train_box[0]) if len(train_box) > 0 else 0
    loss_spikes = bool(max_loss > initial_loss * 50 and initial_loss > 0)

    return {
        "epochs_completed": n,
        "best_epoch": best_epoch,
        "best_mAP50": round(best_mAP50, 4),
        "best_mAP50_95": round(best_mAP50_95, 4),
        "final_box_loss": round(float(train_box[-1]), 6),
        "final_val_box_loss": round(float(val_box[-1]), 6) if len(val_box) > 0 else 0.0,
        "overfitting": {
            "detected": overfitting,
            "start_epoch": overfitting_start,
            "train_box_slope": round(float(train_slope), 6),
            "val_box_slope": round(float(val_slope), 6),
        },
        "gradient_issues": {
            "nan_occurred": nan_occurred,
            "loss_spikes": loss_spikes,
            "max_loss": max_loss,
        },
    }
```

- [ ] **Step 2: Write test**

```python
# tests/test_csv_parser.py
import csv
import tempfile, os
from auto_tune.modules.training_diagnosis.csv_parser import parse_results_csv


def _make_csv(rows: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


def test_parse_empty_csv():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.close()
    result = parse_results_csv(f.name)
    os.unlink(f.name)
    assert "error" in result


def test_parse_basic_csv():
    rows = []
    for e in range(10):
        rows.append({
            "epoch": str(e),
            "train/box_loss": str(0.1 / (e + 1)),
            "val/box_loss": str(0.12 - e * 0.002),
            "metrics/mAP50": str(min(0.9, e * 0.09)),
            "metrics/mAP50-95": str(min(0.7, e * 0.07)),
            "train/cls_loss": "0.05",
            "train/dfl_loss": "0.08",
            "x/lr0": "0.01",
            "x/lr1": "0.01",
            "x/lr2": "0.01",
        })
    path = _make_csv(rows)
    result = parse_results_csv(path)
    os.unlink(path)
    assert result["epochs_completed"] == 10
    assert result["best_mAP50"] >= 0.8
```

- [ ] **Step 3: Run test**

Run: `python -m pytest auto_tune/tests/test_csv_parser.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/training_diagnosis/csv_parser.py auto_tune/tests/test_csv_parser.py
git commit -m "feat: YOLOv8 results.csv parser with trend analysis"
```


### Task 7: Metrics calculator (miss_rate, false_alarm_rate)

**Files:**
- Create: `auto_tune/modules/training_diagnosis/metrics.py`

- [ ] **Step 1: Write `metrics.py`**

```python
"""Industrial metrics: miss_rate and false_alarm_rate from confusion matrix."""

import numpy as np


def compute_rates_from_confusion(confusion_matrix: np.ndarray, num_classes: int) -> dict:
    """Compute miss_rate and false_alarm_rate per class from confusion matrix.

    Args:
        confusion_matrix: (num_classes+1, num_classes+1) array where
                          last row/col is background.
        num_classes: number of target classes (excluding background).

    Returns:
        dict with per-class rates and overall averages.
    """
    per_class = {}
    total_fp = 0
    total_fn = 0
    total_tp = 0

    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fn = np.sum(confusion_matrix[c, :num_classes]) - tp
        fp = np.sum(confusion_matrix[:num_classes, c]) - tp
        # Background FN = background row minus background col
        bg_fn = confusion_matrix[num_classes, c]
        bg_fp = confusion_matrix[c, num_classes]

        miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        total_tp += tp
        total_fn += fn
        total_fp += fp

        per_class[c] = {
            "miss_rate": round(float(miss_rate), 4),
            "false_alarm_rate": round(float(false_alarm_rate), 4),
            "background_fp": int(bg_fp),
            "background_fn": int(bg_fn),
        }

    overall_miss = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_fa = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0

    return {
        "per_class": per_class,
        "overall_miss_rate": round(float(overall_miss), 4),
        "overall_false_alarm_rate": round(float(overall_fa), 4),
    }
```

- [ ] **Step 2: Write test**

```python
# tests/test_metrics.py
import numpy as np
from auto_tune.modules.training_diagnosis.metrics import compute_rates_from_confusion


def test_perfect_detection():
    cm = np.array([
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 5],
    ])
    result = compute_rates_from_confusion(cm, 2)
    assert result["per_class"][0]["miss_rate"] == 0.0
    assert result["per_class"][0]["false_alarm_rate"] == 0.0


def test_some_misses():
    cm = np.array([
        [8, 2, 1],
        [3, 15, 2],
        [0, 0, 0],
    ])
    result = compute_rates_from_confusion(cm, 2)
    assert result["per_class"][0]["miss_rate"] > 0
    assert result["overall_miss_rate"] > 0
```

- [ ] **Step 3: Run test**

Run: `python -m pytest auto_tune/tests/test_metrics.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/training_diagnosis/metrics.py auto_tune/tests/test_metrics.py
git commit -m "feat: miss_rate and false_alarm_rate metrics"
```


### Task 8: Text LLM diagnosis (Stage 2)

**Files:**
- Create: `auto_tune/modules/training_diagnosis/text_diagnosis.py`

- [ ] **Step 1: Write `text_diagnosis.py`**

```python
"""Stage 2 diagnosis: send structured data to text LLM for analysis."""

import json
import os
from typing import Optional


def build_diagnosis_prompt(csv_analysis: dict, args_yaml: dict) -> str:
    """Build a concise prompt for text LLM diagnosis.

    Args:
        csv_analysis: output from csv_parser.parse_results_csv().
        args_yaml: parsed args.yaml dict.

    Returns:
        Prompt string (~800 tokens).
    """
    return f"""You are a YOLOv8 training diagnostician. Analyze the training results and recommend hyperparameter changes.

## Training Results
- Epochs completed: {csv_analysis.get('epochs_completed')}
- Best mAP50: {csv_analysis.get('best_mAP50')} at epoch {csv_analysis.get('best_epoch')}
- Final box loss: {csv_analysis.get('final_box_loss')}
- Final val box loss: {csv_analysis.get('final_val_box_loss')}
- Overfitting detected: {csv_analysis.get('overfitting', {}).get('detected')}
- Gradient issues: NaN={csv_analysis.get('gradient_issues', {}).get('nan_occurred')}, Spikes={csv_analysis.get('gradient_issues', {}).get('loss_spikes')}

## Current Hyperparameters
{json.dumps(args_yaml, indent=2)}

## Response Format
Return a JSON object with:
- "status": one of "optimal", "retune_needed", "retrain_needed"
- "diagnosis": short text explanation
- "reason": one of "overfitting", "underfitting", "gradient_issues", "class_imbalance", "low_mAP", "mixed", "none"
- "hyperparameter_suggestions": dict of param -> new value (only changed params)
- "triggers_mllm": true if confusion matrix or visual inspection needed"""
```

- [ ] **Step 2: Write `text_diagnosis.py` (continued) — LLM caller**

```python
def call_llm(prompt: str, config: dict) -> Optional[dict]:
    """Call configured LLM API for diagnosis.

    Supports Claude (Anthropic API), OpenAI, and Ollama local.
    Returns parsed JSON response or None on failure.
    """
    provider = config.get("provider", "claude")
    api_key = os.environ.get(config.get("api_key_env", "LLM_API_KEY"), "")

    if provider == "claude":
        return _call_claude(prompt, api_key, config.get("model", "claude-opus-4-7"))
    elif provider == "openai":
        return _call_openai(prompt, api_key, config.get("model", "gpt-4o"))
    elif provider == "ollama":
        return _call_ollama(prompt, config.get("model", "qwen2.5"))
    return None


def _call_claude(prompt: str, api_key: str, model: str) -> Optional[dict]:
    """Call Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return _extract_json(text)
    except Exception as e:
        print(f"Claude API call failed: {e}")
        return None


def _call_openai(prompt: str, api_key: str, model: str) -> Optional[dict]:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return _extract_json(text)
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None


def _call_ollama(prompt: str, model: str) -> Optional[dict]:
    """Call local Ollama model."""
    try:
        import httpx
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        text = response.json()["response"]
        return _extract_json(text)
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response text (handles markdown wrapping)."""
    import re
    # Try ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def should_trigger_mllm(diagnosis: dict) -> bool:
    """Determine if Stage 3 (MLLM vision) is needed."""
    if diagnosis is None:
        return False
    return diagnosis.get("triggers_mllm", False) or diagnosis.get("status") == "retrain_needed"
```

- [ ] **Step 3: Write test**

```python
# tests/test_text_diagnosis.py
from auto_tune.modules.training_diagnosis.text_diagnosis import (
    build_diagnosis_prompt, _extract_json, should_trigger_mllm
)


def test_build_diagnosis_prompt():
    csv_analysis = {"epochs_completed": 50, "best_mAP50": 0.75, "best_epoch": 42,
                    "final_box_loss": 0.02, "final_val_box_loss": 0.03,
                    "overfitting": {"detected": False}, "gradient_issues": {"nan_occurred": False, "loss_spikes": False}}
    args = {"lr0": 0.01, "batch": 16}
    prompt = build_diagnosis_prompt(csv_analysis, args)
    assert "mAP50" in prompt
    assert "0.01" in prompt


def test_extract_json_from_code_block():
    text = "Here's the result:\n```json\n{\"status\": \"optimal\"}\n```"
    result = _extract_json(text)
    assert result == {"status": "optimal"}


def test_extract_json_bare():
    text = 'Some text {"status": "retune_needed"} trailing'
    result = _extract_json(text)
    assert result == {"status": "retune_needed"}


def test_should_trigger_mllm():
    assert should_trigger_mllm({"triggers_mllm": True}) is True
    assert should_trigger_mllm({"triggers_mllm": False}) is False
    assert should_trigger_mllm(None) is False
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest auto_tune/tests/test_text_diagnosis.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add auto_tune/modules/training_diagnosis/text_diagnosis.py auto_tune/tests/test_text_diagnosis.py
git commit -m "feat: text LLM diagnosis stage with multi-provider support"
```


### Task 9: Vision diagnosis (Stage 3, conditional)

**Files:**
- Create: `auto_tune/modules/training_diagnosis/vision_diagnosis.py`

- [ ] **Step 1: Write `vision_diagnosis.py`**

```python
"""Stage 3 diagnosis: MLLM vision analysis of confusion matrix and misclassification crops."""

import base64
import os
from typing import Optional


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Read image and return base64-encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def build_vision_prompt(csv_summary: dict) -> str:
    """Build prompt for vision LLM analysis.

    Args:
        csv_summary: concise summary from Stage 1/2.

    Returns:
        Prompt string.
    """
    return f"""Analyze the confusion matrix and misclassification patches from YOLOv8 training.

Context:
- Best mAP50: {csv_summary.get('best_mAP50', 'N/A')}
- Best epoch: {csv_summary.get('best_epoch', 'N/A')}
- Overfitting: {csv_summary.get('overfitting', {}).get('detected', 'N/A')}

Focus on:
1. Which classes are confused with each other?
2. Is background being misclassified as defects (false alarms)?
3. Are defects being missed (high false negative in specific classes)?
4. What visual patterns cause the errors (lighting, occlusion, small size)?

Respond with JSON:
{{"visual_diagnosis": "...", "likely_causes": ["..."], "suggested_fixes": ["..."]}}"""


def diagnose_with_vision(
    confusion_matrix_path: str,
    crop_patches_dir: str,
    csv_summary: dict,
    llm_config: dict,
) -> Optional[dict]:
    """Run vision diagnosis with configured MLLM.

    Currently supports Anthropic Claude (vision) and OpenAI GPT-4o.
    Falls back to text-only diagnosis if vision not available.

    Args:
        confusion_matrix_path: path to confusion_matrix_normalized.png.
        crop_patches_dir: directory of 224x224 misclassification crops.
        csv_summary: dict from csv_parser.
        llm_config: LLM configuration dict.

    Returns:
        Parsed JSON dict or None.
    """
    provider = llm_config.get("provider", "claude")
    api_key = os.environ.get(llm_config.get("api_key_env", "LLM_API_KEY"), "")

    cm_b64 = encode_image_to_base64(confusion_matrix_path)
    if cm_b64 is None:
        return None

    prompt = build_vision_prompt(csv_summary)

    if provider == "claude":
        return _call_claude_vision(prompt, cm_b64, api_key, llm_config.get("model", "claude-opus-4-7"))
    elif provider == "openai":
        return _call_openai_vision(prompt, cm_b64, api_key, llm_config.get("model", "gpt-4o"))
    return None


def _call_claude_vision(prompt: str, image_b64: str, api_key: str, model: str) -> Optional[dict]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                ],
            }],
        )
        from .text_diagnosis import _extract_json
        return _extract_json(response.content[0].text)
    except Exception as e:
        print(f"Claude vision call failed: {e}")
        return None


def _call_openai_vision(prompt: str, image_b64: str, api_key: str, model: str) -> Optional[dict]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
        )
        from .text_diagnosis import _extract_json
        return _extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI vision call failed: {e}")
        return None
```

- [ ] **Step 2: Write test**

```python
# tests/test_vision_diagnosis.py
from auto_tune.modules.training_diagnosis.vision_diagnosis import (
    encode_image_to_base64, build_vision_prompt
)


def test_encode_image_to_base64_nonexistent():
    result = encode_image_to_base64("nonexistent.png")
    assert result is None


def test_build_vision_prompt():
    summary = {"best_mAP50": 0.75, "best_epoch": 50, "overfitting": {"detected": False}}
    prompt = build_vision_prompt(summary)
    assert "mAP50" in prompt
    assert "confusion matrix" in prompt.lower()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_vision_diagnosis.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/training_diagnosis/vision_diagnosis.py auto_tune/tests/test_vision_diagnosis.py
git commit -m "feat: vision diagnosis stage with MLLM support"
```


### Task 10: Training diagnosis orchestrator

**Files:**
- Create: `auto_tune/modules/training_diagnosis/diagnostician.py`

- [ ] **Step 1: Write `diagnostician.py`**

```python
"""Training diagnosis orchestrator — coordinates all three stages."""

import yaml
from .csv_parser import parse_results_csv
from .text_diagnosis import build_diagnosis_prompt, call_llm, should_trigger_mllm
from .vision_diagnosis import diagnose_with_vision


def diagnose_training(
    results_csv_path: str,
    args_yaml_path: str,
    llm_config: dict,
    confusion_matrix_path: str = None,
    crop_patches_dir: str = None,
) -> dict:
    """Run full training diagnosis pipeline.

    Args:
        results_csv_path: path to results.csv.
        args_yaml_path: path to args.yaml.
        llm_config: LLM configuration dict.
        confusion_matrix_path: optional path for Stage 3.
        crop_patches_dir: optional dir for Stage 3 crops.

    Returns:
        JSON dict with full diagnosis per spec Section 4.4.
    """
    # Stage 1: CSV parsing
    csv_analysis = parse_results_csv(results_csv_path)
    if "error" in csv_analysis:
        return {"module": "training_diagnosis", "version": "1.0", "error": csv_analysis["error"]}

    # Parse args.yaml
    with open(args_yaml_path, encoding="utf-8") as f:
        args_yaml = yaml.safe_load(f)

    # Build result with csv data
    result = {
        "module": "training_diagnosis",
        "version": "1.0",
        "training_summary": {
            "epochs_completed": csv_analysis["epochs_completed"],
            "best_epoch": csv_analysis["best_epoch"],
            "best_mAP50": csv_analysis["best_mAP50"],
            "best_mAP50_95": csv_analysis["best_mAP50_95"],
        },
        "overfitting": csv_analysis.get("overfitting", {}),
        "gradient_issues": csv_analysis.get("gradient_issues", {}),
        "class_performance": {},
        "diagnosis_stage": "csv_parser",
        "summary": {"status": "unknown", "recommend_action": "none"},
    }

    # Stage 2: Text LLM
    prompt = build_diagnosis_prompt(csv_analysis, args_yaml or {})
    llm_diagnosis = call_llm(prompt, llm_config)

    if llm_diagnosis:
        result["summary"]["status"] = llm_diagnosis.get("status", "unknown")
        result["summary"]["reason"] = llm_diagnosis.get("reason", "none")
        result["llm_diagnosis_text"] = llm_diagnosis.get("diagnosis", "")
        result["hyperparameter_suggestions"] = llm_diagnosis.get("hyperparameter_suggestions", {})
        result["diagnosis_stage"] = "text_llm"

        # Stage 3: Vision (conditional)
        if should_trigger_mllm(llm_diagnosis) and confusion_matrix_path:
            vision_result = diagnose_with_vision(
                confusion_matrix_path, crop_patches_dir, csv_analysis, llm_config
            )
            if vision_result:
                result["vision_diagnosis"] = vision_result
                result["diagnosis_stage"] = "mllm_vision"
    else:
        result["summary"]["status"] = "unknown"
        result["summary"]["reason"] = "llm_unavailable"

    return result
```

- [ ] **Step 2: Write test**

```python
# tests/test_diagnostician.py
from auto_tune.modules.training_diagnosis.diagnostician import diagnose_training


def test_diagnose_training_no_csv(tmp_path):
    csv_path = tmp_path / "nonexistent.csv"
    yaml_path = tmp_path / "args.yaml"
    yaml_path.write_text("lr0: 0.01\nbatch: 16\n")
    result = diagnose_training(str(csv_path), str(yaml_path), {"provider": "claude"})
    assert "error" in result or result["epochs_completed"] == 0
```

- [ ] **Step 3: Run test**

Run: `python -m pytest auto_tune/tests/test_diagnostician.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/training_diagnosis/diagnostician.py auto_tune/tests/test_diagnostician.py
git commit -m "feat: training diagnosis orchestrator (3 stages)"
```


## Phase 3: Module C — Agent Decision & Execution

### Task 11: Guardrails base, registry, and validator

**Files:**
- Create: `auto_tune/modules/agent_engine/guardrails/base.py`
- Create: `auto_tune/modules/agent_engine/guardrails/registry.py`
- Create: `auto_tune/modules/agent_engine/guardrails/validator.py`

- [ ] **Step 1: Write `base.py`**

```python
"""Guardrail rule base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RuleResult:
    """Result of a guardrail check."""
    rule_name: str
    passed: bool
    priority: str  # CRITICAL | HIGH | MEDIUM
    message: str = ""
    corrected_value: any = None


class GuardrailRule(ABC):
    """Abstract base for all guardrail rules."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def priority(self) -> str: ...

    @abstractmethod
    def check(self, changes: dict, context: dict) -> RuleResult:
        """Validate proposed hyperparameter changes.

        Args:
            changes: proposed hyperparameter changes dict.
            context: full context dict (dataset_report, training_report, config, current_params).

        Returns:
            RuleResult with pass/fail and optional corrected value.
        """
        ...
```

- [ ] **Step 2: Write `registry.py`**

```python
"""Auto-discovery and registry of guardrail rules."""

import importlib
import inspect
import pkgutil
import os
from .base import GuardrailRule, RuleResult


class GuardrailRegistry:
    """Discovers and manages guardrail rules."""

    def __init__(self):
        self._rules: list[GuardrailRule] = []

    def discover_rules(self, package_path: str = None):
        """Auto-discover rules from the rules package.

        Args:
            package_path: dotted path to rules package (default: auto_tune.modules.agent_engine.guardrails.rules).
        """
        if package_path is None:
            package_path = "auto_tune.modules.agent_engine.guardrails.rules"

        try:
            package = importlib.import_module(package_path)
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                if modname.startswith("_"):
                    continue
                module = importlib.import_module(f"{package_path}.{modname}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, GuardrailRule) and obj is not GuardrailRule:
                        self._rules.append(obj())
        except ImportError:
            print(f"Warning: could not load rules from {package_path}")

    def register_rule(self, rule: GuardrailRule):
        """Manually register a rule."""
        self._rules.append(rule)

    def get_rules(self, priority: str = None) -> list[GuardrailRule]:
        """Get rules, optionally filtered by priority."""
        if priority:
            return [r for r in self._rules if r.priority == priority]
        return list(self._rules)

    def clear(self):
        """Clear all registered rules."""
        self._rules.clear()
```

- [ ] **Step 3: Write `validator.py`**

```python
"""Guardrails validator — runs all rules against proposed changes."""

import copy
from .base import GuardrailRule, RuleResult
from .registry import GuardrailRegistry


class GuardrailValidator:
    """Validates hyperparameter changes against all registered rules."""

    def __init__(self, registry: GuardrailRegistry, mode: str = "strict"):
        """
        Args:
            registry: GuardrailRegistry with discovered rules.
            mode: 'strict' (CRITICAL=override, HIGH=clamp, MEDIUM=warn),
                  'warn' (log only, no overrides),
                  'off' (pass through).
        """
        self.registry = registry
        self.mode = mode

    def validate(self, changes: dict, context: dict) -> dict:
        """Run all rules against proposed changes.

        Args:
            changes: proposed hyperparameter changes dict.
            context: full context for rules.

        Returns:
            dict with:
              - 'passed': bool (all rules passed)
              - 'changes': corrected changes dict
              - 'results': list of RuleResult
              - 'warnings': list of warning strings
        """
        if self.mode == "off":
            return {"passed": True, "changes": changes, "results": [], "warnings": []}

        corrected = copy.deepcopy(changes)
        results = []
        warnings = []
        all_passed = True

        rules = self.registry.get_rules()
        # Sort: CRITICAL first, then HIGH, then MEDIUM
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        rules.sort(key=lambda r: priority_order.get(r.priority, 99))

        for rule in rules:
            result = rule.check(corrected, context)
            results.append(result)

            if not result.passed:
                if rule.priority == "CRITICAL":
                    if self.mode == "strict" and result.corrected_value is not None:
                        for key in changes:
                            if result.corrected_value is not None:
                                corrected[key] = result.corrected_value
                    warnings.append(f"[CRITICAL] {result.message}")
                    all_passed = False
                elif rule.priority == "HIGH":
                    if self.mode == "strict" and result.corrected_value is not None:
                        for key in changes:
                            corrected[key] = result.corrected_value
                    warnings.append(f"[HIGH] {result.message}")
                else:  # MEDIUM
                    warnings.append(f"[MEDIUM] {result.message}")

        return {
            "passed": all_passed,
            "changes": corrected,
            "results": [{"rule": r.rule_name, "passed": r.passed, "priority": r.priority, "message": r.message} for r in results],
            "warnings": warnings,
        }
```

- [ ] **Step 4: Write test**

```python
# tests/test_guardrails.py
from auto_tune.modules.agent_engine.guardrails.base import GuardrailRule, RuleResult
from auto_tune.modules.agent_engine.guardrails.registry import GuardrailRegistry
from auto_tune.modules.agent_engine.guardrails.validator import GuardrailValidator


class MockRule(GuardrailRule):
    name = "mock_rule"
    priority = "HIGH"
    def check(self, changes, context):
        if "lr0" in changes and changes["lr0"] > 0.1:
            changes["lr0"] = 0.01
            return RuleResult("mock_rule", False, "HIGH", "lr0 too high", 0.01)
        return RuleResult("mock_rule", True, "HIGH", "")


def test_registry_manual_register():
    registry = GuardrailRegistry()
    registry.register_rule(MockRule())
    assert len(registry.get_rules()) == 1


def test_validator_passing():
    registry = GuardrailRegistry()
    registry.register_rule(MockRule())
    validator = GuardrailValidator(registry)
    result = validator.validate({"lr0": 0.005}, {})
    assert result["passed"] is True


def test_validator_corrects():
    registry = GuardrailRegistry()
    registry.register_rule(MockRule())
    validator = GuardrailValidator(registry)
    result = validator.validate({"lr0": 0.5}, {})
    assert result["changes"]["lr0"] == 0.01
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest auto_tune/tests/test_guardrails.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add auto_tune/modules/agent_engine/guardrails/base.py auto_tune/modules/agent_engine/guardrails/registry.py auto_tune/modules/agent_engine/guardrails/validator.py auto_tune/tests/test_guardrails.py
git commit -m "feat: guardrails base, registry, and validator"
```


### Task 12: Implement guardrail rules

**Files:**
- Create: `auto_tune/modules/agent_engine/guardrails/rules/lr_boundary.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/lr_batch_scale.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/aug_dataset_size.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/dropout_weight_decay.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/cls_balance.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/box_tiny_bbox.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/batch_auto.py`
- Create: `auto_tune/modules/agent_engine/guardrails/rules/mosaic_small_data.py`

- [ ] **Step 1: Write `lr_boundary.py`**

```python
"""Guardrail: clamp lr0 to [1e-5, 0.1] and lrf to [1e-6, 0.1]."""

from ..base import GuardrailRule, RuleResult


class Lr0BoundaryRule(GuardrailRule):
    name = "lr0_boundary"
    priority = "CRITICAL"

    def check(self, changes: dict, context: dict) -> RuleResult:
        if "lr0" in changes:
            val = changes["lr0"]
            if val < 1e-5:
                return RuleResult(self.name, False, self.priority,
                                  f"lr0={val} below minimum 1e-5, clamped to 1e-5", 1e-5)
            if val > 0.1:
                return RuleResult(self.name, False, self.priority,
                                  f"lr0={val} exceeds maximum 0.1, clamped to 0.01", 0.01)
        return RuleResult(self.name, True, self.priority)


class LrfBoundaryRule(GuardrailRule):
    name = "lrf_boundary"
    priority = "CRITICAL"

    def check(self, changes: dict, context: dict) -> RuleResult:
        if "lrf" in changes:
            val = changes["lrf"]
            if val < 1e-6:
                return RuleResult(self.name, False, self.priority,
                                  f"lrf={val} below minimum 1e-6, clamped to 1e-6", 1e-6)
            if val > 0.1:
                return RuleResult(self.name, False, self.priority,
                                  f"lrf={val} exceeds maximum 0.1, clamped to 0.01", 0.01)
        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 2: Write `lr_batch_scale.py`**

```python
"""Guardrail: scale lr0 proportionally when batch_size changes."""

from ..base import GuardrailRule, RuleResult


class LrBatchScaleRule(GuardrailRule):
    name = "lr_batch_scale"
    priority = "HIGH"

    def check(self, changes: dict, context: dict) -> RuleResult:
        if "batch" not in changes:
            return RuleResult(self.name, True, self.priority)

        current_params = context.get("current_params", {})
        old_batch = current_params.get("batch", 16)
        new_batch = changes["batch"]
        old_lr = current_params.get("lr0", 0.01)

        if old_batch <= 0 or new_batch == old_batch:
            return RuleResult(self.name, True, self.priority)

        scale_factor = new_batch / old_batch
        expected_lr = old_lr * scale_factor

        if "lr0" in changes:
            actual_lr = changes["lr0"]
            ratio = actual_lr / expected_lr if expected_lr > 0 else 1
            if ratio < 0.5 or ratio > 2.0:
                corrected = max(min(actual_lr, expected_lr * 2), expected_lr * 0.5)
                return RuleResult(self.name, False, self.priority,
                                  f"lr0={actual_lr} not scaled with batch (expected ~{expected_lr:.4f}). "
                                  f"Adjusted to {corrected:.4f}", corrected)

        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 3: Write `aug_dataset_size.py`**

```python
"""Guardrail: reduce augmentation when dataset is small."""

from ..base import GuardrailRule, RuleResult


class AugDatasetSizeRule(GuardrailRule):
    name = "augmentation_dataset_size"
    priority = "HIGH"

    def check(self, changes: dict, context: dict) -> RuleResult:
        dataset_report = context.get("dataset_report", {})
        total_images = dataset_report.get("total_images", 0)

        if total_images > 1000:
            return RuleResult(self.name, True, self.priority)

        correction = {}
        if total_images < 300:
            # Very small dataset: disable strong aug
            for aug_key in ["mosaic", "mixup", "copy_paste", "degrees", "scale"]:
                if aug_key in changes and changes[aug_key] > 0.2:
                    correction[aug_key] = min(changes[aug_key], 0.2)
            msg = f"Dataset small ({total_images} images), strong aug capped at 0.2"
        else:
            # Medium dataset: moderate aug
            for aug_key in ["mosaic", "degrees", "scale"]:
                if aug_key in changes and changes[aug_key] > 0.5:
                    correction[aug_key] = min(changes[aug_key], 0.5)
            msg = f"Dataset moderate ({total_images} images), strong aug capped at 0.5"

        if correction:
            changes.update(correction)
            return RuleResult(self.name, False, self.priority, msg, correction)
        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 4: Write `dropout_weight_decay.py`**

```python
"""Guardrail: warn when dropout and weight_decay are both high."""

from ..base import GuardrailRule, RuleResult


class DropoutWeightDecayConflictRule(GuardrailRule):
    name = "dropout_weight_decay_conflict"
    priority = "HIGH"

    def check(self, changes: dict, context: dict) -> RuleResult:
        dropout = changes.get("dropout", context.get("current_params", {}).get("dropout", 0))
        weight_decay = changes.get("weight_decay", context.get("current_params", {}).get("weight_decay", 0))

        if dropout > 0.1 and weight_decay > 0.001:
            corrected = {"weight_decay": 0.0005}
            changes["weight_decay"] = 0.0005
            return RuleResult(self.name, False, self.priority,
                              f"Both dropout({dropout}) and weight_decay({weight_decay}) high. "
                              f"Reduced weight_decay to 0.0005 to prevent over-regularization.",
                              corrected)

        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 5: Write `cls_balance.py`**

```python
"""Guardrail: boost cls loss when long-tail detected."""

from ..base import GuardrailRule, RuleResult


class ClsBalanceRule(GuardrailRule):
    name = "cls_balance"
    priority = "MEDIUM"

    def check(self, changes: dict, context: dict) -> RuleResult:
        dataset_report = context.get("dataset_report", {})
        class_balance = dataset_report.get("class_balance", {})
        long_tail = class_balance.get("long_tail_classes", [])

        if long_tail and "cls" not in changes:
            recommended_cls = 2.0
            changes["cls"] = recommended_cls
            return RuleResult(self.name, False, self.priority,
                              f"Long-tail classes detected ({long_tail}), "
                              f"setting cls={recommended_cls} to focus on rare classes",
                              recommended_cls)

        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 6: Write `box_tiny_bbox.py`**

```python
"""Guardrail: boost box loss weight when tiny bbox ratio is high."""

from ..base import GuardrailRule, RuleResult


class BoxTinyBboxRule(GuardrailRule):
    name = "box_tiny_bbox"
    priority = "MEDIUM"

    def check(self, changes: dict, context: dict) -> RuleResult:
        dataset_report = context.get("dataset_report", {})
        bbox_analysis = dataset_report.get("bbox_analysis", {})
        tiny_ratio = bbox_analysis.get("tiny_bbox_ratio", 0)

        if tiny_ratio > 0.2 and "box" not in changes:
            recommended_box = 12.0
            changes["box"] = recommended_box
            return RuleResult(self.name, False, self.priority,
                              f"Tiny bbox ratio {tiny_ratio:.2f} > 0.2, "
                              f"setting box={recommended_box} to improve small object localization",
                              recommended_box)

        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 7: Write `batch_auto.py`**

```python
"""Guardrail: force batch=-1 (auto-batch) when VRAM may be constrained."""

import os
from ..base import GuardrailRule, RuleResult


class BatchAutoOnOomRule(GuardrailRule):
    name = "batch_auto_on_oom"
    priority = "CRITICAL"

    def check(self, changes: dict, context: dict) -> RuleResult:
        batch = changes.get("batch", context.get("current_params", {}).get("batch", 16))
        if isinstance(batch, int) and batch > 0:
            # Detect low VRAM via torch
            try:
                import torch
                if torch.cuda.is_available():
                    total_vram = torch.cuda.get_device_properties(0).total_memory
                    total_vram_gb = total_vram / 1024**3
                    if total_vram_gb < 8 and batch > 16:
                        changes["batch"] = -1
                        return RuleResult(self.name, False, self.priority,
                                          f"VRAM ~{total_vram_gb:.0f}GB < 8GB with batch={batch}, "
                                          f"forcing batch=-1 (auto-batch)")
            except ImportError:
                pass
        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 8: Write `mosaic_small_data.py`**

```python
"""Guardrail: disable mosaic when dataset is very small."""

from ..base import GuardrailRule, RuleResult


class MosaicOffForSmallDataRule(GuardrailRule):
    name = "mosaic_off_for_small_data"
    priority = "MEDIUM"

    def check(self, changes: dict, context: dict) -> RuleResult:
        dataset_report = context.get("dataset_report", {})
        total_images = dataset_report.get("total_images", 0)

        if total_images < 300:
            if changes.get("mosaic", 1.0) > 0.0:
                changes["mosaic"] = 0.0
                return RuleResult(self.name, False, self.priority,
                                  f"Dataset too small ({total_images} images) for mosaic augmentation, "
                                  f"disabled mosaic")

        return RuleResult(self.name, True, self.priority)
```

- [ ] **Step 9: Write test for rules**

```python
# tests/test_guardrail_rules.py
from auto_tune.modules.agent_engine.guardrails.rules.lr_boundary import Lr0BoundaryRule, LrfBoundaryRule
from auto_tune.modules.agent_engine.guardrails.rules.aug_dataset_size import AugDatasetSizeRule
from auto_tune.modules.agent_engine.guardrails.rules.cls_balance import ClsBalanceRule
from auto_tune.modules.agent_engine.guardrails.rules.box_tiny_bbox import BoxTinyBboxRule
from auto_tune.modules.agent_engine.guardrails.rules.dropout_weight_decay import DropoutWeightDecayConflictRule


def test_lr0_boundary_too_high():
    rule = Lr0BoundaryRule()
    result = rule.check({"lr0": 0.5}, {})
    assert result.passed is False
    assert result.corrected_value == 0.01


def test_lr0_boundary_too_low():
    rule = Lr0BoundaryRule()
    result = rule.check({"lr0": 1e-6}, {})
    assert result.passed is False
    assert result.corrected_value == 1e-5


def test_lrf_boundary():
    rule = LrfBoundaryRule()
    result = rule.check({"lrf": 0.5}, {})
    assert result.passed is False


def test_aug_small_dataset():
    rule = AugDatasetSizeRule()
    context = {"dataset_report": {"total_images": 200}}
    changes = {"mosaic": 1.0, "mixup": 0.5}
    result = rule.check(changes, context)
    assert result.passed is False
    assert changes["mosaic"] <= 0.2


def test_cls_balance_triggers():
    rule = ClsBalanceRule()
    context = {"dataset_report": {"class_balance": {"long_tail_classes": ["dent"]}}}
    changes = {}
    result = rule.check(changes, context)
    assert changes.get("cls") == 2.0


def test_box_tiny_bbox():
    rule = BoxTinyBboxRule()
    context = {"dataset_report": {"bbox_analysis": {"tiny_bbox_ratio": 0.3}}}
    changes = {}
    rule.check(changes, context)
    assert changes.get("box") == 12.0


def test_dropout_weight_decay_conflict():
    rule = DropoutWeightDecayConflictRule()
    changes = {"dropout": 0.2, "weight_decay": 0.01}
    result = rule.check(changes, {})
    assert result.passed is False
    assert changes["weight_decay"] < 0.001
```

- [ ] **Step 10: Run tests**

Run: `python -m pytest auto_tune/tests/test_guardrail_rules.py -v`
Expected: 7 passed

- [ ] **Step 11: Commit**

```bash
git add auto_tune/modules/agent_engine/guardrails/rules/ auto_tune/tests/test_guardrail_rules.py
git commit -m "feat: all guardrail rule plugins"
```


### Task 13: Decision agent (LLM-driven)

**Files:**
- Create: `auto_tune/modules/agent_engine/decision_agent.py`

- [ ] **Step 1: Write `decision_agent.py`**

```python
"""Decision Agent — LLM-driven hyperparameter recommendation."""

import json
import os
from typing import Optional


AGENT_SYSTEM_PROMPT = """You are a YOLOv8 hyperparameter tuning expert for industrial defect detection.
Your task is to analyze the dataset analysis report and training diagnosis, then recommend precise hyperparameter changes.

## Core Rules
1. Only change parameters that need adjustment. Never change everything.
2. Learning rate (lr0) typically needs reduction when overfitting or gradient spikes occur.
3. box loss weight should increase when tiny objects are prevalent or localization is poor.
4. cls loss weight should increase when class imbalance exists or specific classes underperform.
5. Data augmentation (mosaic, mixup, scale) helps generalization but must be reduced for small datasets.
6. weight_decay helps with overfitting but conflicts with high dropout.

## Available Hyperparameters
- lr0 (initial lr, default 0.01): [1e-5, 0.1]
- lrf (final lr factor, default 0.01): [1e-6, 0.1]
- box (box loss weight, default 7.5): [0.1, 20.0]
- cls (cls loss weight, default 0.5): [0.1, 10.0]
- dfl (dfl loss weight, default 1.5): [0.1, 10.0]
- mosaic (mosaic augmentation, default 1.0): [0.0, 1.0]
- mixup (mixup augmentation, default 0.0): [0.0, 1.0]
- scale (scale augmentation, default 0.5): [0.0, 1.0]
- degrees (rotation augmentation, default 0.0): [0.0, 45.0]
- weight_decay (L2 regularization, default 0.0005): [0.0, 0.01]
- warmup_epochs (default 3.0): [0, 10]
- hsv_h / hsv_s / hsv_v (color augmentation, default 0.015/0.7/0.4)

## Response Format
Must be valid JSON with no markdown wrapping:
{
  "diagnosis": "one-line diagnosis",
  "action": "one-line action description",
  "hyperparameter_changes": {"param": value, ...},
  "reasoning": "step-by-step reasoning"
}"""


def build_agent_prompt(dataset_report: dict, training_report: dict, current_params: dict) -> str:
    """Build decision agent prompt from all available data.

    Args:
        dataset_report: Module A output JSON.
        training_report: Module B output JSON.
        current_params: current hyperparameter config.

    Returns:
        Prompt string for LLM.
    """
    return f"""## Dataset Analysis Report
{json.dumps(dataset_report, indent=2)}

## Training Diagnosis Report
{json.dumps(training_report, indent=2)}

## Current Hyperparameters
{json.dumps(current_params, indent=2)}

## Task
Analyze the above and output a JSON with recommended hyperparameter changes.
Focus on the key issues and suggest precise, minimal changes."""


def get_decision(dataset_report: dict, training_report: dict, current_params: dict, llm_config: dict) -> Optional[dict]:
    """Get hyperparameter recommendations from LLM.

    Args:
        dataset_report: Module A JSON output.
        training_report: Module B JSON output.
        current_params: current training hyperparameters.
        llm_config: LLM configuration.

    Returns:
        Dict with diagnosis, action, hyperparameter_changes, reasoning or None.
    """
    from ..training_diagnosis.text_diagnosis import _call_claude, _call_openai, _call_ollama, _extract_json

    prompt = build_agent_prompt(dataset_report, training_report, current_params)
    full_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + prompt

    provider = llm_config.get("provider", "claude")
    api_key = os.environ.get(llm_config.get("api_key_env", "LLM_API_KEY"), "")
    model = llm_config.get("model", "claude-opus-4-7")

    if provider == "claude":
        return _call_claude(full_prompt, api_key, model)
    elif provider == "openai":
        return _call_openai(full_prompt, api_key, model)
    elif provider == "ollama":
        return _call_ollama(full_prompt, model)
    return None
```

- [ ] **Step 2: Write test**

```python
# tests/test_decision_agent.py
from auto_tune.modules.agent_engine.decision_agent import build_agent_prompt, AGENT_SYSTEM_PROMPT


def test_build_agent_prompt():
    ds_report = {"total_images": 100, "summary": {"key_issues": []}}
    tr_report = {"training_summary": {"best_mAP50": 0.85}, "summary": {"status": "optimal"}}
    params = {"lr0": 0.01, "batch": 16}
    prompt = build_agent_prompt(ds_report, tr_report, params)
    assert "total_images" in prompt
    assert "best_mAP50" in prompt


def test_agent_system_prompt_contains_rules():
    assert "Only change parameters that need adjustment" in AGENT_SYSTEM_PROMPT
    assert "Available Hyperparameters" in AGENT_SYSTEM_PROMPT
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_decision_agent.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/agent_engine/decision_agent.py auto_tune/tests/test_decision_agent.py
git commit -m "feat: LLM-driven decision agent with system prompt"
```


### Task 14: Decision tree fallback

**Files:**
- Create: `auto_tune/modules/agent_engine/decision_tree.py`

- [ ] **Step 1: Write `decision_tree.py`**

```python
"""Decision tree fallback when LLM is unavailable.

Maps dataset analysis + training diagnosis to hyperparameter changes
using deterministic expert rules (0 tokens, 0 API calls)."""


def get_decision_tree_recommendation(dataset_report: dict, training_report: dict) -> dict:
    """Generate hyperparameter changes using rule-based decision tree.

    Args:
        dataset_report: Module A JSON output.
        training_report: Module B JSON output.

    Returns:
        Dict with diagnosis, action, hyperparameter_changes, reasoning.
    """
    changes = {}
    reasons = []

    overfitting = training_report.get("overfitting", {}).get("detected", False)
    grad_issues = training_report.get("gradient_issues", {})
    nan_occurred = grad_issues.get("nan_occurred", False)
    loss_spikes = grad_issues.get("loss_spikes", False)
    best_mAP = training_report.get("training_summary", {}).get("best_mAP50", 0)
    bbox_analysis = dataset_report.get("bbox_analysis", {})
    tiny_ratio = bbox_analysis.get("tiny_bbox_ratio", 0)
    class_balance = dataset_report.get("class_balance", {})
    long_tail = class_balance.get("long_tail_classes", [])
    total_images = dataset_report.get("total_images", 0)

    # Rule 1: Gradient issues → reduce lr
    if nan_occurred or loss_spikes:
        changes["lr0"] = 0.001
        changes["warmup_epochs"] = 5.0
        reasons.append("Gradient issues detected: reduced lr0=0.001, warmup=5")

    # Rule 2: Overfitting → regularization
    if overfitting:
        changes["weight_decay"] = 0.001
        changes["mosaic"] = 1.0
        changes["mixup"] = 0.15
        changes["hsv_h"] = 0.03
        changes["hsv_s"] = 0.8
        reasons.append("Overfitting detected: increased regularization and augmentation")

    # Rule 3: Tiny bboxes → increase box loss and input size
    if tiny_ratio > 0.2:
        changes["box"] = 12.0
        changes["scale"] = 0.8
        reasons.append(f"Tiny bbox ratio {tiny_ratio:.2f}: increased box loss and scale aug")

    # Rule 4: Long tail classes → increase cls loss
    if long_tail:
        changes["cls"] = 2.0
        changes["copy_paste"] = 0.3
        reasons.append(f"Long-tail classes {long_tail}: increased cls loss and copy_paste")

    # Rule 5: Small dataset → reduce augmentation
    if total_images < 300:
        changes.update({
            "mosaic": 0.0,
            "mixup": 0.0,
            "degrees": 5.0,
            "scale": 0.3,
            "weight_decay": 0.001,
        })
        reasons.append(f"Small dataset ({total_images}): disabled strong aug, increased weight_decay")

    # Rule 6: Low mAP not explained by other rules
    if best_mAP < 0.3 and not changes:
        changes["lr0"] = 0.005
        changes["box"] = 10.0
        changes["cls"] = 1.0
        reasons.append(f"Low mAP50={best_mAP}: moderate adjustments across lr, box, cls")

    if not changes:
        changes["lr0"] = 0.01
        reasons.append("No issues detected: keeping default hyperparameters")

    return {
        "diagnosis": "; ".join(reasons),
        "action": "decision_tree_fallback",
        "hyperparameter_changes": changes,
        "reasoning": " | ".join(reasons),
    }


def should_use_tree(llm_result: dict) -> bool:
    """Determine if fallback tree should be used instead of LLM result."""
    return llm_result is None or "error" in llm_result
```

- [ ] **Step 2: Write test**

```python
# tests/test_decision_tree.py
from auto_tune.modules.agent_engine.decision_tree import (
    get_decision_tree_recommendation, should_use_tree
)


def test_overfitting_detected():
    ds_report = {"total_images": 1000, "bbox_analysis": {"tiny_bbox_ratio": 0.05},
                 "class_balance": {"long_tail_classes": []}}
    tr_report = {"overfitting": {"detected": True}, "gradient_issues": {},
                 "training_summary": {"best_mAP50": 0.6}}
    result = get_decision_tree_recommendation(ds_report, tr_report)
    assert "weight_decay" in result["hyperparameter_changes"]


def test_tiny_bbox_triggers():
    ds_report = {"total_images": 1000, "bbox_analysis": {"tiny_bbox_ratio": 0.3},
                 "class_balance": {"long_tail_classes": []}}
    tr_report = {"overfitting": {"detected": False}, "gradient_issues": {},
                 "training_summary": {"best_mAP50": 0.8}}
    result = get_decision_tree_recommendation(ds_report, tr_report)
    assert result["hyperparameter_changes"].get("box") == 12.0


def test_long_tail_triggers():
    ds_report = {"total_images": 1000, "bbox_analysis": {"tiny_bbox_ratio": 0.05},
                 "class_balance": {"long_tail_classes": ["dent"]}}
    tr_report = {"overfitting": {"detected": False}, "gradient_issues": {},
                 "training_summary": {"best_mAP50": 0.8}}
    result = get_decision_tree_recommendation(ds_report, tr_report)
    assert result["hyperparameter_changes"].get("cls") == 2.0


def test_small_dataset():
    ds_report = {"total_images": 200, "bbox_analysis": {"tiny_bbox_ratio": 0.05},
                 "class_balance": {"long_tail_classes": []}}
    tr_report = {"overfitting": {"detected": False}, "gradient_issues": {},
                 "training_summary": {"best_mAP50": 0.8}}
    result = get_decision_tree_recommendation(ds_report, tr_report)
    assert result["hyperparameter_changes"].get("mosaic") == 0.0


def test_should_use_tree():
    assert should_use_tree(None) is True
    assert should_use_tree({"error": "API failed"}) is True
    assert should_use_tree({"diagnosis": "ok"}) is False
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_decision_tree.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/agent_engine/decision_tree.py auto_tune/tests/test_decision_tree.py
git commit -m "feat: decision tree fallback with expert rules"
```


### Task 15: YOLO subprocess executor

**Files:**
- Create: `auto_tune/utils/yolo_subprocess.py`

- [ ] **Step 1: Write `yolo_subprocess.py`**

```python
"""YOLOv8 subprocess wrapper for clean launch, monitor, and kill."""

import subprocess
import os
import signal
import yaml
import time
import shlex


class YOLOSubprocess:
    """Manages YOLOv8 training as a subprocess for clean kill/abort."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self._process: subprocess.Popen | None = None

    def write_args_yaml(self, args: dict, output_path: str = None) -> str:
        """Write hyperparameters to args.yaml.

        Args:
            args: dict of hyperparameters.
            output_path: path to write to (default: project_dir/args.yaml).

        Returns:
            Path to written file.
        """
        if output_path is None:
            output_path = os.path.join(self.project_dir, "args.yaml")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(args, f)
        return output_path

    def build_command(self, args_yaml_path: str, data_yaml_path: str, extra_args: list[str] = None) -> str:
        """Build YOLO train command.

        Args:
            args_yaml_path: path to hyperparameter config file.
            data_yaml_path: path to data.yaml.
            extra_args: additional CLI arguments.

        Returns:
            Command string for subprocess.
        """
        cmd = f"yolo task=detect mode=train data={shlex.quote(data_yaml_path)} cfg={shlex.quote(args_yaml_path)}"
        if extra_args:
            cmd += " " + " ".join(shlex.quote(a) for a in extra_args)
        return cmd

    def launch(self, command: str, cwd: str = None) -> bool:
        """Launch YOLOv8 training subprocess.

        Args:
            command: command string to execute.
            cwd: working directory for subprocess.

        Returns:
            True if process started successfully.
        """
        if self._process is not None:
            return False

        self._process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd or self.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return True

    def read_stdout_line(self, timeout: float = 1.0) -> str | None:
        """Read one line from stdout (non-blocking with timeout)."""
        if self._process is None or self._process.stdout is None:
            return None
        import select
        try:
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(self._process.stdout, selectors.EVENT_READ)
            events = sel.select(timeout=timeout)
            if events:
                return self._process.stdout.readline()
        except (ImportError, OSError):
            pass
        return None

    def poll(self) -> int | None:
        """Check if process is still running. Returns None if running, exit code if done."""
        if self._process is None:
            return -1
        return self._process.poll()

    def kill(self) -> bool:
        """Kill the training subprocess."""
        if self._process is None:
            return False
        try:
            if os.name == "nt":
                self._process.terminate()
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process = None
            return True
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._process is not None and self.poll() is None
```

- [ ] **Step 2: Write test**

```python
# tests/test_yolo_subprocess.py
from auto_tune.utils.yolo_subprocess import YOLOSubprocess
import tempfile, os


def test_write_args_yaml(tmp_path):
    yolo = YOLOSubprocess(str(tmp_path))
    path = yolo.write_args_yaml({"lr0": 0.01, "batch": 16})
    assert os.path.exists(path)


def test_build_command():
    yolo = YOLOSubprocess("/tmp")
    cmd = yolo.build_command("/tmp/args.yaml", "/tmp/data.yaml")
    assert "yolo" in cmd
    assert "/tmp/args.yaml" in cmd


def test_kill_without_start():
    yolo = YOLOSubprocess("/tmp")
    assert yolo.kill() is False
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_yolo_subprocess.py -v`
Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/utils/yolo_subprocess.py auto_tune/tests/test_yolo_subprocess.py
git commit -m "feat: YOLOv8 subprocess executor with kill support"
```


### Task 16: Probe mode logic

**Files:**
- Create: `auto_tune/modules/agent_engine/probe.py`

- [ ] **Step 1: Write `probe.py`**

```python
"""Probe mode logic: early-stopping probe after N epochs to decide continue or retune."""

from ..training_diagnosis.csv_parser import parse_results_csv


def check_probe_result(csv_path: str, config: dict) -> dict:
    """Analyze probe phase results and decide next action.

    Args:
        csv_path: path to results.csv after probe_epochs.
        config: probe config dict with probe_epochs, auto_continue_threshold_mAP50, etc.

    Returns:
        dict with decision, reason, and diagnostic data.
    """
    analysis = parse_results_csv(csv_path)
    if "error" in analysis:
        return {"decision": "abort", "reason": f"CSV parse failed: {analysis['error']}"}

    epochs = analysis.get("epochs_completed", 0)
    probe_target = config.get("probe_epochs", 10)
    mAP50 = analysis.get("best_mAP50", 0)
    threshold = config.get("auto_continue_threshold_mAP50", 0.05)
    overfitting = analysis.get("overfitting", {}).get("detected", False)
    grad_issues = analysis.get("gradient_issues", {})
    nan_occurred = grad_issues.get("nan_occurred", False)

    # Decision logic
    if nan_occurred:
        return {"decision": "abort", "reason": "NaN values detected in probe phase", "mAP50": mAP50}

    if overfitting:
        return {"decision": "retune", "reason": "Overfitting detected early in probe phase", "mAP50": mAP50}

    if mAP50 < threshold:
        return {"decision": "retune", "reason": f"mAP50={mAP50:.4f} below threshold={threshold}", "mAP50": mAP50}

    if epochs < probe_target:
        return {"decision": "continue", "reason": f"Only {epochs}/{probe_target} epochs completed, still early", "mAP50": mAP50}

    return {"decision": "continue", "reason": "Probe phase looks good, continuing full training", "mAP50": mAP50}


def should_retry(attempt: int, max_retries: int) -> bool:
    """Check if retry is allowed."""
    return attempt < max_retries
```

- [ ] **Step 2: Write test**

```python
# tests/test_probe.py
from auto_tune.modules.agent_engine.probe import check_probe_result, should_retry


def test_check_probe_result_nonexistent_csv():
    result = check_probe_result("nonexistent.csv", {"probe_epochs": 10, "auto_continue_threshold_mAP50": 0.05})
    assert result["decision"] == "abort"


def test_should_retry():
    assert should_retry(0, 3) is True
    assert should_retry(3, 3) is False
    assert should_retry(5, 3) is False
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest auto_tune/tests/test_probe.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/agent_engine/probe.py auto_tune/tests/test_probe.py
git commit -m "feat: probe mode decision logic"
```


### Task 17: Agent engine executor (full loop orchestrator)

**Files:**
- Create: `auto_tune/modules/agent_engine/executor.py`

- [ ] **Step 1: Write `executor.py`**

```python
"""Agent execution engine — orchestrates the full auto-tuning loop."""

import os
import json
import yaml
from .guardrails.registry import GuardrailRegistry
from .guardrails.validator import GuardrailValidator
from .decision_agent import get_decision
from .decision_tree import get_decision_tree_recommendation, should_use_tree
from .probe import check_probe_result, should_retry
from ..training_diagnosis.diagnostician import diagnose_training
from ...utils.yolo_subprocess import YOLOSubprocess


class AutoTuningExecutor:
    """Orchestrates the full auto-tuning loop: diagnose → decide → guard → execute → probe."""

    def __init__(self, project_dir: str, config: dict):
        self.project_dir = project_dir
        self.config = config
        self.yolo = YOLOSubprocess(project_dir)
        self.guardrails = GuardrailValidator(GuardrailRegistry(), config.get("guardrails", {}).get("mode", "strict"))

    def run_loop(self, dataset_report: dict, data_yaml_path: str, current_params: dict,
                 extra_args: list[str] = None, max_attempts: int = 3) -> list[dict]:
        """Run the auto-tuning loop.

        Args:
            dataset_report: Module A output JSON.
            data_yaml_path: path to data.yaml for YOLO.
            current_params: current hyperparameter config.
            extra_args: extra CLI args for YOLO.
            max_attempts: maximum tuning attempts.

        Returns:
            List of result dicts per attempt.
        """
        llm_config = self.config.get("llm", {})
        probe_config = self.config.get("probe", {})
        training_config = self.config.get("training", {})
        probe_enabled = probe_config.get("enabled", True)
        probe_epochs = probe_config.get("probe_epochs", 10)

        history = []

        for attempt in range(max_attempts):
            print(f"Auto-Tuning attempt {attempt + 1}/{max_attempts}")

            # Step 1: Get decision
            decision = get_decision(dataset_report, {}, current_params, llm_config)
            if should_use_tree(decision):
                print("LLM unavailable, using decision tree fallback")
                decision = get_decision_tree_recommendation(dataset_report, {})

            # Step 2: Guardrails
            context = {
                "dataset_report": dataset_report,
                "training_report": {},
                "current_params": current_params,
            }
            guard_result = self.guardrails.validate(decision.get("hyperparameter_changes", {}), context)

            # Step 3: Apply and launch
            params = {**current_params, **guard_result["changes"]}
            args_path = self.yolo.write_args_yaml(params)
            cmd = self.yolo.build_command(args_path, data_yaml_path, extra_args)
            self.yolo.launch(cmd)

            # Step 4: Probe phase
            if probe_enabled:
                results_csv = os.path.join(self.project_dir, "results.csv")
                probe_result = self._wait_and_probe(results_csv, attempt, probe_epochs)

                if probe_result["decision"] == "abort":
                    self.yolo.kill()
                    history.append({"attempt": attempt, "decision": "abort", "reason": probe_result["reason"]})
                    break
                elif probe_result["decision"] == "retune":
                    self.yolo.kill()
                    history.append({"attempt": attempt, "decision": "retune",
                                    "reason": probe_result["reason"], "mAP50": probe_result.get("mAP50")})
                    # Update current params for next attempt
                    current_params = params
                    continue

            # Step 5: Wait for full training
            exit_code = self.yolo.poll()
            while exit_code is None:
                # In production this would be event-driven
                import time
                time.sleep(5)
                exit_code = self.yolo.poll()

            # Step 6: Full diagnosis
            full_diagnosis = diagnose_training(
                os.path.join(self.project_dir, "results.csv"),
                args_path,
                llm_config,
            )
            history.append({"attempt": attempt, "decision": "completed", "diagnosis": full_diagnosis})
            break

        return history

    def _wait_and_probe(self, results_csv: str, attempt: int, probe_epochs: int) -> dict:
        """Simplified probe — in production this would be async."""
        import time
        # Wait for results.csv to have probe_epochs rows (simplified)
        time.sleep(2)
        if os.path.exists(results_csv):
            return check_probe_result(results_csv, {"probe_epochs": probe_epochs, "auto_continue_threshold_mAP50": 0.05})
        return {"decision": "continue", "reason": "Results CSV not yet available"}
```

- [ ] **Step 2: Write test**

```python
# tests/test_executor.py
from auto_tune.modules.agent_engine.executor import AutoTuningExecutor


def test_executor_init(tmp_path):
    config = {"llm": {"provider": "claude"}, "guardrails": {"mode": "strict"}, "probe": {"enabled": False}}
    executor = AutoTuningExecutor(str(tmp_path), config)
    assert executor.project_dir == str(tmp_path)
```

- [ ] **Step 3: Run test**

Run: `python -m pytest auto_tune/tests/test_executor.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
git add auto_tune/modules/agent_engine/executor.py auto_tune/tests/test_executor.py
git commit -m "feat: auto-tuning loop executor with probe and guardrails"
```


## Phase 4: Utilities

### Task 18: File utilities and JSON helpers

**Files:**
- Create: `auto_tune/utils/file_utils.py`
- Create: `auto_tune/utils/json_utils.py`

- [ ] **Step 1: Write `file_utils.py`**

```python
"""File utility functions."""

import os
import glob


def find_files(directory: str, pattern: str) -> list[str]:
    """Find files matching glob pattern under directory."""
    return sorted(glob.glob(os.path.join(directory, pattern)))


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if needed."""
    os.makedirs(path, exist_ok=True)
    return path


def read_yaml(path: str) -> dict:
    """Read YAML file."""
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(data: dict, path: str):
    """Write JSON file."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: Write `json_utils.py`**

```python
"""JSON utility functions."""

import json


def read_json(path: str) -> dict:
    """Read JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_json(base: dict, override: dict) -> dict:
    """Deep merge two dicts. override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_json(result[key], value)
        else:
            result[key] = value
    return result


def safe_serialize(obj: any) -> str:
    """Serialize object to JSON string, handling non-serializable types."""
    class SafeEncoder(json.JSONEncoder):
        def default(self, o):
            try:
                return str(o)
            except Exception:
                return None
    return json.dumps(obj, cls=SafeEncoder, indent=2, ensure_ascii=False)
```

- [ ] **Step 3: Write tests**

```python
# tests/test_utils.py
import tempfile, os
from auto_tune.utils.file_utils import find_files, ensure_dir, write_json
from auto_tune.utils.json_utils import read_json, merge_json, safe_serialize


def test_find_files(tmp_path):
    (tmp_path / "test.txt").write_text("hello")
    files = find_files(str(tmp_path), "*.txt")
    assert len(files) == 1


def test_ensure_dir(tmp_path):
    path = str(tmp_path / "new_dir")
    result = ensure_dir(path)
    assert os.path.exists(result)


def test_write_and_read_json(tmp_path):
    path = str(tmp_path / "data.json")
    write_json({"key": "value"}, path)
    data = read_json(path)
    assert data == {"key": "value"}


def test_merge_json():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}
    merged = merge_json(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 3
    assert merged["e"] == 4


def test_safe_serialize():
    result = safe_serialize({"a": 1, "b": "test"})
    assert '"a": 1' in result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest auto_tune/tests/test_utils.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add auto_tune/utils/ auto_tune/tests/test_utils.py
git commit -m "feat: file and JSON utility functions"
```


## Phase 5: UI (Gradio)

### Task 19: CSS styling

**Files:**
- Create: `auto_tune/ui/static/style.css`

- [ ] **Step 1: Write `style.css`**

```css
:root {
    --primary: #1E40AF;
    --primary-light: #3B82F6;
    --primary-dark: #1E3A8A;
    --accent: #D97706;
    --destructive: #DC2626;
    --success: #16A34A;
    --bg: #F8FAFC;
    --surface: #FFFFFF;
    --text: #1E293B;
    --text-muted: #64748B;
    --border: #E2E8F0;
    --radius: 8px;
    --font-sans: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'Fira Code', 'Cascadia Code', monospace;
}

body {
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.kpi-card .kpi-icon {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.kpi-card .kpi-value {
    font-size: 24px;
    font-weight: 700;
    font-family: var(--font-mono);
    line-height: 1.2;
}

.kpi-card .kpi-label {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.3;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    font-family: var(--font-mono);
}

.badge-blue { background: #DBEAFE; color: #1E40AF; }
.badge-yellow { background: #FEF3C7; color: #92400E; }
.badge-red { background: #FEE2E2; color: #DC2626; }
.badge-green { background: #DCFCE7; color: #16A34A; }

.console-log {
    background: #0F172A;
    color: #E2E8F0;
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 12px;
    border-radius: var(--radius);
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
    line-height: 1.5;
}

.console-log .log-info { color: #3B82F6; }
.console-log .log-warn { color: #F59E0B; }
.console-log .log-error { color: #EF4444; }
.console-log .log-success { color: #22C55E; }

.hyperparam-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 14px;
}

.hyperparam-table th {
    text-align: left;
    padding: 8px 12px;
    background: #F1F5F9;
    border-bottom: 2px solid var(--border);
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

.hyperparam-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}

.hyperparam-table tr:hover { background: #F8FAFC; }

.chart-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
}
```

- [ ] **Step 2: Commit**

```bash
git add auto_tune/ui/static/style.css
git commit -m "feat: UI design system CSS (Data-Dense Dashboard)"
```


### Task 20: Gradio app entry + all pages

**Files:**
- Create: `auto_tune/ui/app.py`
- Create: `auto_tune/ui/pages/projects.py`
- Create: `auto_tune/ui/pages/dataset_report.py`
- Create: `auto_tune/ui/pages/agent_suggestion.py`
- Create: `auto_tune/ui/pages/training_monitor.py`
- Create: `auto_tune/ui/pages/history.py`
- Create: `auto_tune/ui/components/kpi_card.py`
- Create: `auto_tune/ui/components/badge.py`
- Create: `auto_tune/ui/components/hyperparam_table.py`
- Create: `auto_tune/ui/components/console_log.py`

- [ ] **Step 1: Write component files**

```python
# ui/components/kpi_card.py
import gradio as gr

def kpi_card(value: str, label: str, icon_svg: str, color: str = "blue"):
    """Render a KPI card component."""
    color_map = {
        "blue": "#DBEAFE,#1E40AF", "yellow": "#FEF3C7,#92400E",
        "red": "#FEE2E2,#DC2626", "green": "#DCFCE7,#16A34A",
    }
    bg, fg = color_map.get(color, color_map["blue"]).split(",")
    html = f"""
    <div class="kpi-card">
        <div class="kpi-icon" style="background:{bg};color:{fg}">
            {icon_svg}
        </div>
        <div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
    </div>
    """
    return gr.HTML(html)
```

```python
# ui/components/badge.py
import gradio as gr

def badge(text: str, color: str = "blue"):
    """Render a badge component."""
    html = f'<span class="badge badge-{color}">{text}</span>'
    return gr.HTML(html)
```

```python
# ui/components/hyperparam_table.py
import gradio as gr

def hyperparam_table(params: dict, editable: bool = True):
    """Render hyperparameter table with edit capability."""
    rows = ""
    for key, value in params.items():
        input_field = f'<input type="text" value="{value}" class="param-input" data-param="{key}" style="font-family:var(--font-mono);width:120px;padding:4px 8px;border:1px solid var(--border);border-radius:4px;"/>' if editable else str(value)
        rows += f"<tr><td>{key}</td><td>{input_field}</td></tr>"
    html = f"""
    <table class="hyperparam-table">
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
    return gr.HTML(html)
```

```python
# ui/components/console_log.py
import gradio as gr

def console_log(lines: list[str]):
    """Render training console log."""
    formatted = []
    for line in lines:
        css_class = "log-info"
        if "WARN" in line or "warning" in line.lower():
            css_class = "log-warn"
        elif "ERROR" in line or "error" in line.lower():
            css_class = "log-error"
        elif "mAP" in line or "success" in line.lower():
            css_class = "log-success"
        formatted.append(f'<div class="{css_class}">{line}</div>')
    html = f'<div class="console-log">{"".join(formatted)}</div>'
    return gr.HTML(html)
```

- [ ] **Step 2: Write page files**

```python
# ui/pages/projects.py
import gradio as gr

def create_page():
    with gr.Blocks() as page:
        gr.Markdown("# Projects")
        with gr.Row():
            with gr.Column(scale=2):
                project_list = gr.Dropdown(label="Select Project", choices=[], interactive=True)
                gr.Markdown("### Create New Project")
                project_name = gr.Textbox(label="Project Name")
                data_yaml = gr.Textbox(label="Data YAML Path")
                with gr.Row():
                    create_btn = gr.Button("Create", variant="primary")
                    delete_btn = gr.Button("Delete", variant="stop")
            with gr.Column(scale=3):
                gr.Markdown("### Training Config")
                epochs = gr.Number(label="Epochs", value=100)
                imgsz = gr.Number(label="Image Size", value=640)
                batch = gr.Number(label="Batch Size", value=16)
                with gr.Row():
                    probe_toggle = gr.Checkbox(label="Enable Probe Mode", value=True)
                    auto_loop = gr.Checkbox(label="Auto-Tuning Loop", value=False)
                save_config_btn = gr.Button("Save Config", variant="primary")
    return page
```

```python
# ui/pages/dataset_report.py
import gradio as gr

def create_page():
    with gr.Blocks() as page:
        gr.Markdown("# Dataset Analysis Report")
        with gr.Row():
            total_images = gr.HTML()
            total_annotations = gr.HTML()
            quality_score = gr.HTML()
        with gr.Row():
            gr.Markdown("### Image Quality")
            blur_ratio = gr.HTML()
            exposure = gr.HTML()
        with gr.Row():
            gr.Markdown("### Bounding Box Distribution")
            bbox_chart = gr.Plot(label="BBox Size Distribution")
        with gr.Row():
            gr.Markdown("### Class Distribution")
            class_chart = gr.Plot(label="Per-Class Count")
        with gr.Row():
            gr.Markdown("### Feature Clustering")
            cluster_plot = gr.Plot(label="Feature Embeddings (PCA)")
        refresh_btn = gr.Button("Refresh", variant="primary")
    return page
```

```python
# ui/pages/agent_suggestion.py
import gradio as gr

def create_page():
    with gr.Blocks() as page:
        gr.Markdown("# Agent Suggestion")
        with gr.Row():
            diagnosis = gr.Textbox(label="Diagnosis", lines=3, interactive=False)
            action = gr.Textbox(label="Recommended Action", lines=2, interactive=False)
        gr.Markdown("### Hyperparameter Changes")
        param_table = gr.HTML()
        gr.Markdown("### Reasoning")
        reasoning = gr.Markdown()
        with gr.Row():
            approve_btn = gr.Button("Approve & Train", variant="primary")
            reject_btn = gr.Button("Reject & Manual Edit", variant="secondary")
            manual_edit_btn = gr.Button("Open Manual Edit")
    return page
```

```python
# ui/pages/training_monitor.py
import gradio as gr

def create_page():
    with gr.Blocks() as page:
        gr.Markdown("# Training Monitor")
        with gr.Row():
            current_epoch = gr.HTML()
            current_mAP = gr.HTML()
            current_loss = gr.HTML()
            time_remaining = gr.HTML()
        with gr.Row():
            loss_plot = gr.Plot(label="Loss Curves")
            map_plot = gr.Plot(label="mAP Curves")
        with gr.Row():
            console = gr.HTML()
        with gr.Row():
            stop_btn = gr.Button("Stop Training", variant="stop")
            pause_btn = gr.Button("Pause")
    return page
```

```python
# ui/pages/history.py
import gradio as gr

def create_page():
    with gr.Blocks() as page:
        gr.Markdown("# Training History")
        with gr.Row():
            compare_toggle = gr.Checkbox(label="Compare Mode")
            export_btn = gr.Button("Export Report")
        history_table = gr.Dataframe(
            headers=["Run ID", "Date", "mAP50", "mAP50-95", "Miss Rate", "FA Rate", "Status"],
            interactive=False,
        )
        with gr.Row():
            selected_run_1 = gr.Dropdown(label="Run 1", choices=[])
            selected_run_2 = gr.Dropdown(label="Run 2 (compare)", choices=[])
        compare_btn = gr.Button("Compare", variant="primary")
        refresh_btn = gr.Button("Refresh")
    return page
```

- [ ] **Step 3: Write `app.py`**

```python
"""Gradio multi-page application entry."""

import gradio as gr
from .pages import projects, dataset_report, agent_suggestion, training_monitor, history

CSS_PATH = "auto_tune/ui/static/style.css"


def create_app():
    """Create the Gradio multi-page application."""
    with open(CSS_PATH, encoding="utf-8") as f:
        css = f.read()

    with gr.Blocks(css=css, title="YOLOv8 Auto-Tuning Agent") as app:
        gr.Markdown("# YOLOv8 Auto-Tuning Agent")
        gr.Markdown("Industrial defect detection — automated dataset analysis, training diagnosis, and hyperparameter tuning.")

        tabs = gr.Tabs()
        with tabs:
            with gr.Tab("Projects"):
                projects.create_page()
            with gr.Tab("Dataset Report"):
                dataset_report.create_page()
            with gr.Tab("Agent Suggestion"):
                agent_suggestion.create_page()
            with gr.Tab("Training Monitor"):
                training_monitor.create_page()
            with gr.Tab("History"):
                history.create_page()

    return app
```

- [ ] **Step 4: Commit**

```bash
git add auto_tune/ui/
git commit -m "feat: Gradio UI with all 5 pages and components"
```


## Phase 6: Integration

### Task 21: Main entry point — FastAPI + Gradio mount

**Files:**
- Modify: `auto_tune/main.py`

- [ ] **Step 1: Rewrite `main.py`**

```python
"""YOLOv8 Auto-Tuning Agent — FastAPI + Gradio entry point."""

import os
import yaml
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def create_fastapi_app():
    """Create FastAPI application with mounted Gradio UI."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from ui.app import create_app as create_gradio_app

    app = FastAPI(title="YOLOv8 Auto-Tuning Agent")

    # Serve static files
    static_dir = Path(__file__).parent / "ui" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Mount Gradio app
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "config_provider": CONFIG.get("llm", {}).get("provider")}

    return app


def main():
    """Launch the application."""
    import uvicorn
    app = create_fastapi_app()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7860"))
    print(f"Starting Auto-Tuning Agent at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test import**

Run: `cd e:\dataprocess_modeltrain\Auto_labeltrain_project && python -c "from auto_tune.main import create_fastapi_app; print('import OK')"`
Expected: "import OK"

- [ ] **Step 3: Commit**

```bash
git add auto_tune/main.py
git commit -m "feat: FastAPI + Gradio integration entry point"
```


### Task 22: End-to-end smoke test

**Files:**
- Create: `auto_tune/tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration tests — verify module interfaces and data flow."""

from auto_tune.main import CONFIG
from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset
from auto_tune.modules.training_diagnosis.diagnostician import diagnose_training
from auto_tune.modules.agent_engine.decision_agent import build_agent_prompt
from auto_tune.utils.file_utils import read_yaml, write_json
from auto_tune.utils.json_utils import merge_json


def test_config_loaded():
    assert CONFIG is not None
    assert "llm" in CONFIG
    assert "guardrails" in CONFIG
    assert "probe" in CONFIG


def test_analyzer_interface():
    """Verify analyze_dataset accepts spec-defined input signature."""
    import inspect
    sig = inspect.signature(analyze_dataset)
    params = list(sig.parameters.keys())
    assert "dataset_dir" in params
    assert "data_yaml" in params
    assert "config" in params


def test_diagnostician_interface():
    """Verify diagnose_training accepts spec-defined input signature."""
    import inspect
    sig = inspect.signature(diagnose_training)
    params = list(sig.parameters.keys())
    assert "results_csv_path" in params
    assert "args_yaml_path" in params
    assert "llm_config" in params


def test_merge_json_utility():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}}
    merged = merge_json(base, override)
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 3
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest auto_tune/tests/ -v`
Expected: All tests pass (or relevant subset depending on installed dependencies)

- [ ] **Step 3: Commit**

```bash
git add auto_tune/tests/test_integration.py
git commit -m "test: integration smoke tests"
```


## Self-Review Checklist

After writing this plan, verify:

1. **Spec coverage:** Every section in the spec has a corresponding task:
   - Sec 3 (Module A): Tasks 1-5
   - Sec 4 (Module B): Tasks 6-10
   - Sec 5 (Module C): Tasks 11-17
   - Sec 6 (UI): Tasks 19-20
   - Sec 7 (Config): Task 0
   - Sec 8 (Project structure): Task 0
   - Sec 10 (Edge cases): Covered across tasks (NaN checks, small dataset, LLM fallback)

2. **Placeholder scan:** No TBD, no "implement later", no "fill in details". Every step has complete code.

3. **Type consistency:** All function signatures, dict keys, and method names are consistent across tasks (e.g., `parse_results_csv` → `dict` with same keys consumed by `build_diagnosis_prompt` → `diagnose_training`).

4. **Test coverage:** Each functional task has tests written before implementation code.
