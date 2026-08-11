"""Dataset analyzer orchestrator — runs all analysis steps, produces JSON report.

Supports two directory layouts:
1. Structured: dataset_dir/images/train/ + dataset_dir/labels/train/
2. Flat: dataset_dir contains both images and .txt files (matched by stem)
"""

import os
import glob
import datetime
import cv2
import numpy as np
from .image_quality import analyze_image
from .bbox_geometry import (
    parse_yolo_label, compute_bbox_size_stats, compute_aspect_ratio_range,
    compute_overlap_analysis, compute_spatial_bias
)
from .class_stats import compute_class_distribution, compute_class_balance
from .feature_cluster import extract_features as _extract_features, cluster_outliers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: str) -> bool:
    """Check if file has an image extension."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _detect_layout(dataset_dir: str) -> tuple[str, bool]:
    """Detect layout type and return (actual_dataset_dir, is_flat)."""
    if os.path.isdir(os.path.join(dataset_dir, "images", "train")):
        return os.path.join(dataset_dir, "images", "train"), False
    if os.path.isdir(os.path.join(dataset_dir, "images")):
        return os.path.join(dataset_dir, "images"), False
    return dataset_dir, True


def _gather_flat(dataset_dir: str) -> tuple[list[str], dict[str, str | None]]:
    """In flat layout, gather images and match .txt labels by filename stem.

    Returns:
        img_paths: sorted image file paths.
        label_map: dict mapping image stem -> label path, or None if no .txt.
    """
    img_paths = []
    txt_stems = set()

    for fname in os.listdir(dataset_dir):
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem, ext = os.path.splitext(fname)
        if _is_image(fname):
            img_paths.append(fpath)
        elif ext.lower() == ".txt":
            txt_stems.add(stem)

    img_paths.sort()
    label_map = {}
    for path in img_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        label_map[stem] = os.path.join(dataset_dir, stem + ".txt") if stem in txt_stems else None
    return img_paths, label_map


def _get_image_size(img_path: str):
    """Read actual image dimensions. Returns (w, h) or None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def analyze_dataset(dataset_dir: str, data_yaml: dict, config: dict) -> dict:
    """Run full dataset analysis pipeline.

    Args:
        dataset_dir: path to dataset root.
            Structured: contains images/train/ and labels/train/.
            Flat:      contains images and .txt files directly.
        data_yaml: parsed data.yaml dict with 'names', 'train'/'val' paths.
        config: dict with analyzer thresholds.

    Returns:
        JSON-serializable dict per spec Section 3.3.
    """
    actual_dir, is_flat = _detect_layout(dataset_dir)
    class_names = {int(k): v for k, v in data_yaml.get("names", {}).items()}

    # Gather files based on layout
    if is_flat:
        img_paths, label_map = _gather_flat(actual_dir)
        label_files = [p for p in label_map.values() if p is not None]
        img_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
    else:
        img_paths = sorted(glob.glob(os.path.join(actual_dir, "*.*")))
        img_paths = [p for p in img_paths if _is_image(p)]
        label_dir = os.path.join(dataset_dir, "labels")
        if os.path.isdir(os.path.join(label_dir, "train")):
            label_dir = os.path.join(label_dir, "train")
        label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        img_by_stem = {}
        label_map = {}

    if not img_paths:
        return {"error": "No images found", "total_images": 0}

    # --- Label coverage breakdown (flat mode only has accurate counts) ---
    if is_flat:
        with_labels = empty_labels = 0
        for txt_path in label_map.values():
            if txt_path is None:
                continue
            with open(txt_path, encoding="utf-8") as f:
                if f.read().strip():
                    with_labels += 1
                else:
                    empty_labels += 1
        without_labels = len(img_paths) - with_labels - empty_labels
        label_coverage = {
            "total_images": len(img_paths),
            "with_labels": with_labels,
            "empty_labels": empty_labels,
            "without_labels": without_labels,
            "label_rate": round(with_labels / len(img_paths), 4),
        }
    else:
        empty_count = sum(1 for p in label_files if os.path.getsize(p) == 0)
        label_coverage = {
            "total_images": len(img_paths),
            "with_labels": len(label_files) - empty_count,
            "empty_labels": empty_count,
            "without_labels": max(0, len(img_paths) - len(label_files)),
            "label_rate": round((len(label_files) - empty_count) / len(img_paths), 4) if img_paths else 0.0,
        }

    # Step 1: Image quality analysis (sample-based for speed)
    quality_results = []
    size_cache = {}
    quality_sample_size = config.get("quality_sample_size", 500)
    sample_size = min(quality_sample_size, len(img_paths))
    rng = np.random.default_rng()
    sampled_paths = rng.choice(img_paths, sample_size, replace=False)
    blur_thresh = config.get("blur_threshold", 100.0)
    under_pixel = config.get("under_exposure_pixel_threshold", 50)
    over_pixel = config.get("over_exposure_pixel_threshold", 200)
    for path in sampled_paths:
        img = cv2.imread(str(path))
        if img is not None:
            quality_results.append(analyze_image(img,
                blur_threshold=blur_thresh,
                under_pixel_threshold=under_pixel,
                over_pixel_threshold=over_pixel))
            stem = os.path.splitext(os.path.basename(path))[0]
            size_cache[stem] = (img.shape[1], img.shape[0])

    under_ratio_thresh = config.get("under_exposure_ratio_threshold", 0.3)
    over_ratio_thresh = config.get("over_exposure_ratio_threshold", 0.3)
    snr_thresh = config.get("snr_threshold", 10)
    blur_ratio = np.mean([q["is_blurry"] for q in quality_results]) if quality_results else 0
    under_ratio = np.mean([q["under_exposure"] > under_ratio_thresh for q in quality_results]) if quality_results else 0
    over_ratio = np.mean([q["over_exposure"] > over_ratio_thresh for q in quality_results]) if quality_results else 0
    low_snr = np.mean([q["snr"] < snr_thresh for q in quality_results]) if quality_results else 0

    # Step 2: Parse labels and compute geometry (use real image dimensions in flat mode)
    default_w = config.get("img_width", 640)
    default_h = config.get("img_height", 640)
    img_area = default_w * default_h
    all_boxes = []
    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        if stem in size_cache:
            img_w, img_h = size_cache[stem]
        elif is_flat and stem in img_by_stem:
            size = _get_image_size(img_by_stem[stem])
            if size:
                img_w, img_h = size
                size_cache[stem] = (img_w, img_h)
            else:
                img_w, img_h = default_w, default_h
        else:
            img_w, img_h = default_w, default_h
        boxes = parse_yolo_label(label_path, img_width=img_w, img_height=img_h)
        all_boxes.extend(boxes)

    bbox_tiny = config.get("bbox_tiny_threshold", 0.01)
    bbox_small = config.get("bbox_small_threshold", 0.05)
    bbox_medium = config.get("bbox_medium_threshold", 0.2)
    bbox_stats = compute_bbox_size_stats(all_boxes, img_area,
        tiny_threshold=bbox_tiny, small_threshold=bbox_small, medium_threshold=bbox_medium)
    aspect_range = compute_aspect_ratio_range(all_boxes)
    overlap = compute_overlap_analysis(all_boxes, config.get("high_iou_threshold", 0.7))
    spatial_center_low = config.get("spatial_center_low", 0.25)
    spatial_center_high = config.get("spatial_center_high", 0.75)
    spatial_edge_low = config.get("spatial_edge_low", 0.1)
    spatial_edge_high = config.get("spatial_edge_high", 0.9)
    spatial = compute_spatial_bias(all_boxes,
        center_low=spatial_center_low, center_high=spatial_center_high,
        edge_low=spatial_edge_low, edge_high=spatial_edge_high)

    # Step 3: Class distribution
    class_counts = compute_class_distribution(label_files)
    long_tail_ratio = config.get("long_tail_ratio", 0.3)
    class_balance = compute_class_balance(class_counts, class_names, long_tail_ratio=long_tail_ratio)

    # Step 4: Feature clustering (sample-based)
    cluster_sample_size = config.get("cluster_sample_size", 200)
    cluster_sample = min(cluster_sample_size, len(img_paths))
    cluster_paths = rng.choice(img_paths, cluster_sample, replace=False)
    cluster_images = []
    for path in cluster_paths:
        img = cv2.imread(str(path))
        if img is not None:
            cluster_images.append(cv2.resize(img, (224, 224)))
    features = _extract_features(cluster_images)
    outliers = cluster_outliers(
        features,
        config.get("dbscan_eps", 0.3),
        config.get("dbscan_min_samples", 5),
    )

    # Step 5: Build summary
    warn_label_rate = config.get("warn_label_rate", 0.5)
    warn_tiny_bbox = config.get("warn_tiny_bbox_ratio", 0.2)
    warn_center = config.get("warn_center_concentration", 0.6)
    warn_blur = config.get("warn_blur_ratio", 0.1)
    key_issues = []
    if label_coverage["label_rate"] < warn_label_rate:
        key_issues.append("low_label_coverage")
    if not class_balance["is_balanced"]:
        for cls in class_balance["long_tail_classes"]:
            key_issues.append(f"long_tail_class_{cls}")
    if bbox_stats["tiny_bbox_ratio"] > warn_tiny_bbox:
        key_issues.append("tiny_bbox_high_ratio")
    if spatial["center_concentration_score"] > warn_center:
        key_issues.append("center_spatial_bias")
    if blur_ratio > warn_blur:
        key_issues.append("high_blur_ratio")

    blur_w = config.get("quality_blur_weight", 0.3)
    under_w = config.get("quality_under_weight", 0.2)
    over_w = config.get("quality_over_weight", 0.2)
    imbalance_w = config.get("quality_imbalance_weight", 0.3)
    quality_score = round(
        1.0 - (blur_ratio * blur_w + under_ratio * under_w + over_ratio * over_w
               + (1 - class_balance["is_balanced"]) * imbalance_w),
        2,
    )

    return {
        "module": "dataset_analyzer",
        "version": "1.0",
        "analysis_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "dataset_path": dataset_dir,
        "total_images": len(img_paths),
        "label_coverage": label_coverage,
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
            "dataset_quality_score": max(0.0, quality_score),
            "key_issues": key_issues,
        },
    }
