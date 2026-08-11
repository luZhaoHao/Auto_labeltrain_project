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


def compute_bbox_size_stats(boxes: list[dict], img_area: int,
                            tiny_threshold: float = 0.01,
                            small_threshold: float = 0.05,
                            medium_threshold: float = 0.2) -> dict:
    """Categorize bounding boxes by size relative to image area.

    Default categories: tiny (<0.01), small (0.01-0.05), medium (0.05-0.2), large (>0.2).

    Args:
        boxes: list of box dicts with 'area' key.
        img_area: total image area in pixels.
        tiny_threshold: max relative area for tiny (default 0.01).
        small_threshold: max relative area for small (default 0.05).
        medium_threshold: max relative area for medium (default 0.2).

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
        "tiny_bbox_ratio": float(np.mean(rel_areas < tiny_threshold)),
        "small_bbox_ratio": float(np.mean((rel_areas >= tiny_threshold) & (rel_areas < small_threshold))),
        "medium_bbox_ratio": float(np.mean((rel_areas >= small_threshold) & (rel_areas < medium_threshold))),
        "large_bbox_ratio": float(np.mean(rel_areas >= medium_threshold)),
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

    Returns dict with high_iou_ratio.
    """
    if len(boxes) < 2:
        return {"high_iou_ratio": 0.0}

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
    }


def compute_spatial_bias(boxes: list[dict],
                         center_low: float = 0.25, center_high: float = 0.75,
                         edge_low: float = 0.1, edge_high: float = 0.9) -> dict:
    """Analyze spatial distribution of box centers.

    Args:
        boxes: list of box dicts with 'x_center', 'y_center' (pixel coords).
        center_low: lower normalized bound for center region (default 0.25).
        center_high: upper normalized bound for center region (default 0.75).
        edge_low: lower normalized bound for edge detection (default 0.1).
        edge_high: upper normalized bound for edge detection (default 0.9).

    Returns dict with center_concentration_score and edge_distribution_ratio.
    """
    if not boxes:
        return {"center_concentration_score": 0.0, "edge_distribution_ratio": 0.0}

    centers = np.array([[b["x_center"], b["y_center"]] for b in boxes])
    cx, cy = centers[:, 0], centers[:, 1]
    center_mask = (cx > center_low) & (cx < center_high) & (cy > center_low) & (cy < center_high)
    edge_region_mask = (cx < edge_low) | (cx > edge_high) | (cy < edge_low) | (cy > edge_high)

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
