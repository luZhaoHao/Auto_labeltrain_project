"""Tests for bbox_geometry module."""

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


def test_compute_aspect_ratio_range():
    boxes = [
        {"x_center": 50, "y_center": 50, "width": 20, "height": 10, "area": 200, "aspect_ratio": 2.0, "class_id": 0},
        {"x_center": 50, "y_center": 50, "width": 10, "height": 20, "area": 200, "aspect_ratio": 0.5, "class_id": 1},
    ]
    result = compute_aspect_ratio_range(boxes)
    assert result[0] == 0.5
    assert result[1] == 2.0
