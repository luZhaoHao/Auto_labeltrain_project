"""Tests for analyzer orchestrator."""
import cv2
import numpy as np
from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset


def _dummy_image(path, w=100, h=100):
    """Create a small synthetic image for testing."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_analyze_dataset_no_images(tmp_path):
    data_yaml = {"names": {0: "crack"}, "train": str(tmp_path)}
    result = analyze_dataset(str(tmp_path), data_yaml, {"blur_threshold": 100.0})
    assert "error" in result


def test_analyze_dataset_output_structure():
    """Verify function signature matches expected interface."""
    import inspect
    sig = inspect.signature(analyze_dataset)
    assert "dataset_dir" in sig.parameters
    assert "data_yaml" in sig.parameters
    assert "config" in sig.parameters


def test_analyze_dataset_flat_layout(tmp_path):
    """Flat layout: images and .txt in same folder, mixed label coverage."""
    for i in range(5):
        _dummy_image(tmp_path / f"img{i:03d}.jpg")
    # img000: has 2 labels
    with open(tmp_path / "img000.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    # img001: empty txt
    with open(tmp_path / "img001.txt", "w") as f:
        f.write("")
    # img002: has 1 label
    with open(tmp_path / "img002.txt", "w") as f:
        f.write("0 0.5 0.5 0.3 0.3\n")
    # img003: empty txt
    with open(tmp_path / "img003.txt", "w") as f:
        f.write("")
    # img004: no txt at all

    data_yaml = {"names": {0: "crack", 1: "stain"}}
    config = {"blur_threshold": 100.0, "img_width": 100, "img_height": 100}
    result = analyze_dataset(str(tmp_path), data_yaml, config)

    assert "error" not in result
    assert result["label_coverage"]["total_images"] == 5
    assert result["label_coverage"]["with_labels"] == 2      # img000, img002
    assert result["label_coverage"]["empty_labels"] == 2     # img001, img003
    assert result["label_coverage"]["without_labels"] == 1   # img004
    assert result["label_coverage"]["label_rate"] == 0.4
    assert result["total_annotations"] == 3                  # 2 + 0 + 1 + 0 + 0


def test_analyze_dataset_flat_all_unlabeled(tmp_path):
    """Flat layout: images with no .txt files at all."""
    for i in range(3):
        _dummy_image(tmp_path / f"img{i}.jpg")

    data_yaml = {"names": {0: "crack"}}
    config = {"blur_threshold": 100.0}
    result = analyze_dataset(str(tmp_path), data_yaml, config)

    assert "error" not in result
    assert result["label_coverage"]["with_labels"] == 0
    assert result["label_coverage"]["empty_labels"] == 0
    assert result["label_coverage"]["without_labels"] == 3
    assert result["label_coverage"]["label_rate"] == 0.0
    assert result["total_annotations"] == 0
    # Image quality should still be analyzed
    assert "image_quality" in result


def test_analyze_dataset_structured_layout(tmp_path):
    """Structured layout still works (backward compat)."""
    img_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    _dummy_image(img_dir / "img001.jpg")
    _dummy_image(img_dir / "img002.jpg")

    with open(label_dir / "img001.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
    with open(label_dir / "img002.txt", "w") as f:
        f.write("")

    data_yaml = {"names": {0: "crack"}}
    config = {"blur_threshold": 100.0, "img_width": 100, "img_height": 100}
    result = analyze_dataset(str(tmp_path), data_yaml, config)

    assert "error" not in result
    assert result["total_images"] == 2
    assert result["label_coverage"]["with_labels"] == 1
    assert result["label_coverage"]["empty_labels"] == 1
    assert result["total_annotations"] == 1
