"""Tests for image_quality module."""

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
    # constant image: std=0 → snr=0.0 (division by zero guard)
    assert result["snr"] == 0.0


def test_analyze_image_returns_all_keys():
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = analyze_image(image)
    assert "laplacian_var" in result
    assert "under_exposure" in result
    assert "over_exposure" in result
    assert "snr" in result
