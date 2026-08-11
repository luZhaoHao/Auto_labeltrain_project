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


def estimate_exposure(image: np.ndarray,
                      under_pixel_threshold: int = 50,
                      over_pixel_threshold: int = 200) -> dict:
    """Estimate exposure using grayscale histogram percentiles.

    Args:
        image: BGR image array.
        under_pixel_threshold: pixel value below this is underexposed (default 50).
        over_pixel_threshold: pixel value above this is overexposed (default 200).

    Returns:
        dict with 'under_exposure', 'over_exposure' ratios [0,1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_pixels = np.sum(gray < under_pixel_threshold) / gray.size
    high_pixels = np.sum(gray > over_pixel_threshold) / gray.size
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


def analyze_image(image: np.ndarray,
                  blur_threshold: float = 100.0,
                  under_pixel_threshold: int = 50,
                  over_pixel_threshold: int = 200) -> dict:
    """Run all image quality checks on a single image.

    Args:
        image: BGR image array.
        blur_threshold: threshold for blur detection (default 100.0).
        under_pixel_threshold: pixel value below this is underexposed (default 50).
        over_pixel_threshold: pixel value above this is overexposed (default 200).

    Returns:
        dict combining blur, exposure, and SNR results.
    """
    blur = estimate_blur(image, blur_threshold)
    exposure = estimate_exposure(image, under_pixel_threshold, over_pixel_threshold)
    snr = estimate_snr(image)
    return {**blur, **exposure, **snr}
