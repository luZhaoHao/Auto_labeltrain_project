"""Generate error crop images by comparing val_batch_labels vs val_batch_pred."""

import os
import cv2
import numpy as np


def _find_val_batch_pairs(run_dir: str) -> list[tuple[str, str]]:
    """Find matching (labels, pred) image pairs in run_dir.

    Returns list of (label_path, pred_path) tuples.
    """
    pairs = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("val_batch") and fname.endswith("_labels.jpg"):
            stem = fname.replace("_labels.jpg", "")
            pred_name = f"{stem}_pred.jpg"
            pred_path = os.path.join(run_dir, pred_name)
            if os.path.exists(pred_path):
                pairs.append((os.path.join(run_dir, fname), pred_path))
    return pairs


def _compute_diff_regions(labels_path: str, pred_path: str,
                          diff_thresh: int = 30,
                          min_area: int = 1600) -> list[tuple[int, int, int, int]]:
    """Compare label and pred images, return bounding boxes of diff regions.

    Args:
        labels_path: path to val_batchX_labels.jpg
        pred_path: path to val_batchX_pred.jpg
        diff_thresh: pixel difference threshold (0-255)
        min_area: minimum contour area to keep

    Returns:
        list of (x, y, w, h) bounding boxes in label image coordinates.
    """
    img_a = cv2.imread(labels_path)
    img_b = cv2.imread(pred_path)
    if img_a is None or img_b is None:
        return []

    h, w = img_a.shape[:2]

    # Ensure same size
    if img_b.shape != img_a.shape:
        img_b = cv2.resize(img_b, (w, h))

    # Absolute difference
    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, mask = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)

    # Light close to merge nearby pixels (no open — thin lines get erased)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= min_area:
            # Add small padding
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def generate_error_crops(run_dir: str, output_dir: str = None) -> str | None:
    """Generate composite error crop image from val_batch comparisons.

    Crops regions where predictions differ significantly from labels,
    resizes each to 224px height, and arranges them in a horizontal strip.

    Args:
        run_dir: training run directory containing val_batch images.
        output_dir: where to save the composite. Defaults to run_dir.

    Returns:
        Path to the composite image, or None if no differences found.
    """
    pairs = _find_val_batch_pairs(run_dir)
    if not pairs:
        return None

    all_crops = []
    for labels_path, pred_path in pairs:
        boxes = _compute_diff_regions(labels_path, pred_path)
        if not boxes:
            continue

        img = cv2.imread(labels_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        for x, y, bw, bh in boxes:
            crop = img[y:y + bh, x:x + bw]
            if crop.size == 0:
                continue
            # Resize to fixed height 224, keep aspect ratio
            aspect = crop.shape[1] / crop.shape[0]
            new_w = int(224 * aspect)
            crop_resized = cv2.resize(crop, (new_w, 224))
            all_crops.append(crop_resized)

    if not all_crops:
        return None

    # Limit to max 8 crops
    if len(all_crops) > 8:
        all_crops = all_crops[:8]

    # Stack horizontally with 10px gap
    gap = 10
    total_w = sum(c.shape[1] for c in all_crops) + gap * (len(all_crops) - 1)
    max_h = max(c.shape[0] for c in all_crops)
    composite = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255

    x_offset = 0
    for crop in all_crops:
        h_crop = crop.shape[0]
        y_offset = (max_h - h_crop) // 2
        composite[y_offset:y_offset + h_crop, x_offset:x_offset + crop.shape[1]] = crop
        x_offset += crop.shape[1] + gap

    # Save
    if output_dir is None:
        output_dir = run_dir
    out_path = os.path.join(output_dir, "error_crops.jpg")
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
