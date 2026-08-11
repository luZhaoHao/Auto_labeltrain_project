"""Class distribution analysis for YOLO datasets."""

from collections import Counter


def compute_class_distribution(label_files: list[str]) -> dict[int, int]:
    """Count class instances across all label files.

    Args:
        label_files: list of paths to YOLO-format .txt label files.

    Returns:
        dict mapping class_id -> count.
    """
    counter = Counter()
    for path in label_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
    return dict(counter)


def compute_class_balance(class_counts: dict[int, int], class_names: dict[int, str],
                          long_tail_ratio: float = 0.3) -> dict:
    """Compute class balance metrics.

    Args:
        class_counts: dict of class_id -> count.
        class_names: dict of class_id -> class_name.
        long_tail_ratio: fraction of mean count below which a class is long-tail (default 0.3).

    Returns:
        dict with per-class distribution, is_balanced flag, long_tail_classes list.
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

    threshold = total / len(class_counts) * long_tail_ratio
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
