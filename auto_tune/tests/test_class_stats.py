"""Tests for class_stats module."""

import tempfile
import os
from auto_tune.modules.dataset_analyzer.class_stats import (
    compute_class_distribution, compute_class_balance
)


def test_compute_class_distribution():
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
