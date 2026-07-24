"""Tests for feature_cluster module."""

import numpy as np
from auto_tune.modules.dataset_analyzer.feature_cluster import (
    cluster_outliers
)


def test_cluster_outliers_few_samples():
    result = cluster_outliers(np.random.rand(2, 5))
    assert result["outlier_count"] == 0


def test_cluster_outliers_all_similar():
    features = np.tile(np.random.rand(1, 5), (20, 1))
    result = cluster_outliers(features, eps=0.5, min_samples=2)
    assert "outlier_count" in result
    assert "silhouette_score" in result
