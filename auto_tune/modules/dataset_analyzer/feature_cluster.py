"""Feature extraction and clustering for outlier detection."""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def extract_features(images: list[np.ndarray]) -> np.ndarray:
    """Extract feature vectors using HOG-like features.

    Uses gradient orientation histograms as lightweight feature descriptor.
    For production, swap with ResNet18 embedding.

    Args:
        images: list of BGR image arrays (resized to 224x224 caller side).

    Returns:
        (N, D) feature matrix.
    """
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(9 * ang / (2 * np.pi + 1e-6))
        hist = np.zeros(9)
        for i in range(9):
            hist[i] = np.sum(mag[bins == i])
        hist = hist / (np.sum(hist) + 1e-6)
        features.append(hist)
    return np.array(features)


def cluster_outliers(features: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> dict:
    """Run PCA + DBSCAN to detect outlier samples.

    Args:
        features: (N, D) feature matrix.
        eps: DBSCAN eps parameter.
        min_samples: DBSCAN min_samples parameter.

    Returns:
        dict with outlier_count, outlier_ratio, silhouette_score.
    """
    if len(features) < 3:
        return {"outlier_count": 0, "outlier_ratio": 0.0, "silhouette_score": 0.0}

    pca = PCA(n_components=min(2, features.shape[1]))
    reduced = pca.fit_transform(features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced)
    labels = clustering.labels_

    outlier_mask = labels == -1
    outlier_count = int(np.sum(outlier_mask))

    unique_labels = set(labels) - {-1}
    sil_score = 0.0
    if len(unique_labels) >= 2 and outlier_count < len(labels):
        sil_score = float(silhouette_score(reduced[~outlier_mask], labels[~outlier_mask]))

    return {
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / len(features), 4),
        "silhouette_score": round(sil_score, 4),
    }
