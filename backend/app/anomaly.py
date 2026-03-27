"""
Anomaly detection via K-means: flag points whose distance to their assigned
cluster center is unusually large (z-score on distances).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

__all__ = ["detect_anomalies"]


def detect_anomalies(
    vectors: list[dict],
    *,
    k_clusters: int = 3,
    threshold: float = 2.5,
    random_state: int = 42,
) -> list[dict]:
    """
    Cluster embeddings with K-means; vectors far from their own centroid are anomalies.

    Each item in ``vectors`` must have key ``embedding`` (1-D sequence or ndarray).

    Parameters
    ----------
    vectors:
        Memory records (or any dicts) including ``embedding``.
    k_clusters:
        Number of K-means clusters (capped by sample count).
    threshold:
        Z-score on distance-to-centroid; above this => anomaly.
    random_state:
        RNG seed for K-means init.
    """
    if not vectors:
        return []

    embeddings = np.asarray([v["embedding"] for v in vectors], dtype=np.float64)
    if embeddings.ndim != 2:
        embeddings = embeddings.reshape(len(vectors), -1)

    n = embeddings.shape[0]
    if n < 2:
        return []

    n_clusters = max(1, min(k_clusters, n))
    if n_clusters == 1 or n == 2:
        # Degenerate: use global centroid distance spread
        center = np.mean(embeddings, axis=0, keepdims=True)
        dist = np.linalg.norm(embeddings - center, axis=1)
        mu, sigma = float(np.mean(dist)), float(np.std(dist)) + 1e-6
        z = (dist - mu) / sigma
        return [vectors[i] for i in range(n) if z[i] > threshold]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    dist = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist[i] = float(np.linalg.norm(embeddings[i] - centers[labels[i]]))

    mu, sigma = float(np.mean(dist)), float(np.std(dist)) + 1e-6
    z = (dist - mu) / sigma

    return [vectors[i] for i in range(n) if z[i] > threshold]
