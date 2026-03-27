"""Demo-only: 3D PCA for latent vectors; 2D K-means + anomalies for cluster view."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

_backend = Path(__file__).resolve().parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from app.anomaly import detect_anomalies

from demo.kit import EMBED_DIM


def _apply_w(w: np.ndarray, emb: np.ndarray) -> np.ndarray:
    return w @ np.asarray(emb, dtype=np.float64).reshape(-1)


def fit_pca_global_3d(vectors: list[dict], w: np.ndarray) -> PCA:
    """PCA basis fit on final memory cloud — replay earlier frames through this basis."""
    if not vectors:
        return PCA(n_components=min(3, EMBED_DIM))
    x = np.vstack([_apply_w(w, v["embedding"]) for v in vectors])
    n_comp = min(3, x.shape[0], x.shape[1])
    n_comp = max(1, n_comp)
    return PCA(n_components=n_comp).fit(x)


def build_morph_snapshot_3d(
    vectors: list[dict],
    w: np.ndarray,
    *,
    pca_ref: PCA,
) -> dict:
    """
    64D embeddings → W·x → PCA into **3D** for shape visualization.
    No K-means here — pure geometry of the transformed vectors.
    """
    if not vectors:
        return {
            "points": [],
            "embed_dim": EMBED_DIM,
            "explained_variance_ratio": [0.0, 0.0, 0.0],
        }

    x = np.vstack([_apply_w(w, v["embedding"]) for v in vectors])
    x3 = pca_ref.transform(x)
    # Pad to 3 columns if PCA used fewer (e.g. n_samples < 3)
    if x3.shape[1] < 3:
        pad = np.zeros((x3.shape[0], 3 - x3.shape[1]), dtype=np.float64)
        x3 = np.hstack([x3, pad])

    evr = pca_ref.explained_variance_ratio_.tolist()
    while len(evr) < 3:
        evr.append(0.0)
    evr = evr[:3]

    points = []
    for i, v in enumerate(vectors):
        points.append(
            {
                "id": v["id"],
                "x": float(x3[i, 0]),
                "y": float(x3[i, 1]),
                "z": float(x3[i, 2]),
                "agent_id": str(v.get("agent_id", "seed")),
                "snippet": (v.get("content") or "")[:160],
            }
        )

    return {
        "points": points,
        "embed_dim": EMBED_DIM,
        "explained_variance_ratio": evr,
    }


def fit_pca_global_2d(vectors: list[dict], w: np.ndarray) -> PCA:
    if not vectors:
        return PCA(n_components=min(2, EMBED_DIM))
    x = np.vstack([_apply_w(w, v["embedding"]) for v in vectors])
    n_comp = min(2, x.shape[0], x.shape[1])
    n_comp = max(1, n_comp)
    return PCA(n_components=n_comp).fit(x)


def build_cluster_snapshot_2d(
    vectors: list[dict],
    w: np.ndarray,
    *,
    pca_ref: PCA | None = None,
    k_clusters: int = 3,
    anomaly_threshold: float = 2.5,
) -> tuple[dict, PCA]:
    """2D projection + K-means cluster labels + anomaly flags (final panel)."""
    if not vectors:
        return (
            {
                "points": [],
                "centroids": [],
                "anomaly_ids": [],
                "explained_variance_ratio": [0.0, 0.0],
            },
            PCA(n_components=2),
        )

    x = np.vstack([_apply_w(w, v["embedding"]) for v in vectors])
    if pca_ref is None:
        n_comp = min(2, x.shape[0], x.shape[1])
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp).fit(x)
    else:
        pca = pca_ref
    x2 = pca.transform(x)

    n = len(vectors)
    n_k = max(1, min(k_clusters, n))
    km = KMeans(n_clusters=n_k, random_state=42, n_init=10).fit(x)
    labels = km.labels_
    c2 = pca.transform(km.cluster_centers_)

    anomalies = detect_anomalies(vectors, k_clusters=k_clusters, threshold=anomaly_threshold)
    anom_ids = [a["id"] for a in anomalies]

    points = []
    for i, v in enumerate(vectors):
        points.append(
            {
                "id": v["id"],
                "x": float(x2[i, 0]),
                "y": float(x2[i, 1]) if x2.shape[1] > 1 else 0.0,
                "cluster": int(labels[i]),
                "is_anomaly": v["id"] in anom_ids,
                "agent_id": str(v.get("agent_id", "seed")),
                "snippet": (v.get("content") or "")[:160],
            }
        )

    centroids = [{"x": float(c[0]), "y": float(c[1]) if len(c) > 1 else 0.0} for c in c2]
    evr = pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else [0.0, 0.0]

    return (
        {
            "points": points,
            "centroids": centroids,
            "anomaly_ids": anom_ids,
            "explained_variance_ratio": evr[:2],
        },
        pca,
    )
