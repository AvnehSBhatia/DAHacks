"""Demo-only: minimal embeddings, vector list, and latent W (not a general library)."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import numpy as np

EMBED_DIM = 64


def _bytes_to_vec(seed: bytes, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float64)
    block = seed
    i = 0
    while i < dim:
        block = hashlib.sha256(block).digest()
        for b in block:
            if i >= dim:
                break
            out[i] = (b / 255.0) - 0.5
            i += 1
    out -= out.mean()
    n = np.linalg.norm(out)
    if n > 1e-9:
        out /= n
    return out


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return _bytes_to_vec(h, dim)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class VectorStore:
    def __init__(self) -> None:
        self.vectors: list[dict[str, Any]] = []

    def insert(self, item: dict[str, Any]) -> str:
        iid = str(item.get("id") or uuid.uuid4())
        row = {**item, "id": iid}
        self.vectors.append(row)
        return iid

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        for v in self.vectors:
            base = _cosine(q, v["embedding"])
            bonus = 1.0 + 0.05 * float(v.get("outcome_score", 0))
            scored.append((base * bonus, v))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [v for _, v in scored[:k]]


class LatentEngine:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.W = np.eye(dim, dtype=np.float64)

    def transform(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        return self.W @ v

    def update(self, context_text: str, action_text: str, reward: float, lr: float = 0.01) -> None:
        c = embed(context_text, self.dim)
        a = embed(action_text, self.dim)
        diff = c - a
        outer = np.outer(diff, diff)
        if reward > 0:
            self.W -= lr * outer
        else:
            self.W += lr * outer
