"""
Hackathon demo only: grounded micro-QA, shared memory, latent W, three agents (one hallucinates).
Not a general-purpose agent backend.
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any

import numpy as np

from demo.kit import EMBED_DIM, LatentEngine, VectorStore, embed
from demo.viz import (
    build_cluster_snapshot_2d,
    build_morph_snapshot_3d,
    fit_pca_global_2d,
    fit_pca_global_3d,
)

# --- Ground-truth corpus (seeded into shared memory) ---

SEED_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "seed-fr",
        "content": "The capital of France is Paris.",
        "agent_id": "corpus",
    },
    {
        "id": "seed-light",
        "content": "The speed of light in vacuum is approximately 299792458 meters per second.",
        "agent_id": "corpus",
    },
    {
        "id": "seed-water",
        "content": "Pure water boils at 100 degrees Celsius at standard atmospheric pressure (1 atm).",
        "agent_id": "corpus",
    },
]

AGENTS: list[dict[str, Any]] = [
    {"id": "alpha", "name": "Agent α", "hallucinate": False},
    {"id": "beta", "name": "Agent β", "hallucinate": False},
    {
        "id": "gamma",
        "name": "Agent γ (hallucination-prone)",
        "hallucinate": True,
    },
]

_HALLUCINATIONS = [
    "The capital of France is London.",
    "Light travels at about 1000 meters per second in vacuum.",
    "Water boils at 50 degrees Celsius at standard pressure.",
]


def _decide(question: str, contexts: list[str], hallucinate: bool, rng: random.Random) -> str:
    if hallucinate and rng.random() < 0.78:
        return rng.choice(_HALLUCINATIONS)
    q = question.lower()
    if "france" in q and "capital" in q:
        return "The capital of France is Paris."
    if "speed" in q and "light" in q:
        return "The speed of light in vacuum is approximately 299792458 m/s."
    if "water" in q and ("boil" in q or "celsius" in q or "100" in q):
        return "Water boils at 100°C at 1 atm."
    if contexts:
        return contexts[0][:280]
    return "I need more context to answer."


def _evaluate(answer: str, question: str) -> float:
    a = answer.lower().replace(" ", "")
    q = question.lower()
    if "france" in q and "capital" in q:
        return 1.0 if "paris" in answer.lower() and "london" not in answer.lower() else -1.0
    if "speed" in q and "light" in q:
        return 1.0 if "299792458" in a else -1.0
    if "water" in q and "boil" in q:
        return 1.0 if "100" in answer.lower() else -1.0
    return 0.3


def _frobenius_delta(w: np.ndarray) -> float:
    return float(np.linalg.norm(w - np.eye(w.shape[0]), ord="fro"))


def run_demo(prompt: str, *, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    store = VectorStore()
    latent = LatentEngine(EMBED_DIM)

    for ch in SEED_CHUNKS:
        store.insert(
            {
                "id": ch["id"],
                "embedding": embed(ch["content"]),
                "content": ch["content"],
                "agent_id": ch["agent_id"],
                "outcome_score": 0.0,
                "kind": "seed",
            }
        )

    steps_out: list[dict[str, Any]] = []
    timeline: list[list[dict[str, Any]]] = []

    w0 = _frobenius_delta(latent.W)

    for agent in AGENTS:
        qv = embed(prompt)
        qt = latent.transform(qv)
        retrieved = store.search(qt, k=5)
        contexts = [r["content"] for r in retrieved]
        action = _decide(prompt, contexts, bool(agent["hallucinate"]), rng)
        reward = _evaluate(action, prompt)

        row = {
            "id": str(uuid.uuid4()),
            "embedding": embed(prompt + action),
            "content": action,
            "agent_id": agent["id"],
            "outcome_score": reward,
            "kind": "experience",
        }
        store.insert(row)
        latent.update(prompt, action, reward)

        mem_copy = copy.deepcopy(store.vectors)
        timeline.append(mem_copy)

        steps_out.append(
            {
                "agent": {
                    "id": agent["id"],
                    "name": agent["name"],
                    "hallucination_prone": agent["hallucinate"],
                },
                "action": action,
                "reward": reward,
                "retrieved_snippets": contexts[:3],
                "pulse": {"receive": True, "write": True},
                "w_frobenius_delta": _frobenius_delta(latent.W),
            }
        )

    final_vecs = timeline[-1]
    pca3 = fit_pca_global_3d(final_vecs, latent.W)

    morph_frames: list[dict[str, Any]] = []
    for mem in timeline:
        morph_frames.append(build_morph_snapshot_3d(mem, latent.W, pca_ref=pca3))

    pca2 = fit_pca_global_2d(final_vecs, latent.W)
    final_clusters, _ = build_cluster_snapshot_2d(
        final_vecs,
        latent.W,
        pca_ref=pca2,
        k_clusters=3,
        anomaly_threshold=2.5,
    )

    return {
        "prompt": prompt,
        "steps": steps_out,
        "morph_frames": morph_frames,
        "final_clusters": final_clusters,
        "w_frobenius_delta_start": w0,
        "w_frobenius_delta_end": _frobenius_delta(latent.W),
    }
