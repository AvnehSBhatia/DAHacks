"""
Run the real LatentSpace + AgentNetwork pipeline and export PCA visuals + raw 64-D vectors.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, AsyncGenerator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.demo import (
    ADVERSARIAL_SYSTEM,
    SUBTLE_ADVERSARIAL_SYSTEM,
    adversarial_generate,
    honest_generate,
    make_featherless_generate_fn,
    subtle_adversarial_generate,
)
from backend.featherless_agents import AGENT_PROFILES
from demo.kit import EMBED_DIM
from demo.viz import (
    build_cluster_snapshot_2d,
    build_morph_snapshot_3d,
    fit_pca_global_2d,
    fit_pca_global_3d,
)
from models.agent_system import Agent, AgentNetwork
from models.latent_space import LatentSpace
from models.paths import resolve_repo_path


def _optional_ckpt(rel: str) -> str | None:
    p = resolve_repo_path(rel)
    return str(p) if p.is_file() else None


def _vec_to_list(v: torch.Tensor) -> list[float]:
    return [float(x) for x in v.detach().cpu().float().reshape(-1)]


def _anchors_as_viz_rows(space: LatentSpace) -> list[dict[str, Any]]:
    """Records compatible with demo.viz (embedding = 64-D numpy)."""
    rows: list[dict[str, Any]] = []
    for a in space.anchors:
        rows.append(
            {
                "id": a.id,
                "embedding": a.vector.detach().cpu().numpy().astype(np.float64),
                "content": (a.text or "")[:500],
                "agent_id": a.agent_id,
                "outcome_score": 1.0 if not a.anomaly else -1.0,
                "kind": "anchor",
            }
        )
    return rows


def _snapshot_full_vectors(space: LatentSpace) -> list[dict[str, Any]]:
    """Pass-through of every latent anchor for custom visuals."""
    out: list[dict[str, Any]] = []
    for a in space.anchors:
        vo = a.vector_original if a.vector_original is not None else a.vector
        out.append(
            {
                "id": a.id,
                "agent_id": a.agent_id,
                "vector": _vec_to_list(a.vector),
                "vector_original": _vec_to_list(vo),
                "text": (a.text or "")[:500],
                "anomaly": bool(a.anomaly),
                "penalized": bool(a.penalized),
                "weight": float(a.weight),
                "impact": float(a.impact),
                "timestamp": float(a.timestamp),
            }
        )
    return out


W_IDENTITY = np.eye(EMBED_DIM, dtype=np.float64)


def _timing_enabled() -> bool:
    return os.environ.get("DAHACKS_TIMING", "1").strip().lower() not in ("0", "false", "no")


def _tlog(msg: str) -> None:
    if _timing_enabled():
        print(f"[timing] {msg}", flush=True)

AGENT_ROLES: list[str] = [
    "Careful science educator who cites mechanisms and avoids speculation.",
    "Concise technical writer who prefers bullet points and definitions.",
    "Friendly tutor who uses analogies and checks understanding.",
    "Skeptical reviewer who highlights uncertainties and edge cases.",
    "Fast responder who prioritizes brevity and memorable phrases.",
    "Systems thinker who connects concepts across domains.",
    "Pedantic fact-checker who insists on precise terminology.",
    "Narrative explainer who uses step-by-step reasoning.",
    "Minimalist answerer who gives the shortest correct reply.",
    "Cross-disciplinary synthesizer who links biology, chemistry, and earth science.",
]

# Rotate generator styles so the space mixes grounded and adversarial anchors
GEN_CYCLE: list[Callable[..., str]] = [
    honest_generate,
    honest_generate,
    honest_generate,
    subtle_adversarial_generate,
    adversarial_generate,
]


def _build_agents_spec(
    n: int,
    use_featherless: bool,
) -> list[tuple[str, str, Any, bool, str]]:
    """
    Each entry: ``(agent_id, display_name, generate_fn, hallucination_prone, role_string)``.

    For ``n == 3`` we fix α / β / γ (honest, honest, adversarial) and always run **3** full cycles
    (α→β→γ repeated three times); the ``cycles`` request field is ignored for that case.
    """
    if n == 3:
        rows: list[tuple[str, str, Any, bool, bool]] = [
            ("alpha", "Agent α", honest_generate, False, True),
            ("beta", "Agent β", honest_generate, False, True),
            ("gamma", "Agent γ", adversarial_generate, True, False),
        ]
        out: list[tuple[str, str, Any, bool, str]] = []
        for i, (aid, display_name, gen_raw, prone, honest_slot) in enumerate(rows):
            if use_featherless:
                if honest_slot:
                    prof = AGENT_PROFILES[i % len(AGENT_PROFILES)]
                    gen = make_featherless_generate_fn(prof["system"])
                    role = f"{prof['id']}: {prof['system']}"
                else:
                    gen = make_featherless_generate_fn(ADVERSARIAL_SYSTEM)
                    role = AGENT_ROLES[i % len(AGENT_ROLES)]
            else:
                gen = gen_raw
                role = AGENT_ROLES[i % len(AGENT_ROLES)]
            out.append((aid, display_name, gen, prone, role))
        return out

    spec: list[tuple[str, str, Any, bool, str]] = []
    for i in range(n):
        gen_raw = GEN_CYCLE[i % len(GEN_CYCLE)]
        prone = gen_raw in (subtle_adversarial_generate, adversarial_generate)
        aid = f"agent_{i}"

        if use_featherless:
            if gen_raw is honest_generate:
                prof = AGENT_PROFILES[i % len(AGENT_PROFILES)]
                gen = make_featherless_generate_fn(prof["system"])
                role = f"{prof['id']}: {prof['system']}"
            elif gen_raw is subtle_adversarial_generate:
                gen = make_featherless_generate_fn(SUBTLE_ADVERSARIAL_SYSTEM)
                role = AGENT_ROLES[i % len(AGENT_ROLES)]
            else:
                gen = make_featherless_generate_fn(ADVERSARIAL_SYSTEM)
                role = AGENT_ROLES[i % len(AGENT_ROLES)]
        else:
            gen = gen_raw
            role = AGENT_ROLES[i % len(AGENT_ROLES)]

        short = role if len(role) <= 52 else f"{role[:48]}…"
        spec.append((aid, f"Agent {i} — {short}", gen, prone, role))
    return spec


def run_latent_demo(
    prompt: str,
    *,
    context: str = "",
    seed: int = 42,
    num_agents: int = 10,
    stagger_s: float = 0.5,
    cycles: int = 3,
) -> dict[str, Any]:
    """
    Multi-agent loop on shared LatentSpace; returns PCA visuals + full 64-D ``latent`` payload.

    Between each ``network.run`` call we sleep ``stagger_s`` seconds (except before the first).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    t_run = time.perf_counter()
    _tlog(f"run_latent_demo start (agents={num_agents}, cycles={cycles}, stagger_s={stagger_s})")

    enc = os.environ.get("DAHACKS_ENCODER") or _optional_ckpt("models/checkpoints/encoder_2x384_to_64.pt")
    rnet = os.environ.get("DAHACKS_RESPONSE_NET") or _optional_ckpt("models/checkpoints/response_latent_net.pt")

    t0 = time.perf_counter()
    space = LatentSpace(
        encoder_path=enc,
        response_net_path=rnet,
        eta=0.02,
        anomaly_threshold=0.42,
        max_anchors=500,
        decay_k=0.002,
    )
    _tlog(f"LatentSpace init (encoder + models): {time.perf_counter() - t0:.3f}s")

    space.set_base_vector(
        "You are a reliable multi-agent knowledge system. Provide accurate, factual responses."
    )

    t0 = time.perf_counter()
    network = AgentNetwork(space, update_every=5, retrieval_k=5)
    _tlog(f"AgentNetwork construct: {time.perf_counter() - t0:.3f}s")

    n = max(1, min(int(num_agents), 32))
    use_featherless = bool(os.environ.get("FEATHERLESS_API_KEY"))

    agents_spec = _build_agents_spec(n, use_featherless)

    t0 = time.perf_counter()
    for aid, _display_name, gen, _prone, role in agents_spec:
        network.register(Agent(aid, role, gen))
    _tlog(f"register {n} agents: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    z_session = space.embed_pair_to_latent(prompt, context if context.strip() else prompt)
    _tlog(f"session embed_pair_to_latent (prompt+context): {time.perf_counter() - t0:.3f}s")

    steps_out: list[dict[str, Any]] = []
    timeline: list[list[dict[str, Any]]] = []
    timeline_vectors: list[list[dict[str, Any]]] = []

    order = [a[0] for a in agents_spec]
    base_v = getattr(space, "_base_vector")
    w_start = float(torch.norm(space.ground_truth - base_v).item())

    gap = max(0.0, float(stagger_s))
    n_cycles = 3 if n == 3 else max(1, int(cycles))
    first_step = True
    step_idx = 0

    for _c in range(n_cycles):
        for aid in order:
            if aid not in network.list_agents():
                continue
            if not first_step and gap > 0:
                _tlog(f"stagger sleep: {gap:.3f}s (before agent {aid}, cycle {_c})")
                time.sleep(gap)
            first_step = False
            step_idx += 1
            t_step = time.perf_counter()
            result = network.run(prompt, agent_id=aid, extra_context=context)
            _tlog(
                f"step {step_idx} cycle={_c} agent={aid} network.run total: "
                f"{time.perf_counter() - t_step:.3f}s"
            )
            t_snap = time.perf_counter()
            mem_copy = copy.deepcopy(_anchors_as_viz_rows(space))
            _tlog(f"  deepcopy viz rows ({len(mem_copy)} anchors): {time.perf_counter() - t_snap:.3f}s")
            t_sv = time.perf_counter()
            sv = _snapshot_full_vectors(space)
            _tlog(f"  snapshot full vectors: {time.perf_counter() - t_sv:.3f}s")
            meta = next(
                (x for x in agents_spec if x[0] == aid),
                (aid, aid, None, False, ""),
            )
            steps_out.append(
                {
                    "agent": {
                        "id": aid,
                        "name": meta[1],
                        "hallucination_prone": bool(meta[3]),
                    },
                    "action": result.output,
                    "reward": 1.0 if not result.anomaly else -1.0,
                    "retrieved_snippets": result.context[:3],
                    "pulse": {"receive": True, "write": True},
                    "w_frobenius_delta": float(result.score),
                }
            )
            timeline.append(mem_copy)
            timeline_vectors.append(sv)

    t0 = time.perf_counter()
    space.update_cycle()
    _tlog(f"final space.update_cycle: {time.perf_counter() - t0:.3f}s")
    w_end = float(torch.norm(space.ground_truth - base_v).item())

    t0 = time.perf_counter()
    final_rows = _anchors_as_viz_rows(space)
    _tlog(f"final_rows anchors_as_viz: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    pca3 = fit_pca_global_3d(final_rows, W_IDENTITY)
    _tlog(f"fit_pca_global_3d: {time.perf_counter() - t0:.3f}s")

    morph_frames: list[dict[str, Any]] = []
    t_morph = time.perf_counter()
    for i, mem in enumerate(timeline):
        t_fr = time.perf_counter()
        morph_frames.append(build_morph_snapshot_3d(mem, W_IDENTITY, pca_ref=pca3))
        _tlog(f"  morph frame {i + 1}/{len(timeline)}: {time.perf_counter() - t_fr:.3f}s")
    _tlog(f"morph_frames total ({len(timeline)} frames): {time.perf_counter() - t_morph:.3f}s")

    t0 = time.perf_counter()
    pca2 = fit_pca_global_2d(final_rows, W_IDENTITY)
    _tlog(f"fit_pca_global_2d: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    final_clusters, _ = build_cluster_snapshot_2d(
        final_rows,
        W_IDENTITY,
        pca_ref=pca2,
        k_clusters=3,
        anomaly_threshold=2.5,
    )
    _tlog(f"build_cluster_snapshot_2d: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    anchors_final = _snapshot_full_vectors(space)
    _tlog(f"latent.anchors_final snapshot build: {time.perf_counter() - t0:.3f}s")

    gt = space.ground_truth
    base = base_v

    _tlog(f"run_latent_demo done: {time.perf_counter() - t_run:.3f}s total")

    return {
        "prompt": prompt,
        "context": context,
        "steps": steps_out,
        "morph_frames": morph_frames,
        "final_clusters": final_clusters,
        "w_frobenius_delta_start": w_start,
        "w_frobenius_delta_end": w_end,
        "latent": {
            "dim": EMBED_DIM,
            "session_z": _vec_to_list(z_session),
            "ground_truth": _vec_to_list(gt),
            "base_vector": _vec_to_list(base),
            "anchors_final": anchors_final,
            "timeline_vectors": timeline_vectors,
            "encoder_loaded": enc is not None,
            "response_net_loaded": rnet is not None,
            "num_agents": n,
            "stagger_s": gap,
            "cycles": n_cycles,
        },
    }


async def stream_latent_demo(
    prompt: str,
    *,
    context: str = "",
    seed: int = 42,
    num_agents: int = 3,
    stagger_s: float = 0.5,
    cycles: int = 3,
) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator version of run_latent_demo.
    Yields JSON updates progressively.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    _tlog(f"stream_latent_demo start (agents={num_agents}, cycles={cycles})")

    enc = os.environ.get("DAHACKS_ENCODER") or _optional_ckpt("models/checkpoints/encoder_2x384_to_64.pt")
    rnet = os.environ.get("DAHACKS_RESPONSE_NET") or _optional_ckpt("models/checkpoints/response_latent_net.pt")

    space = LatentSpace(
        encoder_path=enc,
        response_net_path=rnet,
        eta=0.02,
        anomaly_threshold=0.42,
        max_anchors=500,
        decay_k=0.002,
    )
    space.set_base_vector("You are a reliable multi-agent knowledge system. Provide accurate, factual responses.")
    network = AgentNetwork(space, update_every=5, retrieval_k=5)

    n = max(1, min(int(num_agents), 32))
    use_featherless = bool(os.environ.get("FEATHERLESS_API_KEY"))

    agents_spec = _build_agents_spec(n, use_featherless)

    for aid, _display_name, gen, _prone, role in agents_spec:
        network.register(Agent(aid, role, gen))

    z_session = space.embed_pair_to_latent(prompt, context if context.strip() else prompt)
    order = [a[0] for a in agents_spec]
    base_v = getattr(space, "_base_vector")
    w_start = float(torch.norm(space.ground_truth - base_v).item())

    steps_out = []
    timeline = []
    timeline_vectors = []

    # Send initial boot sequence Event
    init_data = {
        "type": "init",
        "payload": {
            "w_frobenius_delta_start": w_start,
            "session_z": _vec_to_list(z_session),
        }
    }
    yield f"data: {json.dumps(init_data)}\n\n"

    gap = max(0.0, float(stagger_s))
    n_cycles = 3 if n == 3 else max(1, int(cycles))
    first_step = True
    step_idx = 0

    for _c in range(n_cycles):
        for aid in order:
            if aid not in network.list_agents():
                continue

            meta = next(
                (x for x in agents_spec if x[0] == aid),
                (aid, aid, None, False, ""),
            )

            if not first_step and gap > 0:
                await asyncio.sleep(gap)
            first_step = False
            step_idx += 1
            
            # Send 'thinking' event to frontend
            think_data = {
                "type": "thinking",
                "agent_id": aid,
                "agent_name": meta[1]
            }
            yield f"data: {json.dumps(think_data)}\n\n"
            
            # Since network.run is synchronous, we use asyncio.to_thread
            result = await asyncio.to_thread(network.run, prompt, agent_id=aid, extra_context=context)

            mem_copy = copy.deepcopy(_anchors_as_viz_rows(space))
            sv = _snapshot_full_vectors(space)

            step_obj = {
                "agent": {
                    "id": aid,
                    "name": meta[1],
                    "hallucination_prone": bool(meta[3]),
                },
                "action": result.output,
                "reward": 1.0 if not result.anomaly else -1.0,
                "retrieved_snippets": result.context[:3],
                "pulse": {"receive": True, "write": True},
                "w_frobenius_delta": float(result.score),
            }
            steps_out.append(step_obj)
            timeline.append(mem_copy)
            timeline_vectors.append(sv)
            
            # Compute current PCA directly on what we have so far
            # If d < 2 no PCA is reasonable, but we can compute it on sv + base + gt
            # We'll calculate a fast ad-hoc PCA for the stream
            current_pca = build_fast_pca_frame(mem_copy)

            # Send the step completed event
            step_data = {
                "type": "step",
                "payload": {
                    "step": step_obj,
                    "current_frame": current_pca,
                    "latent": {
                        "anchors_final": sv,
                        "ground_truth": _vec_to_list(space.ground_truth)
                    }
                }
            }
            yield f"data: {json.dumps(step_data)}\n\n"

    space.update_cycle()
    w_end = float(torch.norm(space.ground_truth - base_v).item())

    final_rows = _anchors_as_viz_rows(space)
    pca3 = fit_pca_global_3d(final_rows, W_IDENTITY)
    
    morph_frames = []
    for mem in timeline:
        morph_frames.append(build_morph_snapshot_3d(mem, W_IDENTITY, pca_ref=pca3))

    pca2 = fit_pca_global_2d(final_rows, W_IDENTITY)
    final_clusters, _ = build_cluster_snapshot_2d(
        final_rows, W_IDENTITY, pca_ref=pca2, k_clusters=3, anomaly_threshold=2.5
    )
    anchors_final = _snapshot_full_vectors(space)

    final_payload = {
        "prompt": prompt,
        "context": context,
        "steps": steps_out,
        "morph_frames": morph_frames,
        "final_clusters": final_clusters,
        "w_frobenius_delta_start": w_start,
        "w_frobenius_delta_end": w_end,
        "latent": {
            "dim": EMBED_DIM,
            "session_z": _vec_to_list(z_session),
            "ground_truth": _vec_to_list(space.ground_truth),
            "base_vector": _vec_to_list(base_v),
            "anchors_final": anchors_final,
            "timeline_vectors": timeline_vectors,
            "encoder_loaded": enc is not None,
            "response_net_loaded": rnet is not None,
            "num_agents": n,
            "stagger_s": gap,
            "cycles": n_cycles,
        },
    }
    
    complete_event = {
        "type": "complete",
        "payload": final_payload
    }
    yield f"data: {json.dumps(complete_event)}\n\n"


def build_fast_pca_frame(rows: list[dict[str, Any]]):
    """Fast ad-hoc PCA calculation for streaming frames."""
    if not rows:
        return {"points": [], "embed_dim": EMBED_DIM, "explained_variance_ratio": [0,0,0]}
    
    from demo.viz import fit_pca_global_3d, build_morph_snapshot_3d
    pca3 = fit_pca_global_3d(rows, W_IDENTITY)
    return build_morph_snapshot_3d(rows, W_IDENTITY, pca_ref=pca3)
