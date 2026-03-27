"""
Cohesive orchestration: prompt + context → 64-D session vector, N persona-agents
each pull/push into a shared LatentSpace, outlier detection + removal, then a random
survivor answers using score-weighted retrieval.

Configuration: JSON file (see backend/system_config.json). Use ``session.final_synthesize``:
``retrieval`` (default) prints ranked anchors from the latent query; ``agent`` calls
each agent's ``generate_fn`` with plain anchor texts (for real LLMs). Run:

    ./run_cohesive.sh
    python backend/cohesive_system.py --config backend/system_config.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from backend.demo import adversarial_generate, honest_generate, subtle_adversarial_generate
from models.agent_system import Agent, AgentNetwork
from models.latent_space import LatentSpace, Vec
from models.paths import resolve_repo_path

GENERATORS = {
    "honest": honest_generate,
    "subtle": subtle_adversarial_generate,
    "adversarial": adversarial_generate,
}


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cross_agent_outlier_ids(
    space: LatentSpace,
    agent_ids: list[str],
    iqr_factor: float,
) -> set[str]:
    """Flag agents whose anomaly_rate is above Q3 + iqr_factor * IQR (needs ≥3 agents with data)."""
    rates: list[tuple[str, float]] = []
    for aid in agent_ids:
        rep = space.agent_anomaly_score(aid)
        if rep.get("verdict") == "no_data":
            continue
        rates.append((aid, float(rep["anomaly_rate"])))
    if len(rates) < 3:
        return set()
    by_rate = sorted(rates, key=lambda x: x[1])
    vals = [x[1] for x in by_rate]
    try:
        q1, q3 = statistics.quantiles(vals, n=4)[0], statistics.quantiles(vals, n=4)[2]
    except Exception:
        n = len(vals)
        q1, q3 = vals[n // 4], vals[(3 * n) // 4]
    iqr = max(q3 - q1, 1e-8)
    hi = q3 + iqr_factor * iqr
    return {aid for aid, r in rates if r > hi}


def agents_to_flag_for_removal(
    space: LatentSpace,
    network: AgentNetwork,
    agent_ids: list[str],
    *,
    iqr_factor: float,
    min_trust: float,
    prune_health_at_or_below: float,
) -> set[str]:
    """
    Remove agents that are bad actors, below ``min_trust``, statistical outliers
    (IQR), or at or below ``prune_health_at_or_below`` on the latent health score
    (0.5 = bottom 50% of the 0–1 health scale for typical agents).
    """
    remove: set[str] = set()
    for aid in agent_ids:
        rep = space.agent_anomaly_score(aid)
        if rep.get("verdict") == "bad_actor":
            remove.add(aid)
        try:
            if network.get_agent(aid).trust_score < min_trust:
                remove.add(aid)
        except KeyError:
            pass
        health = float(rep.get("health", 1.0))
        if health <= prune_health_at_or_below:
            remove.add(aid)
    remove |= cross_agent_outlier_ids(space, agent_ids, iqr_factor)
    return remove


def run_cohesive(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    sess = config.get("session", {})
    out_cfg = config.get("outliers", {})
    personas_cfg = config.get("personas", [])
    gen_cfg = config.get("generators", "honest")

    encoder = str(resolve_repo_path(paths["encoder"])) if paths.get("encoder") else None
    response_net = str(resolve_repo_path(paths["response_net"])) if paths.get("response_net") else None

    prompt = sess["prompt"]
    context = sess.get("context", "")
    num_models = int(sess["num_models"])
    cycles = int(sess["cycles_per_model"])
    final_k = int(sess.get("final_retrieval_k", 12))
    final_synthesize = str(sess.get("final_synthesize", "retrieval")).strip().lower()
    if final_synthesize not in ("retrieval", "agent"):
        raise ValueError("session.final_synthesize must be 'retrieval' or 'agent'")
    base_vector = sess.get(
        "base_vector",
        "You are a reliable multi-agent knowledge system. Provide accurate, factual responses.",
    )
    retrieval_k = int(sess.get("interaction_retrieval_k", 5))
    update_every = int(sess.get("update_every", 5))
    seed = int(sess.get("seed", 42))

    iqr_factor = float(out_cfg.get("iqr_factor", 1.5))
    min_trust = float(out_cfg.get("min_trust", 0.2))
    removal_passes = int(out_cfg.get("removal_passes", 2))
    prune_health_at_or_below = float(out_cfg.get("prune_health_at_or_below", 0.5))

    random.seed(seed)
    torch.manual_seed(seed)

    if len(personas_cfg) < num_models:
        raise ValueError(f"Need at least {num_models} personas, got {len(personas_cfg)}")

    # --- Latent space (shared by all agents) ---
    space = LatentSpace(
        encoder_path=encoder,
        response_net_path=response_net,
    )
    space.set_base_vector(base_vector)

    network = AgentNetwork(
        space,
        update_every=update_every,
        retrieval_k=retrieval_k,
        bad_actor_threshold=float(out_cfg.get("bad_actor_threshold", 0.35)),
    )

    # --- Register agents with unique personas ---
    agent_ids: list[str] = []
    prefix = sess.get("agent_id_prefix", "model")

    def resolve_generator(i: int):
        if isinstance(gen_cfg, list):
            key = gen_cfg[i % len(gen_cfg)]
        else:
            key = gen_cfg
        if key not in GENERATORS:
            raise ValueError(f"Unknown generator '{key}'. Use: {list(GENERATORS)}")
        return GENERATORS[key]

    for i in range(num_models):
        aid = f"{prefix}_{i}"
        agent_ids.append(aid)
        role = personas_cfg[i]
        gen_fn = resolve_generator(i)
        network.register(Agent(aid, role, gen_fn))

    # --- Session 64-D vector from prompt + context (encoder path in LatentSpace) ---
    z_session: Vec = space.embed_pair_to_latent(prompt, context if context else prompt)
    print(f"\n[Session] 64-D latent from (prompt, context) — norm={z_session.norm().item():.4f}")

    # --- Each agent pulls/push cycles: same prompt, context passed as extra in run() ---
    print(f"\n[Phase 1] {cycles} pull/push cycles per agent ({len(agent_ids)} agents)…")
    for c in range(cycles):
        for aid in agent_ids:
            if aid not in network.list_agents():
                continue
            network.run(prompt, agent_id=aid, extra_context=context)
        if (c + 1) % max(1, cycles // 3) == 0 or c == cycles - 1:
            print(f"  … completed cycle {c + 1}/{cycles}")

    # --- Outlier detection & removal ---
    print("\n[Phase 2] Outlier detection and agent removal…")
    print(
        f"  (health ≤ {prune_health_at_or_below} → prune list; penalize anchors when health in that band or bad_actor)"
    )
    for p in range(removal_passes):
        alive = network.list_agents()
        if not alive:
            break
        to_remove = agents_to_flag_for_removal(
            space,
            network,
            alive,
            iqr_factor=iqr_factor,
            min_trust=min_trust,
            prune_health_at_or_below=prune_health_at_or_below,
        )
        for aid in sorted(to_remove):
            rep = space.agent_anomaly_score(aid)
            health = float(rep.get("health", 1.0))
            verdict = rep.get("verdict")
            should_penalize = verdict == "bad_actor" or (
                verdict not in ("no_data", None) and health <= prune_health_at_or_below
            )
            if should_penalize and verdict != "no_data":
                n_pen = space.penalize_agent_anchors(aid)
                if n_pen:
                    print(
                        f"  Penalized {n_pen} anchor(s) for {aid} "
                        f"(health={health:.3f}, verdict={verdict})"
                    )
            network.remove_agent(aid)
            print(f"  Removed outlier / low-trust / low-health agent: {aid}")
        if not to_remove:
            print(f"  Pass {p + 1}: no agents removed")
        network.refresh_trust_scores()

    surviving = network.list_agents()
    if not surviving:
        raise RuntimeError("All agents were removed; relax outlier thresholds or use gentler generators.")

    space.update_cycle()
    network.refresh_trust_scores()

    # --- Random survivor + weighted retrieval in latent space ---
    chosen = random.choice(surviving)
    print(f"\n[Phase 3] Final answer from random survivor: {chosen}")
    print(f"  Retrieving top-{final_k} anchors weighted by similarity × decay × impact…")
    if final_synthesize == "agent":
        final = network.final_answer_weighted(
            prompt=prompt,
            agent_id=chosen,
            query_z=z_session,
            k=final_k,
            synthesize="agent",
        )
    else:
        final = network.final_answer_weighted(
            prompt=prompt,
            agent_id=chosen,
            query_z=z_session,
            k=final_k,
            synthesize="retrieval",
        )

    audit = network.audit_agents()

    return {
        "session_z_norm": z_session.norm().item(),
        "final_agent": chosen,
        "final_output": final.output,
        "weighted_context_lines": final.weighted_context_lines,
        "surviving_agents": surviving,
        "audit": audit,
        "space_stats": network.network_stats(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run cohesive latent multi-agent system")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "system_config.json",
        help="JSON command file (paths, session, personas, outliers)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run_cohesive(cfg)

    print("\n" + "=" * 60)
    print("  FINAL OUTPUT")
    print("=" * 60)
    print(result["final_output"])
    print("\n" + "=" * 60)
    print("  AUDIT (survivors + removed may be inferred from audit)")
    print("=" * 60)
    for r in result["audit"]:
        print(
            f"  {r.get('agent_id')}: verdict={r.get('verdict')}  "
            f"health={r.get('health')}  anomaly_rate={r.get('anomaly_rate')}  trust≈{r.get('trust_score')}"
        )


if __name__ == "__main__":
    main()
