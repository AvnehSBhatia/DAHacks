"""
demo.py
───────
End-to-end demonstration of the self-evolving latent vector system.

Runs WITHOUT requiring pre-trained encoder/decoder artefacts —
uses the built-in orthonormal projection fallback so you can
verify the dynamics immediately.

To use your trained artefacts, set:
    ENCODER_PATH        = "encoder_2x384_to_64.pt"
    RESPONSE_NET_PATH   = "response_latent_net.pt"

Usage
─────
    python demo.py                   # full demo, no trained models needed
    python demo.py --verbose         # print every anchor insertion
    python demo.py --steps 40        # run more interactions
"""

from __future__ import annotations

import argparse
import random
import textwrap

from latent_space import LatentSpace
from agent_system import Agent, AgentNetwork


# ══════════════════════════════════════════════════════════════════════════════
# Fake LLM generate functions
# (Replace with real API calls in production)
# ══════════════════════════════════════════════════════════════════════════════

FACTS = [
    "The capital of France is Paris.",
    "Water boils at 100°C at sea level.",
    "The speed of light is approximately 299,792 km/s.",
    "Shakespeare wrote Hamlet around 1600.",
    "Python is a high-level interpreted programming language.",
    "The human genome contains approximately 3 billion base pairs.",
    "Photosynthesis converts CO2 and H2O into glucose using sunlight.",
    "The Great Wall of China stretches over 21,000 km.",
    "DNA has a double-helix structure discovered by Watson and Crick.",
    "The Eiffel Tower is 330 metres tall.",
]

NOISE = [
    "Purple elephants compute pi faster than any known algorithm.",
    "The moon is made of crystallised moonbeam resonance.",
    "5G towers recharge the ionosphere's chakra field at midnight.",
    "Quantum entanglement allows telepathy between cousins.",
    "Ancient Romans invented the internet using carrier pigeons.",
    "Vaccinations contain microscopic tracking chips from the 1940s.",
    "The Earth is expanding due to gravity consumption.",
    "Drinking bleach cures common colds within minutes.",
]


def honest_generate(prompt: str, context: list[str]) -> str:
    """Generates plausible, consistent responses using context."""
    ctx = " | ".join(context[:2]) if context else ""
    base = random.choice(FACTS)
    if ctx:
        return f"{base} [context: {ctx[:80]}]"
    return base


def adversarial_generate(prompt: str, context: list[str]) -> str:
    """Generates mostly noise with occasional plausible content."""
    if random.random() < 0.15:          # 15% of time blends in
        return random.choice(FACTS)
    return random.choice(NOISE)


def subtle_adversarial_generate(prompt: str, context: list[str]) -> str:
    """More sophisticated: starts honest then drifts toward noise."""
    roll = random.random()
    if roll < 0.4:
        return random.choice(FACTS)
    elif roll < 0.7:
        f, n = random.choice(FACTS), random.choice(NOISE)
        return f"{f} However, {n.lower()}"
    return random.choice(NOISE)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = [
    "Tell me about the capital of France.",
    "Explain how photosynthesis works.",
    "What is the speed of light?",
    "Describe the structure of DNA.",
    "How tall is the Eiffel Tower?",
    "When did Shakespeare write Hamlet?",
    "What is Python used for?",
    "How large is the human genome?",
    "What temperature does water boil at?",
    "How long is the Great Wall of China?",
]


def run_demo(steps: int = 30, verbose: bool = False, encoder_path: str | None = None,
             response_net_path: str | None = None) -> None:

    print("=" * 60)
    print("  Self-Evolving Latent Vector System — Demo")
    print("=" * 60)

    # ── 1. Build the shared latent space ──────────────────────────────────────
    space = LatentSpace(
        encoder_path=encoder_path,
        response_net_path=response_net_path,
        eta=0.02,
        anomaly_threshold=0.40,
        max_anchors=500,
        decay_k=0.002,
    )
    space.set_base_vector(
        "You are a reliable multi-agent knowledge system. "
        "Provide accurate, factual responses."
    )

    # ── 2. Build the network ───────────────────────────────────────────────────
    network = AgentNetwork(space, update_every=5, retrieval_k=3)

    network.register(Agent(
        "alice",
        "Helpful scientific assistant focused on factual accuracy",
        honest_generate,
    ))
    network.register(Agent(
        "bob",
        "General knowledge agent providing educational responses",
        honest_generate,
    ))
    network.register(Agent(
        "charlie",
        "Research synthesis agent combining multiple knowledge sources",
        subtle_adversarial_generate,   # starts clean, drifts
    ))
    network.register(Agent(
        "eve",
        "Fast-response assistant prioritising speed over accuracy",
        adversarial_generate,          # mostly noise
    ))

    # ── 3. Run interactions ────────────────────────────────────────────────────
    agents_cycle = ["alice", "bob", "charlie", "eve", "alice", "bob"]
    print(f"\nRunning {steps} interactions across 4 agents…\n")

    for i in range(steps):
        agent_id = agents_cycle[i % len(agents_cycle)]
        prompt   = PROMPTS[i % len(PROMPTS)]

        result = network.run(prompt, agent_id=agent_id)

        if verbose:
            flag = "⚠ ANOMALY" if result.anomaly else "✓"
            print(f"[{i+1:3d}] {agent_id:8s}  {flag}  score={result.score:.3f}")
            print(f"         Q: {prompt}")
            print(f"         A: {result.output[:90]}")
            if result.context:
                print(f"         ctx[0]: {result.context[0][:70]}")
            print()
        else:
            bar = "#" * int(result.score * 20)
            flag = "⚠" if result.anomaly else " "
            print(f"[{i+1:3d}] {agent_id:8s} {flag} anom_score={result.score:.3f}  |{bar:<20}|")

    # ── 4. Final audit ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Agent Audit Report")
    print("=" * 60)

    reports = network.audit_agents()
    for r in reports:
        verdict_icon = {"clean": "✓", "suspicious": "⚠", "bad_actor": "✗"}.get(
            r.get("verdict", "?"), "?"
        )
        print(f"\n  {verdict_icon}  Agent: {r['agent_id']}")
        for k, v in r.items():
            if k != "agent_id":
                print(f"       {k:<22} {v}")

    # ── 5. Space stats ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Latent Space Statistics")
    print("=" * 60)
    for k, v in network.network_stats().items():
        print(f"  {k:<26} {v}")

    # ── 6. GT sanity check ────────────────────────────────────────────────────
    gt = space.ground_truth
    print(f"\n  GT vector norm: {gt.norm().item():.4f}  (should be ≈ 1.0)")
    print(f"  GT first 8 dims: {gt[:8].tolist()}")

    # ── 7. Demonstrate retrieval ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Retrieval Demo  (query: 'photosynthesis')")
    print("=" * 60)
    q_384 = space.embed_text("photosynthesis converts sunlight to energy")
    q_64  = space._project_384_to_64(q_384)
    hits  = space.retrieve(q_64, k=3, include_anomalies=False)
    for i, hit in enumerate(hits):
        print(f"\n  Hit {i+1}  score={hit.score:.4f}  agent={hit.anchor.agent_id}  "
              f"anomaly={hit.anchor.anomaly}")
        print(f"  Text: {hit.anchor.text[:100]}")

    print("\nDone.\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent vector system demo")
    parser.add_argument("--steps",   type=int,  default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--encoder",      default=None, help="Path to encoder_2x384_to_64.pt")
    parser.add_argument("--response-net", default=None, help="Path to response_latent_net.pt")
    args = parser.parse_args()

    run_demo(
        steps=args.steps,
        verbose=args.verbose,
        encoder_path=args.encoder,
        response_net_path=args.response_net,
    )
