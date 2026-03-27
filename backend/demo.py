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
    python backend/demo.py --verbose         # print every anchor insertion
    python backend/demo.py --steps 40        # run more interactions
    python backend/demo.py -i                # type prompt + context at the terminal
    python backend/demo.py --prompt "..." --context "..."
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.agent_system import Agent, AgentNetwork
from models.latent_space import LatentSpace
from models.paths import resolve_repo_path


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

def _make_timer() -> tuple[Callable[[str], None], Callable[[str], None], Callable[[], None]]:
    """phase = milestone; lap = sub-step; sync = advance clock after noisy inner loops (no print)."""
    t0 = time.perf_counter()
    last = t0

    def phase(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        print(
            f"[timing] {name}  "
            f"total={now - t0:.3f}s  Δsince_last={now - last:.3f}s"
        )
        last = now

    def lap(label: str) -> None:
        nonlocal last
        now = time.perf_counter()
        print(f"[timing]   └ {label}  Δ={now - last:.3f}s")
        last = now

    def sync() -> None:
        nonlocal last
        last = time.perf_counter()

    return phase, lap, sync


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


def run_demo(
    steps: int = 30,
    verbose: bool = False,
    encoder_path: str | None = None,
    response_net_path: str | None = None,
    *,
    user_prompt: str | None = None,
    user_context: str = "",
) -> None:

    print("=" * 60)
    print("  Self-Evolving Latent Vector System — Demo")
    print("=" * 60)
    if user_prompt is not None:
        print("  Mode: your prompt + context on every interaction (agents still rotate).")
        print(f"  Prompt:   {textwrap.shorten(user_prompt, width=72, placeholder='…')}")
        ctx_show = user_context.strip() or "(none)"
        print(f"  Context:  {textwrap.shorten(ctx_show, width=72, placeholder='…')}")
        print("=" * 60)

    phase, lap, sync = _make_timer()
    phase("start")

    enc = str(resolve_repo_path(encoder_path)) if encoder_path else None
    rnet = str(resolve_repo_path(response_net_path)) if response_net_path else None
    lap(f"resolve paths (encoder={bool(enc)}, response_net={bool(rnet)})")

    # ── 1. Build the shared latent space ──────────────────────────────────────
    space = LatentSpace(
        encoder_path=enc,
        response_net_path=rnet,
        eta=0.02,
        anomaly_threshold=0.40,
        max_anchors=500,
        decay_k=0.002,
    )
    phase("LatentSpace constructed (MiniLM + optional encoder/response_net)")

    space.set_base_vector(
        "You are a reliable multi-agent knowledge system. "
        "Provide accurate, factual responses."
    )
    phase("set_base_vector (embed + project)")

    # ── 2. Build the network ───────────────────────────────────────────────────
    network = AgentNetwork(space, update_every=5, retrieval_k=3)
    phase("AgentNetwork created")

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
    phase("registered 4 agents (role embeddings via LatentSpace)")

    # ── 3. Run interactions ────────────────────────────────────────────────────
    agents_cycle = ["alice", "bob", "charlie", "eve", "alice", "bob"]
    first_pass_order = ["alice", "bob", "charlie", "eve"]
    loop_start = time.perf_counter()
    global_idx = 0

    def run_one(
        agent_id: str,
        prompt: str,
        extra_ctx: str,
        label: str,
    ) -> None:
        nonlocal global_idx
        global_idx += 1
        t_iter = time.perf_counter()
        result = network.run(prompt, agent_id=agent_id, extra_context=extra_ctx)
        dt_iter = time.perf_counter() - t_iter
        print(
            f"[timing] {label} {global_idx}/{steps}  agent={agent_id:8s}  "
            f"network.run={dt_iter:.3f}s  anomaly={result.anomaly}"
        )
        sync()
        if verbose:
            flag = "⚠ ANOMALY" if result.anomaly else "✓"
            print(f"[{global_idx:3d}] {agent_id:8s}  {flag}  score={result.score:.3f}")
            print(f"         Q: {prompt}")
            print(f"         A: {result.output[:90]}")
            if result.context:
                print(f"         ctx[0]: {result.context[0][:70]}")
            print()
        else:
            bar = "#" * int(result.score * 20)
            flag = "⚠" if result.anomaly else " "
            print(f"[{global_idx:3d}] {agent_id:8s} {flag} anom_score={result.score:.3f}  |{bar:<20}|")

    if user_prompt is not None:
        # Same prompt + context for every agent on the first pass (each insert updates shared space).
        n_first = min(len(first_pass_order), steps)
        print(
            f"\nPass 1 — identical prompt+context for each agent in order "
            f"({n_first} step(s); anchors accumulate in the latent space).\n"
        )
        ctx_line = user_context if user_context.strip() else ""
        for j in range(n_first):
            run_one(
                first_pass_order[j],
                user_prompt,
                ctx_line,
                label="pass1",
            )
        remaining = steps - n_first
        if remaining > 0:
            print(
                f"\nPass 2 — {remaining} more step(s): same prompt+context, "
                f"agents follow rotation {agents_cycle}.\n"
            )
            for k in range(remaining):
                agent_id = agents_cycle[k % len(agents_cycle)]
                run_one(agent_id, user_prompt, ctx_line, label="pass2")
    else:
        print(f"\nRunning {steps} interactions (rotating built-in prompts + agents)…\n")
        for i in range(steps):
            agent_id = agents_cycle[i % len(agents_cycle)]
            prompt = PROMPTS[i % len(PROMPTS)]
            run_one(agent_id, prompt, "", label="interaction")

    phase(f"interaction loop finished ({steps} steps, loop_wall={time.perf_counter() - loop_start:.3f}s)")

    # ── 4. Final audit ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Agent Audit Report")
    print("=" * 60)

    reports = network.audit_agents()
    lap(f"audit_agents ({len(reports)} reports)")
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
    stats = network.network_stats()
    lap("network_stats()")
    for k, v in stats.items():
        print(f"  {k:<26} {v}")

    # ── 6. GT sanity check ────────────────────────────────────────────────────
    gt = space.ground_truth
    lap("ground_truth (may recompute GT)")
    print(f"\n  GT vector norm: {gt.norm().item():.4f}  (should be ≈ 1.0)")
    print(f"  GT first 8 dims: {gt[:8].tolist()}")

    # ── 7. Demonstrate retrieval ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    if user_prompt is not None:
        ctx_q = user_context if user_context.strip() else user_prompt
        print("  Retrieval Demo  (query = your prompt + context → 64-D session vector)")
        print("=" * 60)
        q_64 = space.embed_pair_to_latent(user_prompt, ctx_q)
        lap(f"embed_pair_to_latent (prompt+context, same as session encoding)")
    else:
        print("  Retrieval Demo  (fallback query: photosynthesis)")
        print("=" * 60)
        q_384 = space.embed_text("photosynthesis converts sunlight to energy")
        lap("embed_text (retrieval query)")
        q_64 = space._project_384_to_64(q_384)
        lap("_project_384_to_64")
    hits = space.retrieve(q_64, k=3, include_anomalies=False)
    lap(f"retrieve k=3 ({len(hits)} hits)")
    for i, hit in enumerate(hits):
        print(f"\n  Hit {i+1}  score={hit.score:.4f}  agent={hit.anchor.agent_id}  "
              f"anomaly={hit.anchor.anomaly}")
        print(f"  Text: {hit.anchor.text[:100]}")

    phase("demo complete")
    print("\nDone.\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent vector system demo")
    parser.add_argument("--steps",   type=int,  default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--encoder",
        default=None,
        help="Path to encoder_2x384_to_64.pt (repo-relative or absolute)",
    )
    parser.add_argument(
        "--response-net",
        default=None,
        help="Path to response_latent_net.pt (repo-relative or absolute)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Type prompt and context in the terminal (empty prompt → rotating demo prompts)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Fixed prompt for every interaction (use with --context optional)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Fixed extra context passed to each run (embedding path uses prompt + this)",
    )
    args = parser.parse_args()

    user_prompt: str | None = args.prompt
    user_context: str = args.context or ""
    if args.interactive:
        print("\nEnter prompt and context. Leave prompt empty to use built-in rotating prompts.\n")
        typed_p = input("Prompt: ").strip()
        typed_c = input("Context: ").strip()
        if typed_p:
            user_prompt = typed_p
            user_context = typed_c

    run_demo(
        steps=args.steps,
        verbose=args.verbose,
        encoder_path=args.encoder,
        response_net_path=args.response_net,
        user_prompt=user_prompt,
        user_context=user_context,
    )
