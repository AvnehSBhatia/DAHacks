"""
agent_system.py
───────────────
Multi-agent wrapper over LatentSpace.

Each Agent:
    • has a role / attribute description that becomes its query vector
    • retrieves context from the shared LatentSpace before generating
    • inserts its output back as a new anchor
    • gets continuously scored for anomaly rate and GT divergence

AgentNetwork:
    • manages a pool of agents
    • runs the full interaction loop
    • surfaces verdicts on bad actors
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from latent_space import Anchor, LatentSpace, RetrievalHit, Vec


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    """
    A single participant in the multi-agent network.

    Parameters
    ──────────
    agent_id        unique string identifier
    role_text       natural-language description of this agent's purpose
                    (used to build its fixed query vector)
    generate_fn     callable(prompt, context_texts) -> str
                    the agent's actual generation logic —
                    could be an LLM call, rule-based, adversarial, etc.
    trust_score     starts at 1.0; updated by the network over time
    """
    agent_id:     str
    role_text:    str
    generate_fn:  Callable[[str, list[str]], str]
    trust_score:  float = 1.0

    # ── built lazily after the agent is registered ────────────────────────────
    query_vector: Vec | None = field(default=None, repr=False)

    # ── per-agent history ─────────────────────────────────────────────────────
    n_interactions:    int   = 0
    n_anomalies:       int   = 0
    cumulative_impact: float = 0.0

    def mean_impact(self) -> float:
        if self.n_interactions == 0:
            return 0.0
        return self.cumulative_impact / self.n_interactions


# ══════════════════════════════════════════════════════════════════════════════
# AgentNetwork
# ══════════════════════════════════════════════════════════════════════════════

class AgentNetwork:
    """
    Orchestrates a pool of agents sharing a single LatentSpace.

    Quick-start
    ───────────
        space   = LatentSpace(encoder_path="encoder_2x384_to_64.pt",
                              response_net_path="response_latent_net.pt")
        network = AgentNetwork(space)

        network.register(Agent("alice", "helpful assistant", my_llm_fn))
        network.register(Agent("bob",   "coding agent",      code_fn))
        network.register(Agent("eve",   "adversarial agent", bad_fn))

        space.set_base_vector("You are a helpful multi-agent system.")

        result = network.run("What is the capital of France?", agent_id="alice")
        print(result.output)

        for verdict in network.audit_agents():
            print(verdict)
    """

    def __init__(
        self,
        space: LatentSpace,
        update_every: int   = 5,      # run update_cycle() every N interactions
        retrieval_k:  int   = 5,      # how many anchors to retrieve
        bad_actor_threshold: float = 0.35,  # anomaly_rate above which = suspicious
    ) -> None:
        self.space                = space
        self.update_every         = update_every
        self.retrieval_k          = retrieval_k
        self.bad_actor_threshold  = bad_actor_threshold

        self._agents:      dict[str, Agent] = {}
        self._interaction_count: int        = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Agent registration
    # ══════════════════════════════════════════════════════════════════════════

    def register(self, agent: Agent) -> None:
        """Register an agent and precompute its query vector from role_text."""
        # Build query vector: project role text to 64D
        role_384     = self.space.embed_text(agent.role_text)
        agent.query_vector = self.space._project_384_to_64(role_384)
        self._agents[agent.agent_id] = agent
        print(f"[Network] Registered agent '{agent.agent_id}' | role: {agent.role_text[:60]}")

    def get_agent(self, agent_id: str) -> Agent:
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return self._agents[agent_id]

    # ══════════════════════════════════════════════════════════════════════════
    # Full interaction loop
    # ══════════════════════════════════════════════════════════════════════════

    @dataclass
    class InteractionResult:
        agent_id:   str
        prompt:     str
        output:     str
        context:    list[str]
        anchor:     Anchor
        anomaly:    bool
        score:      float         # composite anomaly score for this output

    def run(
        self,
        prompt: str,
        agent_id: str,
        extra_context: str = "",
    ) -> "AgentNetwork.InteractionResult":
        """
        Full pipeline for one agent interaction:

            1. Build query vector from agent role
            2. Retrieve top-K anchors from the space
            3. Call agent.generate_fn(prompt, context_texts)
            4. Embed the output (via ResponseLatentNet or fallback)
            5. Insert into space (anomaly check included)
            6. Optionally run update_cycle()
            7. Update agent trust score

        Returns an InteractionResult with all diagnostics attached.
        """
        agent = self.get_agent(agent_id)
        self._interaction_count += 1

        # ── Step 1: query vector ───────────────────────────────────────────────
        qv = agent.query_vector   # [64] precomputed from role

        # ── Step 2: retrieve context ───────────────────────────────────────────
        hits: list[RetrievalHit] = self.space.retrieve(
            qv, k=self.retrieval_k
        )
        context_texts = [h.anchor.text for h in hits]

        # ── Step 3: generate ───────────────────────────────────────────────────
        output = agent.generate_fn(prompt, context_texts)

        # ── Step 4: embed output ───────────────────────────────────────────────
        combined_text = (extra_context + " " + prompt + " " + output).strip()
        if self.space._response_net is not None:
            # Use the ResponseLatentNet fusion path
            resp_384 = self.space.embed_text(combined_text)
            z_new    = self.space.embed_response_to_latent(resp_384, qv)
        else:
            z_new = self.space.embed_pair_to_latent(prompt, output)

        # ── Step 5: insert with anomaly check ─────────────────────────────────
        anchor, anom_result = self.space.insert(
            z_new, text=output, agent_id=agent_id
        )

        # ── Step 6: update agent stats ─────────────────────────────────────────
        agent.n_interactions += 1
        if anom_result.is_anomaly:
            agent.n_anomalies += 1

        # ── Step 7: maintenance ────────────────────────────────────────────────
        if self._interaction_count % self.update_every == 0:
            self.space.update_cycle()
            self._update_trust_scores()

        anom_rate = agent.n_anomalies / max(1, agent.n_interactions)

        return AgentNetwork.InteractionResult(
            agent_id=agent_id,
            prompt=prompt,
            output=output,
            context=context_texts,
            anchor=anchor,
            anomaly=anom_result.is_anomaly,
            score=anom_result.distance,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Trust scores
    # ══════════════════════════════════════════════════════════════════════════

    def _update_trust_scores(self) -> None:
        """
        Recompute trust for every registered agent using the space's
        agent_anomaly_score() method.  Trust decays toward 0 for bad actors.
        """
        for aid, agent in self._agents.items():
            report = self.space.agent_anomaly_score(aid)
            if "verdict" not in report or report["verdict"] == "no_data":
                continue
            # Trust = 1 − anomaly_rate weighted by GT divergence
            anom_r  = report.get("anomaly_rate",   0.0)
            gt_div  = report.get("gt_divergence",  0.0)
            raw     = 1.0 - (0.6 * anom_r + 0.4 * gt_div)
            # EMA smooth so a brief glitch doesn't instantly destroy a clean agent
            agent.trust_score = 0.85 * agent.trust_score + 0.15 * max(0.0, raw)

    # ══════════════════════════════════════════════════════════════════════════
    # Auditing
    # ══════════════════════════════════════════════════════════════════════════

    def audit_agents(self) -> list[dict]:
        """
        Return a full anomaly report for every registered agent,
        sorted from worst (highest anomaly rate) to best.
        """
        reports = []
        for aid in self._agents:
            report = self.space.agent_anomaly_score(aid)
            report["trust_score"] = round(self._agents[aid].trust_score, 3)
            reports.append(report)
        reports.sort(key=lambda r: r.get("anomaly_rate", 0.0), reverse=True)
        return reports

    def bad_actors(self, threshold: float | None = None) -> list[str]:
        """Return agent IDs whose verdict is 'bad_actor' or 'suspicious'."""
        threshold = threshold or self.bad_actor_threshold
        return [
            r["agent_id"]
            for r in self.audit_agents()
            if r.get("verdict") in ("bad_actor", "suspicious")
        ]

    def network_stats(self) -> dict:
        base = self.space.stats()
        base["total_interactions"] = self._interaction_count
        base["registered_agents"]  = list(self._agents.keys())
        return base
