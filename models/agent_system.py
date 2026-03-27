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

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from .latent_space import Anchor, LatentSpace, RetrievalHit, Vec


def _timing_enabled() -> bool:
    return os.environ.get("DAHACKS_TIMING", "1").strip().lower() not in ("0", "false", "no")


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

    def remove_agent(self, agent_id: str) -> bool:
        """Unregister an agent (e.g. after outlier removal). Returns False if missing."""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        print(f"[Network] Removed agent '{agent_id}' from this network")
        return True

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

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
        t_run = time.perf_counter()

        # ── Step 1: query vector ───────────────────────────────────────────────
        qv = agent.query_vector   # [64] precomputed from role

        # ── Step 2: retrieve context ───────────────────────────────────────────
        t0 = time.perf_counter()
        hits: list[RetrievalHit] = self.space.retrieve(
            qv, k=self.retrieval_k
        )
        t_retrieve = time.perf_counter() - t0
        context_texts = [h.anchor.text for h in hits]

        # ── Step 3: generate ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        output = agent.generate_fn(prompt, context_texts)
        t_generate = time.perf_counter() - t0

        # ── Step 4: embed output ───────────────────────────────────────────────
        t0 = time.perf_counter()
        combined_text = (extra_context + " " + prompt + " " + output).strip()
        if self.space._response_net is not None:
            # Use the ResponseLatentNet fusion path
            resp_384 = self.space.embed_text(combined_text)
            z_new    = self.space.embed_response_to_latent(resp_384, qv)
        else:
            z_new = self.space.embed_pair_to_latent(prompt, output)
        t_embed = time.perf_counter() - t0

        # ── Step 5: insert with anomaly check ─────────────────────────────────
        t0 = time.perf_counter()
        anchor, anom_result = self.space.insert(
            z_new, text=output, agent_id=agent_id
        )
        t_insert = time.perf_counter() - t0

        # ── Step 6: update agent stats ─────────────────────────────────────────
        agent.n_interactions += 1
        if anom_result.is_anomaly:
            agent.n_anomalies += 1

        # ── Step 7: maintenance ────────────────────────────────────────────────
        t_maint = 0.0
        if self._interaction_count % self.update_every == 0:
            t0 = time.perf_counter()
            self.space.update_cycle()
            self._update_trust_scores()
            t_maint = time.perf_counter() - t0

        if _timing_enabled():
            t_total = time.perf_counter() - t_run
            print(
                f"[timing] agent={agent_id} retrieve={t_retrieve:.3f}s "
                f"generate={t_generate:.3f}s embed={t_embed:.3f}s insert={t_insert:.3f}s "
                f"maint={t_maint:.3f}s total={t_total:.3f}s",
                flush=True,
            )

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

    @dataclass
    class FinalAnswerResult:
        agent_id: str
        output: str
        weighted_context_lines: list[str]
        hits: list[RetrievalHit]

    @staticmethod
    def _anchor_text_for_display(anchor_body: str, *, max_len: int = 420) -> str:
        """Strip demo stub noise: session task is printed once above; drop nested prior-anchor tails."""
        t = (anchor_body or "").strip().replace("\n", " ")
        if t.startswith("[task:"):
            close = t.find("] ")
            if close != -1:
                t = t[close + 2 :].strip()
        if "[prior anchors:" in t:
            t = t.split("[prior anchors:")[0].strip()
        if len(t) > max_len:
            t = t[: max_len - 1] + "…"
        return t

    def _format_final_from_retrieval(
        self,
        prompt: str,
        agent_id: str,
        hits: list[RetrievalHit],
    ) -> str:
        """Readable Phase-3 report: prompt + ranked anchors (no stub generator chaining)."""
        if not hits:
            return (
                f"No anchors retrieved for this session (prompt: {prompt[:220]}…).\n"
                "Latent space may be empty or all hits filtered as anomalies."
            )
        lines: list[str] = []
        for h in hits:
            t = self._anchor_text_for_display(h.anchor.text or "")
            lines.append(
                f"  • score={h.score:.4f}  agent={h.anchor.agent_id}\n    {t}"
            )
        return (
            f"Session prompt:\n  {prompt}\n\n"
            f"Finalist agent: {agent_id}\n\n"
            f"Top anchors (session 64-D query, weighted retrieval):\n"
            + "\n".join(lines)
        )

    def final_answer_weighted(
        self,
        prompt: str,
        agent_id: str,
        query_z: Vec,
        k: int,
        *,
        synthesize: Literal["retrieval", "agent"] = "retrieval",
    ) -> "AgentNetwork.FinalAnswerResult":
        """
        Retrieve top-k anchors by cosine × decay × impact using ``query_z``.

        ``synthesize="retrieval"`` (default): format the final answer directly from
        ranked anchors — avoids feeding score-prefixed strings into stub generators,
        which previously produced duplicated ``[task:]`` / random-fact noise.

        ``synthesize="agent"``: call ``agent.generate_fn(prompt, anchor_texts)`` with
        **plain** anchor texts (scores are only in ``weighted_context_lines``).
        """
        agent = self.get_agent(agent_id)
        hits: list[RetrievalHit] = self.space.retrieve(query_z, k=k, include_anomalies=False)
        weighted_lines: list[str] = []
        for h in hits:
            weighted_lines.append(f"[sim×weight={h.score:.4f} | agent={h.anchor.agent_id}] {h.anchor.text}")
        anchor_texts = [h.anchor.text for h in hits]
        if synthesize == "retrieval":
            output = self._format_final_from_retrieval(prompt, agent_id, hits)
        else:
            output = agent.generate_fn(prompt, anchor_texts)
        return AgentNetwork.FinalAnswerResult(
            agent_id=agent_id,
            output=output,
            weighted_context_lines=weighted_lines,
            hits=hits,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Trust scores
    # ══════════════════════════════════════════════════════════════════════════

    def refresh_trust_scores(self) -> None:
        """Public alias for periodic trust recomputation (e.g. after outlier removal)."""
        self._update_trust_scores()

    def _update_trust_scores(self) -> None:
        """
        Recompute trust for every registered agent using the space's
        agent_anomaly_score() method.  Trust decays toward 0 for bad actors.
        """
        for aid, agent in self._agents.items():
            report = self.space.agent_anomaly_score(aid)
            if "verdict" not in report or report["verdict"] == "no_data":
                continue
            # Align EMA with latent-space ``health`` (same formula as agent_anomaly_score)
            raw = float(report.get("health", 1.0 - (
                0.6 * report.get("anomaly_rate", 0.0) + 0.4 * report.get("gt_divergence", 0.0)
            )))
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
