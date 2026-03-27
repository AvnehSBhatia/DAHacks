"""
latent_space.py
───────────────
Core latent vector system.

Integrates with your existing encoder artefacts:
    • Encoder2x384To64   (encoder_2x384_to_64.pt)
    • Decoder64To2x384   (decoder_64_to_2x384.pt)
    • ResponseLatentNet  (response_latent_net.pt)
    • SentenceTransformer all-MiniLM-L6-v2

Public surface
──────────────
    LatentSpace          — the main object
    Anchor               — dataclass stored in the space
    AnomalyResult        — returned by anomaly check

Deformation
───────────
Each anchor is stretched along the consensus direction n̂ = normalize(GT − base)
on insertion, with the stretch decaying back to 1.0 over time:

    stretch(t) = 1 + (s0 − 1) / (1 + exp(k_def × age))

    v_deformed = normalize(v + (stretch − 1) × (v · n̂) × n̂)

Anomalous anchors get the opposite treatment — their stretch factor is
inverted below 1.0 (compression), pushing them further from the consensus
direction and making them easier to flag on subsequent anomaly checks.

    stretch_anomaly = 1 / stretch(t)   →  compresses along n̂
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .device import autodetect_device_str, qr_reduced
from .sentence_transformer_loader import load_sentence_transformer

Vec = torch.Tensor   # shape [D] — always on CPU, float32, L2-normalised


# ══════════════════════════════════════════════════════════════════════════════
# Anchor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Anchor:
    """
    A single knowledge anchor in the 64-D latent space.

    Fields
    ──────
    vector          normalised 64-D tensor (CPU float32)
    vector_original the raw vector at insertion, before any deformation
    text            raw text that was embedded
    agent_id        which agent produced this anchor
    timestamp       unix time of insertion
    impact          contextual value  ∈ [0, 1]
    weight          effective influence after decay  ∈ [0, 1]
    anomaly         True if this anchor failed the anomaly gate
    penalized       True if downweighted after post-hoc review
    stretch_s0      initial stretch factor applied at insertion
    """
    vector:          Vec
    text:            str
    agent_id:        str
    timestamp:       float = field(default_factory=time.time)
    impact:          float = 0.5
    weight:          float = 1.0
    anomaly:         bool  = False
    penalized:       bool  = False
    id:              str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vector_original: Vec   = field(default=None, repr=False)
    stretch_s0:      float = 1.0

    def __post_init__(self):
        if self.vector_original is None:
            self.vector_original = self.vector.clone()

    # ── weight decay ──────────────────────────────────────────────────────────

    def logistic_weight(self, now: float, k_base: float = 0.01) -> float:
        """
        w(t) = w0 / (1 + exp(k × age))
        High-impact anchors decay slowly (small k).
        """
        age = now - self.timestamp
        k   = k_base * (1.0 - 0.9 * self.impact)
        return self.weight / (1.0 + math.exp(k * age))

    def apply_decay(self, now: float, k_base: float = 0.01) -> None:
        self.weight = self.logistic_weight(now, k_base)

    # ── stretch decay ─────────────────────────────────────────────────────────

    def current_stretch(self, now: float, k_def: float = 0.015) -> float:
        """
        stretch(t) = 1 + (s0 − 1) / (1 + exp(k_def × age))

        Decays back toward 1.0 (no deformation) over time.
        For anomalies s0 < 1, so this decays from compression back to neutral.
        """
        age = now - self.timestamp
        return 1.0 + (self.stretch_s0 - 1.0) / (1.0 + math.exp(k_def * age))


# ══════════════════════════════════════════════════════════════════════════════
# Named tuples
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalHit(NamedTuple):
    anchor: Anchor
    score:  float

class AnomalyResult(NamedTuple):
    is_anomaly:  bool
    distance:    float
    threshold:   float


# ══════════════════════════════════════════════════════════════════════════════
# LatentSpace
# ══════════════════════════════════════════════════════════════════════════════

class LatentSpace:
    """
    A self-evolving 64-D Euclidean anchor store with directional deformation.

    Parameters
    ──────────
    minilm_model            HF name for SentenceTransformer
    encoder_path            path to encoder_2x384_to_64.pt   (optional)
    response_net_path       path to response_latent_net.pt   (optional)
    dim                     latent dimension (64)
    eta                     gravitational learning rate  (0.01–0.05)
    anomaly_threshold       composite cosine distance threshold
    impact_alpha            EMA weight for impact updates
    max_anchors             prune when store exceeds this
    penalty_factor          weight/impact multiplier for anomalies
    decay_k                 base weight decay rate
    stretch_s0              initial stretch for clean anchors (>1 = elongate)
    stretch_k               deformation decay rate (faster than decay_k)
    min_anchors_for_deform    don't deform until GT is stable enough
    min_anchors_for_detection  skip anomaly scoring below this count (defaults to min_anchors_for_deform)
    device                    None = autodetect (cuda → mps → cpu), or 'cuda'|'mps'|'cpu'
    """

    def __init__(
        self,
        minilm_model:           str   = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_path:           str | None = None,
        response_net_path:      str | None = None,
        dim:                    int   = 64,
        eta:                    float = 0.02,
        anomaly_threshold:      float = 0.45,
        impact_alpha:           float = 0.15,
        max_anchors:            int   = 2000,
        penalty_factor:         float = 0.1,
        decay_k:                float = 0.005,
        stretch_s0:             float = 1.3,
        stretch_k:              float = 0.015,
        min_anchors_for_deform: int   = 10,
        min_anchors_for_detection: int | None = None,
        device:                 str | None = None,
    ) -> None:
        self.dim                    = dim
        self.eta                    = eta
        self.anomaly_threshold      = anomaly_threshold
        self.impact_alpha           = impact_alpha
        self.max_anchors            = max_anchors
        self.penalty_factor         = penalty_factor
        self.decay_k                = decay_k
        self.stretch_s0             = stretch_s0
        self.stretch_k              = stretch_k
        self.min_anchors_for_deform = min_anchors_for_deform
        self.min_anchors_for_detection = (
            min_anchors_for_detection
            if min_anchors_for_detection is not None
            else min_anchors_for_deform
        )
        dev_str = device if device is not None else autodetect_device_str()
        self.device                 = torch.device(dev_str)

        print(f"[LatentSpace] Device: {dev_str}  (set TORCH_DEVICE to override)")
        self._st = load_sentence_transformer(minilm_model, dev_str)

        self._encoder      = None
        self._response_net = None
        if encoder_path:
            self._encoder = self._load_encoder(encoder_path)
        if response_net_path:
            self._response_net = self._load_response_net(response_net_path)

        self.anchors:       list[Anchor] = []
        self._base_vector:  Vec          = torch.zeros(dim)
        self._gt:           Vec          = torch.zeros(dim)
        self._gt_dirty:     bool         = True
        self._deform_axis:  Vec | None   = None   # n̂, invalidated each GT update

        self.total_inserted   = 0
        self.total_rejected   = 0
        self.total_gt_updates = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Deformation helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _deform(self, v: Vec, s: float, axis: Vec) -> Vec:
        """
        Stretch v by factor s along axis n̂:
            v' = normalize(v + (s − 1) × (v · n̂) × n̂)

        s > 1 → elongate along n̂  (consensus amplification for clean anchors)
        s < 1 → compress along n̂  (anomaly penalty — pushes them off-axis)
        s = 1 → identity
        """
        if abs(s - 1.0) < 1e-6:
            return v
        projection = (v @ axis).item()
        v_deformed = v + (s - 1.0) * projection * axis
        return F.normalize(v_deformed, dim=0)

    def _consensus_axis(self) -> Vec | None:
        """
        n̂ = normalize(GT − base_vector)

        This is the direction the space has drifted since initialisation —
        the natural axis to amplify clean signal and penalise noise along.
        Returns None if GT == base (no directional signal yet).
        Cached between GT recomputes.
        """
        if self._deform_axis is not None:
            return self._deform_axis
        diff = self.ground_truth - self._base_vector
        norm = diff.norm()
        if norm < 1e-6:
            return None
        self._deform_axis = diff / norm
        return self._deform_axis

    # ══════════════════════════════════════════════════════════════════════════
    # Embedding
    # ══════════════════════════════════════════════════════════════════════════

    def embed_text(self, text: str) -> Vec:
        with torch.no_grad():
            v = self._st.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return v.cpu().to(torch.float32)

    def embed_pair_to_latent(self, prompt: str, context: str) -> Vec:
        ep = self.embed_text(prompt)
        ec = self.embed_text(context)
        if self._encoder is not None:
            x = torch.stack([ep, ec], dim=0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z = self._encoder(x).squeeze(0).cpu()
            return F.normalize(z, dim=0)
        mean_384 = F.normalize((ep + ec) / 2.0, dim=0)
        return self._project_384_to_64(mean_384)

    def embed_response_to_latent(self, response_384: Vec, context_64: Vec) -> Vec:
        if self._response_net is not None:
            r = response_384.unsqueeze(0).to(self.device)
            z = context_64.unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_out, _, _ = self._response_net(r, z)
            return F.normalize(z_out.squeeze(0).cpu(), dim=0)
        return self._project_384_to_64(response_384)

    def _project_384_to_64(self, v384: Vec) -> Vec:
        if not hasattr(self, "_proj"):
            g = torch.Generator()
            g.manual_seed(42)
            P = torch.randn(384, self.dim, generator=g)
            Q, _ = torch.linalg.qr(P)
            self._proj = Q[:, : self.dim]
        return F.normalize(v384 @ self._proj, dim=0)

    # ══════════════════════════════════════════════════════════════════════════
    # Ground Truth
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def ground_truth(self) -> Vec:
        if self._gt_dirty:
            self._recompute_gt()
        return self._gt

    def _recompute_gt(self) -> None:
        accum = self._base_vector.clone()
        for a in self.anchors:
            w = a.weight * a.impact
            if a.anomaly or a.penalized:
                w *= self.penalty_factor
            accum = accum + w * a.vector
        norm = accum.norm()
        self._gt          = accum / norm.clamp(min=1e-8)
        self._gt_dirty    = False
        self._deform_axis = None   # GT moved — invalidate cached axis
        self.total_gt_updates += 1

    def set_base_vector(self, text: str) -> None:
        self._base_vector = self._project_384_to_64(self.embed_text(text))
        self._gt_dirty    = True
        self._deform_axis = None

    # ══════════════════════════════════════════════════════════════════════════
    # Anomaly detection
    # ══════════════════════════════════════════════════════════════════════════

    def check_anomaly(self, z: Vec, k: int = 5) -> AnomalyResult:
        """
        Composite score = 0.6 × neighbour_distance + 0.4 × GT_divergence.

        Skips the anomaly gate (returns clean) when the space is too sparse
        to produce a stable ground-truth estimate. The warm-up size is
        ``min_anchors_for_detection`` (defaults to ``min_anchors_for_deform``).
        """
        if len(self.anchors) < self.min_anchors_for_detection:
            return AnomalyResult(False, 0.0, self.anomaly_threshold)

        stacked = torch.stack([a.vector for a in self.anchors])
        cos_sim = F.cosine_similarity(z.unsqueeze(0), stacked)
        top_k_sim, _ = cos_sim.topk(min(k, len(self.anchors)))
        mean_sim = top_k_sim.mean().item()
        distance = 1.0 - mean_sim

        gt_sim = F.cosine_similarity(
            z.unsqueeze(0), self.ground_truth.unsqueeze(0)
        ).item()
        gt_distance = 1.0 - gt_sim

        composite = 0.6 * distance + 0.4 * gt_distance

        return AnomalyResult(
            is_anomaly=composite > self.anomaly_threshold,
            distance=composite,
            threshold=self.anomaly_threshold,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Insertion
    # ══════════════════════════════════════════════════════════════════════════

    def insert(
        self,
        z: Vec,
        text: str,
        agent_id: str,
        initial_impact: float = 0.5,
        force: bool = False,
    ) -> tuple[Anchor, AnomalyResult]:
        """
        Insert a new anchor with directional deformation applied at birth.

        Clean anchors:     s0 = stretch_s0  > 1  → stretched toward GT
        Anomalous anchors: s0 = 1/stretch_s0 < 1  → compressed away from GT

        Both decay back to s=1 (neutral) over time via stretch_k,
        so the deformation is a transient amplifier, not a permanent warp.
        """
        z      = F.normalize(z, dim=0)
        result = self.check_anomaly(z)
        is_bad = result.is_anomaly and not force

        enough = len(self.anchors) >= self.min_anchors_for_deform
        axis   = self._consensus_axis() if enough else None

        if axis is not None:
            s0         = (1.0 / self.stretch_s0) if is_bad else self.stretch_s0
            z_deformed = self._deform(z, s0, axis)
        else:
            s0         = 1.0
            z_deformed = z

        anchor = Anchor(
            vector=z_deformed,
            text=text,
            agent_id=agent_id,
            impact=initial_impact,
            weight=1.0,
            anomaly=is_bad,
            vector_original=z,
            stretch_s0=s0,
        )

        if is_bad:
            anchor.weight *= self.penalty_factor
            anchor.impact *= self.penalty_factor
            self.total_rejected += 1
        else:
            self.total_inserted += 1

        self.anchors.append(anchor)
        self._gt_dirty = True

        if len(self.anchors) > self.max_anchors:
            self._prune()

        return anchor, result

    # ══════════════════════════════════════════════════════════════════════════
    # Retrieval
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query: Vec,
        k: int = 5,
        include_anomalies: bool = False,
        decay_now: float | None = None,
    ) -> list[RetrievalHit]:
        if not self.anchors:
            return []

        now        = decay_now or time.time()
        candidates = [
            a for a in self.anchors
            if include_anomalies or not a.anomaly
        ]
        if not candidates:
            return []

        stacked = torch.stack([a.vector for a in candidates])
        cos_sim = F.cosine_similarity(query.unsqueeze(0), stacked).tolist()

        hits: list[RetrievalHit] = []
        for a, sim in zip(candidates, cos_sim):
            decay_w = a.logistic_weight(now, self.decay_k)
            pen_w = 0.5 if a.penalized else 1.0
            score = sim * decay_w * a.impact * pen_w
            hits.append(RetrievalHit(anchor=a, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ══════════════════════════════════════════════════════════════════════════
    # Impact updates
    # ══════════════════════════════════════════════════════════════════════════

    def update_impact(self, anchor_id: str, utility_score: float) -> None:
        for a in self.anchors:
            if a.id == anchor_id:
                a.impact = (1 - self.impact_alpha) * a.impact + self.impact_alpha * utility_score
                a.impact = max(0.0, min(1.0, a.impact))
                self._gt_dirty = True
                return

    def batch_update_impacts(self, agent_id: str, utility_score: float) -> None:
        for a in self.anchors:
            if a.agent_id == agent_id:
                a.impact = (1 - self.impact_alpha) * a.impact + self.impact_alpha * utility_score
                a.impact = max(0.0, min(1.0, a.impact))
        self._gt_dirty = True

    # ══════════════════════════════════════════════════════════════════════════
    # Gravitational update
    # ══════════════════════════════════════════════════════════════════════════

    def gravitational_step(self) -> None:
        """
        Pull each anchor's original vector toward GT, then re-apply the
        current time-decayed deformation on top.

        Gravity acts on the true semantic position (vector_original).
        Deformation is a separate layer that decays independently.
        This keeps the two mechanisms cleanly separated.
        """
        gt   = self.ground_truth
        now  = time.time()
        axis = self._consensus_axis()

        for a in self.anchors:
            eta = self.eta * (self.penalty_factor if (a.anomaly or a.penalized) else 1.0)

            # pull the pre-deform vector toward GT
            a.vector_original = F.normalize(
                a.vector_original + eta * (gt - a.vector_original), dim=0
            )

            # re-apply decayed deformation on top of updated original
            if axis is not None:
                s_now    = a.current_stretch(now, self.stretch_k)
                a.vector = self._deform(a.vector_original, s_now, axis)
            else:
                a.vector = a.vector_original.clone()

        self._gt_dirty = True

    # ══════════════════════════════════════════════════════════════════════════
    # Decay step
    # ══════════════════════════════════════════════════════════════════════════

    def decay_step(self, now: float | None = None) -> None:
        """Apply weight decay and advance deformation decay for all anchors."""
        now  = now or time.time()
        axis = self._consensus_axis()

        for a in self.anchors:
            a.apply_decay(now, self.decay_k)
            if axis is not None:
                s_now    = a.current_stretch(now, self.stretch_k)
                a.vector = self._deform(a.vector_original, s_now, axis)

        self._gt_dirty = True

    # ══════════════════════════════════════════════════════════════════════════
    # Full update cycle
    # ══════════════════════════════════════════════════════════════════════════

    def update_cycle(self, now: float | None = None) -> None:
        """
        1. Recompute GT  (invalidates cached deform axis)
        2. Gravitational pull on original vectors, re-deform
        3. Weight + deformation decay
        4. Re-normalise
        """
        now = now or time.time()
        self._recompute_gt()
        self.gravitational_step()
        self.decay_step(now)
        for a in self.anchors:
            a.vector = F.normalize(a.vector, dim=0)
        self._gt_dirty = True

    # ══════════════════════════════════════════════════════════════════════════
    # Pruning
    # ══════════════════════════════════════════════════════════════════════════

    def _prune(self) -> None:
        def score(a: Anchor) -> float:
            penalty = 10.0 if (a.anomaly or a.penalized) else 1.0
            return (a.weight * a.impact) / penalty

        self.anchors.sort(key=score, reverse=True)
        self.anchors = self.anchors[: self.max_anchors]
        self._gt_dirty = True

    # ══════════════════════════════════════════════════════════════════════════
    # Agent-level anomaly scoring
    # ══════════════════════════════════════════════════════════════════════════

    def agent_anomaly_score(self, agent_id: str) -> dict:
        mine = [a for a in self.anchors if a.agent_id == agent_id]
        if not mine:
            return {"agent_id": agent_id, "verdict": "no_data"}

        n_anom       = sum(1 for a in mine if a.anomaly)
        anom_rate    = n_anom / len(mine)
        mean_impact  = sum(a.impact for a in mine) / len(mine)
        mean_weight  = sum(a.weight for a in mine) / len(mine)
        now          = time.time()
        mean_stretch = sum(a.current_stretch(now, self.stretch_k) for a in mine) / len(mine)

        centroid = F.normalize(
            torch.stack([a.vector for a in mine]).mean(0), dim=0
        )
        gt_div = (1.0 - F.cosine_similarity(
            centroid.unsqueeze(0), self.ground_truth.unsqueeze(0)
        ).item())

        # Health ∈ [0,1]: 1 = aligned with consensus; ≤0.5 = bottom half → suspicious + prune target
        health = 1.0 - (0.6 * anom_rate + 0.4 * gt_div)

        if anom_rate >= 0.5 or gt_div > 0.6:
            verdict = "bad_actor"
        elif health <= 0.5:
            verdict = "suspicious"
        elif anom_rate > 0.25 or gt_div > 0.35:
            verdict = "suspicious"
        else:
            verdict = "clean"

        return {
            "agent_id":          agent_id,
            "total_anchors":     len(mine),
            "anomalous_anchors": n_anom,
            "anomaly_rate":      round(anom_rate, 3),
            "mean_impact":       round(mean_impact, 3),
            "mean_weight":       round(mean_weight, 3),
            "gt_divergence":     round(gt_div, 3),
            "mean_stretch":      round(mean_stretch, 3),  # <1 = penalized, >1 = amplified
            "health":            round(health, 3),
            "verdict":           verdict,
        }

    def penalize_agent_anchors(self, agent_id: str, *, factor: float = 0.5) -> int:
        """
        Mark anchors from ``agent_id`` as post-hoc penalized: ``impact`` and ``weight``
        are scaled by ``factor``, and ``penalized`` is set so retrieval and GT treat
        them as second-class (see ``retrieve`` and ``gravitational_step``).
        Idempotent: already-penalized anchors are skipped.
        """
        n = 0
        for a in self.anchors:
            if a.agent_id != agent_id or a.penalized:
                continue
            a.penalized = True
            a.impact = max(0.0, a.impact * factor)
            a.weight = max(0.0, a.weight * factor)
            n += 1
        if n:
            self._gt_dirty = True
        return n

    # ══════════════════════════════════════════════════════════════════════════
    # Stats
    # ══════════════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        n      = len(self.anchors)
        n_anom = sum(1 for a in self.anchors if a.anomaly)
        agents = set(a.agent_id for a in self.anchors)
        axis   = self._consensus_axis()
        return {
            "total_anchors":     n,
            "anomalous_anchors": n_anom,
            "clean_anchors":     n - n_anom,
            "anomaly_rate":      round(n_anom / n, 3) if n else 0.0,
            "unique_agents":     len(agents),
            "gt_updates":        self.total_gt_updates,
            "total_inserted":    self.total_inserted,
            "total_rejected":    self.total_rejected,
            "deform_axis[:4]":   str(axis[:4].tolist()) if axis is not None else "not set",
            "stretch_s0":        self.stretch_s0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Model loading helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _load_encoder(self, path: str):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[LatentSpace] Could not load encoder from {path}: {e}")
            return None

        import torch.nn as nn
        DIM, LATENT = data.get("dim", 384), data.get("latent", 64)
        ENC_IN = 2 * DIM

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(ENC_IN, LATENT)
            def forward(self, x):
                return self.fc(x.reshape(x.size(0), -1))

        enc = _Enc()
        enc.load_state_dict(data["state_dict"])
        enc.eval()
        enc = enc.to(self.device)
        print(f"[LatentSpace] Loaded encoder from {path}")
        return enc

    def _load_response_net(self, path: str):
        import torch.nn as nn
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"[LatentSpace] Could not load response net from {path}: {e}")
            return None

        DIM, LATENT = data.get("dim", 384), data.get("latent", 64)

        class _RLN(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_head = nn.Sequential(
                    nn.Linear(LATENT, 512), nn.GELU(), nn.Linear(512, DIM * LATENT)
                )
                self.fusion = nn.Sequential(
                    nn.Linear(DIM + LATENT, 512), nn.GELU(),
                    nn.Linear(512, 256),           nn.GELU(),
                    nn.Linear(256, LATENT),
                )
            def forward(self, r, z):
                b = z.size(0)
                raw = self.W_head(z).view(b, DIM, LATENT)
                W, _ = qr_reduced(raw)
                proj = torch.einsum("bd,bdk->bk", r, W)
                z_out = self.fusion(torch.cat([r, z], dim=-1))
                return z_out, proj, W

        net = _RLN().to(self.device)
        net.load_state_dict(data["state_dict"])
        net.eval()
        print(f"[LatentSpace] Loaded ResponseLatentNet from {path}")
        return net