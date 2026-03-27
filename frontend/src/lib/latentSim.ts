/**
 * Client-side dynamics: gravitational pull toward GT + logistic-style weight decay.
 * Simulation timestep DT = 0.001 s; caller integrates multiple substeps per animation frame.
 */
import type { LatentPayload } from "../api";
import { addScaled, cosineSimilarity, normalize, sub } from "./mathVec";

export const SIM_DT = 0.001;

export type SimParticle = {
  id: string;
  agentId: string;
  v: number[];
  weight: number;
  weight0: number;
  impact: number;
  anomaly: boolean;
  penalized: boolean;
  text: string;
  timestamp: number;
  /** Simulated age in seconds (starts at 0, advances by SIM_DT per substep) */
  simAge: number;
  weightHistory: number[];
};

export type SimState = {
  particles: SimParticle[];
  gt: number[];
  base: number[];
  session: number[];
};

/** η · (penalty) — aligned loosely with LatentSpace gravitational_step (higher = more visible motion) */
const ETA = 0.12;
const DECAY_K = 0.012;
const MAX_HISTORY = 180;

export function buildSimState(latent: LatentPayload): SimState {
  const gt = normalize(latent.ground_truth);
  const base = normalize(latent.base_vector);
  const session = normalize(latent.session_z);
  const particles: SimParticle[] = latent.anchors_final.map((a) => ({
    id: a.id,
    agentId: a.agent_id,
    v: normalize(a.vector),
    weight: a.weight,
    weight0: a.weight,
    impact: a.impact,
    anomaly: a.anomaly,
    penalized: a.penalized,
    text: a.text,
    timestamp: a.timestamp ?? Date.now() / 1000,
    simAge: 0,
    weightHistory: [a.weight],
  }));
  return { particles, gt, base, session };
}

/**
 * One integration step of duration SIM_DT (seconds).
 * Pulls each particle toward GT; anomalies / penalized anchors pull weaker (suppressed).
 */
export function integrationStep(state: SimState): void {
  const { particles, gt } = state;
  for (const p of particles) {
    const diff = sub(gt, p.v);
    const pullScale =
      ETA *
      p.weight *
      (p.penalized ? 0.35 : 1) *
      (p.anomaly ? 0.45 : 1) *
      (0.5 + 0.5 * p.impact);
    const vNext = normalize(addScaled(p.v, diff, pullScale));
    p.v = vNext;

    // Logistic-style weight decay (smooth, never hits exactly zero)
    const ageFactor = 1 / (1 + Math.exp(-0.5 * p.simAge));
    const wDecay = Math.exp(-DECAY_K * SIM_DT * (1 - 0.85 * p.impact) * (1 + ageFactor));
    p.weight = Math.max(0.03, Math.min(1.5, p.weight * wDecay));
    p.simAge += SIM_DT;
    p.weightHistory.push(p.weight);
    if (p.weightHistory.length > MAX_HISTORY) p.weightHistory.shift();
  }
}

/** Run many substeps (bounded per frame for performance). */
export function integrate(state: SimState, substeps: number): void {
  const n = Math.max(0, Math.floor(substeps));
  for (let i = 0; i < n; i++) integrationStep(state);
}

export function nearestNeighborCos(
  particles: SimParticle[],
  idx: number,
): { cos: number; id: string } {
  const p = particles[idx];
  let best = -2;
  let bestId = "";
  for (let j = 0; j < particles.length; j++) {
    if (j === idx) continue;
    const c = cosineSimilarity(p.v, particles[j].v);
    if (c > best) {
      best = c;
      bestId = particles[j].id;
    }
  }
  return { cos: best, id: bestId };
}

export function decayRateDisplay(impact: number): string {
  return `μ=${impact.toFixed(3)} · w ← w·exp(−${DECAY_K}·Δt·(1−0.85μ)(1+σ(age))), Δt=${SIM_DT}s`;
}
