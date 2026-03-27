import type { SimParticle } from "../lib/latentSim";
import { decayRateDisplay } from "../lib/latentSim";
import { cosineSimilarity } from "../lib/mathVec";
import { MiniSparkline } from "./MiniSparkline";
import React from "react";

type Props = {
  particle: SimParticle | null;
  gt: number[];
  position3d: [number, number, number];
  neighborCos: number;
  neighborId: string;
};

export function VectorDetailPanel({
  particle,
  gt,
  position3d,
  neighborCos,
  neighborId,
}: Props) {
  if (!particle) {
    return (
      <div className="detail-panel detail-panel--empty">
        <p className="detail-panel__hint">Hover or click a point in the field to inspect its latent vector.</p>
      </div>
    );
  }

  const gtSim = cosineSimilarity(particle.v, gt);
  const now = Date.now() / 1000;
  const ageWall = Math.max(0, now - particle.timestamp);

  return (
    <div className="detail-panel">
      <header className="detail-panel__head">
        <span className="detail-panel__badge">Anchor</span>
        <h3 className="detail-panel__title">{particle.id}</h3>
      </header>
      <dl className="detail-grid">
        <dt>Agent</dt>
        <dd>{particle.agentId}</dd>
        <dt>Position (PCA₃)</dt>
        <dd className="mono">
          [{position3d.map((x) => x.toFixed(4)).join(", ")}]
        </dd>
        <dt>Weight</dt>
        <dd>
          {particle.weight.toFixed(4)}{" "}
          <span className="muted">(initial {particle.weight0.toFixed(4)})</span>
        </dd>
        <dt>Impact μ</dt>
        <dd>{particle.impact.toFixed(4)}</dd>
        <dt>Decay model</dt>
        <dd className="detail-formula">{decayRateDisplay(particle.impact)}</dd>
        <dt>Wall time since insert</dt>
        <dd>{ageWall.toFixed(2)} s</dd>
        <dt>Simulated age</dt>
        <dd>{particle.simAge.toFixed(3)} s</dd>
        <dt>Flags</dt>
        <dd>
          {particle.anomaly ? <span className="tag tag--warn">anomaly</span> : null}{" "}
          {particle.penalized ? <span className="tag tag--muted">penalized</span> : null}
          {!particle.anomaly && !particle.penalized ? (
            <span className="muted">clean</span>
          ) : null}
        </dd>
        <dt>Similarity to ground truth</dt>
        <dd>{gtSim.toFixed(4)}</dd>
        <dt>Nearest neighbor</dt>
        <dd>
          {neighborId ? (
            <>
              <span className="mono">{neighborId}</span> · cos = {neighborCos.toFixed(4)}
            </>
          ) : (
            "—"
          )}
        </dd>
      </dl>
      <div className="detail-panel__spark">
        <p className="detail-panel__spark-label">Weight vs. sim time</p>
        <MiniSparkline values={particle.weightHistory} />
      </div>
      <div className="detail-panel__text">
        <p className="detail-panel__text-label">Data Flow & Math Pipeline</p>
        <div style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)", padding: "0.5rem", background: "var(--card-bg)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", marginBottom: "0.75rem" }}>
          <div>1. Text → SentenceTransformer → Neural Net </div>
          <div>2. 64-D Latent Tensor generated</div>
          <div>3. <span style={{color:"var(--accent)"}}>cos(θ) = (A · B) / (||A|| ||B||)</span> calculated for nearest neighbor</div>
          <div>4. 3x64 PCA Projection Matrix computes 3D (x,y,z) coordinate</div>
        </div>
      </div>
      <div className="detail-panel__text">
        <p className="detail-panel__text-label">64-Dimensional Vector Tensor [V_0 ... V_63]</p>
        <div className="math-matrix-container">
          <div className="math-matrix">
            {particle.v.map((val, idx) => (
              <span key={idx} className="math-matrix-cell" title={`V_${idx} = ${val}`}>
                {val.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="detail-panel__text">
        <p className="detail-panel__text-label">Content</p>
        <p className="detail-panel__text-body">{particle.text || "—"}</p>
      </div>
    </div>
  );
}
