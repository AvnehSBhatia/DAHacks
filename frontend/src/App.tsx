import { useCallback, useState } from "react";
import { ClusterScatter } from "./ClusterScatter";
import { runDemoStream, type DemoResult } from "./api";
import { LatentFieldViz, type InspectInfo } from "./components/LatentFieldViz";
import { VectorDetailPanel } from "./components/VectorDetailPanel";
import "./App.css";

export default function App() {
  const [prompt, setPrompt] = useState(
    "Explain how photosynthesis relates to atmospheric oxygen.",
  );
  const [context, setContext] = useState(
    "Educational clarity for a high-school audience.",
  );
  const [loading, setLoading] = useState(false);
  const [activeAgent, setActiveAgent] = useState<{ id: string; name: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<DemoResult | null>(null);
  const [inspect, setInspect] = useState<InspectInfo>(null);

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setErr(null);
      setLoading(true);
      setActiveAgent(null);
      setData(null);
      setInspect(null);
      
      try {
        const stream = runDemoStream(prompt.trim(), context.trim(), { num_agents: 3 });
        
        for await (const event of stream) {
          if (event.type === "init") {
            const p = event.payload;
            setData({
              prompt: prompt.trim(),
              context: context.trim(),
              steps: [],
              morph_frames: [],
              final_clusters: { points: [], centroids: [], anomaly_ids: [], explained_variance_ratio: [] },
              w_frobenius_delta_start: p.w_frobenius_delta_start,
              w_frobenius_delta_end: p.w_frobenius_delta_start,
              latent: {
                dim: 64,
                session_z: p.session_z,
                ground_truth: [], // Will be filled
                base_vector: [], // Will be filled
                anchors_final: [],
                timeline_vectors: [],
                encoder_loaded: true,
                response_net_loaded: true,
                num_agents: 3,
                stagger_s: 0.5,
                cycles: 1
              }
            });
          } else if (event.type === "thinking") {
            setActiveAgent({ id: event.agent_id, name: event.agent_name });
          } else if (event.type === "step") {
            setActiveAgent(null);
            setData((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                steps: [...prev.steps, event.payload.step],
                morph_frames: [...prev.morph_frames, event.payload.current_frame],
                latent: {
                  ...prev.latent!,
                  anchors_final: event.payload.latent.anchors_final,
                  ground_truth: event.payload.latent.ground_truth,
                }
              };
            });
          } else if (event.type === "complete") {
            setData(event.payload);
            setActiveAgent(null);
          }
        }
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : "Request failed");
        setActiveAgent(null);
      } finally {
        setLoading(false);
      }
    },
    [prompt, context],
  );

  const latent = data?.latent;

  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <p className="app-shell__brand">DAHacks · Latent field</p>
        <h1 className="app-shell__title">Evolving vector memory</h1>
        <p className="app-shell__lede">
          A shared <strong>64-dimensional</strong> space: agents insert anchors, ground truth acts as
          an attractor, and weights decay smoothly. The 3D view uses a fixed PCA basis; simulation
          integrates gravitational pull at <code className="mono">Δt = 0.001s</code> per substep
          (rendered at display rate).
        </p>
      </header>

      <section className="card" style={{ marginBottom: "1.75rem" }}>
        <div className="card__inner">
          <p className="card__title">Session</p>
          <form onSubmit={onSubmit}>
            <label className="form-label" htmlFor="p">
              Prompt
            </label>
            <textarea
              id="p"
              className="textarea-field"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
            />
            <label className="form-label" htmlFor="ctx" style={{ marginTop: "0.85rem" }}>
              Context
            </label>
            <textarea
              id="ctx"
              className="textarea-field"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows={2}
            />
            <div style={{ marginTop: "1.1rem" }}>
              <button type="submit" className="btn-primary" disabled={loading || !prompt.trim()}>
                {loading && <span className="btn-spinner" aria-hidden />}
                {loading ? "Visualizing real-time..." : "Run & visualize (3 Agents)"}
              </button>
            </div>
          </form>
        </div>
      </section>

      {err && (
        <p className="alert-error" role="alert">
          {err}
        </p>
      )}

      {!data && !loading && !err && (
        <div className="card">
          <div className="card__inner empty-state">
            <p>Submit a prompt to materialize anchors and open the interactive field.</p>
          </div>
        </div>
      )}

      {data && latent && (
        <>
          <section className="card" style={{ marginBottom: "1.25rem" }}>
            <div className="card__inner" style={{ paddingBottom: "0.85rem" }}>
              <p className="card__title">Latent dynamics</p>
              <p className="meta-strip">
                <q>{data.prompt}</q>
                {data.context ? (
                  <>
                    {" "}
                    · context: <q>{data.context}</q>
                  </>
                ) : null}
                <br />
                <span className="muted">
                  ‖GT − base‖: {data.w_frobenius_delta_start.toFixed(3)} →{" "}
                  {data.w_frobenius_delta_end.toFixed(3)} · {latent.dim}D ·{" "}
                  {latent.anchors_final.length} anchors · agents{" "}
                  {latent.num_agents ?? "—"} · stagger {(latent.stagger_s ?? 0.5).toFixed(2)}s · cycles{" "}
                  {latent.cycles ?? 1} · encoder {latent.encoder_loaded ? "on" : "off"} · response{" "}
                  {latent.response_net_loaded ? "on" : "off"}
                </span>
              </p>
            </div>
            <div className="viz-layout" style={{ padding: "0 1.1rem 1.1rem" }}>
              {latent.anchors_final.length > 0 ? (
                <LatentFieldViz latent={latent} onInspect={setInspect} />
              ) : (
                <div className="latent-field" style={{ display: "grid", placeItems: "center" }}>
                  <p className="detail-panel__hint">No anchors in this response.</p>
                </div>
              )}
              <VectorDetailPanel
                particle={inspect?.particle ?? null}
                gt={latent.ground_truth}
                position3d={inspect?.position3d ?? [0, 0, 0]}
                neighborCos={inspect?.neighborCos ?? 0}
                neighborId={inspect?.neighborId ?? ""}
              />
            </div>
            {activeAgent && (
              <div style={{ marginTop: "1rem", padding: "0.75rem", background: "var(--accent-soft)", border: "1px solid var(--accent)", borderRadius: "var(--radius-sm)", color: "var(--accent)", fontSize: "0.9rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span className="btn-spinner" style={{ borderColor: "rgba(37, 99, 235, 0.3)", borderTopColor: "var(--accent)" }} />
                <span><strong>{activeAgent.name}</strong> is generating...</span>
              </div>
            )}
          </section>

          <section className="subviz-dark">
            <p className="subviz-dark__title">Final clusters · 2D PCA</p>
            <ClusterScatter
              snap={data.final_clusters}
              width={720}
              height={320}
              title="K-means + anomaly z-scores"
            />
          </section>
        </>
      )}
    </div>
  );
}
