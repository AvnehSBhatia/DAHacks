import { useCallback, useEffect, useState } from "react";
import { AgentTopology } from "./AgentTopology";
import { ClusterScatter } from "./ClusterScatter";
import type { DemoResult } from "./api";
import { VectorSpace3D } from "./VectorSpace3D";

type MainTab = "prompt" | "visualization";

type Props = {
  runDemoFn: (prompt: string) => Promise<DemoResult>;
};

export function DemoExperience({ runDemoFn }: Props) {
  const [mainTab, setMainTab] = useState<MainTab>("prompt");
  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<DemoResult | null>(null);
  const [frame, setFrame] = useState(0);

  const morphFrames = data?.morph_frames ?? [];

  useEffect(() => {
    if (!morphFrames.length) return;
    const t = setInterval(() => {
      setFrame((f) => (f + 1) % morphFrames.length);
    }, 2200);
    return () => clearInterval(t);
  }, [morphFrames]);

  const currentMorph = morphFrames[frame] ?? null;
  const finalClusters = data?.final_clusters ?? null;

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setErr(null);
      setLoading(true);
      setData(null);
      try {
        const r = await runDemoFn(prompt.trim());
        setData(r);
        setFrame(0);
        setMainTab("visualization");
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : "Request failed");
      } finally {
        setLoading(false);
      }
    },
    [prompt, runDemoFn],
  );

  return (
    <>
      <nav className="tab-bar" role="tablist" aria-label="Primary">
        <button
          type="button"
          role="tab"
          id="tab-prompt"
          aria-selected={mainTab === "prompt"}
          aria-controls="panel-prompt"
          tabIndex={mainTab === "prompt" ? 0 : -1}
          className="tab-btn"
          onClick={() => setMainTab("prompt")}
        >
          Prompt
        </button>
        <button
          type="button"
          role="tab"
          id="tab-visualization"
          aria-selected={mainTab === "visualization"}
          aria-controls="panel-visualization"
          tabIndex={mainTab === "visualization" ? 0 : -1}
          className="tab-btn"
          onClick={() => setMainTab("visualization")}
        >
          Visualization
        </button>
      </nav>

      {mainTab === "prompt" && (
        <div
          id="panel-prompt"
          role="tabpanel"
          aria-labelledby="tab-prompt"
          className="tab-panel"
        >
          <header className="app-header">
            <p className="app-kicker">DAHacks demo</p>
            <h1 className="app-title">Shared latent memory</h1>
            <p className="app-lede">
              Three agents run in sequence on the same vector memory and latent matrix{" "}
              <code>W</code>. Agent γ is tuned to hallucinate—watch clusters and anomalies.
            </p>
          </header>

          <div className="card card-glow">
            <div className="card-inner">
              <p className="section-title">Prompt</p>
              <form onSubmit={onSubmit}>
                <label className="form-label" htmlFor="p">
                  Grounded question (try France, speed of light, or boiling water)
                </label>
                <textarea
                  id="p"
                  className="textarea-field"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={3}
                  placeholder="Ask something answerable from the seeded corpus…"
                />
                <div style={{ marginTop: "1rem" }}>
                  <button
                    type="submit"
                    className="btn-primary"
                    disabled={loading || !prompt.trim()}
                  >
                    {loading && <span className="btn-spinner" aria-hidden />}
                    {loading ? "Running pipeline…" : "Run demo"}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {err && (
            <p className="alert-error" style={{ marginTop: "1.25rem" }} role="alert">
              {err}
            </p>
          )}
        </div>
      )}

      {mainTab === "visualization" && (
        <div
          id="panel-visualization"
          role="tabpanel"
          aria-labelledby="tab-visualization"
          className="tab-panel"
        >
          <header className="app-header" style={{ marginBottom: "1.25rem" }}>
            <p className="app-kicker">Results</p>
            <h1 className="app-title" style={{ fontSize: "clamp(1.25rem, 3vw, 1.65rem)" }}>
              Visualization
            </h1>
            {data ? (
              <p className="app-lede" style={{ fontSize: "0.92rem" }}>
                Prompt: <q>{data.prompt}</q>
              </p>
            ) : (
              <p className="app-lede">
                Run a prompt first—charts appear here after each demo.
              </p>
            )}
          </header>

          {!data && (
            <div className="card">
              <div className="card-inner empty-viz">
                <p>No run yet. Submit a question on the Prompt tab.</p>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setMainTab("prompt")}
                >
                  Go to Prompt
                </button>
              </div>
            </div>
          )}

          {data && (
            <div className="results-stack">
              <div className="card">
                <div className="card-inner">
                  <p className="section-title">Live visualization</p>
                  <h2 className="section-heading">3D latent shape & agent topology</h2>
                  <p className="section-desc">
                    Left: each point is a{" "}
                    <strong>{currentMorph?.embed_dim ?? 64}D</strong> memory vector after{" "}
                    <code>W·x</code>, shown in <strong>3 PCA dimensions</strong> (not
                    K-means). Frame advances automatically.{" "}
                    <code>
                      ‖W − I‖_F: {data.w_frobenius_delta_start.toFixed(3)} →{" "}
                      {data.w_frobenius_delta_end.toFixed(3)}
                    </code>
                  </p>
                  <div className="frame-pills" role="tablist" aria-label="Morph frames">
                    {morphFrames.map((_, i) => (
                      <button
                        key={i}
                        type="button"
                        className={`frame-pill ${i === frame ? "is-active" : ""}`}
                        onClick={() => setFrame(i)}
                      >
                        Step {i + 1}
                      </button>
                    ))}
                  </div>
                  <div className="grid-two">
                    <div>
                      <p className="section-title" style={{ marginBottom: "0.65rem" }}>
                        64D vectors → 3D PCA convex hull
                      </p>
                      <div className="viz-wrap">
                        <VectorSpace3D
                          frame={currentMorph}
                          frameLabel={`Morph · step ${frame + 1} / ${morphFrames.length}`}
                        />
                      </div>
                      <div className="legend">
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#8b9cb8" }}
                          />
                          corpus
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#5b9cfa" }}
                          />
                          α
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#3ecf8e" }}
                          />
                          β
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#e3b341" }}
                          />
                          γ
                        </span>
                      </div>
                    </div>
                    <div>
                      <p className="section-title" style={{ marginBottom: "0.65rem" }}>
                        Topology
                      </p>
                      <div className="viz-wrap">
                        <AgentTopology
                          activeStep={frame}
                          pulse={data.steps[frame]?.pulse}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-inner">
                  <p className="section-title">Clustering</p>
                  <h2 className="section-heading">Final memory — K-means & anomalies</h2>
                  <p className="section-desc">
                    Colors are cluster IDs; red ring = high z-score distance to centroid (see{" "}
                    <code>detect_anomalies</code>).
                  </p>
                  <div className="viz-wrap">
                    {finalClusters && (
                      <ClusterScatter
                        snap={finalClusters}
                        width={720}
                        height={340}
                        title="Final state · K-means in 2D PCA"
                      />
                    )}
                  </div>
                  <div className="legend">
                    <span className="legend-item">
                      <span
                        className="legend-dot"
                        style={{ border: "1px dashed rgba(139, 156, 184, 0.8)" }}
                      />
                      Dashed = centroid
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
