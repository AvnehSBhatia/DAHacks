import { useCallback, useEffect, useState } from "react";
import { AgentTopology } from "./AgentTopology";
import type { DemoResult } from "./api";
import { PipelineStoryboard } from "./PipelineStoryboard";
import { VectorSpace3D } from "./VectorSpace3D";

/** 3D hull and topology share this canvas height so the row aligns. */
const GRID_PAIR_PLOT_HEIGHT = 320;

export type MainTab = "prompt" | "visualization";

type Props = {
  runDemoFn: (prompt: string) => Promise<DemoResult>;
  /** Parent renders tabs (e.g. auth top bar); state must stay in sync. */
  liftedTabs?: { mainTab: MainTab; setMainTab: (t: MainTab) => void };
};

export function MainTabNav({
  mainTab,
  onChange,
}: {
  mainTab: MainTab;
  onChange: (t: MainTab) => void;
}) {
  return (
    <nav className="tab-bar" role="tablist" aria-label="Primary">
      <button
        type="button"
        role="tab"
        id="tab-prompt"
        aria-selected={mainTab === "prompt"}
        aria-controls="panel-prompt"
        tabIndex={mainTab === "prompt" ? 0 : -1}
        className="tab-btn"
        onClick={() => onChange("prompt")}
      >
        Prompt / Input
      </button>
      <button
        type="button"
        role="tab"
        id="tab-visualization"
        aria-selected={mainTab === "visualization"}
        aria-controls="panel-visualization"
        tabIndex={mainTab === "visualization" ? 0 : -1}
        className="tab-btn"
        onClick={() => onChange("visualization")}
      >
        Visual Analytics
      </button>
    </nav>
  );
}

export function DemoExperience({ runDemoFn, liftedTabs }: Props) {
  const [internalTab, setInternalTab] = useState<MainTab>("prompt");
  const mainTab = liftedTabs ? liftedTabs.mainTab : internalTab;
  const setMainTab = (t: MainTab) => {
    if (liftedTabs) liftedTabs.setMainTab(t);
    else setInternalTab(t);
  };
  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [loading, setLoading] = useState(false);
  const [initSequence, setInitSequence] = useState<number>(0);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<DemoResult | null>(null);
  const [frame, setFrame] = useState(0);

  const morphFrames = data?.morph_frames ?? [];

  useEffect(() => {
    if (!morphFrames.length) return;
    const last = morphFrames.length - 1;
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f >= last) {
          window.clearInterval(id);
          return last;
        }
        return f + 1;
      });
    }, 2200);
    return () => clearInterval(id);
  }, [morphFrames]);

  // Handle the fake initialization sequence
  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => {
      setInitSequence(prev => Math.min(prev + 1, 100));
    }, 40);
    return () => clearInterval(timer);
  }, [loading]);

  const currentMorph = morphFrames[frame] ?? null;

  const stepCount = data?.steps.length ?? 0;
  /** 3 agents × 3 fixed cycles → 9 steps: rotate highlight α/β/γ via frame % 3. */
  const threeAgentTripleCycle = stepCount === 9;
  const topologySynced =
    (stepCount > 0 && stepCount <= 3) || threeAgentTripleCycle;
  const activeTopologyStep = threeAgentTripleCycle
    ? frame % 3
    : stepCount > 0 && stepCount <= 3
      ? Math.min(frame, stepCount - 1)
      : -1;
  const stepIdx = Math.min(frame, Math.max(0, stepCount - 1));
  const corruptionRevealed =
    !threeAgentTripleCycle || frame >= stepCount - 1;

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setErr(null);
      setLoading(true);
      setInitSequence(0);
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
      {!liftedTabs && <MainTabNav mainTab={mainTab} onChange={setMainTab} />}

      {mainTab === "prompt" && (
        <div
          id="panel-prompt"
          role="tabpanel"
          aria-labelledby="tab-prompt"
          className="tab-panel"
        >
          <header className="app-header">
            <p className="app-kicker">System OS v0.9</p>
            <h1 className="app-title">Latent Memory Simulation</h1>
            <p className="app-lede">
              Initialize a vector generation sequence. The engine orchestrates autonomous agents
              to map semantic tensors against a baseline metric axis.
            </p>
          </header>

          <div className="card">
            <div className="card-inner">
              <p className="section-title">Calibration Target</p>
              <form onSubmit={onSubmit}>
                <label className="form-label" htmlFor="p">
                  Topic Prompt
                </label>
                <textarea
                  id="p"
                  className="textarea-field"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={3}
                  placeholder="Enter input parameters..."
                />
                <div style={{ marginTop: "1.2rem" }}>
                  <button
                    type="submit"
                    className="btn-primary"
                    disabled={loading || !prompt.trim()}
                  >
                    {loading ? "Initializing..." : "Execute Simulation Sequence"}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {err && (
            <p className="alert-error" style={{ marginTop: "1.25rem" }} role="alert">
              [CRITICAL ERROR] {err}
            </p>
          )}

          {loading && (
             <div className="card" style={{ marginTop: "1.25rem", border: "1px solid var(--accent)", boxShadow: "0 0 20px rgba(0, 255, 204, 0.1)" }}>
               <div className="card-inner">
                  <p className="section-title">System Boot Process</p>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent)" }}>
                    <div>{`[+] Allocating tensor buffers... ${initSequence}%`}</div>
                    <div style={{ opacity: initSequence > 15 ? 1 : 0 }}>{`[+] Loading Encoder2x384To64 weights...`}</div>
                    <div style={{ opacity: initSequence > 35 ? 1 : 0 }}>{`[+] Normalizing Base Vector = normalize(avg(Embed(P)))`}</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: "2px", margin: "10px 0", color: "#666", fontSize: "0.6rem", opacity: initSequence > 50 ? 1 : 0 }}>
                       {Array(64).fill(0).map((_, i) => <span key={i}>{(Math.random() * 2 - 1).toFixed(4)}</span>)}
                    </div>
                    <div style={{ opacity: initSequence > 70 ? 1 : 0 }}>{`[+] Waking 3 Agents for iterative thought pipeline...`}</div>
                    <div style={{ opacity: initSequence > 90 ? 1 : 0 }}>{`[+] Connection established. Awaiting LLM response chunk.`}</div>
                    
                    <div style={{ marginTop: "1rem", height: "4px", background: "var(--line)", width: "100%", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${initSequence}%`, background: "var(--accent)", transition: "width 0.1s linear" }} />
                    </div>
                  </div>
               </div>
             </div>
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
            <p className="app-kicker">Diagnostics Console</p>
            <h1 className="app-title" style={{ fontSize: "clamp(1.25rem, 3vw, 1.65rem)" }}>
              Data Topology Map
            </h1>
            {data ? (
              <p className="app-lede" style={{ fontSize: "0.92rem", color: "var(--accent)" }}>
                &gt; Sequence run: <q>{data.prompt}</q>
              </p>
            ) : (
              <p className="app-lede">
                System awaiting input vector on Prompt screen.
              </p>
            )}
          </header>

          {!data && (
            <div className="card">
              <div className="card-inner empty-viz">
                <p>Telemetry offline. Provide initial parameters.</p>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setMainTab("prompt")}
                >
                  Return to Input
                </button>
              </div>
            </div>
          )}

          {data && (
            <div className="results-stack">
              <div className="card">
                <div className="card-inner">
                  <p className="section-title">Telemetry Stream</p>
                  <h2 className="section-heading">Multi-Agent State Trajectory</h2>
                  <p className="section-desc">
                    Matrix decomposition over time. The 64D manifold is projected to 3 PCA dimensions.
                  </p>
                  <div className="frame-pills" role="tablist" aria-label="Morph frames">
                    {morphFrames.map((_, i) => (
                      <button
                        key={i}
                        type="button"
                        className={`frame-pill ${i === frame ? "is-active" : ""}`}
                        style={i === frame ? { background: "var(--accent)", color: "#000", border: 'none' } : {}}
                        onClick={() => setFrame(i)}
                      >
                        CYC_{i + 1}
                      </button>
                    ))}
                  </div>
                  <div className="grid-two">
                    <div>
                      <p className="section-title" style={{ marginBottom: "0.65rem" }}>
                        SVD Hull Visualizer
                      </p>
                      <div className="viz-wrap viz-wrap--grid-pair">
                        <VectorSpace3D
                          frame={currentMorph}
                          frameLabel={`T=${frame + 1} // ${morphFrames.length}`}
                          plotHeight={GRID_PAIR_PLOT_HEIGHT}
                          highlightLastVectorRed={
                            threeAgentTripleCycle &&
                            frame === morphFrames.length - 1
                          }
                        />
                      </div>
                      <div className="legend">
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#8b9cb8" }}
                          />
                          Corpus
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#5b9cfa" }}
                          />
                          A-1
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#3ecf8e" }}
                          />
                          A-2
                        </span>
                        <span className="legend-item">
                          <span
                            className="legend-dot"
                            style={{ background: "#e3b341" }}
                          />
                          A-3
                        </span>
                      </div>
                    </div>
                    <div>
                      <p className="section-title" style={{ marginBottom: "0.65rem" }}>
                        Node Density Flow
                      </p>
                      <div className="viz-wrap viz-wrap--grid-pair">
                        <AgentTopology
                          activeStep={
                            topologySynced ? activeTopologyStep : -1
                          }
                          pulse={
                            topologySynced
                              ? data.steps[stepIdx]?.pulse
                              : undefined
                          }
                          corruptionRevealed={corruptionRevealed}
                          plotHeight={GRID_PAIR_PLOT_HEIGHT}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-inner">
                  <PipelineStoryboard
                    data={data}
                    frame={frame}
                    morphFrame={currentMorph}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
