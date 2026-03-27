import { useCallback, useState } from "react";
import { ClusterScatter } from "./ClusterScatter";
import { runDemoStream, type DemoResult } from "./api";
import { LatentFieldViz, type InspectInfo } from "./components/LatentFieldViz";
import { VectorDetailPanel } from "./components/VectorDetailPanel";
import { useAuth0 } from "@auth0/auth0-react";
import { DemoExperience } from "./DemoExperience";
import { runDemo } from "./api";
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
    <div className="app">
      <DemoExperience runDemoFn={runDemoFn} />
    </div>
  );
}

function AppWithAuth() {
  const {
    isLoading,
    isAuthenticated,
    loginWithRedirect,
    logout,
    user,
    getAccessTokenSilently,
  } = useAuth0();

  const audience = import.meta.env.VITE_AUTH0_AUDIENCE as string | undefined;

  const getToken = useCallback(
    () =>
      getAccessTokenSilently({
        authorizationParams: {
          audience: audience ?? undefined,
        },
      }),
    [audience, getAccessTokenSilently],
  );

  const runDemoFn = useCallback(
    (prompt: string) => runDemo(prompt, getToken),
    [getToken],
  );

  if (isLoading) {
    return (
      <div className="app app-auth-loading">
        <p className="app-lede" style={{ textAlign: "center", marginTop: "2rem" }}>
          Signing in…
        </p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="app">
        <header className="app-header">
          <p className="app-kicker">DAHacks demo</p>
          <h1 className="app-title">Shared latent memory</h1>
          <p className="app-lede">
            Sign in with Auth0 to run the agent pipeline. Your identity is sent as a secure
            token with each demo request.
          </p>
        </header>
        <div className="card card-glow">
          <div className="card-inner" style={{ textAlign: "center" }}>
            <button
              type="button"
              className="btn-primary"
              onClick={() => loginWithRedirect()}
            >
              Log in
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app app-with-auth">
      <div className="auth-bar" role="region" aria-label="Account">
        <span className="auth-bar-user" title={user?.sub}>
          {user?.email ?? user?.name ?? user?.sub}
        </span>
        <button
          type="button"
          className="btn-secondary auth-bar-logout"
          onClick={() =>
            logout({ logoutParams: { returnTo: window.location.origin } })
          }
        >
          Log out
        </button>
      </div>
      <DemoExperience runDemoFn={runDemoFn} />
    </div>
  );
}

export default function App() {
  const domain = import.meta.env.VITE_AUTH0_DOMAIN;
  const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID;

  if (domain && !clientId) {
    return (
      <div className="app">
        <div className="card card-glow" style={{ marginTop: "2rem" }}>
          <div className="card-inner">
            <p className="section-title">Auth0 configuration</p>
            <p className="section-desc" style={{ marginBottom: "0.75rem" }}>
              <code>AUTH0_DOMAIN</code> is set in the repo root <code>.env</code>, but{" "}
              <code>AUTH0_CLIENT_ID</code> is missing. Add your SPA application&apos;s Client ID
              from Auth0 (Applications → your app → Client ID), restart{" "}
              <code>npm run dev</code>, then log in.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (domain && clientId) {
    return <AppWithAuth />;
  }
  return <AppWithoutAuth />;
}
