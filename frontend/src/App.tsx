import { useAuth0 } from "@auth0/auth0-react";
import { useCallback, useState } from "react";
import { DemoExperience, MainTabNav, type MainTab } from "./DemoExperience";
import { runDemo } from "./api";
import "./App.css";

function AppWithoutAuth() {
  const runDemoFn = useCallback((prompt: string) => runDemo(prompt), []);
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

  const [mainTab, setMainTab] = useState<MainTab>("prompt");

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
      <div className="app app-auth-landing">
        <header className="app-header">
          <p className="app-kicker">DAHacks demo</p>
          <h1 className="app-title">Shared latent memory</h1>
          <p className="app-lede">
            Sign in with Auth0 to run the agent pipeline. Your identity is sent as a secure
            token with each demo request.
          </p>
        </header>
        <div className="auth-login-actions">
          <button
            type="button"
            className="btn-primary"
            onClick={() => loginWithRedirect()}
          >
            Log in
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app app-with-auth">
      <div className="app-top-bar">
        <MainTabNav mainTab={mainTab} onChange={setMainTab} />
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
      </div>
      <DemoExperience
        runDemoFn={runDemoFn}
        liftedTabs={{ mainTab, setMainTab }}
      />
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
