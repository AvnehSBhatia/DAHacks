import { Auth0Provider } from "@auth0/auth0-react";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const auth0Domain = import.meta.env.VITE_AUTH0_DOMAIN;
const auth0ClientId = import.meta.env.VITE_AUTH0_CLIENT_ID;
const auth0Audience = import.meta.env.VITE_AUTH0_AUDIENCE;

const root = (
  <React.StrictMode>
    {auth0Domain && auth0ClientId ? (
      <Auth0Provider
        domain={auth0Domain}
        clientId={auth0ClientId}
        authorizationParams={{
          redirect_uri: window.location.origin,
          ...(auth0Audience ? { audience: auth0Audience } : {}),
        }}
        cacheLocation="localstorage"
        useRefreshTokens={true}
      >
        <App />
      </Auth0Provider>
    ) : (
      <App />
    )}
  </React.StrictMode>
);

ReactDOM.createRoot(document.getElementById("root")!).render(root);
