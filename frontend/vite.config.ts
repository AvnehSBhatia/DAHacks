import path from "node:path";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** Load repo-root .env so Auth0 vars match the FastAPI app (no duplicate frontend/.env.local). */
export default defineConfig(({ mode }) => {
  const root = path.resolve(__dirname, "..");
  const e = loadEnv(mode, root, "");

  const domain = e.VITE_AUTH0_DOMAIN || e.AUTH0_DOMAIN || "";
  const audience = e.VITE_AUTH0_AUDIENCE || e.AUTH0_AUDIENCE || "";
  const clientId = e.VITE_AUTH0_CLIENT_ID || e.AUTH0_CLIENT_ID || "";
  // Default 8000 matches common `uvicorn ... --port 8000`; override with API_PORT in repo .env
  const apiPort = e.API_PORT || e.VITE_API_PORT || "8000";
  const apiOrigin = `http://127.0.0.1:${apiPort}`;

  return {
    plugins: [react(), basicSsl()],
    envDir: root,
    define: {
      "import.meta.env.VITE_AUTH0_DOMAIN": JSON.stringify(domain),
      "import.meta.env.VITE_AUTH0_CLIENT_ID": JSON.stringify(clientId),
      "import.meta.env.VITE_AUTH0_AUDIENCE": JSON.stringify(audience),
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": apiOrigin,
        "/health": apiOrigin,
      },
    },
  };
});
