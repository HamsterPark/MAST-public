import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// Dev: proxy /api → the standalone FastAPI (python -m mast.api on :7870) so the
// SPA runs same-origin in dev and prod alike (no CORS surprises). Override the
// target with MAST_API_TARGET if the API runs elsewhere.
const apiTarget = process.env.MAST_API_TARGET ?? "http://127.0.0.1:7870";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
      // WebSocket/SSE channels (added Phase 3)
      "/ws": { target: apiTarget, ws: true, changeOrigin: true },
      "/sse": { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    // Built SPA is bundled into the PyInstaller _internal/ and served by the
    // FastAPI app in production (Phase 5).
    outDir: "dist",
    sourcemap: true,
  },
});
