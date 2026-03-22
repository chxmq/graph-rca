import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true, // Listen on all network interfaces (0.0.0.0)
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:8010",
        changeOrigin: true,
      },
    },
  },
});

