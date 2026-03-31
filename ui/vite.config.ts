import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import path from "path";

export default defineConfig({
  plugins: [solid()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 400,
    rollupOptions: {
      output: {
        manualChunks: {
          solid: ["solid-js", "@solidjs/router"],
          query: ["@tanstack/solid-query"],
          echarts: ["echarts"],
        },
      },
    },
  },
});
