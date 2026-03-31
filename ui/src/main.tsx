import { render } from "solid-js/web";
import { QueryClientProvider } from "@tanstack/solid-query";
import App from "./App";
import "./index.css";
import { queryClient } from "./lib/queryClient";
import { AnalyticsModeProvider } from "./lib/analyticsMode";

render(
  () => (
    <QueryClientProvider client={queryClient}>
      <AnalyticsModeProvider>
        <App />
      </AnalyticsModeProvider>
    </QueryClientProvider>
  ),
  document.getElementById("root")!,
);
