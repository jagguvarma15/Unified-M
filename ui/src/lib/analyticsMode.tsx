import { createContext, createSignal, useContext, type JSX } from "solid-js";

interface AnalyticsModeContextValue {
  analyticsEnabled: () => boolean;
  setAnalyticsEnabled: (enabled: boolean) => void;
}

const AnalyticsModeContext = createContext<AnalyticsModeContextValue | null>(
  null,
);

export function AnalyticsModeProvider(props: { children: JSX.Element }) {
  const [analyticsEnabled, setAnalyticsEnabled] = createSignal(false);
  return (
    <AnalyticsModeContext.Provider
      value={{ analyticsEnabled, setAnalyticsEnabled }}
    >
      {props.children}
    </AnalyticsModeContext.Provider>
  );
}

export function useAnalyticsMode() {
  const ctx = useContext(AnalyticsModeContext);
  if (!ctx)
    throw new Error(
      "useAnalyticsMode must be used within AnalyticsModeProvider",
    );
  return ctx;
}
