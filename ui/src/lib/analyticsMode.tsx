import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

interface AnalyticsModeContextValue {
  analyticsEnabled: boolean;
  setAnalyticsEnabled: (enabled: boolean) => void;
}

const AnalyticsModeContext = createContext<AnalyticsModeContextValue | null>(null);

export function AnalyticsModeProvider({ children }: { children: ReactNode }) {
  const [analyticsEnabled, setAnalyticsEnabledState] = useState<boolean>(false);

  const setAnalyticsEnabled = (enabled: boolean) => {
    setAnalyticsEnabledState(enabled);
  };

  const value = useMemo(
    () => ({ analyticsEnabled, setAnalyticsEnabled }),
    [analyticsEnabled],
  );

  return <AnalyticsModeContext.Provider value={value}>{children}</AnalyticsModeContext.Provider>;
}

export function useAnalyticsMode() {
  const ctx = useContext(AnalyticsModeContext);
  if (!ctx) {
    throw new Error("useAnalyticsMode must be used within AnalyticsModeProvider");
  }
  return ctx;
}
