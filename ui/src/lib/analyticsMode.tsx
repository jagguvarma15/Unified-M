import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

const STORAGE_KEY = "unified_m_analytics_enabled";

interface AnalyticsModeContextValue {
  analyticsEnabled: boolean;
  setAnalyticsEnabled: (enabled: boolean) => void;
}

const AnalyticsModeContext = createContext<AnalyticsModeContextValue | null>(null);

export function AnalyticsModeProvider({ children }: { children: ReactNode }) {
  const [analyticsEnabled, setAnalyticsEnabledState] = useState<boolean>(true);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "false") setAnalyticsEnabledState(false);
  }, []);

  const setAnalyticsEnabled = (enabled: boolean) => {
    setAnalyticsEnabledState(enabled);
    localStorage.setItem(STORAGE_KEY, enabled ? "true" : "false");
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

