import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, Suspense, type ReactNode } from "react";
import Layout from "./components/Layout";
import { ToastProvider } from "./lib/toast";
import ToastContainer from "./components/Toast";
import EmptyState from "./components/EmptyState";
import { useAnalyticsMode } from "./lib/analyticsMode";

import Dashboard from "./pages/Dashboard";
import Calibration from "./pages/Calibration";
import Stability from "./pages/Stability";
import DataQuality from "./pages/DataQuality";

const Data = lazy(() => import("./pages/Data"));
const Contributions = lazy(() => import("./pages/Contributions"));
const Optimization = lazy(() => import("./pages/Optimization"));
const ResponseCurves = lazy(() => import("./pages/ResponseCurves"));
const Runs = lazy(() => import("./pages/Runs"));
const Diagnostics = lazy(() => import("./pages/Diagnostics"));
const ROASAnalysis = lazy(() => import("./pages/ROASAnalysis"));
const ScenarioPlanner = lazy(() => import("./pages/ScenarioPlanner"));
const Settings = lazy(() => import("./pages/Settings"));
const Datapoint = lazy(() => import("./pages/Datapoint"));
const ChannelInsights = lazy(() => import("./pages/ChannelInsights"));
const SpendPacing = lazy(() => import("./pages/SpendPacing"));
const Report = lazy(() => import("./pages/Report"));

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
    </div>
  );
}

function AnalyticsGate({ children }: { children: ReactNode }) {
  const { analyticsEnabled } = useAnalyticsMode();
  if (analyticsEnabled) return children;
  return (
    <EmptyState
      title="Analytics Hidden"
      message="Enable the sample-data toggle in Run Pipeline to view analytics pages."
      action={{ label: "Go to Data", href: "/data" }}
    />
  );
}

export default function App() {
  const { analyticsEnabled } = useAnalyticsMode();
  return (
    <ToastProvider>
      <BrowserRouter>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={analyticsEnabled ? <Dashboard /> : <Data />} />
              <Route path="/data" element={<Data />} />
              <Route path="/contributions" element={<AnalyticsGate><Contributions /></AnalyticsGate>} />
              <Route path="/optimization" element={<AnalyticsGate><Optimization /></AnalyticsGate>} />
              <Route path="/curves" element={<AnalyticsGate><ResponseCurves /></AnalyticsGate>} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/diagnostics" element={<AnalyticsGate><Diagnostics /></AnalyticsGate>} />
              <Route path="/roas" element={<AnalyticsGate><ROASAnalysis /></AnalyticsGate>} />
              <Route path="/scenarios" element={<AnalyticsGate><ScenarioPlanner /></AnalyticsGate>} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/datapoint" element={<Datapoint />} />
              <Route path="/calibration" element={<AnalyticsGate><Calibration /></AnalyticsGate>} />
              <Route path="/stability" element={<AnalyticsGate><Stability /></AnalyticsGate>} />
              <Route path="/data-quality" element={<AnalyticsGate><DataQuality /></AnalyticsGate>} />
              <Route path="/channel-insights" element={<AnalyticsGate><ChannelInsights /></AnalyticsGate>} />
              <Route path="/spend-pacing" element={<AnalyticsGate><SpendPacing /></AnalyticsGate>} />
              <Route path="/report" element={<AnalyticsGate><Report /></AnalyticsGate>} />
            </Route>
          </Routes>
        </Suspense>
        <ToastContainer />
      </BrowserRouter>
    </ToastProvider>
  );
}
