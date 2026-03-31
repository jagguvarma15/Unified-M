import { Router, Route } from "@solidjs/router";
import { lazy, Suspense, Show, type JSX } from "solid-js";
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
    <div class="flex items-center justify-center h-64">
      <div
        class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"
        role="status"
        aria-label="Loading"
      />
    </div>
  );
}

function AnalyticsGate(props: { children: JSX.Element }) {
  const { analyticsEnabled } = useAnalyticsMode();
  return (
    <Show
      when={analyticsEnabled()}
      fallback={
        <EmptyState
          title="Analytics Hidden"
          message="Enable the sample-data toggle in Run Pipeline to view analytics pages."
          action={{ label: "Go to Data", href: "/data" }}
        />
      }
    >
      {props.children}
    </Show>
  );
}

function AppRoutes() {
  const { analyticsEnabled } = useAnalyticsMode();
  return (
    <Suspense fallback={<PageLoader />}>
      <Route path="/" component={Layout}>
        <Route
          path="/"
          component={() => (
            <Show when={analyticsEnabled()} fallback={<Data />}>
              <Dashboard />
            </Show>
          )}
        />
        <Route path="/data" component={Data} />
        <Route path="/contributions" component={() => <AnalyticsGate><Contributions /></AnalyticsGate>} />
        <Route path="/optimization" component={() => <AnalyticsGate><Optimization /></AnalyticsGate>} />
        <Route path="/curves" component={() => <AnalyticsGate><ResponseCurves /></AnalyticsGate>} />
        <Route path="/runs" component={Runs} />
        <Route path="/diagnostics" component={() => <AnalyticsGate><Diagnostics /></AnalyticsGate>} />
        <Route path="/roas" component={() => <AnalyticsGate><ROASAnalysis /></AnalyticsGate>} />
        <Route path="/scenarios" component={() => <AnalyticsGate><ScenarioPlanner /></AnalyticsGate>} />
        <Route path="/settings" component={Settings} />
        <Route path="/datapoint" component={Datapoint} />
        <Route path="/calibration" component={() => <AnalyticsGate><Calibration /></AnalyticsGate>} />
        <Route path="/stability" component={() => <AnalyticsGate><Stability /></AnalyticsGate>} />
        <Route path="/data-quality" component={() => <AnalyticsGate><DataQuality /></AnalyticsGate>} />
        <Route path="/channel-insights" component={() => <AnalyticsGate><ChannelInsights /></AnalyticsGate>} />
        <Route path="/spend-pacing" component={() => <AnalyticsGate><SpendPacing /></AnalyticsGate>} />
        <Route path="/report" component={() => <AnalyticsGate><Report /></AnalyticsGate>} />
      </Route>
    </Suspense>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Router>
        <AppRoutes />
      </Router>
      <ToastContainer />
    </ToastProvider>
  );
}
