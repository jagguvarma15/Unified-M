import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, Suspense } from "react";
import Layout from "./components/Layout";

// Eagerly load the dashboard (landing page)
import Dashboard from "./pages/Dashboard";

// Lazy-load secondary pages for smaller initial bundle
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
const Calibration = lazy(() => import("./pages/Calibration"));
const Stability = lazy(() => import("./pages/Stability"));
const DataQuality = lazy(() => import("./pages/DataQuality"));

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="/data" element={<Data />} />
          <Route path="/contributions" element={<Contributions />} />
          <Route path="/optimization" element={<Optimization />} />
          <Route path="/curves" element={<ResponseCurves />} />
          <Route path="/runs" element={<Runs />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          <Route path="/roas" element={<ROASAnalysis />} />
            <Route path="/scenarios" element={<ScenarioPlanner />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/datapoint" element={<Datapoint />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/stability" element={<Stability />} />
            <Route path="/data-quality" element={<DataQuality />} />
        </Route>
      </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
