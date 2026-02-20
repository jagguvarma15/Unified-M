import { useEffect, useState } from "react";
import {
  Settings as SettingsIcon,
  Download,
  RefreshCw,
  Server,
  Cpu,
  HardDrive,
  CheckCircle2,
  XCircle,
  Blocks,
  Database,
  Cloud,
  Megaphone,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from "recharts";
import EmptyState from "../components/EmptyState";
import { api, type ParametersData, type HealthData, type AdaptersData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Settings() {
  const [params, setParams] = useState<ParametersData | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [adapters, setAdapters] = useState<AdaptersData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<"parameters" | "adstock" | "saturation" | "adapters" | "system">("parameters");

  useEffect(() => {
    Promise.allSettled([
      api.parameters().then(setParams),
      api.health().then(setHealth),
      api.adapters().then(setAdapters),
    ]).finally(() => setLoading(false));
  }, []);

  const handleRefreshCache = async () => {
    setRefreshing(true);
    try {
      await fetch("/api/v1/refresh", { method: "POST" });
      // Reload params
      const newParams = await api.parameters();
      setParams(newParams);
    } catch {
      // silently ignore
    } finally {
      setRefreshing(false);
    }
  };

  const handleExportJSON = () => {
    if (!params) return;
    const blob = new Blob([JSON.stringify(params, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "model_parameters.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  const tabs = [
    { key: "parameters" as const, label: "Coefficients" },
    { key: "adstock" as const, label: "Adstock" },
    { key: "saturation" as const, label: "Saturation" },
    { key: "adapters" as const, label: "Adapters" },
    { key: "system" as const, label: "System" },
  ];

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Settings & Parameters</h1>
          <p className="text-sm text-slate-500 mt-1">
            Model configuration, parameter inspection, and system info
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleRefreshCache}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
            Refresh Cache
          </button>
          <button
            onClick={handleExportJSON}
            disabled={!params}
            className="flex items-center gap-2 px-3 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50"
          >
            <Download size={14} />
            Export JSON
          </button>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 mt-6 bg-slate-100 rounded-lg p-1 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-6">
        {activeTab === "parameters" && <CoefficientsTab params={params} />}
        {activeTab === "adstock" && <AdstockTab params={params} />}
        {activeTab === "saturation" && <SaturationTab params={params} />}
        {activeTab === "adapters" && <AdaptersTab adapters={adapters} />}
        {activeTab === "system" && <SystemTab health={health} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab components
// ---------------------------------------------------------------------------

function CoefficientsTab({ params }: { params: ParametersData | null }) {
  if (!params?.coefficients || Object.keys(params.coefficients).length === 0) {
    return <EmptyState title="No coefficients" message="Run a model to see parameter estimates." />;
  }

  const coefs = Object.entries(params.coefficients)
    .map(([channel, value]) => ({ channel, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Channel Coefficients</h2>
        <ResponsiveContainer width="100%" height={Math.max(200, coefs.length * 48)}>
          <BarChart data={coefs} layout="vertical" margin={{ left: 80, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis type="category" dataKey="channel" tick={{ fontSize: 13 }} width={75} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} name="Coefficient">
              {coefs.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {params.intercept !== undefined && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-2">Intercept (Baseline)</h2>
          <p className="text-2xl font-bold text-slate-900 tabular-nums">
            {params.intercept.toLocaleString(undefined, { maximumFractionDigits: 4 })}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Base response level independent of media spend
          </p>
        </div>
      )}

      {/* Coefficient table */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Parameter Values</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Coefficient</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">|Coefficient|</th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Relative Strength</th>
              </tr>
            </thead>
            <tbody>
              {coefs.map((c, i) => {
                const maxAbs = Math.max(...coefs.map((x) => Math.abs(x.value)));
                const pct = maxAbs > 0 ? (Math.abs(c.value) / maxAbs) * 100 : 0;
                return (
                  <tr key={c.channel} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td className="py-3 px-4 flex items-center gap-2 font-medium">
                      <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      {c.channel}
                    </td>
                    <td className={`text-right py-3 px-4 tabular-nums font-mono ${
                      c.value >= 0 ? "text-emerald-600" : "text-red-500"
                    }`}>
                      {c.value.toFixed(6)}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums font-mono">
                      {Math.abs(c.value).toFixed(6)}
                    </td>
                    <td className="py-3 px-4">
                      <div className="w-32 bg-slate-100 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-indigo-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function AdstockTab({ params }: { params: ParametersData | null }) {
  if (!params?.adstock || Object.keys(params.adstock).length === 0) {
    return <EmptyState title="No adstock parameters" message="Run a model with adstock transforms to see decay parameters." />;
  }

  const adstock = Object.entries(params.adstock).map(([channel, p]) => {
    const q = p as { decay?: number; max_lag?: number };
    const decay = q.decay ?? 0;
    const max_lag = q.max_lag ?? 0;
    return {
      channel,
      decay,
      max_lag,
      halfLife: decay > 0 ? Math.log(0.5) / Math.log(decay) : 0,
    };
  });

  // Build decay curves for visualization
  const maxLag = Math.max(0, ...adstock.map((a) => a.max_lag));
  const decayCurves = Array.from({ length: maxLag + 1 }, (_, t) => {
    const row: Record<string, number | string> = { lag: t };
    for (const a of adstock) {
      row[a.channel] = Math.pow(a.decay, t);
    }
    return row;
  });

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Adstock Decay Curves</h2>
        <p className="text-xs text-slate-500 mb-4">
          Shows how the effect of advertising decays over time for each channel
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={decayCurves}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="lag"
              tick={{ fontSize: 12 }}
              label={{ value: "Lag (periods)", position: "insideBottomRight", offset: -5, fontSize: 12 }}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              domain={[0, 1]}
              label={{ value: "Weight", angle: -90, position: "insideLeft", fontSize: 12 }}
            />
            <Tooltip />
            {/* Using recharts Line inside BarChart is not standard, use separate LineChart approach */}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Adstock Parameters</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Decay Rate</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Max Lag</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Half-Life</th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Decay Speed</th>
              </tr>
            </thead>
            <tbody>
              {adstock.map((a, i) => (
                <tr key={a.channel} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td className="py-3 px-4 flex items-center gap-2 font-medium">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {a.channel}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums font-mono">{a.decay.toFixed(3)}</td>
                  <td className="text-right py-3 px-4 tabular-nums">{a.max_lag}</td>
                  <td className="text-right py-3 px-4 tabular-nums">{a.halfLife.toFixed(1)} periods</td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-slate-100 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            a.decay > 0.7 ? "bg-amber-500" : a.decay > 0.4 ? "bg-emerald-500" : "bg-indigo-500"
                          }`}
                          style={{ width: `${a.decay * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-500">
                        {a.decay > 0.7 ? "Slow" : a.decay > 0.4 ? "Medium" : "Fast"}
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SaturationTab({ params }: { params: ParametersData | null }) {
  if (!params?.saturation || Object.keys(params.saturation).length === 0) {
    return <EmptyState title="No saturation parameters" message="Run a model with saturation transforms to see Hill curve parameters." />;
  }

  const saturation = Object.entries(params.saturation).map(([channel, p]) => {
    const q = p as { K?: number; S?: number };
    return { channel, K: q.K ?? 0, S: q.S ?? 0 };
  });

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Saturation (Hill) Parameters</h2>
        <p className="text-xs text-slate-500 mb-4">
          K = half-saturation point (spend at 50% of max effect), S = shape (steepness of curve)
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">K (Half-Sat)</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">S (Shape)</th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Saturation Speed</th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {saturation.map((s, i) => (
                <tr key={s.channel} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td className="py-3 px-4 flex items-center gap-2 font-medium">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {s.channel}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums font-mono">
                    {s.K.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums font-mono">{s.S.toFixed(3)}</td>
                  <td className="py-3 px-4">
                    <div className="w-24 bg-slate-100 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          s.S > 1.5 ? "bg-red-500" : s.S > 0.8 ? "bg-amber-500" : "bg-emerald-500"
                        }`}
                        style={{ width: `${Math.min(100, s.S * 50)}%` }}
                      />
                    </div>
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-600">
                    {s.S > 1.5
                      ? "Quickly saturates — strong diminishing returns"
                      : s.S > 0.8
                        ? "Moderate saturation — standard curve"
                        : "Slow saturation — linear-like response"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function AdaptersTab({ adapters }: { adapters: AdaptersData | null }) {
  if (!adapters) {
    return <EmptyState title="Loading adapters..." message="Could not fetch adapter information from the API." />;
  }

  const cacheInfo = adapters.cache;

  return (
    <div className="space-y-6">
      {/* Model Backends */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Blocks size={16} />
          Model Backends
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {adapters.model_backends.map((b) => (
            <div
              key={b.name}
              className={`flex items-center justify-between rounded-lg border p-4 ${
                b.available ? "border-emerald-200 bg-emerald-50/50" : "border-slate-200 bg-slate-50"
              }`}
            >
              <div className="flex items-center gap-3">
                {b.available ? (
                  <CheckCircle2 size={18} className="text-emerald-500" />
                ) : (
                  <XCircle size={18} className="text-slate-400" />
                )}
                <div>
                  <p className="text-sm font-semibold text-slate-900">{b.name}</p>
                  <p className="text-xs text-slate-500">
                    {b.available ? "Installed & ready" : b.install_hint || "Not installed"}
                  </p>
                </div>
              </div>
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  b.available ? "bg-emerald-100 text-emerald-700" : "bg-slate-200 text-slate-600"
                }`}
              >
                {b.available ? "Active" : "Unavailable"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Connectors */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">Supported Connectors</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Database size={14} /> Databases
            </div>
            <ul className="space-y-1">
              {adapters.connectors.database.map((d) => (
                <li key={d} className="text-sm text-slate-700 flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Cloud size={14} /> Cloud Storage
            </div>
            <ul className="space-y-1">
              {adapters.connectors.cloud.map((c) => (
                <li key={c} className="text-sm text-slate-700 flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {c}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Megaphone size={14} /> Ad Platforms
            </div>
            <ul className="space-y-1">
              {adapters.connectors.ad_platforms.map((a) => (
                <li key={a} className="text-sm text-slate-700 flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {a}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Cache */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Cpu size={16} />
          Cache Backend
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Object.entries(cacheInfo).map(([key, val]) => (
            <div key={key} className="p-3 bg-slate-50 rounded-lg">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-0.5">{key.replace(/_/g, " ")}</p>
              <p className="text-sm font-semibold text-slate-900 tabular-nums">{String(val)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SystemTab({ health }: { health: HealthData | null }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Server size={16} />
          API Server Status
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 bg-slate-50 rounded-lg">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Status</p>
            <p className="text-sm font-semibold flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${health ? "bg-emerald-500" : "bg-red-500"}`} />
              {health?.status ?? "Offline"}
            </p>
          </div>
          <div className="p-4 bg-slate-50 rounded-lg">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Version</p>
            <p className="text-sm font-semibold">{health?.version ?? "—"}</p>
          </div>
          <div className="p-4 bg-slate-50 rounded-lg">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Timestamp</p>
            <p className="text-sm font-semibold font-mono">{health?.timestamp ?? "—"}</p>
          </div>
          <div className="p-4 bg-slate-50 rounded-lg">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Latest Run</p>
            <p className="text-sm font-semibold font-mono truncate">{health?.latest_run ?? "None"}</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Cpu size={16} />
          API Endpoints
        </h2>
        <div className="space-y-2 text-sm">
          {[
            { method: "GET", path: "/health", desc: "Health check" },
            { method: "GET", path: "/api/v1/runs", desc: "List pipeline runs" },
            { method: "GET", path: "/api/v1/contributions", desc: "Channel contributions" },
            { method: "GET", path: "/api/v1/reconciliation", desc: "Reconciled estimates" },
            { method: "GET", path: "/api/v1/optimization", desc: "Budget optimization" },
            { method: "GET", path: "/api/v1/response-curves", desc: "Response curves" },
            { method: "GET", path: "/api/v1/parameters", desc: "Model parameters" },
            { method: "GET", path: "/api/v1/diagnostics", desc: "Model diagnostics" },
            { method: "GET", path: "/api/v1/roas", desc: "ROAS analysis" },
            { method: "GET", path: "/api/v1/waterfall", desc: "Waterfall decomposition" },
            { method: "GET", path: "/api/v1/data/status", desc: "Data source status" },
            { method: "POST", path: "/api/v1/data/upload", desc: "Upload data file" },
            { method: "POST", path: "/api/v1/pipeline/run", desc: "Trigger pipeline" },
            { method: "POST", path: "/api/v1/refresh", desc: "Refresh cache" },
          ].map((ep) => (
            <div key={ep.path} className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-slate-50">
              <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                ep.method === "GET" ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
              }`}>
                {ep.method}
              </span>
              <code className="text-xs font-mono text-slate-700 flex-1">{ep.path}</code>
              <span className="text-xs text-slate-500">{ep.desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <HardDrive size={16} />
          Quick Actions
        </h2>
        <div className="flex flex-wrap gap-3">
          <a
            href="/docs"
            target="_blank"
            className="px-4 py-2 bg-slate-100 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
          >
            OpenAPI Docs
          </a>
          <a
            href="/redoc"
            target="_blank"
            className="px-4 py-2 bg-slate-100 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
          >
            ReDoc
          </a>
        </div>
      </div>
    </div>
  );
}
