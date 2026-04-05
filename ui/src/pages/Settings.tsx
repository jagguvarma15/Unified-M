import { createSignal, onMount, createEffect, Show, For } from "solid-js";
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
} from "../lib/icons";
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
import ReactChart, { h } from "../lib/ReactChart";
import EmptyState from "../components/EmptyState";
import { api, type ParametersData, type HealthData, type AdaptersData } from "../lib/api";
import { COLORS } from "../lib/colors";
import { useHealthQuery } from "../lib/queries";

export default function Settings() {
  const [params, setParams] = createSignal<ParametersData | null>(null);
  const [health, setHealth] = createSignal<HealthData | null>(null);
  const [adapters, setAdapters] = createSignal<AdaptersData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [refreshing, setRefreshing] = createSignal(false);
  const [activeTab, setActiveTab] = createSignal<"parameters" | "adstock" | "saturation" | "adapters" | "system">("parameters");
  const healthQuery = useHealthQuery();

  onMount(() => {
    Promise.allSettled([
      api.parameters().then(setParams),
      api.health().then(setHealth),
      api.adapters().then(setAdapters),
    ]).finally(() => setLoading(false));
  });

  createEffect(() => {
    if (healthQuery.isError) {
      setParams(null);
      setAdapters(null);
      setHealth(null);
    }
  });

  const handleRefreshCache = async () => {
    setRefreshing(true);
    try {
      await fetch("/api/v1/refresh", { method: "POST" });
      const newParams = await api.parameters();
      setParams(newParams);
    } catch {
      // silently ignore
    } finally {
      setRefreshing(false);
    }
  };

  const handleExportJSON = () => {
    if (!params()) return;
    const blob = new Blob([JSON.stringify(params(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "model_parameters.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const tabs = [
    { key: "parameters" as const, label: "Coefficients" },
    { key: "adstock" as const, label: "Adstock" },
    { key: "saturation" as const, label: "Saturation" },
    { key: "adapters" as const, label: "Adapters" },
    { key: "system" as const, label: "System" },
  ];

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <div>
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-2xl font-bold text-slate-900">Settings & Parameters</h1>
            <p class="text-sm text-slate-500 mt-1">
              Model configuration, parameter inspection, and system info
            </p>
          </div>
          <div class="flex gap-2">
            <button
              onClick={handleRefreshCache}
              disabled={refreshing()}
              class="flex items-center gap-2 px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors disabled:opacity-50"
            >
              <RefreshCw size={14} class={refreshing() ? "animate-spin" : ""} />
              Refresh Cache
            </button>
            <button
              onClick={handleExportJSON}
              disabled={!params()}
              class="flex items-center gap-2 px-3 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              <Download size={14} />
              Export JSON
            </button>
          </div>
        </div>

        {/* Tab navigation */}
        <div class="flex gap-1 mt-6 bg-slate-100 rounded-lg p-1 w-fit">
          <For each={tabs}>
            {(tab) => (
              <button
                onClick={() => setActiveTab(tab.key)}
                class={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab() === tab.key
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            )}
          </For>
        </div>

        {/* Tab content */}
        <div class="mt-6">
          <Show when={activeTab() === "parameters"}><CoefficientsTab params={params()} /></Show>
          <Show when={activeTab() === "adstock"}><AdstockTab params={params()} /></Show>
          <Show when={activeTab() === "saturation"}><SaturationTab params={params()} /></Show>
          <Show when={activeTab() === "adapters"}><AdaptersTab adapters={adapters()} /></Show>
          <Show when={activeTab() === "system"}><SystemTab health={health()} /></Show>
        </div>
      </div>
    </Show>
  );
}

// ---------------------------------------------------------------------------
// Tab components
// ---------------------------------------------------------------------------

function CoefficientsTab({ params }: { params: ParametersData | null }) {
  if (!params?.coefficients || Object.keys(params.coefficients).length === 0) {
    return <EmptyState title="No coefficients" message="Run a model to see parameter estimates." />;
  }

  const toFinite = (value: unknown): number => {
    const n = typeof value === "number" ? value : Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  const coefs = Object.entries(params.coefficients)
    .map(([channel, value]) => ({ channel, value: toFinite(value) }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const maxAbs = Math.max(0, ...coefs.map((x) => Math.abs(x.value)));

  return (
    <div class="space-y-6">
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Channel Coefficients</h2>
        <ReactChart>
          {() => h(ResponsiveContainer, { width: "100%", height: Math.max(200, coefs.length * 48) },
            h(BarChart, { data: coefs, layout: "vertical", margin: { left: 80, right: 20 } },
              h(CartesianGrid, { strokeDasharray: "3 3", stroke: "#e2e8f0", horizontal: false }),
              h(XAxis, { type: "number", tick: { fontSize: 12 } }),
              h(YAxis, { type: "category", dataKey: "channel", tick: { fontSize: 13 }, width: 75 }),
              h(Tooltip, { formatter: (v: unknown) => toFinite(v).toFixed(4) }),
              h(Bar, { dataKey: "value", radius: [0, 6, 6, 0], name: "Coefficient" },
                ...coefs.map((_, i) => h(Cell, { key: i, fill: COLORS[i % COLORS.length] }))
              )
            )
          )}
        </ReactChart>
      </div>

      {typeof params.intercept === "number" && Number.isFinite(params.intercept) && (
        <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 class="text-sm font-semibold text-slate-700 mb-2">Intercept (Baseline)</h2>
          <p class="text-2xl font-bold text-slate-900 tabular-nums">
            {params.intercept.toLocaleString(undefined, { maximumFractionDigits: 4 })}
          </p>
          <p class="text-xs text-slate-500 mt-1">
            Base response level independent of media spend
          </p>
        </div>
      )}

      {/* Coefficient table */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Parameter Values</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-slate-200">
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">Coefficient</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">|Coefficient|</th>
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Relative Strength</th>
              </tr>
            </thead>
            <tbody>
              {coefs.map((c, i) => {
                const pct = maxAbs > 0 ? (Math.abs(c.value) / maxAbs) * 100 : 0;
                return (
                  <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td class="py-3 px-4 flex items-center gap-2 font-medium">
                      <span class="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      {c.channel}
                    </td>
                    <td class={`text-right py-3 px-4 tabular-nums font-mono ${
                      c.value >= 0 ? "text-emerald-600" : "text-red-500"
                    }`}>
                      {c.value.toFixed(6)}
                    </td>
                    <td class="text-right py-3 px-4 tabular-nums font-mono">
                      {Math.abs(c.value).toFixed(6)}
                    </td>
                    <td class="py-3 px-4">
                      <div class="w-32 bg-slate-100 rounded-full h-2">
                        <div
                          class="h-2 rounded-full bg-indigo-500"
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

  const maxLag = Math.max(0, ...adstock.map((a) => a.max_lag));
  const decayCurves = Array.from({ length: maxLag + 1 }, (_, t) => {
    const row: Record<string, number | string> = { lag: t };
    for (const a of adstock) {
      row[a.channel] = Math.pow(a.decay, t);
    }
    return row;
  });

  return (
    <div class="space-y-6">
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Adstock Decay Curves</h2>
        <p class="text-xs text-slate-500 mb-4">
          Shows how the effect of advertising decays over time for each channel
        </p>
        <ReactChart>
          {() => h(ResponsiveContainer, { width: "100%", height: 350 },
            h(BarChart, { data: decayCurves },
              h(CartesianGrid, { strokeDasharray: "3 3", stroke: "#e2e8f0" }),
              h(XAxis, { dataKey: "lag", tick: { fontSize: 12 }, label: { value: "Lag (periods)", position: "insideBottomRight", offset: -5, fontSize: 12 } }),
              h(YAxis, { tick: { fontSize: 12 }, domain: [0, 1], label: { value: "Weight", angle: -90, position: "insideLeft", fontSize: 12 } }),
              h(Tooltip)
            )
          )}
        </ReactChart>
      </div>

      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Adstock Parameters</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-slate-200">
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">Decay Rate</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">Max Lag</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">Half-Life</th>
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Decay Speed</th>
              </tr>
            </thead>
            <tbody>
              {adstock.map((a, i) => (
                <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td class="py-3 px-4 flex items-center gap-2 font-medium">
                    <span class="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {a.channel}
                  </td>
                  <td class="text-right py-3 px-4 tabular-nums font-mono">{a.decay.toFixed(3)}</td>
                  <td class="text-right py-3 px-4 tabular-nums">{a.max_lag}</td>
                  <td class="text-right py-3 px-4 tabular-nums">{a.halfLife.toFixed(1)} periods</td>
                  <td class="py-3 px-4">
                    <div class="flex items-center gap-2">
                      <div class="w-24 bg-slate-100 rounded-full h-2">
                        <div
                          class={`h-2 rounded-full ${
                            a.decay > 0.7 ? "bg-amber-500" : a.decay > 0.4 ? "bg-emerald-500" : "bg-indigo-500"
                          }`}
                          style={{ width: `${a.decay * 100}%` }}
                        />
                      </div>
                      <span class="text-xs text-slate-500">
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
    <div class="space-y-6">
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Saturation (Hill) Parameters</h2>
        <p class="text-xs text-slate-500 mb-4">
          K = half-saturation point (spend at 50% of max effect), S = shape (steepness of curve)
        </p>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-slate-200">
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">K (Half-Sat)</th>
                <th class="text-right py-3 px-4 font-semibold text-slate-600">S (Shape)</th>
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Saturation Speed</th>
                <th class="text-left py-3 px-4 font-semibold text-slate-600">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {saturation.map((s, i) => (
                <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td class="py-3 px-4 flex items-center gap-2 font-medium">
                    <span class="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {s.channel}
                  </td>
                  <td class="text-right py-3 px-4 tabular-nums font-mono">
                    {s.K.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td class="text-right py-3 px-4 tabular-nums font-mono">{s.S.toFixed(3)}</td>
                  <td class="py-3 px-4">
                    <div class="w-24 bg-slate-100 rounded-full h-2">
                      <div
                        class={`h-2 rounded-full ${
                          s.S > 1.5 ? "bg-red-500" : s.S > 0.8 ? "bg-amber-500" : "bg-emerald-500"
                        }`}
                        style={{ width: `${Math.min(100, s.S * 50)}%` }}
                      />
                    </div>
                  </td>
                  <td class="py-3 px-4 text-xs text-slate-600">
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
    <div class="space-y-6">
      {/* Model Backends */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Blocks size={16} />
          Model Backends
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {adapters.model_backends.map((b) => (
            <div
              class={`flex items-center justify-between rounded-lg border p-4 ${
                b.available ? "border-emerald-200 bg-emerald-50/50" : "border-slate-200 bg-slate-50"
              }`}
            >
              <div class="flex items-center gap-3">
                {b.available ? (
                  <CheckCircle2 size={18} class="text-emerald-500" />
                ) : (
                  <XCircle size={18} class="text-slate-400" />
                )}
                <div>
                  <p class="text-sm font-semibold text-slate-900">{b.name}</p>
                  <p class="text-xs text-slate-500">
                    {b.available ? "Installed & ready" : b.install_hint || "Not installed"}
                  </p>
                </div>
              </div>
              <span
                class={`px-2 py-0.5 rounded text-xs font-medium ${
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
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4">Supported Connectors</h2>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Database size={14} /> Databases
            </div>
            <ul class="space-y-1">
              {adapters.connectors.database.map((d) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Cloud size={14} /> Cloud Storage
            </div>
            <ul class="space-y-1">
              {adapters.connectors.cloud.map((c) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {c}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              <Megaphone size={14} /> Ad Platforms
            </div>
            <ul class="space-y-1">
              {adapters.connectors.ad_platforms.map((a) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {a}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Cache */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Cpu size={16} />
          Cache Backend
        </h2>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Object.entries(cacheInfo).map(([key, val]) => (
            <div class="p-3 bg-slate-50 rounded-lg">
              <p class="text-xs text-slate-500 uppercase tracking-wider mb-0.5">{key.replace(/_/g, " ")}</p>
              <p class="text-sm font-semibold text-slate-900 tabular-nums">{String(val)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SystemTab({ health }: { health: HealthData | null }) {
  return (
    <div class="space-y-6">
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Server size={16} />
          API Server Status
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="p-4 bg-slate-50 rounded-lg">
            <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Status</p>
            <p class="text-sm font-semibold flex items-center gap-2">
              <span class={`w-2 h-2 rounded-full ${health ? "bg-emerald-500" : "bg-red-500"}`} />
              {health?.status ?? "Offline"}
            </p>
          </div>
          <div class="p-4 bg-slate-50 rounded-lg">
            <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Version</p>
            <p class="text-sm font-semibold">{health?.version ?? "—"}</p>
          </div>
          <div class="p-4 bg-slate-50 rounded-lg">
            <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Timestamp</p>
            <p class="text-sm font-semibold font-mono">{health?.timestamp ?? "—"}</p>
          </div>
          <div class="p-4 bg-slate-50 rounded-lg">
            <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">Latest Run</p>
            <p class="text-sm font-semibold font-mono truncate">{health?.latest_run ?? "None"}</p>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <Cpu size={16} />
          API Endpoints
        </h2>
        <div class="space-y-2 text-sm">
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
            <div class="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-slate-50">
              <span class={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                ep.method === "GET" ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
              }`}>
                {ep.method}
              </span>
              <code class="text-xs font-mono text-slate-700 flex-1">{ep.path}</code>
              <span class="text-xs text-slate-500">{ep.desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
          <HardDrive size={16} />
          Quick Actions
        </h2>
        <div class="flex flex-wrap gap-3">
          <a
            href="/docs"
            target="_blank"
            class="px-4 py-2 bg-slate-100 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
          >
            OpenAPI Docs
          </a>
          <a
            href="/redoc"
            target="_blank"
            class="px-4 py-2 bg-slate-100 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
          >
            ReDoc
          </a>
        </div>
      </div>
    </div>
  );
}
