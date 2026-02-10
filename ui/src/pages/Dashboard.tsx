import { useEffect, useState } from "react";
import { BarChart2, Percent, Layers, TrendingUp, DollarSign, Activity } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  CartesianGrid,
  ResponsiveContainer,
  LineChart,
  Line,
  ReferenceLine,
} from "recharts";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import {
  api,
  type ContributionsData,
  type ReconciliationData,
  type OptimizationData,
  type RunsData,
  type WaterfallData,
  type DiagnosticsData,
  type ROASData,
} from "../lib/api";
import { COLORS } from "../lib/colors";

// ---------------------------------------------------------------------------

export default function Dashboard() {
  const [contributions, setContributions] = useState<ContributionsData | null>(null);
  const [reconciliation, setReconciliation] = useState<ReconciliationData | null>(null);
  const [optimization, setOptimization] = useState<OptimizationData | null>(null);
  const [runs, setRuns] = useState<RunsData | null>(null);
  const [waterfall, setWaterfall] = useState<WaterfallData | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsData | null>(null);
  const [roas, setRoas] = useState<ROASData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      api.contributions().then(setContributions),
      api.reconciliation().then(setReconciliation),
      api.optimization().then(setOptimization),
      api.runs(1).then(setRuns),
      api.waterfall().then(setWaterfall),
      api.diagnostics().then(setDiagnostics),
      api.roas().then(setRoas),
    ]).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  const latestRun = runs?.runs?.[0];
  if (!latestRun) return <EmptyState />;

  const metrics = latestRun.metrics;
  const contribShares = getContribShares(contributions);
  const timeline = getTimeline(contributions);
  const reconBars = getReconBars(reconciliation);
  const waterfallBars = buildWaterfall(waterfall);

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-sm text-slate-500 mt-1">
            Unified Marketing Measurement overview
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Activity size={14} className="text-emerald-400" />
          Run: {latestRun.run_id.slice(0, 12)}...
        </div>
      </div>

      {/* ---- Metric cards ---- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
        <MetricCard
          label="R-squared"
          value={metrics?.r_squared?.toFixed(3) ?? "\u2014"}
          icon={BarChart2}
          color="indigo"
        />
        <MetricCard
          label="MAPE"
          value={metrics?.mape ? `${metrics.mape.toFixed(1)}%` : "\u2014"}
          icon={Percent}
          color="emerald"
        />
        <MetricCard
          label="Channels"
          value={latestRun.n_channels}
          icon={Layers}
          color="amber"
        />
        <MetricCard
          label="Optim. Uplift"
          value={
            optimization
              ? `${optimization.improvement_pct >= 0 ? "+" : ""}${optimization.improvement_pct.toFixed(1)}%`
              : "\u2014"
          }
          icon={TrendingUp}
          color="indigo"
        />
        <MetricCard
          label="Total Spend"
          value={
            roas
              ? `$${(roas.summary.total_spend / 1000).toFixed(0)}k`
              : "\u2014"
          }
          icon={DollarSign}
          color="emerald"
        />
        <MetricCard
          label="Blended ROAS"
          value={
            roas
              ? `${roas.summary.blended_roas.toFixed(2)}x`
              : "\u2014"
          }
          icon={TrendingUp}
          color="amber"
        />
      </div>

      {/* ---- Actual vs Predicted mini ---- */}
      {diagnostics && diagnostics.chart.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-700">
              Model Fit: Actual vs Predicted
            </h2>
            <a href="/diagnostics" className="text-xs text-indigo-600 hover:text-indigo-700 font-medium">
              View diagnostics &rarr;
            </a>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={diagnostics.chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => (v / 1000).toFixed(0) + "k"} />
              <Tooltip
                formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              />
              <Line type="monotone" dataKey="actual" stroke="#334155" strokeWidth={1.5} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#6366f1" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ---- Charts row ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Contribution donut */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Contribution Share
          </h2>
          {contribShares.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={contribShares}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius="50%"
                  outerRadius="78%"
                  paddingAngle={2}
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  {contribShares.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: number) =>
                    v.toLocaleString(undefined, { maximumFractionDigits: 0 })
                  }
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-400 py-20 text-center">
              No contribution data
            </p>
          )}
        </div>

        {/* Waterfall chart */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Response Waterfall Decomposition
          </h2>
          {waterfallBars.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={waterfallBars} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v / 1000).toFixed(0) + "k"} />
                <Tooltip
                  formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                />
                <Bar dataKey="invisible" stackId="stack" fill="transparent" />
                <Bar dataKey="value" stackId="stack" radius={[4, 4, 0, 0]}>
                  {waterfallBars.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-400 py-20 text-center">
              No waterfall data
            </p>
          )}
        </div>
      </div>

      {/* ---- Reconciled lift + ROAS row ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Reconciled lift bars */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Reconciled Lift by Channel
          </h2>
          {reconBars.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={reconBars}
                layout="vertical"
                margin={{ left: 60 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e2e8f0"
                  horizontal={false}
                />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="channel"
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  formatter={(_: number, __: string, entry: any) => {
                    const d = entry.payload;
                    return [
                      `${d.lift.toFixed(4)}  [${d.ciLo.toFixed(4)}, ${d.ciHi.toFixed(4)}]`,
                      "Lift (95% CI)",
                    ];
                  }}
                />
                <Bar dataKey="lift" radius={[0, 4, 4, 0]}>
                  {reconBars.map((d, i) => (
                    <Cell
                      key={i}
                      fill={
                        d.confidence > 0.7
                          ? "#6366f1"
                          : d.confidence > 0.4
                            ? "#f59e0b"
                            : "#ef4444"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-400 py-20 text-center">
              No reconciliation data
            </p>
          )}
        </div>

        {/* ROAS by channel */}
        {roas && roas.channels.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-700">
                ROAS by Channel
              </h2>
              <a href="/roas" className="text-xs text-indigo-600 hover:text-indigo-700 font-medium">
                Full analysis &rarr;
              </a>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={roas.channels} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v) => `${v.toFixed(1)}x`} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: number) => `${v.toFixed(2)}x`} />
                <ReferenceLine x={roas.summary.blended_roas} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "Avg", fontSize: 10 }} />
                <Bar dataKey="roas" radius={[0, 4, 4, 0]} name="ROAS">
                  {roas.channels.map((c, i) => (
                    <Cell
                      key={i}
                      fill={c.roas >= roas.summary.blended_roas ? "#10b981" : "#f59e0b"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ---- Timeline ---- */}
      {timeline.channels.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Contributions Over Time
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeline.rows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {timeline.channels.map((ch, i) => (
                <Area
                  key={ch}
                  type="monotone"
                  dataKey={ch}
                  stackId="1"
                  fill={COLORS[i % COLORS.length]}
                  stroke={COLORS[i % COLORS.length]}
                  fillOpacity={0.7}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RESERVED = new Set(["date", "actual", "predicted", "baseline"]);

function channelKeys(row: Record<string, unknown>): string[] {
  return Object.keys(row).filter((k) => !RESERVED.has(k));
}

function getContribShares(data: ContributionsData | null) {
  if (!data?.data?.length) return [];
  const channels = channelKeys(data.data[0]);
  return channels
    .map((ch) => ({
      name: ch,
      value: Math.abs(
        data.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0),
      ),
    }))
    .filter((t) => t.value > 0)
    .sort((a, b) => b.value - a.value);
}

function getTimeline(data: ContributionsData | null) {
  if (!data?.data?.length) return { rows: [], channels: [] as string[] };
  const channels = channelKeys(data.data[0]);
  const step = Math.max(1, Math.floor(data.data.length / 120));
  const rows = data.data
    .filter((_, i) => i % step === 0)
    .map((r) => ({
      date: String(r.date).slice(0, 10),
      ...Object.fromEntries(channels.map((ch) => [ch, Number(r[ch]) || 0])),
    }));
  return { rows, channels };
}

function getReconBars(data: ReconciliationData | null) {
  if (!data?.channel_estimates) return [];
  return Object.entries(data.channel_estimates).map(([channel, est]) => ({
    channel,
    lift: est.lift_estimate,
    ciLo: est.ci_lower,
    ciHi: est.ci_upper,
    confidence: est.confidence_score,
  }));
}

function buildWaterfall(data: WaterfallData | null) {
  if (!data) return [];
  const bars: { name: string; value: number; invisible: number; color: string }[] = [];

  // Baseline bar
  bars.push({ name: "Baseline", value: data.baseline, invisible: 0, color: "#94a3b8" });

  // Channel bars (stacked waterfall)
  let running = data.baseline;
  for (const ch of data.channels) {
    if (ch.value >= 0) {
      bars.push({ name: ch.name, value: ch.value, invisible: running, color: "#6366f1" });
    } else {
      bars.push({ name: ch.name, value: Math.abs(ch.value), invisible: running + ch.value, color: "#ef4444" });
    }
    running += ch.value;
  }

  // Total bar
  bars.push({ name: "Total", value: data.total, invisible: 0, color: "#10b981" });

  return bars;
}
