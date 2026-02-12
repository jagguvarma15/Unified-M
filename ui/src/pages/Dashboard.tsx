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
import PageHeader from "../components/PageHeader";
import { MetricCardSkeleton } from "../components/Skeleton";
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
      <div>
        <PageHeader title="Dashboard" description="Unified Marketing Measurement overview" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {[...Array(6)].map((_, i) => (
            <MetricCardSkeleton key={i} />
          ))}
        </div>
        <div className="mt-6 flex items-center justify-center h-48 rounded-xl border border-slate-200/60 bg-white/50">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-slate-300 border-t-indigo-500" />
        </div>
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
      <PageHeader
        title="Dashboard"
        description="Unified Marketing Measurement overview"
        detail={
          <span className="inline-flex items-center gap-1.5">
            <Activity size={12} className="text-emerald-500" aria-hidden />
            Run: <code className="font-mono text-slate-500">{latestRun.run_id.slice(0, 12)}…</code>
          </span>
        }
        hint="Metrics from latest pipeline run"
      />

      {/* ---- Metric cards ---- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <MetricCard
          label="R-squared"
          value={metrics?.r_squared?.toFixed(3) ?? "\u2014"}
          icon={BarChart2}
          color="indigo"
          tooltip="Variance in the target explained by the model. Higher is better (0–1)."
        />
        <MetricCard
          label="MAPE"
          value={metrics?.mape ? `${metrics.mape.toFixed(1)}%` : "\u2014"}
          icon={Percent}
          color="emerald"
          tooltip="Mean Absolute Percentage Error. Lower is better."
        />
        <MetricCard
          label="Channels"
          value={latestRun.n_channels}
          icon={Layers}
          color="amber"
          tooltip="Number of media channels in the model."
        />
        <MetricCard
          label="Optim. Uplift"
          value={
            optimization != null && optimization.improvement_pct != null
              ? `${optimization.improvement_pct >= 0 ? "+" : ""}${optimization.improvement_pct.toFixed(1)}%`
              : "\u2014"
          }
          icon={TrendingUp}
          color="indigo"
          tooltip="Expected response gain from reallocating to the optimal budget mix."
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
          tooltip="Sum of spend across all channels in the latest run."
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
          tooltip="Return on ad spend: total contribution ÷ total spend."
        />
      </div>

      {/* ---- Actual vs Predicted mini ---- */}
      {diagnostics && diagnostics.chart.length > 0 && (
        <div className="mt-6 rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-slate-700">Model Fit: Actual vs Predicted</h2>
              <p className="mt-0.5 text-xs text-slate-500">Daily actual outcome vs model prediction</p>
            </div>
            <a href="/diagnostics" className="text-xs font-medium text-indigo-600 hover:text-indigo-700 transition-colors">
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
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">Contribution Share</h2>
          <p className="mb-4 mt-0.5 text-xs text-slate-500">How much each channel contributes to the outcome</p>
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
                  label={({ name, percent }) => {
                    if (percent < CONTRIB_LABEL_MIN_PERCENT) return null;
                    const pctStr = `${(percent * 100).toFixed(0)}%`;
                    const shortName = name.startsWith("Others (") ? "Others" : name;
                    return `${shortName} ${pctStr}`;
                  }}
                  labelLine={true}
                >
                  {contribShares.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: number, name: string) => {
                    const total = contribShares.reduce((s, d) => s + d.value, 0);
                    const pct = total > 0 ? (Number(v) / total) * 100 : 0;
                    return [v.toLocaleString(undefined, { maximumFractionDigits: 0 }) + ` (${pct.toFixed(1)}%)`, name];
                  }}
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
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">Response Waterfall Decomposition</h2>
          <p className="mb-4 mt-0.5 text-xs text-slate-500">Baseline + channel lift building to total response</p>
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
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">Reconciled Lift by Channel</h2>
          <p className="mb-4 mt-0.5 text-xs text-slate-500">Experiment-calibrated lift with 95% CI</p>
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
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-slate-700">ROAS by Channel</h2>
                <p className="mt-0.5 text-xs text-slate-500">Return on ad spend per channel vs blended average</p>
              </div>
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
        <div className="mt-6 rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">Contributions Over Time</h2>
          <p className="mb-4 mt-0.5 text-xs text-slate-500">Stacked daily contribution by channel</p>
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

/** Threshold below which a slice is grouped into "Others"; 8% keeps only meaningful segments and avoids tiny slices with missing/cramped labels */
const CONTRIB_OTHERS_THRESHOLD = 0.08;
/** Min slice size to show inline label; match Others threshold so every visible slice gets a label */
const CONTRIB_LABEL_MIN_PERCENT = 0.08;

function getContribShares(data: ContributionsData | null) {
  if (!data?.data?.length) return [];
  const channels = channelKeys(data.data[0]);
  const raw = channels
    .map((ch) => ({
      name: ch,
      value: Math.abs(
        data.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0),
      ),
    }))
    .filter((t) => t.value > 0)
    .sort((a, b) => b.value - a.value);

  const total = raw.reduce((s, t) => s + t.value, 0);
  if (total <= 0) return [];

  const main: { name: string; value: number }[] = [];
  let othersValue = 0;
  const othersNames: string[] = [];

  for (const t of raw) {
    const pct = t.value / total;
    if (pct < CONTRIB_OTHERS_THRESHOLD) {
      othersValue += t.value;
      othersNames.push(t.name);
    } else {
      main.push(t);
    }
  }

  if (othersValue > 0) {
    const othersLabel =
      othersNames.length > 1
        ? `Others (${othersNames.length})`
        : othersNames[0];
    main.push({ name: othersLabel, value: othersValue });
  }

  return main;
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
