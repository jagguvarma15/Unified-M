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
  ScatterChart,
  Scatter,
  ZAxis,
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
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";
import { formatCurrency, formatPercent, formatROAS } from "../lib/format";
import ChartCard from "../components/ChartCard";

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
  if (!latestRun)
    return (
      <EmptyState
        title="No pipeline runs"
        message="Run the pipeline to generate your first set of results."
        action={{ label: "Go to Data", href: "/data" }}
        secondaryAction={{ label: "Connect Datapoint", href: "/datapoint" }}
      />
    );

  const metrics = latestRun.metrics;
  const contribShares = getContribShares(contributions);
  const timeline = getTimeline(contributions);
  const reconBars = getReconBars(reconciliation);
  const waterfallBars = buildWaterfall(waterfall);
  const sparklineActual = diagnostics?.chart?.map((d) => d.actual ?? 0).slice(-30) ?? [];
  const sparklineContribution =
    timeline.rows.length > 0
      ? timeline.rows.slice(-24).map((r) => {
          let sum = 0;
          for (const k of timeline.channels) {
            sum += Number((r as Record<string, unknown>)[k]) || 0;
          }
          return sum;
        })
      : [];

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
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 min-w-0">
        <MetricCard
          label="R-squared"
          value={metrics?.r_squared?.toFixed(3) ?? "\u2014"}
          icon={BarChart2}
          color="indigo"
          tooltip="Variance in the target explained by the model. Higher is better (0–1)."
        />
        <MetricCard
          label="MAPE"
          value={metrics?.mape != null ? formatPercent(metrics.mape, 1) : "\u2014"}
          icon={Percent}
          color="emerald"
          tooltip="Mean Absolute Percentage Error. Lower is better."
          sparkline={sparklineActual.length > 0 ? sparklineActual : undefined}
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
              ? (optimization.improvement_pct >= 0 ? "+" : "") + formatPercent(optimization.improvement_pct, 1)
              : "\u2014"
          }
          icon={TrendingUp}
          color="indigo"
          tooltip="Expected response gain from reallocating to the optimal budget mix."
        />
        <MetricCard
          label="Total Spend"
          value={roas ? formatCurrency(roas.summary.total_spend, true) : "\u2014"}
          icon={DollarSign}
          color="emerald"
          tooltip="Sum of spend across all channels in the latest run."
          sparkline={sparklineContribution.length > 0 ? sparklineContribution : undefined}
        />
        <MetricCard
          label="Blended ROAS"
          value={roas ? formatROAS(roas.summary.blended_roas) : "\u2014"}
          icon={TrendingUp}
          color="amber"
          tooltip="Return on ad spend: total contribution ÷ total spend."
        />
      </div>

      {/* ---- Actual vs Predicted mini ---- */}
      {diagnostics && diagnostics.chart.length > 0 && (
        <ChartCard
          className="mt-6"
          title="Model Fit: Actual vs Predicted"
          description="Daily actual outcome vs model prediction"
          actionHref="/diagnostics"
          actionLabel="View diagnostics →"
          minHeight={260}
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={diagnostics.chart}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => (v / 1000).toFixed(0) + "k"} />
              <Tooltip
                contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                formatter={(v: number) => [v.toLocaleString(undefined, { maximumFractionDigits: 0 }), ""]}
              />
              <Line type="monotone" dataKey="actual" stroke="#334155" strokeWidth={1.5} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#6366f1" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* ---- Charts row ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Contribution donut */}
        <ChartCard
          title="Contribution Share"
          description="How much each channel contributes to the outcome"
          minHeight={320}
        >
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
        </ChartCard>

        {/* Waterfall chart */}
        <ChartCard
          title="Response Waterfall Decomposition"
          description="Baseline + channel lift building to total response"
          minHeight={320}
        >
          {waterfallBars.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={waterfallBars} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
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
        </ChartCard>
      </div>

      {/* ---- Reconciled lift + ROAS row ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Reconciled lift bars */}
        <ChartCard
          title="Reconciled Lift by Channel"
          description="Experiment-calibrated lift with 95% CI"
          minHeight={320}
        >
          {reconBars.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={reconBars}
                layout="vertical"
                margin={{ left: 60 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={CHART_GRID}
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
        </ChartCard>

        {/* ROAS by channel */}
        {roas && roas.channels.length > 0 && (
          <ChartCard
            title="ROAS by Channel"
            description="Return on ad spend per channel vs blended average"
            actionHref="/roas"
            actionLabel="Full analysis →"
            minHeight={320}
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={roas.channels} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v: number) => formatROAS(v, 1)} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }} formatter={(v: number) => [formatROAS(v), "ROAS"]} />
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
          </ChartCard>
        )}
      </div>

      {/* ---- Current vs Optimal Allocation + Channel Efficiency ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Current vs Optimal Allocation */}
        {optimization && optimization.current_allocation && optimization.optimal_allocation && (
          <ChartCard
            title="Budget Allocation: Current vs Optimal"
            description="Side-by-side comparison of where budget is vs where it should be"
            actionHref="/optimization"
            actionLabel="Optimizer →"
            minHeight={320}
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={getAllocComparison(optimization)}
                layout="vertical"
                margin={{ left: 70, right: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => formatCurrency(v, true)}
                />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={60} />
                <Tooltip
                  formatter={(v: number, name: string) => [`$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, name]}
                  contentStyle={{ background: "rgba(15,23,42,0.9)", border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="current" name="Current" fill="#94a3b8" radius={[0, 3, 3, 0]} barSize={10} />
                <Bar dataKey="optimal" name="Optimal" fill="#6366f1" radius={[0, 3, 3, 0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Channel Efficiency Scatter: spend vs ROAS */}
        {roas && roas.channels.length > 0 && (
          <ChartCard
            title="Channel Efficiency Map"
            description="Spend vs ROAS — top-right quadrant is the sweet spot"
            actionHref="/channel-insights"
            actionLabel="Insights →"
            minHeight={320}
          >
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                <XAxis
                  type="number"
                  dataKey="spend"
                  name="Total Spend"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => formatCurrency(v, true)}
                  label={{ value: "Spend", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "#94a3b8" }}
                />
                <YAxis
                  type="number"
                  dataKey="roas"
                  name="ROAS"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: number) => formatROAS(v, 1)}
                  label={{ value: "ROAS", angle: -90, position: "insideLeft", fontSize: 10, fill: "#94a3b8" }}
                />
                <ZAxis type="number" dataKey="contribution" range={[60, 400]} name="Contribution" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ background: "rgba(15,23,42,0.9)", border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                  formatter={(v: number, name: string) => {
                    if (name === "Total Spend" || name === "Contribution")
                      return [`$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, name];
                    return [`${v.toFixed(2)}x`, name];
                  }}
                  labelFormatter={(_, payload) => {
                    const p = payload?.[0]?.payload;
                    return p?.channel ?? "";
                  }}
                />
                <ReferenceLine y={roas.summary.blended_roas} stroke="#94a3b8" strokeDasharray="4 4" />
                <Scatter
                  data={roas.channels.map((c) => ({
                    channel: c.channel.replace(/_spend$/, ""),
                    spend: c.total_spend,
                    roas: c.roas,
                    contribution: c.total_contribution,
                  }))}
                  fill="#6366f1"
                >
                  {roas.channels.map((c, i) => (
                    <Cell
                      key={i}
                      fill={c.roas >= roas.summary.blended_roas ? "#10b981" : "#f59e0b"}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>

      {/* ---- Residuals distribution ---- */}
      {diagnostics && diagnostics.chart.length > 0 && (
        <ChartCard
          className="mt-6"
          title="Prediction Residuals"
          description="Difference between actual and predicted — should hover near zero"
          actionHref="/diagnostics"
          actionLabel="View diagnostics →"
          rightSlot={
            diagnostics.residual_stats && (
              <div className="flex items-center gap-3 text-xs text-slate-500">
                {diagnostics.residual_stats.mean != null && (
                  <span>Mean: <span className="font-mono font-medium text-slate-700">{diagnostics.residual_stats.mean.toFixed(1)}</span></span>
                )}
                {diagnostics.residual_stats.std != null && (
                  <span>Std: <span className="font-mono font-medium text-slate-700">{diagnostics.residual_stats.std.toFixed(1)}</span></span>
                )}
              </div>
            )
          }
          minHeight={220}
        >
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart
              data={diagnostics.chart.map((d) => ({
                date: String(d.date).slice(0, 10),
                residual: (d.actual ?? 0) - (d.predicted ?? 0),
              }))}
              margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
            >
              <defs>
                <linearGradient id="residPos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="residNeg" x1="0" y1="1" x2="0" y2="0">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => (v / 1000).toFixed(0) + "k"} />
              <Tooltip
                formatter={(v: number) => [v.toLocaleString(undefined, { maximumFractionDigits: 0 }), "Residual"]}
                contentStyle={{ background: "rgba(15,23,42,0.9)", border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
              />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
              <Area
                type="monotone"
                dataKey="residual"
                stroke="#6366f1"
                strokeWidth={1.5}
                fill="url(#residPos)"
                fillOpacity={1}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* ---- Timeline ---- */}
      {timeline.channels.length > 0 && (
        <ChartCard
          className="mt-6"
          title="Contributions Over Time"
          description="Stacked daily contribution by channel"
          minHeight={340}
        >
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
        </ChartCard>
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

function getAllocComparison(opt: OptimizationData) {
  const channels = new Set([
    ...Object.keys(opt.current_allocation ?? {}),
    ...Object.keys(opt.optimal_allocation ?? {}),
  ]);
  return Array.from(channels)
    .map((ch) => ({
      channel: ch.replace(/_spend$/, ""),
      current: opt.current_allocation?.[ch] ?? 0,
      optimal: opt.optimal_allocation?.[ch] ?? 0,
    }))
    .sort((a, b) => b.optimal - a.optimal);
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
