import { createSignal, createMemo } from "solid-js";
import { Activity } from "../lib/icons";
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
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";
import { MetricCardSkeleton } from "../components/Skeleton";
import ChannelDetailPanel from "../components/ChannelDetailPanel";
import {
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
import { api } from "../lib/api";
import { qk } from "../lib/queryKeys";
import { downsampleEvenly } from "../lib/downsample";
import {
  formatCompactNumber,
  formatSpendTick,
  getDateAxisProps,
} from "../lib/chartFormat";
import { trackEvent } from "../lib/telemetry";
import { Show, For } from "solid-js";
import {
  useContributionsQuery,
  useReconciliationQuery,
  useOptimizationQuery,
  useRunsQuery,
  useWaterfallQuery,
  useDiagnosticsQuery,
  useRoasQuery,
} from "../lib/queries";

// ---------------------------------------------------------------------------

interface ChannelDetail {
  channel: string;
  spend?: number;
  contribution?: number;
  roas?: number;
  lift?: number;
  optimal?: number;
  current?: number;
}

export default function Dashboard() {
  const contributionsQ = useContributionsQuery();
  const reconciliationQ = useReconciliationQuery();
  const optimizationQ = useOptimizationQuery();
  const runsQ = useRunsQuery(1);
  const waterfallQ = useWaterfallQuery();
  const diagnosticsQ = useDiagnosticsQuery();
  const roasQ = useRoasQuery();

  const loading = () =>
    contributionsQ.isLoading &&
    reconciliationQ.isLoading &&
    optimizationQ.isLoading &&
    runsQ.isLoading &&
    waterfallQ.isLoading &&
    diagnosticsQ.isLoading &&
    roasQ.isLoading;

  const contributions = () =>
    (contributionsQ.data ?? null) as ContributionsData | null;
  const reconciliation = () =>
    (reconciliationQ.data ?? null) as ReconciliationData | null;
  const optimization = () =>
    (optimizationQ.data ?? null) as OptimizationData | null;
  const runs = () => (runsQ.data ?? null) as RunsData | null;
  const waterfall = () => (waterfallQ.data ?? null) as WaterfallData | null;
  const diagnostics = () =>
    (diagnosticsQ.data ?? null) as DiagnosticsData | null;
  const roas = () => (roasQ.data ?? null) as ROASData | null;
  const latestRun = () => runs()?.runs?.[0] ?? null;

  const contribShares = createMemo(() => getContribShares(contributions()));
  const timeline = createMemo(() => getTimeline(contributions()));
  const reconBars = createMemo(() => getReconBars(reconciliation()));
  const waterfallBars = createMemo(() => buildWaterfall(waterfall()));

  // Sparkline data from contributions timeline
  const kpiSparklines = createMemo(() => {
    const t = timeline();
    if (!t.rows.length) return { total: [] as number[], channels: 0 };
    const totals = t.rows.map((r: Record<string, unknown>) => {
      let sum = 0;
      for (const ch of t.channels) sum += Number(r[ch]) || 0;
      return sum;
    });
    return { total: totals, channels: t.channels.length };
  });

  // Channel detail slide-over
  const [selectedChannel, setSelectedChannel] =
    createSignal<ChannelDetail | null>(null);
  const [detailOpen, setDetailOpen] = createSignal(false);

  const openChannelDetail = (channelName: string) => {
    const roasData = roas();
    const optData = optimization();
    const reconData = reconciliation();
    const ch = roasData?.channels.find(
      (c) =>
        c.channel === channelName ||
        c.channel.replace(/_spend$/, "") === channelName,
    );
    const recon = reconData?.channel_estimates?.[channelName];
    const detail: ChannelDetail = {
      channel: channelName,
      spend: ch?.total_spend,
      contribution: ch?.total_contribution,
      roas: ch?.roas,
      lift: recon?.lift_estimate,
      current:
        optData?.current_allocation?.[channelName] ??
        optData?.current_allocation?.[channelName + "_spend"],
      optimal:
        optData?.optimal_allocation?.[channelName] ??
        optData?.optimal_allocation?.[channelName + "_spend"],
    };
    setSelectedChannel(detail);
    setDetailOpen(true);
    trackEvent("channel_drilldown", { channel: channelName });
  };

  return (
    <Show
      when={!loading()}
      fallback={
        <div>
          <PageHeader
            title="Dashboard"
            description="Unified Marketing Measurement overview"
          />
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <For each={[0, 1, 2, 3, 4, 5]}>{() => <MetricCardSkeleton />}</For>
          </div>
          <div class="mt-6 flex items-center justify-center h-48 rounded-xl border border-slate-200/60 bg-white/50">
            <div class="animate-spin rounded-full h-8 w-8 border-2 border-slate-300 border-t-indigo-500" />
          </div>
        </div>
      }
    >
      <Show
        when={latestRun()}
        fallback={
          <EmptyState
            title="No pipeline runs"
            message="Run the pipeline to generate your first set of results."
            action={{ label: "Go to Data", href: "/data" }}
            secondaryAction={{ label: "Connect Datapoint", href: "/datapoint" }}
            steps={[
              {
                label: "Upload or connect data",
                description: "CSV, database, or ad platform",
                action: { label: "Data", href: "/data" },
              },
              {
                label: "Run the pipeline",
                description: "Train the MMM model",
                action: { label: "Runs", href: "/runs" },
              },
              {
                label: "Explore results",
                description: "Dashboard auto-populates",
              },
            ]}
          />
        }
      >
        {(run) => {
          const metrics = run().metrics;
          return (
            <div>
              <PageHeader
                title="Dashboard"
                description="Unified Marketing Measurement overview"
                detail={
                  <span class="inline-flex items-center gap-1.5">
                    <Activity size={12} class="text-emerald-500" aria-hidden />
                    Run:{" "}
                    <code class="font-mono text-slate-500">
                      {run().run_id.slice(0, 12)}…
                    </code>
                  </span>
                }
                hint="Metrics from latest pipeline run"
              />

              {/* Metric cards with sparklines + delta badges */}
              <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 min-w-0">
                <MetricCard
                  label="R²"
                  value={metrics?.r_squared?.toFixed(3) ?? "—"}
                  tooltip="Variance explained (0–1). Higher is better."
                  sparkline={
                    kpiSparklines().total.length > 4
                      ? kpiSparklines().total.slice(-20)
                      : undefined
                  }
                  changePct={
                    metrics?.r_squared != null
                      ? (metrics.r_squared - 0.85) * 100
                      : undefined
                  }
                  changeLabel="vs baseline"
                />
                <MetricCard
                  label="MAPE"
                  value={
                    metrics?.mape != null ? formatPercent(metrics.mape, 1) : "—"
                  }
                  tooltip="Mean Absolute % Error. Lower is better."
                  changePct={
                    metrics?.mape != null ? -(metrics.mape - 8) : undefined
                  }
                  changeLabel="vs prior"
                />
                <MetricCard
                  label="Channels"
                  value={run().n_channels}
                  tooltip="Media channels in the model."
                />
                <MetricCard
                  label="Optim. Uplift"
                  value={
                    optimization()?.improvement_pct != null
                      ? (optimization()!.improvement_pct! >= 0 ? "+" : "") +
                        formatPercent(optimization()!.improvement_pct!, 1)
                      : "—"
                  }
                  tooltip="Expected gain from optimal budget mix."
                  changePct={optimization()?.improvement_pct ?? undefined}
                  changeLabel="vs current"
                  color="emerald"
                />
                <MetricCard
                  label="Total Spend"
                  value={
                    roas()
                      ? formatCurrency(roas()!.summary.total_spend, true)
                      : "—"
                  }
                  tooltip="Sum of spend across channels."
                  sparkline={
                    kpiSparklines().total.length > 4
                      ? kpiSparklines().total.slice(-20)
                      : undefined
                  }
                />
                <MetricCard
                  label="Blended ROAS"
                  value={
                    roas() ? formatROAS(roas()!.summary.blended_roas) : "—"
                  }
                  tooltip="Total contribution ÷ total spend."
                  changePct={
                    roas()?.summary.blended_roas != null
                      ? (roas()!.summary.blended_roas - 2.5) * 10
                      : undefined
                  }
                  changeLabel="vs benchmark"
                  color={
                    roas()?.summary.blended_roas != null &&
                    roas()!.summary.blended_roas >= 2.5
                      ? "emerald"
                      : "amber"
                  }
                />
              </div>

              {/* ---- Actual vs Predicted mini ---- */}
              <Show when={diagnostics() && diagnostics()!.chart.length > 0}>
                <ChartCard
                  class="mt-6"
                  title="Model Fit: Actual vs Predicted"
                  description="Daily actual outcome vs model prediction"
                  actionHref="/diagnostics"
                  actionLabel="View diagnostics →"
                  minHeight={260}
                  exportData={diagnostics()!.chart}
                  exportName="model-fit"
                >
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 220 },
                        h(
                          LineChart,
                          {
                            data: diagnostics()!.chart,
                            onClick: () =>
                              trackEvent("chart_interaction", {
                                chart_id: "dashboard_model_fit",
                                interaction: "click",
                              }),
                          },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: CHART_GRID,
                          }),
                          h(XAxis, {
                            dataKey: "date",
                            ...getDateAxisProps(diagnostics()!.chart.length),
                          }),
                          h(YAxis, {
                            tick: { fontSize: 10 },
                            tickFormatter: (v: number) =>
                              formatCompactNumber(v),
                          }),
                          h(Tooltip, {
                            contentStyle: {
                              background: CHART_TOOLTIP_BG,
                              border: "none",
                              borderRadius: 8,
                              fontSize: 12,
                              color: "#e2e8f0",
                            },
                            formatter: (v: number) => [
                              v.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              }),
                              "",
                            ],
                          }),
                          h(Line, {
                            type: "monotone",
                            dataKey: "actual",
                            stroke: "#334155",
                            strokeWidth: 1.5,
                            dot: false,
                            name: "Actual",
                          }),
                          h(Line, {
                            type: "monotone",
                            dataKey: "predicted",
                            stroke: "#6366f1",
                            strokeWidth: 1.5,
                            dot: false,
                            strokeDasharray: "5 3",
                            name: "Predicted",
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                </ChartCard>
              </Show>

              {/* ---- Charts row ---- */}
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                {/* Contribution donut */}
                <ChartCard
                  title="Contribution Share"
                  description="How much each channel contributes to the outcome"
                  minHeight={320}
                  exportData={contribShares()}
                  exportName="contribution-share"
                >
                  <Show
                    when={contribShares().length > 0}
                    fallback={
                      <p class="text-sm text-slate-400 py-20 text-center">
                        No contribution data
                      </p>
                    }
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            PieChart,
                            null,
                            h(
                              Pie,
                              {
                                data: contribShares(),
                                dataKey: "value",
                                nameKey: "name",
                                cx: "50%",
                                cy: "50%",
                                innerRadius: "50%",
                                outerRadius: "78%",
                                paddingAngle: 2,
                                label: ({ name, percent }: any) => {
                                  if (percent < CONTRIB_LABEL_MIN_PERCENT)
                                    return null;
                                  const pctStr = `${(percent * 100).toFixed(0)}%`;
                                  const shortName = name.startsWith("Others (")
                                    ? "Others"
                                    : name;
                                  return `${shortName} ${pctStr}`;
                                },
                                labelLine: true,
                                onClick: (_: any, idx: number) => {
                                  const item = contribShares()[idx];
                                  if (item && !item.name.startsWith("Others"))
                                    openChannelDetail(item.name);
                                },
                                cursor: "pointer",
                              },
                              ...contribShares().map((_, i) =>
                                h(Cell, {
                                  key: i,
                                  fill: COLORS[i % COLORS.length],
                                }),
                              ),
                            ),
                            h(Tooltip, {
                              formatter: (v: number, name: string) => {
                                const total = contribShares().reduce(
                                  (s, d) => s + d.value,
                                  0,
                                );
                                const pct =
                                  total > 0 ? (Number(v) / total) * 100 : 0;
                                return [
                                  v.toLocaleString(undefined, {
                                    maximumFractionDigits: 0,
                                  }) + ` (${pct.toFixed(1)}%)`,
                                  name,
                                ];
                              },
                            }),
                          ),
                        )
                      }
                    </ReactChart>
                  </Show>
                </ChartCard>

                {/* Waterfall chart */}
                <ChartCard
                  title="Response Waterfall Decomposition"
                  description="Baseline + channel lift building to total response"
                  minHeight={320}
                  exportData={waterfallBars()}
                  exportName="waterfall"
                >
                  <Show
                    when={waterfallBars().length > 0}
                    fallback={
                      <p class="text-sm text-slate-400 py-20 text-center">
                        No waterfall data
                      </p>
                    }
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            BarChart,
                            {
                              data: waterfallBars(),
                              margin: { left: 10, right: 10 },
                            },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: CHART_GRID,
                            }),
                            h(XAxis, {
                              dataKey: "name",
                              tick: { fontSize: 11 },
                            }),
                            h(YAxis, {
                              tick: { fontSize: 11 },
                              tickFormatter: (v: number) =>
                                formatCompactNumber(v),
                            }),
                            h(Tooltip, {
                              formatter: (v: number) =>
                                v.toLocaleString(undefined, {
                                  maximumFractionDigits: 0,
                                }),
                            }),
                            h(Bar, {
                              dataKey: "invisible",
                              stackId: "stack",
                              fill: "transparent",
                            }),
                            h(
                              Bar,
                              {
                                dataKey: "value",
                                stackId: "stack",
                                radius: [4, 4, 0, 0],
                                cursor: "pointer",
                                onClick: (_: any, idx: number) => {
                                  const item = waterfallBars()[idx];
                                  if (
                                    item &&
                                    item.name !== "Baseline" &&
                                    item.name !== "Total"
                                  )
                                    openChannelDetail(item.name);
                                },
                              },
                              ...waterfallBars().map((d, i) =>
                                h(Cell, { key: i, fill: d.color }),
                              ),
                            ),
                          ),
                        )
                      }
                    </ReactChart>
                  </Show>
                </ChartCard>
              </div>

              {/* ---- Reconciled lift + ROAS row ---- */}
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                {/* Reconciled lift bars */}
                <ChartCard
                  title="Reconciled Lift by Channel"
                  description="Experiment-calibrated lift with 95% CI"
                  minHeight={320}
                  exportData={reconBars()}
                  exportName="reconciled-lift"
                >
                  <Show
                    when={reconBars().length > 0}
                    fallback={
                      <p class="text-sm text-slate-400 py-20 text-center">
                        No reconciliation data
                      </p>
                    }
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            BarChart,
                            {
                              data: reconBars(),
                              layout: "vertical",
                              margin: { left: 60 },
                            },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: CHART_GRID,
                              horizontal: false,
                            }),
                            h(XAxis, {
                              type: "number",
                              tick: { fontSize: 12 },
                            }),
                            h(YAxis, {
                              type: "category",
                              dataKey: "channel",
                              tick: { fontSize: 12 },
                            }),
                            h(Tooltip, {
                              formatter: (
                                _: number,
                                __: string,
                                entry: any,
                              ) => {
                                const d = entry.payload;
                                return [
                                  `${d.lift.toFixed(4)}  [${d.ciLo.toFixed(4)}, ${d.ciHi.toFixed(4)}]`,
                                  "Lift (95% CI)",
                                ];
                              },
                            }),
                            h(
                              Bar,
                              {
                                dataKey: "lift",
                                radius: [0, 4, 4, 0],
                                cursor: "pointer",
                                onClick: (_: any, idx: number) => {
                                  const item = reconBars()[idx];
                                  if (item) openChannelDetail(item.channel);
                                },
                              },
                              ...reconBars().map((d, i) =>
                                h(Cell, {
                                  key: i,
                                  fill:
                                    d.confidence > 0.7
                                      ? "#6366f1"
                                      : d.confidence > 0.4
                                        ? "#f59e0b"
                                        : "#ef4444",
                                }),
                              ),
                            ),
                          ),
                        )
                      }
                    </ReactChart>
                  </Show>
                </ChartCard>

                {/* ROAS by channel */}
                <Show when={roas() && roas()!.channels.length > 0}>
                  <ChartCard
                    title="ROAS by Channel"
                    description="Return on ad spend per channel vs blended average"
                    actionHref="/roas"
                    actionLabel="Full analysis →"
                    minHeight={320}
                    exportData={roas()!.channels}
                    exportName="roas-by-channel"
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            BarChart,
                            {
                              data: roas()!.channels,
                              layout: "vertical",
                              margin: { left: 60 },
                            },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: CHART_GRID,
                              horizontal: false,
                            }),
                            h(XAxis, {
                              type: "number",
                              tick: { fontSize: 12 },
                              tickFormatter: (v: number) => formatROAS(v, 1),
                            }),
                            h(YAxis, {
                              type: "category",
                              dataKey: "channel",
                              tick: { fontSize: 12 },
                            }),
                            h(Tooltip, {
                              contentStyle: {
                                background: CHART_TOOLTIP_BG,
                                border: "none",
                                borderRadius: 8,
                                fontSize: 12,
                                color: "#e2e8f0",
                              },
                              formatter: (v: number) => [formatROAS(v), "ROAS"],
                            }),
                            h(ReferenceLine, {
                              x: roas()!.summary.blended_roas,
                              stroke: "#94a3b8",
                              strokeDasharray: "4 4",
                              label: { value: "Avg", fontSize: 10 },
                            }),
                            h(
                              Bar,
                              {
                                dataKey: "roas",
                                radius: [0, 4, 4, 0],
                                name: "ROAS",
                                cursor: "pointer",
                                onClick: (_: any, idx: number) => {
                                  const ch = roas()!.channels[idx];
                                  if (ch) openChannelDetail(ch.channel);
                                },
                              },
                              ...roas()!.channels.map((c, i) =>
                                h(Cell, {
                                  key: i,
                                  fill:
                                    c.roas >= roas()!.summary.blended_roas
                                      ? "#10b981"
                                      : "#f59e0b",
                                }),
                              ),
                            ),
                          ),
                        )
                      }
                    </ReactChart>
                  </ChartCard>
                </Show>
              </div>

              {/* ---- Current vs Optimal Allocation + Channel Efficiency ---- */}
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                <Show
                  when={
                    optimization() &&
                    optimization()!.current_allocation &&
                    optimization()!.optimal_allocation
                  }
                >
                  <ChartCard
                    title="Budget Allocation: Current vs Optimal"
                    description="Side-by-side comparison of where budget is vs where it should be"
                    actionHref="/optimization"
                    actionLabel="Optimizer →"
                    minHeight={320}
                    exportData={getAllocComparison(optimization()!)}
                    exportName="budget-allocation"
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            BarChart,
                            {
                              data: getAllocComparison(optimization()!),
                              layout: "vertical",
                              margin: { left: 70, right: 10 },
                            },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: CHART_GRID,
                              horizontal: false,
                            }),
                            h(XAxis, {
                              type: "number",
                              tick: { fontSize: 11 },
                              tickFormatter: (v: number) => formatSpendTick(v),
                            }),
                            h(YAxis, {
                              type: "category",
                              dataKey: "channel",
                              tick: { fontSize: 11 },
                              width: 60,
                            }),
                            h(Tooltip, {
                              formatter: (v: number, name: string) => [
                                `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                                name,
                              ],
                              contentStyle: {
                                background: "rgba(15,23,42,0.9)",
                                border: "none",
                                borderRadius: 8,
                                fontSize: 12,
                                color: "#e2e8f0",
                              },
                            }),
                            h(Legend, { wrapperStyle: { fontSize: 11 } }),
                            h(Bar, {
                              dataKey: "current",
                              name: "Current",
                              fill: "#94a3b8",
                              radius: [0, 3, 3, 0],
                              barSize: 10,
                            }),
                            h(Bar, {
                              dataKey: "optimal",
                              name: "Optimal",
                              fill: "#6366f1",
                              radius: [0, 3, 3, 0],
                              barSize: 10,
                            }),
                          ),
                        )
                      }
                    </ReactChart>
                  </ChartCard>
                </Show>

                <Show when={roas() && roas()!.channels.length > 0}>
                  <ChartCard
                    title="Channel Efficiency Map"
                    description="Spend vs ROAS — top-right quadrant is the sweet spot"
                    actionHref="/channel-insights"
                    actionLabel="Insights →"
                    minHeight={320}
                    exportData={roas()!.channels.map((c) => ({
                      channel: c.channel,
                      spend: c.total_spend,
                      roas: c.roas,
                      contribution: c.total_contribution,
                    }))}
                    exportName="channel-efficiency"
                  >
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            ScatterChart,
                            {
                              margin: {
                                left: 10,
                                right: 20,
                                top: 10,
                                bottom: 10,
                              },
                            },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: CHART_GRID,
                            }),
                            h(XAxis, {
                              type: "number",
                              dataKey: "spend",
                              name: "Total Spend",
                              tick: { fontSize: 11 },
                              tickFormatter: (v: number) => formatSpendTick(v),
                              label: {
                                value: "Spend",
                                position: "insideBottomRight",
                                offset: -5,
                                fontSize: 10,
                                fill: "#94a3b8",
                              },
                            }),
                            h(YAxis, {
                              type: "number",
                              dataKey: "roas",
                              name: "ROAS",
                              tick: { fontSize: 11 },
                              tickFormatter: (v: number) => formatROAS(v, 1),
                              label: {
                                value: "ROAS",
                                angle: -90,
                                position: "insideLeft",
                                fontSize: 10,
                                fill: "#94a3b8",
                              },
                            }),
                            h(ZAxis, {
                              type: "number",
                              dataKey: "contribution",
                              range: [60, 400],
                              name: "Contribution",
                            }),
                            h(Tooltip, {
                              cursor: { strokeDasharray: "3 3" },
                              contentStyle: {
                                background: "rgba(15,23,42,0.9)",
                                border: "none",
                                borderRadius: 8,
                                fontSize: 12,
                                color: "#e2e8f0",
                              },
                              formatter: (v: number, name: string) => {
                                if (
                                  name === "Total Spend" ||
                                  name === "Contribution"
                                )
                                  return [
                                    `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                                    name,
                                  ];
                                return [`${v.toFixed(2)}x`, name];
                              },
                              labelFormatter: (_: any, payload: any) => {
                                const p = payload?.[0]?.payload;
                                return p?.channel ?? "";
                              },
                            }),
                            h(ReferenceLine, {
                              y: roas()!.summary.blended_roas,
                              stroke: "#94a3b8",
                              strokeDasharray: "4 4",
                            }),
                            h(
                              Scatter,
                              {
                                data: roas()!.channels.map((c) => ({
                                  channel: c.channel.replace(/_spend$/, ""),
                                  spend: c.total_spend,
                                  roas: c.roas,
                                  contribution: c.total_contribution,
                                })),
                                fill: "#6366f1",
                                cursor: "pointer",
                                onClick: (e: any) => {
                                  if (e?.channel) openChannelDetail(e.channel);
                                },
                              },
                              ...roas()!.channels.map((c, i) =>
                                h(Cell, {
                                  key: i,
                                  fill:
                                    c.roas >= roas()!.summary.blended_roas
                                      ? "#10b981"
                                      : "#f59e0b",
                                }),
                              ),
                            ),
                          ),
                        )
                      }
                    </ReactChart>
                  </ChartCard>
                </Show>
              </div>

              {/* ---- Residuals distribution ---- */}
              <Show when={diagnostics() && diagnostics()!.chart.length > 0}>
                <ChartCard
                  class="mt-6"
                  title="Prediction Residuals"
                  description="Difference between actual and predicted — should hover near zero"
                  actionHref="/diagnostics"
                  actionLabel="View diagnostics →"
                  rightSlot={
                    diagnostics()!.residual_stats ? (
                      <div class="flex items-center gap-3 text-xs text-slate-500">
                        <Show
                          when={diagnostics()!.residual_stats!.mean != null}
                        >
                          <span>
                            Mean:{" "}
                            <span class="font-mono font-medium text-slate-700">
                              {diagnostics()!.residual_stats!.mean!.toFixed(1)}
                            </span>
                          </span>
                        </Show>
                        <Show when={diagnostics()!.residual_stats!.std != null}>
                          <span>
                            Std:{" "}
                            <span class="font-mono font-medium text-slate-700">
                              {diagnostics()!.residual_stats!.std!.toFixed(1)}
                            </span>
                          </span>
                        </Show>
                      </div>
                    ) : undefined
                  }
                  minHeight={220}
                  exportName="residuals"
                >
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 180 },
                        h(
                          AreaChart,
                          {
                            data: downsampleEvenly(
                              diagnostics()!.chart,
                              240,
                            ).map((d) => ({
                              date: String(d.date).slice(0, 10),
                              residual: (d.actual ?? 0) - (d.predicted ?? 0),
                            })),
                            margin: { left: 10, right: 10, top: 5, bottom: 5 },
                          },
                          h(
                            "defs",
                            null,
                            h(
                              "linearGradient",
                              {
                                id: "residPos",
                                x1: "0",
                                y1: "0",
                                x2: "0",
                                y2: "1",
                              },
                              h("stop", {
                                offset: "5%",
                                stopColor: "#10b981",
                                stopOpacity: 0.25,
                              }),
                              h("stop", {
                                offset: "95%",
                                stopColor: "#10b981",
                                stopOpacity: 0,
                              }),
                            ),
                          ),
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: "#e2e8f0",
                          }),
                          h(XAxis, {
                            dataKey: "date",
                            ...getDateAxisProps(diagnostics()!.chart.length),
                          }),
                          h(YAxis, {
                            tick: { fontSize: 10 },
                            tickFormatter: (v: number) =>
                              formatCompactNumber(v),
                          }),
                          h(Tooltip, {
                            formatter: (v: number) => [
                              v.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              }),
                              "Residual",
                            ],
                            contentStyle: {
                              background: "rgba(15,23,42,0.9)",
                              border: "none",
                              borderRadius: 8,
                              fontSize: 12,
                              color: "#e2e8f0",
                            },
                          }),
                          h(ReferenceLine, {
                            y: 0,
                            stroke: "#94a3b8",
                            strokeDasharray: "4 4",
                          }),
                          h(Area, {
                            type: "monotone",
                            dataKey: "residual",
                            stroke: "#6366f1",
                            strokeWidth: 1.5,
                            fill: "url(#residPos)",
                            fillOpacity: 1,
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                </ChartCard>
              </Show>

              {/* ---- Timeline ---- */}
              <Show when={timeline().channels.length > 0}>
                <ChartCard
                  class="mt-6"
                  title="Contributions Over Time"
                  description="Stacked daily contribution by channel"
                  minHeight={340}
                  exportData={timeline().rows}
                  exportName="contributions-timeline"
                >
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 300 },
                        h(
                          AreaChart,
                          { data: timeline().rows },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: "#e2e8f0",
                          }),
                          h(XAxis, {
                            dataKey: "date",
                            ...getDateAxisProps(timeline().rows.length),
                          }),
                          h(YAxis, {
                            tick: { fontSize: 11 },
                            tickFormatter: (v: number) =>
                              formatCompactNumber(v),
                          }),
                          h(Tooltip),
                          h(Legend),
                          ...timeline().channels.map((ch, i) =>
                            h(Area, {
                              key: ch,
                              type: "monotone",
                              dataKey: ch,
                              stackId: "1",
                              fill: COLORS[i % COLORS.length],
                              stroke: COLORS[i % COLORS.length],
                              fillOpacity: 0.7,
                            }),
                          ),
                        ),
                      )
                    }
                  </ReactChart>
                </ChartCard>
              </Show>

              {/* Channel detail slide-over */}
              <ChannelDetailPanel
                open={detailOpen()}
                channel={selectedChannel()}
                onClose={() => setDetailOpen(false)}
              />
            </div>
          );
        }}
      </Show>
    </Show>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RESERVED = new Set(["date", "actual", "predicted", "baseline"]);

function channelKeys(row: Record<string, unknown>): string[] {
  return Object.keys(row).filter((k) => !RESERVED.has(k));
}

const CONTRIB_OTHERS_THRESHOLD = 0.08;
const CONTRIB_LABEL_MIN_PERCENT = 0.08;

function getContribShares(data: ContributionsData | null) {
  if (!data?.data?.length) return [];
  const channels = channelKeys(data.data[0]);
  const raw = channels
    .map((ch) => ({
      name: ch,
      value: Math.abs(data.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0)),
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
  const rows = downsampleEvenly(data.data, 180).map((r) => ({
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
  const bars: {
    name: string;
    value: number;
    invisible: number;
    color: string;
  }[] = [];

  bars.push({
    name: "Baseline",
    value: data.baseline,
    invisible: 0,
    color: "#94a3b8",
  });

  let running = data.baseline;
  for (const ch of data.channels) {
    if (ch.value >= 0) {
      bars.push({
        name: ch.name,
        value: ch.value,
        invisible: running,
        color: "#6366f1",
      });
    } else {
      bars.push({
        name: ch.name,
        value: Math.abs(ch.value),
        invisible: running + ch.value,
        color: "#ef4444",
      });
    }
    running += ch.value;
  }

  bars.push({
    name: "Total",
    value: data.total,
    invisible: 0,
    color: "#10b981",
  });

  return bars;
}
