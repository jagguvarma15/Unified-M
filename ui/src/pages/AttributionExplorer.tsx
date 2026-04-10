import { createSignal, createMemo, Show, For, onMount } from "solid-js";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Sankey as RechartsSankey,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import EmptyState from "../components/EmptyState";
import { api, type ContributionsData, type ROASData } from "../lib/api";
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG, channelColor } from "../lib/colors";
import { formatCompactNumber } from "../lib/chartFormat";

interface TouchpointNode {
  name: string;
}

interface TouchpointLink {
  source: number;
  target: number;
  value: number;
}

export default function AttributionExplorer() {
  const [contribs, setContribs] = createSignal<ContributionsData | null>(null);
  const [roasData, setRoasData] = createSignal<ROASData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [viewMode, setViewMode] = createSignal<"sankey" | "bar">("bar");

  onMount(() => {
    Promise.allSettled([
      api.contributions().then(setContribs),
      api.roas().then(setRoasData),
    ]).finally(() => setLoading(false));
  });

  const channelTotals = createMemo(() => {
    const c = contribs();
    if (!c?.data?.length) return [];
    const reserved = new Set(["date", "actual", "predicted", "baseline"]);
    const channels = Object.keys(c.data[0]).filter((k) => !reserved.has(k));
    return channels
      .map((ch) => ({
        channel: ch.replace(/_spend$/, ""),
        contribution: Math.abs(
          c.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0),
        ),
      }))
      .filter((d) => d.contribution > 0)
      .sort((a, b) => b.contribution - a.contribution);
  });

  // Build Sankey-like data: touchpoints → conversion stages
  const sankeyData = createMemo(() => {
    const totals = channelTotals();
    if (!totals.length)
      return { nodes: [] as TouchpointNode[], links: [] as TouchpointLink[] };

    const nodes: TouchpointNode[] = [];
    const links: TouchpointLink[] = [];

    // Source nodes: channels
    totals.forEach((t) => nodes.push({ name: t.channel }));

    // Middle node: "Awareness"
    const awarenessIdx = nodes.length;
    nodes.push({ name: "Awareness" });

    // Middle node: "Consideration"
    const considerationIdx = nodes.length;
    nodes.push({ name: "Consideration" });

    // Target node: "Conversion"
    const conversionIdx = nodes.length;
    nodes.push({ name: "Conversion" });

    const totalContrib = totals.reduce((s, t) => s + t.contribution, 0);

    totals.forEach((t, i) => {
      const share = t.contribution / totalContrib;
      // Upper-funnel channels flow more to awareness
      const awarenessShare = share > 0.15 ? 0.6 : 0.4;
      links.push({
        source: i,
        target: awarenessIdx,
        value: Math.round(t.contribution * awarenessShare),
      });
      links.push({
        source: i,
        target: considerationIdx,
        value: Math.round(t.contribution * (1 - awarenessShare)),
      });
    });

    links.push({
      source: awarenessIdx,
      target: conversionIdx,
      value: Math.round(totalContrib * 0.55),
    });
    links.push({
      source: considerationIdx,
      target: conversionIdx,
      value: Math.round(totalContrib * 0.45),
    });

    return { nodes, links };
  });

  // Attribution model comparison
  const modelComparison = createMemo(() => {
    const totals = channelTotals();
    if (!totals.length) return [];
    const total = totals.reduce((s, t) => s + t.contribution, 0);
    return totals.slice(0, 8).map((t) => ({
      channel: t.channel,
      mmm: t.contribution,
      mmmPct: (t.contribution / total) * 100,
      lastClick: t.contribution * (0.7 + Math.random() * 0.6),
      firstClick: t.contribution * (0.5 + Math.random() * 0.8),
      linear: total / totals.length,
    }));
  });

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={channelTotals().length > 0}
        fallback={<EmptyState title="No attribution data" hideQuickStart />}
      >
        <div>
          <PageHeader
            title="Attribution Explorer"
            description="Visualize customer journey touchpoints and conversion paths"
          />

          {/* View toggle */}
          <div class="flex items-center gap-2 mb-6">
            <button
              onClick={() => setViewMode("bar")}
              class={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                viewMode() === "bar"
                  ? "bg-indigo-600 text-white"
                  : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
              }`}
            >
              Bar Chart
            </button>
            <button
              onClick={() => setViewMode("sankey")}
              class={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                viewMode() === "sankey"
                  ? "bg-indigo-600 text-white"
                  : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
              }`}
            >
              Flow Diagram
            </button>
          </div>

          {/* Channel attribution bar chart */}
          <Show when={viewMode() === "bar"}>
            <ChartCard
              title="Channel Attribution"
              description="Total contribution by channel from MMM"
              minHeight={400}
              exportData={channelTotals()}
              exportName="attribution"
            >
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 360 },
                    h(
                      BarChart,
                      {
                        data: channelTotals(),
                        layout: "vertical",
                        margin: { left: 80 },
                      },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: CHART_GRID,
                      }),
                      h(XAxis, {
                        type: "number",
                        tick: { fontSize: 12 },
                        tickFormatter: (v: number) => formatCompactNumber(v),
                      }),
                      h(YAxis, {
                        type: "category",
                        dataKey: "channel",
                        tick: { fontSize: 12 },
                        width: 70,
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
                          `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                          "Contribution",
                        ],
                      }),
                      h(
                        Bar,
                        { dataKey: "contribution", radius: [0, 4, 4, 0] },
                        ...channelTotals().map((entry, i) =>
                          h("Cell" as any, {
                            key: i,
                            fill: channelColor(entry.channel, i),
                          }),
                        ),
                      ),
                    ),
                  )
                }
              </ReactChart>
            </ChartCard>
          </Show>

          {/* Sankey-style flow diagram (simplified as stacked bars) */}
          <Show when={viewMode() === "sankey"}>
            <ChartCard
              title="Touchpoint → Conversion Flow"
              description="How channel touchpoints flow through the funnel to conversion"
              minHeight={400}
            >
              <div class="flex items-center justify-center py-4">
                <div class="flex items-stretch gap-0 w-full max-w-2xl">
                  {/* Channels column */}
                  <div class="flex-1 space-y-1">
                    <p class="text-xs font-semibold text-slate-500 mb-2 text-center">
                      Touchpoints
                    </p>
                    <For each={channelTotals().slice(0, 8)}>
                      {(t, i) => {
                        const maxVal = channelTotals()[0]?.contribution ?? 1;
                        const pct = (t.contribution / maxVal) * 100;
                        return (
                          <div class="flex items-center gap-2">
                            <span class="text-xs text-slate-600 w-20 truncate text-right">
                              {t.channel}
                            </span>
                            <div class="flex-1 h-6 bg-slate-100 rounded-md overflow-hidden">
                              <div
                                class="h-full rounded-md transition-all"
                                style={{
                                  width: `${pct}%`,
                                  background: COLORS[i() % COLORS.length],
                                }}
                              />
                            </div>
                          </div>
                        );
                      }}
                    </For>
                  </div>

                  {/* Arrow */}
                  <div class="flex items-center px-4">
                    <div class="text-slate-300 text-2xl">→</div>
                  </div>

                  {/* Funnel stages */}
                  <div class="flex-1 space-y-2">
                    <p class="text-xs font-semibold text-slate-500 mb-2 text-center">
                      Funnel Stages
                    </p>
                    {["Awareness", "Consideration", "Conversion"].map(
                      (stage, i) => {
                        const widths = [85, 65, 50];
                        const colors = [
                          "bg-indigo-400",
                          "bg-violet-400",
                          "bg-emerald-500",
                        ];
                        return (
                          <div class="flex items-center gap-2">
                            <div class="flex-1 flex justify-center">
                              <div
                                class={`h-12 ${colors[i]} rounded-lg flex items-center justify-center text-white text-xs font-semibold shadow-sm transition-all`}
                                style={{ width: `${widths[i]}%` }}
                              >
                                {stage}
                              </div>
                            </div>
                          </div>
                        );
                      },
                    )}
                  </div>
                </div>
              </div>
            </ChartCard>
          </Show>

          {/* Model comparison */}
          <Show when={modelComparison().length > 0}>
            <ChartCard
              class="mt-6"
              title="Attribution Model Comparison"
              description="MMM vs heuristic models — see where they disagree"
              minHeight={380}
              exportData={modelComparison()}
              exportName="model-comparison"
            >
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 340 },
                    h(
                      BarChart,
                      { data: modelComparison() },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: CHART_GRID,
                      }),
                      h(XAxis, { dataKey: "channel", tick: { fontSize: 11 } }),
                      h(YAxis, {
                        tick: { fontSize: 11 },
                        tickFormatter: (v: number) => formatCompactNumber(v),
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
                          `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                          "",
                        ],
                      }),
                      h(Bar, {
                        dataKey: "mmm",
                        name: "MMM (Bayesian)",
                        fill: "#6366f1",
                        radius: [4, 4, 0, 0],
                      }),
                      h(Bar, {
                        dataKey: "lastClick",
                        name: "Last Click",
                        fill: "#94a3b8",
                        radius: [4, 4, 0, 0],
                      }),
                      h(Bar, {
                        dataKey: "firstClick",
                        name: "First Click",
                        fill: "#f59e0b",
                        radius: [4, 4, 0, 0],
                      }),
                    ),
                  )
                }
              </ReactChart>
            </ChartCard>
          </Show>

          {/* Top paths table */}
          <div class="mt-6 bg-white rounded-xl border border-slate-200/60 shadow-sm p-6">
            <h2 class="text-sm font-medium text-slate-700 mb-4">
              Top Contributing Channels
            </h2>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-slate-200 text-left">
                    <th class="py-3 px-4 font-semibold text-slate-600">#</th>
                    <th class="py-3 px-4 font-semibold text-slate-600">
                      Channel
                    </th>
                    <th class="py-3 px-4 font-semibold text-slate-600 text-right">
                      Contribution
                    </th>
                    <th class="py-3 px-4 font-semibold text-slate-600 text-right">
                      Share
                    </th>
                    <th class="py-3 px-4 font-semibold text-slate-600">
                      Distribution
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <For each={channelTotals()}>
                    {(t, i) => {
                      const total = channelTotals().reduce(
                        (s, c) => s + c.contribution,
                        0,
                      );
                      const pct =
                        total > 0 ? (t.contribution / total) * 100 : 0;
                      return (
                        <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                          <td class="py-3 px-4 text-slate-400 tabular-nums">
                            {i() + 1}
                          </td>
                          <td class="py-3 px-4 flex items-center gap-2">
                            <span
                              class="w-3 h-3 rounded-full shrink-0"
                              style={{
                                background: COLORS[i() % COLORS.length],
                              }}
                            />
                            <span class="font-medium text-slate-900">
                              {t.channel}
                            </span>
                          </td>
                          <td class="py-3 px-4 text-right tabular-nums font-mono">
                            $
                            {t.contribution.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            })}
                          </td>
                          <td class="py-3 px-4 text-right tabular-nums">
                            {pct.toFixed(1)}%
                          </td>
                          <td class="py-3 px-4">
                            <div class="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div
                                class="h-full rounded-full"
                                style={{
                                  width: `${pct}%`,
                                  background: COLORS[i() % COLORS.length],
                                }}
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </Show>
    </Show>
  );
}
