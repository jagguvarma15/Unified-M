import { createSignal, onMount, Show, For } from "solid-js";
import { DollarSign, TrendingUp, BarChart2 } from "../lib/icons";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type ROASData } from "../lib/api";
import { COLORS, channelColor } from "../lib/colors";
import { ROASAnalysisSkeleton } from "../components/Skeleton";

export default function ROASAnalysis() {
  const [data, setData] = createSignal<ROASData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [sortBy, setSortBy] = createSignal<
    "roas" | "contribution" | "spend" | "cpa"
  >("roas");

  onMount(() => {
    api
      .roas()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  const sorted = () => {
    const d = data();
    if (!d) return [];
    return [...d.channels].sort((a, b) => {
      switch (sortBy()) {
        case "roas":
          return b.roas - a.roas;
        case "contribution":
          return b.total_contribution - a.total_contribution;
        case "spend":
          return b.total_spend - a.total_spend;
        case "cpa":
          return (a.cpa ?? 0) - (b.cpa ?? 0);
        default:
          return 0;
      }
    });
  };

  const maxRoas = () => {
    const d = data();
    if (!d) return 0;
    return Math.max(...d.channels.map((c) => c.roas));
  };

  const radarData = () => {
    const d = data();
    if (!d) return [];
    const maxContrib = Math.max(...d.channels.map((c) => c.total_contribution));
    const maxSpend = Math.max(...d.channels.map((c) => c.total_spend));
    const maxMROI = Math.max(
      ...d.channels.map((c) => Math.abs(c.marginal_roi ?? 0)),
    );
    const mr = maxRoas();
    return d.channels.map((ch) => ({
      channel: ch.channel,
      ROAS: mr > 0 ? (ch.roas / mr) * 100 : 0,
      Contribution:
        maxContrib > 0 ? (ch.total_contribution / maxContrib) * 100 : 0,
      Spend: maxSpend > 0 ? (ch.total_spend / maxSpend) * 100 : 0,
      "Marginal ROI":
        maxMROI > 0 ? (Math.abs(ch.marginal_roi ?? 0) / maxMROI) * 100 : 0,
    }));
  };

  const radarChartData = () => {
    const rd = radarData();
    const d = data();
    if (!d) return [];
    return [
      {
        metric: "ROAS",
        ...Object.fromEntries(rd.map((r) => [r.channel, r.ROAS])),
      },
      {
        metric: "Contribution",
        ...Object.fromEntries(rd.map((r) => [r.channel, r.Contribution])),
      },
      {
        metric: "Spend Share",
        ...Object.fromEntries(rd.map((r) => [r.channel, r.Spend])),
      },
      {
        metric: "Marginal ROI",
        ...Object.fromEntries(rd.map((r) => [r.channel, r["Marginal ROI"]])),
      },
    ];
  };

  return (
    <Show
      when={!loading()}
      fallback={<ROASAnalysisSkeleton />}
    >
      <Show
        when={data() && data()!.channels.length > 0}
        fallback={
          <EmptyState
            title="No ROAS data"
            message="Run the pipeline to calculate return on ad spend for each channel."
            hideQuickStart
          />
        }
      >
        {() => (
          <div>
            <h1 class="text-2xl font-bold text-slate-900">
              ROAS & ROI Analysis
            </h1>
            <p class="text-sm text-slate-500 mt-1">
              Return on ad spend and efficiency metrics across all channels
            </p>

            {/* Summary cards */}
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
              <MetricCard
                label="Blended ROAS"
                value={`${data()!.summary.blended_roas.toFixed(2)}x`}
                icon={TrendingUp}
                color="indigo"
              />
              <MetricCard
                label="Total Spend"
                value={`$${data()!.summary.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                icon={DollarSign}
                color="emerald"
              />
              <MetricCard
                label="Total Contribution"
                value={data()!.summary.total_contribution.toLocaleString(
                  undefined,
                  { maximumFractionDigits: 0 },
                )}
                icon={BarChart2}
                color="amber"
              />
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
              {/* ROAS by channel bar */}
              <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
                <h2 class="text-sm font-medium text-slate-700 mb-4">
                  ROAS by Channel
                </h2>
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      {
                        width: "100%",
                        height: Math.max(200, data()!.channels.length * 48),
                      },
                      h(
                        BarChart,
                        {
                          data: sorted(),
                          layout: "vertical",
                          margin: { left: 80, right: 20 },
                        },
                        h(CartesianGrid, {
                          strokeDasharray: "3 3",
                          stroke: "#e2e8f0",
                          horizontal: false,
                        }),
                        h(XAxis, {
                          type: "number",
                          tick: { fontSize: 12 },
                          tickFormatter: (v: number) => `${v.toFixed(1)}x`,
                        }),
                        h(YAxis, {
                          type: "category",
                          dataKey: "channel",
                          tick: { fontSize: 13 },
                          width: 75,
                        }),
                        h(Tooltip, {
                          formatter: (v: number) => `${v.toFixed(2)}x`,
                        }),
                        h(
                          Bar,
                          {
                            dataKey: "roas",
                            radius: [0, 6, 6, 0],
                            name: "ROAS",
                          },
                          ...sorted().map((entry, i) =>
                            h(Cell, {
                              key: i,
                              fill: channelColor(entry.channel, i),
                            }),
                          ),
                        ),
                      ),
                    )
                  }
                </ReactChart>
              </div>

              {/* Channel efficiency radar */}
              <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
                <h2 class="text-sm font-medium text-slate-700 mb-4">
                  Channel Efficiency Radar
                </h2>
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      {
                        width: "100%",
                        height: Math.max(300, data()!.channels.length * 48),
                      },
                      h(
                        RadarChart,
                        { data: radarChartData() },
                        h(PolarGrid, { stroke: "#e2e8f0" }),
                        h(PolarAngleAxis, {
                          dataKey: "metric",
                          tick: { fontSize: 11 },
                        }),
                        h(PolarRadiusAxis, {
                          tick: { fontSize: 10 },
                          domain: [0, 100],
                        }),
                        ...data()!.channels.map((ch, i) =>
                          h(Radar, {
                            key: ch.channel,
                            name: ch.channel,
                            dataKey: ch.channel,
                            stroke: channelColor(ch.channel, i),
                            fill: channelColor(ch.channel, i),
                            fillOpacity: 0.15,
                            strokeWidth: 2,
                          }),
                        ),
                        h(Legend),
                      ),
                    )
                  }
                </ReactChart>
              </div>
            </div>

            {/* Spend vs Contribution comparison */}
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-medium text-slate-700 mb-4">
                Spend vs Contribution by Channel
              </h2>
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 360 },
                    h(
                      BarChart,
                      { data: sorted() },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: "#e2e8f0",
                      }),
                      h(XAxis, { dataKey: "channel", tick: { fontSize: 13 } }),
                      h(YAxis, {
                        tick: { fontSize: 12 },
                        tickFormatter: (v: number) =>
                          `$${(v / 1000).toFixed(0)}k`,
                      }),
                      h(Tooltip, {
                        formatter: (v: number) =>
                          `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                      }),
                      h(Legend),
                      h(Bar, {
                        dataKey: "total_spend",
                        name: "Total Spend",
                        fill: "#94a3b8",
                        radius: [4, 4, 0, 0],
                      }),
                      h(Bar, {
                        dataKey: "total_contribution",
                        name: "Total Contribution",
                        fill: "#6366f1",
                        radius: [4, 4, 0, 0],
                      }),
                    ),
                  )
                }
              </ReactChart>
            </div>

            {/* Detailed table */}
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <div class="flex items-center justify-between mb-4">
                <h2 class="text-sm font-medium text-slate-700">
                  Channel Performance Table
                </h2>
                <div class="flex items-center gap-2">
                  <span class="text-xs text-slate-500">Sort by:</span>
                  <select
                    value={sortBy()}
                    onInput={(e) =>
                      setSortBy(
                        e.currentTarget
                          .value as typeof sortBy extends () => infer T
                          ? T
                          : never,
                      )
                    }
                    class="text-xs border border-slate-300 rounded-lg px-2 py-1.5 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  >
                    <option value="roas">ROAS</option>
                    <option value="contribution">Contribution</option>
                    <option value="spend">Spend</option>
                    <option value="cpa">CPA</option>
                  </select>
                </div>
              </div>
              <div class="overflow-x-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr class="border-b border-slate-200">
                      <th class="text-left py-3 px-4 font-semibold text-slate-600">
                        Channel
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        Spend
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        Contribution
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        ROAS
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        Marginal ROI
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        CPA
                      </th>
                      <th class="text-left py-3 px-4 font-semibold text-slate-600">
                        Efficiency
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={sorted()}>
                      {(ch, i) => (
                        <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                          <td class="py-3 px-4 flex items-center gap-2 font-medium">
                            <span
                              class="w-3 h-3 rounded-full flex-shrink-0"
                              style={{
                                background: channelColor(ch.channel, i()),
                              }}
                            />
                            {ch.channel}
                          </td>
                          <td class="text-right py-3 px-4 tabular-nums">
                            $
                            {ch.total_spend.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            })}
                          </td>
                          <td class="text-right py-3 px-4 tabular-nums">
                            {ch.total_contribution.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            })}
                          </td>
                          <td
                            class={`text-right py-3 px-4 tabular-nums font-medium ${
                              ch.roas >= data()!.summary.blended_roas
                                ? "text-emerald-600"
                                : "text-amber-600"
                            }`}
                          >
                            {ch.roas.toFixed(2)}x
                          </td>
                          <td class="text-right py-3 px-4 tabular-nums">
                            {(ch.marginal_roi ?? 0).toFixed(4)}
                          </td>
                          <td class="text-right py-3 px-4 tabular-nums">
                            ${(ch.cpa ?? 0).toFixed(2)}
                          </td>
                          <td class="py-3 px-4">
                            <EfficiencyBar value={ch.roas} max={maxRoas()} />
                          </td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </Show>
    </Show>
  );
}

function EfficiencyBar(props: { value: number; max: number }) {
  const pct = () => (props.max > 0 ? (props.value / props.max) * 100 : 0);
  const color = () =>
    pct() > 70 ? "bg-emerald-500" : pct() > 40 ? "bg-amber-500" : "bg-red-400";
  return (
    <div class="w-24 bg-slate-100 rounded-full h-2">
      <div
        class={`h-2 rounded-full ${color()} transition-all`}
        style={{ width: `${Math.min(100, pct())}%` }}
      />
    </div>
  );
}
