import { createSignal, Show, For } from "solid-js";
import { DollarSign, TrendingUp, ArrowUpDown, Zap } from "../lib/icons";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { OptimizationSkeleton } from "../components/Skeleton";
import { COLORS } from "../lib/colors";
import { useOptimizationQuery } from "../lib/queries";

export default function Optimization() {
  const query = useOptimizationQuery();
  const data = () => query.data ?? null;
  const [view, setView] = createSignal<"grouped" | "change" | "allocation">(
    "grouped",
  );

  const channels = () =>
    data() ? Object.keys(data()!.optimal_allocation) : [];

  const chartData = () =>
    channels().map((ch) => ({
      channel: ch,
      Current: data()!.current_allocation[ch] ?? 0,
      Optimal: data()!.optimal_allocation[ch] ?? 0,
      Change:
        (data()!.optimal_allocation[ch] ?? 0) -
        (data()!.current_allocation[ch] ?? 0),
    }));

  const optimalPie = () =>
    channels().map((ch) => ({
      name: ch,
      value: data()!.optimal_allocation[ch] ?? 0,
    }));

  const changeSorted = () =>
    [...chartData()].sort((a, b) => Math.abs(b.Change) - Math.abs(a.Change));

  const channelDetails = () =>
    channels().map((ch) => {
      const cur = data()!.current_allocation[ch] ?? 0;
      const opt = data()!.optimal_allocation[ch] ?? 0;
      const diff = opt - cur;
      const pct = cur > 0 ? (diff / cur) * 100 : 0;
      const curShare =
        data()!.total_budget > 0 ? (cur / data()!.total_budget) * 100 : 0;
      const optShare =
        data()!.total_budget > 0 ? (opt / data()!.total_budget) * 100 : 0;
      return { channel: ch, cur, opt, diff, pct, curShare, optShare };
    });

  return (
    <Show when={!query.isLoading} fallback={<OptimizationSkeleton />}>
      <Show
        when={data()}
        fallback={
          <EmptyState
            title="No optimization results"
            message="Run the pipeline to get budget allocation recommendations based on your model."
            hideQuickStart
          />
        }
      >
        <div>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-2xl font-bold text-slate-900">
                Budget Optimization
              </h1>
              <p class="text-sm text-slate-500 mt-1">
                Optimal budget allocation to maximize expected response
              </p>
            </div>
            <a
              href="/scenarios"
              class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
            >
              <Zap size={16} />
              Scenario Planner
            </a>
          </div>

          {/* ---- Metric cards ---- */}
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
            <MetricCard
              label="Total Budget"
              value={`$${data()!.total_budget.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              icon={DollarSign}
              color="indigo"
            />
            <MetricCard
              label="Expected Response"
              value={data()!.expected_response.toLocaleString(undefined, {
                maximumFractionDigits: 0,
              })}
              icon={TrendingUp}
              color="emerald"
            />
            <MetricCard
              label="Improvement"
              value={`${(data()!.improvement_pct ?? 0) >= 0 ? "+" : ""}${(data()!.improvement_pct ?? 0).toFixed(1)}%`}
              icon={ArrowUpDown}
              color={(data()!.improvement_pct ?? 0) >= 0 ? "emerald" : "red"}
            />
            <MetricCard
              label="ROI"
              value={`${(data()!.expected_response / data()!.total_budget).toFixed(2)}x`}
              icon={TrendingUp}
              color="amber"
            />
          </div>

          {/* ---- Chart view selector ---- */}
          <div class="flex gap-1 mt-6 bg-slate-100 rounded-lg p-1 w-fit">
            <For
              each={[
                { key: "grouped" as const, label: "Current vs Optimal" },
                { key: "change" as const, label: "Budget Change" },
                { key: "allocation" as const, label: "Optimal Allocation" },
              ]}
            >
              {(tab) => (
                <button
                  onClick={() => setView(tab.key)}
                  class={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    view() === tab.key
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {tab.label}
                </button>
              )}
            </For>
          </div>

          {/* ---- Charts ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-4">
            <Show when={view() === "grouped"}>
              <>
                <h2 class="text-sm font-semibold text-slate-700 mb-4">
                  Current vs Optimal Allocation
                </h2>
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      { width: "100%", height: 360 },
                      h(
                        BarChart,
                        { data: chartData() },
                        h(CartesianGrid, {
                          strokeDasharray: "3 3",
                          stroke: "#e2e8f0",
                        }),
                        h(XAxis, {
                          dataKey: "channel",
                          tick: { fontSize: 13 },
                        }),
                        h(YAxis, {
                          tick: { fontSize: 12 },
                          tickFormatter: (v: number) =>
                            `$${(v / 1000).toFixed(0)}k`,
                        }),
                        h(Tooltip, {
                          formatter: (v: number) =>
                            `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                        }),
                        h(Legend, null),
                        h(Bar, {
                          dataKey: "Current",
                          fill: "#94a3b8",
                          radius: [4, 4, 0, 0],
                        }),
                        h(Bar, {
                          dataKey: "Optimal",
                          fill: "#6366f1",
                          radius: [4, 4, 0, 0],
                        }),
                      ),
                    )
                  }
                </ReactChart>
              </>
            </Show>

            <Show when={view() === "change"}>
              <>
                <h2 class="text-sm font-semibold text-slate-700 mb-4">
                  Budget Change by Channel
                </h2>
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      { width: "100%", height: 360 },
                      h(
                        BarChart,
                        {
                          data: changeSorted(),
                          layout: "vertical",
                          margin: { left: 80 },
                        },
                        h(CartesianGrid, {
                          strokeDasharray: "3 3",
                          stroke: "#e2e8f0",
                          horizontal: false,
                        }),
                        h(XAxis, {
                          type: "number",
                          tick: { fontSize: 12 },
                          tickFormatter: (v: number) =>
                            `$${(v / 1000).toFixed(0)}k`,
                        }),
                        h(YAxis, {
                          type: "category",
                          dataKey: "channel",
                          tick: { fontSize: 13 },
                          width: 75,
                        }),
                        h(Tooltip, {
                          formatter: (v: number) =>
                            `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                        }),
                        h(
                          Bar,
                          {
                            dataKey: "Change",
                            radius: [0, 4, 4, 0],
                            name: "Change ($)",
                          },
                          ...changeSorted().map((d, i) =>
                            h(Cell, {
                              key: i,
                              fill: d.Change >= 0 ? "#10b981" : "#ef4444",
                            }),
                          ),
                        ),
                      ),
                    )
                  }
                </ReactChart>
              </>
            </Show>

            <Show when={view() === "allocation"}>
              <>
                <h2 class="text-sm font-semibold text-slate-700 mb-4">
                  Optimal Budget Allocation
                </h2>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 320 },
                        h(
                          PieChart,
                          null,
                          h(
                            Pie,
                            {
                              data: optimalPie(),
                              dataKey: "value",
                              nameKey: "name",
                              cx: "50%",
                              cy: "50%",
                              innerRadius: "45%",
                              outerRadius: "80%",
                              paddingAngle: 2,
                              label: ({
                                name,
                                percent,
                              }: {
                                name: string;
                                percent: number;
                              }) => `${name} ${(percent * 100).toFixed(0)}%`,
                              labelLine: false,
                            },
                            ...optimalPie().map((_, i) =>
                              h(Cell, {
                                key: i,
                                fill: COLORS[i % COLORS.length],
                              }),
                            ),
                          ),
                          h(Tooltip, {
                            formatter: (v: number) =>
                              `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                  <div class="flex flex-col justify-center space-y-2">
                    <For
                      each={[...optimalPie()].sort((a, b) => b.value - a.value)}
                    >
                      {(ch, i) => (
                        <div class="flex items-center gap-3">
                          <span
                            class="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ background: COLORS[i() % COLORS.length] }}
                          />
                          <span class="text-sm text-slate-700 flex-1">
                            {ch.name}
                          </span>
                          <span class="text-sm font-mono font-medium text-slate-900">
                            $
                            {ch.value.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            })}
                          </span>
                          <span class="text-xs text-slate-500 w-12 text-right">
                            {data()!.total_budget > 0
                              ? `${((ch.value / data()!.total_budget) * 100).toFixed(0)}%`
                              : "—"}
                          </span>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              </>
            </Show>
          </div>

          {/* ---- Recommendation table ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-semibold text-slate-700 mb-4">
              Channel Recommendations
            </h2>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-slate-200">
                    <th class="text-left py-3 px-4 font-semibold text-slate-600">
                      Channel
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Current
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Current %
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Optimal
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Optimal %
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Change
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Change %
                    </th>
                    <th class="text-left py-3 px-4 font-semibold text-slate-600">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <For each={channelDetails()}>
                    {(ch, i) => (
                      <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                        <td class="py-3 px-4 flex items-center gap-2 font-medium">
                          <span
                            class="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ background: COLORS[i() % COLORS.length] }}
                          />
                          {ch.channel}
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums">
                          $
                          {ch.cur.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums text-slate-500">
                          {ch.curShare.toFixed(1)}%
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums">
                          $
                          {ch.opt.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums text-slate-500">
                          {ch.optShare.toFixed(1)}%
                        </td>
                        <td
                          class={`text-right py-3 px-4 tabular-nums font-medium ${ch.diff > 0 ? "text-emerald-600" : ch.diff < 0 ? "text-red-500" : "text-slate-500"}`}
                        >
                          {ch.diff >= 0 ? "+" : ""}$
                          {ch.diff.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                        <td
                          class={`text-right py-3 px-4 tabular-nums font-medium ${ch.pct > 0 ? "text-emerald-600" : ch.pct < 0 ? "text-red-500" : "text-slate-500"}`}
                        >
                          {ch.pct >= 0 ? "+" : ""}
                          {ch.pct.toFixed(1)}%
                        </td>
                        <td class="py-3 px-4">
                          <span
                            class={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              ch.pct > 5
                                ? "bg-emerald-100 text-emerald-700"
                                : ch.pct < -5
                                  ? "bg-red-100 text-red-700"
                                  : "bg-slate-100 text-slate-600"
                            }`}
                          >
                            {ch.pct > 5
                              ? "Increase"
                              : ch.pct < -5
                                ? "Decrease"
                                : "Maintain"}
                          </span>
                        </td>
                      </tr>
                    )}
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
