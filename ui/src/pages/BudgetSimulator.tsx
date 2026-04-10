import { createSignal, createEffect, Show, For, onMount } from "solid-js";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  LineChart,
  Line,
  ReferenceLine,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import EmptyState from "../components/EmptyState";
import MetricCard from "../components/MetricCard";
import {
  api,
  type OptimizationData,
  type ResponseCurvesData,
} from "../lib/api";
import {
  COLORS,
  CHART_GRID,
  CHART_TOOLTIP_BG,
  channelColor,
} from "../lib/colors";
import { formatCompactNumber, formatSpendTick } from "../lib/chartFormat";
import { formatCurrency } from "../lib/format";

export default function BudgetSimulator() {
  const [optData, setOptData] = createSignal<OptimizationData | null>(null);
  const [curvesData, setCurvesData] = createSignal<ResponseCurvesData | null>(
    null,
  );
  const [loading, setLoading] = createSignal(true);

  // Real-time slider state
  const [totalBudgetMult, setTotalBudgetMult] = createSignal(1.0);
  const [channelMults, setChannelMults] = createSignal<Record<string, number>>(
    {},
  );
  const [locked, setLocked] = createSignal<Record<string, boolean>>({});

  onMount(() => {
    Promise.allSettled([
      api.optimization().then(setOptData),
      api.responseCurves().then(setCurvesData),
    ]).finally(() => setLoading(false));
  });

  createEffect(() => {
    const d = optData();
    if (!d) return;
    const channels = Object.keys(d.current_allocation);
    const mults: Record<string, number> = {};
    channels.forEach((ch) => (mults[ch] = 1.0));
    setChannelMults(mults);
  });

  const channels = () => {
    const d = optData();
    if (!d) return [];
    return Object.keys(d.current_allocation);
  };

  const baseBudget = () => {
    const d = optData();
    if (!d) return 0;
    return Object.values(d.current_allocation).reduce((a, b) => a + b, 0);
  };

  const simBudget = () => baseBudget() * totalBudgetMult();

  const channelBudget = (ch: string) => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_allocation[ch] ?? 0;
    return base * totalBudgetMult() * (channelMults()[ch] ?? 1);
  };

  const totalAllocated = () =>
    channels().reduce((s, ch) => s + channelBudget(ch), 0);

  const estimatedResponse = () => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_response ?? d.expected_response;
    const budgetRatio = totalAllocated() / baseBudget();
    return base * Math.pow(budgetRatio, 0.7);
  };

  const estimatedROI = () => {
    const spent = totalAllocated();
    return spent > 0 ? estimatedResponse() / spent : 0;
  };

  const upliftPct = () => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_response ?? d.expected_response;
    return base > 0 ? ((estimatedResponse() - base) / base) * 100 : 0;
  };

  const efficiencyData = () => {
    const d = optData();
    if (!d) return [];
    const base = d.current_response ?? d.expected_response;
    const result: { budget: number; response: number; label?: string }[] = [];
    for (let mult = 0.3; mult <= 2.5; mult += 0.05) {
      const budget = baseBudget() * mult;
      result.push({
        budget: Math.round(budget),
        response: Math.round(base * Math.pow(mult, 0.7)),
      });
    }
    return result;
  };

  const channelCompare = () =>
    channels().map((ch) => ({
      channel: ch.replace(/_spend$/, ""),
      current: optData()?.current_allocation[ch] ?? 0,
      simulated: channelBudget(ch),
    }));

  const updateChannelMult = (ch: string, value: number) => {
    setChannelMults((prev) => ({ ...prev, [ch]: value }));
  };

  const resetAll = () => {
    setTotalBudgetMult(1.0);
    const mults: Record<string, number> = {};
    channels().forEach((ch) => (mults[ch] = 1.0));
    setChannelMults(mults);
  };

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
        when={optData()}
        fallback={<EmptyState title="No optimization data" hideQuickStart />}
      >
        <div>
          <div class="flex items-center justify-between">
            <PageHeader
              title="Budget Simulator"
              description="Real-time what-if analysis with budget sliders"
            />
            <button
              onClick={resetAll}
              class="px-3 py-1.5 rounded-md border border-slate-200 bg-white text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Reset All
            </button>
          </div>

          {/* Summary KPIs */}
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            <MetricCard
              label="Simulated Budget"
              value={formatCurrency(totalAllocated(), true)}
              changePct={(totalAllocated() / baseBudget() - 1) * 100}
              changeLabel="vs current"
              color="indigo"
            />
            <MetricCard
              label="Est. Response"
              value={estimatedResponse().toLocaleString(undefined, {
                maximumFractionDigits: 0,
              })}
              changePct={upliftPct()}
              changeLabel="vs current"
              color={upliftPct() >= 0 ? "emerald" : "red"}
            />
            <MetricCard
              label="Est. ROI"
              value={`${estimatedROI().toFixed(2)}x`}
              color="amber"
            />
            <MetricCard
              label="Budget Change"
              value={`${totalBudgetMult() >= 1 ? "+" : ""}${((totalBudgetMult() - 1) * 100).toFixed(0)}%`}
              color={totalBudgetMult() >= 1 ? "emerald" : "red"}
            />
          </div>

          {/* Total budget slider */}
          <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-6 mb-6">
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-sm font-medium text-slate-700">
                Total Budget Multiplier
              </h2>
              <span class="text-lg font-bold tabular-nums text-indigo-600">
                {(totalBudgetMult() * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min={30}
              max={250}
              step={5}
              value={totalBudgetMult() * 100}
              onInput={(e) =>
                setTotalBudgetMult(Number(e.currentTarget.value) / 100)
              }
              class="w-full h-3 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div class="flex justify-between text-[10px] text-slate-400 mt-1 tabular-nums">
              <span>30%</span>
              <span>100%</span>
              <span>250%</span>
            </div>
          </div>

          {/* Per-channel sliders */}
          <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-6 mb-6">
            <h2 class="text-sm font-medium text-slate-700 mb-4">
              Channel-Level Adjustments
            </h2>
            <div class="space-y-4">
              <For each={channels()}>
                {(ch, i) => {
                  const base = () => optData()!.current_allocation[ch] ?? 0;
                  const mult = () => channelMults()[ch] ?? 1;
                  const adjusted = () => base() * totalBudgetMult() * mult();
                  const diff = () => adjusted() - base();
                  const diffPct = () =>
                    base() > 0 ? (diff() / base()) * 100 : 0;

                  return (
                    <div class="flex items-center gap-4">
                      <div class="w-28 flex items-center gap-2">
                        <span
                          class="w-3 h-3 rounded-full shrink-0"
                          style={{
                            background: channelColor(
                              ch.replace(/_spend$/, ""),
                              i(),
                            ),
                          }}
                        />
                        <span class="text-sm font-medium text-slate-700 truncate">
                          {ch.replace(/_spend$/, "")}
                        </span>
                      </div>
                      <div class="flex-1">
                        <input
                          type="range"
                          min={0}
                          max={300}
                          step={5}
                          value={mult() * 100}
                          onInput={(e) =>
                            updateChannelMult(
                              ch,
                              Number(e.currentTarget.value) / 100,
                            )
                          }
                          class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                      </div>
                      <div class="w-20 text-right">
                        <span class="text-sm font-mono tabular-nums">
                          {formatCurrency(adjusted(), true)}
                        </span>
                      </div>
                      <div class="w-20 text-right">
                        <span
                          class={`text-xs font-medium tabular-nums ${diff() > 0 ? "text-emerald-600" : diff() < 0 ? "text-red-500" : "text-slate-400"}`}
                        >
                          {diff() >= 0 ? "+" : ""}
                          {diffPct().toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                }}
              </For>
            </div>
          </div>

          {/* Charts */}
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Comparison bar */}
            <ChartCard
              title="Current vs Simulated Allocation"
              minHeight={350}
              exportData={channelCompare()}
              exportName="budget-simulator"
            >
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 300 },
                    h(
                      BarChart,
                      {
                        data: channelCompare(),
                        layout: "vertical",
                        margin: { left: 70 },
                      },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: CHART_GRID,
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
                          background: CHART_TOOLTIP_BG,
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
                        dataKey: "simulated",
                        name: "Simulated",
                        fill: "#6366f1",
                        radius: [0, 3, 3, 0],
                        barSize: 10,
                      }),
                    ),
                  )
                }
              </ReactChart>
            </ChartCard>

            {/* Efficiency frontier */}
            <ChartCard
              title="Budget Efficiency Frontier"
              description="Expected response at different budget levels"
              minHeight={350}
            >
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 300 },
                    h(
                      LineChart,
                      { data: efficiencyData() },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: CHART_GRID,
                      }),
                      h(XAxis, {
                        dataKey: "budget",
                        tick: { fontSize: 11 },
                        tickFormatter: (v: number) => formatSpendTick(v),
                      }),
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
                          v.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          }),
                          "Response",
                        ],
                        labelFormatter: (v: string) =>
                          `Budget: $${Number(v).toLocaleString()}`,
                      }),
                      h(ReferenceLine, {
                        x: totalAllocated(),
                        stroke: "#6366f1",
                        strokeWidth: 2,
                        strokeDasharray: "4 4",
                        label: {
                          value: "Simulated",
                          fontSize: 10,
                          fill: "#6366f1",
                        },
                      }),
                      h(ReferenceLine, {
                        x: baseBudget(),
                        stroke: "#94a3b8",
                        strokeDasharray: "4 4",
                        label: {
                          value: "Current",
                          fontSize: 10,
                          fill: "#94a3b8",
                        },
                      }),
                      h(Line, {
                        type: "monotone",
                        dataKey: "response",
                        stroke: "#10b981",
                        strokeWidth: 2,
                        dot: false,
                      }),
                    ),
                  )
                }
              </ReactChart>
            </ChartCard>
          </div>
        </div>
      </Show>
    </Show>
  );
}
