import { createSignal, onMount, createEffect, Show, For } from "solid-js";
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
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import { Calculator } from "../lib/icons";
import EmptyState from "../components/EmptyState";
import {
  api,
  type OptimizationData,
  type ResponseCurvesData,
} from "../lib/api";
import { COLORS, channelColor } from "../lib/colors";

interface ScenarioAllocation {
  [channel: string]: number;
}

interface Scenario {
  id: string;
  name: string;
  budget: number;
  allocation: ScenarioAllocation;
  expectedResponse: number;
}

export default function ScenarioPlanner() {
  const [optData, setOptData] = createSignal<OptimizationData | null>(null);
  const [curvesData, setCurvesData] = createSignal<ResponseCurvesData | null>(
    null,
  );
  const [loading, setLoading] = createSignal(true);
  const [scenarios, setScenarios] = createSignal<Scenario[]>([]);
  const [activeScenarioId, setActiveScenarioId] =
    createSignal<string>("current");

  onMount(() => {
    Promise.allSettled([
      api.optimization().then(setOptData),
      api.responseCurves().then(setCurvesData),
    ]).finally(() => setLoading(false));
  });

  // Build initial scenarios once data loads
  createEffect(() => {
    const d = optData();
    if (!d) return;

    const channels = Object.keys(d.current_allocation);
    const currentBudget = Object.values(d.current_allocation).reduce(
      (a, b) => a + b,
      0,
    );

    const initialScenarios: Scenario[] = [
      {
        id: "current",
        name: "Current",
        budget: currentBudget,
        allocation: { ...d.current_allocation },
        expectedResponse: d.current_response ?? 0,
      },
      {
        id: "optimal",
        name: "Optimal",
        budget: d.total_budget,
        allocation: { ...d.optimal_allocation },
        expectedResponse: d.expected_response,
      },
      {
        id: "conservative",
        name: "Conservative (-15%)",
        budget: currentBudget * 0.85,
        allocation: Object.fromEntries(
          channels.map((ch) => [ch, (d.optimal_allocation[ch] ?? 0) * 0.85]),
        ),
        expectedResponse: d.expected_response * 0.88,
      },
      {
        id: "aggressive",
        name: "Aggressive (+20%)",
        budget: currentBudget * 1.2,
        allocation: Object.fromEntries(
          channels.map((ch) => [ch, (d.optimal_allocation[ch] ?? 0) * 1.2]),
        ),
        expectedResponse: d.expected_response * 1.15,
      },
    ];

    setScenarios(initialScenarios);
  });

  const activeScenario = () =>
    scenarios().find((s) => s.id === activeScenarioId()) ?? scenarios()[0];

  const handleSliderChange = (channel: string, value: number) => {
    setScenarios((prev) =>
      prev.map((s) => {
        if (s.id !== activeScenarioId()) return s;
        const newAllocation = { ...s.allocation, [channel]: value };
        const newBudget = Object.values(newAllocation).reduce(
          (a, b) => a + b,
          0,
        );
        return { ...s, allocation: newAllocation, budget: newBudget };
      }),
    );
  };

  const addScenario = () => {
    const as_ = activeScenario();
    if (!as_) return;
    const id = `custom-${Date.now()}`;
    setScenarios((prev) => [
      ...prev,
      {
        id,
        name: `Custom ${prev.length - 3}`,
        budget: as_.budget,
        allocation: { ...as_.allocation },
        expectedResponse: as_.expectedResponse,
      },
    ]);
    setActiveScenarioId(id);
  };

  const removeScenario = (id: string) => {
    if (id === "current" || id === "optimal") return;
    setScenarios((prev) => prev.filter((s) => s.id !== id));
    if (activeScenarioId() === id) setActiveScenarioId("current");
  };

  const channels = () => {
    const d = optData();
    if (!d) return [];
    return Object.keys(d.current_allocation);
  };

  const maxSpend = () => {
    const d = optData();
    if (!d) return 0;
    const chs = channels();
    return (
      Math.max(
        ...chs.map((ch) =>
          Math.max(
            ...scenarios().map((s) => s.allocation[ch] ?? 0),
            d.current_allocation[ch] ?? 0,
            d.optimal_allocation[ch] ?? 0,
          ),
        ),
      ) * 1.5
    );
  };

  const comparisonData = () =>
    channels().map((ch) => {
      const row: Record<string, string | number> = { channel: ch };
      for (const s of scenarios()) {
        row[s.name] = s.allocation[ch] ?? 0;
      }
      return row;
    });

  const efficiencyData = () => {
    const d = optData();
    const c = curvesData();
    if (!d || !c) return [];
    const baseBudget = d.total_budget;
    const result: { budget: number; response: number }[] = [];
    for (let mult = 0.5; mult <= 2.0; mult += 0.1) {
      const budget = baseBudget * mult;
      const response = d.expected_response * Math.pow(mult, 0.7);
      result.push({
        budget: Math.round(budget),
        response: Math.round(response),
      });
    }
    return result;
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
        fallback={
          <EmptyState
            title="No scenario data"
            message="Run the pipeline first to start exploring budget allocation scenarios."
            hideQuickStart
          />
        }
      >
        {() => (
          <div>
            <div class="flex items-center justify-between">
              <div>
                <h1 class="text-2xl font-bold text-slate-900">
                  Scenario Planner
                </h1>
                <p class="text-sm text-slate-500 mt-1">
                  Compare budget allocation scenarios and explore what-if
                  analyses
                </p>
              </div>
              <button
                onClick={addScenario}
                class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
              >
                <Calculator size={16} />
                New Scenario
              </button>
            </div>

            {/* Scenario tabs */}
            <div class="flex gap-2 mt-6 flex-wrap">
              <For each={scenarios()}>
                {(s) => (
                  <button
                    onClick={() => setActiveScenarioId(s.id)}
                    class={`relative group px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      activeScenarioId() === s.id
                        ? "bg-indigo-600 text-white shadow-sm"
                        : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    {s.name}
                    <Show when={s.id !== "current" && s.id !== "optimal"}>
                      <span
                        onClick={(e) => {
                          e.stopPropagation();
                          removeScenario(s.id);
                        }}
                        class="ml-2 text-xs opacity-60 hover:opacity-100 cursor-pointer"
                      >
                        &times;
                      </span>
                    </Show>
                  </button>
                )}
              </For>
            </div>

            {/* Active scenario editor */}
            <Show when={activeScenario()}>
              {(as_) => (
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
                  <div class="flex items-center justify-between mb-5">
                    <div>
                      <h2 class="text-sm font-medium text-slate-700">
                        {as_().name} — Budget Allocation
                      </h2>
                      <p class="text-xs text-slate-500 mt-0.5">
                        Total: $
                        {as_().budget.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        })}
                      </p>
                    </div>
                    <div class="text-right">
                      <p class="text-xs text-slate-500">Est. Response</p>
                      <p class="text-lg font-bold text-indigo-600">
                        {as_().expectedResponse.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        })}
                      </p>
                    </div>
                  </div>

                  <div class="space-y-4">
                    <For each={channels()}>
                      {(ch, i) => {
                        const value = () => as_().allocation[ch] ?? 0;
                        const currentValue = () =>
                          optData()!.current_allocation[ch] ?? 0;
                        const diff = () => value() - currentValue();
                        const pct = () =>
                          currentValue() > 0
                            ? (diff() / currentValue()) * 100
                            : 0;
                        const isEditable = () =>
                          as_().id !== "current" && as_().id !== "optimal";

                        return (
                          <div class="flex items-center gap-4">
                            <div class="w-28 flex items-center gap-2">
                              <span
                                class="w-3 h-3 rounded-full flex-shrink-0"
                                style={{
                                  background: channelColor(ch, i()),
                                }}
                              />
                              <span class="text-sm font-medium text-slate-700 truncate">
                                {ch}
                              </span>
                            </div>
                            <div class="flex-1">
                              <input
                                type="range"
                                min={0}
                                max={maxSpend()}
                                step={100}
                                value={value()}
                                onInput={(e) =>
                                  handleSliderChange(
                                    ch,
                                    Number(e.currentTarget.value),
                                  )
                                }
                                disabled={!isEditable()}
                                class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600 disabled:opacity-50"
                              />
                            </div>
                            <div class="w-24 text-right">
                              <span class="text-sm font-mono tabular-nums">
                                ${(value() / 1000).toFixed(1)}k
                              </span>
                            </div>
                            <div class="w-20 text-right">
                              <span
                                class={`text-xs font-medium tabular-nums ${
                                  diff() > 0
                                    ? "text-emerald-600"
                                    : diff() < 0
                                      ? "text-red-500"
                                      : "text-slate-400"
                                }`}
                              >
                                {diff() >= 0 ? "+" : ""}
                                {pct().toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>
              )}
            </Show>

            {/* Scenario comparison chart */}
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-medium text-slate-700 mb-4">
                Scenario Comparison by Channel
              </h2>
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 360 },
                    h(
                      BarChart,
                      { data: comparisonData() },
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
                      ...scenarios().map((s, i) =>
                        h(Bar, {
                          key: s.id,
                          dataKey: s.name,
                          fill: COLORS[i % COLORS.length],
                          radius: [4, 4, 0, 0],
                          fillOpacity: activeScenarioId() === s.id ? 1 : 0.5,
                        }),
                      ),
                    ),
                  )
                }
              </ReactChart>
            </div>

            {/* Scenario summary table */}
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-medium text-slate-700 mb-4">
                Scenario Summary
              </h2>
              <div class="overflow-x-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr class="border-b border-slate-200">
                      <th class="text-left py-3 px-4 font-semibold text-slate-600">
                        Scenario
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        Total Budget
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        Est. Response
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        ROI
                      </th>
                      <th class="text-right py-3 px-4 font-semibold text-slate-600">
                        vs Current
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={scenarios()}>
                      {(s, i) => {
                        const currentResponse = () =>
                          scenarios()[0]?.expectedResponse ?? 0;
                        const diff = () =>
                          currentResponse() > 0
                            ? ((s.expectedResponse - currentResponse()) /
                                currentResponse()) *
                              100
                            : 0;
                        const roi = () =>
                          s.budget > 0 ? s.expectedResponse / s.budget : 0;

                        return (
                          <tr
                            class={`border-b border-slate-100 transition-colors cursor-pointer ${
                              activeScenarioId() === s.id
                                ? "bg-indigo-50"
                                : "hover:bg-slate-50"
                            }`}
                            onClick={() => setActiveScenarioId(s.id)}
                          >
                            <td class="py-3 px-4 flex items-center gap-2 font-medium">
                              <span
                                class="w-3 h-3 rounded-full flex-shrink-0"
                                style={{
                                  background: COLORS[i() % COLORS.length],
                                }}
                              />
                              {s.name}
                            </td>
                            <td class="text-right py-3 px-4 tabular-nums">
                              $
                              {s.budget.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              })}
                            </td>
                            <td class="text-right py-3 px-4 tabular-nums">
                              {s.expectedResponse.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              })}
                            </td>
                            <td class="text-right py-3 px-4 tabular-nums">
                              {roi().toFixed(2)}x
                            </td>
                            <td
                              class={`text-right py-3 px-4 tabular-nums font-medium ${
                                diff() > 0
                                  ? "text-emerald-600"
                                  : diff() < 0
                                    ? "text-red-500"
                                    : "text-slate-500"
                              }`}
                            >
                              {s.id === "current"
                                ? "—"
                                : `${diff() >= 0 ? "+" : ""}${diff().toFixed(1)}%`}
                            </td>
                          </tr>
                        );
                      }}
                    </For>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Efficiency frontier */}
            <Show when={efficiencyData().length > 0}>
              <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
                <h2 class="text-sm font-medium text-slate-700 mb-4">
                  Budget Efficiency Frontier
                </h2>
                <p class="text-xs text-slate-500 mb-4">
                  Expected optimal response at different budget levels
                  (diminishing returns)
                </p>
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
                          stroke: "#e2e8f0",
                        }),
                        h(XAxis, {
                          dataKey: "budget",
                          tick: { fontSize: 12 },
                          tickFormatter: (v: number) =>
                            `$${(v / 1000).toFixed(0)}k`,
                          label: {
                            value: "Total Budget",
                            position: "insideBottomRight",
                            offset: -5,
                            fontSize: 12,
                          },
                        }),
                        h(YAxis, {
                          tick: { fontSize: 12 },
                          label: {
                            value: "Expected Response",
                            angle: -90,
                            position: "insideLeft",
                            fontSize: 12,
                          },
                        }),
                        h(Tooltip, {
                          formatter: (v: number) =>
                            v.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            }),
                          labelFormatter: (v: string) =>
                            `Budget: $${Number(v).toLocaleString()}`,
                        }),
                        h(Line, {
                          type: "monotone",
                          dataKey: "response",
                          stroke: "#6366f1",
                          strokeWidth: 2.5,
                          dot: false,
                          name: "Optimal Response",
                        }),
                      ),
                    )
                  }
                </ReactChart>
              </div>
            </Show>
          </div>
        )}
      </Show>
    </Show>
  );
}
