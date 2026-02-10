import { useEffect, useState, useCallback } from "react";
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
import { Calculator, RefreshCw } from "lucide-react";
import EmptyState from "../components/EmptyState";
import { api, type OptimizationData, type ResponseCurvesData } from "../lib/api";
import { COLORS } from "../lib/colors";

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
  const [optData, setOptData] = useState<OptimizationData | null>(null);
  const [curvesData, setCurvesData] = useState<ResponseCurvesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<string>("current");

  useEffect(() => {
    Promise.allSettled([
      api.optimization().then(setOptData),
      api.responseCurves().then(setCurvesData),
    ]).finally(() => setLoading(false));
  }, []);

  // Build initial scenarios once data loads
  useEffect(() => {
    if (!optData) return;

    const channels = Object.keys(optData.current_allocation);
    const currentBudget = Object.values(optData.current_allocation).reduce((a, b) => a + b, 0);

    const initialScenarios: Scenario[] = [
      {
        id: "current",
        name: "Current",
        budget: currentBudget,
        allocation: { ...optData.current_allocation },
        expectedResponse: optData.current_response ?? 0,
      },
      {
        id: "optimal",
        name: "Optimal",
        budget: optData.total_budget,
        allocation: { ...optData.optimal_allocation },
        expectedResponse: optData.expected_response,
      },
      {
        id: "conservative",
        name: "Conservative (-15%)",
        budget: currentBudget * 0.85,
        allocation: Object.fromEntries(
          channels.map((ch) => [ch, (optData.optimal_allocation[ch] ?? 0) * 0.85])
        ),
        expectedResponse: optData.expected_response * 0.88,
      },
      {
        id: "aggressive",
        name: "Aggressive (+20%)",
        budget: currentBudget * 1.2,
        allocation: Object.fromEntries(
          channels.map((ch) => [ch, (optData.optimal_allocation[ch] ?? 0) * 1.2])
        ),
        expectedResponse: optData.expected_response * 1.15,
      },
    ];

    setScenarios(initialScenarios);
  }, [optData]);

  const activeScenario = scenarios.find((s) => s.id === activeScenarioId) ?? scenarios[0];

  const handleSliderChange = useCallback(
    (channel: string, value: number) => {
      setScenarios((prev) =>
        prev.map((s) => {
          if (s.id !== activeScenarioId) return s;
          const newAllocation = { ...s.allocation, [channel]: value };
          const newBudget = Object.values(newAllocation).reduce((a, b) => a + b, 0);
          return { ...s, allocation: newAllocation, budget: newBudget };
        })
      );
    },
    [activeScenarioId]
  );

  const addScenario = () => {
    if (!activeScenario) return;
    const id = `custom-${Date.now()}`;
    setScenarios((prev) => [
      ...prev,
      {
        id,
        name: `Custom ${prev.length - 3}`,
        budget: activeScenario.budget,
        allocation: { ...activeScenario.allocation },
        expectedResponse: activeScenario.expectedResponse,
      },
    ]);
    setActiveScenarioId(id);
  };

  const removeScenario = (id: string) => {
    if (id === "current" || id === "optimal") return;
    setScenarios((prev) => prev.filter((s) => s.id !== id));
    if (activeScenarioId === id) setActiveScenarioId("current");
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (!optData) return <EmptyState />;

  const channels = Object.keys(optData.current_allocation);
  const maxSpend = Math.max(
    ...channels.map((ch) =>
      Math.max(
        ...scenarios.map((s) => s.allocation[ch] ?? 0),
        optData.current_allocation[ch] ?? 0,
        optData.optimal_allocation[ch] ?? 0
      )
    )
  ) * 1.5;

  // Comparison chart data
  const comparisonData = channels.map((ch) => {
    const row: Record<string, string | number> = { channel: ch };
    for (const s of scenarios) {
      row[s.name] = s.allocation[ch] ?? 0;
    }
    return row;
  });

  // Budget efficiency curve data (if response curves available)
  const efficiencyData: { budget: number; response: number }[] = [];
  if (curvesData) {
    const baseBudget = optData.total_budget;
    for (let mult = 0.5; mult <= 2.0; mult += 0.1) {
      const budget = baseBudget * mult;
      // Simple proportional scaling estimate
      const response = optData.expected_response * Math.pow(mult, 0.7);
      efficiencyData.push({
        budget: Math.round(budget),
        response: Math.round(response),
      });
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Scenario Planner</h1>
          <p className="text-sm text-slate-500 mt-1">
            Compare budget allocation scenarios and explore what-if analyses
          </p>
        </div>
        <button
          onClick={addScenario}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
        >
          <Calculator size={16} />
          New Scenario
        </button>
      </div>

      {/* Scenario tabs */}
      <div className="flex gap-2 mt-6 flex-wrap">
        {scenarios.map((s) => (
          <button
            key={s.id}
            onClick={() => setActiveScenarioId(s.id)}
            className={`relative group px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeScenarioId === s.id
                ? "bg-indigo-600 text-white shadow-sm"
                : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            {s.name}
            {s.id !== "current" && s.id !== "optimal" && (
              <span
                onClick={(e) => {
                  e.stopPropagation();
                  removeScenario(s.id);
                }}
                className="ml-2 text-xs opacity-60 hover:opacity-100 cursor-pointer"
              >
                &times;
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Active scenario editor */}
      {activeScenario && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-sm font-semibold text-slate-700">
                {activeScenario.name} — Budget Allocation
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Total: ${activeScenario.budget.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-500">Est. Response</p>
              <p className="text-lg font-bold text-indigo-600">
                {activeScenario.expectedResponse.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {channels.map((ch, i) => {
              const value = activeScenario.allocation[ch] ?? 0;
              const currentValue = optData.current_allocation[ch] ?? 0;
              const diff = value - currentValue;
              const pct = currentValue > 0 ? (diff / currentValue) * 100 : 0;
              const isEditable = activeScenario.id !== "current" && activeScenario.id !== "optimal";

              return (
                <div key={ch} className="flex items-center gap-4">
                  <div className="w-28 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    <span className="text-sm font-medium text-slate-700 truncate">{ch}</span>
                  </div>
                  <div className="flex-1">
                    <input
                      type="range"
                      min={0}
                      max={maxSpend}
                      step={100}
                      value={value}
                      onChange={(e) => handleSliderChange(ch, Number(e.target.value))}
                      disabled={!isEditable}
                      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600 disabled:opacity-50"
                    />
                  </div>
                  <div className="w-24 text-right">
                    <span className="text-sm font-mono tabular-nums">
                      ${(value / 1000).toFixed(1)}k
                    </span>
                  </div>
                  <div className="w-20 text-right">
                    <span className={`text-xs font-medium tabular-nums ${
                      diff > 0 ? "text-emerald-600" : diff < 0 ? "text-red-500" : "text-slate-400"
                    }`}>
                      {diff >= 0 ? "+" : ""}{pct.toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Scenario comparison chart */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Scenario Comparison by Channel
        </h2>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="channel" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
            <Tooltip
              formatter={(v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
            <Legend />
            {scenarios.map((s, i) => (
              <Bar
                key={s.id}
                dataKey={s.name}
                fill={COLORS[i % COLORS.length]}
                radius={[4, 4, 0, 0]}
                fillOpacity={activeScenarioId === s.id ? 1 : 0.5}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scenario summary table */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Scenario Summary
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Scenario</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Total Budget</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Est. Response</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">ROI</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">vs Current</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((s, i) => {
                const currentResponse = scenarios[0]?.expectedResponse ?? 0;
                const diff = currentResponse > 0
                  ? ((s.expectedResponse - currentResponse) / currentResponse) * 100
                  : 0;
                const roi = s.budget > 0 ? s.expectedResponse / s.budget : 0;

                return (
                  <tr
                    key={s.id}
                    className={`border-b border-slate-100 transition-colors cursor-pointer ${
                      activeScenarioId === s.id ? "bg-indigo-50" : "hover:bg-slate-50"
                    }`}
                    onClick={() => setActiveScenarioId(s.id)}
                  >
                    <td className="py-3 px-4 flex items-center gap-2 font-medium">
                      <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                      {s.name}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      ${s.budget.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      {s.expectedResponse.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      {roi.toFixed(2)}x
                    </td>
                    <td className={`text-right py-3 px-4 tabular-nums font-medium ${
                      diff > 0 ? "text-emerald-600" : diff < 0 ? "text-red-500" : "text-slate-500"
                    }`}>
                      {s.id === "current" ? "—" : `${diff >= 0 ? "+" : ""}${diff.toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Efficiency frontier */}
      {efficiencyData.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Budget Efficiency Frontier
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Expected optimal response at different budget levels (diminishing returns)
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={efficiencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="budget"
                tick={{ fontSize: 12 }}
                tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                label={{ value: "Total Budget", position: "insideBottomRight", offset: -5, fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                label={{ value: "Expected Response", angle: -90, position: "insideLeft", fontSize: 12 }}
              />
              <Tooltip
                formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                labelFormatter={(v) => `Budget: $${Number(v).toLocaleString()}`}
              />
              <Line
                type="monotone"
                dataKey="response"
                stroke="#6366f1"
                strokeWidth={2.5}
                dot={false}
                name="Optimal Response"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
