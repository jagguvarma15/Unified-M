import { useEffect, useState } from "react";
import { DollarSign, TrendingUp, ArrowUpDown } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type OptimizationData } from "../lib/api";

export default function Optimization() {
  const [data, setData] = useState<OptimizationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .optimization()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (!data) return <EmptyState />;

  const channels = Object.keys(data.optimal_allocation);
  const chartData = channels.map((ch) => ({
    channel: ch,
    Current: data.current_allocation[ch] ?? 0,
    Optimal: data.optimal_allocation[ch] ?? 0,
  }));

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">
        Budget Optimization
      </h1>
      <p className="text-sm text-slate-500 mt-1">
        Optimal budget allocation to maximize expected response
      </p>

      {/* ---- Metric cards ---- */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
        <MetricCard
          label="Total Budget"
          value={`$${data.total_budget.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          icon={DollarSign}
          color="indigo"
        />
        <MetricCard
          label="Expected Response"
          value={data.expected_response.toLocaleString(undefined, {
            maximumFractionDigits: 0,
          })}
          icon={TrendingUp}
          color="emerald"
        />
        <MetricCard
          label="Improvement"
          value={`${data.improvement_pct >= 0 ? "+" : ""}${data.improvement_pct.toFixed(1)}%`}
          icon={ArrowUpDown}
          color={data.improvement_pct >= 0 ? "emerald" : "red"}
        />
      </div>

      {/* ---- Grouped bar chart ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Current vs Optimal Allocation
        </h2>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="channel" tick={{ fontSize: 13 }} />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(v) =>
                `$${(v / 1000).toFixed(0)}k`
              }
            />
            <Tooltip
              formatter={(v: number) =>
                `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
              }
            />
            <Legend />
            <Bar dataKey="Current" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Optimal" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Recommendation table ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Channel Recommendations
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Channel
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Current
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Optimal
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Change
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Change %
                </th>
              </tr>
            </thead>
            <tbody>
              {channels.map((ch) => {
                const cur = data.current_allocation[ch] ?? 0;
                const opt = data.optimal_allocation[ch] ?? 0;
                const diff = opt - cur;
                const pct = cur > 0 ? (diff / cur) * 100 : 0;
                return (
                  <tr
                    key={ch}
                    className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                  >
                    <td className="py-3 px-4 font-medium">{ch}</td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      ${cur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      ${opt.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td
                      className={`text-right py-3 px-4 tabular-nums font-medium ${
                        diff > 0
                          ? "text-emerald-600"
                          : diff < 0
                            ? "text-red-500"
                            : "text-slate-500"
                      }`}
                    >
                      {diff >= 0 ? "+" : ""}
                      ${diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td
                      className={`text-right py-3 px-4 tabular-nums font-medium ${
                        pct > 0
                          ? "text-emerald-600"
                          : pct < 0
                            ? "text-red-500"
                            : "text-slate-500"
                      }`}
                    >
                      {pct >= 0 ? "+" : ""}
                      {pct.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
