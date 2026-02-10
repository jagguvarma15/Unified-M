import { useEffect, useState } from "react";
import { DollarSign, TrendingUp, ArrowUpDown, Zap } from "lucide-react";
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
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type OptimizationData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Optimization() {
  const [data, setData] = useState<OptimizationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<"grouped" | "change" | "allocation">("grouped");

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
    Change: (data.optimal_allocation[ch] ?? 0) - (data.current_allocation[ch] ?? 0),
  }));

  // Allocation pie data
  const optimalPie = channels.map((ch) => ({
    name: ch,
    value: data.optimal_allocation[ch] ?? 0,
  }));

  // Change sorted by absolute delta
  const changeSorted = [...chartData].sort((a, b) => Math.abs(b.Change) - Math.abs(a.Change));

  // Calculate per-channel metrics
  const channelDetails = channels.map((ch) => {
    const cur = data.current_allocation[ch] ?? 0;
    const opt = data.optimal_allocation[ch] ?? 0;
    const diff = opt - cur;
    const pct = cur > 0 ? (diff / cur) * 100 : 0;
    const curShare = data.total_budget > 0 ? (cur / data.total_budget) * 100 : 0;
    const optShare = data.total_budget > 0 ? (opt / data.total_budget) * 100 : 0;
    return { channel: ch, cur, opt, diff, pct, curShare, optShare };
  });

  return (
    <div>
      <div className="flex items-center justify-between">
    <div>
      <h1 className="text-2xl font-bold text-slate-900">
        Budget Optimization
      </h1>
      <p className="text-sm text-slate-500 mt-1">
        Optimal budget allocation to maximize expected response
      </p>
        </div>
        <a
          href="/scenarios"
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors text-sm"
        >
          <Zap size={16} />
          Scenario Planner
        </a>
      </div>

      {/* ---- Metric cards ---- */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
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
        <MetricCard
          label="ROI"
          value={`${(data.expected_response / data.total_budget).toFixed(2)}x`}
          icon={TrendingUp}
          color="amber"
        />
      </div>

      {/* ---- Chart view selector ---- */}
      <div className="flex gap-1 mt-6 bg-slate-100 rounded-lg p-1 w-fit">
        {([
          { key: "grouped" as const, label: "Current vs Optimal" },
          { key: "change" as const, label: "Budget Change" },
          { key: "allocation" as const, label: "Optimal Allocation" },
        ]).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setView(tab.key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              view === tab.key
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ---- Charts ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-4">
        {view === "grouped" && (
          <>
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
          </>
        )}

        {view === "change" && (
          <>
            <h2 className="text-sm font-semibold text-slate-700 mb-4">
              Budget Change by Channel
            </h2>
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={changeSorted} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 13 }} width={75} />
                <Tooltip formatter={(v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
                <Bar dataKey="Change" radius={[0, 4, 4, 0]} name="Change ($)">
                  {changeSorted.map((d, i) => (
                    <Cell key={i} fill={d.Change >= 0 ? "#10b981" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        )}

        {view === "allocation" && (
          <>
            <h2 className="text-sm font-semibold text-slate-700 mb-4">
              Optimal Budget Allocation
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie
                    data={optimalPie}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius="45%"
                    outerRadius="80%"
                    paddingAngle={2}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {optimalPie.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v: number) =>
                      `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                    }
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-col justify-center space-y-2">
                {optimalPie.sort((a, b) => b.value - a.value).map((ch, i) => (
                  <div key={ch.name} className="flex items-center gap-3">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    <span className="text-sm text-slate-700 flex-1">{ch.name}</span>
                    <span className="text-sm font-mono font-medium text-slate-900">
                      ${ch.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </span>
                    <span className="text-xs text-slate-500 w-12 text-right">
                      {data.total_budget > 0 ? `${((ch.value / data.total_budget) * 100).toFixed(0)}%` : "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
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
                  Current %
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Optimal
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Optimal %
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Change
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Change %
                </th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {channelDetails.map((ch, i) => (
                  <tr
                  key={ch.channel}
                    className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                  >
                  <td className="py-3 px-4 flex items-center gap-2 font-medium">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {ch.channel}
                  </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                    ${ch.cur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums text-slate-500">
                    {ch.curShare.toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                    ${ch.opt.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums text-slate-500">
                    {ch.optShare.toFixed(1)}%
                    </td>
                    <td
                      className={`text-right py-3 px-4 tabular-nums font-medium ${
                      ch.diff > 0
                          ? "text-emerald-600"
                        : ch.diff < 0
                            ? "text-red-500"
                            : "text-slate-500"
                      }`}
                    >
                    {ch.diff >= 0 ? "+" : ""}
                    ${ch.diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td
                      className={`text-right py-3 px-4 tabular-nums font-medium ${
                      ch.pct > 0
                          ? "text-emerald-600"
                        : ch.pct < 0
                            ? "text-red-500"
                            : "text-slate-500"
                      }`}
                    >
                    {ch.pct >= 0 ? "+" : ""}
                    {ch.pct.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4">
                    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                      ch.pct > 5
                        ? "bg-emerald-100 text-emerald-700"
                        : ch.pct < -5
                          ? "bg-red-100 text-red-700"
                          : "bg-slate-100 text-slate-600"
                    }`}>
                      {ch.pct > 5 ? "Increase" : ch.pct < -5 ? "Decrease" : "Maintain"}
                    </span>
                    </td>
                  </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
