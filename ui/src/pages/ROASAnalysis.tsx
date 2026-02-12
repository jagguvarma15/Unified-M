import { useEffect, useState } from "react";
import { DollarSign, TrendingUp, BarChart2 } from "lucide-react";
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
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type ROASData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function ROASAnalysis() {
  const [data, setData] = useState<ROASData | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState<"roas" | "contribution" | "spend" | "cpa">("roas");

  useEffect(() => {
    api
      .roas()
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

  if (!data || data.channels.length === 0) return <EmptyState />;

  const sorted = [...data.channels].sort((a, b) => {
    switch (sortBy) {
      case "roas": return b.roas - a.roas;
      case "contribution": return b.total_contribution - a.total_contribution;
      case "spend": return b.total_spend - a.total_spend;
      case "cpa": return (a.cpa ?? 0) - (b.cpa ?? 0);
      default: return 0;
    }
  });

  // Radar chart data (normalize each metric to 0-100)
  const maxRoas = Math.max(...data.channels.map((c) => c.roas));
  const maxContrib = Math.max(...data.channels.map((c) => c.total_contribution));
  const maxSpend = Math.max(...data.channels.map((c) => c.total_spend));
  const maxMROI = Math.max(...data.channels.map((c) => Math.abs(c.marginal_roi ?? 0)));

  const radarData = data.channels.map((ch) => ({
    channel: ch.channel,
    ROAS: maxRoas > 0 ? (ch.roas / maxRoas) * 100 : 0,
    Contribution: maxContrib > 0 ? (ch.total_contribution / maxContrib) * 100 : 0,
    Spend: maxSpend > 0 ? (ch.total_spend / maxSpend) * 100 : 0,
    "Marginal ROI": maxMROI > 0 ? (Math.abs(ch.marginal_roi ?? 0) / maxMROI) * 100 : 0,
  }));

  // Efficiency quadrant data
  const efficiency = data.channels.map((ch, i) => ({
    ...ch,
    color: COLORS[i % COLORS.length],
  }));

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">ROAS & ROI Analysis</h1>
      <p className="text-sm text-slate-500 mt-1">
        Return on ad spend and efficiency metrics across all channels
      </p>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
        <MetricCard
          label="Blended ROAS"
          value={`${data.summary.blended_roas.toFixed(2)}x`}
          icon={TrendingUp}
          color="indigo"
        />
        <MetricCard
          label="Total Spend"
          value={`$${data.summary.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          icon={DollarSign}
          color="emerald"
        />
        <MetricCard
          label="Total Contribution"
          value={data.summary.total_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          icon={BarChart2}
          color="amber"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* ROAS by channel bar */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            ROAS by Channel
          </h2>
          <ResponsiveContainer width="100%" height={Math.max(200, data.channels.length * 48)}>
            <BarChart data={sorted} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v) => `${v.toFixed(1)}x`} />
              <YAxis type="category" dataKey="channel" tick={{ fontSize: 13 }} width={75} />
              <Tooltip formatter={(v: number) => `${v.toFixed(2)}x`} />
              <Bar dataKey="roas" radius={[0, 6, 6, 0]} name="ROAS">
                {sorted.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Channel efficiency radar */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Channel Efficiency Radar
          </h2>
          <ResponsiveContainer width="100%" height={Math.max(300, data.channels.length * 48)}>
            <RadarChart data={[
              { metric: "ROAS", ...Object.fromEntries(radarData.map((r) => [r.channel, r.ROAS])) },
              { metric: "Contribution", ...Object.fromEntries(radarData.map((r) => [r.channel, r.Contribution])) },
              { metric: "Spend Share", ...Object.fromEntries(radarData.map((r) => [r.channel, r.Spend])) },
              { metric: "Marginal ROI", ...Object.fromEntries(radarData.map((r) => [r.channel, r["Marginal ROI"]])) },
            ]}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
              {data.channels.map((ch, i) => (
                <Radar
                  key={ch.channel}
                  name={ch.channel}
                  dataKey={ch.channel}
                  stroke={COLORS[i % COLORS.length]}
                  fill={COLORS[i % COLORS.length]}
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              ))}
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Spend vs Contribution comparison */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Spend vs Contribution by Channel
        </h2>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={sorted}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="channel" tick={{ fontSize: 13 }} />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip
              formatter={(v: number) =>
                `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
              }
            />
            <Legend />
            <Bar dataKey="total_spend" name="Total Spend" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="total_contribution" name="Total Contribution" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed table */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-slate-700">
            Channel Performance Table
          </h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="text-xs border border-slate-300 rounded-lg px-2 py-1.5 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            >
              <option value="roas">ROAS</option>
              <option value="contribution">Contribution</option>
              <option value="spend">Spend</option>
              <option value="cpa">CPA</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Spend</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Contribution</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">ROAS</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Marginal ROI</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">CPA</th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Efficiency</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((ch, i) => (
                <tr key={ch.channel} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td className="py-3 px-4 flex items-center gap-2 font-medium">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    {ch.channel}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    ${ch.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    {ch.total_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className={`text-right py-3 px-4 tabular-nums font-medium ${
                    ch.roas >= data.summary.blended_roas ? "text-emerald-600" : "text-amber-600"
                  }`}>
                    {ch.roas.toFixed(2)}x
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    {(ch.marginal_roi ?? 0).toFixed(4)}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    ${(ch.cpa ?? 0).toFixed(2)}
                  </td>
                  <td className="py-3 px-4">
                    <EfficiencyBar value={ch.roas} max={maxRoas} />
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

function EfficiencyBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  const color = pct > 70 ? "bg-emerald-500" : pct > 40 ? "bg-amber-500" : "bg-red-400";
  return (
    <div className="w-24 bg-slate-100 rounded-full h-2">
      <div
        className={`h-2 rounded-full ${color} transition-all`}
        style={{ width: `${Math.min(100, pct)}%` }}
      />
    </div>
  );
}
