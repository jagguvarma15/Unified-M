import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import EmptyState from "../components/EmptyState";
import { api, type ContributionsData } from "../lib/api";
import { COLORS } from "../lib/colors";

const RESERVED = new Set(["date", "actual", "predicted", "baseline"]);

export default function Contributions() {
  const [data, setData] = useState<ContributionsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .contributions()
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

  if (!data?.data?.length) return <EmptyState />;

  const channels = Object.keys(data.data[0]).filter((k) => !RESERVED.has(k));

  // Total per channel (sorted)
  const channelTotals = channels
    .map((ch) => ({
      channel: ch,
      total: data.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0),
    }))
    .sort((a, b) => b.total - a.total);

  const allTotal = channelTotals.reduce((s, c) => s + Math.abs(c.total), 0);

  // Timeline (downsampled)
  const step = Math.max(1, Math.floor(data.data.length / 120));
  const timeline = data.data
    .filter((_, i) => i % step === 0)
    .map((r) => ({
      date: String(r.date).slice(0, 10),
      ...Object.fromEntries(channels.map((ch) => [ch, Number(r[ch]) || 0])),
    }));

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">
        Channel Contributions
      </h1>
      <p className="text-sm text-slate-500 mt-1">
        Breakdown of total response by marketing channel
      </p>

      {/* ---- Horizontal bar chart ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Total Contribution by Channel
        </h2>
        <ResponsiveContainer width="100%" height={Math.max(200, channels.length * 52)}>
          <BarChart
            data={channelTotals}
            layout="vertical"
            margin={{ left: 80, right: 20 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e2e8f0"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fontSize: 12 }}
              tickFormatter={(v) => v.toLocaleString()}
            />
            <YAxis
              type="category"
              dataKey="channel"
              tick={{ fontSize: 13 }}
              width={75}
            />
            <Tooltip
              formatter={(v: number) =>
                v.toLocaleString(undefined, { maximumFractionDigits: 0 })
              }
            />
            <Bar dataKey="total" radius={[0, 6, 6, 0]}>
              {channelTotals.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Stacked area timeline ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Contributions Over Time
        </h2>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            {channels.map((ch, i) => (
              <Area
                key={ch}
                type="monotone"
                dataKey={ch}
                stackId="1"
                fill={COLORS[i % COLORS.length]}
                stroke={COLORS[i % COLORS.length]}
                fillOpacity={0.7}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Summary table ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Channel Summary
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Channel
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Total Contribution
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Share
                </th>
              </tr>
            </thead>
            <tbody>
              {channelTotals.map((ch, i) => (
                <tr
                  key={ch.channel}
                  className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                >
                  <td className="py-3 px-4 flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ background: COLORS[i % COLORS.length] }}
                    />
                    {ch.channel}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    {ch.total.toLocaleString(undefined, {
                      maximumFractionDigits: 0,
                    })}
                  </td>
                  <td className="text-right py-3 px-4 tabular-nums">
                    {allTotal > 0
                      ? `${((Math.abs(ch.total) / allTotal) * 100).toFixed(1)}%`
                      : "—"}
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
