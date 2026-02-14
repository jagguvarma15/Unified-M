import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Loader2, ArrowUp, ArrowDown, Minus } from "lucide-react";
import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";
import MetricCard from "../components/MetricCard";
import { DollarSign, TrendingUp, Target } from "lucide-react";
import { api, type SpendPacingData } from "../lib/api";
import { CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";

const STATUS_BADGE: Record<string, { cls: string; label: string }> = {
  "on-track": { cls: "bg-emerald-50 text-emerald-700 border-emerald-200", label: "On Track" },
  over: { cls: "bg-amber-50 text-amber-700 border-amber-200", label: "Over" },
  under: { cls: "bg-red-50 text-red-700 border-red-200", label: "Under" },
};

export default function SpendPacing() {
  const [data, setData] = useState<SpendPacingData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .spendPacing()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!data || !data.channels?.length) {
    return (
      <EmptyState
        title="No pacing data"
        message="Run the optimizer first so planned allocations are available for comparison."
      />
    );
  }

  const offPace = data.channels.filter((c) => c.status !== "on-track");
  const pacingColor = data.pacing_pct > 115 ? "red" : data.pacing_pct < 85 ? "amber" : "emerald";

  return (
    <div>
      <PageHeader
        title="Spend Pacing"
        description="Budget vs actual spend tracker — compare optimal plan with real spend"
      />

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <MetricCard
          label="Total Planned"
          value={`$${data.total_planned.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          icon={Target}
          color="indigo"
          tooltip="Sum of optimal allocation from the latest optimization"
        />
        <MetricCard
          label="Total Actual"
          value={`$${data.total_actual.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          icon={DollarSign}
          color="emerald"
          tooltip="Sum of actual spend recorded in media_spend data"
        />
        <MetricCard
          label="Pacing"
          value={`${data.pacing_pct}%`}
          icon={TrendingUp}
          color={pacingColor as "indigo" | "emerald" | "amber" | "red"}
          delta={offPace.length > 0 ? `${offPace.length} channel${offPace.length > 1 ? "s" : ""} off-pace` : "All channels on track"}
          tooltip="Actual / Planned as a percentage — 100% means perfectly on-pace"
        />
      </div>

      {/* Cumulative spend chart */}
      {data.cumulative && data.cumulative.length > 1 && (
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm mb-6">
          <h2 className="text-base font-semibold text-slate-800 mb-1">Cumulative Spend Over Time</h2>
          <p className="text-xs text-slate-500 mb-4">Actual cumulative media spend trajectory</p>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={data.cumulative} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <defs>
                <linearGradient id="spendGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                formatter={(v: number) => [`$${v.toLocaleString()}`, "Cumulative Spend"]}
              />
              <Area type="monotone" dataKey="actual" stroke="#6366f1" fill="url(#spendGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Channel pacing table */}
      <div className="rounded-xl border border-slate-200/60 bg-white shadow-sm overflow-hidden">
        <div className="p-5 border-b border-slate-100">
          <h2 className="text-base font-semibold text-slate-800">Channel Pacing Details</h2>
          <p className="text-xs text-slate-500 mt-0.5">Channels &gt;15% off-pace are flagged</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                <th className="text-left py-3 px-4 font-semibold text-slate-600">Channel</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Planned</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Actual</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Diff</th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">Pacing</th>
                <th className="text-center py-3 px-4 font-semibold text-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.channels.map((ch) => {
                const badge = STATUS_BADGE[ch.status] || STATUS_BADGE["on-track"];
                const DirIcon = ch.diff > 0 ? ArrowUp : ch.diff < 0 ? ArrowDown : Minus;
                return (
                  <tr key={ch.channel} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                    <td className="py-3 px-4 font-medium text-slate-800">
                      {ch.channel.replace(/_spend$/, "")}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      ${ch.planned.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      ${ch.actual.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums">
                      <span className={`inline-flex items-center gap-0.5 ${ch.diff > 0 ? "text-amber-600" : ch.diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                        <DirIcon size={12} />
                        ${Math.abs(ch.diff).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </span>
                    </td>
                    <td className="text-right py-3 px-4 tabular-nums font-medium">
                      {ch.pacing_pct}%
                    </td>
                    <td className="text-center py-3 px-4">
                      <span className={`inline-flex items-center text-xs font-medium px-2.5 py-0.5 rounded-full border ${badge.cls}`}>
                        {badge.label}
                      </span>
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
