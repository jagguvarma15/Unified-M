import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertTriangle, TrendingUp, TrendingDown, Minus, Loader2 } from "lucide-react";
import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";
import Badge from "../components/Badge";
import ChartCard from "../components/ChartCard";
import { api, type ChannelInsightsData, type ChannelInsight } from "../lib/api";
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";
import { formatCurrency } from "../lib/format";

const STATUS_VARIANT: Record<string, "info" | "success" | "warning"> = {
  "under-invested": "info",
  efficient: "success",
  "over-saturated": "warning",
};

export default function ChannelInsights() {
  const [data, setData] = useState<ChannelInsightsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .channelInsights()
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

  if (!data?.channels?.length) {
    return (
      <EmptyState
        title="No channel insights yet"
        message="Run the pipeline and optimization to generate channel saturation and marginal ROI data."
        action={{ label: "Go to Data", href: "/data" }}
      />
    );
  }

  const channels = data.channels;
  const overSaturated = channels.filter((c) => c.status === "over-saturated");
  const marginalData = channels.map((c) => ({
    channel: c.channel.replace(/_spend$/, ""),
    marginal_roi: c.marginal_roi,
    status: c.status,
  }));

  return (
    <div>
      <PageHeader
        title="Channel Insights"
        description="Saturation alerts and marginal ROI per channel"
        hint="Channels past diminishing returns are flagged"
      />

      {/* Alerts banner */}
      {overSaturated.length > 0 && (
        <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-center gap-2 text-amber-700 font-medium text-sm mb-1">
            <AlertTriangle size={16} />
            {overSaturated.length} channel{overSaturated.length > 1 ? "s" : ""} over-saturated
          </div>
          <p className="text-xs text-amber-600">
            {overSaturated.map((c) => c.channel.replace(/_spend$/, "")).join(", ")} —
            reallocating budget may improve overall ROI.
          </p>
        </div>
      )}

      {/* Marginal ROI bar chart */}
      <ChartCard
        className="mb-6"
        title="Marginal ROI by Channel"
        description="Additional return from the next dollar of spend (sorted highest first)"
        minHeight={320}
      >
        <ResponsiveContainer width="100%" height={Math.max(280, marginalData.length * 44)}>
          <BarChart data={marginalData} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v: number) => v.toFixed(3)} />
            <YAxis type="category" dataKey="channel" tick={{ fontSize: 12 }} width={90} />
            <Tooltip
              contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
              formatter={(v: number) => [v.toFixed(6), "Marginal ROI"]}
            />
            <Bar dataKey="marginal_roi" radius={[0, 4, 4, 0]}>
              {marginalData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    entry.status === "over-saturated"
                      ? "#f59e0b"
                      : entry.status === "under-invested"
                        ? "#6366f1"
                        : "#10b981"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Per-channel cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {channels.map((ch, i) => (
          <ChannelCard key={ch.channel} insight={ch} color={COLORS[i % COLORS.length]} />
        ))}
      </div>
    </div>
  );
}

function ChannelCard({ insight, color }: { insight: ChannelInsight; color: string }) {
  const variant = STATUS_VARIANT[insight.status] ?? "default";
  const StatusIcon =
    insight.status === "over-saturated"
      ? TrendingDown
      : insight.status === "under-invested"
        ? TrendingUp
        : Minus;
  const name = insight.channel.replace(/_spend$/, "");

  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-sm focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-slate-800 text-sm">{name}</h3>
        <Badge variant={variant} icon={<StatusIcon size={12} />}>
          {insight.status}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <p className="text-slate-500">Current Spend</p>
          <p className="font-semibold text-slate-800 tabular-nums">
            {formatCurrency(insight.current_spend)}
          </p>
        </div>
        <div>
          <p className="text-slate-500">Optimal Spend</p>
          <p className="font-semibold text-slate-800 tabular-nums">
            {formatCurrency(insight.optimal_spend)}
          </p>
        </div>
        <div>
          <p className="text-slate-500">Marginal ROI</p>
          <p className="font-semibold text-slate-800 tabular-nums">{insight.marginal_roi.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-slate-500">Headroom</p>
          <p className="font-semibold text-slate-800 tabular-nums">{insight.headroom_pct}%</p>
        </div>
      </div>

      {/* Saturation progress bar */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
          <span>Saturation</span>
          <span>{Math.min(100, Math.round(100 - insight.headroom_pct))}%</span>
        </div>
        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.min(100, Math.round(100 - insight.headroom_pct))}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>
    </div>
  );
}
