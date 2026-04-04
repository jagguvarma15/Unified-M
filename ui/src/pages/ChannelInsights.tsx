import { createSignal, onMount, Show, For } from "solid-js";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertTriangle, TrendingUp, TrendingDown, Minus, Loader2 } from "../lib/icons";
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
  const [data, setData] = createSignal<ChannelInsightsData | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .channelInsights()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  const channels = () => data()?.channels ?? [];

  const overSaturated = () => channels().filter((c) => c.status === "over-saturated");

  const marginalData = () =>
    channels().map((c) => ({
      channel: c.channel.replace(/_spend$/, ""),
      marginal_roi: c.marginal_roi,
      status: c.status,
    }));

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <Loader2 class="h-8 w-8 animate-spin text-indigo-600" />
        </div>
      }
    >
      <Show
        when={data()?.channels?.length}
        fallback={
          <EmptyState
            title="No channel insights yet"
            message="Run the pipeline and optimization to generate channel saturation and marginal ROI data."
            action={{ label: "Go to Data", href: "/data" }}
          />
        }
      >
        <div>
          <PageHeader
            title="Channel Insights"
            description="Saturation alerts and marginal ROI per channel"
            hint="Channels past diminishing returns are flagged"
          />

          {/* Alerts banner */}
          <Show when={overSaturated().length > 0}>
            <div class="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4">
              <div class="flex items-center gap-2 text-amber-700 font-medium text-sm mb-1">
                <AlertTriangle size={16} />
                {overSaturated().length} channel{overSaturated().length > 1 ? "s" : ""} over-saturated
              </div>
              <p class="text-xs text-amber-600">
                {overSaturated().map((c) => c.channel.replace(/_spend$/, "")).join(", ")} —
                reallocating budget may improve overall ROI.
              </p>
            </div>
          </Show>

          {/* Marginal ROI bar chart */}
          <ChartCard
            class="mb-6"
            title="Marginal ROI by Channel"
            description="Additional return from the next dollar of spend (sorted highest first)"
            minHeight={320}
          >
            <ResponsiveContainer width="100%" height={Math.max(280, marginalData().length * 44)}>
              <BarChart data={marginalData()} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v: number) => v.toFixed(3)} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 12 }} width={90} />
                <Tooltip
                  contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                  formatter={(v: number) => [v.toFixed(6), "Marginal ROI"]}
                />
                <Bar dataKey="marginal_roi" radius={[0, 4, 4, 0]}>
                  {marginalData().map((entry, i) => (
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
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <For each={channels()}>
              {(ch, i) => (
                <ChannelCard insight={ch} color={COLORS[i() % COLORS.length]} />
              )}
            </For>
          </div>
        </div>
      </Show>
    </Show>
  );
}

function ChannelCard(props: { insight: ChannelInsight; color: string }) {
  const variant = () => STATUS_VARIANT[props.insight.status] ?? "default";
  const StatusIcon = () => {
    const Ic = props.insight.status === "over-saturated"
      ? TrendingDown
      : props.insight.status === "under-invested"
        ? TrendingUp
        : Minus;
    return <Ic size={12} />;
  };
  const name = () => props.insight.channel.replace(/_spend$/, "");

  return (
    <div class="rounded-xl border border-slate-200/60 bg-white p-5 shadow-sm focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2">
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-semibold text-slate-800 text-sm">{name()}</h3>
        <Badge variant={variant() as any} icon={StatusIcon()}>
          {props.insight.status}
        </Badge>
      </div>

      <div class="grid grid-cols-2 gap-3 text-xs">
        <div>
          <p class="text-slate-500">Current Spend</p>
          <p class="font-semibold text-slate-800 tabular-nums">
            {formatCurrency(props.insight.current_spend)}
          </p>
        </div>
        <div>
          <p class="text-slate-500">Optimal Spend</p>
          <p class="font-semibold text-slate-800 tabular-nums">
            {formatCurrency(props.insight.optimal_spend)}
          </p>
        </div>
        <div>
          <p class="text-slate-500">Marginal ROI</p>
          <p class="font-semibold text-slate-800 tabular-nums">{props.insight.marginal_roi.toFixed(4)}</p>
        </div>
        <div>
          <p class="text-slate-500">Headroom</p>
          <p class="font-semibold text-slate-800 tabular-nums">{props.insight.headroom_pct}%</p>
        </div>
      </div>

      {/* Saturation progress bar */}
      <div class="mt-3">
        <div class="flex items-center justify-between text-[11px] text-slate-500 mb-1">
          <span>Saturation</span>
          <span>{Math.min(100, Math.round(100 - props.insight.headroom_pct))}%</span>
        </div>
        <div class="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all"
            style={{
              width: `${Math.min(100, Math.round(100 - props.insight.headroom_pct))}%`,
              "background-color": props.color,
            }}
          />
        </div>
      </div>
    </div>
  );
}
