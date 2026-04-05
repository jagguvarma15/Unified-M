import { createSignal, onMount, Show, For } from "solid-js";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import {
  Loader2,
  ArrowUp,
  ArrowDown,
  Minus,
  DollarSign,
  TrendingUp,
  Target,
} from "../lib/icons";
import EmptyState from "../components/EmptyState";
import PageHeader from "../components/PageHeader";
import MetricCard from "../components/MetricCard";
import Badge from "../components/Badge";
import ChartCard from "../components/ChartCard";
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableHeaderCell,
  TableCell,
} from "../components/Table";
import { api, type SpendPacingData } from "../lib/api";
import { CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";
import { formatCurrency } from "../lib/format";

const PACING_VARIANT: Record<string, "success" | "warning" | "error"> = {
  "on-track": "success",
  over: "warning",
  under: "error",
};

export default function SpendPacing() {
  const [data, setData] = createSignal<SpendPacingData | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .spendPacing()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  const offPace = () =>
    data()?.channels?.filter((c) => c.status !== "on-track") ?? [];
  const pacingColor = () => {
    const d = data();
    if (!d) return "emerald";
    return d.pacing_pct > 115 ? "red" : d.pacing_pct < 85 ? "amber" : "emerald";
  };

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
        when={data() && data()!.channels?.length}
        fallback={
          <EmptyState
            title="No pacing data"
            message="Run the optimizer first so planned allocations are available for comparison."
            action={{ label: "Go to Optimizer", href: "/optimization" }}
          />
        }
      >
        {() => (
          <div>
            <PageHeader
              title="Spend Pacing"
              description="Budget vs actual spend tracker — compare optimal plan with real spend"
            />

            {/* Summary cards */}
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <MetricCard
                label="Total Planned"
                value={formatCurrency(data()!.total_planned)}
                icon={Target}
                color="indigo"
                tooltip="Sum of optimal allocation from the latest optimization"
              />
              <MetricCard
                label="Total Actual"
                value={formatCurrency(data()!.total_actual)}
                icon={DollarSign}
                color="emerald"
                tooltip="Sum of actual spend recorded in media_spend data"
              />
              <MetricCard
                label="Pacing"
                value={`${data()!.pacing_pct}%`}
                icon={TrendingUp}
                color={pacingColor() as "indigo" | "emerald" | "amber" | "red"}
                delta={
                  offPace().length > 0
                    ? `${offPace().length} channel${offPace().length > 1 ? "s" : ""} off-pace`
                    : "All channels on track"
                }
                tooltip="Actual / Planned as a percentage — 100% means perfectly on-pace"
              />
            </div>

            {/* Cumulative spend chart */}
            <Show when={data()!.cumulative && data()!.cumulative!.length > 1}>
              <ChartCard
                class="mb-6"
                title="Cumulative Spend Over Time"
                description="Actual cumulative media spend trajectory"
                minHeight={320}
              >
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      { width: "100%", height: 260 },
                      h(
                        AreaChart,
                        {
                          data: data()!.cumulative,
                          margin: { left: 10, right: 10, top: 5, bottom: 5 },
                        },
                        h(
                          "defs",
                          null,
                          h(
                            "linearGradient",
                            {
                              id: "spendGrad",
                              x1: "0",
                              y1: "0",
                              x2: "0",
                              y2: "1",
                            },
                            h("stop", {
                              offset: "5%",
                              stopColor: "#6366f1",
                              stopOpacity: 0.15,
                            }),
                            h("stop", {
                              offset: "95%",
                              stopColor: "#6366f1",
                              stopOpacity: 0,
                            }),
                          ),
                        ),
                        h(CartesianGrid, {
                          strokeDasharray: "3 3",
                          stroke: CHART_GRID,
                        }),
                        h(XAxis, { dataKey: "date", tick: { fontSize: 11 } }),
                        h(YAxis, {
                          tick: { fontSize: 11 },
                          tickFormatter: (v: number) =>
                            `$${(v / 1000).toFixed(0)}k`,
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
                            `$${v.toLocaleString()}`,
                            "Cumulative Spend",
                          ],
                        }),
                        h(Area, {
                          type: "monotone",
                          dataKey: "actual",
                          stroke: "#6366f1",
                          fill: "url(#spendGrad)",
                          strokeWidth: 2,
                        }),
                      ),
                    )
                  }
                </ReactChart>
              </ChartCard>
            </Show>

            {/* Channel pacing table */}
            <ChartCard
              title="Channel Pacing Details"
              description="Channels >15% off-pace are flagged"
              minHeight={200}
            >
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Channel</TableHeaderCell>
                    <TableHeaderCell align="right">Planned</TableHeaderCell>
                    <TableHeaderCell align="right">Actual</TableHeaderCell>
                    <TableHeaderCell align="right">Diff</TableHeaderCell>
                    <TableHeaderCell align="right">Pacing</TableHeaderCell>
                    <TableHeaderCell align="center">Status</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <For each={data()!.channels}>
                    {(ch) => {
                      const variant = PACING_VARIANT[ch.status] ?? "default";
                      const label =
                        ch.status === "on-track"
                          ? "On Track"
                          : ch.status === "over"
                            ? "Over"
                            : "Under";
                      const DirIcon =
                        ch.diff > 0 ? ArrowUp : ch.diff < 0 ? ArrowDown : Minus;
                      return (
                        <TableRow>
                          <TableCell class="font-medium text-slate-800">
                            {ch.channel.replace(/_spend$/, "")}
                          </TableCell>
                          <TableCell align="right" class="tabular-nums">
                            {formatCurrency(ch.planned)}
                          </TableCell>
                          <TableCell align="right" class="tabular-nums">
                            {formatCurrency(ch.actual)}
                          </TableCell>
                          <TableCell align="right" class="tabular-nums">
                            <span
                              class={`inline-flex items-center gap-0.5 ${ch.diff > 0 ? "text-amber-600" : ch.diff < 0 ? "text-red-600" : "text-slate-500"}`}
                            >
                              <DirIcon size={12} />
                              {formatCurrency(Math.abs(ch.diff))}
                            </span>
                          </TableCell>
                          <TableCell
                            align="right"
                            class="tabular-nums font-medium"
                          >
                            {ch.pacing_pct}%
                          </TableCell>
                          <TableCell align="center">
                            <Badge variant={variant as any}>{label}</Badge>
                          </TableCell>
                        </TableRow>
                      );
                    }}
                  </For>
                </TableBody>
              </Table>
            </ChartCard>
          </div>
        )}
      </Show>
    </Show>
  );
}
