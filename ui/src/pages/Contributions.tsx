import { Show, For } from "solid-js";
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
import ReactChart, { h } from "../lib/ReactChart";
import EmptyState from "../components/EmptyState";
import { COLORS } from "../lib/colors";
import { useContributionsQuery } from "../lib/queries";
import { downsampleEvenly } from "../lib/downsample";
import { formatCompactNumber, getDateAxisProps } from "../lib/chartFormat";

const RESERVED = new Set(["date", "actual", "predicted", "baseline"]);

export default function Contributions() {
  const query = useContributionsQuery();
  const data = () => query.data ?? null;

  const channels = () => {
    const d = data();
    if (!d?.data?.length) return [];
    return Object.keys(d.data[0]).filter((k) => !RESERVED.has(k));
  };

  const channelTotals = () =>
    channels()
      .map((ch) => ({
        channel: ch,
        total: data()!.data.reduce((s, r) => s + (Number(r[ch]) || 0), 0),
      }))
      .sort((a, b) => b.total - a.total);

  const allTotal = () =>
    channelTotals().reduce((s, c) => s + Math.abs(c.total), 0);

  const timeline = () => {
    const d = data();
    if (!d?.data?.length) return [];
    const chs = channels();
    return downsampleEvenly(d.data, 180).map((r) => ({
      date: String(r.date).slice(0, 10),
      ...Object.fromEntries(chs.map((ch) => [ch, Number(r[ch]) || 0])),
    }));
  };

  return (
    <Show
      when={!query.isLoading}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={data()?.data?.length}
        fallback={
          <EmptyState
            title="No contribution data"
            message="Run the pipeline to see how each channel contributes to your KPI over time."
            hideQuickStart
          />
        }
      >
        <div>
          <h1 class="text-2xl font-bold text-slate-900">
            Channel Contributions
          </h1>
          <p class="text-sm text-slate-500 mt-1">
            Breakdown of total response by marketing channel
          </p>

          {/* ---- Horizontal bar chart ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-semibold text-slate-700 mb-4">
              Total Contribution by Channel
            </h2>
            <ReactChart>
              {() =>
                h(
                  ResponsiveContainer,
                  {
                    width: "100%",
                    height: Math.max(200, channels().length * 52),
                  },
                  h(
                    BarChart,
                    {
                      data: channelTotals(),
                      layout: "vertical",
                      margin: { left: 80, right: 20 },
                    },
                    h(CartesianGrid, {
                      strokeDasharray: "3 3",
                      stroke: "#e2e8f0",
                      horizontal: false,
                    }),
                    h(XAxis, {
                      type: "number",
                      tick: { fontSize: 12 },
                      tickFormatter: (v: number) => formatCompactNumber(v),
                    }),
                    h(YAxis, {
                      type: "category",
                      dataKey: "channel",
                      tick: { fontSize: 13 },
                      width: 75,
                    }),
                    h(Tooltip, {
                      formatter: (v: number) =>
                        v.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        }),
                    }),
                    h(
                      Bar,
                      { dataKey: "total", radius: [0, 6, 6, 0] },
                      ...channelTotals().map((_, i) =>
                        h(Cell, { key: i, fill: COLORS[i % COLORS.length] }),
                      ),
                    ),
                  ),
                )
              }
            </ReactChart>
          </div>

          {/* ---- Stacked area timeline ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-semibold text-slate-700 mb-4">
              Contributions Over Time
            </h2>
            <ReactChart>
              {() =>
                h(
                  ResponsiveContainer,
                  { width: "100%", height: 350 },
                  h(
                    AreaChart,
                    { data: timeline() },
                    h(CartesianGrid, {
                      strokeDasharray: "3 3",
                      stroke: "#e2e8f0",
                    }),
                    h(XAxis, {
                      dataKey: "date",
                      ...getDateAxisProps(timeline().length),
                    }),
                    h(YAxis, {
                      tick: { fontSize: 11 },
                      tickFormatter: (v: number) => formatCompactNumber(v),
                    }),
                    h(Tooltip),
                    h(Legend),
                    ...channels().map((ch, i) =>
                      h(Area, {
                        key: ch,
                        type: "monotone",
                        dataKey: ch,
                        stackId: "1",
                        fill: COLORS[i % COLORS.length],
                        stroke: COLORS[i % COLORS.length],
                        fillOpacity: 0.7,
                      }),
                    ),
                  ),
                )
              }
            </ReactChart>
          </div>

          {/* ---- Summary table ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-semibold text-slate-700 mb-4">
              Channel Summary
            </h2>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-slate-200">
                    <th class="text-left py-3 px-4 font-semibold text-slate-600">
                      Channel
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Total Contribution
                    </th>
                    <th class="text-right py-3 px-4 font-semibold text-slate-600">
                      Share
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <For each={channelTotals()}>
                    {(ch, i) => (
                      <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                        <td class="py-3 px-4 flex items-center gap-2">
                          <span
                            class="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ background: COLORS[i() % COLORS.length] }}
                          />
                          {ch.channel}
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums">
                          {ch.total.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                        <td class="text-right py-3 px-4 tabular-nums">
                          {allTotal() > 0
                            ? `${((Math.abs(ch.total) / allTotal()) * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </Show>
    </Show>
  );
}
