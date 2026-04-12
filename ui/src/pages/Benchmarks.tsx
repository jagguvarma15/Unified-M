import { createSignal, For, Show } from "solid-js";
import PageHeader from "../components/PageHeader";
import MetricCard from "../components/MetricCard";
import { Award, TrendingUp, BarChart3, Info } from "../lib/icons";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";

interface BenchmarkRow {
  channel: string;
  label: string;
  yourROAS: number;
  industryMedian: number;
  topQuartile: number;
  percentile: number; // your position vs peers
}

const BENCHMARK_DATA: BenchmarkRow[] = [
  {
    channel: "google_ads",
    label: "Google Ads",
    yourROAS: 3.1,
    industryMedian: 2.4,
    topQuartile: 4.2,
    percentile: 72,
  },
  {
    channel: "meta_ads",
    label: "Meta Ads",
    yourROAS: 1.9,
    industryMedian: 2.0,
    topQuartile: 3.5,
    percentile: 44,
  },
  {
    channel: "tiktok_ads",
    label: "TikTok Ads",
    yourROAS: 2.4,
    industryMedian: 1.8,
    topQuartile: 3.1,
    percentile: 78,
  },
  {
    channel: "linkedin_ads",
    label: "LinkedIn Ads",
    yourROAS: 1.2,
    industryMedian: 1.5,
    topQuartile: 2.8,
    percentile: 31,
  },
  {
    channel: "pinterest_ads",
    label: "Pinterest Ads",
    yourROAS: 2.7,
    industryMedian: 2.1,
    topQuartile: 3.6,
    percentile: 68,
  },
  {
    channel: "snapchat_ads",
    label: "Snapchat Ads",
    yourROAS: 1.6,
    industryMedian: 1.4,
    topQuartile: 2.4,
    percentile: 56,
  },
];

type ViewMode = "chart" | "table";

function percentileColor(p: number) {
  if (p >= 75) return "text-emerald-600 bg-emerald-50";
  if (p >= 50) return "text-indigo-600 bg-indigo-50";
  if (p >= 25) return "text-amber-600 bg-amber-50";
  return "text-red-600 bg-red-50";
}

function roasVsMedian(row: BenchmarkRow) {
  const diff = ((row.yourROAS - row.industryMedian) / row.industryMedian) * 100;
  return diff;
}

export default function Benchmarks() {
  const [view, setView] = createSignal<ViewMode>("chart");
  const [selectedChannel, setSelectedChannel] = createSignal<string>("all");

  const aboveMedian = () =>
    BENCHMARK_DATA.filter((r) => r.yourROAS >= r.industryMedian).length;
  const avgPercentile = () =>
    Math.round(
      BENCHMARK_DATA.reduce((s, r) => s + r.percentile, 0) /
        BENCHMARK_DATA.length,
    );
  const topChannel = () =>
    BENCHMARK_DATA.reduce((best, r) =>
      r.percentile > best.percentile ? r : best,
    );

  const chartData = () =>
    BENCHMARK_DATA.map((r) => ({
      name: r.label.replace(" Ads", ""),
      "Your ROAS": r.yourROAS,
      "Industry Median": r.industryMedian,
      "Top Quartile": r.topQuartile,
    }));

  return (
    <div>
      <div class="flex items-center justify-between">
        <PageHeader
          title="Benchmarks"
          description="Compare your channel ROAS against industry peers. Benchmarks are based on aggregated anonymized data from similar advertisers."
        />
        <div class="flex items-center gap-2">
          <div
            class="flex rounded-lg overflow-hidden border border-slate-200 bg-white"
            role="group"
          >
            <button
              onClick={() => setView("chart")}
              class={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium transition-colors ${
                view() === "chart"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <BarChart3 size={14} />
              Chart
            </button>
            <button
              onClick={() => setView("table")}
              class={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium transition-colors border-l border-slate-200 ${
                view() === "table"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              Table
            </button>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div class="mb-4 flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-xs text-slate-500">
        <Info size={14} class="mt-0.5 shrink-0 text-slate-400" />
        <span>
          Benchmarks use synthetic industry-wide aggregates. Your actual
          competitive position may vary by vertical, audience, and creative
          strategy. Data is refreshed monthly.
        </span>
      </div>

      {/* Summary KPIs */}
      <div class="grid grid-cols-3 gap-3 mb-6">
        <MetricCard
          icon={Award}
          label="Avg. Peer Percentile"
          value={`${avgPercentile()}th`}
          color={avgPercentile() >= 50 ? "emerald" : "amber"}
        />
        <MetricCard
          icon={TrendingUp}
          label="Channels Above Median"
          value={`${aboveMedian()} / ${BENCHMARK_DATA.length}`}
          color={aboveMedian() >= 4 ? "emerald" : "amber"}
        />
        <MetricCard
          label="Top Performing Channel"
          value={topChannel().label}
          color="indigo"
        />
      </div>

      {/* Chart View */}
      <Show when={view() === "chart"}>
        <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <h2 class="text-base font-semibold text-slate-900 mb-1">
            ROAS by Channel vs. Industry Benchmarks
          </h2>
          <p class="text-xs text-slate-500 mb-5">
            Your ROAS (indigo) vs. industry median (slate) and top quartile
            (emerald). Taller indigo bars relative to median = stronger
            performance.
          </p>
          <ReactChart>
            {() =>
              h(
                ResponsiveContainer,
                { width: "100%", height: 380 },
                h(
                  BarChart,
                  {
                    data: chartData(),
                    margin: { top: 10, right: 20, bottom: 5, left: 0 },
                  },
                  h(CartesianGrid, {
                    strokeDasharray: "3 3",
                    stroke: "#e2e8f0",
                  }),
                  h(XAxis, {
                    dataKey: "name",
                    tick: { fontSize: 12 },
                  }),
                  h(YAxis, {
                    tickFormatter: (v: number) => `${v.toFixed(1)}x`,
                    tick: { fontSize: 11 },
                  }),
                  h(Tooltip, {
                    formatter: (value: number, name: string) => [
                      `${value.toFixed(2)}x`,
                      name,
                    ],
                  }),
                  h(Legend, { wrapperStyle: { fontSize: 12 } }),
                  h(Bar, {
                    dataKey: "Your ROAS",
                    fill: "#6366f1",
                    radius: [3, 3, 0, 0],
                  }),
                  h(Bar, {
                    dataKey: "Industry Median",
                    fill: "#94a3b8",
                    radius: [3, 3, 0, 0],
                  }),
                  h(Bar, {
                    dataKey: "Top Quartile",
                    fill: "#10b981",
                    radius: [3, 3, 0, 0],
                  }),
                ),
              )
            }
          </ReactChart>
        </div>

        {/* Per-channel cards */}
        <div class="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3">
          <For each={BENCHMARK_DATA}>
            {(row) => {
              const diff = roasVsMedian(row);
              return (
                <div class="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                  <div class="flex items-center justify-between mb-3">
                    <span class="text-sm font-semibold text-slate-900">
                      {row.label}
                    </span>
                    <span
                      class={`rounded-full px-2 py-0.5 text-[11px] font-bold ${percentileColor(row.percentile)}`}
                    >
                      {row.percentile}th pct
                    </span>
                  </div>
                  <div class="space-y-1.5 text-sm">
                    <div class="flex justify-between">
                      <span class="text-slate-500">Your ROAS</span>
                      <span class="font-semibold text-indigo-600 tabular-nums">
                        {row.yourROAS.toFixed(2)}x
                      </span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-slate-500">Industry Median</span>
                      <span class="tabular-nums text-slate-700">
                        {row.industryMedian.toFixed(2)}x
                      </span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-slate-500">Top Quartile</span>
                      <span class="tabular-nums text-emerald-600">
                        {row.topQuartile.toFixed(2)}x
                      </span>
                    </div>
                  </div>
                  <div class="mt-3">
                    <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                      <span>vs. Median</span>
                      <span
                        class={diff >= 0 ? "text-emerald-600" : "text-red-500"}
                      >
                        {diff >= 0 ? "+" : ""}
                        {diff.toFixed(1)}%
                      </span>
                    </div>
                    {/* Percentile bar */}
                    <div class="h-1.5 w-full rounded-full bg-slate-100">
                      <div
                        class="h-1.5 rounded-full transition-all"
                        style={{
                          width: `${row.percentile}%`,
                          background:
                            row.percentile >= 75
                              ? "#10b981"
                              : row.percentile >= 50
                                ? "#6366f1"
                                : row.percentile >= 25
                                  ? "#f59e0b"
                                  : "#ef4444",
                        }}
                      />
                    </div>
                  </div>
                </div>
              );
            }}
          </For>
        </div>
      </Show>

      {/* Table View */}
      <Show when={view() === "table"}>
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table class="min-w-full divide-y divide-slate-200">
            <thead class="bg-slate-50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Channel
                </th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Your ROAS
                </th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Industry Median
                </th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Top Quartile
                </th>
                <th class="px-4 py-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  vs. Median
                </th>
                <th class="px-4 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Peer Percentile
                </th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100">
              <For each={BENCHMARK_DATA}>
                {(row) => {
                  const diff = roasVsMedian(row);
                  return (
                    <tr class="hover:bg-slate-50 transition-colors">
                      <td class="px-4 py-3 text-sm font-medium text-slate-900">
                        {row.label}
                      </td>
                      <td class="px-4 py-3 text-sm text-right font-semibold text-indigo-600 tabular-nums">
                        {row.yourROAS.toFixed(2)}x
                      </td>
                      <td class="px-4 py-3 text-sm text-right text-slate-600 tabular-nums">
                        {row.industryMedian.toFixed(2)}x
                      </td>
                      <td class="px-4 py-3 text-sm text-right text-emerald-600 tabular-nums">
                        {row.topQuartile.toFixed(2)}x
                      </td>
                      <td
                        class={`px-4 py-3 text-sm text-right tabular-nums font-medium ${
                          diff >= 0 ? "text-emerald-600" : "text-red-500"
                        }`}
                      >
                        {diff >= 0 ? "+" : ""}
                        {diff.toFixed(1)}%
                      </td>
                      <td class="px-4 py-3 text-center">
                        <span
                          class={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${percentileColor(row.percentile)}`}
                        >
                          {row.percentile}th
                        </span>
                      </td>
                    </tr>
                  );
                }}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </div>
  );
}
