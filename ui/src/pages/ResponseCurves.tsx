import { createSignal, onMount, Show } from "solid-js";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Cell,
  ZAxis,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import EmptyState from "../components/EmptyState";
import { ResponseCurvesSkeleton } from "../components/Skeleton";
import {
  api,
  type ResponseCurvesData,
  type ChannelInsightsData,
} from "../lib/api";
import { COLORS, channelColor } from "../lib/colors";
import { formatCompactNumber, formatSpendTick } from "../lib/chartFormat";
import { trackEvent } from "../lib/telemetry";

interface BubblePoint {
  channel: string;
  currentSpend: number;
  marginalRoi: number;
  contribution: number;
  color: string;
  quadrant: string;
}

function classifyQuadrant(
  spend: number,
  mRoi: number,
  medianSpend: number,
  medianRoi: number,
): string {
  if (mRoi >= medianRoi && spend < medianSpend) return "Grow";
  if (mRoi >= medianRoi && spend >= medianSpend) return "Maintain";
  if (mRoi < medianRoi && spend >= medianSpend) return "Reduce";
  return "Review";
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export default function ResponseCurves() {
  const [data, setData] = createSignal<ResponseCurvesData | null>(null);
  const [insights, setInsights] = createSignal<ChannelInsightsData | null>(
    null,
  );
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    Promise.all([
      api.responseCurves().catch(() => null),
      api.channelInsights().catch(() => null),
    ]).then(([curves, ins]) => {
      if (curves) setData(curves);
      if (ins) setInsights(ins);
      setLoading(false);
    });
  });

  const channels = () => (data() ? Object.keys(data()!) : []);

  const responseRows = () => {
    const d = data();
    if (!d) return [];
    const chs = Object.keys(d);
    const maxLen = Math.max(...chs.map((ch) => d[ch].spend?.length ?? 0));
    const step = Math.max(1, Math.floor(maxLen / 100));
    const rows: Record<string, number | string>[] = [];
    for (let i = 0; i < maxLen; i += step) {
      const row: Record<string, number | string> = {};
      let hasSpend = false;
      for (const ch of chs) {
        const sp = d[ch].spend?.[i];
        const re = d[ch].response?.[i];
        if (sp !== undefined) {
          row[`${ch}_spend`] = sp;
          row[ch] = re ?? 0;
          if (!hasSpend) {
            row.spend = sp;
            hasSpend = true;
          }
        }
      }
      if (hasSpend) rows.push(row);
    }
    return rows;
  };

  const bubbleData = (): BubblePoint[] => {
    const ins = insights();
    if (!ins?.channels?.length) return [];
    const spends = ins.channels.map((c) => c.current_spend);
    const rois = ins.channels.map((c) => c.marginal_roi);
    const medSpend = median(spends);
    const medRoi = median(rois);
    // Use current_spend * marginal_roi as a contribution proxy
    return ins.channels.map((c, i) => ({
      channel: c.channel,
      currentSpend: c.current_spend,
      marginalRoi: c.marginal_roi,
      contribution: c.current_spend * Math.max(c.marginal_roi, 0),
      color: COLORS[i % COLORS.length],
      quadrant: classifyQuadrant(
        c.current_spend,
        c.marginal_roi,
        medSpend,
        medRoi,
      ),
    }));
  };

  const bubbleMedians = () => {
    const pts = bubbleData();
    if (!pts.length) return { spend: 0, roi: 0 };
    return {
      spend: median(pts.map((p) => p.currentSpend)),
      roi: median(pts.map((p) => p.marginalRoi)),
    };
  };

  const hasMarginal = () =>
    channels().some(
      (ch) =>
        data()![ch].marginal_response &&
        data()![ch].marginal_response!.length > 0,
    );

  const marginalRows = () => {
    const d = data();
    if (!d || !hasMarginal()) return [];
    const chs = Object.keys(d);
    const maxLen = Math.max(...chs.map((ch) => d[ch].spend?.length ?? 0));
    const step = Math.max(1, Math.floor(maxLen / 100));
    const rows: Record<string, number | string>[] = [];
    for (let i = 0; i < maxLen; i += step) {
      const row: Record<string, number | string> = {};
      let hasSpend = false;
      for (const ch of chs) {
        const sp = d[ch].spend?.[i];
        const mr = d[ch].marginal_response?.[i];
        if (sp !== undefined && mr !== undefined) {
          row.spend = sp;
          row[ch] = mr;
          hasSpend = true;
        }
      }
      if (hasSpend) rows.push(row);
    }
    return rows;
  };

  return (
    <Show when={!loading()} fallback={<ResponseCurvesSkeleton />}>
      <Show
        when={data() && Object.keys(data()!).length > 0}
        fallback={
          <EmptyState
            title="No response curves yet"
            message="Run the pipeline to generate saturation curves — they show how each channel's returns diminish with more spend."
            hideQuickStart
          />
        }
      >
        <div>
          <h1 class="text-2xl font-bold text-slate-900">Response Curves</h1>
          <p class="text-sm text-slate-500 mt-1">
            Saturation curves showing diminishing returns per channel
          </p>

          {/* ---- Investment Efficiency Bubble Chart ---- */}
          <Show when={bubbleData().length > 0}>
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-medium text-slate-700 mb-1">
                Investment Efficiency
              </h2>
              <p class="text-xs text-slate-500 mb-4">
                Bubble size = estimated contribution. Quadrants show
                under/over-investment.
              </p>
              <ReactChart>
                {() => {
                  const pts = bubbleData();
                  const med = bubbleMedians();
                  return h(
                    ResponsiveContainer,
                    { width: "100%", height: 420 },
                    h(
                      ScatterChart,
                      {
                        margin: {
                          top: 30,
                          right: 40,
                          bottom: 20,
                          left: 20,
                        },
                        onClick: () =>
                          trackEvent("chart_interaction", {
                            chart_id: "investment_efficiency_bubble",
                            interaction: "click",
                          }),
                      },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: "#e2e8f0",
                      }),
                      h(XAxis, {
                        type: "number",
                        dataKey: "currentSpend",
                        name: "Current Spend",
                        tick: { fontSize: 12 },
                        tickFormatter: (v: number) => formatSpendTick(v),
                        label: {
                          value: "Current Spend",
                          position: "insideBottomRight",
                          offset: -5,
                          fontSize: 12,
                        },
                      }),
                      h(YAxis, {
                        type: "number",
                        dataKey: "marginalRoi",
                        name: "Marginal ROI",
                        tick: { fontSize: 12 },
                        tickFormatter: (v: number) => formatCompactNumber(v),
                        label: {
                          value: "Marginal ROI",
                          angle: -90,
                          position: "insideLeft",
                          fontSize: 12,
                        },
                      }),
                      h(ZAxis, {
                        type: "number",
                        dataKey: "contribution",
                        range: [120, 900],
                        name: "Contribution",
                      }),
                      h(Tooltip, {
                        content: ({ payload }: any) => {
                          if (!payload?.length) return null;
                          const p = payload[0].payload as BubblePoint;
                          return h(
                            "div",
                            {
                              className:
                                "bg-white border border-slate-200 rounded-lg shadow-lg p-3 text-sm",
                            },
                            h(
                              "p",
                              { className: "font-semibold text-slate-900" },
                              p.channel,
                            ),
                            h(
                              "p",
                              { className: "text-slate-600" },
                              `Spend: $${p.currentSpend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                            ),
                            h(
                              "p",
                              { className: "text-slate-600" },
                              `Marginal ROI: ${p.marginalRoi.toFixed(2)}`,
                            ),
                            h(
                              "p",
                              { className: "text-slate-600" },
                              `Contribution: $${p.contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                            ),
                            h(
                              "p",
                              {
                                className: `font-medium mt-1 ${
                                  p.quadrant === "Grow"
                                    ? "text-emerald-600"
                                    : p.quadrant === "Maintain"
                                      ? "text-indigo-600"
                                      : p.quadrant === "Reduce"
                                        ? "text-amber-600"
                                        : "text-slate-500"
                                }`,
                              },
                              p.quadrant,
                            ),
                          );
                        },
                      }),
                      // Quadrant dividers
                      h(ReferenceLine, {
                        x: med.spend,
                        stroke: "#94a3b8",
                        strokeDasharray: "4 4",
                      }),
                      h(ReferenceLine, {
                        y: med.roi,
                        stroke: "#94a3b8",
                        strokeDasharray: "4 4",
                      }),
                      h(
                        Scatter,
                        { data: pts, name: "Channels" },
                        ...pts.map((pt, i) =>
                          h(Cell, {
                            key: pt.channel,
                            fill: pt.color,
                            fillOpacity: 0.75,
                            stroke: pt.color,
                            strokeWidth: 1.5,
                          }),
                        ),
                      ),
                    ),
                  );
                }}
              </ReactChart>
              {/* Quadrant legend */}
              <div class="flex flex-wrap gap-4 mt-3 text-xs text-slate-600 justify-center">
                <span class="flex items-center gap-1">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  Top-left: Grow (high mROI, low spend)
                </span>
                <span class="flex items-center gap-1">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500" />
                  Top-right: Maintain
                </span>
                <span class="flex items-center gap-1">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-amber-500" />
                  Bottom-right: Reduce (saturated)
                </span>
                <span class="flex items-center gap-1">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-slate-400" />
                  Bottom-left: Review
                </span>
              </div>
            </div>
          </Show>

          {/* ---- Response curves ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-medium text-slate-700 mb-4">
              Response vs Spend (all channels)
            </h2>
            <ReactChart>
              {() =>
                h(
                  ResponsiveContainer,
                  { width: "100%", height: 400 },
                  h(
                    LineChart,
                    {
                      data: responseRows(),
                      onClick: () =>
                        trackEvent("chart_interaction", {
                          chart_id: "response_curves",
                          interaction: "click",
                        }),
                    },
                    h(CartesianGrid, {
                      strokeDasharray: "3 3",
                      stroke: "#e2e8f0",
                    }),
                    h(XAxis, {
                      dataKey: "spend",
                      tick: { fontSize: 12 },
                      tickFormatter: (v: any) =>
                        typeof v === "number" ? formatSpendTick(v) : String(v),
                      label: {
                        value: "Spend",
                        position: "insideBottomRight",
                        offset: -5,
                        fontSize: 12,
                      },
                    }),
                    h(YAxis, {
                      tick: { fontSize: 12 },
                      tickFormatter: (v: number) => formatCompactNumber(v),
                      label: {
                        value: "Response",
                        angle: -90,
                        position: "insideLeft",
                        fontSize: 12,
                      },
                    }),
                    h(Tooltip, {
                      formatter: (v: number) =>
                        v.toLocaleString(undefined, {
                          maximumFractionDigits: 2,
                        }),
                      labelFormatter: (v: any) =>
                        `Spend: $${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                    }),
                    h(Legend),
                    ...channels().map((ch, i) =>
                      h(Line, {
                        key: ch,
                        type: "monotone",
                        dataKey: ch,
                        stroke: channelColor(ch, i),
                        strokeWidth: 2.5,
                        dot: false,
                        name: ch,
                      }),
                    ),
                  ),
                )
              }
            </ReactChart>
          </div>

          {/* ---- Marginal response ---- */}
          <Show when={hasMarginal() && marginalRows().length > 0}>
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-medium text-slate-700 mb-4">
                Marginal Response (per additional dollar)
              </h2>
              <ReactChart>
                {() =>
                  h(
                    ResponsiveContainer,
                    { width: "100%", height: 350 },
                    h(
                      LineChart,
                      { data: marginalRows() },
                      h(CartesianGrid, {
                        strokeDasharray: "3 3",
                        stroke: "#e2e8f0",
                      }),
                      h(XAxis, {
                        dataKey: "spend",
                        tick: { fontSize: 12 },
                        tickFormatter: (v: any) =>
                          typeof v === "number"
                            ? formatSpendTick(v)
                            : String(v),
                      }),
                      h(YAxis, {
                        tick: { fontSize: 12 },
                        tickFormatter: (v: number) => formatCompactNumber(v),
                      }),
                      h(Tooltip),
                      h(Legend),
                      ...channels().map((ch, i) =>
                        h(Line, {
                          key: ch,
                          type: "monotone",
                          dataKey: ch,
                          stroke: channelColor(ch, i),
                          strokeWidth: 2,
                          dot: false,
                          name: ch,
                        }),
                      ),
                    ),
                  )
                }
              </ReactChart>
            </div>
          </Show>
        </div>
      </Show>
    </Show>
  );
}
