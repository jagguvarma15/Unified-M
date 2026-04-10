import { createSignal, onMount, Show } from "solid-js";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from "recharts";
import { Activity, AlertTriangle, CheckCircle2 } from "../lib/icons";
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type DiagnosticsData } from "../lib/api";
import { formatCompactNumber, getDateAxisProps } from "../lib/chartFormat";
import { trackEvent } from "../lib/telemetry";

export default function Diagnostics() {
  const [data, setData] = createSignal<DiagnosticsData | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .diagnostics()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={data()}
        fallback={
          <EmptyState
            title="No diagnostics available"
            message="Train a model first to see fit metrics, residual plots, and convergence diagnostics."
            hideQuickStart
          />
        }
      >
        {() => {
          const d = data()!;
          const m = d.metrics;

          const residuals = d.chart.map((r) => r.residual ?? 0);
          const histBins = buildHistogram(residuals, 30);

          const scatterData = d.chart.map((r) => ({
            actual: r.actual,
            predicted: r.predicted,
          }));

          const allVals = [
            ...scatterData.map((x) => x.actual),
            ...scatterData.map((x) => x.predicted),
          ];
          const minVal = Math.min(...allVals);
          const maxVal = Math.max(...allVals);

          const dwInterpretation =
            m.durbin_watson > 1.5 && m.durbin_watson < 2.5
              ? {
                  label: "No autocorrelation",
                  color: "text-emerald-600",
                  icon: CheckCircle2,
                }
              : m.durbin_watson <= 1.5
                ? {
                    label: "Positive autocorrelation",
                    color: "text-amber-600",
                    icon: AlertTriangle,
                  }
                : {
                    label: "Negative autocorrelation",
                    color: "text-amber-600",
                    icon: AlertTriangle,
                  };

          const DWIcon = dwInterpretation.icon;

          return (
            <div>
              <h1 class="text-2xl font-bold text-slate-900">
                Model Diagnostics
              </h1>
              <p class="text-sm text-slate-500 mt-1">
                Evaluate model fit quality, residual patterns, and statistical
                assumptions
              </p>

              {/* Metric cards */}
              <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
                <MetricCard
                  label="R-squared"
                  value={m.r_squared.toFixed(3)}
                  icon={Activity}
                  color="indigo"
                />
                <MetricCard
                  label="MAPE"
                  value={`${m.mape.toFixed(1)}%`}
                  icon={Activity}
                  color="emerald"
                />
                <MetricCard
                  label="RMSE"
                  value={m.rmse.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  icon={Activity}
                  color="amber"
                />
                <MetricCard
                  label="MAE"
                  value={m.mae.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  icon={Activity}
                  color="amber"
                />
                <MetricCard
                  label="Durbin-Watson"
                  value={m.durbin_watson.toFixed(3)}
                  icon={Activity}
                  color="indigo"
                />
                <MetricCard
                  label="Observations"
                  value={m.n_observations.toLocaleString()}
                  icon={Activity}
                  color="indigo"
                />
              </div>

              {/* DW interpretation banner */}
              <div
                class={`mt-4 flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium ${
                  dwInterpretation.color === "text-emerald-600"
                    ? "bg-emerald-50 border border-emerald-200"
                    : "bg-amber-50 border border-amber-200"
                }`}
              >
                <DWIcon size={16} class={dwInterpretation.color} />
                <span class={dwInterpretation.color}>
                  Durbin-Watson: {m.durbin_watson.toFixed(3)} —{" "}
                  {dwInterpretation.label}
                </span>
              </div>

              {/* Actual vs Predicted timeline */}
              <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
                <h2 class="text-sm font-semibold text-slate-700 mb-4">
                  Actual vs Predicted Over Time
                </h2>
                <ReactChart>
                  {() =>
                    h(
                      ResponsiveContainer,
                      { width: "100%", height: 350 },
                      h(
                        LineChart,
                        {
                          data: d.chart,
                          onClick: () =>
                            trackEvent("chart_interaction", {
                              chart_id: "diagnostics_timeseries",
                              interaction: "click",
                            }),
                        },
                        h(CartesianGrid, {
                          strokeDasharray: "3 3",
                          stroke: "#e2e8f0",
                        }),
                        h(XAxis, {
                          dataKey: "date",
                          ...getDateAxisProps(d.chart.length),
                        }),
                        h(YAxis, {
                          tick: { fontSize: 11 },
                          tickFormatter: (v: number) => formatCompactNumber(v),
                        }),
                        h(Tooltip, {
                          formatter: (v: number) =>
                            v.toLocaleString(undefined, {
                              maximumFractionDigits: 0,
                            }),
                        }),
                        h(Legend),
                        h(Line, {
                          type: "monotone",
                          dataKey: "actual",
                          stroke: "#334155",
                          strokeWidth: 2,
                          dot: false,
                          name: "Actual",
                        }),
                        h(Line, {
                          type: "monotone",
                          dataKey: "predicted",
                          stroke: "#6366f1",
                          strokeWidth: 2,
                          dot: false,
                          strokeDasharray: "6 3",
                          name: "Predicted",
                        }),
                      ),
                    )
                  }
                </ReactChart>
              </div>

              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                {/* Actual vs Predicted scatter */}
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
                  <h2 class="text-sm font-semibold text-slate-700 mb-4">
                    Actual vs Predicted (Scatter)
                  </h2>
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 320 },
                        h(
                          ScatterChart,
                          { margin: { bottom: 20, left: 10 } },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: "#e2e8f0",
                          }),
                          h(XAxis, {
                            type: "number",
                            dataKey: "actual",
                            name: "Actual",
                            tick: { fontSize: 11 },
                            domain: [minVal * 0.95, maxVal * 1.05],
                            label: {
                              value: "Actual",
                              position: "insideBottom",
                              offset: -10,
                              fontSize: 12,
                            },
                          }),
                          h(YAxis, {
                            type: "number",
                            dataKey: "predicted",
                            name: "Predicted",
                            tick: { fontSize: 11 },
                            domain: [minVal * 0.95, maxVal * 1.05],
                            label: {
                              value: "Predicted",
                              angle: -90,
                              position: "insideLeft",
                              fontSize: 12,
                            },
                          }),
                          h(Tooltip, {
                            formatter: (v: number) =>
                              v.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              }),
                          }),
                          h(ReferenceLine, {
                            segment: [
                              { x: minVal, y: minVal },
                              { x: maxVal, y: maxVal },
                            ],
                            stroke: "#94a3b8",
                            strokeDasharray: "4 4",
                            strokeWidth: 1.5,
                          }),
                          h(Scatter, {
                            data: scatterData,
                            fill: "#6366f1",
                            fillOpacity: 0.6,
                            r: 3,
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                  <p class="text-xs text-slate-400 text-center mt-2">
                    Points close to the diagonal line indicate good fit
                  </p>
                </div>

                {/* Residuals over time */}
                <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
                  <h2 class="text-sm font-semibold text-slate-700 mb-4">
                    Residuals Over Time
                  </h2>
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 320 },
                        h(
                          BarChart,
                          { data: d.chart },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: "#e2e8f0",
                          }),
                          h(XAxis, {
                            dataKey: "date",
                            ...getDateAxisProps(d.chart.length),
                          }),
                          h(YAxis, {
                            tick: { fontSize: 11 },
                            tickFormatter: (v: number) =>
                              formatCompactNumber(v),
                          }),
                          h(Tooltip, {
                            formatter: (v: number) =>
                              v.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              }),
                          }),
                          h(ReferenceLine, {
                            y: 0,
                            stroke: "#64748b",
                            strokeWidth: 1.5,
                          }),
                          h(
                            Bar,
                            { dataKey: "residual", name: "Residual" },
                            ...d.chart.map((row, i) =>
                              h(Cell, {
                                key: i,
                                fill:
                                  (row.residual ?? 0) >= 0
                                    ? "#10b981"
                                    : "#ef4444",
                                fillOpacity: 0.7,
                              }),
                            ),
                          ),
                        ),
                      )
                    }
                  </ReactChart>
                  <p class="text-xs text-slate-400 text-center mt-2">
                    Randomly distributed residuals suggest a well-specified
                    model
                  </p>
                </div>
              </div>

              {/* Residual histogram */}
              <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
                <h2 class="text-sm font-semibold text-slate-700 mb-4">
                  Residual Distribution
                </h2>
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div class="lg:col-span-2">
                    <ReactChart>
                      {() =>
                        h(
                          ResponsiveContainer,
                          { width: "100%", height: 280 },
                          h(
                            BarChart,
                            { data: histBins },
                            h(CartesianGrid, {
                              strokeDasharray: "3 3",
                              stroke: "#e2e8f0",
                            }),
                            h(XAxis, {
                              dataKey: "label",
                              tick: { fontSize: 10 },
                            }),
                            h(YAxis, { tick: { fontSize: 11 } }),
                            h(Tooltip),
                            h(ReferenceLine, {
                              x: findBinForValue(histBins, 0),
                              stroke: "#64748b",
                              strokeWidth: 1.5,
                              strokeDasharray: "4 4",
                            }),
                            h(Bar, {
                              dataKey: "count",
                              fill: "#6366f1",
                              fillOpacity: 0.7,
                              radius: [4, 4, 0, 0],
                            }),
                          ),
                        )
                      }
                    </ReactChart>
                  </div>
                  <div class="space-y-3">
                    <h3 class="text-sm font-semibold text-slate-700">
                      Residual Statistics
                    </h3>
                    <div class="space-y-2 text-sm">
                      <div class="flex justify-between">
                        <span class="text-slate-500">Mean</span>
                        <span class="font-mono font-medium">
                          {(d.residual_stats?.mean ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-500">Std Dev</span>
                        <span class="font-mono font-medium">
                          {(d.residual_stats?.std ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-500">Min</span>
                        <span class="font-mono font-medium">
                          {(d.residual_stats?.min ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-500">Max</span>
                        <span class="font-mono font-medium">
                          {(d.residual_stats?.max ?? 0).toFixed(2)}
                        </span>
                      </div>
                    </div>
                    <div class="pt-3 border-t border-slate-200">
                      <h4 class="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">
                        Interpretation Guide
                      </h4>
                      <ul class="text-xs text-slate-500 space-y-1.5">
                        <li>• Mean near 0 = unbiased predictions</li>
                        <li>• Bell-shaped = normally distributed errors</li>
                        <li>• DW near 2 = no autocorrelation</li>
                        <li>• R² &gt; 0.9 = excellent fit</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        }}
      </Show>
    </Show>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildHistogram(values: number[], nBins: number) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / nBins;

  const bins = Array.from({ length: nBins }, (_, i) => ({
    label: (min + binWidth * (i + 0.5)).toFixed(0),
    binStart: min + binWidth * i,
    binEnd: min + binWidth * (i + 1),
    count: 0,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
    bins[idx].count++;
  }

  return bins;
}

function findBinForValue(
  bins: { label: string; binStart: number; binEnd: number }[],
  value: number,
): string | undefined {
  const bin = bins.find((b) => value >= b.binStart && value < b.binEnd);
  return bin?.label;
}
