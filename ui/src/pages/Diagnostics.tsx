import { useEffect, useState } from "react";
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
import { Activity, AlertTriangle, CheckCircle2 } from "lucide-react";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type DiagnosticsData } from "../lib/api";

export default function Diagnostics() {
  const [data, setData] = useState<DiagnosticsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .diagnostics()
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

  if (!data) return <EmptyState />;

  const m = data.metrics;

  // Residual histogram
  const residuals = data.chart.map((r) => r.residual);
  const histBins = buildHistogram(residuals, 30);

  // Scatter data for actual vs predicted
  const scatterData = data.chart.map((r) => ({
    actual: r.actual,
    predicted: r.predicted,
  }));

  const allVals = [...scatterData.map((d) => d.actual), ...scatterData.map((d) => d.predicted)];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);

  // Durbin-Watson interpretation
  const dwInterpretation =
    m.durbin_watson > 1.5 && m.durbin_watson < 2.5
      ? { label: "No autocorrelation", color: "text-emerald-600", icon: CheckCircle2 }
      : m.durbin_watson <= 1.5
        ? { label: "Positive autocorrelation", color: "text-amber-600", icon: AlertTriangle }
        : { label: "Negative autocorrelation", color: "text-amber-600", icon: AlertTriangle };

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">Model Diagnostics</h1>
      <p className="text-sm text-slate-500 mt-1">
        Evaluate model fit quality, residual patterns, and statistical assumptions
      </p>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
        <MetricCard label="R-squared" value={m.r_squared.toFixed(3)} icon={Activity} color="indigo" />
        <MetricCard label="MAPE" value={`${m.mape.toFixed(1)}%`} icon={Activity} color="emerald" />
        <MetricCard label="RMSE" value={m.rmse.toLocaleString(undefined, { maximumFractionDigits: 0 })} icon={Activity} color="amber" />
        <MetricCard label="MAE" value={m.mae.toLocaleString(undefined, { maximumFractionDigits: 0 })} icon={Activity} color="amber" />
        <MetricCard label="Durbin-Watson" value={m.durbin_watson.toFixed(3)} icon={Activity} color="indigo" />
        <MetricCard label="Observations" value={m.n_observations.toLocaleString()} icon={Activity} color="indigo" />
      </div>

      {/* DW interpretation banner */}
      <div className={`mt-4 flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium ${
        dwInterpretation.color === "text-emerald-600"
          ? "bg-emerald-50 border border-emerald-200"
          : "bg-amber-50 border border-amber-200"
      }`}>
        <dwInterpretation.icon size={16} className={dwInterpretation.color} />
        <span className={dwInterpretation.color}>
          Durbin-Watson: {m.durbin_watson.toFixed(3)} — {dwInterpretation.label}
        </span>
      </div>

      {/* Actual vs Predicted timeline */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Actual vs Predicted Over Time
        </h2>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={data.chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => v.toLocaleString()} />
            <Tooltip
              formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#334155"
              strokeWidth={2}
              dot={false}
              name="Actual"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              strokeDasharray="6 3"
              name="Predicted"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Actual vs Predicted scatter */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Actual vs Predicted (Scatter)
          </h2>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                type="number"
                dataKey="actual"
                name="Actual"
                tick={{ fontSize: 11 }}
                domain={[minVal * 0.95, maxVal * 1.05]}
                label={{ value: "Actual", position: "insideBottom", offset: -10, fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                tick={{ fontSize: 11 }}
                domain={[minVal * 0.95, maxVal * 1.05]}
                label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 12 }}
              />
              <Tooltip
                formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              />
              <ReferenceLine
                segment={[
                  { x: minVal, y: minVal },
                  { x: maxVal, y: maxVal },
                ]}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                strokeWidth={1.5}
              />
              <Scatter data={scatterData} fill="#6366f1" fillOpacity={0.6} r={3} />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 text-center mt-2">
            Points close to the diagonal line indicate good fit
          </p>
        </div>

        {/* Residuals over time */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Residuals Over Time
          </h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data.chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => v.toLocaleString()} />
              <Tooltip
                formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              />
              <ReferenceLine y={0} stroke="#64748b" strokeWidth={1.5} />
              <Bar dataKey="residual" name="Residual">
                {data.chart.map((row, i) => (
                  <Cell key={i} fill={row.residual >= 0 ? "#10b981" : "#ef4444"} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 text-center mt-2">
            Randomly distributed residuals suggest a well-specified model
          </p>
        </div>
      </div>

      {/* Residual histogram */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Residual Distribution
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={histBins}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <ReferenceLine x={findBinForValue(histBins, 0)} stroke="#64748b" strokeWidth={1.5} strokeDasharray="4 4" />
                <Bar dataKey="count" fill="#6366f1" fillOpacity={0.7} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-700">Residual Statistics</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Mean</span>
                <span className="font-mono font-medium">{data.residual_stats.mean.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Std Dev</span>
                <span className="font-mono font-medium">{data.residual_stats.std.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Min</span>
                <span className="font-mono font-medium">{data.residual_stats.min.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Max</span>
                <span className="font-mono font-medium">{data.residual_stats.max.toFixed(2)}</span>
              </div>
            </div>
            <div className="pt-3 border-t border-slate-200">
              <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">
                Interpretation Guide
              </h4>
              <ul className="text-xs text-slate-500 space-y-1.5">
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

function findBinForValue(bins: { label: string; binStart: number; binEnd: number }[], value: number): string | undefined {
  const bin = bins.find((b) => value >= b.binStart && value < b.binEnd);
  return bin?.label;
}
