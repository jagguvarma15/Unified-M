import { createSignal, onMount, Show, For } from "solid-js";
import { CheckCircle, XCircle, Target, AlertTriangle } from "../lib/icons";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  BarChart,
  Bar,
} from "recharts";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type CalibrationData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Calibration() {
  const [data, setData] = createSignal<CalibrationData | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .calibration()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setData(null);
      })
      .finally(() => setLoading(false));
  });

  const toFinite = (value: unknown): number => {
    const n = typeof value === "number" ? value : Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="min-h-[60vh] flex items-center justify-center">
          <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show when={!error()} fallback={
        <div class="min-h-[60vh] flex items-center justify-center">
          <EmptyState
            icon={<AlertTriangle class="w-10 h-10 text-amber-400" />}
            title="Calibration data unavailable"
            description={error()!}
          />
        </div>
      }>
        <Show when={data() && (data()!.n_tests ?? 0) > 0} fallback={
          <div class="min-h-[60vh] flex items-center justify-center">
            <EmptyState
              icon={<Target class="w-10 h-10 text-gray-400" />}
              title="No calibration data yet"
              description="Run an experiment calibration to see predictions vs. measured lift."
            />
          </div>
        }>
          {(d) => {
            const points = () => data()!.points ?? [];
            const scatterData = () => points().map((p) => ({
              ...p,
              x: p.measured_lift ?? 0,
              y: p.predicted_lift ?? 0,
            }));
            const minX = () => scatterData().length ? Math.min(...scatterData().map((d) => toFinite(d.x))) : 0;
            const maxX = () => scatterData().length ? Math.max(...scatterData().map((d) => toFinite(d.x))) : 0;
            const barData = () => points().map((p) => ({
              channel: p.channel ?? "",
              error_pct: Math.round(p.error_pct ?? 0),
              within_ci: p.within_ci ?? false,
            }));
            const qualityColor = () =>
              data()!.calibration_quality === "good"
                ? "text-green-600"
                : data()!.calibration_quality === "fair"
                  ? "text-amber-600"
                  : "text-red-600";

            return (
              <div class="min-h-[60vh] space-y-6">
                <h1 class="text-2xl font-bold text-gray-900">
                  Calibration: MMM vs. Experiments
                </h1>

                {/* Summary cards */}
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <MetricCard icon={Target} label="Tests Compared" value={data()!.n_tests} />
                  <MetricCard icon={CheckCircle} label="Coverage (within CI)" value={`${((data()!.coverage ?? 0) * 100).toFixed(0)}%`} />
                  <MetricCard icon={AlertTriangle} label="Median Lift Error" value={`${(data()!.median_lift_error ?? 0).toFixed(1)}%`} />
                  <div class="bg-white rounded-xl shadow-sm border border-gray-200 px-5 py-4">
                    <p class="text-xs text-gray-500 mb-1">Quality</p>
                    <p class={`text-xl font-bold capitalize ${qualityColor()}`}>
                      {data()!.calibration_quality}
                    </p>
                  </div>
                </div>

                {/* Scatter: predicted vs measured */}
                <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                  <h2 class="text-lg font-semibold text-gray-900 mb-4">
                    Predicted vs. Measured Lift
                  </h2>
                  <p class="text-sm text-gray-500 mb-4">
                    Points near the diagonal mean the MMM prediction matched the
                    experiment result. Green = within CI, Red = outside CI.
                  </p>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        name="Measured Lift"
                        label={{ value: "Measured Lift", position: "insideBottom", offset: -10 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        name="Predicted Lift"
                        label={{ value: "Predicted Lift", angle: -90, position: "insideLeft" }}
                      />
                      <Tooltip
                        content={({ payload }) => {
                          if (!payload?.length) return null;
                          const p = payload[0].payload;
                          return (
                            <div class="bg-white border border-gray-200 rounded shadow-lg p-3 text-sm">
                              <p class="font-semibold">{p.channel}</p>
                              <p>Measured: {toFinite(p.measured_lift).toFixed(4)}</p>
                              <p>Predicted: {toFinite(p.predicted_lift).toFixed(4)}</p>
                              <p>Error: {toFinite(p.error_pct).toFixed(1)}%</p>
                              <p>Within CI: <span class={p.within_ci ? "text-green-600" : "text-red-600"}>{p.within_ci ? "Yes" : "No"}</span></p>
                            </div>
                          );
                        }}
                      />
                      <ReferenceLine
                        segment={[
                          { x: minX() * 0.8, y: minX() * 0.8 },
                          { x: maxX() * 1.2, y: maxX() * 1.2 },
                        ]}
                        stroke="#9ca3af"
                        strokeDasharray="6 4"
                        label="Perfect"
                      />
                      <Scatter data={scatterData()}>
                        {scatterData().map((entry, i) => (
                          <Cell key={i} fill={entry.within_ci ? "#16a34a" : "#dc2626"} r={8} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* Error by channel bar chart */}
                <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                  <h2 class="text-lg font-semibold text-gray-900 mb-4">
                    Lift Error by Channel
                  </h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={barData()} margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="channel" />
                      <YAxis label={{ value: "Error %", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Bar dataKey="error_pct" name="Error %">
                        {barData().map((entry, i) => (
                          <Cell key={i} fill={entry.within_ci ? "#16a34a" : "#dc2626"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Detail table */}
                <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                  <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                      <tr>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Test ID</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Channel</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Measured</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Predicted</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Error %</th>
                        <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">In CI?</th>
                      </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                      <For each={points()}>
                        {(p, i) => (
                          <tr class="hover:bg-gray-50">
                            <td class="px-4 py-3 text-sm font-mono">{p.test_id ?? ""}</td>
                            <td class="px-4 py-3 text-sm">{p.channel ?? ""}</td>
                            <td class="px-4 py-3 text-sm text-right">{(p.measured_lift ?? 0).toFixed(4)}</td>
                            <td class="px-4 py-3 text-sm text-right">{(p.predicted_lift ?? 0).toFixed(4)}</td>
                            <td class="px-4 py-3 text-sm text-right">{(p.error_pct ?? 0).toFixed(1)}%</td>
                            <td class="px-4 py-3 text-center">
                              <Show when={p.within_ci} fallback={<XCircle class="w-5 h-5 text-red-600 inline" />}>
                                <CheckCircle class="w-5 h-5 text-green-600 inline" />
                              </Show>
                            </td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              </div>
            );
          }}
        </Show>
      </Show>
    </Show>
  );
}
