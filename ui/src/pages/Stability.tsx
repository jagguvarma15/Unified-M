import { createSignal, onMount, Show, For } from "solid-js";
import { Shield, AlertTriangle, TrendingUp, Activity } from "../lib/icons";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type StabilityData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Stability() {
  const [data, setData] = createSignal<StabilityData | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .stability()
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

  const hasStabilityData = () =>
    data() &&
    (data()!.recommendation_stability ||
      data()!.parameter_drift ||
      data()!.contribution_stability);

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
            title="Stability data unavailable"
            description={error()!}
          />
        </div>
      }>
        <Show when={hasStabilityData()} fallback={
          <div class="min-h-[60vh] flex items-center justify-center">
            <EmptyState
              icon={<Shield class="w-10 h-10 text-gray-400" />}
              title="No stability data yet"
              description="Run at least two pipeline runs to see recommendation stability."
            />
          </div>
        }>
          {() => {
            const recStability = () => data()!.recommendation_stability;
            const drift = () => data()!.parameter_drift;
            const contribStab = () => data()!.contribution_stability;

            const recChanges = () =>
              recStability()
                ? Object.entries(recStability()!.channel_changes).map(([ch, v]) => ({
                    channel: ch.replace("_spend", ""),
                    change_pct: v.change_pct,
                    absChange: Math.abs(v.change_pct),
                  }))
                : [];

            const contribData = () =>
              contribStab()
                ? Object.entries(contribStab()!).map(([ch, cv]) => ({
                    channel: ch.replace("_spend", ""),
                    cv: Number((cv * 100).toFixed(1)),
                  }))
                : [];

            return (
              <div class="min-h-[60vh] space-y-6">
                <h1 class="text-2xl font-bold text-gray-900">Recommendation Stability</h1>

                {/* Summary cards */}
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Show when={recStability()}>
                    <>
                      <MetricCard
                        icon={recStability()!.is_stable ? Shield : AlertTriangle}
                        label="Recommendation Status"
                        value={recStability()!.is_stable ? "Stable" : "Unstable"}
                      />
                      <MetricCard
                        icon={TrendingUp}
                        label="Max Channel Change"
                        value={`${recStability()!.max_change_pct.toFixed(1)}%`}
                      />
                      <MetricCard
                        icon={Activity}
                        label="Alert Threshold"
                        value={`${recStability()!.alert_threshold_pct}%`}
                      />
                    </>
                  </Show>
                  <Show when={drift()}>
                    <MetricCard
                      icon={AlertTriangle}
                      label="Parameter Drift Alerts"
                      value={drift()!.n_drift_alerts}
                    />
                  </Show>
                </div>

                {/* Recommendation changes */}
                <Show when={recChanges().length > 0}>
                  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h2 class="text-lg font-semibold text-gray-900 mb-2">
                      Allocation Changes vs. Previous Run
                    </h2>
                    <p class="text-sm text-gray-500 mb-4">
                      Large swings ("whipsaw") erode stakeholder trust. Red bars exceed
                      the {recStability()?.alert_threshold_pct}% threshold.
                    </p>
                    <ReactChart>
                      {() => h(ResponsiveContainer, { width: "100%", height: 300 },
                        h(BarChart, { data: recChanges(), margin: { top: 5, right: 30, bottom: 5, left: 20 } },
                          h(CartesianGrid, { strokeDasharray: "3 3" }),
                          h(XAxis, { dataKey: "channel" }),
                          h(YAxis, { label: { value: "% Change", angle: -90, position: "insideLeft" } }),
                          h(Tooltip, { formatter: (v: number) => `${v.toFixed(1)}%` }),
                          ...(recStability() ? [
                            h(ReferenceLine, { y: recStability()!.alert_threshold_pct, stroke: "#dc2626", strokeDasharray: "4 4", label: "Threshold" }),
                            h(ReferenceLine, { y: -recStability()!.alert_threshold_pct, stroke: "#dc2626", strokeDasharray: "4 4" }),
                          ] : []),
                          h(Bar, { dataKey: "change_pct", name: "Change %" },
                            ...recChanges().map((entry, i) => h(Cell, { key: i, fill: Math.abs(entry.change_pct) > (recStability()?.alert_threshold_pct ?? 20) ? "#dc2626" : "#4f46e5" }))
                          )
                        )
                      )}
                    </ReactChart>
                  </div>
                </Show>

                {/* Contribution coefficient of variation */}
                <Show when={contribData().length > 0}>
                  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h2 class="text-lg font-semibold text-gray-900 mb-2">
                      Contribution Stability (CV%)
                    </h2>
                    <p class="text-sm text-gray-500 mb-4">
                      Coefficient of Variation of rolling contributions. Lower is more stable.
                    </p>
                    <ReactChart>
                      {() => h(ResponsiveContainer, { width: "100%", height: 300 },
                        h(BarChart, { data: contribData(), margin: { top: 5, right: 30, bottom: 5, left: 20 } },
                          h(CartesianGrid, { strokeDasharray: "3 3" }),
                          h(XAxis, { dataKey: "channel" }),
                          h(YAxis, { label: { value: "CV %", angle: -90, position: "insideLeft" } }),
                          h(Tooltip, { formatter: (v: number) => `${v}%` }),
                          h(Bar, { dataKey: "cv", name: "CV %", fill: "#6366f1" },
                            ...contribData().map((_, i) => h(Cell, { key: i, fill: COLORS[i % COLORS.length] }))
                          )
                        )
                      )}
                    </ReactChart>
                  </div>
                </Show>

                {/* Parameter drift alerts */}
                <Show when={drift() && drift()!.alerts.length > 0}>
                  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h2 class="text-lg font-semibold text-gray-900 mb-4">Parameter Drift Alerts</h2>
                    <div class="overflow-x-auto">
                      <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                          <tr>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Channel</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Previous</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Current</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Delta (σ)</th>
                          </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
                          <For each={drift()!.alerts}>
                            {(a) => (
                              <tr class="hover:bg-gray-50">
                                <td class="px-4 py-3 text-sm">{a.channel}</td>
                                <td class="px-4 py-3 text-sm text-right">{a.previous.toFixed(4)}</td>
                                <td class="px-4 py-3 text-sm text-right">{a.current.toFixed(4)}</td>
                                <td class="px-4 py-3 text-sm text-right text-red-600 font-semibold">{a.delta_sigma.toFixed(1)}σ</td>
                              </tr>
                            )}
                          </For>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </Show>
              </div>
            );
          }}
        </Show>
      </Show>
    </Show>
  );
}
