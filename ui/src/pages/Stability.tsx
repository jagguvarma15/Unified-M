import { useEffect, useState } from "react";
import { Shield, AlertTriangle, TrendingUp, Activity } from "lucide-react";
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
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type StabilityData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Stability() {
  const [data, setData] = useState<StabilityData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
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
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (error) {
    return (
      <EmptyState
        icon={<AlertTriangle className="w-10 h-10 text-amber-400" />}
        title="Stability data unavailable"
        description={error}
      />
    );
  }

  const hasStabilityData =
    data &&
    (data.recommendation_stability ||
      data.parameter_drift ||
      data.contribution_stability);

  if (!data || !hasStabilityData) {
    return (
      <EmptyState
        icon={<Shield className="w-10 h-10 text-gray-400" />}
        title="No stability data yet"
        description="Run at least two pipeline runs to see recommendation stability."
      />
    );
  }

  const recStability = data.recommendation_stability;
  const drift = data.parameter_drift;
  const contribStab = data.contribution_stability;

  // Recommendation changes bar chart
  const recChanges = recStability
    ? Object.entries(recStability.channel_changes).map(([ch, v]) => ({
        channel: ch.replace("_spend", ""),
        change_pct: v.change_pct,
        absChange: Math.abs(v.change_pct),
      }))
    : [];

  // Contribution stability bar chart
  const contribData = contribStab
    ? Object.entries(contribStab).map(([ch, cv]) => ({
        channel: ch.replace("_spend", ""),
        cv: Number((cv * 100).toFixed(1)),
      }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">
        Recommendation Stability
      </h1>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {recStability && (
          <>
            <MetricCard
              icon={
                recStability.is_stable ? (
                  <Shield className="w-5 h-5 text-green-600" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-amber-600" />
                )
              }
              label="Recommendation Status"
              value={recStability.is_stable ? "Stable" : "Unstable"}
            />
            <MetricCard
              icon={<TrendingUp className="w-5 h-5" />}
              label="Max Channel Change"
              value={`${recStability.max_change_pct.toFixed(1)}%`}
            />
            <MetricCard
              icon={<Activity className="w-5 h-5" />}
              label="Alert Threshold"
              value={`${recStability.alert_threshold_pct}%`}
            />
          </>
        )}
        {drift && (
          <MetricCard
            icon={<AlertTriangle className="w-5 h-5 text-red-600" />}
            label="Parameter Drift Alerts"
            value={drift.n_drift_alerts}
          />
        )}
      </div>

      {/* Recommendation changes */}
      {recChanges.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">
            Allocation Changes vs. Previous Run
          </h2>
          <p className="text-sm text-gray-500 mb-4">
            Large swings ("whipsaw") erode stakeholder trust. Red bars exceed
            the {recStability?.alert_threshold_pct}% threshold.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={recChanges} margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="channel" />
              <YAxis
                label={{ value: "% Change", angle: -90, position: "insideLeft" }}
              />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              {recStability && (
                <>
                  <ReferenceLine
                    y={recStability.alert_threshold_pct}
                    stroke="#dc2626"
                    strokeDasharray="4 4"
                    label="Threshold"
                  />
                  <ReferenceLine
                    y={-recStability.alert_threshold_pct}
                    stroke="#dc2626"
                    strokeDasharray="4 4"
                  />
                </>
              )}
              <Bar dataKey="change_pct" name="Change %">
                {recChanges.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      Math.abs(entry.change_pct) >
                      (recStability?.alert_threshold_pct ?? 20)
                        ? "#dc2626"
                        : "#4f46e5"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Contribution coefficient of variation */}
      {contribData.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">
            Contribution Stability (CV%)
          </h2>
          <p className="text-sm text-gray-500 mb-4">
            Coefficient of Variation of rolling contributions. Lower is more
            stable.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={contribData} margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="channel" />
              <YAxis
                label={{ value: "CV %", angle: -90, position: "insideLeft" }}
              />
              <Tooltip formatter={(v: number) => `${v}%`} />
              <Bar dataKey="cv" name="CV %" fill="#6366f1">
                {contribData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Parameter drift alerts */}
      {drift && drift.alerts.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Parameter Drift Alerts
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Channel
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                    Previous
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                    Current
                  </th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                    Delta (σ)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {drift.alerts.map((a, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm">{a.channel}</td>
                    <td className="px-4 py-3 text-sm text-right">
                      {a.previous.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right">
                      {a.current.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-red-600 font-semibold">
                      {a.delta_sigma.toFixed(1)}σ
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
