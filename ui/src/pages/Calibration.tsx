import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Target, AlertTriangle } from "lucide-react";
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
  Legend,
} from "recharts";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type CalibrationData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function Calibration() {
  const [data, setData] = useState<CalibrationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
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
        title="Calibration data unavailable"
        description={error}
      />
    );
  }

  const points = data?.points ?? [];
  const nTests = data?.n_tests ?? 0;
  if (!data || nTests === 0) {
    return (
      <EmptyState
        icon={<Target className="w-10 h-10 text-gray-400" />}
        title="No calibration data yet"
        description="Run an experiment calibration to see predictions vs. measured lift."
      />
    );
  }

  const qualityColor =
    data.calibration_quality === "good"
      ? "text-green-600"
      : data.calibration_quality === "fair"
        ? "text-amber-600"
        : "text-red-600";

  // Scatter data for predicted vs measured
  const scatterData = points.map((p) => ({
    ...p,
    x: p.measured_lift,
    y: p.predicted_lift,
  }));

  // Error bar chart by channel
  const barData = points.map((p) => ({
    channel: p.channel,
    error_pct: Math.round(p.error_pct),
    within_ci: p.within_ci,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">
        Calibration: MMM vs. Experiments
      </h1>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={<Target className="w-5 h-5" />}
          label="Tests Compared"
          value={data.n_tests}
        />
        <MetricCard
          icon={<CheckCircle className="w-5 h-5 text-green-600" />}
          label="Coverage (within CI)"
          value={`${(data.coverage * 100).toFixed(0)}%`}
        />
        <MetricCard
          icon={<AlertTriangle className="w-5 h-5" />}
          label="Median Lift Error"
          value={`${data.median_lift_error.toFixed(1)}%`}
        />
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-5 py-4">
          <p className="text-xs text-gray-500 mb-1">Quality</p>
          <p className={`text-xl font-bold capitalize ${qualityColor}`}>
            {data.calibration_quality}
          </p>
        </div>
      </div>

      {/* Scatter: predicted vs measured */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Predicted vs. Measured Lift
        </h2>
        <p className="text-sm text-gray-500 mb-4">
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
              label={{
                value: "Measured Lift",
                position: "insideBottom",
                offset: -10,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Predicted Lift"
              label={{
                value: "Predicted Lift",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div className="bg-white border border-gray-200 rounded shadow-lg p-3 text-sm">
                    <p className="font-semibold">{p.channel}</p>
                    <p>Measured: {p.measured_lift.toFixed(4)}</p>
                    <p>Predicted: {p.predicted_lift.toFixed(4)}</p>
                    <p>Error: {p.error_pct.toFixed(1)}%</p>
                    <p>
                      Within CI:{" "}
                      {p.within_ci ? (
                        <span className="text-green-600">Yes</span>
                      ) : (
                        <span className="text-red-600">No</span>
                      )}
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine
              segment={[
                { x: Math.min(...scatterData.map((d) => d.x)) * 0.8, y: Math.min(...scatterData.map((d) => d.x)) * 0.8 },
                { x: Math.max(...scatterData.map((d) => d.x)) * 1.2, y: Math.max(...scatterData.map((d) => d.x)) * 1.2 },
              ]}
              stroke="#9ca3af"
              strokeDasharray="6 4"
              label="Perfect"
            />
            <Scatter data={scatterData}>
              {scatterData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.within_ci ? "#16a34a" : "#dc2626"}
                  r={8}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Error by channel bar chart */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Lift Error by Channel
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barData} margin={{ top: 5, right: 30, bottom: 5, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" />
            <YAxis
              label={{ value: "Error %", angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Bar dataKey="error_pct" name="Error %">
              {barData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.within_ci ? "#16a34a" : "#dc2626"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detail table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Test ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Channel
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Measured
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Predicted
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Error %
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                In CI?
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {points.map((p, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-4 py-3 text-sm font-mono">{p.test_id}</td>
                <td className="px-4 py-3 text-sm">{p.channel}</td>
                <td className="px-4 py-3 text-sm text-right">
                  {p.measured_lift.toFixed(4)}
                </td>
                <td className="px-4 py-3 text-sm text-right">
                  {p.predicted_lift.toFixed(4)}
                </td>
                <td className="px-4 py-3 text-sm text-right">
                  {p.error_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-center">
                  {p.within_ci ? (
                    <CheckCircle className="w-5 h-5 text-green-600 inline" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600 inline" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
