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
} from "recharts";
import EmptyState from "../components/EmptyState";
import { api, type ResponseCurvesData } from "../lib/api";
import { COLORS } from "../lib/colors";

export default function ResponseCurves() {
  const [data, setData] = useState<ResponseCurvesData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .responseCurves()
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

  if (!data || Object.keys(data).length === 0) return <EmptyState />;

  const channels = Object.keys(data);

  // Build unified data for the multi-line chart.
  // All channels share the same spend-index (longest array).
  const maxLen = Math.max(
    ...channels.map((ch) => data[ch].spend?.length ?? 0),
  );
  const step = Math.max(1, Math.floor(maxLen / 100));

  const responseRows: Record<string, number | string>[] = [];
  for (let i = 0; i < maxLen; i += step) {
    const row: Record<string, number | string> = {};
    let hasSpend = false;
    for (const ch of channels) {
      const sp = data[ch].spend?.[i];
      const re = data[ch].response?.[i];
      if (sp !== undefined) {
        row[`${ch}_spend`] = sp;
        row[ch] = re ?? 0;
        if (!hasSpend) {
          row.spend = sp;
          hasSpend = true;
        }
      }
    }
    if (hasSpend) responseRows.push(row);
  }

  // Build marginal response rows
  const hasMarginal = channels.some(
    (ch) => data[ch].marginal_response && data[ch].marginal_response!.length > 0,
  );
  const marginalRows: Record<string, number | string>[] = [];
  if (hasMarginal) {
    for (let i = 0; i < maxLen; i += step) {
      const row: Record<string, number | string> = {};
      let hasSpend = false;
      for (const ch of channels) {
        const sp = data[ch].spend?.[i];
        const mr = data[ch].marginal_response?.[i];
        if (sp !== undefined && mr !== undefined) {
          row.spend = sp;
          row[ch] = mr;
          hasSpend = true;
        }
      }
      if (hasSpend) marginalRows.push(row);
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">Response Curves</h1>
      <p className="text-sm text-slate-500 mt-1">
        Saturation curves showing diminishing returns per channel
      </p>

      {/* ---- Response curves ---- */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Response vs Spend (all channels)
        </h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={responseRows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="spend"
              tick={{ fontSize: 12 }}
              tickFormatter={(v) =>
                typeof v === "number" ? `$${(v / 1000).toFixed(0)}k` : v
              }
              label={{
                value: "Spend",
                position: "insideBottomRight",
                offset: -5,
                fontSize: 12,
              }}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              label={{
                value: "Response",
                angle: -90,
                position: "insideLeft",
                fontSize: 12,
              }}
            />
            <Tooltip
              formatter={(v: number) =>
                v.toLocaleString(undefined, { maximumFractionDigits: 2 })
              }
              labelFormatter={(v) =>
                `Spend: $${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
              }
            />
            <Legend />
            {channels.map((ch, i) => (
              <Line
                key={ch}
                type="monotone"
                dataKey={ch}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2.5}
                dot={false}
                name={ch}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Marginal response ---- */}
      {hasMarginal && marginalRows.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">
            Marginal Response (per additional dollar)
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={marginalRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="spend"
                tick={{ fontSize: 12 }}
                tickFormatter={(v) =>
                  typeof v === "number" ? `$${(v / 1000).toFixed(0)}k` : v
                }
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              {channels.map((ch, i) => (
                <Line
                  key={ch}
                  type="monotone"
                  dataKey={ch}
                  stroke={COLORS[i % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  name={ch}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
