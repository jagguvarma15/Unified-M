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
} from "recharts";
import EmptyState from "../components/EmptyState";
import { api, type ResponseCurvesData } from "../lib/api";
import { COLORS } from "../lib/colors";
import { formatCompactNumber, formatSpendTick } from "../lib/chartFormat";
import { trackEvent } from "../lib/telemetry";

export default function ResponseCurves() {
  const [data, setData] = createSignal<ResponseCurvesData | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .responseCurves()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
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

  const hasMarginal = () => channels().some(
    (ch) => data()![ch].marginal_response && data()![ch].marginal_response!.length > 0,
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
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show when={data() && Object.keys(data()!).length > 0} fallback={<EmptyState />}>
        <div>
          <h1 class="text-2xl font-bold text-slate-900">Response Curves</h1>
          <p class="text-sm text-slate-500 mt-1">
            Saturation curves showing diminishing returns per channel
          </p>

          {/* ---- Response curves ---- */}
          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
            <h2 class="text-sm font-semibold text-slate-700 mb-4">
              Response vs Spend (all channels)
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={responseRows()}
                onClick={() => trackEvent("chart_interaction", { chart_id: "response_curves", interaction: "click" })}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="spend"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => (typeof v === "number" ? formatSpendTick(v) : String(v))}
                  label={{ value: "Spend", position: "insideBottomRight", offset: -5, fontSize: 12 }}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v: number) => formatCompactNumber(v)}
                  label={{ value: "Response", angle: -90, position: "insideLeft", fontSize: 12 }}
                />
                <Tooltip
                  formatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  labelFormatter={(v) => `Spend: $${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                />
                <Legend />
                {channels().map((ch, i) => (
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
          <Show when={hasMarginal() && marginalRows().length > 0}>
            <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
              <h2 class="text-sm font-semibold text-slate-700 mb-4">
                Marginal Response (per additional dollar)
              </h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={marginalRows()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="spend"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(v) => (typeof v === "number" ? formatSpendTick(v) : String(v))}
                  />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v: number) => formatCompactNumber(v)} />
                  <Tooltip />
                  <Legend />
                  {channels().map((ch, i) => (
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
          </Show>
        </div>
      </Show>
    </Show>
  );
}
