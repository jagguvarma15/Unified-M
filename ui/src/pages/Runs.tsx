import { useEffect, useState } from "react";
import {
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  GitCompareArrows,
  X,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import EmptyState from "../components/EmptyState";
import { api, type RunsData, type RunManifest, type RunComparisonData } from "../lib/api";
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";

export default function Runs() {
  const [data, setData] = useState<RunsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<RunComparisonData | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    api
      .runs(20)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const toggleSelect = (runId: string) => {
    setSelected((prev) => {
      if (prev.includes(runId)) return prev.filter((id) => id !== runId);
      if (prev.length >= 2) return [prev[1], runId];
      return [...prev, runId];
    });
    setComparison(null);
    setCompareError(null);
  };

  const handleCompare = async () => {
    if (selected.length !== 2) return;
    setComparing(true);
    setCompareError(null);
    try {
      const result = await api.compareRuns(selected[0], selected[1]);
      setComparison(result);
    } catch (err) {
      setComparison(null);
      setCompareError(err instanceof Error ? err.message : "Compare failed. Check that both runs exist and the API is reachable.");
    } finally {
      setComparing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (!data?.runs?.length) {
    return (
      <EmptyState
        title="No pipeline runs"
        message="Run the pipeline to generate your first set of results."
      />
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Pipeline Runs</h1>
          <p className="text-sm text-slate-500 mt-1">
            Every run is versioned with a full audit trail.
            {selected.length > 0 && (
              <span className="text-indigo-600 ml-1">
                {selected.length}/2 selected for comparison
              </span>
            )}
          </p>
        </div>
        {selected.length === 2 && (
          <button
            onClick={handleCompare}
            disabled={comparing}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors text-sm"
          >
            {comparing ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <GitCompareArrows size={15} />
            )}
            Compare Runs
          </button>
        )}
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200/60 mt-6 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                <th className="w-10 py-3 px-3" />
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Status
                </th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Run ID
                </th>
                <th className="text-left py-3 px-4 font-semibold text-slate-600">
                  Backend
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Rows
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Channels
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  MAPE
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  R&sup2;
                </th>
                <th className="text-right py-3 px-4 font-semibold text-slate-600">
                  Duration
                </th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((run) => (
                <RunRow
                  key={run.run_id}
                  run={run}
                  isSelected={selected.includes(run.run_id)}
                  onToggle={() => toggleSelect(run.run_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Compare error */}
      {compareError && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          <p className="font-medium">Comparison failed</p>
          <p className="mt-1">{compareError}</p>
          <p className="mt-2 text-xs text-red-600">
            Ensure the API server is running (e.g. <code className="bg-red-100 px-1 rounded">PYTHONPATH=src python -m cli serve</code>) and that both run directories exist under <code className="bg-red-100 px-1 rounded">runs/</code>.
          </p>
        </div>
      )}

      {/* Comparison panel */}
      {comparison && <ComparisonPanel data={comparison} onClose={() => { setComparison(null); setCompareError(null); }} />}
    </div>
  );
}

function RunRow({
  run,
  isSelected,
  onToggle,
}: {
  run: RunManifest;
  isSelected: boolean;
  onToggle: () => void;
}) {
  const m = run.metrics;
  return (
    <tr
      className={`border-b border-slate-100 hover:bg-slate-50 transition-colors ${isSelected ? "bg-indigo-50/50" : ""}`}
    >
      <td className="py-3 px-3 text-center">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
        />
      </td>
      <td className="py-3 px-4">
        <StatusBadge status={run.status} />
      </td>
      <td className="py-3 px-4 font-mono text-xs text-slate-600 max-w-[180px] truncate">
        {run.run_id}
      </td>
      <td className="py-3 px-4">
        <span className="inline-flex items-center rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700">
          {run.model_backend}
        </span>
      </td>
      <td className="text-right py-3 px-4 tabular-nums">{run.n_rows}</td>
      <td className="text-right py-3 px-4 tabular-nums">{run.n_channels}</td>
      <td className="text-right py-3 px-4 tabular-nums">
        {m?.mape != null ? `${m.mape.toFixed(1)}%` : "\u2014"}
      </td>
      <td className="text-right py-3 px-4 tabular-nums">
        {m?.r_squared != null ? m.r_squared.toFixed(3) : "\u2014"}
      </td>
      <td className="text-right py-3 px-4 tabular-nums text-slate-500">
        {run.duration_seconds != null
          ? `${run.duration_seconds.toFixed(1)}s`
          : "\u2014"}
      </td>
    </tr>
  );
}

function ComparisonPanel({
  data,
  onClose,
}: {
  data: RunComparisonData;
  onClose: () => void;
}) {
  const verification = data.verification;
  const metricsA = data.metrics_a;
  const metricsB = data.metrics_b;
  const metricsDelta = data.metrics_delta ?? {};
  const coeffA = data.coefficients_a;
  const coeffB = data.coefficients_b;
  const coefficientDiff = data.coefficient_diff ?? {};
  const allocA = data.allocation_a ?? {};
  const allocB = data.allocation_b ?? {};
  const allocationDiff = data.allocation_diff ?? {};
  const contributionDiff = data.contribution_diff ?? {};

  // Chart/table data from API (coefficient_diff and allocation_diff are authoritative)
  const coeffDiff: { channel: string; diff: number }[] = Object.entries(coefficientDiff)
    .map(([ch, diff]) => ({ channel: ch.replace(/_spend$/, ""), diff }))
    .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

  const allocChannels = [...new Set([...Object.keys(allocA), ...Object.keys(allocB)])].sort();
  const allocDiffRows = allocChannels.map((ch) => ({
    channel: ch.replace(/_spend$/, ""),
    a: allocA[ch] ?? 0,
    b: allocB[ch] ?? 0,
    diff: allocationDiff[ch] ?? (allocB[ch] ?? 0) - (allocA[ch] ?? 0),
  }));

  const metricKeys = ["r_squared", "mape", "rmse", "mae"];

  return (
    <div className="mt-6 rounded-xl border border-indigo-200 bg-white shadow-sm overflow-hidden">
      <div className="flex items-center justify-between p-5 border-b border-slate-100 bg-indigo-50/30">
        <div>
          <h2 className="text-base font-semibold text-slate-800">Run Comparison</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            <span className="font-mono" title={data.run_a}>{data.run_a?.toString().slice(0, 18)}</span>
            {" vs "}
            <span className="font-mono" title={data.run_b}>{data.run_b?.toString().slice(0, 18)}</span>
          </p>
        </div>
        <button onClick={onClose} className="p-1 rounded-md hover:bg-slate-200 transition-colors" aria-label="Close comparison">
          <X size={18} className="text-slate-500" />
        </button>
      </div>

      <div className="p-5 space-y-6">
        {/* Verification: run ids and data hashes for audit */}
        {verification && (
          <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">Verification</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
              <div>
                <span className="text-slate-500">Run A:</span>{" "}
                <span className="text-slate-800" title={verification.run_a}>{verification.run_a}</span>
                {verification.data_hash_a && (
                  <span className="block mt-0.5 text-slate-500" title={verification.data_hash_a}>
                    data_hash: {verification.data_hash_a.slice(0, 12)}…
                  </span>
                )}
                {verification.model_backend_a && (
                  <span className="block text-slate-500">backend: {verification.model_backend_a}</span>
                )}
              </div>
              <div>
                <span className="text-slate-500">Run B:</span>{" "}
                <span className="text-slate-800" title={verification.run_b}>{verification.run_b}</span>
                {verification.data_hash_b && (
                  <span className="block mt-0.5 text-slate-500" title={verification.data_hash_b}>
                    data_hash: {verification.data_hash_b.slice(0, 12)}…
                  </span>
                )}
                {verification.model_backend_b && (
                  <span className="block text-slate-500">backend: {verification.model_backend_b}</span>
                )}
              </div>
            </div>
            {(verification.data_hash_changed || verification.model_backend_changed) && (
              <p className="mt-2 text-xs text-amber-700">
                {verification.data_hash_changed && "Data hash changed. "}
                {verification.model_backend_changed && "Model backend changed."}
              </p>
            )}
          </div>
        )}

        {/* Metrics delta at a glance */}
        {Object.keys(metricsDelta).length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Metric Deltas (B − A)</h3>
            <p className="text-xs text-slate-500 mb-2">Positive = Run B higher. For R² higher is better; for MAPE/RMSE/MAE lower is better.</p>
            <div className="flex flex-wrap gap-2">
              {metricKeys.filter((k) => metricsDelta[k] != null).map((key) => {
                const d = metricsDelta[key]!;
                const improved = key === "r_squared" ? d > 0 : d < 0;
                return (
                  <span
                    key={key}
                    className={`inline-flex items-center rounded-md border px-2.5 py-1 text-xs font-medium tabular-nums ${
                      improved ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-slate-200 bg-slate-50 text-slate-700"
                    }`}
                  >
                    {key.replace("_", " ")}: {d >= 0 ? "+" : ""}{key === "mape" ? d.toFixed(2) + "%" : d.toFixed(4)}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* Metric comparison */}
        {metricsA && metricsB && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Side-by-Side Metrics</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {metricKeys.map((key) => {
                const va = metricsA[key];
                const vb = metricsB[key];
                if (va == null && vb == null) return null;
                const suffix = key === "mape" ? "%" : "";
                const precision = key === "r_squared" ? 4 : 2;
                return (
                  <div key={key} className="rounded-lg border border-slate-200 p-3">
                    <p className="text-[11px] text-slate-500 uppercase tracking-wide">{key.replace("_", " ")}</p>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-sm font-bold tabular-nums text-slate-800">
                        {va != null ? va.toFixed(precision) + suffix : "—"}
                      </span>
                      <span className="text-xs text-slate-400">vs</span>
                      <span className="text-sm font-bold tabular-nums text-indigo-700">
                        {vb != null ? vb.toFixed(precision) + suffix : "—"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Coefficient diff chart */}
        {coeffDiff.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Coefficient Diff (B − A)</h3>
            <ResponsiveContainer width="100%" height={Math.max(200, coeffDiff.length * 36)}>
              <BarChart data={coeffDiff} layout="vertical" margin={{ left: 90, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                />
                <Bar dataKey="diff" radius={[0, 4, 4, 0]}>
                  {coeffDiff.map((entry, i) => (
                    <Cell key={i} fill={entry.diff >= 0 ? "#10b981" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Allocation diff table */}
        {allocDiffRows.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Optimal Allocation Comparison</h3>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="text-left py-2 px-3 font-semibold text-slate-600">Channel</th>
                    <th className="text-right py-2 px-3 font-semibold text-slate-600">Run A</th>
                    <th className="text-right py-2 px-3 font-semibold text-slate-600">Run B</th>
                    <th className="text-right py-2 px-3 font-semibold text-slate-600">Diff (B − A)</th>
                  </tr>
                </thead>
                <tbody>
                  {allocDiffRows.map((row) => (
                    <tr key={row.channel} className="border-b border-slate-100">
                      <td className="py-2 px-3 font-medium text-slate-700">{row.channel}</td>
                      <td className="text-right py-2 px-3 tabular-nums">${row.a.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className="text-right py-2 px-3 tabular-nums">${row.b.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className={`text-right py-2 px-3 tabular-nums font-medium ${row.diff > 0 ? "text-emerald-600" : row.diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                        {row.diff > 0 ? "+" : ""}{row.diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Contribution diff (from contributions.parquet totals) */}
        {Object.keys(contributionDiff).length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Contribution Totals Diff (B − A)</h3>
            <p className="text-xs text-slate-500 mb-2">Per-channel contribution sum from each run’s contributions.parquet.</p>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="text-left py-2 px-3 font-semibold text-slate-600">Channel</th>
                    <th className="text-right py-2 px-3 font-semibold text-slate-600">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(contributionDiff)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .map(([ch, diff]) => (
                      <tr key={ch} className="border-b border-slate-100">
                        <td className="py-2 px-3 font-medium text-slate-700">{ch.replace(/_spend$/, "")}</td>
                        <td className={`text-right py-2 px-3 tabular-nums font-medium ${diff > 0 ? "text-emerald-600" : diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                          {diff > 0 ? "+" : ""}{diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return (
        <span className="inline-flex items-center gap-1 text-emerald-600">
          <CheckCircle2 size={15} /> OK
        </span>
      );
    case "failed":
      return (
        <span className="inline-flex items-center gap-1 text-red-500">
          <XCircle size={15} /> Failed
        </span>
      );
    case "running":
      return (
        <span className="inline-flex items-center gap-1 text-amber-500">
          <Loader2 size={15} className="animate-spin" /> Running
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 text-slate-400">
          <Clock size={15} /> {status}
        </span>
      );
  }
}
