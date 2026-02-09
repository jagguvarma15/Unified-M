import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Clock, Loader2 } from "lucide-react";
import EmptyState from "../components/EmptyState";
import { api, type RunsData, type RunManifest } from "../lib/api";

export default function Runs() {
  const [data, setData] = useState<RunsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .runs(20)
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
      <h1 className="text-2xl font-bold text-slate-900">Pipeline Runs</h1>
      <p className="text-sm text-slate-500 mt-1">
        Every run is versioned with a full audit trail
      </p>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200/60 mt-6 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
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
                <RunRow key={run.run_id} run={run} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RunRow({ run }: { run: RunManifest }) {
  const m = run.metrics;
  return (
    <tr className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
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
