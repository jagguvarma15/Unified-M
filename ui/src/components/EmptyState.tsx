import { Database } from "lucide-react";
import type { ReactNode } from "react";

interface Props {
  title?: string;
  message?: string;
  /** Alias for message (used by Calibration, Stability, DataQuality pages) */
  description?: string;
  /** Optional custom icon (otherwise Database) */
  icon?: ReactNode;
}

export default function EmptyState({
  title = "No data available",
  message,
  description,
  icon,
}: Props) {
  const displayMessage = message ?? description ?? "Run the pipeline first to generate results.";
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="rounded-full bg-slate-100 p-5 ring-4 ring-slate-200/60">
        {icon ?? <Database size={28} className="text-slate-400" />}
      </div>
      <h3 className="mt-5 text-lg font-semibold text-slate-800">{title}</h3>
      <p className="mt-1.5 max-w-sm text-sm text-slate-500">{displayMessage}</p>
      <div className="mt-8 max-w-md overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900 p-5 text-left shadow-lg">
        <p className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
          Quick start
        </p>
        <p className="mt-2 font-mono text-xs leading-relaxed text-slate-300">
          <span className="text-slate-500"># generate demo data + train</span>
          <br />
          <span className="text-emerald-400">$</span> PYTHONPATH=src python -m cli demo
          <br />
          <br />
          <span className="text-slate-500"># start the API server</span>
          <br />
          <span className="text-emerald-400">$</span> PYTHONPATH=src python -m cli serve
          <br />
          <br />
          <span className="text-slate-500"># start the UI (separate terminal)</span>
          <br />
          <span className="text-emerald-400">$</span> cd ui && bun dev
        </p>
      </div>
    </div>
  );
}
