import { Database, ExternalLink, Play } from "lucide-react";
import type { ReactNode } from "react";

interface Action {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface Props {
  title?: string;
  message?: string;
  /** Alias for message (used by Calibration, Stability, DataQuality pages) */
  description?: string;
  /** Optional custom icon (otherwise Database) */
  icon?: ReactNode;
  /** Primary CTA (e.g. "Run pipeline" → /data or "Upload data" → /data) */
  action?: Action;
  /** Secondary link (e.g. "View docs") */
  secondaryAction?: Action;
  /** Hide the quick-start code block */
  hideQuickStart?: boolean;
}

export default function EmptyState({
  title = "No data available",
  message,
  description,
  icon,
  action,
  secondaryAction,
  hideQuickStart = false,
}: Props) {
  const displayMessage = message ?? description ?? "Run the pipeline first to generate results.";

  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="rounded-full bg-slate-100 p-5 ring-4 ring-slate-200/60" aria-hidden>
        {icon ?? <Database size={28} className="text-slate-400" />}
      </div>
      <h3 className="mt-5 text-lg font-semibold text-slate-800">{title}</h3>
      <p className="mt-1.5 max-w-sm text-sm text-slate-500">{displayMessage}</p>

      {(action || secondaryAction) && (
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          {action &&
            (action.href ? (
              <a
                href={action.href}
                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <Play size={16} aria-hidden />
                {action.label}
              </a>
            ) : (
              <button
                type="button"
                onClick={action.onClick}
                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <Play size={16} aria-hidden />
                {action.label}
              </button>
            ))}
          {secondaryAction &&
            (secondaryAction.href ? (
              <a
                href={secondaryAction.href}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <ExternalLink size={16} aria-hidden />
                {secondaryAction.label}
              </a>
            ) : (
              <button
                type="button"
                onClick={secondaryAction.onClick}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <ExternalLink size={16} aria-hidden />
                {secondaryAction.label}
              </button>
            ))}
        </div>
      )}

      {!hideQuickStart && (
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
      )}
    </div>
  );
}
