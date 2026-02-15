import type { ReactNode } from "react";

interface Props {
  title: string;
  description?: string;
  /** Optional "View full" link (href) */
  actionHref?: string;
  actionLabel?: string;
  /** Optional right-side slot (e.g. stats, filters) */
  rightSlot?: ReactNode;
  children: ReactNode;
  /** Min height for consistent chart panels (Grafana-style) */
  minHeight?: number;
  className?: string;
}

/**
 * Consistent panel wrapper for all chart sections (Grafana / Tremor style).
 * Title, description, optional action link, and consistent border/shadow.
 */
export default function ChartCard({
  title,
  description,
  actionHref,
  actionLabel = "View full →",
  rightSlot,
  children,
  minHeight = 280,
  className = "",
}: Props) {
  return (
    <div
      className={`rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm transition-shadow hover:shadow-md ${className}`}
      style={minHeight ? { minHeight } : undefined}
    >
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">{title}</h2>
          {description && <p className="mt-0.5 text-xs text-slate-500">{description}</p>}
        </div>
        <div className="flex items-center gap-2">
          {rightSlot}
          {actionHref && (
            <a
              href={actionHref}
              className="text-xs font-medium text-indigo-600 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 rounded"
            >
              {actionLabel}
            </a>
          )}
        </div>
      </div>
      {children}
    </div>
  );
}
