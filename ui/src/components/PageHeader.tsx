import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  /** Optional detail line (e.g. "Run: abc123 · Updated 2m ago") */
  detail?: ReactNode;
  /** Optional short hint/tooltip for the page */
  hint?: string;
}

export default function PageHeader({ title, description, detail, hint }: PageHeaderProps) {
  return (
    <header className="mb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">{title}</h1>
          {description && (
            <p className="mt-1 text-sm text-slate-500">{description}</p>
          )}
          {detail && (
            <p className="mt-1.5 text-xs text-slate-400">{detail}</p>
          )}
        </div>
        {hint && (
          <p className="text-xs text-slate-400 max-w-[200px] hidden sm:block" title={hint}>
            {hint}
          </p>
        )}
      </div>
    </header>
  );
}
