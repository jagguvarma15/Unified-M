import type { ReactNode } from "react";

type Variant = "default" | "success" | "warning" | "error" | "info";

const variantStyles: Record<Variant, string> = {
  default: "bg-slate-100 text-slate-700 border-slate-200",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  error: "bg-red-50 text-red-700 border-red-200",
  info: "bg-indigo-50 text-indigo-700 border-indigo-200",
};

interface Props {
  children: ReactNode;
  variant?: Variant;
  className?: string;
  /** Optional small icon before label (e.g. Lucide 12px) */
  icon?: ReactNode;
}

/**
 * Status badge used across dashboards (Tremor / shadcn style).
 * Use for: on-track, over-saturated, completed, failed, etc.
 */
export default function Badge({ children, variant = "default", className = "", icon }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${variantStyles[variant]} ${className}`}
      role="status"
    >
      {icon}
      {children}
    </span>
  );
}
