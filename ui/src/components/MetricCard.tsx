import type { LucideIcon } from "lucide-react";
import Tooltip from "./Tooltip";

interface Props {
  label: string;
  value: string | number;
  icon: LucideIcon;
  delta?: string;
  color?: "indigo" | "emerald" | "amber" | "red";
  /** Short explanation for the metric (shown in tooltip on label hover) */
  tooltip?: string;
}

const iconBg: Record<string, string> = {
  indigo: "bg-indigo-50 text-indigo-600",
  emerald: "bg-emerald-50 text-emerald-600",
  amber: "bg-amber-50 text-amber-600",
  red: "bg-red-50 text-red-600",
};

export default function MetricCard({
  label,
  value,
  icon: Icon,
  delta,
  color = "indigo",
  tooltip,
}: Props) {
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-sm transition-shadow duration-200 hover:shadow-md">
      <div className="flex items-center justify-between">
        {tooltip ? (
          <Tooltip content={tooltip} side="top">
            <span className="cursor-help text-sm font-medium text-slate-500 border-b border-dotted border-slate-400/50">
              {label}
            </span>
          </Tooltip>
        ) : (
          <p className="text-sm font-medium text-slate-500">{label}</p>
        )}
        <div className={`rounded-lg p-2 ${iconBg[color]}`}>
          <Icon size={18} aria-hidden />
        </div>
      </div>
      <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">{value}</p>
      {delta && <p className="mt-1 text-xs text-slate-500">{delta}</p>}
    </div>
  );
}
