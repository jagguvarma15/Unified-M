import type { LucideIcon } from "lucide-react";
import Tooltip from "./Tooltip";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  value: string | number;
  /** Optional: omit for a clean label + value only card */
  icon?: LucideIcon;
  delta?: string;
  color?: "indigo" | "emerald" | "amber" | "red";
  /** Short explanation for the metric (shown in tooltip on label hover) */
  tooltip?: string;
  /** Optional sparkline data (Tremor-style micro chart) */
  sparkline?: number[];
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
  sparkline,
}: Props) {
  const trend =
    sparkline && sparkline.length >= 2
      ? sparkline[sparkline.length - 1] > sparkline[0]
        ? "up"
        : sparkline[sparkline.length - 1] < sparkline[0]
          ? "down"
          : "neutral"
      : undefined;

  return (
    <div className="min-w-0 rounded-xl border border-slate-200/60 bg-white p-5 shadow-sm transition-shadow duration-200 hover:shadow-md focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 overflow-hidden">
      <div className="flex min-h-[24px] items-center gap-2">
        {Icon && (
          <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${iconBg[color]}`} aria-hidden>
            <Icon size={14} />
          </div>
        )}
        <div className="min-w-0 flex-1 overflow-hidden">
          {tooltip ? (
            <Tooltip content={tooltip} side="top">
              <span className="block truncate cursor-help text-sm font-medium text-slate-500 border-b border-dotted border-slate-400/50">
                {label}
              </span>
            </Tooltip>
          ) : (
            <p className="truncate text-sm font-medium text-slate-500">{label}</p>
          )}
        </div>
      </div>
      <p className="mt-2 truncate text-2xl font-bold tabular-nums text-slate-900" title={typeof value === "string" ? value : String(value)}>
        {value}
      </p>
      {delta && <p className="mt-1 truncate text-xs text-slate-500">{delta}</p>}
      {sparkline && sparkline.length > 0 && (
        <div className="mt-2 flex justify-end">
          <Sparkline data={sparkline} trend={trend} height={20} width={64} className="shrink-0" />
        </div>
      )}
    </div>
  );
}
