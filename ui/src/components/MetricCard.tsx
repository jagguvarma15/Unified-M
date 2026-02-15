import type { LucideIcon } from "lucide-react";
import Tooltip from "./Tooltip";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  value: string | number;
  icon: LucideIcon;
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
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-sm transition-shadow duration-200 hover:shadow-md focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 overflow-hidden">
      <div className="flex items-center justify-between gap-2 min-h-[28px]">
        <div className="min-w-0 flex-1">
          {tooltip ? (
            <Tooltip content={tooltip} side="top">
              <span className="cursor-help text-sm font-medium text-slate-500 border-b border-dotted border-slate-400/50 truncate block">
                {label}
              </span>
            </Tooltip>
          ) : (
            <p className="text-sm font-medium text-slate-500 truncate">{label}</p>
          )}
        </div>
        <div className="flex flex-shrink-0 items-center gap-1">
          {sparkline && sparkline.length > 0 && (
            <Sparkline data={sparkline} trend={trend} height={22} width={40} className="shrink-0" />
          )}
          <div className={`shrink-0 rounded-lg p-1.5 ${iconBg[color]}`}>
            <Icon size={16} aria-hidden />
          </div>
        </div>
      </div>
      <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900 truncate">{value}</p>
      {delta && <p className="mt-1 text-xs text-slate-500 truncate">{delta}</p>}
    </div>
  );
}
