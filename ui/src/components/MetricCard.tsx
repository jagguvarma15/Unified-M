import type { LucideIcon } from "lucide-react";
import Tooltip from "./Tooltip";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  delta?: string;
  color?: "indigo" | "emerald" | "amber" | "red";
  tooltip?: string;
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

  const labelEl = tooltip ? (
    <Tooltip content={tooltip} side="top">
      <span className="cursor-help border-b border-dotted border-slate-300">{label}</span>
    </Tooltip>
  ) : (
    label
  );

  return (
    <div className="min-w-0 rounded-lg border border-slate-200/80 bg-white px-4 py-3 overflow-hidden">
      <p className="flex items-center gap-2 text-xs font-medium text-slate-500 truncate">
        {Icon && (
          <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${iconBg[color]}`} aria-hidden>
            <Icon size={12} />
          </span>
        )}
        {labelEl}
      </p>
      <p className="mt-1 truncate text-lg font-semibold tabular-nums text-slate-900" title={String(value)}>
        {value}
      </p>
      {delta && <p className="mt-0.5 text-xs text-slate-500 truncate">{delta}</p>}
      {sparkline && sparkline.length > 0 && (
        <div className="mt-1.5 flex justify-end">
          <Sparkline data={sparkline} trend={trend} height={16} width={56} className="shrink-0 opacity-80" />
        </div>
      )}
    </div>
  );
}
