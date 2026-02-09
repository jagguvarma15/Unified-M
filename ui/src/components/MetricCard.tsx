import type { LucideIcon } from "lucide-react";

interface Props {
  label: string;
  value: string | number;
  icon: LucideIcon;
  delta?: string;
  color?: "indigo" | "emerald" | "amber" | "red";
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
}: Props) {
  return (
    <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-200/60">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <div className={`p-2 rounded-lg ${iconBg[color]}`}>
          <Icon size={18} />
        </div>
      </div>
      <p className="text-2xl font-bold text-slate-900 mt-2">{value}</p>
      {delta && <p className="text-xs text-slate-500 mt-1">{delta}</p>}
    </div>
  );
}
