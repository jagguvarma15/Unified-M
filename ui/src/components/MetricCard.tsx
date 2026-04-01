import type { LucideIcon } from "lucide-react";
import { Show } from "solid-js";
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

export default function MetricCard(props: Props) {
  const color = () => props.color ?? "indigo";

  const trend = () => {
    const sp = props.sparkline;
    if (!sp || sp.length < 2) return undefined;
    if (sp[sp.length - 1] > sp[0]) return "up";
    if (sp[sp.length - 1] < sp[0]) return "down";
    return "neutral";
  };

  const labelEl = () => {
    if (props.tooltip) {
      return (
        <Tooltip content={props.tooltip} side="top">
          <span class="cursor-help border-b border-dotted border-slate-300">{props.label}</span>
        </Tooltip>
      );
    }
    return <>{props.label}</>;
  };

  return (
    <div class="min-w-0 rounded-lg border border-slate-200/80 bg-white px-4 py-3 overflow-hidden">
      <p class="flex items-center gap-2 text-xs font-medium text-slate-500 truncate">
        <Show when={props.icon}>
          {(Icon) => (
            <span class={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${iconBg[color()]}`} aria-hidden>
              <Icon() size={12} />
            </span>
          )}
        </Show>
        {labelEl()}
      </p>
      <p class="mt-1 truncate text-lg font-semibold tabular-nums text-slate-900" title={String(props.value)}>
        {props.value}
      </p>
      <Show when={props.delta}>
        <p class="mt-0.5 text-xs text-slate-500 truncate">{props.delta}</p>
      </Show>
      <Show when={props.sparkline && props.sparkline.length > 0}>
        <div class="mt-1.5 flex justify-end">
          <Sparkline data={props.sparkline!} trend={trend()} height={16} width={56} class="shrink-0 opacity-80" />
        </div>
      </Show>
    </div>
  );
}
