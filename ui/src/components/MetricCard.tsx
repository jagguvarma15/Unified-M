import type { LucideIcon } from "../lib/icons";
import { Show } from "solid-js";
import Tooltip from "./Tooltip";
import Sparkline from "./Sparkline";
import { density, CARD_PAD, VALUE_TEXT } from "../lib/density";

interface Props {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  delta?: string;
  /** Numeric % change vs prior period — renders trend arrow + badge */
  changePct?: number;
  /** Optional label for the change badge (e.g. "vs Q1") */
  changeLabel?: string;
  color?: "indigo" | "emerald" | "amber" | "red";
  tooltip?: string;
  sparkline?: number[];
  onClick?: () => void;
  /** Haus-style confidence range bar: lo/hi define the scale, current is the marker */
  rangeBar?: { lo: number; hi: number; current: number };
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

  const changeColor = () => {
    const pct = props.changePct;
    if (pct == null) return "";
    if (pct > 0) return "text-emerald-600 bg-emerald-50";
    if (pct < 0) return "text-red-600 bg-red-50";
    return "text-slate-500 bg-slate-50";
  };

  const changeArrow = () => {
    const pct = props.changePct;
    if (pct == null) return "";
    if (pct > 0) return "↑";
    if (pct < 0) return "↓";
    return "→";
  };

  const labelEl = () => {
    if (props.tooltip) {
      return (
        <Tooltip content={props.tooltip} side="top">
          <span class="cursor-help border-b border-dotted border-slate-300">
            {props.label}
          </span>
        </Tooltip>
      );
    }
    return <>{props.label}</>;
  };

  return (
    <div
      class={`min-w-0 rounded-lg border border-slate-200/80 bg-white overflow-hidden transition-shadow hover:shadow-md ${CARD_PAD[density()]} ${props.onClick ? "cursor-pointer" : ""}`}
      onClick={props.onClick}
    >
      <div class="flex items-start justify-between gap-2">
        <p class="flex items-center gap-2 text-xs font-medium text-slate-500 truncate">
          <Show when={props.icon}>
            {(Icon) => {
              const Ic = Icon();
              return (
                <span
                  class={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${iconBg[color()]}`}
                  aria-hidden
                >
                  <Ic size={12} />
                </span>
              );
            }}
          </Show>
          {labelEl()}
        </p>
        <Show when={props.sparkline && props.sparkline.length > 0}>
          <Sparkline
            data={props.sparkline!}
            trend={trend()}
            height={20}
            width={48}
            class="shrink-0 opacity-80"
          />
        </Show>
      </div>

      <div class="mt-1 flex items-end justify-between gap-2">
        <p
          class={`truncate font-semibold tabular-nums text-slate-900 ${VALUE_TEXT[density()]}`}
          title={String(props.value)}
        >
          {props.value}
        </p>
        <Show when={props.changePct != null}>
          <span
            class={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold tabular-nums whitespace-nowrap ${changeColor()}`}
            title={props.changeLabel ?? "vs prior period"}
          >
            {changeArrow()} {Math.abs(props.changePct!).toFixed(1)}%
          </span>
        </Show>
      </div>

      <Show when={props.delta}>
        <p class="mt-0.5 text-xs text-slate-500 truncate">{props.delta}</p>
      </Show>

      <Show when={props.rangeBar} keyed>
        {(rb) => {
          const span = rb.hi - rb.lo;
          if (span <= 0) return null;
          const pct = Math.max(
            2,
            Math.min(98, ((rb.current - rb.lo) / span) * 100),
          );
          const trackColor =
            color() === "emerald"
              ? "bg-emerald-100"
              : color() === "amber"
                ? "bg-amber-100"
                : "bg-indigo-100";
          const fillColor =
            color() === "emerald"
              ? "bg-emerald-400"
              : color() === "amber"
                ? "bg-amber-400"
                : "bg-indigo-400";
          return (
            <div
              class={`mt-2 h-1 rounded-full ${trackColor} relative overflow-visible`}
            >
              <div
                class={`absolute top-[-2px] h-[8px] w-[3px] rounded-full ${fillColor}`}
                style={{ left: `calc(${pct}% - 1px)` }}
              />
            </div>
          );
        }}
      </Show>
    </div>
  );
}
