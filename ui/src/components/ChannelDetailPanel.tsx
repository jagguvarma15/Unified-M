import { Show } from "solid-js";
import { X, TrendingUp, DollarSign, Target } from "../lib/icons";

interface ChannelDetail {
  channel: string;
  spend?: number;
  contribution?: number;
  roas?: number;
  lift?: number;
  optimal?: number;
  current?: number;
}

interface Props {
  open: boolean;
  channel: ChannelDetail | null;
  onClose: () => void;
}

function fmt(v: number | undefined, prefix = "", suffix = ""): string {
  if (v == null) return "—";
  if (Math.abs(v) >= 1_000_000) return `${prefix}${(v / 1_000_000).toFixed(1)}M${suffix}`;
  if (Math.abs(v) >= 1_000) return `${prefix}${(v / 1_000).toFixed(1)}k${suffix}`;
  return `${prefix}${v.toFixed(2)}${suffix}`;
}

export default function ChannelDetailPanel(props: Props) {
  return (
    <Show when={props.open && props.channel}>
      <div class="fixed inset-0 z-50 flex justify-end">
        <div class="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={props.onClose} />
        <div class="relative w-full max-w-md bg-white shadow-2xl border-l border-slate-200 overflow-y-auto animate-in slide-in-from-right">
          <div class="sticky top-0 z-10 bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between">
            <div>
              <h2 class="text-lg font-bold text-slate-900">{props.channel!.channel}</h2>
              <p class="text-xs text-slate-500">Channel deep-dive</p>
            </div>
            <button
              onClick={props.onClose}
              class="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          <div class="p-6 space-y-6">
            <div class="grid grid-cols-2 gap-4">
              <StatBox
                label="Total Spend"
                value={fmt(props.channel!.spend, "$")}
                icon={<DollarSign size={16} class="text-indigo-500" />}
              />
              <StatBox
                label="Contribution"
                value={fmt(props.channel!.contribution, "$")}
                icon={<TrendingUp size={16} class="text-emerald-500" />}
              />
              <StatBox
                label="ROAS"
                value={props.channel!.roas != null ? `${props.channel!.roas.toFixed(2)}x` : "—"}
                icon={<Target size={16} class="text-amber-500" />}
              />
              <StatBox
                label="Lift"
                value={props.channel!.lift != null ? props.channel!.lift.toFixed(4) : "—"}
                icon={<TrendingUp size={16} class="text-violet-500" />}
              />
            </div>

            <Show when={props.channel!.current != null && props.channel!.optimal != null}>
              <div class="rounded-lg border border-slate-200 p-4">
                <h3 class="text-sm font-semibold text-slate-700 mb-3">Budget Allocation</h3>
                <div class="space-y-3">
                  <BudgetBar label="Current" value={props.channel!.current!} max={Math.max(props.channel!.current ?? 0, props.channel!.optimal ?? 0)} color="bg-slate-400" />
                  <BudgetBar label="Optimal" value={props.channel!.optimal!} max={Math.max(props.channel!.current ?? 0, props.channel!.optimal ?? 0)} color="bg-indigo-500" />
                </div>
                <Show when={props.channel!.current != null && props.channel!.optimal != null}>
                  {(() => {
                    const diff = props.channel!.optimal! - props.channel!.current!;
                    const pct = props.channel!.current! > 0 ? (diff / props.channel!.current!) * 100 : 0;
                    return (
                      <p class={`mt-3 text-xs font-medium ${diff > 0 ? "text-emerald-600" : diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                        {diff >= 0 ? "+" : ""}{fmt(diff, "$")} ({pct >= 0 ? "+" : ""}{pct.toFixed(1)}%) to reach optimal
                      </p>
                    );
                  })()}
                </Show>
              </div>
            </Show>

            <div class="rounded-lg bg-slate-50 border border-slate-200 p-4">
              <h3 class="text-sm font-semibold text-slate-700 mb-2">Quick Actions</h3>
              <div class="space-y-2">
                <a href="/curves" class="block text-sm text-indigo-600 hover:text-indigo-700 font-medium">
                  View response curve →
                </a>
                <a href="/channel-insights" class="block text-sm text-indigo-600 hover:text-indigo-700 font-medium">
                  Full channel insights →
                </a>
                <a href="/optimization" class="block text-sm text-indigo-600 hover:text-indigo-700 font-medium">
                  Optimize budget →
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Show>
  );
}

function StatBox(props: { label: string; value: string; icon: any }) {
  return (
    <div class="rounded-lg border border-slate-200 bg-white p-3">
      <div class="flex items-center gap-2 text-xs font-medium text-slate-500">
        {props.icon}
        {props.label}
      </div>
      <p class="mt-1 text-lg font-bold tabular-nums text-slate-900">{props.value}</p>
    </div>
  );
}

function BudgetBar(props: { label: string; value: number; max: number; color: string }) {
  const pct = () => props.max > 0 ? (props.value / props.max) * 100 : 0;
  return (
    <div>
      <div class="flex items-center justify-between text-xs mb-1">
        <span class="text-slate-600 font-medium">{props.label}</span>
        <span class="font-mono tabular-nums text-slate-700">${(props.value / 1000).toFixed(1)}k</span>
      </div>
      <div class="h-2 rounded-full bg-slate-100 overflow-hidden">
        <div class={`h-full rounded-full ${props.color} transition-all`} style={{ width: `${pct()}%` }} />
      </div>
    </div>
  );
}
