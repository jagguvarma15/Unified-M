import type { JSX } from "solid-js";
import { Show } from "solid-js";

interface Props {
  title: string;
  description?: string;
  /** Optional "View full" link (href) */
  actionHref?: string;
  actionLabel?: string;
  /** Optional right-side slot (e.g. stats, filters) */
  rightSlot?: JSX.Element;
  children: JSX.Element;
  /** Min height for consistent chart panels (Grafana-style) */
  minHeight?: number;
  class?: string;
}

/**
 * Consistent panel wrapper for all chart sections (Grafana / Tremor style).
 * Title, description, optional action link, and consistent border/shadow.
 */
export default function ChartCard(props: Props) {
  return (
    <div
      class={`rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm transition-shadow hover:shadow-md ${props.class ?? ""}`}
      style={props.minHeight ? { "min-height": `${props.minHeight}px` } : undefined}
    >
      <div class="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 class="text-sm font-semibold tracking-tight text-slate-700">{props.title}</h2>
          <Show when={props.description}>
            <p class="mt-0.5 text-xs text-slate-500">{props.description}</p>
          </Show>
        </div>
        <div class="flex items-center gap-2">
          {props.rightSlot}
          <Show when={props.actionHref}>
            <a
              href={props.actionHref}
              class="text-xs font-medium text-indigo-600 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 rounded"
            >
              {props.actionLabel ?? "View full →"}
            </a>
          </Show>
        </div>
      </div>
      {props.children}
    </div>
  );
}
