import { Database, ExternalLink, Play } from "../lib/icons";
import type { JSX } from "solid-js";
import { Show } from "solid-js";

interface Action {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface Props {
  title?: string;
  message?: string;
  /** Alias for message (used by Calibration, Stability, DataQuality pages) */
  description?: string;
  /** Optional custom icon (otherwise Database) */
  icon?: JSX.Element;
  /** Primary CTA (e.g. "Run pipeline" → /data or "Upload data" → /data) */
  action?: Action;
  /** Secondary link (e.g. "View docs") */
  secondaryAction?: Action;
  /** Hide the quick-start code block */
  hideQuickStart?: boolean;
}

export default function EmptyState(props: Props) {
  const displayMessage = () => props.message ?? props.description ?? "Run the pipeline first to generate results.";

  return (
    <div class="flex flex-col items-center justify-center py-20 text-center">
      <div class="rounded-full bg-slate-100 p-5 ring-4 ring-slate-200/60" aria-hidden>
        {props.icon ?? <Database size={28} class="text-slate-400" />}
      </div>
      <h3 class="mt-5 text-lg font-semibold text-slate-800">{props.title ?? "No data available"}</h3>
      <p class="mt-1.5 max-w-sm text-sm text-slate-500">{displayMessage()}</p>

      <Show when={props.action || props.secondaryAction}>
        <div class="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Show when={props.action}>
            <Show
              when={props.action!.href}
              fallback={
                <button
                  type="button"
                  onClick={props.action!.onClick}
                  class="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
                >
                  <Play size={16} aria-hidden />
                  {props.action!.label}
                </button>
              }
            >
              <a
                href={props.action!.href}
                class="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <Play size={16} aria-hidden />
                {props.action!.label}
              </a>
            </Show>
          </Show>
          <Show when={props.secondaryAction}>
            <Show
              when={props.secondaryAction!.href}
              fallback={
                <button
                  type="button"
                  onClick={props.secondaryAction!.onClick}
                  class="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
                >
                  <ExternalLink size={16} aria-hidden />
                  {props.secondaryAction!.label}
                </button>
              }
            >
              <a
                href={props.secondaryAction!.href}
                class="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
              >
                <ExternalLink size={16} aria-hidden />
                {props.secondaryAction!.label}
              </a>
            </Show>
          </Show>
        </div>
      </Show>

      <Show when={!props.hideQuickStart}>
        <div class="mt-8 max-w-md overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900 p-5 text-left shadow-lg">
          <p class="text-[11px] font-medium uppercase tracking-wider text-slate-500">
            Quick start
          </p>
          <p class="mt-2 font-mono text-xs leading-relaxed text-slate-300">
            <span class="text-slate-500"># generate demo data + train</span>
            <br />
            <span class="text-emerald-400">$</span> PYTHONPATH=src python -m cli demo
            <br />
            <br />
            <span class="text-slate-500"># start the API server</span>
            <br />
            <span class="text-emerald-400">$</span> PYTHONPATH=src python -m cli serve
            <br />
            <br />
            <span class="text-slate-500"># start the UI (separate terminal)</span>
            <br />
            <span class="text-emerald-400">$</span> cd ui && bun dev
          </p>
        </div>
      </Show>
    </div>
  );
}
