import { Database, ExternalLink, Play, ArrowRight } from "../lib/icons";
import type { JSX } from "solid-js";
import { Show, For } from "solid-js";

interface Action {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface GuidedStep {
  label: string;
  description?: string;
  action?: Action;
}

interface Props {
  title?: string;
  message?: string;
  description?: string;
  icon?: JSX.Element;
  action?: Action;
  secondaryAction?: Action;
  hideQuickStart?: boolean;
  /** Guided action CTA steps */
  steps?: GuidedStep[];
}

function AnimatedIllustration() {
  return (
    <div class="relative rounded-full bg-gradient-to-br from-indigo-50 to-slate-50 p-6 ring-4 ring-slate-200/40" aria-hidden>
      <div class="relative">
        <Database size={32} class="text-slate-400" />
        <div class="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-indigo-400 animate-ping" />
        <div class="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-indigo-500" />
      </div>
      <div class="absolute inset-0 rounded-full animate-pulse bg-indigo-100/30" />
    </div>
  );
}

export default function EmptyState(props: Props) {
  const displayMessage = () => props.message ?? props.description ?? "Run the pipeline first to generate results.";

  return (
    <div class="flex flex-col items-center justify-center py-20 text-center">
      <AnimatedIllustration />
      <h3 class="mt-5 text-lg font-semibold text-slate-800">{props.title ?? "No data available"}</h3>
      <p class="mt-1.5 max-w-sm text-sm text-slate-500">{displayMessage()}</p>

      {/* Guided steps */}
      <Show when={props.steps && props.steps.length > 0}>
        <div class="mt-8 w-full max-w-md">
          <div class="rounded-xl border border-slate-200 bg-white divide-y divide-slate-100 shadow-sm">
            <For each={props.steps}>
              {(step, i) => (
                <div class="flex items-center gap-4 px-5 py-4">
                  <div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-bold text-indigo-600">
                    {i() + 1}
                  </div>
                  <div class="flex-1 text-left">
                    <p class="text-sm font-medium text-slate-800">{step.label}</p>
                    <Show when={step.description}>
                      <p class="text-xs text-slate-500 mt-0.5">{step.description}</p>
                    </Show>
                  </div>
                  <Show when={step.action}>
                    <Show
                      when={step.action!.href}
                      fallback={
                        <button
                          onClick={step.action!.onClick}
                          class="text-xs font-medium text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
                        >
                          {step.action!.label}
                          <ArrowRight size={12} />
                        </button>
                      }
                    >
                      <a
                        href={step.action!.href}
                        class="text-xs font-medium text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
                      >
                        {step.action!.label}
                        <ArrowRight size={12} />
                      </a>
                    </Show>
                  </Show>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>

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
