import { Show, type JSX } from "solid-js";

interface PageHeaderProps {
  title: string;
  description?: string;
  detail?: JSX.Element;
  hint?: string;
}

export default function PageHeader(props: PageHeaderProps) {
  return (
    <header class="mb-6">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 class="text-2xl font-bold tracking-tight text-slate-900">{props.title}</h1>
          <Show when={props.description}>
            <p class="mt-1 text-sm text-slate-500">{props.description}</p>
          </Show>
          <Show when={props.detail}>
            <p class="mt-1.5 text-xs text-slate-400">{props.detail}</p>
          </Show>
        </div>
        <Show when={props.hint}>
          <p class="text-xs text-slate-400 max-w-[200px] hidden sm:block" title={props.hint}>
            {props.hint}
          </p>
        </Show>
      </div>
    </header>
  );
}
