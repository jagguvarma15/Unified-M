import { Show, For, createSignal, createEffect, onCleanup } from "solid-js";
import { X, CheckCircle2, AlertCircle, Info, AlertTriangle, Undo2 } from "../lib/icons";
import { useToast, type ToastType } from "../lib/toast";

const ICONS: Record<ToastType, typeof CheckCircle2> = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
  warning: AlertTriangle,
};

const COLORS: Record<ToastType, string> = {
  success: "border-emerald-400 bg-emerald-50 text-emerald-800",
  error: "border-red-400 bg-red-50 text-red-800",
  info: "border-indigo-400 bg-indigo-50 text-indigo-800",
  warning: "border-amber-400 bg-amber-50 text-amber-800",
};

const ICON_COLORS: Record<ToastType, string> = {
  success: "text-emerald-500",
  error: "text-red-500",
  info: "text-indigo-500",
  warning: "text-amber-500",
};

const PROGRESS_COLORS: Record<ToastType, string> = {
  success: "bg-emerald-400",
  error: "bg-red-400",
  info: "bg-indigo-400",
  warning: "bg-amber-400",
};

function ProgressBar(props: { duration: number; createdAt: number; type: ToastType }) {
  const [pct, setPct] = createSignal(100);

  createEffect(() => {
    if (props.duration <= 0) return;
    const interval = setInterval(() => {
      const elapsed = Date.now() - props.createdAt;
      const remaining = Math.max(0, 100 - (elapsed / props.duration) * 100);
      setPct(remaining);
      if (remaining <= 0) clearInterval(interval);
    }, 50);
    onCleanup(() => clearInterval(interval));
  });

  return (
    <Show when={props.duration > 0}>
      <div class="absolute bottom-0 left-0 right-0 h-0.5 bg-black/5 rounded-b-lg overflow-hidden">
        <div
          class={`h-full transition-all ease-linear ${PROGRESS_COLORS[props.type]}`}
          style={{ width: `${pct()}%`, opacity: 0.6 }}
        />
      </div>
    </Show>
  );
}

export default function ToastContainer() {
  const { toasts, removeToast } = useToast();

  return (
    <Show when={toasts().length > 0}>
      <div
        class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm"
        role="region"
        aria-label="Notifications"
        aria-live="polite"
        aria-atomic="false"
      >
        <For each={toasts()}>
          {(toast) => {
            const Icon = ICONS[toast.type];
            return (
              <div
                role="alert"
                class={`relative flex items-start gap-2.5 rounded-lg border px-4 py-3 shadow-lg animate-in slide-in-from-right overflow-hidden ${COLORS[toast.type]}`}
              >
                <Icon size={16} aria-hidden class={`mt-0.5 shrink-0 ${ICON_COLORS[toast.type]}`} />
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium">{toast.message}</p>
                  <Show when={toast.onUndo}>
                    <button
                      onClick={() => { toast.onUndo!(); removeToast(toast.id); }}
                      class="mt-1 inline-flex items-center gap-1 text-xs font-semibold underline hover:no-underline"
                    >
                      <Undo2 size={11} /> Undo
                    </button>
                  </Show>
                </div>
                <button
                  onClick={() => removeToast(toast.id)}
                  aria-label="Dismiss notification"
                  class="shrink-0 rounded p-0.5 opacity-60 hover:opacity-100 transition-opacity"
                >
                  <X size={14} aria-hidden />
                </button>
                <ProgressBar duration={toast.duration} createdAt={toast.createdAt} type={toast.type} />
              </div>
            );
          }}
        </For>
      </div>
    </Show>
  );
}
