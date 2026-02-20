import { X, CheckCircle2, AlertCircle, Info, AlertTriangle } from "lucide-react";
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

export default function ToastContainer() {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.type];
        return (
          <div
            key={toast.id}
            className={`flex items-start gap-2.5 rounded-lg border px-4 py-3 shadow-lg animate-in slide-in-from-right ${COLORS[toast.type]}`}
          >
            <Icon size={16} className={`mt-0.5 shrink-0 ${ICON_COLORS[toast.type]}`} />
            <p className="flex-1 text-sm font-medium">{toast.message}</p>
            <button
              onClick={() => removeToast(toast.id)}
              className="shrink-0 rounded p-0.5 opacity-60 hover:opacity-100 transition-opacity"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
