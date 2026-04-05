import { createContext, createSignal, useContext, type JSX } from "solid-js";

export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
  createdAt: number;
  onUndo?: () => void;
}

interface ToastContextValue {
  toasts: () => Toast[];
  addToast: (
    type: ToastType,
    message: string,
    options?: { onUndo?: () => void; duration?: number },
  ) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

let _nextId = 0;

export function ToastProvider(props: { children: JSX.Element }) {
  const [toasts, setToasts] = createSignal<Toast[]>([]);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const addToast = (
    type: ToastType,
    message: string,
    options?: { onUndo?: () => void; duration?: number },
  ) => {
    const id = String(++_nextId);
    const duration = options?.duration ?? (type === "error" ? 0 : 5000);
    setToasts((prev) => [
      ...prev,
      {
        id,
        type,
        message,
        duration,
        createdAt: Date.now(),
        onUndo: options?.onUndo,
      },
    ]);
    if (duration > 0) {
      setTimeout(() => removeToast(id), duration);
    }
  };

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {props.children}
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}
