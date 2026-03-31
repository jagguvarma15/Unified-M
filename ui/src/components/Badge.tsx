import type { JSX } from "solid-js";

type Variant = "default" | "success" | "warning" | "error" | "info";

const variantStyles: Record<Variant, string> = {
  default: "bg-slate-100 text-slate-700 border-slate-200",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  error: "bg-red-50 text-red-700 border-red-200",
  info: "bg-indigo-50 text-indigo-700 border-indigo-200",
};

interface Props {
  children: JSX.Element;
  variant?: Variant;
  class?: string;
  icon?: JSX.Element;
}

export default function Badge(props: Props) {
  return (
    <span
      class={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${variantStyles[props.variant ?? "default"]} ${props.class ?? ""}`}
      role="status"
    >
      {props.icon}
      {props.children}
    </span>
  );
}
