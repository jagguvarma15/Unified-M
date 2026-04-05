import { createSignal } from "solid-js";

export type Density = "compact" | "default" | "comfortable";

const STORAGE_KEY = "ui-density";

function load(): Density {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "compact" || v === "comfortable") return v;
  } catch {}
  return "default";
}

const [density, _set] = createSignal<Density>(load());

export { density };

export function setDensity(d: Density) {
  _set(d);
  try {
    localStorage.setItem(STORAGE_KEY, d);
  } catch {}
}

// ── Lookup tables (all Tailwind classes listed explicitly so JIT keeps them) ──

export const CARD_PAD: Record<Density, string> = {
  compact: "px-3 py-2",
  default: "px-4 py-3",
  comfortable: "px-5 py-4",
};

export const CHART_PAD: Record<Density, string> = {
  compact: "p-4",
  default: "p-6",
  comfortable: "p-7",
};

export const CHART_HEADER_MB: Record<Density, string> = {
  compact: "mb-3",
  default: "mb-4",
  comfortable: "mb-5",
};

export const VALUE_TEXT: Record<Density, string> = {
  compact: "text-base",
  default: "text-lg",
  comfortable: "text-xl",
};

export const TH_PAD: Record<Density, string> = {
  compact: "py-2 px-3",
  default: "py-3 px-4",
  comfortable: "py-4 px-5",
};

export const TD_PAD: Record<Density, string> = {
  compact: "py-1.5 px-3",
  default: "py-3 px-4",
  comfortable: "py-4 px-5",
};

export const PAGE_HEADER_MB: Record<Density, string> = {
  compact: "mb-4",
  default: "mb-6",
  comfortable: "mb-8",
};
