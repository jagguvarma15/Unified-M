import { createSignal, Show } from "solid-js";
import { Calendar, ChevronDown } from "../lib/icons";
import {
  useDateRange,
  formatDateShort,
  type RangePreset,
} from "../lib/dateRange";

const PRESETS: { key: RangePreset; label: string }[] = [
  { key: "30d", label: "Last 30 days" },
  { key: "90d", label: "Last 90 days" },
  { key: "1y", label: "Last 1 year" },
  { key: "custom", label: "Custom range" },
];

export default function DateRangePicker() {
  const { range, setPreset, setCustomRange } = useDateRange();
  const [open, setOpen] = createSignal(false);
  const [customFrom, setCustomFrom] = createSignal("");
  const [customTo, setCustomTo] = createSignal("");

  const label = () => {
    const r = range();
    const preset = PRESETS.find((p) => p.key === r.preset);
    if (preset && r.preset !== "custom") return preset.label;
    return `${formatDateShort(r.from)} – ${formatDateShort(r.to)}`;
  };

  const applyCustom = () => {
    const from = new Date(customFrom());
    const to = new Date(customTo());
    if (!isNaN(from.getTime()) && !isNaN(to.getTime()) && from < to) {
      setCustomRange(from, to);
      setOpen(false);
    }
  };

  return (
    <div class="relative">
      <button
        onClick={() => setOpen(!open())}
        class="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
      >
        <Calendar size={14} />
        {label()}
        <ChevronDown
          size={12}
          class={`transition-transform ${open() ? "rotate-180" : ""}`}
        />
      </button>

      <Show when={open()}>
        <div class="absolute right-0 top-full mt-1 z-50 w-64 rounded-lg border border-slate-200 bg-white p-3 shadow-xl">
          <div class="space-y-1">
            {PRESETS.filter((p) => p.key !== "custom").map((p) => (
              <button
                onClick={() => {
                  setPreset(p.key);
                  setOpen(false);
                }}
                class={`w-full text-left rounded-md px-3 py-2 text-sm transition-colors ${
                  range().preset === p.key
                    ? "bg-indigo-50 text-indigo-700 font-medium"
                    : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div class="mt-3 border-t border-slate-100 pt-3">
            <p class="text-xs font-medium text-slate-500 mb-2">Custom range</p>
            <div class="flex gap-2">
              <input
                type="date"
                value={customFrom()}
                onInput={(e) => setCustomFrom(e.currentTarget.value)}
                class="flex-1 rounded-md border border-slate-300 px-2 py-1 text-xs"
              />
              <input
                type="date"
                value={customTo()}
                onInput={(e) => setCustomTo(e.currentTarget.value)}
                class="flex-1 rounded-md border border-slate-300 px-2 py-1 text-xs"
              />
            </div>
            <button
              onClick={applyCustom}
              class="mt-2 w-full rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 transition-colors"
            >
              Apply
            </button>
          </div>
        </div>
      </Show>
    </div>
  );
}
