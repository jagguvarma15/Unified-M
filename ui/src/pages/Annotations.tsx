import { createSignal, For, Show } from "solid-js";
import PageHeader from "../components/PageHeader";
import { Plus, Trash2, X, Info, Milestone } from "../lib/icons";
import { useToast } from "../lib/toast";
import {
  annotations,
  addAnnotation,
  removeAnnotation,
  TYPE_COLORS,
  TYPE_LABELS,
  type AnnotationType,
} from "../lib/annotationStore";

const CHANNELS = [
  { value: "", label: "All Channels" },
  { value: "google_ads", label: "Google Ads" },
  { value: "meta_ads", label: "Meta Ads" },
  { value: "tiktok_ads", label: "TikTok Ads" },
  { value: "linkedin_ads", label: "LinkedIn Ads" },
  { value: "pinterest_ads", label: "Pinterest Ads" },
  { value: "snapchat_ads", label: "Snapchat Ads" },
];

const ANNOTATION_TYPES: AnnotationType[] = [
  "launch",
  "campaign",
  "macro",
  "other",
];

const TYPE_BG: Record<AnnotationType, string> = {
  launch: "bg-indigo-100 text-indigo-700",
  campaign: "bg-emerald-100 text-emerald-700",
  macro: "bg-amber-100 text-amber-700",
  other: "bg-slate-100 text-slate-600",
};

export default function Annotations() {
  const { addToast } = useToast();
  const [showAdd, setShowAdd] = createSignal(false);
  const [filterType, setFilterType] = createSignal<AnnotationType | "all">(
    "all",
  );

  // Form state
  const [date, setDate] = createSignal(new Date().toISOString().slice(0, 10));
  const [label, setLabel] = createSignal("");
  const [type, setType] = createSignal<AnnotationType>("campaign");
  const [channel, setChannel] = createSignal("");

  const filtered = () => {
    const t = filterType();
    if (t === "all") return annotations();
    return annotations().filter((a) => a.type === t);
  };

  const handleAdd = () => {
    if (!label().trim() || !date()) return;
    addAnnotation({
      date: date(),
      label: label().trim(),
      type: type(),
      color: TYPE_COLORS[type()],
      channel: channel() || undefined,
    });
    setLabel("");
    setChannel("");
    setShowAdd(false);
    addToast("success", `Annotation "${label()}" added`);
  };

  const handleDelete = (id: string, lbl: string) => {
    removeAnnotation(id);
    addToast("info", `Annotation "${lbl}" removed`);
  };

  const resetForm = () => {
    setDate(new Date().toISOString().slice(0, 10));
    setLabel("");
    setType("campaign");
    setChannel("");
    setShowAdd(false);
  };

  return (
    <div>
      <div class="flex items-center justify-between">
        <PageHeader
          title="Annotations"
          description="Mark key events on your time-series charts. Annotations appear as vertical markers across all chart views."
        />
        <button
          onClick={() => setShowAdd(true)}
          class="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
        >
          <Plus size={14} /> New Annotation
        </button>
      </div>

      {/* Info banner */}
      <div class="mb-5 flex items-start gap-2 rounded-lg border border-indigo-100 bg-indigo-50 px-3 py-2.5 text-xs text-indigo-700">
        <Info size={14} class="mt-0.5 shrink-0" />
        <span>
          Annotations are displayed as vertical markers on contribution, ROAS,
          and spend pacing charts. They help correlate model shifts with
          real-world events like product launches, media spends, or platform
          changes.
        </span>
      </div>

      {/* Filter tabs */}
      <div class="flex gap-1 mb-4">
        {(["all", ...ANNOTATION_TYPES] as const).map((t) => (
          <button
            onClick={() => setFilterType(t)}
            class={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize ${
              filterType() === t
                ? "bg-indigo-600 text-white"
                : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            {t === "all" ? "All" : TYPE_LABELS[t as AnnotationType]}
            {t !== "all" && (
              <span class="ml-1.5 text-xs opacity-70">
                ({annotations().filter((a) => a.type === t).length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Annotations list */}
      <Show
        when={filtered().length > 0}
        fallback={
          <div class="rounded-xl border border-dashed border-slate-300 bg-white py-16 text-center">
            <Milestone size={32} class="mx-auto text-slate-300" />
            <p class="mt-3 text-sm text-slate-500">No annotations yet</p>
            <p class="mt-1 text-xs text-slate-400">
              Add events to track launches, campaigns, and macro changes
            </p>
          </div>
        }
      >
        <div class="space-y-2">
          <For each={filtered()}>
            {(ann) => (
              <div class="flex items-center gap-4 rounded-lg border border-slate-200 bg-white p-4 hover:border-slate-300 transition-colors">
                {/* Color dot */}
                <div
                  class="h-3 w-3 rounded-full shrink-0"
                  style={{ background: ann.color }}
                />
                {/* Date */}
                <span class="text-sm font-mono text-slate-500 shrink-0 w-24">
                  {ann.date}
                </span>
                {/* Label */}
                <span class="flex-1 text-sm font-medium text-slate-900">
                  {ann.label}
                </span>
                {/* Type badge */}
                <span
                  class={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${TYPE_BG[ann.type]}`}
                >
                  {TYPE_LABELS[ann.type]}
                </span>
                {/* Channel badge */}
                <Show when={ann.channel}>
                  <span class="rounded-full px-2.5 py-0.5 text-xs font-medium bg-slate-100 text-slate-600">
                    {ann.channel!.replace(/_/g, " ")}
                  </span>
                </Show>
                {/* Delete */}
                <button
                  onClick={() => handleDelete(ann.id, ann.label)}
                  class="text-slate-300 hover:text-red-500 transition-colors p-1 shrink-0"
                  aria-label="Delete annotation"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Summary count */}
      <Show when={annotations().length > 0}>
        <p class="mt-3 text-xs text-slate-400 text-right">
          {annotations().length} annotation
          {annotations().length !== 1 ? "s" : ""} total
        </p>
      </Show>

      {/* Add annotation modal */}
      <Show when={showAdd()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div class="absolute inset-0 bg-black/20" onClick={resetForm} />
          <div class="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">
                New Annotation
              </h3>
              <button
                onClick={resetForm}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Date
              </label>
              <input
                type="date"
                value={date()}
                onInput={(e) => setDate(e.currentTarget.value)}
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Label
              </label>
              <input
                type="text"
                value={label()}
                onInput={(e) => setLabel(e.currentTarget.value)}
                placeholder="e.g. Spring Sale Campaign"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Type
                </label>
                <select
                  value={type()}
                  onChange={(e) =>
                    setType(e.currentTarget.value as AnnotationType)
                  }
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={ANNOTATION_TYPES}>
                    {(t) => <option value={t}>{TYPE_LABELS[t]}</option>}
                  </For>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Channel (optional)
                </label>
                <select
                  value={channel()}
                  onChange={(e) => setChannel(e.currentTarget.value)}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={CHANNELS}>
                    {(ch) => <option value={ch.value}>{ch.label}</option>}
                  </For>
                </select>
              </div>
            </div>

            {/* Color preview */}
            <div class="flex items-center gap-2 text-xs text-slate-500">
              <div
                class="h-3 w-3 rounded-full"
                style={{ background: TYPE_COLORS[type()] }}
              />
              <span>
                Will appear as a {TYPE_LABELS[type()].toLowerCase()} marker on
                charts
              </span>
            </div>

            <div class="flex justify-end gap-2 pt-2">
              <button
                onClick={resetForm}
                class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleAdd}
                disabled={!label().trim() || !date()}
                class="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors"
              >
                Add Annotation
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
