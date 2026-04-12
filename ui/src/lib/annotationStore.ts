import { createSignal } from "solid-js";

export type AnnotationType = "launch" | "campaign" | "macro" | "other";

export interface Annotation {
  id: string;
  date: string; // YYYY-MM-DD
  label: string;
  type: AnnotationType;
  color: string;
  channel?: string;
}

export const TYPE_COLORS: Record<AnnotationType, string> = {
  launch: "#6366f1",
  campaign: "#10b981",
  macro: "#f59e0b",
  other: "#94a3b8",
};

export const TYPE_LABELS: Record<AnnotationType, string> = {
  launch: "Product Launch",
  campaign: "Campaign",
  macro: "Macro Event",
  other: "Other",
};

const STORAGE_KEY = "unified-m-annotations";

const DEMO_ANNOTATIONS: Annotation[] = [
  {
    id: "a1",
    date: "2026-01-15",
    label: "Q1 Product Launch",
    type: "launch",
    color: "#6366f1",
  },
  {
    id: "a2",
    date: "2026-02-14",
    label: "Valentine's Day Campaign",
    type: "campaign",
    color: "#10b981",
    channel: "meta_ads",
  },
  {
    id: "a3",
    date: "2026-03-01",
    label: "iOS Privacy Update",
    type: "macro",
    color: "#f59e0b",
  },
];

function loadAnnotations(): Annotation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Annotation[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {}
  return DEMO_ANNOTATIONS;
}

const [annotations, setAnnotationsRaw] =
  createSignal<Annotation[]>(loadAnnotations());

function persist(anns: Annotation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(anns));
  } catch {}
}

export { annotations };

export function addAnnotation(ann: Omit<Annotation, "id">) {
  const next: Annotation = { ...ann, id: `ann-${Date.now()}` };
  setAnnotationsRaw((prev) => {
    const updated = [...prev, next].sort((a, b) =>
      a.date.localeCompare(b.date),
    );
    persist(updated);
    return updated;
  });
}

export function removeAnnotation(id: string) {
  setAnnotationsRaw((prev) => {
    const updated = prev.filter((a) => a.id !== id);
    persist(updated);
    return updated;
  });
}
