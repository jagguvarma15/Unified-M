import { createSignal } from "solid-js";

export interface SavedScenario {
  id: string;
  name: string;
  createdAt: string;
  dateRange: { start: string; end: string };
  budgetMode: "fixed" | "flexible";
  totalBudgetMult: number;
  channelMults: Record<string, number>;
  channelMins: Record<string, number>;
  channelMaxs: Record<string, number>;
  totalBudget: number;
  estimatedResponse: number;
  estimatedROI: number;
}

const STORAGE_KEY = "unified-m-scenarios";

function load(): SavedScenario[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as SavedScenario[];
      if (Array.isArray(parsed)) return parsed;
    }
  } catch {}
  return [];
}

const [savedScenarios, setSavedScenariosRaw] =
  createSignal<SavedScenario[]>(load());

function persist(items: SavedScenario[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {}
}

export { savedScenarios };

export function saveScenario(
  scenario: Omit<SavedScenario, "id" | "createdAt">,
) {
  const next: SavedScenario = {
    ...scenario,
    id: `sc-${Date.now()}`,
    createdAt: new Date().toISOString(),
  };
  setSavedScenariosRaw((prev) => {
    const updated = [next, ...prev];
    persist(updated);
    return updated;
  });
  return next;
}

export function removeScenario(id: string) {
  setSavedScenariosRaw((prev) => {
    const updated = prev.filter((s) => s.id !== id);
    persist(updated);
    return updated;
  });
}

export function encodeScenarioUrl(scenario: SavedScenario): string {
  const payload = {
    n: scenario.name,
    dr: scenario.dateRange,
    bm: scenario.budgetMode,
    tm: scenario.totalBudgetMult,
    cm: scenario.channelMults,
    ci: scenario.channelMins,
    cx: scenario.channelMaxs,
  };
  const encoded = btoa(JSON.stringify(payload));
  return `${window.location.origin}/what-if?shared=${encoded}`;
}

export function decodeScenarioUrl(
  param: string,
): Partial<SavedScenario> | null {
  try {
    const payload = JSON.parse(atob(param));
    return {
      name: payload.n,
      dateRange: payload.dr,
      budgetMode: payload.bm,
      totalBudgetMult: payload.tm,
      channelMults: payload.cm,
      channelMins: payload.ci,
      channelMaxs: payload.cx,
    };
  } catch {
    return null;
  }
}
