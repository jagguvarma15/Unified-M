import { createContext, createSignal, useContext, type JSX } from "solid-js";

export type RangePreset = "30d" | "90d" | "1y" | "custom";

export interface DateRange {
  from: Date;
  to: Date;
  preset: RangePreset;
}

function defaultRange(): DateRange {
  const to = new Date();
  const from = new Date();
  from.setDate(from.getDate() - 90);
  return { from, to, preset: "90d" };
}

function rangeFromPreset(preset: RangePreset): DateRange {
  const to = new Date();
  const from = new Date();
  switch (preset) {
    case "30d":
      from.setDate(from.getDate() - 30);
      break;
    case "90d":
      from.setDate(from.getDate() - 90);
      break;
    case "1y":
      from.setFullYear(from.getFullYear() - 1);
      break;
    default:
      from.setDate(from.getDate() - 90);
  }
  return { from, to, preset };
}

export function priorPeriod(range: DateRange): DateRange {
  const durationMs = range.to.getTime() - range.from.getTime();
  const priorTo = new Date(range.from.getTime() - 1);
  const priorFrom = new Date(priorTo.getTime() - durationMs);
  return { from: priorFrom, to: priorTo, preset: "custom" };
}

export function formatDateShort(d: Date): string {
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

interface DateRangeContextValue {
  range: () => DateRange;
  setPreset: (preset: RangePreset) => void;
  setCustomRange: (from: Date, to: Date) => void;
}

const DateRangeContext = createContext<DateRangeContextValue>();

export function DateRangeProvider(props: { children: JSX.Element }) {
  const [range, setRange] = createSignal<DateRange>(defaultRange());

  const setPreset = (preset: RangePreset) => {
    if (preset !== "custom") setRange(rangeFromPreset(preset));
  };

  const setCustomRange = (from: Date, to: Date) => {
    setRange({ from, to, preset: "custom" });
  };

  return (
    <DateRangeContext.Provider value={{ range, setPreset, setCustomRange }}>
      {props.children}
    </DateRangeContext.Provider>
  );
}

export function useDateRange() {
  const ctx = useContext(DateRangeContext);
  if (!ctx) throw new Error("useDateRange must be within DateRangeProvider");
  return ctx;
}
