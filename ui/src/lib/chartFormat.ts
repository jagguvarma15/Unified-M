const COMPACT_NUMBER = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const SHORT_MONTH_DAY = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});

const SHORT_MONTH = new Intl.DateTimeFormat("en-US", {
  month: "short",
});

function parseDateLike(value: unknown): Date | null {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function formatCompactNumber(value: number): string {
  return COMPACT_NUMBER.format(value);
}

export function formatSpendTick(value: number): string {
  if (Math.abs(value) < 1000) return `$${value.toFixed(0)}`;
  return `$${COMPACT_NUMBER.format(value)}`;
}

export function makeDateTickFormatter(totalPoints: number) {
  return (value: unknown): string => {
    const d = parseDateLike(value);
    if (!d) return String(value ?? "");

    if (totalPoints > 220) return SHORT_MONTH.format(d);
    if (totalPoints > 90) return SHORT_MONTH_DAY.format(d);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
  };
}

export function getDateAxisProps(totalPoints: number) {
  return {
    tick: { fontSize: 11, fill: "#64748b" },
    minTickGap: totalPoints > 180 ? 24 : 14,
    tickMargin: 8,
    tickFormatter: makeDateTickFormatter(totalPoints),
  };
}
