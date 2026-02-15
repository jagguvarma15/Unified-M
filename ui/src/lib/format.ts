/**
 * Number and currency formatting (Tremor / Grafana style).
 * K/M abbreviations and consistent decimals for dashboards.
 */

export function formatCompact(value: number, decimals = 1): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return (value / 1_000_000).toFixed(decimals) + "M";
  if (abs >= 1_000) return (value / 1_000).toFixed(decimals) + "k";
  return value.toFixed(decimals);
}

export function formatCurrency(value: number, compact = true, decimals = 0): string {
  if (compact && Math.abs(value) >= 1_000) {
    const suffix = value >= 1_000_000 ? "M" : "k";
    const scaled = value >= 1_000_000 ? value / 1_000_000 : value / 1_000;
    return `$${scaled.toFixed(decimals)}${suffix}`;
  }
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: decimals,
    minimumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

export function formatNumber(value: number, options?: { decimals?: number; compact?: boolean }): string {
  const { decimals = 0, compact = false } = options ?? {};
  if (compact) return formatCompact(value, decimals);
  return value.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: 0 });
}

export function formatROAS(value: number, decimals = 2): string {
  return `${value.toFixed(decimals)}x`;
}
