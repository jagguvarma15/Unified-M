/**
 * Chart and UI color palette for Unified-M.
 */

/** Generic chart series palette (cycled via index). */
export const COLORS = [
  "#6366f1", // indigo-500
  "#10b981", // emerald-500
  "#f59e0b", // amber-500
  "#ec4899", // pink-500
  "#8b5cf6", // violet-500
  "#06b6d4", // cyan-500
  "#f97316", // orange-500
  "#84cc16", // lime-500
];

export const CHART_GRID = "#e2e8f0";
export const CHART_TOOLTIP_BG = "rgba(15, 23, 42, 0.9)";

// ---------------------------------------------------------------------------
// Semantic accent tokens
// ---------------------------------------------------------------------------

/** Secondary accent for positive / optimization actions (teal-500). */
export const ACCENT_POSITIVE = "#14b8a6";
/** Secondary accent hover (teal-600). */
export const ACCENT_POSITIVE_HOVER = "#0d9488";

// ---------------------------------------------------------------------------
// Fixed channel-to-color map
// Guarantees the same channel always gets the same color across all charts.
// ---------------------------------------------------------------------------

const CHANNEL_COLOR_MAP: Record<string, string> = {
  // Ad platforms
  google_ads: "#4285F4",
  google: "#4285F4",
  meta_ads: "#0081FB",
  meta: "#0081FB",
  facebook: "#0081FB",
  tiktok_ads: "#010101",
  tiktok: "#010101",
  linkedin_ads: "#0A66C2",
  linkedin: "#0A66C2",
  pinterest_ads: "#E60023",
  pinterest: "#E60023",
  snapchat_ads: "#FFFC00",
  snapchat: "#FFFC00",
  twitter_ads: "#1DA1F2",
  twitter: "#1DA1F2",
  x_ads: "#1DA1F2",
  apple_search_ads: "#0071E3",
  apple: "#0071E3",

  // Analytics / Revenue
  shopify: "#96BF48",
  salesforce: "#00A1E0",

  // Generic channels (display, email, TV, etc.)
  display: "#8b5cf6",
  email: "#f59e0b",
  tv: "#ec4899",
  radio: "#f97316",
  print: "#84cc16",
  seo: "#06b6d4",
  organic: "#10b981",
  direct: "#6366f1",
  referral: "#a855f7",
  affiliate: "#14b8a6",
  video: "#e11d48",
  audio: "#d946ef",
  ooh: "#eab308",
};

/**
 * Returns a deterministic color for a channel name.
 * Falls back to the generic COLORS palette if the name is unknown.
 */
export function channelColor(name: string, index: number = 0): string {
  const key = name.toLowerCase().replace(/[\s\-]+/g, "_");
  if (CHANNEL_COLOR_MAP[key]) return CHANNEL_COLOR_MAP[key];
  // Try partial match (e.g. "Google Ads (brand)" → "google_ads")
  for (const [mapKey, color] of Object.entries(CHANNEL_COLOR_MAP)) {
    if (key.includes(mapKey) || mapKey.includes(key)) return color;
  }
  return COLORS[index % COLORS.length];
}
