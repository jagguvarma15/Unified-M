interface Props {
  data: number[];
  width?: number;
  height?: number;
  strokeColor?: string;
  fillColor?: string;
  strokeWidth?: number;
  className?: string;
  /** Show trend up/down tint (green/red) */
  trend?: "up" | "down" | "neutral";
}

/**
 * Tiny SVG sparkline for metric cards and tables (Tremor-style).
 * No Recharts dependency; pure SVG path.
 */
export default function Sparkline({
  data,
  width = 64,
  height = 24,
  strokeColor = "#6366f1",
  fillColor = "rgba(99, 102, 241, 0.15)",
  strokeWidth = 1.5,
  className = "",
  trend,
}: Props) {
  if (!data.length) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 2;
  const w = width - padding * 2;
  const h = height - padding * 2;
  const step = data.length > 1 ? w / (data.length - 1) : 0;

  const points = data.map((v, i) => {
    const x = padding + i * step;
    const y = padding + h - ((v - min) / range) * h;
    return `${x},${y}`;
  });
  const pathD = `M ${points.join(" L ")}`;
  const areaD = `M ${padding},${height - padding} L ${points.join(" L ")} L ${padding + w},${height - padding} Z`;

  const trendStroke =
    trend === "up" ? "#10b981" : trend === "down" ? "#ef4444" : strokeColor;

  return (
    <svg
      width={width}
      height={height}
      className={className}
      aria-hidden
    >
      <path d={areaD} fill={fillColor} />
      <path
        d={pathD}
        fill="none"
        stroke={trendStroke}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
