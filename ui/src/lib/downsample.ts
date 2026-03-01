export function downsampleEvenly<T>(rows: T[], maxPoints: number): T[] {
  if (maxPoints <= 0 || rows.length <= maxPoints) return rows;
  if (maxPoints === 1) return [rows[0]];

  const step = (rows.length - 1) / (maxPoints - 1);
  const out: T[] = [];
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.round(i * step);
    out.push(rows[idx]);
  }
  return out;
}

