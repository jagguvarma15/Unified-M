interface SkeletonProps {
  class?: string;
}

export function Skeleton(props: SkeletonProps) {
  return (
    <div
      class={`animate-pulse rounded-lg bg-slate-200/80 ${props.class ?? ""}`}
      aria-hidden
    />
  );
}

export function MetricCardSkeleton() {
  return (
    <div class="rounded-lg border border-slate-200/80 bg-white px-4 py-3">
      <Skeleton class="h-3.5 w-20" />
      <Skeleton class="mt-1.5 h-5 w-16" />
    </div>
  );
}

/** Skeleton matching the ChartCard layout (title + chart area). */
export function ChartCardSkeleton(props: { height?: number }) {
  return (
    <div class="rounded-xl border border-slate-200/60 bg-white p-6">
      <Skeleton class="h-4 w-40 mb-2" />
      <Skeleton class="h-3 w-56 mb-4" />
      <Skeleton
        class={`w-full rounded-lg`}
        style={{ height: `${props.height ?? 300}px` }}
      />
    </div>
  );
}

/** Skeleton matching a typical table layout. */
export function TableSkeleton(props: { rows?: number; cols?: number }) {
  const rows = props.rows ?? 5;
  const cols = props.cols ?? 4;
  return (
    <div class="rounded-xl border border-slate-200/60 bg-white p-6">
      <Skeleton class="h-4 w-36 mb-4" />
      <div class="space-y-3">
        {/* Header row */}
        <div class="flex gap-4">
          {Array.from({ length: cols }).map(() => (
            <Skeleton class="h-3 flex-1" />
          ))}
        </div>
        {/* Data rows */}
        {Array.from({ length: rows }).map(() => (
          <div class="flex gap-4">
            {Array.from({ length: cols }).map(() => (
              <Skeleton class="h-3.5 flex-1" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Dashboard-shaped skeleton: metric cards + chart + table. */
export function DashboardSkeleton() {
  return (
    <div class="space-y-6" aria-hidden>
      {/* Page title */}
      <div>
        <Skeleton class="h-7 w-48" />
        <Skeleton class="h-3.5 w-72 mt-2" />
      </div>
      {/* Metric cards row */}
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </div>
      {/* Charts */}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCardSkeleton height={280} />
        <ChartCardSkeleton height={280} />
      </div>
      <TableSkeleton rows={4} cols={5} />
    </div>
  );
}

/** Optimization page skeleton: metrics + bar chart + table. */
export function OptimizationSkeleton() {
  return (
    <div class="space-y-6" aria-hidden>
      <div>
        <Skeleton class="h-7 w-56" />
        <Skeleton class="h-3.5 w-80 mt-2" />
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-3 gap-4">
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </div>
      <ChartCardSkeleton height={350} />
      <TableSkeleton rows={6} cols={5} />
    </div>
  );
}

/** ROAS Analysis skeleton: 3 metrics + 2 charts + full-width chart + table. */
export function ROASAnalysisSkeleton() {
  return (
    <div class="space-y-6" aria-hidden>
      <div>
        <Skeleton class="h-7 w-52" />
        <Skeleton class="h-3.5 w-80 mt-2" />
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCardSkeleton height={300} />
        <ChartCardSkeleton height={300} />
      </div>
      <ChartCardSkeleton height={360} />
      <TableSkeleton rows={6} cols={7} />
    </div>
  );
}

/** Response Curves skeleton: bubble chart + line charts. */
export function ResponseCurvesSkeleton() {
  return (
    <div class="space-y-6" aria-hidden>
      <div>
        <Skeleton class="h-7 w-44" />
        <Skeleton class="h-3.5 w-72 mt-2" />
      </div>
      <ChartCardSkeleton height={420} />
      <ChartCardSkeleton height={400} />
      <ChartCardSkeleton height={350} />
    </div>
  );
}
