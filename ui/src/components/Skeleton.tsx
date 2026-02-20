interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-slate-200/80 ${className}`}
      aria-hidden
    />
  );
}

/** Placeholder for metric card during load */
export function MetricCardSkeleton() {
  return (
    <div className="rounded-lg border border-slate-200/80 bg-white px-4 py-3">
      <Skeleton className="h-3.5 w-20" />
      <Skeleton className="mt-1.5 h-5 w-16" />
    </div>
  );
}
