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
