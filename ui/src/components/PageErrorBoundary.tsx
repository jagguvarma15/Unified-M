import { ErrorBoundary, type JSX } from "solid-js";

interface Props {
  children: JSX.Element;
  fallback?: JSX.Element;
}

export default function PageErrorBoundary(props: Props) {
  return (
    <ErrorBoundary
      fallback={(err) =>
        props.fallback ?? (
          <div class="rounded-xl border border-red-200 bg-red-50 p-6 text-red-800">
            <h2 class="text-lg font-semibold">Something went wrong</h2>
            <p class="mt-2 text-sm">{(err as Error)?.message ?? String(err)}</p>
          </div>
        )
      }
    >
      {props.children}
    </ErrorBoundary>
  );
}
