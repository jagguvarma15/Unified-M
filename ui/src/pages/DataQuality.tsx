import { createSignal, onMount, Show, For } from "solid-js";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Shield,
  Database,
} from "../lib/icons";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type DataQualityData, type GateResult } from "../lib/api";

function GateIcon(props: { gate: GateResult }) {
  if (props.gate.passed) {
    return <CheckCircle class="w-5 h-5 text-green-600" />;
  }
  if (props.gate.severity === "warning") {
    return <AlertTriangle class="w-5 h-5 text-amber-500" />;
  }
  return <XCircle class="w-5 h-5 text-red-600" />;
}

function GateBadge(props: { severity: string; passed: boolean }) {
  if (props.passed) {
    return (
      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
        PASS
      </span>
    );
  }
  if (props.severity === "warning") {
    return (
      <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
        WARN
      </span>
    );
  }
  return (
    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
      FAIL
    </span>
  );
}

export default function DataQuality() {
  const [data, setData] = createSignal<DataQualityData | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [expanded, setExpanded] = createSignal<string | null>(null);

  onMount(() => {
    api
      .dataQuality()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setData(null);
      })
      .finally(() => setLoading(false));
  });

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="min-h-[60vh] flex items-center justify-center">
          <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={!error()}
        fallback={
          <div class="min-h-[60vh] flex items-center justify-center">
            <EmptyState
              icon={<AlertTriangle class="w-10 h-10 text-amber-400" />}
              title="Data quality report unavailable"
              description={error()!}
            />
          </div>
        }
      >
        <Show
          when={data()}
          fallback={
            <div class="min-h-[60vh] flex items-center justify-center">
              <EmptyState
                icon={<Database class="w-10 h-10 text-gray-400" />}
                title="No data quality report"
                description="Run the pipeline to generate a data quality report."
              />
            </div>
          }
        >
          {() => (
            <div class="min-h-[60vh] space-y-6">
              <h1 class="text-2xl font-bold text-gray-900">Data Quality</h1>

              {/* Summary banner */}
              <div
                class={`rounded-xl p-5 flex items-center gap-4 ${
                  data()!.overall_pass
                    ? "bg-green-50 border border-green-200"
                    : "bg-red-50 border border-red-200"
                }`}
              >
                <Show
                  when={data()!.overall_pass}
                  fallback={<XCircle class="w-8 h-8 text-red-600" />}
                >
                  <Shield class="w-8 h-8 text-green-600" />
                </Show>
                <div>
                  <p class="text-lg font-semibold">
                    {data()!.overall_pass
                      ? "All quality gates passed"
                      : `${data()!.n_failed} gate(s) failed`}
                  </p>
                  <p class="text-sm text-gray-600">
                    {data()!.timestamp
                      ? `Checked at ${new Date(data()!.timestamp!).toLocaleString()} · `
                      : ""}
                    {data()!.n_passed ?? 0} passed, {data()!.n_warnings ?? 0}{" "}
                    warnings, {data()!.n_failed ?? 0} failed
                  </p>
                </div>
              </div>

              {/* Summary cards */}
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MetricCard
                  icon={CheckCircle}
                  label="Passed"
                  value={data()!.n_passed ?? 0}
                />
                <MetricCard
                  icon={AlertTriangle}
                  label="Warnings"
                  value={data()!.n_warnings ?? 0}
                />
                <MetricCard
                  icon={XCircle}
                  label="Failed"
                  value={data()!.n_failed ?? 0}
                />
              </div>

              {/* Gate details */}
              <div class="bg-white rounded-xl shadow-sm border border-gray-200 divide-y divide-gray-200">
                <For each={data()!.gates ?? []}>
                  {(gate) => {
                    const gateKey = gate.gate_name ?? gate.name ?? "";
                    return (
                      <div>
                        <button
                          class="w-full px-5 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
                          onClick={() =>
                            setExpanded(expanded() === gateKey ? null : gateKey)
                          }
                        >
                          <div class="flex items-center gap-3">
                            <GateIcon gate={gate} />
                            <div class="text-left">
                              <p class="text-sm font-semibold text-gray-900">
                                {(gate.gate_name ?? gate.name)
                                  .replace(/_/g, " ")
                                  .replace(/\b\w/g, (c: string) =>
                                    c.toUpperCase(),
                                  )}
                              </p>
                              <p class="text-xs text-gray-500">
                                {gate.message}
                              </p>
                            </div>
                          </div>
                          <GateBadge
                            severity={gate.severity}
                            passed={gate.passed}
                          />
                        </button>
                        <Show when={expanded() === gateKey}>
                          <div class="px-5 pb-4 bg-gray-50">
                            <pre class="text-xs text-gray-700 bg-white rounded border border-gray-200 p-3 overflow-x-auto whitespace-pre-wrap">
                              {JSON.stringify(gate.details, null, 2)}
                            </pre>
                          </div>
                        </Show>
                      </div>
                    );
                  }}
                </For>
              </div>
            </div>
          )}
        </Show>
      </Show>
    </Show>
  );
}
