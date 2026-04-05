import { createSignal, onMount, Show, For } from "solid-js";
import { Loader2, Printer, Copy, Check, TrendingUp } from "../lib/icons";
import EmptyState from "../components/EmptyState";
import { api, type ReportSummaryData } from "../lib/api";

export default function Report() {
  const [data, setData] = createSignal<ReportSummaryData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [copied, setCopied] = createSignal(false);
  let printRef!: HTMLDivElement;

  onMount(() => {
    api
      .reportSummary()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  });

  const handlePrint = () => window.print();

  const handleCopy = async () => {
    const d = data();
    if (!d) return;
    const lines: string[] = [
      "UNIFIED-M EXECUTIVE SUMMARY",
      `Generated: ${new Date(d.generated_at).toLocaleDateString()}`,
      `Run: ${d.run_id || "—"}`,
      "",
      "KEY METRICS",
    ];
    if (d.metrics.r_squared != null)
      lines.push(`  R²: ${d.metrics.r_squared.toFixed(3)}`);
    if (d.metrics.mape != null)
      lines.push(`  MAPE: ${d.metrics.mape.toFixed(1)}%`);
    if (d.roas_summary?.blended_roas != null)
      lines.push(`  Blended ROAS: ${d.roas_summary.blended_roas.toFixed(2)}`);
    if (d.roas_summary?.total_spend != null)
      lines.push(
        `  Total Spend: $${d.roas_summary.total_spend.toLocaleString()}`,
      );
    if (d.improvement_pct)
      lines.push(`  Optimization Uplift: +${d.improvement_pct.toFixed(1)}%`);
    lines.push("");
    if (d.top_channels.length) {
      lines.push("TOP CHANNELS");
      d.top_channels.forEach((c) =>
        lines.push(
          `  ${c.channel}: ${c.share_pct}% share ($${c.contribution.toLocaleString()})`,
        ),
      );
      lines.push("");
    }
    if (d.recommendations.length) {
      lines.push("KEY RECOMMENDATIONS");
      d.recommendations.forEach((r) => lines.push(`  • ${r}`));
    }
    await navigator.clipboard.writeText(lines.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <Loader2 class="h-8 w-8 animate-spin text-indigo-600" />
        </div>
      }
    >
      <Show
        when={data() && (data()!.run_id || data()!.top_channels.length > 0)}
        fallback={
          <EmptyState
            title="No report available"
            message="Run the pipeline to generate an executive summary."
          />
        }
      >
        {() => (
          <div>
            {/* Toolbar (hidden on print) */}
            <div class="flex items-center justify-between mb-6 print:hidden">
              <div>
                <h1 class="text-2xl font-bold tracking-tight text-slate-900">
                  Executive Summary
                </h1>
                <p class="mt-1 text-sm text-slate-500">
                  One-click report for stakeholders
                </p>
              </div>
              <div class="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  class="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
                >
                  <Show when={copied()} fallback={<Copy size={15} />}>
                    <Check size={15} class="text-emerald-500" />
                  </Show>
                  {copied() ? "Copied" : "Copy as Text"}
                </button>
                <button
                  onClick={handlePrint}
                  class="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  <Printer size={15} />
                  Print
                </button>
              </div>
            </div>

            {/* Report content */}
            <div ref={printRef} class="space-y-6">
              {/* Header */}
              <div class="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
                <div class="flex items-center gap-3 mb-4">
                  <div class="rounded-lg bg-indigo-100 p-2">
                    <TrendingUp size={20} class="text-indigo-600" />
                  </div>
                  <div>
                    <h2 class="text-lg font-bold text-slate-900">
                      Unified-M Marketing Report
                    </h2>
                    <p class="text-xs text-slate-500">
                      {new Date(data()!.generated_at).toLocaleDateString(
                        undefined,
                        {
                          weekday: "long",
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                        },
                      )}
                      <Show when={data()!.run_id}>
                        {" · Run "}
                        <span class="font-mono">
                          {data()!.run_id!.slice(0, 12)}
                        </span>
                      </Show>
                    </p>
                  </div>
                </div>

                {/* Metric pills */}
                <div class="flex flex-wrap gap-3">
                  <Show when={data()!.metrics.r_squared != null}>
                    <MetricPill
                      label="R²"
                      value={data()!.metrics.r_squared!.toFixed(3)}
                    />
                  </Show>
                  <Show when={data()!.metrics.mape != null}>
                    <MetricPill
                      label="MAPE"
                      value={`${data()!.metrics.mape!.toFixed(1)}%`}
                    />
                  </Show>
                  <Show when={data()!.roas_summary?.blended_roas != null}>
                    <MetricPill
                      label="Blended ROAS"
                      value={data()!.roas_summary!.blended_roas.toFixed(2)}
                    />
                  </Show>
                  <Show when={data()!.roas_summary?.total_spend != null}>
                    <MetricPill
                      label="Total Spend"
                      value={`$${data()!.roas_summary!.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    />
                  </Show>
                  <Show when={data()!.roas_summary?.total_contribution != null}>
                    <MetricPill
                      label="Total Contribution"
                      value={`$${data()!.roas_summary!.total_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    />
                  </Show>
                  <Show when={data()!.improvement_pct > 0}>
                    <MetricPill
                      label="Optimization Uplift"
                      value={`+${data()!.improvement_pct.toFixed(1)}%`}
                      accent
                    />
                  </Show>
                </div>
              </div>

              {/* Top channels */}
              <Show when={data()!.top_channels.length > 0}>
                <div class="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
                  <h3 class="text-base font-semibold text-slate-800 mb-4">
                    Top Channels by Contribution
                  </h3>
                  <div class="space-y-3">
                    <For each={data()!.top_channels}>
                      {(ch) => (
                        <div class="flex items-center gap-3">
                          <span class="w-32 text-sm text-slate-700 truncate font-medium">
                            {ch.channel.replace(/_spend$/, "")}
                          </span>
                          <div class="flex-1 h-5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              class="h-full bg-indigo-500 rounded-full transition-all"
                              style={{ width: `${ch.share_pct}%` }}
                            />
                          </div>
                          <span class="w-16 text-right text-sm tabular-nums font-semibold text-slate-800">
                            {ch.share_pct}%
                          </span>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              </Show>

              {/* Recommendations */}
              <Show when={data()!.recommendations.length > 0}>
                <div class="rounded-xl border border-indigo-200 bg-indigo-50/50 p-6 shadow-sm">
                  <h3 class="text-base font-semibold text-indigo-900 mb-3">
                    Key Recommendations
                  </h3>
                  <ul class="space-y-2">
                    <For each={data()!.recommendations}>
                      {(rec, i) => (
                        <li class="flex items-start gap-2 text-sm text-indigo-800">
                          <span class="mt-0.5 flex-shrink-0 h-5 w-5 rounded-full bg-indigo-200 text-indigo-700 text-xs font-bold flex items-center justify-center">
                            {i() + 1}
                          </span>
                          {rec}
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
              </Show>
            </div>
          </div>
        )}
      </Show>
    </Show>
  );
}

function MetricPill(props: { label: string; value: string; accent?: boolean }) {
  return (
    <div
      class={`rounded-lg border px-3 py-2 ${props.accent ? "border-indigo-200 bg-indigo-50" : "border-slate-200 bg-slate-50"}`}
    >
      <p class="text-[11px] text-slate-500">{props.label}</p>
      <p
        class={`text-sm font-bold tabular-nums ${props.accent ? "text-indigo-700" : "text-slate-800"}`}
      >
        {props.value}
      </p>
    </div>
  );
}
