import { createMemo, createSignal, Show, For } from "solid-js";
import type { JSX } from "solid-js";
import {
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  GitCompareArrows,
  X,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import EmptyState from "../components/EmptyState";
import { type RunManifest, type RunComparisonData } from "../lib/api";
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";
import { useCompareRunsMutation, useRunsQuery } from "../lib/queries";
import { VirtualizedList } from "../components/VirtualizedList";

export default function Runs() {
  const [selected, setSelected] = createSignal<string[]>([]);
  const [comparison, setComparison] = createSignal<RunComparisonData | null>(null);
  const [compareError, setCompareError] = createSignal<string | null>(null);
  const runsQuery = useRunsQuery(200);
  const compareRuns = useCompareRunsMutation();
  const runs = () => runsQuery.data?.runs ?? [];
  const useVirtualized = () => runs().length > 60;
  const gridCols = "64px minmax(120px,140px) minmax(260px,1fr) 140px 110px 110px 110px 110px 110px";

  const toggleSelect = (runId: string) => {
    setSelected((prev) => {
      if (prev.includes(runId)) return prev.filter((id) => id !== runId);
      if (prev.length >= 2) return [prev[1], runId];
      return [...prev, runId];
    });
    setComparison(null);
    setCompareError(null);
  };

  const handleCompare = async () => {
    if (selected().length !== 2) return;
    setCompareError(null);
    try {
      const result = await compareRuns.mutateAsync({ runA: selected()[0], runB: selected()[1] });
      setComparison(result);
    } catch (err) {
      setComparison(null);
      setCompareError(err instanceof Error ? err.message : "Compare failed. Check that both runs exist and the API is reachable.");
    }
  };

  return (
    <Show
      when={!runsQuery.isLoading}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={runsQuery.data?.runs?.length}
        fallback={
          <EmptyState
            title="No pipeline runs"
            message="Run the pipeline to generate your first set of results."
          />
        }
      >
        <div>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-2xl font-bold text-slate-900">Pipeline Runs</h1>
              <p class="text-sm text-slate-500 mt-1">
                Every run is versioned with a full audit trail.
                <Show when={selected().length > 0}>
                  <span class="text-indigo-600 ml-1">
                    {selected().length}/2 selected for comparison
                  </span>
                </Show>
              </p>
            </div>
            <Show when={selected().length === 2}>
              <button
                onClick={handleCompare}
                disabled={compareRuns.isPending}
                class="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors text-sm"
              >
                <Show when={compareRuns.isPending} fallback={<GitCompareArrows size={15} />}>
                  <Loader2 size={15} class="animate-spin" />
                </Show>
                Compare Runs
              </button>
            </Show>
          </div>

          <div class="bg-white rounded-xl shadow-sm border border-slate-200/60 mt-6 overflow-hidden">
            <div class="overflow-x-auto">
              <div class="min-w-[1020px]">
                <div
                  class="grid bg-slate-50 border-b border-slate-200 text-sm font-semibold text-slate-600"
                  style={{ "grid-template-columns": gridCols }}
                >
                  <div class="py-3 px-3" />
                  <div class="py-3 px-4">Status</div>
                  <div class="py-3 px-4">Run ID</div>
                  <div class="py-3 px-4">Backend</div>
                  <div class="py-3 px-4 text-right">Rows</div>
                  <div class="py-3 px-4 text-right">Channels</div>
                  <div class="py-3 px-4 text-right">MAPE</div>
                  <div class="py-3 px-4 text-right">R&sup2;</div>
                  <div class="py-3 px-4 text-right">Duration</div>
                </div>

                <Show
                  when={useVirtualized()}
                  fallback={
                    <div>
                      <For each={runs()}>
                        {(run) => (
                          <RunGridRow
                            run={run}
                            columns={gridCols}
                            isSelected={selected().includes(run.run_id)}
                            onToggle={() => toggleSelect(run.run_id)}
                          />
                        )}
                      </For>
                    </div>
                  }
                >
                  <VirtualizedList
                    rows={runs()}
                    rowHeight={50}
                    height={Math.min(600, runs().length * 50)}
                    renderRow={(run, _, style) => (
                      <RunGridRow
                        run={run}
                        style={style}
                        columns={gridCols}
                        isSelected={selected().includes(run.run_id)}
                        onToggle={() => toggleSelect(run.run_id)}
                      />
                    )}
                  />
                </Show>
              </div>
            </div>
          </div>

          <Show when={compareError()}>
            <div class="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              <p class="font-medium">Comparison failed</p>
              <p class="mt-1">{compareError()}</p>
            </div>
          </Show>

          <Show when={comparison()}>
            <ComparisonPanel data={comparison()!} onClose={() => { setComparison(null); setCompareError(null); }} />
          </Show>
        </div>
      </Show>
    </Show>
  );
}

function RunGridRow(props: {
  run: RunManifest;
  isSelected: boolean;
  onToggle: () => void;
  columns: string;
  style?: JSX.CSSProperties;
}) {
  const m = props.run.metrics;
  return (
    <div
      style={{ ...props.style, "grid-template-columns": props.columns }}
      class={`grid border-b border-slate-100 hover:bg-slate-50 transition-colors text-sm ${props.isSelected ? "bg-indigo-50/50" : ""}`}
    >
      <div class="py-3 px-3 text-center">
        <input
          type="checkbox"
          checked={props.isSelected}
          onChange={props.onToggle}
          class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
        />
      </div>
      <div class="py-3 px-4"><StatusBadge status={props.run.status} /></div>
      <div class="py-3 px-4 font-mono text-xs text-slate-600 truncate">{props.run.run_id}</div>
      <div class="py-3 px-4">
        <span class="inline-flex items-center rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700">
          {props.run.model_backend}
        </span>
      </div>
      <div class="text-right py-3 px-4 tabular-nums">{props.run.n_rows}</div>
      <div class="text-right py-3 px-4 tabular-nums">{props.run.n_channels}</div>
      <div class="text-right py-3 px-4 tabular-nums">{m?.mape != null ? `${m.mape.toFixed(1)}%` : "\u2014"}</div>
      <div class="text-right py-3 px-4 tabular-nums">{m?.r_squared != null ? m.r_squared.toFixed(3) : "\u2014"}</div>
      <div class="text-right py-3 px-4 tabular-nums text-slate-500">
        {props.run.duration_seconds != null ? `${props.run.duration_seconds.toFixed(1)}s` : "\u2014"}
      </div>
    </div>
  );
}

function VerificationBadge(props: { label: string; changed?: boolean }) {
  return (
    <span
      class={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
        props.changed ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"
      }`}
    >
      <Show when={props.changed} fallback={<CheckCircle2 size={11} />}>
        <XCircle size={11} />
      </Show>
      {props.label}
    </span>
  );
}

function ComparisonPanel(props: { data: RunComparisonData; onClose: () => void }) {
  const verification = props.data.verification;
  const metricsA = props.data.metrics_a;
  const metricsB = props.data.metrics_b;
  const metricsDelta = props.data.metrics_delta ?? {};
  const coefficientDiff = props.data.coefficient_diff ?? {};
  const allocA = props.data.allocation_a ?? {};
  const allocB = props.data.allocation_b ?? {};
  const allocationDiff = props.data.allocation_diff ?? {};
  const contributionDiff = props.data.contribution_diff ?? {};

  const coeffDiff = Object.entries(coefficientDiff)
    .map(([ch, diff]) => ({ channel: ch.replace(/_spend$/, ""), diff }))
    .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

  const allocChannels = [...new Set([...Object.keys(allocA), ...Object.keys(allocB)])].sort();

  const allocOverlay = allocChannels.map((ch) => ({
    channel: ch.replace(/_spend$/, ""),
    "Run A": allocA[ch] ?? 0,
    "Run B": allocB[ch] ?? 0,
  }));

  const allocDiffRows = allocChannels.map((ch) => ({
    channel: ch.replace(/_spend$/, ""),
    a: allocA[ch] ?? 0,
    b: allocB[ch] ?? 0,
    diff: allocationDiff[ch] ?? (allocB[ch] ?? 0) - (allocA[ch] ?? 0),
  }));

  const metricKeys = ["r_squared", "mape", "rmse", "mae"];

  return (
    <div class="mt-6 rounded-xl border border-indigo-200 bg-white shadow-sm overflow-hidden">
      <div class="flex items-center justify-between p-5 border-b border-slate-100 bg-indigo-50/30">
        <div>
          <h2 class="text-base font-semibold text-slate-800">Run Comparison</h2>
          <p class="text-xs text-slate-500 mt-0.5">
            <span class="font-mono" title={props.data.run_a}>{props.data.run_a?.toString().slice(0, 18)}</span>
            {" vs "}
            <span class="font-mono" title={props.data.run_b}>{props.data.run_b?.toString().slice(0, 18)}</span>
          </p>
        </div>
        <button onClick={props.onClose} class="p-1 rounded-md hover:bg-slate-200 transition-colors" aria-label="Close">
          <X size={18} class="text-slate-500" />
        </button>
      </div>

      <div class="p-5 space-y-6">
        {/* Verification with badges */}
        <Show when={verification}>
          <div class="rounded-lg border border-slate-200 bg-slate-50/80 p-4">
            <div class="flex items-center gap-2 mb-3">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-500">Verification</h3>
              <VerificationBadge label={verification!.data_hash_changed ? "Data changed" : "Same data"} changed={verification!.data_hash_changed} />
              <VerificationBadge label={verification!.model_backend_changed ? "Backend changed" : "Same backend"} changed={verification!.model_backend_changed} />
            </div>
            <div class="grid grid-cols-2 gap-4">
              <div class="rounded-md border border-slate-200 bg-white p-3">
                <p class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">Run A</p>
                <p class="text-xs font-mono text-slate-700 truncate" title={verification!.run_a}>{verification!.run_a}</p>
                <Show when={verification!.data_hash_a}>
                  <p class="text-[11px] font-mono text-slate-400 mt-0.5">hash: {verification!.data_hash_a!.slice(0, 12)}</p>
                </Show>
                <Show when={verification!.model_backend_a}>
                  <p class="text-[11px] text-slate-400">backend: {verification!.model_backend_a}</p>
                </Show>
              </div>
              <div class="rounded-md border border-slate-200 bg-white p-3">
                <p class="text-[10px] font-semibold uppercase tracking-wider text-indigo-400 mb-1">Run B</p>
                <p class="text-xs font-mono text-slate-700 truncate" title={verification!.run_b}>{verification!.run_b}</p>
                <Show when={verification!.data_hash_b}>
                  <p class="text-[11px] font-mono text-slate-400 mt-0.5">hash: {verification!.data_hash_b!.slice(0, 12)}</p>
                </Show>
                <Show when={verification!.model_backend_b}>
                  <p class="text-[11px] text-slate-400">backend: {verification!.model_backend_b}</p>
                </Show>
              </div>
            </div>
          </div>
        </Show>

        {/* Side-by-side metrics with delta */}
        <Show when={metricsA && metricsB}>
          <div>
            <h3 class="text-sm font-semibold text-slate-700 mb-3">Metrics Side-by-Side</h3>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <For each={metricKeys}>
                {(key) => {
                  const va = metricsA![key];
                  const vb = metricsB![key];
                  if (va == null && vb == null) return null;
                  const d = metricsDelta[key];
                  const improved = d != null && (key === "r_squared" ? d > 0 : d < 0);
                  const suffix = key === "mape" ? "%" : "";
                  const precision = key === "r_squared" ? 4 : 2;
                  return (
                    <div
                      class={`rounded-lg border p-3 ${d != null && improved ? "border-emerald-200 bg-emerald-50/40" : "border-slate-200"}`}
                    >
                      <p class="text-[11px] text-slate-500 uppercase tracking-wide">{key.replace("_", " ")}</p>
                      <div class="flex items-baseline justify-between mt-1.5 gap-2">
                        <span class="text-sm font-bold tabular-nums text-slate-800">
                          {va != null ? va.toFixed(precision) + suffix : "\u2014"}
                        </span>
                        <span class="text-[10px] text-slate-300 font-medium">vs</span>
                        <span class="text-sm font-bold tabular-nums text-indigo-700">
                          {vb != null ? vb.toFixed(precision) + suffix : "\u2014"}
                        </span>
                      </div>
                      <Show when={d != null}>
                        <p class={`mt-1 text-[11px] font-medium tabular-nums ${improved ? "text-emerald-600" : "text-slate-500"}`}>
                          {d >= 0 ? "+" : ""}{key === "mape" ? d.toFixed(2) + "%" : d.toFixed(4)}
                        </p>
                      </Show>
                    </div>
                  );
                }}
              </For>
            </div>
          </div>
        </Show>

        {/* Coefficient change chart */}
        <Show when={coeffDiff.length > 0}>
          <div>
            <h3 class="text-sm font-semibold text-slate-700 mb-3">Coefficient Change (B - A)</h3>
            <ResponsiveContainer width="100%" height={Math.max(200, coeffDiff.length * 36)}>
              <BarChart data={coeffDiff} layout="vertical" margin={{ left: 90, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                  formatter={(v: number) => v.toFixed(6)}
                />
                <Bar dataKey="diff" radius={[0, 4, 4, 0]}>
                  {coeffDiff.map((entry, i) => (
                    <Cell key={i} fill={entry.diff >= 0 ? "#10b981" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Show>

        {/* Allocation overlay chart */}
        <Show when={allocOverlay.length > 0}>
          <div>
            <h3 class="text-sm font-semibold text-slate-700 mb-3">Allocation Overlay</h3>
            <ResponsiveContainer width="100%" height={Math.max(220, allocOverlay.length * 40)}>
              <BarChart data={allocOverlay} layout="vertical" margin={{ left: 90, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ background: CHART_TOOLTIP_BG, border: "none", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }}
                  formatter={(v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                />
                <Bar dataKey="Run A" fill="#94a3b8" radius={[0, 3, 3, 0]} barSize={14} />
                <Bar dataKey="Run B" fill="#6366f1" radius={[0, 3, 3, 0]} barSize={14} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Show>

        {/* Allocation detail table */}
        <Show when={allocDiffRows.length > 0}>
          <div>
            <h3 class="text-sm font-semibold text-slate-700 mb-3">Allocation Detail</h3>
            <div class="overflow-x-auto rounded-lg border border-slate-200">
              <table class="w-full text-sm">
                <thead>
                  <tr class="bg-slate-50 border-b border-slate-200">
                    <th class="text-left py-2 px-3 font-semibold text-slate-600">Channel</th>
                    <th class="text-right py-2 px-3 font-semibold text-slate-600">Run A</th>
                    <th class="text-right py-2 px-3 font-semibold text-slate-600">Run B</th>
                    <th class="text-right py-2 px-3 font-semibold text-slate-600">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={allocDiffRows}>
                    {(row) => (
                      <tr class="border-b border-slate-100">
                        <td class="py-2 px-3 font-medium text-slate-700">{row.channel}</td>
                        <td class="text-right py-2 px-3 tabular-nums">${row.a.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td class="text-right py-2 px-3 tabular-nums text-indigo-700">${row.b.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td class={`text-right py-2 px-3 tabular-nums font-medium ${row.diff > 0 ? "text-emerald-600" : row.diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                          {row.diff > 0 ? "+" : ""}${Math.abs(row.diff).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </div>
        </Show>

        {/* Contribution shift */}
        <Show when={Object.keys(contributionDiff).length > 0}>
          <div>
            <h3 class="text-sm font-semibold text-slate-700 mb-3">Contribution Shift (B - A)</h3>
            <div class="overflow-x-auto rounded-lg border border-slate-200">
              <table class="w-full text-sm">
                <thead>
                  <tr class="bg-slate-50 border-b border-slate-200">
                    <th class="text-left py-2 px-3 font-semibold text-slate-600">Channel</th>
                    <th class="text-right py-2 px-3 font-semibold text-slate-600">Diff</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={Object.entries(contributionDiff).sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))}>
                    {([ch, diff]) => (
                      <tr class="border-b border-slate-100">
                        <td class="py-2 px-3 font-medium text-slate-700">{ch.replace(/_spend$/, "")}</td>
                        <td class={`text-right py-2 px-3 tabular-nums font-medium ${diff > 0 ? "text-emerald-600" : diff < 0 ? "text-red-600" : "text-slate-500"}`}>
                          {diff > 0 ? "+" : ""}{diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </div>
          </div>
        </Show>
      </div>
    </div>
  );
}

function StatusBadge(props: { status: string }) {
  switch (props.status) {
    case "completed":
      return (
        <span class="inline-flex items-center gap-1 text-emerald-600">
          <CheckCircle2 size={15} /> OK
        </span>
      );
    case "failed":
      return (
        <span class="inline-flex items-center gap-1 text-red-500">
          <XCircle size={15} /> Failed
        </span>
      );
    case "running":
      return (
        <span class="inline-flex items-center gap-1 text-amber-500">
          <Loader2 size={15} class="animate-spin" /> Running
        </span>
      );
    default:
      return (
        <span class="inline-flex items-center gap-1 text-slate-400">
          <Clock size={15} /> {props.status}
        </span>
      );
  }
}
