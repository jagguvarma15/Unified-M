import { useEffect, useState, useRef } from "react";
import { Loader2, Printer, Copy, Check, TrendingUp } from "lucide-react";
import EmptyState from "../components/EmptyState";
import { api, type ReportSummaryData } from "../lib/api";

export default function Report() {
  const [data, setData] = useState<ReportSummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const printRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .reportSummary()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handlePrint = () => window.print();

  const handleCopy = async () => {
    if (!data) return;
    const lines: string[] = [
      "UNIFIED-M EXECUTIVE SUMMARY",
      `Generated: ${new Date(data.generated_at).toLocaleDateString()}`,
      `Run: ${data.run_id || "—"}`,
      "",
      "KEY METRICS",
    ];
    if (data.metrics.r_squared != null) lines.push(`  R²: ${data.metrics.r_squared.toFixed(3)}`);
    if (data.metrics.mape != null) lines.push(`  MAPE: ${data.metrics.mape.toFixed(1)}%`);
    if (data.roas_summary?.blended_roas != null) lines.push(`  Blended ROAS: ${data.roas_summary.blended_roas.toFixed(2)}`);
    if (data.roas_summary?.total_spend != null) lines.push(`  Total Spend: $${data.roas_summary.total_spend.toLocaleString()}`);
    if (data.improvement_pct) lines.push(`  Optimization Uplift: +${data.improvement_pct.toFixed(1)}%`);
    lines.push("");
    if (data.top_channels.length) {
      lines.push("TOP CHANNELS");
      data.top_channels.forEach((c) => lines.push(`  ${c.channel}: ${c.share_pct}% share ($${c.contribution.toLocaleString()})`));
      lines.push("");
    }
    if (data.recommendations.length) {
      lines.push("KEY RECOMMENDATIONS");
      data.recommendations.forEach((r) => lines.push(`  • ${r}`));
    }
    await navigator.clipboard.writeText(lines.join("\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!data || (!data.run_id && !data.top_channels.length)) {
    return (
      <EmptyState
        title="No report available"
        message="Run the pipeline to generate an executive summary."
      />
    );
  }

  const metrics = data.metrics || {};
  const roas = data.roas_summary || {};

  return (
    <div>
      {/* Toolbar (hidden on print) */}
      <div className="flex items-center justify-between mb-6 print:hidden">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Executive Summary</h1>
          <p className="mt-1 text-sm text-slate-500">One-click report for stakeholders</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
          >
            {copied ? <Check size={15} className="text-emerald-500" /> : <Copy size={15} />}
            {copied ? "Copied" : "Copy as Text"}
          </button>
          <button
            onClick={handlePrint}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <Printer size={15} />
            Print
          </button>
        </div>
      </div>

      {/* Report content */}
      <div ref={printRef} className="space-y-6">
        {/* Header */}
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-4">
            <div className="rounded-lg bg-indigo-100 p-2">
              <TrendingUp size={20} className="text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">Unified-M Marketing Report</h2>
              <p className="text-xs text-slate-500">
                {new Date(data.generated_at).toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
                {data.run_id && <> · Run <span className="font-mono">{data.run_id.slice(0, 12)}</span></>}
              </p>
            </div>
          </div>

          {/* Metric pills */}
          <div className="flex flex-wrap gap-3">
            {metrics.r_squared != null && (
              <MetricPill label="R²" value={metrics.r_squared.toFixed(3)} />
            )}
            {metrics.mape != null && (
              <MetricPill label="MAPE" value={`${metrics.mape.toFixed(1)}%`} />
            )}
            {roas.blended_roas != null && (
              <MetricPill label="Blended ROAS" value={roas.blended_roas.toFixed(2)} />
            )}
            {roas.total_spend != null && (
              <MetricPill label="Total Spend" value={`$${roas.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            )}
            {roas.total_contribution != null && (
              <MetricPill label="Total Contribution" value={`$${roas.total_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            )}
            {data.improvement_pct > 0 && (
              <MetricPill label="Optimization Uplift" value={`+${data.improvement_pct.toFixed(1)}%`} accent />
            )}
          </div>
        </div>

        {/* Top channels */}
        {data.top_channels.length > 0 && (
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-sm">
            <h3 className="text-base font-semibold text-slate-800 mb-4">Top Channels by Contribution</h3>
            <div className="space-y-3">
              {data.top_channels.map((ch) => (
                <div key={ch.channel} className="flex items-center gap-3">
                  <span className="w-32 text-sm text-slate-700 truncate font-medium">
                    {ch.channel.replace(/_spend$/, "")}
                  </span>
                  <div className="flex-1 h-5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full transition-all"
                      style={{ width: `${ch.share_pct}%` }}
                    />
                  </div>
                  <span className="w-16 text-right text-sm tabular-nums font-semibold text-slate-800">
                    {ch.share_pct}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        {data.recommendations.length > 0 && (
          <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-6 shadow-sm">
            <h3 className="text-base font-semibold text-indigo-900 mb-3">Key Recommendations</h3>
            <ul className="space-y-2">
              {data.recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-indigo-800">
                  <span className="mt-0.5 flex-shrink-0 h-5 w-5 rounded-full bg-indigo-200 text-indigo-700 text-xs font-bold flex items-center justify-center">
                    {i + 1}
                  </span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricPill({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`rounded-lg border px-3 py-2 ${accent ? "border-indigo-200 bg-indigo-50" : "border-slate-200 bg-slate-50"}`}>
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className={`text-sm font-bold tabular-nums ${accent ? "text-indigo-700" : "text-slate-800"}`}>{value}</p>
    </div>
  );
}
