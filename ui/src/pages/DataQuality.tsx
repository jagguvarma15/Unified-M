import { useEffect, useState } from "react";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Shield,
  Database,
} from "lucide-react";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type DataQualityData, type GateResult } from "../lib/api";

function GateIcon({ gate }: { gate: GateResult }) {
  if (gate.passed) {
    return <CheckCircle className="w-5 h-5 text-green-600" />;
  }
  if (gate.severity === "warning") {
    return <AlertTriangle className="w-5 h-5 text-amber-500" />;
  }
  return <XCircle className="w-5 h-5 text-red-600" />;
}

function GateBadge({ severity, passed }: { severity: string; passed: boolean }) {
  if (passed) {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
        PASS
      </span>
    );
  }
  if (severity === "warning") {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
        WARN
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
      FAIL
    </span>
  );
}

export default function DataQuality() {
  const [data, setData] = useState<DataQualityData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api
      .dataQuality()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <EmptyState
        icon={<AlertTriangle className="w-10 h-10 text-amber-400" />}
        title="Data quality report unavailable"
        description={error}
      />
    );
  }

  if (!data) {
    return (
      <EmptyState
        icon={<Database className="w-10 h-10 text-gray-400" />}
        title="No data quality report"
        description="Run the pipeline to generate a data quality report."
      />
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Data Quality</h1>

      {/* Summary banner */}
      <div
        className={`rounded-xl p-5 flex items-center gap-4 ${
          data.overall_pass
            ? "bg-green-50 border border-green-200"
            : "bg-red-50 border border-red-200"
        }`}
      >
        {data.overall_pass ? (
          <Shield className="w-8 h-8 text-green-600" />
        ) : (
          <XCircle className="w-8 h-8 text-red-600" />
        )}
        <div>
          <p className="text-lg font-semibold">
            {data.overall_pass
              ? "All quality gates passed"
              : `${data.n_failed} gate(s) failed`}
          </p>
          <p className="text-sm text-gray-600">
            Checked at {new Date(data.timestamp).toLocaleString()} &middot;{" "}
            {data.n_passed} passed, {data.n_warnings} warnings,{" "}
            {data.n_failed} failed
          </p>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          icon={<CheckCircle className="w-5 h-5 text-green-600" />}
          label="Passed"
          value={data.n_passed}
        />
        <MetricCard
          icon={<AlertTriangle className="w-5 h-5 text-amber-500" />}
          label="Warnings"
          value={data.n_warnings}
        />
        <MetricCard
          icon={<XCircle className="w-5 h-5 text-red-600" />}
          label="Failed"
          value={data.n_failed}
        />
      </div>

      {/* Gate details */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 divide-y divide-gray-200">
        {data.gates.map((gate) => (
          <div key={gate.gate_name}>
            <button
              className="w-full px-5 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
              onClick={() =>
                setExpanded(
                  expanded === gate.gate_name ? null : gate.gate_name
                )
              }
            >
              <div className="flex items-center gap-3">
                <GateIcon gate={gate} />
                <div className="text-left">
                  <p className="text-sm font-semibold text-gray-900">
                    {gate.gate_name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                  </p>
                  <p className="text-xs text-gray-500">{gate.message}</p>
                </div>
              </div>
              <GateBadge severity={gate.severity} passed={gate.passed} />
            </button>
            {expanded === gate.gate_name && (
              <div className="px-5 pb-4 bg-gray-50">
                <pre className="text-xs text-gray-700 bg-white rounded border border-gray-200 p-3 overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(gate.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
