import { useEffect, useState, useRef } from "react";
import {
  Upload,
  CheckCircle2,
  XCircle,
  Play,
  Loader2,
  FileText,
  AlertCircle,
} from "lucide-react";
import { api, type DataStatus, type DataSourceStatus } from "../lib/api";

const KNOWN_DATA_TYPES: Record<
  string,
  { label: string; description: string; required: boolean; group: "required" | "optional" }
> = {
  media_spend: {
    label: "Media Spend",
    description: "Daily/weekly spend by channel (date, channel, spend, impressions, clicks)",
    required: true,
    group: "required",
  },
  outcomes: {
    label: "Outcomes",
    description: "Target metrics (date, revenue, conversions)",
    required: true,
    group: "required",
  },
  controls: {
    label: "Control Variables",
    description: "Non-media factors (date, seasonality, promo, etc.)",
    required: false,
    group: "optional",
  },
  incrementality_tests: {
    label: "Incrementality Tests",
    description: "Test results (test_id, channel, start_date, end_date, lift_estimate, lift_ci_lower, lift_ci_upper)",
    required: false,
    group: "optional",
  },
  attribution: {
    label: "Attribution Data",
    description: "Attributed conversions/revenue (date, channel, attributed_conversions, attributed_revenue)",
    required: false,
    group: "optional",
  },
};

function getTypeInfo(key: string) {
  if (key in KNOWN_DATA_TYPES) return KNOWN_DATA_TYPES[key];
  return {
    label: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    description: "Custom data source",
    required: false,
    group: "custom" as const,
  };
}

export default function Data() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<string | null>(null);
  const fileInputs = useRef<Record<string, HTMLInputElement | null>>({});

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const data = await api.dataStatus();
      setStatus(data);
    } catch (err) {
      console.error("Failed to load data status:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (
    dataType: string,
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(dataType);
    setRunResult(null);

    try {
      await api.uploadFile(dataType, file);
      await loadStatus(); // Refresh status
    } catch (err) {
      setRunResult(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(null);
      // Reset file input
      if (fileInputs.current[dataType]) {
        fileInputs.current[dataType]!.value = "";
      }
    }
  };

  const handleRunPipeline = async () => {
    if (!status) return;

    // Check required files
    const requiredKeys = Object.keys(KNOWN_DATA_TYPES).filter(
      (k) => KNOWN_DATA_TYPES[k].required,
    );
    const missing = requiredKeys.filter(
      (k) => !(status[k] as DataSourceStatus | undefined)?.exists,
    );

    if (missing.length > 0) {
      setRunResult(
        `Missing required data: ${missing.map((m) => KNOWN_DATA_TYPES[m]?.label ?? m).join(", ")}`,
      );
      return;
    }

    setRunning(true);
    setRunResult(null);

    try {
      const result = await api.triggerPipeline("builtin", "revenue");
      setRunResult(`Pipeline job started (job: ${result.job_id}). Track progress via the Run Pipeline panel.`);
      setTimeout(loadStatus, 5000);
    } catch (err) {
      setRunResult(`Pipeline failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  // Build grouped data sources from the dynamic status response
  const allKeys = status ? Object.keys(status) : [];
  const grouped = {
    required: allKeys.filter((k) => getTypeInfo(k).group === "required"),
    optional: allKeys.filter((k) => getTypeInfo(k).group === "optional"),
    custom: allKeys.filter((k) => getTypeInfo(k).group === "custom"),
  };

  const hasRequired = status
    ? grouped.required.every((k) => (status[k] as DataSourceStatus | undefined)?.exists)
    : false;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Data Management</h1>
          <p className="text-sm text-slate-500 mt-1">
            Upload and manage your data sources for the pipeline
          </p>
        </div>
        <button
          onClick={handleRunPipeline}
          disabled={!hasRequired || running}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {running ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Pipeline
            </>
          )}
        </button>
      </div>

      {runResult && (
        <div
          className={`mb-6 p-4 rounded-lg ${
            runResult.includes("failed") || runResult.includes("Missing")
              ? "bg-red-50 text-red-800 border border-red-200"
              : "bg-emerald-50 text-emerald-800 border border-emerald-200"
          }`}
        >
          <pre className="text-sm whitespace-pre-wrap font-mono">{runResult}</pre>
        </div>
      )}

      {/* Sections: Required / Optional / Custom */}
      {(["required", "optional", "custom"] as const).map((section) => {
        const keys = grouped[section];
        if (!keys.length) return null;
        const sectionLabel =
          section === "required" ? "Required Data Sources" :
          section === "optional" ? "Optional Data Sources" :
          "Custom Data Sources";
        return (
          <div key={section} className="mb-8">
            <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
              {sectionLabel}
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {keys.map((key) => {
                const info = getTypeInfo(key);
                const sourceStatus = status?.[key] as DataSourceStatus | undefined;
                const exists = sourceStatus?.exists ?? false;
                const isUploading = uploading === key;

                return (
                  <div
                    key={key}
                    className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <FileText size={20} className="text-slate-600" />
                        <h3 className="font-semibold text-slate-900">{info.label}</h3>
                        {info.required && (
                          <span className="text-xs px-2 py-0.5 bg-red-100 text-red-700 rounded-full">
                            Required
                          </span>
                        )}
                        {section === "custom" && (
                          <span className="text-xs px-2 py-0.5 bg-violet-100 text-violet-700 rounded-full">
                            Custom
                          </span>
                        )}
                      </div>
                      {exists ? (
                        <CheckCircle2 size={20} className="text-emerald-500" />
                      ) : (
                        <XCircle size={20} className="text-slate-300" />
                      )}
                    </div>

                    <p className="text-xs text-slate-500 mb-4">{info.description}</p>

                    {exists && sourceStatus && (
                      <div className="mb-4 p-3 bg-slate-50 rounded-lg text-xs">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <span className="text-slate-500">Rows:</span>{" "}
                            <span className="font-medium">{sourceStatus.rows?.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Size:</span>{" "}
                            <span className="font-medium">
                              {sourceStatus.size_bytes
                                ? `${(sourceStatus.size_bytes / 1024).toFixed(1)} KB`
                                : "—"}
                            </span>
                          </div>
                        </div>
                        {sourceStatus.columns && (
                          <div className="mt-2">
                            <span className="text-slate-500">Columns:</span>{" "}
                            <span className="font-mono text-xs">
                              {sourceStatus.columns.slice(0, 5).join(", ")}
                              {sourceStatus.columns.length > 5 && "..."}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {sourceStatus?.error && (
                      <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 flex items-center gap-2">
                        <AlertCircle size={14} />
                        {sourceStatus.error}
                      </div>
                    )}

                    <label className="block">
                      <input
                        ref={(el) => {
                          fileInputs.current[key] = el;
                        }}
                        type="file"
                        accept=".csv,.parquet"
                        onChange={(e) => handleFileSelect(key, e)}
                        className="hidden"
                        disabled={isUploading}
                      />
                      <div
                        className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg border-2 border-dashed transition-colors cursor-pointer ${
                          isUploading
                            ? "border-indigo-300 bg-indigo-50"
                            : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50"
                        }`}
                        onClick={() => fileInputs.current[key]?.click()}
                      >
                        {isUploading ? (
                          <>
                            <Loader2 size={16} className="animate-spin text-indigo-600" />
                            <span className="text-sm font-medium text-indigo-600">
                              Uploading...
                            </span>
                          </>
                        ) : (
                          <>
                            <Upload size={16} className="text-slate-600" />
                            <span className="text-sm font-medium text-slate-700">
                              {exists ? "Replace" : "Upload"} File
                            </span>
                          </>
                        )}
                      </div>
                    </label>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Instructions */}
      <div className="mt-4 bg-slate-50 rounded-xl p-6 border border-slate-200">
        <h3 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
          <AlertCircle size={18} className="text-slate-600" />
          Data Format Requirements
        </h3>
        <ul className="text-sm text-slate-600 space-y-2">
          <li>
            • <strong>Media Spend:</strong> Must include <code>date</code>,{" "}
            <code>channel</code>, and <code>spend</code> columns
          </li>
          <li>
            • <strong>Outcomes:</strong> Must include <code>date</code> and at least one outcome
            column (<code>revenue</code>, <code>conversions</code>, etc.)
          </li>
          <li>
            • <strong>Date columns:</strong> Should be in YYYY-MM-DD format or parseable datetime
          </li>
          <li>
            • <strong>File formats:</strong> CSV or Parquet (Parquet recommended for large files)
          </li>
          <li>
            • Files are automatically converted to Parquet and stored in{" "}
            <code>data/processed/</code>
          </li>
        </ul>
      </div>
    </div>
  );
}
