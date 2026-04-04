import { createSignal, Show, For } from "solid-js";
import { useQueryClient } from "@tanstack/solid-query";
import {
  Upload,
  CheckCircle2,
  XCircle,
  Play,
  Loader2,
  FileText,
  AlertCircle,
} from "../lib/icons";
import type { DataSourceStatus } from "../lib/api";
import { qk } from "../lib/queryKeys";
import { useDataStatusQuery, useTriggerPipelineMutation, useUploadFileMutation } from "../lib/queries";

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
  const [uploading, setUploading] = createSignal<string | null>(null);
  const [running, setRunning] = createSignal(false);
  const [runResult, setRunResult] = createSignal<string | null>(null);
  const fileInputs: Record<string, HTMLInputElement | null> = {};
  const queryClient = useQueryClient();
  const statusQuery = useDataStatusQuery();
  const uploadFile = useUploadFileMutation();
  const triggerPipeline = useTriggerPipelineMutation();

  const status = () => statusQuery.data;
  const loading = () => statusQuery.isLoading;

  const handleFileSelect = async (
    dataType: string,
    event: Event & { target: HTMLInputElement },
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(dataType);
    setRunResult(null);

    try {
      await uploadFile.mutateAsync({ dataType, file });
      await statusQuery.refetch();
      await queryClient.invalidateQueries({ queryKey: qk.dataStatus });
    } catch (err) {
      setRunResult(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(null);
      if (fileInputs[dataType]) {
        fileInputs[dataType]!.value = "";
      }
    }
  };

  const handleRunPipeline = async () => {
    const s = status();
    if (!s) return;

    const requiredKeys = Object.keys(KNOWN_DATA_TYPES).filter(
      (k) => KNOWN_DATA_TYPES[k].required,
    );
    const missing = requiredKeys.filter(
      (k) => !(s[k] as DataSourceStatus | undefined)?.exists,
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
      const result = await triggerPipeline.mutateAsync({ model: "builtin", target: "revenue" });
      setRunResult(`Pipeline job started (job: ${result.job_id}). Track progress via the Run Pipeline panel.`);
      setTimeout(() => {
        void statusQuery.refetch();
      }, 5000);
    } catch (err) {
      setRunResult(`Pipeline failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRunning(false);
    }
  };

  const allKeys = () => (status() ? Object.keys(status()!) : []);
  const grouped = () => ({
    required: allKeys().filter((k) => getTypeInfo(k).group === "required"),
    optional: allKeys().filter((k) => getTypeInfo(k).group === "optional"),
    custom: allKeys().filter((k) => getTypeInfo(k).group === "custom"),
  });

  const hasRequired = () =>
    status()
      ? grouped().required.every((k) => (status()![k] as DataSourceStatus | undefined)?.exists)
      : false;

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <div>
        <div class="flex items-center justify-between mb-6">
          <div>
            <h1 class="text-2xl font-bold text-slate-900">Data Management</h1>
            <p class="text-sm text-slate-500 mt-1">
              Upload and manage your data sources for the pipeline
            </p>
          </div>
          <button
            onClick={handleRunPipeline}
            disabled={!hasRequired() || running()}
            class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Show when={running()} fallback={<><Play size={18} />Run Pipeline</>}>
              <><Loader2 size={18} class="animate-spin" />Running...</>
            </Show>
          </button>
        </div>

        <Show when={runResult()}>
          <div
            class={`mb-6 p-4 rounded-lg ${
              runResult()!.includes("failed") || runResult()!.includes("Missing")
                ? "bg-red-50 text-red-800 border border-red-200"
                : "bg-emerald-50 text-emerald-800 border border-emerald-200"
            }`}
          >
            <pre class="text-sm whitespace-pre-wrap font-mono">{runResult()}</pre>
          </div>
        </Show>

        {/* Sections: Required / Optional / Custom */}
        <For each={["required", "optional", "custom"] as const}>
          {(section) => {
            const keys = () => grouped()[section];
            const sectionLabel =
              section === "required" ? "Required Data Sources" :
              section === "optional" ? "Optional Data Sources" :
              "Custom Data Sources";

            return (
              <Show when={keys().length > 0}>
                <div class="mb-8">
                  <h2 class="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
                    {sectionLabel}
                  </h2>
                  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <For each={keys()}>
                      {(key) => {
                        const info = getTypeInfo(key);
                        const sourceStatus = () => status()?.[key] as DataSourceStatus | undefined;
                        const exists = () => sourceStatus()?.exists ?? false;
                        const isUploading = () => uploading() === key;

                        return (
                          <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
                            <div class="flex items-start justify-between mb-3">
                              <div class="flex items-center gap-2">
                                <FileText size={20} class="text-slate-600" />
                                <h3 class="font-semibold text-slate-900">{info.label}</h3>
                                <Show when={info.required}>
                                  <span class="text-xs px-2 py-0.5 bg-red-100 text-red-700 rounded-full">
                                    Required
                                  </span>
                                </Show>
                                <Show when={section === "custom"}>
                                  <span class="text-xs px-2 py-0.5 bg-violet-100 text-violet-700 rounded-full">
                                    Custom
                                  </span>
                                </Show>
                              </div>
                              <Show when={exists()} fallback={<XCircle size={20} class="text-slate-300" />}>
                                <CheckCircle2 size={20} class="text-emerald-500" />
                              </Show>
                            </div>

                            <p class="text-xs text-slate-500 mb-4">{info.description}</p>

                            <Show when={exists() && sourceStatus()}>
                              <div class="mb-4 p-3 bg-slate-50 rounded-lg text-xs">
                                <div class="grid grid-cols-2 gap-2">
                                  <div>
                                    <span class="text-slate-500">Rows:</span>{" "}
                                    <span class="font-medium">{sourceStatus()!.rows?.toLocaleString()}</span>
                                  </div>
                                  <div>
                                    <span class="text-slate-500">Size:</span>{" "}
                                    <span class="font-medium">
                                      {sourceStatus()!.size_bytes
                                        ? `${(sourceStatus()!.size_bytes! / 1024).toFixed(1)} KB`
                                        : "—"}
                                    </span>
                                  </div>
                                </div>
                                <Show when={sourceStatus()!.columns}>
                                  <div class="mt-2">
                                    <span class="text-slate-500">Columns:</span>{" "}
                                    <span class="font-mono text-xs">
                                      {sourceStatus()!.columns!.slice(0, 5).join(", ")}
                                      {sourceStatus()!.columns!.length > 5 && "..."}
                                    </span>
                                  </div>
                                </Show>
                              </div>
                            </Show>

                            <Show when={sourceStatus()?.error}>
                              <div class="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 flex items-center gap-2">
                                <AlertCircle size={14} />
                                {sourceStatus()!.error}
                              </div>
                            </Show>

                            <label class="block">
                              <input
                                ref={(el) => { fileInputs[key] = el; }}
                                type="file"
                                accept=".csv,.parquet"
                                onChange={(e) => handleFileSelect(key, e)}
                                class="hidden"
                                disabled={isUploading()}
                              />
                              <div
                                class={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg border-2 border-dashed transition-colors cursor-pointer ${
                                  isUploading()
                                    ? "border-indigo-300 bg-indigo-50"
                                    : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50"
                                }`}
                                onClick={() => fileInputs[key]?.click()}
                              >
                                <Show
                                  when={isUploading()}
                                  fallback={
                                    <>
                                      <Upload size={16} class="text-slate-600" />
                                      <span class="text-sm font-medium text-slate-700">
                                        {exists() ? "Replace" : "Upload"} File
                                      </span>
                                    </>
                                  }
                                >
                                  <Loader2 size={16} class="animate-spin text-indigo-600" />
                                  <span class="text-sm font-medium text-indigo-600">
                                    Uploading...
                                  </span>
                                </Show>
                              </div>
                            </label>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>
              </Show>
            );
          }}
        </For>

        {/* Instructions */}
        <div class="mt-4 bg-slate-50 rounded-xl p-6 border border-slate-200">
          <h3 class="font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <AlertCircle size={18} class="text-slate-600" />
            Data Format Requirements
          </h3>
          <ul class="text-sm text-slate-600 space-y-2">
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
    </Show>
  );
}
