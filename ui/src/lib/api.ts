/**
 * API client for Unified-M backend.
 * All endpoints are read-only from the artifact store unless noted.
 */

const BASE = "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Types (aligned with backend responses)
// ---------------------------------------------------------------------------

export interface HealthData {
  status: string;
  timestamp: string;
  latest_run: string | null;
  version: string;
  cache?: Record<string, unknown>;
}

export interface RunManifest {
  run_id: string;
  status: string;
  model_backend: string;
  n_rows: number;
  n_channels: number;
  metrics?: { r_squared?: number; mape?: number; rmse?: number };
  duration_seconds?: number;
}

export interface RunsData {
  runs: RunManifest[];
}

export interface ContributionsData {
  data: Record<string, unknown>[];
  n_rows?: number;
}

export interface ReconciliationData {
  channel_estimates: Record<
    string,
    {
      lift_estimate: number;
      ci_lower: number;
      ci_upper: number;
      confidence_score: number;
    }
  >;
}

export interface OptimizationData {
  current_allocation: Record<string, number>;
  optimal_allocation: Record<string, number>;
  total_budget: number;
  current_response?: number;
  expected_response: number;
  improvement_pct?: number;
}

export interface ResponseCurveChannel {
  spend?: number[];
  response?: number[];
  marginal_response?: number[];
}

export interface ResponseCurvesData {
  [channel: string]: ResponseCurveChannel;
}

export interface ParametersData {
  coefficients?: Record<string, number>;
  intercept?: number;
  adstock?: Record<string, unknown>;
  saturation?: Record<string, unknown>;
}

export interface DiagnosticsData {
  metrics: Record<string, number>;
  chart: { date: string; actual: number; predicted: number; residual?: number }[];
  residual_stats?: Record<string, number>;
}

export interface ROASData {
  channels: {
    channel: string;
    roas: number;
    total_spend: number;
    total_contribution: number;
    cpa?: number;
    marginal_roi?: number;
  }[];
  summary: { total_spend: number; total_contribution: number; blended_roas: number };
}

export interface WaterfallData {
  baseline: number;
  channels: { name: string; value: number }[];
  total: number;
}

export interface DataSourceStatus {
  key: string;
  connected: boolean;
  path?: string;
  n_rows?: number;
  rows?: number;
  last_updated?: string;
  exists?: boolean;
  size_bytes?: number;
  columns?: string[];
  error?: string;
}

export interface DataStatus {
  media_spend?: DataSourceStatus;
  outcomes?: DataSourceStatus;
  controls?: DataSourceStatus;
  incrementality_tests?: DataSourceStatus;
  attribution?: DataSourceStatus;
  [key: string]: DataSourceStatus | undefined;
}

export interface CalibrationPoint {
  test_id?: string;
  channel?: string;
  measured_lift?: number;
  predicted_lift?: number;
  error_pct?: number;
  within_ci?: boolean;
  [key: string]: unknown;
}

export interface CalibrationData {
  n_tests: number;
  points: CalibrationPoint[];
  coverage?: number;
  median_lift_error?: number;
  mean_lift_error?: number;
  calibration_quality?: string;
}

export interface StabilityData {
  allocation_change_pct?: number;
  channel_changes?: Record<string, { change_pct: number }>;
  is_stable?: boolean;
  max_change_pct?: number;
  alert_threshold_pct?: number;
  n_drift_alerts?: number;
  alerts?: Array<{ channel: string; previous: number; current: number; delta_sigma: number }>;
  recommendation_stability?: {
    channel_changes: Record<string, { change_pct: number }>;
    is_stable: boolean;
    max_change_pct: number;
    alert_threshold_pct: number;
  };
  parameter_drift?: {
    n_drift_alerts: number;
    alerts: Array<{ channel: string; previous: number; current: number; delta_sigma: number }>;
  };
  contribution_stability?: Record<string, number>;
  [key: string]: unknown;
}

export interface GateResult {
  name: string;
  gate_name?: string;
  passed: boolean;
  severity: string;
  message?: string;
  details?: unknown;
}

export interface DataQualityData {
  timestamp: string;
  overall_pass: boolean;
  n_passed: number;
  n_failed: number;
  n_warnings: number;
  gates: GateResult[];
}

// ---------------------------------------------------------------------------
// Channel Insights
// ---------------------------------------------------------------------------

export interface ChannelInsight {
  channel: string;
  current_spend: number;
  optimal_spend: number;
  marginal_roi: number;
  saturation_point: number;
  headroom_pct: number;
  status: "under-invested" | "efficient" | "over-saturated";
  coefficient: number;
}

export interface ChannelInsightsData {
  channels: ChannelInsight[];
}

// ---------------------------------------------------------------------------
// Spend Pacing
// ---------------------------------------------------------------------------

export interface PacingChannel {
  channel: string;
  planned: number;
  actual: number;
  diff: number;
  pacing_pct: number;
  status: "on-track" | "over" | "under";
}

export interface SpendPacingData {
  total_planned: number;
  total_actual: number;
  pacing_pct: number;
  channels: PacingChannel[];
  cumulative: { date: string; actual: number }[];
}

// ---------------------------------------------------------------------------
// Executive Report
// ---------------------------------------------------------------------------

export interface ReportSummaryData {
  run_id: string | null;
  generated_at: string;
  metrics: Record<string, number>;
  roas_summary: Record<string, number>;
  top_channels: { channel: string; contribution: number; share_pct: number }[];
  recommendations: string[];
  improvement_pct: number;
}

// ---------------------------------------------------------------------------
// Run Comparison (advanced, verifiable)
// ---------------------------------------------------------------------------

export interface RunComparisonVerification {
  run_a: string;
  run_b: string;
  timestamp_a: string;
  timestamp_b: string;
  data_hash_a: string;
  data_hash_b: string;
  data_hash_changed: boolean;
  model_backend_a: string;
  model_backend_b: string;
  model_backend_changed: boolean;
}

export interface RunComparisonData {
  run_a: string;
  run_b: string;
  verification?: RunComparisonVerification;
  config_changes?: Record<string, { before: unknown; after: unknown }>;
  n_rows_a?: number;
  n_rows_b?: number;
  n_rows_change?: number;
  n_channels_a?: number;
  n_channels_b?: number;
  n_channels_change?: number;
  metrics_a?: Record<string, number>;
  metrics_b?: Record<string, number>;
  metrics_delta?: Record<string, number>;
  coefficients_a?: Record<string, number>;
  coefficients_b?: Record<string, number>;
  coefficient_diff?: Record<string, number>;
  allocation_a?: Record<string, number>;
  allocation_b?: Record<string, number>;
  current_allocation_a?: Record<string, number>;
  current_allocation_b?: Record<string, number>;
  allocation_diff?: Record<string, number>;
  contribution_totals_a?: Record<string, number>;
  contribution_totals_b?: Record<string, number>;
  contribution_diff?: Record<string, number>;
  data_hash_changed?: boolean;
  model_backend_changed?: boolean;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Pipeline Jobs
// ---------------------------------------------------------------------------

export interface PipelineJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  current_step: string;
  progress_pct: number;
  logs: string[];
  error: string | null;
  run_id: string | null;
  metrics: Record<string, number>;
  created_at: string;
  finished_at: string | null;
}

// ---------------------------------------------------------------------------
// Connectors
// ---------------------------------------------------------------------------

export interface SavedConnector {
  id: string;
  name: string;
  type: string;
  subtype: string;
  config?: Record<string, unknown>;
  created_at: string;
  last_tested: string | null;
  status: "untested" | "connected" | "failed";
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

export interface AdapterBackend {
  name: string;
  available: boolean;
  install_hint: string | null;
}

export interface AdaptersData {
  model_backends: AdapterBackend[];
  connectors: {
    database: string[];
    cloud: string[];
    ad_platforms: string[];
  };
  cache: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

function postForm<T>(path: string, data: Record<string, string>): Promise<T> {
  const formData = new FormData();
  for (const [k, v] of Object.entries(data)) {
    if (v != null) formData.append(k, v);
  }
  return fetch(`${BASE}${path}`, { method: "POST", body: formData }).then((r) => {
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
  });
}

export const api = {
  health: () => get<HealthData>("/health"),

  runs: (limit = 20) => get<RunsData>(`/api/v1/runs?limit=${limit}`),

  contributions: () => get<ContributionsData>("/api/v1/contributions"),
  reconciliation: () => get<ReconciliationData>("/api/v1/reconciliation"),
  optimization: () => get<OptimizationData>("/api/v1/optimization"),
  responseCurves: (channel?: string) =>
    get<ResponseCurvesData>(
      channel ? `/api/v1/response-curves?channel=${encodeURIComponent(channel)}` : "/api/v1/response-curves"
    ),
  parameters: () => get<ParametersData>("/api/v1/parameters"),
  diagnostics: () => get<DiagnosticsData>("/api/v1/diagnostics"),
  roas: () => get<ROASData>("/api/v1/roas"),
  waterfall: () => get<WaterfallData>("/api/v1/waterfall"),

  dataStatus: () => get<DataStatus>("/api/v1/data/status"),
  uploadFile: (dataType: string, file: File) => {
    const formData = new FormData();
    formData.append("data_type", dataType);
    formData.append("file", file);
    return fetch(`${BASE}/api/v1/data/upload`, { method: "POST", body: formData }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    });
  },

  // Pipeline jobs (async)
  triggerPipeline: (model = "builtin", target = "revenue") =>
    postForm<{ job_id: string; status: string }>("/api/v1/pipeline/run", { model, target }),
  listJobs: (limit = 20) => get<{ jobs: PipelineJob[] }>(`/api/v1/pipeline/jobs?limit=${limit}`),
  getJob: (jobId: string) => get<PipelineJob>(`/api/v1/pipeline/jobs/${encodeURIComponent(jobId)}`),

  // Connectors CRUD
  listConnectors: () => get<{ connectors: SavedConnector[] }>("/api/v1/connectors"),
  getConnector: (id: string) => get<SavedConnector>(`/api/v1/connectors/${id}`),
  createConnector: (name: string, type: string, subtype: string, config: Record<string, unknown>) =>
    postForm<SavedConnector>("/api/v1/connectors", {
      name,
      connector_type: type,
      subtype,
      connector_config: JSON.stringify(config),
    }),
  updateConnector: (id: string, name?: string, config?: Record<string, unknown>) => {
    const formData = new FormData();
    if (name != null) formData.append("name", name);
    if (config != null) formData.append("connector_config", JSON.stringify(config));
    return fetch(`${BASE}/api/v1/connectors/${id}`, { method: "PUT", body: formData }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json() as Promise<SavedConnector>;
    });
  },
  deleteConnector: (id: string) =>
    fetch(`${BASE}/api/v1/connectors/${id}`, { method: "DELETE" }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    }),
  testConnector: (id: string) =>
    postForm<{ status: string; connected: boolean; message: string }>(
      `/api/v1/connectors/${id}/test`,
      {},
    ),
  fetchFromConnector: (id: string, queryOrPath: string, dataType: string) =>
    postForm<{ status: string; rows: number; columns: string[]; data_type: string }>(
      `/api/v1/connectors/${id}/fetch`,
      { query_or_path: queryOrPath, data_type: dataType },
    ),

  // Adapters
  adapters: () => get<AdaptersData>("/api/v1/adapters"),

  calibration: () => get<CalibrationData>("/api/v1/calibration"),
  stability: () => get<StabilityData>("/api/v1/stability"),
  dataQuality: () => get<DataQualityData>("/api/v1/data-quality"),
  channelInsights: () => get<ChannelInsightsData>("/api/v1/channel-insights"),
  spendPacing: () => get<SpendPacingData>("/api/v1/spend-pacing"),
  reportSummary: () => get<ReportSummaryData>("/api/v1/report/summary"),
  compareRuns: (runA: string, runB: string) =>
    get<RunComparisonData>(`/api/v1/compare-runs?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`),
};
