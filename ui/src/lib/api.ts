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
// API
// ---------------------------------------------------------------------------

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
  triggerPipeline: (model = "builtin", target = "revenue") => {
    const formData = new FormData();
    formData.append("model", model);
    formData.append("target", target);
    return fetch(`${BASE}/api/v1/pipeline/run`, { method: "POST", body: formData }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json() as Promise<{ run_id: string; metrics: { mape?: number; r_squared?: number } }>;
    });
  },
  calibration: () => get<CalibrationData>("/api/v1/calibration"),
  stability: () => get<StabilityData>("/api/v1/stability"),
  dataQuality: () => get<DataQualityData>("/api/v1/data-quality"),
};
