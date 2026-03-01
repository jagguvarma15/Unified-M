/*
 * Generated API contract.
 * Source of truth should be OpenAPI generation:
 *   bun run typegen:api
 */

type JsonBody<T> = {
  responses: {
    200: {
      content: {
        "application/json": T;
      };
    };
  };
};

export interface paths {
  "/health": { get: JsonBody<{ status: string; timestamp: string; latest_run: string | null; version: string; cache?: Record<string, unknown> }> };
  "/": { get: JsonBody<{ name: string; version: string; docs: string }> };
  "/api/v1/runs": { get: JsonBody<{ runs: Array<{ run_id: string; status: string; model_backend: string; n_rows: number; n_channels: number; metrics?: { r_squared?: number; mape?: number; rmse?: number }; duration_seconds?: number }> }> };
  "/api/v1/contributions": { get: JsonBody<{ data: Record<string, unknown>[]; n_rows?: number }> };
  "/api/v1/reconciliation": { get: JsonBody<{ channel_estimates: Record<string, { lift_estimate: number; ci_lower: number; ci_upper: number; confidence_score: number }> }> };
  "/api/v1/optimization": { get: JsonBody<{ current_allocation: Record<string, number>; optimal_allocation: Record<string, number>; total_budget: number; current_response?: number; expected_response: number; improvement_pct?: number }> };
  "/api/v1/response-curves": { get: JsonBody<Record<string, { spend?: number[]; response?: number[]; marginal_response?: number[] }>> };
  "/api/v1/parameters": { get: JsonBody<{ coefficients?: Record<string, number>; intercept?: number; adstock?: Record<string, unknown>; saturation?: Record<string, unknown>; adstock_params?: Record<string, unknown>; saturation_params?: Record<string, unknown> }> };
  "/api/v1/diagnostics": { get: JsonBody<{ metrics: Record<string, number>; chart: { date: string; actual: number; predicted: number; residual?: number }[]; residual_stats?: Record<string, number> }> };
  "/api/v1/roas": { get: JsonBody<{ channels: { channel: string; roas: number; total_spend: number; total_contribution: number; cpa?: number; marginal_roi?: number }[]; summary: { total_spend: number; total_contribution: number; blended_roas: number } }> };
  "/api/v1/waterfall": { get: JsonBody<{ baseline: number; channels: { name: string; value: number }[]; total: number }> };
  "/api/v1/data/status": { get: JsonBody<Record<string, { key?: string; connected?: boolean; path?: string; n_rows?: number; rows?: number; last_updated?: string; exists?: boolean; size_bytes?: number; columns?: string[]; error?: string }>> };
  "/api/v1/data/upload": { post: JsonBody<Record<string, unknown>> };
  "/api/v1/pipeline/run": { post: JsonBody<{ job_id: string; status: string }> };
  "/api/v1/pipeline/jobs": { get: JsonBody<{ jobs: Array<{ job_id: string; status: "pending" | "running" | "completed" | "failed"; current_step: string; progress_pct: number; logs: string[]; error: string | null; run_id: string | null; metrics: Record<string, number>; created_at: string; finished_at: string | null }> }> };
  "/api/v1/pipeline/jobs/{job_id}": { get: JsonBody<{ job_id: string; status: "pending" | "running" | "completed" | "failed"; current_step: string; progress_pct: number; logs: string[]; error: string | null; run_id: string | null; metrics: Record<string, number>; created_at: string; finished_at: string | null }> };
  "/api/v1/connectors": { get: JsonBody<{ connectors: Array<{ id: string; name: string; type: string; subtype: string; config?: Record<string, unknown>; created_at: string; last_tested: string | null; status: "untested" | "connected" | "failed" }> }>; post: JsonBody<{ id: string; name: string; type: string; subtype: string; config?: Record<string, unknown>; created_at: string; last_tested: string | null; status: "untested" | "connected" | "failed" }> };
  "/api/v1/connectors/{connector_id}": { get: JsonBody<{ id: string; name: string; type: string; subtype: string; config?: Record<string, unknown>; created_at: string; last_tested: string | null; status: "untested" | "connected" | "failed" }> };
  "/api/v1/connectors/{connector_id}/test": { post: JsonBody<{ status: string; connected: boolean; message: string }> };
  "/api/v1/connectors/{connector_id}/fetch": { post: JsonBody<{ status: string; rows: number; columns: string[]; data_type: string; path?: string | null }> };
  "/api/v1/refresh": { post: JsonBody<{ status: string }> };
  "/api/v1/adapters": { get: JsonBody<{ model_backends: Array<{ name: string; available: boolean; install_hint: string | null }>; connectors: { database: string[]; cloud: string[]; ad_platforms: string[] }; cache: Record<string, unknown> }> };
  "/api/v1/calibration": { get: JsonBody<{ n_tests: number; points: Array<Record<string, unknown>>; coverage?: number; median_lift_error?: number; mean_lift_error?: number; calibration_quality?: string }> };
  "/api/v1/stability": { get: JsonBody<Record<string, unknown>> };
  "/api/v1/data-quality": { get: JsonBody<{ timestamp: string; overall_pass: boolean; n_passed: number; n_failed: number; n_warnings: number; gates: Array<{ name: string; gate_name?: string; passed: boolean; severity: string; message?: string; details?: unknown }> }> };
  "/api/v1/channel-insights": { get: JsonBody<{ channels: Array<{ channel: string; current_spend: number; optimal_spend: number; marginal_roi: number; saturation_point: number; headroom_pct: number; status: "under-invested" | "efficient" | "over-saturated"; coefficient: number }> }> };
  "/api/v1/spend-pacing": { get: JsonBody<{ total_planned: number; total_actual: number; pacing_pct: number; channels: Array<{ channel: string; planned: number; actual: number; diff: number; pacing_pct: number; status: "on-track" | "over" | "under" }>; cumulative: { date: string; actual: number }[] }> };
  "/api/v1/report/summary": { get: JsonBody<{ run_id: string | null; generated_at: string; metrics: Record<string, number>; roas_summary: Record<string, number>; top_channels: { channel: string; contribution: number; share_pct: number }[]; recommendations: string[]; improvement_pct: number }> };
  "/api/v1/compare-runs": { get: JsonBody<Record<string, unknown>> };
}
