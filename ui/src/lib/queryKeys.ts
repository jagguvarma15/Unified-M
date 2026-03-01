export const qk = {
  health: ["health"] as const,
  runs: (limit: number) => ["runs", limit] as const,
  contributions: ["contributions"] as const,
  reconciliation: ["reconciliation"] as const,
  optimization: ["optimization"] as const,
  waterfall: ["waterfall"] as const,
  diagnostics: ["diagnostics"] as const,
  roas: ["roas"] as const,
  dataStatus: ["data-status"] as const,
  compareRuns: (runA: string, runB: string) => ["compare-runs", runA, runB] as const,
  pipelineJobs: (limit: number) => ["pipeline-jobs", limit] as const,
};

