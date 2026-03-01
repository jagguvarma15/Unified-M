import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { qk } from "./queryKeys";

export function useHealthQuery() {
  return useQuery({
    queryKey: qk.health,
    queryFn: api.health,
    refetchInterval: 10_000,
  });
}

export function useRunsQuery(limit = 20) {
  return useQuery({
    queryKey: qk.runs(limit),
    queryFn: () => api.runs(limit),
  });
}

export function useContributionsQuery() {
  return useQuery({
    queryKey: qk.contributions,
    queryFn: api.contributions,
  });
}

export function useReconciliationQuery() {
  return useQuery({
    queryKey: qk.reconciliation,
    queryFn: api.reconciliation,
  });
}

export function useOptimizationQuery() {
  return useQuery({
    queryKey: qk.optimization,
    queryFn: api.optimization,
  });
}

export function useWaterfallQuery() {
  return useQuery({
    queryKey: qk.waterfall,
    queryFn: api.waterfall,
  });
}

export function useDiagnosticsQuery() {
  return useQuery({
    queryKey: qk.diagnostics,
    queryFn: api.diagnostics,
  });
}

export function useRoasQuery() {
  return useQuery({
    queryKey: qk.roas,
    queryFn: api.roas,
  });
}

export function useDataStatusQuery() {
  return useQuery({
    queryKey: qk.dataStatus,
    queryFn: api.dataStatus,
  });
}

export function useCompareRunsMutation() {
  return useMutation({
    mutationFn: ({ runA, runB }: { runA: string; runB: string }) => api.compareRuns(runA, runB),
  });
}

export function useUploadFileMutation() {
  return useMutation({
    mutationFn: ({ dataType, file }: { dataType: string; file: File }) => api.uploadFile(dataType, file),
  });
}

export function useTriggerPipelineMutation() {
  return useMutation({
    mutationFn: ({ model, target }: { model?: string; target?: string }) =>
      api.triggerPipeline(model ?? "builtin", target ?? "revenue"),
  });
}

