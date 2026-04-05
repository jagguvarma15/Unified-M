import { createMutation, createQuery } from "@tanstack/solid-query";
import { api } from "./api";
import { qk } from "./queryKeys";

export function useHealthQuery() {
  return createQuery(() => ({
    queryKey: qk.health,
    queryFn: api.health,
    refetchInterval: 10_000,
  }));
}

export function useRunsQuery(limit = 20) {
  return createQuery(() => ({
    queryKey: qk.runs(limit),
    queryFn: ({ signal }) => api.runs(limit, signal),
    refetchInterval: 15_000,
    staleTime: 0,
    refetchOnMount: true,
  }));
}

export function useContributionsQuery() {
  return createQuery(() => ({
    queryKey: qk.contributions,
    queryFn: api.contributions,
  }));
}

export function useReconciliationQuery() {
  return createQuery(() => ({
    queryKey: qk.reconciliation,
    queryFn: api.reconciliation,
  }));
}

export function useOptimizationQuery() {
  return createQuery(() => ({
    queryKey: qk.optimization,
    queryFn: api.optimization,
  }));
}

export function useWaterfallQuery() {
  return createQuery(() => ({
    queryKey: qk.waterfall,
    queryFn: api.waterfall,
  }));
}

export function useDiagnosticsQuery() {
  return createQuery(() => ({
    queryKey: qk.diagnostics,
    queryFn: api.diagnostics,
  }));
}

export function useRoasQuery() {
  return createQuery(() => ({
    queryKey: qk.roas,
    queryFn: api.roas,
  }));
}

export function useDataStatusQuery() {
  return createQuery(() => ({
    queryKey: qk.dataStatus,
    queryFn: api.dataStatus,
    refetchInterval: 15_000,
  }));
}

export function useCompareRunsMutation() {
  return createMutation(() => ({
    mutationFn: ({ runA, runB }: { runA: string; runB: string }) => api.compareRuns(runA, runB),
  }));
}

export function useUploadFileMutation() {
  return createMutation(() => ({
    mutationFn: ({ dataType, file }: { dataType: string; file: File }) => api.uploadFile(dataType, file),
  }));
}

export function useTriggerPipelineMutation() {
  return createMutation(() => ({
    mutationFn: ({ model, target }: { model?: string; target?: string }) =>
      api.triggerPipeline(model ?? "builtin", target ?? "revenue"),
  }));
}
