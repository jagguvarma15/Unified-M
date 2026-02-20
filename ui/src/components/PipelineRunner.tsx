import { useState, useEffect, useCallback, useRef } from "react";
import {
  X,
  Play,
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  Link2,
  Database,
  ShieldCheck,
  Sparkles,
  Brain,
  GitMerge,
  Target,
  FileCheck,
} from "lucide-react";
import { api, type PipelineJob } from "../lib/api";
import { useToast } from "../lib/toast";

const STEPS = [
  { key: "connect", label: "Connect", icon: Database },
  { key: "quality_gates", label: "Quality Gates", icon: ShieldCheck },
  { key: "transform", label: "Transform", icon: Sparkles },
  { key: "train", label: "Train Model", icon: Brain },
  { key: "reconcile", label: "Reconcile", icon: GitMerge },
  { key: "optimise", label: "Optimise", icon: Target },
  { key: "finalise", label: "Finalise", icon: FileCheck },
] as const;

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function PipelineRunner({ open, onClose }: Props) {
  const [model, setModel] = useState("builtin");
  const [target, setTarget] = useState("revenue");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<PipelineJob | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const { addToast } = useToast();

  const isRunning = job?.status === "pending" || job?.status === "running";
  const isDone = job?.status === "completed" || job?.status === "failed";

  const startPipeline = useCallback(async () => {
    setStarting(true);
    try {
      const res = await api.triggerPipeline(model, target);
      setJobId(res.job_id);
      setJob(null);
      addToast("info", "Pipeline started");
    } catch (e: any) {
      addToast("error", `Failed to start pipeline: ${e.message}`);
    } finally {
      setStarting(false);
    }
  }, [model, target, addToast]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const j = await api.getJob(jobId);
        if (!cancelled) setJob(j);
        if (j.status === "completed") {
          addToast("success", `Pipeline completed (run: ${j.run_id?.slice(0, 12)})`);
        } else if (j.status === "failed") {
          addToast("error", `Pipeline failed: ${j.error || "Unknown error"}`);
        }
        if (j.status === "completed" || j.status === "failed") {
          clearInterval(pollRef.current);
        }
      } catch {
        // keep polling
      }
    };

    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(pollRef.current);
    };
  }, [jobId, addToast]);

  useEffect(() => {
    if (!open) {
      clearInterval(pollRef.current);
    }
  }, [open]);

  if (!open) return null;

  const currentIdx = job ? STEPS.findIndex((s) => s.key === job.current_step) : -1;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/20" onClick={onClose} />
      <div className="relative w-full max-w-md bg-white shadow-xl flex flex-col animate-in slide-in-from-right">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
          <h2 className="text-base font-semibold text-slate-900">Run Pipeline</h2>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:text-slate-600 transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {/* Config */}
          {!jobId && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Model backend</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="builtin">Built-in (OLS)</option>
                  <option value="pymc">PyMC-Marketing</option>
                  <option value="meridian">Google Meridian</option>
                  <option value="numpyro">NumPyro</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Target column</label>
                <select
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="revenue">Revenue</option>
                  <option value="conversions">Conversions</option>
                </select>
              </div>
              <button
                onClick={startPipeline}
                disabled={starting}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-60 transition-colors"
              >
                {starting ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                {starting ? "Starting..." : "Start Pipeline"}
              </button>
            </div>
          )}

          {/* Stepper */}
          {jobId && (
            <div className="space-y-1">
              {STEPS.map((step, i) => {
                const Icon = step.icon;
                let status: "pending" | "running" | "done" | "failed" = "pending";
                if (job) {
                  if (i < currentIdx) status = "done";
                  else if (i === currentIdx) status = isRunning ? "running" : isDone ? (job.status === "completed" ? "done" : "failed") : "running";
                  else if (isDone && job.status === "completed") status = "done";
                }

                return (
                  <div key={step.key} className="flex items-center gap-3 py-2">
                    <div className="relative flex h-8 w-8 shrink-0 items-center justify-center">
                      {status === "done" && <CheckCircle2 size={18} className="text-emerald-500" />}
                      {status === "running" && <Loader2 size={18} className="text-indigo-500 animate-spin" />}
                      {status === "failed" && <AlertCircle size={18} className="text-red-500" />}
                      {status === "pending" && <Circle size={18} className="text-slate-300" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium ${status === "done" ? "text-emerald-700" : status === "running" ? "text-indigo-700" : status === "failed" ? "text-red-700" : "text-slate-400"}`}>
                        {step.label}
                      </p>
                    </div>
                    <Icon size={14} className="shrink-0 text-slate-300" />
                  </div>
                );
              })}
            </div>
          )}

          {/* Progress bar */}
          {job && isRunning && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-slate-500">
                <span>{job.current_step || "Starting"}</span>
                <span>{job.progress_pct}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full bg-indigo-500 transition-all duration-500"
                  style={{ width: `${job.progress_pct}%` }}
                />
              </div>
            </div>
          )}

          {/* Result */}
          {job?.status === "completed" && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 space-y-2">
              <p className="text-sm font-medium text-emerald-800">Pipeline completed</p>
              {job.metrics && (
                <div className="grid grid-cols-2 gap-2 text-xs text-emerald-700">
                  {job.metrics.mape != null && <div>MAPE: {Number(job.metrics.mape).toFixed(1)}%</div>}
                  {job.metrics.r_squared != null && <div>R²: {Number(job.metrics.r_squared).toFixed(3)}</div>}
                </div>
              )}
              {job.run_id && (
                <a href="/" className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 hover:text-indigo-800">
                  <Link2 size={12} /> View Dashboard
                </a>
              )}
            </div>
          )}

          {job?.status === "failed" && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4">
              <p className="text-sm font-medium text-red-800">Pipeline failed</p>
              {job.error && <p className="mt-1 text-xs text-red-600">{job.error}</p>}
            </div>
          )}

          {/* Logs */}
          {job && job.logs.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-slate-500 mb-2">Logs</h3>
              <div className="rounded-md bg-slate-900 p-3 max-h-40 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
                {job.logs.map((log, i) => (
                  <div key={i}>{log}</div>
                ))}
              </div>
            </div>
          )}

          {/* Reset */}
          {isDone && (
            <button
              onClick={() => { setJobId(null); setJob(null); }}
              className="w-full rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Run Another
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
