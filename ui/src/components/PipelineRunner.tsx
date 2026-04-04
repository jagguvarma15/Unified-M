import { createSignal, createEffect, onMount, onCleanup, Show, For } from "solid-js";
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
} from "../lib/icons";
import { api, type PipelineJob } from "../lib/api";
import { useToast } from "../lib/toast";
import { useAnalyticsMode } from "../lib/analyticsMode";

const STEPS = [
  { key: "connect", label: "Connect", icon: Database },
  { key: "quality_gates", label: "Quality Gates", icon: ShieldCheck },
  { key: "transform", label: "Transform", icon: Sparkles },
  { key: "train", label: "Train Model", icon: Brain },
  { key: "reconcile", label: "Reconcile", icon: GitMerge },
  { key: "optimise", label: "Optimise", icon: Target },
  { key: "finalise", label: "Finalise", icon: FileCheck },
] as const;

type RunTemplate = {
  id: string;
  label: string;
  description: string;
  model: string;
  target: string;
  useSampleData: boolean;
  budget?: number;
};

const RUN_TEMPLATES: RunTemplate[] = [
  {
    id: "quick_demo",
    label: "Quick Demo",
    description: "Fast baseline run on generated sample data.",
    model: "builtin",
    target: "revenue",
    useSampleData: true,
  },
  {
    id: "revenue_planning",
    label: "Revenue Planning",
    description: "Production-style run on uploaded data with revenue target.",
    model: "builtin",
    target: "revenue",
    useSampleData: false,
  },
  {
    id: "conversion_planning",
    label: "Conversion Planning",
    description: "Optimize for conversion efficiency on uploaded data.",
    model: "builtin",
    target: "conversions",
    useSampleData: false,
  },
  {
    id: "bayesian_deep_dive",
    label: "Bayesian Deep Dive",
    description: "Higher-fidelity Bayesian run (slower) for deeper diagnostics.",
    model: "pymc",
    target: "revenue",
    useSampleData: false,
  },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function PipelineRunner(props: Props) {
  const [templateId, setTemplateId] = createSignal(RUN_TEMPLATES[0].id);
  const [model, setModel] = createSignal("builtin");
  const [target, setTarget] = createSignal("revenue");
  const [budgetInput, setBudgetInput] = createSignal("");
  const [useSampleData, setUseSampleData] = createSignal(true);
  const [jobId, setJobId] = createSignal<string | null>(null);
  const [job, setJob] = createSignal<PipelineJob | null>(null);
  const [starting, setStarting] = createSignal(false);
  let pollInterval: ReturnType<typeof setInterval> | undefined;
  const { addToast } = useToast();
  const { setAnalyticsEnabled } = useAnalyticsMode();

  const isRunning = () => job()?.status === "pending" || job()?.status === "running";
  const isDone = () => job()?.status === "completed" || job()?.status === "failed";
  const showConfig = () => !isRunning();
  const selectedTemplate = () => RUN_TEMPLATES.find((t) => t.id === templateId()) ?? RUN_TEMPLATES[0];

  const applyTemplate = (template: RunTemplate) => {
    setTemplateId(template.id);
    setModel(template.model);
    setTarget(template.target);
    setUseSampleData(template.useSampleData);
    setBudgetInput(template.budget != null ? String(template.budget) : "");
    setAnalyticsEnabled(template.useSampleData);
  };

  const startPipeline = async () => {
    setStarting(true);
    try {
      const parsedBudget = budgetInput().trim() === "" ? undefined : Number(budgetInput());
      if (parsedBudget != null && (!Number.isFinite(parsedBudget) || parsedBudget <= 0)) {
        throw new Error("Budget must be a positive number");
      }
      setAnalyticsEnabled(useSampleData());
      const res = await api.triggerPipeline(model(), target(), useSampleData(), parsedBudget);
      setJobId(res.job_id);
      setJob(null);
      addToast("info", useSampleData() ? "Sample data pipeline started" : "Pipeline started");
    } catch (e: any) {
      addToast("error", `Failed to start pipeline: ${e.message}`);
    } finally {
      setStarting(false);
    }
  };

  createEffect(() => {
    const id = jobId();
    if (!id) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const j = await api.getJob(id);
        if (!cancelled) setJob(j);
        if (j.status === "completed") {
          addToast("success", `Pipeline completed (run: ${j.run_id?.slice(0, 12)})`);
        } else if (j.status === "failed") {
          addToast("error", `Pipeline failed: ${j.error || "Unknown error"}`);
        }
        if (j.status === "completed" || j.status === "failed") {
          clearInterval(pollInterval);
        }
      } catch {
        // keep polling
      }
    };

    poll();
    pollInterval = setInterval(poll, 2000);
    onCleanup(() => {
      cancelled = true;
      clearInterval(pollInterval);
    });
  });

  createEffect(() => {
    if (!props.open) {
      clearInterval(pollInterval);
    }
  });

  const currentIdx = () => {
    const j = job();
    return j ? STEPS.findIndex((s) => s.key === j.current_step) : -1;
  };

  return (
    <Show when={props.open}>
      <div class="fixed inset-0 z-40 flex justify-end">
        <div class="absolute inset-0 bg-black/20" onClick={props.onClose} />
        <div class="relative w-full max-w-md bg-white shadow-xl flex flex-col animate-in slide-in-from-right">
          {/* Header */}
          <div class="flex items-center justify-between border-b border-slate-200 px-5 py-4">
            <h2 class="text-base font-semibold text-slate-900">Run Pipeline</h2>
            <button onClick={props.onClose} class="rounded p-1 text-slate-400 hover:text-slate-600 transition-colors">
              <X size={18} />
            </button>
          </div>

          <div class="flex-1 overflow-y-auto px-5 py-4 space-y-5">
            {/* Config */}
            <Show when={showConfig()}>
              <div class="space-y-3">
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Execution template</label>
                  <select
                    value={templateId()}
                    onChange={(e) => {
                      const tmpl = RUN_TEMPLATES.find((t) => t.id === e.target.value);
                      if (tmpl) applyTemplate(tmpl);
                    }}
                    class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  >
                    <For each={RUN_TEMPLATES}>
                      {(template) => (
                        <option value={template.id}>{template.label}</option>
                      )}
                    </For>
                  </select>
                  <p class="mt-1 text-[11px] text-slate-500">{selectedTemplate().description}</p>
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Model backend</label>
                  <select
                    value={model()}
                    onChange={(e) => setModel(e.target.value)}
                    class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  >
                    <option value="builtin">Built-in (OLS)</option>
                    <option value="pymc">PyMC-Marketing</option>
                    <option value="meridian">Google Meridian</option>
                    <option value="numpyro">NumPyro</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Target column</label>
                  <select
                    value={target()}
                    onChange={(e) => setTarget(e.target.value)}
                    class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  >
                    <option value="revenue">Revenue</option>
                    <option value="conversions">Conversions</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-medium text-slate-600 mb-1">Total budget (optional)</label>
                  <input
                    type="number"
                    min={0}
                    step="any"
                    value={budgetInput()}
                    onInput={(e) => setBudgetInput(e.target.value)}
                    placeholder="Auto from spend history"
                    class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  />
                </div>
                <label class="flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
                  <span class="text-xs font-medium text-slate-700">Use sample data (demo run)</span>
                  <input
                    type="checkbox"
                    checked={useSampleData()}
                    onChange={(e) => {
                      const enabled = e.target.checked;
                      setUseSampleData(enabled);
                      setAnalyticsEnabled(enabled);
                    }}
                    class="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </label>
                <Show when={!useSampleData()}>
                  <p class="text-[11px] text-amber-700">
                    Analytics views will be hidden while this is off.
                  </p>
                </Show>
                <button
                  onClick={startPipeline}
                  disabled={starting()}
                  class="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-60 transition-colors"
                >
                  <Show when={starting()} fallback={<Play size={16} />}>
                    <Loader2 size={16} class="animate-spin" />
                  </Show>
                  {starting() ? "Starting..." : jobId() ? "Run Pipeline" : "Start Pipeline"}
                </button>
              </div>
            </Show>

            {/* Stepper */}
            <Show when={jobId()}>
              <div class="space-y-1">
                <For each={STEPS}>
                  {(step, i) => {
                    const Icon = step.icon;
                    const status = () => {
                      const j = job();
                      const idx = currentIdx();
                      if (!j) return "pending" as const;
                      if (i() < idx) return "done" as const;
                      if (i() === idx) return isRunning() ? "running" as const : isDone() ? (j.status === "completed" ? "done" as const : "failed" as const) : "running" as const;
                      if (isDone() && j.status === "completed") return "done" as const;
                      return "pending" as const;
                    };

                    return (
                      <div class="flex items-center gap-3 py-2">
                        <div class="relative flex h-8 w-8 shrink-0 items-center justify-center">
                          <Show when={status() === "done"}>
                            <CheckCircle2 size={18} class="text-emerald-500" />
                          </Show>
                          <Show when={status() === "running"}>
                            <Loader2 size={18} class="text-indigo-500 animate-spin" />
                          </Show>
                          <Show when={status() === "failed"}>
                            <AlertCircle size={18} class="text-red-500" />
                          </Show>
                          <Show when={status() === "pending"}>
                            <Circle size={18} class="text-slate-300" />
                          </Show>
                        </div>
                        <div class="flex-1 min-w-0">
                          <p class={`text-sm font-medium ${status() === "done" ? "text-emerald-700" : status() === "running" ? "text-indigo-700" : status() === "failed" ? "text-red-700" : "text-slate-400"}`}>
                            {step.label}
                          </p>
                        </div>
                        <Icon size={14} class="shrink-0 text-slate-300" />
                      </div>
                    );
                  }}
                </For>
              </div>
            </Show>

            {/* Progress bar */}
            <Show when={job() && isRunning()}>
              <div class="space-y-1">
                <div class="flex justify-between text-xs text-slate-500">
                  <span>{job()!.current_step || "Starting"}</span>
                  <span>{job()!.progress_pct}%</span>
                </div>
                <div class="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    class="h-full rounded-full bg-indigo-500 transition-all duration-500"
                    style={{ width: `${job()!.progress_pct}%` }}
                  />
                </div>
              </div>
            </Show>

            {/* Result */}
            <Show when={job()?.status === "completed"}>
              <div class="rounded-lg border border-emerald-200 bg-emerald-50 p-4 space-y-2">
                <p class="text-sm font-medium text-emerald-800">Pipeline completed</p>
                <Show when={job()!.metrics}>
                  <div class="grid grid-cols-2 gap-2 text-xs text-emerald-700">
                    <Show when={job()!.metrics!.mape != null}>
                      <div>MAPE: {Number(job()!.metrics!.mape).toFixed(1)}%</div>
                    </Show>
                    <Show when={job()!.metrics!.r_squared != null}>
                      <div>R²: {Number(job()!.metrics!.r_squared).toFixed(3)}</div>
                    </Show>
                  </div>
                </Show>
                <Show when={job()!.run_id}>
                  <a href="/" class="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 hover:text-indigo-800">
                    <Link2 size={12} /> View Dashboard
                  </a>
                </Show>
              </div>
            </Show>

            <Show when={job()?.status === "failed"}>
              <div class="rounded-lg border border-red-200 bg-red-50 p-4">
                <p class="text-sm font-medium text-red-800">Pipeline failed</p>
                <Show when={job()!.error}>
                  <p class="mt-1 text-xs text-red-600">{job()!.error}</p>
                </Show>
              </div>
            </Show>

            {/* Logs */}
            <Show when={job() && job()!.logs.length > 0}>
              <div>
                <h3 class="text-xs font-medium text-slate-500 mb-2">Logs</h3>
                <div class="rounded-md bg-slate-900 p-3 max-h-40 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
                  <For each={job()!.logs}>
                    {(log) => <div>{log}</div>}
                  </For>
                </div>
              </div>
            </Show>

            {/* Reset */}
            <Show when={isDone()}>
              <button
                onClick={() => { setJobId(null); setJob(null); }}
                class="w-full rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Run Another
              </button>
            </Show>
          </div>
        </div>
      </div>
    </Show>
  );
}
