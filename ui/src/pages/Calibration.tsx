import { createSignal, onMount, Show, For } from "solid-js";
import {
  CheckCircle,
  XCircle,
  Target,
  AlertTriangle,
  Plus,
  ChevronUp,
  ChevronDown,
  Trash2,
  X,
  FlaskConical,
  Calculator,
} from "../lib/icons";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  BarChart,
  Bar,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import MetricCard from "../components/MetricCard";
import EmptyState from "../components/EmptyState";
import { api, type CalibrationData } from "../lib/api";
import { useToast } from "../lib/toast";

// ─── Experiment Planner types ────────────────────────────────────────────────

type ExpStatus = "planned" | "running" | "complete" | "calibrated";

interface Experiment {
  id: string;
  name: string;
  channel: string;
  budget: number; // $
  weeks: number;
  status: ExpStatus;
  startDate?: string;
}

const STATUS_STYLES: Record<
  ExpStatus,
  { label: string; badge: string; dot: string }
> = {
  planned: {
    label: "Planned",
    badge: "bg-slate-100 text-slate-600",
    dot: "bg-slate-400",
  },
  running: {
    label: "Running",
    badge: "bg-blue-100 text-blue-700",
    dot: "bg-blue-500",
  },
  complete: {
    label: "Complete",
    badge: "bg-emerald-100 text-emerald-700",
    dot: "bg-emerald-500",
  },
  calibrated: {
    label: "Calibrated",
    badge: "bg-indigo-100 text-indigo-700",
    dot: "bg-indigo-500",
  },
};

const STATUS_ORDER: ExpStatus[] = [
  "planned",
  "running",
  "complete",
  "calibrated",
];

const CHANNELS = [
  "google_ads",
  "meta_ads",
  "tiktok_ads",
  "linkedin_ads",
  "pinterest_ads",
  "snapchat_ads",
];

const DEMO_EXPERIMENTS: Experiment[] = [
  {
    id: "exp1",
    name: "Meta Geo Holdout Q2",
    channel: "meta_ads",
    budget: 80000,
    weeks: 6,
    status: "running",
    startDate: "2026-03-17",
  },
  {
    id: "exp2",
    name: "TikTok Incrementality Test",
    channel: "tiktok_ads",
    budget: 40000,
    weeks: 4,
    status: "planned",
  },
  {
    id: "exp3",
    name: "Google Brand vs. Generic",
    channel: "google_ads",
    budget: 120000,
    weeks: 8,
    status: "complete",
    startDate: "2026-01-06",
  },
  {
    id: "exp4",
    name: "LinkedIn B2B Lift Study",
    channel: "linkedin_ads",
    budget: 25000,
    weeks: 5,
    status: "calibrated",
    startDate: "2025-11-10",
  },
];

// ─── Power calculator ─────────────────────────────────────────────────────────

/** Mock confidence estimator: educational-grade approximation */
function calcConfidence(budget: number, weeks: number): number {
  if (budget <= 0 || weeks <= 0) return 0;
  const raw = 40 + (budget / 5000) * 0.8 + weeks * 2.5;
  return Math.min(95, Math.round(raw * 10) / 10);
}

function confidenceColor(c: number) {
  if (c >= 80) return "text-emerald-600";
  if (c >= 60) return "text-amber-600";
  return "text-red-500";
}

// ─── Calibration helpers ──────────────────────────────────────────────────────

const toFinite = (value: unknown): number => {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : 0;
};

// ─── Component ────────────────────────────────────────────────────────────────

export default function Calibration() {
  const [mainTab, setMainTab] = createSignal<"calibration" | "planner">(
    "calibration",
  );

  // ── Calibration state ──
  const [data, setData] = createSignal<CalibrationData | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(true);

  onMount(() => {
    api
      .calibration()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setData(null);
      })
      .finally(() => setLoading(false));
  });

  // ── Planner state ──
  const { addToast } = useToast();
  const [experiments, setExperiments] =
    createSignal<Experiment[]>(DEMO_EXPERIMENTS);
  const [showAddExp, setShowAddExp] = createSignal(false);

  // Add form
  const [newName, setNewName] = createSignal("");
  const [newChannel, setNewChannel] = createSignal("google_ads");
  const [newBudget, setNewBudget] = createSignal(50000);
  const [newWeeks, setNewWeeks] = createSignal(4);

  // Power calc (independent of add form)
  const [calcChannel, setCalcChannel] = createSignal("google_ads");
  const [calcBudget, setCalcBudget] = createSignal(50000);
  const [calcWeeks, setCalcWeeks] = createSignal(4);
  const calcResult = () => calcConfidence(calcBudget(), calcWeeks());

  const moveExp = (id: string, dir: -1 | 1) => {
    setExperiments((prev) => {
      const idx = prev.findIndex((e) => e.id === id);
      if (idx < 0) return prev;
      const next = [...prev];
      const swap = idx + dir;
      if (swap < 0 || swap >= next.length) return prev;
      [next[idx], next[swap]] = [next[swap], next[idx]];
      return next;
    });
  };

  const advanceStatus = (id: string) => {
    setExperiments((prev) =>
      prev.map((e) => {
        if (e.id !== id) return e;
        const idx = STATUS_ORDER.indexOf(e.status);
        if (idx >= STATUS_ORDER.length - 1) return e;
        return {
          ...e,
          status: STATUS_ORDER[idx + 1],
          startDate: e.startDate ?? new Date().toISOString().slice(0, 10),
        };
      }),
    );
  };

  const deleteExp = (id: string) => {
    const exp = experiments().find((e) => e.id === id);
    setExperiments((prev) => prev.filter((e) => e.id !== id));
    if (exp) addToast("info", `Experiment "${exp.name}" removed`);
  };

  const addExperiment = () => {
    if (!newName().trim()) return;
    const exp: Experiment = {
      id: `exp-${Date.now()}`,
      name: newName(),
      channel: newChannel(),
      budget: newBudget(),
      weeks: newWeeks(),
      status: "planned",
    };
    setExperiments((prev) => [...prev, exp]);
    setShowAddExp(false);
    setNewName("");
    addToast("success", `Experiment "${exp.name}" added to backlog`);
  };

  const plannedCount = () =>
    experiments().filter((e) => e.status === "planned").length;
  const runningCount = () =>
    experiments().filter((e) => e.status === "running").length;
  const calibratedCount = () =>
    experiments().filter((e) => e.status === "calibrated").length;

  return (
    <div>
      {/* Page-level tabs */}
      <div class="flex items-center justify-between mb-5">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Calibration</h1>
          <p class="text-sm text-slate-500 mt-0.5">
            Validate model accuracy and plan future experiments
          </p>
        </div>
        <div class="flex gap-1">
          <button
            onClick={() => setMainTab("calibration")}
            class={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mainTab() === "calibration"
                ? "bg-indigo-600 text-white"
                : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            <Target size={14} /> MMM Calibration
          </button>
          <button
            onClick={() => setMainTab("planner")}
            class={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mainTab() === "planner"
                ? "bg-indigo-600 text-white"
                : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            <FlaskConical size={14} /> Experiment Planner
            <Show when={runningCount() > 0}>
              <span class="ml-0.5 inline-flex items-center justify-center w-4 h-4 rounded-full bg-blue-500 text-white text-[9px] font-bold">
                {runningCount()}
              </span>
            </Show>
          </button>
        </div>
      </div>

      {/* ── MMM Calibration tab ── */}
      <Show when={mainTab() === "calibration"}>
        <Show
          when={!loading()}
          fallback={
            <div class="min-h-[60vh] flex items-center justify-center">
              <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600" />
            </div>
          }
        >
          <Show
            when={!error()}
            fallback={
              <div class="min-h-[60vh] flex items-center justify-center">
                <EmptyState
                  icon={<AlertTriangle class="w-10 h-10 text-amber-400" />}
                  title="Calibration data unavailable"
                  description={error()!}
                />
              </div>
            }
          >
            <Show
              when={data() && (data()!.n_tests ?? 0) > 0}
              fallback={
                <div class="min-h-[60vh] flex items-center justify-center">
                  <EmptyState
                    icon={<Target class="w-10 h-10 text-gray-400" />}
                    title="No calibration data yet"
                    description="Run an experiment calibration to see predictions vs. measured lift."
                  />
                </div>
              }
            >
              {(_d) => {
                const points = () => data()!.points ?? [];
                const scatterData = () =>
                  points().map((p) => ({
                    ...p,
                    x: p.measured_lift ?? 0,
                    y: p.predicted_lift ?? 0,
                  }));
                const minX = () =>
                  scatterData().length
                    ? Math.min(...scatterData().map((d) => toFinite(d.x)))
                    : 0;
                const maxX = () =>
                  scatterData().length
                    ? Math.max(...scatterData().map((d) => toFinite(d.x)))
                    : 0;
                const barData = () =>
                  points().map((p) => ({
                    channel: p.channel ?? "",
                    error_pct: Math.round((p.error_pct as number | undefined) ?? 0),
                    within_ci: p.within_ci ?? false,
                  }));
                const qualityColor = () =>
                  data()!.calibration_quality === "good"
                    ? "text-green-600"
                    : data()!.calibration_quality === "fair"
                      ? "text-amber-600"
                      : "text-red-600";

                return (
                  <div class="min-h-[60vh] space-y-6">
                    {/* Summary cards */}
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <MetricCard
                        icon={Target}
                        label="Tests Compared"
                        value={data()!.n_tests}
                      />
                      <MetricCard
                        icon={CheckCircle}
                        label="Coverage (within CI)"
                        value={`${((data()!.coverage ?? 0) * 100).toFixed(0)}%`}
                      />
                      <MetricCard
                        icon={AlertTriangle}
                        label="Median Lift Error"
                        value={`${(data()!.median_lift_error ?? 0).toFixed(1)}%`}
                      />
                      <div class="bg-white rounded-xl shadow-sm border border-gray-200 px-5 py-4">
                        <p class="text-xs text-gray-500 mb-1">Quality</p>
                        <p
                          class={`text-xl font-bold capitalize ${qualityColor()}`}
                        >
                          {data()!.calibration_quality}
                        </p>
                      </div>
                    </div>

                    {/* Scatter */}
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                      <h2 class="text-lg font-semibold text-gray-900 mb-4">
                        Predicted vs. Measured Lift
                      </h2>
                      <p class="text-sm text-gray-500 mb-4">
                        Points near the diagonal mean the MMM prediction
                        matched the experiment result. Green = within CI, Red =
                        outside CI.
                      </p>
                      <ReactChart>
                        {() =>
                          h(
                            ResponsiveContainer,
                            { width: "100%", height: 400 },
                            h(
                              ScatterChart,
                              {
                                margin: {
                                  top: 10,
                                  right: 30,
                                  bottom: 20,
                                  left: 20,
                                },
                              },
                              h(CartesianGrid, { strokeDasharray: "3 3" }),
                              h(XAxis, {
                                type: "number",
                                dataKey: "x",
                                name: "Measured Lift",
                                label: {
                                  value: "Measured Lift",
                                  position: "insideBottom",
                                  offset: -10,
                                },
                              }),
                              h(YAxis, {
                                type: "number",
                                dataKey: "y",
                                name: "Predicted Lift",
                                label: {
                                  value: "Predicted Lift",
                                  angle: -90,
                                  position: "insideLeft",
                                },
                              }),
                              h(Tooltip, {
                                content: ({ payload }: any) => {
                                  if (!payload?.length) return null;
                                  const p = payload[0].payload;
                                  return h(
                                    "div",
                                    {
                                      className:
                                        "bg-white border border-gray-200 rounded shadow-lg p-3 text-sm",
                                    },
                                    h(
                                      "p",
                                      { className: "font-semibold" },
                                      p.channel,
                                    ),
                                    h(
                                      "p",
                                      null,
                                      `Measured: ${toFinite(p.measured_lift).toFixed(4)}`,
                                    ),
                                    h(
                                      "p",
                                      null,
                                      `Predicted: ${toFinite(p.predicted_lift).toFixed(4)}`,
                                    ),
                                    h(
                                      "p",
                                      null,
                                      `Error: ${toFinite(p.error_pct).toFixed(1)}%`,
                                    ),
                                    h(
                                      "p",
                                      null,
                                      "Within CI: ",
                                      h(
                                        "span",
                                        {
                                          className: p.within_ci
                                            ? "text-green-600"
                                            : "text-red-600",
                                        },
                                        p.within_ci ? "Yes" : "No",
                                      ),
                                    ),
                                  );
                                },
                              }),
                              h(ReferenceLine, {
                                segment: [
                                  { x: minX() * 0.8, y: minX() * 0.8 },
                                  { x: maxX() * 1.2, y: maxX() * 1.2 },
                                ],
                                stroke: "#9ca3af",
                                strokeDasharray: "6 4",
                                label: "Perfect",
                              }),
                              h(
                                Scatter,
                                { data: scatterData() },
                                ...scatterData().map((entry, i) =>
                                  h(Cell, {
                                    key: i,
                                    fill: (entry as any).within_ci
                                      ? "#16a34a"
                                      : "#dc2626",
                                    r: 8,
                                  }),
                                ),
                              ),
                            ),
                          )
                        }
                      </ReactChart>
                    </div>

                    {/* Bar chart */}
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                      <h2 class="text-lg font-semibold text-gray-900 mb-4">
                        Lift Error by Channel
                      </h2>
                      <ReactChart>
                        {() =>
                          h(
                            ResponsiveContainer,
                            { width: "100%", height: 300 },
                            h(
                              BarChart,
                              {
                                data: barData(),
                                margin: {
                                  top: 5,
                                  right: 30,
                                  bottom: 5,
                                  left: 20,
                                },
                              },
                              h(CartesianGrid, { strokeDasharray: "3 3" }),
                              h(XAxis, { dataKey: "channel" }),
                              h(YAxis, {
                                label: {
                                  value: "Error %",
                                  angle: -90,
                                  position: "insideLeft",
                                },
                              }),
                              h(Tooltip, null),
                              h(
                                Bar,
                                { dataKey: "error_pct", name: "Error %" },
                                ...barData().map((entry, i) =>
                                  h(Cell, {
                                    key: i,
                                    fill: entry.within_ci
                                      ? "#16a34a"
                                      : "#dc2626",
                                  }),
                                ),
                              ),
                            ),
                          )
                        }
                      </ReactChart>
                    </div>

                    {/* Detail table */}
                    <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                      <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                          <tr>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                              Test ID
                            </th>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                              Channel
                            </th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                              Measured
                            </th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                              Predicted
                            </th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                              Error %
                            </th>
                            <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                              In CI?
                            </th>
                          </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                          <For each={points()}>
                            {(p) => (
                              <tr class="hover:bg-gray-50">
                                <td class="px-4 py-3 text-sm font-mono">
                                  {p.test_id ?? ""}
                                </td>
                                <td class="px-4 py-3 text-sm">
                                  {p.channel ?? ""}
                                </td>
                                <td class="px-4 py-3 text-sm text-right">
                                  {(p.measured_lift ?? 0).toFixed(4)}
                                </td>
                                <td class="px-4 py-3 text-sm text-right">
                                  {(p.predicted_lift ?? 0).toFixed(4)}
                                </td>
                                <td class="px-4 py-3 text-sm text-right">
                                  {(p.error_pct ?? 0).toFixed(1)}%
                                </td>
                                <td class="px-4 py-3 text-center">
                                  <Show
                                    when={p.within_ci}
                                    fallback={
                                      <XCircle class="w-5 h-5 text-red-600 inline" />
                                    }
                                  >
                                    <CheckCircle class="w-5 h-5 text-green-600 inline" />
                                  </Show>
                                </td>
                              </tr>
                            )}
                          </For>
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              }}
            </Show>
          </Show>
        </Show>
      </Show>

      {/* ── Experiment Planner tab ── */}
      <Show when={mainTab() === "planner"}>
        <div class="space-y-6">
          {/* Summary KPIs */}
          <div class="grid grid-cols-3 gap-3">
            <MetricCard
              label="Planned"
              value={plannedCount()}
            />
            <MetricCard
              label="Running"
              value={runningCount()}
              color={runningCount() > 0 ? "indigo" : undefined}
            />
            <MetricCard
              label="Calibrated"
              value={calibratedCount()}
              color={calibratedCount() > 0 ? "emerald" : undefined}
            />
          </div>

          {/* Power Calculator */}
          <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <div class="flex items-center gap-2 mb-4">
              <Calculator size={16} class="text-indigo-500" />
              <h2 class="text-base font-semibold text-slate-900">
                Power Calculator
              </h2>
            </div>
            <p class="text-xs text-slate-500 mb-4">
              Estimate the statistical confidence achievable for a given test
              budget and duration. Higher budget and longer duration increase
              your ability to detect true lift.
            </p>
            <div class="grid grid-cols-3 gap-4 items-end">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Channel
                </label>
                <select
                  value={calcChannel()}
                  onChange={(e) => setCalcChannel(e.currentTarget.value)}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={CHANNELS}>
                    {(ch) => (
                      <option value={ch}>{ch.replace(/_/g, " ")}</option>
                    )}
                  </For>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Test Budget ($)
                </label>
                <input
                  type="number"
                  min="1000"
                  step="5000"
                  value={calcBudget()}
                  onInput={(e) =>
                    setCalcBudget(Number(e.currentTarget.value))
                  }
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Duration (weeks)
                </label>
                <input
                  type="number"
                  min="1"
                  max="52"
                  value={calcWeeks()}
                  onInput={(e) =>
                    setCalcWeeks(Number(e.currentTarget.value))
                  }
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Result */}
            <div class="mt-4 flex items-center gap-4 rounded-lg bg-slate-50 border border-slate-200 px-4 py-3">
              <div>
                <p class="text-xs text-slate-500">Estimated Confidence</p>
                <p
                  class={`text-3xl font-bold tabular-nums ${confidenceColor(calcResult())}`}
                >
                  {calcResult()}%
                </p>
              </div>
              <div class="flex-1">
                <div class="h-2.5 w-full rounded-full bg-slate-200">
                  <div
                    class="h-2.5 rounded-full transition-all duration-300"
                    style={{
                      width: `${calcResult()}%`,
                      background:
                        calcResult() >= 80
                          ? "#10b981"
                          : calcResult() >= 60
                            ? "#f59e0b"
                            : "#ef4444",
                    }}
                  />
                </div>
                <p class="text-xs text-slate-400 mt-1">
                  {calcResult() >= 80
                    ? "Strong — likely to detect meaningful lift"
                    : calcResult() >= 60
                      ? "Moderate — consider increasing budget or duration"
                      : "Weak — test may be underpowered"}
                </p>
              </div>
            </div>
          </div>

          {/* Experiments Backlog */}
          <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-base font-semibold text-slate-900">
                Experiment Backlog
              </h2>
              <button
                onClick={() => setShowAddExp(true)}
                class="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
              >
                <Plus size={13} /> Add Experiment
              </button>
            </div>
            <p class="text-xs text-slate-500 mb-4">
              Drag to reprioritize (use arrows). Advance status as experiments
              progress. Calibrated experiments feed back into MMM accuracy.
            </p>

            <Show
              when={experiments().length > 0}
              fallback={
                <div class="rounded-lg border border-dashed border-slate-200 py-10 text-center text-sm text-slate-400">
                  No experiments yet. Add one to get started.
                </div>
              }
            >
              <div class="space-y-2">
                <For each={experiments()}>
                  {(exp, idx) => {
                    const st = STATUS_STYLES[exp.status];
                    const nextStatus =
                      STATUS_ORDER[STATUS_ORDER.indexOf(exp.status) + 1];
                    return (
                      <div class="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-3 hover:border-slate-300 transition-colors">
                        {/* Priority controls */}
                        <div class="flex flex-col gap-0.5">
                          <button
                            onClick={() => moveExp(exp.id, -1)}
                            disabled={idx() === 0}
                            class="text-slate-300 hover:text-slate-600 disabled:opacity-20 p-0.5"
                            aria-label="Move up"
                          >
                            <ChevronUp size={12} />
                          </button>
                          <button
                            onClick={() => moveExp(exp.id, 1)}
                            disabled={idx() === experiments().length - 1}
                            class="text-slate-300 hover:text-slate-600 disabled:opacity-20 p-0.5"
                            aria-label="Move down"
                          >
                            <ChevronDown size={12} />
                          </button>
                        </div>

                        {/* Priority # */}
                        <span class="text-xs font-mono text-slate-400 w-4 text-center shrink-0">
                          {idx() + 1}
                        </span>

                        {/* Status dot */}
                        <div
                          class={`h-2 w-2 rounded-full shrink-0 ${st.dot}`}
                        />

                        {/* Info */}
                        <div class="flex-1 min-w-0">
                          <p class="text-sm font-medium text-slate-900 truncate">
                            {exp.name}
                          </p>
                          <p class="text-xs text-slate-400">
                            {exp.channel.replace(/_/g, " ")} ·{" "}
                            ${exp.budget.toLocaleString()} · {exp.weeks}w
                            {exp.startDate ? ` · Started ${exp.startDate}` : ""}
                          </p>
                        </div>

                        {/* Status badge */}
                        <span
                          class={`rounded-full px-2.5 py-0.5 text-xs font-semibold shrink-0 ${st.badge}`}
                        >
                          {st.label}
                        </span>

                        {/* Advance button */}
                        <Show when={nextStatus}>
                          <button
                            onClick={() => advanceStatus(exp.id)}
                            class="rounded-md border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors shrink-0"
                            title={`Mark as ${STATUS_STYLES[nextStatus!].label}`}
                          >
                            → {STATUS_STYLES[nextStatus!].label}
                          </button>
                        </Show>

                        {/* Delete */}
                        <button
                          onClick={() => deleteExp(exp.id)}
                          class="text-slate-300 hover:text-red-500 transition-colors p-1 shrink-0"
                          aria-label="Delete experiment"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    );
                  }}
                </For>
              </div>
            </Show>
          </div>
        </div>
      </Show>

      {/* Add Experiment Modal */}
      <Show when={showAddExp()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div
            class="absolute inset-0 bg-black/20"
            onClick={() => setShowAddExp(false)}
          />
          <div class="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">
                New Experiment
              </h3>
              <button
                onClick={() => setShowAddExp(false)}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Experiment Name
              </label>
              <input
                type="text"
                value={newName()}
                onInput={(e) => setNewName(e.currentTarget.value)}
                placeholder="e.g. Meta Geo Holdout Q3"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Channel
              </label>
              <select
                value={newChannel()}
                onChange={(e) => setNewChannel(e.currentTarget.value)}
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                <For each={CHANNELS}>
                  {(ch) => (
                    <option value={ch}>{ch.replace(/_/g, " ")}</option>
                  )}
                </For>
              </select>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Budget ($)
                </label>
                <input
                  type="number"
                  min="0"
                  step="5000"
                  value={newBudget()}
                  onInput={(e) => setNewBudget(Number(e.currentTarget.value))}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Duration (weeks)
                </label>
                <input
                  type="number"
                  min="1"
                  max="52"
                  value={newWeeks()}
                  onInput={(e) => setNewWeeks(Number(e.currentTarget.value))}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Live confidence preview */}
            <div class="rounded-lg bg-slate-50 border border-slate-200 px-3 py-2.5 text-xs text-slate-500 flex items-center gap-2">
              <Calculator size={12} class="shrink-0 text-slate-400" />
              Estimated confidence:{" "}
              <span
                class={`font-semibold ${confidenceColor(calcConfidence(newBudget(), newWeeks()))}`}
              >
                {calcConfidence(newBudget(), newWeeks())}%
              </span>
            </div>

            <div class="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowAddExp(false)}
                class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={addExperiment}
                disabled={!newName().trim()}
                class="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors"
              >
                Add to Backlog
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
