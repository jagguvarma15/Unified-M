import { createSignal, createEffect, onMount, Show, For } from "solid-js";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  LineChart,
  Line,
  ReferenceLine,
} from "recharts";
import ReactChart, { h } from "../lib/ReactChart";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import EmptyState from "../components/EmptyState";
import MetricCard from "../components/MetricCard";
import {
  api,
  type OptimizationData,
  type ResponseCurvesData,
} from "../lib/api";
import { COLORS, CHART_GRID, CHART_TOOLTIP_BG } from "../lib/colors";
import { formatCompactNumber, formatSpendTick } from "../lib/chartFormat";
import { formatCurrency } from "../lib/format";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Calendar,
  DollarSign,
  SlidersHorizontal,
  Target,
  Save,
  Copy,
  Trash2,
  History,
  Download,
} from "../lib/icons";
import { useToast } from "../lib/toast";
import {
  savedScenarios,
  saveScenario,
  removeScenario,
  encodeScenarioUrl,
  decodeScenarioUrl,
  type SavedScenario,
} from "../lib/scenarioStore";

const STEPS = [
  { label: "Date Range", icon: Calendar },
  { label: "Budget", icon: DollarSign },
  { label: "Channel Mix", icon: SlidersHorizontal },
  { label: "Summary", icon: Target },
] as const;

export default function WhatIfStudio() {
  const { addToast } = useToast();
  const [optData, setOptData] = createSignal<OptimizationData | null>(null);
  const [curvesData, setCurvesData] = createSignal<ResponseCurvesData | null>(
    null,
  );
  const [loading, setLoading] = createSignal(true);

  // Wizard state
  const [step, setStep] = createSignal(0);

  // Step 1: Date range
  const [dateStart, setDateStart] = createSignal("2025-01-01");
  const [dateEnd, setDateEnd] = createSignal("2025-12-31");

  // Step 2: Budget constraint
  const [budgetMode, setBudgetMode] = createSignal<"fixed" | "flexible">(
    "flexible",
  );
  const [totalBudgetMult, setTotalBudgetMult] = createSignal(1.0);

  // Step 3: Channel mix
  const [channelMults, setChannelMults] = createSignal<Record<string, number>>(
    {},
  );
  const [channelMins, setChannelMins] = createSignal<Record<string, number>>(
    {},
  );
  const [channelMaxs, setChannelMaxs] = createSignal<Record<string, number>>(
    {},
  );

  // Saved scenarios panel
  const [showSaved, setShowSaved] = createSignal(false);
  const [scenarioName, setScenarioName] = createSignal("");

  onMount(() => {
    Promise.allSettled([
      api.optimization().then(setOptData),
      api.responseCurves().then(setCurvesData),
    ]).finally(() => setLoading(false));

    // Check for shared scenario in URL
    const params = new URLSearchParams(window.location.search);
    const shared = params.get("shared");
    if (shared) {
      try {
        const decoded = decodeScenarioUrl(shared);
        if (decoded) {
          if (decoded.dateRange) {
            setDateStart(decoded.dateRange.start);
            setDateEnd(decoded.dateRange.end);
          }
          if (decoded.budgetMode) setBudgetMode(decoded.budgetMode);
          if (decoded.totalBudgetMult)
            setTotalBudgetMult(decoded.totalBudgetMult);
          if (decoded.channelMults) setChannelMults(decoded.channelMults);
          if (decoded.channelMins) setChannelMins(decoded.channelMins);
          if (decoded.channelMaxs) setChannelMaxs(decoded.channelMaxs);
          setStep(3); // Jump to summary
          addToast("info", "Loaded shared scenario");
        }
      } catch {}
    }
  });

  // Init channel mults when data loads
  createEffect(() => {
    const d = optData();
    if (!d) return;
    const channels = Object.keys(d.current_allocation);
    const mults: Record<string, number> = {};
    const mins: Record<string, number> = {};
    const maxs: Record<string, number> = {};
    channels.forEach((ch) => {
      if (!channelMults()[ch]) mults[ch] = 1.0;
      else mults[ch] = channelMults()[ch];
      if (!channelMins()[ch]) mins[ch] = 0;
      else mins[ch] = channelMins()[ch];
      if (!channelMaxs()[ch]) maxs[ch] = 3.0;
      else maxs[ch] = channelMaxs()[ch];
    });
    setChannelMults(mults);
    setChannelMins(mins);
    setChannelMaxs(maxs);
  });

  const channels = () => {
    const d = optData();
    return d ? Object.keys(d.current_allocation) : [];
  };

  const baseBudget = () => {
    const d = optData();
    if (!d) return 0;
    return Object.values(d.current_allocation).reduce((a, b) => a + b, 0);
  };

  const channelBudget = (ch: string) => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_allocation[ch] ?? 0;
    return base * totalBudgetMult() * (channelMults()[ch] ?? 1);
  };

  const totalAllocated = () =>
    channels().reduce((s, ch) => s + channelBudget(ch), 0);

  const estimatedResponse = () => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_response ?? d.expected_response;
    const budgetRatio = totalAllocated() / baseBudget();
    return base * Math.pow(budgetRatio, 0.7);
  };

  const estimatedROI = () => {
    const spent = totalAllocated();
    return spent > 0 ? estimatedResponse() / spent : 0;
  };

  const upliftPct = () => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_response ?? d.expected_response;
    return base > 0 ? ((estimatedResponse() - base) / base) * 100 : 0;
  };

  const currentROI = () => {
    const d = optData();
    if (!d) return 0;
    const base = d.current_response ?? d.expected_response;
    return baseBudget() > 0 ? base / baseBudget() : 0;
  };

  const channelCompare = () =>
    channels().map((ch) => ({
      channel: ch.replace(/_spend$/, ""),
      current: optData()?.current_allocation[ch] ?? 0,
      optimized: channelBudget(ch),
    }));

  const efficiencyData = () => {
    const d = optData();
    if (!d) return [];
    const base = d.current_response ?? d.expected_response;
    const result: { budget: number; response: number }[] = [];
    for (let mult = 0.3; mult <= 2.5; mult += 0.05) {
      result.push({
        budget: Math.round(baseBudget() * mult),
        response: Math.round(base * Math.pow(mult, 0.7)),
      });
    }
    return result;
  };

  const comparisonRows = () => {
    const d = optData();
    if (!d) return [];
    return channels().map((ch) => {
      const current = d.current_allocation[ch] ?? 0;
      const optimized = channelBudget(ch);
      const delta = optimized - current;
      const deltaPct = current > 0 ? (delta / current) * 100 : 0;
      return {
        channel: ch.replace(/_spend$/, ""),
        current,
        optimized,
        delta,
        deltaPct,
      };
    });
  };

  const updateChannelMult = (ch: string, value: number) => {
    setChannelMults((prev) => ({ ...prev, [ch]: value }));
  };

  const updateChannelMin = (ch: string, value: number) => {
    setChannelMins((prev) => ({ ...prev, [ch]: value }));
  };

  const updateChannelMax = (ch: string, value: number) => {
    setChannelMaxs((prev) => ({ ...prev, [ch]: value }));
  };

  const resetAll = () => {
    setTotalBudgetMult(1.0);
    const mults: Record<string, number> = {};
    const mins: Record<string, number> = {};
    const maxs: Record<string, number> = {};
    channels().forEach((ch) => {
      mults[ch] = 1.0;
      mins[ch] = 0;
      maxs[ch] = 3.0;
    });
    setChannelMults(mults);
    setChannelMins(mins);
    setChannelMaxs(maxs);
  };

  const canAdvance = () => {
    if (step() === 0)
      return dateStart() && dateEnd() && dateStart() < dateEnd();
    return true;
  };

  const handleSaveScenario = () => {
    const name =
      scenarioName().trim() || `Scenario ${savedScenarios().length + 1}`;
    const saved = saveScenario({
      name,
      dateRange: { start: dateStart(), end: dateEnd() },
      budgetMode: budgetMode(),
      totalBudgetMult: totalBudgetMult(),
      channelMults: { ...channelMults() },
      channelMins: { ...channelMins() },
      channelMaxs: { ...channelMaxs() },
      totalBudget: totalAllocated(),
      estimatedResponse: estimatedResponse(),
      estimatedROI: estimatedROI(),
    });
    setScenarioName("");
    addToast("success", `Saved "${saved.name}"`);
  };

  const handleShareScenario = () => {
    const scenario: SavedScenario = {
      id: "share",
      createdAt: new Date().toISOString(),
      name: scenarioName().trim() || "Shared Scenario",
      dateRange: { start: dateStart(), end: dateEnd() },
      budgetMode: budgetMode(),
      totalBudgetMult: totalBudgetMult(),
      channelMults: { ...channelMults() },
      channelMins: { ...channelMins() },
      channelMaxs: { ...channelMaxs() },
      totalBudget: totalAllocated(),
      estimatedResponse: estimatedResponse(),
      estimatedROI: estimatedROI(),
    };
    const url = encodeScenarioUrl(scenario);
    navigator.clipboard.writeText(url).then(
      () => addToast("success", "Share link copied to clipboard"),
      () => addToast("error", "Failed to copy link"),
    );
  };

  const loadSavedScenario = (s: SavedScenario) => {
    setDateStart(s.dateRange.start);
    setDateEnd(s.dateRange.end);
    setBudgetMode(s.budgetMode);
    setTotalBudgetMult(s.totalBudgetMult);
    setChannelMults({ ...s.channelMults });
    setChannelMins({ ...s.channelMins });
    setChannelMaxs({ ...s.channelMaxs });
    setShowSaved(false);
    setStep(3);
    addToast("info", `Loaded "${s.name}"`);
  };

  const handleExportCsv = () => {
    const rows = comparisonRows();
    const header = "Channel,Current,Optimized,Delta,Delta %\n";
    const csv = rows
      .map(
        (r) =>
          `${r.channel},${r.current.toFixed(0)},${r.optimized.toFixed(0)},${r.delta.toFixed(0)},${r.deltaPct.toFixed(1)}%`,
      )
      .join("\n");
    const blob = new Blob([header + csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "what-if-scenario.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <Show
        when={optData()}
        fallback={
          <EmptyState
            title="No optimization data"
            message="Run the pipeline and optimizer to start building what-if scenarios for your budget."
            hideQuickStart
          />
        }
      >
        <div>
          {/* Header */}
          <div class="flex items-center justify-between mb-6">
            <PageHeader
              title="What-If Studio"
              description="Guided scenario planning: set constraints, adjust channels, compare outcomes"
            />
            <div class="flex items-center gap-2">
              <button
                onClick={() => setShowSaved(!showSaved())}
                class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                <History size={14} />
                Saved ({savedScenarios().length})
              </button>
              <button
                onClick={resetAll}
                class="px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Saved Scenarios Panel */}
          <Show when={showSaved()}>
            <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-5 mb-6">
              <h3 class="text-sm font-medium text-slate-700 mb-3">
                Saved Scenarios
              </h3>
              <Show
                when={savedScenarios().length > 0}
                fallback={
                  <p class="text-sm text-slate-400">
                    No saved scenarios yet. Complete the wizard and save one.
                  </p>
                }
              >
                <div class="space-y-2">
                  <For each={savedScenarios()}>
                    {(s) => (
                      <div class="flex items-center justify-between py-2 px-3 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
                        <div
                          class="flex-1 cursor-pointer"
                          onClick={() => loadSavedScenario(s)}
                        >
                          <p class="text-sm font-medium text-slate-700">
                            {s.name}
                          </p>
                          <p class="text-xs text-slate-400">
                            {formatCurrency(s.totalBudget, true)} budget
                            &middot; {s.estimatedROI.toFixed(2)}x ROI &middot;{" "}
                            {new Date(s.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                        <button
                          onClick={() => removeScenario(s.id)}
                          class="p-1 text-slate-400 hover:text-red-500 transition-colors"
                          title="Delete scenario"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                  </For>
                </div>
              </Show>
            </div>
          </Show>

          {/* Step Indicator */}
          <div class="flex items-center justify-between mb-8">
            <For each={STEPS}>
              {(s, i) => {
                const isActive = () => step() === i();
                const isComplete = () => step() > i();
                return (
                  <>
                    <Show when={i() > 0}>
                      <div
                        class={`flex-1 h-0.5 mx-2 transition-colors ${
                          step() > i() ? "bg-indigo-500" : "bg-slate-200"
                        }`}
                      />
                    </Show>
                    <button
                      onClick={() => {
                        if (i() <= step() || canAdvance()) setStep(i());
                      }}
                      class={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                        isActive()
                          ? "bg-indigo-600 text-white shadow-md shadow-indigo-200"
                          : isComplete()
                            ? "bg-indigo-50 text-indigo-700 hover:bg-indigo-100"
                            : "bg-slate-100 text-slate-400"
                      }`}
                    >
                      <Show
                        when={!isComplete()}
                        fallback={
                          <span class="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500 text-white">
                            <Check size={12} />
                          </span>
                        }
                      >
                        <span
                          class={`flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold ${
                            isActive()
                              ? "bg-white/20 text-white"
                              : "bg-slate-200 text-slate-500"
                          }`}
                        >
                          {i() + 1}
                        </span>
                      </Show>
                      <span class="hidden sm:inline">{s.label}</span>
                    </button>
                  </>
                );
              }}
            </For>
          </div>

          {/* Step Content */}
          {/* Step 1: Date Range */}
          <Show when={step() === 0}>
            <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-8">
              <div class="max-w-lg mx-auto">
                <div class="flex items-center gap-3 mb-6">
                  <div class="flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600">
                    <Calendar size={20} />
                  </div>
                  <div>
                    <h2 class="text-lg font-semibold text-slate-800">
                      Select Date Range
                    </h2>
                    <p class="text-sm text-slate-500">
                      Choose the analysis period for your scenario
                    </p>
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm font-medium text-slate-600 mb-1.5">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={dateStart()}
                      onInput={(e) => setDateStart(e.currentTarget.value)}
                      class="w-full px-3 py-2.5 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-400 transition-colors"
                    />
                  </div>
                  <div>
                    <label class="block text-sm font-medium text-slate-600 mb-1.5">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={dateEnd()}
                      onInput={(e) => setDateEnd(e.currentTarget.value)}
                      class="w-full px-3 py-2.5 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-400 transition-colors"
                    />
                  </div>
                </div>

                {/* Quick range presets */}
                <div class="mt-4 flex flex-wrap gap-2">
                  {[
                    { label: "Q1", start: "2025-01-01", end: "2025-03-31" },
                    { label: "Q2", start: "2025-04-01", end: "2025-06-30" },
                    { label: "Q3", start: "2025-07-01", end: "2025-09-30" },
                    { label: "Q4", start: "2025-10-01", end: "2025-12-31" },
                    {
                      label: "Full Year",
                      start: "2025-01-01",
                      end: "2025-12-31",
                    },
                    {
                      label: "Last 90 Days",
                      start: "2025-07-12",
                      end: "2025-10-09",
                    },
                  ].map((p) => (
                    <button
                      onClick={() => {
                        setDateStart(p.start);
                        setDateEnd(p.end);
                      }}
                      class={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                        dateStart() === p.start && dateEnd() === p.end
                          ? "bg-indigo-600 text-white"
                          : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>

                <Show
                  when={dateStart() && dateEnd() && dateStart() >= dateEnd()}
                >
                  <p class="mt-3 text-sm text-red-500">
                    Start date must be before end date
                  </p>
                </Show>
              </div>
            </div>
          </Show>

          {/* Step 2: Budget Constraint */}
          <Show when={step() === 1}>
            <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-8">
              <div class="max-w-lg mx-auto">
                <div class="flex items-center gap-3 mb-6">
                  <div class="flex items-center justify-center w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600">
                    <DollarSign size={20} />
                  </div>
                  <div>
                    <h2 class="text-lg font-semibold text-slate-800">
                      Budget Constraint
                    </h2>
                    <p class="text-sm text-slate-500">
                      Set your overall budget approach
                    </p>
                  </div>
                </div>

                {/* Mode toggle */}
                <div class="grid grid-cols-2 gap-3 mb-6">
                  <button
                    onClick={() => {
                      setBudgetMode("fixed");
                      setTotalBudgetMult(1.0);
                    }}
                    class={`p-4 rounded-xl border-2 text-left transition-all ${
                      budgetMode() === "fixed"
                        ? "border-indigo-500 bg-indigo-50"
                        : "border-slate-200 hover:border-slate-300"
                    }`}
                  >
                    <p class="text-sm font-semibold text-slate-800">
                      Fixed Budget
                    </p>
                    <p class="text-xs text-slate-500 mt-1">
                      Keep total spend the same, redistribute across channels
                    </p>
                  </button>
                  <button
                    onClick={() => setBudgetMode("flexible")}
                    class={`p-4 rounded-xl border-2 text-left transition-all ${
                      budgetMode() === "flexible"
                        ? "border-indigo-500 bg-indigo-50"
                        : "border-slate-200 hover:border-slate-300"
                    }`}
                  >
                    <p class="text-sm font-semibold text-slate-800">
                      Flexible Budget
                    </p>
                    <p class="text-xs text-slate-500 mt-1">
                      Scale total budget up or down with a multiplier
                    </p>
                  </button>
                </div>

                {/* Budget multiplier (flexible only) */}
                <Show when={budgetMode() === "flexible"}>
                  <div class="bg-slate-50 rounded-xl p-5">
                    <div class="flex items-center justify-between mb-3">
                      <span class="text-sm font-medium text-slate-600">
                        Total Budget Multiplier
                      </span>
                      <span class="text-lg font-bold tabular-nums text-indigo-600">
                        {(totalBudgetMult() * 100).toFixed(0)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min={30}
                      max={250}
                      step={5}
                      value={totalBudgetMult() * 100}
                      onInput={(e) =>
                        setTotalBudgetMult(Number(e.currentTarget.value) / 100)
                      }
                      class="w-full h-3 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                    />
                    <div class="flex justify-between text-[10px] text-slate-400 mt-1 tabular-nums">
                      <span>30%</span>
                      <span>100%</span>
                      <span>250%</span>
                    </div>
                    <div class="mt-3 flex items-center justify-between text-sm">
                      <span class="text-slate-500">Current budget:</span>
                      <span class="font-medium tabular-nums">
                        {formatCurrency(baseBudget(), true)}
                      </span>
                    </div>
                    <div class="flex items-center justify-between text-sm mt-1">
                      <span class="text-slate-500">Simulated budget:</span>
                      <span class="font-bold text-indigo-600 tabular-nums">
                        {formatCurrency(baseBudget() * totalBudgetMult(), true)}
                      </span>
                    </div>
                  </div>
                </Show>

                <Show when={budgetMode() === "fixed"}>
                  <div class="bg-slate-50 rounded-xl p-5 text-center">
                    <p class="text-sm text-slate-500">Total budget locked at</p>
                    <p class="text-2xl font-bold text-slate-800 mt-1">
                      {formatCurrency(baseBudget(), true)}
                    </p>
                    <p class="text-xs text-slate-400 mt-1">
                      Adjust individual channel allocations in the next step
                    </p>
                  </div>
                </Show>
              </div>
            </div>
          </Show>

          {/* Step 3: Channel Mix */}
          <Show when={step() === 2}>
            <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-6">
              <div class="flex items-center gap-3 mb-5">
                <div class="flex items-center justify-center w-10 h-10 rounded-xl bg-amber-50 text-amber-600">
                  <SlidersHorizontal size={20} />
                </div>
                <div class="flex-1">
                  <h2 class="text-lg font-semibold text-slate-800">
                    Channel Mix
                  </h2>
                  <p class="text-sm text-slate-500">
                    Set per-channel multipliers and min/max constraints
                  </p>
                </div>
                <div class="text-right">
                  <p class="text-xs text-slate-500">Total Allocated</p>
                  <p class="text-lg font-bold text-indigo-600 tabular-nums">
                    {formatCurrency(totalAllocated(), true)}
                  </p>
                </div>
              </div>

              {/* Channel header */}
              <div class="flex items-center gap-4 px-1 mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                <div class="w-28">Channel</div>
                <div class="flex-1">Multiplier</div>
                <div class="w-16 text-center">Min</div>
                <div class="w-16 text-center">Max</div>
                <div class="w-20 text-right">Spend</div>
                <div class="w-16 text-right">Change</div>
              </div>

              <div class="space-y-3">
                <For each={channels()}>
                  {(ch, i) => {
                    const base = () => optData()!.current_allocation[ch] ?? 0;
                    const mult = () => channelMults()[ch] ?? 1;
                    const adjusted = () => base() * totalBudgetMult() * mult();
                    const diff = () => adjusted() - base();
                    const diffPct = () =>
                      base() > 0 ? (diff() / base()) * 100 : 0;

                    return (
                      <div class="flex items-center gap-4 py-1">
                        <div class="w-28 flex items-center gap-2">
                          <span
                            class="w-3 h-3 rounded-full shrink-0"
                            style={{
                              background: COLORS[i() % COLORS.length],
                            }}
                          />
                          <span class="text-sm font-medium text-slate-700 truncate">
                            {ch.replace(/_spend$/, "")}
                          </span>
                        </div>
                        <div class="flex-1 flex items-center gap-2">
                          <input
                            type="range"
                            min={0}
                            max={300}
                            step={5}
                            value={mult() * 100}
                            onInput={(e) =>
                              updateChannelMult(
                                ch,
                                Number(e.currentTarget.value) / 100,
                              )
                            }
                            class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                          />
                          <span class="text-xs font-mono tabular-nums text-slate-500 w-10 text-right">
                            {(mult() * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div class="w-16">
                          <input
                            type="number"
                            min={0}
                            max={3}
                            step={0.1}
                            value={(channelMins()[ch] ?? 0).toFixed(1)}
                            onInput={(e) =>
                              updateChannelMin(
                                ch,
                                Number(e.currentTarget.value),
                              )
                            }
                            class="w-full px-1.5 py-1 rounded border border-slate-200 text-xs text-center tabular-nums focus:outline-none focus:ring-1 focus:ring-indigo-400"
                          />
                        </div>
                        <div class="w-16">
                          <input
                            type="number"
                            min={0}
                            max={5}
                            step={0.1}
                            value={(channelMaxs()[ch] ?? 3).toFixed(1)}
                            onInput={(e) =>
                              updateChannelMax(
                                ch,
                                Number(e.currentTarget.value),
                              )
                            }
                            class="w-full px-1.5 py-1 rounded border border-slate-200 text-xs text-center tabular-nums focus:outline-none focus:ring-1 focus:ring-indigo-400"
                          />
                        </div>
                        <div class="w-20 text-right">
                          <span class="text-sm font-mono tabular-nums">
                            {formatCurrency(adjusted(), true)}
                          </span>
                        </div>
                        <div class="w-16 text-right">
                          <span
                            class={`text-xs font-medium tabular-nums ${
                              diff() > 0
                                ? "text-emerald-600"
                                : diff() < 0
                                  ? "text-red-500"
                                  : "text-slate-400"
                            }`}
                          >
                            {diff() >= 0 ? "+" : ""}
                            {diffPct().toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    );
                  }}
                </For>
              </div>
            </div>
          </Show>

          {/* Step 4: Summary + Run */}
          <Show when={step() === 3}>
            <div>
              {/* KPI summary */}
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                <MetricCard
                  label="Optimized Budget"
                  value={formatCurrency(totalAllocated(), true)}
                  changePct={(totalAllocated() / baseBudget() - 1) * 100}
                  changeLabel="vs current"
                  color="indigo"
                />
                <MetricCard
                  label="Est. Response"
                  value={estimatedResponse().toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  changePct={upliftPct()}
                  changeLabel="vs current"
                  color={upliftPct() >= 0 ? "emerald" : "red"}
                />
                <MetricCard
                  label="Est. ROI"
                  value={`${estimatedROI().toFixed(2)}x`}
                  delta={`Current: ${currentROI().toFixed(2)}x`}
                  color="amber"
                />
                <MetricCard
                  label="Budget Change"
                  value={`${totalBudgetMult() >= 1 ? "+" : ""}${((totalBudgetMult() - 1) * 100).toFixed(0)}%`}
                  color={totalBudgetMult() >= 1 ? "emerald" : "red"}
                />
              </div>

              {/* Before/After Comparison Table */}
              <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-6 mb-6">
                <div class="flex items-center justify-between mb-4">
                  <h3 class="text-sm font-medium text-slate-700">
                    Current vs. Optimized Comparison
                  </h3>
                  <button
                    onClick={handleExportCsv}
                    class="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium text-slate-600 hover:bg-slate-100 transition-colors"
                  >
                    <Download size={12} />
                    Export CSV
                  </button>
                </div>
                <div class="overflow-x-auto">
                  <table class="w-full text-sm">
                    <thead>
                      <tr class="border-b border-slate-200">
                        <th class="text-left py-2.5 px-3 font-semibold text-slate-600">
                          Channel
                        </th>
                        <th class="text-right py-2.5 px-3 font-semibold text-slate-600">
                          Current
                        </th>
                        <th class="text-right py-2.5 px-3 font-semibold text-slate-600">
                          Optimized
                        </th>
                        <th class="text-right py-2.5 px-3 font-semibold text-slate-600">
                          Delta
                        </th>
                        <th class="text-right py-2.5 px-3 font-semibold text-slate-600">
                          Change %
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={comparisonRows()}>
                        {(row) => (
                          <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                            <td class="py-2.5 px-3 font-medium text-slate-700">
                              {row.channel}
                            </td>
                            <td class="text-right py-2.5 px-3 tabular-nums text-slate-500">
                              {formatCurrency(row.current, true)}
                            </td>
                            <td class="text-right py-2.5 px-3 tabular-nums font-medium text-slate-800">
                              {formatCurrency(row.optimized, true)}
                            </td>
                            <td
                              class={`text-right py-2.5 px-3 tabular-nums font-medium ${
                                row.delta > 0
                                  ? "text-emerald-600"
                                  : row.delta < 0
                                    ? "text-red-500"
                                    : "text-slate-400"
                              }`}
                            >
                              {row.delta >= 0 ? "+" : ""}
                              {formatCurrency(row.delta, true)}
                            </td>
                            <td
                              class={`text-right py-2.5 px-3 tabular-nums font-medium ${
                                row.deltaPct > 0
                                  ? "text-emerald-600"
                                  : row.deltaPct < 0
                                    ? "text-red-500"
                                    : "text-slate-400"
                              }`}
                            >
                              {row.deltaPct >= 0 ? "+" : ""}
                              {row.deltaPct.toFixed(1)}%
                            </td>
                          </tr>
                        )}
                      </For>
                      {/* Totals row */}
                      <tr class="border-t-2 border-slate-300 font-semibold">
                        <td class="py-2.5 px-3 text-slate-800">Total</td>
                        <td class="text-right py-2.5 px-3 tabular-nums text-slate-600">
                          {formatCurrency(baseBudget(), true)}
                        </td>
                        <td class="text-right py-2.5 px-3 tabular-nums text-indigo-600">
                          {formatCurrency(totalAllocated(), true)}
                        </td>
                        <td
                          class={`text-right py-2.5 px-3 tabular-nums ${
                            totalAllocated() - baseBudget() >= 0
                              ? "text-emerald-600"
                              : "text-red-500"
                          }`}
                        >
                          {totalAllocated() - baseBudget() >= 0 ? "+" : ""}
                          {formatCurrency(
                            totalAllocated() - baseBudget(),
                            true,
                          )}
                        </td>
                        <td
                          class={`text-right py-2.5 px-3 tabular-nums ${
                            totalAllocated() - baseBudget() >= 0
                              ? "text-emerald-600"
                              : "text-red-500"
                          }`}
                        >
                          {totalAllocated() - baseBudget() >= 0 ? "+" : ""}
                          {(
                            ((totalAllocated() - baseBudget()) / baseBudget()) *
                            100
                          ).toFixed(1)}
                          %
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Charts */}
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <ChartCard
                  title="Current vs Optimized Allocation"
                  minHeight={350}
                  exportData={channelCompare()}
                  exportName="what-if-comparison"
                >
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 300 },
                        h(
                          BarChart,
                          {
                            data: channelCompare(),
                            layout: "vertical",
                            margin: { left: 70 },
                          },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: CHART_GRID,
                          }),
                          h(XAxis, {
                            type: "number",
                            tick: { fontSize: 11 },
                            tickFormatter: (v: number) => formatSpendTick(v),
                          }),
                          h(YAxis, {
                            type: "category",
                            dataKey: "channel",
                            tick: { fontSize: 11 },
                            width: 60,
                          }),
                          h(Tooltip, {
                            formatter: (v: number, name: string) => [
                              `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                              name,
                            ],
                            contentStyle: {
                              background: CHART_TOOLTIP_BG,
                              border: "none",
                              borderRadius: 8,
                              fontSize: 12,
                              color: "#e2e8f0",
                            },
                          }),
                          h(Legend, { wrapperStyle: { fontSize: 11 } }),
                          h(Bar, {
                            dataKey: "current",
                            name: "Current",
                            fill: "#94a3b8",
                            radius: [0, 3, 3, 0],
                            barSize: 10,
                          }),
                          h(Bar, {
                            dataKey: "optimized",
                            name: "Optimized",
                            fill: "#6366f1",
                            radius: [0, 3, 3, 0],
                            barSize: 10,
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                </ChartCard>

                <ChartCard
                  title="Budget Efficiency Frontier"
                  description="Expected response at different budget levels"
                  minHeight={350}
                >
                  <ReactChart>
                    {() =>
                      h(
                        ResponsiveContainer,
                        { width: "100%", height: 300 },
                        h(
                          LineChart,
                          { data: efficiencyData() },
                          h(CartesianGrid, {
                            strokeDasharray: "3 3",
                            stroke: CHART_GRID,
                          }),
                          h(XAxis, {
                            dataKey: "budget",
                            tick: { fontSize: 11 },
                            tickFormatter: (v: number) => formatSpendTick(v),
                          }),
                          h(YAxis, {
                            tick: { fontSize: 11 },
                            tickFormatter: (v: number) =>
                              formatCompactNumber(v),
                          }),
                          h(Tooltip, {
                            contentStyle: {
                              background: CHART_TOOLTIP_BG,
                              border: "none",
                              borderRadius: 8,
                              fontSize: 12,
                              color: "#e2e8f0",
                            },
                            formatter: (v: number) => [
                              v.toLocaleString(undefined, {
                                maximumFractionDigits: 0,
                              }),
                              "Response",
                            ],
                            labelFormatter: (v: string) =>
                              `Budget: $${Number(v).toLocaleString()}`,
                          }),
                          h(ReferenceLine, {
                            x: totalAllocated(),
                            stroke: "#6366f1",
                            strokeWidth: 2,
                            strokeDasharray: "4 4",
                            label: {
                              value: "Optimized",
                              fontSize: 10,
                              fill: "#6366f1",
                            },
                          }),
                          h(ReferenceLine, {
                            x: baseBudget(),
                            stroke: "#94a3b8",
                            strokeDasharray: "4 4",
                            label: {
                              value: "Current",
                              fontSize: 10,
                              fill: "#94a3b8",
                            },
                          }),
                          h(Line, {
                            type: "monotone",
                            dataKey: "response",
                            stroke: "#10b981",
                            strokeWidth: 2,
                            dot: false,
                          }),
                        ),
                      )
                    }
                  </ReactChart>
                </ChartCard>
              </div>

              {/* Save / Share actions */}
              <div class="bg-white rounded-xl border border-slate-200/60 shadow-sm p-6">
                <h3 class="text-sm font-medium text-slate-700 mb-4">
                  Save & Share
                </h3>
                <div class="flex items-end gap-3">
                  <div class="flex-1">
                    <label class="block text-xs font-medium text-slate-500 mb-1">
                      Scenario Name
                    </label>
                    <input
                      type="text"
                      value={scenarioName()}
                      onInput={(e) => setScenarioName(e.currentTarget.value)}
                      placeholder={`Scenario ${savedScenarios().length + 1}`}
                      class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-400 transition-colors"
                    />
                  </div>
                  <button
                    onClick={handleSaveScenario}
                    class="flex items-center gap-1.5 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium text-sm hover:bg-indigo-700 transition-colors"
                  >
                    <Save size={14} />
                    Save Scenario
                  </button>
                  <button
                    onClick={handleShareScenario}
                    class="flex items-center gap-1.5 px-4 py-2 bg-white border border-slate-200 text-slate-700 rounded-lg font-medium text-sm hover:bg-slate-50 transition-colors"
                  >
                    <Copy size={14} />
                    Share Link
                  </button>
                </div>
              </div>
            </div>
          </Show>

          {/* Navigation buttons */}
          <div class="flex items-center justify-between mt-6">
            <Show when={step() > 0} fallback={<div />}>
              <button
                onClick={() => setStep((s) => Math.max(0, s - 1))}
                class="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-slate-200 bg-white text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              >
                <ChevronLeft size={16} />
                Back
              </button>
            </Show>
            <Show when={step() < 3}>
              <button
                onClick={() => setStep((s) => Math.min(3, s + 1))}
                disabled={!canAdvance()}
                class="flex items-center gap-1.5 px-5 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed ml-auto"
              >
                Next
                <ChevronRight size={16} />
              </button>
            </Show>
          </div>
        </div>
      </Show>
    </Show>
  );
}
