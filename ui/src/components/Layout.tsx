import { A, useLocation } from "@solidjs/router";
import PageErrorBoundary from "./PageErrorBoundary";
import PipelineRunner from "./PipelineRunner";
import {
  LayoutDashboard,
  BarChart3,
  Target,
  TrendingUp,
  History,
  Database,
  Stethoscope,
  DollarSign,
  Calculator,
  Settings,
  ChevronDown,
  Link2,
  Crosshair,
  Shield,
  ClipboardCheck,
  Zap,
  Gauge,
  FileText,
  Play,
  type LucideIcon,
} from "../lib/icons";
import { createSignal, createEffect, For, Show, Suspense, type JSX } from "solid-js";
import { useHealthQuery } from "../lib/queries";
import { useAnalyticsMode } from "../lib/analyticsMode";
import { useQueryClient } from "@tanstack/solid-query";
import { trackPageView } from "../lib/telemetry";
import { density, setDensity, type Density } from "../lib/density";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

// Always-visible items — never hidden, never collapsible
const PINNED_NAV: NavItem[] = [
  { to: "/data", label: "Data", icon: Database },
  { to: "/runs", label: "Runs", icon: History },
];

const NAV_SECTIONS: NavSection[] = [
  {
    title: "Overview",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard },
      { to: "/datapoint", label: "Connections", icon: Link2 },
    ],
  },
  {
    title: "Analysis",
    items: [
      { to: "/contributions", label: "Contributions", icon: BarChart3 },
      { to: "/curves", label: "Response Curves", icon: TrendingUp },
      { to: "/roas", label: "ROAS Analysis", icon: DollarSign },
      { to: "/channel-insights", label: "Channel Insights", icon: Zap },
      { to: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
    ],
  },
  {
    title: "Optimization",
    items: [
      { to: "/optimization", label: "Budget Optimizer", icon: Target },
      { to: "/scenarios", label: "Scenario Planner", icon: Calculator },
      { to: "/spend-pacing", label: "Spend Pacing", icon: Gauge },
    ],
  },
  {
    title: "Monitoring",
    items: [
      { to: "/calibration", label: "Calibration", icon: Crosshair },
      { to: "/stability", label: "Stability", icon: Shield },
      { to: "/data-quality", label: "Data Quality", icon: ClipboardCheck },
    ],
  },
  {
    title: "Reports",
    items: [
      { to: "/report", label: "Executive Summary", icon: FileText },
    ],
  },
  {
    title: "Configuration",
    items: [
      { to: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

export default function Layout(props: { children?: JSX.Element }) {
  const [collapsed, setCollapsed] = createSignal<Record<string, boolean>>({});
  const [pipelineOpen, setPipelineOpen] = createSignal(false);
  const health = useHealthQuery();
  const { analyticsEnabled, setAnalyticsEnabled } = useAnalyticsMode();
  const queryClient = useQueryClient();
  let didClear = false;
  const location = useLocation();

  createEffect(() => {
    if (health.isError) {
      setAnalyticsEnabled(false);
      if (!didClear) {
        queryClient.clear();
        didClear = true;
      }
    } else {
      didClear = false;
      // Enable analytics when API is up and at least one run exists
      if (health.data?.latest_run) {
        setAnalyticsEnabled(true);
      }
    }
  });

  createEffect(() => {
    trackPageView(location.pathname);
  });

  const toggleSection = (title: string) => {
    setCollapsed((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  return (
    <div class="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <aside class="w-64 flex-shrink-0 flex flex-col bg-slate-900 text-slate-200 ring-1 ring-slate-800/50">
        <div class="px-5 pt-6 pb-4">
          <h1 class="text-lg font-bold tracking-tight text-white">
            Unified-M
          </h1>
          <p class="mt-0.5 text-[11px] text-slate-400">
            Marketing Measurement
          </p>
        </div>

        {/* Run Pipeline button */}
        <div class="px-3 pb-3">
          <button
            onClick={() => setPipelineOpen(true)}
            class="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
          >
            <Play size={14} />
            Run Pipeline
          </button>
        </div>

        <nav class="flex-1 px-2.5 overflow-y-auto py-2 space-y-0.5">
          {/* Pinned items — always visible regardless of analytics state */}
          <For each={PINNED_NAV}>
            {(item) => (
              <A
                href={item.to}
                activeClass="bg-indigo-600 text-white"
                inactiveClass="text-slate-300 hover:bg-slate-800 hover:text-white"
                class="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors"
              >
                {item.icon({ size: 16, class: "shrink-0" })}
                {item.label}
              </A>
            )}
          </For>

          {/* Divider */}
          <div class="my-2 border-t border-slate-800" />

          {/* Collapsible sections — analytics-gated */}
          <For each={NAV_SECTIONS}>
            {(section) => {
              const items = (): NavItem[] => {
                if (!analyticsEnabled()) {
                  // Before first run: only show Configuration
                  if (section.title !== "Configuration") return [];
                }
                return section.items;
              };

              return (
                <Show when={items().length > 0}>
                  <div class="pt-1">
                    <button
                      onClick={() => toggleSection(section.title)}
                      aria-expanded={!collapsed()[section.title]}
                      aria-label={`${collapsed()[section.title] ? "Expand" : "Collapse"} ${section.title}`}
                      class="flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-400 transition-colors"
                    >
                      {section.title}
                      <ChevronDown
                        size={12}
                        aria-hidden
                        class={`shrink-0 transition-transform ${collapsed()[section.title] ? "-rotate-90" : ""}`}
                      />
                    </button>
                    <Show when={!collapsed()[section.title]}>
                      <div class="mt-0.5 space-y-0.5">
                        <For each={items()}>
                          {(item) => (
                            <A
                              href={item.to}
                              end={item.to === "/"}
                              activeClass="bg-indigo-600 text-white"
                              inactiveClass="text-slate-300 hover:bg-slate-800 hover:text-white"
                              class="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors"
                            >
                              {item.icon({ size: 16, class: "shrink-0" })}
                              {item.label}
                            </A>
                          )}
                        </For>
                      </div>
                    </Show>
                  </div>
                </Show>
              );
            }}
          </For>
        </nav>

        <div class="border-t border-slate-700/60 p-3 space-y-2">
          <div class="flex items-center gap-2 text-[11px] text-slate-400">
            <span
              class={`h-1.5 w-1.5 shrink-0 rounded-full ${
                health.data ? "bg-emerald-400" : "bg-red-400"
              }`}
              aria-hidden
            />
            {health.data ? "API connected" : "API offline"}
          </div>
          <Show when={health.data?.latest_run}>
            <p class="truncate text-[11px] text-slate-500" title={health.data!.latest_run!}>
              {health.data!.latest_run!.slice(0, 14)}…
            </p>
          </Show>
          <Show when={health.data}>
            <p class="text-[10px] text-slate-600">v{health.data!.version}</p>
          </Show>

          {/* Density toggle */}
          <div class="flex items-center gap-1.5 pt-0.5">
            <span class="text-[10px] text-slate-500 select-none">Density</span>
            <div class="flex rounded-md overflow-hidden border border-slate-700" role="group" aria-label="UI density">
              {(
                [
                  { key: "compact" as Density, label: "S", title: "Compact" },
                  { key: "default" as Density, label: "M", title: "Default" },
                  { key: "comfortable" as Density, label: "L", title: "Comfortable" },
                ] as const
              ).map(({ key, label, title }) => (
                <button
                  onClick={() => setDensity(key)}
                  title={title}
                  aria-pressed={density() === key}
                  class={`px-2 py-0.5 text-[10px] font-semibold transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-400 ${
                    density() === key
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main class="flex-1 overflow-auto">
        <div class="mx-auto max-w-7xl px-6 py-8 min-h-[400px]">
          <Suspense
            fallback={
              <div class="flex items-center justify-center h-64">
                <div
                  class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"
                  role="status"
                  aria-label="Loading"
                />
              </div>
            }
          >
            <PageErrorBoundary>
              {props.children}
            </PageErrorBoundary>
          </Suspense>
        </div>
      </main>

      {/* Pipeline Runner slide-out */}
      <PipelineRunner open={pipelineOpen()} onClose={() => setPipelineOpen(false)} />
    </div>
  );
}
