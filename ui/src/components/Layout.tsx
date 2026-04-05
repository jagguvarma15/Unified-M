import { A, useLocation } from "@solidjs/router";
import PageErrorBoundary from "./PageErrorBoundary";
import PipelineRunner from "./PipelineRunner";
import DateRangePicker from "./DateRangePicker";
import CommandPalette from "./CommandPalette";
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
  PanelLeftClose,
  PanelLeftOpen,
  Command,
  Bell,
  MapIcon,
  SlidersHorizontal,
  GripVertical,
  type LucideIcon,
} from "../lib/icons";
import {
  createSignal,
  createEffect,
  For,
  Show,
  Suspense,
  type JSX,
} from "solid-js";
import { useHealthQuery } from "../lib/queries";
import { useAnalyticsMode } from "../lib/analyticsMode";
import { useQueryClient } from "@tanstack/solid-query";
import { trackPageView } from "../lib/telemetry";
import { density, setDensity, type Density } from "../lib/density";
import Tooltip from "./Tooltip";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

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
      { to: "/attribution", label: "Attribution Explorer", icon: MapIcon },
    ],
  },
  {
    title: "Optimization",
    items: [
      { to: "/optimization", label: "Budget Optimizer", icon: Target },
      { to: "/scenarios", label: "Scenario Planner", icon: Calculator },
      {
        to: "/budget-simulator",
        label: "Budget Simulator",
        icon: SlidersHorizontal,
      },
      { to: "/spend-pacing", label: "Spend Pacing", icon: Gauge },
    ],
  },
  {
    title: "Monitoring",
    items: [
      { to: "/calibration", label: "Calibration", icon: Crosshair },
      { to: "/stability", label: "Stability", icon: Shield },
      { to: "/data-quality", label: "Data Quality", icon: ClipboardCheck },
      { to: "/alerts", label: "Alerts Center", icon: Bell },
    ],
  },
  {
    title: "Reports",
    items: [
      { to: "/report", label: "Executive Summary", icon: FileText },
      { to: "/report-builder", label: "Report Builder", icon: GripVertical },
    ],
  },
  {
    title: "Configuration",
    items: [{ to: "/settings", label: "Settings", icon: Settings }],
  },
];

const SIDEBAR_STORAGE_KEY = "sidebar-collapsed";

export default function Layout(props: { children?: JSX.Element }) {
  const [sectionCollapsed, setSectionCollapsed] = createSignal<
    Record<string, boolean>
  >({});
  const [sidebarCollapsed, setSidebarCollapsed] = createSignal(
    (() => {
      try {
        return localStorage.getItem(SIDEBAR_STORAGE_KEY) === "true";
      } catch (_e) {
        return false;
      }
    })(),
  );
  const [pipelineOpen, setPipelineOpen] = createSignal(false);
  const health = useHealthQuery();
  const { analyticsEnabled, setAnalyticsEnabled } = useAnalyticsMode();
  const queryClient = useQueryClient();
  let didClear = false;
  let prevLatestRun: string | null = null;
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
      const latestRun = health.data?.latest_run ?? null;
      if (latestRun) {
        setAnalyticsEnabled(true);
        if (latestRun !== prevLatestRun) {
          prevLatestRun = latestRun;
          queryClient.invalidateQueries();
        }
      }
    }
  });

  createEffect(() => {
    trackPageView(location.pathname);
  });

  const toggleSection = (title: string) => {
    setSectionCollapsed((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  const toggleSidebar = () => {
    const next = !sidebarCollapsed();
    setSidebarCollapsed(next);
    try {
      localStorage.setItem(SIDEBAR_STORAGE_KEY, String(next));
    } catch {}
  };

  const sc = () => sidebarCollapsed();

  return (
    <div class="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <aside
        class={`flex-shrink-0 flex flex-col bg-slate-900 text-slate-200 ring-1 ring-slate-800/50 transition-all duration-200 ${sc() ? "w-[52px]" : "w-64"}`}
      >
        {/* Brand */}
        <div class={`pt-5 pb-3 ${sc() ? "px-2" : "px-5"}`}>
          <Show
            when={!sc()}
            fallback={
              <div class="flex items-center justify-center">
                <span class="text-sm font-bold text-white">M</span>
              </div>
            }
          >
            <h1 class="text-lg font-bold tracking-tight text-white">
              Unified-M
            </h1>
            <p class="mt-0.5 text-[11px] text-slate-400">
              Marketing Measurement
            </p>
          </Show>
        </div>

        {/* Run Pipeline */}
        <div class={sc() ? "px-1.5 pb-2" : "px-3 pb-3"}>
          <Show
            when={!sc()}
            fallback={
              <Tooltip content="Run Pipeline" side="right">
                <button
                  onClick={() => setPipelineOpen(true)}
                  class="flex w-full items-center justify-center rounded-lg bg-indigo-600 p-2 text-white hover:bg-indigo-700 transition-colors"
                >
                  <Play size={14} />
                </button>
              </Tooltip>
            }
          >
            <button
              onClick={() => setPipelineOpen(true)}
              class="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
            >
              <Play size={14} />
              Run Pipeline
            </button>
          </Show>
        </div>

        <nav
          class={`flex-1 overflow-y-auto py-2 space-y-0.5 ${sc() ? "px-1.5" : "px-2.5"}`}
        >
          {/* Pinned */}
          <For each={PINNED_NAV}>
            {(item) => (
              <Show
                when={!sc()}
                fallback={
                  <Tooltip content={item.label} side="right">
                    <A
                      href={item.to}
                      activeClass="bg-indigo-600 text-white"
                      inactiveClass="text-slate-300 hover:bg-slate-800 hover:text-white"
                      class="flex items-center justify-center rounded-md p-2 transition-colors"
                    >
                      {item.icon({ size: 16, class: "shrink-0" })}
                    </A>
                  </Tooltip>
                }
              >
                <A
                  href={item.to}
                  activeClass="bg-indigo-600 text-white"
                  inactiveClass="text-slate-300 hover:bg-slate-800 hover:text-white"
                  class="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors"
                >
                  {item.icon({ size: 16, class: "shrink-0" })}
                  {item.label}
                </A>
              </Show>
            )}
          </For>

          <div class="my-2 border-t border-slate-800" />

          {/* Sections */}
          <For each={NAV_SECTIONS}>
            {(section) => {
              const items = (): NavItem[] => {
                if (!analyticsEnabled()) {
                  if (section.title !== "Configuration") return [];
                }
                return section.items;
              };

              return (
                <Show when={items().length > 0}>
                  <div class="pt-1">
                    <Show
                      when={!sc()}
                      fallback={
                        <div class="my-1 border-t border-slate-800/40" />
                      }
                    >
                      <button
                        onClick={() => toggleSection(section.title)}
                        aria-expanded={!sectionCollapsed()[section.title]}
                        aria-label={`${sectionCollapsed()[section.title] ? "Expand" : "Collapse"} ${section.title}`}
                        class="flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-400 transition-colors"
                      >
                        {section.title}
                        <ChevronDown
                          size={12}
                          aria-hidden
                          class={`shrink-0 transition-transform ${sectionCollapsed()[section.title] ? "-rotate-90" : ""}`}
                        />
                      </button>
                    </Show>
                    <Show when={sc() || !sectionCollapsed()[section.title]}>
                      <div class={`${sc() ? "" : "mt-0.5"} space-y-0.5`}>
                        <For each={items()}>
                          {(item) => (
                            <Show
                              when={!sc()}
                              fallback={
                                <Tooltip content={item.label} side="right">
                                  <A
                                    href={item.to}
                                    end={item.to === "/"}
                                    activeClass="bg-indigo-600 text-white"
                                    inactiveClass="text-slate-300 hover:bg-slate-800 hover:text-white"
                                    class="flex items-center justify-center rounded-md p-2 transition-colors"
                                  >
                                    {item.icon({ size: 16, class: "shrink-0" })}
                                  </A>
                                </Tooltip>
                              }
                            >
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
                            </Show>
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

        {/* Footer */}
        <div
          class={`border-t border-slate-700/60 space-y-2 ${sc() ? "p-1.5" : "p-3"}`}
        >
          {/* Collapse toggle */}
          <button
            onClick={toggleSidebar}
            title={sc() ? "Expand sidebar" : "Collapse sidebar"}
            class={`flex items-center rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors ${sc() ? "justify-center p-2 w-full" : "gap-2 px-2.5 py-1.5 w-full text-[11px]"}`}
          >
            {sc() ? PanelLeftOpen({ size: 14 }) : PanelLeftClose({ size: 14 })}
            <Show when={!sc()}>Collapse</Show>
          </button>

          <Show when={!sc()}>
            <div class="flex items-center gap-2 text-[11px] text-slate-400">
              <span
                class={`h-1.5 w-1.5 shrink-0 rounded-full ${health.data ? "bg-emerald-400" : "bg-red-400"}`}
                aria-hidden
              />
              {health.data ? "API connected" : "API offline"}
            </div>
            <Show when={health.data?.latest_run}>
              <p
                class="truncate text-[11px] text-slate-500"
                title={health.data!.latest_run!}
              >
                {health.data!.latest_run!.slice(0, 14)}…
              </p>
            </Show>
            <Show when={health.data}>
              <p class="text-[10px] text-slate-600">v{health.data!.version}</p>
            </Show>

            {/* Density toggle */}
            <div class="flex items-center gap-1.5 pt-0.5">
              <span class="text-[10px] text-slate-500 select-none">
                Density
              </span>
              <div
                class="flex rounded-md overflow-hidden border border-slate-700"
                role="group"
                aria-label="UI density"
              >
                {(
                  [
                    { key: "compact" as Density, label: "S", title: "Compact" },
                    { key: "default" as Density, label: "M", title: "Default" },
                    {
                      key: "comfortable" as Density,
                      label: "L",
                      title: "Comfortable",
                    },
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

            {/* Command palette hint */}
            <div class="flex items-center gap-1.5 pt-0.5">
              <kbd class="inline-flex items-center gap-0.5 rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-[10px] font-medium text-slate-400">
                ⌘K
              </kbd>
              <span class="text-[10px] text-slate-500">Command palette</span>
            </div>
          </Show>
        </div>
      </aside>

      {/* Main */}
      <main class="flex-1 overflow-auto">
        {/* Top bar with date range picker */}
        <div class="sticky top-0 z-20 bg-white/80 backdrop-blur-sm border-b border-slate-200/60 px-6 py-2 flex items-center justify-end gap-3">
          <DateRangePicker />
        </div>
        <div class="mx-auto max-w-7xl px-6 py-6 min-h-[400px]">
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
            <PageErrorBoundary>{props.children}</PageErrorBoundary>
          </Suspense>
        </div>
      </main>

      {/* Pipeline Runner slide-out */}
      <PipelineRunner
        open={pipelineOpen()}
        onClose={() => setPipelineOpen(false)}
      />

      {/* Command Palette */}
      <CommandPalette />
    </div>
  );
}
