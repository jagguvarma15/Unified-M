import { A, useLocation } from "@solidjs/router";
import PageErrorBoundary from "./PageErrorBoundary";
import PipelineRunner from "./PipelineRunner";
import CommandPalette from "./CommandPalette";
import TopNavbar from "./TopNavbar";
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
  PanelLeftClose,
  PanelLeftOpen,
  MapIcon,
  GripVertical,
  FlaskConical,
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
import { useAnalyticsMode } from "../lib/analyticsMode";
import { useQueryClient } from "@tanstack/solid-query";
import { trackPageView } from "../lib/telemetry";
import { density, setDensity, type Density } from "../lib/density";
import { useHealthQuery } from "../lib/queries";
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

const PINNED_NAV: NavItem[] = [{ to: "/data", label: "Data", icon: Database }];

const NAV_SECTIONS: NavSection[] = [
  {
    title: "Measure",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard },
      { to: "/contributions", label: "Contributions", icon: BarChart3 },
      { to: "/roas", label: "ROAS Analysis", icon: DollarSign },
      { to: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
    ],
  },
  {
    title: "Plan",
    items: [
      { to: "/optimization", label: "Budget Optimizer", icon: Target },
      { to: "/what-if", label: "What-If Studio", icon: Calculator },
      { to: "/spend-pacing", label: "Spend Pacing", icon: Gauge },
    ],
  },
  {
    title: "Experiment",
    items: [
      { to: "/calibration", label: "Calibration", icon: Crosshair },
      {
        to: "/experiment-roadmap",
        label: "Experiment Roadmap",
        icon: FlaskConical,
      },
      { to: "/stability", label: "Stability", icon: Shield },
    ],
  },
  {
    title: "Explore",
    items: [
      { to: "/curves", label: "Response Curves", icon: TrendingUp },
      { to: "/channel-insights", label: "Channel Insights", icon: Zap },
      { to: "/attribution", label: "Attribution Explorer", icon: MapIcon },
      { to: "/data-quality", label: "Data Quality", icon: ClipboardCheck },
    ],
  },
  {
    title: "Share",
    items: [
      { to: "/report", label: "Executive Summary", icon: FileText },
      { to: "/report-builder", label: "Report Builder", icon: GripVertical },
    ],
  },
];

const BOTTOM_NAV: NavItem[] = [
  { to: "/datapoint", label: "Connections", icon: Link2 },
  { to: "/settings", label: "Settings", icon: Settings },
  { to: "/runs", label: "Runs", icon: History },
];

const MOBILE_TAB_NAV: NavItem[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/optimization", label: "Optimize", icon: Target },
  { to: "/what-if", label: "Scenarios", icon: Calculator },
  { to: "/curves", label: "Explore", icon: TrendingUp },
  { to: "/settings", label: "Settings", icon: Settings },
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
    <div class="flex flex-col h-screen overflow-hidden bg-slate-50">
      {/* ── Top Navbar (full width) ── */}
      <TopNavbar
        onRunPipeline={() => setPipelineOpen(true)}
        onToggleSidebar={toggleSidebar}
        sidebarCollapsed={sc()}
      />

      {/* ── Body: Sidebar + Main ── */}
      <div class="flex flex-1 overflow-hidden">
        {/* Sidebar — hidden on mobile, visible md+ */}
        <aside
          class={`hidden md:flex flex-shrink-0 flex-col bg-slate-900 text-slate-200 ring-1 ring-slate-800/50 transition-all duration-200 ${sc() ? "w-[52px]" : "w-60"}`}
        >
          <nav
            class={`flex-1 overflow-y-auto py-3 space-y-0.5 ${sc() ? "px-1.5" : "px-2.5"}`}
          >
            {/* Pinned items (Data + Runs) */}
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
                    if (section.title !== "Measure") return [];
                    if (section.title === "Measure")
                      return section.items.filter((i) => i.to === "/");
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
                                      {item.icon({
                                        size: 16,
                                        class: "shrink-0",
                                      })}
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
            {/* Bottom utility items */}
            <div class="my-2 border-t border-slate-800" />
            <div class="space-y-0.5">
              <For each={BOTTOM_NAV}>
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
            </div>
          </nav>

          {/* Sidebar Footer — collapse toggle + density */}
          <div
            class={`border-t border-slate-700/60 space-y-2 ${sc() ? "p-1.5" : "p-3"}`}
          >
            <button
              onClick={toggleSidebar}
              title={sc() ? "Expand sidebar" : "Collapse sidebar"}
              class={`flex items-center rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors ${sc() ? "justify-center p-2 w-full" : "gap-2 px-2.5 py-1.5 w-full text-[11px]"}`}
            >
              {sc()
                ? PanelLeftOpen({ size: 14 })
                : PanelLeftClose({ size: 14 })}
              <Show when={!sc()}>Collapse</Show>
            </button>

            {/* Density toggle */}
            <Show when={!sc()}>
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
                      {
                        key: "compact" as Density,
                        label: "S",
                        title: "Compact",
                      },
                      {
                        key: "default" as Density,
                        label: "M",
                        title: "Default",
                      },
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

              <Show when={health.data?.latest_run}>
                <p
                  class="truncate text-[10px] text-slate-500 pt-0.5"
                  title={health.data!.latest_run!}
                >
                  Run: {health.data!.latest_run!.slice(0, 14)}…
                </p>
              </Show>
            </Show>
          </div>
        </aside>

        {/* Main content */}
        <main class="flex-1 overflow-auto">
          <div class="mx-auto max-w-7xl px-4 py-4 md:px-6 md:py-6 pb-20 md:pb-6 min-h-[400px]">
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
      </div>

      {/* ── Mobile Bottom Tab Bar (visible below md) ── */}
      <nav class="md:hidden fixed bottom-0 inset-x-0 z-40 border-t border-slate-200 bg-white/95 backdrop-blur-sm safe-area-pb">
        <div class="flex items-stretch justify-around">
          <For each={MOBILE_TAB_NAV}>
            {(item) => (
              <A
                href={item.to}
                end={item.to === "/"}
                activeClass="text-indigo-600"
                inactiveClass="text-slate-400"
                class="flex flex-1 flex-col items-center gap-0.5 py-2 text-[10px] font-medium transition-colors"
              >
                {item.icon({ size: 20, class: "shrink-0" })}
                {item.label}
              </A>
            )}
          </For>
        </div>
      </nav>

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
