import { NavLink, Outlet } from "react-router-dom";
import PageErrorBoundary from "./PageErrorBoundary";
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
  Plug,
  Crosshair,
  Shield,
  ClipboardCheck,
  Zap,
  Gauge,
  FileText,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { api, type HealthData } from "../lib/api";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: "Overview",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard },
      { to: "/data", label: "Data", icon: Database },
      { to: "/datapoint", label: "Connect to Datapoint", icon: Plug },
      { to: "/runs", label: "Runs", icon: History },
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

export default function Layout() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null));

    const id = setInterval(() => {
      api
        .health()
        .then(setHealth)
        .catch(() => setHealth(null));
    }, 10_000);

    return () => clearInterval(id);
  }, []);

  const toggleSection = (title: string) => {
    setCollapsed((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* ---- Sidebar ---- */}
      <aside className="w-64 flex-shrink-0 flex flex-col bg-slate-900 text-slate-200 ring-1 ring-slate-800/50">
        <div className="px-5 pt-6 pb-4">
          <h1 className="text-lg font-bold tracking-tight text-white">
            Unified-M
          </h1>
          <p className="mt-0.5 text-[11px] text-slate-400">
            Marketing Measurement
          </p>
        </div>

        <nav className="flex-1 px-2.5 space-y-3 overflow-y-auto py-2">
          {NAV_SECTIONS.map((section) => (
            <div key={section.title}>
              <button
                onClick={() => toggleSection(section.title)}
                className="flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-400 transition-colors"
              >
                {section.title}
                <ChevronDown
                  size={12}
                  className={`shrink-0 transition-transform ${collapsed[section.title] ? "-rotate-90" : ""}`}
                />
              </button>
              {!collapsed[section.title] && (
                <div className="mt-0.5 space-y-0.5">
                  {section.items.map(({ to, label, icon: Icon }) => (
                    <NavLink
                      key={to}
                      to={to}
                      end={to === "/"}
                      className={({ isActive }) =>
                        `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors ${
                          isActive
                            ? "bg-indigo-600 text-white"
                            : "text-slate-300 hover:bg-slate-800 hover:text-white"
                        }`
                      }
                    >
                      <Icon size={16} className="shrink-0" />
                      {label}
                    </NavLink>
                  ))}
                </div>
              )}
            </div>
          ))}
        </nav>

        <div className="border-t border-slate-700/60 p-3">
          <div className="flex items-center gap-2 text-[11px] text-slate-400">
            <span
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                health ? "bg-emerald-400" : "bg-red-400"
              }`}
              aria-hidden
            />
            {health ? "API connected" : "API offline"}
          </div>
          {health?.latest_run && (
            <p className="mt-1 truncate text-[11px] text-slate-500" title={health.latest_run}>
              {health.latest_run.slice(0, 14)}…
            </p>
          )}
          {health && (
            <p className="mt-0.5 text-[10px] text-slate-600">v{health.version}</p>
          )}
        </div>
      </aside>

      {/* ---- Main ---- */}
      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl px-6 py-8 min-h-[400px]">
          <PageErrorBoundary>
            <Outlet />
          </PageErrorBoundary>
        </div>
      </main>
    </div>
  );
}
