import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  BarChart3,
  Target,
  TrendingUp,
  History,
  Activity,
  Database,
  Stethoscope,
  DollarSign,
  Calculator,
  Settings,
  ChevronDown,
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
      { to: "/runs", label: "Runs", icon: History },
    ],
  },
  {
    title: "Analysis",
    items: [
  { to: "/contributions", label: "Contributions", icon: BarChart3 },
  { to: "/curves", label: "Response Curves", icon: TrendingUp },
      { to: "/roas", label: "ROAS Analysis", icon: DollarSign },
      { to: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
    ],
  },
  {
    title: "Optimization",
    items: [
      { to: "/optimization", label: "Budget Optimizer", icon: Target },
      { to: "/scenarios", label: "Scenario Planner", icon: Calculator },
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
    <div className="flex h-screen overflow-hidden">
      {/* ---- Sidebar ---- */}
      <aside className="w-64 flex-shrink-0 bg-slate-900 text-slate-200 flex flex-col">
        <div className="px-6 pt-7 pb-5">
          <h1 className="text-xl font-bold text-white tracking-tight">
            Unified-M
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Marketing Measurement Platform
          </p>
        </div>

        <nav className="flex-1 px-3 space-y-4 overflow-y-auto pb-4">
          {NAV_SECTIONS.map((section) => (
            <div key={section.title}>
              <button
                onClick={() => toggleSection(section.title)}
                className="flex items-center justify-between w-full px-3 py-1.5 text-[10px] font-semibold text-slate-500 uppercase tracking-wider hover:text-slate-300 transition-colors"
              >
                {section.title}
                <ChevronDown
                  size={12}
                  className={`transition-transform ${collapsed[section.title] ? "-rotate-90" : ""}`}
                />
              </button>
              {!collapsed[section.title] && (
                <div className="mt-1 space-y-0.5">
                  {section.items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-indigo-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`
              }
            >
                      <Icon size={17} />
              {label}
            </NavLink>
                  ))}
                </div>
              )}
            </div>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-700/50">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Activity
              size={14}
              className={health ? "text-emerald-400" : "text-red-400"}
            />
            {health ? "API connected" : "API offline"}
          </div>
          {health?.latest_run && (
            <p className="text-xs text-slate-500 mt-1 truncate">
              Run: {health.latest_run.slice(0, 20)}
            </p>
          )}
          {health && (
            <p className="text-[10px] text-slate-600 mt-0.5">
              v{health.version}
            </p>
          )}
        </div>
      </aside>

      {/* ---- Main ---- */}
      <main className="flex-1 overflow-auto bg-slate-50">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
