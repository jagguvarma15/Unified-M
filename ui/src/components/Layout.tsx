import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  BarChart3,
  Target,
  TrendingUp,
  History,
  Activity,
} from "lucide-react";
import { useEffect, useState } from "react";
import { api, type HealthData } from "../lib/api";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/contributions", label: "Contributions", icon: BarChart3 },
  { to: "/optimization", label: "Optimization", icon: Target },
  { to: "/curves", label: "Response Curves", icon: TrendingUp },
  { to: "/runs", label: "Runs", icon: History },
];

export default function Layout() {
  const [health, setHealth] = useState<HealthData | null>(null);

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

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ---- Sidebar ---- */}
      <aside className="w-64 flex-shrink-0 bg-slate-900 text-slate-200 flex flex-col">
        <div className="px-6 pt-7 pb-5">
          <h1 className="text-xl font-bold text-white tracking-tight">
            Unified-M
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Marketing Measurement
          </p>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-indigo-600 text-white"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
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
