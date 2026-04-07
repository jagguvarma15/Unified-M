import { createSignal, Show } from "solid-js";
import { A } from "@solidjs/router";
import DateRangePicker from "./DateRangePicker";
import {
  Play,
  Bell,
  Search,
  ChevronDown,
  LogOut,
  Building2,
  Settings,
} from "../lib/icons";
import { useHealthQuery } from "../lib/queries";
import { alertCount } from "../lib/alertStore";

interface Props {
  onRunPipeline: () => void;
  onToggleSidebar: () => void;
  sidebarCollapsed: boolean;
}

export default function TopNavbar(props: Props) {
  const health = useHealthQuery();
  const [userMenuOpen, setUserMenuOpen] = createSignal(false);

  const openPalette = () => {
    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "k",
        metaKey: true,
        bubbles: true,
        cancelable: true,
      }),
    );
  };

  const closeUserMenu = () => setUserMenuOpen(false);

  return (
    <header class="flex-shrink-0 h-14 bg-white border-b border-slate-200 flex items-center gap-2 px-4 z-30 shadow-sm relative">
      {/* Logo */}
      <A
        href="/"
        class="flex items-center gap-2 min-w-0 shrink-0"
        aria-label="Unified-M home"
      >
        <img
          src="/favicon.png"
          alt="Unified-M"
          class="h-7 w-7 rounded-lg object-contain shrink-0"
        />
        <span class="hidden sm:block text-sm font-semibold text-slate-900 tracking-tight">
          Unified&#8209;M
        </span>
      </A>

      <div class="hidden sm:block h-5 w-px bg-slate-200 mx-1 shrink-0" />

      {/* Push everything else to the right */}
      <div class="flex-1" />

      {/* ── Right-side actions ── */}

      {/* Date Range Picker */}
      <DateRangePicker />

      {/* Run Pipeline */}
      <button
        onClick={props.onRunPipeline}
        class="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 active:bg-indigo-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 shrink-0"
        aria-label="Run Pipeline"
      >
        {Play({ size: 13 })}
        <span class="hidden sm:inline">Run Pipeline</span>
      </button>

      {/* Alerts Bell */}
      <A
        href="/alerts"
        class="relative flex items-center justify-center h-8 w-8 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition-colors shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        aria-label={`Alerts${alertCount() > 0 ? ` (${alertCount()} unread)` : ""}`}
      >
        {Bell({ size: 17 })}
        <Show when={alertCount() > 0}>
          <span
            aria-hidden
            class="absolute -top-0.5 -right-0.5 h-4 min-w-[1rem] rounded-full bg-red-500 flex items-center justify-center text-[9px] font-bold text-white px-0.5 leading-none"
          >
            {alertCount() > 9 ? "9+" : alertCount()}
          </span>
        </Show>
      </A>

      {/* ⌘K Search button */}
      <button
        onClick={openPalette}
        class="hidden md:flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-500 hover:bg-slate-100 hover:border-slate-300 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 shrink-0"
        aria-label="Open command palette"
      >
        {Search({ size: 13 })}
        <span>Search</span>
        <kbd class="ml-1 inline-flex items-center gap-0.5 rounded border border-slate-200 bg-white px-1 py-0.5 text-[9px] font-medium text-slate-400 leading-none">
          ⌘K
        </kbd>
      </button>

      {/* API health dot */}
      <div
        class="hidden lg:flex items-center gap-1.5 text-[11px] shrink-0"
        title={health.data ? `API online · v${health.data.version}` : "API offline"}
      >
        <span
          aria-hidden
          class={`h-2 w-2 rounded-full ${health.data ? "bg-emerald-400" : "bg-red-400"}`}
        />
        <span class={health.data ? "text-emerald-700" : "text-red-600"}>
          {health.data ? "Live" : "Offline"}
        </span>
      </div>

      {/* User / Workspace dropdown */}
      <div class="relative shrink-0">
        <button
          onClick={() => setUserMenuOpen((v) => !v)}
          class="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-slate-600 hover:bg-slate-100 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          aria-label="User menu"
          aria-expanded={userMenuOpen()}
        >
          <div class="h-7 w-7 rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center">
            <span class="text-[11px] font-semibold text-white">JD</span>
          </div>
          <span class="hidden sm:block text-xs font-medium text-slate-700">
            Jane Doe
          </span>
          {ChevronDown({
            size: 12,
            class: `hidden sm:block text-slate-400 transition-transform ${userMenuOpen() ? "rotate-180" : ""}`,
          })}
        </button>

        <Show when={userMenuOpen()}>
          {/* Backdrop to close on outside click */}
          <div
            class="fixed inset-0 z-40"
            onClick={closeUserMenu}
            aria-hidden
          />

          <div class="absolute right-0 top-full mt-1.5 w-56 rounded-xl border border-slate-200 bg-white shadow-xl z-50 overflow-hidden">
            {/* Identity */}
            <div class="px-4 py-3 border-b border-slate-100 bg-slate-50/60">
              <p class="text-sm font-semibold text-slate-900">Jane Doe</p>
              <p class="text-xs text-slate-500 mt-0.5">jane@company.com</p>
            </div>

            {/* Workspace */}
            <div class="p-1.5">
              <div class="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">
                {Building2({ size: 14, class: "text-slate-400 shrink-0" })}
                <div class="min-w-0">
                  <p class="text-xs font-medium text-slate-700 truncate">
                    Default Workspace
                  </p>
                  <p class="text-[10px] text-slate-400">Switch workspace</p>
                </div>
              </div>
            </div>

            <div class="border-t border-slate-100 p-1.5">
              <A
                href="/settings"
                onClick={closeUserMenu}
                class="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
              >
                {Settings({ size: 14, class: "text-slate-400 shrink-0" })}
                Settings
              </A>
              <button class="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-slate-700 hover:bg-slate-50 transition-colors">
                {LogOut({ size: 14, class: "text-slate-400 shrink-0" })}
                Sign Out
              </button>
            </div>
          </div>
        </Show>
      </div>
    </header>
  );
}
