import { createSignal, createEffect, For, Show, onCleanup } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { Search } from "../lib/icons";

interface CommandItem {
  id: string;
  label: string;
  section: string;
  href: string;
  keywords?: string;
}

const COMMANDS: CommandItem[] = [
  // Measure
  {
    id: "dashboard",
    label: "Dashboard",
    section: "Measure",
    href: "/",
    keywords: "home overview",
  },
  {
    id: "contributions",
    label: "Contributions",
    section: "Measure",
    href: "/contributions",
    keywords: "channel share",
  },
  {
    id: "roas",
    label: "ROAS Analysis",
    section: "Measure",
    href: "/roas",
    keywords: "return spend",
  },
  {
    id: "diagnostics",
    label: "Diagnostics",
    section: "Measure",
    href: "/diagnostics",
    keywords: "model fit residual",
  },
  // Plan
  {
    id: "optimization",
    label: "Budget Optimizer",
    section: "Plan",
    href: "/optimization",
    keywords: "allocate budget",
  },
  {
    id: "what-if",
    label: "What-If Studio",
    section: "Plan",
    href: "/what-if",
    keywords: "scenario simulate what-if",
  },
  {
    id: "spend-pacing",
    label: "Spend Pacing",
    section: "Plan",
    href: "/spend-pacing",
    keywords: "pace track",
  },
  // Experiment
  {
    id: "calibration",
    label: "Calibration",
    section: "Experiment",
    href: "/calibration",
    keywords: "experiment test",
  },
  {
    id: "experiment-roadmap",
    label: "Experiment Roadmap",
    section: "Experiment",
    href: "/experiment-roadmap",
    keywords: "plan lift test roadmap",
  },
  {
    id: "stability",
    label: "Stability",
    section: "Experiment",
    href: "/stability",
    keywords: "drift parameter",
  },
  // Explore
  {
    id: "curves",
    label: "Response Curves",
    section: "Explore",
    href: "/curves",
    keywords: "saturation diminishing",
  },
  {
    id: "channel-insights",
    label: "Channel Insights",
    section: "Explore",
    href: "/channel-insights",
    keywords: "marginal roi",
  },
  {
    id: "attribution",
    label: "Attribution Explorer",
    section: "Explore",
    href: "/attribution",
    keywords: "sankey touchpoint",
  },
  {
    id: "data-quality",
    label: "Data Quality",
    section: "Explore",
    href: "/data-quality",
    keywords: "gate check",
  },
  // Share
  {
    id: "report",
    label: "Executive Summary",
    section: "Share",
    href: "/report",
    keywords: "summary pdf",
  },
  {
    id: "report-builder",
    label: "Report Builder",
    section: "Share",
    href: "/report-builder",
    keywords: "drag drop canvas pdf ppt",
  },
  // Utilities
  {
    id: "connections",
    label: "Connections",
    section: "Utilities",
    href: "/datapoint",
    keywords: "connectors sources",
  },
  {
    id: "settings",
    label: "Settings",
    section: "Utilities",
    href: "/settings",
    keywords: "config preferences",
  },
  {
    id: "data",
    label: "Data",
    section: "Utilities",
    href: "/data",
    keywords: "upload import",
  },
  {
    id: "runs",
    label: "Runs",
    section: "Utilities",
    href: "/runs",
    keywords: "pipeline history",
  },
];

export default function CommandPalette() {
  const [open, setOpen] = createSignal(false);
  const [query, setQuery] = createSignal("");
  const [selectedIndex, setSelectedIndex] = createSignal(0);
  const navigate = useNavigate();

  const filtered = () => {
    const q = query().toLowerCase().trim();
    if (!q) return COMMANDS;
    return COMMANDS.filter(
      (c) =>
        c.label.toLowerCase().includes(q) ||
        c.section.toLowerCase().includes(q) ||
        c.keywords?.toLowerCase().includes(q),
    );
  };

  const grouped = () => {
    const groups: Record<string, CommandItem[]> = {};
    for (const item of filtered()) {
      (groups[item.section] ??= []).push(item);
    }
    return Object.entries(groups);
  };

  const flatItems = () => filtered();

  const handleKeydown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      setOpen(true);
      setQuery("");
      setSelectedIndex(0);
    }
    if (!open()) return;
    if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, flatItems().length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const items = flatItems();
      if (items[selectedIndex()]) {
        navigate(items[selectedIndex()].href);
        setOpen(false);
      }
    }
  };

  createEffect(() => {
    document.addEventListener("keydown", handleKeydown);
    onCleanup(() => document.removeEventListener("keydown", handleKeydown));
  });

  const select = (item: CommandItem) => {
    navigate(item.href);
    setOpen(false);
  };

  return (
    <Show when={open()}>
      <div class="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
        <div
          class="absolute inset-0 bg-black/30 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        />
        <div class="relative w-full max-w-lg rounded-xl bg-white shadow-2xl border border-slate-200 overflow-hidden">
          <div class="flex items-center gap-3 border-b border-slate-200 px-4 py-3">
            <Search size={18} class="text-slate-400 shrink-0" />
            <input
              ref={(el) => setTimeout(() => el.focus(), 10)}
              type="text"
              value={query()}
              onInput={(e) => {
                setQuery(e.currentTarget.value);
                setSelectedIndex(0);
              }}
              placeholder="Search pages, commands..."
              class="flex-1 bg-transparent text-sm text-slate-900 placeholder-slate-400 outline-none"
            />
            <kbd class="hidden sm:inline-flex items-center gap-0.5 rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
              ESC
            </kbd>
          </div>

          <div class="max-h-80 overflow-y-auto py-2">
            <Show
              when={flatItems().length > 0}
              fallback={
                <p class="px-4 py-8 text-center text-sm text-slate-400">
                  No results found
                </p>
              }
            >
              <For each={grouped()}>
                {([section, items]) => (
                  <div>
                    <p class="px-4 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      {section}
                    </p>
                    <For each={items}>
                      {(item) => {
                        const idx = () => flatItems().indexOf(item);
                        return (
                          <button
                            onClick={() => select(item)}
                            onMouseEnter={() => setSelectedIndex(idx())}
                            class={`flex w-full items-center gap-3 px-4 py-2 text-sm transition-colors ${
                              selectedIndex() === idx()
                                ? "bg-indigo-50 text-indigo-700"
                                : "text-slate-700 hover:bg-slate-50"
                            }`}
                          >
                            {item.label}
                          </button>
                        );
                      }}
                    </For>
                  </div>
                )}
              </For>
            </Show>
          </div>

          <div class="border-t border-slate-100 px-4 py-2 flex items-center justify-between text-[10px] text-slate-400">
            <span>
              Navigate with <kbd class="font-mono">↑↓</kbd> · Select with{" "}
              <kbd class="font-mono">↵</kbd>
            </span>
            <span>
              <kbd class="font-mono">⌘K</kbd> to toggle
            </span>
          </div>
        </div>
      </div>
    </Show>
  );
}
