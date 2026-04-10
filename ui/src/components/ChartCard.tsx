import type { JSX } from "solid-js";
import { createSignal, Show } from "solid-js";
import { density, CHART_PAD, CHART_HEADER_MB } from "../lib/density";
import { Download, ImageIcon } from "../lib/icons";

interface Props {
  title: string;
  description?: string;
  actionHref?: string;
  actionLabel?: string;
  rightSlot?: JSX.Element;
  children: JSX.Element;
  minHeight?: number;
  class?: string;
  /** Pass chart data for CSV export — array of objects */
  exportData?: Record<string, unknown>[];
  /** Filename prefix for exports */
  exportName?: string;
}

function downloadCsv(data: Record<string, unknown>[], filename: string) {
  if (!data.length) return;
  const keys = Object.keys(data[0]);
  const rows = [keys.join(",")];
  for (const row of data) {
    rows.push(
      keys
        .map((k) => {
          const v = row[k];
          const s = v == null ? "" : String(v);
          return s.includes(",") || s.includes('"')
            ? `"${s.replace(/"/g, '""')}"`
            : s;
        })
        .join(","),
    );
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filename}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadPng(container: HTMLElement, filename: string) {
  const svg = container.querySelector("svg");
  if (!svg) return;
  const svgData = new XMLSerializer().serializeToString(svg);
  const canvas = document.createElement("canvas");
  const rect = svg.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename}.png`;
    a.click();
  };
  img.src =
    "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
}

export default function ChartCard(props: Props) {
  let containerRef!: HTMLDivElement;
  const [showExportMenu, setShowExportMenu] = createSignal(false);
  const exportName = () =>
    props.exportName ?? props.title.toLowerCase().replace(/\s+/g, "-");

  return (
    <div
      ref={containerRef}
      class={`rounded-xl border border-slate-200/60 bg-white shadow-sm transition-shadow hover:shadow-md ${CHART_PAD[density()]} ${props.class ?? ""}`}
      style={
        props.minHeight ? { "min-height": `${props.minHeight}px` } : undefined
      }
    >
      <div
        class={`flex flex-wrap items-start justify-between gap-3 ${CHART_HEADER_MB[density()]}`}
      >
        <div>
          <h2 class="text-sm font-medium tracking-tight text-slate-700">
            {props.title}
          </h2>
          <Show when={props.description}>
            <p class="mt-0.5 text-xs text-slate-500">{props.description}</p>
          </Show>
        </div>
        <div class="flex items-center gap-2">
          {props.rightSlot}

          {/* Export menu */}
          <div class="relative">
            <button
              onClick={() => setShowExportMenu(!showExportMenu())}
              class="flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-500 hover:bg-slate-50 hover:text-slate-700 transition-colors"
              title="Export chart"
            >
              <Download size={12} />
            </button>
            <Show when={showExportMenu()}>
              <div class="absolute right-0 top-full mt-1 z-30 w-36 rounded-lg border border-slate-200 bg-white py-1 shadow-lg">
                <Show when={props.exportData}>
                  <button
                    onClick={() => {
                      downloadCsv(props.exportData!, exportName());
                      setShowExportMenu(false);
                    }}
                    class="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
                  >
                    <Download size={12} /> Export CSV
                  </button>
                </Show>
                <button
                  onClick={() => {
                    downloadPng(containerRef, exportName());
                    setShowExportMenu(false);
                  }}
                  class="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50"
                >
                  <ImageIcon size={12} /> Export PNG
                </button>
              </div>
            </Show>
          </div>

          <Show when={props.actionHref}>
            <a
              href={props.actionHref}
              class="text-xs font-medium text-indigo-600 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 rounded"
            >
              {props.actionLabel ?? "View full →"}
            </a>
          </Show>
        </div>
      </div>
      {props.children}
    </div>
  );
}
