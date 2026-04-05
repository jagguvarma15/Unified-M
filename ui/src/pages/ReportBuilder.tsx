import { createSignal, Show, For } from "solid-js";
import PageHeader from "../components/PageHeader";
import { GripVertical, Plus, Trash2, Download, FileText } from "../lib/icons";

interface ReportBlock {
  id: string;
  type: "text" | "chart" | "metric" | "table";
  title: string;
  content: string;
}

const BLOCK_TEMPLATES: { type: ReportBlock["type"]; label: string; defaultTitle: string; defaultContent: string }[] = [
  { type: "text", label: "Text Block", defaultTitle: "Section Header", defaultContent: "Add your narrative here..." },
  { type: "chart", label: "Chart", defaultTitle: "Chart Widget", defaultContent: "contributions" },
  { type: "metric", label: "KPI Card", defaultTitle: "Key Metric", defaultContent: "r_squared" },
  { type: "table", label: "Data Table", defaultTitle: "Data Table", defaultContent: "channel_summary" },
];

const CHART_OPTIONS = [
  { value: "contributions", label: "Contribution Share" },
  { value: "waterfall", label: "Waterfall Decomposition" },
  { value: "roas", label: "ROAS by Channel" },
  { value: "allocation", label: "Budget Allocation" },
  { value: "model_fit", label: "Model Fit" },
  { value: "residuals", label: "Prediction Residuals" },
  { value: "timeline", label: "Contributions Over Time" },
];

const METRIC_OPTIONS = [
  { value: "r_squared", label: "R²" },
  { value: "mape", label: "MAPE" },
  { value: "total_spend", label: "Total Spend" },
  { value: "blended_roas", label: "Blended ROAS" },
  { value: "optim_uplift", label: "Optimization Uplift" },
];

export default function ReportBuilder() {
  const [blocks, setBlocks] = createSignal<ReportBlock[]>([
    { id: "1", type: "text", title: "Executive Summary", content: "This report summarizes the key findings from the latest MMM run." },
    { id: "2", type: "metric", title: "Model Performance", content: "r_squared" },
    { id: "3", type: "chart", title: "Contribution Breakdown", content: "contributions" },
    { id: "4", type: "table", title: "Channel Performance", content: "channel_summary" },
  ]);
  const [selectedBlock, setSelectedBlock] = createSignal<string | null>(null);
  const [reportTitle, setReportTitle] = createSignal("Marketing Performance Report");
  const [dragIdx, setDragIdx] = createSignal<number | null>(null);

  const addBlock = (type: ReportBlock["type"]) => {
    const tmpl = BLOCK_TEMPLATES.find((t) => t.type === type)!;
    const newBlock: ReportBlock = {
      id: `block-${Date.now()}`,
      type,
      title: tmpl.defaultTitle,
      content: tmpl.defaultContent,
    };
    setBlocks((prev) => [...prev, newBlock]);
    setSelectedBlock(newBlock.id);
  };

  const removeBlock = (id: string) => {
    setBlocks((prev) => prev.filter((b) => b.id !== id));
    if (selectedBlock() === id) setSelectedBlock(null);
  };

  const updateBlock = (id: string, field: "title" | "content", value: string) => {
    setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, [field]: value } : b)));
  };

  const moveBlock = (fromIdx: number, toIdx: number) => {
    setBlocks((prev) => {
      const arr = [...prev];
      const [item] = arr.splice(fromIdx, 1);
      arr.splice(toIdx, 0, item);
      return arr;
    });
  };

  const exportAsText = () => {
    const lines = [`# ${reportTitle()}\n`];
    for (const b of blocks()) {
      lines.push(`## ${b.title}`);
      if (b.type === "text") lines.push(b.content);
      else if (b.type === "chart") lines.push(`[Chart: ${CHART_OPTIONS.find((c) => c.value === b.content)?.label ?? b.content}]`);
      else if (b.type === "metric") lines.push(`[Metric: ${METRIC_OPTIONS.find((m) => m.value === b.content)?.label ?? b.content}]`);
      else lines.push(`[Table: ${b.content}]`);
      lines.push("");
    }
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${reportTitle().replace(/\s+/g, "-").toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const blockTypeLabel = (type: ReportBlock["type"]) => {
    switch (type) {
      case "text": return "Text";
      case "chart": return "Chart";
      case "metric": return "KPI";
      case "table": return "Table";
    }
  };

  const blockTypeColor = (type: ReportBlock["type"]) => {
    switch (type) {
      case "text": return "bg-slate-100 text-slate-600";
      case "chart": return "bg-indigo-50 text-indigo-600";
      case "metric": return "bg-emerald-50 text-emerald-600";
      case "table": return "bg-amber-50 text-amber-600";
    }
  };

  return (
    <div>
      <div class="flex items-center justify-between">
        <PageHeader
          title="Report Builder"
          description="Drag-and-drop report canvas — build custom reports"
        />
        <div class="flex items-center gap-2">
          <button
            onClick={exportAsText}
            class="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 shadow-sm transition-colors"
          >
            <Download size={14} /> Export
          </button>
        </div>
      </div>

      {/* Report title */}
      <div class="mb-6">
        <input
          type="text"
          value={reportTitle()}
          onInput={(e) => setReportTitle(e.currentTarget.value)}
          class="w-full text-2xl font-bold text-slate-900 bg-transparent border-b-2 border-transparent focus:border-indigo-500 outline-none py-1 transition-colors"
          placeholder="Report Title"
        />
      </div>

      <div class="flex gap-6">
        {/* Canvas */}
        <div class="flex-1 space-y-3">
          <For each={blocks()}>
            {(block, i) => (
              <div
                class={`group relative rounded-xl border bg-white p-4 transition-all cursor-pointer ${
                  selectedBlock() === block.id
                    ? "border-indigo-400 ring-2 ring-indigo-100 shadow-md"
                    : "border-slate-200 hover:border-slate-300 hover:shadow-sm"
                }`}
                onClick={() => setSelectedBlock(block.id)}
                draggable
                onDragStart={() => setDragIdx(i())}
                onDragOver={(e) => e.preventDefault()}
                onDrop={() => {
                  if (dragIdx() != null && dragIdx() !== i()) moveBlock(dragIdx()!, i());
                  setDragIdx(null);
                }}
              >
                <div class="flex items-start gap-3">
                  <div class="mt-1 cursor-grab text-slate-300 hover:text-slate-500 transition-colors" draggable>
                    <GripVertical size={16} />
                  </div>
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                      <span class={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${blockTypeColor(block.type)}`}>
                        {blockTypeLabel(block.type)}
                      </span>
                      <span class="text-sm font-semibold text-slate-700">{block.title}</span>
                    </div>

                    {block.type === "text" && (
                      <p class="text-sm text-slate-500 line-clamp-2">{block.content}</p>
                    )}
                    {block.type === "chart" && (
                      <div class="h-32 rounded-lg bg-gradient-to-br from-indigo-50 to-slate-50 flex items-center justify-center">
                        <p class="text-xs text-indigo-400 font-medium">
                          {CHART_OPTIONS.find((c) => c.value === block.content)?.label ?? "Chart"}
                        </p>
                      </div>
                    )}
                    {block.type === "metric" && (
                      <div class="inline-flex items-center gap-2 rounded-lg bg-slate-50 border border-slate-200 px-3 py-2">
                        <span class="text-xs text-slate-500">{METRIC_OPTIONS.find((m) => m.value === block.content)?.label ?? "Metric"}</span>
                        <span class="text-lg font-bold text-slate-900">—</span>
                      </div>
                    )}
                    {block.type === "table" && (
                      <div class="h-16 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-center">
                        <p class="text-xs text-slate-400">Data Table: {block.content}</p>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeBlock(block.id); }}
                    class="opacity-0 group-hover:opacity-100 text-slate-300 hover:text-red-500 transition-all p-1"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            )}
          </For>

          {/* Add block buttons */}
          <div class="flex gap-2 pt-2">
            <For each={BLOCK_TEMPLATES}>
              {(tmpl) => (
                <button
                  onClick={() => addBlock(tmpl.type)}
                  class="flex items-center gap-1.5 rounded-lg border-2 border-dashed border-slate-200 px-3 py-2 text-xs font-medium text-slate-500 hover:border-indigo-300 hover:text-indigo-600 transition-colors"
                >
                  <Plus size={12} /> {tmpl.label}
                </button>
              )}
            </For>
          </div>
        </div>

        {/* Properties panel */}
        <div class="w-72 shrink-0">
          <div class="sticky top-20 rounded-xl border border-slate-200 bg-white shadow-sm p-4 space-y-4">
            <h3 class="text-sm font-semibold text-slate-700">Properties</h3>
            <Show
              when={selectedBlock() && blocks().find((b) => b.id === selectedBlock())}
              fallback={<p class="text-xs text-slate-400 py-8 text-center">Select a block to edit</p>}
            >
              {(() => {
                const block = () => blocks().find((b) => b.id === selectedBlock())!;
                return (
                  <div class="space-y-3">
                    <div>
                      <label class="block text-xs font-medium text-slate-600 mb-1">Title</label>
                      <input
                        type="text"
                        value={block().title}
                        onInput={(e) => updateBlock(block().id, "title", e.currentTarget.value)}
                        class="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                      />
                    </div>

                    <Show when={block().type === "text"}>
                      <div>
                        <label class="block text-xs font-medium text-slate-600 mb-1">Content</label>
                        <textarea
                          value={block().content}
                          onInput={(e) => updateBlock(block().id, "content", e.currentTarget.value)}
                          rows={4}
                          class="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                        />
                      </div>
                    </Show>

                    <Show when={block().type === "chart"}>
                      <div>
                        <label class="block text-xs font-medium text-slate-600 mb-1">Chart Type</label>
                        <select
                          value={block().content}
                          onChange={(e) => updateBlock(block().id, "content", e.currentTarget.value)}
                          class="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                        >
                          <For each={CHART_OPTIONS}>
                            {(opt) => <option value={opt.value}>{opt.label}</option>}
                          </For>
                        </select>
                      </div>
                    </Show>

                    <Show when={block().type === "metric"}>
                      <div>
                        <label class="block text-xs font-medium text-slate-600 mb-1">Metric</label>
                        <select
                          value={block().content}
                          onChange={(e) => updateBlock(block().id, "content", e.currentTarget.value)}
                          class="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                        >
                          <For each={METRIC_OPTIONS}>
                            {(opt) => <option value={opt.value}>{opt.label}</option>}
                          </For>
                        </select>
                      </div>
                    </Show>
                  </div>
                );
              })()}
            </Show>
          </div>
        </div>
      </div>
    </div>
  );
}
