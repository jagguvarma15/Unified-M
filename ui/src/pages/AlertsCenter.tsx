import { createSignal, createEffect, Show, For } from "solid-js";
import PageHeader from "../components/PageHeader";
import MetricCard from "../components/MetricCard";
import {
  Bell,
  Plus,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  X,
  AlertCircle,
} from "../lib/icons";
import { useToast } from "../lib/toast";
import { setAlertCount } from "../lib/alertStore";

interface AlertRule {
  id: string;
  name: string;
  channel: string;
  metric: string;
  operator: ">" | "<" | ">=" | "<=";
  threshold: number;
  enabled: boolean;
  lastTriggered: string | null;
}

interface AlertEvent {
  id: string;
  ruleId: string;
  ruleName: string;
  channel: string;
  metric: string;
  value: number;
  threshold: number;
  severity: "warning" | "critical";
  timestamp: string;
  acknowledged: boolean;
}

const METRICS = [
  { value: "roas", label: "ROAS" },
  { value: "spend", label: "Spend" },
  { value: "cpa", label: "CPA" },
  { value: "mape", label: "MAPE" },
  { value: "contribution_pct", label: "Contribution %" },
  { value: "saturation", label: "Saturation Level" },
];

const CHANNELS = [
  "All Channels",
  "google_ads",
  "meta_ads",
  "tiktok_ads",
  "linkedin_ads",
  "pinterest_ads",
  "snapchat_ads",
  "twitter_ads",
  "apple_search_ads",
];

const DEMO_RULES: AlertRule[] = [
  {
    id: "r1",
    name: "Low ROAS Alert",
    channel: "All Channels",
    metric: "roas",
    operator: "<",
    threshold: 1.5,
    enabled: true,
    lastTriggered: "2026-04-03T14:22:00Z",
  },
  {
    id: "r2",
    name: "High CPA Warning",
    channel: "meta_ads",
    metric: "cpa",
    operator: ">",
    threshold: 50,
    enabled: true,
    lastTriggered: null,
  },
  {
    id: "r3",
    name: "Saturation Alert",
    channel: "google_ads",
    metric: "saturation",
    operator: ">",
    threshold: 0.85,
    enabled: false,
    lastTriggered: "2026-03-28T09:15:00Z",
  },
];

const DEMO_EVENTS: AlertEvent[] = [
  {
    id: "e1",
    ruleId: "r1",
    ruleName: "Low ROAS Alert",
    channel: "tiktok_ads",
    metric: "roas",
    value: 1.2,
    threshold: 1.5,
    severity: "critical",
    timestamp: "2026-04-03T14:22:00Z",
    acknowledged: false,
  },
  {
    id: "e2",
    ruleId: "r1",
    ruleName: "Low ROAS Alert",
    channel: "snapchat_ads",
    metric: "roas",
    value: 1.35,
    threshold: 1.5,
    severity: "warning",
    timestamp: "2026-04-02T10:45:00Z",
    acknowledged: true,
  },
  {
    id: "e3",
    ruleId: "r3",
    ruleName: "Saturation Alert",
    channel: "google_ads",
    metric: "saturation",
    value: 0.92,
    threshold: 0.85,
    severity: "warning",
    timestamp: "2026-03-28T09:15:00Z",
    acknowledged: true,
  },
];

export default function AlertsCenter() {
  const [rules, setRules] = createSignal<AlertRule[]>(DEMO_RULES);
  const [events, setEvents] = createSignal<AlertEvent[]>(DEMO_EVENTS);
  const [showAdd, setShowAdd] = createSignal(false);

  // Keep global alert badge in sync
  createEffect(() => {
    setAlertCount(events().filter((e) => !e.acknowledged).length);
  });
  const [tab, setTab] = createSignal<"events" | "rules">("events");
  const { addToast } = useToast();

  // Add rule form
  const [newName, setNewName] = createSignal("");
  const [newChannel, setNewChannel] = createSignal("All Channels");
  const [newMetric, setNewMetric] = createSignal("roas");
  const [newOperator, setNewOperator] =
    createSignal<AlertRule["operator"]>("<");
  const [newThreshold, setNewThreshold] = createSignal(0);

  const activeAlerts = () => events().filter((e) => !e.acknowledged).length;
  const totalRules = () => rules().length;
  const enabledRules = () => rules().filter((r) => r.enabled).length;

  const addRule = () => {
    if (!newName().trim()) return;
    const rule: AlertRule = {
      id: `r-${Date.now()}`,
      name: newName(),
      channel: newChannel(),
      metric: newMetric(),
      operator: newOperator(),
      threshold: newThreshold(),
      enabled: true,
      lastTriggered: null,
    };
    setRules((prev) => [...prev, rule]);
    setShowAdd(false);
    setNewName("");
    addToast("success", `Alert rule "${rule.name}" created`);
  };

  const deleteRule = (id: string) => {
    const rule = rules().find((r) => r.id === id);
    setRules((prev) => prev.filter((r) => r.id !== id));
    setEvents((prev) => prev.filter((e) => e.ruleId !== id));
    if (rule) {
      addToast("info", `Rule "${rule.name}" deleted`, {
        onUndo: () => {
          setRules((prev) => [...prev, rule]);
        },
      });
    }
  };

  const toggleRule = (id: string) => {
    setRules((prev) =>
      prev.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r)),
    );
  };

  const acknowledgeEvent = (id: string) => {
    setEvents((prev) =>
      prev.map((e) => (e.id === id ? { ...e, acknowledged: true } : e)),
    );
  };

  const acknowledgeAll = () => {
    setEvents((prev) => prev.map((e) => ({ ...e, acknowledged: true })));
    addToast("success", "All alerts acknowledged");
  };

  const severityColor = (severity: string) =>
    severity === "critical"
      ? "text-red-600 bg-red-50 border-red-200"
      : "text-amber-600 bg-amber-50 border-amber-200";

  const severityIcon = (severity: string) =>
    severity === "critical" ? AlertCircle : AlertTriangle;

  return (
    <div>
      <div class="flex items-center justify-between">
        <PageHeader
          title="Alerts Center"
          description="Configure threshold alerts per channel and metric"
        />
        <div class="flex items-center gap-2">
          <Show when={activeAlerts() > 0}>
            <button
              onClick={acknowledgeAll}
              class="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
            >
              <CheckCircle2 size={14} /> Acknowledge All
            </button>
          </Show>
          <button
            onClick={() => setShowAdd(true)}
            class="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
          >
            <Plus size={14} /> New Rule
          </button>
        </div>
      </div>

      {/* Summary KPIs */}
      <div class="grid grid-cols-3 gap-3 mb-6">
        <MetricCard
          label="Active Alerts"
          value={activeAlerts()}
          color={activeAlerts() > 0 ? "red" : "emerald"}
        />
        <MetricCard label="Total Rules" value={totalRules()} />
        <MetricCard
          label="Enabled Rules"
          value={enabledRules()}
          color="emerald"
        />
      </div>

      {/* Tabs */}
      <div class="flex gap-1 mb-4">
        <button
          onClick={() => setTab("events")}
          class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab() === "events"
              ? "bg-indigo-600 text-white"
              : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          Alert Events
          <Show when={activeAlerts() > 0}>
            <span class="ml-1.5 inline-flex items-center justify-center w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold">
              {activeAlerts()}
            </span>
          </Show>
        </button>
        <button
          onClick={() => setTab("rules")}
          class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab() === "rules"
              ? "bg-indigo-600 text-white"
              : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          Alert Rules ({totalRules()})
        </button>
      </div>

      {/* Events list */}
      <Show when={tab() === "events"}>
        <Show
          when={events().length > 0}
          fallback={
            <div class="rounded-xl border border-dashed border-slate-300 bg-white py-16 text-center">
              <Bell size={32} class="mx-auto text-slate-300" />
              <p class="mt-3 text-sm text-slate-500">No alert events yet</p>
              <p class="mt-1 text-xs text-slate-400">
                Events appear when configured thresholds are exceeded
              </p>
            </div>
          }
        >
          <div class="space-y-2">
            <For each={events()}>
              {(event) => {
                const SevIcon = severityIcon(event.severity);
                return (
                  <div
                    class={`rounded-lg border p-4 transition-all ${
                      event.acknowledged
                        ? "bg-white border-slate-200 opacity-60"
                        : `${severityColor(event.severity)} shadow-sm`
                    }`}
                  >
                    <div class="flex items-start gap-3">
                      <SevIcon
                        size={18}
                        class={
                          event.severity === "critical"
                            ? "text-red-500 mt-0.5"
                            : "text-amber-500 mt-0.5"
                        }
                      />
                      <div class="flex-1">
                        <div class="flex items-center gap-2">
                          <span class="font-semibold text-sm">
                            {event.ruleName}
                          </span>
                          <span
                            class={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
                              event.severity === "critical"
                                ? "bg-red-100 text-red-700"
                                : "bg-amber-100 text-amber-700"
                            }`}
                          >
                            {event.severity}
                          </span>
                        </div>
                        <p class="text-sm mt-0.5">
                          <span class="font-medium">{event.channel}</span>{" "}
                          {event.metric} ={" "}
                          <span class="font-mono tabular-nums font-semibold">
                            {event.value}
                          </span>{" "}
                          (threshold: {event.threshold})
                        </p>
                        <p class="text-xs text-slate-500 mt-1">
                          {new Date(event.timestamp).toLocaleString()}
                        </p>
                      </div>
                      <Show when={!event.acknowledged}>
                        <button
                          onClick={() => acknowledgeEvent(event.id)}
                          class="shrink-0 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
                        >
                          <CheckCircle2 size={12} /> Ack
                        </button>
                      </Show>
                    </div>
                  </div>
                );
              }}
            </For>
          </div>
        </Show>
      </Show>

      {/* Rules list */}
      <Show when={tab() === "rules"}>
        <div class="space-y-2">
          <For each={rules()}>
            {(rule) => (
              <div class="flex items-center gap-4 rounded-lg border border-slate-200 bg-white p-4">
                <button
                  onClick={() => toggleRule(rule.id)}
                  class={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors ${
                    rule.enabled ? "bg-indigo-600" : "bg-slate-300"
                  }`}
                >
                  <span
                    class={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform mt-0.5 ${
                      rule.enabled ? "translate-x-4 ml-0.5" : "translate-x-0.5"
                    }`}
                  />
                </button>
                <div class="flex-1">
                  <p class="text-sm font-semibold text-slate-900">
                    {rule.name}
                  </p>
                  <p class="text-xs text-slate-500">
                    {rule.channel} ·{" "}
                    {METRICS.find((m) => m.value === rule.metric)?.label ??
                      rule.metric}{" "}
                    {rule.operator} {rule.threshold}
                  </p>
                </div>
                <Show when={rule.lastTriggered}>
                  <span class="text-xs text-slate-400">
                    Last: {new Date(rule.lastTriggered!).toLocaleDateString()}
                  </span>
                </Show>
                <button
                  onClick={() => deleteRule(rule.id)}
                  class="text-slate-300 hover:text-red-500 transition-colors p-1"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Add Rule Modal */}
      <Show when={showAdd()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div
            class="absolute inset-0 bg-black/20"
            onClick={() => setShowAdd(false)}
          />
          <div class="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">
                New Alert Rule
              </h3>
              <button
                onClick={() => setShowAdd(false)}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Rule Name
              </label>
              <input
                type="text"
                value={newName()}
                onInput={(e) => setNewName(e.currentTarget.value)}
                placeholder="e.g. Low ROAS Alert"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Channel
                </label>
                <select
                  value={newChannel()}
                  onChange={(e) => setNewChannel(e.currentTarget.value)}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={CHANNELS}>
                    {(ch) => (
                      <option value={ch}>{ch.replace(/_/g, " ")}</option>
                    )}
                  </For>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Metric
                </label>
                <select
                  value={newMetric()}
                  onChange={(e) => setNewMetric(e.currentTarget.value)}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={METRICS}>
                    {(m) => <option value={m.value}>{m.label}</option>}
                  </For>
                </select>
              </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Condition
                </label>
                <select
                  value={newOperator()}
                  onChange={(e) =>
                    setNewOperator(
                      e.currentTarget.value as AlertRule["operator"],
                    )
                  }
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <option value="<">Less than</option>
                  <option value=">">Greater than</option>
                  <option value="<=">Less than or equal</option>
                  <option value=">=">Greater than or equal</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">
                  Threshold
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={newThreshold()}
                  onInput={(e) =>
                    setNewThreshold(Number(e.currentTarget.value))
                  }
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>

            <div class="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowAdd(false)}
                class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={addRule}
                disabled={!newName().trim()}
                class="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors"
              >
                Create Rule
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
