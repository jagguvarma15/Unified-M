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
  Clock,
  Mail,
  MessageSquare,
  History,
  BellOff,
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
  snoozedUntil?: string; // ISO timestamp
}

type AlertTab = "events" | "rules" | "digest";

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
  {
    id: "e4",
    ruleId: "r2",
    ruleName: "High CPA Warning",
    channel: "meta_ads",
    metric: "cpa",
    value: 67,
    threshold: 50,
    severity: "warning",
    timestamp: "2026-03-21T08:00:00Z",
    acknowledged: true,
  },
  {
    id: "e5",
    ruleId: "r1",
    ruleName: "Low ROAS Alert",
    channel: "pinterest_ads",
    metric: "roas",
    value: 1.1,
    threshold: 1.5,
    severity: "critical",
    timestamp: "2026-03-15T16:30:00Z",
    acknowledged: true,
  },
];

function isSnoozed(event: AlertEvent): boolean {
  if (!event.snoozedUntil) return false;
  return new Date(event.snoozedUntil) > new Date();
}

export default function AlertsCenter() {
  const [rules, setRules] = createSignal<AlertRule[]>(DEMO_RULES);
  const [events, setEvents] = createSignal<AlertEvent[]>(DEMO_EVENTS);
  const [showAdd, setShowAdd] = createSignal(false);
  const [tab, setTab] = createSignal<AlertTab>("events");
  const { addToast } = useToast();

  // Digest settings
  const [digestEmail, setDigestEmail] = createSignal(true);
  const [digestSlack, setDigestSlack] = createSignal(false);
  const [digestFrequency, setDigestFrequency] = createSignal<
    "daily" | "weekly"
  >("weekly");

  // Keep global alert badge in sync (active = unacknowledged and not snoozed)
  createEffect(() => {
    setAlertCount(
      events().filter((e) => !e.acknowledged && !isSnoozed(e)).length,
    );
  });

  // Add rule form
  const [newName, setNewName] = createSignal("");
  const [newChannel, setNewChannel] = createSignal("All Channels");
  const [newMetric, setNewMetric] = createSignal("roas");
  const [newOperator, setNewOperator] =
    createSignal<AlertRule["operator"]>("<");
  const [newThreshold, setNewThreshold] = createSignal(0);

  const activeAlerts = () =>
    events().filter((e) => !e.acknowledged && !isSnoozed(e)).length;
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

  const snoozeEvent = (id: string, hours: number) => {
    const until = new Date(Date.now() + hours * 60 * 60 * 1000).toISOString();
    setEvents((prev) =>
      prev.map((e) => (e.id === id ? { ...e, snoozedUntil: until } : e)),
    );
    addToast("info", `Alert snoozed for ${hours}h`);
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

  // Sort history descending by timestamp
  const historyEvents = () =>
    [...events()].sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );

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
        <button
          onClick={() => setTab("digest")}
          class={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab() === "digest"
              ? "bg-indigo-600 text-white"
              : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          <History size={13} /> Digest & History
        </button>
      </div>

      {/* ── Events tab ── */}
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
            <For each={events().filter((e) => !e.acknowledged)}>
              {(event) => {
                const SevIcon = severityIcon(event.severity);
                const snoozed = () => isSnoozed(event);
                return (
                  <div
                    class={`rounded-lg border p-4 transition-all ${
                      snoozed()
                        ? "bg-white border-slate-200 opacity-60"
                        : `${severityColor(event.severity)} shadow-sm`
                    }`}
                  >
                    <div class="flex items-start gap-3">
                      <Show
                        when={!snoozed()}
                        fallback={
                          <BellOff size={18} class="text-slate-400 mt-0.5" />
                        }
                      >
                        <SevIcon
                          size={18}
                          class={
                            event.severity === "critical"
                              ? "text-red-500 mt-0.5"
                              : "text-amber-500 mt-0.5"
                          }
                        />
                      </Show>
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
                          <Show when={snoozed()}>
                            <span class="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase bg-slate-100 text-slate-500">
                              Snoozed
                            </span>
                          </Show>
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
                        <Show when={event.snoozedUntil && snoozed()}>
                          <p class="text-xs text-slate-400 mt-0.5">
                            Snoozed until{" "}
                            {new Date(event.snoozedUntil!).toLocaleString()}
                          </p>
                        </Show>
                      </div>
                      <Show when={!snoozed()}>
                        <div class="flex items-center gap-1.5 shrink-0">
                          {/* Snooze menu */}
                          <div class="relative group">
                            <button class="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors">
                              <Clock size={11} /> Snooze
                            </button>
                            <div class="absolute right-0 top-full mt-1 hidden group-hover:flex flex-col z-10 bg-white border border-slate-200 rounded-lg shadow-lg overflow-hidden text-xs min-w-[100px]">
                              {[
                                { label: "1 hour", hours: 1 },
                                { label: "4 hours", hours: 4 },
                                { label: "24 hours", hours: 24 },
                                { label: "48 hours", hours: 48 },
                              ].map(({ label, hours }) => (
                                <button
                                  onClick={() => snoozeEvent(event.id, hours)}
                                  class="px-3 py-1.5 text-left text-slate-700 hover:bg-slate-50 transition-colors"
                                >
                                  {label}
                                </button>
                              ))}
                            </div>
                          </div>
                          <button
                            onClick={() => acknowledgeEvent(event.id)}
                            class="flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
                          >
                            <CheckCircle2 size={11} /> Ack
                          </button>
                        </div>
                      </Show>
                    </div>
                  </div>
                );
              }}
            </For>

            {/* Acknowledged events */}
            <Show when={events().filter((e) => e.acknowledged).length > 0}>
              <p class="text-xs text-slate-400 mt-4 mb-2 uppercase tracking-wider font-semibold">
                Acknowledged
              </p>
              <For each={events().filter((e) => e.acknowledged)}>
                {(event) => {
                  const SevIcon = severityIcon(event.severity);
                  return (
                    <div class="rounded-lg border border-slate-200 bg-white p-4 opacity-50">
                      <div class="flex items-start gap-3">
                        <SevIcon
                          size={16}
                          class="text-slate-400 mt-0.5 shrink-0"
                        />
                        <div class="flex-1">
                          <div class="flex items-center gap-2">
                            <span class="font-medium text-sm text-slate-700">
                              {event.ruleName}
                            </span>
                            <CheckCircle2 size={13} class="text-emerald-500" />
                          </div>
                          <p class="text-xs text-slate-500 mt-0.5">
                            {event.channel} · {event.metric} = {event.value} ·{" "}
                            {new Date(event.timestamp).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                }}
              </For>
            </Show>
          </div>
        </Show>
      </Show>

      {/* ── Rules tab ── */}
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

      {/* ── Digest & History tab ── */}
      <Show when={tab() === "digest"}>
        <div class="space-y-5">
          {/* Weekly Digest config */}
          <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <h2 class="text-base font-semibold text-slate-900 mb-1">
              Weekly Digest
            </h2>
            <p class="text-xs text-slate-500 mb-4">
              Automatically send a summary of alert activity and model health to
              your team.
            </p>

            <div class="space-y-3">
              {/* Frequency */}
              <div class="flex items-center justify-between">
                <span class="text-sm text-slate-700">Digest Frequency</span>
                <div class="flex rounded-lg overflow-hidden border border-slate-200">
                  {(["daily", "weekly"] as const).map((f) => (
                    <button
                      onClick={() => setDigestFrequency(f)}
                      class={`px-3 py-1.5 text-xs font-medium capitalize transition-colors ${
                        digestFrequency() === f
                          ? "bg-indigo-600 text-white"
                          : "text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              {/* Email toggle */}
              <div class="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2.5">
                <div class="flex items-center gap-2">
                  <Mail size={14} class="text-slate-500" />
                  <span class="text-sm text-slate-700">Email digest</span>
                </div>
                <button
                  onClick={() => setDigestEmail((v) => !v)}
                  class={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors ${
                    digestEmail() ? "bg-indigo-600" : "bg-slate-300"
                  }`}
                >
                  <span
                    class={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform mt-0.5 ${
                      digestEmail() ? "translate-x-4 ml-0.5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>

              {/* Slack toggle */}
              <div class="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2.5">
                <div class="flex items-center gap-2">
                  <MessageSquare size={14} class="text-slate-500" />
                  <span class="text-sm text-slate-700">Slack push</span>
                  <span class="text-[10px] text-slate-400 rounded-full bg-slate-200 px-1.5 py-0.5">
                    Connect Slack in Settings
                  </span>
                </div>
                <button
                  onClick={() => setDigestSlack((v) => !v)}
                  class={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors ${
                    digestSlack() ? "bg-indigo-600" : "bg-slate-300"
                  }`}
                >
                  <span
                    class={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform mt-0.5 ${
                      digestSlack() ? "translate-x-4 ml-0.5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>

              <Show when={digestEmail() || digestSlack()}>
                <div class="rounded-lg border border-indigo-100 bg-indigo-50 px-3 py-2 text-xs text-indigo-700">
                  {digestFrequency() === "weekly"
                    ? "Every Monday morning"
                    : "Every morning at 8:00 AM"}{" "}
                  you'll receive a digest of all alert activity, model health,
                  and spend anomalies.
                </div>
              </Show>
            </div>
          </div>

          {/* Alert History Timeline */}
          <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-base font-semibold text-slate-900">
                Alert History
              </h2>
              <span class="text-xs text-slate-400">
                {events().length} total events
              </span>
            </div>

            <div class="relative">
              {/* Timeline line */}
              <div class="absolute left-[7px] top-0 bottom-0 w-px bg-slate-200" />

              <div class="space-y-3">
                <For each={historyEvents()}>
                  {(event) => {
                    const SevIcon = severityIcon(event.severity);
                    return (
                      <div class="flex gap-4 pl-7 relative">
                        {/* Timeline dot */}
                        <div
                          class={`absolute left-0 top-1 h-3.5 w-3.5 rounded-full border-2 border-white ring-1 ${
                            event.severity === "critical"
                              ? "bg-red-500 ring-red-300"
                              : "bg-amber-400 ring-amber-200"
                          } ${event.acknowledged ? "opacity-40" : ""}`}
                        />
                        <div
                          class={`flex-1 rounded-lg border px-3 py-2.5 ${
                            event.acknowledged
                              ? "border-slate-100 bg-slate-50"
                              : event.severity === "critical"
                                ? "border-red-200 bg-red-50"
                                : "border-amber-200 bg-amber-50"
                          }`}
                        >
                          <div class="flex items-start justify-between gap-2">
                            <div>
                              <span class="text-xs font-semibold text-slate-800">
                                {event.ruleName}
                              </span>
                              <span
                                class={`ml-2 rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase ${
                                  event.severity === "critical"
                                    ? "bg-red-100 text-red-700"
                                    : "bg-amber-100 text-amber-700"
                                }`}
                              >
                                {event.severity}
                              </span>
                              <Show when={event.acknowledged}>
                                <span class="ml-1.5 rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase bg-emerald-100 text-emerald-700">
                                  acked
                                </span>
                              </Show>
                            </div>
                            <span class="text-[10px] text-slate-400 shrink-0 tabular-nums">
                              {new Date(event.timestamp).toLocaleDateString(
                                undefined,
                                {
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                },
                              )}
                            </span>
                          </div>
                          <p class="text-xs text-slate-500 mt-0.5">
                            {event.channel.replace(/_/g, " ")} · {event.metric}{" "}
                            = {event.value} (threshold: {event.threshold})
                          </p>
                        </div>
                      </div>
                    );
                  }}
                </For>
              </div>
            </div>
          </div>
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
