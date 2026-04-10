import { createSignal, onMount, createEffect, Show, For } from "solid-js";
import {
  User,
  Mail,
  Globe,
  Key,
  Bell,
  Building2,
  Save,
  Camera,
  Lock,
  Server,
  Cpu,
  HardDrive,
  CheckCircle2,
  XCircle,
  Blocks,
  Database,
  Cloud,
  Megaphone,
  RefreshCw,
  Download,
} from "../lib/icons";
import EmptyState from "../components/EmptyState";
import { api, type HealthData, type AdaptersData } from "../lib/api";
import { useHealthQuery } from "../lib/queries";
import { useToast } from "../lib/toast";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabKey = "profile" | "workspace" | "notifications" | "adapters" | "system";

const TABS: { key: TabKey; label: string }[] = [
  { key: "profile", label: "Profile" },
  { key: "workspace", label: "Workspace" },
  { key: "notifications", label: "Notifications" },
  { key: "adapters", label: "Adapters" },
  { key: "system", label: "System" },
];

// ---------------------------------------------------------------------------
// Main Settings page
// ---------------------------------------------------------------------------

export default function Settings() {
  const [adapters, setAdapters] = createSignal<AdaptersData | null>(null);
  const [health, setHealth] = createSignal<HealthData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [activeTab, setActiveTab] = createSignal<TabKey>("profile");
  const healthQuery = useHealthQuery();

  onMount(() => {
    Promise.allSettled([
      api.health().then(setHealth),
      api.adapters().then(setAdapters),
    ]).finally(() => setLoading(false));
  });

  createEffect(() => {
    if (healthQuery.isError) {
      setAdapters(null);
      setHealth(null);
    }
  });

  return (
    <Show
      when={!loading()}
      fallback={
        <div class="flex items-center justify-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <div>
        {/* Page header */}
        <div class="mb-6">
          <h1 class="text-2xl font-bold text-slate-900">Settings</h1>
          <p class="text-sm text-slate-500 mt-1">
            Manage your profile, workspace, and platform configuration
          </p>
        </div>

        {/* Tab bar */}
        <div class="flex gap-1 bg-slate-100 rounded-xl p-1 w-fit mb-8">
          <For each={TABS}>
            {(tab) => (
              <button
                onClick={() => setActiveTab(tab.key)}
                class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab() === tab.key
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            )}
          </For>
        </div>

        {/* Tab content */}
        <Show when={activeTab() === "profile"}>
          <ProfileTab />
        </Show>
        <Show when={activeTab() === "workspace"}>
          <WorkspaceTab />
        </Show>
        <Show when={activeTab() === "notifications"}>
          <NotificationsTab />
        </Show>
        <Show when={activeTab() === "adapters"}>
          <AdaptersTab adapters={adapters()} />
        </Show>
        <Show when={activeTab() === "system"}>
          <SystemTab health={health()} />
        </Show>
      </div>
    </Show>
  );
}

// ---------------------------------------------------------------------------
// Profile Tab
// ---------------------------------------------------------------------------

function ProfileTab() {
  const { addToast } = useToast();
  const [name, setName] = createSignal("Jane Doe");
  const [email, setEmail] = createSignal("jane@company.com");
  const [role, setRole] = createSignal("Marketing Analyst");
  const [timezone, setTimezone] = createSignal("America/New_York");
  const [currentPw, setCurrentPw] = createSignal("");
  const [newPw, setNewPw] = createSignal("");
  const [confirmPw, setConfirmPw] = createSignal("");

  const initials = () =>
    name()
      .split(" ")
      .map((w) => w[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);

  const saveProfile = () => {
    addToast("success", "Profile updated successfully");
  };

  const changePassword = () => {
    if (!currentPw() || !newPw()) {
      addToast("error", "Please fill in all password fields");
      return;
    }
    if (newPw() !== confirmPw()) {
      addToast("error", "New passwords do not match");
      return;
    }
    if (newPw().length < 8) {
      addToast("error", "Password must be at least 8 characters");
      return;
    }
    setCurrentPw("");
    setNewPw("");
    setConfirmPw("");
    addToast("success", "Password changed successfully");
  };

  const TIMEZONES = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Australia/Sydney",
    "Pacific/Auckland",
  ];

  return (
    <div class="space-y-6 max-w-2xl">
      {/* Avatar + identity */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {User({ size: 15, class: "text-slate-400" })}
          Personal Information
        </h2>

        {/* Avatar row */}
        <div class="flex items-center gap-5 mb-6">
          <div class="relative">
            <div class="h-16 w-16 rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center">
              <span class="text-xl font-bold text-white">{initials()}</span>
            </div>
            <button
              class="absolute -bottom-1 -right-1 h-6 w-6 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center hover:bg-slate-50 transition-colors"
              title="Change avatar"
              aria-label="Change avatar"
            >
              {Camera({ size: 12, class: "text-slate-500" })}
            </button>
          </div>
          <div>
            <p class="text-sm font-semibold text-slate-900">{name()}</p>
            <p class="text-xs text-slate-500 mt-0.5">{email()}</p>
            <span class="mt-1.5 inline-block px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 text-[11px] font-medium">
              {role()}
            </span>
          </div>
        </div>

        {/* Fields */}
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Full Name
            </label>
            <div class="relative">
              <span class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
                {User({ size: 14 })}
              </span>
              <input
                type="text"
                value={name()}
                onInput={(e) => setName(e.currentTarget.value)}
                class="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Email Address
            </label>
            <div class="relative">
              <span class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
                {Mail({ size: 14 })}
              </span>
              <input
                type="email"
                value={email()}
                onInput={(e) => setEmail(e.currentTarget.value)}
                class="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Job Title / Role
            </label>
            <input
              type="text"
              value={role()}
              onInput={(e) => setRole(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Timezone
            </label>
            <div class="relative">
              <span class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
                {Globe({ size: 14 })}
              </span>
              <select
                value={timezone()}
                onChange={(e) => setTimezone(e.currentTarget.value)}
                class="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent appearance-none bg-white"
              >
                <For each={TIMEZONES}>
                  {(tz) => <option value={tz}>{tz}</option>}
                </For>
              </select>
            </div>
          </div>
        </div>

        <div class="mt-5 flex justify-end">
          <button
            onClick={saveProfile}
            class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            {Save({ size: 14 })}
            Save Profile
          </button>
        </div>
      </div>

      {/* Change password */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {Lock({ size: 15, class: "text-slate-400" })}
          Change Password
        </h2>
        <div class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Current Password
            </label>
            <input
              type="password"
              value={currentPw()}
              onInput={(e) => setCurrentPw(e.currentTarget.value)}
              placeholder="••••••••"
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">
                New Password
              </label>
              <input
                type="password"
                value={newPw()}
                onInput={(e) => setNewPw(e.currentTarget.value)}
                placeholder="Min. 8 characters"
                class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1.5">
                Confirm New Password
              </label>
              <input
                type="password"
                value={confirmPw()}
                onInput={(e) => setConfirmPw(e.currentTarget.value)}
                placeholder="Re-enter new password"
                class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>
        </div>
        <div class="mt-5 flex justify-end">
          <button
            onClick={changePassword}
            class="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white rounded-lg text-sm font-medium hover:bg-slate-900 transition-colors"
          >
            {Key({ size: 14 })}
            Update Password
          </button>
        </div>
      </div>

      {/* API token */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-1 flex items-center gap-2">
          {Key({ size: 15, class: "text-slate-400" })}
          Personal API Token
        </h2>
        <p class="text-xs text-slate-500 mb-5">
          Use this token to authenticate API requests. Keep it secret.
        </p>
        <div class="flex items-center gap-3">
          <code class="flex-1 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs font-mono text-slate-600 truncate select-all">
            umm_sk_••••••••••••••••••••••••••••••••
          </code>
          <button
            onClick={() => navigator.clipboard?.writeText("umm_sk_demo_token")}
            class="px-3 py-2 bg-slate-100 border border-slate-200 rounded-lg text-xs font-medium text-slate-700 hover:bg-slate-200 transition-colors whitespace-nowrap"
          >
            Copy
          </button>
          <button class="px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-xs font-medium text-red-700 hover:bg-red-100 transition-colors whitespace-nowrap">
            Regenerate
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Workspace Tab
// ---------------------------------------------------------------------------

function WorkspaceTab() {
  const { addToast } = useToast();
  const [orgName, setOrgName] = createSignal("Acme Corp");
  const [brand, setBrand] = createSignal("Default Workspace");
  const [currency, setCurrency] = createSignal("USD");
  const [kpiTarget, setKpiTarget] = createSignal("revenue");
  const [fiscalStart, setFiscalStart] = createSignal("january");
  const [reportingTz, setReportingTz] = createSignal("America/New_York");

  const CURRENCIES = ["USD", "EUR", "GBP", "AUD", "CAD", "SGD", "INR", "JPY"];
  const KPI_TARGETS = [
    { value: "revenue", label: "Revenue" },
    { value: "conversions", label: "Conversions" },
    { value: "leads", label: "Leads" },
    { value: "orders", label: "Orders" },
    { value: "installs", label: "App Installs" },
  ];
  const MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
  ];

  const save = () => addToast("success", "Workspace settings saved");

  return (
    <div class="space-y-6 max-w-2xl">
      {/* Organization */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {Building2({ size: 15, class: "text-slate-400" })}
          Organization
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Organization Name
            </label>
            <input
              type="text"
              value={orgName()}
              onInput={(e) => setOrgName(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Workspace / Brand
            </label>
            <input
              type="text"
              value={brand()}
              onInput={(e) => setBrand(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Reporting defaults */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {Globe({ size: 15, class: "text-slate-400" })}
          Reporting Defaults
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Default Currency
            </label>
            <select
              value={currency()}
              onChange={(e) => setCurrency(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
            >
              <For each={CURRENCIES}>
                {(c) => <option value={c}>{c}</option>}
              </For>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Default KPI Target
            </label>
            <select
              value={kpiTarget()}
              onChange={(e) => setKpiTarget(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
            >
              <For each={KPI_TARGETS}>
                {(k) => <option value={k.value}>{k.label}</option>}
              </For>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Fiscal Year Start
            </label>
            <select
              value={fiscalStart()}
              onChange={(e) => setFiscalStart(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white capitalize"
            >
              <For each={MONTHS}>
                {(m) => (
                  <option value={m} class="capitalize">
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </option>
                )}
              </For>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-600 mb-1.5">
              Reporting Timezone
            </label>
            <select
              value={reportingTz()}
              onChange={(e) => setReportingTz(e.currentTarget.value)}
              class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
            >
              {[
                "America/New_York",
                "America/Los_Angeles",
                "Europe/London",
                "Europe/Paris",
                "Asia/Singapore",
                "Asia/Tokyo",
                "Australia/Sydney",
              ].map((tz) => (
                <option value={tz}>{tz}</option>
              ))}
            </select>
          </div>
        </div>

        <div class="mt-5 flex justify-end">
          <button
            onClick={save}
            class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            {Save({ size: 14 })}
            Save Workspace
          </button>
        </div>
      </div>

      {/* Danger zone */}
      <div class="bg-white rounded-xl border border-red-200 shadow-sm p-6">
        <h2 class="text-sm font-semibold text-red-700 mb-1">Danger Zone</h2>
        <p class="text-xs text-slate-500 mb-4">
          These actions are irreversible. Proceed with caution.
        </p>
        <div class="flex flex-wrap gap-3">
          <button class="px-4 py-2 rounded-lg border border-red-200 text-sm font-medium text-red-600 hover:bg-red-50 transition-colors">
            Reset All Runs
          </button>
          <button class="px-4 py-2 rounded-lg border border-red-200 text-sm font-medium text-red-600 hover:bg-red-50 transition-colors">
            Delete Workspace
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Notifications Tab
// ---------------------------------------------------------------------------

function NotificationsTab() {
  const { addToast } = useToast();
  const [emailDigest, setEmailDigest] = createSignal(true);
  const [digestFreq, setDigestFreq] = createSignal("weekly");
  const [alertEmail, setAlertEmail] = createSignal("jane@company.com");
  const [slackWebhook, setSlackWebhook] = createSignal("");
  const [notifyOnComplete, setNotifyOnComplete] = createSignal(true);
  const [notifyOnFail, setNotifyOnFail] = createSignal(true);
  const [notifyOnAlert, setNotifyOnAlert] = createSignal(true);
  const [notifyOnDrift, setNotifyOnDrift] = createSignal(false);

  const save = () => addToast("success", "Notification preferences saved");

  return (
    <div class="space-y-6 max-w-2xl">
      {/* Email digest */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {Mail({ size: 15, class: "text-slate-400" })}
          Email Notifications
        </h2>

        <div class="space-y-4">
          <div class="flex items-center justify-between py-2">
            <div>
              <p class="text-sm font-medium text-slate-800">Weekly Digest</p>
              <p class="text-xs text-slate-500 mt-0.5">
                Receive a summary of model performance and recommendations
              </p>
            </div>
            <Toggle value={emailDigest()} onChange={setEmailDigest} />
          </div>

          <Show when={emailDigest()}>
            <div class="ml-0 pl-4 border-l-2 border-indigo-100 space-y-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1.5">
                  Delivery Frequency
                </label>
                <select
                  value={digestFreq()}
                  onChange={(e) => setDigestFreq(e.currentTarget.value)}
                  class="w-full sm:w-48 px-3 py-2 rounded-lg border border-slate-200 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="biweekly">Bi-weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1.5">
                  Recipient Email
                </label>
                <input
                  type="email"
                  value={alertEmail()}
                  onInput={(e) => setAlertEmail(e.currentTarget.value)}
                  class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
              </div>
            </div>
          </Show>
        </div>
      </div>

      {/* Event notifications */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-5 flex items-center gap-2">
          {Bell({ size: 15, class: "text-slate-400" })}
          Event Triggers
        </h2>
        <div class="divide-y divide-slate-100">
          {[
            {
              label: "Pipeline completed",
              desc: "Notify when a model run finishes successfully",
              value: notifyOnComplete,
              set: setNotifyOnComplete,
            },
            {
              label: "Pipeline failed",
              desc: "Notify when a run errors or is aborted",
              value: notifyOnFail,
              set: setNotifyOnFail,
            },
            {
              label: "Alert triggered",
              desc: "Notify when a channel metric breaches a threshold",
              value: notifyOnAlert,
              set: setNotifyOnAlert,
            },
            {
              label: "Model drift detected",
              desc: "Notify when parameter stability drops below threshold",
              value: notifyOnDrift,
              set: setNotifyOnDrift,
            },
          ].map((item) => (
            <div class="flex items-center justify-between py-3">
              <div>
                <p class="text-sm font-medium text-slate-800">{item.label}</p>
                <p class="text-xs text-slate-500 mt-0.5">{item.desc}</p>
              </div>
              <Toggle value={item.value()} onChange={item.set} />
            </div>
          ))}
        </div>
      </div>

      {/* Slack */}
      <div class="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
        <h2 class="text-sm font-medium text-slate-700 mb-1">
          Slack Integration
        </h2>
        <p class="text-xs text-slate-500 mb-5">
          Post alerts and digests to a Slack channel via an Incoming Webhook
          URL.
        </p>
        <div>
          <label class="block text-xs font-medium text-slate-600 mb-1.5">
            Webhook URL
          </label>
          <input
            type="url"
            value={slackWebhook()}
            onInput={(e) => setSlackWebhook(e.currentTarget.value)}
            placeholder="https://hooks.slack.com/services/…"
            class="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <p class="mt-1.5 text-[11px] text-slate-400">
            Leave blank to disable Slack notifications.
          </p>
        </div>
        <div class="mt-5 flex justify-end">
          <button
            onClick={save}
            class="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            {Save({ size: 14 })}
            Save Preferences
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Adapters Tab  (existing, unchanged in structure)
// ---------------------------------------------------------------------------

function AdaptersTab({ adapters }: { adapters: AdaptersData | null }) {
  const [refreshing, setRefreshing] = createSignal(false);

  const handleClearCache = async () => {
    setRefreshing(true);
    try {
      await fetch("/api/cache/clear", { method: "POST" });
    } catch {
      // ignore
    } finally {
      setTimeout(() => setRefreshing(false), 800);
    }
  };

  if (!adapters) {
    return (
      <EmptyState
        title="Loading adapters…"
        message="Could not fetch adapter information from the API."
      />
    );
  }

  const cacheInfo = adapters.cache;

  return (
    <div class="space-y-6">
      {/* Model Backends */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-medium text-slate-700 mb-4 flex items-center gap-2">
          {Blocks({ size: 16, class: "text-slate-400" })}
          Model Backends
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {adapters.model_backends.map((b) => (
            <div
              class={`flex items-center justify-between rounded-xl border p-4 ${
                b.available
                  ? "border-emerald-200 bg-emerald-50/40"
                  : "border-slate-200 bg-slate-50"
              }`}
            >
              <div class="flex items-center gap-3">
                {b.available
                  ? CheckCircle2({
                      size: 18,
                      class: "text-emerald-500 shrink-0",
                    })
                  : XCircle({ size: 18, class: "text-slate-400 shrink-0" })}
                <div>
                  <p class="text-sm font-semibold text-slate-900">{b.name}</p>
                  <p class="text-xs text-slate-500">
                    {b.available
                      ? "Installed & ready"
                      : b.install_hint || "Not installed"}
                  </p>
                </div>
              </div>
              <span
                class={`px-2 py-0.5 rounded-full text-xs font-medium ${
                  b.available
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-slate-200 text-slate-600"
                }`}
              >
                {b.available ? "Active" : "Unavailable"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Connectors */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-medium text-slate-700 mb-4">
          Supported Connectors
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              {Database({ size: 13 })} Databases
            </div>
            <ul class="space-y-1.5">
              {adapters.connectors.database.map((d) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              {Cloud({ size: 13 })} Cloud Storage
            </div>
            <ul class="space-y-1.5">
              {adapters.connectors.cloud.map((c) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                  {c}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div class="flex items-center gap-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              {Megaphone({ size: 13 })} Ad Platforms
            </div>
            <ul class="space-y-1.5">
              {adapters.connectors.ad_platforms.map((a) => (
                <li class="text-sm text-slate-700 flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                  {a}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Cache */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-sm font-medium text-slate-700 flex items-center gap-2">
            {Cpu({ size: 16, class: "text-slate-400" })}
            Cache Backend
          </h2>
          <button
            onClick={handleClearCache}
            disabled={refreshing()}
            class="flex items-center gap-1.5 px-3 py-1.5 border border-slate-200 rounded-lg text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            {RefreshCw({ size: 12, class: refreshing() ? "animate-spin" : "" })}
            Clear Cache
          </button>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Object.entries(cacheInfo).map(([key, val]) => (
            <div class="p-3 bg-slate-50 rounded-lg">
              <p class="text-xs text-slate-500 uppercase tracking-wider mb-0.5">
                {key.replace(/_/g, " ")}
              </p>
              <p class="text-sm font-semibold text-slate-900 tabular-nums">
                {String(val)}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// System Tab
// ---------------------------------------------------------------------------

function SystemTab({ health }: { health: HealthData | null }) {
  return (
    <div class="space-y-6">
      {/* API status */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-medium text-slate-700 mb-4 flex items-center gap-2">
          {Server({ size: 16, class: "text-slate-400" })}
          API Server Status
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            {
              label: "Status",
              value: (
                <span class="flex items-center gap-2">
                  <span
                    class={`w-2 h-2 rounded-full ${health ? "bg-emerald-500" : "bg-red-500"}`}
                  />
                  {health?.status ?? "Offline"}
                </span>
              ),
            },
            { label: "Version", value: health?.version ?? "—" },
            { label: "Timestamp", value: health?.timestamp ?? "—" },
            { label: "Latest Run", value: health?.latest_run ?? "None" },
          ].map((item) => (
            <div class="p-4 bg-slate-50 rounded-xl">
              <p class="text-xs text-slate-500 uppercase tracking-wider mb-1">
                {item.label}
              </p>
              <p class="text-sm font-semibold font-mono truncate">
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Endpoints */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-sm font-medium text-slate-700 flex items-center gap-2">
            {Cpu({ size: 16, class: "text-slate-400" })}
            API Endpoints
          </h2>
          <div class="flex gap-2">
            <a
              href="/docs"
              target="_blank"
              class="px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-medium text-slate-700 hover:bg-slate-200 transition-colors"
            >
              OpenAPI Docs ↗
            </a>
            <a
              href="/redoc"
              target="_blank"
              class="px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-medium text-slate-700 hover:bg-slate-200 transition-colors"
            >
              ReDoc ↗
            </a>
          </div>
        </div>
        <div class="space-y-1">
          {[
            { method: "GET", path: "/health", desc: "Health check" },
            { method: "GET", path: "/api/v1/runs", desc: "List pipeline runs" },
            {
              method: "GET",
              path: "/api/v1/contributions",
              desc: "Channel contributions",
            },
            {
              method: "GET",
              path: "/api/v1/reconciliation",
              desc: "Reconciled estimates",
            },
            {
              method: "GET",
              path: "/api/v1/optimization",
              desc: "Budget optimization",
            },
            {
              method: "GET",
              path: "/api/v1/response-curves",
              desc: "Response curves",
            },
            {
              method: "GET",
              path: "/api/v1/parameters",
              desc: "Model parameters",
            },
            {
              method: "GET",
              path: "/api/v1/diagnostics",
              desc: "Model diagnostics",
            },
            { method: "GET", path: "/api/v1/roas", desc: "ROAS analysis" },
            {
              method: "GET",
              path: "/api/v1/waterfall",
              desc: "Waterfall decomposition",
            },
            {
              method: "GET",
              path: "/api/v1/data/status",
              desc: "Data source status",
            },
            {
              method: "POST",
              path: "/api/v1/data/upload",
              desc: "Upload data file",
            },
            {
              method: "POST",
              path: "/api/v1/pipeline/run",
              desc: "Trigger pipeline",
            },
            { method: "POST", path: "/api/cache/clear", desc: "Clear cache" },
          ].map((ep) => (
            <div class="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-slate-50 transition-colors">
              <span
                class={`text-[10px] font-mono font-bold px-2 py-0.5 rounded shrink-0 ${
                  ep.method === "GET"
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-amber-100 text-amber-700"
                }`}
              >
                {ep.method}
              </span>
              <code class="text-xs font-mono text-slate-700 flex-1">
                {ep.path}
              </code>
              <span class="text-xs text-slate-500 hidden sm:block">
                {ep.desc}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Platform info */}
      <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60">
        <h2 class="text-sm font-medium text-slate-700 mb-4 flex items-center gap-2">
          {HardDrive({ size: 16, class: "text-slate-400" })}
          Platform Info
        </h2>
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            { label: "Product", value: "Unified-M" },
            { label: "Version", value: health?.version ?? "0.3.0" },
            { label: "UI Framework", value: "SolidJS 1.9" },
            { label: "API Framework", value: "FastAPI" },
            { label: "Model Engine", value: "Ridge / PyMC" },
            { label: "License", value: "Proprietary" },
          ].map((item) => (
            <div class="p-3 bg-slate-50 rounded-xl">
              <p class="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">
                {item.label}
              </p>
              <p class="text-sm font-semibold text-slate-900">{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared helper — Toggle switch
// ---------------------------------------------------------------------------

function Toggle(props: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={props.value}
      onClick={() => props.onChange(!props.value)}
      class={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        props.value ? "bg-indigo-600" : "bg-slate-200"
      }`}
    >
      <span
        class={`pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow-md transition-transform ${
          props.value ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  );
}
