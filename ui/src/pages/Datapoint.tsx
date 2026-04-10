import { createSignal, onMount, Show, For } from "solid-js";
import {
  Database,
  Cloud,
  Upload,
  Plus,
  TestTube,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  Download,
  X,
  Megaphone,
  DollarSign,
  BarChart3,
  Server,
  Zap,
  ArrowRight,
  Check,
  Clock,
  Target,
  Sparkles,
  AlertTriangle,
  Globe,
  Activity,
  Building2,
} from "../lib/icons";
import PageHeader from "../components/PageHeader";
import { api, type SavedConnector, type DataSourceStatus } from "../lib/api";
import { useToast } from "../lib/toast";

// ---------------------------------------------------------------------------
// Connector Type Registry (all tiers)
// ---------------------------------------------------------------------------

interface ConnectorSubtype {
  key: string;
  label: string;
  tier: number;
  fields: { key: string; label: string; type?: string; placeholder?: string }[];
}

interface ConnectorTypeGroup {
  type: string;
  label: string;
  icon: any;
  category: string;
  subtypes: ConnectorSubtype[];
}

const CONNECTOR_TYPES: ConnectorTypeGroup[] = [
  {
    type: "ad_platform",
    label: "Ad Platforms",
    icon: Megaphone,
    category: "Ad Platforms",
    subtypes: [
      {
        key: "google_ads",
        label: "Google Ads (incl. YouTube)",
        tier: 1,
        fields: [
          { key: "client_id", label: "Client ID" },
          { key: "client_secret", label: "Client Secret", type: "password" },
          { key: "refresh_token", label: "Refresh Token", type: "password" },
          { key: "customer_id", label: "Customer ID" },
        ],
      },
      {
        key: "meta_ads",
        label: "Meta Ads (FB + IG)",
        tier: 1,
        fields: [
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "account_id", label: "Ad Account ID" },
        ],
      },
      {
        key: "tiktok_ads",
        label: "TikTok Ads",
        tier: 1,
        fields: [
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "advertiser_id", label: "Advertiser ID" },
        ],
      },
      {
        key: "linkedin_ads",
        label: "LinkedIn Ads",
        tier: 1,
        fields: [
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "account_id", label: "Account ID" },
        ],
      },
      {
        key: "pinterest_ads",
        label: "Pinterest Ads",
        tier: 1,
        fields: [
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "advertiser_id", label: "Advertiser ID" },
        ],
      },
      {
        key: "snapchat_ads",
        label: "Snapchat Ads",
        tier: 1,
        fields: [
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "ad_account_id", label: "Ad Account ID" },
        ],
      },
      {
        key: "twitter_ads",
        label: "X (Twitter) Ads",
        tier: 1,
        fields: [
          { key: "consumer_key", label: "Consumer Key" },
          {
            key: "consumer_secret",
            label: "Consumer Secret",
            type: "password",
          },
          { key: "access_token", label: "Access Token", type: "password" },
          { key: "access_secret", label: "Access Secret", type: "password" },
        ],
      },
      {
        key: "apple_search_ads",
        label: "Apple Search Ads",
        tier: 1,
        fields: [
          { key: "api_key", label: "API Key", type: "password" },
          { key: "org_id", label: "Org ID" },
        ],
      },
    ],
  },
  {
    type: "analytics",
    label: "Analytics",
    icon: BarChart3,
    category: "Analytics",
    subtypes: [
      {
        key: "ga4",
        label: "Google Analytics 4",
        tier: 2,
        fields: [
          { key: "property_id", label: "Property ID" },
          {
            key: "service_account_json",
            label: "Service Account Key (JSON)",
            type: "password",
          },
        ],
      },
      {
        key: "adobe_analytics",
        label: "Adobe Analytics",
        tier: 2,
        fields: [
          { key: "api_key", label: "API Key", type: "password" },
          { key: "company_id", label: "Company ID" },
        ],
      },
    ],
  },
  {
    type: "revenue",
    label: "CRM / Revenue",
    icon: DollarSign,
    category: "CRM/Revenue",
    subtypes: [
      {
        key: "shopify",
        label: "Shopify",
        tier: 2,
        fields: [
          {
            key: "shop_domain",
            label: "Shop Domain",
            placeholder: "your-store.myshopify.com",
          },
          { key: "api_token", label: "Private App Token", type: "password" },
        ],
      },
      {
        key: "salesforce",
        label: "Salesforce CRM",
        tier: 2,
        fields: [
          { key: "instance_url", label: "Instance URL" },
          { key: "client_id", label: "Connected App Client ID" },
          { key: "client_secret", label: "Client Secret", type: "password" },
          { key: "username", label: "Username" },
          { key: "password", label: "Password", type: "password" },
        ],
      },
    ],
  },
  {
    type: "warehouse",
    label: "Data Warehouse",
    icon: Server,
    category: "Warehouse",
    subtypes: [
      {
        key: "bigquery",
        label: "BigQuery",
        tier: 3,
        fields: [
          { key: "project_id", label: "Project ID" },
          { key: "dataset", label: "Dataset" },
          {
            key: "service_account_json",
            label: "Service Account Key (JSON)",
            type: "password",
          },
        ],
      },
      {
        key: "snowflake",
        label: "Snowflake",
        tier: 3,
        fields: [
          {
            key: "account_url",
            label: "Account URL",
            placeholder: "abc123.snowflakecomputing.com",
          },
          { key: "warehouse", label: "Warehouse" },
          { key: "database", label: "Database" },
          { key: "schema", label: "Schema" },
          { key: "user", label: "Username" },
          { key: "password", label: "Password", type: "password" },
        ],
      },
      {
        key: "redshift",
        label: "Redshift",
        tier: 3,
        fields: [
          { key: "host", label: "Host" },
          { key: "port", label: "Port" },
          { key: "database", label: "Database" },
          { key: "user", label: "Username" },
          { key: "password", label: "Password", type: "password" },
        ],
      },
      {
        key: "databricks",
        label: "Databricks",
        tier: 3,
        fields: [
          { key: "host", label: "Workspace URL" },
          { key: "token", label: "Personal Access Token", type: "password" },
          { key: "cluster_id", label: "Cluster ID" },
        ],
      },
      {
        key: "duckdb",
        label: "DuckDB",
        tier: 3,
        fields: [
          {
            key: "path",
            label: "File Path / S3 URI",
            placeholder: "/data/analytics.duckdb or s3://bucket/path",
          },
        ],
      },
    ],
  },
  {
    type: "database",
    label: "Database",
    icon: Database,
    category: "Database",
    subtypes: [
      {
        key: "postgresql",
        label: "PostgreSQL",
        tier: 3,
        fields: [
          { key: "host", label: "Host" },
          { key: "port", label: "Port" },
          { key: "database", label: "Database" },
          { key: "user", label: "Username" },
          { key: "password", label: "Password", type: "password" },
        ],
      },
      {
        key: "mysql",
        label: "MySQL",
        tier: 3,
        fields: [
          { key: "host", label: "Host" },
          { key: "port", label: "Port" },
          { key: "database", label: "Database" },
          { key: "user", label: "Username" },
          { key: "password", label: "Password", type: "password" },
        ],
      },
    ],
  },
  {
    type: "cloud",
    label: "Cloud Storage",
    icon: Cloud,
    category: "Cloud",
    subtypes: [
      {
        key: "s3",
        label: "AWS S3",
        tier: 3,
        fields: [
          { key: "bucket", label: "Bucket" },
          { key: "aws_access_key_id", label: "Access Key ID" },
          {
            key: "aws_secret_access_key",
            label: "Secret Access Key",
            type: "password",
          },
          { key: "region_name", label: "Region" },
        ],
      },
      {
        key: "azure",
        label: "Azure Blob",
        tier: 3,
        fields: [
          { key: "account_name", label: "Account Name" },
          { key: "container_name", label: "Container" },
          { key: "account_key", label: "Account Key", type: "password" },
        ],
      },
    ],
  },
  {
    type: "external",
    label: "External Signals",
    icon: Zap,
    category: "External",
    subtypes: [
      {
        key: "holidays",
        label: "Holidays / Events",
        tier: 4,
        fields: [{ key: "country", label: "Country Code", placeholder: "US" }],
      },
      {
        key: "weather",
        label: "Weather (NOAA / Open-Meteo)",
        tier: 4,
        fields: [
          { key: "latitude", label: "Latitude" },
          { key: "longitude", label: "Longitude" },
        ],
      },
      {
        key: "fred",
        label: "FRED Economic Data",
        tier: 4,
        fields: [
          { key: "api_key", label: "API Key", type: "password" },
          { key: "series_id", label: "Series ID", placeholder: "UNRATE" },
        ],
      },
    ],
  },
];

const CATEGORIES = [
  "Ad Platforms",
  "Analytics",
  "CRM/Revenue",
  "Warehouse",
  "Database",
  "Cloud",
  "External",
  "File Upload",
];

const STATUS_COLORS: Record<string, string> = {
  connected: "bg-emerald-400",
  failed: "bg-red-400",
  untested: "bg-slate-300",
};

// ---------------------------------------------------------------------------
// Use-case Templates
// ---------------------------------------------------------------------------

interface UseCaseTemplate {
  key: string;
  label: string;
  icon: any;
  color: string;
  description: string;
  connectors: string[];
  kpis: string[];
  recommended: string[];
}

const USE_CASE_TEMPLATES: UseCaseTemplate[] = [
  {
    key: "dtc_ecommerce",
    label: "DTC E-commerce",
    icon: Globe,
    color: "indigo",
    description: "Online-first brands selling direct to consumer",
    connectors: ["google_ads", "meta_ads", "shopify", "ga4"],
    kpis: ["Revenue", "ROAS", "CAC"],
    recommended: [
      "media_spend",
      "outcomes",
      "controls",
      "incrementality_tests",
    ],
  },
  {
    key: "b2b_saas",
    label: "B2B SaaS",
    icon: Target,
    color: "violet",
    description: "Software companies with longer sales cycles",
    connectors: ["google_ads", "linkedin_ads", "salesforce", "ga4"],
    kpis: ["Pipeline", "MQLs", "CAC"],
    recommended: ["media_spend", "outcomes", "controls"],
  },
  {
    key: "retail",
    label: "Retail / Omnichannel",
    icon: Building2,
    color: "emerald",
    description: "Physical + digital retail with multiple channels",
    connectors: ["google_ads", "meta_ads", "ga4", "bigquery"],
    kpis: ["Revenue", "Foot Traffic", "ROAS"],
    recommended: [
      "media_spend",
      "outcomes",
      "controls",
      "attribution",
      "incrementality_tests",
    ],
  },
  {
    key: "cpg",
    label: "CPG / FMCG",
    icon: Activity,
    color: "amber",
    description: "Consumer packaged goods with retail distribution",
    connectors: ["meta_ads", "google_ads", "bigquery", "fred"],
    kpis: ["Sales Volume", "Share of Voice", "Brand Lift"],
    recommended: ["media_spend", "outcomes", "controls", "attribution"],
  },
];

// ---------------------------------------------------------------------------
// Recommended data inputs for completeness scoring
// ---------------------------------------------------------------------------

const RECOMMENDED_INPUTS: {
  key: string;
  label: string;
  required: boolean;
}[] = [
  { key: "media_spend", label: "Media Spend", required: true },
  { key: "outcomes", label: "Outcomes / KPI", required: true },
  { key: "controls", label: "Control Variables", required: false },
  {
    key: "incrementality_tests",
    label: "Incrementality Tests",
    required: false,
  },
  { key: "attribution", label: "Attribution Data", required: false },
];

const DATA_TYPES = [
  "media_spend",
  "outcomes",
  "controls",
  "incrementality_tests",
  "attribution",
];

const MMM_FIELDS = [
  { value: "date", label: "Date" },
  { value: "channel_spend", label: "Channel Spend" },
  { value: "target_kpi", label: "Target KPI" },
  { value: "impressions", label: "Impressions" },
  { value: "clicks", label: "Clicks" },
  { value: "conversions", label: "Conversions" },
  { value: "control", label: "Control Variable" },
  { value: "skip", label: "— Skip —" },
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function healthBadge(connector: SavedConnector) {
  if (connector.status === "failed")
    return {
      label: "Error",
      dotClass: "bg-red-500",
      badgeClass: "bg-red-50 text-red-700 border-red-200",
    };
  if (connector.status === "connected") {
    if (connector.last_tested) {
      const age = Date.now() - new Date(connector.last_tested).getTime();
      const STALE_MS = 24 * 60 * 60 * 1000; // 24 h
      if (age > STALE_MS)
        return {
          label: "Stale",
          dotClass: "bg-amber-500",
          badgeClass: "bg-amber-50 text-amber-700 border-amber-200",
        };
    }
    return {
      label: "Live",
      dotClass: "bg-emerald-500",
      badgeClass: "bg-emerald-50 text-emerald-700 border-emerald-200",
    };
  }
  return {
    label: "Untested",
    dotClass: "bg-slate-400",
    badgeClass: "bg-slate-50 text-slate-600 border-slate-200",
  };
}

function ConnectorCard(props: {
  connector: SavedConnector;
  onTest: () => void;
  onDelete: () => void;
  onFetch: () => void;
}) {
  const group = CONNECTOR_TYPES.find((g) =>
    g.subtypes.some((s) => s.key === props.connector.subtype),
  );
  const Icon = group?.icon ?? Database;
  const badge = () => healthBadge(props.connector);

  return (
    <div class="rounded-lg border border-slate-200 bg-white p-4 space-y-3 hover:shadow-sm transition-shadow">
      <div class="flex items-start justify-between">
        <div class="flex items-center gap-3">
          <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
            <Icon size={18} />
          </div>
          <div>
            <p class="text-sm font-semibold text-slate-900">
              {props.connector.name}
            </p>
            <p class="text-xs text-slate-500">
              {props.connector.subtype} · {props.connector.type}
            </p>
          </div>
        </div>
        <span
          class={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${badge().badgeClass}`}
        >
          <span class={`h-1.5 w-1.5 rounded-full ${badge().dotClass}`} />
          {badge().label}
        </span>
      </div>

      <Show when={props.connector.last_tested}>
        <p class="text-xs text-slate-400">
          Tested: {new Date(props.connector.last_tested!).toLocaleString()}
        </p>
      </Show>

      <div class="flex gap-2">
        <button
          onClick={props.onTest}
          class="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <TestTube size={12} /> Test
        </button>
        <button
          onClick={props.onFetch}
          class="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <Download size={12} /> Fetch
        </button>
        <button
          onClick={props.onDelete}
          class="flex items-center gap-1.5 rounded-md border border-red-200 px-2.5 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 transition-colors ml-auto"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function Datapoint() {
  const [connectors, setConnectors] = createSignal<SavedConnector[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [showAdd, setShowAdd] = createSignal(false);
  const [showFetch, setShowFetch] = createSignal<string | null>(null);
  const [dataStatus, setDataStatus] = createSignal<
    Record<string, DataSourceStatus>
  >({});
  const [activeTemplate, setActiveTemplate] =
    createSignal<UseCaseTemplate | null>(null);
  const { addToast } = useToast();

  // Add form — wizard steps
  const [wizardStep, setWizardStep] = createSignal(0);
  const [selectedCategory, setSelectedCategory] = createSignal<string | null>(
    null,
  );
  const [selectedSubtype, setSelectedSubtype] =
    createSignal<ConnectorSubtype | null>(null);
  const [selectedGroup, setSelectedGroup] =
    createSignal<ConnectorTypeGroup | null>(null);
  const [name, setName] = createSignal("");
  const [configFields, setConfigFields] = createSignal<Record<string, string>>(
    {},
  );
  const [saving, setSaving] = createSignal(false);
  const [testResult, setTestResult] = createSignal<"success" | "error" | null>(
    null,
  );

  // Column mapping
  const [showMapping, setShowMapping] = createSignal(false);
  const [mappingColumns, setMappingColumns] = createSignal<string[]>([
    "date",
    "fb_spend",
    "google_spend",
    "total_revenue",
    "impressions",
  ]);
  const [columnMappings, setColumnMappings] = createSignal<
    Record<string, string>
  >({});

  // Fetch form
  const [fetchQuery, setFetchQuery] = createSignal("");
  const [fetchDataType, setFetchDataType] = createSignal("media_spend");
  const [fetching, setFetching] = createSignal(false);

  // File upload
  const [uploadFile, setUploadFile] = createSignal<File | null>(null);
  const [uploadDataType, setUploadDataType] = createSignal("media_spend");
  const [uploading, setUploading] = createSignal(false);

  const refresh = () => {
    api
      .listConnectors()
      .then((r) => setConnectors(r.connectors))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  onMount(() => {
    refresh();
    api
      .dataStatus()
      .then((s) => setDataStatus(s as Record<string, DataSourceStatus>))
      .catch(() => {});
  });

  const openAddWizard = () => {
    setWizardStep(0);
    setSelectedCategory(null);
    setSelectedSubtype(null);
    setSelectedGroup(null);
    setName("");
    setConfigFields({});
    setTestResult(null);
    setShowAdd(true);
  };

  const selectCategory = (cat: string) => {
    setSelectedCategory(cat);
    setWizardStep(1);
  };

  const selectSubtype = (group: ConnectorTypeGroup, sub: ConnectorSubtype) => {
    setSelectedGroup(group);
    setSelectedSubtype(sub);
    setConfigFields({});
    setWizardStep(2);
  };

  const handleTestConnection = async () => {
    setTestResult(null);
    await new Promise((r) => setTimeout(r, 800));
    setTestResult("success");
  };

  const handleCreate = async () => {
    if (!name().trim() || !selectedGroup() || !selectedSubtype()) return;
    setSaving(true);
    try {
      await api.createConnector(
        name(),
        selectedGroup()!.type,
        selectedSubtype()!.key,
        configFields(),
      );
      addToast("success", `Connection "${name()}" saved`);
      setShowAdd(false);
      refresh();
    } catch (e: any) {
      addToast("error", e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async (id: string) => {
    try {
      const res = await api.testConnector(id);
      if (res.connected) {
        addToast("success", "Connection successful");
      } else {
        addToast("error", res.message || "Connection failed");
      }
      refresh();
    } catch (e: any) {
      addToast("error", e.message);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteConnector(id);
      addToast("info", "Connection deleted", {
        onUndo: () => addToast("info", "Undo not supported for this action"),
      });
      refresh();
    } catch (e: any) {
      addToast("error", e.message);
    }
  };

  const handleFetchData = async () => {
    if (!showFetch() || !fetchQuery().trim()) return;
    setFetching(true);
    try {
      const res = await api.fetchFromConnector(
        showFetch()!,
        fetchQuery(),
        fetchDataType(),
      );
      addToast("success", `Imported ${res.rows} rows as ${res.data_type}`);
      setShowFetch(null);
      setFetchQuery("");
    } catch (e: any) {
      addToast("error", e.message);
    } finally {
      setFetching(false);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile()) return;
    setUploading(true);
    try {
      const res = await api.uploadFile(uploadDataType(), uploadFile()!);
      addToast("success", `Uploaded ${res.rows} rows as ${uploadDataType()}`);
      setUploadFile(null);
    } catch (e: any) {
      addToast("error", e.message);
    } finally {
      setUploading(false);
    }
  };

  // Completeness scoring
  const completenessInfo = () => {
    const tpl = activeTemplate();
    const inputs = tpl
      ? RECOMMENDED_INPUTS.filter((i) => tpl.recommended.includes(i.key))
      : RECOMMENDED_INPUTS;
    const ds = dataStatus();
    const filled = inputs.filter((i) => {
      const src = ds[i.key];
      return src && (src.exists || (src.n_rows ?? src.rows ?? 0) > 0);
    });
    const pct = inputs.length
      ? Math.round((filled.length / inputs.length) * 100)
      : 0;
    const gaps = inputs.filter((i) => {
      const src = ds[i.key];
      return !src || (!src.exists && (src.n_rows ?? src.rows ?? 0) === 0);
    });
    return { pct, filled: filled.length, total: inputs.length, gaps };
  };

  // Data freshness rows
  const freshnessRows = () => {
    const ds = dataStatus();
    return Object.entries(ds)
      .filter(([, v]) => v && (v.exists || (v.n_rows ?? v.rows ?? 0) > 0))
      .map(([key, v]) => ({
        key,
        rows: v.n_rows ?? v.rows ?? 0,
        lastUpdated: v.last_updated ?? null,
        sizeKb: v.size_bytes ? Math.round(v.size_bytes / 1024) : null,
      }));
  };

  const applyTemplate = (tpl: UseCaseTemplate) => {
    setActiveTemplate(tpl);
    // Pre-select category for the first recommended connector
    const firstConn = tpl.connectors[0];
    const group = CONNECTOR_TYPES.find((g) =>
      g.subtypes.some((s) => s.key === firstConn),
    );
    if (group) {
      setSelectedCategory(group.category);
      const sub = group.subtypes.find((s) => s.key === firstConn);
      if (sub) {
        selectSubtype(group, sub);
        setShowAdd(true);
        return;
      }
    }
    // Fallback: open the wizard
    openAddWizard();
  };

  const filteredSubtypes = () => {
    if (!selectedCategory()) return [];
    if (selectedCategory() === "File Upload") return [];
    return CONNECTOR_TYPES.filter((g) => g.category === selectedCategory());
  };

  return (
    <div>
      <PageHeader
        title="Connections"
        description="Manage data source connections — ad platforms, analytics, warehouses, and more"
      />

      {/* ---- Use-case Templates ---- */}
      <div class="mb-6">
        <h2 class="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-1.5">
          <Sparkles size={14} /> Quick Start Templates
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <For each={USE_CASE_TEMPLATES}>
            {(tpl) => {
              const Icon = tpl.icon;
              const isActive = () => activeTemplate()?.key === tpl.key;
              return (
                <button
                  onClick={() => applyTemplate(tpl)}
                  class={`relative rounded-lg border p-4 text-left transition-all ${
                    isActive()
                      ? "border-indigo-300 bg-indigo-50/60 ring-1 ring-indigo-200"
                      : "border-slate-200 bg-white hover:border-indigo-200 hover:bg-indigo-50/30"
                  }`}
                >
                  <div class="flex items-center gap-2 mb-2">
                    <div
                      class={`flex h-8 w-8 items-center justify-center rounded-lg ${
                        tpl.color === "indigo"
                          ? "bg-indigo-100 text-indigo-600"
                          : tpl.color === "violet"
                            ? "bg-violet-100 text-violet-600"
                            : tpl.color === "emerald"
                              ? "bg-emerald-100 text-emerald-600"
                              : "bg-amber-100 text-amber-600"
                      }`}
                    >
                      <Icon size={16} />
                    </div>
                    <span class="text-sm font-semibold text-slate-900">
                      {tpl.label}
                    </span>
                  </div>
                  <p class="text-xs text-slate-500 mb-2">{tpl.description}</p>
                  <div class="flex flex-wrap gap-1">
                    <For each={tpl.kpis}>
                      {(k) => (
                        <span class="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                          {k}
                        </span>
                      )}
                    </For>
                  </div>
                  <Show when={isActive()}>
                    <span class="absolute top-2 right-2 flex h-5 w-5 items-center justify-center rounded-full bg-indigo-600 text-white">
                      <Check size={12} />
                    </span>
                  </Show>
                </button>
              );
            }}
          </For>
        </div>
      </div>

      {/* ---- Completeness Score + Data Freshness ---- */}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {/* Completeness Score */}
        <div class="rounded-lg border border-slate-200 bg-white p-5">
          <h3 class="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-1.5">
            <Target size={14} /> Data Completeness
          </h3>
          <div class="flex items-center gap-4 mb-3">
            <div class="relative h-16 w-16 shrink-0">
              <svg class="h-16 w-16 -rotate-90" viewBox="0 0 36 36">
                <path
                  d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#e2e8f0"
                  stroke-width="3"
                />
                <path
                  d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke={
                    completenessInfo().pct >= 80
                      ? "#10b981"
                      : completenessInfo().pct >= 50
                        ? "#f59e0b"
                        : "#ef4444"
                  }
                  stroke-width="3"
                  stroke-dasharray={`${completenessInfo().pct}, 100`}
                  stroke-linecap="round"
                />
              </svg>
              <span class="absolute inset-0 flex items-center justify-center text-sm font-bold text-slate-900">
                {completenessInfo().pct}%
              </span>
            </div>
            <div>
              <p class="text-sm text-slate-700">
                Your model has{" "}
                <span class="font-semibold">{completenessInfo().pct}%</span> of
                recommended data inputs
              </p>
              <p class="text-xs text-slate-500 mt-0.5">
                {completenessInfo().filled} of {completenessInfo().total}{" "}
                sources connected
              </p>
            </div>
          </div>
          <Show when={completenessInfo().gaps.length > 0}>
            <div class="space-y-1.5">
              <p class="text-xs font-medium text-slate-500 uppercase tracking-wider">
                Missing
              </p>
              <For each={completenessInfo().gaps}>
                {(gap) => (
                  <div class="flex items-center gap-2 text-xs">
                    <span
                      class={`h-1.5 w-1.5 rounded-full ${gap.required ? "bg-red-400" : "bg-amber-400"}`}
                    />
                    <span class="text-slate-700">{gap.label}</span>
                    <span class="text-slate-400">
                      {gap.required ? "(required)" : "(recommended)"}
                    </span>
                  </div>
                )}
              </For>
            </div>
          </Show>
        </div>

        {/* Data Freshness */}
        <div class="rounded-lg border border-slate-200 bg-white p-5">
          <h3 class="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-1.5">
            <Clock size={14} /> Data Freshness
          </h3>
          <Show
            when={freshnessRows().length > 0}
            fallback={
              <p class="text-xs text-slate-500">
                No data sources loaded yet. Upload data or connect a source to
                see freshness info.
              </p>
            }
          >
            <div class="overflow-x-auto">
              <table class="w-full text-xs">
                <thead>
                  <tr class="border-b border-slate-100 text-left text-slate-500">
                    <th class="pb-2 pr-4 font-medium">Source</th>
                    <th class="pb-2 pr-4 font-medium text-right">Rows</th>
                    <th class="pb-2 pr-4 font-medium text-right">Size</th>
                    <th class="pb-2 font-medium">Last Synced</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={freshnessRows()}>
                    {(row) => {
                      const stale = () => {
                        if (!row.lastUpdated) return false;
                        return (
                          Date.now() - new Date(row.lastUpdated).getTime() >
                          24 * 60 * 60 * 1000
                        );
                      };
                      return (
                        <tr class="border-b border-slate-50">
                          <td class="py-2 pr-4 font-medium text-slate-700">
                            {row.key}
                          </td>
                          <td class="py-2 pr-4 text-right text-slate-600">
                            {row.rows.toLocaleString()}
                          </td>
                          <td class="py-2 pr-4 text-right text-slate-600">
                            {row.sizeKb != null
                              ? `${row.sizeKb.toLocaleString()} KB`
                              : "—"}
                          </td>
                          <td class="py-2">
                            <Show
                              when={row.lastUpdated}
                              fallback={<span class="text-slate-400">—</span>}
                            >
                              <span
                                class={`inline-flex items-center gap-1 ${stale() ? "text-amber-600" : "text-slate-600"}`}
                              >
                                <Show when={stale()}>
                                  <AlertTriangle size={10} />
                                </Show>
                                {new Date(row.lastUpdated!).toLocaleDateString(
                                  "en-US",
                                  {
                                    month: "short",
                                    day: "numeric",
                                    hour: "2-digit",
                                    minute: "2-digit",
                                  },
                                )}
                              </span>
                            </Show>
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            </div>
          </Show>
        </div>
      </div>

      {/* Quick file upload strip */}
      <div class="rounded-lg border border-slate-200 bg-white p-4 flex flex-wrap items-end gap-4">
        <div class="flex items-center gap-2 text-sm font-medium text-slate-700">
          <Upload size={16} /> Quick Upload
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1">Data type</label>
          <select
            value={uploadDataType()}
            onChange={(e) => setUploadDataType(e.target.value)}
            class="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          >
            <For each={DATA_TYPES}>
              {(dt) => <option value={dt}>{dt}</option>}
            </For>
          </select>
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1">File</label>
          <input
            type="file"
            accept=".csv,.parquet,.xlsx,.xls"
            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            class="text-sm"
          />
        </div>
        <button
          onClick={handleFileUpload}
          disabled={!uploadFile() || uploading()}
          class="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors"
        >
          <Show when={uploading()} fallback={<Upload size={14} />}>
            <Loader2 size={14} class="animate-spin" />
          </Show>
          Upload
        </button>
      </div>

      {/* Saved connections grid */}
      <div class="mt-6 flex items-center justify-between">
        <h2 class="text-sm font-semibold text-slate-900">Saved Connections</h2>
        <button
          onClick={openAddWizard}
          class="flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 transition-colors"
        >
          <Plus size={14} /> Add Source
        </button>
      </div>

      <Show
        when={!loading()}
        fallback={
          <div class="mt-4 flex justify-center py-12">
            <Loader2 size={24} class="animate-spin text-slate-400" />
          </div>
        }
      >
        <Show
          when={connectors().length > 0}
          fallback={
            <div class="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 py-12 text-center">
              <Database size={32} class="mx-auto text-slate-300" />
              <p class="mt-2 text-sm text-slate-500">
                No saved connections yet
              </p>
              <button
                onClick={openAddWizard}
                class="mt-3 text-sm font-medium text-indigo-600 hover:text-indigo-700"
              >
                Add your first connection
              </button>
            </div>
          }
        >
          <div class="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <For each={connectors()}>
              {(c) => (
                <ConnectorCard
                  connector={c}
                  onTest={() => handleTest(c.id)}
                  onDelete={() => handleDelete(c.id)}
                  onFetch={() => {
                    setShowFetch(c.id);
                    setFetchQuery("");
                  }}
                />
              )}
            </For>
          </div>
        </Show>
      </Show>

      {/* ================================================================ */}
      {/* Add Source Wizard Modal                                          */}
      {/* ================================================================ */}
      <Show when={showAdd()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div
            class="absolute inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => setShowAdd(false)}
          />
          <div class="relative w-full max-w-xl max-h-[85vh] overflow-y-auto rounded-xl bg-white shadow-xl">
            {/* Header */}
            <div class="sticky top-0 z-10 bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between rounded-t-xl">
              <div>
                <h3 class="text-base font-semibold text-slate-900">
                  {wizardStep() === 0 && "Choose Source Type"}
                  {wizardStep() === 1 && `Select ${selectedCategory()}`}
                  {wizardStep() === 2 &&
                    `Configure: ${selectedSubtype()?.label}`}
                </h3>
                {/* Breadcrumb */}
                <div class="flex items-center gap-1 mt-1 text-xs text-slate-400">
                  <button
                    onClick={() => {
                      setWizardStep(0);
                      setSelectedCategory(null);
                    }}
                    class="hover:text-slate-600"
                  >
                    Source Type
                  </button>
                  <Show when={wizardStep() >= 1}>
                    <span>›</span>
                    <button
                      onClick={() => setWizardStep(1)}
                      class="hover:text-slate-600"
                    >
                      {selectedCategory()}
                    </button>
                  </Show>
                  <Show when={wizardStep() >= 2}>
                    <span>›</span>
                    <span class="text-slate-600">
                      {selectedSubtype()?.label}
                    </span>
                  </Show>
                </div>
              </div>
              <button
                onClick={() => setShowAdd(false)}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div class="p-6">
              {/* Step 0: Choose category */}
              <Show when={wizardStep() === 0}>
                <div class="grid grid-cols-2 gap-3">
                  <For each={CATEGORIES}>
                    {(cat) => {
                      const group = CONNECTOR_TYPES.find(
                        (g) => g.category === cat,
                      );
                      const Icon = group?.icon ?? Upload;
                      return (
                        <button
                          onClick={() => selectCategory(cat)}
                          class="flex items-center gap-3 rounded-lg border border-slate-200 p-4 text-left hover:border-indigo-300 hover:bg-indigo-50/50 transition-all group"
                        >
                          <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-600 group-hover:bg-indigo-100 group-hover:text-indigo-600 transition-colors">
                            <Icon size={20} />
                          </div>
                          <div>
                            <p class="text-sm font-semibold text-slate-900">
                              {cat}
                            </p>
                            <p class="text-xs text-slate-500">
                              {cat === "File Upload"
                                ? "CSV, Excel, Parquet"
                                : `${CONNECTOR_TYPES.filter((g) => g.category === cat).reduce((s, g) => s + g.subtypes.length, 0)} sources`}
                            </p>
                          </div>
                        </button>
                      );
                    }}
                  </For>
                </div>
              </Show>

              {/* Step 1: Choose subtype */}
              <Show
                when={
                  wizardStep() === 1 && selectedCategory() !== "File Upload"
                }
              >
                <div class="space-y-2">
                  <For each={filteredSubtypes()}>
                    {(group) => (
                      <For each={group.subtypes}>
                        {(sub) => (
                          <button
                            onClick={() => selectSubtype(group, sub)}
                            class="flex w-full items-center gap-3 rounded-lg border border-slate-200 p-3 text-left hover:border-indigo-300 hover:bg-indigo-50/30 transition-all"
                          >
                            <div class="flex-1">
                              <span class="text-sm font-medium text-slate-900">
                                {sub.label}
                              </span>
                              <span class="ml-2 text-[10px] font-semibold text-slate-400 bg-slate-100 rounded px-1.5 py-0.5">
                                Tier {sub.tier}
                              </span>
                            </div>
                            <ArrowRight size={14} class="text-slate-300" />
                          </button>
                        )}
                      </For>
                    )}
                  </For>
                </div>
              </Show>

              {/* Step 2: Configure */}
              <Show when={wizardStep() === 2 && selectedSubtype()}>
                <div class="space-y-4">
                  <div>
                    <label class="block text-xs font-medium text-slate-600 mb-1">
                      Connection Name
                    </label>
                    <input
                      type="text"
                      value={name()}
                      onInput={(e) => setName(e.target.value)}
                      placeholder={`My ${selectedSubtype()!.label}`}
                      class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>

                  <div class="grid grid-cols-2 gap-3">
                    <For each={selectedSubtype()!.fields}>
                      {(field) => (
                        <div
                          class={
                            field.key.includes("json") ||
                            field.key.includes("url")
                              ? "col-span-2"
                              : ""
                          }
                        >
                          <label class="block text-xs font-medium text-slate-600 mb-1">
                            {field.label}
                          </label>
                          <input
                            type={field.type ?? "text"}
                            value={configFields()[field.key] ?? ""}
                            placeholder={field.placeholder}
                            onInput={(e) =>
                              setConfigFields((p) => ({
                                ...p,
                                [field.key]: e.target.value,
                              }))
                            }
                            class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                          />
                        </div>
                      )}
                    </For>
                  </div>

                  {/* Test connection */}
                  <div class="flex items-center gap-3 pt-2">
                    <button
                      onClick={handleTestConnection}
                      class="flex items-center gap-1.5 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
                    >
                      <TestTube size={14} /> Test Connection
                    </button>
                    <Show when={testResult() === "success"}>
                      <span class="flex items-center gap-1 text-sm text-emerald-600 font-medium">
                        <CheckCircle2 size={14} /> Connected
                      </span>
                    </Show>
                    <Show when={testResult() === "error"}>
                      <span class="flex items-center gap-1 text-sm text-red-600 font-medium">
                        <XCircle size={14} /> Failed
                      </span>
                    </Show>
                  </div>

                  {/* Column mapping link */}
                  <button
                    onClick={() => setShowMapping(true)}
                    class="flex items-center gap-1.5 text-sm font-medium text-indigo-600 hover:text-indigo-700"
                  >
                    <ArrowRight size={14} /> Column Mapping Wizard
                  </button>

                  {/* Actions */}
                  <div class="flex justify-end gap-2 pt-4 border-t border-slate-200">
                    <button
                      onClick={() => setShowAdd(false)}
                      class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleCreate}
                      disabled={!name().trim() || saving()}
                      class="flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                    >
                      <Show when={saving()} fallback={<Plus size={14} />}>
                        <Loader2 size={14} class="animate-spin" />
                      </Show>
                      Save & Sync
                    </button>
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </Show>

      {/* ================================================================ */}
      {/* Column Mapping Wizard                                            */}
      {/* ================================================================ */}
      <Show when={showMapping()}>
        <div class="fixed inset-0 z-50 flex items-center justify-center">
          <div
            class="absolute inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => setShowMapping(false)}
          />
          <div class="relative w-full max-w-lg rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="text-base font-semibold text-slate-900">
                  Column Mapping
                </h3>
                <p class="text-xs text-slate-500 mt-0.5">
                  Map your source columns to MMM schema fields
                </p>
              </div>
              <button
                onClick={() => setShowMapping(false)}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div class="space-y-3">
              <div class="grid grid-cols-[1fr_auto_1fr] gap-3 items-center text-xs font-semibold text-slate-500 uppercase tracking-wider">
                <span>Your Column</span>
                <span />
                <span>MMM Field</span>
              </div>
              <For each={mappingColumns()}>
                {(col) => (
                  <div class="grid grid-cols-[1fr_auto_1fr] gap-3 items-center">
                    <div class="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-mono text-slate-700">
                      {col}
                    </div>
                    <span class="text-slate-300">→</span>
                    <select
                      value={columnMappings()[col] ?? "skip"}
                      onChange={(e) =>
                        setColumnMappings((p) => ({
                          ...p,
                          [col]: e.currentTarget.value,
                        }))
                      }
                      class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                    >
                      <For each={MMM_FIELDS}>
                        {(f) => <option value={f.value}>{f.label}</option>}
                      </For>
                    </select>
                  </div>
                )}
              </For>
            </div>

            <div class="flex justify-end gap-2 pt-3 border-t border-slate-200">
              <button
                onClick={() => setShowMapping(false)}
                class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowMapping(false);
                  addToast("success", "Column mappings saved");
                }}
                class="flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
              >
                <Check size={14} /> Apply Mapping
              </button>
            </div>
          </div>
        </div>
      </Show>

      {/* Fetch Data Modal */}
      <Show when={showFetch()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div
            class="absolute inset-0 bg-black/20"
            onClick={() => setShowFetch(null)}
          />
          <div class="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">Fetch Data</h3>
              <button
                onClick={() => setShowFetch(null)}
                class="text-slate-400 hover:text-slate-600"
              >
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                SQL Query or File Path
              </label>
              <textarea
                value={fetchQuery()}
                onInput={(e) => setFetchQuery(e.target.value)}
                rows={3}
                placeholder="SELECT * FROM media_spend"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">
                Import as
              </label>
              <select
                value={fetchDataType()}
                onChange={(e) => setFetchDataType(e.target.value)}
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                <For each={DATA_TYPES}>
                  {(dt) => <option value={dt}>{dt}</option>}
                </For>
              </select>
            </div>

            <div class="flex justify-end gap-2">
              <button
                onClick={() => setShowFetch(null)}
                class="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleFetchData}
                disabled={!fetchQuery().trim() || fetching()}
                class="flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                <Show when={fetching()} fallback={<Download size={14} />}>
                  <Loader2 size={14} class="animate-spin" />
                </Show>
                Fetch
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
