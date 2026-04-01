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
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import { api, type SavedConnector } from "../lib/api";
import { useToast } from "../lib/toast";

const CONNECTOR_TYPES = [
  {
    type: "database",
    label: "Database",
    icon: Database,
    subtypes: [
      { key: "postgresql", label: "PostgreSQL" },
      { key: "mysql", label: "MySQL" },
      { key: "sqlserver", label: "SQL Server" },
      { key: "sqlite", label: "SQLite" },
    ],
  },
  {
    type: "cloud",
    label: "Cloud Storage",
    icon: Cloud,
    subtypes: [
      { key: "s3", label: "AWS S3" },
      { key: "azure", label: "Azure Blob" },
    ],
  },
] as const;

const STATUS_COLORS: Record<string, string> = {
  connected: "bg-emerald-400",
  failed: "bg-red-400",
  untested: "bg-slate-300",
};

const DATA_TYPES = ["media_spend", "outcomes", "controls", "incrementality_tests", "attribution"];

function ConnectorCard(props: {
  connector: SavedConnector;
  onTest: () => void;
  onDelete: () => void;
  onFetch: () => void;
}) {
  const typeInfo = CONNECTOR_TYPES.find((t) => t.type === props.connector.type);
  const Icon = typeInfo?.icon ?? Database;

  return (
    <div class="rounded-lg border border-slate-200 bg-white p-4 space-y-3">
      <div class="flex items-start justify-between">
        <div class="flex items-center gap-3">
          <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
            <Icon size={18} />
          </div>
          <div>
            <p class="text-sm font-semibold text-slate-900">{props.connector.name}</p>
            <p class="text-xs text-slate-500">
              {props.connector.subtype} &middot; {props.connector.type}
            </p>
          </div>
        </div>
        <span class={`mt-1 h-2 w-2 shrink-0 rounded-full ${STATUS_COLORS[props.connector.status] ?? "bg-slate-300"}`} />
      </div>

      <Show when={props.connector.last_tested}>
        <p class="text-xs text-slate-400">
          Tested: {new Date(props.connector.last_tested!).toLocaleString()}
        </p>
      </Show>

      <div class="flex gap-2">
        <button onClick={props.onTest} class="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors">
          <TestTube size={12} /> Test
        </button>
        <button onClick={props.onFetch} class="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors">
          <Download size={12} /> Fetch
        </button>
        <button onClick={props.onDelete} class="flex items-center gap-1.5 rounded-md border border-red-200 px-2.5 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 transition-colors ml-auto">
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

export default function Datapoint() {
  const [connectors, setConnectors] = createSignal<SavedConnector[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [showAdd, setShowAdd] = createSignal(false);
  const [showFetch, setShowFetch] = createSignal<string | null>(null);
  const { addToast } = useToast();

  // Add form
  const [name, setName] = createSignal("");
  const [connType, setConnType] = createSignal("database");
  const [subtype, setSubtype] = createSignal("postgresql");
  const [configFields, setConfigFields] = createSignal<Record<string, string>>({});
  const [saving, setSaving] = createSignal(false);

  // Fetch form
  const [fetchQuery, setFetchQuery] = createSignal("");
  const [fetchDataType, setFetchDataType] = createSignal("media_spend");
  const [fetching, setFetching] = createSignal(false);

  // File upload
  const [uploadFile, setUploadFile] = createSignal<File | null>(null);
  const [uploadDataType, setUploadDataType] = createSignal("media_spend");
  const [uploading, setUploading] = createSignal(false);

  const refresh = () => {
    api.listConnectors().then((r) => setConnectors(r.connectors)).catch(() => {}).finally(() => setLoading(false));
  };

  onMount(refresh);

  const selectedType = () => CONNECTOR_TYPES.find((t) => t.type === connType());

  const handleCreate = async () => {
    if (!name().trim()) return;
    setSaving(true);
    try {
      await api.createConnector(name(), connType(), subtype(), configFields());
      addToast("success", `Connection "${name()}" saved`);
      setShowAdd(false);
      setName("");
      setConfigFields({});
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
      addToast("info", "Connection deleted");
      refresh();
    } catch (e: any) {
      addToast("error", e.message);
    }
  };

  const handleFetchData = async () => {
    if (!showFetch() || !fetchQuery().trim()) return;
    setFetching(true);
    try {
      const res = await api.fetchFromConnector(showFetch()!, fetchQuery(), fetchDataType());
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

  const fieldFor = (key: string, label: string, type = "text") => (
    <div>
      <label class="block text-xs font-medium text-slate-600 mb-1">{label}</label>
      <input
        type={type}
        value={configFields()[key] ?? ""}
        onInput={(e) => setConfigFields((p) => ({ ...p, [key]: e.target.value }))}
        class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
      />
    </div>
  );

  const renderConfigForm = () => {
    if (connType() === "database") {
      return (
        <div class="grid grid-cols-2 gap-3">
          {fieldFor("host", "Host")}
          {fieldFor("port", "Port")}
          {fieldFor("database", "Database")}
          {fieldFor("user", "Username")}
          {fieldFor("password", "Password", "password")}
        </div>
      );
    }
    if (subtype() === "s3") {
      return (
        <div class="grid grid-cols-2 gap-3">
          {fieldFor("bucket", "Bucket")}
          {fieldFor("aws_access_key_id", "Access Key ID")}
          {fieldFor("aws_secret_access_key", "Secret Access Key", "password")}
          {fieldFor("region_name", "Region")}
        </div>
      );
    }
    return (
      <div class="grid grid-cols-2 gap-3">
        {fieldFor("account_name", "Account Name")}
        {fieldFor("container_name", "Container")}
        {fieldFor("account_key", "Account Key", "password")}
      </div>
    );
  };

  return (
    <div>
      <PageHeader
        title="Connections"
        description="Manage data source connections and file uploads"
      />

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
          onClick={() => setShowAdd(true)}
          class="flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 transition-colors"
        >
          <Plus size={14} /> Add Connection
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
              <p class="mt-2 text-sm text-slate-500">No saved connections yet</p>
              <button
                onClick={() => setShowAdd(true)}
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
                  onFetch={() => { setShowFetch(c.id); setFetchQuery(""); }}
                />
              )}
            </For>
          </div>
        </Show>
      </Show>

      {/* Add Connection Modal */}
      <Show when={showAdd()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div class="absolute inset-0 bg-black/20" onClick={() => setShowAdd(false)} />
          <div class="relative w-full max-w-lg rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">New Connection</h3>
              <button onClick={() => setShowAdd(false)} class="text-slate-400 hover:text-slate-600">
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Name</label>
              <input
                type="text"
                value={name()}
                onInput={(e) => setName(e.target.value)}
                placeholder="My PostgreSQL"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">Type</label>
                <select
                  value={connType()}
                  onChange={(e) => {
                    setConnType(e.target.value);
                    const t = CONNECTOR_TYPES.find((ct) => ct.type === e.target.value);
                    if (t) setSubtype(t.subtypes[0].key);
                    setConfigFields({});
                  }}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={CONNECTOR_TYPES}>
                    {(t) => <option value={t.type}>{t.label}</option>}
                  </For>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-600 mb-1">Provider</label>
                <select
                  value={subtype()}
                  onChange={(e) => { setSubtype(e.target.value); setConfigFields({}); }}
                  class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  <For each={selectedType()?.subtypes ?? []}>
                    {(s) => <option value={s.key}>{s.label}</option>}
                  </For>
                </select>
              </div>
            </div>

            {renderConfigForm()}

            <div class="flex justify-end gap-2 pt-2">
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
                Save Connection
              </button>
            </div>
          </div>
        </div>
      </Show>

      {/* Fetch Data Modal */}
      <Show when={showFetch()}>
        <div class="fixed inset-0 z-40 flex items-center justify-center">
          <div class="absolute inset-0 bg-black/20" onClick={() => setShowFetch(null)} />
          <div class="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div class="flex items-center justify-between">
              <h3 class="text-base font-semibold text-slate-900">Fetch Data</h3>
              <button onClick={() => setShowFetch(null)} class="text-slate-400 hover:text-slate-600">
                <X size={18} />
              </button>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">SQL Query or File Path</label>
              <textarea
                value={fetchQuery()}
                onInput={(e) => setFetchQuery(e.target.value)}
                rows={3}
                placeholder="SELECT * FROM media_spend"
                class="w-full rounded-md border border-slate-300 px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Import as</label>
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
