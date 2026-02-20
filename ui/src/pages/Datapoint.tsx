import { useEffect, useState } from "react";
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

function ConnectorCard({
  connector,
  onTest,
  onDelete,
  onFetch,
}: {
  connector: SavedConnector;
  onTest: () => void;
  onDelete: () => void;
  onFetch: () => void;
}) {
  const typeInfo = CONNECTOR_TYPES.find((t) => t.type === connector.type);
  const Icon = typeInfo?.icon ?? Database;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
            <Icon size={18} />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">{connector.name}</p>
            <p className="text-xs text-slate-500">
              {connector.subtype} &middot; {connector.type}
            </p>
          </div>
        </div>
        <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${STATUS_COLORS[connector.status] ?? "bg-slate-300"}`} />
      </div>

      {connector.last_tested && (
        <p className="text-xs text-slate-400">
          Tested: {new Date(connector.last_tested).toLocaleString()}
        </p>
      )}

      <div className="flex gap-2">
        <button
          onClick={onTest}
          className="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <TestTube size={12} /> Test
        </button>
        <button
          onClick={onFetch}
          className="flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <Download size={12} /> Fetch
        </button>
        <button
          onClick={onDelete}
          className="flex items-center gap-1.5 rounded-md border border-red-200 px-2.5 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 transition-colors ml-auto"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

export default function Datapoint() {
  const [connectors, setConnectors] = useState<SavedConnector[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [showFetch, setShowFetch] = useState<string | null>(null);
  const { addToast } = useToast();

  // Add form
  const [name, setName] = useState("");
  const [connType, setConnType] = useState("database");
  const [subtype, setSubtype] = useState("postgresql");
  const [configFields, setConfigFields] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  // Fetch form
  const [fetchQuery, setFetchQuery] = useState("");
  const [fetchDataType, setFetchDataType] = useState("media_spend");
  const [fetching, setFetching] = useState(false);

  // File upload
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadDataType, setUploadDataType] = useState("media_spend");
  const [uploading, setUploading] = useState(false);

  const refresh = () => {
    api.listConnectors().then((r) => setConnectors(r.connectors)).catch(() => {}).finally(() => setLoading(false));
  };

  useEffect(refresh, []);

  const selectedType = CONNECTOR_TYPES.find((t) => t.type === connType);

  const handleCreate = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await api.createConnector(name, connType, subtype, configFields);
      addToast("success", `Connection "${name}" saved`);
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
    if (!showFetch || !fetchQuery.trim()) return;
    setFetching(true);
    try {
      const res = await api.fetchFromConnector(showFetch, fetchQuery, fetchDataType);
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
    if (!uploadFile) return;
    setUploading(true);
    try {
      const res = await api.uploadFile(uploadDataType, uploadFile);
      addToast("success", `Uploaded ${res.rows} rows as ${uploadDataType}`);
      setUploadFile(null);
    } catch (e: any) {
      addToast("error", e.message);
    } finally {
      setUploading(false);
    }
  };

  const fieldFor = (key: string, label: string, type = "text") => (
    <div key={key}>
      <label className="block text-xs font-medium text-slate-600 mb-1">{label}</label>
      <input
        type={type}
        value={configFields[key] ?? ""}
        onChange={(e) => setConfigFields((p) => ({ ...p, [key]: e.target.value }))}
        className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
      />
    </div>
  );

  const renderConfigForm = () => {
    if (connType === "database") {
      return (
        <div className="grid grid-cols-2 gap-3">
          {fieldFor("host", "Host")}
          {fieldFor("port", "Port")}
          {fieldFor("database", "Database")}
          {fieldFor("user", "Username")}
          {fieldFor("password", "Password", "password")}
        </div>
      );
    }
    if (subtype === "s3") {
      return (
        <div className="grid grid-cols-2 gap-3">
          {fieldFor("bucket", "Bucket")}
          {fieldFor("aws_access_key_id", "Access Key ID")}
          {fieldFor("aws_secret_access_key", "Secret Access Key", "password")}
          {fieldFor("region_name", "Region")}
        </div>
      );
    }
    return (
      <div className="grid grid-cols-2 gap-3">
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
      <div className="rounded-lg border border-slate-200 bg-white p-4 flex flex-wrap items-end gap-4">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
          <Upload size={16} /> Quick Upload
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Data type</label>
          <select
            value={uploadDataType}
            onChange={(e) => setUploadDataType(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          >
            {DATA_TYPES.map((dt) => (
              <option key={dt} value={dt}>{dt}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">File</label>
          <input
            type="file"
            accept=".csv,.parquet,.xlsx,.xls"
            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            className="text-sm"
          />
        </div>
        <button
          onClick={handleFileUpload}
          disabled={!uploadFile || uploading}
          className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors"
        >
          {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
          Upload
        </button>
      </div>

      {/* Saved connections grid */}
      <div className="mt-6 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-900">Saved Connections</h2>
        <button
          onClick={() => setShowAdd(true)}
          className="flex items-center gap-1.5 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 transition-colors"
        >
          <Plus size={14} /> Add Connection
        </button>
      </div>

      {loading ? (
        <div className="mt-4 flex justify-center py-12">
          <Loader2 size={24} className="animate-spin text-slate-400" />
        </div>
      ) : connectors.length === 0 ? (
        <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 py-12 text-center">
          <Database size={32} className="mx-auto text-slate-300" />
          <p className="mt-2 text-sm text-slate-500">No saved connections yet</p>
          <button
            onClick={() => setShowAdd(true)}
            className="mt-3 text-sm font-medium text-indigo-600 hover:text-indigo-700"
          >
            Add your first connection
          </button>
        </div>
      ) : (
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {connectors.map((c) => (
            <ConnectorCard
              key={c.id}
              connector={c}
              onTest={() => handleTest(c.id)}
              onDelete={() => handleDelete(c.id)}
              onFetch={() => { setShowFetch(c.id); setFetchQuery(""); }}
            />
          ))}
        </div>
      )}

      {/* Add Connection Modal */}
      {showAdd && (
        <div className="fixed inset-0 z-40 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/20" onClick={() => setShowAdd(false)} />
          <div className="relative w-full max-w-lg rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-900">New Connection</h3>
              <button onClick={() => setShowAdd(false)} className="text-slate-400 hover:text-slate-600">
                <X size={18} />
              </button>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My PostgreSQL"
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Type</label>
                <select
                  value={connType}
                  onChange={(e) => {
                    setConnType(e.target.value);
                    const t = CONNECTOR_TYPES.find((ct) => ct.type === e.target.value);
                    if (t) setSubtype(t.subtypes[0].key);
                    setConfigFields({});
                  }}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  {CONNECTOR_TYPES.map((t) => (
                    <option key={t.type} value={t.type}>{t.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Provider</label>
                <select
                  value={subtype}
                  onChange={(e) => { setSubtype(e.target.value); setConfigFields({}); }}
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                >
                  {selectedType?.subtypes.map((s) => (
                    <option key={s.key} value={s.key}>{s.label}</option>
                  ))}
                </select>
              </div>
            </div>

            {renderConfigForm()}

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowAdd(false)}
                className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={!name.trim() || saving}
                className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                Save Connection
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Fetch Data Modal */}
      {showFetch && (
        <div className="fixed inset-0 z-40 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/20" onClick={() => setShowFetch(null)} />
          <div className="relative w-full max-w-md rounded-xl bg-white shadow-xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-slate-900">Fetch Data</h3>
              <button onClick={() => setShowFetch(null)} className="text-slate-400 hover:text-slate-600">
                <X size={18} />
              </button>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">SQL Query or File Path</label>
              <textarea
                value={fetchQuery}
                onChange={(e) => setFetchQuery(e.target.value)}
                rows={3}
                placeholder="SELECT * FROM media_spend"
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Import as</label>
              <select
                value={fetchDataType}
                onChange={(e) => setFetchDataType(e.target.value)}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
              >
                {DATA_TYPES.map((dt) => (
                  <option key={dt} value={dt}>{dt}</option>
                ))}
              </select>
            </div>

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowFetch(null)}
                className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleFetchData}
                disabled={!fetchQuery.trim() || fetching}
                className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {fetching ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                Fetch
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
