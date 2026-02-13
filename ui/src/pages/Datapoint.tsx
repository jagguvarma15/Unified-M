import { useState } from "react";
import {
  Database,
  Cloud,
  Upload,
  TestTube,
  CheckCircle2,
  XCircle,
  Loader2,
  FileText,
} from "lucide-react";

interface ConnectionTestResult {
  status: string;
  connected: boolean;
  message: string;
}

interface FetchResult {
  status: string;
  rows: number;
  columns: string[];
  path: string;
  data_type: string;
}

const DATA_TYPES = [
  { key: "media_spend", label: "Media Spend", required: true },
  { key: "outcomes", label: "Outcomes", required: true },
  { key: "controls", label: "Control Variables", required: false },
  { key: "incrementality_tests", label: "Incrementality Tests", required: false },
  { key: "attribution", label: "Attribution Data", required: false },
  { key: "custom", label: "Custom (name below)", required: false },
] as const;

/** Custom name allowed: letter then letters/numbers/underscores, max 64 chars */
const CUSTOM_DATA_TYPE_REGEX = /^[a-zA-Z][a-zA-Z0-9_]{0,63}$/;
function isValidCustomName(name: string): boolean {
  return name.trim() !== "" && CUSTOM_DATA_TYPE_REGEX.test(name.trim());
}

export default function Datapoint() {
  const [connectionType, setConnectionType] = useState<"database" | "cloud" | "file">("file");
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);
  const [fetchResult, setFetchResult] = useState<FetchResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Database form state
  const [dbType, setDbType] = useState("postgresql");
  const [dbHost, setDbHost] = useState("");
  const [dbPort, setDbPort] = useState("");
  const [dbDatabase, setDbDatabase] = useState("");
  const [dbUser, setDbUser] = useState("");
  const [dbPassword, setDbPassword] = useState("");
  const [dbQuery, setDbQuery] = useState("");

  // Cloud form state
  const [cloudType, setCloudType] = useState("s3");
  const [cloudBucket, setCloudBucket] = useState("");
  const [cloudAccessKey, setCloudAccessKey] = useState("");
  const [cloudSecretKey, setCloudSecretKey] = useState("");
  const [cloudRegion, setCloudRegion] = useState("us-east-1");
  const [cloudPath, setCloudPath] = useState("");

  // File upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadDataType, setUploadDataType] = useState("media_spend");
  const [customDataTypeName, setCustomDataTypeName] = useState("");

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);

    try {
      let config: Record<string, unknown> = {};

      if (connectionType === "database") {
        config = {
          db_type: dbType,
          host: dbHost,
          port: parseInt(dbPort) || (dbType === "postgresql" ? 5432 : 3306),
          database: dbDatabase,
          user: dbUser,
          password: dbPassword,
        };
      } else if (connectionType === "cloud") {
        if (cloudType === "s3") {
          config = {
            cloud_type: "s3",
            bucket: cloudBucket,
            aws_access_key_id: cloudAccessKey,
            aws_secret_access_key: cloudSecretKey,
            region_name: cloudRegion,
          };
        } else if (cloudType === "azure") {
          config = {
            cloud_type: "azure",
            account_name: cloudBucket,
            container_name: cloudPath.split("/")[0] || "",
            account_key: cloudAccessKey,
          };
        }
      }

      const formData = new FormData();
      formData.append("connection_type", connectionType);
      formData.append("connection_config", JSON.stringify(config));

      const res = await fetch("/api/v1/datapoint/test", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      setTestResult(result);
    } catch (err) {
      setTestResult({
        status: "error",
        connected: false,
        message: err instanceof Error ? err.message : "Connection test failed",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleFetchData = async (dataType: string) => {
    setFetching(true);
    setFetchResult(null);

    try {
      let config: Record<string, unknown> = {};
      let queryOrPath = "";

      if (connectionType === "database") {
        config = {
          db_type: dbType,
          host: dbHost,
          port: parseInt(dbPort) || (dbType === "postgresql" ? 5432 : 3306),
          database: dbDatabase,
          user: dbUser,
          password: dbPassword,
        };
        queryOrPath = dbQuery;
      } else if (connectionType === "cloud") {
        if (cloudType === "s3") {
          config = {
            cloud_type: "s3",
            bucket: cloudBucket,
            aws_access_key_id: cloudAccessKey,
            aws_secret_access_key: cloudSecretKey,
            region_name: cloudRegion,
          };
        } else if (cloudType === "azure") {
          config = {
            cloud_type: "azure",
            account_name: cloudBucket,
            container_name: cloudPath.split("/")[0] || "",
            account_key: cloudAccessKey,
          };
        }
        queryOrPath = cloudPath;
      }

      const formData = new FormData();
      formData.append("connection_type", connectionType);
      formData.append("connection_config", JSON.stringify(config));
      formData.append("query_or_path", queryOrPath);
      formData.append("data_type", dataType);

      const res = await fetch("/api/v1/datapoint/fetch", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Fetch failed");
      }

      const result = await res.json();
      setFetchResult(result);
    } catch (err) {
      setFetchResult({
        status: "error",
        rows: 0,
        columns: [],
        path: "",
        data_type: dataType,
      });
      alert(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setFetching(false);
    }
  };

  const resolveDataType = (): string => {
    if (uploadDataType === "custom") {
      return customDataTypeName.trim();
    }
    return uploadDataType;
  };

  const handleFileUpload = async () => {
    if (!uploadFile) return;
    const dataType = resolveDataType();
    if (uploadDataType === "custom" && !isValidCustomName(customDataTypeName)) {
      alert("Custom data type name must start with a letter and use only letters, numbers, and underscores (max 64 chars).");
      return;
    }

    setUploading(true);
    setFetchResult(null);

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("data_type", dataType);

      const res = await fetch("/api/v1/datapoint/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Upload failed");
      }

      const result = await res.json();
      setFetchResult(result);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to upload file");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-900">Connect to a Datapoint</h1>
      <p className="text-sm text-slate-500 mt-1">
        Connect to databases, cloud storage, or upload files to import data
      </p>

      {/* Connection type selector */}
      <div className="mt-6 flex gap-2 bg-slate-100 rounded-lg p-1 w-fit">
        {[
          { key: "file" as const, label: "File Upload", icon: Upload },
          { key: "database" as const, label: "Database", icon: Database },
          { key: "cloud" as const, label: "Cloud Storage", icon: Cloud },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setConnectionType(key)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              connectionType === key
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* File Upload Form */}
      {connectionType === "file" && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Upload File</h2>
          <p className="text-xs text-slate-500 mb-4">
            Supported formats: CSV, Parquet, Excel (.xlsx, .xls)
          </p>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Data Type
              </label>
              <select
                value={uploadDataType}
                onChange={(e) => setUploadDataType(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                {DATA_TYPES.map((dt) => (
                  <option key={dt.key} value={dt.key}>
                    {dt.label} {dt.required && "(Required)"}
                  </option>
                ))}
              </select>
            </div>

            {uploadDataType === "custom" && (
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Custom Data Type Name
                </label>
                <input
                  type="text"
                  value={customDataTypeName}
                  onChange={(e) => setCustomDataTypeName(e.target.value)}
                  placeholder="e.g. promo_flags, weather"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Letters, numbers, underscores only; must start with a letter (max 64 chars).
                </p>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                File
              </label>
              <input
                type="file"
                accept=".csv,.parquet,.xlsx,.xls"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={handleFileUpload}
              disabled={
                !uploadFile ||
                uploading ||
                (uploadDataType === "custom" && !isValidCustomName(customDataTypeName))
              }
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={16} />
                  Upload File
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Database Form */}
      {connectionType === "database" && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Database Connection</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Database Type
              </label>
              <select
                value={dbType}
                onChange={(e) => setDbType(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="sqlserver">SQL Server</option>
                <option value="sqlite">SQLite</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Host / Server
              </label>
              <input
                type="text"
                value={dbHost}
                onChange={(e) => setDbHost(e.target.value)}
                placeholder="localhost"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Port
              </label>
              <input
                type="number"
                value={dbPort}
                onChange={(e) => setDbPort(e.target.value)}
                placeholder={dbType === "postgresql" ? "5432" : "3306"}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Database Name
              </label>
              <input
                type="text"
                value={dbDatabase}
                onChange={(e) => setDbDatabase(e.target.value)}
                placeholder="database_name"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Username
              </label>
              <input
                type="text"
                value={dbUser}
                onChange={(e) => setDbUser(e.target.value)}
                placeholder="username"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Password
              </label>
              <input
                type="password"
                value={dbPassword}
                onChange={(e) => setDbPassword(e.target.value)}
                placeholder="password"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="mt-4">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              SQL Query
            </label>
            <textarea
              value={dbQuery}
              onChange={(e) => setDbQuery(e.target.value)}
              placeholder="SELECT * FROM media_spend WHERE date >= '2024-01-01'"
              rows={4}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          <div className="flex gap-2 mt-4">
            <button
              onClick={handleTestConnection}
              disabled={testing}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 disabled:opacity-50 transition-colors"
            >
              {testing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Testing...
                </>
              ) : (
                <>
                  <TestTube size={16} />
                  Test Connection
                </>
              )}
            </button>
          </div>

          {testResult && (
            <div
              className={`mt-4 p-3 rounded-lg flex items-center gap-2 ${
                testResult.connected
                  ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                  : "bg-red-50 text-red-800 border border-red-200"
              }`}
            >
              {testResult.connected ? (
                <CheckCircle2 size={16} />
              ) : (
                <XCircle size={16} />
              )}
              <span className="text-sm">{testResult.message}</span>
            </div>
          )}

          {testResult?.connected && (
            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Select Data Type to Import
                </label>
                <div className="flex flex-wrap gap-2">
                  {DATA_TYPES.filter((dt) => dt.key !== "custom").map((dt) => (
                    <button
                      key={dt.key}
                      onClick={() => handleFetchData(dt.key)}
                      disabled={fetching || !dbQuery}
                      className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {fetching ? "Fetching..." : `Import as ${dt.label}`}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Or import as custom (name it)
                </label>
                <div className="flex flex-wrap items-center gap-2">
                  <input
                    type="text"
                    value={customDataTypeName}
                    onChange={(e) => setCustomDataTypeName(e.target.value)}
                    placeholder="e.g. promo_flags, weather"
                    className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono w-48 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                  <button
                    onClick={() => {
                      if (!isValidCustomName(customDataTypeName)) {
                        alert("Custom name: start with a letter, use only letters, numbers, underscores (max 64 chars).");
                        return;
                      }
                      handleFetchData(customDataTypeName.trim());
                    }}
                    disabled={fetching || !dbQuery || !customDataTypeName.trim()}
                    className="px-3 py-1.5 bg-slate-600 text-white rounded-lg text-sm font-medium hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {fetching ? "Fetching..." : "Import as custom"}
                  </button>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  Letters, numbers, underscores; must start with a letter.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cloud Storage Form */}
      {connectionType === "cloud" && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Cloud Storage Connection</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Cloud Provider
              </label>
              <select
                value={cloudType}
                onChange={(e) => setCloudType(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              >
                <option value="s3">AWS S3</option>
                <option value="azure">Azure Blob Storage</option>
              </select>
            </div>

            {cloudType === "s3" && (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    S3 Bucket Name
                  </label>
                  <input
                    type="text"
                    value={cloudBucket}
                    onChange={(e) => setCloudBucket(e.target.value)}
                    placeholder="my-bucket"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      AWS Access Key ID
                    </label>
                    <input
                      type="text"
                      value={cloudAccessKey}
                      onChange={(e) => setCloudAccessKey(e.target.value)}
                      placeholder="AKIA..."
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      AWS Secret Access Key
                    </label>
                    <input
                      type="password"
                      value={cloudSecretKey}
                      onChange={(e) => setCloudSecretKey(e.target.value)}
                      placeholder="secret"
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Region
                  </label>
                  <input
                    type="text"
                    value={cloudRegion}
                    onChange={(e) => setCloudRegion(e.target.value)}
                    placeholder="us-east-1"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    File Path in Bucket
                  </label>
                  <input
                    type="text"
                    value={cloudPath}
                    onChange={(e) => setCloudPath(e.target.value)}
                    placeholder="data/media_spend.parquet"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>
              </>
            )}

            {cloudType === "azure" && (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Storage Account Name
                  </label>
                  <input
                    type="text"
                    value={cloudBucket}
                    onChange={(e) => setCloudBucket(e.target.value)}
                    placeholder="mystorageaccount"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Account Key
                  </label>
                  <input
                    type="password"
                    value={cloudAccessKey}
                    onChange={(e) => setCloudAccessKey(e.target.value)}
                    placeholder="account key"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Container/Blob Path
                  </label>
                  <input
                    type="text"
                    value={cloudPath}
                    onChange={(e) => setCloudPath(e.target.value)}
                    placeholder="container/data/media_spend.parquet"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>
              </>
            )}

            <button
              onClick={handleTestConnection}
              disabled={testing}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 disabled:opacity-50 transition-colors"
            >
              {testing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Testing...
                </>
              ) : (
                <>
                  <TestTube size={16} />
                  Test Connection
                </>
              )}
            </button>

            {testResult && (
              <div
                className={`p-3 rounded-lg flex items-center gap-2 ${
                  testResult.connected
                    ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                    : "bg-red-50 text-red-800 border border-red-200"
                }`}
              >
                {testResult.connected ? (
                  <CheckCircle2 size={16} />
                ) : (
                  <XCircle size={16} />
                )}
                <span className="text-sm">{testResult.message}</span>
              </div>
            )}

            {testResult?.connected && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Select Data Type to Import
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {DATA_TYPES.filter((dt) => dt.key !== "custom").map((dt) => (
                      <button
                        key={dt.key}
                        onClick={() => handleFetchData(dt.key)}
                        disabled={fetching || !cloudPath}
                        className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        {fetching ? "Fetching..." : `Import as ${dt.label}`}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Or import as custom (name it)
                  </label>
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      type="text"
                      value={customDataTypeName}
                      onChange={(e) => setCustomDataTypeName(e.target.value)}
                      placeholder="e.g. promo_flags, weather"
                      className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm font-mono w-48 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        if (!isValidCustomName(customDataTypeName)) {
                          alert("Custom name: start with a letter, use only letters, numbers, underscores (max 64 chars).");
                          return;
                        }
                        handleFetchData(customDataTypeName.trim());
                      }}
                      disabled={fetching || !cloudPath || !customDataTypeName.trim()}
                      className="px-3 py-1.5 bg-slate-600 text-white rounded-lg text-sm font-medium hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {fetching ? "Fetching..." : "Import as custom"}
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    Letters, numbers, underscores; must start with a letter.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Fetch Result */}
      {fetchResult && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200/60 mt-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
            <FileText size={16} />
            Import Result
          </h2>
          <div
            className={`p-4 rounded-lg ${
              fetchResult.status === "success"
                ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                : "bg-red-50 text-red-800 border border-red-200"
            }`}
          >
            {fetchResult.status === "success" ? (
              <div className="space-y-2">
                <p className="font-medium">Data imported successfully!</p>
                <p className="text-sm">
                  Rows: {fetchResult.rows.toLocaleString()} | Columns: {fetchResult.columns.length}
                </p>
                <p className="text-sm">Data Type: {fetchResult.data_type}</p>
                <p className="text-xs font-mono">{fetchResult.path}</p>
              </div>
            ) : (
              <p>Import failed. Please check the error message above.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
