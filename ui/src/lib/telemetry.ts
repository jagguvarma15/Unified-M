type Primitive = string | number | boolean | null;
type EventProps = Record<string, Primitive>;

interface TelemetryEvent {
  event: string;
  timestamp: string;
  path?: string;
  props?: EventProps;
}

const TELEMETRY_ENDPOINT = "/api/v1/telemetry";
const MAX_BATCH_SIZE = 25;
const FLUSH_INTERVAL_MS = 6000;
const MAX_QUEUE_SIZE = 300;
const queue: TelemetryEvent[] = [];
let flushTimer: number | null = null;

function sanitizeProps(props?: Record<string, unknown>): EventProps | undefined {
  if (!props) return undefined;
  const out: EventProps = {};
  for (const [k, v] of Object.entries(props)) {
    if (v == null) {
      out[k] = null;
    } else if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
      out[k] = v;
    } else {
      out[k] = String(v);
    }
  }
  return out;
}

function scheduleFlush() {
  if (flushTimer != null) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    void flushTelemetry();
  }, FLUSH_INTERVAL_MS);
}

async function flushTelemetry() {
  if (queue.length === 0) return;

  const batch = queue.splice(0, MAX_BATCH_SIZE);
  try {
    await fetch(TELEMETRY_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: batch }),
      keepalive: true,
    });
  } catch {
    queue.unshift(...batch);
    if (queue.length > MAX_QUEUE_SIZE) queue.splice(0, queue.length - MAX_QUEUE_SIZE);
  }

  if (queue.length > 0) scheduleFlush();
}

export function trackEvent(event: string, props?: Record<string, unknown>) {
  const payload: TelemetryEvent = {
    event,
    timestamp: new Date().toISOString(),
    path: window.location.pathname,
    props: sanitizeProps(props),
  };
  queue.push(payload);
  if (queue.length > MAX_QUEUE_SIZE) queue.shift();
  if (queue.length >= MAX_BATCH_SIZE) {
    void flushTelemetry();
  } else {
    scheduleFlush();
  }
}

export function trackPageView(path: string) {
  trackEvent("page_view", { path });
}

export function trackApiLatency(path: string, durationMs: number, ok: boolean, status: number) {
  trackEvent("api_request", {
    endpoint: path,
    duration_ms: Math.round(durationMs),
    ok,
    status,
  });
}

window.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    void flushTelemetry();
  }
});
