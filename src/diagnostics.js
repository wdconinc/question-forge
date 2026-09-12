// diagnostics.js — formats an in-memory diagnostic log (console messages,
// uncaught errors, environment info) into a plain-text report a user can
// attach to a bug report.
//
// Deliberately dependency-free and DOM-free, like qti_export.js: the actual
// log capture (wrapping console.*, listening for "error" /
// "unhandledrejection") lives in a small bootstrap <script> in index.html,
// installed before any CDN library loads so it can catch their failures too.
// This module only turns the resulting plain-data entries into text, so it
// stays runnable — identically — in Node (for unit tests) and in the browser.
// index.html is responsible for reading window.__qfDiagnostics, gathering
// navigator/localStorage info, and saving the report to a file; nothing here
// ever sends the report anywhere on its own.

// Redacts common secret shapes (auth headers, api keys/tokens/passwords in
// "key: value" or "key=value" form) so a pasted log can't leak the AI
// connection token or similar. Best-effort, not a guarantee — anything logged
// as an opaque blob (e.g. an entire request object) may not match.
export function redact(str) {
  return String(str)
    .replace(/(Bearer\s+)\S+/gi, "$1[REDACTED]")
    .replace(/("?(?:api[_-]?key|token|password|secret)"?\s*[:=]\s*"?)[^\s"',}&]+/gi, "$1[REDACTED]");
}

// Turns one arg passed to console.log/warn/error/etc into a loggable string.
export function stringifyArg(arg) {
  if (typeof arg === "string") return arg;
  if (arg instanceof Error) return arg.stack || `${arg.name}: ${arg.message}`;
  try { return JSON.stringify(arg); }
  catch { return String(arg); }
}

export function formatEntry(entry) {
  const ts = new Date(entry.t).toISOString();
  const level = String(entry.level || "log").toUpperCase().padEnd(5);
  return `[${ts}] ${level} ${redact(entry.message)}`;
}

// Builds the full report text. All inputs are plain data — no window/
// navigator/localStorage access here — so callers (index.html) gather that
// themselves and pass it in.
//
//   entries:     [{ t: <ms epoch>, level: "log"|"warn"|"error"|..., message }]
//   startedAt:   ISO timestamp string, or null if unknown
//   generatedAt: ISO timestamp string for "now"
//   env:         plain object of environment facts (userAgent, viewport, ...)
//   storageBytes: size of the persisted app state in bytes, or null
export function buildDiagnosticsReport({
  entries = [],
  startedAt = null,
  generatedAt = new Date().toISOString(),
  env = {},
  storageBytes = null,
} = {}) {
  const lines = [];
  lines.push("QuestionForge diagnostic log");
  lines.push(`Generated:      ${generatedAt}`);
  lines.push(`Session started: ${startedAt || "unknown"}`);
  for (const [key, value] of Object.entries(env)) {
    lines.push(`${key}: ${redact(value)}`);
  }
  lines.push(
    `localStorage state size: ${storageBytes === null ? "unavailable" : `${storageBytes} bytes`}`
  );
  lines.push("");
  lines.push(`── Log (${entries.length} ${entries.length === 1 ? "entry" : "entries"}) ${"─".repeat(40)}`);
  for (const entry of entries) lines.push(formatEntry(entry));

  return lines.join("\n");
}
