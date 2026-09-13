import test from "node:test";
import assert from "node:assert/strict";
import { redact, stringifyArg, formatEntry, buildDiagnosticsReport } from "./diagnostics.js";

test("redact masks a Bearer token", () => {
  assert.equal(
    redact('fetch failed, Authorization: Bearer abc123.def456'),
    'fetch failed, Authorization: Bearer [REDACTED]'
  );
});

test("redact masks token/password/secret key-value pairs in either quoting style", () => {
  assert.equal(redact('{"token":"abc123"}'), '{"token":"[REDACTED]"}');
  assert.equal(redact("token=abc123&other=1"), "token=[REDACTED]&other=1");
  assert.equal(redact("password: hunter2"), "password: [REDACTED]");
});

test("redact leaves ordinary text untouched", () => {
  assert.equal(redact("Pyodide init failed: NetworkError"), "Pyodide init failed: NetworkError");
});

test("stringifyArg passes strings through unchanged", () => {
  assert.equal(stringifyArg("hello"), "hello");
});

test("stringifyArg renders an Error as its stack (or name: message if no stack)", () => {
  const err = new Error("boom");
  err.stack = undefined;
  assert.equal(stringifyArg(err), "Error: boom");
});

test("stringifyArg JSON-stringifies plain objects", () => {
  assert.equal(stringifyArg({ a: 1 }), '{"a":1}');
});

test("stringifyArg falls back to String() for values JSON can't handle", () => {
  const circular = {};
  circular.self = circular;
  assert.equal(stringifyArg(circular), String(circular));
});

test("formatEntry includes an ISO timestamp, padded level, and redacted message", () => {
  const line = formatEntry({ t: Date.UTC(2026, 0, 1, 0, 0, 0), level: "error", message: "token=abc123" });
  assert.match(line, /^\[2026-01-01T00:00:00\.000Z\] ERROR token=\[REDACTED\]$/);
});

test("formatEntry defaults to level 'log' when none is given", () => {
  const line = formatEntry({ t: 0, message: "hi" });
  assert.match(line, /^\[.+\] LOG {3}hi$/);
});

test("buildDiagnosticsReport includes header, env facts, storage size, and entry count", () => {
  const report = buildDiagnosticsReport({
    entries: [{ t: 0, level: "warn", message: "careful" }],
    startedAt: "2026-01-01T00:00:00.000Z",
    generatedAt: "2026-01-01T00:05:00.000Z",
    env: { userAgent: "TestAgent/1.0", viewport: "800x600" },
    storageBytes: 1234,
  });
  assert.match(report, /QuestionForge diagnostic log/);
  assert.match(report, /Generated:\s+2026-01-01T00:05:00\.000Z/);
  assert.match(report, /Session started: 2026-01-01T00:00:00\.000Z/);
  assert.match(report, /userAgent: TestAgent\/1\.0/);
  assert.match(report, /viewport: 800x600/);
  assert.match(report, /localStorage state size: 1234 bytes/);
  assert.match(report, /Log \(1 entry\)/);
  assert.match(report, /WARN {2}careful/);
});

test("buildDiagnosticsReport reports storage size as unavailable when null", () => {
  const report = buildDiagnosticsReport({ storageBytes: null });
  assert.match(report, /localStorage state size: unavailable/);
});

test("buildDiagnosticsReport pluralizes the entry count correctly", () => {
  assert.match(buildDiagnosticsReport({ entries: [] }), /Log \(0 entries\)/);
  assert.match(
    buildDiagnosticsReport({ entries: [{ t: 0, message: "a" }, { t: 0, message: "b" }] }),
    /Log \(2 entries\)/
  );
});

test("buildDiagnosticsReport defaults startedAt to 'unknown' when not provided", () => {
  assert.match(buildDiagnosticsReport({}), /Session started: unknown/);
});

test("buildDiagnosticsReport redacts secrets in env values", () => {
  const report = buildDiagnosticsReport({
    env: { url: "https://example.com/app?access_token=abc123&other=1" },
  });
  assert.match(report, /url: https:\/\/example\.com\/app\?access_token=\[REDACTED\]&other=1/);
});
