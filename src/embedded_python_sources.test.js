import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// Pyodide executes the <script type="text/plain" id="src-*-py"> blocks embedded
// in index.html, not the python/*.py files on disk — those files are kept only
// as a readable mirror (see INIT_PY_SOURCE / EXAM_CORE_SOURCE / RENDER_PY_SOURCE
// in index.html). Nothing enforces that the mirror stays in sync, so this test
// catches drift the way server/test_qvalidate.py::TestRuntimeParity does for
// server/questions.py vs python/__init__.py.

const REPO_ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const INDEX_HTML = readFileSync(path.join(REPO_ROOT, "index.html"), "utf8");

const MIRRORS = [
  ["src-init-py", "python/__init__.py"],
  ["src-exam-core-py", "python/exam_core.py"],
  ["src-render-py", "python/render.py"],
];

function extractScriptBlock(id) {
  const re = new RegExp(`<script type="text/plain" id="${id}">\\n([\\s\\S]*?)</script>`);
  const match = INDEX_HTML.match(re);
  assert.ok(match, `could not find <script id="${id}"> block in index.html`);
  return match[1];
}

for (const [id, relPath] of MIRRORS) {
  test(`index.html #${id} stays byte-identical to ${relPath}`, () => {
    const embedded = extractScriptBlock(id);
    const onDisk = readFileSync(path.join(REPO_ROOT, relPath), "utf8");
    assert.equal(
      onDisk,
      embedded,
      `${relPath} has drifted from index.html's #${id} block — the embedded copy is what Pyodide actually runs, so treat it as the source of truth and copy it back into ${relPath}`
    );
  });
}
