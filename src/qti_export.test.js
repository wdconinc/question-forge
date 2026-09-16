import test from "node:test";
import assert from "node:assert/strict";
import {
  sanitizeIdentifier,
  versionedQid,
  versionedTitle,
  groupVersionsByQid,
  buildItemNode,
  groupItemsIntoSections,
  buildObjectBankXml,
  buildManifestXml,
  buildQtiPackage,
  serialize,
  el,
} from "./qti_export.js";

// Minimal dependency-free well-formedness check: every opening tag must have a
// matching closing tag (or be self-closing), correctly nested. Node has no
// DOMParser, so this stands in for "the output XML parses cleanly".
function assertWellFormedXmlFragment(xml) {
  const tagRe = /<(\/?)([A-Za-z][\w:.-]*)([^>]*?)(\/?)>/g;
  const stack = [];
  let m;
  while ((m = tagRe.exec(xml)) !== null) {
    const [, closing, name, , selfClosing] = m;
    if (selfClosing === "/") continue;
    if (closing === "/") {
      const top = stack.pop();
      assert.equal(top, name, `mismatched closing tag </${name}>`);
    } else {
      stack.push(name);
    }
  }
  assert.equal(stack.length, 0, `unclosed tags remain: ${stack.join(",")}`);
}

const stubLatexToMathML = async (tex) =>
  `<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>${tex}</mtext></math>`;

test("sanitizeIdentifier keeps already-safe ids unchanged", () => {
  assert.equal(sanitizeIdentifier("q01_units"), "q01_units");
});

test("sanitizeIdentifier prefixes ids that start with a digit", () => {
  assert.equal(sanitizeIdentifier("1abc"), "q_1abc");
});

test("sanitizeIdentifier falls back to q_item for an empty, null, or undefined id", () => {
  // Invalid characters are replaced with "_", not removed, so a non-empty
  // input never sanitizes down to an empty string (e.g. "!!!" -> "___", which
  // already starts with "_" and needs no further fallback) — only a raw id
  // that is itself empty/nullish can hit this fallback.
  assert.equal(sanitizeIdentifier(""), "q_item");
  assert.equal(sanitizeIdentifier(null), "q_item");
  assert.equal(sanitizeIdentifier(undefined), "q_item");
});

test("sanitizeIdentifier disambiguates collisions", () => {
  // "!" and "@" both collapse to "_", so these two distinct raw ids collide
  // on the sanitized string "q_1" and must be disambiguated.
  const used = new Set();
  assert.equal(sanitizeIdentifier("q!1", used), "q_1");
  assert.equal(sanitizeIdentifier("q@1", used), "q_1_2");
});

test("buildItemNode produces a well-formed <item> with cc_profile metadata, 5 response_labels, and the correct varequal", async () => {
  const question = {
    qid: "q02_kinematics_1d",
    title: "q02_kinematics_1d",
    question: "A car starts from rest and accelerates at $2.0$ m/s².",
    choices: ["10 m", "20 m", "30 m", "40 m", "50 m"],
    answer: "c",
  };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assert.equal(id, "q02_kinematics_1d");
  assertWellFormedXmlFragment(xml);
  assert.match(xml, /ident="q02_kinematics_1d"/);
  assert.match(xml, /<fieldlabel>cc_profile<\/fieldlabel><fieldentry>cc\.multiple_choice\.v0p1<\/fieldentry>/);
  for (const letter of ["A", "B", "C", "D", "E"]) {
    assert.match(xml, new RegExp(`ident="${letter}"`));
  }
  assert.match(xml, /<varequal respident="response1">C<\/varequal>/);
  assert.match(xml, /<setvar action="Set" varname="SCORE">100<\/setvar>/);
});

test("buildItemNode escapes special characters in plain text", async () => {
  const question = {
    qid: "q_special",
    question: "If v < 5 m/s & a > 0, find x.",
    choices: ["A & B", "C", "D", "E", "F"],
    answer: "a",
  };
  const { node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assertWellFormedXmlFragment(xml);
  assert.ok(!xml.includes("v < 5"), "raw '<' must be escaped");
  assert.match(xml, /v &lt; 5/);
  assert.match(xml, /A &amp; B/);
});

test("buildItemNode converts math runs via latexToMathML and splices them verbatim as inline HTML", async () => {
  const question = { qid: "q_math", question: "Find $g$ given the table.", choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const { node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assertWellFormedXmlFragment(xml);
  // The <math> markup is HTML embedded inside mattext's text/html content, so
  // its tag angle-brackets appear escaped (&lt;math...) in the outer QTI XML —
  // matching the real Canvas-generated example this was verified against.
  // Attribute quotes within XML *text* content (unlike attribute values)
  // don't need escaping, so the xmlns="..." keeps its literal double-quotes.
  assert.match(xml, /&lt;math xmlns="http:\/\/www\.w3\.org\/1998\/Math\/MathML"&gt;&lt;mtext&gt;g&lt;\/mtext&gt;&lt;\/math&gt;/);
});

test("buildItemNode falls back to escaped text and records a failure when MathML conversion fails", async () => {
  const failing = async () => { throw new Error("boom"); };
  const question = { qid: "q_bad_math", question: String.raw`Compute $\frac{1}{2}$.`, choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const failures = [];
  const { node } = await buildItemNode(question, { latexToMathML: failing, usedIds: new Set(), failures });
  assertWellFormedXmlFragment(serialize(node));
  assert.equal(failures.length, 1);
  assert.equal(failures[0].qid, "q_bad_math");
});

test("buildItemNode rejects an invalid answer letter", async () => {
  const question = { qid: "q_bad_answer", question: "x", choices: ["1", "2", "3", "4", "5"], answer: "z" };
  await assert.rejects(() =>
    buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] })
  );
});

test("buildItemNode splices an optional svg figure into the stem, wrapped in a max-width div", async () => {
  const question = {
    qid: "q_diagram",
    question: "A block on an incline.",
    svg: '<svg viewBox="0 0 10 10" width="100" height="50"><rect width="10" height="10"/></svg>',
    choices: ["1", "2", "3", "4", "5"],
    answer: "a",
  };
  const { node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assertWellFormedXmlFragment(xml);
  // Same escaping story as the MathML test above: the <div>/<svg> markup is
  // HTML embedded inside mattext's text/html content, escaped exactly once.
  assert.match(xml, /&lt;div style="max-width:100%;overflow-x:auto;text-align:center"&gt;&lt;svg viewBox="0 0 10 10" width="100" height="50"&gt;&lt;rect width="10" height="10"\/&gt;&lt;\/svg&gt;&lt;\/div&gt;/);
});

test("buildItemNode omits the figure div entirely when the question has no svg", async () => {
  const question = { qid: "q_no_diagram", question: "x", choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const { node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assert.ok(!xml.includes("max-width:100%;overflow-x:auto"), "no svg means no figure wrapper");
});

test("buildItemNode (numerical) produces a well-formed <item> with response_num/render_fib, cc.fib.v0p1, and a vargte/varlte tolerance range", async () => {
  const question = {
    qid: "q_numeric",
    title: "q_numeric",
    type: "numerical",
    question: "What is the final speed?",
    answer: 12.3,
    tolerance: 0.5,
    unit: "m/s",
  };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assert.equal(id, "q_numeric");
  assertWellFormedXmlFragment(xml);
  assert.match(xml, /<fieldlabel>cc_profile<\/fieldlabel><fieldentry>cc\.fib\.v0p1<\/fieldentry>/);
  assert.match(xml, /<response_num ident="response1" rcardinality="Single" numtype="Decimal">/);
  assert.match(xml, /<render_fib fibtype="Decimal"/);
  assert.match(xml, /<vargte respident="response1">11\.8<\/vargte>/);
  assert.match(xml, /<varlte respident="response1">12\.8<\/varlte>/);
  assert.match(xml, /<setvar action="Set" varname="SCORE">100<\/setvar>/);
  assert.ok(!xml.includes("response_label"), "numerical items must not have MC response_labels");
  assert.ok(!xml.includes("render_choice"), "numerical items must not use render_choice");
});

test("buildItemNode (numerical) splices an optional svg figure into the stem too", async () => {
  const question = {
    qid: "q_numeric_diagram",
    type: "numerical",
    question: "What is the final speed?",
    svg: '<svg viewBox="0 0 10 10" width="100" height="50"></svg>',
    answer: 12.3,
    tolerance: 0.5,
    unit: "m/s",
  };
  const { node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  const xml = serialize(node);
  assertWellFormedXmlFragment(xml);
  assert.match(xml, /&lt;div style="max-width:100%;overflow-x:auto;text-align:center"&gt;&lt;svg viewBox="0 0 10 10" width="100" height="50"&gt;&lt;\/svg&gt;&lt;\/div&gt;/);
});

test("buildItemNode (numerical) rejects a non-finite answer or tolerance", async () => {
  const base = { qid: "q_bad_numeric", type: "numerical", question: "x" };
  await assert.rejects(() =>
    buildItemNode({ ...base, answer: "not-a-number", tolerance: 0.5 }, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] })
  );
  await assert.rejects(() =>
    buildItemNode({ ...base, answer: 1, tolerance: "not-a-number" }, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] })
  );
});

test("buildItemNode (numerical) rejects a negative tolerance", async () => {
  const question = { qid: "q_negative_tolerance", type: "numerical", question: "x", answer: 10, tolerance: -0.5 };
  await assert.rejects(() =>
    buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] })
  );
});

test("buildObjectBankXml wraps every item directly under one <objectbank> (no <section> nesting)", async () => {
  const q1 = await buildItemNode({ qid: "q01", question: "A", choices: ["1", "2", "3", "4", "5"], answer: "a" }, {
    latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [],
  });
  const q2 = await buildItemNode({ qid: "q02", question: "B", choices: ["1", "2", "3", "4", "5"], answer: "b" }, {
    latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [],
  });
  const { id, xml } = buildObjectBankXml([q1, q2], { bankId: "my_bank" });
  assertWellFormedXmlFragment(xml);
  assert.equal(id, "my_bank");
  assert.match(xml, /<objectbank ident="my_bank">/);
  assert.ok(!xml.includes("<section"), "question banks must not nest items in a <section>");
  assert.match(xml, /<item ident="q01"/);
  assert.match(xml, /<item ident="q02"/);
});

test("buildManifestXml declares a question-bank resource with empty organizations", () => {
  const xml = buildManifestXml("my_bank", { manifestId: "my_export", bankHref: "questestinterop.xml" });
  assertWellFormedXmlFragment(xml);
  // Confirmed against the 1EdTech CC 1.1 spec directly: a question bank must
  // NOT be referenced from <organizations> — it must stay empty.
  assert.match(xml, /<organizations\/>/);
  assert.match(xml, /type="imsqti_xmlv1p2\/imscc_xmlv1p1\/question-bank"/);
  assert.match(xml, /href="questestinterop\.xml"/);
});

test("buildQtiPackage zips a manifest plus one combined questestinterop.xml via an injected zip factory", async () => {
  // Minimal fake JSZip so this test needs no real dependency and no browser JSZip.
  class FakeZip {
    constructor() { this.files = {}; }
    file(path, content) { this.files[path] = content; return this; }
    async generateAsync() { return { __files: this.files }; }
  }

  const questions = [
    { qid: "q01", question: "What is $g$?", choices: ["a", "b", "c", "d", "e"], answer: "a" },
    { qid: "q02", question: "Plain text, no math.", choices: ["a", "b", "c", "d", "e"], answer: "b" },
  ];
  const { blob, failures } = await buildQtiPackage(questions, { latexToMathML: stubLatexToMathML, zipFactory: FakeZip });
  assert.equal(failures.length, 0);
  assert.ok("imsmanifest.xml" in blob.__files);
  assert.ok("questestinterop.xml" in blob.__files);
  assertWellFormedXmlFragment(blob.__files["imsmanifest.xml"]);
  assertWellFormedXmlFragment(blob.__files["questestinterop.xml"]);
  assert.match(blob.__files["questestinterop.xml"], /<item ident="q01"/);
  assert.match(blob.__files["questestinterop.xml"], /<item ident="q02"/);
});

test("buildQtiPackage disambiguates two qids that sanitize to the same identifier", async () => {
  class FakeZip {
    constructor() { this.files = {}; }
    file(path, content) { this.files[path] = content; return this; }
    async generateAsync() { return { __files: this.files }; }
  }
  const questions = [
    { qid: "q-1", question: "A", choices: ["a", "b", "c", "d", "e"], answer: "a" },
    { qid: "q_1", question: "B", choices: ["a", "b", "c", "d", "e"], answer: "a" },
  ];
  const { blob } = await buildQtiPackage(questions, { latexToMathML: stubLatexToMathML, zipFactory: FakeZip });
  const idMatches = [...blob.__files["questestinterop.xml"].matchAll(/<item ident="([^"]+)"/g)].map(m => m[1]);
  assert.equal(idMatches.length, 2);
  assert.equal(new Set(idMatches).size, 2);
});

// ── Seed-versioned export (one item per question per paper seed) ─────────────

test("versionedQid/versionedTitle suffix only when a seed is present", () => {
  assert.equal(versionedQid("q01_units", 42), "q01_units__seed42");
  assert.equal(versionedTitle("Units", 42), "Units (seed 42)");
  // No seed means "not one version among several" — a single-paper export must
  // keep the plain qid and title it had before versioning existed.
  assert.equal(versionedQid("q01_units", undefined), "q01_units");
  assert.equal(versionedTitle("Units", undefined), "Units");
  assert.equal(versionedQid("q01_units", null), "q01_units");
  assert.equal(versionedTitle("Units", null), "Units");
  // Seed 0 is a real seed, not an absent one.
  assert.equal(versionedQid("q01_units", 0), "q01_units__seed0");
  assert.equal(versionedTitle("Units", 0), "Units (seed 0)");
});

test("a missing qid or title falls back the same way whether or not a seed is present", () => {
  // Interpolating a nullish base straight into the suffix template used to bake
  // the literal text "null"/"undefined" into the identifier, so a seeded export
  // skipped the fallback an unseeded one applies. Missing is missing either way.
  for (const missing of [null, undefined, ""]) {
    assert.equal(sanitizeIdentifier(versionedQid(missing, undefined)), "q_item");
    assert.equal(sanitizeIdentifier(versionedQid(missing, 42)), "q_item__seed42");
    assert.ok(!versionedQid(missing, 42).includes("undefined"));
    assert.ok(!versionedQid(missing, 42).includes("null"));
    // A seed marker with nothing to prefix it should not carry a dangling space.
    assert.equal(versionedTitle(missing, 42), "(seed 42)");
    assert.equal(versionedTitle(missing, undefined), "");
  }
});

test("an item built from a question with no qid still gets a usable ident", async () => {
  const question = { question: "x", seed: 42, choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  assert.equal(id, "q_item__seed42");
  assert.ok(!serialize(node).includes("undefined"), "no literal 'undefined' may reach the XML");
});

test("groupVersionsByQid gathers every version of a question, keeping first-seen qid order", () => {
  // Papers are materialized one at a time, so versions arrive paper-major:
  // all of paper A, then all of paper B. The bank needs them question-major.
  const input = [
    { qid: "q01", seed: 42 }, { qid: "q02", seed: 42 },
    { qid: "q01", seed: 137 }, { qid: "q02", seed: 137 },
  ];
  assert.deepEqual(groupVersionsByQid(input), [
    { qid: "q01", seed: 42 }, { qid: "q01", seed: 137 },
    { qid: "q02", seed: 42 }, { qid: "q02", seed: 137 },
  ]);
});

test("groupVersionsByQid leaves a single-version list in its original order", () => {
  const input = [{ qid: "q03" }, { qid: "q01" }, { qid: "q02" }];
  assert.deepEqual(groupVersionsByQid(input).map(q => q.qid), ["q03", "q01", "q02"]);
});

test("buildItemNode stamps the seed on both the ident and the library title", async () => {
  const question = {
    qid: "q02_kinematics_1d",
    title: "Kinematics 1D",
    seed: 137,
    question: "A car accelerates.",
    choices: ["1", "2", "3", "4", "5"],
    answer: "a",
  };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  assert.equal(id, "q02_kinematics_1d__seed137");
  assert.match(serialize(node), /<item ident="q02_kinematics_1d__seed137" title="Kinematics 1D \(seed 137\)"/);
});

test("buildItemNode (numerical) stamps the seed too", async () => {
  const question = { qid: "q_numeric", title: "Final speed", seed: 271, type: "numerical", question: "x", answer: 12.3, tolerance: 0.5 };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  assert.equal(id, "q_numeric__seed271");
  assert.match(serialize(node), /title="Final speed \(seed 271\)"/);
});

test("buildItemNode leaves a seedless question's ident and title exactly as before", async () => {
  const question = { qid: "q01_units", title: "Units", question: "x", choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const { id, node } = await buildItemNode(question, { latexToMathML: stubLatexToMathML, usedIds: new Set(), failures: [] });
  assert.equal(id, "q01_units");
  assert.match(serialize(node), /<item ident="q01_units" title="Units"/);
  assert.ok(!serialize(node).includes("seed"), "a single-paper export must not mention a seed");
});

// ── Grouping randomized versions into a named <section> ──────────────────────

test("groupItemsIntoSections wraps a multi-item, all-seeded qid group in one named <section>", () => {
  const items = [
    { id: "q01__seed42", node: el("item", { ident: "q01__seed42" }), qid: "q01", title: "Units", seed: 42 },
    { id: "q01__seed137", node: el("item", { ident: "q01__seed137" }), qid: "q01", title: "Units", seed: 137 },
  ];
  const children = groupItemsIntoSections(items, new Set());
  assert.equal(children.length, 1);
  assert.equal(children[0].node.tag, "section");
  assert.equal(children[0].node.attrs.title, "Units");
  assert.equal(children[0].node.children.length, 2);
  assert.deepEqual(children[0].node.children.map(c => c.attrs.ident), ["q01__seed42", "q01__seed137"]);
});

test("groupItemsIntoSections leaves a single-version qid as a bare <item> (no section)", () => {
  const items = [{ id: "q01", node: el("item", { ident: "q01" }), qid: "q01", title: "Units", seed: undefined }];
  const children = groupItemsIntoSections(items, new Set());
  assert.equal(children.length, 1);
  assert.equal(children[0].node.tag, "item");
});

test("groupItemsIntoSections leaves an unseeded multi-item qid group as bare <item>s", () => {
  // Shouldn't happen via the app's own UI (an unseeded export never repeats a
  // qid), but the module's own contract is "seeded and >1 is what triggers a
  // section" — not "more than one item with the same qid".
  const items = [
    { id: "q01", node: el("item", { ident: "q01" }), qid: "q01", title: "Units", seed: undefined },
    { id: "q01_2", node: el("item", { ident: "q01_2" }), qid: "q01", title: "Units", seed: undefined },
  ];
  const children = groupItemsIntoSections(items, new Set());
  assert.equal(children.length, 2);
  assert.ok(children.every(c => c.node.tag === "item"));
});

test("groupItemsIntoSections disambiguates a section ident against the shared usedIds set", () => {
  const used = new Set(["section_q01"]);
  const items = [
    { id: "q01__seed42", node: el("item", { ident: "q01__seed42" }), qid: "q01", title: "Units", seed: 42 },
    { id: "q01__seed137", node: el("item", { ident: "q01__seed137" }), qid: "q01", title: "Units", seed: 137 },
  ];
  const [child] = groupItemsIntoSections(items, used);
  assert.equal(child.node.attrs.ident, "section_q01_2");
});

test("buildQtiPackage wraps a question's full set of randomized versions in one named <section>, titled with the base (unsuffixed) title", async () => {
  class FakeZip {
    constructor() { this.files = {}; }
    file(path, content) { this.files[path] = content; return this; }
    async generateAsync() { return { __files: this.files }; }
  }
  const mc = { question: "x", choices: ["1", "2", "3", "4", "5"], answer: "a" };
  const questions = [
    { qid: "q01", title: "Units", seed: 42, ...mc },
    { qid: "q02", title: "Kinematics", seed: 42, ...mc },
    { qid: "q01", title: "Units", seed: 137, ...mc },
    { qid: "q02", title: "Kinematics", seed: 137, ...mc },
  ];
  const { blob } = await buildQtiPackage(questions, { latexToMathML: stubLatexToMathML, zipFactory: FakeZip });
  const xml = blob.__files["questestinterop.xml"];
  assertWellFormedXmlFragment(xml);
  assert.match(xml, /<section ident="[^"]+" title="Units">/);
  assert.match(xml, /<section ident="[^"]+" title="Kinematics">/);
  const unitsSection = xml.match(/<section ident="[^"]+" title="Units">([\s\S]*?)<\/section>/)[1];
  assert.match(unitsSection, /ident="q01__seed42"/);
  assert.match(unitsSection, /ident="q01__seed137"/);
  assert.ok(!unitsSection.includes("q02"), "each section must only contain its own question's versions");
});

test("buildQtiPackage leaves a single-paper (unseeded) export with no <section> at all", async () => {
  class FakeZip {
    constructor() { this.files = {}; }
    file(path, content) { this.files[path] = content; return this; }
    async generateAsync() { return { __files: this.files }; }
  }
  const questions = [
    { qid: "q01", title: "Units", question: "x", choices: ["1", "2", "3", "4", "5"], answer: "a" },
    { qid: "q02", title: "Kinematics", question: "x", choices: ["1", "2", "3", "4", "5"], answer: "b" },
  ];
  const { blob } = await buildQtiPackage(questions, { latexToMathML: stubLatexToMathML, zipFactory: FakeZip });
  const xml = blob.__files["questestinterop.xml"];
  assertWellFormedXmlFragment(xml);
  assert.ok(!xml.includes("<section"), "an unseeded export must not introduce any section");
});

test("buildQtiPackage lands every version of a question contiguously, with distinct seed-tagged idents", async () => {
  class FakeZip {
    constructor() { this.files = {}; }
    file(path, content) { this.files[path] = content; return this; }
    async generateAsync() { return { __files: this.files }; }
  }
  const mc = { question: "x", choices: ["1", "2", "3", "4", "5"], answer: "a" };
  // Paper-major input, as exportQtiForPapers builds it.
  const questions = [
    { qid: "q01", title: "Units", seed: 42, ...mc },
    { qid: "q02", title: "Kinematics", seed: 42, ...mc },
    { qid: "q01", title: "Units", seed: 137, ...mc },
    { qid: "q02", title: "Kinematics", seed: 137, ...mc },
  ];
  const { blob } = await buildQtiPackage(questions, { latexToMathML: stubLatexToMathML, zipFactory: FakeZip });
  const xml = blob.__files["questestinterop.xml"];
  assertWellFormedXmlFragment(xml);
  const idents = [...xml.matchAll(/<item ident="([^"]+)"/g)].map(m => m[1]);
  // Contiguous per question — that adjacency is what makes selecting a whole
  // group into one D2L question pool a single drag.
  assert.deepEqual(idents, ["q01__seed42", "q01__seed137", "q02__seed42", "q02__seed137"]);
  // Seed suffixes keep the idents distinct on their own, so none of them falls
  // back to sanitizeIdentifier's opaque "_2" collision suffix.
  assert.ok(!idents.some(id => /_\d+$/.test(id.replace(/__seed-?\d+$/, ""))), "no opaque collision suffixes");
  assert.match(xml, /title="Units \(seed 42\)"/);
  assert.match(xml, /title="Units \(seed 137\)"/);
});
