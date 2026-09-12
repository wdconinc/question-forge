import test from "node:test";
import assert from "node:assert/strict";
import {
  sanitizeIdentifier,
  buildItemNode,
  buildObjectBankXml,
  buildManifestXml,
  buildQtiPackage,
  serialize,
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
