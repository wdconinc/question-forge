// qti_export.js — pure IMS Common Cartridge 1.1 question-bank package builder
// for question-forge, targeting D2L Brightspace's "Import/Export/Copy
// Components" course import.
//
// This app's first two attempts targeted bare IMS QTI 2.1 content packages
// and both failed against a real D2L course with "We didn't find any
// questions or sections", even after fixing a genuine QTI 2.1 conformance gap
// (a missing assessmentTest). The actual cause, confirmed from D2L's own
// processing log: course-level import always runs packages through
// `Converters.CommonCartridge` — it does not treat a bare QTI 2.1 package as
// QTI at all, so unrecognized resources silently fall back to being imported
// as a generic external link. Common Cartridge's own assessment/question-bank
// resource types are QTI 1.2-based, not 2.1 (confirmed against the 1EdTech
// Common Cartridge 1.1 spec directly, and against a real Canvas-generated CC
// package on GitHub, instructure/common-cartridge-viewer).
//
// Deliberately dependency-free and DOM-free: Node has no document/XMLSerializer,
// and this project ships zero runtime npm dependencies (everything else loads
// via CDN <script> tags in index.html). A small hand-rolled element tree +
// serializer gives the same escaping guarantees as real DOM APIs while staying
// runnable — identically — in Node (for unit tests) and in the browser.
//
// MathJax (`latexToMathML`) and JSZip (`zipFactory`) are both injected by the
// caller rather than imported, so this module never touches a global that only
// exists in a browser.

const LETTERS = ["a", "b", "c", "d", "e"];
const CHOICE_IDS = ["A", "B", "C", "D", "E"];

// ── Minimal XML element tree + serializer ───────────────────────────────────

export function el(tag, attrs = {}, children = []) {
  return { tag, attrs, children: Array.isArray(children) ? children : [children] };
}

export function text(value) {
  return { text: String(value) };
}

// Splices an already-serialized XML/HTML string in verbatim (used for MathML
// output, which must not be re-escaped or re-parsed).
export function raw(xmlString) {
  return { raw: String(xmlString) };
}

function escapeXmlText(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeXmlAttr(s) {
  return escapeXmlText(s).replace(/"/g, "&quot;");
}

export function serialize(node) {
  if (node == null) return "";
  if (typeof node === "string") return escapeXmlText(node);
  if ("text" in node) return escapeXmlText(node.text);
  if ("raw" in node) return node.raw;

  const attrStr = Object.entries(node.attrs || {})
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => ` ${k}="${escapeXmlAttr(v)}"`)
    .join("");
  const children = node.children || [];
  if (children.length === 0) return `<${node.tag}${attrStr}/>`;
  return `<${node.tag}${attrStr}>${children.map(serialize).join("")}</${node.tag}>`;
}

// ── Identifier sanitization ──────────────────────────────────────────────────

// Stands in for a question with no usable qid at all. Shared with
// versionedQid below so a seeded and an unseeded export fall back the same way.
const FALLBACK_ID = "q_item";

// Coerces a qid into a valid XML identifier (must start with a letter or
// underscore) and disambiguates collisions against `usedIds`, if provided.
export function sanitizeIdentifier(rawId, usedIds) {
  let id = String(rawId ?? "").replace(/[^A-Za-z0-9_.-]/g, "_");
  if (!id) id = FALLBACK_ID;
  else if (!/^[A-Za-z_]/.test(id)) id = "q_" + id;
  if (usedIds) {
    const base = id;
    let n = 2;
    while (usedIds.has(id)) { id = `${base}_${n}`; n += 1; }
    usedIds.add(id);
  }
  return id;
}

// ── Seed-versioned identity ──────────────────────────────────────────────────

// A question carrying a `seed` is one randomized *version* of that question:
// the same generator re-run against a different paper seed, so it asks the same
// thing with different numbers. Every version is exported as its own bank item,
// which is what lets an instructor drop all versions of one question into a
// single D2L question pool and have each student draw a different one.
//
// The suffix goes at the END of both the title and the identifier so all
// versions of a question share a prefix: they stay adjacent under the bank's
// import order AND under an alphabetical sort of the Question Library, so
// selecting the whole group is one drag either way. Suffixing the identifier
// too keeps versions from colliding on the same qid and picking up
// sanitizeIdentifier's opaque "_2"/"_3" fallback instead.
//
// A seed of null/undefined means "not a version" — a single-paper export, whose
// items keep the plain qid and title they have always had.

// Both coerce their base BEFORE appending the suffix: interpolating a nullish
// qid/title straight into the template would bake the literal text "null" or
// "undefined" into the identifier, skipping the fallback the unseeded path
// applies. A missing base is missing whether or not a seed is present.

export function versionedQid(qid, seed) {
  const base = String(qid ?? "") || FALLBACK_ID;
  return seed === null || seed === undefined ? base : `${base}__seed${seed}`;
}

export function versionedTitle(title, seed) {
  const base = String(title ?? "");
  if (seed === null || seed === undefined) return base;
  return base ? `${base} (seed ${seed})` : `(seed ${seed})`;
}

// Groups every version of the same question together, in the order each qid was
// first seen. A single-paper export (one version per qid) comes back in exactly
// its input order, so this is a no-op there.
export function groupVersionsByQid(questions) {
  const groups = new Map();
  for (const q of questions) {
    const key = String(q.qid ?? "");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(q);
  }
  return [...groups.values()].flat();
}

// Resolves the <item> ident/title pair shared by every question type.
function itemIdentity(question, usedIds) {
  return {
    id: sanitizeIdentifier(versionedQid(question.qid, question.seed), usedIds),
    title: versionedTitle(question.title || question.qid, question.seed),
  };
}

// ── Math-delimiter scanning ──────────────────────────────────────────────────
// Matches this app's live MathJax config (index.html): inlineMath $...$ / \(...\),
// plus MathJax 3's un-overridden defaults for displayMath $$...$$ / \[...\].

const MATH_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)/g;

function splitMathSegments(str) {
  const segments = [];
  let lastIndex = 0;
  const re = new RegExp(MATH_RE.source, "g");
  let m;
  while ((m = re.exec(str)) !== null) {
    if (m.index > lastIndex) segments.push({ type: "text", value: str.slice(lastIndex, m.index) });
    if (m[1] !== undefined) segments.push({ type: "math", value: m[1], display: true });
    else if (m[2] !== undefined) segments.push({ type: "math", value: m[2], display: true });
    else if (m[3] !== undefined) segments.push({ type: "math", value: m[3], display: false });
    else segments.push({ type: "math", value: m[4], display: false });
    lastIndex = re.lastIndex;
  }
  if (lastIndex < str.length) segments.push({ type: "text", value: str.slice(lastIndex) });
  return segments;
}

// Builds an HTML string (math runs converted to inline MathML and spliced
// verbatim — HTML5 parses <math> as foreign content natively, no special
// wrapping needed) for embedding as QTI 1.2 `<mattext texttype="text/html">`
// content. On a MathML conversion failure, falls back to the original
// delimited text and records the failure in `failures` rather than aborting
// the whole export.
//
// Deliberately returns UNESCAPED HTML (literal `<`/`&`/`>`, e.g. from the
// <p>/<math> tags themselves): the caller places this whole string as the
// text content of an XML <mattext> element via `text()`, which escapes it
// exactly once — matching a real Canvas-generated CC package's shape
// (`&lt;div&gt;&lt;p&gt;...&lt;/p&gt;&lt;/div&gt;`, a single level of
// escaping applied uniformly to the whole HTML string). Escaping here too
// would double-escape.
async function buildMattextHtml(str, { qid, latexToMathML, failures }) {
  const parts = [];
  for (const seg of splitMathSegments(str || "")) {
    if (seg.type === "text") {
      if (seg.value.length) parts.push(seg.value);
      continue;
    }
    try {
      parts.push(await latexToMathML(seg.value, { display: seg.display }));
    } catch (err) {
      failures.push({ qid, snippet: seg.value, error: String(err?.message || err) });
      parts.push(seg.display ? `$$${seg.value}$$` : `$${seg.value}$`);
    }
  }
  return `<p>${parts.join("")}</p>`;
}

function mattext(html) {
  return el("material", {}, [el("mattext", { texttype: "text/html" }, [text(html)])]);
}

// Spliced into the stem's mattext HTML exactly like the browser preview and
// exam_core.py/render.py embed it (same <div> wrapper, same rationale in
// their svg_figure_html). UNVERIFIED against a real D2L import: D2L's HTML
// sanitizer may or may not preserve an inline <svg> the way it preserves the
// MathML this module already relies on.
function svgFigureHtml(question) {
  if (!question.svg) return "";
  return `<div style="max-width:100%;overflow-x:auto;text-align:center">${question.svg}</div>`;
}

// ── QTI 1.2 <item> builder ───────────────────────────────────────────────────

// question: either
//   { qid, title?, seed?, question, choices: [5 strings], answer: "a".."e" }   (multiple choice)
// or
//   { qid, title?, seed?, type: "numerical", question, answer: number, tolerance: number, unit? }
// An optional `seed` marks this as one randomized version among several and
// suffixes both the ident and the title (see versionedQid/versionedTitle).
// opts: { latexToMathML, usedIds?: Set, failures?: Array }
// Returns an `el(...)` node (an <item>, not a full document — it's nested
// inside the single combined <objectbank> built by buildObjectBankXml).
export async function buildItemNode(question, opts) {
  if (question.type === "numerical") {
    return buildNumericalItemNode(question, opts);
  }
  return buildMultipleChoiceItemNode(question, opts);
}

async function buildMultipleChoiceItemNode(question, opts) {
  const { latexToMathML } = opts;
  const usedIds = opts.usedIds || new Set();
  const failures = opts.failures || [];
  if (typeof latexToMathML !== "function") {
    throw new Error("buildItemNode requires opts.latexToMathML");
  }

  const { id, title } = itemIdentity(question, usedIds);
  const correctIdx = LETTERS.indexOf(question.answer);
  if (correctIdx === -1) {
    throw new Error(`buildItemNode: question ${question.qid} has invalid answer "${question.answer}"`);
  }
  const correctChoiceId = CHOICE_IDS[correctIdx];

  const stemHtml = await buildMattextHtml(question.question, { qid: question.qid, latexToMathML, failures })
    + svgFigureHtml(question);
  const responseLabels = [];
  for (let i = 0; i < 5; i++) {
    const choiceHtml = await buildMattextHtml(question.choices[i], { qid: question.qid, latexToMathML, failures });
    responseLabels.push(el("response_label", { ident: CHOICE_IDS[i] }, [mattext(choiceHtml)]));
  }

  // cc_profile: cc.multiple_choice.v0p1 is the Common Cartridge question-type
  // identifier the importer's CC converter keys off of — without it, an
  // otherwise well-formed QTI 1.2 item was observed to be silently ignored.
  const itemmetadata = el("itemmetadata", {}, [
    el("qtimetadata", {}, [
      el("qtimetadatafield", {}, [el("fieldlabel", {}, [text("cc_profile")]), el("fieldentry", {}, [text("cc.multiple_choice.v0p1")])]),
    ]),
  ]);

  const presentation = el("presentation", {}, [
    mattext(stemHtml),
    el("response_lid", { ident: "response1", rcardinality: "Single" }, [el("render_choice", {}, responseLabels)]),
  ]);

  const resprocessing = el("resprocessing", {}, [
    el("outcomes", {}, [el("decvar", { varname: "SCORE", vartype: "Decimal", minvalue: "0", maxvalue: "100" })]),
    el("respcondition", { continue: "No" }, [
      el("conditionvar", {}, [el("varequal", { respident: "response1" }, [text(correctChoiceId)])]),
      el("setvar", { action: "Set", varname: "SCORE" }, [text("100")]),
    ]),
  ]);

  return { id, node: el("item", { ident: id, title }, [itemmetadata, presentation, resprocessing]) };
}

// Numeric fill-in-the-blank item: QTI 1.2 ASI response_str/render_fib, graded
// via a bounded list of literal exact-match alternatives (see
// enumerateNumericAnswers) rather than a true numeric range.
//
// CONFIRMED against a real D2L import: this always lands as a D2L "Short
// Answer" question, never a native "Numeric" type. That's not a mislabeling
// bug to fix — D2L's Question Library has no numeric type reachable from ANY
// file-based import path (not CSV bulk-upload, not Common Cartridge 1.1, not
// Thin Common Cartridge 1.3; all three published type lists top out at
// fill-in-blank/short-answer). cc.fib.v0p1 is the correct, spec-compliant
// profile choice here. The actual bug was resprocessing: an earlier version
// matched the tolerance range via <and><vargte/><varlte/></and>, but the fib
// profile only defines literal <varequal> matching (same mechanism the
// multiple-choice exporter above relies on) — D2L's Common Cartridge
// converter had no way to parse a numeric range condition, so the imported
// question likely had no working correct answer at all, on top of the wrong
// type label.
//
// Still open: whether D2L's Short Answer grading actually honors multiple
// <varequal> alternatives the way this assumes (same "confirm with a real
// test-import" caveat the multiple-choice exporter needed before it was
// verified working).
async function buildNumericalItemNode(question, opts) {
  const { latexToMathML } = opts;
  const usedIds = opts.usedIds || new Set();
  const failures = opts.failures || [];
  if (typeof latexToMathML !== "function") {
    throw new Error("buildItemNode requires opts.latexToMathML");
  }

  const { id, title } = itemIdentity(question, usedIds);
  const answer = Number(question.answer);
  const tolerance = Number(question.tolerance ?? 0);
  if (!Number.isFinite(answer)) {
    throw new Error(`buildItemNode: question ${question.qid} has invalid numerical answer "${question.answer}"`);
  }
  if (!Number.isFinite(tolerance)) {
    throw new Error(`buildItemNode: question ${question.qid} has invalid tolerance "${question.tolerance}"`);
  }
  if (tolerance < 0) {
    throw new Error(`buildItemNode: question ${question.qid} has a negative tolerance "${question.tolerance}"`);
  }
  // Mirrors server/qvalidate.py's contract for this field: an integer >= 1,
  // required only when present at all — a question with no opinion on
  // precision still defaults to 3 (matching phys_fmt's own default).
  let sigFigs = 3;
  if (question.sig_figs !== undefined && question.sig_figs !== null) {
    const parsedSigFigs = Number(question.sig_figs);
    if (!Number.isInteger(parsedSigFigs) || parsedSigFigs < 1) {
      throw new Error(`buildItemNode: question ${question.qid} has invalid sig_figs "${question.sig_figs}"`);
    }
    sigFigs = parsedSigFigs;
  }

  // Told to the student, not just baked into grading: without it, "type the
  // number you computed" and "must exactly match one of N rounded strings"
  // are different tasks, and only one of them is winnable.
  const precisionNote = question.unit
    ? `<p><em>Enter only the numeric value (no units), rounded to ${sigFigs} significant figures.</em></p>`
    : `<p><em>Enter your answer rounded to ${sigFigs} significant figures.</em></p>`;
  const stemHtml = await buildMattextHtml(question.question, { qid: question.qid, latexToMathML, failures })
    + svgFigureHtml(question)
    + precisionNote;

  const itemmetadata = el("itemmetadata", {}, [
    el("qtimetadata", {}, [
      el("qtimetadatafield", {}, [el("fieldlabel", {}, [text("cc_profile")]), el("fieldentry", {}, [text("cc.fib.v0p1")])]),
    ]),
  ]);

  const presentation = el("presentation", {}, [
    mattext(stemHtml),
    el("response_str", { ident: "response1", rcardinality: "Single" }, [
      el("render_fib", { fibtype: "String", rows: "1", columns: "10", prompt: "Box" }),
    ]),
  ]);

  const alternatives = enumerateNumericAnswers(answer, tolerance, sigFigs);
  const varequals = alternatives.map((v) => el("varequal", { respident: "response1" }, [text(v)]));
  const conditionvar = el("conditionvar", {}, varequals.length > 1 ? [el("or", {}, varequals)] : varequals);

  const resprocessing = el("resprocessing", {}, [
    el("outcomes", {}, [el("decvar", { varname: "SCORE", vartype: "Decimal", minvalue: "0", maxvalue: "100" })]),
    el("respcondition", { continue: "No" }, [
      conditionvar,
      el("setvar", { action: "Set", varname: "SCORE" }, [text("100")]),
    ]),
  ]);

  return { id, node: el("item", { ident: id, title }, [itemmetadata, presentation, resprocessing]) };
}

// Every value at the answer's own significant-figure precision that falls
// within [answer - tolerance, answer + tolerance], formatted the same plain
// fixed-decimal way a student would type it (see sigFigsPrecision). Always
// returns at least one string. Falls back to a single alternative — the
// answer rounded to its own precision, i.e. zero-tolerance exact match —
// when the range can't be reasonably enumerated as literal alternatives:
// either the magnitude is extreme enough that a student would type it in
// scientific notation instead (too many equivalent spellings to enumerate),
// or the range/step ratio would produce an unreasonably long answer list.
const MAX_NUMERIC_ALTERNATIVES = 25;

function enumerateNumericAnswers(answer, tolerance, sigFigs, maxCount = MAX_NUMERIC_ALTERNATIVES) {
  const { decimalPlaces, extreme } = sigFigsPrecision(answer, sigFigs);
  const singleAlternative = () => [answer.toFixed(decimalPlaces)];
  if (extreme) return singleAlternative();

  const step = Math.pow(10, -decimalPlaces);
  const epsilon = 1e-6;
  const kMin = Math.ceil((answer - tolerance) / step - epsilon);
  const kMax = Math.floor((answer + tolerance) / step + epsilon);
  const count = kMax - kMin + 1;
  if (!(count >= 1) || count > maxCount) return singleAlternative();

  const values = [];
  for (let k = kMin; k <= kMax; k++) {
    values.push((k * step).toFixed(decimalPlaces));
  }
  return values;
}

// Mirrors questions.phys_fmt's plain-decimal precision rule (python/__init__.py):
// decimalPlaces = max(0, sigFigs - 1 - exponent). phys_fmt instead prints
// values with |exponent| >= 4 in scientific notation for humans — not a
// string a student would type into a text box, so callers get `extreme: true`
// and fall back rather than enumerating plain-decimal strings nobody would use.
// decimalPlaces is also clamped to 100 — Number#toFixed's own hard limit —
// since an extreme enough exponent would otherwise throw a RangeError even
// on the single-alternative fallback `extreme` routes callers to.
function sigFigsPrecision(value, sigFigs) {
  if (value === 0) return { decimalPlaces: Math.max(0, sigFigs - 1), extreme: false };
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const decimalPlaces = Math.min(100, Math.max(0, sigFigs - 1 - exponent));
  return { decimalPlaces, extreme: exponent >= 4 || exponent <= -4 };
}

// ── questestinterop.xml builder (one <objectbank> holding every item) ───────

// A Common Cartridge question bank is a single <objectbank> containing <item>
// elements directly (no <section> nesting, unlike a CC "assessment"/quiz
// resource) — confirmed against the 1EdTech CC 1.1 spec and a real IMS QTI
// ASI reference. Only one question-bank resource is allowed per cartridge, so
// this always builds the *entire* export as one file.
//
// items: [{ id, node }] as produced by buildItemNode, in materialized order.
export function buildObjectBankXml(items, opts = {}) {
  const bankId = sanitizeIdentifier(opts.bankId || "question-forge-bank");

  const objectbank = el("objectbank", { ident: bankId }, items.map(item => item.node));

  const questestinterop = el(
    "questestinterop",
    {
      xmlns: "http://www.imsglobal.org/xsd/ims_qtiasiv1p2",
      "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
      "xsi:schemaLocation":
        "http://www.imsglobal.org/xsd/ims_qtiasiv1p2 http://www.imsglobal.org/profile/cc/ccv1p1/ccv1p1_qtiasiv1p2p1_v1p0.xsd",
    },
    [objectbank]
  );

  return { id: bankId, xml: '<?xml version="1.0" encoding="UTF-8"?>\n' + serialize(questestinterop) };
}

// ── imsmanifest.xml builder ──────────────────────────────────────────────────

// A CC question-bank resource must NOT be referenced from <organizations> at
// all (explicit in the 1EdTech CC 1.1 spec: "it is not included in the
// organization") — <organizations/> stays empty. This is the opposite of what
// this app's second attempt tried (populating organizations for a bare QTI
// 2.1 assessmentTest resource) — that fix targeted the wrong resource type;
// under Common Cartridge's own question-bank type, an empty organizations is
// correct, not the bug.
export function buildManifestXml(bankId, opts = {}) {
  const manifestId = sanitizeIdentifier(opts.manifestId || "question-forge-export");
  const bankHref = opts.bankHref || "questestinterop.xml";

  const manifest = el(
    "manifest",
    {
      identifier: "MAN_" + manifestId,
      xmlns: "http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1",
      "xmlns:lom": "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource",
      "xmlns:lomimscc": "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest",
      "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
      "xsi:schemaLocation":
        "http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1 http://www.imsglobal.org/profile/cc/ccv1p1/ccv1p1_imscp_v1p2_v1p0.xsd " +
        "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/resource http://www.imsglobal.org/profile/cc/ccv1p1/LOM/ccv1p1_lomresource_v1p0.xsd " +
        "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest http://www.imsglobal.org/profile/cc/ccv1p1/LOM/ccv1p1_lommanifest_v1p0.xsd",
    },
    [
      el("metadata", {}, [el("schema", {}, [text("IMS Common Cartridge")]), el("schemaversion", {}, [text("1.1.0")])]),
      el("organizations", {}, []),
      el("resources", {}, [
        el("resource", { identifier: "RES_" + bankId, type: "imsqti_xmlv1p2/imscc_xmlv1p1/question-bank", href: bankHref }, [
          el("file", { href: bankHref }),
        ]),
      ]),
    ]
  );

  return '<?xml version="1.0" encoding="UTF-8"?>\n' + serialize(manifest);
}

// ── Full package ──────────────────────────────────────────────────────────────

// materializedQuestions: [{ qid, title?, seed?, question, topic?, difficulty?, ...
//   (choices, answer) for multiple choice, or
//   (type: "numerical", answer, tolerance, unit?) for numerical entry ]
// May contain the same qid more than once — one entry per randomized version,
// each with its own `seed`. Versions are regrouped so every version of a
// question lands contiguously in the bank, ready to be selected as a block into
// a D2L question pool.
// opts: { latexToMathML, manifestId?, zipFactory? } — zipFactory defaults to
// the browser global JSZip; tests inject a fake to avoid needing a real
// dependency.
export async function buildQtiPackage(materializedQuestions, opts = {}) {
  const { latexToMathML, manifestId, zipFactory } = opts;
  if (typeof latexToMathML !== "function") {
    throw new Error("buildQtiPackage requires opts.latexToMathML");
  }

  const usedIds = new Set();
  const failures = [];
  const items = [];
  for (const q of groupVersionsByQid(materializedQuestions)) {
    const { id, node } = await buildItemNode(q, { latexToMathML, usedIds, failures });
    items.push({ id, node });
  }

  const bankHref = "questestinterop.xml";
  const bankFile = buildObjectBankXml(items, { bankId: manifestId });
  const manifestXml = buildManifestXml(bankFile.id, { manifestId, bankHref });

  const ZipCtor = zipFactory || (typeof JSZip !== "undefined" ? JSZip : undefined);
  if (!ZipCtor) throw new Error("buildQtiPackage requires JSZip (pass opts.zipFactory, or load it globally)");

  const zip = new ZipCtor();
  zip.file("imsmanifest.xml", manifestXml);
  zip.file(bankHref, bankFile.xml);

  const blob = await zip.generateAsync({ type: "blob" });
  return { blob, failures };
}
