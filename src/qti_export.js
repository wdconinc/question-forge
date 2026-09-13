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

// Coerces a qid into a valid XML identifier (must start with a letter or
// underscore) and disambiguates collisions against `usedIds`, if provided.
export function sanitizeIdentifier(rawId, usedIds) {
  let id = String(rawId ?? "").replace(/[^A-Za-z0-9_.-]/g, "_");
  if (!id) id = "q_item";
  else if (!/^[A-Za-z_]/.test(id)) id = "q_" + id;
  if (usedIds) {
    const base = id;
    let n = 2;
    while (usedIds.has(id)) { id = `${base}_${n}`; n += 1; }
    usedIds.add(id);
  }
  return id;
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

// ── QTI 1.2 <item> builder ───────────────────────────────────────────────────

// question: either
//   { qid, title?, question, choices: [5 strings], answer: "a".."e" }         (multiple choice)
// or
//   { qid, title?, type: "numerical", question, answer: number, tolerance: number, unit? }
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

  const id = sanitizeIdentifier(question.qid, usedIds);
  const correctIdx = LETTERS.indexOf(question.answer);
  if (correctIdx === -1) {
    throw new Error(`buildItemNode: question ${question.qid} has invalid answer "${question.answer}"`);
  }
  const correctChoiceId = CHOICE_IDS[correctIdx];

  const stemHtml = await buildMattextHtml(question.question, { qid: question.qid, latexToMathML, failures });
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

  return { id, node: el("item", { ident: id, title: question.title || question.qid }, [itemmetadata, presentation, resprocessing]) };
}

// Numeric fill-in-the-blank item: standard QTI 1.2 ASI response_num/render_fib,
// with a tolerance range matched via <and><vargte/><varlte/></and>.
//
// UNVERIFIED cc_profile choice: Common Cartridge 1.1's formal profile list has
// no dedicated numeric-tolerance type — cc.fib.v0p1 is defined there for
// literal string matching, not numeric ranges. It's the closest available
// signal, used here as a best-effort extrapolation (same spirit as the
// cc.multiple_choice.v0p1 discovery above), not a confirmed-correct value.
// Treat a numerical export as unverified until confirmed with a real D2L
// test-import, same as the multiple-choice exporter needed.
async function buildNumericalItemNode(question, opts) {
  const { latexToMathML } = opts;
  const usedIds = opts.usedIds || new Set();
  const failures = opts.failures || [];
  if (typeof latexToMathML !== "function") {
    throw new Error("buildItemNode requires opts.latexToMathML");
  }

  const id = sanitizeIdentifier(question.qid, usedIds);
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

  const stemHtml = await buildMattextHtml(question.question, { qid: question.qid, latexToMathML, failures });

  const itemmetadata = el("itemmetadata", {}, [
    el("qtimetadata", {}, [
      el("qtimetadatafield", {}, [el("fieldlabel", {}, [text("cc_profile")]), el("fieldentry", {}, [text("cc.fib.v0p1")])]),
    ]),
  ]);

  const presentation = el("presentation", {}, [
    mattext(stemHtml),
    el("response_num", { ident: "response1", rcardinality: "Single", numtype: "Decimal" }, [
      el("render_fib", { fibtype: "Decimal", rows: "1", columns: "10", prompt: "Box" }),
    ]),
  ]);

  const resprocessing = el("resprocessing", {}, [
    el("outcomes", {}, [el("decvar", { varname: "SCORE", vartype: "Decimal", minvalue: "0", maxvalue: "100" })]),
    el("respcondition", { continue: "No" }, [
      el("conditionvar", {}, [
        el("and", {}, [
          el("vargte", { respident: "response1" }, [text(String(answer - tolerance))]),
          el("varlte", { respident: "response1" }, [text(String(answer + tolerance))]),
        ]),
      ]),
      el("setvar", { action: "Set", varname: "SCORE" }, [text("100")]),
    ]),
  ]);

  return { id, node: el("item", { ident: id, title: question.title || question.qid }, [itemmetadata, presentation, resprocessing]) };
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

// materializedQuestions: [{ qid, title?, question, topic?, difficulty?, ...
//   (choices, answer) for multiple choice, or
//   (type: "numerical", answer, tolerance, unit?) for numerical entry ]
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
  for (const q of materializedQuestions) {
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
