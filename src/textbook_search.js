/**
 * Two-stage textbook retrieval for the browser.
 *
 * Stage 1 is a compact per-book index (titles, learning objectives, key terms,
 * the book's own section summaries, plus precomputed BM25 statistics).  Stage 2
 * fetches only the chapter shards the ranked sections live in.
 *
 * Shards are per *chapter*, not per section: a model that asks for 4.3 usually
 * wants 4.4 next, and it is already in the cached shard.
 *
 * All I/O is injected -- `fetchJson` arrives in an options object rather than
 * being imported -- so this module runs unchanged under `node --test` with a
 * stub, the same contract src/qti_export.js follows.
 *
 * Retrieval deliberately selects *whole sections* rather than fragments.  A
 * p90 section is ~1600 tokens against a 1M-token window, and writing a good
 * exam question needs a complete worked example, the section's notation and its
 * problem set -- an OpenStax example runs Problem -> Strategy -> Solution ->
 * Discussion, and any fixed-size chunker splits Strategy from Solution.
 */

import { combineStats, rankSections, tokenize } from "./textbook_index.js";

/** Hard ceiling on injected context, across all accumulated searches in a turn. */
export const MAX_CONTEXT_CHARS = 120000;

/**
 * Most sections one lookup may return, matching the `max_sections` range in the
 * search_textbook schema.  Explicit section_ids are capped here too: naming
 * eight sections is still a request the prompt budget has to absorb, and a
 * second, larger limit hidden in that branch was exactly the kind of quiet
 * contract drift that overfills the context.
 */
export const MAX_SECTIONS = 6;

const DEFAULT_INCLUDE = ["objectives", "summary", "equations", "examples", "problems"];
const KNOWN_INCLUDE = new Set([...DEFAULT_INCLUDE, "body", "conceptual", "glossary"]);

// Calculus sections carry 50+ end-of-section exercises.  Dumping all of them
// crowds out every other section in the budget for no extra signal.
const MAX_PROBLEMS = 12;
const MAX_CONCEPTUAL = 6;
const MAX_EXAMPLES = 3;

/**
 * Cached, injected-fetch accessor for the static corpus under `baseUrl`.
 * Promises (not values) are cached, so concurrent callers share one request.
 */
export function createCorpus({ fetchJson, baseUrl = "./corpus", version = "" } = {}) {
  if (typeof fetchJson !== "function") throw new Error("createCorpus: fetchJson is required");
  const cache = new Map();
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  const get = (path) => {
    if (!cache.has(path)) {
      cache.set(path, Promise.resolve(fetchJson(`${baseUrl}/${path}${suffix}`)));
    }
    return cache.get(path);
  };
  return {
    manifest: () => get("manifest.json"),
    index: (slug) => get(`${slug}/index.json`),
    shard: (slug, shard) => get(`${slug}/${shard}.json`),
    fetchCount: () => cache.size,
  };
}

function activeBooks(manifest, wanted) {
  const all = (manifest && manifest.books) || [];
  if (!wanted || !wanted.length) return all;
  const want = new Set(wanted);
  const picked = all.filter((b) => want.has(b.slug));
  return picked.length ? picked : all;
}

/** Accept "4.3" or "college-physics-2e:4.3". */
function parseHandle(handle, books) {
  const raw = String(handle).trim();
  const colon = raw.indexOf(":");
  if (colon > 0) {
    const slug = raw.slice(0, colon);
    if (books.some((b) => b.slug === slug)) return { slug, number: raw.slice(colon + 1) };
  }
  return { slug: null, number: raw };
}

/**
 * Resolve a textbook search into full section records.
 *
 * `section_ids` bypasses ranking entirely.  That matters because the chat
 * transport gives the model exactly one chance to ask for context per turn, so
 * it has to be able to name sections it already knows from the catalog instead
 * of guessing a query and hoping.
 */
export async function searchTextbook(args = {}, deps = {}) {
  const { corpus } = deps;
  if (!corpus) throw new Error("searchTextbook: corpus is required");

  const query = String(args.query || "").trim();
  const sectionIds = Array.isArray(args.section_ids) ? args.section_ids : [];
  const chapters = Array.isArray(args.chapters) ? args.chapters : [];
  const maxSections = Math.max(1, Math.min(Number(args.max_sections) || 3, MAX_SECTIONS));

  const manifest = await corpus.manifest();
  const books = activeBooks(manifest, args.books);
  const bySlug = new Map(books.map((b) => [b.slug, b]));

  let selected = [];
  const missing = [];

  if (sectionIds.length) {
    // Explicitly named sections are served up to the same ceiling as a ranked
    // search.  They are not narrowed to `maxSections`, which the model may not
    // have set: having named them, it has stated its intent.
    for (const handle of sectionIds.slice(0, MAX_SECTIONS)) {
      const { slug, number } = parseHandle(handle, books);
      const candidates = slug ? [bySlug.get(slug)] : books;
      let found = null;
      for (const book of candidates) {
        if (!book) continue;
        const index = await corpus.index(book.slug);
        const rec = (index.sections || []).find((s) => s.id === number);
        if (rec) { found = { slug: book.slug, id: rec.id, sh: rec.sh, section: rec }; break; }
      }
      if (found) selected.push(found);
      else missing.push(String(handle));
    }
  } else if (query) {
    // Rank every enabled book, then merge on one global scale.
    //
    // Reciprocal-rank fusion was the obvious choice and is wrong here: it fuses
    // different rankers over *the same* documents, but these lists cover
    // disjoint books, so it simply promoted each book's best hit to within
    // 1/61 of every other book's and let book order decide.  A query about the
    // chain rule came back with Fission and Dimensional Analysis and no
    // Calculus at all.  Coverage-weighted BM25 is comparable across books
    // because the coverage factor is scale-free.
    const indexes = await Promise.all(books.map((b) => corpus.index(b.slug)));
    const stats = combineStats(indexes);
    const pooled = [];
    books.forEach((book, i) => {
      for (const r of rankSections(query, indexes[i], { topK: maxSections, chapters, stats })) {
        pooled.push({ slug: book.slug, id: r.id, sh: r.section.sh, section: r.section, relevance: r.relevance });
      }
    });
    pooled.sort((a, b) => b.relevance - a.relevance);
    selected = pooled.slice(0, maxSections);
  }

  // Fetch each needed chapter shard once, not once per section.
  const shardKeys = [...new Set(selected.map((s) => `${s.slug}|${s.sh}`))];
  const shards = new Map();
  await Promise.all(shardKeys.map(async (key) => {
    const [slug, sh] = key.split("|");
    shards.set(key, await corpus.shard(slug, sh));
  }));

  const hits = [];
  for (const sel of selected) {
    const shard = shards.get(`${sel.slug}|${sel.sh}`);
    const record = shard && shard.sections ? shard.sections[sel.id] : null;
    if (!record) { missing.push(`${sel.slug}:${sel.id}`); continue; }
    const book = bySlug.get(sel.slug) || {};
    hits.push({
      slug: sel.slug,
      bookTitle: book.title || sel.slug,
      attribution: book.attribution || "",
      license: book.license || "",
      number: record.number,
      title: record.title,
      chapter: record.chapter,
      chapterTitle: record.chapterTitle,
      record,
    });
  }

  return { query, hits, missing, terms: tokenize(query) };
}

// --- prompt rendering -------------------------------------------------------

function renderParts(hit, include) {
  const r = hit.record;
  const parts = [];
  const add = (key, text) => { if (text) parts.push({ key, text }); };

  if (include.has("objectives") && r.objectives && r.objectives.length) {
    add("objectives", "Learning objectives:\n" + r.objectives.map((o) => `  - ${o}`).join("\n"));
  }
  if (include.has("summary") && r.summary) add("summary", `Summary:\n${r.summary}`);
  if (include.has("equations") && r.equations && r.equations.length) {
    add("equations", "Key equations:\n" + r.equations.slice(0, 12).map((e) => `  ${e}`).join("\n"));
  }
  if (include.has("glossary") && r.glossary && r.glossary.length) {
    add("glossary", "Glossary:\n" + r.glossary.map((g) => `  - ${g.term}: ${g.meaning}`).join("\n"));
  }
  if (include.has("examples") && r.examples && r.examples.length) {
    add("examples", r.examples.slice(0, MAX_EXAMPLES).map(
      (e) => `Worked example${e.title ? ` — "${e.title}"` : ""}:\n${e.text}`
    ).join("\n\n"));
  }
  if (include.has("problems") && r.problems && r.problems.length) {
    const shown = r.problems.slice(0, MAX_PROBLEMS);
    const more = r.problems.length - shown.length;
    add("problems", `End-of-section problems (${r.problems.length} in the book${more > 0 ? `, showing ${shown.length}` : ""}):\n` +
      shown.map((p) => `  [P${p.n}] ${p.problem}${p.solution ? `  → ${p.solution}` : ""}`).join("\n"));
  }
  if (include.has("conceptual") && r.conceptual && r.conceptual.length) {
    add("conceptual", "Conceptual questions:\n" +
      r.conceptual.slice(0, MAX_CONCEPTUAL).map((p) => `  [Q${p.n}] ${p.problem}`).join("\n"));
  }
  if (include.has("body") && r.body) add("body", `Full text:\n${r.body}`);
  return parts;
}

/**
 * Render retrieved sections as the block injected into the system prompt.
 *
 * Every section carries a [slug:number] handle so a citation can be checked
 * against something real, and each distinct book contributes one attribution
 * line -- CC BY-NC-SA requires the credit to travel with the text.
 */
export function formatSectionsForPrompt(result, opts = {}) {
  const maxChars = opts.maxChars || MAX_CONTEXT_CHARS;
  const requested = Array.isArray(opts.include) && opts.include.length
    ? opts.include.filter((k) => KNOWN_INCLUDE.has(k))
    : DEFAULT_INCLUDE;
  const include = new Set(requested.length ? requested : DEFAULT_INCLUDE);

  if (!result || !result.hits || !result.hits.length) {
    return `=== TEXTBOOK EXCERPTS ===\nNo section of the enabled textbooks matched ${
      result && result.query ? `"${result.query}"` : "that request"
    }. Say so rather than inventing textbook content, and try different terms or name sections from the catalog.\n`;
  }

  const head = "=== TEXTBOOK EXCERPTS ===\n" +
    (result.query ? `Retrieved for: "${result.query}"\n` : "") +
    "Reference only. Write original, parametrized questions in this style; do not " +
    "reproduce this text verbatim. Cite only the section handles shown below.\n";

  const missing = result.missing && result.missing.length
    ? `Not found in the enabled textbooks: ${result.missing.join(", ")}\n`
    : "";
  const renderFooter = (srcs) =>
    "\n" + "-".repeat(60) + "\nSources: " + [...srcs.values()].join(" | ") + "\n";

  // The attribution footer and the missing-sections line are appended after the
  // packing loop, so they have to be budgeted for before it -- otherwise the
  // returned string overruns maxChars and the caller, which compares
  // accumulated length against MAX_CONTEXT_CHARS, silently drops the whole
  // excerpt.  Reserve the worst case (every hit contributing attribution); the
  // footer actually emitted covers a subset of those and so is never longer.
  const worstCaseSources = new Map();
  for (const hit of result.hits) {
    if (hit.attribution) {
      worstCaseSources.set(hit.slug, `${hit.attribution} (${hit.license || "CC BY-NC-SA 4.0"})`);
    }
  }
  const budget = maxChars - renderFooter(worstCaseSources).length - missing.length;

  const chunks = [];
  let used = head.length;
  const sources = new Map();

  for (const hit of result.hits) {
    const header = `\n${"-".repeat(60)}\n[${hit.slug}:${hit.number}] ${hit.bookTitle}` +
      ` — Ch.${hit.chapter} ${hit.chapterTitle}\n§${hit.number} ${hit.title}\n`;
    if (used + header.length >= budget) break;

    let body = header;
    for (const part of renderParts(hit, include)) {
      const candidate = part.text + "\n";
      if (used + body.length + candidate.length <= budget) {
        body += candidate;
      } else if (part.key === "body") {
        // Prose is the one unbounded part; trim it on a line boundary rather
        // than dropping the whole section.
        const room = budget - used - body.length - 40;
        if (room > 400) {
          const cut = part.text.lastIndexOf("\n", room);
          body += part.text.slice(0, cut > 400 ? cut : room) + "\n…[section text truncated]\n";
        }
        break;
      } else {
        break;
      }
    }
    chunks.push(body);
    used += body.length;
    if (hit.attribution) sources.set(hit.slug, `${hit.attribution} (${hit.license || "CC BY-NC-SA 4.0"})`);
  }

  return head + chunks.join("") + renderFooter(sources) + missing;
}

/**
 * Parse a chapter pin like "4-6, 9" into [4, 5, 6, 9].
 *
 * Lenient by design: this is a free-text field an instructor types between
 * classes, so "4 - 6", "9,4-6" and "6-4" all mean the same thing, and garbage
 * yields an empty pin (= every chapter) rather than an error that silently
 * narrows retrieval to nothing.
 */
export function parseChapterSpec(spec) {
  const out = new Set();
  for (const part of String(spec || "").split(",")) {
    const range = part.trim().match(/^(\d+)\s*[-–]\s*(\d+)$/);
    if (range) {
      const [lo, hi] = [Number(range[1]), Number(range[2])].sort((a, b) => a - b);
      // A fat-fingered "1-900" should not enumerate 900 chapters.
      if (hi - lo <= 200) for (let n = lo; n <= hi; n++) out.add(n);
      continue;
    }
    const one = part.trim().match(/^\d+$/);
    if (one) out.add(Number(part.trim()));
  }
  return [...out].sort((a, b) => a - b);
}

/** Render [4,5,6,9] back to "4-6, 9" for the input field. */
export function formatChapterSpec(chapters) {
  const sorted = [...new Set((chapters || []).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    parts.push(j - i >= 2 ? `${sorted[i]}-${sorted[j]}` : sorted.slice(i, j + 1).join(", "));
    i = j + 1;
  }
  return parts.join(", ");
}

/**
 * The always-on chapter catalog.
 *
 * Chapter level, not section level, on purpose: seven books come to ~740
 * sections, which is ~18k tokens on every turn including the ones that just fix
 * a typo.  Chapters cost ~2k and still let the model name section ids, which it
 * can then request directly.
 */
export function buildCatalog(manifest, opts = {}) {
  const books = activeBooks(manifest, opts.books);
  if (!books.length) return "";
  const pinned = opts.chapters && opts.chapters.length ? new Set(opts.chapters.map(Number)) : null;

  const lines = ["=== TEXTBOOK CATALOG ===",
    "Available through the search_textbook tool. Cite sections exactly as [slug:number]."];
  for (const book of books) {
    lines.push("", `[${book.slug}] ${book.title} — ${book.license || "CC BY-NC-SA 4.0"}`);
    for (const ch of book.chapters || []) {
      if (pinned && !pinned.has(Number(ch.n))) continue;
      lines.push(`  Ch ${ch.n}. ${ch.title} (${ch.sections} sections → ${ch.n}.0–${ch.n}.${ch.sections - 1})`);
    }
  }
  if (pinned) lines.push("", `This question set is pinned to chapters ${[...pinned].join(", ")}; prefer them unless asked otherwise.`);
  return lines.join("\n") + "\n";
}
