import test from "node:test";
import assert from "node:assert/strict";
import {
  createCorpus,
  searchTextbook,
  formatSectionsForPrompt,
  buildCatalog,
  chapterSectionIds,
  MAX_CONTEXT_CHARS,
  MAX_SECTIONS,
} from "./textbook_search.js";

// A two-book corpus held in memory.  No network, no files: createCorpus takes
// fetchJson as an injected option precisely so this works, the same way
// qti_export takes zipFactory.
function makeFixture() {
  const section = (id, sh, ch, t, tf) => ({
    id, sh, ch, t, len: 100, tf, o: [], s: "", k: [],
  });
  const record = (number, chapter, title, extra = {}) => ({
    number, chapter, chapterTitle: `Chapter ${chapter}`, title,
    objectives: [`Understand ${title}.`],
    summary: `A summary of ${title}.`,
    body: `Body text for ${title}. `.repeat(20),
    equations: [`$$E_{${number}} = mc^2$$`],
    examples: [{ title: `Example for ${title}`, text: "Worked solution text." }],
    problems: [{ n: 1, problem: `Problem about ${title}`, solution: "42 N" }],
    conceptual: [{ n: 1, problem: `Why ${title}?`, solution: "" }],
    glossary: [{ term: title.toLowerCase(), meaning: "a defined thing" }],
    figures: [], terms: [title.toLowerCase()],
    ...extra,
  });

  const files = {
    "manifest.json": {
      schemaVersion: 1,
      books: [
        {
          slug: "book-a", title: "Book A", license: "CC BY-NC-SA 4.0",
          attribution: "OpenStax, Book A. Access for free at example.org",
          chapters: [{ n: 1, title: "Mechanics", sections: 3 }, { n: 2, title: "Waves", sections: 2 }],
        },
        {
          slug: "book-b", title: "Book B", license: "CC BY-NC-SA 4.0",
          attribution: "OpenStax, Book B. Access for free at example.org",
          chapters: [{ n: 1, title: "Calculus", sections: 2 }],
        },
      ],
    },
    "book-a/index.json": {
      slug: "book-a", docCount: 3, avgLen: 100,
      df: { friction: 1, motion: 3, wave: 1 },
      sections: [
        section("1.1", "ch01", 1, "Motion", { motion: 30 }),
        section("1.2", "ch01", 1, "Friction", { friction: 30, motion: 5 }),
        section("2.1", "ch02", 2, "Waves", { wave: 30, motion: 5 }),
      ],
    },
    "book-b/index.json": {
      slug: "book-b", docCount: 2, avgLen: 100,
      df: { derivative: 1, motion: 1 },
      sections: [
        section("1.1", "ch01", 1, "Derivatives", { derivative: 30 }),
        section("1.2", "ch01", 1, "Integrals", { motion: 2 }),
      ],
    },
    "book-a/ch01.json": {
      slug: "book-a", attribution: "OpenStax, Book A. Access for free at example.org",
      license: "CC BY-NC-SA 4.0",
      sections: { "1.1": record("1.1", 1, "Motion"), "1.2": record("1.2", 1, "Friction") },
    },
    "book-a/ch02.json": {
      slug: "book-a", attribution: "OpenStax, Book A. Access for free at example.org",
      license: "CC BY-NC-SA 4.0",
      sections: { "2.1": record("2.1", 2, "Waves") },
    },
    "book-b/ch01.json": {
      slug: "book-b", attribution: "OpenStax, Book B. Access for free at example.org",
      license: "CC BY-NC-SA 4.0",
      sections: { "1.1": record("1.1", 1, "Derivatives"), "1.2": record("1.2", 1, "Integrals") },
    },
  };

  const calls = [];
  const fetchJson = async (url) => {
    const path = url.replace(/^.*?corpus\//, "").replace(/\?.*$/, "");
    calls.push(path);
    if (!(path in files)) throw new Error(`404 ${path}`);
    return JSON.parse(JSON.stringify(files[path]));
  };
  return { corpus: createCorpus({ fetchJson, baseUrl: "./corpus" }), calls, files };
}

const shardCalls = (calls) => calls.filter((p) => /ch\d+\.json$/.test(p));

test("createCorpus requires an injected fetchJson", () => {
  assert.throws(() => createCorpus({}), /fetchJson is required/);
});

test("an explicit section id fetches exactly one shard", async () => {
  const { corpus, calls } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1"] }, { corpus });
  assert.equal(r.hits.length, 1);
  assert.equal(r.hits[0].title, "Motion");
  assert.deepEqual(shardCalls(calls), ["book-a/ch01.json"]);
});

test("sections sharing a chapter are fetched in one request, not one each", async () => {
  const { corpus, calls } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1", "book-a:1.2"] }, { corpus });
  assert.equal(r.hits.length, 2);
  assert.deepEqual(shardCalls(calls), ["book-a/ch01.json"]);
});

test("sections in different chapters and books fetch one shard each", async () => {
  const { corpus, calls } = makeFixture();
  await searchTextbook({ section_ids: ["book-a:1.1", "book-a:2.1", "book-b:1.1"] }, { corpus });
  assert.deepEqual(shardCalls(calls).sort(),
    ["book-a/ch01.json", "book-a/ch02.json", "book-b/ch01.json"]);
});

test("a repeated search re-uses the cache instead of refetching", async () => {
  const { corpus, calls } = makeFixture();
  await searchTextbook({ section_ids: ["book-a:1.1"] }, { corpus });
  const after = calls.length;
  await searchTextbook({ section_ids: ["book-a:1.2"] }, { corpus });
  assert.equal(calls.length, after, "second search in the same chapter must fetch nothing");
});

test("an unqualified section id resolves against the enabled books", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["2.1"] }, { corpus });
  assert.equal(r.hits[0].slug, "book-a");
  assert.equal(r.hits[0].title, "Waves");
});

test("unknown section ids are reported, not silently dropped", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:9.9"] }, { corpus });
  assert.equal(r.hits.length, 0);
  assert.deepEqual(r.missing, ["book-a:9.9"]);
});

test("a query ranks across every enabled book", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "friction", max_sections: 1 }, { corpus });
  assert.equal(r.hits.length, 1);
  assert.equal(r.hits[0].title, "Friction");
});

test("the books filter restricts the search", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "derivative", books: ["book-a"] }, { corpus });
  assert.equal(r.hits.length, 0, "book-b holds the only match and was not enabled");
});

test("the chapters filter restricts the search in both directions", async () => {
  const { corpus } = makeFixture();
  // "wave" lives only in chapter 2, so the filter is observable either way.
  const inside = await searchTextbook({ query: "wave", books: ["book-a"], chapters: [2] }, { corpus });
  assert.deepEqual(inside.hits.map((h) => h.number), ["2.1"]);
  assert.equal(inside.browsed, false);
  // Filtered to a chapter the term is absent from, nothing ranks -- so the
  // lookup degrades to browsing that chapter rather than returning nothing.
  const outside = await searchTextbook({ query: "wave", books: ["book-a"], chapters: [1] }, { corpus });
  assert.equal(outside.browsed, true);
  assert.deepEqual(outside.hits.map((h) => h.number), ["1.1", "1.2"]);
});

test("a term common to every section scores below the floor", async () => {
  // "motion" appears in all three book-a sections, so its IDF is near zero.
  // This is the floor doing its job, not a bug: a term that discriminates
  // nothing is not evidence that a section is relevant.
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "motion", books: ["book-a"] }, { corpus });
  assert.equal(r.hits.length, 0);
});

test("a query no section covers returns no hits at all", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "photosynthesis chlorophyll" }, { corpus });
  assert.equal(r.hits.length, 0);
});

test("max_sections is clamped to a sane range", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "motion", max_sections: 99 }, { corpus });
  assert.ok(r.hits.length <= MAX_SECTIONS, "must not return more than the hard ceiling");
});

test("explicit section_ids obey the same ceiling as a ranked search", async () => {
  // This branch used to slice to 8, contradicting the tool schema's documented
  // 1-6 range and letting a single lookup overfill the prompt budget.
  const { corpus } = makeFixture();
  const ids = ["book-a:1.1", "book-a:1.2", "book-a:2.1", "book-b:1.1", "book-b:1.2",
               "book-a:1.1", "book-a:1.2", "book-a:2.1"];
  const r = await searchTextbook({ section_ids: ids }, { corpus });
  assert.ok(r.hits.length + r.missing.length <= MAX_SECTIONS,
    `resolved ${r.hits.length} hits + ${r.missing.length} missing from ${ids.length} ids`);
});

test("formatSectionsForPrompt emits a citable handle and one attribution per book", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1", "book-b:1.1"] }, { corpus });
  const out = formatSectionsForPrompt(r);
  assert.match(out, /\[book-a:1\.1\]/);
  assert.match(out, /\[book-b:1\.1\]/);
  assert.match(out, /Sources:.*Book A.*\|.*Book B/s);
  assert.match(out, /CC BY-NC-SA 4\.0/);
});

test("include selects which parts of a section are rendered", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1"] }, { corpus });
  const only = formatSectionsForPrompt(r, { include: ["problems"] });
  assert.match(only, /End-of-section problems/);
  assert.doesNotMatch(only, /Learning objectives/);
  assert.doesNotMatch(only, /Worked example/);
});

test("include ignores unknown part names rather than rendering nothing", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1"] }, { corpus });
  const out = formatSectionsForPrompt(r, { include: ["nonsense"] });
  assert.match(out, /Learning objectives/, "falls back to the default parts");
});

test("formatSectionsForPrompt never exceeds maxChars", async () => {
  // This asserted `<= budget + 200` and so did not catch that the attribution
  // footer and the missing-sections line were appended *after* the packing
  // loop.  An over-budget string makes the caller -- which compares accumulated
  // length against MAX_CONTEXT_CHARS -- drop the entire excerpt, so the bound
  // has to be exact.
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1", "book-a:1.2", "book-a:2.1"] }, { corpus });
  for (const budget of [400, 900, 2000, 5000]) {
    const out = formatSectionsForPrompt(r, { include: ["summary", "body"], maxChars: budget });
    assert.ok(out.length <= budget, `budget ${budget} produced ${out.length} chars`);
  }
});

test("the attribution footer is still emitted when the budget is tight", async () => {
  // Budgeting must not be achieved by dropping the credit: CC BY-NC-SA requires
  // it to travel with the text.
  const { corpus } = makeFixture();
  const r = await searchTextbook({ section_ids: ["book-a:1.1"] }, { corpus });
  const out = formatSectionsForPrompt(r, { include: ["body"], maxChars: 600 });
  assert.ok(out.length <= 600);
  assert.match(out, /Sources: .*Book A/);
});

test("maxChars accounts for the missing-sections line too", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook(
    { section_ids: ["book-a:1.1", "book-a:9.9", "book-b:8.8"] }, { corpus });
  const out = formatSectionsForPrompt(r, { maxChars: 1200 });
  assert.ok(out.length <= 1200, `produced ${out.length} chars`);
  assert.match(out, /Not found in the enabled textbooks/);
});

test("an empty result tells the model to say so rather than invent content", () => {
  const out = formatSectionsForPrompt({ query: "phlogiston", hits: [], missing: [] });
  assert.match(out, /No section/);
  assert.match(out, /Never invent textbook content/);
  assert.match(out, /phlogiston/);
});

// The turn gets one lookup, so an empty result that suggests searching again
// sends the model into a call the server drops on the floor -- the turn then
// ends with no text and no tool call at all.
test("an empty result does not invite another search", () => {
  const out = formatSectionsForPrompt({ query: "phlogiston", hits: [], missing: [] });
  assert.match(out, /do not try to search again/);
  assert.doesNotMatch(out, /try different terms/);
});

test("a chapter with no query browses that chapter instead of matching nothing", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ chapters: [1], books: ["book-a"] }, { corpus });
  assert.equal(r.browsed, true);
  assert.deepEqual(r.hits.map((h) => `${h.slug}:${h.number}`), ["book-a:1.1", "book-a:1.2"]);
  assert.match(formatSectionsForPrompt(r), /sections in book order/);
});

test("the chapter browse honours max_sections", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ chapters: [1], max_sections: 1 }, { corpus });
  assert.equal(r.hits.length, 1);
});

test("a query that ranks nothing falls back to the requested chapter", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "phlogiston", chapters: [2], books: ["book-a"] }, { corpus });
  assert.equal(r.browsed, true);
  assert.deepEqual(r.hits.map((h) => `${h.slug}:${h.number}`), ["book-a:2.1"]);
});

test("a query that ranks is not replaced by the chapter browse", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ query: "friction", chapters: [1], books: ["book-a"] }, { corpus });
  assert.equal(r.browsed, false);
  assert.equal(r.hits[0].number, "1.2");
});

test("a call with neither query, sections nor chapters still returns no hits", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({}, { corpus });
  assert.equal(r.hits.length, 0);
  assert.equal(r.browsed, false);
});

test("buildCatalog lists chapters with their section ranges", () => {
  const { files } = makeFixture();
  const out = buildCatalog(files["manifest.json"]);
  assert.match(out, /\[book-a\] Book A/);
  assert.match(out, /Ch 1\. Mechanics \(3 sections → 1\.0–1\.2\)/);
  assert.match(out, /\[book-b\] Book B/);
});

test("buildCatalog narrows to books/chapters with a selected section and says so", () => {
  const { files } = makeFixture();
  // book-a chapter 1 "Mechanics" has 3 sections (1.0, 1.1, 1.2); only pinning
  // 1.0 leaves the chapter listed but flagged partial. Chapter 2 "Waves" (2.0,
  // 2.1) has nothing pinned, so it drops out entirely -- same as the old
  // flat chapter pin's on/off behaviour, just per book and per section now.
  const out = buildCatalog(files["manifest.json"], { books: ["book-a"], sections: { "book-a": ["1.0"] } });
  assert.match(out, /Ch 1\. Mechanics.*limited to selected sections/);
  assert.doesNotMatch(out, /Ch 2\. Waves/);
  assert.match(out, /limited to a hand-picked subset/);
  assert.doesNotMatch(out, /Book B/);
});

test("buildCatalog does not flag a chapter that is fully selected", () => {
  const { files } = makeFixture();
  const out = buildCatalog(files["manifest.json"], { sections: { "book-a": ["1.0", "1.1", "1.2"] } });
  assert.match(out, /Ch 1\. Mechanics \(3 sections/);
  assert.doesNotMatch(out, /Ch 1\..*limited to selected sections/);
  // book-b carries no entry in the scope at all, so it stays fully unrestricted.
  assert.match(out, /Ch 1\. Calculus \(2 sections/);
});

test("the context ceiling is a real number other modules can rely on", () => {
  assert.equal(typeof MAX_CONTEXT_CHARS, "number");
  assert.ok(MAX_CONTEXT_CHARS > 0);
});

test("chapterSectionIds numbers from N.0 through N.(count-1)", () => {
  assert.deepEqual(chapterSectionIds({ n: 4, sections: 3 }), ["4.0", "4.1", "4.2"]);
  assert.deepEqual(chapterSectionIds({ n: 9, sections: 1 }), ["9.0"]);
});

test("sectionScope restricts a ranked query to the pinned sections", async () => {
  const { corpus } = makeFixture();
  // "friction" only lives in 1.2, which is outside this pin, so ranking must
  // not reach into it -- and like a chapter pin that ranks nothing, this
  // falls back to browsing the pinned section itself rather than returning
  // nothing at all.
  const excluded = await searchTextbook(
    { query: "friction", books: ["book-a"], sectionScope: { "book-a": ["1.1"] } }, { corpus });
  assert.equal(excluded.browsed, true);
  assert.deepEqual(excluded.hits.map((h) => h.number), ["1.1"]);
  // Widen the pin to include 1.2 and ranking finds it directly, no browse needed.
  const included = await searchTextbook(
    { query: "friction", books: ["book-a"], sectionScope: { "book-a": ["1.1", "1.2"] } }, { corpus });
  assert.deepEqual(included.hits.map((h) => h.number), ["1.2"]);
  assert.equal(included.browsed, false);
});

test("sectionScope hides an explicitly named section outside the pin", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook(
    { section_ids: ["book-a:1.2"], sectionScope: { "book-a": ["1.1"] } }, { corpus });
  assert.equal(r.hits.length, 0);
  assert.deepEqual(r.missing, ["book-a:1.2"]);
});

test("sectionScope alone (no query, no chapters) browses the pinned sections", async () => {
  const { corpus } = makeFixture();
  const r = await searchTextbook({ sectionScope: { "book-a": ["1.1", "2.1"] } }, { corpus });
  assert.equal(r.browsed, true);
  assert.deepEqual(r.hits.map((h) => `${h.slug}:${h.number}`).sort(), ["book-a:1.1", "book-a:2.1"]);
});

test("a chapter request still respects another book's section pin", async () => {
  const { corpus } = makeFixture();
  // Ask for chapter 1 across both books, but book-a is pinned to just 1.2.
  const r = await searchTextbook({ chapters: [1], sectionScope: { "book-a": ["1.2"] } }, { corpus });
  assert.deepEqual(r.hits.map((h) => `${h.slug}:${h.number}`).sort(),
    ["book-a:1.2", "book-b:1.1", "book-b:1.2"]);
});
