#!/usr/bin/env python3
"""
Build a searchable OpenStax textbook corpus for QuestionForge.

Run from the question-forge/ repo root:

    python scripts/build_textbook_corpus.py \
        --book college-physics-2e \
        --book university-physics-volume-1 \
        --output _site/corpus

Output layout (see README "Textbook corpus"):

    <output>/manifest.json          books, chapter lists, attribution, licence
    <output>/LICENSE                CC BY-NC-SA 4.0, scoping the derived corpus
    <output>/<slug>/index.json      stage-1 search index (one record per section)
    <output>/<slug>/chNN.json       stage-2 chapter shards (full section text)

Why this shape
--------------
OpenStax CNXML is *already* chunked and semantically labelled: every book section
is a module, and every top-level <section> inside it carries a class attribute
("learning-objectives", "section-summary", "problems-exercises", ...).  So there
is no chunking problem to solve and no need for embeddings -- the stage-1 index
is a linear scan over a few hundred human-authored summaries, which BM25 handles
in well under a millisecond in the browser.

Term frequencies and document frequencies are computed *here*, not in the
browser.  That keeps the Python/JS tokenizer-parity surface down to "both sides
must agree on a six-word query" instead of "both sides must agree on 5 MB of
prose".  src/textbook_index.js:tokenize() must stay in sync with tokenize()
below; src/textbook_index.test.js pins that with a committed fixture.

Rejected alternatives
---------------------
* Downloading the repo tarball: the College Physics bundle carries 490 MB of
  media/ that we never use.  A blobless sparse clone fetches only the CNXML
  (~17 MB, ~2 s).
* Committing the generated corpus: ~18 MB of generated text per regeneration in
  a ~400 KB repo.  It is built in CI by .github/workflows/pages.yml instead.
* Fetching CNXML from jsDelivr at runtime: moves XML parsing and MathML
  conversion into the browser on every turn and makes question generation depend
  on a third-party CDN.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone

CNX = "{http://cnx.rice.edu/cnxml}"
COL = "{http://cnx.rice.edu/collxml}"
MD = "{http://cnx.rice.edu/mdml}"
MML = "{http://www.w3.org/1998/Math/MathML}"

# slug -> bundle repository holding that collection
BUNDLES = {
    "college-physics-2e": "osbooks-college-physics-bundle",
    "college-physics-ap-courses-2e": "osbooks-college-physics-bundle",
    "university-physics-volume-1": "osbooks-university-physics-bundle",
    "university-physics-volume-2": "osbooks-university-physics-bundle",
    "university-physics-volume-3": "osbooks-university-physics-bundle",
    "calculus-volume-1": "osbooks-calculus-bundle",
    "calculus-volume-2": "osbooks-calculus-bundle",
    "calculus-volume-3": "osbooks-calculus-bundle",
}

CANONICAL_URL = "https://openstax.org/details/books/{slug}"

# Sections longer than this are split on paragraph boundaries.  College Physics
# never hits it (max measured: 23 711 chars); the larger Calculus modules can.
MAX_CHUNK_CHARS = 24000

# CNXML section/@class -> the key we store it under.  Anything unrecognised
# falls through to prose.
SECTION_KINDS = {
    "learning-objectives": "objectives",
    "section-summary": "summary",
    "key-concepts": "summary",
    "key-equations": "equations",
    "conceptual-questions": "conceptual",
    "problems-exercises": "problems",
    "section-exercises": "problems",
    "review-exercises": "problems",
    "ap-test-prep": "problems",
    "glossary": "glossary",
}


# ---------------------------------------------------------------------------
# MathML -> LaTeX
#
# This is the highest-risk part of the build.  The app's whole math pipeline is
# LaTeX in $...$ delimiters: MathJax renders it in the preview, and
# src/qti_export.js:splitMathSegments() looks for exactly those delimiters when
# building the D2L package.  Feeding the model raw MathML (or a "a ∝ F_net"
# style linearisation) teaches it to emit that form instead, which silently
# corrupts both.  So we convert properly, and test it against a committed
# fixture in src/textbook_index.test.js.
#
# Only the presentation subset the OpenStax books actually use is handled.
# Anything unrecognised degrades to its concatenated symbol text rather than
# raising -- a slightly ugly excerpt is recoverable, a failed build is not.
# ---------------------------------------------------------------------------

GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta",
    "θ": r"\theta", "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "µ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "ς": r"\sigma", "τ": r"\tau",
    "υ": r"\upsilon", "φ": r"\phi", "ϕ": r"\phi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

OPS = {
    "∝": r"\propto", "×": r"\times", "⋅": r"\cdot", "·": r"\cdot",
    "−": "-", "–": "-", "—": "-", "±": r"\pm", "∓": r"\mp",
    "≈": r"\approx", "≃": r"\simeq", "≅": r"\cong", "∼": r"\sim",
    "≠": r"\neq", "≤": r"\le", "≥": r"\ge", "≪": r"\ll", "≫": r"\gg",
    "≡": r"\equiv", "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    "°": r"^\circ", "′": r"'", "″": r"''", "⊥": r"\perp", "∥": r"\parallel",
    "∘": r"\circ", "√": r"\sqrt", "∠": r"\angle", "…": r"\ldots",
    "⟨": r"\langle", "⟩": r"\rangle", "ℏ": r"\hbar", "ℓ": r"\ell",
}

FUNCS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "log", "ln", "exp", "lim", "max", "min",
    "det", "gcd", "sup", "inf",
}

ACCENTS = {"¯": r"\bar", "^": r"\hat", "→": r"\vec", "˙": r"\dot", "~": r"\tilde"}

_TEX_ESCAPE = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex_escape(s):
    return "".join(_TEX_ESCAPE.get(c, c) for c in s)


def _local(elem):
    return elem.tag.split("}")[-1] if isinstance(elem.tag, str) else ""


def _map_chars(s):
    """Map Greek letters and operator glyphs inside a run of <mi>/<mo> text."""
    return "".join(GREEK.get(c, OPS.get(c, c)) for c in s)


def _map_greek_only(s):
    """Leave text alone but spell out any stray Greek glyphs it contains."""
    return "".join(GREEK.get(c, c) for c in s)


def mathml_to_latex(elem):
    """Convert a presentation-MathML element to a LaTeX fragment (no $ delimiters)."""
    tag = _local(elem)
    text = (elem.text or "").strip()
    kids = [k for k in elem if isinstance(k.tag, str)]

    def sub(i):
        return mathml_to_latex(kids[i]) if i < len(kids) else ""

    def joined():
        parts = [mathml_to_latex(k) for k in kids]
        return " ".join(p for p in parts if p)

    if tag == "mi":
        if not text:
            return ""
        if text in GREEK:
            out = GREEK[text]
        elif text.lower() in FUNCS:
            out = "\\" + text.lower()
        elif len(text) == 1:
            out = _map_chars(text)
        else:
            out = rf"\mathrm{{{_tex_escape(text)}}}"
        return _wrap_variant(elem, out)

    if tag == "mn":
        return _wrap_variant(elem, _map_chars(text))

    if tag == "mo":
        if not text:
            return ""
        return _wrap_variant(elem, OPS.get(text, _map_chars(text) if text in GREEK else _tex_escape(text)))

    if tag == "mtext":
        if not text:
            return ""
        # OpenStax routinely puts *variables* in <mtext>, e.g.
        # <mtext mathvariant="bold">a</mtext> for the acceleration vector, and
        # <mtext>Δ</mtext> for the delta operator.  Rendering those as \text{}
        # gives upright \mathbf{\text{a}} where the book shows an italic bold
        # a.  Single glyphs are therefore treated as identifiers; only genuine
        # multi-character labels ("net", "max") stay as \text{}.
        if len(text) == 1:
            return _wrap_variant(elem, GREEK.get(text, OPS.get(text, _tex_escape(text))))
        if text in GREEK:
            return _wrap_variant(elem, GREEK[text])
        if text.lower() in FUNCS:
            return _wrap_variant(elem, "\\" + text.lower())
        return _wrap_variant(elem, rf"\text{{{_tex_escape(_map_greek_only(text))}}}")

    if tag == "mspace":
        return r"\;"

    if tag in ("math", "mrow", "mstyle", "mpadded", "menclose", "semantics"):
        if tag == "semantics" and kids:
            # first child is the presentation form; <annotation> follows
            return _wrap_variant(elem, mathml_to_latex(kids[0]))
        return _wrap_variant(elem, joined())

    if tag == "msub":
        return f"{_brace(sub(0))}_{{{sub(1)}}}"
    if tag == "msup":
        return f"{_brace(sub(0))}^{{{sub(1)}}}"
    if tag == "msubsup":
        return f"{_brace(sub(0))}_{{{sub(1)}}}^{{{sub(2)}}}"
    if tag == "mfrac":
        return rf"\frac{{{sub(0)}}}{{{sub(1)}}}"
    if tag == "msqrt":
        return rf"\sqrt{{{joined()}}}"
    if tag == "mroot":
        return rf"\sqrt[{sub(1)}]{{{sub(0)}}}"

    if tag in ("mover", "munder", "munderover"):
        base = sub(0)
        mark = (kids[1].text or "").strip() if len(kids) > 1 else ""
        if tag == "mover" and mark in ACCENTS:
            return f"{ACCENTS[mark]}{{{base}}}"
        if tag == "mover":
            return rf"\overset{{{sub(1)}}}{{{_brace(base)}}}"
        if tag == "munder":
            return rf"\underset{{{sub(1)}}}{{{_brace(base)}}}"
        return f"{_brace(base)}_{{{sub(1)}}}^{{{sub(2)}}}"

    if tag == "mfenced":
        inner = " , ".join(mathml_to_latex(k) for k in kids)
        return r"\left{} {} \right{}".format(_delim(elem.get("open", "(")), inner, _delim(elem.get("close", ")")))

    if tag == "mtable":
        rows = [" & ".join(mathml_to_latex(c) for c in r if isinstance(c.tag, str)) for r in kids]
        return r"\begin{{matrix}} {} \end{{matrix}}".format(r" \\ ".join(rows))
    if tag in ("mtr", "mtd"):
        return joined()

    if tag == "annotation":
        return ""

    return joined()


def _delim(ch):
    """A \\left/\\right delimiter.  Braces must be escaped; "" means none."""
    if not ch:
        return "."
    return {"{": r"\{", "}": r"\}", "|": r"|"}.get(ch, ch)


def _brace(s):
    """Brace a sub-expression unless it is already a single token."""
    if len(s) <= 1 or re.fullmatch(r"\\[A-Za-z]+", s) or re.fullmatch(r"\{.*\}", s):
        return s
    return f"{{{s}}}"


def _wrap_variant(elem, latex):
    if not latex:
        return ""
    variant = elem.get("mathvariant", "")
    # Bold is meaningful in these books: it marks vectors.  Keep it.
    if variant == "bold":
        return rf"\mathbf{{{latex}}}"
    if variant == "bold-italic":
        return rf"\boldsymbol{{{latex}}}"
    return latex


def math_to_tex(elem, display=False):
    """Convert a <m:math> element to a delimited LaTeX string."""
    body = re.sub(r"\s+", " ", mathml_to_latex(elem)).strip()
    # The books habitually close an equation with the sentence's punctuation
    # *inside* the math ("F = ma \text{.}").  Drop it: the surrounding prose
    # already carries it, and it reads as noise to the model.
    body = re.sub(r"(?:\s*(?:\\text\{[.,;:]\}|[.,;:]))+$", "", body).strip()
    # The books encode "0.13" as <mn>0</mn><mo>.</mo><mn>13</mn>, which comes out
    # as "0 . 13" and reads as three tokens.  Rejoin decimals and thousands.
    body = re.sub(r"(?<=\d) \. (?=\d)", ".", body)
    body = re.sub(r"(?<=\d) , (?=\d\d\d\b)", ",", body)
    if not body:
        return ""
    return (f"$${body}$$") if display else (f"${body}$")


# ---------------------------------------------------------------------------
# CNXML -> structured section records
# ---------------------------------------------------------------------------

def _norm(s):
    out = re.sub(r"[ \t]+", " ", (s or ""))
    # <link target-id="..."/> cross-references carry no text, so dropping them
    # leaves "as shown in ." behind.  Pull punctuation back onto the word.
    return re.sub(r"(\w)\s+([.,;:])", r"\1\2", out).strip()


# Element ids -> a readable noun, rebuilt per module by extract_module().
# CNXML cross-references are empty elements (<link target-id="fs-id123"/>), so
# dropping them leaves "as shown in ." and "(see )." all through the worked
# examples -- the highest-value text in the book.  Naming the target's element
# type restores a readable sentence without resolving figure numbers.
_LINK_LABELS = {
    "figure": "the figure", "table": "the table", "example": "the example",
    "equation": "the equation", "section": "the section", "note": "the note",
    "exercise": "the exercise", "list": "the list",
}
_link_targets = {}


def _index_link_targets(root):
    targets = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        eid = el.get("id")
        if eid:
            label = _LINK_LABELS.get(_local(el))
            if label:
                targets[eid] = label
    return targets


def inline_text(elem, skip_tags=()):
    """Flatten an element to text: math becomes $...$, markup is dropped.

    <image alt="..."> is deliberately *not* included.  The alt text runs to
    ~1 000 characters per figure ("A boy in a wagon is pushed by two girls
    toward the right...") and accounts for roughly a fifth of a section's
    bytes.  Worse, it makes the model write questions about pictures that the
    exam will not contain.  Figure *captions* are kept; alt text is not.
    """
    out = []
    if elem.text:
        out.append(elem.text)
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        name = _local(child)
        if child.tag == MML + "math":
            out.append(math_to_tex(child))
        elif name == "link" and not (child.text or "").strip() and len(child) == 0:
            out.append(_link_targets.get(child.get("target-id", ""), "the figure"))
        elif name in ("media", "image", "iframe") or name in skip_tags:
            pass
        else:
            out.append(inline_text(child, skip_tags))
        if child.tail:
            out.append(child.tail)
    return "".join(out)


def _para_text(para):
    """A <para>, rendered.  A <para> may carry a <title> -- inside <example>,
    the "Strategy" / "Solution" / "Discussion" headings are *not* sections but
    <para><title>Strategy</title>...</para> siblings.  Without this they come
    out as orphan one-word lines."""
    title = para.find(CNX + "title")
    body = _norm(inline_text(para, skip_tags=("title",)))
    if title is not None:
        label = _norm(inline_text(title))
        if label:
            return f"**{label}** {body}" if body else f"**{label}**"
    return body


def render_blocks(elem, skip=()):
    """Render the block-level children of an element to markdown-ish text."""
    parts = []
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        name = _local(child)
        if name in skip:
            continue
        if name == "para":
            parts.append(_para_text(child))
        elif name == "title":
            parts.append("### " + _norm(inline_text(child)))
        elif name == "list":
            bullet = "1." if child.get("list-type") == "enumerated" else "-"
            for item in child.findall(CNX + "item"):
                parts.append(f"{bullet} {_norm(inline_text(item))}")
        elif name == "equation":
            math = child.find(MML + "math")
            if math is not None:
                tex = math_to_tex(math, display=True)
                if tex:
                    parts.append(tex)
        elif name == "figure":
            cap = child.find(CNX + "caption")
            cap_text = _norm(inline_text(cap)) if cap is not None else ""
            if cap_text:
                parts.append(f"[Figure: {cap_text}]")
        elif name == "table":
            cap = child.find(CNX + "caption")
            cap_text = _norm(inline_text(cap)) if cap is not None else ""
            parts.append(f"[Table: {cap_text}]" if cap_text else "[Table]")
        elif name == "note":
            inner = render_blocks(child)
            if inner:
                parts.append("\n".join("> " + ln for ln in inner.split("\n") if ln))
        elif name in ("section", "example", "exercise", "glossary"):
            continue  # handled by the caller
        else:
            txt = _norm(inline_text(child))
            if txt:
                parts.append(txt)
    return "\n".join(p for p in parts if p)


def extract_example(elem):
    """A worked example, in either of the two shapes the books use.

    College Physics writes <example><title/><para/>..., with "Strategy" and
    "Solution" as titled <para>s.  Calculus wraps the whole thing in an
    exercise: <example><exercise><problem><title/>...</problem><solution/>.
    render_blocks() skips <exercise> (it belongs to the problem sets), so the
    second shape yields an empty example unless it is unwrapped here.
    """
    title_el = elem.find(CNX + "title")
    title = _norm(inline_text(title_el)) if title_el is not None else ""

    inner = elem.find(CNX + "exercise")
    if inner is None:
        return {"title": title, "text": render_blocks(elem, skip=("title",))}

    problem = inner.find(CNX + "problem")
    solution = inner.find(CNX + "solution")
    if not title and problem is not None:
        ptitle = problem.find(CNX + "title")
        if ptitle is not None:
            title = _norm(inline_text(ptitle))
    parts = []
    if problem is not None:
        parts.append(render_blocks(problem, skip=("title",)))
    if solution is not None:
        sol = render_blocks(solution, skip=("title",))
        if sol:
            parts.append("**Solution** " + sol)
    return {"title": title, "text": "\n".join(p for p in parts if p)}


def extract_exercise(elem, n):
    problem = elem.find(CNX + "problem")
    solution = elem.find(CNX + "solution")
    text = ""
    if problem is not None:
        ptitle = problem.find(CNX + "title")
        body = render_blocks(problem, skip=("title",))
        label = _norm(inline_text(ptitle)) if ptitle is not None else ""
        text = (f"**{label}** {body}").strip() if label else body
    return {
        "n": n,
        "problem": text,
        "solution": render_blocks(solution) if solution is not None else "",
    }


def extract_glossary(elem):
    out = []
    for d in elem.findall(CNX + "definition"):
        term = d.find(CNX + "term")
        meaning = d.find(CNX + "meaning")
        if term is None:
            continue
        out.append({
            "term": _norm(inline_text(term)),
            "meaning": _norm(inline_text(meaning)) if meaning is not None else "",
        })
    return out


def _collect(elem, record):
    """Pull examples, exercises and equations out of an arbitrary subtree."""
    for ex in elem.iter(CNX + "example"):
        record["examples"].append(extract_example(ex))
    for eq in elem.iter(CNX + "equation"):
        math = eq.find(MML + "math")
        if math is not None:
            tex = math_to_tex(math, display=True)
            if tex:
                record["equations"].append(tex)


def _walk_sections(elem, rec, body_parts, level=2):
    """Recurse the <section> tree, routing each one by its class.

    The classed sections are not always top-level: 36 College Physics modules
    wrap "Section Summary" / "Conceptual Questions" / "Problem Exercises" inside
    an *unclassed* section (m42073 is one), so a findall() over content's direct
    children silently loses every exercise in them.
    """
    for sec in elem.findall(CNX + "section"):
        kind = SECTION_KINDS.get(sec.get("class") or "", "prose")
        title_el = sec.find(CNX + "title")
        title = _norm(inline_text(title_el)) if title_el is not None else ""

        if kind == "objectives":
            rec["objectives"] += [
                _norm(inline_text(i)) for i in sec.iter(CNX + "item") if _norm(inline_text(i))
            ]
        elif kind == "summary":
            text = render_blocks(sec, skip=("title",))
            if text:
                rec["summary"] = (rec["summary"] + "\n" + text).strip() if rec["summary"] else text
        elif kind in ("problems", "conceptual"):
            bucket = rec["problems"] if kind == "problems" else rec["conceptual"]
            # .iter(): two modules nest their exercises one level deeper still.
            for ex in sec.iter(CNX + "exercise"):
                bucket.append(extract_exercise(ex, len(bucket) + 1))
        elif kind == "glossary":
            rec["glossary"] += extract_glossary(sec)
        elif kind == "equations":
            text = render_blocks(sec, skip=("title",))
            if text:
                body_parts.append("{} {}\n{}".format("#" * level, title or "Key Equations", text))
        else:
            text = render_blocks(sec, skip=("title",))
            head = "{} {}".format("#" * level, title) if title else ""
            chunk = "\n".join(x for x in (head, text) if x)
            if chunk:
                body_parts.append(chunk)
            _walk_sections(sec, rec, body_parts, min(level + 1, 5))


def extract_module(path, module_id):
    """Parse one index.cnxml into a structured section record."""
    global _link_targets
    root = ET.parse(path).getroot()
    _link_targets = _index_link_targets(root)
    title_el = root.find(CNX + "title")
    content = root.find(CNX + "content")
    if content is None:
        return None

    rec = {
        "moduleId": module_id,
        "title": _norm(inline_text(title_el)) if title_el is not None else module_id,
        "objectives": [],
        "summary": "",
        "body": "",
        "equations": [],
        "examples": [],
        "problems": [],
        "conceptual": [],
        "glossary": [],
        "figures": [],
        "terms": [],
    }

    body_parts = [render_blocks(content)]
    _walk_sections(content, rec, body_parts)

    # Exercises sitting directly under <content>, outside any section.
    for ex in content.findall(CNX + "exercise"):
        rec["problems"].append(extract_exercise(ex, len(rec["problems"]) + 1))

    # <glossary> is a sibling of <content> under <document>, not a child of it.
    for gl in root.iter(CNX + "glossary"):
        rec["glossary"] += extract_glossary(gl)

    # Examples and equations are gathered once, from the whole module, so that
    # nesting depth cannot duplicate or drop them.
    _collect(content, rec)

    rec["figures"] = [
        _norm(inline_text(c)) for c in content.iter(CNX + "caption") if _norm(inline_text(c))
    ]

    # <term> appears both inline in prose and inside <definition>; take both.
    seen = set()
    for t in list(content.iter(CNX + "term")) + [
        g for gl in root.iter(CNX + "glossary") for g in gl.iter(CNX + "term")
    ]:
        val = _norm(inline_text(t))
        if val and len(val) < 60 and val.lower() not in seen:
            seen.add(val.lower())
            rec["terms"].append(val)

    rec["body"] = re.sub(r"\n{3,}", "\n\n", "\n".join(p for p in body_parts if p)).strip()
    rec["equations"] = list(dict.fromkeys(rec["equations"]))[:40]
    seen_ex = set()
    uniq = []
    for e in rec["examples"]:
        key = (e["title"], e["text"][:120])
        if key not in seen_ex:
            seen_ex.add(key)
            uniq.append(e)
    rec["examples"] = uniq
    return rec


# ---------------------------------------------------------------------------
# Collection (book) structure
# ---------------------------------------------------------------------------

def parse_collection(path):
    """Read a .collection.xml into {title, license, chapters[], frontmatter[]}.

    The collection file holds chapter titles and bare module references only --
    section titles live inside each module -- so the table of contents can only
    be completed after every module has been read.
    """
    root = ET.parse(path).getroot()
    meta = root.find(COL + "metadata")
    title = _norm(meta.findtext(MD + "title")) if meta is not None else path.stem
    lic_el = meta.find(MD + "license") if meta is not None else None
    book = {
        "title": title,
        "license": _norm(lic_el.text) if lic_el is not None else "",
        "license_url": (lic_el.get("url") if lic_el is not None else "") or "",
        "chapters": [],
        "frontmatter": [],
    }

    def walk(node, chapter):
        for child in node:
            name = _local(child)
            if name == "subcollection":
                sub_title = _norm(child.findtext(MD + "title") or "")
                content = child.find(COL + "content")
                nested = content is not None and any(
                    _local(c) == "subcollection" for c in content
                )
                if nested:
                    # A part/unit wrapper (e.g. "Thermodynamics" grouping
                    # "Temperature and Heat", "The Second Law of
                    # Thermodynamics", ...) -- not a chapter itself, so it
                    # must not consume a chapter number or get indexed as an
                    # empty one. Unwrap it and keep walking its real chapters.
                    walk(content, chapter)
                else:
                    book["chapters"].append({"title": sub_title, "modules": []})
                    walk(content, book["chapters"][-1])
            elif name == "module":
                mid = child.get("document")
                if not mid:
                    continue
                (chapter["modules"] if chapter else book["frontmatter"]).append(mid)

    walk(root.find(COL + "content"), None)
    return book


def assign_numbers(book):
    """Give every module an N.M section number.

    The chapter opener ("Introduction to ...") becomes N.0 and the rest count
    from N.1, which is how OpenStax numbers them on the web.
    """
    out = []
    for ci, chapter in enumerate(book["chapters"], start=1):
        for mi, mid in enumerate(chapter["modules"]):
            out.append({
                "moduleId": mid,
                "chapter": ci,
                "chapterTitle": chapter["title"],
                "number": f"{ci}.{mi}",
                "shard": f"ch{ci:02d}",
            })
    return out


def clone_bundle(repo, cache_dir):
    """Blobless sparse clone: CNXML only, never the media/ tree.

    The College Physics bundle carries 490 MB of images we have no use for.
    --filter=blob:none plus a sparse checkout of modules/ and collections/
    fetches ~17 MB in under two seconds.
    """
    dest = cache_dir / repo
    if (dest / ".git").exists():
        return dest
    url = f"https://github.com/openstax/{repo}"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "modules", "collections"],
        cwd=dest, check=True, stdout=subprocess.DEVNULL,
    )
    return dest


def bundle_sha(path):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


# ---------------------------------------------------------------------------
# Search index
#
# Term and document frequencies are computed here so the browser only ever
# tokenises the *query*.  src/textbook_index.js:tokenize() must stay in sync
# with this function; src/textbook_index.test.js pins that against a fixture.
# ---------------------------------------------------------------------------

# SIM905 is suppressed below: a wrapped word list reads far better than 60
# quoted literals, and splitting a module-level constant once at import is free.
STOPWORDS = frozenset("""
a an and are as at be been but by can for from had has have how in into is it its
may more most not of on or that the their them then there these they this to was
were what when where which who will with would you your
""".split())  # noqa: SIM905

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Field weights, folded into the stored term frequencies at build time.
FIELD_WEIGHTS = {"title": 3, "objectives": 2, "terms": 2, "summary": 2, "body": 1}

# The books' own vocabulary drifts between chapters; a short curated map covers
# the cases where an instructor's phrasing and the book's differ.
SYNONYMS = {
    "moment of inertia": "rotational inertia",
    "rotational inertia": "moment of inertia",
    "centripetal": "circular motion uniform",
    "emf": "electromotive force",
    "electromotive force": "emf",
    "spring constant": "hooke law elastic",
    "simple harmonic": "oscillation periodic shm",
    "projectile": "trajectory two dimensional motion",
    "free body diagram": "force diagram newton",
    "kinetic friction": "friction coefficient sliding",
    "terminal velocity": "drag air resistance",
    "ideal gas": "pressure volume temperature mole",
    "specific heat": "calorimetry thermal energy",
    "antiderivative": "indefinite integral",
    "related rates": "implicit differentiation",
    "riemann sum": "definite integral area",
}


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stopwords and short tokens."""
    out = []
    for tok in TOKEN_RE.findall((text or "").lower()):
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
            tok = tok[:-1]
        if len(tok) >= 3:
            out.append(tok)
    return out


def section_tf(rec):
    """Weighted term frequencies for one section, plus its weighted length."""
    fields = {
        "title": rec["title"],
        "objectives": " ".join(rec["objectives"]),
        "terms": " ".join(rec["terms"]) + " " + " ".join(g["term"] for g in rec["glossary"]),
        "summary": rec["summary"],
        "body": rec["body"],
    }
    tf = Counter()
    for field, text in fields.items():
        weight = FIELD_WEIGHTS[field]
        for tok in tokenize(text):
            tf[tok] += weight

    lowered = (rec["title"] + " " + rec["summary"] + " " + " ".join(rec["terms"])).lower()
    for phrase, expansion in SYNONYMS.items():
        if phrase in lowered:
            for tok in tokenize(expansion):
                tf[tok] += FIELD_WEIGHTS["terms"]

    total = sum(tf.values())

    # Keep the top terms, but never drop a title or key-term token: those are
    # the ones a targeted query is most likely to use.
    protected = set(tokenize(fields["title"])) | set(tokenize(fields["terms"]))
    keep = {t for t, _ in tf.most_common(60)} | protected
    return {t: c for t, c in tf.items() if t in keep}, total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CC_BY_NC_SA = """This directory contains text derived from OpenStax textbooks.

The derived corpus is made available under the same licence as its source,
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0): https://creativecommons.org/licenses/by-nc-sa/4.0/

Per-book attribution, canonical URLs and the exact upstream commit each book was
built from are recorded in manifest.json.

This licence covers the contents of this directory only.  The QuestionForge
application code is licensed separately; see the repository root.
"""


def _cap_body(body):
    """Cap a section's prose on a paragraph boundary.

    Sections are left whole rather than split into synthetic sub-records: their
    numbers come from the book, and inventing "1.2a" would produce citations a
    reader cannot look up.  Only the prose is capped -- summary, equations,
    examples, problems and glossary are stored separately and never truncated --
    and the opening paragraphs, where a section establishes its notation and
    definitions, are what survive.
    """
    if len(body) <= MAX_CHUNK_CHARS:
        return body, False
    cut = body.rfind("\n", 0, MAX_CHUNK_CHARS)
    return body[: cut if cut > MAX_CHUNK_CHARS // 2 else MAX_CHUNK_CHARS].rstrip(), True


def build_book(slug, repo_dir, report=False):
    col_path = repo_dir / "collections" / (f"{slug}.collection.xml")
    if not col_path.exists():
        raise SystemExit(f"no collection file for {slug!r} at {col_path}")
    book = parse_collection(col_path)
    entries = assign_numbers(book)

    index_sections = []
    shards = {}
    df = Counter()
    total_len = 0
    truncated = 0

    for entry in entries:
        path = repo_dir / "modules" / entry["moduleId"] / "index.cnxml"
        if not path.exists():
            print("  WARNING: missing module {}".format(entry["moduleId"]), file=sys.stderr)
            continue
        rec = extract_module(path, entry["moduleId"])
        if rec is None:
            continue
        rec["body"], was_cut = _cap_body(rec["body"])
        truncated += was_cut
        rec.update({
            "id": entry["number"],
            "number": entry["number"],
            "chapter": entry["chapter"],
            "chapterTitle": entry["chapterTitle"],
        })

        tf, weighted_len = section_tf(rec)
        for term in tf:
            df[term] += 1
        total_len += weighted_len

        index_sections.append({
            "id": entry["number"],
            "m": entry["moduleId"],
            "sh": entry["shard"],
            "ch": entry["chapter"],
            "chT": entry["chapterTitle"],
            "t": rec["title"],
            "o": rec["objectives"],
            "s": _norm(rec["summary"].replace("\n", " "))[:400],
            "k": rec["terms"][:14],
            "n": {
                "w": len(rec["body"].split()),
                "e": len(rec["examples"]),
                "q": len(rec["conceptual"]),
                "p": len(rec["problems"]),
            },
            "len": weighted_len,
            "tf": tf,
        })
        shards.setdefault(entry["shard"], {})[entry["number"]] = rec

    attribution = ("OpenStax, {}. Access for free at https://openstax.org/details/books/{}".format(book["title"], slug))
    index = {
        "slug": slug,
        "title": book["title"],
        "schemaVersion": 1,
        "docCount": len(index_sections),
        "avgLen": round(total_len / len(index_sections), 2) if index_sections else 0,
        "df": dict(sorted(df.items())),
        "sections": index_sections,
    }
    meta = {
        "slug": slug,
        "title": book["title"],
        "license": book["license"],
        "license_url": book["license_url"],
        "attribution": attribution,
        "canonical_url": CANONICAL_URL.format(slug=slug),
        "sections": len(index_sections),
        "chapters": [
            {"n": i, "title": c["title"], "sections": len(c["modules"])}
            for i, c in enumerate(book["chapters"], start=1)
        ],
    }
    if report and truncated:
        print(f"  {truncated} section(s) capped at {MAX_CHUNK_CHARS} chars")
    return index, shards, meta


def write_json(path, obj, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    else:
        text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return len(text.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", action="append", required=True,
                    help="OpenStax collection slug (repeatable). Known: {}".format(", ".join(sorted(BUNDLES))))
    ap.add_argument("--output", required=True, help="output directory, e.g. _site/corpus")
    ap.add_argument("--cache-dir", default=".cache/osbooks",
                    help="where bundle clones are kept between runs")
    ap.add_argument("--report", action="store_true", help="print per-book size figures")
    args = ap.parse_args()

    unknown = [b for b in args.book if b not in BUNDLES]
    if unknown:
        raise SystemExit("unknown book slug(s): {}\nknown: {}".format(", ".join(unknown), ", ".join(sorted(BUNDLES))))

    out = pathlib.Path(args.output)
    cache = pathlib.Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    books_meta = []
    for slug in args.book:
        repo = BUNDLES[slug]
        print(f"[{slug}] cloning {repo}…")
        repo_dir = clone_bundle(repo, cache)
        print(f"[{slug}] extracting…")
        index, shards, meta = build_book(slug, repo_dir, report=args.report)
        meta["sourceCommit"] = bundle_sha(repo_dir)
        meta["sourceRepo"] = f"openstax/{repo}"

        book_dir = out / slug
        idx_bytes = write_json(book_dir / "index.json", index)
        shard_bytes = sum(
            write_json(book_dir / (f"{shard}.json"), {
                "slug": slug,
                "attribution": meta["attribution"],
                "license": meta["license"],
                "sections": sections,
            })
            for shard, sections in sorted(shards.items())
        )
        books_meta.append(meta)
        print(f"  {len(index['sections'])} sections, {len(meta['chapters'])} chapters"
              f" | index {idx_bytes / 1024:.0f} KB, shards {shard_bytes / 1e6:.1f} MB")
        if args.report:
            import gzip
            raw = (book_dir / "index.json").read_bytes()
            print("  index.json gzipped: %.0f KB" % (len(gzip.compress(raw)) / 1024))

    write_json(out / "manifest.json", {
        "schemaVersion": 1,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "books": books_meta,
    }, pretty=True)
    (out / "LICENSE").write_text(CC_BY_NC_SA, encoding="utf-8")
    print(f"Wrote {len(books_meta)} book(s) to {out}")


if __name__ == "__main__":
    main()
