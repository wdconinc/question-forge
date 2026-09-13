# QuestionForge — ExamForge Web App

A **privacy-first, single-page in-browser exam authoring tool** for PHYS 1020 (and similar courses). Uses [Pyodide](https://pyodide.org/) to run Python question-generation code directly in your browser. Question authoring, preview, and export run entirely in your browser, with state saved to `localStorage`. An optional AI Chat assistant sends question content to a runner server and on to Google's Gemini API — see [PRIVACY.md](PRIVACY.md).

🔗 **Live app:** https://wdconinc.github.io/question-forge/

---

## Features

- **30 pre-loaded PHYS 1020 questions** — units, kinematics, Newton's laws, energy, momentum, rotation, SHM, fluids, thermodynamics
- **CodeMirror editors** — syntax-highlighted Jinja2 template and Python generator editors per question
- **Live preview** — renders each question with MathJax math typesetting via Pyodide
- **Multi-paper support** — generate papers A, B, C with different seeds; balanced answer-position distribution
- **Export Markdown** — download ZIP of `exam_A.md`, `exam_B.md`, etc.
- **Export ZIP** — download full Python project (render.py + questions/) for offline use
- **Import ZIP** — load questions from a ZIP archive
- **Drag-to-reorder** questions in the sidebar with SortableJS
- **Persistent state** — all edits saved to `localStorage`

## Usage

1. Open https://wdconinc.github.io/question-forge/ — Pyodide loads in the browser (~30 s first time)
2. Click a question in the sidebar to edit its Jinja2 template and Python generator
3. Click **▶ Preview** to run the question and see the rendered output
4. Adjust seeds for papers A/B/C in the top bar
5. Click **▶ Render All** to generate all enabled papers
6. Click **⬇ Export MD** to download exam Markdown files
7. Click **⬇ Export ZIP** to download the full Python project

## Building the question bank locally

```bash
python scripts/build_default_bank.py \
    --source /path/to/questions \
    --output src/default_bank.js
```

## Textbook corpus

The AI grounds generated questions in the OpenStax books the courses are taught
from, so questions match the book's notation, level and problem style and can
cite a real section number.

The corpus is **generated at deploy time** by
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) and is not checked
in — seven books come to ~17 MB, which does not belong in a ~400 KB repository.
To build it locally:

```bash
python scripts/build_textbook_corpus.py --book college-physics-2e --output corpus --report
```

Add `--book` once per book. Available slugs: `college-physics-2e`,
`college-physics-ap-courses-2e`, `university-physics-volume-1` (and `-2`, `-3`),
`calculus-volume-1` (and `-2`, `-3`). The script needs only the Python standard
library and `git`; it fetches CNXML with a blobless sparse clone, so it never
downloads the ~490 MB of book images.

Retrieval runs entirely in the browser: a compact per-book index is ranked with
BM25, and only the matching chapter shards are fetched. Nothing about your
questions is sent anywhere to search the textbook.

Note that `corpus/` is loaded with `fetch()`, so — like the existing lazy module
imports — it needs a real HTTP server. Use `python3 -m http.server 8420` rather
than opening `index.html` from disk.

## Privacy

See [PRIVACY.md](PRIVACY.md). Authoring, previewing, and exporting happen entirely in your browser. The optional AI Chat assistant sends your question content to a runner server, which forwards it to Google's Gemini API.

## Colors

University of Manitoba colors: **#f0ab00** gold, **#1e3a5f** navy.

## License

The application code is MIT — see [LICENSE](LICENSE) if present.

**The generated textbook corpus is licensed separately.** It is derived from
OpenStax textbooks, which are published under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and the
derivative carries the same licence. `corpus/LICENSE` states this, and
`corpus/manifest.json` records per-book attribution, canonical URLs and the
exact upstream commit each book was built from. QuestionForge is a free academic
tool, which keeps it within the NonCommercial term; a commercial fork would
inherit that restriction.

Questions the AI generates are original, parametrized problems *informed by* the
text — the same relationship as an instructor who has read the chapter — and the
system prompt explicitly forbids reproducing textbook problems verbatim. They are
not intended as derivative works of the book, but if you paste textbook prose
into a question yourself, the book's licence follows it.

---

*Built with ❤️ using Pyodide, CodeMirror 6, MathJax 3, JSZip, and SortableJS.*
*As your friendly Caltech PhD physicist would say: all the physics, none of the servers!*
