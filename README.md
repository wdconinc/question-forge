# QuestionForge — ExamForge Web App

A **privacy-first, single-page in-browser exam authoring tool** for PHYS 1020 (and similar courses). Uses [Pyodide](https://pyodide.org/) to run Python question-generation code directly in your browser. No server required. All data stays in `localStorage`.

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

## Privacy

See [PRIVACY.md](PRIVACY.md). All processing happens in your browser. No data is ever sent to any server.

## Colors

University of Manitoba colors: **#f0ab00** gold, **#1e3a5f** navy.

## License

MIT — See [LICENSE](LICENSE) if present.

---

*Built with ❤️ using Pyodide, CodeMirror 6, MathJax 3, JSZip, and SortableJS.*
*As your friendly Caltech PhD physicist would say: all the physics, none of the servers!*
