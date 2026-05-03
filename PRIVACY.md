# Privacy Statement — QuestionForge

## Summary

**All data processing happens entirely in your browser. No data ever leaves your device.**

## Details

- **No server-side processing.** QuestionForge is a static single-page application hosted on GitHub Pages. It has no backend, no API endpoints, and no database.
- **No data transmission.** Your exam questions, templates, Python code, and settings are never sent to any external server. They never leave your browser.
- **localStorage only.** All application state (questions, settings, seeds) is stored in your browser's `localStorage` under the key `questionforge_state`. This data stays on your device.
- **Pyodide runs locally.** The Python execution environment (Pyodide) is downloaded from a CDN and runs entirely within your browser's JavaScript engine. Your Python question-generation code is executed locally, not on any remote server.
- **CDN resources.** The following libraries are loaded from CDNs on first use:
  - Pyodide (cdn.jsdelivr.net) — Python runtime
  - CodeMirror (esm.sh) — code editors
  - MathJax (cdn.jsdelivr.net) — math typesetting
  - JSZip (cdn.jsdelivr.net) — ZIP file creation
  - FileSaver.js (cdn.jsdelivr.net) — file downloads
  - SortableJS (cdn.jsdelivr.net) — drag-to-reorder UI
  These CDNs may log access requests (IP address, timestamp) per their own privacy policies. No question content is transmitted to them.
- **Exports are local.** When you click "Export MD" or "Export ZIP", files are generated in your browser and downloaded directly to your device. No copy is retained anywhere else.
- **Clearing your data.** To delete all stored data, clear `localStorage` for this site in your browser settings, or use the browser's developer tools to remove the `questionforge_state` key.

## Contact

This tool is maintained by the Department of Physics & Astronomy, University of Manitoba.
