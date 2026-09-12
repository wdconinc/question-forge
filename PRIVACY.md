# Privacy Statement — QuestionForge

## Summary

**QuestionForge has two modes. Authoring, previewing, and exporting questions happen entirely in your browser. The optional AI Chat assistant sends your question content to a runner server, which forwards it to Google's Gemini API.**

## Authoring, previewing, and exporting

- **No server-side processing.** The QuestionForge app is a static single-page application hosted on GitHub Pages. Authoring, previewing, rendering, and exporting run entirely in your browser, with no backend involved. GitHub logs requests for the page itself, per its own privacy policy.
- **localStorage only.** All application state (questions, settings, seeds) is stored in your browser's `localStorage` under the key `questionforge_state`. AI Chat settings and your chat history are stored separately under `questionForge_ai`. This data stays on your device.
- **Pyodide runs locally.** The Python execution environment (Pyodide) is downloaded from a CDN and runs entirely within your browser's JavaScript engine. Your Python question-generation code is executed locally when you preview or render.
- **CDN resources.** The following libraries are loaded from `cdn.jsdelivr.net` when the page loads, each pinned with a Subresource Integrity hash:
  - Pyodide — Python runtime
  - CodeMirror 5 — code editors
  - MathJax — math typesetting
  - marked — Markdown rendering
  - DOMPurify — HTML sanitizing
  - JSZip — ZIP file creation
  - FileSaver.js — file downloads
  - SortableJS — drag-to-reorder UI
  Pyodide additionally downloads its WebAssembly runtime and the `numpy`, `micropip`, and `jinja2` packages from the same CDN when Python first starts. This CDN may log access requests (IP address, timestamp) per its own privacy policy. No question content is transmitted to it.
- **Exports are local.** When you click "Export MD" or "Export ZIP", files are generated in your browser and downloaded directly to your device. No copy is retained anywhere else.
- **Diagnostic logs stay on your device too.** QuestionForge keeps a rolling, in-memory log of console messages and errors to help troubleshoot issues. This log is never sent anywhere automatically; it is cleared when you close or reload the page. Clicking "Export Logs" writes it to a `.txt` file on your device, which you can choose to attach to a bug report. The app makes a best-effort attempt to redact obvious secrets (e.g. the AI connection token) from the log, but you should still review the file before sharing it, since it may include content you typed (question text, AI chat messages).
- **Clearing your data.** To delete all stored data, clear `localStorage` for this site in your browser settings, or remove the `questionforge_state` and `questionForge_ai` keys with the browser's developer tools.

## AI Chat

- **The app contacts a runner server on load.** QuestionForge ships with a default runner server (`question-forge-server.fly.dev`) and a shared access token already filled in. When the page loads, and every 15 seconds for as long as it stays open, your browser sends an unauthenticated `GET /health` request to that server to display connection status. These requests carry no question content, but the server operator and its host can see your IP address, your browser's user agent, and the time of each request. Clicking "Disconnect" ends the current conversation, but a background check reconnects within 15 seconds; there is currently no setting that stops these health requests.
- **Your content is sent only when you send a message.** Nothing you write is transmitted until you send a chat message — or click "Fix with AI", which sends one immediately.
- **What is sent with each message.** The full conversation so far; the active question's Jinja2 template and Python code; its question ID; your customized system prompt, if you changed it; the question set's context prompt and its question type (quiz or homework); the current preview error, if any; and a listing of up to 80 question IDs, titles, and topics from your bank.
- **What the assistant can pull in.** If the assistant asks for more context, your browser automatically sends a follow-up request containing either the Jinja2 template text of every question in every question set — including disabled ones — or the full template and Python code of one other question it names.
- **The runner server stores nothing.** It holds each request in memory only for as long as it takes to answer, then discards it. It has no database and writes nothing to disk. Its only retained state is a single timestamp used to slow down failed sign-in attempts.
- **What the runner server logs.** The server prints diagnostics to its host's log stream: the model name and the number of messages in a request, but not their content; response status codes and error messages; the ID of any question the assistant asks to read; up to 200 characters of any error raised while checking generated code; and up to 500 characters of any error returned by Google. Your IP address is logged only when a request presents an incorrect access token. No retention period is configured for these logs.
- **Google processes your prompts.** The runner server forwards everything above to Google's Gemini API at `generativelanguage.googleapis.com`, by default the `gemini-2.5-flash` model. Google's handling and retention of that data are governed by the Gemini API terms applying to the operator's account. The Google API key stays on the server and is never sent to your browser.
- **Where the default server runs.** The default runner server is configured to run on Fly.io in Ashburn, Virginia, United States. Content sent to AI Chat is processed outside Canada.
- **The access token is not personal.** It is a single passphrase shared by all users and visible in the page source. It does not identify you, and it is not a confidentiality control.
- **Using a different server.** The server URL and token are editable under "🤖 AI Runner Connection". The server is open source — see `server/` in this repository — and can be self-hosted.

## Contact

This tool is maintained by the Department of Physics & Astronomy, University of Manitoba.
