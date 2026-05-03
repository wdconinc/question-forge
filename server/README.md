# QuestionForge AI Runner Server

A lightweight FastAPI server that adds AI assistance to the QuestionForge browser app.  
The AI can read the active question's Jinja2 template and Python generator, and write back
updates directly into the editors via function-calling tools.

## Quick start

```bash
cd server
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set API_TOKEN, LITELLM_MODEL, and your provider API key
python main.py
```

Or with `uv` (recommended — pins Python 3.13 via `.python-version`):
```bash
cd server
cp .env.example .env
# Edit .env
uv sync
uv run main.py
```

The server starts on `http://localhost:8000` by default.

## Configuration (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Google AI Studio key ([aistudio.google.com](https://aistudio.google.com)) |
| `API_TOKEN` | ✅ | — | 8-char alphanumeric token shared with browser users |
| `LITELLM_MODEL` | ❌ | `gemini-2.5-flash` | Gemini model name; `gemini/` prefix (LiteLLM style) is stripped automatically |
| `AUTH_HOLDOFF_SECS` | ❌ | `10` | Seconds an IP is locked out after a failed auth attempt (rate-limits brute force) |
| `PORT` | ❌ | `8000` | Server port |

### Available Gemini models

```
LITELLM_MODEL=gemini-2.5-flash        # fast, free tier available (default)
LITELLM_MODEL=gemini-2.5-flash-lite   # lighter/cheaper
LITELLM_MODEL=gemini-2.5-pro          # highest capability
LITELLM_MODEL=gemini/gemini-2.5-flash # LiteLLM-style prefix also accepted
```

> **Note:** The server calls the Gemini REST API directly via `httpx` (no litellm
> dependency) to keep the container footprint small. Only Google Gemini models are
> supported. To use OpenAI/Anthropic/Ollama, run a litellm proxy and point
> `LITELLM_MODEL` + server URL at it.

## API

### `GET /health`
Returns `{"ok": true, "model": "..."}`. Unauthenticated — use to test connectivity.

### `POST /chat`
Requires `Authorization: Bearer <token>` header.

Request body:
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "template": "current jinja2 template text",
  "python_code": "current python code",
  "question_id": "q01_kinematics"
}
```

Streams Server-Sent Events:
| Event data type | Fields | Description |
|---|---|---|
| `text` | `delta` | Streamed text from the AI |
| `tool_call` | `tool`, `content` | AI called `update_template` or `update_python` |
| `done` | — | Stream complete |
| `error` | `message` | Server-side error |

## Running on a remote server

### Fly.io (recommended)

[Install `flyctl`](https://fly.io/docs/hands-on/install-flyctl/), then from the `server/` directory:

```bash
# 1. Authenticate
fly auth login

# 2. Create a new app (pick a unique name)
fly apps create question-forge-server   # or any name you like

# 3. Update the app name in fly.toml to match
#    app = "question-forge-server"

# 4. Set secrets (never committed to git)
fly secrets set \
  API_TOKEN=your8chartoken \
  LITELLM_MODEL=gemini/gemini-2.5-flash \
  GOOGLE_API_KEY=your_google_ai_studio_key

# 5. Deploy
fly deploy
```

The server will be live at `https://<app-name>.fly.dev`.  
Paste that URL into the QuestionForge **AI Chat → ⚙ Connection settings** dialog.

To redeploy after code changes:
```bash
fly deploy
```

To view logs:
```bash
fly logs
```

**Scaling / cost:** the default `fly.toml` uses a shared-cpu-1x 256 MB machine with
`auto_stop_machines = "stop"` — it sleeps when idle (free tier eligible) and wakes on
the next request (≈2 s cold start).

---

### Self-hosted (nginx reverse proxy)

The browser's URL field can point to any reachable HTTPS URL.

```nginx
location /ai/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_buffering off;          # required for SSE
    proxy_cache off;
    proxy_read_timeout 120s;
}
```



- The `API_TOKEN` is verified with `hmac.compare_digest` (constant-time, prevents timing attacks).
- Failed auth attempts trigger a per-IP holdoff (`AUTH_HOLDOFF_SECS`, default 10 s), returning HTTP 429. At 1 attempt/10 s, a 2-word passphrase (170 K² ≈ 29 B combinations) would take ~4,600 years to brute-force; the default 8-char alphanumeric token ~890,000 years.
- Provider API keys never leave the server.
- CORS is currently open (`*`); restrict `allow_origins` for production deployments.
