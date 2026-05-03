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
| `API_TOKEN` | ✅ | — | 8-char alphanumeric token shared with browser users |
| `LITELLM_MODEL` | ✅ | `gpt-4o` | Any [LiteLLM model string](https://docs.litellm.ai/docs/providers) |
| `OPENAI_API_KEY` | if using OpenAI | — | |
| `ANTHROPIC_API_KEY` | if using Anthropic | — | |
| `GOOGLE_API_KEY` | if using Gemini | — | Google AI Studio key |
| `PORT` | ❌ | `8000` | Server port |

### Model examples

```
LITELLM_MODEL=gpt-4o                          # OpenAI
LITELLM_MODEL=claude-3-5-sonnet-20241022      # Anthropic
LITELLM_MODEL=gemini/gemini-2.0-flash         # Google Gemini (fast, free tier available)
LITELLM_MODEL=gemini/gemini-1.5-pro           # Google Gemini Pro
LITELLM_MODEL=ollama/llama3                   # Local Ollama
LITELLM_MODEL=azure/gpt-4o                    # Azure OpenAI
```

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

The browser's URL field (in the AI Chat connection settings) can point to any reachable URL.
Use HTTPS and a reverse proxy (nginx/caddy) for production.

```nginx
location /ai/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_buffering off;          # required for SSE
    proxy_cache off;
    proxy_read_timeout 120s;
}
```

## Security notes

- The `API_TOKEN` is verified with `hmac.compare_digest` (constant-time, prevents timing attacks).
- Provider API keys never leave the server.
- CORS is currently open (`*`); restrict `allow_origins` for production deployments.
