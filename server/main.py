"""
QuestionForge AI Runner Server
================================
A lightweight FastAPI server that:
 - Authenticates browsers with an 8-char token (Authorization: Bearer header)
 - Streams LLM responses back as SSE
 - Exposes update_template / update_python function-calling tools so the AI
   can edit the active question's editors in the browser
"""

from __future__ import annotations

import sys

# Remove any paths injected via PYTHONPATH that belong to a different Python
# version (e.g. /opt/local/lib/python3.14t/site-packages leaking into a 3.13
# venv).  The venv's own site-packages always start with sys.prefix.
sys.path = [
    p for p in sys.path
    if not p or p.startswith(sys.prefix) or p.startswith(sys.base_prefix)
    or not any(seg.startswith("python3.") and seg != f"python{sys.version_info.major}.{sys.version_info.minor}" for seg in p.split("/"))
]

import hmac
import json
import os
from typing import AsyncIterator

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

API_TOKEN: str = os.environ.get("API_TOKEN", "")
LITELLM_MODEL: str = os.environ.get("LITELLM_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="QuestionForge AI Runner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_token(request: Request) -> None:
    """Raise 401 if the bearer token doesn't match API_TOKEN."""
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="Server has no API_TOKEN configured.")
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    provided = auth[len("Bearer "):]
    if not hmac.compare_digest(provided.encode(), API_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Invalid token.")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    template: str = ""
    python_code: str = ""
    question_id: str = ""

# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_template",
            "description": (
                "Replace the entire Jinja2 template for the current question. "
                "Use this when the user asks to modify, fix, or rewrite the template."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete new Jinja2 template text.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_python",
            "description": (
                "Replace the entire Python question-generator code for the current question. "
                "Use this when the user asks to modify, fix, or rewrite the Python code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete new Python generator code.",
                    }
                },
                "required": ["content"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _system_prompt(req: ChatRequest) -> str:
    qid = req.question_id or "(unknown)"
    return f"""You are an expert physics exam question author helping edit a parametrized \
multiple-choice question for an algebra-based introductory physics course \
(OpenStax College Physics 2e).

Current question ID: {qid}

=== JINJA2 TEMPLATE ===
{req.template or "(empty)"}

=== PYTHON GENERATOR ===
{req.python_code or "(empty)"}

Guidelines:
- The Python generator must define a `generate(rng)` function that returns a dict with
  keys: question (str), choices (list[str], exactly 5), answer (str, one of A-E),
  topic (str), difficulty (int 1-3).
- The Jinja2 template renders the question text and answer choices.
  Parameters are passed as keyword arguments by `render_template(qid, params)`.
- When asked to modify code or template, use the update_template or update_python tools.
- Otherwise reply in plain text (Markdown is fine).
"""

# ---------------------------------------------------------------------------
# /health  (unauthenticated)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": LITELLM_MODEL}

# ---------------------------------------------------------------------------
# /chat  (SSE streaming)
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    _check_token(request)

    messages = [{"role": "system", "content": _system_prompt(req)}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    async def _stream() -> AsyncIterator[dict]:
        try:
            response = await litellm.acompletion(
                model=LITELLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            tool_calls: dict[int, dict] = {}  # index → {name, arguments_buf}

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Text content
                if delta.content:
                    yield {"data": json.dumps({"type": "text", "delta": delta.content})}

                # Tool calls (streamed in pieces)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls:
                            tool_calls[idx] = {"name": "", "arguments_buf": ""}
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments_buf"] += tc.function.arguments

            # Emit completed tool calls
            for tc in tool_calls.values():
                try:
                    args = json.loads(tc["arguments_buf"])
                except json.JSONDecodeError:
                    args = {}
                yield {
                    "data": json.dumps({
                        "type": "tool_call",
                        "tool": tc["name"],
                        "content": args.get("content", ""),
                    })
                }

            yield {"data": json.dumps({"type": "done"})}

        except Exception as exc:  # noqa: BLE001
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(_stream())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
