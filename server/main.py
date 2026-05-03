"""
QuestionForge AI Runner Server
================================
A lightweight FastAPI server that:
 - Authenticates browsers with an 8-char token (Authorization: Bearer header)
 - Streams LLM responses back as SSE
 - Exposes function-calling tools so the AI can edit the active question's
   editors in the browser

Uses the Gemini REST API directly via httpx — no litellm dependency, keeping
the container memory footprint small enough for Fly.io free tier.
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

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

API_TOKEN: str    = os.environ.get("API_TOKEN", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

# Accept "gemini/gemini-2.5-flash" (LiteLLM style) or bare "gemini-2.5-flash"
_raw_model = os.environ.get("LITELLM_MODEL", "gemini-2.5-flash")
GEMINI_MODEL: str = _raw_model.split("/")[-1]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

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
    system_prompt: str = ""       # optional override for the base system prompt
    question_set_prompt: str = "" # optional per-exam context appended to system prompt

# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_question",
            "description": (
                "Update the Jinja2 template and/or Python generator for the current question "
                "in a single operation. Provide `template` to change the template, `python_code` "
                "to change the Python code, or both. Always use this tool when the user asks to "
                "modify any part of the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "description": "The complete new Jinja2 template text. Omit if not changing the template.",
                    },
                    "python_code": {
                        "type": "string",
                        "description": "The complete new Python generator code. Omit if not changing the Python code.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_question",
            "description": (
                "Create a brand-new parametrized multiple-choice question and add it to the exam. "
                "Use this when the user asks to create, add, or write a new question. "
                "The question_id must be a short snake_case identifier, e.g. 'q_friction_ramp'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": "Short snake_case identifier for the new question, e.g. 'q_projectile_angle'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Human-readable display title for the question.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Physics topic, e.g. 'Kinematics', 'Thermodynamics'.",
                    },
                    "template": {
                        "type": "string",
                        "description": "The complete Jinja2 template for the question.",
                    },
                    "python_code": {
                        "type": "string",
                        "description": "The complete Python generator code defining generate(rng) -> dict.",
                    },
                },
                "required": ["question_id", "title", "template", "python_code"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are an expert physics exam question author helping edit a parametrized \
multiple-choice question for an algebra-based introductory physics course \
(OpenStax College Physics 2e).

Guidelines:
- The Python generator must define a `generate(rng)` function that returns a dict with
  keys: question (str), choices (list[str], exactly 5), answer (str, one of A-E),
  topic (str), difficulty (int 1-3).
- The Jinja2 template renders the question text and answer choices.
  Parameters are passed as keyword arguments by `render_template(qid, params)`.
- When asked to modify code or template, use the update_question tool.
  You may update both template and python_code in a single call when both need changing.
- When asked to create a new question, use the create_question tool with a complete
  template and python_code.
- Otherwise reply in plain text (Markdown is fine).\
"""

def _system_prompt(req: ChatRequest) -> str:
    qid = req.question_id or "(unknown)"
    base = req.system_prompt.strip() if req.system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
    prompt = f"""{base}

Current question ID: {qid}

=== JINJA2 TEMPLATE ===
{req.template or "(empty)"}

=== PYTHON GENERATOR ===
{req.python_code or "(empty)"}
"""
    if req.question_set_prompt.strip():
        prompt += f"""
=== QUESTION SET CONTEXT ===
{req.question_set_prompt.strip()}
"""
    return prompt

# ---------------------------------------------------------------------------
# /health  (unauthenticated)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": GEMINI_MODEL}

# ---------------------------------------------------------------------------
# /chat  (SSE streaming)
# ---------------------------------------------------------------------------

def _to_gemini_tools() -> list[dict]:
    """Convert OpenAI-style TOOLS list to Gemini functionDeclarations."""
    return [{
        "functionDeclarations": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"].get("parameters", {}),
            }
            for t in TOOLS
        ]
    }]


def _to_gemini_contents(system_prompt: str, messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    """Return (systemInstruction text, contents list) in Gemini format."""
    contents = []
    for m in messages:
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return system_prompt, contents


# ---------------------------------------------------------------------------
# /chat  (SSE streaming)
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    _check_token(request)

    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured on server.")

    system_text, contents = _to_gemini_contents(_system_prompt(req), req.messages)

    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "tools": _to_gemini_tools(),
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0.7},
    }

    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:streamGenerateContent"
    params = {"key": GOOGLE_API_KEY, "alt": "sse"}

    async def _stream() -> AsyncIterator[dict]:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, params=params, json=body) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield {"data": json.dumps({"type": "error", "message": err.decode()})}
                        return

                    function_calls: list[dict] = []

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        for candidate in chunk.get("candidates", []):
                            parts = candidate.get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part and part["text"]:
                                    yield {"data": json.dumps({"type": "text", "delta": part["text"]})}
                                if "functionCall" in part:
                                    function_calls.append(part["functionCall"])

            # Emit completed tool calls
            for fc in function_calls:
                name = fc.get("name", "")
                args = fc.get("args", {})
                payload: dict = {"type": "tool_call", "tool": name}
                if "template" in args:
                    payload["template"] = args["template"]
                if "python_code" in args:
                    payload["python_code"] = args["python_code"]
                if "question_id" in args:
                    payload["question_id"] = args["question_id"]
                if "title" in args:
                    payload["title"] = args["title"]
                if "topic" in args:
                    payload["topic"] = args.get("topic", "")
                if "content" in args:
                    payload["content"] = args["content"]
                yield {"data": json.dumps(payload)}

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
