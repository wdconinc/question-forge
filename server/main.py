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
import time
import concurrent.futures
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment as JinjaEnv, StrictUndefined
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
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth rate limiter (global)
# ---------------------------------------------------------------------------
# After any failed authentication attempt, ALL subsequent attempts are held
# off for AUTH_HOLDOFF_SECS seconds.  IP-based limiting is not used because
# IP spoofing trivially bypasses it; a global limit is simpler and equally
# effective against dictionary attacks.  DoS risk (legitimate users briefly
# locked out by an attacker) is accepted.

AUTH_HOLDOFF_SECS: float = float(os.environ.get("AUTH_HOLDOFF_SECS", "10"))
MAX_FIX_ATTEMPTS: int   = int(os.environ.get("MAX_FIX_ATTEMPTS", "2"))

_last_auth_failure: float = 0.0   # monotonic timestamp of most recent failure


def _get_client_ip(request: Request) -> str:
    """Best-effort client IP for logging only."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_token(request: Request) -> None:
    """Raise 401/429 if authentication fails or the server is in cooldown."""
    global _last_auth_failure

    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="Server has no API_TOKEN configured.")

    # Global cooldown after any recent failure
    elapsed = time.monotonic() - _last_auth_failure
    if elapsed < AUTH_HOLDOFF_SECS:
        retry_after = int(AUTH_HOLDOFF_SECS - elapsed) + 1
        print(f"[auth] global rate-limit active (retry in {retry_after}s)", flush=True)
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        _last_auth_failure = time.monotonic()
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    provided = auth[len("Bearer "):]
    if not hmac.compare_digest(provided.encode(), API_TOKEN.encode()):
        _last_auth_failure = time.monotonic()
        ip = _get_client_ip(request)
        print(f"[auth] failed attempt from {ip}", flush=True)
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
    preview_error: str = ""       # current error shown in the preview panel (if any)
    question_bank_summary: str = "" # brief listing of existing question IDs, titles, and topics

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

## Python generator function

The Python code must define a `generate(rng: numpy.random.Generator) -> dict` function.
The function receives a seeded NumPy random generator and must return a dict with
exactly these keys:

  question   : str        — full question text (plain text or Markdown / LaTeX)
  choices    : list[str]  — exactly 5 answer choice strings, e.g. ["1.23 m", ...]
  answer     : str        — the correct choice letter (lowercase): always 'a' when
                           using make_choices (correct value placed at index 0);
                           use 'b'–'e' only when building choices manually
  topic      : str        — brief topic label, e.g. "Ch. 4 — Newton's 2nd Law"
  difficulty : int        — difficulty level 1 (easy) to 3 (hard)

The exam framework automatically shuffles answer positions before printing, so
there is no need to randomize the correct answer position yourself.

## Helper functions (import from `questions`)

```python
from questions import render_template, make_choices, phys_fmt
```

- `render_template(question_id: str, params: dict) -> str`
  Renders the Jinja2 template associated with this question. `params` is passed as
  keyword arguments to the template context. Returns the rendered string stripped of
  leading/trailing whitespace.

- `make_choices(correct_val: float, distractors: list[float], fmt: callable) -> list[str]`
  Builds a list of 5 unique, well-spaced choice strings. The correct answer is always
  at index 0 (answer = 'a'). `fmt` is a callable that converts a float to a display
  string, e.g. `lambda v: f"{v:.2f} m"`.

- `phys_fmt(v: float, sig: int = 3) -> str`
  Formats a number with `sig` significant figures for a printed exam. Automatically
  uses LaTeX scientific notation (e.g. `$1.23 \\times 10^{4}$`) for very large or
  very small values.

## Jinja2 template

The template renders the question text. Variables from the `params` dict are
available as top-level template variables. Use `{{ variable }}` for substitution
and `{% if %} / {% elif %} / {% else %} / {% endif %}` for conditionals.
LaTeX math is written inline as `$...$`.

## Workflow guidelines

- When asked to modify code or template, use the update_question tool.
  You may update both template and python_code in a single call when both need changing.
- When asked to create a new question, use the create_question tool with a complete
  template and python_code.
- Otherwise reply in plain text (Markdown is fine).

## Creating multiple questions

When the user asks to create more than one question:

- First check the question set context (if provided) and the existing question bank
  for topics, coverage gaps, or other guidance on what to create.
- If you have enough context to choose topics independently (e.g. a syllabus, chapter
  list, or topic breakdown is available), **select suitable topics yourself** and call
  create_question once per question without asking the user first.
- If there is **not** enough context to determine appropriate topics, ask the user a
  single focused question (e.g. "Which topics or chapters should these questions cover?")
  and wait for their answer before proceeding.
- When creating multiple questions, vary difficulty levels and sub-topics to produce a
  balanced set, and avoid duplicating topics already present in the existing question bank.
- Each new question **must** have a unique `question_id` that does not already exist in the
  bank. Use a descriptive snake_case name, e.g. `q_friction_ramp`. If the bank listing is
  provided, check it and choose IDs that are not already there. The `python_code` must call
  `render_template` with the **exact same** `question_id` string you provide in this call,
  because the template file is stored under that name on disk.
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
    if req.question_bank_summary.strip():
        prompt += f"""
=== EXISTING QUESTIONS IN BANK ===
{req.question_bank_summary.strip()}
"""
    if req.question_set_prompt.strip():
        prompt += f"""
=== QUESTION SET CONTEXT ===
{req.question_set_prompt.strip()}
"""
    if req.preview_error.strip():
        prompt += f"""
=== CURRENT PREVIEW ERROR ===
The question currently fails to preview with this error:
{req.preview_error.strip()}

Please fix the template and/or python_code so the preview runs without errors.
"""
    return prompt

# ---------------------------------------------------------------------------
# /health  (unauthenticated)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": GEMINI_MODEL}


# ---------------------------------------------------------------------------
# /test-stream  (unauthenticated, hardcoded SSE for client-side debugging)
# ---------------------------------------------------------------------------

@app.get("/test-stream")
async def test_stream() -> EventSourceResponse:
    """Returns a fixed SSE sequence so the browser SSE reader can be tested
    without hitting the Gemini API at all."""
    import asyncio

    async def _fixed() -> AsyncIterator[dict]:
        await asyncio.sleep(0.1)
        yield {"data": json.dumps({"type": "text", "delta": "Hello from "})}
        await asyncio.sleep(0.1)
        yield {"data": json.dumps({"type": "text", "delta": "test-stream!"})}
        await asyncio.sleep(0.1)
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(_fixed(), ping=0)

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
# Question validation helpers
# ---------------------------------------------------------------------------

def _validate_question(template: str, python_code: str) -> tuple[bool, str]:
    """Execute python_code, call generate_params(), render template.

    Runs in a thread pool with a 5-second timeout to guard against infinite
    loops in AI-generated code.  Returns (ok, error_message).
    """
    def _run() -> tuple[bool, str]:
        try:
            namespace: dict = {}
            exec(compile(python_code, "<ai_generated>", "exec"), namespace)  # noqa: S102
            generate = namespace.get("generate_params")
            if generate is None:
                return False, "python_code must define a generate_params() function"
            params = generate()
            if not isinstance(params, dict):
                return False, f"generate_params() must return a dict, got {type(params).__name__}"
            env = JinjaEnv(undefined=StrictUndefined)
            rendered = env.from_string(template).render(**params)
            if not rendered.strip():
                return False, "Template rendered to an empty string"
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=5)
        except concurrent.futures.TimeoutError:
            return False, "Execution timed out (>5 s)"


async def _gemini_fix_call(
    system_text: str,
    contents: list[dict],
    fc_name: str,
    fc_args: dict,
    error: str,
) -> dict | None:
    """Non-streaming Gemini call that feeds a validation error back as a
    functionResponse and asks the model to return a corrected function call.
    Returns the new args dict, or None if Gemini didn't return a function call.
    """
    fix_contents = contents + [
        {"role": "model", "parts": [{"functionCall": {"name": fc_name, "args": fc_args}}]},
        {"role": "user", "parts": [{"functionResponse": {
            "name": fc_name,
            "response": {
                "error": (
                    f"Validation failed: {error}\n\n"
                    "Please fix the template and python_code so they work together without errors. "
                    "Requirements:\n"
                    "  • python_code must define a generate_params() function that returns a dict\n"
                    "  • All Jinja2 variables in the template must be keys in that dict\n"
                    "  • The template must render without exceptions using the generated params"
                )
            },
        }}]},
    ]
    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": fix_contents,
        "tools": _to_gemini_tools(),
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [fc_name],
            }
        },
        "generationConfig": {"temperature": 0.2},
    }
    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, params={"key": GOOGLE_API_KEY}, json=body)
        if resp.status_code != 200:
            print(f"[chat] fix call HTTP {resp.status_code}", flush=True)
            return None
        data = resp.json()
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "functionCall" in part and part["functionCall"].get("name") == fc_name:
                    return part["functionCall"].get("args", {})
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] fix call exception: {exc}", flush=True)
    return None


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

    print(f"[chat] model={GEMINI_MODEL} msgs={len(req.messages)}", flush=True)

    async def _stream() -> AsyncIterator[dict]:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, params=params, json=body) as resp:
                    print(f"[chat] gemini status={resp.status_code}", flush=True)
                    if resp.status_code != 200:
                        err = await resp.aread()
                        err_text = err.decode()
                        print(f"[chat] gemini error: {err_text[:500]}", flush=True)
                        yield {"data": json.dumps({"type": "error", "message": f"Gemini {resp.status_code}: {err_text[:300]}"})}
                        return

                    function_calls: list[dict] = []
                    n_text = 0

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
                                    n_text += 1
                                    yield {"data": json.dumps({"type": "text", "delta": part["text"]})}
                                if "functionCall" in part:
                                    function_calls.append(part["functionCall"])

            print(f"[chat] done: n_text={n_text} tool_calls={len(function_calls)}", flush=True)

            # Validate and auto-fix update_question / create_question calls
            for i, fc in enumerate(function_calls):
                name = fc.get("name", "")
                args = fc.get("args", {})
                if name not in ("update_question", "create_question"):
                    continue
                template = args.get("template", "")
                python_code = args.get("python_code", "")
                if not (template and python_code):
                    continue

                ok, err = _validate_question(template, python_code)
                if ok:
                    print(f"[chat] validation ok for {name}", flush=True)
                    continue

                print(f"[chat] validation failed for {name}: {err[:200]}", flush=True)
                for fix_attempt in range(MAX_FIX_ATTEMPTS):
                    print(f"[chat] fix attempt {fix_attempt + 1}/{MAX_FIX_ATTEMPTS}", flush=True)
                    fixed_args = await _gemini_fix_call(
                        system_text=system_text,
                        contents=contents,
                        fc_name=name,
                        fc_args=args,
                        error=err,
                    )
                    if fixed_args is None:
                        print("[chat] fix call returned no function call", flush=True)
                        break
                    ok, err = _validate_question(
                        fixed_args.get("template", template),
                        fixed_args.get("python_code", python_code),
                    )
                    args = fixed_args
                    function_calls[i] = {**fc, "args": fixed_args}
                    if ok:
                        print(f"[chat] fixed on attempt {fix_attempt + 1}", flush=True)
                        break
                    print(f"[chat] fix attempt {fix_attempt + 1} still failing: {err[:200]}", flush=True)

                if not ok:
                    yield {"data": json.dumps({
                        "type": "text",
                        "delta": (
                            f"\n\n⚠️ *Warning: I could not verify this code runs without errors "
                            f"after {MAX_FIX_ATTEMPTS} fix attempt(s). "
                            f"Last error: `{err}`. Please review carefully before accepting.*"
                        ),
                    })}

            # Emit (possibly fixed) tool calls
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
            print(f"[chat] exception: {exc}", flush=True)
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(_stream(), ping=0)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
