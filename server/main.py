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
    if not p or p.startswith((sys.prefix, sys.base_prefix))
    or not any(seg.startswith("python3.") and seg != f"python{sys.version_info.major}.{sys.version_info.minor}" for seg in p.split("/"))
]

import hmac
import json
import math
import os
import queue
import threading
import time
import types
from collections.abc import AsyncIterator

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment as JinjaEnv
from jinja2 import StrictUndefined
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
    question_type: str = "quiz"   # authoring style for this set: "quiz" or "homework"
    question_set_prompt: str = "" # optional per-exam context appended to system prompt
    bank_summary: str = ""        # optional summary of all questions in the bank
    preview_error: str = ""       # current error shown in the preview panel (if any)
    question_bank_summary: str = "" # brief listing of existing question IDs, titles, and topics
    requested_question_id: str = ""       # ID of a non-active question the AI requested
    requested_question_template: str = "" # Jinja2 template of the requested question
    requested_question_python_code: str = "" # Python code of the requested question

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
            "name": "get_question_bank",
            "description": (
                "Retrieve the full list of all questions in the exam bank, including their "
                "exam-order position, IDs, topics, and question text. Call this when you need "
                "to check topic coverage, identify missing or overrepresented topics, or ensure "
                "consistency in notation across all questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_question",
            "description": (
                "Retrieve the full Jinja2 template and Python generator code for a specific "
                "question that is not currently active in the editor. Use this when you need "
                "to read, compare, or reference another question's implementation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": "The ID of the question to retrieve.",
                    },
                },
                "required": ["question_id"],
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
multiple-choice question.

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

# Guidance appended for each selectable "question type" (see ChatRequest.question_type).
# This supplements the base system prompt and the per-set question_set_prompt — it never
# replaces either of them.
QUESTION_TYPE_PROMPTS = {
    "quiz": (
        "Write this question in **quiz/test style**: concise and self-contained, "
        "answerable within a couple of minutes, testing a single concept or "
        "calculation. Prefer a direct numeric or conceptual question without "
        "multi-part scaffolding."
    ),
    "homework": (
        "Write this question in **homework/practice style**: it may involve a "
        "multi-step derivation or several intermediate calculations, giving the "
        "student room to practice applying a concept rather than testing quick "
        "recall. The final answer must still be one of exactly 5 multiple-choice "
        "options (the exam framework requires this), but the question text itself "
        "can walk through a longer scenario, and distractors should reflect common "
        "step-by-step mistakes (e.g. a sign error, a unit conversion slip, using the "
        "wrong formula) rather than just nearby numeric values."
    ),
}

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
    question_type = (req.question_type or "").strip().lower() or "quiz"
    type_guidance = QUESTION_TYPE_PROMPTS.get(question_type)
    if type_guidance:
        prompt += f"""
=== QUESTION TYPE GUIDANCE ===
{type_guidance}
"""
    if req.question_set_prompt.strip():
        prompt += f"""
=== QUESTION SET CONTEXT ===
{req.question_set_prompt.strip()}
"""
    if req.bank_summary.strip():
        prompt += f"""
=== ALL QUESTIONS IN BANK ===
{req.bank_summary.strip()}
"""
    if req.preview_error.strip():
        prompt += f"""
=== CURRENT PREVIEW ERROR ===
The question currently fails to preview with this error:
{req.preview_error.strip()}

Please fix the template and/or python_code so the preview runs without errors.
"""
    if req.requested_question_id.strip():
        prompt += f"""
=== REQUESTED QUESTION: {req.requested_question_id.strip()} ===

--- JINJA2 TEMPLATE ---
{req.requested_question_template or "(empty)"}

--- PYTHON GENERATOR ---
{req.requested_question_python_code or "(empty)"}
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
#
# The real exam renderer runs entirely in-browser via Pyodide (see
# python/exam_core.py + the questions/__init__.py embedded in index.html):
# it seeds a numpy.random.Generator, execs the question's python_code, and
# calls generate(rng) with `questions.render_template` monkey-patched to
# render that question's own template. To catch the same errors server-side
# without depending on Pyodide (or on index.html, which isn't shipped in the
# server's Docker image), we reimplement that contract here with plain
# CPython + numpy and lightweight stand-ins for the questions/ helpers.
# ---------------------------------------------------------------------------

# Serializes validation runs: each one temporarily registers a "questions"
# module in sys.modules (so `from questions import ...` resolves inside the
# exec'd code) whose render_template is monkey-patched to the current
# template. That's global, mutable state, so concurrent /chat requests must
# not validate at the same time. The previous entry (if any) is saved and
# restored around the run rather than blindly popped, and the lock is never
# held waiting on a hung worker thread (see _validate_question) so one
# runaway AI-generated infinite loop can't wedge every future validation.
_VALIDATION_LOCK = threading.Lock()


def _stub_make_choices(correct_val: float, distractors: list, fmt, min_spacing: float = 0.12) -> list[str]:
    """Simplified stand-in for questions.make_choices() — just enough to run
    AI-generated code without crashing. Doesn't enforce uniqueness/spacing;
    that real logic lives in the browser's questions/__init__.py."""
    values = ([correct_val, *list(distractors)] + [correct_val] * 4)[:5]
    return [fmt(v) for v in values]


def _stub_phys_fmt(v: float, sig: int = 3) -> str:
    if not math.isfinite(v) or v == 0:
        return "0"
    return f"{v:.{sig}g}"


def _run_generate(template: str, python_code: str) -> tuple[bool, str]:
    """Exec python_code, call generate(rng), and confirm it returns a
    well-formed question dict. Assumes `questions` is already registered in
    sys.modules by the caller. Runs on the caller's thread — the caller is
    responsible for the timeout, since a genuine infinite loop in
    AI-generated code cannot be interrupted from here."""
    try:
        namespace: dict = {}
        exec(compile(python_code, "<ai_generated>", "exec"), namespace)  # noqa: S102
        generate = namespace.get("generate")
        if generate is None:
            return False, "python_code must define a generate(rng) function"
        result = generate(np.random.default_rng())

        if not isinstance(result, dict):
            return False, f"generate(rng) must return a dict, got {type(result).__name__}"
        if not str(result.get("question", "")).strip():
            return False, "generate(rng) must return a non-empty 'question'"
        if not str(result.get("topic", "")).strip():
            return False, "generate(rng) must return a non-empty 'topic'"
        choices = result.get("choices")
        if not (isinstance(choices, list) and len(choices) == 5):
            return False, "generate(rng) must return exactly 5 'choices'"
        if result.get("answer") not in ("a", "b", "c", "d", "e"):
            return False, "generate(rng) must return an 'answer' of 'a'-'e'"
        difficulty = result.get("difficulty")
        if not (isinstance(difficulty, int) and not isinstance(difficulty, bool) and 1 <= difficulty <= 3):
            return False, "generate(rng) must return a 'difficulty' int between 1 and 3"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _validate_question(template: str, python_code: str) -> tuple[bool, str]:
    """Validate that python_code's generate(rng) renders `template` without
    errors, using render_template (backed by `template`) for any
    `from questions import ...` the code performs.

    Runs on a daemon thread with a 5-second timeout to guard against infinite
    loops in AI-generated code. A timeout gives up waiting but the worker
    thread itself is not killed (Python cannot forcibly stop a running
    thread) — it is abandoned to finish or loop forever on its own. It must
    be a daemon thread (not a concurrent.futures.ThreadPoolExecutor one,
    which the stdlib joins at interpreter exit) so a leaked infinite loop
    can't also hang server shutdown. The `questions` module patch is restored
    immediately regardless of the timeout, since any `from questions import
    ...` in the worker already bound its own reference at exec time and is
    unaffected by later changes to sys.modules.  Returns (ok, error_message).
    """
    jinja_env = JinjaEnv(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        undefined=StrictUndefined,
    )
    compiled_template = jinja_env.from_string(template)
    questions_module = types.ModuleType("questions")
    questions_module.render_template = (
        lambda name, params: compiled_template.render(**params).strip()
    )
    questions_module.make_choices = _stub_make_choices
    questions_module.phys_fmt = _stub_phys_fmt

    with _VALIDATION_LOCK:
        previous_questions_module = sys.modules.get("questions")
        sys.modules["questions"] = questions_module
        result_box: queue.Queue = queue.Queue(maxsize=1)
        worker = threading.Thread(
            target=lambda: result_box.put(_run_generate(template, python_code)),
            daemon=True,
        )
        try:
            worker.start()
            worker.join(timeout=5)
            if worker.is_alive():
                return False, "Execution timed out (>5 s)"
            return result_box.get_nowait()
        finally:
            if previous_questions_module is None:
                sys.modules.pop("questions", None)
            else:
                sys.modules["questions"] = previous_questions_module


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
                    "  • python_code must define a generate(rng: numpy.random.Generator) -> dict function\n"
                    "  • It must return a dict with 'question', 'choices' (exactly 5), 'answer' ('a'-'e'),\n"
                    "    'topic', and 'difficulty'\n"
                    "  • If it calls render_template(question_id, params), every Jinja2 variable in the\n"
                    "    template must be a key in params, and the template must render without exceptions"
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

    # Build the tools list; exclude get_question_bank when bank data is already
    # in the system prompt (prevents the AI from calling it in a loop).
    # Similarly, exclude get_question when that question's data is already provided.
    gemini_tools = _to_gemini_tools()
    excluded: set[str] = set()
    if req.bank_summary.strip():
        excluded.add("get_question_bank")
    if req.requested_question_id.strip():
        excluded.add("get_question")
    if excluded:
        gemini_tools = [{
            "functionDeclarations": [
                fd for fd in tool_group["functionDeclarations"]
                if fd["name"] not in excluded
            ]
        } for tool_group in gemini_tools]

    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "tools": gemini_tools,
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0.7},
    }

    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:streamGenerateContent"
    params = {"key": GOOGLE_API_KEY, "alt": "sse"}

    print(f"[chat] model={GEMINI_MODEL} msgs={len(req.messages)}", flush=True)

    async def _stream() -> AsyncIterator[dict]:
        try:
            async with (
                httpx.AsyncClient(timeout=120) as client,
                client.stream("POST", url, params=params, json=body) as resp,
            ):
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
                            if part.get("text"):
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
                if name == "get_question_bank":
                    if req.bank_summary.strip():
                        # Bank is already in the system prompt; suppress to avoid loops
                        print("[chat] suppressing get_question_bank — bank_summary already provided", flush=True)
                        continue
                    # Signal the browser to provide the bank summary and retry
                    print("[chat] AI requested get_question_bank", flush=True)
                    yield {"data": json.dumps({"type": "tool_call", "tool": "get_question_bank"})}
                    continue
                if name == "get_question":
                    if req.requested_question_id.strip():
                        # Question is already in the system prompt; suppress to avoid loops
                        print("[chat] suppressing get_question — requested_question already provided", flush=True)
                        continue
                    # Signal the browser to provide the question data and retry
                    requested_id = args.get("question_id", "")
                    print(f"[chat] AI requested get_question: {requested_id}", flush=True)
                    yield {"data": json.dumps({"type": "tool_call", "tool": "get_question", "question_id": requested_id})}
                    continue
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
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
