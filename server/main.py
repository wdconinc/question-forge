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

import asyncio
import hmac
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import qvalidate

load_dotenv()

API_TOKEN: str      = os.environ.get("API_TOKEN", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MAX_TOKENS: int = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192"))

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps a model id (as sent by the browser / configured via LITELLM_MODEL) to
# the provider that serves it. Only models whose provider has a configured
# API key are advertised to the browser via GET /models, but any model id
# can still be requested directly (e.g. a newer snapshot not yet listed here)
# as long as its provider key is set.

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gemini-2.5-flash":      {"provider": "gemini", "label": "Gemini 2.5 Flash"},
    "gemini-2.5-flash-lite": {"provider": "gemini", "label": "Gemini 2.5 Flash Lite"},
    "gemini-2.5-pro":        {"provider": "gemini", "label": "Gemini 2.5 Pro"},
    "claude-opus-5":         {"provider": "anthropic", "label": "Claude Opus 5"},
    "claude-sonnet-5":       {"provider": "anthropic", "label": "Claude Sonnet 5"},
    "claude-haiku-4-5":      {"provider": "anthropic", "label": "Claude Haiku 4.5"},
}

def _normalize_model(raw: str) -> str:
    """Strip whitespace and an optional "provider/" prefix (LiteLLM style),
    e.g. " gemini/gemini-2.5-flash " -> "gemini-2.5-flash"."""
    return raw.strip().split("/")[-1]


# Accept "gemini/gemini-2.5-flash" (LiteLLM style) or a bare model id.
DEFAULT_MODEL: str = _normalize_model(os.environ.get("LITELLM_MODEL", "gemini-2.5-flash"))


def _provider_for_model(model: str) -> str:
    info = MODEL_REGISTRY.get(model)
    if info:
        return info["provider"]
    # Unknown model id (e.g. a newer snapshot not yet in the registry) —
    # guess the provider from a conventional name prefix.
    return "anthropic" if model.startswith("claude") else "gemini"


def _provider_api_key(provider: str) -> str:
    return ANTHROPIC_API_KEY if provider == "anthropic" else GOOGLE_API_KEY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Fire and forget: cold starts are frequent (auto_stop_machines), so the canary
    # must not add its subprocess latency to the first request.
    asyncio.create_task(_validation_canary())
    yield


app = FastAPI(title="QuestionForge AI Runner", version="0.1.0", lifespan=_lifespan)

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
VALIDATE_TIMEOUT: float = float(os.environ.get("VALIDATE_TIMEOUT", "10"))

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
    model: str = ""          # optional model id override (defaults to DEFAULT_MODEL)
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
    textbook_catalog: str = ""    # chapter-level table of contents of the enabled textbooks
    textbook_context: str = ""    # textbook excerpts the browser retrieved, client-resolved
    textbook_query: str = ""      # the query/section ids those excerpts came from

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
            "name": "search_textbook",
            "description": (
                "Look up the course textbook and get back full section text: learning "
                "objectives, the section summary, key equations, worked examples with "
                "their solutions, and end-of-section problems with answers. "
                "Call this BEFORE writing a new question so the question matches the "
                "book's notation, level and problem style. "
                "Pass `query` with precise terminology (e.g. 'coefficient of kinetic "
                "friction inclined plane'), or `section_ids` when the catalog already "
                "tells you which sections you need (e.g. ['6.3','6.4']). Prefer "
                "`section_ids` when you know them. You get ONE textbook lookup per "
                "reply, so ask for everything you need in a single call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search over the textbook index.",
                    },
                    "section_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Section numbers from the catalog, e.g. ['6.3'] or "
                            "['college-physics-2e:6.3']. Overrides `query`."
                        ),
                    },
                    "books": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these book slugs, as shown in the catalog.",
                    },
                    "chapters": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Restrict the search to these chapter numbers.",
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["objectives", "summary", "equations", "examples",
                                     "problems", "conceptual", "glossary", "body"],
                        },
                        "description": (
                            "Which parts to return. Defaults to objectives, summary, "
                            "equations, examples and problems. Add 'body' only when you "
                            "need the full prose."
                        ),
                    },
                    "max_sections": {
                        "type": "integer",
                        "description": "How many sections to return, 1-6. Default 3.",
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
                "Create a brand-new parametrized multiple-choice or numerical-entry question "
                "and add it to the exam. Use this when the user asks to create, add, or write "
                "a new question. The question_id must be a short snake_case identifier, "
                "e.g. 'q_friction_ramp'."
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
multiple-choice or numerical-entry question.

## Python generator function

The Python code must define a `generate(rng: numpy.random.Generator) -> dict` function.
The function receives a seeded NumPy random generator and must return a dict.
There are two question types, chosen by the `type` key (default "multiple_choice"
if omitted) — use whichever the current question already is:

### Multiple choice

  question   : str        — full question text (plain text or Markdown / LaTeX)
  choices    : list[str]  — exactly 5 answer choice strings, e.g. ["1.23 m", ...]
  answer     : str        — the correct choice letter (lowercase): always 'a' when
                           using make_choices (correct value placed at index 0);
                           use 'b'–'e' only when building choices manually
  topic      : str        — brief topic label, e.g. "Ch. 4 — Newton's 2nd Law"
  difficulty : int        — difficulty level 1 (easy) to 4 (hardest)

The exam framework automatically shuffles answer positions before printing, so
there is no need to randomize the correct answer position yourself.

### Numerical entry

  type       : str        — must be "numerical"
  question   : str        — full question text (plain text or Markdown / LaTeX)
  answer     : float       — the correct numeric value
  tolerance  : float       — absolute ± tolerance, resolved via resolve_tolerance()
  unit       : str        — optional display unit, e.g. "m/s"
  sig_figs   : int        — optional display precision (default 3)
  topic      : str        — brief topic label
  difficulty : int        — difficulty level 1 (easy) to 3 (hard)

Numerical questions have no lettered choices and are graded as "within tolerance
of answer", not by exact match.

## Helper functions (import from `questions`)

```python
from questions import render_template, make_choices, phys_fmt, resolve_tolerance
```

- `render_template(question_id: str, params: dict) -> str`
  Renders the Jinja2 template associated with this question. `params` is passed as
  keyword arguments to the template context. Returns the rendered string stripped of
  leading/trailing whitespace.

- `make_choices(correct_val: float, distractors: list[float], fmt: callable) -> list[str]`
  Builds a list of 5 unique, well-spaced choice strings. The correct answer is always
  at index 0 (answer = 'a'). `fmt` is a callable that converts a float to a display
  string, e.g. `lambda v: f"{v:.2f} m"`. Multiple choice only.

- `phys_fmt(v: float, sig: int = 3) -> str`
  Formats a number with `sig` significant figures for a printed exam. Automatically
  uses LaTeX scientific notation (e.g. `$1.23 \\times 10^{4}$`) for very large or
  very small values.

- `resolve_tolerance(value: float, abs_tol: float = None, rel_tol: float = None) -> float`
  Numerical-entry only. Returns an absolute tolerance from either an absolute value
  (`abs_tol`) or a fraction of `value` (`rel_tol`, e.g. 0.02 for ±2%). Exactly one
  of the two must be given.

## Jinja2 template

The template renders the question text. Variables from the `params` dict are
available as top-level template variables. Use `{{ variable }}` for substitution
and `{% if %} / {% elif %} / {% else %} / {% endif %}` for conditionals.
LaTeX math is written inline as `$...$`.

## Runtime limits

Questions run in the browser under Pyodide, which provides numpy, jinja2 and the
Python standard library only. There is no networking (`socket`, `urllib`, `http`,
`ssl`), no `subprocess` and no real threading, so a generator must rely on plain
arithmetic, numpy and `math`.

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
    if req.textbook_catalog.strip():
        # Deliberately NOT folded into DEFAULT_SYSTEM_PROMPT: that constant is
        # duplicated verbatim in index.html and the two copies have already
        # drifted, and the client sends "" when the user has not edited it, so
        # whichever copy wins depends on whether they ever opened the textarea.
        # A dynamic block sidesteps the trap entirely.
        prompt += f"""
{req.textbook_catalog.strip()}

Grounding rules:
- Call search_textbook BEFORE authoring a new question, and base the question on
  what comes back. You get one textbook lookup per reply, so request everything
  you need at once.
- Use the book's own symbols, subscripts and unit conventions.
- Match the difficulty and phrasing of that section's end-of-section problems.
  Never reproduce a textbook problem verbatim: parametrize the numbers and write
  the wording yourself.
- Set the question's `topic` to the section number and title, e.g.
  "6.3 - Centripetal Force".
- In your chat reply, say which sections you used, e.g. "Grounded in 6.3, 6.4."
  Cite ONLY sections whose text you were actually given. If the excerpts do not
  cover what was asked, say so and search again with different terms rather than
  inventing textbook content.
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
    if req.textbook_context.strip():
        prompt += f"""
{req.textbook_context.strip()}
"""
    return prompt

# ---------------------------------------------------------------------------
# /health  (unauthenticated)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": DEFAULT_MODEL}


@app.get("/models")
async def models() -> dict:
    """List the models this server can serve right now, based on which
    provider API keys are configured. Unauthenticated, like /health, so the
    browser can populate its model picker before the user enters a token."""
    available = [
        {"id": model_id, "label": info["label"], "provider": info["provider"]}
        for model_id, info in MODEL_REGISTRY.items()
        if _provider_api_key(info["provider"])
    ]
    return {"default": DEFAULT_MODEL, "models": available}


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

_QVALIDATE = str(Path(__file__).resolve().parent / "qvalidate.py")

# One numpy child at a time: a 256 MB machine cannot host several, and validation is
# never on the latency-critical path (the Gemini stream has already finished).
_VALIDATE_SEM = asyncio.Semaphore(1)

# Defined in qvalidate so the worker, main.py and the tests cannot drift apart. Importing
# it is cheap: qvalidate defers numpy and jinja2 until validate() is actually called.
_SANDBOX_ENV = qvalidate.SANDBOX_ENV


async def _validate_question(
    template: str,
    python_code: str,
    expected_name: str | None = None,
) -> tuple[str, str]:
    """Run the question in a sandboxed subprocess; return (state, message).

    state is one of:
      "ok"          — the question satisfies the exam contract
      "invalid"     — the question is broken; worth asking the model to fix it
      "unavailable" — this sandbox is broken (missing dependency, crashed child).
                      Never spend model calls "fixing" a question over this.

    The work happens in server/qvalidate.py, which mirrors the browser's Pyodide
    preview (index.html:1893-1916).  It runs out-of-process for three reasons: the
    parent never imports numpy, a runaway can actually be SIGKILLed, and the child
    cannot reach GOOGLE_API_KEY / API_TOKEN.  Note this is a pre-filter, not an
    oracle — the browser preview remains the authority on whether a question runs.
    """
    job = json.dumps({
        "template": template,
        "python_code": python_code,
        "expected_name": expected_name or "",
    }).encode()

    async with _VALIDATE_SEM:
        try:
            proc = await asyncio.create_subprocess_exec(
                # -I isolates from PYTHONPATH and user site-packages; -B stops bytecode
                # writes, which the child's RLIMIT_FSIZE=0 would otherwise block.
                sys.executable, "-I", "-B", _QVALIDATE,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_SANDBOX_ENV,
            )
        except OSError as exc:
            return "unavailable", f"could not start the validation sandbox: {exc}"

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(job), timeout=VALIDATE_TIMEOUT
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return "invalid", (
                f"the question did not finish running within {VALIDATE_TIMEOUT:.0f} s — "
                "generate(rng) is probably stuck in an infinite loop"
            )

    try:
        result = json.loads(stdout.decode())
        state = result["state"]
        message = result.get("message", "")
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        detail = stderr.decode(errors="replace").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        if proc.returncode is not None and proc.returncode < 0:
            # Killed by a signal — RLIMIT_CPU (SIGXCPU) or our own kill. The question
            # exhausted its budget, so it is worth asking the model to fix it.
            return "invalid", (
                "the question exceeded the sandbox CPU or memory budget — generate(rng) "
                f"is probably stuck in a loop or allocating without bound ({tail})"
            )
        return "unavailable", (
            f"validation sandbox exited with code {proc.returncode}: {tail}"
        )

    if state not in ("ok", "invalid", "unavailable"):
        return "unavailable", f"validation sandbox returned an unknown state {state!r}"
    return state, message


async def _validation_canary() -> None:
    """Prove the sandbox works at boot, so a broken deploy is one line in `fly logs`.

    Without this, a forgotten COPY in the Dockerfile silently turns every question
    into a validation failure and burns two Gemini calls apiece — exactly the bug
    this validator was rewritten to fix.
    """
    state, message = await _validate_question(
        template="{{ n }} apples.",
        python_code=(
            "import numpy as np\n"
            "from questions import render_template\n"
            "def generate(rng: np.random.Generator) -> dict:\n"
            "    n = int(rng.integers(2, 9))\n"
            "    return {\n"
            '        "question": render_template("canary", {"n": n}),\n'
            '        "choices": ["a", "b", "c", "d", "e"],\n'
            '        "answer": "a",\n'
            '        "topic": "canary",\n'
            '        "difficulty": 1,\n'
            "    }\n"
        ),
        expected_name="canary",
    )
    if state == "ok":
        print("[validate] environment OK", flush=True)
    else:
        print(
            f"[validate] SANDBOX BROKEN ({state}): {message} — "
            "AI-authored questions will not be validated",
            flush=True,
        )


_FIX_REQUIREMENTS_TEXT = (
    "Requirements:\n"
    "  • python_code must define generate(rng: numpy.random.Generator) -> dict\n"
    "  • It must return question (str), choices (a list of exactly 5 distinct "
    "strings), answer (one of 'a'-'e'), topic (str) and difficulty (int 1-4)\n"
    "  • Build the question text with render_template(question_id, params), "
    "passing this question's own id\n"
    "  • Every Jinja2 variable used in the template must be a key in that params "
    "dict, and the template must render for any rng seed\n"
    "  • Only numpy, jinja2 and the Python standard library are available, "
    "and the question runs in the browser under Pyodide: no networking "
    "(socket, urllib, http, ssl), no subprocess and no threading\n"
    "  • Use plain arithmetic, numpy and the math module — a question "
    "generator never needs to reach outside the process"
)


def _format_validation_error(err: str, limit: int = 400) -> str:
    """Collapse whitespace and cap length for embedding in a user-facing warning.

    ``err`` carries an arbitrary exception message from AI-generated code, which
    can be multi-line and unbounded (a deep Jinja2 chain, a repr of a large
    value).  Left raw it would bloat the SSE payload and stretch the warning row
    in the browser, so normalise it here rather than at either consumer.
    """
    collapsed = " ".join(err.split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit - 1].rstrip() + "…"
    return collapsed


async def _gemini_fix_call(
    model: str,
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
                    f"{_FIX_REQUIREMENTS_TEXT}"
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
    url = f"{GEMINI_BASE}/{model}:generateContent"
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
# Anthropic (Claude) — tool format conversion + fix call
# ---------------------------------------------------------------------------

def _to_anthropic_tools(excluded: set[str] | None = None) -> list[dict]:
    """Convert the OpenAI-style TOOLS list to Anthropic's tool schema."""
    excluded = excluded or set()
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"].get("parameters", {}),
        }
        for t in TOOLS
        if t["function"]["name"] not in excluded
    ]


def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict]:
    return [
        {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
        for m in messages
    ]


async def _anthropic_fix_call(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_text: str,
    messages: list[dict],
    fc_name: str,
    fc_args: dict,
    error: str,
) -> dict | None:
    """Non-streaming Claude call that feeds a validation error back as a
    tool_result and forces the model to return a corrected tool call.
    Returns the new input dict, or None if Claude didn't return a tool call.
    """
    fix_messages = messages + [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "fix_call", "name": fc_name, "input": fc_args},
        ]},
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": "fix_call",
                "is_error": True,
                "content": (
                    f"Validation failed: {error}\n\n"
                    "Please fix the template and python_code so they work together without errors. "
                    f"{_FIX_REQUIREMENTS_TEXT}"
                ),
            },
        ]},
    ]
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_text,
            messages=fix_messages,
            tools=_to_anthropic_tools(),
            tool_choice={"type": "tool", "name": fc_name},
            temperature=0.2,
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == fc_name:
                return block.input
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] anthropic fix call exception: {exc}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Shared post-processing: validate/auto-fix tool calls, emit SSE events
# ---------------------------------------------------------------------------

async def _validate_and_fix_calls(
    function_calls: list[dict],
    req: ChatRequest,
    fix_call: Callable[[str, dict, str], Awaitable[dict | None]],
    validation_warnings: dict[int, str],
) -> None:
    """Validate update_question/create_question calls and try to auto-fix
    them via `fix_call(name, args, error) -> dict | None`. Mutates
    `function_calls` in place with any fixed args, and records an entry in
    `validation_warnings` (keyed by function-call index) for any call that
    still fails validation after MAX_FIX_ATTEMPTS — the emit loop attaches
    these to the tool_call payload itself, since a free-floating text delta
    would be lost (the browser overwrites the assistant bubble with its
    "Proposing ..." line when the tool_call arrives).
    """
    for i, fc in enumerate(function_calls):
        name = fc.get("name", "")
        args = fc.get("args", {})
        if name not in ("update_question", "create_question"):
            continue
        # An update_question only has to carry the field it changes; the browser
        # applies it on top of the editor's current content, so validate that same
        # pair.  Single-field updates are the common case and used to skip
        # validation entirely.  A create_question has no editor content to fall
        # back on — borrowing the open question's template would validate a pair
        # that never exists — so it must supply both itself.
        fallback_template = req.template if name == "update_question" else ""
        fallback_python = req.python_code if name == "update_question" else ""
        template = args.get("template") or fallback_template
        python_code = args.get("python_code") or fallback_python
        if not (template and python_code):
            continue
        expected_name = args.get("question_id") or req.question_id or None

        state, err = await _validate_question(template, python_code, expected_name)
        if state == "ok":
            print(f"[chat] validation ok for {name}", flush=True)
            continue
        if state == "unavailable":
            # Our problem, not the model's — never burn fix calls on it.
            print(f"[chat] validation unavailable for {name}: {err}", flush=True)
            continue

        print(f"[chat] validation failed for {name}: {err[:200]}", flush=True)
        for fix_attempt in range(MAX_FIX_ATTEMPTS):
            print(f"[chat] fix attempt {fix_attempt + 1}/{MAX_FIX_ATTEMPTS}", flush=True)
            fixed_args = await fix_call(name, args, err)
            if fixed_args is None:
                print("[chat] fix call returned no function call", flush=True)
                break
            # Merge, don't replace: a fix that returns only python_code must not
            # drop question_id/title, nor a template repaired on an earlier attempt.
            args = {**args, **fixed_args}
            template = args.get("template") or fallback_template
            python_code = args.get("python_code") or fallback_python
            state, err = await _validate_question(template, python_code, expected_name)
            if state == "ok":
                # Only now is the rewrite worth showing the user.
                function_calls[i] = {**fc, "args": args}
                print(f"[chat] fixed on attempt {fix_attempt + 1}", flush=True)
                break
            if state == "unavailable":
                print(f"[chat] validation unavailable mid-fix: {err}", flush=True)
                break
            print(f"[chat] fix attempt {fix_attempt + 1} still failing: {err[:200]}", flush=True)

        if state != "ok":
            # Emit the model's ORIGINAL tool call untouched — a failed repair is
            # usually worse than what it started from.
            print(f"[chat] giving up on {name}; emitting the original tool call", flush=True)
            validation_warnings[i] = (
                f"I could not verify this code runs without errors after "
                f"{MAX_FIX_ATTEMPTS} fix attempt(s). "
                f"Last error: {_format_validation_error(err)} "
                f"— please review carefully before accepting."
            )


async def _emit_tool_calls(
    function_calls: list[dict],
    req: ChatRequest,
    validation_warnings: dict[int, str],
) -> AsyncIterator[dict]:
    """Emit the (possibly fixed) tool calls as SSE tool_call events, handling
    the get_question_bank / get_question round-trip requests."""
    for i, fc in enumerate(function_calls):
        name = fc.get("name", "")
        args = fc.get("args", {})
        if name == "get_question_bank":
            if req.bank_summary.strip():
                print("[chat] suppressing get_question_bank — bank_summary already provided", flush=True)
                continue
            print("[chat] AI requested get_question_bank", flush=True)
            yield {"data": json.dumps({"type": "tool_call", "tool": "get_question_bank"})}
            continue
        if name == "get_question":
            if req.requested_question_id.strip():
                print("[chat] suppressing get_question — requested_question already provided", flush=True)
                continue
            requested_id = args.get("question_id", "")
            print(f"[chat] AI requested get_question: {requested_id}", flush=True)
            yield {"data": json.dumps({"type": "tool_call", "tool": "get_question", "question_id": requested_id})}
            continue
        if name == "search_textbook":
            if req.textbook_context.strip():
                print("[chat] suppressing search_textbook — textbook_context already provided", flush=True)
                continue
            print(f"[chat] AI requested search_textbook: {str(args.get('query', ''))[:80]}", flush=True)
            yield {"data": json.dumps({
                "type": "tool_call",
                "tool": "search_textbook",
                "query": args.get("query", ""),
                "section_ids": args.get("section_ids", []),
                "books": args.get("books", []),
                "chapters": args.get("chapters", []),
                "include": args.get("include", []),
                "max_sections": args.get("max_sections", 3),
            })}
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
        if i in validation_warnings:
            payload["validation_warning"] = validation_warnings[i]
        yield {"data": json.dumps(payload)}


def _excluded_tools(req: ChatRequest) -> set[str]:
    """Tools to drop from the request — suppresses get_question_bank /
    get_question once their data is already inlined in the system prompt,
    which otherwise invites the model into a redundant call loop."""
    excluded: set[str] = set()
    if req.bank_summary.strip():
        excluded.add("get_question_bank")
    if req.requested_question_id.strip():
        excluded.add("get_question")
    if req.textbook_context.strip():
        excluded.add("search_textbook")
    elif not req.textbook_catalog.strip():
        # No catalog means no corpus was built, or the question set has every
        # book unchecked.  Offering the tool would invite a call the client
        # cannot answer.
        excluded.add("search_textbook")
    return excluded


async def _stream_gemini(model: str, req: ChatRequest) -> AsyncIterator[dict]:
    system_text, contents = _to_gemini_contents(_system_prompt(req), req.messages)

    excluded = _excluded_tools(req)
    gemini_tools = _to_gemini_tools()
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

    url = f"{GEMINI_BASE}/{model}:streamGenerateContent"
    params = {"key": GOOGLE_API_KEY, "alt": "sse"}

    print(f"[chat] model={model} (gemini) msgs={len(req.messages)}", flush=True)

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

        async def _fix(name: str, args: dict, error: str) -> dict | None:
            return await _gemini_fix_call(model, system_text, contents, name, args, error)

        validation_warnings: dict[int, str] = {}
        await _validate_and_fix_calls(function_calls, req, _fix, validation_warnings)
        async for ev in _emit_tool_calls(function_calls, req, validation_warnings):
            yield ev
        yield {"data": json.dumps({"type": "done"})}

    except Exception as exc:  # noqa: BLE001
        print(f"[chat] exception: {exc}", flush=True)
        yield {"data": json.dumps({"type": "error", "message": str(exc)})}


async def _stream_anthropic(model: str, req: ChatRequest) -> AsyncIterator[dict]:
    system_text = _system_prompt(req)
    messages = _to_anthropic_messages(req.messages)
    tools = _to_anthropic_tools(_excluded_tools(req))
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    print(f"[chat] model={model} (anthropic) msgs={len(req.messages)}", flush=True)

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_text,
            messages=messages,
            tools=tools,
        ) as stream:
            async for text in stream.text_stream:
                yield {"data": json.dumps({"type": "text", "delta": text})}
            final = await stream.get_final_message()
    except anthropic.APIStatusError as exc:
        print(f"[chat] anthropic error: {exc}", flush=True)
        yield {"data": json.dumps({"type": "error", "message": f"Claude {exc.status_code}: {str(exc.message)[:300]}"})}
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] exception: {exc}", flush=True)
        yield {"data": json.dumps({"type": "error", "message": str(exc)})}
        return

    function_calls = [
        {"name": block.name, "args": block.input}
        for block in final.content
        if block.type == "tool_use"
    ]
    print(f"[chat] done: tool_calls={len(function_calls)}", flush=True)

    async def _fix(name: str, args: dict, error: str) -> dict | None:
        return await _anthropic_fix_call(client, model, system_text, messages, name, args, error)

    validation_warnings: dict[int, str] = {}
    await _validate_and_fix_calls(function_calls, req, _fix, validation_warnings)
    async for ev in _emit_tool_calls(function_calls, req, validation_warnings):
        yield ev
    yield {"data": json.dumps({"type": "done"})}


# ---------------------------------------------------------------------------
# /chat  (SSE streaming)
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    _check_token(request)

    model = _normalize_model(req.model) or DEFAULT_MODEL
    provider = _provider_for_model(model)

    if provider == "anthropic" and not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured on server.")
    if provider == "gemini" and not GOOGLE_API_KEY:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not configured on server.")

    stream_fn = _stream_anthropic if provider == "anthropic" else _stream_gemini
    return EventSourceResponse(stream_fn(model, req), ping=0)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
