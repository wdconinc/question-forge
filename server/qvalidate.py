"""
Question validation sandbox
===========================
Executes an AI-authored question (Jinja2 template + Python generator) exactly the
way the browser does, and reports whether it satisfies the exam contract.

Reference implementations this mirrors:
  - index.html:1893-1916        (Pyodide preview path)
  - python/exam_core.py:110-130 (render-all path)

Both compile the template from a string, monkey-patch ``questions.render_template``
to render *that* template, exec the python into a bare namespace, and call
``generate(numpy.random.default_rng(seed))``.  The template is exercised as a side
effect of ``generate()`` — it is never rendered standalone, because the dict
``generate()`` returns is NOT the template's parameter dict.

This is a cheap **pre-filter, not an oracle**.  Pyodide and CPython are not the same
runtime (Pyodide has no working threading/socket/subprocess), so a pass here does not
guarantee a pass in the browser.  The browser preview remains the authority.

Run as a script, this module is the sandbox worker: it drops resource limits, reads
one JSON job from stdin and writes one JSON result to stdout.  ``server/main.py``
spawns it as a subprocess with a scrubbed environment so that model-authored code
cannot reach GOOGLE_API_KEY / API_TOKEN, and so a runaway can actually be killed.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import sys

# Jinja2 settings must match the browser preview (index.html:1897-1901) and the
# render-all path (python/exam_core.py:113-117) exactly.  A mismatch here means
# whitespace-sensitive templates validate differently than they render.
_ENV_KW = {
    "trim_blocks": True,
    "lstrip_blocks": True,
    "keep_trailing_newline": False,
}

# Fixed, never random: a validator that fails one call in twenty and passes the retry
# is worse than no validator.  42/137/271 are the seedA/seedB/seedC defaults
# (index.html:1284), so validation covers the seed the first preview will actually use.
SEEDS = (42, 137, 271, 7, 99)

LETTERS = ("a", "b", "c", "d", "e")

# The browser runtime loads numpy + micropip, then jinja2 via micropip
# (index.html:1362-1366).  Everything else it has is the Python standard library.
# A missing *third-party* module is therefore a real failure, not a validator gap —
# but a missing one of these three means OUR environment is broken.
_RUNTIME_MODULES = frozenset({"numpy", "jinja2", "questions"})

# Question generators do scalar arithmetic. Left at its default OpenBLAS spawns a thread
# per core and reserves over a gigabyte of address space, which both wastes a
# shared-cpu-1x machine and aborts outright under RLIMIT_AS. The worker applies these
# itself rather than trusting the parent, because losing them turns every validation into
# a hard numpy abort - silently disabling validation altogether.
_THREAD_CAPS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

# The environment main.py hands the worker. Deliberately minimal: model-authored code
# must not be able to read GOOGLE_API_KEY or the shared API_TOKEN.
SANDBOX_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
    **_THREAD_CAPS,
}

# Modules the browser cannot run. Pyodide has no raw sockets, no subprocess and no real
# threads, so question code importing any of these would pass here and then fail in the
# preview — a false positive, which is the failure mode this validator exists to avoid.
# Blocking them also narrows what model-authored code can reach.
#
# This is a parity lint and defence in depth, NOT the security boundary. It is bypassable
# (obfuscated __import__, a module already bound to another name). The boundary is the
# subprocess itself: a scrubbed environment with no API keys, RLIMIT_NPROC=0 so nothing
# can fork or exec, RLIMIT_FSIZE=0 so nothing can write, and a parent that can SIGKILL it.
_BLOCKED_IMPORTS = frozenset({
    "socket", "ssl", "subprocess", "multiprocessing", "_thread", "threading",
    "concurrent", "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib",
    "xmlrpc", "webbrowser", "ctypes", "importlib",
})

_MAX_MSG = 400

OK = "ok"
INVALID = "invalid"
UNAVAILABLE = "unavailable"


def _truncate(msg: str) -> str:
    """Messages end up in a Gemini prompt and a log line; keep them short."""
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= _MAX_MSG else msg[: _MAX_MSG - 1] + "…"


def _missing_module(exc: ImportError) -> str:
    return getattr(exc, "name", "") or ""


def _classify_import_error(exc: ImportError, prefix: str = "") -> tuple[str, str]:
    """Map an ImportError to (state, message).

    A missing numpy/jinja2/questions means OUR sandbox is broken, so it must report
    "unavailable" and skip the fix loop rather than blaming the model for it.
    """
    missing = _missing_module(exc)
    root = missing.partition(".")[0]
    if root in _RUNTIME_MODULES:
        return UNAVAILABLE, _truncate(f"validation sandbox is missing {missing}")
    if root in _BLOCKED_IMPORTS:
        return INVALID, _truncate(f"{prefix}{exc}")
    return INVALID, _truncate(
        f"{prefix}cannot import {missing or exc!r}: the browser runtime provides only "
        "numpy, jinja2 and the Python standard library. Rewrite without it."
    )


@contextlib.contextmanager
def _blocked_imports():
    """Refuse _BLOCKED_IMPORTS for the duration of the block.

    Patches builtins.__import__ rather than installing a sys.meta_path finder: the
    import statement always calls __import__, whereas meta_path is skipped entirely
    for anything already in sys.modules (numpy and jinja2 drag in several of these).
    Blocking importlib closes the importlib.import_module route at the same time.
    """
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.partition(".")[0]
        if level == 0 and root in _BLOCKED_IMPORTS:
            raise ImportError(
                f"{root} is not usable in the browser runtime (Pyodide has no raw "
                "sockets, no subprocess and no real threads) and is not available here",
                name=root,
            )
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _check_result(d: object, seed: int) -> str:
    """Return an error message, or "" if *d* satisfies the contract.

    question/choices/answer are required — index.html:1910-1912 subscripts them
    directly.  topic/difficulty are optional; index.html:1913-1914 defaults them.
    """
    at = f"(seed {seed})"
    if not isinstance(d, dict):
        return f"{at} generate(rng) must return a dict, got {type(d).__name__}"

    for key in ("question", "choices", "answer"):
        if key not in d:
            return f"{at} the dict returned by generate(rng) is missing the '{key}' key"

    question = d["question"]
    if not isinstance(question, str) or not question.strip():
        return f"{at} 'question' must be a non-empty string"

    choices = d["choices"]
    if isinstance(choices, str) or not isinstance(choices, (list, tuple)):
        return f"{at} 'choices' must be a list of 5 strings, got {type(choices).__name__}"
    if len(choices) != 5:
        return (
            f"{at} 'choices' must have exactly 5 entries, got {len(choices)} — "
            "make_choices() drops distractors that collide with the correct value or "
            "each other. Pick better-separated distractors, or build the 5 choices by hand."
        )
    for i, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            return f"{at} choices[{i}] must be a non-empty string, got {choice!r}"
    if len(set(choices)) != 5:
        return f"{at} the 5 entries in 'choices' must be distinct, got {list(choices)}"

    answer = d["answer"]
    if not isinstance(answer, str) or answer not in LETTERS:
        return f"{at} 'answer' must be one of 'a'-'e', got {answer!r}"

    if "topic" in d and not isinstance(d["topic"], str):
        return f"{at} 'topic' must be a string, got {type(d['topic']).__name__}"

    if "difficulty" in d:
        try:
            difficulty = int(d["difficulty"])
        except (TypeError, ValueError):
            return f"{at} 'difficulty' must be an integer 1-4, got {d['difficulty']!r}"
        if not 1 <= difficulty <= 4:
            return f"{at} 'difficulty' must be between 1 (easy) and 4 (hardest), got {difficulty}"

    return ""


def validate(
    template: str,
    python_code: str,
    expected_name: str | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[str, str]:
    """Run *python_code* against *template* and return (state, message).

    state is one of "ok" (contract satisfied), "invalid" (the question is broken —
    worth asking the model to fix), or "unavailable" (this sandbox is broken — do
    NOT spend model calls trying to fix the question).
    """
    try:
        import jinja2
        import numpy

        import questions
    except ImportError as exc:
        return UNAVAILABLE, _truncate(
            f"validation sandbox is missing {_missing_module(exc) or exc}"
        )

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, **_ENV_KW)
    try:
        compiled = env.from_string(template)
    except jinja2.TemplateSyntaxError as exc:
        return INVALID, _truncate(f"the Jinja2 template has a syntax error: {exc}")

    # Both runtimes replace render_template on the MODULE OBJECT before exec, so that
    # `from questions import render_template` binds the patched version.  The `name`
    # argument is ignored (only one template exists) — but we record it, because the
    # ZIP export (index.html:2157-2176) ships the real FileSystemLoader-backed helper
    # and looks the template up as "<name>.j2" on disk.
    seen_names: list[str] = []

    def _render_template(name: str, params: dict) -> str:
        seen_names.append(name)
        return compiled.render(**params).strip()

    original = questions.render_template
    questions.render_template = _render_template
    try:
        namespace: dict = {}
        try:
            with _blocked_imports():
                exec(compile(python_code, f"{expected_name or 'question'}.py", "exec"), namespace)  # noqa: S102
        except ImportError as exc:
            return _classify_import_error(exc)
        except Exception as exc:  # noqa: BLE001
            return INVALID, _truncate(f"python_code failed to load: {type(exc).__name__}: {exc}")

        generate = namespace.get("generate")
        if generate is None:
            return INVALID, _truncate(
                "python_code must define a generate(rng: numpy.random.Generator) -> dict function"
            )
        if not callable(generate):
            return INVALID, _truncate(
                f"`generate` must be a function, not a {type(generate).__name__}"
            )

        for seed in seeds:
            seen_names.clear()
            try:
                with _blocked_imports():
                    result = generate(numpy.random.default_rng(seed))
            except jinja2.UndefinedError as exc:
                return INVALID, _truncate(
                    f"(seed {seed}) rendering the template raised {exc} — every variable used "
                    "in the template must be a key in the params dict passed to render_template()"
                )
            except ImportError as exc:
                return _classify_import_error(exc, prefix=f"(seed {seed}) ")
            except Exception as exc:  # noqa: BLE001
                return INVALID, _truncate(
                    f"(seed {seed}) generate(rng) raised {type(exc).__name__}: {exc}"
                )

            error = _check_result(result, seed)
            if error:
                return INVALID, _truncate(error)

            if expected_name:
                wrong = [n for n in seen_names if n != expected_name]
                if wrong:
                    return INVALID, _truncate(
                        f"python_code calls render_template({wrong[0]!r}, ...) but this question's "
                        f"id is {expected_name!r}. The template is stored as {expected_name}.j2, so "
                        f"the name passed to render_template must be {expected_name!r}."
                    )

        return OK, ""
    finally:
        questions.render_template = original


# ---------------------------------------------------------------------------
# Sandbox worker entry point
# ---------------------------------------------------------------------------

def _drop_privileges() -> None:
    """Bound CPU, address space, file writes and forks before touching untrusted code.

    Set here in Python rather than via subprocess preexec_fn, which is thread-unsafe
    and awkward under asyncio.  Each limit is best-effort: a platform that refuses one
    should not stop validation from running at all — the parent's kill is the backstop.
    """
    try:
        import resource
    except ImportError:      # non-POSIX; the parent timeout still applies
        return

    for name, limit in (
        ("RLIMIT_CPU", (5, 6)),
        ("RLIMIT_FSIZE", (0, 0)),
        ("RLIMIT_NPROC", (0, 0)),
        # Virtual address space, not residency: numpy/OpenBLAS reserve far more than
        # they use, and below ~1 GiB `import numpy` aborts outright. 2 GiB still bounds
        # a runaway allocation long before it matters on a 256 MB machine.
        ("RLIMIT_AS", (2 << 30, 2 << 30)),
    ):
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            _, hard = resource.getrlimit(which)
            want_soft, want_hard = limit
            if hard != resource.RLIM_INFINITY:
                want_soft = min(want_soft, hard)
                want_hard = min(want_hard, hard)
            resource.setrlimit(which, (want_soft, want_hard))
        except (ValueError, OSError):
            continue


def _main() -> int:
    # The parent launches us with -I, which implies -P: the script's own directory is
    # NOT on sys.path, so `import questions` would fail without this.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Must precede the first numpy import, which validate() defers until it is called.
    for key, value in _THREAD_CAPS.items():
        os.environ.setdefault(key, value)
    _drop_privileges()

    try:
        job = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"qvalidate: malformed job: {exc}", file=sys.stderr)
        return 2

    state, message = validate(
        template=job.get("template", ""),
        python_code=job.get("python_code", ""),
        expected_name=job.get("expected_name") or None,
    )
    sys.stdout.write(json.dumps({"state": state, "message": message}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
