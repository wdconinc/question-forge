"""
Contract tests for qvalidate.py — run with `python server/test_qvalidate.py`.

The headline test is test_default_bank: every question shipped in
src/default_bank.js must validate.  Before this suite existed the server
validated a `generate_params()` contract that no question implements, so all 30
failed with "ModuleNotFoundError: No module named 'questions'" and each one cost
two wasted Gemini calls.

Uses unittest rather than pytest — the repo has no pytest dependency.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import qvalidate

REPO = pathlib.Path(__file__).resolve().parent.parent
BANK_JS = REPO / "src" / "default_bank.js"
INDEX_HTML = REPO / "index.html"
WORKER = str(pathlib.Path(__file__).resolve().parent / "qvalidate.py")


def import_main():
    """Import server/main.py, or None when fastapi & friends aren't installed."""
    try:
        import main
    except ImportError:
        return None
    return main


def load_bank() -> dict:
    """src/default_bank.js is `export const DEFAULT_BANK = <json.dumps(...)>;`."""
    text = BANK_JS.read_text()
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


# A minimal question that satisfies the contract, used as the base for mutations.
GOOD_TEMPLATE = "A cart accelerates at {{ a }} m/s^2 for {{ t }} s.\n"
GOOD_PYTHON = '''
import numpy as np
from questions import make_choices, render_template

def generate(rng: np.random.Generator) -> dict:
    a = float(rng.choice([1.5, 2.0, 2.5]))
    t = float(rng.choice([3.0, 4.0, 5.0]))
    x = 0.5 * a * t**2
    params = {"a": f"{a:.1f}", "t": f"{t:.1f}"}
    return {
        "question": render_template("q_good", params),
        "choices": make_choices(x, [a * t**2, a * t, 0.5 * a * t, a * t**2 / 4],
                                lambda v: f"{v:.1f} m"),
        "answer": "a",
        "topic": "Kinematics",
        "difficulty": 2,
    }
'''


class TestDefaultBank(unittest.TestCase):
    """The regression test: the shipped bank must satisfy the validated contract."""

    def test_every_question_validates(self):
        bank = load_bank()
        self.assertEqual(len(bank["question_order"]), 31)
        failures = []
        for qid in bank["question_order"]:
            q = bank["questions"][qid]
            state, message = qvalidate.validate(
                q["template"], q["python_code"], expected_name=qid
            )
            if state != "ok":
                failures.append(f"{qid}: [{state}] {message}")
        self.assertEqual(failures, [], "\n".join(failures))


class TestContract(unittest.TestCase):
    def check(self, template=GOOD_TEMPLATE, python_code=GOOD_PYTHON,
              expected_name="q_good", **kw):
        return qvalidate.validate(template, python_code, expected_name=expected_name, **kw)

    def assert_invalid(self, needle, **kw):
        state, message = self.check(**kw)
        self.assertEqual(state, "invalid", f"got {state}: {message}")
        self.assertIn(needle, message)
        return message

    def test_baseline_is_ok(self):
        self.assertEqual(self.check(), ("ok", ""))

    def test_generate_params_is_rejected(self):
        # The contract the old validator demanded. It must now fail.
        self.assert_invalid(
            "must define a generate(rng",
            python_code=GOOD_PYTHON.replace("def generate(", "def generate_params("),
        )

    def test_generate_must_accept_rng(self):
        self.assert_invalid(
            "generate(rng) raised TypeError",
            python_code=GOOD_PYTHON.replace(
                "def generate(rng: np.random.Generator)", "def generate()"),
        )

    def test_four_choices_rejected(self):
        msg = self.assert_invalid(
            "exactly 5 entries, got 4",
            python_code=GOOD_PYTHON.replace(
                'lambda v: f"{v:.1f} m")', 'lambda v: f"{v:.1f} m")[:4]'),
        )
        self.assertIn("seed 42", msg)   # failures must be reproducible

    def test_duplicate_choices_rejected(self):
        self.assert_invalid(
            "must be distinct",
            python_code=GOOD_PYTHON.replace(
                '"choices": make_choices(x, [a * t**2, a * t, 0.5 * a * t, a * t**2 / 4],\n'
                '                                lambda v: f"{v:.1f} m")',
                '"choices": ["1 m"] * 5'),
        )

    def test_bad_answer_letter_rejected(self):
        self.assert_invalid('must be one of', python_code=GOOD_PYTHON.replace(
            '"answer": "a"', '"answer": "f"'))

    def test_answer_b_through_e_allowed(self):
        # make_choices puts the correct value at index 0, but hand-built choices may
        # legitimately use b-e; DEFAULT_SYSTEM_PROMPT says so explicitly.
        state, message = self.check(python_code=GOOD_PYTHON.replace(
            '"answer": "a"', '"answer": "d"'))
        self.assertEqual(state, "ok", message)

    def test_difficulty_out_of_range_rejected(self):
        self.assert_invalid("between 1", python_code=GOOD_PYTHON.replace(
            '"difficulty": 2', '"difficulty": 9'))

    def test_difficulty_four_allowed(self):
        # q31/q32/q33 in the shipped bank return 4, and both STARS maps cover 1-4.
        state, message = self.check(python_code=GOOD_PYTHON.replace(
            '"difficulty": 2', '"difficulty": 4'))
        self.assertEqual(state, "ok", message)

    def test_missing_template_variable_rejected(self):
        self.assert_invalid("'t' is undefined", python_code=GOOD_PYTHON.replace(
            'params = {"a": f"{a:.1f}", "t": f"{t:.1f}"}', 'params = {"a": f"{a:.1f}"}'))

    def test_template_syntax_error_rejected(self):
        self.assert_invalid("syntax error", template="{% if %}oops{% endif %}")

    def test_python_syntax_error_rejected(self):
        self.assert_invalid("failed to load", python_code="def generate(rng)\n    pass\n")

    def test_render_template_name_must_match_question_id(self):
        self.assert_invalid(
            "The template is stored as",
            python_code=GOOD_PYTHON.replace('render_template("q_good"',
                                            'render_template("q_typo"'),
        )

    def test_unknown_third_party_import_rejected(self):
        # Pyodide loads numpy + jinja2 only, so an unavailable import is a real failure
        # rather than a validator gap. Uses a module that exists nowhere, because a dev
        # machine may well have scipy/sympy installed while the container does not.
        self.assert_invalid(
            "browser runtime provides only",
            python_code="import totally_not_a_real_module\n" + GOOD_PYTHON,
        )

    def test_blocked_modules_are_rejected(self):
        """Pyodide has no sockets/subprocess/threads, so these must fail here too.

        Otherwise the validator passes code that then breaks in the preview — the
        false-positive direction this validator exists to prevent.
        """
        for module in ("socket", "subprocess", "threading", "multiprocessing",
                       "urllib", "ssl", "ctypes", "importlib"):
            with self.subTest(module=module):
                self.assert_invalid(
                    "not usable in the browser runtime",
                    python_code=f"import {module}\n" + GOOD_PYTHON,
                )

    def test_blocked_modules_rejected_inside_generate(self):
        """The guard must cover generate()'s body, not just module import time."""
        self.assert_invalid(
            "not usable in the browser runtime",
            python_code=GOOD_PYTHON.replace(
                "    a = float(rng.choice([1.5, 2.0, 2.5]))",
                "    import socket  # noqa\n    a = float(rng.choice([1.5, 2.0, 2.5]))"),
        )

    def test_dunder_import_bypass_is_blocked(self):
        """builtins.__import__ is patched, so the obvious bypass fails too."""
        self.assert_invalid(
            "not usable in the browser runtime",
            python_code='__import__("socket")\n' + GOOD_PYTHON,
        )

    def test_allowed_modules_still_work(self):
        """The denylist must not create false failures: math is used by 4 bank questions."""
        state, message = self.check(
            python_code=GOOD_PYTHON.replace("import numpy as np", "import math\nimport numpy as np"))
        self.assertEqual(state, "ok", message)

    def test_seed_sensitivity(self):
        # q27_temperature used to return 3 choices at seeds 195/207: make_choices could
        # not build fallbacks when the correct value is 0 (32 F -> 0 C).  The additive
        # fallback fixed that, so both seeds must now validate — this is the regression
        # guard for it, and still the reason validation sweeps several seeds.
        bank = load_bank()
        q = bank["questions"]["q27_temperature"]
        for seed in (195, 207):
            state, message = qvalidate.validate(
                q["template"], q["python_code"],
                expected_name="q27_temperature", seeds=(seed,))
            self.assertEqual(state, "ok", f"seed {seed}: {message}")


# A minimal numerical-entry question that satisfies that contract (no 'choices',
# see DEFAULT_SYSTEM_PROMPT's "Numerical entry" section in server/main.py).
GOOD_NUMERICAL_TEMPLATE = "A cart accelerates at {{ a }} m/s^2 for {{ t }} s.\n"
GOOD_NUMERICAL_PYTHON = '''
import numpy as np
from questions import resolve_tolerance, render_template

def generate(rng: np.random.Generator) -> dict:
    a = float(rng.choice([1.5, 2.0, 2.5]))
    t = float(rng.choice([3.0, 4.0, 5.0]))
    v = a * t
    params = {"a": f"{a:.1f}", "t": f"{t:.1f}"}
    return {
        "type": "numerical",
        "question": render_template("q_good", params),
        "answer": v,
        "tolerance": resolve_tolerance(v, rel_tol=0.02),
        "unit": "m/s",
        "topic": "Kinematics",
        "difficulty": 2,
    }
'''


class TestNumericalContract(unittest.TestCase):
    """type: "numerical" questions have no 'choices' and a different answer shape —
    a regression suite for the bug where _check_result required 'choices'
    unconditionally and rejected every valid numerical-entry question."""

    def check(self, template=GOOD_NUMERICAL_TEMPLATE, python_code=GOOD_NUMERICAL_PYTHON,
              expected_name="q_good", **kw):
        return qvalidate.validate(template, python_code, expected_name=expected_name, **kw)

    def assert_invalid(self, needle, **kw):
        state, message = self.check(**kw)
        self.assertEqual(state, "invalid", f"got {state}: {message}")
        self.assertIn(needle, message)
        return message

    def test_baseline_is_ok(self):
        self.assertEqual(self.check(), ("ok", ""))

    def test_missing_choices_is_not_required(self):
        # The bug this class guards against: a numerical question has no 'choices'
        # key at all, and must not be rejected for lacking one.
        state, message = self.check()
        self.assertEqual(state, "ok", message)

    def test_missing_tolerance_rejected(self):
        self.assert_invalid(
            "missing the 'tolerance' key",
            python_code=GOOD_NUMERICAL_PYTHON.replace(
                '"tolerance": resolve_tolerance(v, rel_tol=0.02),\n', ""),
        )

    def test_non_numeric_answer_rejected(self):
        self.assert_invalid(
            "'answer' must be a number",
            python_code=GOOD_NUMERICAL_PYTHON.replace('"answer": v,', '"answer": str(v),'),
        )

    def test_negative_tolerance_rejected(self):
        self.assert_invalid(
            "must be non-negative",
            python_code=GOOD_NUMERICAL_PYTHON.replace(
                '"tolerance": resolve_tolerance(v, rel_tol=0.02),',
                '"tolerance": -1.0,'),
        )

    def test_difficulty_out_of_range_rejected(self):
        self.assert_invalid("between 1", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"difficulty": 2', '"difficulty": 9'))

    def test_numpy_integer_answer_accepted(self):
        # rng.integers() returns np.int64, which is not an int subclass — the
        # concrete isinstance check rejected it while accepting np.float64.
        state, message = self.check(python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": np.int64(12),'))
        self.assertEqual(state, "ok", message)

    def test_non_finite_answer_rejected(self):
        # qti_export.js throws on this, and phys_fmt() formats it as "0".
        self.assert_invalid("'answer' must be finite", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": float("inf"),'))

    def test_nan_answer_rejected(self):
        self.assert_invalid("'answer' must be finite", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": float("nan"),'))

    def test_non_finite_tolerance_rejected(self):
        self.assert_invalid("'tolerance' must be finite",
                            python_code=GOOD_NUMERICAL_PYTHON.replace(
                                '"tolerance": resolve_tolerance(v, rel_tol=0.02),',
                                '"tolerance": float("inf"),'))

    def test_bytes_answer_rejected(self):
        # float(b"12.3") == 12.3, so a coercion-based check would let this through;
        # phys_fmt() then dies with "must be real number, not bytes".
        self.assert_invalid("'answer' must be a number", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": b"12.3",'))

    def test_missing_answer_rejected(self):
        self.assert_invalid("missing the 'answer' key", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"answer": v,', ''))

    def test_non_string_unit_rejected(self):
        self.assert_invalid("'unit' must be a string", python_code=GOOD_NUMERICAL_PYTHON.replace(
            '"unit": "m/s",', '"unit": 3,'))

    def test_unknown_type_named_clearly(self):
        # A typo in 'type' used to fall through to the multiple-choice path and
        # report "missing 'choices'", which says nothing about the real mistake.
        self.assert_invalid("'type' must be 'multiple_choice' or 'numerical'",
                            python_code=GOOD_NUMERICAL_PYTHON.replace(
                                '"type": "numerical",', '"type": "numeric",'))

    def test_multiple_choice_still_validated_without_a_type(self):
        # Every question that predates numerical entry omits 'type'.
        self.assertEqual(
            qvalidate.validate(GOOD_TEMPLATE, GOOD_PYTHON, expected_name="q_good"), ("ok", ""))
        state, _ = qvalidate.validate(
            GOOD_TEMPLATE, GOOD_PYTHON.replace('"answer": "a"', '"answer": "f"'),
            expected_name="q_good")
        self.assertEqual(state, "invalid")


# A valid sized <svg>, and the same picture with the root's width/height dropped —
# used as the base for TestSvgContract's mutations.
GOOD_SVG = '<svg viewBox="0 0 10 10" width="100" height="50" xmlns="http://www.w3.org/2000/svg"></svg>'


def with_svg(python_code, svg_literal):
    """Insert `"svg": <svg_literal>,` into GOOD_PYTHON/GOOD_NUMERICAL_PYTHON's return dict."""
    return python_code.replace('"difficulty": 2,', f'"difficulty": 2,\n        "svg": {svg_literal},')


class TestSvgContract(unittest.TestCase):
    """The optional 'svg' key — see DEFAULT_SYSTEM_PROMPT's "Optional SVG diagram"
    section in server/main.py. A <svg> with only a viewBox and no explicit
    width/height renders at zero size with no exception, so this is the one
    field-shape check worth enforcing here rather than leaving it to a blank
    preview."""

    def check(self, template=GOOD_TEMPLATE, python_code=GOOD_PYTHON, expected_name="q_good"):
        return qvalidate.validate(template, python_code, expected_name=expected_name)

    def test_sized_svg_is_ok(self):
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, repr(GOOD_SVG)))
        self.assertEqual(state, "ok", message)

    def test_sized_svg_is_ok_on_a_numerical_question(self):
        state, message = self.check(python_code=with_svg(GOOD_NUMERICAL_PYTHON, repr(GOOD_SVG)))
        self.assertEqual(state, "ok", message)

    def test_missing_svg_key_is_ok(self):
        # The overwhelming majority of questions have no diagram at all.
        self.assertEqual(self.check(), ("ok", ""))

    def test_empty_string_svg_is_ok(self):
        # A generator may only draw a diagram for some random branches and
        # return "" for the rest — that is "no diagram", not a validation error.
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, '""'))
        self.assertEqual(state, "ok", message)

    def test_non_string_svg_rejected(self):
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, "42"))
        self.assertEqual(state, "invalid")
        self.assertIn("'svg' must be a string", message)

    def test_non_svg_string_rejected(self):
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, '"just some text"'))
        self.assertEqual(state, "invalid")
        self.assertIn("complete '<svg", message)

    def test_missing_width_rejected(self):
        no_width = '<svg viewBox="0 0 10 10" height="50"></svg>'
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, repr(no_width)))
        self.assertEqual(state, "invalid")
        self.assertIn("width", message)
        self.assertIn("zero size", message)

    def test_missing_height_rejected(self):
        no_height = '<svg viewBox="0 0 10 10" width="100"></svg>'
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, repr(no_height)))
        self.assertEqual(state, "invalid")
        self.assertIn("height", message)

    def test_stroke_width_on_a_child_does_not_satisfy_the_root_width_check(self):
        # Regression guard: a naive "'width=' in svg" substring check would be
        # fooled by stroke-width="2" on a nested element and never catch a
        # root <svg> that has no width attribute of its own.
        no_root_width = '<svg viewBox="0 0 10 10" height="50"><line stroke-width="2"/></svg>'
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, repr(no_root_width)))
        self.assertEqual(state, "invalid")
        self.assertIn("width", message)

    def test_incomplete_svg_element_rejected(self):
        # Missing the closing tag entirely.
        state, message = self.check(
            python_code=with_svg(GOOD_PYTHON, '\'<svg width="100" height="50">\''))
        self.assertEqual(state, "invalid")
        self.assertIn("complete '<svg", message)

    def test_self_closing_svg_root_is_a_complete_element(self):
        # A self-closing root (<svg .../>) has no separate closing tag by
        # construction, but it IS a complete, renderable element — it must not
        # be rejected as "incomplete" just because "</svg>" never appears.
        self_closing = '<svg viewBox="0 0 10 10" width="100" height="50"/>'
        state, message = self.check(python_code=with_svg(GOOD_PYTHON, repr(self_closing)))
        self.assertEqual(state, "ok", message)


def run_worker(template, python_code, expected_name, env_extra=None, timeout=30):
    """Drive qvalidate.py the way main.py does: a subprocess with a scrubbed env."""
    env = dict(qvalidate.SANDBOX_ENV)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-I", "-B", WORKER],
        input=json.dumps({"template": template, "python_code": python_code,
                          "expected_name": expected_name}),
        capture_output=True, text=True, timeout=timeout, env=env, check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return "unavailable", f"exit {proc.returncode}: {proc.stderr.strip()[-200:]}"
    return result["state"], result.get("message", "")


class TestWorker(unittest.TestCase):
    def test_worker_round_trip(self):
        self.assertEqual(run_worker(GOOD_TEMPLATE, GOOD_PYTHON, "q_good"), ("ok", ""))

    def test_runaway_terminates(self):
        """Regression guard for the ThreadPoolExecutor hang.

        The old validator raised TimeoutError and then blocked forever in
        ThreadPoolExecutor.__exit__(), which calls shutdown(wait=True). Because it ran
        synchronously inside the SSE generator, one `while True:` from the model froze
        the uvicorn event loop and took the whole server down. Here the child's own
        RLIMIT_CPU stops it well before the parent's timeout.
        """
        started = time.monotonic()
        state, _ = run_worker(GOOD_TEMPLATE, "while True:\n    pass\n", "q_good", timeout=30)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 20, "runaway question was not stopped")
        self.assertNotEqual(state, "ok")

    def test_runaway_reaches_main_as_invalid(self):
        """End to end through main.py: a runaway is the model's fault, not the sandbox's.

        Misclassifying it as "unavailable" would silently skip the fix loop.
        """
        main = import_main()
        if main is None:
            self.skipTest("server dependencies not installed")
        started = time.monotonic()
        state, message = asyncio.run(
            main._validate_question(GOOD_TEMPLATE, "while True:\n    pass\n", "q_good"))
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(state, "invalid", message)

    def test_main_validates_the_bank(self):
        """The subprocess path main.py actually uses, not just the library."""
        main = import_main()
        if main is None:
            self.skipTest("server dependencies not installed")
        bank = load_bank()
        qid = bank["question_order"][0]
        q = bank["questions"][qid]
        state, message = asyncio.run(
            main._validate_question(q["template"], q["python_code"], qid))
        self.assertEqual(state, "ok", message)

    def test_worker_survives_a_bare_environment(self):
        """numpy must import even if the parent forgets the BLAS thread caps.

        Without them OpenBLAS reserves more address space than RLIMIT_AS allows and
        aborts before Python can catch anything, so every question would come back
        "unavailable" and validation would be silently off. The worker sets them itself.
        """
        proc = subprocess.run(
            [sys.executable, "-I", "-B", WORKER],
            input=json.dumps({"template": GOOD_TEMPLATE, "python_code": GOOD_PYTHON,
                              "expected_name": "q_good"}),
            capture_output=True, text=True, timeout=60, check=False,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-300:])
        self.assertEqual(json.loads(proc.stdout)["state"], "ok", proc.stdout)

    def test_missing_dependency_is_unavailable_not_invalid(self):
        """A broken sandbox must not be blamed on the model.

        If a Dockerfile COPY or a requirements bump is ever forgotten, every question
        would fail. Reporting that as "invalid" would spend MAX_FIX_ATTEMPTS Gemini
        calls per question asking the model to repair code that was never broken —
        precisely the failure this rewrite removes.
        """
        import importlib
        real = sys.modules.pop("questions", None)
        blocker = _BlockImport("questions")
        sys.meta_path.insert(0, blocker)
        try:
            importlib.invalidate_caches()
            state, message = qvalidate.validate(GOOD_TEMPLATE, GOOD_PYTHON, "q_good")
        finally:
            sys.meta_path.remove(blocker)
            if real is not None:
                sys.modules["questions"] = real
            importlib.invalidate_caches()
        self.assertEqual(state, "unavailable", message)
        self.assertIn("questions", message)


class _BlockImport:
    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ImportError("blocked for test", name=fullname)


class TestRuntimeParity(unittest.TestCase):
    def test_jinja_settings_match_the_browser(self):
        """The server must compile templates exactly as index.html's preview does.

        A mismatch means whitespace-sensitive templates validate differently than
        they render, which would make the validator lie in both directions.
        """
        html = INDEX_HTML.read_text()
        block = re.search(r"_tpl = _j2\.Environment\((.*?)\)\.from_string", html, re.DOTALL)
        self.assertIsNotNone(block, "preview Environment(...) block not found in index.html")
        kwargs = block.group(1)
        for key, value in qvalidate._ENV_KW.items():
            self.assertIn(f"{key}={value}", kwargs.replace(" ", "").replace("\n", ""))
        self.assertIn("StrictUndefined", kwargs)

    def test_questions_helpers_match_the_canonical_copy(self):
        self.assertEqual(
            (pathlib.Path(__file__).resolve().parent / "questions.py").read_text(),
            (REPO / "python" / "__init__.py").read_text(),
            "server/questions.py must stay a byte-exact copy of python/__init__.py",
        )


# ---------------------------------------------------------------------------
# Chat turn contract
# ---------------------------------------------------------------------------
# A turn in which the model only asked for data the server had already inlined
# used to produce no SSE events at all: every call was suppressed, no text was
# streamed, and the browser rendered the turn as "(Empty response from AI)".
# Three consecutive user messages were lost that way on question-forge-server
# before these tests existed.


def _drain(agen):
    """Collect an SSE async generator's decoded payloads."""
    async def _run():
        return [json.loads(ev["data"]) async for ev in agen]
    return asyncio.run(_run())


def _fake_round(script):
    """Build a stand-in for main._gemini_round that replays `script`.

    Each entry is (text, function_calls, finish); the call count is recorded on
    the returned function so a test can assert whether a continuation ran.
    """
    calls = []
    seen_tools = []

    async def _round(model, system_text, contents, tools, out):
        calls.append(contents)
        seen_tools.append(tools)
        text, function_calls, finish = script[min(len(calls) - 1, len(script) - 1)]
        out["finish"] = finish
        out["function_calls"] = list(function_calls)
        if text:
            out["n_text"] += 1
            yield {"data": json.dumps({"type": "text", "delta": text})}

    _round.calls = calls
    _round.tools = seen_tools
    return _round


class TestChatTurnContract(unittest.TestCase):
    def setUp(self):
        self.main = import_main()
        if self.main is None:
            self.skipTest("server dependencies not installed")
        self._real_round = self.main._gemini_round
        self.addCleanup(setattr, self.main, "_gemini_round", self._real_round)

    def _req(self, **kw):
        return self.main.ChatRequest(**{
            "messages": [
                self.main.ChatMessage(role="user", content="Create 30 numerical entry questions."),
            ],
            "textbook_catalog": "=== TEXTBOOK CATALOG ===\n  Ch 5. Electric Charges",
            **kw,
        })

    # -- _emit_tool_calls reporting ----------------------------------------

    def test_a_suppressed_call_is_reported_and_emits_nothing(self):
        req = self._req(textbook_context="=== TEXTBOOK EXCERPTS ===\nstuff")
        out = {}
        events = _drain(self.main._emit_tool_calls(
            [{"name": "search_textbook", "args": {"chapters": [5]}}], req, {}, out))
        self.assertEqual(events, [])
        self.assertEqual(out["emitted"], 0)
        self.assertEqual([fc["name"] for fc in out["suppressed"]], ["search_textbook"])

    def test_a_pending_search_is_forwarded_with_every_selector(self):
        """chapters-only is a valid lookup now, so it must survive the hop."""
        out = {}
        events = _drain(self.main._emit_tool_calls(
            [{"name": "search_textbook", "args": {"chapters": [5], "max_sections": 6}}],
            self._req(), {}, out))
        self.assertEqual(events[0]["chapters"], [5])
        self.assertEqual(events[0]["max_sections"], 6)
        self.assertEqual(out["emitted"], 1)

    def test_a_real_call_counts_as_emitted(self):
        out = {}
        events = _drain(self.main._emit_tool_calls(
            [{"name": "create_question", "args": {"question_id": "q_x", "title": "X"}}],
            self._req(), {}, out))
        self.assertEqual(len(events), 1)
        self.assertEqual(out["emitted"], 1)
        self.assertEqual(out["suppressed"], [])

    # -- grounding rules ----------------------------------------------------

    def test_excerpts_in_the_prompt_forbid_a_second_search(self):
        prompt = self.main._system_prompt(self._req(
            textbook_context="=== TEXTBOOK EXCERPTS ===\nstuff",
            textbook_query="chapter 5",
        ))
        self.assertIn("do NOT call search_textbook again", prompt)
        self.assertIn("chapter 5", prompt)
        self.assertNotIn("Call search_textbook BEFORE", prompt)

    def test_without_excerpts_the_model_is_told_to_search(self):
        prompt = self.main._system_prompt(self._req())
        self.assertIn("Call search_textbook BEFORE", prompt)

    # -- the turn itself ----------------------------------------------------

    def test_a_suppressed_only_turn_is_continued_rather_than_ending_empty(self):
        req = self._req(textbook_context="=== TEXTBOOK EXCERPTS ===\nstuff")
        suppressed = [{"name": "search_textbook", "args": {"chapters": [5]}}]
        round_ = _fake_round([("", suppressed, "STOP"), ("Grounded in 5.1.", [], "STOP")])
        self.main._gemini_round = round_
        events = _drain(self.main._stream_gemini("gemini-2.5-flash", req))
        self.assertEqual(len(round_.calls), 2, "the suppressed call was not answered in-band")
        # The refactor into rounds must not lose the tool exclusion: offering
        # search_textbook again is what started the loop.
        for tools in round_.tools:
            names = [fd["name"] for group in tools for fd in group["functionDeclarations"]]
            self.assertNotIn("search_textbook", names)
            self.assertIn("create_question", names)
        self.assertEqual([e["type"] for e in events], ["text", "done"])
        # The continuation must carry the tool result, or the model just repeats itself.
        self.assertIn("functionResponse", json.dumps(round_.calls[1]))

    def test_a_turn_that_stays_empty_explains_itself(self):
        req = self._req(textbook_context="=== TEXTBOOK EXCERPTS ===\nstuff")
        suppressed = [{"name": "search_textbook", "args": {}}]
        round_ = _fake_round([("", suppressed, "STOP")])
        self.main._gemini_round = round_
        events = _drain(self.main._stream_gemini("gemini-2.5-flash", req))
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("done", [e["type"] for e in events])

    def test_max_tokens_with_no_output_is_reported_not_swallowed(self):
        round_ = _fake_round([("", [], "MAX_TOKENS")])
        self.main._gemini_round = round_
        events = _drain(self.main._stream_gemini("gemini-2.5-flash", self._req()))
        self.assertEqual(len(round_.calls), 1, "nothing was suppressed, so nothing to continue")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("output limit", events[-1]["message"])

    def test_max_tokens_usage_is_surfaced_in_the_error(self):
        """The reasoning/reply token split must reach the user, not just the
        server log -- that split is exactly what turns "the model broke" into
        a diagnosable "reasoning ate the budget, ask for fewer questions"."""
        async def round_(model, system_text, contents, tools, out):
            out["finish"] = "MAX_TOKENS"
            out["usage"] = {
                "promptTokenCount": 1500,
                "thoughtsTokenCount": 8192,
                "candidatesTokenCount": 0,
                "totalTokenCount": 9692,
            }
            return
            yield  # unreachable; presence of `yield` makes this an async generator
        self.main._gemini_round = round_
        events = _drain(self.main._stream_gemini("gemini-2.5-flash", self._req()))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("8192", events[-1]["message"])
        self.assertIn("1500", events[-1]["message"])

    def test_empty_turn_message_without_usage_still_explains_itself(self):
        """Some Gemini previews omit usageMetadata even with thinking on --
        the message must degrade gracefully, not KeyError or print "None"."""
        msg = self.main._empty_turn_message("MAX_TOKENS", {})
        self.assertIn("output limit", msg)
        self.assertNotIn("None", msg)

    def test_generation_config_caps_the_thinking_budget(self):
        """Gemini spends reasoning tokens out of the same maxOutputTokens
        ceiling as the reply, so the thinking budget must sit strictly below
        the output ceiling -- otherwise reasoning alone can hit MAX_TOKENS
        with nothing left over for the reply (or the tool call) it was
        supposed to produce."""
        config = self.main._gemini_generation_config(0.7)
        self.assertLess(
            config["thinkingConfig"]["thinkingBudget"],
            config["maxOutputTokens"],
            "the thinking budget must leave headroom for the actual reply",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
