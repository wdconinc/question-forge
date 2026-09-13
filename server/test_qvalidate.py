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


NUMERICAL_TEMPLATE = "A sled starts at {{ v0 }} m/s and accelerates at {{ a }} m/s^2.\n"
NUMERICAL_PYTHON = '''
import numpy as np
from questions import resolve_tolerance, render_template

def generate(rng: np.random.Generator) -> dict:
    v0 = float(rng.choice([0.0, 2.0, 5.0]))
    a = float(rng.choice([1.0, 2.0, 3.0]))
    v = v0 + a * 4.0
    params = {"v0": f"{v0:.1f}", "a": f"{a:.1f}"}
    return {
        "type": "numerical",
        "question": render_template("q_num", params),
        "answer": v,
        "tolerance": resolve_tolerance(v, rel_tol=0.02),
        "unit": "m/s",
        "topic": "Kinematics",
        "difficulty": 1,
    }
'''


class TestNumericalContract(unittest.TestCase):
    """Numerical entry is a second contract, selected by type == "numerical".

    The shape is the one DEFAULT_SYSTEM_PROMPT documents to the model and
    src/qti_export.js:149 documents to its caller.
    """

    def check(self, template=NUMERICAL_TEMPLATE, python_code=NUMERICAL_PYTHON,
              expected_name="q_num", **kw):
        return qvalidate.validate(template, python_code, expected_name=expected_name, **kw)

    def assert_invalid(self, needle, **kw):
        state, message = self.check(**kw)
        self.assertEqual(state, "invalid", f"got {state}: {message}")
        self.assertIn(needle, message)
        return message

    def test_baseline_is_ok(self):
        self.assertEqual(self.check(), ("ok", ""))

    def test_numerical_needs_no_choices(self):
        # The whole point: the multiple-choice contract must not be applied here.
        state, message = self.check()
        self.assertEqual(state, "ok", message)
        self.assertNotIn("choices", message)

    def test_missing_answer_rejected(self):
        self.assert_invalid("missing the 'answer' key", python_code=NUMERICAL_PYTHON.replace(
            '"answer": v,', ''))

    def test_missing_tolerance_rejected(self):
        self.assert_invalid("missing the 'tolerance' key", python_code=NUMERICAL_PYTHON.replace(
            '"tolerance": resolve_tolerance(v, rel_tol=0.02),', ''))

    def test_non_numeric_answer_rejected(self):
        self.assert_invalid("'answer' must be a number", python_code=NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": "twelve",'))

    def test_non_finite_answer_rejected(self):
        # qti_export.js rejects this outright; catch it before the user accepts.
        self.assert_invalid("'answer' must be finite", python_code=NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": float("inf"),'))

    def test_negative_tolerance_rejected(self):
        self.assert_invalid("'tolerance' must be non-negative",
                            python_code=NUMERICAL_PYTHON.replace(
                                '"tolerance": resolve_tolerance(v, rel_tol=0.02),',
                                '"tolerance": -1.0,'))

    def test_non_string_unit_rejected(self):
        self.assert_invalid("'unit' must be a string", python_code=NUMERICAL_PYTHON.replace(
            '"unit": "m/s",', '"unit": 3,'))

    def test_bad_sig_figs_rejected(self):
        self.assert_invalid("'sig_figs' must be a positive integer",
                            python_code=NUMERICAL_PYTHON.replace(
                                '"unit": "m/s",', '"unit": "m/s", "sig_figs": 0,'))

    def test_bytes_answer_rejected(self):
        # float(b"12.3") == 12.3, so a coercion-based check let this through;
        # phys_fmt() then dies at render time with "must be real number, not bytes".
        self.assert_invalid("'answer' must be a number", python_code=NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": b"12.3",'))

    def test_bytearray_tolerance_rejected(self):
        self.assert_invalid("'tolerance' must be a number",
                            python_code=NUMERICAL_PYTHON.replace(
                                '"tolerance": resolve_tolerance(v, rel_tol=0.02),',
                                '"tolerance": bytearray(b"0.5"),'))

    def test_numpy_scalars_accepted(self):
        # Generators compute with numpy; np.int64 is not an int subclass, and
        # rejecting it would be a false failure.
        state, message = self.check(python_code=NUMERICAL_PYTHON.replace(
            '"answer": v,', '"answer": np.int64(12),').replace(
            '"tolerance": resolve_tolerance(v, rel_tol=0.02),', '"tolerance": np.float64(0.5),'))
        self.assertEqual(state, "ok", message)

    def test_unknown_type_named_clearly(self):
        # Without this the browser's "anything not numerical is MC" default turns
        # a typo into a baffling "missing 'choices'".
        self.assert_invalid("'type' must be 'multiple_choice' or 'numerical'",
                            python_code=NUMERICAL_PYTHON.replace(
                                '"type": "numerical",', '"type": "numeric",'))

    def test_multiple_choice_still_validated_without_a_type(self):
        # Every pre-existing bank question omits 'type' and must stay on the MC path.
        state, message = qvalidate.validate(GOOD_TEMPLATE, GOOD_PYTHON, expected_name="q_good")
        self.assertEqual(state, "ok", message)
        state, message = qvalidate.validate(
            GOOD_TEMPLATE,
            GOOD_PYTHON.replace('"answer": "a"', '"answer": "f"'),
            expected_name="q_good")
        self.assertEqual(state, "invalid", message)



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


if __name__ == "__main__":
    unittest.main(verbosity=2)
