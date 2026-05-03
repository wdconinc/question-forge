# Question modules for PHYS 1020 Final Exam
# Each module exports: generate(rng: numpy.random.Generator) -> dict
# Return dict keys:
#   question  : str   — full question text (Markdown)
#   choices   : list[str] of exactly 5 items — answer choices (a)–(e)
#   answer    : str   — correct choice letter, one of 'a','b','c','d','e'
#   topic     : str   — brief topic label (e.g. "Ch.7 Work-Energy Theorem")

import math
import os
import jinja2

_QUESTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_QUESTIONS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)


def render_template(name: str, params: dict) -> str:
    """Render the Jinja2 template *name*.j2 from the questions/ directory.

    Parameters
    ----------
    name : str
        Template base name without extension, e.g. ``"q06_newton2"``.
    params : dict
        Variables passed to the template context.

    Returns
    -------
    str
        Rendered question text, stripped of leading/trailing whitespace.
    """
    tpl = _jinja_env.get_template(f"{name}.j2")
    return tpl.render(**params).strip()


def phys_fmt(v: float, sig: int = 3) -> str:
    """Format a number with *sig* significant figures for a printed exam.

    Avoids Python's default 'e+03' notation:
    - Values in [0.001, 10000): formatted directly (e.g. "1740", "29.4", "0.085")
    - Values >= 10000 or < 0.001: formatted as LaTeX "$M.mm \\times 10^{n}$"
    """
    if not math.isfinite(v) or v == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(v))))
    if exp >= 4 or exp <= -4:
        mantissa = v / 10 ** exp
        return f"${mantissa:.{sig - 1}f} \\times 10^{{{exp}}}$"
    else:
        decimal_places = max(0, sig - 1 - exp)
        return f"{v:.{decimal_places}f}"


def make_choices(correct_val: float, distractors: list, fmt, min_spacing: float = 0.12) -> list:
    """
    Build [correct, d1, d2, d3, d4] with guaranteed uniqueness and minimum
    relative spacing between any pair of choices.

    correct_val : float — the numerically correct answer
    distractors : list of 4 floats — proposed distractor values
    fmt         : callable(float) -> str — formatter
    min_spacing : minimum required relative difference |a-b|/max(|a|,|b|)
                  between any two choice values (default 12 %).

    Returns list of 5 unique strings; correct is always at index 0.
    If a distractor is too close to the correct value or any already-placed
    choice, it is replaced by correct_val * fallback_multiplier.
    """
    def _too_close(v, existing_vals):
        """Return True if v is within min_spacing of any value in existing_vals."""
        if v == 0:
            return any(ev == 0 for ev in existing_vals)
        for ev in existing_vals:
            if ev == 0:
                continue
            # Opposite signs are always distinguishable
            if (v < 0) != (ev < 0):
                continue
            if abs(v - ev) / max(abs(v), abs(ev)) < min_spacing:
                return True
        return False

    correct_str = fmt(correct_val)
    used_strs = {correct_str}
    used_vals = [correct_val]
    result = [correct_str]

    fallbacks = [0.5, 0.25, 1.5, 2.0, 0.75, 3.0, 0.1, 4.0, 1.25, 0.333,
                 5.0, 0.2, 6.0, 0.4, 7.0, 8.0, 0.6, 9.0, 10.0, 0.05,
                 2.3, 2.7, 3.5, 4.5, 11.0, 12.0, 0.3, 0.15, 0.08]
    fb_idx = 0

    for d in distractors:
        s = fmt(d)
        if s not in used_strs and not _too_close(d, used_vals):
            used_strs.add(s)
            used_vals.append(d)
            result.append(s)
        else:
            placed = False
            while fb_idx < len(fallbacks):
                candidate = correct_val * fallbacks[fb_idx]
                fb_idx += 1
                if candidate == 0:
                    continue
                cs = fmt(candidate)
                if cs not in used_strs and not _too_close(candidate, used_vals):
                    used_strs.add(cs)
                    used_vals.append(candidate)
                    result.append(cs)
                    placed = True
                    break
            if not placed:
                for mult in [1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8]:
                    cs = fmt(correct_val * mult)
                    candidate = correct_val * mult
                    if cs not in used_strs and not _too_close(candidate, used_vals):
                        used_strs.add(cs)
                        used_vals.append(candidate)
                        result.append(cs)
                        placed = True
                        break

    return result
