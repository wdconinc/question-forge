# exam_core.py — runs inside Pyodide
# Browser-compatible exam assembly logic (no argparse, no file I/O)
import sys
import json
import numpy as np

LETTERS = ["a", "b", "c", "d", "e"]


def local_shuffle_order(n, rng):
    """Shuffle with max displacement of 2 (adjacent-swap, two rounds)."""
    order = list(range(n))
    for i in range(0, n - 1, 2):
        if rng.random() < 0.5:
            order[i], order[i+1] = order[i+1], order[i]
    for i in range(1, n - 1, 2):
        if rng.random() < 0.5:
            order[i], order[i+1] = order[i+1], order[i]
    return order


def assign_answer_positions(n, rng):
    assert n % 5 == 0
    pool = np.array(LETTERS * (n // 5))
    rng.shuffle(pool)
    return pool.tolist()


def apply_answer_positions(data_list, positions, rng):
    for datum, target in zip(data_list, positions):
        choices = list(datum["choices"])
        curr_idx = LETTERS.index(datum["answer"])
        tgt_idx  = LETTERS.index(target)
        correct  = choices[curr_idx]
        distractors = [choices[i] for i in range(5) if i != curr_idx]
        rng.shuffle(distractors)
        result = [None] * 5
        result[tgt_idx] = correct
        d_iter = iter(distractors)
        for i in range(5):
            if result[i] is None:
                result[i] = next(d_iter)
        datum["choices"] = result
        datum["answer"]  = target


def render_question_block(q_num, data):
    lines = [f"**{q_num}.** {data['question']}", ""]
    for letter, choice in zip(LETTERS, data["choices"]):
        lines.append(f"({letter}) {choice}")
    lines.append("")
    return "\n".join(lines), data["answer"]


def build_header(meta):
    paper = meta.get("paper", "A")
    course = meta.get("course", "PHYS 1020")
    exam_type = meta.get("examType", "FINAL EXAMINATION")
    date = meta.get("date", "")
    time_slot = meta.get("timeSlot", "")
    duration = meta.get("duration", "3 hours")
    examiners = meta.get("examiners", "")
    num_q = meta.get("numQuestions", 30)
    return f"""---
**UNIVERSITY OF MANITOBA**
**{exam_type.upper()}**

**{date}**
**(Paper {paper})**

**DEPARTMENT & COURSE NO.:** {course}  
**TIME:** {duration}  
**EXAMINERS:** {examiners}

---

All questions are of equal value. This is **Paper {paper}**. Questions are numbered 1 to {num_q}.

---

**TABLE OF CONSTANTS**

| Constant | Value |
|---|---|
| $G$ | $6.674 \\times 10^{{-11}}\\ \\text{{m}}^3\\,\\text{{kg}}^{{-1}}\\,\\text{{s}}^{{-2}}$ |
| $g$ | $9.80\\ \\text{{m/s}}^2$ |
| Mass of the Earth | $5.97 \\times 10^{{24}}\\ \\text{{kg}}$ |
| Radius of the Earth | $6.38 \\times 10^{{3}}\\ \\text{{km}}$ |
| Speed of sound in air (20 °C) | $343\\ \\text{{m/s}}$ |
| Speed of light | $3.00 \\times 10^{{8}}\\ \\text{{m/s}}$ |

---
"""


def render_paper(question_bank, question_order, meta, seed):
    """Generate one exam paper. Returns (markdown_str, answers_list, ordered_data)."""
    rng = np.random.default_rng(seed)
    data_orig = []
    import jinja2 as _j2
    import questions as _qmod

    for qid in question_order:
        q = question_bank[qid]
        # Monkey-patch render_template BEFORE exec so from-import captures the patched version
        _tpl = _j2.Environment(
            trim_blocks=True, lstrip_blocks=True,
            keep_trailing_newline=False,
            undefined=_j2.StrictUndefined,
        ).from_string(q["template"])
        def _rt(name, params, _t=_tpl):
            return _t.render(**params).strip()
        _orig_rt = _qmod.render_template
        _qmod.render_template = _rt
        try:
            ns = {}
            exec(compile(q["python_code"], qid + ".py", "exec"), ns)
            q_rng = np.random.default_rng(rng.integers(0, 2**63))
            data_orig.append(ns["generate"](q_rng))
        finally:
            _qmod.render_template = _orig_rt

    order = local_shuffle_order(len(question_order), rng)
    data = [data_orig[i] for i in order]
    pos_rng = np.random.default_rng([seed, 0xF1A1A])
    apply_answer_positions(data, assign_answer_positions(len(data), pos_rng), pos_rng)
    lines = [build_header(meta)]
    answers = []
    for i, d in enumerate(data, 1):
        block, ans = render_question_block(i, d)
        lines.append(block)
        answers.append(ans)
    return "\n".join(lines), answers
