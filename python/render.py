#!/usr/bin/env python3
"""
render.py — PHYS 1020 Final Exam renderer
==========================================
Assembles the 30-question final exam from individual question modules
and renders two randomized versions (A and B) as Markdown files.

Usage
-----
    python render.py [--seed-A INT] [--seed-B INT] [--out-dir PATH]

Outputs
-------
    exam_A.md       — Paper A (version 1)
    exam_B.md       — Paper B (version 2, different parameters + shuffled choices)
    answer_key.md   — Answer key for both papers
"""

import argparse
import importlib
import textwrap
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXAM_META = {
    "exam_type": "FINAL EXAMINATION",
    "date": "April 20, 2026",
    "time_slot": "9:00 AM – 11:00 AM",
    "duration": "3 hours",
    "num_questions": 30,
}

# Ordered list of question module names (must live in questions/)
QUESTION_MODULES = [
    "q01_units",
    "q02_kinematics_1d",
    "q03_free_fall",
    "q04_vectors",
    "q05_projectile",
    "q06_newton2",
    "q07_friction",
    "q08_atwood",
    "q09_circular",
    "q10_banked_curve",
    "q31_synthesis_proj_energy",
    "q12_work_energy",
    "q13_energy_conservation",
    "q32_synthesis_incline_energy",
    "q15_momentum",
    "q16_collision",
    "q17_rot_kinematics",
    "q18_torque",
    "q19_rot_dynamics",
    "q20_angular_momentum",
    "q21_shm_period",
    "q22_shm_energy",
    "q23_pressure",
    "q24_buoyancy",
    "q33_synthesis_loop",
    "q26_bernoulli",
    "q27_temperature",
    "q28_calorimetry",
    "q29_heat_transfer",
    "q30_thermo_laws",
]

LETTERS = ["a", "b", "c", "d", "e"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def local_shuffle_order(n: int, rng: np.random.Generator) -> list:
    """Return a permutation of range(n) where every element stays within
    ±2 positions of its original index.

    Uses two rounds of random adjacent swaps:
      Round 1: even-indexed pairs (0,1), (2,3), (4,5), …
      Round 2: odd-indexed pairs  (1,2), (3,4), (5,6), …

    After both rounds the maximum displacement is exactly 2.
    """
    order = list(range(n))
    for i in range(0, n - 1, 2):          # even pairs
        if rng.random() < 0.5:
            order[i], order[i + 1] = order[i + 1], order[i]
    for i in range(1, n - 1, 2):          # odd pairs
        if rng.random() < 0.5:
            order[i], order[i + 1] = order[i + 1], order[i]
    return order


def load_question(module_name: str):
    """Import a question module from the questions/ package."""
    return importlib.import_module(f"questions.{module_name}")


def assign_answer_positions(n: int, rng: np.random.Generator) -> list:
    """Return a shuffled list of n answer letters where each of a/b/c/d/e appears
    exactly n//5 times (n must be divisible by 5).  Uses *rng* so the assignment
    is deterministic given the paper seed."""
    assert n % 5 == 0, f"n={n} must be divisible by 5"
    pool = np.array(LETTERS * (n // 5))
    rng.shuffle(pool)
    return pool.tolist()


def apply_answer_positions(data_list: list, positions: list, rng: np.random.Generator) -> None:
    """Permute each datum's choices in-place so the correct answer lands at the
    letter given by *positions[i]*; the four distractors are randomly reordered."""
    for datum, target in zip(data_list, positions):
        choices = list(datum["choices"])
        curr_idx = LETTERS.index(datum["answer"])
        tgt_idx = LETTERS.index(target)

        correct_choice = choices[curr_idx]
        distractors = [choices[i] for i in range(5) if i != curr_idx]
        rng.shuffle(distractors)

        result = [None] * 5
        result[tgt_idx] = correct_choice
        d_iter = iter(distractors)
        for i in range(5):
            if i != tgt_idx:
                result[i] = next(d_iter)

        datum["choices"] = result
        datum["answer"] = target


def render_question(q_num: int, data: dict) -> tuple[str, str]:
    """Render a single question as a Markdown block.

    Returns (markdown_block, correct_letter).  Choices must already be in their
    final order (call apply_answer_positions beforehand).
    """
    lines = [f"**{q_num}.** {data['question']}", ""]
    for letter, choice in zip(LETTERS, data["choices"]):
        lines.append(f"({letter}) {choice}")
    lines.append("")
    return "\n".join(lines), data["answer"]


def render_exam(paper: str, questions_data: list[dict]) -> tuple[str, list[str]]:
    """Render the full exam Markdown for one paper.

    Returns (exam_markdown, list_of_correct_letters).
    """
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        keep_trailing_newline=True,
    )
    header_tmpl = env.get_template("exam_header.md.j2")
    header = header_tmpl.render(paper=paper, **EXAM_META)

    blocks = [header]
    answers = []

    for i, data in enumerate(questions_data, start=1):
        block, correct = render_question(i, data)
        blocks.append(block)
        answers.append(correct)

    return "\n".join(blocks), answers


def render_answer_key(
    answers_A: list, order_A: list,
    answers_B: list, order_B: list,
    answers_C: list, order_C: list,
    topics: list,
    difficulties: list,
) -> str:
    STARS = {1: "★", 2: "★★", 3: "★★★", 4: "★★★★"}
    lines = [
        "# Answer Key — PHYS 1020 Final Exam",
        "",
        "| Pos | Paper A answer | Paper A orig Q | Paper B answer | Paper B orig Q | Paper C answer | Paper C orig Q | Difficulty | Topic |",
        "|-----|----------------|----------------|----------------|----------------|----------------|----------------|------------|-------|",
    ]
    for pos in range(len(answers_A)):
        qa, qb, qc = order_A[pos] + 1, order_B[pos] + 1, order_C[pos] + 1
        topic = topics[pos]
        stars = STARS[difficulties[pos]]
        lines.append(
            f"| {pos+1:2d}  | {answers_A[pos]}              | Q{qa:02d}           "
            f"| {answers_B[pos]}              | Q{qb:02d}           "
            f"| {answers_C[pos]}              | Q{qc:02d}           "
            f"| {stars:10} | {topic} |"
        )

    # Distribution summary
    from collections import Counter
    dist_A = Counter(answers_A)
    dist_B = Counter(answers_B)
    dist_C = Counter(answers_C)
    lines += [
        "",
        "## Answer distribution",
        "",
        "| Letter | Paper A | Paper B | Paper C |",
        "|--------|---------|---------|---------|",
    ]
    for letter in LETTERS:
        lines.append(f"| {letter}      | {dist_A[letter]:7d} | {dist_B[letter]:7d} | {dist_C[letter]:7d} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Render PHYS 1020 final exam")
    parser.add_argument("--seed-A", type=int, default=42, help="RNG seed for Paper A")
    parser.add_argument("--seed-B", type=int, default=137, help="RNG seed for Paper B")
    parser.add_argument("--seed-C", type=int, default=271, help="RNG seed for Paper C (deferred)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(QUESTION_MODULES)} question modules...")
    modules = []
    for name in QUESTION_MODULES:
        try:
            modules.append(load_question(name))
        except ModuleNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {name}")

    print(f"  Loaded {len(modules)} / {len(QUESTION_MODULES)} modules")

    # Generate questions for Paper A
    print(f"Generating Paper A (seed={args.seed_A})...")
    rng_A = np.random.default_rng(args.seed_A)
    data_A_orig = [mod.generate(rng_A) for mod in modules]
    order_A = local_shuffle_order(len(modules), rng_A)
    data_A = [data_A_orig[i] for i in order_A]
    # Assign balanced answer positions (6× each letter), shuffle distractors
    pos_rng_A = np.random.default_rng([args.seed_A, 0xF1A1A])
    apply_answer_positions(data_A, assign_answer_positions(len(data_A), pos_rng_A), pos_rng_A)

    # Generate questions for Paper B (different parameters)
    print(f"Generating Paper B (seed={args.seed_B})...")
    rng_B = np.random.default_rng(args.seed_B)
    data_B_orig = [mod.generate(rng_B) for mod in modules]
    order_B = local_shuffle_order(len(modules), rng_B)
    data_B = [data_B_orig[i] for i in order_B]
    pos_rng_B = np.random.default_rng([args.seed_B, 0xF1A1A])
    apply_answer_positions(data_B, assign_answer_positions(len(data_B), pos_rng_B), pos_rng_B)

    # Generate questions for Paper C (deferred exam)
    print(f"Generating Paper C (seed={args.seed_C})...")
    rng_C = np.random.default_rng(args.seed_C)
    data_C_orig = [mod.generate(rng_C) for mod in modules]
    order_C = local_shuffle_order(len(modules), rng_C)
    data_C = [data_C_orig[i] for i in order_C]
    pos_rng_C = np.random.default_rng([args.seed_C, 0xF1A1A])
    apply_answer_positions(data_C, assign_answer_positions(len(data_C), pos_rng_C), pos_rng_C)

    # Render
    exam_A_md, answers_A = render_exam("A", data_A)
    exam_B_md, answers_B = render_exam("B", data_B)
    exam_C_md, answers_C = render_exam("C", data_C)
    topics_A = [data_A_orig[i]["topic"] for i in order_A]
    difficulties_A = [data_A_orig[i]["difficulty"] for i in order_A]
    key_md = render_answer_key(
        answers_A, order_A,
        answers_B, order_B,
        answers_C, order_C,
        topics_A, difficulties_A,
    )

    # Write output
    (out_dir / "exam_A.md").write_text(exam_A_md)
    (out_dir / "exam_B.md").write_text(exam_B_md)
    (out_dir / "exam_C.md").write_text(exam_C_md)
    (out_dir / "answer_key.md").write_text(key_md)

    print(f"Written: {out_dir}/exam_A.md, exam_B.md, exam_C.md, answer_key.md")
    print("\nTo convert to Word:")
    print("  pandoc exam_A.md -o exam_A.docx --reference-doc=template.docx --from=markdown-yaml_metadata_block")
    print("  pandoc exam_C.md -o exam_C.docx --reference-doc=template.docx --from=markdown-yaml_metadata_block")


if __name__ == "__main__":
    main()
