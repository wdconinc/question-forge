#!/usr/bin/env python3
"""
Build src/default_bank.js from the phys1020-final-exam-preparation question bank.
Run from the question-forge/ repo root:
    python scripts/build_default_bank.py \
        --source ../phys1020-final-exam-preparation/questions \
        --output src/default_bank.js

Or specify a different source directory.
"""
import json, pathlib, argparse

QUESTION_ORDER = [
    "q01_units", "q02_kinematics_1d", "q03_free_fall", "q04_vectors",
    "q05_projectile", "q06_newton2", "q07_friction", "q08_atwood",
    "q09_circular", "q10_banked_curve", "q31_synthesis_proj_energy",
    "q12_work_energy", "q13_energy_conservation", "q32_synthesis_incline_energy",
    "q15_momentum", "q16_collision", "q17_rot_kinematics", "q18_torque",
    "q19_rot_dynamics", "q20_angular_momentum", "q21_shm_period", "q22_shm_energy",
    "q23_pressure", "q24_buoyancy", "q33_synthesis_loop", "q26_bernoulli",
    "q27_temperature", "q28_calorimetry", "q29_heat_transfer", "q30_thermo_laws",
]

# Override j2 template stem when it differs from the question stem
J2_OVERRIDES = {
    "q30_thermo_laws": "q30_first_law",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    src = pathlib.Path(args.source)
    bank = {"question_order": [], "questions": {}}
    for stem in QUESTION_ORDER:
        py_path = src / f"{stem}.py"
        j2_stem = J2_OVERRIDES.get(stem, stem)
        j2_path = src / f"{j2_stem}.j2"
        if not py_path.exists() or not j2_path.exists():
            print(f"  WARNING: {stem} missing .py or .j2, skipping")
            continue
        bank["question_order"].append(stem)
        bank["questions"][stem] = {
            "id": stem,
            "title": stem,
            "topic": "",
            "difficulty": 2,
            "python_code": py_path.read_text(),
            "template": j2_path.read_text(),
        }
    js = f"export const DEFAULT_BANK = {json.dumps(bank, indent=2)};\n"
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(js)
    print(f"Wrote {len(bank['question_order'])} questions to {args.output}")

if __name__ == "__main__":
    main()
