"""Count teacher-solvable student failures and their same-prefix recoverability."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def compute(raw_root=None, input_dir=None):
    """Summarize standalone failures and recovery from this run's unchanged prefixes."""
    raw = Path(raw_root) if raw_root else ROOT / "frozen"
    inputs = Path(input_dir) if input_dir else ROOT / "inputs/1.7B"
    cohort = read(inputs / "cohort.json")
    if len(cohort) != 500:
        raise ValueError("expected the frozen 500-question cohort")
    rows = []
    student = {}
    for q in cohort:
        h = q["problem_hash"]
        s = read(raw / "standalone/1.7B/student" / (h + "-seed0.json"))
        t = read(raw / "standalone/1.7B/teacher_privileged" / (h + "-seed0.json"))
        for b, r in [("student", s), ("teacher_privileged", t)]:
            field = (
                "teacher_prompt_token_ids"
                if b == "teacher_privileged"
                else "student_prompt_token_ids"
            )
            if (
                r["prompt_token_ids"] != q[field]
                or r["seed"] != q["seed"]
                or r["max_tokens"] != q["max_tokens"]
            ):
                raise ValueError("exact prompt, seed or budget changed")
        student[h] = s
        rows.append(
            dict(
                problem_hash=h,
                student_correct=s["correct"],
                teacher_correct=t["correct"],
                student_tokens=len(s["token_ids"]),
                teacher_tokens=len(t["token_ids"]),
            )
        )
    pairs = [r for r in rows if not r["student_correct"] and r["teacher_correct"]]
    shorter = {r["problem_hash"] for r in pairs if r["student_tokens"] < r["teacher_tokens"]}
    # Derive eligibility from the newly generated answers, not a historical case count.
    cases = [
        json.loads(line)
        for line in (inputs / "failure_prefix_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    case_index = {r["problem_hash"]: r for r in cases}
    if len(case_index) != len(cases) or set(case_index) != shorter:
        raise ValueError("continuation cases do not match this run's shorter failures")
    outcomes = []
    for h, case in case_index.items():
        x = read(raw / "continuation/1.7B/teacher_privileged" / (h + ".json"))
        source_student = student[h]
        pref = list(source_student["token_ids"])
        while pref and pref[-1] in (151643, 151645):
            pref.pop()
        if pref != case["C_token_ids"] or x["C_token_ids"] != pref or x["C_text"] != case["C_text"]:
            raise ValueError("complete original failure prefix changed")
        if (
            x["prompt_token_ids"] != case["prompts"]["teacher_privileged"]
            or x["combined_text"] != case["C_text"] + x["suffix_text"]
        ):
            raise ValueError("same-prefix teacher prompt or output concatenation changed")
        outcomes.append(x)
    report = dict(
        scope="final paper Section 3, Qwen3-1.7B only",
        n=500,
        student_correct=sum(r["student_correct"] for r in rows),
        student_failed=sum(not r["student_correct"] for r in rows),
        privileged_teacher_correct=sum(r["teacher_correct"] for r in rows),
        teacher_correct_student_failed=len(pairs),
        student_shorter_with_correct_teacher=len(shorter),
        continuation_prefix_scope="complete prefixes from this run's shorter teacher-solvable failures",
        continuation=dict(
            eligible=len(outcomes),
            immediate_stop=sum(x["empty_or_immediate_stop"] for x in outcomes),
            continued=sum(not x["empty_or_immediate_stop"] for x in outcomes),
            automatically_correct=sum(x["combined_correct"] for x in outcomes),
        ),
        automatic_grading_only=True,
        figure3="AIME24 360-response teacher-solvable curve, implemented separately in analysis/",
    )
    return report, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--input-dir", type=Path, default=ROOT / "inputs/1.7B")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report, _ = compute(a.raw_root, a.input_dir)
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if a.output:
        with a.output.open("x") as f:
            f.write(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
