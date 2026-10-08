"""Rebuild the Section 3 cohort and derive continuation cases from fresh outputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = (151643, 151645)
sys.path.insert(0, str(ROOT / "src"))
from data_identity import json_digest, normalized_problem_hash


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_cohort(model, checkpoint, assets, output):
    import pyarrow.parquet as pq
    from grading import extract_boxed_answer
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "upstream/opsd"))
    from data_collator import SelfDistillationDataCollator

    spec = read(ROOT / "configs/diagnostic_manifest.json")
    hashes = {row["path"]: row["sha256"] for row in read(ROOT / "ASSETS_MANIFEST.json")["files"]}
    shards = {}
    for name in {item["source_file"] for item in spec["rows"]}:
        path = assets / "training" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes["assets/training/" + name]:
            raise ValueError("diagnostic source parquet changed")
        shards[name] = pq.ParquetFile(path).read().to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    collator = SelfDistillationDataCollator(
        tokenizer,
        max_length=20000,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    cohort = []
    for item in spec["rows"]:
        row = shards[item["source_file"]][item["source_row"]]
        answer = extract_boxed_answer(row["solution"])
        if (
            normalized_problem_hash(row["problem"]) != item["problem_hash"]
            or answer is None
            or hashlib.sha256(answer.encode()).hexdigest() != item["answer_sha256"]
        ):
            raise ValueError("diagnostic question or reference answer changed")
        batch = collator([row])
        student = batch["student_prompts"][0].tolist()
        teacher = batch["teacher_prompts"][0].tolist()
        if (
            json_digest(student) != item["student_prompt_sha256"]
            or json_digest(teacher) != item["teacher_prompt_sha256"]
        ):
            raise ValueError("diagnostic tokenizer or prompt changed")
        cohort.append(
            dict(
                problem_hash=item["problem_hash"],
                problem=row["problem"],
                solution=row["solution"],
                answer=answer,
                student_prompt_token_ids=student,
                teacher_prompt_token_ids=teacher,
                seed=item["seed"],
                max_tokens=item["max_tokens"],
            )
        )
    if len(cohort) != spec["count"]:
        raise ValueError("diagnostic cohort incomplete")
    write(output / "cohort.json", cohort)
    write(
        output / "standalone_contract.json",
        dict(
            model=str(Path(model).resolve()),
            n=len(cohort),
            checkpoints={"student": str(Path(checkpoint).resolve())},
            thinking=False,
            max_model_len=32768,
            stop_ids=list(STOP),
        ),
    )
    print(f"Rebuilt {len(cohort)} diagnostic questions with matching prompt hashes")


def continuation_cases(cohort, raw):
    cases = []
    for question in cohort:
        name = question["problem_hash"] + "-seed0.json"
        student = read(raw / "standalone/1.7B/student" / name)
        teacher = read(raw / "standalone/1.7B/teacher_privileged" / name)
        if (
            student["correct"]
            or not teacher["correct"]
            or len(student["token_ids"]) >= len(teacher["token_ids"])
        ):
            continue
        prefix = list(student["token_ids"])
        while prefix and prefix[-1] in STOP:
            prefix.pop()
        prompt = question["teacher_prompt_token_ids"] + prefix
        if len(prompt) >= 32768:
            raise ValueError("selected continuation has no remaining context")
        cases.append(
            dict(
                problem_hash=question["problem_hash"],
                C_token_ids=prefix,
                C_text=student["text"],
                reference_answer=question["answer"],
                prompts={"teacher_privileged": prompt},
                max_new_tokens=32768 - len(prompt),
                seed=question["seed"],
            )
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cohort = sub.add_parser("cohort")
    cohort.add_argument("--model-path", required=True)
    cohort.add_argument("--student-checkpoint", required=True)
    cohort.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    cohort.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics/inputs/1.7B")
    continuation = sub.add_parser("continuation")
    continuation.add_argument("--input-dir", type=Path, default=ROOT / "diagnostics/inputs/1.7B")
    continuation.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cohort":
        build_cohort(args.model_path, args.student_checkpoint, args.asset_root, args.output_dir)
    else:
        cases = continuation_cases(read(args.input_dir / "cohort.json"), args.raw_root)
        path = args.input_dir / "failure_prefix_cases.jsonl"
        with path.open("x") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
        model = read(args.input_dir / "standalone_contract.json")["model"]
        write(args.input_dir / "continuation_contract.json", dict(model=model, sample_n=len(cases)))
        print(f"Prepared {len(cases)} teacher continuations from fresh student failures")


if __name__ == "__main__":
    main()
