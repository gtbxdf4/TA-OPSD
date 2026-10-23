"""Collect accuracy and generation metrics from fresh evaluation outputs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = ("aime24", "aime25", "amc23")
METHOD_NAMES = {
    "Base": "Base",
    "OPSD": "OPSD",
    "OPSD_SC": "OPSD+SC",
    "OPSD_UL": "OPSD+UL",
    "TA_OPSD": "TA-OPSD",
    "Stable_OPD": "Stable-OPD",
    "TA_OPSD_trained_negatives": "TA-OPSD (OPSD-trained negatives)",
    "TA_OPSD_semantic_negatives": "TA-OPSD (LLM-assisted selection)",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def evaluation(folder, dataset):
    """Join a complete Avg@12 raw-response grid to exact matching author verdict texts.

    Return response-weighted accuracy/length/capping metrics and the joined rows.
    Missing or duplicate responses and mismatched scorer texts are rejected.
    """
    author = read(folder / "author_result.json")
    done = read(folder / "DONE.json")
    questions = 40 if dataset == "amc23" else 30
    if (
        done.get("PASS") is not True
        or author.get("val_n") != 12
        or author.get("num_problems") != questions
    ):
        raise ValueError("complete Avg@12 evaluation required")
    path = folder / "raw_responses.jsonl"
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {(i, j) for i in range(questions) for j in range(12)}
    if (
        len(raw) != len(expected)
        or {(r["problem_index"], r["generation_index"]) for r in raw} != expected
    ):
        raise ValueError("missing or duplicate evaluation answers")
    for row in raw:
        # Score and token statistics must refer to the very same sampled response.
        verdict = author["results"][row["problem_index"]]["generations"][row["generation_index"]]
        if verdict["full_generation"] != row["text"]:
            raise ValueError("scorer and raw answer text differ")
        row.update(correct=verdict["correct"], predicted_answer=verdict["predicted_answer"])
    correct = sum(r["correct"] for r in raw)
    lengths = [len(r["token_ids"]) for r in raw]
    result = dict(
        n=len(raw),
        correct=correct,
        acc=100 * correct / len(raw),
        length=sum(lengths) / len(raw),
        cap=100 * sum(r["finish_reason"] == "length" for r in raw) / len(raw),
    )
    return result, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.manifest)
    base = args.manifest.resolve().parent
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cells, curves, negatives = [], defaultdict(dict), {}
    cached = {}
    for entry in spec["evaluations"]:
        folder = (base / entry["path"]).resolve()
        dataset, method = entry["dataset"], entry["method"]
        if dataset not in DATASETS:
            raise ValueError("unknown paper benchmark")
        metric, raw = evaluation(folder, dataset)
        cached[str(folder)] = raw
        arm = METHOD_NAMES.get(method, method)
        size, step = entry["size"], str(entry["step"])
        if entry.get("main_table"):
            cells.append(
                dict(
                    size=size,
                    method=arm,
                    dataset=dataset,
                    correct=metric["correct"],
                    total=metric["n"],
                    accuracy_pct=metric["acc"],
                )
            )
        if size == "1.7B" and (
            arm in ("OPSD", "OPSD+SC", "OPSD+UL", "TA-OPSD") or method == "Base"
        ):
            arms = ("OPSD", "OPSD+SC", "OPSD+UL", "TA-OPSD") if method == "Base" else (arm,)
            for name in arms:
                curves[step].setdefault(name, {})[dataset] = metric
        if entry.get("negative_table"):
            label = entry["negative_table"]
            negatives.setdefault(label, {"datasets": {}})["datasets"][dataset] = dict(
                n=metric["n"],
                correct=metric["correct"],
                accuracy=metric["acc"],
                mean_length=metric["length"],
                capping_rate=metric["cap"],
            )
    for value in negatives.values():
        if set(value["datasets"]) != set(DATASETS):
            raise ValueError("negative-construction row lacks a benchmark")
        value["macro"] = {
            key: sum(value["datasets"][d][key] for d in DATASETS) / 3
            for key in ("accuracy", "mean_length", "capping_rate")
        }
    write(out / "generation_curves_1p7b.json", dict(steps=curves))
    write(out / "table1_raw_audit.json", dict(cells=cells))
    write(
        out / "table3_negative_selection_step75.json",
        dict(comparison_checkpoint=75, comparison=negatives),
    )
    examples = defaultdict(list)
    for entry in spec.get("examples", []):
        folder = str((base / entry["path"]).resolve())
        row = next(
            r
            for r in cached[folder]
            if r["problem_index"] == entry["problem_index"]
            and r["generation_index"] == entry["generation_index"]
        )
        examples[entry["case"]].append(
            dict(
                method=METHOD_NAMES.get(entry["method"], entry["method"]),
                problem_index=row["problem_index"],
                generation_index=row["generation_index"],
                tokens=len(row["token_ids"]),
                finish_reason=row["finish_reason"],
                correct=row["correct"],
                predicted_answer=row["predicted_answer"],
            )
        )
    write(out / "response_examples.json", dict(cases=examples))
    print(f"Collected {len(cells)} main-table cells")


if __name__ == "__main__":
    main()
