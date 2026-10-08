"""Build paper plotting inputs from new training and evaluation outputs."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = ("aime24", "aime25", "amc23")
ARMS = {
    "Base": "base",
    "OPSD": "T0",
    "OPSD_SC": "T1",
    "OPSD_UL": "T0+UL",
    "TA_OPSD": "T1+UL",
    "Stable_OPD": "STABLE",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def evaluation(folder, dataset, teacher=None):
    """Join raw generations to Avg@12 verdicts and compute response-level behavior."""
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
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    if teacher is not None:
        from diagnostics.classification import short_error

        short = [r for r in raw if short_error(r)]
        eligible = sum(teacher[r["problem_index"]] for r in short)
        result["short_errors"] = dict(
            short=len(short),
            eligible=eligible,
            total=len(raw),
            pct=100 * eligible / len(raw),
            raw_sha256=result["raw_sha256"],
        )
    return result, raw


def training_rows(folder, method):
    """Merge complete distributed monitor shards at valid distillation positions."""
    steps = defaultdict(list)
    layouts = defaultdict(set)
    worlds = {}
    seen = set()
    for path in sorted((folder / "evidence/raw-monitor").rglob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            packet = json.load(stream)
        key = (packet["optimizer_step"], packet["rank"], packet["microbatch_index"])
        if key in seen:
            raise ValueError("duplicate training-monitor microbatch")
        seen.add(key)
        world = packet["world_size"]
        if world not in (4, 8) or worlds.get(key[0], world) != world:
            raise ValueError("training data-parallel layout changed")
        worlds[key[0]] = world
        layouts[key[0]].add((key[1], key[2]))
        steps[key[0]].extend(packet["records"])
    if set(steps) != set(range(1, 101)):
        raise ValueError("100 completed monitored optimizer updates required")
    output = []
    for step, records in sorted(steps.items()):
        if layouts[step] != {(rank, micro) for rank in range(worlds[step]) for micro in range(2)}:
            raise ValueError("incomplete distributed training-monitor update")
        if len(records) != 32:
            raise ValueError("training update must contain 32 student responses")
        sums = defaultdict(float)
        count = 0
        for record in records:
            metrics = record["metrics"]
            for position, valid in enumerate(record["loss_valid_mask"]):
                # Signal plots use valid-token means; SC/UL loss reductions stay separate.
                if not valid:
                    continue
                count += 1
                sums["topk_overlap_fraction"] += metrics["topk_overlap_fraction"][position]
                for token in (151643, 151645):
                    sums[f"student_eos_logp_{token}"] += metrics["student_eos_logp"][str(token)][
                        position
                    ]
                    sums[f"eos_logp_gap_teacher_minus_student_{token}"] += metrics[
                        "eos_logp_gap_teacher_minus_student"
                    ][str(token)][position]
        if not count:
            raise ValueError("training update has no valid response positions")
        means = {key: value / count for key, value in sums.items()}
        output.append(
            dict(
                method=method,
                step=step,
                tokens=count,
                responses=32,
                **means,
                eos_gap_mean=sum(
                    means[f"eos_logp_gap_teacher_minus_student_{t}"] for t in (151643, 151645)
                )
                / 2,
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.manifest)
    base = args.manifest.resolve().parent
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    teacher_rows = read(base / spec["teacher_aime24"])
    teacher = {r["problem_index"]: bool(r["correct"]) for r in teacher_rows}
    if len(teacher_rows) != 30 or set(teacher) != set(range(30)):
        raise ValueError(
            "one independent full-reference teacher answer per AIME24 question required"
        )
    cells, curves, negatives, short = [], defaultdict(dict), {}, defaultdict(dict)
    cached = {}
    for entry in spec["evaluations"]:
        folder = (base / entry["path"]).resolve()
        dataset, method = entry["dataset"], entry["method"]
        if dataset not in DATASETS:
            raise ValueError("unknown paper benchmark")
        metric, raw = evaluation(folder, dataset, teacher if dataset == "aime24" else None)
        cached[str(folder)] = raw
        arm = ARMS.get(method, method)
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
        if size == "1.7B" and (arm in ("T0", "T1", "T0+UL", "T1+UL") or method == "Base"):
            arms = ("T0", "T1", "T0+UL", "T1+UL") if method == "Base" else (arm,)
            for name in arms:
                curves[step].setdefault(name, {})[dataset] = metric
                if dataset == "aime24" and name in ("T0", "T1+UL"):
                    short[step]["OPSD" if name == "T0" else "Ours"] = metric["short_errors"]
        if entry.get("negative_table"):
            label = entry["negative_table"]
            negatives.setdefault(label, {"datasets": {}})["datasets"][dataset] = dict(
                n=metric["n"],
                correct=metric["correct"],
                accuracy=metric["acc"],
                mean_length=metric["length"],
                capping_rate=metric["cap"],
                sha256=metric["raw_sha256"],
            )
    for value in negatives.values():
        if set(value["datasets"]) != set(DATASETS):
            raise ValueError("negative-construction row lacks a benchmark")
        value["macro"] = {
            key: sum(value["datasets"][d][key] for d in DATASETS) / 3
            for key in ("accuracy", "mean_length", "capping_rate")
        }
    write(out / "generation_curves_1p7b.json", dict(steps=curves))
    write(out / "teacher_solvable_short_errors.json", dict(results=short))
    write(out / "table1_raw_audit.json", dict(cells=cells))
    write(
        out / "table3_negative_selection_step75.json",
        dict(comparison_checkpoint=75, comparison=negatives),
    )
    by_size = defaultdict(list)
    budgets = {}
    for entry in spec["training"]:
        folder = (base / entry["path"]).resolve()
        rows = training_rows(folder, ARMS[entry["method"]])
        by_size[entry["size"]].extend(rows)
        replay_tokens = 0
        for path in (folder / "evidence").glob("aux-replay-rank*.jsonl"):
            # Count only supervised suffix tokens; detached prefix caches are excluded.
            for line in path.read_text().splitlines():
                replay_tokens += sum(
                    record.get("negative", {}).get("selected_tokens", 0)
                    for record in json.loads(line)["records"]
                )
        budgets[entry["size"] + "/" + entry["method"]] = dict(
            main_supervised_tokens=sum(row["tokens"] for row in rows),
            negative_supervised_tokens=replay_tokens,
        )
    for size, rows in by_size.items():
        with (out / f"training_{size}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
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
                method=entry["method"],
                problem_index=row["problem_index"],
                generation_index=row["generation_index"],
                tokens=len(row["token_ids"]),
                finish_reason=row["finish_reason"],
                correct=row["correct"],
                predicted_answer=row["predicted_answer"],
            )
        )
    write(out / "response_examples.json", dict(cases=examples))
    write(out / "training_token_budget.json", budgets)
    print(f"Collected {len(cells)} main-table cells and {len(spec['training'])} training traces")


if __name__ == "__main__":
    main()
