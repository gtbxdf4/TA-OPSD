"""Rebuild and run the paper's relative short-error and FREE/ZERO diagnostic."""

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

from scripts.fetch_upstream import UPSTREAM
from src.data_identity import json_digest, normalized_problem_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/eos_intervention.json").read_text())
STOP_IDS = SPEC["stop_ids"]
CONTEXT = SPEC["max_model_len"]
STANDALONE = ("base", "opsd", "sc", "ta-opsd")
PAIR = ("FREE", "ZERO")


def read(path):
    return json.loads(path.read_text())


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def upstream_file(name):
    source = ROOT / "upstream/opsd" / name
    expected = next(row["sha256"] for row in UPSTREAM["files"] if row["path"] == name)
    if sha256(source.read_bytes()) != expected:
        raise ValueError(f"{name} differs from pinned OPSD; run scripts/fetch_upstream.py")
    return source


def author_grader():
    """Load the pinned author's two grading helpers without starting its evaluator."""
    from math_verify import parse, verify

    path = upstream_file("eval/evaluate_math.py")
    names = {"extract_boxed_answer", "grade_answer"}
    funcs = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(funcs) != len(names) or {node.name for node in funcs} != names:
        raise ValueError("unexpected upstream grading helper definitions")
    scope = dict(parse=parse, verify=verify, re=re)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), scope)
    return scope["extract_boxed_answer"], scope["grade_answer"]


def prepare(model, assets, output):
    """Recover the original cohort from row references, without generating a teacher answer."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source = upstream_file("data_collator.py")
    sys.path.insert(0, str(source.parent))
    from data_collator import SelfDistillationDataCollator

    extract, _ = author_grader()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    collator = SelfDistillationDataCollator(
        tokenizer,
        max_length=20000,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    checksums = {row["path"]: row["sha256"] for row in read(ROOT / "ASSETS_MANIFEST.json")["files"]}
    shards = {}
    cohort = []
    for item in SPEC["rows"]:
        name = item["source_file"]
        if name not in shards:
            path = assets / "training" / name
            if sha256(path.read_bytes()) != checksums["assets/training/" + name]:
                raise ValueError(f"diagnostic source parquet changed: {name}")
            shards[name] = pq.ParquetFile(path).read().to_pylist()
        row = shards[name][item["source_row"]]
        answer = extract(row["solution"])
        batch = collator([row])
        normal_ids = batch["student_prompts"][0].tolist()
        # The original standalone budget was shared with the longer privileged prompt.
        # The privileged prompt is used only for this length check, never for generation.
        privileged_len = len(batch["teacher_prompts"][0])
        budget = min(38912, CONTEXT - max(len(normal_ids), privileged_len))
        if (
            normalized_problem_hash(row["problem"]) != item["problem_hash"]
            or not answer
            or sha256(answer.encode()) != item["answer_sha256"]
            or json_digest(normal_ids) != item["student_prompt_sha256"]
            or budget != item["max_tokens"]
        ):
            raise ValueError(
                f"source row, answer, prompt or budget changed: {item['problem_hash']}"
            )
        cohort.append(
            dict(
                problem_hash=item["problem_hash"],
                prompt_token_ids=normal_ids,
                reference_answer=answer,
                seed=item["seed"],
                budget=budget,
            )
        )
    if len(cohort) != SPEC["count"] or len({r["problem_hash"] for r in cohort}) != len(cohort):
        raise ValueError("diagnostic cohort incomplete or duplicated")
    output.mkdir(parents=True, exist_ok=False)
    (output / "cohort.json").write_text(json.dumps(dict(model=str(model), rows=cohort)) + "\n")
    print(f"Prepared {len(cohort)} diagnostic questions")


def continuation_seed(problem_hash, index):
    value = sha256(f"ICLR27-student-EOS-v1:{problem_hash}:{index}".encode())
    return int(value[:8], 16) % 2147483647


def select_cases(cohort, base, opsd):
    """Count relative short errors separately from those with an observable final STOP."""
    denominator, relative, eligible = 0, [], []
    for row in cohort:
        h = row["problem_hash"]
        b, o = base[h], opsd[h]
        if not b["correct"] or b["finish_reason"] != "stop":
            continue
        denominator += 1
        ids = o["token_ids"]
        if o["correct"] or len(ids) >= len(b["token_ids"]):
            continue
        relative.append(row)
        if o["finish_reason"] == "stop" and ids and ids[-1] in STOP_IDS:
            # Remove the final sampled STOP only; the preceding token sequence is immutable.
            eligible.append(dict(row, history_ids=ids[:-1], opsd_text=o["text"]))
    return denominator, relative, eligible


def standalone_requests(cohort, arm):
    return [
        dict(
            arm=arm,
            problem_hash=row["problem_hash"],
            prompt_token_ids=row["prompt_token_ids"],
            seed_index=0,
            seed=row["seed"],
            budget=row["budget"],
        )
        for row in cohort
    ]


def paired_requests(eligible):
    requests = []
    for row in sorted(eligible, key=lambda r: r["problem_hash"]):
        h = row["problem_hash"]
        prompt = row["prompt_token_ids"] + row["history_ids"]
        budget = min(SPEC["continuation_cap"], CONTEXT - len(prompt))
        if budget <= 0:
            raise ValueError(f"no remaining context for {h}")
        for index in range(SPEC["seeds_per_problem"]):
            seed = continuation_seed(h, index)
            requests.extend(
                dict(
                    arm=arm,
                    problem_hash=h,
                    prompt_token_ids=prompt,
                    seed_index=index,
                    seed=seed,
                    budget=budget,
                )
                for arm in PAIR
            )
    return requests


def request_key(row):
    return row["problem_hash"], row["arm"], row["seed_index"]


def validate_result(row, request):
    if any(row[key] != value for key, value in request.items()):
        raise ValueError(f"generation contract mismatch for {request_key(request)}")
    ids = row["token_ids"]
    if (
        row["finish_reason"] not in ("stop", "length")
        or len(ids) > request["budget"]
        or type(row["correct"]) is not bool
        or (row["arm"] == "ZERO" and (not ids or ids[0] in STOP_IDS))
    ):
        raise ValueError(f"invalid response for {request_key(request)}")


def read_results(run_dir, name, requests):
    """Reject missing, duplicate or non-matching generations before computing metrics."""
    expected = {request_key(r): r for r in requests}
    rows = {}
    with (run_dir / f"{name}.jsonl").open() as stream:
        metadata = json.loads(next(stream))
        for line in stream:
            row = json.loads(line)
            key = request_key(row)
            if key in rows or key not in expected:
                raise ValueError(f"duplicate or unexpected response: {key}")
            validate_result(row, expected[key])
            rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError(f"incomplete generations in {name}.jsonl")
    return metadata, rows


def load_selection(run_dir, cohort):
    baselines = []
    for arm in ("base", "opsd"):
        meta, rows = read_results(run_dir, arm, standalone_requests(cohort, arm))
        baselines.append((meta, {h: row for (h, _, _), row in rows.items()}))
    return baselines, select_cases(cohort, baselines[0][1], baselines[1][1])


def generate(run_dir, arm, adapter):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    contract = read(run_dir / "cohort.json")
    cohort = contract["rows"]
    by_hash = {r["problem_hash"]: r for r in cohort}
    model = contract["model"]
    if (arm == "base") != (adapter is None):
        raise ValueError("use no adapter for base and a local step-100 adapter for every other arm")
    adapter_hash = sha256((adapter / "adapter_model.safetensors").read_bytes()) if adapter else None
    histories = {}
    if arm in ("base", "opsd"):
        requests = standalone_requests(cohort, arm)
    else:
        baselines, (_, relative, eligible) = load_selection(run_dir, cohort)
        if arm == "paired":
            if adapter_hash != baselines[1][0]["adapter_sha256"]:
                raise ValueError("FREE and ZERO must use the same adapter as standalone OPSD")
            histories = {r["problem_hash"]: r for r in eligible}
            requests = paired_requests(eligible)
        else:
            requests = standalone_requests(relative, arm)

    output = run_dir / f"{arm}.jsonl"
    if output.exists():
        raise FileExistsError(output)
    extract, grade = author_grader()
    engine = None
    tokenizer = None
    lora = None
    if requests:
        engine = LLM(
            model=model,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=CONTEXT,
            gpu_memory_utilization=0.65,
            enforce_eager=True,
            enable_lora=adapter is not None,
            max_lora_rank=64,
            max_num_seqs=8,
            seed=42,
            logits_processors=(
                ["diagnostics.first_decision_eos:FirstDecisionEOS"] if arm == "paired" else None
            ),
        )
        tokenizer = engine.get_tokenizer()
        lora = LoRARequest(arm, 1, str(adapter.resolve())) if adapter else None
        for row in histories.values():
            prefix = tokenizer.decode(
                row["history_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if prefix != row["opsd_text"]:
                raise ValueError(
                    f"OPSD prefix text does not match preserved IDs: {row['problem_hash']}"
                )

    with output.open("x") as stream:
        stream.write(json.dumps(dict(adapter_sha256=adapter_hash)) + "\n")
        for start in range(0, len(requests), 8):
            batch = requests[start : start + 8]
            prompts = [dict(prompt_token_ids=r["prompt_token_ids"]) for r in batch]
            params = [
                SamplingParams(
                    n=1,
                    max_tokens=r["budget"],
                    seed=r["seed"],
                    stop_token_ids=STOP_IDS,
                    temperature=SPEC["temperature"],
                    top_p=SPEC["top_p"],
                    top_k=SPEC["top_k"],
                    extra_args=(dict(zero_first_stop_ids=STOP_IDS) if r["arm"] == "ZERO" else None),
                )
                for r in batch
            ]
            outputs = engine.generate(prompts, params, lora_request=lora, use_tqdm=False)
            if len(outputs) != len(batch):
                raise ValueError("generation batch incomplete")
            for request, result in zip(batch, outputs):
                if (
                    list(result.prompt_token_ids) != request["prompt_token_ids"]
                    or len(result.outputs) != 1
                ):
                    raise ValueError("generation prompt or sample count mismatch")
                sample = result.outputs[0]
                h = request["problem_hash"]
                ids = list(sample.token_ids)
                text = sample.text
                if arm == "paired":
                    text = tokenizer.decode(
                        histories[h]["history_ids"] + ids, skip_special_tokens=True
                    )
                row = dict(
                    request,
                    token_ids=ids,
                    text=text,
                    finish_reason=sample.finish_reason,
                    correct=bool(grade(extract(text), by_hash[h]["reference_answer"])),
                )
                validate_result(row, request)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{arm}: {start + len(batch)}/{len(requests)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--model-path", required=True, type=Path)
    prep.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    prep.add_argument("--output-dir", required=True, type=Path)
    gen = modes.add_parser("generate")
    gen.add_argument("--run-dir", required=True, type=Path)
    gen.add_argument("--arm", required=True, choices=(*STANDALONE, "paired"))
    gen.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model_path.resolve(), args.asset_root, args.output_dir)
    else:
        generate(args.run_dir, args.arm, args.adapter)


if __name__ == "__main__":
    main()
