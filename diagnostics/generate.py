"""Generate standalone student/teacher answers or same-prefix teacher continuations."""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STOP = [151643, 151645]


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("standalone", "continuation"), required=True)
    p.add_argument(
        "--branch", choices=("student", "teacher_privileged"), default="teacher_privileged"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, default=ROOT / "inputs/1.7B")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.rank < a.workers:
        raise ValueError("invalid rank")
    if (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1
    ):
        raise ValueError("caller must explicitly select one authorized GPU per TP1 worker")
    out = a.output.resolve()
    if out == ROOT or (ROOT / "inputs") in out.parents or (ROOT / "frozen") in out.parents:
        raise ValueError("preserve frozen inputs/results")
    inp = a.input_dir.resolve()
    if a.mode == "standalone":
        c = read(inp / "standalone_contract.json")
        rows = read(inp / "cohort.json")
        adapter = c["checkpoints"].get(a.branch)
    else:
        if a.branch != "teacher_privileged":
            raise ValueError("final paper uses only privileged-teacher continuation")
        c = read(inp / "continuation_contract.json")
        adapter = None
        rows = sorted(
            [
                json.loads(line)
                for line in (inp / "failure_prefix_cases.jsonl").read_text().splitlines()
                if line.strip()
            ],
            key=lambda r: r["problem_hash"],
        )
        if len(rows) != c["sample_n"]:
            raise ValueError("continuation case count differs from its contract")
        for r in rows:
            if r["C_token_ids"] and (
                r["prompts"]["teacher_privileged"][-len(r["C_token_ids"]) :] != r["C_token_ids"]
            ):
                raise ValueError("complete original failed prefix required")
    rows = [r for i, r in enumerate(rows) if i % a.workers == a.rank]

    def dest(r):
        return (
            out
            / a.mode
            / "1.7B"
            / a.branch
            / (r["problem_hash"] + ("-seed0.json" if a.mode == "standalone" else ".json"))
        )

    if any(dest(r).exists() for r in rows):
        raise FileExistsError("choose fresh output")
    from grading import extract_boxed_answer as extract
    from grading import grade_answer as grade
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    engine = LLM(
        model=c["model"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        enable_lora=bool(adapter),
        max_lora_rank=64,
        max_num_seqs=8,
        seed=42,
    )
    req = LoRARequest(a.branch, 1, adapter) if adapter else None
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        prompts = []
        params = []
        for r in batch:
            if a.mode == "standalone":
                field = (
                    "teacher_prompt_token_ids"
                    if a.branch == "teacher_privileged"
                    else "student_prompt_token_ids"
                )
                ids = r[field]
                budget = r["max_tokens"]
            else:
                ids = r["prompts"]["teacher_privileged"]
                budget = r["max_new_tokens"]
            prompts.append({"prompt_token_ids": ids})
            kw = dict(
                n=1,
                max_tokens=budget,
                seed=r["seed"],
                temperature=1.0,
                top_p=0.8,
                top_k=-1,
                stop_token_ids=STOP,
            )
            if a.mode == "continuation":
                kw.update(min_p=0, presence_penalty=0)
            params.append(SamplingParams(**kw))
        outputs = engine.generate(prompts, sampling_params=params, lora_request=req, use_tqdm=False)
        if len(outputs) != len(batch):
            raise ValueError("missing outputs")
        for j, (r, o) in enumerate(zip(batch, outputs)):
            if list(o.prompt_token_ids) != prompts[j]["prompt_token_ids"] or len(o.outputs) != 1:
                raise ValueError("prompt/sample mismatch")
            y = o.outputs[0]
            ids = list(y.token_ids)
            if y.finish_reason not in ("stop", "length"):
                raise ValueError("unexpected termination")
            common = dict(
                problem_hash=r["problem_hash"],
                branch=a.branch,
                model=c["model"],
                adapter=adapter,
                prompt_token_ids=list(o.prompt_token_ids),
                seed=r["seed"],
                finish_reason=y.finish_reason,
                stop_reason=y.stop_reason,
                time=time.time(),
            )
            if a.mode == "standalone":
                pred = extract(y.text)
                record = dict(
                    common,
                    token_ids=ids,
                    text=y.text,
                    predicted_answer=pred,
                    reference_answer=r["answer"],
                    correct=bool(grade(pred, r["answer"])),
                    max_tokens=r["max_tokens"],
                )
            else:
                combined = r["C_text"] + y.text
                pred = extract(combined)
                nonstop = [t for t in ids if t not in STOP]
                record = dict(
                    common,
                    C_token_ids=r["C_token_ids"],
                    C_text=r["C_text"],
                    suffix_token_ids=ids,
                    suffix_text=y.text,
                    combined_text=combined,
                    combined_correct=bool(grade(pred, r["reference_answer"])),
                    reference_answer=r["reference_answer"],
                    empty_or_immediate_stop=not nonstop,
                    non_stop_suffix_tokens=len(nonstop),
                    max_new_tokens=r["max_new_tokens"],
                )
            target = dest(r)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(
            json.dumps(
                {
                    "completed": start + len(batch),
                    "total": len(rows),
                    "mode": a.mode,
                    "branch": a.branch,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
