"""Generate negative candidates from the shared initialization or an OPSD adapter."""

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wseed(prefix, key):
    return int(hashlib.sha256((prefix + key).encode()).hexdigest()[:8], 16) % (2**31)


def digest(h):
    return hashlib.sha256(h.encode()).hexdigest()


def put(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(v, ensure_ascii=False))
    t.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out-root", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--adapter", default=None)
    a.add_argument("--tp", type=int, default=4)
    a.add_argument("--cohort", type=Path, default=ROOT / "assets/mining/cohort.json")
    x = a.parse_args()
    cohort = json.loads(x.cohort.read_text())
    root = Path(x.out_root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    contract = dict(
        tag=x.tag,
        model=x.model,
        adapter=x.adapter,
        tp=x.tp,
        thinking=False,
        generation=dict(
            max_model_len=40960, max_tokens=38912, n=1, temperature=1.1, top_k=20, top_p=0.95
        ),
        engine=dict(
            dtype="bfloat16",
            gpu_memory_utilization=0.25,
            seed=42,
            enforce_eager=True,
            max_num_seqs=4,
            batch=4,
        ),
        seed_rule="workflow.seed(t4-rollout-42:, problem_hash)",
        stop_token_ids=[151643, 151645],
        cohort_sha256=sha(x.cohort),
        cohort_size=len(cohort),
    )
    cpath = root / "CONTRACT.json"
    if not cpath.exists():
        put(cpath, contract)
    else:
        prev = json.loads(cpath.read_text())
        assert prev["model"] == contract["model"] and prev.get("adapter") == contract["adapter"], (
            "contract mismatch",
            prev.get("model"),
            contract["model"],
        )
    done = 0
    pending = []
    for r in cohort:
        h = r["problem_hash"]
        f = raw / (digest(h) + ".json")
        if f.exists():
            done += 1
        else:
            pending.append(h)
    if not pending:
        put(
            root / "DONE.json",
            dict(
                PASS=True,
                tag=x.tag,
                completed=done,
                total=len(cohort),
                time=time.time(),
                contract_sha256=sha(cpath),
            ),
        )
        print("ALL DONE", done)
        return 0
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora = LoRARequest("donor", 1, x.adapter) if x.adapter else None
    engine = LLM(
        model=x.model,
        tensor_parallel_size=x.tp,
        dtype="bfloat16",
        max_model_len=40960,
        gpu_memory_utilization=0.25,
        seed=42,
        enforce_eager=True,
        enable_lora=x.adapter is not None,
        max_lora_rank=64,
        max_num_seqs=4,
    )
    byhash = {r["problem_hash"]: r for r in cohort}
    for start in range(0, len(pending), 4):
        batch = pending[start : start + 4]
        prompts = [{"prompt_token_ids": byhash[h]["student_prompt_token_ids"]} for h in batch]
        params = [
            SamplingParams(
                n=1,
                max_tokens=38912,
                temperature=1.1,
                top_p=0.95,
                top_k=20,
                seed=wseed("t4-rollout-42:", h),
                stop_token_ids=[151643, 151645],
            )
            for h in batch
        ]
        res = engine.generate(prompts, sampling_params=params, lora_request=lora, use_tqdm=False)
        assert len(res) == len(batch)
        for h, p, o in zip(batch, prompts, res):
            out = o.outputs[0]
            assert o.prompt_token_ids == p["prompt_token_ids"] and len(o.outputs) == 1
            put(
                raw / (digest(h) + ".json"),
                dict(
                    problem_hash=h,
                    split=byhash[h]["split"],
                    prompt_token_ids=p["prompt_token_ids"],
                    token_ids=list(out.token_ids),
                    text=out.text,
                    finish_reason=out.finish_reason,
                    stop_reason=out.stop_reason,
                    seed=wseed("t4-rollout-42:", h),
                    model=x.model,
                    adapter=x.adapter,
                    is_partial=False,
                    tag=x.tag,
                ),
            )
        put(
            root / "PROGRESS.json",
            dict(
                time=time.time(), completed=done + start + len(batch), total=len(cohort), tag=x.tag
            ),
        )
        print("progress", done + start + len(batch), "/", len(cohort), flush=True)
    put(
        root / "DONE.json",
        dict(
            PASS=True,
            tag=x.tag,
            completed=len(cohort),
            total=len(cohort),
            time=time.time(),
            contract_sha256=sha(cpath),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
