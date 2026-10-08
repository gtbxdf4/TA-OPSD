#!/usr/bin/env python3
"""Baseline-aware non-thinking adapter evaluation using the pinned author evaluator."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(ROOT / "upstream/opsd")
EVAL_DATA = {
    k: str(ROOT / "assets/evaluation" / (k + ".parquet")) for k in ("aime24", "aime25", "amc23")
}

EVAL_COUNTS = {"aime24": 30, "aime25": 30, "amc23": 40}


def validate_lora_request(expected, request):
    if request is None or not getattr(request, "lora_path", None):
        raise ValueError("LoRARequest missing; base-model fallback forbidden")
    actual = Path(request.lora_path).resolve()
    expected = Path(expected).resolve()
    if actual != expected:
        raise ValueError(f"LoRARequest path mismatch: expected {expected}, got {actual}")
    return str(actual)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--dataset", choices=EVAL_DATA, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tensor-parallel-size", type=int, choices=(4,), required=True)
    p.add_argument("--max-model-len", type=int, choices=(32768,), required=True)
    p.add_argument("--max-new-tokens", type=int, choices=(38912,), required=True)
    p.add_argument("--temperature", type=float, choices=(1.0,), required=True)
    p.add_argument("--top-p", type=float, choices=(0.8,), required=True)
    p.add_argument("--top-k", type=int, choices=(-1,), required=True)
    p.add_argument("--min-p", type=float, choices=(0.0,), required=True)
    p.add_argument("--presence-penalty", type=float, choices=(0.0,), required=True)
    p.add_argument("--val-n", type=int, required=True)
    a = p.parse_args()
    checkpoint = Path(a.checkpoint_dir) if a.checkpoint_dir else None
    if checkpoint is not None and not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("trained adapter required; base fallback is forbidden")
    out = Path(a.output_dir)
    if out.exists():
        raise FileExistsError("fresh evaluation output required")
    out.mkdir(parents=True)

    import datasets
    import vllm

    original_load, original_llm = datasets.load_dataset, vllm.LLM
    observed_dataset_count = None

    def local_data(name, *args, **kwargs):
        nonlocal observed_dataset_count
        path = EVAL_DATA[a.dataset]
        fmt = "json" if path.endswith(".jsonl") else "parquet"
        split = kwargs.get("split") or "test"
        data = original_load(fmt, data_files={split: path}, split=split)
        if a.dataset == "amc23":
            helper = Path(__file__).resolve().parent
            sys.path.insert(0, str(helper))
            from answer_transport_repair import normalize_dataset

            data = normalize_dataset(data)
        observed_dataset_count = len(data)
        if observed_dataset_count != EVAL_COUNTS[a.dataset]:
            raise ValueError(
                f"dataset row count mismatch: expected {EVAL_COUNTS[a.dataset]}, got {observed_dataset_count}"
            )
        return data

    class SeededLLM(original_llm):
        def __init__(self, *args, **kwargs):
            kwargs["seed"] = 20260727
            if kwargs.get("max_model_len") != 32768 or kwargs.get("tensor_parallel_size") != 4:
                raise ValueError("evaluation engine contract drift")
            super().__init__(*args, **kwargs)
            self._baseline_generate_calls = 0

        def generate(self, *args, **kwargs):
            self._baseline_generate_calls += 1
            if self._baseline_generate_calls != 1:
                raise ValueError("expected exactly one batched generate call")
            adapter = (
                validate_lora_request(checkpoint, kwargs.get("lora_request"))
                if checkpoint is not None
                else None
            )
            if checkpoint is None and kwargs.get("lora_request") is not None:
                raise ValueError("unexpected adapter for BASE")
            sampling = args[1] if len(args) > 1 else kwargs.get("sampling_params")
            if sampling is None or sampling.n != a.val_n or sampling.max_tokens != 38912:
                raise ValueError("sampling count/completion contract drift")
            outputs = super().generate(*args, **kwargs)
            if len(outputs) != EVAL_COUNTS[a.dataset]:
                raise ValueError(
                    f"generation problem count mismatch: expected {EVAL_COUNTS[a.dataset]}, got {len(outputs)}"
                )
            raw = out / "raw_responses.jsonl"
            rows = 0
            with raw.open("x", encoding="utf-8") as stream:
                for problem_index, result in enumerate(outputs):
                    if len(result.outputs) != a.val_n:
                        raise ValueError(
                            f"problem {problem_index} has {len(result.outputs)} responses, expected {a.val_n}"
                        )
                    for generation_index, response in enumerate(result.outputs):
                        stream.write(
                            json.dumps(
                                {
                                    "problem_index": problem_index,
                                    "generation_index": generation_index,
                                    "prompt": result.prompt,
                                    "prompt_token_ids": list(result.prompt_token_ids),
                                    "token_ids": list(response.token_ids),
                                    "text": response.text,
                                    "finish_reason": response.finish_reason,
                                    "stop_reason": response.stop_reason,
                                    "adapter": adapter,
                                    "engine_seed": 20260727,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        rows += 1
            if rows != EVAL_COUNTS[a.dataset] * a.val_n:
                raise ValueError("raw response count mismatch")
            return outputs

    datasets.load_dataset, vllm.LLM = local_data, SeededLLM
    result = out / "author_result.json"
    sys.argv = [
        "evaluate_math.py",
        "--base_model",
        a.base_model,
        "--dataset",
        a.dataset,
        "--output_file",
        str(result),
        "--no_thinking",
        "--val_n",
        str(a.val_n),
        "--tensor_parallel_size",
        "4",
        "--max_model_len",
        "32768",
        "--max_new_tokens",
        "38912",
        "--temperature",
        "1.0",
        "--top_p",
        "0.8",
        "--top_k",
        "-1",
        "--min_p",
        "0.0",
        "--presence_penalty",
        "0.0",
    ]
    if checkpoint is not None:
        sys.argv.extend(["--checkpoint_dir", str(checkpoint)])
    try:
        runpy.run_path(f"{SOURCE}/eval/evaluate_math.py", run_name="__main__")
    finally:
        datasets.load_dataset, vllm.LLM = original_load, original_llm
    if not result.is_file():
        raise FileNotFoundError("author result missing")
    author = json.loads(result.read_text())
    expected_problems = EVAL_COUNTS[a.dataset]
    expected_solutions = expected_problems * a.val_n
    if (
        observed_dataset_count != expected_problems
        or author.get("dataset") != a.dataset
        or author.get("enable_thinking") is not False
        or author.get("val_n") != a.val_n
        or author.get("num_problems") != expected_problems
        or author.get("total_solutions") != expected_solutions
        or len(author.get("results", [])) != expected_problems
    ):
        raise ValueError("author result completeness/contract check failed")
    raw = out / "raw_responses.jsonl"
    with raw.open(encoding="utf-8") as stream:
        raw_count = sum(1 for _ in stream)
    if raw_count != expected_solutions:
        raise ValueError(
            f"raw response count mismatch: expected {expected_solutions}, got {raw_count}"
        )
    (out / "DONE.json").write_text(
        json.dumps(
            {
                "PASS": True,
                "dataset": a.dataset,
                "adapter": str(checkpoint) if checkpoint is not None else None,
                "thinking": False,
                "engine_seed": 20260727,
                "max_model_len": 32768,
                "requested_max_new_tokens": 38912,
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "val_n": a.val_n,
                "num_problems": expected_problems,
                "total_solutions": expected_solutions,
                "raw_responses": str(raw),
                "author_result": str(result),
                "completed": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
