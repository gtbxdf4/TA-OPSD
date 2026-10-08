"""Generate full-reference AIME24 teacher answers for Figure 3(a)."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "assets/evaluation/aime24.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise ValueError("select one allocated GPU for this TP1 teacher diagnostic")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    from grading import extract_boxed_answer, grade_answer
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(ROOT / "upstream/opsd"))
    from data_collator import SelfDistillationDataCollator

    rows = pq.read_table(args.dataset).to_pylist()
    if len(rows) != 30:
        raise ValueError("AIME24 must contain 30 questions")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    collator = SelfDistillationDataCollator(
        tokenizer,
        max_length=20000,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_num_seqs=4,
        seed=42,
    )
    results = []
    for start in range(0, 30, 4):
        prompts, params = [], []
        for index in range(start, min(start + 4, 30)):
            row = rows[index]
            solution = row.get("solution")
            if not isinstance(solution, str) or not solution.strip():
                raise ValueError("full reference solution required")
            problem = row.get("problem", row.get("question"))
            ids = collator([dict(problem=problem, solution=solution)])["teacher_prompts"][
                0
            ].tolist()
            if not 0 < len(ids) < 32768:
                raise ValueError("teacher prompt exceeds context")
            seed = (
                int(
                    hashlib.sha256(
                        f"benchmark-prefix-20260924:aime24-{index}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                % 2147483647
            )
            prompts.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    n=1,
                    max_tokens=32768 - len(ids),
                    seed=seed,
                    temperature=1.0,
                    top_p=0.8,
                    top_k=-1,
                    stop_token_ids=[151643, 151645],
                )
            )
        outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise ValueError("missing teacher answers")
        for offset, (prompt, output) in enumerate(zip(prompts, outputs)):
            index = start + offset
            if (
                list(output.prompt_token_ids) != prompt["prompt_token_ids"]
                or len(output.outputs) != 1
            ):
                raise ValueError("teacher prompt/sample mismatch")
            response = output.outputs[0]
            answer = str(rows[index]["answer"])
            prediction = extract_boxed_answer(response.text)
            record = dict(
                problem_index=index,
                correct=bool(grade_answer(prediction, answer)),
                predicted_answer=prediction,
                answer=answer,
                text=response.text,
                token_ids=list(response.token_ids),
                prompt_token_ids=prompt["prompt_token_ids"],
                finish_reason=response.finish_reason,
                seed=params[offset].seed,
            )
            (args.output_dir / f"question-{index}.json").write_text(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            results.append(record)
            print(f"Teacher question {index}: correct={record['correct']}", flush=True)
    (args.output_dir / "teacher_results.json").write_text(
        json.dumps(results, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
