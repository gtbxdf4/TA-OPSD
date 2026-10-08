#!/usr/bin/env python3
"""Rebuild training identities and exact candidate prompts from public parquet."""

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_identity import json_digest, normalized_problem_hash, semantic_index_digest


def main():
    """Recover the 1195-row cohort using IDs, splits and tokenizer identity checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asset-root", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "upstream/opsd"))
    from data_collator import SelfDistillationDataCollator

    assets = args.asset_root.resolve()
    hashes = {
        row["path"]: row["sha256"]
        for row in json.loads((ROOT / "ASSETS_MANIFEST.json").read_text())["files"]
    }
    rows, index = [], {}
    for path in sorted((assets / "training").glob("train-*-of-00002.parquet")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes["assets/training/" + path.name]:
            raise ValueError("training parquet differs from paper source")
        for local_id, row in enumerate(pq.ParquetFile(path).read().to_pylist()):
            problem, solution = str(row["problem"]), str(row["solution"])
            pair = hashlib.sha256(
                json.dumps([problem, solution], ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            entry = index.setdefault(
                pair,
                {
                    "question_hash": hashlib.sha256(problem.encode()).hexdigest(),
                    "source_row_ids": [],
                    "source_locations": [],
                },
            )
            entry["source_row_ids"].append(len(rows))
            entry["source_locations"].append({"file": str(path), "row_index": local_id})
            rows.append(row)
    identity = json.loads((ROOT / "configs/data_identity.json").read_text())
    if (
        len(rows) != identity["source_rows"]
        or semantic_index_digest(index) != identity["semantic_index_sha256"]
    ):
        raise ValueError("training row order or identity changed")
    index_path = assets / "training/question-index-c5b0cfb70cfb.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    manifest = json.loads((ROOT / "configs/mining_manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        collator = SelfDistillationDataCollator(
            tokenizer,
            max_length=20000,
            reason_first=False,
            student_thinking=False,
            teacher_thinking=False,
        )
    cohort = []
    for item in manifest["rows"]:
        row = rows[item["source_row_id"]]
        if normalized_problem_hash(row["problem"]) != item["problem_hash"]:
            raise ValueError("candidate source row identity changed")
        batch = collator([dict(problem=row["problem"], solution=row["solution"])])
        prompt = batch["student_prompts"][0].tolist()
        if json_digest(prompt) != item["student_prompt_sha256"]:
            raise ValueError("candidate tokenizer/prompt identity changed")
        cohort.append(
            dict(
                problem_hash=item["problem_hash"],
                split=item["split"],
                student_prompt_token_ids=prompt,
            )
        )
    if len(cohort) != manifest["count"]:
        raise ValueError("candidate count changed")
    target = assets / "mining/cohort.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "PASS": True,
                "training_rows": len(rows),
                "cohort_rows": len(cohort),
                "index_semantic_sha256": semantic_index_digest(index),
                "cohort_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
