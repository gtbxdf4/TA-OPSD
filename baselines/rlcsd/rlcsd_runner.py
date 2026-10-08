#!/usr/bin/env python3
"""Validate a parent-selected matched RLCSD spec and launch its isolated trainer."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PYTHON = sys.executable
ACCELERATE = str(Path(sys.executable).with_name("accelerate"))
OFFICIAL_RLCSD_COMMIT = "a6f8afa7aa7fad82eca12984a3071840a6af4d50"
REQUIRED = {
    "base_model",
    "init_manifest",
    "data_files",
    "output",
    "source_dir",
    "helper_dir",
    "size",
    "dp",
    "mode",
}
SIZE_DP = {"1.7B": 4, "4B": 4, "8B": 8}


def _read_json_arg(value: str | os.PathLike[str]) -> tuple[dict[str, Any], str | None]:
    text = str(value)
    if text.lstrip().startswith("{"):
        return json.loads(text), None
    path = Path(text)
    return json.loads(path.read_text()), str(path.resolve())


def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED - raw.keys())
    if missing:
        raise ValueError("missing required spec fields: " + ", ".join(missing))
    unknown = sorted(set(raw) - REQUIRED)
    if unknown:
        raise ValueError(
            "unknown spec fields (protocol overrides are forbidden): " + ", ".join(unknown)
        )
    spec = copy.deepcopy(raw)
    if spec["size"] not in SIZE_DP:
        raise ValueError("size must be one of 1.7B, 4B, 8B")
    expected_dp = SIZE_DP[spec["size"]]
    if spec["dp"] != expected_dp:
        raise ValueError(f"{spec['size']} requires dp={expected_dp}")
    if spec["mode"] not in ("smoke", "formal"):
        raise ValueError("mode must be smoke or formal")
    if not isinstance(spec["data_files"], list) or len(spec["data_files"]) != 2:
        raise ValueError("data_files must list the two ordered T-series parquet shards")
    for key in ("base_model", "init_manifest", "output", "source_dir", "helper_dir"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"{key} must be a non-empty path string")

    # One sequence per rank in the differentiable policy/teacher forwards keeps
    # full-vocabulary logits bounded.  A complete 256-sequence generation batch
    # is buffered and accumulated into one optimizer update.
    ga = 256 // spec["dp"]
    spec.update(
        protocol="matched-T-series-nonthinking-RLCSD-v1",
        provenance={
            "paper": "arXiv:2606.11709v2",
            "official_rlcsd_commit": OFFICIAL_RLCSD_COMMIT,
            "engine": "TRL GRPOTrainer adapter over the pinned T-series runtime",
            "engine_equivalence": "semantic adapter; official RLCSD uses vendored verl",
        },
        algorithm={
            "group_size": 8,
            "k": 4,
            "epsilon": 0.2,
            "tau": 1.3,
            "beta": 1.0,
            "lambda": 0.5,
            "delta": 0.02,
            "eta": 0.5,
            "residual_clip": [-2.0, 2.0],
            "rollout_is_level": "token",
            "rollout_is_clip": 2.0,
            "positive_hint": "dataset ground-truth solution",
            "negative_hint": "incorrect same-query sibling; target excluded",
            "teacher": "frozen starting model with adapters disabled",
            "unspecified_numeric_coefficient_policy": "default to 1.0",
            "unspecified_coefficients_applied": {"beta": 1.0},
        },
        batching={
            "unique_prompts_per_update": 32,
            "student_sequences_per_update": 256,
            "per_device_sequences": 1,
            "gradient_accumulation_steps": ga,
            "steps_per_generation": ga,
            "num_iterations": 1,
        },
        generation={"max_completion_length": 1024, "temperature": 1.1, "top_p": 0.95, "top_k": 20},
        sequence={"max_length": 20000},
        optimizer={
            "name": "adamw_torch_fused",
            "learning_rate": 5e-6,
            "max_grad_norm": 0.1,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            # Match the observed T-series GOLD path: PEFT is constructed with
            # .05, then every Dropout module is set to p=0.
            "disable_dropout": True,
            "scheduler": "linear",
            "warmup_steps": 0,
            "max_steps": 100,
            "stop_after_updates": 2 if spec["mode"] == "smoke" else None,
            "save_steps": [25, 50, 75, 100],
            "seed": 42,
        },
        lora={
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "configured_dropout": 0.05,
            "effective_dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        precision="bf16",
        training_thinking=False,
        teacher_thinking=False,
        evaluation="disabled",
    )
    return spec


def load_spec(value: str | os.PathLike[str]) -> dict[str, Any]:
    raw, _ = _read_json_arg(value)
    return normalize_spec(raw)


def build_command(spec_arg: str, spec: dict[str, Any]) -> list[str]:
    trainer = str(Path(__file__).resolve().with_name("train_rlcsd.py"))
    return [
        ACCELERATE,
        "launch",
        "--config_file",
        str(Path(spec["source_dir"]) / "accelerate.yaml"),
        "--num_processes",
        str(spec["dp"]),
        "--gradient_accumulation_steps",
        str(spec["batching"]["gradient_accumulation_steps"]),
        "--main_process_port",
        "0",
        trainer,
        "--spec",
        spec_arg,
    ]


def execute(spec_arg: str, spec: dict[str, Any]) -> None:
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(f"fresh output required; refusing existing path: {output}")
    if str(spec_arg).lstrip().startswith("{"):
        normalized_arg = json.dumps({k: spec[k] for k in sorted(REQUIRED)}, ensure_ascii=False)
    else:
        normalized_arg = str(Path(spec_arg).resolve())
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="offline")
    env.pop("PYTHONPATH", None)
    env["PATH"] = str(Path(PYTHON).parent) + ":" + env.get("PATH", "")
    subprocess.run(build_command(normalized_arg, spec), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON file path or inline JSON object")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.dry_run:
        print(
            json.dumps(
                {"spec": spec, "command": build_command(args.spec, spec)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    execute(args.spec, spec)


if __name__ == "__main__":
    main()
