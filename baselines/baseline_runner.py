#!/usr/bin/env python3
"""Build or execute one isolated official SFT/GRPO baseline run."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
ACCELERATE = str(Path(sys.executable).with_name("accelerate"))
SOURCE = str(ROOT / "upstream/opsd")
POOL = str(ROOT / "runs/baselines")
DATA = str(ROOT / "assets/training/train-0000{}-of-00002.parquet")
MODELS = {
    k: v["original_model"]
    for k, v in {
        x.split("/")[0]: y
        for x, y in json.loads((ROOT / "configs/experiments.json").read_text()).items()
    }.items()
}
MANIFESTS = {k: str(ROOT / "assets/initialization" / k / "SHARED_INIT.json") for k in MODELS}
WEIGHTS = {k: str(ROOT / "assets/initialization" / k / "initial-lora.safetensors") for k in MODELS}
EVAL_DATA = {
    k: str(ROOT / "assets/evaluation" / (k + ".parquet")) for k in ("aime24", "aime25", "amc23")
}


def build_spec(size, method, mode, dp=8):
    if size not in MODELS or method not in ("sft", "grpo") or mode not in ("smoke", "formal"):
        raise ValueError("unsupported baseline selection")
    if dp not in (4, 8):
        raise ValueError("dp must be 4 or 8")
    horizon = 100 if method == "sft" else 500
    save = 25 if method == "sft" else 50
    accum = 4 if dp == 8 else 8
    run = f"{size}-{method}-{mode}-dp{dp}"
    output = f"{POOL}/{size}/{method}/{mode}/{run}"
    bootstrap = str(ROOT / "baselines/baseline_bootstrap.py")
    args = [
        ACCELERATE,
        "launch",
        "--config_file",
        f"{SOURCE}/accelerate.yaml",
        "--num_processes",
        str(dp),
        "--gradient_accumulation_steps",
        str(accum),
        "--main_process_port",
        "0",
        bootstrap,
        "--method",
        method,
        "--official-source",
        SOURCE,
        "--model_name_or_path",
        MODELS[size],
        "--learning_rate",
        "5e-6",
        "--lr_scheduler_type",
        "linear",
        "--warmup_steps",
        "0",
        "--max_steps",
        str(horizon),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(accum),
        "--output_dir",
        output,
        "--gradient_checkpointing",
        "--use_peft",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--lora_dropout",
        "0.05",
        "--lora_target_modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--logging_steps",
        "1",
        "--save_steps",
        str(save),
        "--save_strategy",
        "steps",
        "--report_to",
        "none",
    ]
    if method == "sft":
        args += ["--max_length", "16000"]
    else:
        args += [
            "--run_config",
            run,
            "--num_iterations",
            "2",
            "--max_prompt_length",
            "2048",
            "--max_completion_length",
            "16000",
            "--num_generations",
            "8",
            "--temperature",
            "1.2",
            "--use_vllm",
            "--vllm_mode",
            "colocate",
            "--beta",
            "0.0",
            "--loss_type",
            "grpo",
            "--scale_rewards",
            "group",
            "--wandb_project",
            "OPSD-paper-baselines",
        ]
    return {
        "size": size,
        "method": method,
        "mode": mode,
        "dp": dp,
        "effective_batch": dp * accum,
        "formal_scheduler_horizon": horizon,
        "stop_after_updates": 2 if mode == "smoke" else None,
        "output": output,
        "command": args,
        "environment": {
            "BASELINE_INIT_MANIFEST": MANIFESTS[size],
            "BASELINE_DATA_FILES": DATA,
            "BASELINE_STOP_AFTER_UPDATES": "2" if mode == "smoke" else "",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        },
        "requires": [
            MODELS[size] + "/config.json",
            MANIFESTS[size],
            WEIGHTS[size],
            DATA.format(0),
            DATA.format(1),
            SOURCE + f"/{method}_train.py",
        ],
        "native_receipts": [output + "/DONE.json"],
    }
