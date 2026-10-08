#!/usr/bin/env python3
"""Plan or launch one frozen paper recipe in an independently allocated run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES = ("SFT", "GRPO", "Stable_OPD", "RLCSD")
METHODS = (
    "OPSD",
    "OPSD_SC",
    "OPSD_UL",
    "TA_OPSD",
    "TA_OPSD_trained_negatives",
    "TA_OPSD_semantic_negatives",
    *BASELINES,
)


def replace_option(arguments, option, value):
    """Set one command-line option while preserving all other arguments."""
    if option in arguments:
        arguments[arguments.index(option) + 1] = str(value)
    else:
        arguments.extend([option, str(value)])


def asset_path(value, asset_root):
    """Resolve a recipe's portable asset location under the supplied root."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "assets":
        path = Path(*path.parts[1:])
    return str(asset_root / path)


def plan(size, method, output, model=None, asset_root=None, mode="formal", auxiliary_manifest=None):
    """Build a launch plan with the paper's method-specific DP layout."""
    cases = json.loads((ROOT / "configs/experiments.json").read_text())
    key = size + "/" + ("TA_OPSD" if method in BASELINES else method)
    if key not in cases:
        raise ValueError("this size/method was not reported in the paper")
    case = cases[key]
    assets = Path(asset_root or ROOT / "assets").resolve()
    model = str(Path(model or ROOT / case["original_model"]).resolve())
    out = Path(output).resolve()
    env = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "TA_OPSD_ASSET_ROOT": str(assets),
    }
    init = asset_path(case["init_manifest"], assets)
    data = [asset_path(path, assets) for path in case["data_files"]]
    dp = case["world_size"]
    if method in ("SFT", "GRPO"):
        # Original native DONE receipts: 1.7B DP4; 4B/8B DP8, effective batch32.
        dp = 4 if size == "1.7B" else 8
    launch = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(ROOT / "upstream/opsd/accelerate.yaml"),
        "--num_processes",
        str(dp),
        "--gradient_accumulation_steps",
        "2",
        "--main_process_port",
        "0",
    ]
    files = {}
    required = [init, str(Path(init).parent / "initial-lora.safetensors"), *data]
    if method in ("Stable_OPD", "RLCSD"):
        spec = dict(
            base_model=model,
            init_manifest=init,
            data_files=data,
            output=str(out / "model"),
            source_dir=str(ROOT / "upstream/opsd"),
            size=size,
            dp=dp,
            mode=mode,
        )
        if method == "RLCSD":
            spec["helper_dir"] = str(ROOT / "src")
        files["spec.json"] = spec
        entry = ROOT / (
            "baselines/stable/train_stable_opd.py"
            if method == "Stable_OPD"
            else "baselines/rlcsd/train_rlcsd.py"
        )
        replace_option(
            launch, "--gradient_accumulation_steps", (256 if method == "RLCSD" else 32) // dp
        )
        spec_argument = json.dumps(spec) if method == "Stable_OPD" else str(out / "spec.json")
        command = launch + [str(entry), "--spec", spec_argument]
    elif method in ("SFT", "GRPO"):
        module_spec = importlib.util.spec_from_file_location(
            "paper_baseline_runner", ROOT / "baselines/baseline_runner.py"
        )
        runner = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(runner)
        spec = runner.build_spec(size, method.lower(), mode, dp=dp)
        command = [sys.executable, "-m", "accelerate.commands.launch"] + spec["command"][2:]
        replace_option(command, "--output_dir", out)
        replace_option(command, "--model_name_or_path", model)
        env.update(spec["environment"])
        env.update(
            BASELINE_EVIDENCE=str(out / "evidence"),
            BASELINE_INIT_MANIFEST=init,
            BASELINE_DATA_FILES=str(assets / "training/train-0000{}-of-00002.parquet"),
        )
    else:
        aux = dict(case["auxiliary"])
        if auxiliary_manifest is not None:
            if method not in (
                "OPSD_SC",
                "OPSD_UL",
                "TA_OPSD",
                "TA_OPSD_trained_negatives",
                "TA_OPSD_semantic_negatives",
            ):
                raise ValueError("fresh calibration is not registered for this method")
            aux = json.loads(Path(auxiliary_manifest).read_text())
            if (
                aux.get("PASS") is not True
                or aux.get("method") != method
                or aux.get("size") != size
                or aux.get("world_size") != dp
                or aux.get("stop_ids") != [151643, 151645]
            ):
                raise ValueError("fresh calibration identity differs from requested arm")
            if aux.get("assets"):
                bank = Path(aux["assets"])
                if hashlib.sha256(bank.read_bytes()).hexdigest() != aux["assets_sha256"]:
                    raise ValueError("fresh calibrated negative bank bytes changed")
            if method != "OPSD_UL" and aux["lambda_eos"] <= 0:
                raise ValueError("SC coefficient must be measured and positive")
            if method != "OPSD_SC" and aux["lambda_ul"] <= 0:
                raise ValueError("UL coefficient must be measured and positive")
        elif aux.get("assets"):
            aux["assets"] = asset_path(aux["assets"], assets)
        if aux.get("assets"):
            required.append(aux["assets"])
        files["auxiliary.json"] = aux
        monitor = json.loads((ROOT / case["monitor_contract"]).read_text())
        monitor["model"] = model
        monitor["question_index_path"] = asset_path(monitor["question_index_path"], assets)
        index_path = Path(monitor["question_index_path"])
        if index_path.exists():
            sys.path.insert(0, str(ROOT / "src"))
            from data_identity import validate_index

            expected_identity = json.loads((ROOT / "configs/data_identity.json").read_text())
            monitor["question_index_sha256"] = validate_index(
                index_path, expected_identity["semantic_index_sha256"]
            )
        monitor["source_parquet_sha256"] = {
            asset_path(path, assets): digest
            for path, digest in monitor["source_parquet_sha256"].items()
        }
        required.append(monitor["question_index_path"])
        files["monitor.json"] = monitor
        env.update(
            V17_SOURCE=str(ROOT / "upstream/opsd"),
            V17_EVIDENCE=str(out / "evidence"),
            V17_UUID=size + "-" + method,
            V17_AUX_MANIFEST=str(out / "auxiliary.json"),
            V17_SMOKE_STOP="2" if mode == "smoke" else "0",
            V18_INITIALIZATION_MANIFEST=init,
            V18_MONITOR_CONTRACT=str(out / "monitor.json"),
            V18_MONITOR_SEGMENT="public-release",
            V18_MONITOR_ARM=method,
            PAPER_TRAIN_FILES=json.dumps(data),
        )
        arguments = list(case["training_args"])
        replace_option(arguments, "--model_name_or_path", model)
        replace_option(arguments, "--output_dir", out)
        replace_option(arguments, "--run_config", method)
        command = launch + [str(ROOT / "src/observed_entry_v2.py")] + arguments
    return dict(
        command=command,
        environment=env,
        runtime_files=files,
        output=str(out),
        mode=mode,
        model_id=case["model_id"],
        model_revision=case["model_revision"],
        world_size=dp,
        required_assets=required,
        original_launch=case.get("original_launch"),
        frozen_coefficients=case.get("auxiliary"),
    )


def execution_environment(configured):
    """Keep executables such as Ninja in the active Python environment discoverable."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(configured)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    return environment


def main():
    """Print the plan by default; execute only in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("1.7B", "4B", "8B"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--asset-root")
    parser.add_argument(
        "--auxiliary-manifest",
        type=Path,
        help="Fresh measured SC/UL coefficients for a regenerated negative bank.",
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = plan(
        args.size,
        args.method,
        args.output,
        args.model_path,
        args.asset_root,
        args.mode,
        args.auxiliary_manifest,
    )
    if not args.execute:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    missing = [path for path in result["required_assets"] if not Path(path).is_file()]
    missing += (
        [str(ROOT / "upstream/opsd/opsd_trainer.py")]
        if not (ROOT / "upstream/opsd/opsd_trainer.py").is_file()
        else []
    )
    if missing:
        parser.error(
            "missing inputs; run scripts/prepare_assets.py and scripts/fetch_upstream.py: "
            + ", ".join(missing)
        )
    out = Path(result["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    for name, value in result["runtime_files"].items():
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (out / "LAUNCH.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    environment = execution_environment(result["environment"])
    subprocess.run(result["command"], env=environment, check=True)


if __name__ == "__main__":
    main()
