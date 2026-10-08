"""CPU-only, single seed42 initialization; no training or downstream selection."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from launch import atomic
from shared_initialization_v18 import tensor_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=("1.7B", "4B", "8B"), required=True)
    p.add_argument("--model-path")
    p.add_argument("--output")
    a = p.parse_args()
    release = Path(__file__).resolve().parents[1]
    root = Path(a.output) if a.output else release / "assets/initialization" / a.size
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "SHARED_INIT.json"
    expected = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if (root / "initial-lora.safetensors").exists() and manifest_path.exists():
        from shared_initialization_v18 import read_shared

        read_shared(manifest_path)
        print(manifest_path)
        return
    import torch
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, set_seed
    from trl import ModelConfig, get_peft_config

    case = json.loads((release / "configs/experiments.json").read_text())[a.size + "/OPSD"]
    common = {"model_path": a.model_path or case["original_model"]}
    set_seed(42)
    config = ModelConfig(
        model_name_or_path=common["model_path"],
        use_peft=True,
        lora_r=64,
        lora_alpha=128,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = AutoModelForCausalLM.from_pretrained(
        common["model_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    peft_config = get_peft_config(config)
    model = get_peft_model(model, peft_config)
    state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    weights = root / "initial-lora.safetensors"
    if weights.exists():
        raise ValueError("unreceipted initialization file; preserve and inspect")
    save_file(state, str(weights))
    tensor = tensor_manifest(state)
    if expected and tensor != expected["tensor_manifest"]:
        raise ValueError("generated tensor identity differs from frozen paper initialization")
    if not tensor["B_all_zero"]:
        raise ValueError("nonzero initial B")
    cfg = peft_config.to_dict()
    cfg = {k: sorted(v) if isinstance(v, set) else v for k, v in cfg.items()}
    atomic(
        manifest_path,
        {
            "PASS": True,
            "seed": 42,
            "selection": "pre-registered fresh seed42, not historical performance",
            "weights": weights.name,
            "file_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "tensor_manifest": tensor,
            "model_path": common["model_path"],
            "peft_config": cfg,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                n: importlib.metadata.version(n) for n in ("torch", "transformers", "trl", "peft")
            },
            "cpu_eager_initialization_only": True,
            "training_attention_unchanged": "flash_attention_2",
            "training_performed": False,
            "downstream_used": False,
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "tensor_sha256": tensor["tensor_sha256"],
                "tensor_count": tensor["tensor_count"],
                "B_all_zero": tensor["B_all_zero"],
            }
        )
    )


if __name__ == "__main__":
    main()
