#!/usr/bin/env python3
"""Calibrate SC and UL at initialization using the gradient norms in Appendix B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
STOP_IDS = [151643, 151645]


def norm_ratio(reference, auxiliary):
    """Return the positive finite coefficient matching two gradient norms."""
    if not all(math.isfinite(x) and x > 0 for x in (reference, auxiliary)):
        raise ValueError("nonzero finite gradient norms required")
    return reference / auxiliary


def norm(values):
    """Compute the L2 norm in float64 after collecting LoRA gradients."""
    return math.sqrt(sum(float(value.double().square().sum()) for value in values))


def sha(path):
    """Hash one frozen input without modifying it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrate(smoke, negatives, model_path, method, output):
    """Run the actual author loss and local UL forward at common initialization."""
    if method not in (
        "OPSD_SC",
        "OPSD_UL",
        "TA_OPSD",
        "TA_OPSD_trained_negatives",
        "TA_OPSD_semantic_negatives",
    ):
        raise ValueError("fresh calibration is limited to reported objectives")
    smoke, output = map(Path, (smoke, output))
    negatives = Path(negatives) if negatives is not None else None
    if method != "OPSD_SC" and negatives is None:
        raise ValueError("UL calibration requires a negative library")
    if output.exists():
        raise FileExistsError("fresh calibration directory required")
    evidence = smoke / "evidence"
    launch = json.loads((smoke / "LAUNCH.json").read_text())
    if launch.get("mode") != "smoke" or launch.get("world_size") not in (4, 8):
        raise ValueError("matching actual two-update smoke is required")
    size = next(size for size in ("1.7B", "4B", "8B") if launch["model_id"].endswith(size))
    case = json.loads((ROOT / "configs/experiments.json").read_text())[size + "/OPSD"]
    world = launch["world_size"]
    packets = [
        evidence / f"fixed-step0-inputs-rank{rank}-micro{micro}.pt"
        for micro in range(2)
        for rank in range(world)
    ]
    if any(not path.is_file() for path in packets):
        raise ValueError("one complete step-0 batch with RNG packets is required")
    runtime = [
        json.loads((evidence / f"v2-runtime-rank{rank}.json").read_text()) for rank in range(world)
    ]
    expected_clip = float(
        case["training_args"][case["training_args"].index("--jsd_token_clip") + 1]
    )
    if any(float(row["jsd_token_clip"]) != expected_clip for row in runtime):
        raise ValueError("runtime JSD clip differs from frozen base")
    if any(not row["fixed_teacher"] for row in runtime):
        raise ValueError("author teacher protocol changed")
    for row in runtime:
        fields = row["stop"]["engine_generation_config_fields"]
        extra = fields.get("eos_token_id", [])
        extra = [extra] if isinstance(extra, int) else extra
        primary = row["stop"]["engine_primary_eos"]
        if sorted(set([primary, *extra])) != STOP_IDS:
            raise ValueError("runtime EOS union differs from the recorded stop set")
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from trl.trainer.utils import disable_dropout_in_model

    sys.path.insert(0, str(ROOT / "upstream/opsd"))
    from opsd_trainer import OPSDTrainer

    sys.path.insert(0, str(ROOT / "src"))
    from independent_losses import local_unlikelihood, stop_kl
    from replay_v2 import detached_suffix_logits
    from shared_initialization_v18 import read_shared, validate_runtime_initialization

    values = [torch.load(path, map_location="cpu", weights_only=False) for path in packets]
    if sum(len(row["student_input_ids"]) for row in values) != 32:
        raise ValueError("reference is not one global author batch of 32")
    rows = (
        [json.loads(line) for line in negatives.read_text().splitlines() if line.strip()]
        if method != "OPSD_SC"
        else []
    )
    if method != "OPSD_SC" and (
        not rows
        or any(
            row.get("approved") is not True
            or row.get("split") != "fit"
            or "negative" not in row
            or row.get("positive")
            for row in rows
        )
    ):
        raise ValueError("negative-only fit library required")
    adapter = list(smoke.rglob("checkpoint-2/adapter_config.json"))
    if len(adapter) != 1:
        raise ValueError("unique native two-update LoRA config required")
    initial = evidence / "initial-lora.safetensors"
    if not initial.is_file():
        raise ValueError("captured pre-update common adapter missing")
    asset_root = Path(launch["environment"]["TA_OPSD_ASSET_ROOT"])
    shared, _ = read_shared(asset_root / "initialization" / size / "SHARED_INIT.json")
    validate_runtime_initialization(load_file(str(initial)), shared)
    del shared
    resolved = json.loads((evidence / "resolved_config-rank0.jsonl").open().readline())["args"]
    if resolved["disable_dropout"] is not True or resolved["max_steps"] != 100:
        raise ValueError("reference runtime dropout or optimizer horizon changed")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    ).to(device)
    config = LoraConfig.from_pretrained(adapter[0].parent)
    config.inference_mode = False
    model = get_peft_model(base, config)
    set_peft_model_state_dict(model, load_file(str(initial)))
    disable_dropout_in_model(model)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=resolved["gradient_checkpointing_kwargs"]
    )
    model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    class CalibrationTrainer(OPSDTrainer):
        """Reuse the author loss; only measure and optionally add SC."""

        def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None, **kwargs):
            primary = OPSDTrainer.generalized_jsd_loss(
                self, student_logits, teacher_logits, labels, **kwargs
            )
            auxiliary = stop_kl(student_logits, teacher_logits, labels != -100, STOP_IDS)
            if self.collect_logits:
                base_gradient = torch.autograd.grad(primary, student_logits, retain_graph=True)[0]
                stop_gradient = torch.autograd.grad(auxiliary, student_logits, retain_graph=True)[0]
                self.base_sq += float(base_gradient.float().square().sum())
                self.stop_sq += float(stop_gradient.float().square().sum())
            return primary + self.stop_coefficient * auxiliary

    trainer = object.__new__(CalibrationTrainer)
    trainer.temperature = 1.1
    trainer.beta = 0
    trainer.top_k_loss = 0
    trainer.jsd_token_clip = expected_clip
    trainer.fixed_teacher = True
    trainer.use_ema_teacher = False
    trainer.use_thinking_machines_loss = False
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    trainer.collect_logits = True
    trainer.stop_coefficient = 0.0
    trainer.base_sq = 0.0
    trainer.stop_sq = 0.0

    def prepared(packet):
        torch.set_rng_state(packet["_rng_cpu"])
        torch.cuda.set_rng_state(packet["_rng_cuda"])
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in packet.items()
            if not key.startswith("_rng")
        }

    for packet in values:
        with torch.enable_grad():
            trainer.compute_loss(model, prepared(packet))
    coefficient_sc = norm_ratio(math.sqrt(trainer.base_sq), math.sqrt(trainer.stop_sq))
    trainer.collect_logits = False
    trainer.stop_coefficient = coefficient_sc if method != "OPSD_UL" else 0.0

    def gradients():
        return [
            parameter.grad.detach().float().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in parameters
        ]

    def clear():
        for parameter in parameters:
            parameter.grad = None

    clear()
    for packet in values:
        loss = trainer.compute_loss(model, prepared(packet)) / len(values)
        loss.backward()
    reference_norm = norm(gradients())
    clear()
    coefficient_ul = 0.0
    ul_norm = None
    if method != "OPSD_SC":
        for row in rows:
            span = row["negative"]
            logits = detached_suffix_logits(model, span["prefix_ids"], span["suffix_ids"], device)
            token_ids = torch.tensor([span["suffix_ids"]], device=device)
            selected = torch.tensor([span["selected"]], device=device, dtype=torch.bool)
            (local_unlikelihood(logits, token_ids, selected, STOP_IDS) / len(rows)).backward()
        ul_norm = norm(gradients())
        coefficient_ul = norm_ratio(reference_norm, ul_norm)
        clear()
    output.mkdir(parents=True)
    coefficient_sc = 0.0 if method == "OPSD_UL" else coefficient_sc
    auxiliary = {
        "PASS": True,
        "method": method,
        "size": size,
        "lambda_eos": coefficient_sc,
        "lambda_ul": coefficient_ul,
        "stop_ids": STOP_IDS,
        "world_size": world,
        "assets": str(negatives.resolve()) if rows else None,
        "assets_sha256": sha(negatives) if rows else None,
        "record_count": len(rows),
        "global_replays": 16,
        "calibration_mode": "shared initialization, negative-only",
        "reference_loss": "T0 JSD" if method == "OPSD_UL" else "T1 JSD+EOS",
        "source_smoke_sha256": sha(smoke / "LAUNCH.json"),
        "initial_lora_sha256": sha(initial),
        "measurements": {
            "base_logit_l2": math.sqrt(trainer.base_sq),
            "stop_logit_l2": math.sqrt(trainer.stop_sq),
            "reference_lora_l2": reference_norm,
            "ul_lora_l2": ul_norm,
        },
    }
    (output / "auxiliary.json").write_text(json.dumps(auxiliary, indent=2) + "\n")
    print(
        json.dumps(
            {key: auxiliary[key] for key in ("method", "lambda_eos", "lambda_ul", "record_count")}
        )
    )


def main():
    """Require fresh output to preserve every measured attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--negative-library", type=Path, help="Required for arms containing UL.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "OPSD_SC",
            "OPSD_UL",
            "TA_OPSD",
            "TA_OPSD_trained_negatives",
            "TA_OPSD_semantic_negatives",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.smoke_run, args.negative_library, args.model_path, args.method, args.output_dir)


if __name__ == "__main__":
    main()
