"""Seed before model construction; load and attest one frozen untrained adapter.

No loss, optimizer, data, generation or teacher modifications live in this module.
"""

import hashlib
import json
import os
import time
from pathlib import Path


def tensor_manifest(state):
    """Hash sorted tensor names, shapes, dtypes and contiguous CPU bytes.

    Return per-tensor hashes, their aggregate identity, and whether a nonempty
    collection of LoRA-B tensors is entirely zero. Dtype is part of the identity.
    """
    import torch

    rows = []
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        row = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        rows.append(row)
    b = [t for n, t in state.items() if "lora_B" in n]
    return {
        "tensor_sha256": digest.hexdigest(),
        "tensors": rows,
        "tensor_count": len(rows),
        "B_all_zero": bool(b) and all(torch.count_nonzero(t).item() == 0 for t in b),
    }


def install_state(model, state):
    """Install exact shared LoRA tensors after validating keys, dtype and shape.

    Reject non-LoRA keys and nonzero initial B matrices, then compare the loaded
    adapter manifest byte for byte. Return the same model; do not construct a new
    adapter or alter its trainable parameter selection.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    current = get_peft_model_state_dict(model)
    if set(current) != set(state) or not state:
        raise ValueError("shared adapter key set mismatch")
    for name, value in state.items():
        if (
            current[name].shape != value.shape
            or current[name].dtype != value.dtype
            or not ("lora_A" in name or "lora_B" in name)
        ):
            raise ValueError("shared adapter shape/dtype/name mismatch: " + name)
    expected = tensor_manifest(state)
    if not expected["B_all_zero"]:
        raise ValueError("shared initialization is not an untrained zero-B adapter")
    set_peft_model_state_dict(model, state)
    if tensor_manifest(get_peft_model_state_dict(model)) != expected:
        raise ValueError("shared adapter load did not reproduce every frozen tensor")
    return model


def read_shared(path):
    """Load CPU safetensors only after verifying seed, file and tensor identity.

    Relative weight paths are resolved beside the JSON manifest. DP defaults to
    four when omitted and must be four or eight. Return (state, manifest); reject
    seed, file or tensor mismatches and missing zero-B initialization proof.
    """
    from safetensors.torch import load_file

    manifest = json.loads(Path(path).read_text())
    world_size = manifest.setdefault("world_size", 4)
    if type(world_size) is not int or world_size not in (4, 8):
        raise ValueError("shared initialization world_size must be 4 or 8")
    weights = Path(manifest["weights"])
    if not weights.is_absolute():
        weights = Path(path).parent / weights
    if (
        manifest["seed"] != 42
        or hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["file_sha256"]
    ):
        raise ValueError("shared initialization seed/file identity mismatch")
    state = load_file(str(weights), device="cpu")
    if (
        tensor_manifest(state) != manifest["tensor_manifest"]
        or not manifest["tensor_manifest"]["B_all_zero"]
    ):
        raise ValueError("shared initialization tensor identity mismatch")
    return state, manifest


def runtime_initialization_manifest(state):
    """Frozen accelerate.yaml uses BF16; DeepSpeed casts the entire module."""
    import torch

    return tensor_manifest({name: value.to(dtype=torch.bfloat16) for name, value in state.items()})


def validate_runtime_initialization(actual, shared):
    """Require exact shared tensor identity after the frozen BF16 conversion."""
    observed = tensor_manifest(actual)
    if observed != runtime_initialization_manifest(shared):
        raise ValueError("pre-update rank adapter differs from exact shared BF16 initialization")
    return observed


def install_runtime():
    """Seed model construction and install a process-local TRL PEFT creation hook.

    V18_INITIALIZATION_MANIFEST supplies the frozen adapter and V17_EVIDENCE the
    receipts directory. Call before importing/building the upstream trainer.
    Loading must preserve CPU RNG; installing this hook twice is rejected.
    """
    import trl.trainer.sft_trainer as sft
    from transformers import set_seed

    from launch import atomic

    path = os.environ["V18_INITIALIZATION_MANIFEST"]
    # This function is invoked before the official trainer/model is imported/built.
    set_seed(42)
    state, manifest = read_shared(path)
    original = sft.get_peft_model
    if getattr(original, "_v18_shared", False):
        raise ValueError("duplicate shared-initialization hook")

    def shared_model(*args, **kwargs):
        import torch

        model = original(*args, **kwargs)
        before = torch.get_rng_state().clone()
        install_state(model, state)
        if not torch.equal(before, torch.get_rng_state()):
            raise ValueError("adapter loading changed RNG")
        rank = int(os.environ.get("RANK", "0"))
        atomic(
            Path(os.environ["V17_EVIDENCE"]) / f"INIT_LOAD-rank{rank}-{time.time_ns()}.json",
            {
                "PASS": True,
                "rank": rank,
                "seed_before_model": 42,
                "tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
                "manifest": path,
                "file_sha256": manifest["file_sha256"],
                "B_all_zero": True,
                "loading_preserved_rng": True,
                "resume": os.environ.get("V17_RESUME") or None,
            },
        )
        return model

    shared_model._v18_shared = True
    sft.get_peft_model = shared_model


def verify_before_update(model, global_step):
    """Called before the first loss/backward; resume is separately checkpoint-validated."""
    import pickle
    import random

    import numpy as np
    import torch
    from peft import get_peft_model_state_dict

    from launch import atomic

    rank = int(os.environ.get("RANK", "0"))
    state, manifest = read_shared(os.environ["V18_INITIALIZATION_MANIFEST"])
    actual_state = get_peft_model_state_dict(model)
    actual = tensor_manifest(actual_state)
    expected = runtime_initialization_manifest(state)
    if global_step == 0:
        try:
            actual = validate_runtime_initialization(actual_state, state)
        except ValueError:
            atomic(
                Path(os.environ["V17_EVIDENCE"])
                / f"PRE_UPDATE_FAILED-rank{rank}-{time.time_ns()}.json",
                {
                    "PASS": False,
                    "rank": rank,
                    "actual": actual,
                    "expected_runtime": expected,
                    "frozen_source_tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
                },
            )
            raise
    if global_step > 0 and not os.environ.get("V17_RESUME"):
        raise ValueError("nonzero first update without an explicit same-arm resume")
    rng = {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": hashlib.sha256(
            torch.cuda.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    atomic(
        Path(os.environ["V17_EVIDENCE"])
        / f"PRE_UPDATE-rank{rank}-step{global_step}-{time.time_ns()}.json",
        {
            "PASS": True,
            "rank": rank,
            "global_step": global_step,
            "tensor_manifest": actual,
            "expected_tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
            "expected_runtime_tensor_sha256": expected["tensor_sha256"],
            "runtime_conversion": "DeepSpeed frozen mixed_precision=bf16; exact tensor comparison after cast",
            "rng_sha256": rng,
            "initialization_verified": global_step == 0,
            "resume": os.environ.get("V17_RESUME") or None,
        },
    )
