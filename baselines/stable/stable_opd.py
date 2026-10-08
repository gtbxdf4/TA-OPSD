#!/usr/bin/env python3
"""Narrow Stable-OPD adapter for the matched non-thinking T-series runs.

This is an explicit adaptation of arXiv:2604.08527v1 to the matched T-series:
the frozen teacher is the same initial Qwen+LoRA policy, but receives the
ground-truth solution as privileged context.  It is not the repository's JSD
objective and it does not use the paper's original 33k SFT warm start.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

EXPECTED_SOURCE_COMMIT = "ae7d2519e94920c4eb6206c0c26de46d9c50abae"
EXPECTED_SOURCE_HASHES = {
    "opsd_train.py": "524edd3f9a5ee9ad1496410dd9c51019301e8e1ab72289f5982b34adaf81f26e",
    "opsd_trainer.py": "aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d",
    "data_collator.py": "5b96f3b3ae2f04e2b9dbf03509d3cd723c1b637d9362ab4697135d9f885b8381",
}
SPEC_FIELDS = {
    "base_model",
    "init_manifest",
    "data_files",
    "output",
    "source_dir",
    "size",
    "dp",
    "mode",
}


@dataclass(frozen=True)
class Protocol:
    paper: str = "arXiv:2604.08527v1"
    adaptation: str = (
        "frozen shared initial Qwen+LoRA teacher with privileged ground-truth "
        "context; initial policy reference with unprivileged context"
    )
    lambda_gold: float = 0.1
    beta_ref_kl: float = 0.01
    ppo_epsilon: float = 1.0
    group_size: int = 4
    distinct_prompts_per_update: int = 32
    rollouts_per_update: int = 128
    golden_per_update: int = 32
    max_steps: int = 100
    smoke_updates: int = 2
    save_steps: tuple[int, ...] = (25, 50, 75, 100)
    learning_rate: float = 5e-6
    warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw_torch_fused"
    max_grad_norm: float = 0.1
    seed: int = 42
    max_completion_length: int = 1024
    max_length: int = 20000
    temperature: float = 1.1
    top_p: float = 0.95
    top_k: int = 20
    student_thinking: bool = False
    teacher_thinking: bool = False
    stop_token_ids: tuple[int, ...] = (151643, 151645)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    disable_dropout: bool = True
    effective_lora_dropout: float = 0.0
    checkpoint_use_reentrant: bool = False
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class RunSpec:
    base_model: str
    init_manifest: str
    data_files: tuple[str, ...]
    output: str
    source_dir: str
    size: str
    dp: int
    mode: str
    protocol: Protocol = Protocol()

    @property
    def per_device_prompts(self) -> int:
        return 1

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.protocol.distinct_prompts_per_update // self.dp

    @property
    def distinct_prompts_per_update(self) -> int:
        return self.protocol.distinct_prompts_per_update

    @property
    def rollouts_per_update(self) -> int:
        return self.protocol.rollouts_per_update

    @property
    def golden_per_update(self) -> int:
        return self.protocol.golden_per_update

    @property
    def max_steps(self) -> int:
        return self.protocol.max_steps

    @property
    def stop_after_updates(self) -> int | None:
        return self.protocol.smoke_updates if self.mode == "smoke" else None


def load_run_spec(raw_json: str) -> RunSpec:
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--spec must be one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--spec must decode to one JSON object")
    missing = SPEC_FIELDS - set(value)
    extra = set(value) - SPEC_FIELDS
    if missing:
        raise ValueError("missing spec fields: " + ", ".join(sorted(missing)))
    if extra:
        raise ValueError("unexpected spec fields: " + ", ".join(sorted(extra)))
    if value["size"] not in ("1.7B", "4B", "8B"):
        raise ValueError("size must be 1.7B, 4B, or 8B")
    expected_dp = 8 if value["size"] == "8B" else 4
    if value["dp"] != expected_dp:
        raise ValueError(f"size {value['size']} requires dp={expected_dp}")
    if value["mode"] not in ("smoke", "formal"):
        raise ValueError("mode must be smoke or formal")
    for key in ("base_model", "init_manifest", "output", "source_dir"):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError(f"{key} must be a nonempty absolute path")
    files = value["data_files"]
    if isinstance(files, str):
        files = [files]
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in files)
    ):
        raise ValueError("data_files must be a nonempty absolute-path string or list")
    value["data_files"] = tuple(files)
    return RunSpec(**value)


def masked_response_mean(values, mask):
    """Paper/GRPO normalization: token mean per response, then batch mean."""
    import torch

    mask = mask.to(dtype=values.dtype)
    counts = mask.sum(dim=-1)
    if bool(torch.any(counts == 0).item()):
        raise ValueError("every rollout must contain at least one scored token")
    return ((values * mask).sum(dim=-1) / counts).mean()


def stable_opd_objective(
    current_logp,
    old_logp,
    teacher_logp,
    rollout_mask,
    *,
    golden_sft_loss,
    reference_kl,
    epsilon=1.0,
    lambda_gold=1.0,
    beta_ref_kl=1.0,
):
    """Real Stable-OPD tensor objective; teacher and rollout policy are frozen."""
    advantage = (teacher_logp - old_logp).detach()
    ratio = (current_logp - old_logp.detach()).exp()
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    surrogate = torch_min(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -masked_response_mean(surrogate, rollout_mask)
    loss = policy_loss + lambda_gold * golden_sft_loss + beta_ref_kl * reference_kl
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "golden_sft_loss": golden_sft_loss,
        "reference_kl": reference_kl,
        "advantage": masked_response_mean(advantage, rollout_mask),
        "ratio": masked_response_mean(ratio, rollout_mask),
    }


def torch_min(left, right):
    import torch

    return torch.minimum(left, right)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_manifest(state) -> dict:
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
    b_tensors = [value for name, value in state.items() if "lora_B" in name]
    return {
        "tensor_sha256": digest.hexdigest(),
        "tensors": rows,
        "tensor_count": len(rows),
        "B_all_zero": bool(b_tensors)
        and all(torch.count_nonzero(value).item() == 0 for value in b_tensors),
    }


def read_shared_initialization(path: str):
    from safetensors.torch import load_file

    manifest = json.loads(Path(path).read_text())
    weights = (
        (Path(path).parent / manifest["weights"])
        if not Path(manifest["weights"]).is_absolute()
        else Path(manifest["weights"])
    )
    if manifest.get("seed") != 42:
        raise ValueError("shared initialization must be seed 42")
    if sha256_file(weights) != manifest.get("file_sha256"):
        raise ValueError("shared initialization file hash mismatch")
    state = load_file(str(weights), device="cpu")
    observed = tensor_manifest(state)
    if observed != manifest.get("tensor_manifest") or not observed["B_all_zero"]:
        raise ValueError("shared initialization tensor identity mismatch")
    return state, manifest


def _snapshot_rng():
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def add_exact_reference_adapter(model, state):
    """PEFT add_adapter inherits BF16 base dtype; retain the FP32 frozen LoRA."""
    from peft import set_peft_model_state_dict

    dtypes = {tensor.dtype for tensor in state.values()}
    if len(dtypes) != 1:
        raise ValueError("mixed shared LoRA dtypes are unsupported")
    dtype = next(iter(dtypes))
    model.add_adapter("stable_reference", model.peft_config["default"])
    for name, parameter in model.named_parameters():
        if ".stable_reference." in name and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
    set_peft_model_state_dict(model, state, adapter_name="stable_reference")


def install_exact_initialization_hook(manifest_path: str, evidence_dir: str):
    """Install and attest trainable plus independently frozen initial adapters."""
    import trl.trainer.sft_trainer as sft_module
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state, manifest = read_shared_initialization(manifest_path)
    original = sft_module.get_peft_model
    if getattr(original, "_stable_opd_exact_init", False):
        raise ValueError("duplicate Stable-OPD initialization hook")

    def initialized_model(*args, **kwargs):
        model = original(*args, **kwargs)
        current = get_peft_model_state_dict(model, adapter_name="default")
        if set(current) != set(state) or not state:
            raise ValueError("shared adapter key set mismatch")
        for name, value in state.items():
            if current[name].shape != value.shape or current[name].dtype != value.dtype:
                raise ValueError("shared adapter shape/dtype mismatch: " + name)
        before = _snapshot_rng()
        set_peft_model_state_dict(model, state, adapter_name="default")
        add_exact_reference_adapter(model, state)
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".stable_reference." in name:
                parameter.requires_grad_(False)
        _restore_rng(before)
        actual = get_peft_model_state_dict(model, adapter_name="default")
        frozen = get_peft_model_state_dict(model, adapter_name="stable_reference")
        if tensor_manifest(actual) != manifest["tensor_manifest"]:
            raise ValueError("trainable adapter did not reproduce shared tensors")
        if tensor_manifest(frozen) != manifest["tensor_manifest"]:
            raise ValueError("frozen teacher/reference adapter did not reproduce shared tensors")
        evidence = Path(evidence_dir)
        evidence.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        receipt = {
            "PASS": True,
            "rank": rank,
            "seed": 42,
            "manifest": manifest_path,
            "weights": manifest["weights"],
            "file_sha256": manifest["file_sha256"],
            "tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
            "B_all_zero": True,
            "trainable_and_frozen_copy_exact": True,
            "adapter_install_preserved_rng": True,
        }
        (evidence / f"INIT_LOAD-rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return model

    initialized_model._stable_opd_exact_init = True
    sft_module.get_peft_model = initialized_model
    return manifest


def _freeze_reference_adapter(model):
    for name, parameter in model.named_parameters():
        if ".stable_reference." in name:
            parameter.requires_grad_(False)


@contextmanager
def frozen_initial_adapter(trainer, model):
    unwrapped = trainer.accelerator.unwrap_model(model)
    previous = unwrapped.active_adapter
    unwrapped.set_adapter("stable_reference")
    _freeze_reference_adapter(unwrapped)
    try:
        yield
    finally:
        unwrapped.set_adapter(previous)
        _freeze_reference_adapter(unwrapped)


def _completion_logits(model, input_ids, attention_mask, completion_length):
    """Return only positions predicting completion tokens when supported."""
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=completion_length + 1,
        )
        logits = outputs.logits[:, -(completion_length + 1) : -1, :]
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -(completion_length + 1) : -1, :]
    if logits.shape[1] != completion_length:
        raise ValueError("model returned an unexpected completion-logit shape")
    return logits


def _selected_statistics(logits, token_ids, eos_token_id, *, entropy=False, temperature=1.0):
    import torch

    if temperature <= 0:
        raise ValueError("policy temperature must be positive")
    from torch.utils.checkpoint import checkpoint

    stop_ids = (eos_token_id,) if isinstance(eos_token_id, int) else tuple(eos_token_id)
    stop_index = torch.tensor(stop_ids, device=logits.device, dtype=torch.long)

    def selected_block(block, ids):
        scaled = block / temperature
        # Keep separate casts as in the original BF16 autograd graph: combining
        # them changes the order of low-precision gradient accumulation.
        log_z = torch.logsumexp(scaled.float(), dim=-1)
        chosen = scaled.float().gather(-1, ids.unsqueeze(-1)).squeeze(-1) - log_z
        stop = torch.logsumexp(scaled.float().index_select(-1, stop_index), dim=-1) - log_z
        return chosen, stop

    chosen_parts, stop_parts, top_parts, entropy_parts = [], [], [], []
    for start in range(0, logits.shape[1], 16):
        block = logits[:, start : start + 16, :]
        ids = token_ids[:, start : start + 16]
        if torch.is_grad_enabled() and block.requires_grad:
            chosen, stop = checkpoint(
                selected_block, block, ids, use_reentrant=False, preserve_rng_state=False
            )
        else:
            chosen, stop = selected_block(block, ids)
        chosen_parts.append(chosen)
        stop_parts.append(stop)
        # Preserve the original BF16 scaling/rounding before FP32 statistics.
        # Never cast or retain the entire sequence x vocabulary at once.
        with torch.no_grad():
            scaled = block.detach() / temperature
            top_parts.append(scaled.argmax(dim=-1))
            if entropy:
                chunk = scaled.float()
                chunk_log_z = torch.logsumexp(chunk, dim=-1, keepdim=True)
                logp = chunk - chunk_log_z
                entropy_parts.append(-(logp.exp() * logp).sum(dim=-1))
    return (
        torch.cat(chosen_parts, dim=1),
        torch.cat(stop_parts, dim=1),
        torch.cat(top_parts, dim=1),
        torch.cat(entropy_parts, dim=1) if entropy else None,
    )


def exact_forward_kl(student_logits, reference_logits, mask, temperature, chunk_size=2048):
    """Exact full-vocabulary KL(student || frozen reference), memory bounded.

    The custom backward recomputes vocabulary chunks and returns the analytic
    gradient.  This keeps Eq. 6 exact without retaining additional full-vocab
    softmax/log-softmax tensors in the forward graph.
    """
    import torch

    if temperature <= 0:
        raise ValueError("KL temperature must be positive")
    if student_logits.shape != reference_logits.shape:
        raise ValueError("student/reference logit shapes differ")
    if student_logits.shape[:-1] != mask.shape:
        raise ValueError("KL mask shape differs from prefix-state logits")

    class ExactForwardKL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, student, reference, token_mask, tau, width):
            work_dtype = (
                torch.float32 if student.dtype in (torch.float16, torch.bfloat16) else student.dtype
            )
            student_log_z = None
            reference_log_z = None
            for start in range(0, student.shape[-1], width):
                end = min(start + width, student.shape[-1])
                student_piece = torch.logsumexp(
                    student[..., start:end].to(work_dtype) / tau, dim=-1
                )
                reference_piece = torch.logsumexp(
                    reference[..., start:end].to(work_dtype) / tau, dim=-1
                )
                student_log_z = (
                    student_piece
                    if student_log_z is None
                    else torch.logaddexp(student_log_z, student_piece)
                )
                reference_log_z = (
                    reference_piece
                    if reference_log_z is None
                    else torch.logaddexp(reference_log_z, reference_piece)
                )
            per_token = torch.zeros_like(student_log_z)
            for start in range(0, student.shape[-1], width):
                end = min(start + width, student.shape[-1])
                logp = student[..., start:end].to(work_dtype) / tau
                logp = logp - student_log_z.unsqueeze(-1)
                logq = reference[..., start:end].to(work_dtype) / tau
                logq = logq - reference_log_z.unsqueeze(-1)
                per_token.add_((logp.exp() * (logp - logq)).sum(dim=-1))
            numeric_mask = token_mask.to(work_dtype)
            counts = numeric_mask.sum(dim=-1)
            if bool(torch.any(counts == 0).item()):
                raise ValueError("every rollout must contain a KL-scored token")
            weights = numeric_mask / counts.unsqueeze(-1) / token_mask.shape[0]
            value = (per_token * weights).sum()
            ctx.save_for_backward(
                student, reference, student_log_z, reference_log_z, per_token, weights
            )
            ctx.temperature = tau
            ctx.chunk_size = width
            return value

        @staticmethod
        def backward(ctx, grad_output):
            student, reference, student_log_z, reference_log_z, per_token, weights = (
                ctx.saved_tensors
            )
            tau = ctx.temperature
            width = ctx.chunk_size
            work_dtype = student_log_z.dtype
            grad = torch.empty_like(student)
            for start in range(0, student.shape[-1], width):
                end = min(start + width, student.shape[-1])
                logp = student[..., start:end].to(work_dtype) / tau
                logp = logp - student_log_z.unsqueeze(-1)
                logq = reference[..., start:end].to(work_dtype) / tau
                logq = logq - reference_log_z.unsqueeze(-1)
                chunk_grad = logp.exp() * (logp - logq - per_token.unsqueeze(-1))
                chunk_grad.mul_(weights.unsqueeze(-1) / tau)
                grad[..., start:end] = (chunk_grad * grad_output).to(student.dtype)
            return grad, None, None, None, None

    return ExactForwardKL.apply(
        student_logits, reference_logits.detach(), mask, float(temperature), int(chunk_size)
    )


def paper_repetition_metrics(tokenizer, token_ids, mask):
    """Paper v1 zlib-tail metric (L=10000 chars, threshold=10)."""
    ratios = []
    repeated = []
    for row, keep in zip(token_ids.detach().cpu(), mask.detach().cpu()):
        ids = row[keep].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        tail = text[-10000:]
        raw = tail.encode("utf-8")
        compressed = zlib.compress(raw)
        ratio = len(raw) / max(1, len(compressed))
        ratios.append(ratio)
        repeated.append(float(len(text) > 10000 and ratio > 10.0))
    return {
        "repetition_rate": sum(repeated) / max(1, len(repeated)),
        "compression_ratio": sum(ratios) / max(1, len(ratios)),
    }


def build_data_collator(source_dir: str, tokenizer, protocol: Protocol):
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from data_collator import SelfDistillationDataCollator

    class StableOPDCollator(SelfDistillationDataCollator):
        def __init__(self):
            super().__init__(
                tokenizer=tokenizer,
                max_length=protocol.max_length,
                reason_first=False,
                student_thinking=False,
                teacher_thinking=False,
            )

        def __call__(self, features):
            import torch

            expanded = [feature for feature in features for _ in range(protocol.group_size)]
            batch = super().__call__(expanded)
            full_ids = []
            prompt_lengths = []
            for feature in features:
                user = {
                    "role": "user",
                    "content": (
                        f"Problem: {feature['problem']}\n\nPlease reason step by step, "
                        "and put your final answer within \\boxed{}."
                    ),
                }
                assistant = {"role": "assistant", "content": feature["solution"]}
                prompt = tokenizer.apply_chat_template(
                    [user],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                full = tokenizer.apply_chat_template(
                    [user, assistant],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )[: protocol.max_length]
                full_ids.append(full)
                prompt_lengths.append(min(len(prompt), len(full)))
            width = max(len(ids) for ids in full_ids)
            pad_id = tokenizer.pad_token_id
            ids_tensor = torch.full((len(full_ids), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(ids_tensor)
            labels = torch.full_like(ids_tensor, -100)
            for index, (ids, prompt_length) in enumerate(zip(full_ids, prompt_lengths)):
                ids_tensor[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[index, : len(ids)] = 1
                labels[index, prompt_length : len(ids)] = torch.tensor(
                    ids[prompt_length:], dtype=torch.long
                )
            if not bool((labels != -100).any(dim=1).all().item()):
                raise ValueError("a golden solution was empty after tokenization/truncation")
            batch.update(
                gold_input_ids=ids_tensor,
                gold_attention_mask=attention,
                gold_labels=labels,
            )
            return batch

    return StableOPDCollator()


def build_trainer_class(source_dir: str, protocol: Protocol):
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from opsd_trainer import OPSDTrainer

    class StableOPDTrainer(OPSDTrainer):
        _name = "Stable-OPD-matched-T"

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

            prompt_length = int(inputs["student_prompt_length"])
            sampled_ids = inputs["student_input_ids"][:, prompt_length:]
            mask = inputs["labels"][:, prompt_length:] != -100
            completion_length = sampled_ids.shape[1]

            student_logits = _completion_logits(
                model,
                inputs["student_input_ids"],
                inputs["student_attention_mask"],
                completion_length,
            )
            current_logp, student_eos, student_top1, student_entropy = _selected_statistics(
                student_logits,
                sampled_ids,
                protocol.stop_token_ids,
                entropy=True,
                temperature=protocol.temperature,
            )
            # One fresh rollout batch is consumed by one optimizer update.  The
            # rollout-policy score is therefore the detached pre-update score.
            old_logp = current_logp.detach()

            with torch.no_grad(), frozen_initial_adapter(self, model):
                teacher_logits = _completion_logits(
                    model,
                    inputs["teacher_input_ids"],
                    inputs["teacher_attention_mask"],
                    completion_length,
                )
                teacher_logp, teacher_eos, teacher_top1, _ = _selected_statistics(
                    teacher_logits,
                    sampled_ids,
                    protocol.stop_token_ids,
                    temperature=protocol.temperature,
                )
                del teacher_logits
                reference_logits = _completion_logits(
                    model,
                    inputs["student_input_ids"],
                    inputs["student_attention_mask"],
                    completion_length,
                )

            reference_kl = exact_forward_kl(
                student_logits,
                reference_logits,
                mask,
                protocol.temperature,
            )
            del reference_logits

            gold_outputs = model(
                input_ids=inputs["gold_input_ids"],
                attention_mask=inputs["gold_attention_mask"],
                labels=inputs["gold_labels"],
            )
            golden_sft_loss = gold_outputs.loss
            del gold_outputs

            terms = stable_opd_objective(
                current_logp,
                old_logp,
                teacher_logp,
                mask,
                golden_sft_loss=golden_sft_loss,
                reference_kl=reference_kl,
                epsilon=protocol.ppo_epsilon,
                lambda_gold=protocol.lambda_gold,
                beta_ref_kl=protocol.beta_ref_kl,
            )

            with torch.no_grad():
                student_mean = masked_response_mean(current_logp.detach(), mask)
                teacher_mean = masked_response_mean(teacher_logp.detach(), mask)
                student_eos_mean = masked_response_mean(student_eos.detach(), mask)
                teacher_eos_mean = masked_response_mean(teacher_eos.detach(), mask)
                entropy_mean = masked_response_mean(student_entropy, mask)
                overlap = masked_response_mean(
                    (student_top1 == teacher_top1).to(dtype=torch.float32), mask
                )
                lengths = mask.sum(dim=-1).float()
                stop_ids = torch.tensor(
                    protocol.stop_token_ids, device=sampled_ids.device, dtype=sampled_ids.dtype
                )
                emitted_eos = (sampled_ids.unsqueeze(-1) == stop_ids).any(dim=-1).any(dim=-1)
                cap_hit = (
                    ((~emitted_eos) & (completion_length >= protocol.max_completion_length))
                    .float()
                    .mean()
                )
                repetition = paper_repetition_metrics(self.processing_class, sampled_ids, mask)
                metrics = {
                    "loss/total": terms["loss"],
                    "loss/policy": terms["policy_loss"],
                    "loss/golden_sft": golden_sft_loss,
                    "loss/reference_kl_weighted": protocol.beta_ref_kl * reference_kl,
                    "policy/reference_kl": reference_kl,
                    "policy/ratio": terms["ratio"],
                    "logp/student_sampled": student_mean,
                    "logp/teacher_sampled": teacher_mean,
                    "logp/teacher_minus_student": teacher_mean - student_mean,
                    "eos_logp/student": student_eos_mean,
                    "eos_logp/teacher": teacher_eos_mean,
                    "eos_logp/teacher_minus_student": teacher_eos_mean - student_eos_mean,
                    "policy/student_entropy": entropy_mean,
                    "policy/teacher_student_top1_overlap": overlap,
                    "rollout/response_length": lengths.mean(),
                    "rollout/cap_hit_rate": cap_hit,
                    "rollout/eos_rate": emitted_eos.float().mean(),
                    "rollout/repetition_rate": repetition["repetition_rate"],
                    "rollout/compression_ratio": repetition["compression_ratio"],
                }
                for name, value in metrics.items():
                    number = (
                        float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                    )
                    self._metrics["train"][name].append(number)

            if return_outputs:

                class MinimalOutput:
                    pass

                output = MinimalOutput()
                output.loss = terms["loss"]
                return terms["loss"], output
            return terms["loss"]

        def log(self, logs, start_time=None):
            logs = dict(logs)
            logs["actual_optimizer_updates"] = int(self.state.global_step)
            logs["protocol_max_updates"] = protocol.max_steps
            return super().log(logs, start_time)

    return StableOPDTrainer


def _load_ordered_dataset(paths):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset

    tables = [pq.ParquetFile(path).read().replace_schema_metadata(None) for path in paths]
    dataset = Dataset(pa.concat_tables(tables))
    required = {"problem", "solution"}
    if not required.issubset(dataset.column_names):
        raise ValueError("training data must contain problem and solution columns")
    return dataset


def validate_local_inputs(spec: RunSpec):
    for path in (spec.base_model, spec.init_manifest, spec.source_dir, *spec.data_files):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        path = Path(spec.source_dir) / name
        if sha256_file(path) != expected:
            raise ValueError(f"official source hash mismatch: {name}")
    commit = (Path(spec.source_dir) / "SOURCE_COMMIT").read_text().strip()
    if commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError("official source commit mismatch")
    manifest = json.loads(Path(spec.init_manifest).read_text())
    peft = manifest.get("peft_config", {})
    expected_peft = {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": set(Protocol().lora_targets),
    }
    for key in ("r", "lora_alpha", "lora_dropout"):
        if key in peft and peft[key] != expected_peft[key]:
            raise ValueError(f"initialization manifest {key} mismatch")
    if "target_modules" in peft and set(peft["target_modules"]) != expected_peft["target_modules"]:
        raise ValueError("initialization manifest target_modules mismatch")


def protocol_receipt(spec: RunSpec, manifest: dict) -> dict:
    source_hashes = {
        name: sha256_file(Path(spec.source_dir) / name) for name in EXPECTED_SOURCE_HASHES
    }
    data_hashes = {path: sha256_file(path) for path in spec.data_files}
    code_dir = Path(__file__).resolve().parent
    adapter_hashes = {
        name: sha256_file(code_dir / name) for name in ("stable_opd.py", "train_stable_opd.py")
    }
    return {
        "method": "Stable-OPD",
        "paper": "arXiv:2604.08527v1",
        "not_veto_paper": "not arXiv:2601.07155",
        "not_opsd_jsd": True,
        "reference_divergence": "exact full-vocabulary D_KL(current_policy || frozen_initial_policy) on visited prefixes",
        "microbatch_plan": {
            "per_device_distinct_prompts": spec.per_device_prompts,
            "gradient_accumulation_steps": spec.gradient_accumulation_steps,
            "global_distinct_prompts_per_update": spec.distinct_prompts_per_update,
            "group_size": spec.protocol.group_size,
            "global_rollouts_per_update": spec.rollouts_per_update,
            "global_golden_per_update": spec.golden_per_update,
        },
        "public_starting_point_alignment": "common untrained base+exact shared LoRA; no 33k SFT warm start",
        "user_selected_first_configuration": {
            "lambda_gold": 0.1,
            "beta_ref_kl": 0.01,
            "ppo_epsilon": 1.0,
        },
        "run_spec": {**asdict(spec), "protocol": asdict(spec.protocol)},
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_hashes": source_hashes,
        "adapter_hashes": adapter_hashes,
        "data_hashes": data_hashes,
        "initialization": {
            "manifest": spec.init_manifest,
            "weights": manifest["weights"],
            "file_sha256": manifest["file_sha256"],
            "tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
        },
        "created": time.time(),
    }


def run_training(spec: RunSpec):
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from trl.experimental.gold import GOLDConfig

    validate_local_inputs(spec)
    output = Path(spec.output)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0 and output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    set_seed(spec.protocol.seed)

    evidence = output / "evidence"
    manifest = install_exact_initialization_hook(spec.init_manifest, str(evidence))
    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = _load_ordered_dataset(spec.data_files)
    collator = build_data_collator(spec.source_dir, tokenizer, spec.protocol)
    TrainerClass = build_trainer_class(spec.source_dir, spec.protocol)

    args = GOLDConfig(
        output_dir=spec.output,
        per_device_train_batch_size=spec.per_device_prompts,
        gradient_accumulation_steps=spec.gradient_accumulation_steps,
        learning_rate=spec.protocol.learning_rate,
        lr_scheduler_type=spec.protocol.lr_scheduler_type,
        warmup_steps=spec.protocol.warmup_steps,
        max_steps=spec.protocol.max_steps,
        max_grad_norm=spec.protocol.max_grad_norm,
        optim=spec.protocol.optimizer,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": spec.protocol.checkpoint_use_reentrant},
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=None,
        report_to="none",
        seed=spec.protocol.seed,
        data_seed=spec.protocol.seed,
        remove_unused_columns=False,
        max_length=spec.protocol.max_length,
        max_completion_length=spec.protocol.max_completion_length,
        temperature=spec.protocol.temperature,
        top_p=spec.protocol.top_p,
        top_k=spec.protocol.top_k,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.6,
        vllm_tensor_parallel_size=1,
        beta=0.0,
        lmbda=1.0,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
            "trust_remote_code": True,
        },
    )
    peft_config = LoraConfig(
        r=spec.protocol.lora_r,
        lora_alpha=spec.protocol.lora_alpha,
        lora_dropout=spec.protocol.lora_dropout,
        target_modules=list(spec.protocol.lora_targets),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    callbacks = []
    if spec.stop_after_updates is not None:
        stop = spec.stop_after_updates

        class StopAfterSmoke(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop:
                    control.should_training_stop = True
                return control

        callbacks.append(StopAfterSmoke())

    trainer = TrainerClass(
        model=spec.base_model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
        use_thinking_machines_loss=False,
        fixed_teacher=False,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    if trainer.accelerator.num_processes != spec.dp:
        raise ValueError(
            f"spec dp={spec.dp}, but accelerator world size is {trainer.accelerator.num_processes}"
        )
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        (output / "checkpoint0").mkdir(parents=True, exist_ok=False)
        (output / "checkpoint0" / "STABLE_OPD_PROTOCOL.json").write_text(
            json.dumps(protocol_receipt(spec, manifest), indent=2, ensure_ascii=False) + "\n"
        )
    trainer.accelerator.wait_for_everyone()

    trainer.train()
    trainer.save_model(spec.output)
    trainer.accelerator.wait_for_everyone()

    actual_updates = int(trainer.state.global_step)
    expected_updates = spec.stop_after_updates or spec.max_steps
    if actual_updates != expected_updates:
        raise ValueError(
            f"expected {expected_updates} optimizer updates, observed {actual_updates}"
        )
    required = [output / "adapter_model.safetensors"]
    if spec.mode == "formal":
        required.extend(
            output / f"checkpoint-{step}" / "adapter_model.safetensors"
            for step in spec.protocol.save_steps
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stable-OPD artifacts: " + ", ".join(missing))
    if trainer.is_world_process_zero():
        done = {
            "PASS": True,
            "method": "Stable-OPD",
            "mode": spec.mode,
            "actual_optimizer_updates": actual_updates,
            "formal_scheduler_horizon": spec.max_steps,
            "artifacts": [str(path) for path in required],
            "completed": time.time(),
            "scientific_efficacy_claimed": False,
        }
        (output / "DONE.json").write_text(json.dumps(done, indent=2, ensure_ascii=False) + "\n")
    trainer.accelerator.wait_for_everyone()
