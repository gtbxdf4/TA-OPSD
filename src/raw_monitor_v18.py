"""Detached, chunked measurements of existing full-vocabulary distributions."""

import gzip
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def rng_fingerprint():
    values = {
        "python": pickle.dumps(random.getstate()),
        "numpy": pickle.dumps(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().numpy().tobytes(),
    }
    if torch.cuda.is_initialized():
        values["torch_cuda_current_device"] = torch.cuda.get_rng_state().cpu().numpy().tobytes()
    return {k: hashlib.sha256(v).hexdigest() for k, v in values.items()}


def effective_stop_ids(primary, fields):
    extra = fields.get("eos_token_id", [])
    extra = extra if isinstance(extra, (list, tuple)) else [extra]
    return sorted({int(x) for x in [primary, *extra] if x is not None})


def local_outputs(outputs, local_batch_size, tp_size, group_rank):
    if len(outputs) != local_batch_size * tp_size or not 0 <= group_rank < tp_size:
        raise ValueError("generated TP batch cannot be assigned uniquely to local responses")
    return outputs[group_rank * local_batch_size : (group_rank + 1) * local_batch_size]


def align_response(
    raw_ids,
    sampled_ids,
    labels,
    student_prompt_length,
    teacher_prompt_length,
    finish_reason,
    stop_ids,
):
    size = len(raw_ids)
    if not size or list(sampled_ids[:size]) != list(raw_ids) or len(labels) != len(sampled_ids):
        raise ValueError("raw response IDs and official shifted prediction positions disagree")
    return {
        "token_ids": list(raw_ids),
        "token_positions": list(range(size)),
        "student_prediction_positions": list(
            range(student_prompt_length - 1, student_prompt_length - 1 + size)
        ),
        "teacher_prediction_positions": list(
            range(teacher_prompt_length - 1, teacher_prompt_length - 1 + size)
        ),
        "loss_valid_mask": [x != -100 for x in labels[:size]],
        "generated_length": size,
        "valid_loss_token_count": sum(x != -100 for x in labels[:size]),
        "natural_stop": finish_reason == "stop",
        "right_censored": finish_reason == "length",
        "eos_events": [
            {"position": i, "token_id": x} for i, x in enumerate(raw_ids) if x in stop_ids
        ],
    }


def atomic_gzip(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    encoded = gzip.compress(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(),
        compresslevel=1,
        mtime=0,
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publish with no replacement of an existing shard.
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(encoded)


@torch.no_grad()
def token_metrics(
    student_logits, teacher_logits, token_ids, k, stop_ids, temperature=1.0, chunk_tokens=32
):
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("expected aligned [real_response_tokens, full_vocab] logits")
    size, vocab = student_logits.shape
    if size != token_ids.numel() or not 0 < k <= vocab or temperature <= 0 or chunk_tokens <= 0:
        raise ValueError("invalid token count/k/temperature/chunk")
    if (
        not stop_ids
        or len(set(stop_ids)) != len(stop_ids)
        or any(x < 0 or x >= vocab for x in stop_ids)
    ):
        raise ValueError("invalid stop token set")
    simple = [
        "student_entropy_nats",
        "teacher_entropy_nats",
        "topk_intersection_count",
        "topk_overlap_fraction",
        "student_stop_set_logp",
        "teacher_stop_set_logp",
        "stop_set_logp_gap_teacher_minus_student",
        "student_actual_logp",
        "teacher_actual_logp",
        "actual_logp_gap_teacher_minus_student",
    ]
    result = {key: [] for key in simple}
    for key in ("student_eos_logp", "teacher_eos_logp", "eos_logp_gap_teacher_minus_student"):
        result[key] = {str(x): [] for x in stop_ids}
    for start in range(0, size, chunk_tokens):
        end = min(size, start + chunk_tokens)
        # Only the small current block is promoted; no graph or full-rollout FP32 copy.
        s = student_logits[start:end].detach().float() / temperature
        t = teacher_logits[start:end].detach().float() / temperature
        slp, tlp = s.log_softmax(-1), t.log_softmax(-1)
        stop_s, stop_t = slp[:, stop_ids], tlp[:, stop_ids]
        sk, tk = s.topk(k, dim=-1).indices, t.topk(k, dim=-1).indices
        intersection = (sk.unsqueeze(-1) == tk.unsqueeze(-2)).any(-1).sum(-1)
        targets = token_ids[start:end].to(device=s.device, dtype=torch.long).unsqueeze(-1)
        actual_s, actual_t = (
            slp.gather(-1, targets).squeeze(-1),
            tlp.gather(-1, targets).squeeze(-1),
        )
        mass_s, mass_t = stop_s.logsumexp(-1), stop_t.logsumexp(-1)
        values = {
            "student_entropy_nats": -(slp.exp() * slp).sum(-1),
            "teacher_entropy_nats": -(tlp.exp() * tlp).sum(-1),
            "topk_intersection_count": intersection,
            "topk_overlap_fraction": intersection.float() / k,
            "student_stop_set_logp": mass_s,
            "teacher_stop_set_logp": mass_t,
            "stop_set_logp_gap_teacher_minus_student": mass_t - mass_s,
            "student_actual_logp": actual_s,
            "teacher_actual_logp": actual_t,
            "actual_logp_gap_teacher_minus_student": actual_t - actual_s,
        }
        for key, value in values.items():
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite full-vocabulary monitoring value: " + key)
            result[key].extend(value.cpu().tolist())
        for column, token in enumerate(stop_ids):
            result["student_eos_logp"][str(token)].extend(stop_s[:, column].cpu().tolist())
            result["teacher_eos_logp"][str(token)].extend(stop_t[:, column].cpu().tolist())
            result["eos_logp_gap_teacher_minus_student"][str(token)].extend(
                (stop_t[:, column] - stop_s[:, column]).cpu().tolist()
            )
        del s, t, slp, tlp, stop_s, stop_t, sk, tk, values
    return result
