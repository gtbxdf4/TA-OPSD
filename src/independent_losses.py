"""SC and UL losses with a position mean followed by a response mean."""

import torch
import torch.nn.functional as F


def response_mean(values, mask):
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("expected matching [response, position] tensors")
    counts = mask.sum(-1)
    means = values.masked_fill(~mask.bool(), 0).sum(-1) / counts.clamp_min(1)
    present = counts > 0
    return means.sum() / present.sum().clamp_min(1)


def _binary_log_probabilities(logits, stop_ids):
    if not stop_ids or len(set(stop_ids)) != len(stop_ids):
        raise ValueError("stop IDs must be nonempty and unique")
    vocab = logits.shape[-1]
    if min(stop_ids) < 0 or max(stop_ids) >= vocab or len(stop_ids) == vocab:
        raise ValueError("stop set must be a proper subset of the vocabulary")
    x = logits.float()
    is_continue = torch.ones(vocab, dtype=torch.bool, device=x.device)
    is_continue[list(stop_ids)] = False
    # Sum probability mass, not average token probabilities or mean token logits.
    binary_logits = torch.stack(
        (torch.logsumexp(x[..., list(stop_ids)], -1), torch.logsumexp(x[..., is_continue], -1)),
        dim=-1,
    )
    return F.log_softmax(binary_logits, dim=-1)


def stop_kl(student_logits, teacher_logits, valid_mask, stop_ids):
    # The frozen teacher contributes targets, not gradients.
    teacher = _binary_log_probabilities(teacher_logits.detach(), stop_ids)
    student = _binary_log_probabilities(student_logits, stop_ids)
    positions = (teacher.exp() * (teacher - student)).sum(-1)
    return response_mean(positions, valid_mask.bool())


def local_unlikelihood(logits, token_ids, selected, stop_ids):
    if logits.shape[:-1] != token_ids.shape or selected.shape != token_ids.shape:
        raise ValueError("logits and suffix token/mask alignment mismatch")
    valid = selected.bool() & (token_ids >= 0)
    for token in stop_ids:
        valid = valid & (token_ids != token)
    logp = (
        F.log_softmax(logits.float(), -1)
        .gather(-1, token_ids.clamp_min(0).unsqueeze(-1))
        .squeeze(-1)
    )
    # Keep log(1 - p) finite when p rounds to one in float32.
    p = logp.exp().clamp_max(1 - torch.finfo(torch.float32).eps)
    return response_mean(-torch.log1p(-p), valid)
