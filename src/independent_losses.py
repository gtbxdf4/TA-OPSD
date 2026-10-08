"""SC and UL losses with a position mean followed by a response mean."""

import torch
import torch.nn.functional as F


def response_mean(values, mask):
    """Give each nonempty response equal weight, regardless of its length."""
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("expected matching [response, position] tensors")
    counts = mask.sum(-1)
    means = values.masked_fill(~mask.bool(), 0).sum(-1) / counts.clamp_min(1)
    present = counts > 0
    # Empty rows contribute neither loss nor weight to the response average.
    return means.sum() / present.sum().clamp_min(1)


def _binary_log_probabilities(logits, stop_ids):
    """Collapse vocabulary logits into log probabilities for STOP and CONTINUE."""
    if not stop_ids or len(set(stop_ids)) != len(stop_ids):
        raise ValueError("stop IDs must be nonempty and unique")
    vocab = logits.shape[-1]
    if min(stop_ids) < 0 or max(stop_ids) >= vocab or len(stop_ids) == vocab:
        raise ValueError("stop set must be a proper subset of the vocabulary")
    x = logits.float()
    is_continue = torch.ones(vocab, dtype=torch.bool, device=x.device)
    is_continue[list(stop_ids)] = False
    # Log-sum-exp aggregates probability mass without subtracting p(STOP) from one.
    binary_logits = torch.stack(
        (torch.logsumexp(x[..., list(stop_ids)], -1), torch.logsumexp(x[..., is_continue], -1)),
        dim=-1,
    )
    return F.log_softmax(binary_logits, dim=-1)


def stop_kl(student_logits, teacher_logits, valid_mask, stop_ids):
    """Teacher-to-student binary KL at the valid response positions (Eqs. 2–4)."""
    # The frozen teacher contributes targets, not gradients.
    teacher = _binary_log_probabilities(teacher_logits.detach(), stop_ids)
    student = _binary_log_probabilities(student_logits, stop_ids)
    positions = (teacher.exp() * (teacher - student)).sum(-1)
    return response_mean(positions, valid_mask.bool())


def local_unlikelihood(logits, token_ids, selected, stop_ids):
    """Penalize selected suffix tokens under their failure prefixes (Eq. 5)."""
    if logits.shape[:-1] != token_ids.shape or selected.shape != token_ids.shape:
        raise ValueError("logits and suffix token/mask alignment mismatch")
    valid = selected.bool() & (token_ids >= 0)
    # UL targets repeated content; stopping tokens never receive this penalty.
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
