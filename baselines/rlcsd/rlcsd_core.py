"""Numerically testable RLCSD objective from arXiv:2606.11709v2, Eq. 7--13."""

from dataclasses import dataclass
from typing import Any

import torch


def selected_token_log_mass(logits: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor:
    """Log probability mass of ``token_ids`` under every prefix distribution."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    ids = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    if ids.min() < 0 or ids.max() >= logits.shape[-1]:
        raise ValueError("selected token id is outside the logits vocabulary")
    return torch.logsumexp(logits.index_select(-1, ids), dim=-1) - torch.logsumexp(logits, dim=-1)


@dataclass(frozen=True)
class RLCSDConfig:
    epsilon: float = 0.2
    tau: float = 1.3
    beta: float = 1.0
    lam: float = 0.5
    delta: float = 0.02
    eta: float = 0.5
    residual_clip_low: float = -2.0
    residual_clip_high: float = 2.0

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.epsilon < 0 or self.delta < 0 or self.eta < 0:
            raise ValueError("epsilon, delta, and eta must be non-negative")
        if self.residual_clip_low > self.residual_clip_high:
            raise ValueError("residual clip bounds are reversed")


def marginalize_wrong_log_probs(
    teacher_wrong_multi_log_probs: torch.Tensor,
    wrong_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log of the uniform probability mixture across valid wrong hints."""
    if teacher_wrong_multi_log_probs.ndim != 3:
        raise ValueError("teacher_wrong_multi_log_probs must have shape [batch, K, tokens]")
    if wrong_valid_mask.shape != teacher_wrong_multi_log_probs.shape[:2]:
        raise ValueError("wrong_valid_mask must have shape [batch, K]")
    valid = wrong_valid_mask.to(device=teacher_wrong_multi_log_probs.device, dtype=torch.bool)
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("every target must have at least one valid wrong hint")
    masked = teacher_wrong_multi_log_probs.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    # Mix probabilities uniformly across valid hints, not their log probabilities.
    return torch.logsumexp(masked, dim=1) - counts.to(masked.dtype).log().unsqueeze(-1)


def _masked_rollout_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 15: normalize each path within each rollout, then average rollouts."""
    if values.ndim < 2:
        return (values * mask).sum() / mask.sum().clamp(min=1)
    per_rollout = (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_rollout.mean()


def rlcsd_loss(
    old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_correct_log_probs: torch.Tensor,
    teacher_wrong_multi_log_probs: torch.Tensor,
    wrong_valid_mask: torch.Tensor,
    config: RLCSDConfig = RLCSDConfig(),
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Compute the complete verifier-anchored, two-path RLCSD PPO objective.

    Teacher quantities and branch routing are stop-gradient by construction.
    The two branch means are intentionally normalized independently before
    ``nonselected + eta * selected``; replacing this with a global token mean
    is a different objective.
    """
    if (
        old_log_probs.shape != current_log_probs.shape
        or current_log_probs.shape != response_mask.shape
    ):
        raise ValueError("old/current log-probs and response_mask must share shape [batch, tokens]")
    if teacher_correct_log_probs.shape != current_log_probs.shape:
        raise ValueError("teacher_correct_log_probs must match current_log_probs")
    if teacher_wrong_multi_log_probs.shape[0] != current_log_probs.shape[0]:
        raise ValueError("wrong teacher batch does not match student batch")
    if teacher_wrong_multi_log_probs.shape[2] != current_log_probs.shape[1]:
        raise ValueError("wrong teacher token dimension does not match student")

    dtype = current_log_probs.dtype
    mask = response_mask.to(dtype=dtype)
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    if advantages.shape not in (current_log_probs.shape, (current_log_probs.shape[0], 1)):
        raise ValueError("advantages must have shape [batch] or [batch, tokens]")
    advantages = advantages.to(device=current_log_probs.device, dtype=dtype)

    log_ratio = current_log_probs - old_log_probs
    ratio = log_ratio.exp()
    ratio_clamped = ratio.clamp(1.0 - config.epsilon, 1.0 + config.epsilon)

    with torch.no_grad():
        # Stop gradients through teacher scores and the selected-path decision.
        log_marginal_wrong = marginalize_wrong_log_probs(
            teacher_wrong_multi_log_probs, wrong_valid_mask
        )
        contrast = teacher_correct_log_probs - log_marginal_wrong
        residual = (config.beta * config.lam * torch.tanh(contrast / config.tau)).clamp(
            min=config.residual_clip_low, max=config.residual_clip_high
        )
        selected = (residual.abs() > config.delta) & response_mask.bool()
        residual_for_advantage = residual * selected.to(dtype=residual.dtype)
        raw_modulated = advantages + residual_for_advantage
        modulated = torch.where(
            # The teacher residual must not reverse the verifier's advantage sign.
            advantages > 0,
            raw_modulated.clamp_min(0.0),
            torch.where(
                advantages < 0, raw_modulated.clamp_max(0.0), torch.zeros_like(raw_modulated)
            ),
        )

    selected_float = selected.to(dtype=dtype)
    selected_mask = mask * selected_float
    nonselected_mask = mask * (1.0 - selected_float)
    nonselected_per_token = -torch.minimum(ratio * advantages, ratio_clamped * advantages)
    selected_per_token = -torch.minimum(ratio * modulated, ratio_clamped * modulated)
    if rollout_is_weights is not None:
        weights = rollout_is_weights.to(device=current_log_probs.device, dtype=dtype)
        if weights.shape != current_log_probs.shape:
            raise ValueError("rollout_is_weights must match token log-prob shape")
        nonselected_per_token = nonselected_per_token * weights
        selected_per_token = selected_per_token * weights

    nonselected_loss = _masked_rollout_mean(nonselected_per_token, nonselected_mask)
    # Normalize the two paths separately before weighting the selected path.
    selected_loss = _masked_rollout_mean(selected_per_token, selected_mask)
    loss = nonselected_loss + config.eta * selected_loss

    valid_tokens = mask.sum().clamp(min=1)
    metrics = {
        "loss": float(loss.detach()),
        "loss_nonselected": float(nonselected_loss.detach()),
        "loss_selected": float(selected_loss.detach()),
        "selected_token_share": float((selected_mask.sum() / valid_tokens).detach()),
        "contrast_abs_mean": float(((contrast.abs() * mask).sum() / valid_tokens).detach()),
        "residual_abs_mean": float(((residual.abs() * mask).sum() / valid_tokens).detach()),
        "ppo_clip_share": float(
            (
                (((ratio - 1.0).abs() > config.epsilon).to(dtype) * mask).sum() / valid_tokens
            ).detach()
        ),
    }
    details: dict[str, Any] = {
        "log_marginal_wrong": log_marginal_wrong,
        "contrast": contrast,
        "residual": residual,
        "selected_mask": selected,
        "modulated_advantage": modulated,
    }
    return loss, metrics, details
