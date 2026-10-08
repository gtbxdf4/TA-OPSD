"""Local suffix replay, written independently; prefix KV has no gradient."""

from contextlib import contextmanager
from functools import partial

import torch


def exposure_ids(step, micro, rank, count, world=4):
    """Partition 16 replay examples across two microbatches and all DP ranks."""
    if world not in (4, 8) or not (0 <= micro < 2 and 0 <= rank < world and count > 0):
        raise ValueError("DP4/DP8 GA2 replay layout invalid")
    per = 8 // world
    # A cyclic schedule keeps the global replay count fixed for DP4 and DP8.
    start = (step * 2 + micro) * 8 + rank * per
    return [(start + i) % count for i in range(per)]


@contextmanager
def replay_context(model, device):
    """Enable prefix caching temporarily and preserve the main rollout RNG stream."""
    flags = [
        (m, m.gradient_checkpointing)
        for m in model.modules()
        if hasattr(m, "gradient_checkpointing")
    ]
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        for module, _ in flags:
            module.gradient_checkpointing = False
        try:
            yield
        finally:
            for module, value in flags:
                module.gradient_checkpointing = value


@contextmanager
def joint_backward_context(model):
    """Avoid nested reentrant backward reducing the same ZeRO2 leaf twice.

    Only execution plumbing changes; the author forward/loss is called unmodified.
    Every original checkpoint callable/argument is restored, including RNG settings.
    """
    saved = []
    for module in model.modules():
        callback = getattr(module, "_gradient_checkpointing_func", None)
        if getattr(module, "gradient_checkpointing", False) and callback is not None:
            if not isinstance(callback, partial):
                raise ValueError("unrecognized checkpoint callback")
            saved.append((module, callback))
            kwargs = dict(callback.keywords)
            kwargs["use_reentrant"] = False
            module._gradient_checkpointing_func = partial(callback.func, *callback.args, **kwargs)
    try:
        yield
    finally:
        for module, callback in saved:
            module._gradient_checkpointing_func = callback


def detached_suffix_logits(model, prefix, suffix, device):
    """Return one prediction per suffix token while detaching the earlier prefix."""
    if not prefix or not suffix:
        raise ValueError("nonempty causal prefix and target required")
    with replay_context(model, device):
        cache = None
        if len(prefix) > 1:
            # Cache the unchanged history once; its activations need no backward graph.
            with torch.no_grad():
                cached = model(
                    input_ids=torch.tensor([prefix[:-1]], device=device),
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = cached.past_key_values
                if cache is None:
                    raise ValueError("prefix KV unavailable; no full-prefix-gradient fallback")
                del cached
        # Last prefix token predicts suffix[0]; suffix[-1] is never an input.
        inputs = torch.tensor([[prefix[-1]] + suffix[:-1]], device=device)
        return model(input_ids=inputs, past_key_values=cache, use_cache=False).logits
