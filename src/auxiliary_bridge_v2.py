"""Opt-in mathematical additions to the unmodified author's trainer methods."""

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from independent_losses import local_unlikelihood, stop_kl
from replay_v2 import detached_suffix_logits, exposure_ids, joint_backward_context


def with_auxiliary(original, config):
    """Extend OPSD with SC on current rollouts and UL on the fixed replay bank."""
    if not any(config.get(k, 0) for k in ["lambda_eos", "lambda_ul"]):
        return original

    class AuxiliaryTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._aux_config = dict(config)
            self._aux_rows = []
            if config.get("lambda_ul", 0):
                path = Path(config["assets"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != config["assets_sha256"]:
                    raise ValueError("replay assets hash changed")
                self._aux_rows = [json.loads(line) for line in path.read_text().splitlines()]
                expected = config.get("record_count", 9)
                if (
                    type(expected) is not int
                    or expected < 0
                    or len(self._aux_rows) != expected
                    or any(not row["approved"] or row["split"] != "fit" for row in self._aux_rows)
                ):
                    raise ValueError("expected the frozen algorithm-selected fit records")
            if self.args.gradient_accumulation_steps != 2 or self.accelerator.num_processes not in (
                4,
                8,
            ):
                raise ValueError("replay schedule requires frozen DP4/GA2")
            self._aux_seen_step = -1
            self._aux_micro = 0

        def _aux_emit(self, kind, values):
            rank = int(os.getenv("RANK", "0"))
            path = Path(os.environ["V17_EVIDENCE"]) / f"aux-{kind}-rank{rank}.jsonl"
            values = dict(
                values,
                global_step=self.state.global_step,
                rank=rank,
                time=time.time(),
                experiment_uuid=os.environ["V17_UUID"],
            )
            with path.open("a") as stream:
                stream.write(json.dumps(values, ensure_ascii=False) + "\n")

        def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None, **kwargs):
            # SC uses the same valid positions as the unchanged base distillation loss.
            base = super().generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                **kwargs,
            )
            coefficient = self._aux_config.get("lambda_eos", 0)
            if coefficient == 0:
                return base
            eos = stop_kl(
                student_logits, teacher_logits, labels != -100, self._aux_config["stop_ids"]
            )
            self._aux_emit(
                "eos",
                {
                    "base_loss": float(base.detach()),
                    "stop_kl": float(eos.detach()),
                    "lambda_eos": coefficient,
                },
            )
            return base + coefficient * eos

        def compute_loss(self, model, inputs, *args, **kwargs):
            # The on-policy objective and replay objective share one optimizer update.
            with (
                joint_backward_context(self.accelerator.unwrap_model(model))
                if self._aux_rows
                else nullcontext()
            ):
                result = super().compute_loss(model, inputs, *args, **kwargs)
            if not self._aux_rows:
                return result
            base = result[0] if isinstance(result, tuple) else result
            step = self.state.global_step
            if step != self._aux_seen_step:
                self._aux_seen_step = step
                self._aux_micro = 0
            if self._aux_micro >= 2:
                raise ValueError("unexpected extra training microbatch")
            rank = self.accelerator.process_index
            indices = exposure_ids(
                step, self._aux_micro, rank, len(self._aux_rows), self.accelerator.num_processes
            )
            raw = self.accelerator.unwrap_model(model)
            device = next(raw.parameters()).device
            values = []
            evidence = []
            for index in indices:
                # Each record carries its original causal prefix and local target span.
                row = self._aux_rows[index]
                record = {"id": row["id"], "asset_index": index, "micro": self._aux_micro}
                auxiliary = base.new_zeros(())
                for key, coefficient_name, loss_function in [
                    ("negative", "lambda_ul", local_unlikelihood)
                ]:
                    span = row.get(key)
                    coefficient = self._aux_config.get(coefficient_name, 0)
                    if not span or coefficient == 0:
                        continue
                    prefix = span["prefix_ids"]
                    suffix = span["suffix_ids"]
                    selected = span["selected"]
                    logits = detached_suffix_logits(raw, prefix, suffix, device)
                    target = torch.tensor([suffix], device=device)
                    mask = torch.tensor([selected], device=device, dtype=torch.bool)
                    extra = [self._aux_config["stop_ids"]]
                    loss = loss_function(logits, target, mask, *extra)
                    auxiliary = auxiliary + coefficient * loss
                    record[key] = {
                        "raw_loss": float(loss.detach()),
                        "coefficient": coefficient,
                        "prefix_tokens": len(prefix),
                        "suffix_tokens": len(suffix),
                        "selected_tokens": int(mask.sum()),
                    }
                values.append(auxiliary)
                evidence.append(record)
            combined = base + torch.stack(values).mean()
            # DDP and accumulation complete the global mean over 16 replay examples.
            self._aux_emit(
                "replay",
                {
                    "records": evidence,
                    "replay_loss": float(torch.stack(values).mean().detach()),
                    "rng_preserved": True,
                    "global_exposures_per_optimizer_step": 16,
                },
            )
            self._aux_micro += 1
            if isinstance(result, tuple):
                if hasattr(result[1], "loss"):
                    result[1].loss = combined
                return combined, result[1]
            return combined

    return AuxiliaryTrainer
