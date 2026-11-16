"""The paper's ZERO intervention runs before native vLLM top-p sampling."""

import dataclasses

from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class FirstDecisionEOS(LogitsProcessor):
    """Mask STOP for ZERO requests until their first token has been sampled."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.pending = {}

    def is_argmax_invariant(self):
        return False

    @staticmethod
    def add(params, prompt_ids, output_ids):
        stop_ids = (params.extra_args or {}).get("zero_first_stop_ids")
        # vLLM mutates output_ids in place; keep the reference, not a copy.
        return (stop_ids, output_ids) if stop_ids is not None else None

    def update_state(self, batch_update):
        if batch_update:
            # Remove stale requests before vLLM adds new ones to reused slots.
            for index in batch_update.removed:
                self.pending.pop(index, None)
            batch_update = dataclasses.replace(batch_update, removed=[])
        process_dict_updates(self.pending, batch_update, self.add)
        for index, (_, output_ids) in list(self.pending.items()):
            if output_ids:
                del self.pending[index]

    def apply(self, logits):
        for index, (stop_ids, output_ids) in self.pending.items():
            if not output_ids:
                logits[index, list(stop_ids)] = float("-inf")
        return logits
