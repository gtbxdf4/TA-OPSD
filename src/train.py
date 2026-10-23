"""Common entry: unchanged author/trace path; an explicitly configured auxiliary only."""

import json
import os
from pathlib import Path
import runpy
import sys
import time

code = Path(__file__).parent
from shared_initialization import install_runtime, verify_before_update

install_runtime()
aux = os.environ.get("TA_OPSD_AUX_MANIFEST")
sys.path.insert(0, os.environ["TA_OPSD_SOURCE"])
import opsd_trainer
from launch import atomic

Original = opsd_trainer.OPSDTrainer


class CapturedTrainer(Original):
    """Passive first-batch/engine provenance; does not modify tensors or RNG."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rank = int(os.getenv("RANK", "0"))
        processor = getattr(self.vllm_engine.llm_engine, "processor", None)
        if processor is None:
            processor = getattr(self.vllm_engine.llm_engine, "input_processor", None)
        stop = {"generation_config": self.generation_config.to_dict()}
        if processor is not None:
            stop["engine_generation_config_fields"] = processor.generation_config_fields
            stop["engine_primary_eos"] = processor.input_preprocessor.get_eos_token_id()
        atomic(
            Path(os.environ["TA_OPSD_EVIDENCE"]) / f"runtime-rank{rank}.json",
            {
                "jsd_token_clip": self.jsd_token_clip,
                "beta": self.beta,
                "fixed_teacher": self.fixed_teacher,
                "temperature": self.temperature,
                "rank": rank,
                "time": time.time(),
                "stop": stop,
            },
        )
        if rank == 0:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            path = Path(os.environ["TA_OPSD_EVIDENCE"]) / "initial-lora.safetensors"
            if not path.exists():
                save_file(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in get_peft_model_state_dict(self.model).items()
                    },
                    str(path),
                )

    def compute_loss(self, model, inputs, *args, **kwargs):
        """Attest the first pre-update adapter and capture two step-zero RNG/input packets."""
        if not getattr(self, "_pre_update_verified", False):
            verify_before_update(self.model, self.state.global_step)
            self._pre_update_verified = True
        count = getattr(self, "_capture_count", 0)
        if self.state.global_step == 0 and count < 2:
            import torch

            rank = int(os.getenv("RANK", "0"))
            path = (
                Path(os.environ["TA_OPSD_EVIDENCE"])
                / f"fixed-step0-inputs-rank{rank}-micro{count}.pt"
            )
            if not path.exists():
                values = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in inputs.items()
                }
                values["_rng_cpu"] = torch.get_rng_state()
                values["_rng_cuda"] = torch.cuda.get_rng_state()
                temporary = path.with_suffix(".tmp")
                torch.save(values, temporary)
                temporary.replace(path)
            self._capture_count = count + 1
        return super().compute_loss(model, inputs, *args, **kwargs)


opsd_trainer.OPSDTrainer = CapturedTrainer
if aux:
    from auxiliary_bridge import with_auxiliary

    opsd_trainer.OPSDTrainer = with_auxiliary(
        opsd_trainer.OPSDTrainer, json.loads(Path(aux).read_text())
    )
runpy.run_path(str(code / "observed_entry.py"), run_name="__main__")
