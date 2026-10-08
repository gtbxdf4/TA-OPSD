"""Read-only instrumentation around the unedited author entrypoint.

No sampler, RNG, forward, loss, optimizer, generation argument or return value is changed.
The smoke-only callback stops after two optimizer steps on the same 100-step scheduler.
"""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

SOURCE = Path(os.environ["V17_SOURCE"])
EVIDENCE = Path(os.environ["V17_EVIDENCE"])
EVIDENCE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))

from transformers import TrainerCallback
import opsd_trainer
import datasets

# Only the transport path changes: these are freshly downloaded author parquet bytes.
original_load_dataset = datasets.load_dataset


def local_author_dataset(path, *args, **kwargs):
    if path == "siyanzhao/Openthoughts_math_30k_opsd":
        files = json.loads(os.environ["PAPER_TRAIN_FILES"])
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = [pq.ParquetFile(p).read().replace_schema_metadata(None) for p in files]
        return datasets.DatasetDict({"train": datasets.Dataset(pa.concat_tables(tables))})
    return original_load_dataset(path, *args, **kwargs)


datasets.load_dataset = local_author_dataset


def emit(kind, value):
    rank = int(os.getenv("RANK", "0"))
    value.update(timestamp=time.time(), rank=rank, experiment_uuid=os.environ["V17_UUID"])
    with (EVIDENCE / f"{kind}-rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


class Progress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        scheduler = kwargs.get("lr_scheduler")
        emit(
            "optimizer",
            {
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "learning_rate": scheduler.get_last_lr() if scheduler else None,
            },
        )
        limit = int(os.environ.get("V17_SMOKE_STOP", "0"))
        if limit and state.global_step >= limit:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        emit("metrics", {"global_step": state.global_step, "logs": logs})


OriginalTrainer = opsd_trainer.OPSDTrainer


class ObservedTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        dataset = kwargs["train_dataset"]
        emit(
            "dataset",
            {
                "num_rows": len(dataset),
                "columns": dataset.column_names,
                "fingerprint": dataset._fingerprint,
                "cache_files": dataset.cache_files,
                "no_filter_applied": True,
            },
        )
        super().__init__(*args, **kwargs)
        self.add_callback(Progress())
        emit(
            "resolved_config",
            {
                "args": self.args.to_dict(),
                "generation_config": self.generation_config.to_dict(),
                "sampler_factory": self._get_train_sampler.__qualname__,
            },
        )
        # Observe the exact vLLM request and returned token IDs, without reconstructing generation.
        generate = self.vllm_engine.generate

        def recorded_generate(*gen_args, **gen_kwargs):
            started = time.monotonic()
            outputs = generate(*gen_args, **gen_kwargs)
            for response in outputs:
                emit(
                    "rollouts",
                    {
                        "global_step_before_update": self.state.global_step,
                        "prompt": response.prompt,
                        "prompt_token_ids": response.prompt_token_ids,
                        "prompt_sha256": hashlib.sha256(response.prompt.encode()).hexdigest(),
                        "generation_wall_seconds_batch": time.monotonic() - started,
                        "outputs": [
                            {
                                "text": item.text,
                                "token_ids": list(item.token_ids),
                                "finish_reason": item.finish_reason,
                                "stop_reason": item.stop_reason,
                            }
                            for item in response.outputs
                        ],
                    },
                )
            return outputs

        self.vllm_engine.generate = recorded_generate

    def training_step(self, model, inputs, *args, **kwargs):
        emit(
            "batch",
            {
                "global_step_before_update": self.state.global_step,
                "student_prompt_token_ids": inputs["student_prompts"].detach().cpu().tolist(),
                "student_attention_mask": inputs["student_prompt_attention_mask"]
                .detach()
                .cpu()
                .tolist(),
            },
        )
        return super().training_step(model, inputs, *args, **kwargs)

    def train(self, *args, **kwargs):
        resume = os.environ.get("V17_RESUME")
        if resume:
            kwargs["resume_from_checkpoint"] = resume
        return super().train(*args, **kwargs)


opsd_trainer.OPSDTrainer = ObservedTrainer
runpy.run_path(str(SOURCE / "opsd_train.py"), run_name="__main__")
