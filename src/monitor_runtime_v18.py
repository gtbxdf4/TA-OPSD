"""Passive full main-rollout monitoring; original forward/loss/generation stay in place."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from raw_monitor_v18 import (
    align_response,
    atomic_gzip,
    effective_stop_ids,
    local_outputs,
    rng_fingerprint,
    token_metrics,
)


def row_identity(row):
    problem, solution = str(row["problem"]), str(row["solution"])
    return (
        hashlib.sha256(problem.encode()).hexdigest(),
        hashlib.sha256(
            json.dumps([problem, solution], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    )


class SourceCollator:
    def __init__(self, original, index):
        self.original, self.index = original, index

    def __call__(self, features):
        result = self.original(features)
        rows = []
        for row in features:
            question_hash, paired_hash = row_identity(row)
            entry = self.index.get(paired_hash)
            if entry is None or entry["question_hash"] != question_hash:
                raise ValueError("monitor source row absent from frozen source index")
            rows.append(
                {
                    "question_hash": question_hash,
                    "paired_row_hash": paired_hash,
                    "source_row_ids": entry["source_row_ids"],
                    "problem": str(row["problem"]),
                }
            )
        result["_v18_monitor_rows"] = rows
        return result


def with_raw_monitor(original, config=None, enabled=True):
    if not enabled:
        return original
    if config is None:
        path = Path(os.environ["V18_MONITOR_CONTRACT"])
        config = json.loads(path.read_text())
        contract_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        contract_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config = dict(config)
    index_path = Path(config["question_index_path"])
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != config["question_index_sha256"]:
        raise ValueError("monitor source index hash mismatch")
    index = json.loads(index_path.read_text())

    class RawMonitorTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (
                self.accelerator.num_processes != config["world_size"]
                or self.args.gradient_accumulation_steps != config["gradient_accumulation_steps"]
            ):
                raise ValueError("monitor world size/accumulation differs from scientific contract")
            engine = self.vllm_engine.llm_engine
            processor = getattr(engine, "processor", None) or getattr(
                engine, "input_processor", None
            )
            if processor is None:
                raise ValueError("cannot attest live engine stopping configuration")
            actual_stop = effective_stop_ids(
                processor.input_preprocessor.get_eos_token_id(), processor.generation_config_fields
            )
            if actual_stop != config["stop_ids"]:
                raise ValueError("live engine stop union differs from monitor contract")
            self.data_collator = SourceCollator(self.data_collator, index)
            self._raw_step, self._raw_micro = -1, 0
            self._raw_context = None
            self._raw_inputs = None
            self._raw_segment = os.environ["V18_MONITOR_SEGMENT"]
            self._raw_evidence = Path(os.environ["V17_EVIDENCE"])
            launch_path = self._raw_evidence.parent / "LAUNCH.json"
            self._raw_launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
            generate = self.vllm_engine.generate

            def observed_generate(*gen_args, **gen_kwargs):
                started = time.monotonic()
                outputs = generate(*gen_args, **gen_kwargs)
                if self._raw_context is not None:
                    if self._raw_context["outputs"] is not None:
                        raise ValueError(
                            "multiple main generation calls in one training microbatch"
                        )
                    tp = self.vllm_tensor_parallel_size
                    group_rank = (
                        torch.distributed.get_rank(group=self.vllm_tp_group) if tp > 1 else 0
                    )
                    selected = local_outputs(
                        outputs, self._raw_context["local_batch"], tp, group_rank
                    )
                    self._raw_context["outputs"] = selected
                    self._raw_context["generation_seconds"] = time.monotonic() - started
                return outputs

            self.vllm_engine.generate = observed_generate

        def training_step(self, model, inputs, *args, **kwargs):
            step = self.state.global_step
            if step != self._raw_step:
                self._raw_step, self._raw_micro = step, 0
            if self._raw_micro >= config["gradient_accumulation_steps"]:
                raise ValueError("extra main monitoring microbatch; cannot silently subsample")
            rows = inputs.get("_v18_monitor_rows")
            if rows is None or len(rows) != config["per_device_train_batch_size"]:
                raise ValueError("missing full local source metadata")
            self._raw_context = {
                "outputs": None,
                "local_batch": len(rows),
                "rows": rows,
                "emissions": 0,
                "started": time.monotonic(),
            }
            try:
                result = super().training_step(model, inputs, *args, **kwargs)
                if self._raw_context["emissions"] != 1:
                    raise ValueError("one full main distribution shard required per microbatch")
                self._raw_micro += 1
                return result
            finally:
                self._raw_context = None
                self._raw_inputs = None

        def compute_loss(self, model, inputs, *args, **kwargs):
            if self._raw_context is None:
                raise ValueError("main loss outside recorded training context")
            self._raw_inputs = inputs
            try:
                return super().compute_loss(model, inputs, *args, **kwargs)
            finally:
                self._raw_inputs = None

        def generalized_jsd_loss(self, student_logits, teacher_logits, labels=None, **kwargs):
            # Incoming tensors are already teacher/student shifted by the author's compute_loss.
            # All monitoring is detached; return the identical original loss expression.
            self._record_raw(student_logits, teacher_logits, labels, kwargs)
            return super().generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                **kwargs,
            )

        @torch.no_grad()
        def _record_raw(self, student_logits, teacher_logits, labels, loss_kwargs):
            context, inputs = self._raw_context, self._raw_inputs
            if (
                context is None
                or inputs is None
                or context["outputs"] is None
                or context["emissions"]
            ):
                raise ValueError("missing/duplicate raw main rollout context")
            if (
                student_logits.shape != teacher_logits.shape
                or student_logits.shape[:2] != labels.shape
            ):
                raise ValueError("monitor aligned logits/mask shape mismatch")
            started, before = time.monotonic(), rng_fingerprint()
            sp, tp = inputs["student_prompt_length"], inputs["teacher_prompt_length"]
            sampled = inputs["student_input_ids"][:, sp:]
            records = []
            for i, (source, response) in enumerate(zip(context["rows"], context["outputs"])):
                if len(response.outputs) != 1:
                    raise ValueError("main rollout must have exactly one generation per source row")
                raw = response.outputs[0]
                ids = list(raw.token_ids)
                record = dict(
                    source,
                    local_response_index=i,
                    text=raw.text,
                    decoded_with_special_tokens=self.processing_class.decode(
                        ids, skip_special_tokens=False
                    ),
                    finish_reason=raw.finish_reason,
                    stop_reason=raw.stop_reason,
                    student_prompt_token_ids=inputs["student_input_ids"][i, :sp]
                    .detach()
                    .cpu()
                    .tolist(),
                    teacher_prompt_token_ids=inputs["teacher_input_ids"][i, :tp]
                    .detach()
                    .cpu()
                    .tolist(),
                )
                record.update(
                    align_response(
                        ids,
                        sampled[i].detach().cpu().tolist(),
                        labels[i].detach().cpu().tolist(),
                        sp,
                        tp,
                        raw.finish_reason,
                        config["stop_ids"],
                    )
                )
                record["metrics"] = token_metrics(
                    student_logits[i, : len(ids)],
                    teacher_logits[i, : len(ids)],
                    sampled[i, : len(ids)],
                    k=config["k"],
                    stop_ids=config["stop_ids"],
                    temperature=config["temperature"],
                    chunk_tokens=config["chunk_tokens"],
                )
                records.append(record)
            after = rng_fingerprint()
            if before != after:
                raise ValueError("passive monitor changed RNG")
            rank = self.accelerator.process_index
            path = (
                self._raw_evidence
                / "raw-monitor"
                / self._raw_segment
                / f"rank{rank}"
                / (f"step{self.state.global_step + 1:04d}-micro{self._raw_micro}.json.gz")
            )
            value = {
                "schema": "v18-raw-monitor-v1",
                "time": time.time(),
                "global_step_before_update": self.state.global_step,
                "optimizer_step": self.state.global_step + 1,
                "microbatch_index": self._raw_micro,
                "rank": rank,
                "world_size": self.accelerator.num_processes,
                "arm": os.environ["V18_MONITOR_ARM"],
                "experiment_uuid": os.environ["V17_UUID"],
                "model": config["model"],
                "shared_initialization_tensor_sha256": config[
                    "shared_initialization_tensor_sha256"
                ],
                "contract_sha256": contract_hash,
                "source_commit": config["source_commit"],
                "runtime_code_commit": self._raw_launch.get("runtime_code_commit"),
                "launch_segment": self._raw_segment,
                "stream": "main_opsd_rollout",
                "teacher_privileged": True,
                "temperature": config["temperature"],
                "loss_temperature": loss_kwargs.get("temperature"),
                "dtype": "float32",
                "k": config["k"],
                "stop_ids": config["stop_ids"],
                "rng_before": before,
                "rng_after": after,
                "metrics_elapsed_seconds": time.monotonic() - started,
                "generation_seconds_batch": context["generation_seconds"],
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated()
                if student_logits.is_cuda
                else 0,
                "records": records,
            }
            write_started = time.monotonic()
            size = atomic_gzip(path, value)
            index_row = {
                "path": str(path),
                "bytes": size,
                "write_seconds": time.monotonic() - write_started,
                "metrics_seconds": value["metrics_elapsed_seconds"],
                "global_step_before_update": self.state.global_step,
                "microbatch_index": self._raw_micro,
                "rank": rank,
                "launch_segment": self._raw_segment,
                "time": time.time(),
                "records": len(records),
                "real_tokens": sum(len(r["token_ids"]) for r in records),
            }
            with (self._raw_evidence / f"raw-monitor-index-rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(index_row) + "\n")
                stream.flush()
            context["emissions"] += 1

    return RawMonitorTrainer
