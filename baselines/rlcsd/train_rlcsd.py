#!/usr/bin/env python3
"""Full matched non-thinking RLCSD training entry; run under Accelerate.

The adapter deliberately reuses the pinned TRL GRPO sampler, verifier, PEFT
construction, DeepSpeed config, and shared initialization helper.  Only the
RLCSD teacher-context scoring and policy objective are specialized here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import trl.trainer.grpo_trainer as grpo_module
from accelerate.utils import gather_object
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from rlcsd_core import RLCSDConfig, rlcsd_loss, selected_token_log_mass
from rlcsd_data import (
    build_student_message,
    build_teacher_message,
    choose_negative_siblings,
    extract_boxed_answer,
    token_set_overlap,
    verify_rollout_prompt,
)
from rlcsd_runner import OFFICIAL_RLCSD_COMMIT, load_spec
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


def load_ordered_parquet(paths: list[str]) -> DatasetDict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.ParquetFile(path).read().replace_schema_metadata(None) for path in paths]
    return DatasetDict({"train": Dataset(pa.concat_tables(tables))})


def _disable_adapter(model: torch.nn.Module):
    method = getattr(model, "disable_adapter", None)
    return method() if method is not None else nullcontext()


class StopAtOptimizerUpdate(TrainerCallback):
    def __init__(self, stop: int):
        self.stop = stop

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop:
            control.should_training_stop = True
        return control


class MatchedRLCSDTrainer(GRPOTrainer):
    def __init__(self, *args, rlcsd_spec: dict[str, Any], correctness_fn, **kwargs):
        self.rlcsd_spec = rlcsd_spec
        algorithm = rlcsd_spec["algorithm"]
        self.rlcsd_config = RLCSDConfig(
            epsilon=algorithm["epsilon"],
            tau=algorithm["tau"],
            beta=algorithm["beta"],
            lam=algorithm["lambda"],
            delta=algorithm["delta"],
            eta=algorithm["eta"],
            residual_clip_low=algorithm["residual_clip"][0],
            residual_clip_high=algorithm["residual_clip"][1],
        )
        self.rlcsd_k = int(algorithm["k"])
        self.rlcsd_eos_token_ids = (151643, 151645)
        self.correctness_fn = correctness_fn
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "problem", "solution", "Answer", "uid"]

    def _teacher_prompt_ids(
        self, messages: list[dict[str, str]], completion_width: int
    ) -> tuple[list[int], bool]:
        ids = self.processing_class.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        budget = int(self.rlcsd_spec["sequence"]["max_length"]) - completion_width
        if budget < 1:
            raise ValueError("completion width leaves no token budget for teacher prompt")
        truncated = len(ids) > budget
        return list(ids[-budget:]), truncated

    def _score_teacher_prompts(
        self,
        model: torch.nn.Module,
        prompts: list[list[dict[str, str]]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], list[list[int]]
    ]:
        if not prompts:
            shape = completion_ids.shape
            empty = torch.empty(shape, device=completion_ids.device, dtype=torch.float32)
            return empty, empty, empty.to(dtype=torch.long), empty, {"truncated": 0.0}, []
        width = completion_ids.shape[1]
        prompt_ids_and_flags = [self._teacher_prompt_ids(prompt, width) for prompt in prompts]
        tokenized = [item[0] for item in prompt_ids_and_flags]
        max_prompt = max(len(ids) for ids in tokenized)
        pad_id = self.processing_class.pad_token_id
        prompt_tensor = completion_ids.new_full((len(tokenized), max_prompt), pad_id)
        prompt_mask = completion_mask.new_zeros((len(tokenized), max_prompt))
        for row, ids in enumerate(tokenized):
            prompt_tensor[row, -len(ids) :] = torch.tensor(ids, device=completion_ids.device)
            prompt_mask[row, -len(ids) :] = 1
        full_ids = torch.cat([prompt_tensor, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        unwrapped = self.accelerator.unwrap_model(model)
        all_logps = []
        all_entropies = []
        all_top1 = []
        all_stop_log_mass = []
        with torch.no_grad(), _disable_adapter(unwrapped):
            for row in range(full_ids.shape[0]):
                model_inputs = {
                    "input_ids": full_ids[row : row + 1],
                    "attention_mask": full_mask[row : row + 1],
                    "use_cache": False,
                }
                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = width + 1
                logits = model(**model_inputs).logits[:, :-1, :]
                logits = logits[:, -width:, :] / self.temperature
                target_ids = full_ids[row : row + 1, -width:]
                all_logps.append(selective_log_softmax(logits, target_ids))
                all_entropies.append(entropy_from_logits(logits))
                all_top1.append(logits.argmax(dim=-1))
                all_stop_log_mass.append(selected_token_log_mass(logits, self.rlcsd_eos_token_ids))
                del logits
        # Keep the computed BF16 values, but match the FP32 rollout-cache buffers.
        logps = torch.cat(all_logps, dim=0).float()
        entropies = torch.cat(all_entropies, dim=0).float()
        top1 = torch.cat(all_top1, dim=0)
        stop_log_mass = torch.cat(all_stop_log_mass, dim=0).float()
        return (
            logps,
            entropies,
            top1,
            stop_log_mass,
            {"truncated": float(sum(flag for _, flag in prompt_ids_and_flags))},
            tokenized,
        )

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        output = super()._generate_and_score_completions(inputs)
        actual_prompts = []
        for index, row in enumerate(inputs):
            ids = output["prompt_ids"][index][output["prompt_mask"][index].bool()].tolist()
            verify_rollout_prompt(self.processing_class, row["prompt"], ids)
            actual_prompts.append(ids)
        if self.state.global_step == 0:
            evidence = Path(self.args.output_dir) / "evidence"
            evidence.mkdir(exist_ok=True)
            proof = dict(
                PASS=True,
                enable_thinking=False,
                checked_prompts=len(actual_prompts),
                chat_template_kwargs=self.chat_template_kwargs,
                actual_suffix_token_ids=actual_prompts[0][-16:],
                actual_suffix_text=self.processing_class.decode(
                    actual_prompts[0][-16:], skip_special_tokens=False
                ),
            )
            (evidence / f"ROLLOUT_MODE-rank{self.accelerator.process_index}.json").write_text(
                json.dumps(proof, indent=2) + "\n"
            )
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        rank = self.accelerator.process_index

        local_records = []
        for index, (row, text) in enumerate(zip(inputs, texts, strict=True)):
            reward = float(self.correctness_fn([text], [row["Answer"]])[0])
            local_records.append(
                {
                    "key": f"{rank}:{index}",
                    "origin_rank": rank,
                    "origin_index": index,
                    "uid": str(row["uid"]),
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "answer": row["Answer"],
                    "completion": text,
                    "boxed_answer": extract_boxed_answer(text),
                    "correct": reward > 0.0,
                }
            )
        global_records = gather_object(local_records)
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in global_records:
            groups.setdefault(record["uid"], []).append(record)

        positive_prompts: list[list[dict[str, str]]] = []
        positive_targets: list[int] = []
        negative_prompts: list[list[dict[str, str]]] = []
        negative_target_slots: list[tuple[int, int]] = []
        valid_target = torch.zeros(
            len(local_records), dtype=torch.bool, device=completion_ids.device
        )
        wrong_valid = torch.zeros(
            (len(local_records), self.rlcsd_k), dtype=torch.bool, device=completion_ids.device
        )
        selected_negatives: dict[int, list[dict[str, Any]]] = {}
        valid_group_uids: set[str] = set()
        seed_prefix = f"seed42:step{self.state.global_step}"
        for index, record in enumerate(local_records):
            if not str(record["solution"]).strip():
                raise ValueError(f"empty GT solution for uid={record['uid']}")
            negatives = choose_negative_siblings(
                groups[record["uid"]],
                target_key=record["key"],
                k=self.rlcsd_k,
                seed_material=f"{seed_prefix}:{record['uid']}",
            )
            if not negatives:
                continue
            valid_target[index] = True
            valid_group_uids.add(record["uid"])
            selected_negatives[index] = negatives
            positive_targets.append(index)
            positive_prompts.append(
                build_teacher_message(record["problem"], record["solution"], record["answer"])
            )
            for slot, negative in enumerate(negatives):
                wrong_valid[index, slot] = True
                negative_target_slots.append((index, slot))
                negative_prompts.append(
                    build_teacher_message(
                        record["problem"], negative["completion"], negative["boxed_answer"]
                    )
                )

        token_width = completion_ids.shape[1]
        teacher_correct = torch.zeros_like(completion_ids, dtype=torch.float32)
        teacher_correct_entropy = torch.zeros_like(teacher_correct)
        teacher_correct_top1 = torch.zeros_like(completion_ids)
        teacher_correct_stop_log_mass = torch.zeros_like(teacher_correct)
        teacher_wrong = torch.zeros(
            (len(local_records), self.rlcsd_k, token_width),
            device=completion_ids.device,
            dtype=torch.float32,
        )
        positive_tokenized: list[list[int]] = []
        negative_tokenized: list[list[int]] = []
        positive_stats = {"truncated": 0.0}
        negative_stats = {"truncated": 0.0}
        if positive_targets:
            target_tensor = torch.tensor(positive_targets, device=completion_ids.device)
            (
                pos_logps,
                pos_entropy,
                pos_top1,
                pos_stop_log_mass,
                positive_stats,
                positive_tokenized,
            ) = self._score_teacher_prompts(
                self.model,
                positive_prompts,
                completion_ids.index_select(0, target_tensor),
                completion_mask.index_select(0, target_tensor),
            )
            teacher_correct.index_copy_(0, target_tensor, pos_logps)
            teacher_correct_entropy.index_copy_(0, target_tensor, pos_entropy)
            teacher_correct_top1.index_copy_(0, target_tensor, pos_top1)
            teacher_correct_stop_log_mass.index_copy_(0, target_tensor, pos_stop_log_mass)
        top1_agreements = []
        if negative_target_slots:
            target_tensor = torch.tensor(
                [target for target, _ in negative_target_slots], device=completion_ids.device
            )
            neg_logps, _, neg_top1, _, negative_stats, negative_tokenized = (
                self._score_teacher_prompts(
                    self.model,
                    negative_prompts,
                    completion_ids.index_select(0, target_tensor),
                    completion_mask.index_select(0, target_tensor),
                )
            )
            for flat_index, (target, slot) in enumerate(negative_target_slots):
                teacher_wrong[target, slot] = neg_logps[flat_index]
                target_mask = completion_mask[target].bool()
                if target_mask.any():
                    top1_agreements.append(
                        (
                            teacher_correct_top1[target][target_mask]
                            == neg_top1[flat_index][target_mask]
                        )
                        .float()
                        .mean()
                        .item()
                    )

        # Invalid contexts stay in the DDP batch but are zero-masked. Give their
        # unused marginal a finite dummy slot so the objective rejects no sample.
        wrong_valid[~valid_target, 0] = True
        output["completion_mask"] = completion_mask * valid_target.unsqueeze(-1)
        output["teacher_correct_log_probs"] = teacher_correct
        output["teacher_correct_entropy"] = teacher_correct_entropy
        output["teacher_correct_stop_log_mass"] = teacher_correct_stop_log_mass
        output["teacher_wrong_multi_log_probs"] = teacher_wrong
        output["teacher_wrong_multi_valid_mask"] = wrong_valid

        k_counts = wrong_valid[valid_target].sum(-1).float()
        overlap_values = []
        offset = 0
        for pos_index, target in enumerate(positive_targets):
            for _ in selected_negatives[target]:
                overlap_values.append(
                    token_set_overlap(positive_tokenized[pos_index], negative_tokenized[offset])
                )
                offset += 1
        mode = "train" if self.model.training else "eval"
        context_metrics = {
            "rlcsd/context_valid_sample_share": valid_target.float().mean().item(),
            "rlcsd/context_valid_group_share": len(valid_group_uids) / max(len(groups), 1),
            "rlcsd/contrast_coverage_k_mean": k_counts.mean().item() if k_counts.numel() else 0.0,
            "rlcsd/teacher_scored_context_count": float(
                len(positive_prompts) + len(negative_prompts)
            ),
            "rlcsd/teacher_prompt_truncated_count": positive_stats["truncated"]
            + negative_stats["truncated"],
            "rlcsd/hint_token_set_overlap_jaccard": sum(overlap_values)
            / max(len(overlap_values), 1),
            "rlcsd/teacher_correct_wrong_top1_agreement_sampled_positions": (
                sum(top1_agreements) / max(len(top1_agreements), 1)
            ),
        }
        for name, value in context_metrics.items():
            tensor = torch.tensor(value, device=completion_ids.device)
            self._metrics[mode][name].append(self.accelerator.gather(tensor).nanmean().item())
        return output

    def _student_scores(self, model, input_ids, attention_mask, width):
        """Use the policy forward once for loss, entropy, and full-prefix stop mass."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = width + 1
        logits = model(**model_inputs).logits[:, :-1, :]
        logits = logits[:, -width:, :] / self.temperature
        target_ids = input_ids[:, -width:]
        current_logps = selective_log_softmax(logits, target_ids)
        with torch.no_grad():
            entropies = entropy_from_logits(logits)
            stop_log_mass = selected_token_log_mass(logits, self.rlcsd_eos_token_ids)
        return current_logps, entropies, stop_log_mass

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        width = completion_ids.size(1)
        current_logps, student_entropies, student_stop_log_mass = self._student_scores(
            model, input_ids, attention_mask, width
        )
        old_logps = inputs.get("old_per_token_logps")
        if old_logps is None:
            old_logps = current_logps.detach()
        rollout_is = inputs.get("importance_sampling_ratio")
        loss, metrics, _ = rlcsd_loss(
            old_logps,
            current_logps,
            inputs["advantages"],
            completion_mask,
            inputs["teacher_correct_log_probs"],
            inputs["teacher_wrong_multi_log_probs"],
            inputs["teacher_wrong_multi_valid_mask"],
            self.rlcsd_config,
            rollout_is_weights=rollout_is,
        )
        loss = loss / self.current_gradient_accumulation_steps

        mode = "train" if self.model.training else "eval"
        valid = completion_mask.bool()
        token_count = completion_mask.sum().clamp(min=1)
        entropy = (student_entropies * completion_mask).sum() / token_count
        teacher_entropy = (inputs["teacher_correct_entropy"] * completion_mask).sum() / token_count
        length = completion_mask.sum(-1).float()
        valid_samples = length > 0
        valid_length = length[valid_samples]
        eos_ids = torch.tensor(self.rlcsd_eos_token_ids, device=completion_ids.device)
        eos_mask = valid & torch.isin(completion_ids, eos_ids)
        eos_count = eos_mask.sum().clamp(min=1)
        missing = loss.detach().new_full((), float("nan"))
        student_emitted_stop_lp = (
            (current_logps * eos_mask).sum() / eos_count if eos_mask.any() else missing
        )
        teacher_emitted_stop_lp = (
            (inputs["teacher_correct_log_probs"] * eos_mask).sum() / eos_count
            if eos_mask.any()
            else missing
        )
        student_stop_mass = (student_stop_log_mass.exp() * completion_mask).sum() / token_count
        teacher_stop_mass = (
            inputs["teacher_correct_stop_log_mass"].exp() * completion_mask
        ).sum() / token_count
        metric_tensors = {
            **{
                f"rlcsd/{key}": torch.tensor(value, device=loss.device)
                for key, value in metrics.items()
            },
            "rlcsd/student_entropy_sampled_context": entropy.detach(),
            "rlcsd/teacher_entropy_positive_hint_context": teacher_entropy.detach(),
            "rlcsd/completion_length_mean_valid_context": (
                valid_length.mean() if valid_length.numel() else length.new_zeros(())
            ).detach(),
            "rlcsd/completion_cap_hit_share_valid_context": (
                (valid_length >= width).float().mean()
                if valid_length.numel()
                else length.new_zeros(())
            ).detach(),
            "rlcsd/stop_observed_share_valid_context": (
                eos_mask.any(-1)[valid_samples].float().mean()
                if valid_samples.any()
                else length.new_zeros(())
            ).detach(),
            "rlcsd/emitted_stop_token_count_per_rank_mean": eos_mask.sum().float().detach(),
            "rlcsd/student_emitted_stop_logp_conditional": student_emitted_stop_lp.detach(),
            "rlcsd/teacher_positive_hint_emitted_stop_logp_conditional": teacher_emitted_stop_lp.detach(),
            "rlcsd/emitted_stop_logp_conditional_delta_teacher_minus_student": (
                teacher_emitted_stop_lp - student_emitted_stop_lp
            ).detach(),
            "rlcsd/student_stop_probability_mass_mean_all_valid_prefixes": student_stop_mass.detach(),
            "rlcsd/teacher_positive_hint_stop_probability_mass_mean_all_valid_prefixes": teacher_stop_mass.detach(),
            "rlcsd/stop_probability_mass_mean_all_valid_prefixes_delta_teacher_minus_student": (
                teacher_stop_mass - student_stop_mass
            ).detach(),
            "rlcsd/optimizer_update_target": torch.tensor(
                float(self.state.global_step + 1), device=loss.device
            ),
        }
        for name, value in metric_tensors.items():
            self._metrics[mode][name].append(self.accelerator.gather(value).nanmean().item())
        return loss


def install_shared_initialization(spec: dict[str, Any], output: Path):
    sys.path.insert(0, spec["helper_dir"])
    from shared_initialization_v18 import install_state, read_shared

    state, manifest = read_shared(spec["init_manifest"])
    original = grpo_module.get_peft_model

    def initialized_peft(*args, **kwargs):
        model = original(*args, **kwargs)
        install_state(model, state)
        evidence = output / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        receipt = {
            "PASS": True,
            "rank": int(rank),
            "manifest": spec["init_manifest"],
            "tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
            "B_all_zero": True,
        }
        (evidence / f"INIT_LOAD-rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return model

    grpo_module.get_peft_model = initialized_peft
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON file path or inline JSON object")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(spec["output"])
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        if output.exists():
            raise FileExistsError(f"fresh output required; refusing existing path: {output}")
        output.mkdir(parents=True)
        (output / "checkpoint0").mkdir()
    else:
        for _ in range(600):
            if (output / "checkpoint0").is_dir():
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"rank 0 did not create output: {output}")

    set_seed(spec["optimizer"]["seed"])
    manifest = install_shared_initialization(spec, output)
    sys.path.insert(0, spec["source_dir"])
    from grpo_train import reward_correctness

    tokenizer = AutoTokenizer.from_pretrained(
        spec["base_model"], trust_remote_code=True, padding_side="left", truncation_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = spec["sequence"]["max_length"]
    original_chat_template = tokenizer.apply_chat_template

    def nonthinking_chat_template(*positional, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original_chat_template(*positional, **kwargs)

    tokenizer.apply_chat_template = nonthinking_chat_template

    dataset = load_ordered_parquet(spec["data_files"])["train"]
    if len(dataset) != 29434:
        raise ValueError(f"expected matched 29,434-row T-series training data, got {len(dataset)}")

    def format_row(row, index):
        problem = row.get("Question") or row.get("problem")
        solution = row.get("solution") or row.get("COT_Reason")
        answer = row.get("Answer")
        if not problem or not solution or answer is None:
            raise ValueError(f"training row {index} lacks problem, GT solution, or Answer")
        return {
            "prompt": build_student_message(problem),
            "problem": str(problem),
            "solution": str(solution),
            "Answer": str(answer),
            "uid": str(index),
        }

    dataset = dataset.map(format_row, with_indices=True, remove_columns=dataset.column_names)
    reference = {
        "protocol": spec["protocol"],
        "size": spec["size"],
        "mode": spec["mode"],
        "dp": spec["dp"],
        "base_model": spec["base_model"],
        "initialization_manifest": spec["init_manifest"],
        "initialization_tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
        "official_rlcsd_commit": OFFICIAL_RLCSD_COMMIT,
        "training_thinking": False,
        "dataset_rows": len(dataset),
        "spec": spec,
        "created": time.time(),
    }
    if rank == 0:
        (output / "checkpoint0" / "BASE_REFERENCE.json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=2) + "\n"
        )

    callbacks = []
    if spec["optimizer"]["stop_after_updates"]:
        callbacks.append(StopAtOptimizerUpdate(spec["optimizer"]["stop_after_updates"]))
    training_args = GRPOConfig(
        output_dir=str(output),
        learning_rate=spec["optimizer"]["learning_rate"],
        max_grad_norm=spec["optimizer"]["max_grad_norm"],
        weight_decay=spec["optimizer"]["weight_decay"],
        adam_beta1=spec["optimizer"]["betas"][0],
        adam_beta2=spec["optimizer"]["betas"][1],
        adam_epsilon=spec["optimizer"]["epsilon"],
        lr_scheduler_type=spec["optimizer"]["scheduler"],
        warmup_steps=spec["optimizer"]["warmup_steps"],
        max_steps=spec["optimizer"]["max_steps"],
        per_device_train_batch_size=spec["batching"]["per_device_sequences"],
        gradient_accumulation_steps=spec["batching"]["gradient_accumulation_steps"],
        steps_per_generation=spec["batching"]["steps_per_generation"],
        num_generations=spec["algorithm"]["group_size"],
        num_iterations=spec["batching"]["num_iterations"],
        max_completion_length=spec["generation"]["max_completion_length"],
        max_prompt_length=spec["sequence"]["max_length"]
        - spec["generation"]["max_completion_length"],
        temperature=spec["generation"]["temperature"],
        top_p=spec["generation"]["top_p"],
        top_k=spec["generation"]["top_k"],
        importance_sampling_level="token",
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_cap=spec["algorithm"]["rollout_is_clip"],
        epsilon=spec["algorithm"]["epsilon"],
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        gradient_checkpointing=True,
        bf16=True,
        seed=spec["optimizer"]["seed"],
        data_seed=spec["optimizer"]["seed"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        optim=spec["optimizer"]["name"],
        disable_dropout=spec["optimizer"]["disable_dropout"],
        report_to="none",
        remove_unused_columns=True,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
            "use_cache": False,
        },
        generation_kwargs={"stop_token_ids": [151643, 151645]},
        chat_template_kwargs={"enable_thinking": False},
    )
    peft_config = LoraConfig(
        r=spec["lora"]["r"],
        lora_alpha=spec["lora"]["alpha"],
        lora_dropout=spec["lora"]["dropout"],
        target_modules=spec["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = MatchedRLCSDTrainer(
        model=spec["base_model"],
        reward_funcs=[__import__("rlcsd_data").conversational_reward(reward_correctness)],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=peft_config,
        rlcsd_spec=spec,
        correctness_fn=reward_correctness,
    )
    result = trainer.train()
    trainer.save_model(str(output))
    trainer.accelerator.wait_for_everyone()
    expected_updates = 2 if spec["mode"] == "smoke" else 100
    if int(trainer.state.global_step) != expected_updates:
        raise RuntimeError(
            f"terminal update mismatch: expected {expected_updates}, got {trainer.state.global_step}"
        )
    if trainer.is_world_process_zero():
        artifact = output / "adapter_model.safetensors"
        if not artifact.is_file():
            raise FileNotFoundError(f"trainer finished without adapter artifact: {artifact}")
        if spec["mode"] == "formal":
            missing = [
                str(output / f"checkpoint-{step}" / "adapter_model.safetensors")
                for step in spec["optimizer"]["save_steps"]
                if not (output / f"checkpoint-{step}" / "adapter_model.safetensors").is_file()
            ]
            if missing:
                raise FileNotFoundError("missing formal checkpoints: " + ", ".join(missing))
        done = {
            **reference,
            "PASS": True,
            "completed": time.time(),
            "optimizer_updates": int(trainer.state.global_step),
            "train_loss": float(result.training_loss),
            "artifacts": [str(artifact)],
        }
        temporary_done = output / ".DONE.json.tmp-rank0"
        temporary_done.write_text(json.dumps(done, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary_done, output / "DONE.json")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
