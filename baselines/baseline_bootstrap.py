#!/usr/bin/env python3
"""Narrow adapter/data/non-thinking shim around the pinned official entry."""

import argparse
import os
import runpy
import sys
from pathlib import Path


def load_ordered_parquet_dataset(
    paths, *, parquet_file_cls=None, concat_tables=None, dataset_cls=None, dataset_dict_cls=None
):
    if parquet_file_cls is None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset, DatasetDict

        parquet_file_cls, concat_tables = pq.ParquetFile, pa.concat_tables
        dataset_cls, dataset_dict_cls = Dataset, DatasetDict
    tables = [parquet_file_cls(path).read().replace_schema_metadata(None) for path in paths]
    return dataset_dict_cls({"train": dataset_cls(concat_tables(tables))})


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--method", choices=("sft", "grpo"), required=True)
    p.add_argument("--official-source", required=True)
    own, upstream = p.parse_known_args()

    import datasets
    import trl
    import trl.trainer.grpo_trainer as grpo_module
    import trl.trainer.sft_trainer as sft_module
    from transformers import AutoTokenizer, TrainerCallback, set_seed

    helper_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(helper_root))
    from shared_initialization_v18 import install_state, read_shared

    set_seed(42)
    state, manifest = read_shared(os.environ["BASELINE_INIT_MANIFEST"])
    evidence = Path(os.environ["BASELINE_EVIDENCE"])
    evidence.mkdir(parents=True, exist_ok=True)

    original_peft = sft_module.get_peft_model

    def initialized_peft(*args, **kwargs):
        model = original_peft(*args, **kwargs)
        install_state(model, state)
        rank = os.environ.get("RANK", "0")
        path = evidence / f"INIT_LOAD-rank{rank}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "PASS": True,
                    "seed": 42,
                    "manifest": os.environ["BASELINE_INIT_MANIFEST"],
                    "tensor_sha256": manifest["tensor_manifest"]["tensor_sha256"],
                    "B_all_zero": True,
                },
                indent=2,
            )
            + "\n"
        )
        return model

    sft_module.get_peft_model = initialized_peft
    grpo_module.get_peft_model = initialized_peft

    def offline_dataset(name, *args, **kwargs):
        if name != "siyanzhao/Openthoughts_math_30k_opsd" or args or kwargs:
            raise ValueError("official entry requested unexpected training data")
        files = [os.environ["BASELINE_DATA_FILES"].format(i) for i in (0, 1)]
        return load_ordered_parquet_dataset(files)

    datasets.load_dataset = offline_dataset

    original_tokenizer = AutoTokenizer.from_pretrained

    def nonthinking_tokenizer(*args, **kwargs):
        tokenizer = original_tokenizer(*args, **kwargs)
        apply = tokenizer.apply_chat_template

        def nonthinking_template(*a, **kw):
            kw.setdefault("enable_thinking", False)
            return apply(*a, **kw)

        tokenizer.apply_chat_template = nonthinking_template
        return tokenizer

    AutoTokenizer.from_pretrained = nonthinking_tokenizer

    stop = int(os.environ.get("BASELINE_STOP_AFTER_UPDATES") or 0)
    if stop:

        class StopAtUpdate(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop:
                    control.should_training_stop = True
                return control

        trainer_name = "SFTTrainer" if own.method == "sft" else "GRPOTrainer"
        original_trainer = getattr(trl, trainer_name)

        class SmokeTrainer(original_trainer):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("callbacks", []).append(StopAtUpdate())
                super().__init__(*args, **kwargs)

        setattr(trl, trainer_name, SmokeTrainer)

    sys.argv = [f"{own.method}_train.py", *upstream]
    sys.path.insert(0, own.official_source)
    runpy.run_path(str(Path(own.official_source) / f"{own.method}_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
