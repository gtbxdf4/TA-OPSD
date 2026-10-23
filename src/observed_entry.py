"""Local data loading around the unedited author entrypoint.

No sampler, RNG, forward, loss, optimizer, generation argument or return value is changed.
The smoke-only callback stops after two optimizer steps on the same 100-step scheduler.
"""

import json
import os
from pathlib import Path
import runpy
import sys

SOURCE = Path(os.environ["TA_OPSD_SOURCE"])
EVIDENCE = Path(os.environ["TA_OPSD_EVIDENCE"])
EVIDENCE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))

from transformers import TrainerCallback
import opsd_trainer
import datasets

# Only the transport path changes: these are freshly downloaded author parquet bytes.
original_load_dataset = datasets.load_dataset


def local_author_dataset(path, *args, **kwargs):
    """Load ordered local author parquet shards for the one frozen dataset name."""
    if path == "siyanzhao/Openthoughts_math_30k_opsd":
        files = json.loads(os.environ["PAPER_TRAIN_FILES"])
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = [pq.ParquetFile(p).read().replace_schema_metadata(None) for p in files]
        return datasets.DatasetDict({"train": datasets.Dataset(pa.concat_tables(tables))})
    return original_load_dataset(path, *args, **kwargs)


datasets.load_dataset = local_author_dataset


class Progress(TrainerCallback):
    """Honor the smoke-only early-stop callback."""

    def on_step_end(self, args, state, control, **kwargs):
        limit = int(os.environ.get("TA_OPSD_SMOKE_STOP", "0"))
        if limit and state.global_step >= limit:
            control.should_training_stop = True
            control.should_save = True
        return control


OriginalTrainer = opsd_trainer.OPSDTrainer


class ObservedTrainer(OriginalTrainer):
    """Add only the smoke-stop callback to the upstream trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(Progress())


opsd_trainer.OPSDTrainer = ObservedTrainer
runpy.run_path(str(SOURCE / "opsd_train.py"), run_name="__main__")
