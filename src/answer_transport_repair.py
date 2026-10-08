"""AMC23 answer input normalization and post-generation engine release."""

import math
import os
from pathlib import Path


def answer_string(value):
    if isinstance(value, str):
        return value
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("nonfinite/non-numeric AMC answer")
    return str(value)


def released_class(original):
    class Released(original):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            self.llm_engine.engine_core.shutdown()
            return result

    return Released


def normalize_dataset(data):
    from datasets import Value

    features = data.features.copy()
    features["answer"] = Value("string")
    return data.map(
        lambda row: {"answer": answer_string(row["answer"])},
        features=features,
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def install():
    import datasets
    import observed_eval_entry as entry
    from eval_unit_v2 import sha256

    original_dataset = datasets.load_dataset
    original_class = entry.recorded_llm_class

    def transport(path, *args, **kwargs):
        data = original_dataset(path, *args, **kwargs)
        if os.environ.get("V17_EVAL_DATASET") == "amc23":
            data = normalize_dataset(data)
        return data

    def observed_class(original, context, rawpath, observedpath, attempt_uuid, runtime):
        runtime = dict(
            runtime,
            runtime_files={
                **runtime["runtime_files"],
                str(Path(__file__).resolve()): sha256(__file__),
            },
            answer_input_policy="AMC numeric answer -> exact str(value); problem/prompt untouched",
            release_policy="shutdown after one complete recorded generate; sampling unchanged",
        )
        return released_class(
            original_class(original, context, rawpath, observedpath, attempt_uuid, runtime)
        )

    datasets.load_dataset = transport
    entry.recorded_llm_class = observed_class
