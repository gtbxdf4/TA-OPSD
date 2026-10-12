"""Normalize AMC23 reference answers for the common mathematical scorer."""

import math


def answer_string(value):
    if isinstance(value, str):
        return value
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("nonfinite/non-numeric AMC answer")
    return str(value)


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
