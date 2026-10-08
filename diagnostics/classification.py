"""Original mechanical repetition predicate; no candidate-selection CLI."""

import collections


def repeat(text):
    lines = [" ".join(line.split()) for line in text.splitlines()]
    counts = collections.Counter(
        line for line in lines if len(line) >= 40 and sum(x.isalnum() for x in line) >= 8
    )
    line = max(counts.values(), default=0) >= 3
    tail = text[-480:]
    periodic = any(
        len(tail) >= 3 * p and tail[p:] == tail[:-p] for p in range(1, min(160, len(tail) // 3) + 1)
    )
    return line or periodic


def short_error(response):
    return (
        not response["correct"]
        and response["finish_reason"] == "stop"
        and len(response["token_ids"]) <= 1024
        and not repeat(response["text"])
    )
