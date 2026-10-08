"""Frozen deterministic selection. No model/API calls and no external labels."""

import re
from collections import defaultdict

STOP = (151643, 151645)


def normalized(text):
    return " ".join(text.split())  # Keep numbers, signs and variable names.


def repeated_text_spans(text):
    """Locate the third occurrence of an exact normalized paragraph, line or sentence."""
    found = []
    patterns = (
        (r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", 40, "paragraph"),
        (r"[^\n]+", 40, "line"),
        (r"[^.!?。！？\n]+[.!?。！？]", 40, "sentence"),
    )
    for pattern, minimum, kind in patterns:
        seen = defaultdict(list)
        for match in re.finditer(pattern, text):
            raw = match.group()
            left = len(raw) - len(raw.lstrip())
            right = len(raw.rstrip())
            value = normalized(raw)
            if len(value) < minimum or sum(c.isalnum() for c in value) < 8:
                continue
            interval = (match.start() + left, match.start() + right)
            seen[value].append(interval)
            if len(seen[value]) == 3:
                found.append(
                    dict(
                        start=interval[0],
                        end=interval[1],
                        kind=kind,
                        occurrences=seen[value][:3],
                        rule="third_exact_copy",
                    )
                )
    unique = {(r["start"], r["end"]): r for r in found}
    return sorted(unique.values(), key=lambda r: (r["start"], -r["end"]))


def periodic_token_spans(ids):
    """Find a third consecutive token-pattern copy at the fixed candidate periods."""
    import numpy as np

    a = np.asarray(ids, dtype=np.int64)
    found = []
    for period in (4, 8, 16, 32, 64, 128):
        if len(a) < 3 * period:
            continue
        same = a[:-period] == a[period:]
        edges = np.diff(np.r_[False, same, False].astype(np.int8))
        for begin, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if end - begin >= max(2 * period, 24 - period):
                found.append((int(begin + 2 * period), int(begin + 3 * period)))
    return sorted(set(found))


def clean_token_ids(tok, ids, text, truncated=False):
    """Check token/text identity, allowing only a verified terminal decoding fragment."""
    ids = list(ids)

    def decode(values):
        return tok.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if ids and ids[-1] in STOP and decode(ids[:-1]) == text:
        ids = ids[:-1]
    # At a length cap, incremental decoding can omit one incomplete UTF-8 character.
    # Accept only an exact rendered prefix; never repair an interior mismatch.
    if truncated and decode(ids) == text + "\ufffd":
        for count in range(1, min(3, len(ids)) + 1):
            if decode(ids[:-count]) == text:
                ids = ids[:-count]
                break
    if decode(ids) != text:
        raise ValueError("raw token/text mismatch; do not repair labels")
    return ids


def interior_token_span(tok, ids, text, start, end):
    """Map a character interval to token boundaries contained wholly inside it."""

    # Moving inward avoids including a neighboring character split across tokens.
    def boundary(c, ceil):
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            n = len(
                tok.decode(ids[:mid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
            if n < c:
                lo = mid + 1
            else:
                hi = mid
        choices = []
        for k in range(max(0, lo - 8), min(len(ids), lo + 8) + 1):
            prefix = tok.decode(
                ids[:k], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if text.startswith(prefix) and (len(prefix) >= c if ceil else len(prefix) <= c):
                choices.append((len(prefix), k))
        if not choices:
            raise ValueError("no inward Unicode-safe boundary")
        return (min(choices) if ceil else max(choices))[1]

    a, b = boundary(start, True), boundary(end, False)
    if not 0 < a < b <= len(ids):
        raise ValueError("no nonempty predictable span")
    return a, b


def find_negative(tok, ids, text, prompt_length, truncated=False):
    """Select the earliest admissible local repetition, with at most 256 target tokens."""
    ids = clean_token_ids(tok, ids, text, truncated=truncated)
    spans = []
    for r in repeated_text_spans(text):
        try:
            a, b = interior_token_span(tok, ids, text, r["start"], r["end"])
        except ValueError:
            continue
        spans.append((a, b, r["rule"], r.get("occurrences")))
    for a, b in periodic_token_spans(ids):
        spans.append((a, b, "three_or_more_tandem_token_copies", None))
    for a, b, rule, occurrences in sorted(spans):
        # Keep the full history and reject spans that cannot fit the replay context.
        b = min(b, a + 256)
        if prompt_length + a + 256 > 20000 or b <= a or any(t in STOP for t in ids[a:b]):
            continue
        prefix = tok.decode(ids[:a], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        end = tok.decode(ids[:b], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        while b > a and not text.startswith(end):
            b -= 1
            end = tok.decode(ids[:b], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if (
            b <= a
            or not text.startswith(prefix)
            or sum(c.isalnum() for c in end[len(prefix) :]) < 4
        ):
            continue
        return dict(
            start_token=a,
            end_token=b,
            rule=rule,
            occurrences=occurrences,
            text=end[len(prefix) :],
            char_start=len(prefix),
            char_end=len(end),
        ), ids
    return None, ids


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(
        description="Frozen third-repeat mining without positive recovery or model judging."
    )
    p.add_argument("--raw", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--exclude-question-hashes", required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    exclude = (
        set(json.loads(Path(a.exclude_question_hashes).read_text()))
        if a.exclude_question_hashes
        else set()
    )
    out = []
    raw = Path(a.raw)
    records = (
        (json.loads(p.read_text()) for p in sorted(raw.glob("*.json")))
        if raw.is_dir()
        else (json.loads(line) for line in raw.read_text().splitlines())
    )
    for r in records:
        h = r.get("problem_hash", r.get("question_hash"))
        if h in exclude or r.get("split", "fit") != "fit":
            continue
        ids = r["token_ids"]
        prompt = r["prompt_token_ids"]
        span, ids = find_negative(
            tok, ids, r["text"], len(prompt), truncated=r.get("finish_reason") == "length"
        )
        if not span:
            continue
        start, end = span["start_token"], span["end_token"]
        # C contains the prompt and every response token before the selected copy.
        out.append(
            dict(
                id=r.get("id", "base-step0-" + str(h)),
                problem_sha256=h,
                approved=True,
                split=r.get("split", "fit"),
                negative=dict(
                    prefix_ids=prompt + ids[:start],
                    suffix_ids=ids[start:end],
                    selected=[True] * (end - start),
                ),
            )
        )
    with Path(a.output).open("x") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
