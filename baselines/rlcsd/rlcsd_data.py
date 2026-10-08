"""Pure helpers for RLCSD contrastive context construction."""

from __future__ import annotations

import hashlib
from functools import wraps
from typing import Any


def verify_rollout_prompt(tokenizer, messages, actual_ids):
    expected = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if list(actual_ids) != list(expected):
        raise ValueError(
            "ROLLOUT_PROMPT_CONTRACT_DRIFT: actual engine tokens differ from explicit nonthinking prompt"
        )
    return list(expected)


def conversational_reward(reward):
    """Adapt TRL chat completion containers, without changing author grading."""

    @wraps(reward)
    def wrapped(completions, *args, **kwargs):
        texts = []
        for completion in completions:
            if isinstance(completion, str):
                texts.append(completion)
            elif (
                isinstance(completion, list)
                and len(completion) == 1
                and completion[0].get("role") == "assistant"
                and isinstance(completion[0].get("content"), str)
            ):
                texts.append(completion[0]["content"])
            else:
                raise ValueError("unexpected non-text assistant completion")
        return reward(texts, *args, **kwargs)

    return wrapped


BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TEACHER_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you understand "
    "the reasoning behind each step.\n"
)


def extract_boxed_answer(text: str | None) -> str | None:
    """Extract the final nested ``\\boxed{...}``, matching the official RLCSD helper."""
    if text is None:
        return None
    start = str(text).rfind("\\boxed")
    if start < 0:
        return None
    left = str(text).find("{", start)
    if left < 0:
        return None
    depth = 0
    for index in range(left, len(str(text))):
        char = str(text)[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return str(text)[left + 1 : index].strip()
    return None


def build_student_message(problem: str) -> list[dict[str, str]]:
    # Byte-for-byte pinned T-series SelfDistillationDataCollator wording.
    return [{"role": "user", "content": f"Problem: {problem}\n\n{BOXED_INSTRUCTION}"}]


def build_teacher_message(problem: str, solution: str, answer: str) -> list[dict[str, str]]:
    """Build the official symmetric RLCSD solution+answer teacher wrapper."""
    if not str(solution).strip():
        raise ValueError("teacher solution hint must be non-empty")
    content = (
        f"Problem: {str(problem).strip()}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{str(solution).strip()}\n\n"
        f"Correct final answer: {str(answer).strip()}\n"
        "=== Reference Solution End ==="
        f"{TEACHER_TRANSITION}\n"
        f"{BOXED_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def choose_negative_siblings(
    group_records: list[dict[str, Any]],
    *,
    target_key: str,
    k: int,
    seed_material: str,
) -> list[dict[str, Any]]:
    """Choose up to K non-self verified-incorrect hints without global RNG use."""
    if k <= 0:
        raise ValueError("k must be positive")
    candidates = [
        item
        for item in group_records
        if item["key"] != target_key
        and not bool(item["correct"])
        and item.get("boxed_answer") is not None
    ]

    # Hash-ranking is deterministic, consumes no Python/Torch RNG, and gives a
    # step-varying sample when seed_material includes the optimizer step.
    def rank(item: dict[str, Any]) -> bytes:
        payload = f"{seed_material}\0{target_key}\0{item['key']}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    return sorted(candidates, key=rank)[:k]


def token_set_overlap(left: list[int], right: list[int]) -> float:
    """Jaccard overlap for diagnostic context only; does not touch training RNG/gradients."""
    a, b = set(left), set(right)
    return len(a & b) / max(len(a | b), 1)
