"""Call the pinned OPSD answer helpers without redistributing their implementation.

Only the two helper definitions are compiled from the hash-verified dependency;
its model-loading imports and evaluator CLI are not executed. Grading still
requires the runtime's math_verify package and retains the author's fallback.
"""

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _author_helpers():
    """Compile only the two helper definitions after validating the dependency hash."""
    source = ROOT / "upstream/opsd/eval/evaluate_math.py"
    expected = next(
        row["sha256"]
        for row in json.loads((ROOT / "UPSTREAM.json").read_text())["files"]
        if row["path"] == "eval/evaluate_math.py"
    )
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("pinned OPSD grader file differs; verify scripts/fetch_upstream.py")
    names = {"extract_boxed_answer", "grade_answer"}
    definitions = [
        node
        for node in ast.parse(content.decode()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise ValueError("pinned OPSD answer helpers are missing or duplicated")
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def extract_boxed_answer(text: str) -> str | None:
    """Return the author's last-box extraction result, including nested braces."""
    return _author_helpers()["extract_boxed_answer"](text)


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Call the author's verifier and fallback on already extracted answer strings."""
    from math_verify import parse, verify

    helpers = _author_helpers()
    helpers.update(parse=parse, verify=verify)
    return helpers["grade_answer"](predicted, ground_truth)
