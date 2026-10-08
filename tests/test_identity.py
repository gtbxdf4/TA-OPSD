"""Portable identities must detect changes in input order and selected rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_identity import json_digest, normalized_problem_hash, semantic_index_digest


def test_index_ignores_installation_root_but_preserves_row_position():
    left = {
        "pair": {
            "question_hash": "q",
            "source_row_ids": [0],
            "source_locations": [{"file": "/old/train.parquet", "row_index": 0}],
        }
    }
    right = {
        "pair": {
            "question_hash": "q",
            "source_row_ids": [0],
            "source_locations": [{"file": "/new/train.parquet", "row_index": 0}],
        }
    }
    assert semantic_index_digest(left) == semantic_index_digest(right)
    right["pair"]["source_row_ids"] = [1]
    assert semantic_index_digest(left) != semantic_index_digest(right)


def test_problem_and_token_identity_are_sensitive_to_content():
    assert normalized_problem_hash("A  B") == normalized_problem_hash("A B")
    assert json_digest([1, 2, 3]) != json_digest([1, 3, 2])
