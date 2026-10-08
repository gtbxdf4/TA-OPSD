"""Portable identities for the frozen public training data and mining cohort."""

import hashlib
import json
import unicodedata
from pathlib import Path


def json_digest(value):
    """Hash the canonical JSON representation used by public metadata."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def normalized_problem_hash(problem):
    """Apply the historical NFKC and whitespace-normalized problem identity."""
    normalized = " ".join(unicodedata.normalize("NFKC", str(problem)).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def semantic_index_digest(index):
    """Ignore installation paths while preserving every row and duplicate identity."""
    portable = {
        key: {
            "question_hash": row["question_hash"],
            "source_row_ids": row["source_row_ids"],
            "source_locations": [
                {"file": Path(location["file"]).name, "row_index": location["row_index"]}
                for location in row["source_locations"]
            ],
        }
        for key, row in index.items()
    }
    return json_digest(portable)


def validate_index(path, expected):
    """Validate a regenerated index and return its actual installation-specific hash."""
    path = Path(path)
    if semantic_index_digest(json.loads(path.read_text())) != expected:
        raise ValueError("training question index semantic identity differs")
    return hashlib.sha256(path.read_bytes()).hexdigest()
