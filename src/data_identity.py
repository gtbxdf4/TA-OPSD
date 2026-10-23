"""Portable identities for the frozen public training data and mining cohort."""

import hashlib
import json
import unicodedata


def json_digest(value):
    """Hash the canonical JSON representation used by public metadata."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def normalized_problem_hash(problem):
    """Apply the historical NFKC and whitespace-normalized problem identity."""
    normalized = " ".join(unicodedata.normalize("NFKC", str(problem)).split())
    return hashlib.sha256(normalized.encode()).hexdigest()
