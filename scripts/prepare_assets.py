#!/usr/bin/env python3
"""Acquire or import external paper assets with exact byte-hash validation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = {
    "assets/training/train-00000-of-00002.parquet": (
        "siyanzhao/Openthoughts_math_30k_opsd",
        "data/train-00000-of-00002.parquet",
    ),
    "assets/training/train-00001-of-00002.parquet": (
        "siyanzhao/Openthoughts_math_30k_opsd",
        "data/train-00001-of-00002.parquet",
    ),
    "assets/evaluation/aime24.parquet": (
        "HuggingFaceH4/aime_2024",
        "data/train-00000-of-00001.parquet",
    ),
    "assets/evaluation/aime25.parquet": (
        "yentinglin/aime_2025",
        "data/train-00000-of-00001-243207c6c994e1bd.parquet",
    ),
    "assets/evaluation/amc23.parquet": ("math-ai/amc23", "data/test-00000-of-00001.parquet"),
}


def verify(path, expected):
    """Reject an asset whose bytes differ from the frozen paper manifest."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"asset SHA256 mismatch: {path}")


def main():
    """Copy licensed local assets or download an upstream dataset selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--group",
        choices=(
            "training",
            "evaluation",
            "initialization",
            "negatives",
            "mining",
            "diagnostics",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--revision", help="Upstream dataset revision; exact SHA is checked independently."
    )
    args = parser.parse_args()
    if not args.source_dir and not args.download:
        parser.error("choose --source-dir or --download")
    rows = json.loads((ROOT / "ASSETS_MANIFEST.json").read_text())["files"]
    copied = 0
    for row in rows:
        relative = Path(row["path"])
        group = relative.parts[1] if relative.parts[0] == "assets" else "diagnostics"
        if args.group != "all" and group != args.group:
            continue
        if args.download and row["path"] not in DOWNLOADS:
            continue
        destination = ROOT / relative
        if destination.exists():
            verify(destination, row["sha256"])
            continue
        if args.download:
            from huggingface_hub import hf_hub_download

            repository, filename = DOWNLOADS[row["path"]]
            source = Path(
                hf_hub_download(repository, filename, repo_type="dataset", revision=args.revision)
            )
        else:
            source = args.source_dir / relative
        verify(source, row["sha256"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    print(f"Imported {copied} assets; missing frozen libraries remain an explicit prerequisite.")


if __name__ == "__main__":
    main()
