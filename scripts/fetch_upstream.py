#!/usr/bin/env python3
"""Fetch the pinned OPSD dependency and verify the files used by this package."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Install upstream without redistributing it as part of this repository."""
    spec = json.loads((ROOT / "UPSTREAM.json").read_text())
    destination = ROOT / "upstream/opsd"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(destination)], check=True)
        subprocess.run(["git", "checkout", "--detach", spec["commit"]], cwd=destination, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if commit != spec["commit"]:
        raise ValueError("existing upstream checkout has a different revision")
    for row in spec["files"]:
        path = destination / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("upstream file differs: " + row["path"])
    print(f"Verified OPSD {commit}: {len(spec['files'])} runtime files")


if __name__ == "__main__":
    main()
