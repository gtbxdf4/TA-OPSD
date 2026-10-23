#!/usr/bin/env python3
"""Fetch the pinned OPSD dependency and verify the files used by this package."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    "url": "https://github.com/siyan-zhao/OPSD",
    "commit": "ae7d2519e94920c4eb6206c0c26de46d9c50abae",
    "files": [
        {
            "path": "opsd_train.py",
            "sha256": "524edd3f9a5ee9ad1496410dd9c51019301e8e1ab72289f5982b34adaf81f26e",
        },
        {
            "path": "opsd_trainer.py",
            "sha256": "aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d",
        },
        {
            "path": "data_collator.py",
            "sha256": "5b96f3b3ae2f04e2b9dbf03509d3cd723c1b637d9362ab4697135d9f885b8381",
        },
        {
            "path": "sft_train.py",
            "sha256": "a8d4e7988a527315482f9fcfd42cc4b350a4b847579fa294f78717b4db82727c",
        },
        {
            "path": "grpo_train.py",
            "sha256": "cd4e5e9714f68a902d61392907c78916979e9131bf1b9a471bae95b8c636dde4",
        },
        {
            "path": "accelerate.yaml",
            "sha256": "9ea562ab6c1f0492894ad3f986a3515c1e3371b4b0956bb7376762af8c39bf7b",
        },
        {
            "path": "environment.yml",
            "sha256": "8d4b2a67762ffa085dda408d44b79b9ad9f5ed573d8692a4048dcc1f9bdea01e",
        },
        {
            "path": "eval/evaluate_math.py",
            "sha256": "d1c10fb407461ae8f989c94544a252e9a00776e405e55848d0d4b7ebfd2d6076",
        },
    ],
}


def main():
    """Install upstream without redistributing it as part of this repository."""
    spec = UPSTREAM
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
    # The matched Stable-OPD validator consumes this marker after checking core hashes.
    (destination / "SOURCE_COMMIT").write_text(commit + "\n")
    print(f"Verified OPSD {commit}: {len(spec['files'])} runtime files")


if __name__ == "__main__":
    main()
