"""Apply offline semantic keep/drop decisions without changing negative tokens."""

import argparse
import json
from pathlib import Path


def select(library, decisions):
    by_id = {row["id"]: row for row in library}
    if len(by_id) != len(library):
        raise ValueError("negative IDs must be unique")
    reviewed = {}
    for decision in decisions:
        key = decision["id"]
        if key not in by_id or key in reviewed or type(decision["keep"]) is not bool:
            raise ValueError("unknown, duplicate or invalid review decision")
        reviewed[key] = decision["keep"]
    if set(reviewed) != set(by_id):
        raise ValueError("review every candidate before selecting the subset")
    selected = [row for row in library if reviewed[row["id"]]]
    if not selected:
        raise ValueError("semantic review retained no training negatives")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = [json.loads(line) for line in args.library.read_text().splitlines() if line.strip()]
    decisions = [
        json.loads(line) for line in args.decisions.read_text().splitlines() if line.strip()
    ]
    rows = select(library, decisions)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Retained {len(rows)} of {len(library)} negatives; prefixes and suffixes unchanged")


if __name__ == "__main__":
    main()
