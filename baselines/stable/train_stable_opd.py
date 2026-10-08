#!/usr/bin/env python3
"""Train one isolated matched Stable-OPD arm from an explicit JSON spec."""

import argparse

from stable_opd import load_run_spec, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON object; no implicit paths or overrides")
    args = parser.parse_args()
    run_training(load_run_spec(args.spec))


if __name__ == "__main__":
    main()
