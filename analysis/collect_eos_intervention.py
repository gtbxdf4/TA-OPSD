"""Summarize the paper's relative short-error cohort and paired EOS intervention."""

import argparse
import json
from pathlib import Path
from statistics import mean

from diagnostics.eos_intervention import (
    PAIR,
    SPEC,
    load_selection,
    paired_requests,
    read,
    read_results,
    standalone_requests,
)


def summarize(run_dir):
    cohort = read(run_dir / "cohort.json")["rows"]
    if [r["problem_hash"] for r in cohort] != [r["problem_hash"] for r in SPEC["rows"]]:
        raise ValueError("cohort does not match the published source-row order")
    ((_, base), (opsd_meta, opsd)), (denominator, relative, eligible) = load_selection(
        run_dir, cohort
    )
    meta, pairs = read_results(run_dir, "paired", paired_requests(eligible))
    if not opsd_meta["adapter_sha256"] or meta["adapter_sha256"] != opsd_meta["adapter_sha256"]:
        raise ValueError("standalone OPSD and FREE/ZERO adapter bytes differ")

    selected = [r["problem_hash"] for r in relative]
    arms = {"Base": base, "OPSD": opsd}
    for label, arm in (("OPSD+SC", "sc"), ("TA-OPSD", "ta-opsd")):
        _, rows = read_results(run_dir, arm, standalone_requests(relative, arm))
        arms[label] = {h: row for (h, _, _), row in rows.items()}
    standalone = {
        label: dict(
            n=len(selected),
            correct=sum(rows[h]["correct"] for h in selected),
            mean_tokens=mean(len(rows[h]["token_ids"]) for h in selected) if selected else None,
        )
        for label, rows in arms.items()
    }

    by_question = {}
    improved_lengths = {arm: [] for arm in PAIR}
    totals = {arm: dict(n=0, correct=0) for arm in PAIR}
    for row in eligible:
        h = row["problem_hash"]
        samples = {
            arm: [pairs[h, arm, i] for i in range(SPEC["seeds_per_problem"])] for arm in PAIR
        }
        counts = {arm: sum(r["correct"] for r in samples[arm]) for arm in PAIR}
        delta = counts["ZERO"] - counts["FREE"]
        by_question[h] = dict(correct=counts, delta=delta)
        for arm in PAIR:
            totals[arm]["n"] += len(samples[arm])
            totals[arm]["correct"] += counts[arm]
            if delta > 0:
                improved_lengths[arm].extend(
                    len(row["history_ids"]) + len(r["token_ids"]) for r in samples[arm]
                )
    improved = [h for h, r in by_question.items() if r["delta"] > 0]
    return dict(
        cohort_questions=len(cohort),
        base_correct_natural_stop=denominator,
        relative_short_errors=len(relative),
        intervention_eligible=len(eligible),
        relative_errors_without_observed_terminal_stop=len(relative) - len(eligible),
        standalone=standalone,
        continuation=totals,
        paired_questions=dict(
            improved=len(improved),
            tied=sum(r["delta"] == 0 for r in by_question.values()),
            declined=sum(r["delta"] < 0 for r in by_question.values()),
        ),
        improved_questions_mean_complete_tokens=(
            dict(
                Base=mean(len(base[h]["token_ids"]) for h in improved),
                OPSD=mean(len(opsd[h]["token_ids"]) for h in improved),
                **{arm: mean(improved_lengths[arm]) for arm in PAIR},
            )
            if improved
            else None
        ),
        by_question=by_question,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    output = args.run_dir / "summary.json"
    with output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
