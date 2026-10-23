"""Render evaluation tables and generation curves from collected fresh results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATASETS = ("aime24", "aime25", "amc23")
STEPS = (0, 25, 50, 75, 100)
COLORS = {"OPSD": "#5B3B9E", "OPSD+SC": "#007C9E", "OPSD+UL": "#CB7B00", "TA-OPSD": "#B2183A"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
    }
)


def load(name):
    return json.loads((DATA / name).read_text())


def csv_export(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def macro(records, field):
    return sum(records[d][field] for d in DATASETS) / len(DATASETS)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-10), (a, b)


def curve(ax, x, y, color, label):
    """Connect evaluated points without smoothing or additional observations."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert np.isfinite(y).all()
    ax.plot(x, y, color=color, lw=1.7, marker="o", ms=4, label=label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel("Training step")


def save(fig, out, stem):
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (stem + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def tables(out, generations):
    audit = load("table1_raw_audit.json")
    grouped = defaultdict(dict)
    for cell in audit["cells"]:
        assert cell["dataset"] in DATASETS
        value = 100 * cell["correct"] / cell["total"]
        close(value, cell["accuracy_pct"])
        method = cell["method"]
        group = grouped[(cell["size"], method)]
        assert cell["dataset"] not in group
        group[cell["dataset"]] = value
    table1 = []
    for (size, method), values in grouped.items():
        assert set(values) == set(DATASETS)
        table1.append(
            {
                "model_size": size,
                "Method": method,
                **{d: values[d] for d in DATASETS},
                "mean_accuracy_pct": sum(values.values()) / 3,
            }
        )
    csv_export(out / "table1_accuracy.csv", table1)
    table2 = []
    for arm in ("OPSD", "OPSD+SC", "OPSD+UL", "TA-OPSD"):
        row = generations["steps"]["75"][arm]
        table2.append(
            {
                "Method": arm,
                "checkpoint": 75,
                **{d: row[d]["acc"] for d in DATASETS},
                "mean_accuracy_pct": macro(row, "acc"),
            }
        )
    csv_export(out / "table2_components_step75.csv", table2)
    for arm in ("OPSD", "TA-OPSD"):
        for d in DATASETS:
            close(grouped[("1.7B", arm)][d], generations["steps"]["75"][arm][d]["acc"])
    negative = load("table3_negative_selection_step75.json")
    assert negative["comparison_checkpoint"] == 75
    table3 = []
    for method, result in negative["comparison"].items():
        for d in DATASETS:
            r = result["datasets"][d]
            close(r["accuracy"], 100 * r["correct"] / r["n"])
        accuracy = sum(result["datasets"][d]["accuracy"] for d in DATASETS) / 3
        length = sum(result["datasets"][d]["mean_length"] for d in DATASETS) / 3
        cap = sum(result["datasets"][d]["capping_rate"] for d in DATASETS) / 3
        close(accuracy, result["macro"]["accuracy"])
        close(length, result["macro"]["mean_length"])
        close(cap, result["macro"]["capping_rate"])
        table3.append(
            {
                "Method": method,
                "checkpoint": 75,
                "mean_accuracy_pct": accuracy,
                "mean_length_tokens": length,
                "mean_capping_pct": cap,
            }
        )
    csv_export(out / "table3_negative_construction_step75.csv", table3)
    for method, arm in (("OPSD", "OPSD"), ("TA-OPSD", "TA-OPSD")):
        value = negative["comparison"][method]
        for d in DATASETS:
            curve_row = generations["steps"]["75"][arm][d]
            close(value["datasets"][d]["accuracy"], curve_row["acc"])
            close(value["datasets"][d]["mean_length"], curve_row["length"])
            close(value["datasets"][d]["capping_rate"], curve_row["cap"])
    sections = []
    for title, rows in (
        ("Table 1: accuracy (%)", table1),
        ("Table 2: components at step 75", table2),
        ("Table 3: negative construction at step 75", table3),
    ):
        columns = list(rows[0])
        text = [
            "## " + title,
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in rows:
            text.append(
                "| "
                + " | ".join(
                    f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in columns
                )
                + " |"
            )
        sections.append("\n".join(text))
    (out / "tables.md").write_text("\n\n".join(sections) + "\n")
    return {"table1_rows": len(table1), "table2_rows": len(table2), "table3_rows": len(table3)}


def figures(out, generations):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for arm in ("OPSD", "TA-OPSD"):
        for ax, field in zip(axes, ("length", "cap")):
            curve(
                ax,
                STEPS,
                [macro(generations["steps"][str(s)][arm], field) for s in STEPS],
                COLORS[arm],
                arm,
            )
    axes[0].set_ylabel("Mean response length (tokens)")
    axes[1].set_ylabel("Mean capping rate (%)")
    for ax in axes:
        ax.set_xticks(STEPS)
    axes[0].legend(frameon=False)
    save(fig, out, "generation_outcomes")
    fig, ax = plt.subplots(figsize=(5.1, 3.3), constrained_layout=True)
    cap_rows = []
    for arm in ("OPSD", "OPSD+SC", "OPSD+UL", "TA-OPSD"):
        y = [macro(generations["steps"][str(s)][arm], "cap") for s in STEPS]
        curve(ax, STEPS, y, COLORS[arm], arm)
        cap_rows += [
            {"checkpoint": s, "Method": arm, "mean_capping_pct": cap} for s, cap in zip(STEPS, y)
        ]
    ax.set_ylabel("Mean capping rate (%)")
    ax.set_xticks(STEPS)
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "figure5_component_capping")
    csv_export(out / "figure5_component_capping.csv", cap_rows)


def main():
    global DATA
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "output")
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    DATA = args.data_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = load("generation_curves_1p7b.json")
    for step in data["steps"].values():
        for arm in step.values():
            for d in DATASETS:
                close(arm[d]["acc"], 100 * arm[d]["correct"] / arm[d]["n"])
                assert arm[d]["n"] == (480 if d == "amc23" else 360)
    counts = tables(out, data)
    figures(out, data)
    examples = load("response_examples.json")
    case_rows = [
        {
            "case": name,
            "Method": r["method"],
            **{
                k: r[k]
                for k in (
                    "problem_index",
                    "generation_index",
                    "tokens",
                    "finish_reason",
                    "correct",
                    "predicted_answer",
                )
            },
        }
        for name, records in examples["cases"].items()
        for r in records
    ]
    if case_rows:
        csv_export(out / "response_examples.csv", case_rows)
    summary = {
        "PASS": True,
        "validation": "CPU aggregate consistency and rendering; training/inference are separate",
        **counts,
        "response_examples": len(case_rows),
        "generation_endpoint": 100,
        "table3_checkpoint": 75,
        "generation_step100_macro": {
            a: {k: macro(data["steps"]["100"][a], k) for k in ("acc", "length", "cap")}
            for a in ("OPSD", "TA-OPSD")
        },
    }
    (out / "REPRODUCTION_RESULT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
