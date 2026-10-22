"""Render paper figures and tables from frozen aggregates or collected fresh results."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

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


def curve(ax, x, y, color, label, smooth=False):
    """Draw observed points and optionally a smoothed visual trend.

    Smoothing uses a Gaussian filter followed by a natural cubic spline. It is
    display-only: exported table values and raw points are not recomputed from it.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert np.isfinite(y).all()
    if smooth:
        ax.plot(x, y, color=color, alpha=0.32, lw=0.8, marker="o", ms=2.3)
        trend = gaussian_filter1d(y, sigma=2.5, mode="reflect", truncate=4.0)
        dense = np.linspace(x[0], x[-1], 600)
        ax.plot(
            dense, CubicSpline(x, trend, bc_type="natural")(dense), color=color, lw=1.8, label=label
        )
    else:
        ax.plot(x, y, color=color, lw=1.7, marker="o", ms=4, label=label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel("Training step")


def save(fig, out, stem):
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (stem + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def training():
    """Load both methods' frozen 100-step training curves, requiring 32 responses per step."""
    result = {}
    for size in ("1.7B", "4B", "8B"):
        arms = {"OPSD": [], "TA-OPSD": []}
        with (DATA / f"training_{size}.csv").open() as f:
            for row in csv.DictReader(f):
                if row["Method"] in arms:
                    arms[row["Method"]].append(
                        {k: float(v) for k, v in row.items() if k != "Method"}
                    )
        for rows in arms.values():
            rows.sort(key=lambda r: r["step"])
            assert [int(r["step"]) for r in rows] == list(range(1, 101))
            assert all(int(r["responses"]) == 32 for r in rows)
        result[size] = arms
    return result


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
            assert value["datasets"][d]["sha256"] == curve_row["raw_sha256"]
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


def figures(out, generations, train):
    rows = train["1.7B"]["OPSD"]
    steps = [r["step"] for r in rows]
    student = np.array([r["student_eos_logp_151645"] for r in rows])
    teacher = student + np.array([r["eos_logp_gap_teacher_minus_student_151645"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.7, 4.1), constrained_layout=True)
    curve(ax, steps, student, "#007C9E", "Student", True)
    curve(ax, steps, teacher, "#CB7B00", "Privileged teacher", True)
    ax.set_ylabel("Mean log p(<|im_end|>)")
    ax.legend(frameon=False)
    save(fig, out, "figure2_eos_teacher_student")
    short = load("teacher_solvable_short_errors.json")["results"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), constrained_layout=True)
    for arm in ("OPSD", "TA-OPSD"):
        short_values = []
        for s in STEPS:
            r = short[str(s)][arm]
            close(r["pct"], 100 * r["eligible"] / r["total"])
            assert r["raw_sha256"] == generations["steps"][str(s)][arm]["aime24"]["raw_sha256"]
            short_values.append(r["pct"])
        curve(axes[0], STEPS, short_values, COLORS[arm], arm)
        curve(
            axes[1],
            STEPS,
            [macro(generations["steps"][str(s)][arm], "length") for s in STEPS],
            COLORS[arm],
            arm,
        )
        curve(
            axes[2],
            STEPS,
            [macro(generations["steps"][str(s)][arm], "cap") for s in STEPS],
            COLORS[arm],
            arm,
        )
    for ax, label in zip(
        axes,
        (
            "Teacher-solvable short errors (%)",
            "Mean response length (tokens)",
            "Mean capping rate (%)",
        ),
    ):
        ax.set_ylabel(label)
        ax.set_xticks(STEPS)
    axes[0].legend(frameon=False)
    save(fig, out, "figure3_generation_outcomes")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), constrained_layout=True)
    for col, size in enumerate(("1.7B", "4B", "8B")):
        for arm in ("OPSD", "TA-OPSD"):
            rows = train[size][arm]
            x = [r["step"] for r in rows]
            curve(
                axes[0, col],
                x,
                [100 * r["topk_overlap_fraction"] for r in rows],
                COLORS[arm],
                arm,
                True,
            )
            curve(axes[1, col], x, [r["eos_gap_mean"] for r in rows], COLORS[arm], arm, True)
        axes[0, col].set_title("Qwen3-" + size)
        axes[0, col].set_ylim(25, 100)
        axes[1, col].set_ylim(0, 25)
    axes[0, 0].set_ylabel("Top-100 overlap (%)")
    axes[1, 0].set_ylabel("Teacher − student EOS logp (nats)")
    axes[0, 0].legend(frameon=False)
    save(fig, out, "figure4_training_signals")
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
    p.add_argument("--data-dir", type=Path, default=DATA)
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
    train = training()
    counts = tables(out, data)
    figures(out, data, train)
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
        "gaussian_sigma_steps": 2.5,
        "figure3_endpoint": 100,
        "table3_checkpoint": 75,
        "figure3_step100_macro": {
            a: {k: macro(data["steps"]["100"][a], k) for k in ("acc", "length", "cap")}
            for a in ("OPSD", "TA-OPSD")
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in DATA.iterdir()
            if p.is_file()
        },
    }
    (out / "REPRODUCTION_RESULT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
