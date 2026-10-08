"""Check user-facing recipes against the paper's recorded training layouts."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run


def option(arguments, name):
    """Read a scalar argument from a generated command."""
    return arguments[arguments.index(name) + 1]


def test_all_reported_recipes_plan_with_fresh_output():
    """Each reported size/method resolves without requiring a private path."""
    matrix = {
        "1.7B": (
            "OPSD",
            "OPSD_SC",
            "OPSD_UL",
            "TA_OPSD",
            "SFT",
            "GRPO",
            "Stable_OPD",
            "RLCSD",
            "TA_OPSD_trained_negatives",
            "TA_OPSD_semantic_negatives",
        ),
        "4B": ("OPSD", "TA_OPSD", "SFT", "GRPO", "Stable_OPD", "RLCSD"),
        "8B": ("OPSD", "TA_OPSD", "SFT", "GRPO", "Stable_OPD", "RLCSD"),
    }
    for size, methods in matrix.items():
        for method in methods:
            plan = run.plan(size, method, f"/tmp/new/{size}/{method}", f"/tmp/model/{size}")
            assert plan["output"].endswith(f"/{size}/{method}")
            portable = (
                json.dumps(plan)
                .replace(str(run.ROOT), "<repo>")
                .replace(sys.executable, "<python>")
            )
            assert "/home/" not in portable
            if method.startswith(("OPSD", "TA_OPSD")):
                assert plan["environment"]["V18_MONITOR_ARM"] == method


def test_actual_baseline_data_parallel_layout_and_horizons():
    """The 4B baselines used DP8 even though 4B OPSD used DP4."""
    expected = {"1.7B": 4, "4B": 8, "8B": 8}
    for size, dp in expected.items():
        for method in ("SFT", "GRPO"):
            planned = run.plan(size, method, "/tmp/fresh", "/tmp/model")
            command = planned["command"]
            assert int(option(command, "--num_processes")) == dp
            assert int(option(command, "--gradient_accumulation_steps")) == 32 // dp
            assert int(option(command, "--max_steps")) == (100 if method == "SFT" else 500)
            assert int(option(command, "--per_device_train_batch_size")) == 1
    for size, dp in (("1.7B", 4), ("4B", 4), ("8B", 8)):
        planned = run.plan(size, "TA_OPSD", "/tmp/fresh", "/tmp/model")
        assert int(option(planned["command"], "--num_processes")) == dp
        assert int(option(planned["command"], "--max_steps")) == 100


def test_smoke_keeps_the_formal_horizon():
    """The engineering gate stops at two updates without shortening the LR schedule."""
    opsd = run.plan("1.7B", "TA_OPSD", "/tmp/smoke", "/tmp/model", mode="smoke")
    assert opsd["environment"]["V17_SMOKE_STOP"] == "2"
    assert int(option(opsd["command"], "--max_steps")) == 100
    grpo = run.plan("4B", "GRPO", "/tmp/smoke", "/tmp/model", mode="smoke")
    assert grpo["environment"]["BASELINE_STOP_AFTER_UPDATES"] == "2"
    assert int(option(grpo["command"], "--max_steps")) == 500


def test_fresh_bank_coefficients_are_bound_to_their_exact_bytes(tmp_path):
    """An independently mined library must carry its measured coefficients."""
    library = tmp_path / "negative.jsonl"
    library.write_text('{"approved":true,"negative":{}}\n')
    manifest = tmp_path / "auxiliary.json"
    manifest.write_text(
        json.dumps(
            {
                "PASS": True,
                "method": "TA_OPSD",
                "size": "1.7B",
                "world_size": 4,
                "stop_ids": [151643, 151645],
                "lambda_eos": 2.0,
                "lambda_ul": 0.01,
                "assets": str(library),
                "assets_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        )
    )
    planned = run.plan(
        "1.7B", "TA_OPSD", tmp_path / "fresh", "/tmp/model", auxiliary_manifest=manifest
    )
    assert planned["runtime_files"]["auxiliary.json"]["lambda_eos"] == 2.0
    library.write_text("changed\n")
    try:
        run.plan("1.7B", "TA_OPSD", tmp_path / "fresh2", "/tmp/model", auxiliary_manifest=manifest)
    except ValueError as error:
        assert "bytes changed" in str(error)
    else:
        raise AssertionError("tampered negative library accepted")
