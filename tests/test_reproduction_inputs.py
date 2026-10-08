"""Check fresh continuation selection, semantic replay identity and raw result counts."""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.collect import evaluation, training_rows
from diagnostics.prepare import continuation_cases
from scripts.select_semantic_negatives import select


def test_semantic_selection_keeps_original_tokens():
    rows = [
        dict(id="a", negative=dict(prefix_ids=[1, 2], suffix_ids=[3])),
        dict(id="b", negative=dict(prefix_ids=[4], suffix_ids=[5])),
    ]
    kept = select(rows, [dict(id="a", keep=True), dict(id="b", keep=False)])
    assert kept == rows[:1] and kept[0] is rows[0]
    with pytest.raises(ValueError, match="review every"):
        select(rows, [dict(id="a", keep=True)])
    with pytest.raises(ValueError, match="duplicate"):
        select(rows, [dict(id="a", keep=True), dict(id="a", keep=False)])


def test_continuation_uses_this_runs_complete_failure_prefix(tmp_path):
    question = dict(problem_hash="q", teacher_prompt_token_ids=[1, 2], answer="7", seed=42)
    for branch, record in (
        ("student", dict(correct=False, token_ids=[3, 4, 151645], text="failed step")),
        ("teacher_privileged", dict(correct=True, token_ids=[6, 7, 8, 9])),
    ):
        path = tmp_path / f"standalone/1.7B/{branch}/q-seed0.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
    cases = continuation_cases([question], tmp_path)
    assert len(cases) == 1
    assert cases[0]["C_token_ids"] == [3, 4]
    assert cases[0]["C_text"] == "failed step"
    assert cases[0]["prompts"]["teacher_privileged"] == [1, 2, 3, 4]
    assert cases[0]["max_new_tokens"] == 32764


def test_evaluation_joins_raw_tokens_to_author_verdicts(tmp_path):
    raw, results = [], []
    for i in range(30):
        answers = []
        for j in range(12):
            text = f"answer {i}/{j}"
            raw.append(
                dict(
                    problem_index=i,
                    generation_index=j,
                    text=text,
                    token_ids=[1, 2],
                    finish_reason="length" if (i, j) == (0, 0) else "stop",
                )
            )
            answers.append(dict(full_generation=text, correct=j == 0, predicted_answer=str(j)))
        results.append(dict(generations=answers))
    (tmp_path / "author_result.json").write_text(
        json.dumps(dict(val_n=12, num_problems=30, results=results))
    )
    (tmp_path / "DONE.json").write_text(json.dumps(dict(PASS=True)))
    path = tmp_path / "raw_responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in raw))
    metric, _ = evaluation(tmp_path, "aime24", {i: True for i in range(30)})
    assert metric["correct"] == 30 and metric["n"] == 360
    assert metric["cap"] == pytest.approx(100 / 360)
    assert metric["short_errors"]["eligible"] == 330
    path.write_text("".join(json.dumps(row) + "\n" for row in raw[:-1]))
    with pytest.raises(ValueError, match="missing or duplicate"):
        evaluation(tmp_path, "aime24")


def test_training_aggregation_excludes_masked_positions_and_missing_ranks(tmp_path):
    root = tmp_path / "evidence/raw-monitor"
    root.mkdir(parents=True)
    record = dict(
        loss_valid_mask=[True, False],
        metrics=dict(
            topk_overlap_fraction=[0.75, 0.1],
            student_eos_logp={"151643": [-3, -99], "151645": [-5, -99]},
            eos_logp_gap_teacher_minus_student={"151643": [1, 90], "151645": [3, 90]},
        ),
    )
    for step in range(1, 101):
        for rank in range(4):
            for micro in range(2):
                packet = dict(
                    optimizer_step=step,
                    rank=rank,
                    microbatch_index=micro,
                    world_size=4,
                    records=[record] * 4,
                )
                with gzip.open(root / f"s{step}-r{rank}-m{micro}.json.gz", "wt") as stream:
                    json.dump(packet, stream)
    rows = training_rows(tmp_path, "T0")
    assert len(rows) == 100 and rows[0]["tokens"] == 32
    assert rows[0]["topk_overlap_fraction"] == 0.75
    assert rows[0]["student_eos_logp_151645"] == -5
    assert rows[0]["eos_gap_mean"] == 2
    (root / "s100-r3-m1.json.gz").unlink()
    with pytest.raises(ValueError, match="incomplete distributed"):
        training_rows(tmp_path, "T0")
