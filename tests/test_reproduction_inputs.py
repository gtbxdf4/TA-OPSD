"""Check fresh continuation selection, semantic replay identity and raw result counts."""

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.collect import evaluation, training_rows
from diagnostics import grading
from diagnostics.prepare import continuation_cases
from scripts.select_semantic_negatives import select
from src import generate_candidates


def test_diagnostic_grading_uses_hash_verified_author_helpers(tmp_path, monkeypatch):
    grading._author_helpers.cache_clear()
    try:
        helpers = grading._author_helpers()
        source = grading.ROOT / "upstream/opsd/eval/evaluate_math.py"
        assert helpers["extract_boxed_answer"].__code__.co_filename == str(source)
        assert helpers["grade_answer"].__code__.co_filename == str(source)
        assert grading.extract_boxed_answer(r"\boxed{0} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert grading.extract_boxed_answer("unboxed") is None
        fake_verifier = SimpleNamespace(
            parse=lambda text, **kwargs: text,
            verify=lambda truth, prediction, timeout_seconds: truth == prediction,
        )
        monkeypatch.setitem(sys.modules, "math_verify", fake_verifier)
        assert grading.grade_answer("7", "7") is True

        original_spec = grading.ROOT / "UPSTREAM.json"
        (tmp_path / "UPSTREAM.json").write_bytes(original_spec.read_bytes())
        changed_source = tmp_path / "upstream/opsd/eval/evaluate_math.py"
        changed_source.parent.mkdir(parents=True)
        changed_source.write_text("raise RuntimeError('must never execute')\n")
        monkeypatch.setattr(grading, "ROOT", tmp_path)
        grading._author_helpers.cache_clear()
        with pytest.raises(ValueError, match="grader file differs"):
            grading.extract_boxed_answer("anything")
    finally:
        grading._author_helpers.cache_clear()


@pytest.mark.parametrize("changed", ["prompt", "cohort", "tp", "tag"])
def test_candidate_resume_rejects_changed_contract(tmp_path, monkeypatch, changed):
    cohort = tmp_path / "cohort.json"
    row = dict(problem_hash="same-question", split="fit", student_prompt_token_ids=[1])
    cohort.write_text(json.dumps([row]))
    output = tmp_path / "candidates"
    raw = output / "raw"
    raw.mkdir(parents=True)
    raw_path = raw / (generate_candidates.digest(row["problem_hash"]) + ".json")
    raw_path.write_text(
        json.dumps(dict(row, prompt_token_ids=[1], token_ids=[3], text="old response"))
    )
    argv = [
        "generate_candidates.py",
        "--out-root",
        str(output),
        "--tag",
        "base",
        "--model",
        "same-model",
        "--cohort",
        str(cohort),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    # Existing identical inputs can finish without loading a model or generating tokens.
    assert generate_candidates.main() == 0
    assert generate_candidates.main() == 0
    (output / "DONE.json").unlink()
    contract_bytes = (output / "CONTRACT.json").read_bytes()
    raw_bytes = raw_path.read_bytes()
    if changed in ("prompt", "cohort"):
        row["student_prompt_token_ids" if changed == "prompt" else "problem_hash"] = (
            [2] if changed == "prompt" else "other-question"
        )
        cohort.write_text(json.dumps([row]))
    elif changed == "tp":
        argv.extend(["--tp", "8"])
    else:
        argv[argv.index("--tag") + 1] = "other-tag"
    with pytest.raises(ValueError, match="contract mismatch"):
        generate_candidates.main()
    assert not (output / "DONE.json").exists()
    assert (output / "CONTRACT.json").read_bytes() == contract_bytes
    assert raw_path.read_bytes() == raw_bytes


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
    rows = training_rows(tmp_path, "OPSD")
    assert len(rows) == 100 and rows[0]["tokens"] == 32
    assert rows[0]["Method"] == "OPSD"
    assert rows[0]["topk_overlap_fraction"] == 0.75
    assert rows[0]["student_eos_logp_151645"] == -5
    assert rows[0]["eos_gap_mean"] == 2
    (root / "s100-r3-m1.json.gz").unlink()
    with pytest.raises(ValueError, match="incomplete distributed"):
        training_rows(tmp_path, "OPSD")


def test_monitor_keeps_only_paper_signals_without_changing_rng():
    torch = pytest.importorskip("torch")
    from src.raw_monitor_v18 import token_metrics

    torch.manual_seed(42)
    student, teacher = torch.randn(7, 256), torch.randn(7, 256)
    state = torch.get_rng_state().clone()
    values = token_metrics(student, teacher, torch.arange(7), 100, [20, 21], chunk_tokens=3)
    assert torch.equal(state, torch.get_rng_state())
    assert set(values) == {
        "topk_overlap_fraction",
        "student_eos_logp",
        "teacher_eos_logp",
        "eos_logp_gap_teacher_minus_student",
    }
    expected_student = student.log_softmax(-1)[:, [20, 21]]
    expected_teacher = teacher.log_softmax(-1)[:, [20, 21]]
    for column, token in enumerate((20, 21)):
        assert torch.allclose(
            torch.tensor(values["student_eos_logp"][str(token)]), expected_student[:, column]
        )
        assert torch.allclose(
            torch.tensor(values["teacher_eos_logp"][str(token)]), expected_teacher[:, column]
        )
        assert torch.allclose(
            torch.tensor(values["eos_logp_gap_teacher_minus_student"][str(token)]),
            (expected_teacher - expected_student)[:, column],
        )
    for row in range(7):
        left = set(student[row].topk(100).indices.tolist())
        right = set(teacher[row].topk(100).indices.tolist())
        assert values["topk_overlap_fraction"][row] == pytest.approx(len(left & right) / 100)
