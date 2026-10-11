"""Check coefficient invariants independently of GPU model execution."""

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibrate_fresh import norm_ratio


def test_unit_ratio_and_invalid_norms():
    assert norm_ratio(4.0, 2.0) == 2.0
    for pair in ((0, 2), (1, 0), (math.nan, 1), (2, math.inf)):
        try:
            norm_ratio(*pair)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid norm pair accepted: {pair}")


def test_calibration_uses_pinned_author_loss_with_matching_value_and_gradient():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from independent_losses import stop_kl

    root = Path(__file__).resolve().parents[1]
    source = root / "upstream/opsd/opsd_trainer.py"
    expected = next(
        row["sha256"]
        for row in json.loads((root / "UPSTREAM.json").read_text())["files"]
        if row["path"] == "opsd_trainer.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    author = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "OPSDTrainer"
    )
    loss = next(
        node
        for node in author.body
        if isinstance(node, ast.FunctionDef) and node.name == "generalized_jsd_loss"
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[loss], type_ignores=[]), str(source), "exec"), namespace)
    base = type("AuthorLoss", (), {"generalized_jsd_loss": namespace[loss.name]})

    # Execute the actual nested override without loading a model or a GPU trainer.
    calibration = root / "src/calibrate_fresh.py"
    nested = next(
        node
        for node in ast.walk(ast.parse(calibration.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "CalibrationTrainer"
    )
    namespace.update(OPSDTrainer=base, stop_kl=stop_kl, STOP_IDS=[20, 21])
    exec(compile(ast.Module(body=[nested], type_ignores=[]), str(calibration), "exec"), namespace)
    trainer = object.__new__(namespace[nested.name])
    trainer.collect_logits = True
    trainer.stop_coefficient = 0.25
    trainer.base_sq = trainer.stop_sq = 0.0
    torch.manual_seed(42)
    student = (torch.randn(2, 3, 256) * 0.01).requires_grad_()
    teacher = student.detach() + torch.randn_like(student) * 0.001
    labels = torch.tensor([[7, 8, -100], [9, 10, 11]])
    options = dict(beta=0, temperature=1.1, top_k=0, token_clip=1e-6)
    observed = trainer.generalized_jsd_loss(student, teacher, labels, **options)
    reference = base.generalized_jsd_loss(student, teacher, labels, **options)
    reference += 0.25 * stop_kl(student, teacher, labels != -100, [20, 21])
    observed_gradient = torch.autograd.grad(observed, student, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, student)[0]
    assert torch.equal(observed, reference)
    assert torch.equal(observed_gradient, reference_gradient)
    assert trainer.base_sq > 0 and trainer.stop_sq > 0
