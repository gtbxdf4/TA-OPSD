"""Independent small-tensor checks for the paper's two added objectives."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from independent_losses import local_unlikelihood, stop_kl


def test_stop_binary_kl_and_gradient():
    student = torch.tensor([[[0.3, 0.5, -0.1, 0.8]]], requires_grad=True)
    teacher = torch.tensor([[[0.8, -0.2, 0.4, 0.1]]])
    valid = torch.tensor([[True]])
    observed = stop_kl(student, teacher, valid, [0, 3])
    reference = student.detach().double().requires_grad_(True)
    student_stop = torch.softmax(reference[0, 0], dim=0)[[0, 3]].sum()
    teacher_stop = torch.softmax(teacher[0, 0].double(), dim=0)[[0, 3]].sum()
    expected = teacher_stop * (teacher_stop / student_stop).log()
    expected += (1 - teacher_stop) * ((1 - teacher_stop) / (1 - student_stop)).log()
    # Near zero, a float32 KL needs an absolute rounding-error bound.
    assert math.isclose(observed.item(), expected.item(), rel_tol=1e-6, abs_tol=2e-7)
    observed.backward()
    expected.backward()
    assert torch.allclose(student.grad.double(), reference.grad, rtol=1e-5, atol=2e-7)


def test_ul_only_selected_non_stop_suffix_position():
    logits = torch.tensor([[[1.0, -1.0, 0.0], [2.0, 0.0, -1.0]]], requires_grad=True)
    tokens = torch.tensor([[0, 2]])
    selected = torch.tensor([[True, False]])
    observed = local_unlikelihood(logits, tokens, selected, [2])
    probability = torch.softmax(logits[0, 0], dim=0)[0]
    expected = -torch.log1p(-probability)
    assert math.isclose(observed.item(), expected.item(), rel_tol=1e-6)
    observed.backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))
