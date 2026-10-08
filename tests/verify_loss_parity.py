"""Compare new source with the archived paper loss implementation on CPU."""

import argparse
import importlib.util
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    import torch

    current = load_module(
        "public_losses", Path(__file__).resolve().parents[1] / "src/independent_losses.py"
    )
    old = load_module("archived_losses", args.reference)
    torch.manual_seed(42)
    teacher = torch.randn(2, 4, 256)
    labels = torch.tensor([[True, True, False, True], [True, False, True, True]])
    tokens = torch.tensor([[8, 19, 20, 4], [3, 99, 20, 7]])
    selected = torch.tensor([[True, False, True, True], [False, True, True, True]])
    for name, inputs in (
        ("stop_kl", (teacher, labels, [20, 21])),
        ("local_unlikelihood", (tokens, selected, [20, 21])),
    ):
        student_new = torch.randn(2, 4, 256, requires_grad=True)
        student_old = student_new.detach().clone().requires_grad_(True)
        new = getattr(current, name)(student_new, *inputs)
        previous = getattr(old, name)(student_old, *inputs)
        new.backward()
        previous.backward()
        if not torch.equal(new.detach(), previous.detach()) or not torch.equal(
            student_new.grad, student_old.grad
        ):
            raise AssertionError(f"{name} value or student gradient changed")
        print(name, "BITWISE_VALUE_AND_GRADIENT_PASS")


if __name__ == "__main__":
    main()
