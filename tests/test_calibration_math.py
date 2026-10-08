"""Check coefficient invariants independently of GPU model execution."""

import math
import sys
from pathlib import Path

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
