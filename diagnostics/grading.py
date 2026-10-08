"""Unchanged OPSD last-box extraction and math_verify grading functions."""


def extract_boxed_answer(text: str) -> str | None:
    """Return the last boxed answer, including nested braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]

    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        answer = boxed_str[7:-1]
        return answer.strip()

    return None


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Use the OPSD math verifier and its original string fallback."""
    from math_verify import parse, verify

    if predicted is None:
        return False

    try:
        # math_verify parses expressions in a LaTeX math environment.
        if "$" not in predicted:
            predicted = f"${predicted}$"
        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"

        pred_parsed = parse(predicted, fallback_mode="no_fallback")
        gt_parsed = parse(ground_truth, fallback_mode="no_fallback")

        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm
