# Development and scientific-code documentation

The GPU recipes target Linux, Python 3.10 and the versions in
`requirements-runtime.txt`. Ruff targets Python 3.10 with a 100-column line
length. The selected lint rules are `E4`, `E7`, `E9`, `F` and `I`; `E402` is
disabled because several entrypoints intentionally install runtime hooks before
importing upstream code. The two `observed_entry*.py` files also retain their
explicit import-order and unused-import exceptions. This configuration does not
enforce exhaustive docstring coverage or static type checking.

## Source boundaries

- Keep the OPSD dependency at the commit and file hashes in `UPSTREAM.json`.
  Fetch it with `scripts/fetch_upstream.py`; do not copy its training core or
  scorer into this repository or edit the fetched checkout to pass tests.
- Keep model/data revisions, common initialization, scientific coefficients,
  masks, causal shifts, reduction order and generation/evaluation settings
  unchanged unless the change explicitly concerns that protocol variable.
- Describe Stable-OPD and RLCSD as the matched adaptations implemented here.
  Their module documentation and `PAPER_ANALYSIS_MAP.md` record differences from
  the native recipes. Do not claim native-recipe equivalence from interface tests.
- Keep large model, training and raw evaluation payloads external. Identity
  hashes establish which inputs were used; they do not grant redistribution
  rights. See `NOTICE.md` and `ASSETS.md`.
- Candidate generation may reuse an output root only when its complete stable
  `CONTRACT.json` is identical, including cohort bytes, TP and tag. Changed
  prompts or settings require a fresh output root; preserve the previous raw
  records as evidence.

## Docstrings and comments

Each nontrivial module should state its role. Document public scientific APIs,
trainer wrappers and helpers whose contracts are easy to misuse. A descriptive
test name, a trivial CLI `main`, a one-line accessor or a direct JSON helper does
not need a comment that merely repeats its code.

For a tensor or objective API, describe the applicable parts of this contract:

1. Input/output shapes and units, including whether probabilities are represented
   as logits, log probabilities, fractions or percentages.
2. Causal alignment: which input position predicts each target, and whether
   tensors have already been shifted by the upstream trainer.
3. Mask meaning, padding/stop-token handling, empty-response behavior, and the
   order of token, response, rank and accumulation reductions.
4. Temperature placement, float conversions and numerical clamps. State when a
   helper expects already temperature-scaled scores or applies no scaling.
5. Gradient boundaries: frozen teacher/reference scores, detached prefix KV,
   trainable suffix paths, and whether measurements retain a graph.
6. Initialization/identity requirements, relevant validation errors, filesystem
   side effects and replacement or fresh-output semantics.

Use inline comments to explain a necessary mathematical or runtime decision,
such as why prefix KV is detached or why a low-precision cast remains separate.
Keep a paper equation reference only when it matches the actual operation.
Avoid repeating every assignment, adding speculative claims, or implying that
instrumentation, a completed optimizer update or a checkpoint proves efficacy.
When code changes, update its contract in the same patch.

The plotting reduction is intentionally separate from the training objective:
`analysis/collect.py` averages monitored signals over valid tokens; SC/UL average
positions within each nonempty response, then average responses. RLCSD keeps
zero-mask rows as zero contributions in its batch mean. Plot smoothing is a
display operation and does not replace raw values used in tables.

## Local CPU checks

Install a suitable PyTorch build first. These are the checks run by the current
CPU CI job, including pinned upstream acquisition and frozen aggregate rendering:

```bash
python -m pip install torch==2.8.0
python -m pip install -r requirements-dev.txt -r requirements-analysis.txt
python scripts/fetch_upstream.py
ruff check .
ruff format --check .
python -m compileall -q run.py src scripts baselines diagnostics analysis tests
pytest
python analysis/reproduce.py
```

Keep validation logs and the tested commit with a release review. For a
documentation-only patch, compare Python ASTs after removing only module,
class and function docstrings; this catches accidental executable changes that
format and lint checks cannot rule out. Also verify that configurations,
aggregate inputs and `UPSTREAM.json` are unchanged.

Passing these checks demonstrates the tested CPU interfaces, source formatting
and frozen-statistic figure/table generation in that environment. It does not
demonstrate a fresh GPU training/evaluation replay, distributed-runtime behavior,
model acquisition, or an independently verified scientific gain. Report the
actual Python/platform and any difference from the Python 3.10 Linux target.
