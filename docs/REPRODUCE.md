# Reproducing the paper

Use the pinned environment and model revisions. Main training, baseline,
component-ablation and negative-construction recipes are printed by `run.py`.
The README gives the complete fresh TA-OPSD preparation/calibration/training
sequence; use separate output directories for each run.

## Components and negative-construction ablations

For OPSD+SC, run `calibrate_fresh.py --method OPSD_SC` on the initialization
smoke without `--negative-library`. For OPSD+UL, calibrate with
`--method OPSD_UL` and the same mechanical library used by TA-OPSD. The former
measures SC; the latter matches UL to the base OPSD LoRA gradient. Start the
formal arm with its own resulting manifest.

For Appendix E.1, first train the OPSD donor from the shared initialization.
Generate the same candidate cohort with `generate_candidates.py --adapter
DONOR_CHECKPOINT`, then mine it using the same rules/exclusions. Calibrate with
`--method TA_OPSD_trained_negatives`; the final variant again starts from the
shared initialization rather than from the donor weights.

For Appendix E.2, present each mechanical negative's decoded prefix and suffix
to the offline reviewer. Use the following selection instruction:

> Keep a suffix only when it repeats the same non-progressing or erroneous
> local reasoning step already present in its prefix. Drop repeated problem
> conditions, legitimate formula reuse and transformations that make progress.
> Judge the existing span; do not rewrite its prefix or suffix. Return one
> decision per candidate with its original ID, a boolean keep value and a brief
> reason.

Save the decisions as JSONL:

```json
{"id": "candidate-id", "keep": true, "reason": "same failed derivation recurs"}
```

Record the reviewer model/version and prompt alongside your run. Apply all
keep/drop decisions without altering the original negative records:

```bash
python scripts/select_semantic_negatives.py --library NEGATIVES.jsonl \
  --decisions REVIEW.jsonl --output SEMANTIC_NEGATIVES.jsonl
```

Calibrate that subset with `--method TA_OPSD_semantic_negatives` before formal
training. External review is offline; no reviewer runs during training or
inference. A new review or generation can select a different subset.

## Section 3 termination diagnosis

After the 1.7B OPSD training, reconstruct the 500 held-out questions from public
source rows and the pinned tokenizer:

```bash
python diagnostics/prepare.py cohort --model-path models/Qwen3-1.7B \
  --student-checkpoint runs/1.7B/OPSD/train/OPSD/checkpoint-100
```

Generate one standalone answer per question from the trained student and the
reference-conditioned frozen teacher. Each worker uses one caller-selected GPU;
`--rank`/`--workers` partitions questions if several GPUs are allocated:

```bash
CUDA_VISIBLE_DEVICES=0 python diagnostics/generate.py --mode standalone \
  --branch student --output runs/diagnosis
CUDA_VISIBLE_DEVICES=0 python diagnostics/generate.py --mode standalone \
  --branch teacher_privileged --output runs/diagnosis
```

Select teacher-correct/student-incorrect cases in which the student response
is shorter. Strip only trailing stop tokens and append the complete remaining
response to the teacher prompt:

```bash
python diagnostics/prepare.py continuation --raw-root runs/diagnosis
CUDA_VISIBLE_DEVICES=0 python diagnostics/generate.py --mode continuation \
  --branch teacher_privileged --output runs/diagnosis
python diagnostics/analyze.py --raw-root runs/diagnosis \
  --output runs/diagnosis/summary.json
```

Selection is derived from this run's outputs, not fixed to the historical count
of 93. Immediate stopping, continued reasoning and correctness of the combined
prefix+suffix are counted separately. `--input-dir` lets you keep each diagnosis
in a separate directory.

## Evaluation matrix and raw analysis

`configs/reproduction.json` enumerates the paper's main table, 1.7B component
curves, negative-construction table and training signals. Paths are relative to
that manifest. Evaluate the listed checkpoints/datasets with `src/evaluate.py`
and retain all 12 answers per question. Base evaluations are shared across the
step-zero component curves.

For Figure 3(a), run `diagnostics/benchmark_teacher.py` once on AIME24 using the
full reference solutions in the pinned parquet. The frozen teacher independently
solves each problem; these are distinct from continuations of student prefixes.

`analysis/collect.py` joins raw token records to author verdicts, rejects missing
or duplicate answers, counts length caps from the recorded finish reason, and
computes teacher-solvable short errors. A short error is incorrect, naturally
terminated within 1024 tokens, and free of the mechanical repetition predicate.
It aggregates complete DP4/DP8 training monitor shards at valid response
positions and records the effective backpropagated token budget, including UL
suffix tokens and excluding cached prefixes.

```bash
python analysis/collect.py --manifest configs/reproduction.json \
  --output-dir analysis/fresh-data
python analysis/reproduce.py --data-dir analysis/fresh-data \
  --output-dir figures/fresh
```

To export response examples, add entries to the manifest's `examples` list with
`case`, `method`, `path`, `problem_index` and `generation_index`. Select interesting
responses from the new run instead of assuming historical sample slots retain
the same behavior.

The released `analysis/data` aggregates redraw the original reported curves and
tables without model execution. Use the fresh collection route to analyze new
runs. Neither route promises identical stochastic outputs or scores.
