# TA-OPSD

Code for **Learning When to Continue and When to Stop: Termination-Aware OPSD**.

TA-OPSD augments on-policy self-distillation with two auxiliary objectives:
**stop/continue distillation (SC)** aligns the teacher and student on stopping
probability, and **repetition-aware unlikelihood (UL)** suppresses local loops
under their original failure prefixes. Inference uses the trained student alone.

```text
loss = OPSD + lambda_SC * SC + lambda_UL * UL
```

This repository contains the paper's Qwen3-1.7B/4B/8B recipes, matched baselines,
component and negative-construction ablations, termination diagnostics, and
figure/table code. Model weights, training datasets and raw experiment outputs
are downloaded or generated separately.

## Experimental results

The following values are copied from the paper. Accuracy and capping rates are
percentages; response length is measured in generated tokens. Average accuracy
is the arithmetic mean over AIME24, AIME25 and AMC23.

### Main results

Avg@12 on the three mathematical reasoning benchmarks:

| Model | Method | AIME24 | AIME25 | AMC23 | Average |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 13.61 | 8.61 | 46.04 | 22.75 |
| | SFT | 8.89 | 8.89 | 41.04 | 19.61 |
| | GRPO | 12.78 | 9.17 | 45.00 | 22.31 |
| | OPSD | 13.89 | 9.44 | 45.62 | 22.99 |
| | Stable-OPD | 9.72 | 11.11 | 47.71 | 22.85 |
| | RLCSD | 15.83 | 9.44 | 47.29 | 24.19 |
| | **TA-OPSD (Ours)** | **18.33** (+4.44) | **13.89** (+4.44) | **51.25** (+5.62) | **27.82** (+4.84) |
| Qwen3-4B | Base | 23.89 | 21.39 | 66.04 | 37.11 |
| | SFT | 16.67 | 14.17 | 56.46 | 29.10 |
| | GRPO | 24.17 | 19.72 | 66.46 | 36.78 |
| | OPSD | 19.44 | 17.78 | 70.42 | 35.88 |
| | Stable-OPD | 23.06 | 20.00 | 65.83 | 36.30 |
| | RLCSD | 21.67 | 19.17 | 68.54 | 36.46 |
| | **TA-OPSD (Ours)** | **25.56** (+6.11) | **22.22** (+4.44) | **71.04** (+0.62) | **39.61** (+3.73) |
| Qwen3-8B | Base | 29.72 | 19.17 | 68.12 | 39.00 |
| | SFT | 14.17 | 11.39 | 52.29 | 25.95 |
| | GRPO | 29.72 | 21.67 | 69.58 | 40.32 |
| | OPSD | 44.44 | 31.11 | 82.08 | 52.55 |
| | Stable-OPD | 25.83 | 19.44 | 66.67 | 37.31 |
| | RLCSD | 28.61 | 21.39 | 69.58 | 39.86 |
| | **TA-OPSD (Ours)** | **45.00** (+0.56) | **33.89** (+2.78) | **82.71** (+0.62) | **53.87** (+1.32) |

Parentheses show the absolute improvement over OPSD in percentage points.

### Component ablation

Qwen3-1.7B average accuracy across the three benchmarks:

| Method | Accuracy ↑ |
|---|---:|
| OPSD | 22.99 |
| OPSD+SC | 26.34 |
| OPSD+UL | 23.66 |
| **TA-OPSD (Ours)** | **27.82** |

### Negative construction

Comparison of negative construction strategies on Qwen3-1.7B:

| Method | Accuracy ↑ | Length ↓ | Capping rate ↓ |
|---|---:|---:|---:|
| OPSD | 22.99 | 5,218.87 | 7.55 |
| TA-OPSD | 27.82 | 5,197.91 | 4.77 |
| TA-OPSD (OPSD-trained negatives) | **28.38** | **4,731.88** | 3.50 |
| TA-OPSD (LLM-assisted selection) | 27.82 | 4,771.68 | **3.26** |

### Termination behavior

Qwen3-1.7B generation behavior reported in the paper:

| Metric | OPSD | TA-OPSD |
|---|---:|---:|
| Teacher-solvable short errors on AIME24 | 42/360 (11.7%) | 6/360 (1.7%) |
| Mean response length | 5,539 | 5,249 |
| Mean capping rate | 8.89 | 4.31 |

### Teacher–student alignment

Top-100 candidate-token overlap and the mean teacher-minus-student EOS log
probability gap:

| Model | OPSD overlap ↑ | TA-OPSD overlap ↑ | OPSD EOS gap ↓ | TA-OPSD EOS gap ↓ |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 44.0 | 63.9 | 9.59 | 5.97 |
| Qwen3-4B | 45.5 | 54.1 | 11.78 | 9.95 |
| Qwen3-8B | 45.1 | 61.1 | 21.08 | 18.46 |

Overlap is a percentage; the EOS gap is measured in nats.

### Termination diagnosis

Standalone and same-prefix continuation diagnosis on 500 Qwen3-1.7B problems:

| Observation | Count |
|---|---:|
| Student failures | 236 |
| Teacher-correct / student-failed problems | 210 |
| Student trajectories shorter than their teacher counterparts | 93 (44.3%) |
| Teacher continues from the failed student prefix | 51 (54.8%) |
| Continued responses recover the correct answer | 34 (66.7% of continuations) |

Additional reasoning recovers 36.6% of the 93 originally incorrect responses.

### Response examples

Selected responses to the same AIME24 lottery problem:

| Method | Tokens | Key output | Correct |
|---|---:|---|---|
| OPSD (incorrect) | 13 | Directly states 11 | No |
| OPSD (correct) | 2,251 | Computes 1/115 and returns 116 | Yes |
| TA-OPSD | 1,154 | Counts 115 prizes and returns 116 | Yes |

Selected responses to the same AIME24 logarithm problem:

| Method | Tokens | Termination | Final answer |
|---|---:|---|---|
| OPSD | 32,597 | Budget exhausted | No correct final answer |
| TA-OPSD | 3,026 | Natural | 33; correct |

### Training token budget

TA-OPSD increases the effective backpropagated training-token budget by 6.66%
for the 1.7B run. Replayed negative spans have a mean length of 33.46 tokens and
an observed maximum of 120 tokens, compared with approximately 779 supervised
tokens per OPSD training response.

## Installation

The GPU recipes use Linux, Python 3.10 and CUDA GPUs. The recorded software
versions are pinned in [requirements-runtime.txt](requirements-runtime.txt).
Install a CUDA-compatible PyTorch build first, then the remaining packages;
FlashAttention requires a matching CUDA toolkit and compiler.

```bash
git clone https://github.com/gtbxdf4/TA-OPSD.git
cd TA-OPSD
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0
python -m pip install packaging ninja
python -m pip install --no-build-isolation -r requirements-runtime.txt
python scripts/fetch_upstream.py
```

The bootstrap fetches and verifies OPSD commit
`ae7d2519e94920c4eb6206c0c26de46d9c50abae`. The upstream training core and
scorer are dependencies rather than files redistributed in this repository.
See [NOTICE.md](NOTICE.md).

## Quick start: fresh TA-OPSD training

The following example rebuilds the inputs and negatives, calibrates the two
coefficients at initialization, and trains Qwen3-1.7B. Run GPU commands within
your own allocation. The launcher does not manage GPU resources or a scheduler.

### 1. Prepare the model and data

Download the model revision recorded in `configs/experiments.json`:

```bash
hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir models/Qwen3-1.7B
python scripts/prepare_assets.py --download --group training
python scripts/prepare_assets.py --download --group evaluation
python src/generate_initialization.py --size 1.7B \
  --model-path models/Qwen3-1.7B
python scripts/build_mining_inputs.py --model-path models/Qwen3-1.7B
```

Dataset bytes, ordered source rows, reconstructed prompts and common LoRA
tensors are checked against the released metadata. Acquisition and the other
model revisions are described in [docs/ASSETS.md](docs/ASSETS.md).

### 2. Generate and mine negatives

```bash
python src/generate_candidates.py --model models/Qwen3-1.7B --tp 4 \
  --out-root runs/1.7B/preparation/candidates --tag base
python src/mine_negatives.py --model models/Qwen3-1.7B \
  --raw runs/1.7B/preparation/candidates/raw \
  --exclude-question-hashes configs/mining/1.7B-exclude-exposed.json \
  --output runs/1.7B/preparation/negatives.jsonl
```

Mining retains fit examples after exposure exclusion. It locates the third
repeated occurrence, preserves the preceding prompt and response tokens, and
selects at most 256 suffix tokens. Stop tokens are excluded from UL targets.

### 3. Capture the initialization batch and calibrate

```bash
python run.py --size 1.7B --method OPSD --mode smoke \
  --model-path models/Qwen3-1.7B --output runs/1.7B/OPSD/smoke --execute
python src/calibrate_fresh.py --method TA_OPSD \
  --model-path models/Qwen3-1.7B --smoke-run runs/1.7B/OPSD/smoke \
  --negative-library runs/1.7B/preparation/negatives.jsonl \
  --output-dir runs/1.7B/preparation/calibration
```

The two-update run captures the pre-update batch and common initialization.
SC is calibrated in logit-gradient space; UL is calibrated in trainable LoRA
parameter space. Both coefficients remain fixed during formal training.

### 4. Train

Without `--execute`, `run.py` prints the complete command and required inputs.
Use a different output directory for a smoke and a formal run:

```bash
python run.py --size 1.7B --method TA_OPSD --mode formal \
  --model-path models/Qwen3-1.7B --output runs/1.7B/TA_OPSD/train \
  --auxiliary-manifest runs/1.7B/preparation/calibration/auxiliary.json --execute
```

Formal training starts from the common untrained LoRA tensors, independently
of the smoke checkpoint. Each update uses 32 current student rollouts and
16 local negatives. The prefix is cached without gradients; only the negative
suffix contributes UL gradients.

## Evaluate

```bash
python src/evaluate.py --base-model models/Qwen3-1.7B \
  --checkpoint-dir /path/to/trained-adapter \
  --dataset aime24 --output-dir runs/1.7B/TA_OPSD/eval/aime24 \
  --tensor-parallel-size 4 --max-model-len 32768 --max-new-tokens 38912 \
  --temperature 1 --top-p 0.8 --top-k -1 --min-p 0 \
  --presence-penalty 0 --val-n 12
```

Repeat for `aime25` and `amc23`. Omit `--checkpoint-dir` to evaluate Base.
Avg@12 grades each sampled response with the common OPSD extractor and verifier.
The context window is 32,768 tokens; the effective continuation budget is bounded
by the context remaining after the prompt. Each evaluation retains
`raw_responses.jsonl`, `author_result.json` and `DONE.json`.

## Other models, baselines and ablations

Use `--size 4B` or `--size 8B` with that model's revision, initialization,
exposure exclusions and freshly calibrated coefficients. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the complete recipes.

| Recipe | 1.7B GPUs | 4B GPUs | 8B GPUs |
|---|---:|---:|---:|
| OPSD / TA-OPSD | 4 | 4 | 8 |
| SFT / GRPO | 4 | 8 | 8 |
| Stable-OPD / RLCSD adaptations | 4 | 4 | 8 |

These are the recorded data-parallel layouts. The original experiments used
A800 hardware; memory requirements also depend on context length and the runtime.

- **Baselines:** `SFT`, `GRPO`, `OPSD`, `Stable_OPD`, `RLCSD`.
- **Component ablations (1.7B):** `OPSD_SC` and `OPSD_UL`. Calibrate each arm
  separately with the corresponding `--method`; SC does not require a negative
  library. UL uses the OPSD loss as its reference; TA-OPSD uses OPSD+SC.
- **Trained-donor negatives (Appendix E.1):** pass a trained OPSD checkpoint to
  `generate_candidates.py --adapter`, then mine and calibrate with
  `--method TA_OPSD_trained_negatives`.
- **Semantic negatives (Appendix E.2):** review the existing mechanical negatives
  offline and apply the keep/drop decisions with
  `scripts/select_semantic_negatives.py`. Calibrate the retained library with
  `--method TA_OPSD_semantic_negatives`. The review protocol is in
  [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Diagnostics and figures

[docs/REPRODUCE.md](docs/REPRODUCE.md) covers the 500-question termination
diagnosis, fresh same-prefix continuation, independent benchmark teachers,
and the complete evaluation matrix.

To render the paper's released aggregate statistics on CPU:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/reproduce.py --output-dir figures/paper
```

To compute the same plots and tables from **your new runs**:

```bash
python diagnostics/benchmark_teacher.py --model-path models/Qwen3-1.7B \
  --output-dir runs/1.7B/teacher-aime24
python analysis/collect.py --manifest configs/reproduction.json \
  --output-dir analysis/fresh-data
python analysis/reproduce.py --data-dir analysis/fresh-data \
  --output-dir figures/fresh
```

`configs/reproduction.json` lists all 111 evaluation cells and six monitored
training traces needed for the tables and curves. Update its paths to point to your
training and evaluation outputs before collecting results. See the
[paper-to-code map](docs/PAPER_ANALYSIS_MAP.md).

## Repository layout

```text
run.py          training recipe launcher
src/            SC/UL, replay, initialization, calibration, mining and evaluation
baselines/      paper baseline adaptations
configs/        model revisions, recipes and input identities
scripts/        upstream/data preparation and offline semantic selection
diagnostics/    termination diagnosis and privileged-teacher continuation
analysis/       raw-result collection and figure/table generation
tests/          loss, input, recipe and runtime regression checks
docs/           reproduction details and paper-to-code mappings
```

## Development

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest
```

PyTorch is needed for the loss tests. [VALIDATION.md](docs/VALIDATION.md)
distinguishes source/CPU checks from a fresh GPU training-and-evaluation replay.

## Citation

```bibtex
@misc{taopsd2026,
  title = {Learning When to Continue and When to Stop: Termination-Aware OPSD},
  year = {2026},
  howpublished = {\url{https://github.com/gtbxdf4/TA-OPSD}}
}
```

The paper link and author citation will be added with the public manuscript.
Please also cite OPSD and the baseline papers when using those implementations.

## License

The TA-OPSD additions are under [MIT](LICENSE). Upstream code, models and datasets
retain their own terms; see [NOTICE.md](NOTICE.md).
