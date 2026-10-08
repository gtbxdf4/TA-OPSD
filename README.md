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
  --checkpoint-dir runs/1.7B/TA_OPSD/train/TA_OPSD/checkpoint-75 \
  --dataset aime24 --output-dir runs/1.7B/TA_OPSD/eval/step75/aime24 \
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
training traces needed for the tables and curves. Its paths follow the example
layout above and can be edited to point to your outputs. See the
[paper-to-code map](docs/PAPER_ANALYSIS_MAP.md).

## Results reported in the paper

Mean accuracy across AIME24, AIME25 and AMC23:

| Model | OPSD | TA-OPSD | Improvement |
|---|---:|---:|---:|
| Qwen3-1.7B | 22.99 | **27.82** | +4.84 |
| Qwen3-4B | 35.88 | **39.61** | +3.73 |
| Qwen3-8B | 52.55 | **53.87** | +1.32 |

Values are percentages; improvements use unrounded averages. These are the
reported results, not claims that fresh sampling reproduces identical numbers.
Frozen historical banks are identified by hashes; fresh runs can rebuild banks
from public inputs without them.

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
