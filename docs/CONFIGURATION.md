# Experiment configuration

`configs/experiments.json` records the paper configurations and model revisions.
`run.py` resolves model, asset and output locations without changing their
scientific scalar arguments.

| Main TA-OPSD size | Frozen SC coefficient | Frozen UL coefficient | Frozen negatives | Main-table checkpoint |
|---|---:|---:|---:|---:|
| 1.7B | 5.228043664862209 | 0.015292865006877883 | 52 | 75 |
| 4B | 3.8690446542260792 | 0.008228872973790336 | 43 | 50 |
| 8B | 3.980663829129175 | 0.012583687589434935 | 31 | 100 |

Frozen coefficients belong to their frozen banks. For regenerated banks,
`calibrate_fresh.py` measures the Appendix B ratios at the common initialization
and writes an auxiliary manifest bound to the exact bank SHA. Supply it with
`run.py --auxiliary-manifest`.

Training seed is 42; evaluation engine seed is 20260727. OPSD variants use
100 optimizer updates, effective rollout batch 32, accumulation 2, LoRA rank64
and alpha128, learning rate 5e-6 with linear decay, and 1024-token training
rollouts. Pointwise clipping is 1e-6 for 1.7B/4B and 1e-7 for 8B. SC and UL
average positions within each response, then average responses. Replay supplies
16 negatives per update across the recorded DP4/DP8 layout.

SFT has 100 updates; GRPO has a 500-update horizon and eight completions per
problem. Stable-OPD and RLCSD retain the matched objectives and sampling
procedures in `baselines/`. Their full arguments are printed by `run.py`.

The generated `LAUNCH.json` records the resolved command and environment.
The initial LoRA tensors and pre-update batch are captured under `evidence/`.
A smoke stops after two updates while retaining the formal LR horizon.
Formal runs start from initialization in a new output directory.

OPSD-family checkpoints are `OUTPUT/METHOD/checkpoint-N`; SFT uses
`OUTPUT/checkpoint-N`; GRPO uses
`OUTPUT/SIZE-grpo-MODE-dpDP/checkpoint-N` with the size/mode/DP in the launch
command. Stable-OPD/RLCSD use `OUTPUT/model/checkpoint-N`.
The main-table SFT/Stable-OPD/RLCSD checkpoints follow 75/50/100, and GRPO uses
300/200/400. `configs/reproduction.json` records the evaluated checkpoint paths.
The 1.7B curves use 0/25/50/75/100; component and negative-construction tables
use 75. These selections are retained from the paper release.
