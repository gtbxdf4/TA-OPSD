# Paper-to-code map

| Paper content | Implementation | Fresh outputs |
|---|---|---|
| Base OPSD, Appendix A | pinned `upstream/opsd/opsd_trainer.py` | training checkpoints and captured rollouts |
| SC, Eqs. 2–4 | `src/independent_losses.py::stop_kl` | binary teacher-to-student KL |
| UL, Eq. 5 | `src/independent_losses.py::local_unlikelihood`, `src/replay_v2.py` | detached prefix cache; selected suffix gradients |
| Combined objective, Eq. 6 | `src/auxiliary_bridge_v2.py` | configured SC/UL additions to OPSD |
| Coefficient calibration, Appendix B | `src/calibrate_fresh.py` | measured gradient norms and bank-bound coefficients |
| Negative construction, Appendix C | `scripts/build_mining_inputs.py`, `src/generate_candidates.py`, `src/mine_negatives.py` | public-source prompts and local repeated suffixes |
| Section 3, 500-question diagnosis | `diagnostics/prepare.py`, `generate.py`, `analyze.py` | standalone failures, shorter teacher-solvable cases, same-prefix recovery |
| Figure 2 | raw training monitor → `analysis/collect.py` → `reproduce.py` | mean student/teacher log probability for stop token 151645 |
| Figure 3(a) | `diagnostics/benchmark_teacher.py`, `diagnostics/classification.py`, `analysis/collect.py` | teacher-solvable short-error counts / all 360 AIME24 responses |
| Figure 3(b,c) | `src/evaluate.py`, `analysis/collect.py` | generated token length and budget-exhaustion rate |
| Figure 4 | `src/raw_monitor_v18.py`, `monitor_runtime_v18.py`, `analysis/collect.py` | Top-100 overlap; mean gap over the two stop tokens at valid positions |
| Figure 5 / Table 2 | OPSD, SC, UL and combined recipes | four-arm capping curves and Avg@12 |
| Table 1 | `run.py`, `baselines/`, `src/evaluate.py` | seven methods × three sizes × three benchmarks |
| Table 3 / Appendix E | trained donor / offline semantic selection / fresh calibration | accuracy, length and capping summaries |
| Appendix C token budget | `analysis/collect.py` | main supervised tokens and UL suffix tokens |
| Tables 4–5 / Appendix D | manifest `examples` in `analysis/collect.py` | selected raw sample metadata and correctness |

Across benchmarks, accuracy, length and capping use equal-weight arithmetic
means of AIME24, AIME25 and AMC23. Avg@12 averages individual response verdicts.
Capping is the recorded `finish_reason == "length"`; it is distinct from the
mechanical repetition predicate.

Figures 2 and 4 retain all observed training points. Display curves use a
Gaussian filter (sigma2.5) and a natural cubic spline; interpolation adds no
observations. Figures 3 and 5 connect only the evaluated checkpoints.

`analysis/reproduce.py` defaults to the small frozen inputs in `analysis/data`.
Use `--data-dir` with aggregates emitted by `analysis/collect.py` for fresh runs.
Generated figures and raw results are excluded from Git.
