# Paper-to-code map

| Paper content | Implementation |
|---|---|
| Base OPSD | pinned `upstream/opsd/opsd_trainer.py`, fetched by `scripts/fetch_upstream.py` |
| SC, Eqs. 2–4 | `src/independent_losses.py::stop_kl` |
| UL, Eq. 5 | `src/independent_losses.py::local_unlikelihood`, `src/replay.py` |
| Combined objective, Eq. 6 | `src/auxiliary_bridge.py` |
| Coefficient calibration, Appendix B | `src/calibrate_fresh.py` |
| Negative construction, Appendix C | `scripts/build_mining_inputs.py`, `src/generate_candidates.py`, `src/mine_negatives.py` |
| Main comparisons, Table 1 | `run.py`, `baselines/`, `src/evaluate.py`, `analysis/collect.py` |
| Components, Table 2 and Figure 5 | OPSD/SC/UL/TA-OPSD recipes and `analysis/reproduce.py` |
| Negative construction, Table 3 | trained-donor or offline semantic selection, fresh calibration and evaluation |
| Generation length/capping | `src/evaluate.py`, `analysis/collect.py`, `analysis/reproduce.py` |
| Selected response examples | manifest `examples` entries and `analysis/collect.py` |

Training-distribution monitoring, the 500-question termination diagnosis,
teacher-solvable short-error analysis and historical aggregate files are
currently omitted. Figures 2 and 4 and the diagnostic part of Figure 3 cannot be
reproduced from this checkout alone. No manuscript build files or test suite are
included in this version.

Accuracy, length and capping use equal-weight means of AIME24, AIME25 and AMC23.
Avg@12 averages individual response verdicts. Capping means the recorded
`finish_reason == "length"`, not a mechanical repetition classification.
Curves connect evaluated points. Use fresh aggregates from `analysis/collect.py`;
no default frozen plotting data are included.

The stop set is `[151643, 151645]`; `</think>` is not a stop token. SC and UL
average selected positions within a response, then average nonempty responses.
The UL prefix cache is detached and only the selected suffix contributes loss.

Stable-OPD and RLCSD are the matched adaptations described by their modules,
not claims of equivalence to all settings in their original papers. All methods
use the common upstream evaluation scorer.
