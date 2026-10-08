# Algorithm-to-code map

| Paper component | Implementation | Preserved semantics |
|---|---|---|
| Base OPSD loss L0 | fetched `opsd_trainer.py` | Original full-vocabulary component clipping and valid-token normalization |
| Stop/continue SC | `src/independent_losses.py:stop_kl` | Aggregate stop-token union, teacher-to-student binary KL, response mean |
| Local repetition UL | `src/independent_losses.py:local_unlikelihood` | Negative suffix tokens only; no CE term |
| Auxiliary integration | `src/auxiliary_bridge_v2.py` | Add frozen SC/UL coefficients to the author trainer |
| Replay layout | `src/replay_v2.py` | 16 ordered negative replays per update under recorded DP4/DP8 layouts |
| Prefix/suffix gradient boundary | `src/replay_v2.py` | Prefix C computed without gradients and cached; suffix b carries UL gradient |
| Mechanical mining | `src/mine_negatives.py` | Third exact text copy / tandem token periodicity, inward token alignment, suffix length at most 256 |
| Coefficient ratios | `src/calibrate_fresh.py` | Ratios of measured initialization gradient norms; configuration holds frozen measured coefficients |
| Initialization | `src/shared_initialization_v18.py` | Seed before construction; exact shared LoRA tensors; zero-B verification |
| Passive monitoring | `src/raw_monitor_v18.py`, `monitor_runtime_v18.py` | Original rollout tokens, stop mass and full-vocabulary signals; no sampling or loss changes |
| Evaluation | `src/evaluate.py` plus fetched author evaluator | Raw output identity/count/seed checks, original mathematical scorer |

Qwen stop IDs are `[151643, 151645]`; `</think>` is not added to this set.
The original SC/UL objective averages within responses and then over present
responses. Token-mean experimental branches are outside this paper package.

Stable-OPD is the matched adaptation of arXiv:2604.08527v1 documented in
`baselines/stable/stable_opd.py`. RLCSD uses arXiv:2606.11709v2 equations and
the adaptation in `baselines/rlcsd`. Their data ordering, rollout budget and
reward construction differ from OPSD. These differences are reported rather
than disguised as identical training trajectories.
