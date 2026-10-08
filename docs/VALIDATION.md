# Release validation

The 2026-10-08 release was checked against the paper-specific source extraction
`ta_opsd_iclr2027_20260927`. The original server source and experiment scripts
were not edited. Publication changes are confined to a separate release copy.

Completed checks:

- Python 3.10 syntax, Ruff lint and formatting, JSON parsing and public-file checks.
- 15 CPU tests in the recorded server runtime: SC/UL mathematics and gradients,
  configuration/layout identity, fresh bank binding, initialization-batch smoke
  semantics, environment PATH, terminal UTF-8 handling, fresh continuation
  selection, semantic-token preservation and raw/training aggregation.
- SC and UL values and student gradients are bitwise identical to the original
  extracted implementations on the fixed comparison tensors.
- Public training parquet reconstructs the 29,434 ordered rows and all 1,195
  candidate prompts with matching semantic and prompt hashes.
- The 500-question diagnostic is reconstructed with matching student and
  privileged-teacher prompt hashes and reference-answer identities.
- CPU figure/table reproduction renders Figures 2–5 and verifies 21 main-table,
  four component-table, four negative-construction rows and five saved examples.

The payload contains only source, small configurations/identity metadata and
aggregate figure/table inputs. It excludes datasets, weights, raw responses,
training checkpoints, execution logs, cluster queues and follow-up experiments.

The released entrypoints support a fresh preparation → calibration → training
→ evaluation → analysis run. The publication checks above do not constitute a
new full GPU replay. Fresh GPU calibration/training/evaluation and numerical
agreement with the reported results are separate validation stages. Fresh
sampling and offline semantic selection may produce different negatives and
scores; the code does not promise bitwise regeneration of historical outputs.

`SOURCE_PROVENANCE.json` identifies the original source files and publication
changes. `UPSTREAM.json` checks the fetched third-party runtime. Neither a smoke
nor a successful code check is a measurement of method efficacy.
