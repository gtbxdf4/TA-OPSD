# Models and data

All large inputs live outside Git. `ASSETS_MANIFEST.json` contains exact hashes
for the historical assets; `UPSTREAM.json` pins the OPSD runtime.

| Model | Hugging Face ID | Revision |
|---|---|---|
| 1.7B | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| 4B | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| 8B | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |

`scripts/prepare_assets.py --download --group training` fetches the two
OpenThoughts shards used by OPSD. `--group evaluation` fetches AIME24, AIME25
and AMC23. Downloads must match the manifest hashes. `--revision` can select
an upstream dataset revision; differing bytes are rejected.

Run `src/generate_initialization.py` for each model size. The seed-42 LoRA
construction is checked against the released tensor manifests. Larger model
initializations require enough CPU RAM for the base model and adapter.

`scripts/build_mining_inputs.py` reconstructs the ordered 29,434-row question
index and the 1,195 candidate prompts from source-row metadata. It verifies
question identities, fit/validation roles and tokenizer prompt hashes. With
an external `--asset-root`, use the same root in `run.py` and pass the generated
`mining/cohort.json` to candidate generation via `--cohort`.

Fresh negative banks are generated and mined as in README.md; coefficients
are then measured for those bank bytes. Historical 52/43/31 main banks and
121/5 ablation banks are optional exact-replay inputs. They are not downloaded
by this code repository. Import an authorized historical asset package with
`prepare_assets.py --source-dir PATH --group negatives` if one is available.
The fresh public-input route does not depend on these frozen banks.

The Section 3 diagnostic uses `configs/diagnostic_manifest.json`: 500 source
row references, seeds, budgets and prompt hashes, without question/solution
payloads. `diagnostics/prepare.py cohort` reconstructs its inputs. Continuation
cases are derived from the new standalone student/teacher outputs, so their
count is allowed to differ from the original 93.

Model and dataset access and licensing follow their upstream terms. Do not add
weights, parquet files, negative token libraries or raw responses to Git.
