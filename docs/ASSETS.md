# Models and data

Large inputs stay outside Git. `scripts/fetch_upstream.py` fetches OPSD at commit
`ae7d2519e94920c4eb6206c0c26de46d9c50abae` and verifies its public runtime files.
`ASSETS_MANIFEST.json` verifies the public training and evaluation parquet bytes.

| Model | Hugging Face ID | Revision |
|---|---|---|
| 1.7B | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| 4B | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| 8B | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |

Run `scripts/prepare_assets.py --download --group training` and the same command
with `--group evaluation`. `--revision` can select a dataset revision; downloaded
bytes must still match the manifest. Model and dataset licenses follow upstream
terms.

Run `src/generate_initialization.py` for each size to construct the common seed-42
LoRA. Its tensor identities ensure all methods start from the same untrained
adapter. Initialization may require substantial CPU RAM.

`scripts/build_mining_inputs.py` reconstructs the ordered 29,434 source rows and
1,195 candidate prompts. Candidate problem and prompt identities preserve cohort
selection, exposure exclusion and deterministic sampling. With an external
`--asset-root`, use the same root in `run.py` and pass its `mining/cohort.json`
to candidate generation through `--cohort`.

Generate and mine negatives, then run `src/calibrate_fresh.py` as shown in README.
The generated auxiliary manifest binds coefficients to the exact negative bank;
provide it to `run.py --auxiliary-manifest`. Historic bank files are not shipped.
A fresh public-input reconstruction does not require them. Frozen settings in
`configs/experiments.json` document the paper recipes; do not apply coefficients
from one bank to a newly generated bank without calibration.

Hashes retained in this checkout are used for public dependency/data validation,
shared initialization, prompt/cohort identity, negative selection, sampling or
calibration-bank binding. Pure source-script and run-receipt fingerprints are not
recorded. No weights, parquet files or raw model responses belong in Git.
