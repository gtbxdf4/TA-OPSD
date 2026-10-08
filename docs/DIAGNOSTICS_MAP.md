# Termination diagnostics

The diagnostic data and outputs are separate from training and evaluation.
`configs/diagnostic_manifest.json` identifies the 500 Section 3 source questions
using row references, normalized hashes, prompt hashes, seeds and budgets.

1. `prepare.py cohort` rebuilds exact student and full-reference teacher prompts.
2. `generate.py --mode standalone` samples trained-student and frozen-teacher answers.
3. `prepare.py continuation` derives shorter teacher-solvable student failures.
4. `generate.py --mode continuation` continues their unchanged prefixes with the teacher.
5. `analyze.py` checks prompt/prefix identity and counts the resulting behavior.

For Figure 3(a), `benchmark_teacher.py` instead generates independent teacher
answers for the 30 AIME24 questions. `classification.py` implements the short
incorrect/natural-stop/no-repetition predicate. `analysis/collect.py` combines
these teacher verdicts with all 360 student samples per checkpoint.

`grading.py` retains the common last-box answer extraction and OPSD math verifier.
All diagnostic model, checkpoint, input and output paths are explicit or derived
from the reconstructed local contracts. No historical raw-response cache is
required for a fresh run. See [REPRODUCE.md](REPRODUCE.md) for commands.
