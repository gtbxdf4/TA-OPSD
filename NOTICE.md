# Source and license notices

The original TA-OPSD additions in this package are distributed under the MIT
license in LICENSE.
The training core, collator, SFT/GRPO entrypoints, and mathematical evaluator
are dependencies from [OPSD](https://github.com/siyan-zhao/OPSD), fixed at
`ae7d2519e94920c4eb6206c0c26de46d9c50abae`. No LICENSE/COPYING/NOTICE file was
present in that pinned Git tree at this release audit. The dependency is not
vendored here. `scripts/fetch_upstream.py` obtains it from its original
repository and verifies the runtime file hashes. Its permissions remain
those supplied by its authors; this repository grants no additional rights
to that dependency.

Stable-OPD and RLCSD code here implements the matched adaptations used in the
paper. See their module documentation and docs/ALGORITHM_MAP.md for the
mathematical sources and differences from native paper recipes.

Models and datasets retain their upstream terms. They are external inputs,
not Git-tracked payloads. Asset hashes establish experiment identity and do
not grant redistribution permission. Aggregated numerical figure/table inputs
are included; private filesystem provenance is replaced by logical source
identifiers, with original evidence retained outside this public package.
