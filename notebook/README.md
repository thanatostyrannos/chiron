# notebook/ — pre-registered experiments and results (a register)

The experiment journal. One file per hypothesis, `notebook/<slug>.md`, named for the
hypothesis (`eviction-vs-recall-depth.md`), indexed below. This is a **record**: the
hypothesis card and design freeze *before* the run; the results section is written
once *after* and then freezes too. Corrections are appended, never applied.

Each entry is a G2 hypothesis card, committed before the run:
`HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST / RISKIEST`. No
post-hoc hypothesis fitting; moving a SUCCESS/KILL threshold after seeing results is a
change of standard and must be called out as one. A falsified hypothesis is a
successful experiment — write it up with equal care.

Analysis/plotting scripts here are exempt from strict TDD **only** if reproducible
from committed config and committed data hashes. On reuse they migrate into the rig
and acquire tests.

`G0-LIGHT` exception: experiments under $25 and under 2 hours may proceed on a
one-line rationale logged here instead of a full Discovery Brief.

## Index

| Experiment (slug) | Hypothesis (one line) | Status | Outcome |
|---|---|---|---|
| `uma-carveout-controls-fast-tier` | Raising BIOS UMA FB Size to 96 GB moves the high-bandwidth working-set boundary out from its measured 30 GiB. | complete | **SUCCESS** — fast tier 30 GiB → ≥62 GiB at ~200 GB/s. Kept at 96 GB. Also found: single tensors ≥32 GiB hang or fault. |

Platform-characterisation experiments (like the one above) run before the Hardware
Validation Gate because they establish the instrument. Model ablations are authored in
the Ablation Backlog phase and run only after that gate passes.
