# research/

Three tracks. Read in this order once the Reference Library and Frontier Survey
phases have populated them:

1. `reference/` — cloned upstream code + papers (gitignored except `*.md`/`*.bib`),
   reproduced by `scripts/fetch_reference.sh`. `CODE_MAP.md` is the guided tour;
   `PROVENANCE.md` the ledger. **Authored in the Reference Library phase.**
2. `memory/` — **the memory-systems research track**, the priority deliverable.
   Ten notes, taxonomy → open-problems, mirrored 1:1 by curriculum modules. See
   `memory/README.md` for reading order. **Authored in the Frontier Survey phase.**
3. `notes/` — the frontier survey (how 2026 frontier models work end to end). See
   `notes/README.md`. **Authored in the Frontier Survey phase.**

`synthesis.md` sits at this level and is written last — SCQA → answer-first → three
arguments → evidence, with the MECE issue tree and the questions worth our compute.
**Start here if you are reading the research for the first time**; it names what we
pursue, what we park and why, what is folklore, and the single riskiest assumption.

## State as of 2026-07-26

Seventeen notes, ~95,000 words. **619 distinct arXiv ids machine-verified against the
live API, 0 unresolved** (`scripts/verify_citations.py`; per-track reports in each
directory's `citation-verification.json`). Resolving an id proves a paper exists, not
that it supports the claim beside it — the check is against fabrication, not error.

The synthesis is `status: proposed`. Its recommendation is to build an attribution
instrument rather than another eviction policy, and to exploit the one hardware property
no datacenter GPU exposes: the BIOS UMA carve-out is a measured knob on the fast/slow
bandwidth ratio.

Every material claim carries a G5 tag: `[M]` measured, `[C]` cited (arXiv/URL+date),
`[A]` assumed. No invented numbers.
