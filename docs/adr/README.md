# docs/adr/ — Architecture Decision Records (the register)

An accepted ADR is a **permanent record**: never modified, never renamed, never
deleted. If a decision changes, write a new ADR that supersedes it and append one
supersession line to the old one's Status block. See CLAUDE.md → ADR IMMUTABILITY and
OPERATING_INSTRUCTIONS.md → "Never modify an accepted ADR."

Each ADR is `docs/adr/<slug>.md` (slug states the decision; no numbers), one page:
`# <decision as a sentence>` / Status / Date / Deciders / Context / Decision /
Consequences.

## Status ladder

| Status | Body editable? |
|---|---|
| `Proposed` | yes |
| `Accepted` | **no — frozen** |
| `Superseded by <slug>` | no |
| `Deprecated` | no |
| `Rejected` | no (keep it — knowing why you *didn't* is worth more later) |

## Enforcement

The table below carries a SHA-256 of each ADR's frozen portion (everything below the
Status block). A test in `tests/` recomputes those hashes and fails on mismatch, so an
edit to an accepted ADR cannot be committed unnoticed.

| Slug | Status | Date | Superseded by | Body SHA-256 |
|---|---|---|---|---|
| `aotriton-attention-stays-off-by-default` | Accepted | 2026-07-26 | - | c447bf66bf6fa66830aad15542f656bde9b6bd59f964fa708783e845d8f6c7aa |
| `attribution-instrument-over-eviction-policy` | Proposed | 2026-07-26 | - | - |
| `bios-uma-carveout-at-96gb` | Accepted | 2026-07-26 | - | 326ab7e10ffef1b298615cebeca9f54e00136bc1222fa72afb167d1793e7febd |
| `hipblaslt-is-a-numerics-control` | Accepted | 2026-07-26 | - | 46114831a5ce98e0a52e4bd2080ebf51a56ea6406cdc197492be28f12461b125 |
