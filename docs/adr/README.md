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
| _(none yet — first ADRs land in the Rig Design phase: framework choice, config schema, eval probe set, ablation statistical standard)_ | | | | |
