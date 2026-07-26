# LOG — append-only

A **register** (CLAUDE.md → DOCUMENT CLASSES): entries are appended, never edited or
deleted. One entry per working session or milestone. Gate markers follow
`OPERATING_INSTRUCTIONS.md`. Gate skips are logged here in the override-protocol
format: `SKIPPED G[n] on [date] — rationale: [x] — debt: [what must be revisited]`.

---

## 2026-07-24 — Environment Bootstrap

Ran `scripts/preflight.ps1` on the Z13. 13/19 checks pass; all six failures are the
ROCm/gfx1151 torch stack (system torch is a CUDA wheel and sees no AMD device).
`ENVIRONMENT.md` written with the remediation list. Re-encoded `preflight.ps1` as
UTF-8 with BOM so it parses under Windows PowerShell 5.1.

## 2026-07-24 — Scaffold (written)

uv-workspace layout under `packages/` with the dependency-direction boundary declared;
`research/`, `curriculum/`, `docs/`, `configs/`, `notebook/`, `tests/` seeded with
per-folder READMEs. `ASSUMPTIONS.md` seeded with every assumption baked into the
kickoff. Not committed in this session.

## 2026-07-26 — Scaffold closed → `v0.1.0`

[G4 — Build] Proved the boundary lint contract red-then-green before committing:
inserted a transient `import proteus` under `packages/mnemosyne/src/mnemosyne/`,
confirmed `tests/test_package_boundaries.py` fails naming the offending file, removed
it, confirmed green. `ruff check` clean across `tests/` and `packages/`. Run in an
ephemeral `uv run --no-project --with pytest` environment — **`uv sync` was
deliberately not run**, because the workspace resolves `torch` from PyPI (CPU/CUDA)
until the gfx1151 index is configured, and a wrong torch in the lockfile is a silent
instrument error. The boundary test is filesystem-based and needs no installed
packages, so this costs nothing.

Narrowed the `.claude/` ignore to `settings.local.json` and committed
`.claude/agents/` — the eight subagent definitions are lab state, and a session must
cold-start from the repo alone.

Deferred, not skipped: the GitHub remote. `gh repo create` needs a visibility
decision from the owner and CLAUDE.md hard rule 6 requires asking before a public
push. Local `main` + tag only.
