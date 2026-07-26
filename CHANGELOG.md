# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versions are the annotated
git tags described in CLAUDE.md (semver; milestones are named in the tag annotation).

## [Unreleased]

Next: complete Environment Bootstrap (gfx1151 ROCm torch, capacity ceiling), then the
Reference Library milestone.

## [0.1.0] — 2026-07-26 — milestone `scaffold`

### Added
- Environment Bootstrap: `scripts/preflight.ps1` run on the Z13; `ENVIRONMENT.md`
  written (13/19 checks pass — all green except the ROCm/gfx1151 torch stack, which
  is deferred to a supervised step before the Hardware Validation Gate).
- Scaffold: `git init` on `main`; uv-workspace layout under `packages/`
  (`mnemosyne`, `proteus`, `themis`) with the dependency-direction boundary declared
  and a lint-contract test guarding it; `research/`, `curriculum/`, `docs/` (+`adr/`,
  `diagrams/`), `configs/`, `notebook/`, `tests/` trees seeded with per-folder READMEs.
- `ASSUMPTIONS.md` seeded with every assumption baked into the kickoff.
- Resolved kickoff parameters: PROJECT_NAME=Chiron, ARCH_CODENAME=Proteus,
  CLOUD_BUDGET=$0. GitHub visibility still pending owner decision.
- `.claude/agents/` (the eight subagent definitions) committed rather than ignored, so
  a session cold-starts from the repo alone.
- `LOG.md` (append-only session/milestone log) and `BLOCKERS.md` (what is stopping
  work now), both named in CLAUDE.md's repo map but previously absent.

### Changed
- `.gitignore`: narrowed the `.claude/` rule to `settings.local.json` and
  `.claude/*.local.json`.
- `KICKOFF_PROMPT.md`: removed three violations of the repo's own naming rule —
  `docs/adr/0001+` became `docs/adr/<slug>.md`, the non-semver `v0.1-curriculum` tag
  was dropped in favour of the phase's `v0.4.0`, and a stray `tag.5.` was corrected.

### Fixed
- `scripts/preflight.ps1` re-encoded UTF-8 with BOM so it parses under Windows
  PowerShell 5.1 (the seed shipped as UTF-8 without BOM; 5.1 read the em-dashes as
  ANSI and failed to parse).
