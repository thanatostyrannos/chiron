# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versions are the annotated
git tags described in CLAUDE.md (semver; milestones are named in the tag annotation).

## [Unreleased]

Next: the Reference Library milestone. The Hardware Validation Gate additionally owes a
pre-registered investigation of the ≥32 GiB tensor hang/fault.

## [0.3.0] — 2026-07-26 — experiment `uma-carveout-controls-fast-tier`

### Added
- `notebook/uma-carveout-controls-fast-tier.md` results: BIOS UMA FB Size 16 → 96 GB
  moved the GPU's fast memory tier from a 30 GiB to a **≥62 GiB** working set at
  ~200 GB/s. Pre-registered SUCCESS (≥60 GiB) met; setting kept.
- New ledger entry `large-tensor-fault-32gib`: single tensors ≥32 GiB hard-hang at 0 CPU
  or raise `hipErrorLaunchFailure`. Keep individual buffers under 32 GiB.

### Changed
- `scripts/measure_memory_bandwidth_tiers.py`: `--sizes` for targeted sweeps; a GPU
  fault is now reported as a result and stops the sweep, rather than aborting mid-run
  or recording the post-fault cascade as further data points.

## [0.2.0] — 2026-07-26 — milestone `rocm-toolchain`

### Added
- AMD gfx1151 ROCm nightlies in `C:\venvs\lab`: `torch 2.12.0a0+rocm7.13.0a20260313`
  (HIP 7.2.0), arch `gfx1151`. preflight 18/19. The torch wheel pins
  `rocm==7.13.0a20260313`; that pinned pair is the instrument of record.
- `scripts/activate-lab.ps1` (session-scoped env, no system changes),
  `scripts/measure_capacity_ceiling.py`, `scripts/measure_memory_bandwidth_tiers.py`,
  `scripts/benchmark_gemm.py`.

### Fixed
- `scripts/preflight.ps1` capacity probe reported its own search bound as a
  measurement. It now prints `SATURATED` when it hits the bound and labels the figure
  `alloc-only`, since an untouched `torch.empty` can be a reservation that fails on
  first write.

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
