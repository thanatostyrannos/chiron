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

## 2026-07-26 — ROCm toolchain live; the instrument lied twice

Installed AMD's gfx1151 nightlies into `C:\venvs\lab`: `torch
2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), arch `gfx1151`, device visible. The torch
wheel pins `rocm==7.13.0a20260313` and downgraded the 7.14.0a20260612 SDK installed
minutes earlier — the pinned pair is the instrument, and that pin is why an
unversioned benchmark on this stack is worthless. preflight now 18/19.

Two measurement errors caught, both worth recording because both produced numbers that
looked like results:

1. **Search bounds reported as measurements.** The capacity probe I wrote capped its
   binary search at the reported pool and returned 82.67 GiB — its cap.
   `preflight.ps1` capped at 100 and returned 100 — its cap. The two disagreed, which
   is the only reason either was caught. `preflight.ps1` now prints `SATURATED` and
   labels the figure `alloc-only` instead of returning a bare number.
2. **Untouched allocations counted as capacity.** Both probes only called
   `torch.empty`. A reservation that fails on first write would have passed.
   `scripts/measure_capacity_ceiling.py` now fills and reads back every trial.

Then made a third error live: re-ran that probe with a 104 GiB bound against 112 GB of
physical RAM, drove the host into swap, and aborted it. A bound safely under physical
memory (~90 GiB) is the fix. What survives as evidence is the one trial that completed
honestly: **≥74.40 GiB written, read back, released, 4.04 s.**

Real findings from the session, all single-run `[M]`:
- **Fast tier is ~30 GiB, not 83.** ~195 GB/s up to a 30 GiB footprint; 61 GB/s at 32
  GiB. Sharp boundary, ~2x. It does **not** align with the 16 GiB dedicated carve-out —
  a 24 GiB footprint still ran at 198 GB/s. For a memory-systems lab this inverts the
  headline: capacity is ~83 GiB, but the number that sizes a KV-cache experiment is 30.
- **hipBLASLt: no 5x cliff on this wheel.** 18.6 → 20.9 TFLOPS bf16 (+12%) with the
  env set. Worth setting, but the documented catastrophe did not reproduce.
- **GEMM is 63% of the published figure** (20.9 vs ~33 TFLOPS). Unexplained.

Pre-registered `notebook/uma-carveout-controls-fast-tier.md` with SUCCESS/KILL fixed
before the run, then asked the owner to set BIOS UMA FB Size to 96 GB. The BIOS change
is the experiment, not a fix — the KILL condition sends us back to the default.

## 2026-07-26 — experiment `uma-carveout-controls-fast-tier`: SUCCESS

[G5 — Evidence] Owner set BIOS UMA FB Size to 96 GB. Pre-registration was committed
first (`106ce53`, `v0.2.0`), so the thresholds were frozen before the run.

Fast tier moved **30 GiB → ≥62 GiB** at ~200 GB/s, with no degradation anywhere in the
swept range. Reported pool 82.99 → 107.87 GiB. SUCCESS was ≥60 GiB; KILL was 30 ± 4.
The named confounder (thermal throttling) is cleared: small-footprint bandwidth is
unchanged, so the boundary moved rather than the machine slowing. **Keeping 96 GB.**

The measured boundary is a **floor, not an edge** — the sweep hit a different limit
before finding where bandwidth degrades:

**Unplanned finding — single tensors ≥32 GiB are unsafe.** 31 GiB copies cleanly at
199.9 GB/s; 32 GiB hard-hangs (11 minutes at 0 CPU seconds, host free RAM to 5 GB,
force-killed); 36 GiB raises `hipErrorLaunchFailure`. The GPU recovers fully in a fresh
process. 32 GiB is exactly 2^35 bytes, so `[A]` medium confidence this is a 32-bit
overflow in the copy path rather than a capacity limit — testable by allocating the
same bytes as fp32. **A hang at 0 CPU is silent**: a long training run would stall, not
crash, which makes this a correctness hazard rather than a performance note.

Stopped the fault investigation here rather than chasing it: the 90-minute environment
timebox had expired, and each hang costs ~11 minutes plus a force-kill. Re-decided per
the timebox rule — it becomes a Hardware Validation Gate item with its own
pre-registration. Not skipped, scheduled.

Instrument changes made during the run, both after the primary before/after comparison
was already captured with the `v0.2.0` probe: `--sizes` for targeted sweeps, and a GPU
fault is now caught and reported as a result rather than aborting the sweep (a poisoned
HIP context makes every later reading in that process meaningless, so it stops).
