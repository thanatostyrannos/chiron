---
title: Does the BIOS UMA carve-out control the size of the GPU's fast memory tier?
version: 1.0.0
status: pre-registered
date: 2026-07-26
---

# Does the BIOS UMA carve-out control the size of the GPU's fast memory tier?

**This is a record.** The hypothesis card and design below are frozen as of the
pre-registration date. The Results section is written once after the run and then
freezes too. Corrections are appended, never applied.

`G0-LIGHT` — cost is one reboot and ~20 minutes of measurement, well inside the
under-$25 / under-2-hours exception. Rationale: the fast-tier size is an input to every
KV-cache capacity experiment this lab will run, and it is currently unexplained.

## Control measurement (already taken, 2026-07-26, BIOS UMA FB Size = default)

`[M]` gfx1151, torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, native Windows.
Driver reports **16 GiB dedicated VRAM**; Windows sees 111.6 GB of 128 GB installed;
ROCm reports an **82.99 GiB** pool. Single seed, single run — an anecdote by the house
standard, but the effect size is ~2x and the boundary is sharp across adjacent points.

Capacity, stated at the precision it was actually measured:

| Quantity | Value | Confidence |
|---|---|---|
| Reported pool (`total_memory`) | 82.99 GiB | exact, self-reported by the driver |
| **Written, read back, released** | **≥ 74.40 GiB** in 4.04 s, no paging collapse | `[M]` solid — this memory demonstrably works |
| Allocation-only (untouched) | ≥ 100 GiB | upper bound only; `preflight.ps1`'s search saturated at its own bound of 100 |
| Exact writable ceiling | **unmeasured** | the probe run bounded at 104 GiB drove the host into swap and was aborted; see below |

The writable ceiling exceeds the reported pool, so the driver oversubscribes into
system memory. Two probes initially disagreed (82.67 vs 100 GiB) and **both were
reporting their own search bounds** — one capped at the reported pool, one at a
hardcoded 100. Neither wrote to the memory it claimed. That is why the table above
separates "verified by writing to it" from "an allocator said yes."

Not repaired before this run: a bound safely under physical RAM (~90 GiB) would have
measured it. The before/after comparison rests on the bandwidth curve, which is the
primary metric; capacity is secondary here and its "before" is the ≥74.40 GiB floor.

`scripts/measure_memory_bandwidth_tiers.py` (device-to-device copy, footprint = 2 buffers):

| Footprint (GiB) | GB/s | | Footprint (GiB) | GB/s |
|---|---|---|---|---|
| 2 | 184.3 | | 30 | **194.9** |
| 8 | 209.9 | | 32 | **61.3** |
| 16 | 201.6 | | 34 | 83.8 |
| 24 | 198.0 | | 40 | 92.5 |
| 26 | 198.1 | | 60 | 114.1 |
| 28 | 193.3 | | 72 | 111.6 |

**Fast-tier boundary: 30 GiB footprint.** The plateau is ~185–210 GB/s, consistent
with the ~172 GB/s reported for this silicon `[C]` (ROCm #6034, Mar 2026). Past the
boundary it halves.

The boundary does **not** coincide with the 16 GiB dedicated carve-out: a 24 GiB
footprint — half again the carve-out — still ran at 198 GB/s. So the fast tier is not
simply "dedicated VRAM", and its size is currently unexplained. That is what this
experiment attacks.

## Hypothesis card

```
HYPOTHESIS   Raising BIOS UMA FB Size from its default (driver-reported 16 GiB
             dedicated) to 96 GB moves the high-bandwidth working-set boundary
             from 30 GiB toward the new carve-out size.
FOR          The Z13 (Ryzen AI Max+ 395, gfx1151) as the lab's primary instrument.
BECAUSE      ROCm reaches 82.67 GiB while only 16 GiB is dedicated, so most of the
             pool is GTT/shared. Bandwidth halves past a 30 GiB footprint. If the
             fast tier is the BIOS carve-out (or a fixed multiple of it), enlarging
             the carve-out enlarges the tier — which sizes every KV-cache
             experiment. If it is a WDDM per-process budget or a driver-internal
             limit, the BIOS setting will not move it at all.
MEASURED BY  scripts/measure_memory_bandwidth_tiers.py, --coarse then --fine.
             Primary metric: fast_tier_boundary_gib, the largest footprint
             sustaining >= 170 GB/s. Threshold fixed here, before the run.
SUCCESS      Boundary >= 60 GiB. The carve-out controls the tier; keep BIOS at 96 GB
             and treat ~60+ GiB as the usable fast KV budget.
KILL         Boundary within 30 +/- 4 GiB. The carve-out does not control the tier;
             revert BIOS to default, stop pursuing this axis, and record the fast
             tier as a fixed ~30 GiB property of the platform.
COST         One reboot, ~20 minutes of measurement, $0.
RISKIEST     That the fast tier is a BIOS/UMA property at all, rather than a WDDM
             per-process local-memory budget or a ROCm allocator policy — neither of
             which a BIOS setting would touch.
```

Outcomes between the thresholds (boundary 34–59 GiB) count as **partial**: the
carve-out influences the tier but does not set it, and the mechanism stays open.

## Design freeze

- Identical committed probes before and after; only the BIOS setting changes.
- Same venv, same wheel (`torch 2.12.0a0+rocm7.13.0a20260313`), same session
  environment via `scripts/activate-lab.ps1`, no driver update in between.
- `--coarse` locates the region, `--fine` localizes the boundary to 2 GiB.
- Re-run `scripts/measure_capacity_ceiling.py` too: a 96 GB carve-out could *lower*
  the 82.67 GiB total ceiling by shrinking what remains available as GTT. Capacity
  and bandwidth are separate outcomes and both are recorded.
- Confounder to watch: thermals. The Z13 is a tablet and these are sustained
  memory-bound loops. If the after-run shows a *uniform* slowdown at every footprint
  rather than a moved boundary, suspect throttling and re-run cold.

## Results

Run 2026-07-26, after setting BIOS UMA FB Size to 96 GB (driver-reported dedicated
VRAM 96 GiB, Windows visible RAM 31.6 GB). Same venv, same wheel, no driver change.
The before and after coarse sweeps used the identical probe as committed at `v0.2.0`;
the `--sizes` option was added afterwards, for the follow-up fault isolation only.

### Primary metric — SUCCESS

| | Before (16 GiB dedicated) | After (96 GiB dedicated) |
|---|---|---|
| **fast_tier_boundary_gib** | **30** | **≥62** |
| Bandwidth at 2–24 GiB footprint | 184–210 GB/s | 205 GB/s |
| Bandwidth at 32 GiB footprint | **61 GB/s** | not separately sampled |
| Bandwidth at 40 GiB footprint | 92 GB/s | **203.7 GB/s** |
| Bandwidth at 60 GiB footprint | 114 GB/s | **203.1 GB/s** |
| Bandwidth at 62 GiB footprint | not sampled | **199.9 GB/s** |
| Reported pool (`total_memory`) | 82.99 GiB | **107.87 GiB** |
| Alloc-only ceiling | ≥100 GiB (saturated) | 95 GiB (not saturated) |

SUCCESS was `boundary >= 60 GiB`. Measured ≥62 GiB. **The BIOS carve-out controls the
fast tier.** KILL (`30 ± 4 GiB`) is clearly excluded — the cliff that sat at 30 GiB is
simply gone, with no degradation anywhere in the swept range.

The named confounder is checked and cleared: thermal throttling would have depressed
bandwidth *uniformly*, including at small footprints. Small-footprint bandwidth is
unchanged (205 vs 184–210 GB/s), so the change is the boundary moving, not the machine
slowing.

**Decision: keep BIOS UMA FB Size at 96 GB.** The usable KV-cache budget roughly
doubled, from ~30 GiB to ≥62 GiB, at the cost of Windows dropping to 31.6 GB.

### The boundary is a floor, not an edge

The sweep never found where bandwidth degrades, because it ran into a *different*
limit first. `fast_tier_boundary_gib = 62` means "still fast at 62 GiB", not "slow at
64 GiB". Whether a fast tier of ~96 GiB exists is unmeasured.

### Unplanned finding — single buffers ≥32 GiB fault or hang

Not part of this hypothesis; recorded here because it was found by this run and it
constrains the same experiments.

| Buffer | Footprint | Result |
|---|---|---|
| 31 GiB | 62 GiB | 199.9 GB/s, clean, fresh process |
| 32 GiB | 64 GiB | **hard hang** — 11 min at 0 CPU seconds, killed; host free RAM fell to 5 GB |
| 36 GiB | 72 GiB | **`hipErrorLaunchFailure`** ("unspecified launch failure") |

The GPU recovers fully in a fresh process — a trivial kernel runs correctly afterwards,
so there is no persistent damage. The two symptoms differ (hang vs fault); the 36 GiB
case ran in a process that had already allocated and freed several large buffers, the
32 GiB case was a fresh process, so history may matter.

`[A]` **Hypothesis, untested:** this is a 32-bit overflow in the copy path rather than
a capacity limit. 32 GiB is exactly 2^35 bytes — 2^34 fp16 elements — and a boundary
landing exactly on a power of two is characteristic of an index type rather than a
resource. Confidence: medium. **Cheapest discriminating test:** allocate 32 GiB as
fp32 (2^33 elements, same bytes). If it also fails, the limit is bytes; if it succeeds,
the limit is element count. A second axis: `fill_` alone vs `copy_`, to establish
whether the fault is specific to the copy kernel.

This matters more than it looks. A hang at 0 CPU is silent — a long training run would
simply stop, not crash — and single tensors ≥32 GiB are exactly the regime long-context
KV-cache work lives in. It becomes a Hardware Validation Gate item, pre-registered
separately rather than chased here; the 90-minute timebox on environment work had
already expired when it surfaced.

### Standing caveats

Every number above is **single-run — an anecdote by the house standard** (CLAUDE.md:
≥3 seeds, confidence intervals). They are reported because the effect sizes are large
(~2x) and the boundaries sharp across adjacent points, not because the statistics are
adequate. Nothing here is a research result; it is instrument characterisation, and the
Hardware Validation Gate still has not run — bf16 numerics, determinism, and checkpoint
integrity remain unproven.

Correction appended 2026-07-26: the Control section's capacity figures were revised
before this file was committed, after two probes were found to be reporting their own
search bounds. The freeze point for this record is commit `106ce53` (`v0.2.0`); no
content has been altered since.
