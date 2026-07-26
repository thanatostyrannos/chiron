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

_(written once, after the run)_
