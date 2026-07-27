# We will keep the BIOS UMA FB Size at 96 GB, trading system RAM for a larger fast tier

Status:   Accepted
Date:     2026-07-26
Deciders: Founder (owner — the BIOS change is his to make), Claude (research staff)

## Context

The Z13's 128 GB is unified, split between the OS and a BIOS-configured GPU carve-out.
At the shipped default the driver reported 16 GiB dedicated VRAM while ROCm still
exposed an 82.99 GiB pool by reaching into GTT/shared memory — so capacity looked
abundant.

It was not uniform. `notebook/uma-carveout-controls-fast-tier.md` measured bandwidth
against working-set footprint and found a sharp boundary: ~185–210 GB/s up to a 30 GiB
footprint, collapsing to 61.3 GB/s at 32 GiB. **The binding number was 30 GiB, not 83.**
The boundary did not sit on the 16 GiB dedicated line — a 24 GiB footprint still ran at
198 GB/s — so the mechanism was unexplained and it was not obvious the carve-out
controlled anything.

That made it a real experiment rather than a settings change, and it was pre-registered
as one, with SUCCESS (`boundary ≥ 60 GiB`) and KILL (`boundary stays at 30 ± 4 GiB`)
committed before the BIOS was touched.

Result: with a 96 GB carve-out the curve is **flat at 203–205 GB/s out to ≥62 GiB** with
no degradation anywhere swept, and the reported pool grew 82.99 → 107.87 GiB. The named
confounder is cleared — thermal throttling would have depressed small-footprint bandwidth
too, and it is unchanged. Cost: Windows drops from ~112 GB to 31.6 GB visible.

## Decision

We will keep BIOS UMA FB Size at **96 GB**, and treat **≥62 GiB at ~200 GB/s** as the
fast-tier budget that sizes every long-context and KV-capacity experiment.

We will *not* treat the ~107.87 GiB pool or the ≥74.40 GiB writable ceiling as the
planning number. Capacity beyond the fast tier exists and is slower; using it is a
deliberate choice to be measured, not a default.

## Consequences

**Makes easy.** The usable KV budget roughly doubled. It also hands the lab an unusual
research instrument: the fast/slow bandwidth ratio is a BIOS setting here rather than a
bus, which makes "is the eviction-versus-retention boundary a property of language
models or of a 10–50x PCIe ratio?" a swept variable instead of a doctrine. No discrete
GPU can vary it at all, and `research/synthesis.md` makes this one of two pursued
branches.

**Makes hard.** Windows has 31.6 GB. Anything memory-hungry on the host side — large
dataset preprocessing, many concurrent tools, a second model on CPU — now competes for a
third of what it had. One probe already drove the host into swap and had to be killed.

**Forecloses.** Nothing irreversibly; it is a reboot away. But changing it invalidates
every bandwidth and capacity number recorded under the current setting, so a change is a
re-measurement, not a tweak.

**Debt taken.** The ≥62 GiB figure is a **floor, not a measured edge** — the sweep hit
the ≥32 GiB single-tensor fault before it found a bandwidth knee, so where the fast tier
actually ends is unknown. It is also **one run per arm**, an anecdote by this repo's own
standard. Repayment trigger: the Hardware Validation Gate re-measures with repeats, and
the ≥32 GiB fault is characterised well enough to sweep past it.
