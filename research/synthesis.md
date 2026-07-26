---
title: What we believe, and what is worth our compute
version: 1.0.0
date: 2026-07-26
status: proposed — awaiting founder review
---

# Synthesis — what we believe, and what is worth our compute

Written to the G1 standard: SCQA, answer first, three arguments, then evidence. Sources
are the seventeen survey notes in `research/memory/` and `research/notes/` (~95,000
words, **619 distinct arXiv ids machine-verified, 0 unresolved**) and our own `[M]`
measurements in `ASSUMPTIONS.md`. Every finding ends in a *therefore*.

---

## SCQA

**Situation.** The lab holds a citation-verified map of LLM memory systems, a reference
architecture read from the shipped artifact rather than quoted, and an instrument whose
defining property is inverted relative to production. `[M]` a ≥62 GiB fast tier at
~200 GB/s means that at 300M params a KV cache can be roughly 100× the size of the
weights — a ratio no production system ever sees, and one that makes the expensive
counterfactual in memory research *cheap here*.

**Complication.** Nothing measured on this machine is admissible yet: `bf16-numerics-unproven`
is untested and the Hardware Validation Gate has not run. Worse, the instruments that
would judge any memory policy are themselves uncalibrated, and the most-used one is
adversely selected against the mechanism we study — a needle is a high-salience span that
attracts attention mass, which is exactly what heavy-hitter eviction retains, so
needle-in-a-haystack **structurally cannot fail** for H2O-style policies.

**Question.** What do we build and run next such that every outcome changes the plan
(the G3 test), and what do we refuse to touch?

**Answer.** Build the instrument, not another policy.

---

## Answer, stated once

**Ship an attribution instrument as the lab's deliverable, and add no new eviction policy
to a field that has ~30 of them and no dominance result.** Concretely, in order: close and
widen the Hardware Validation Gate; build Mnemosyne's full-cache oracle-diff harness with
a seed-to-seed null distribution; calibrate every eval by fault injection before it is
allowed to certify anything. Then run the one experiment no discrete-GPU lab can — sweep
the fast/slow bandwidth ratio, which on this platform is a BIOS setting rather than a bus.

Two independent syntheses, written from a decision lens and a contribution lens without
seeing each other, converged on this recommendation and on the same riskiest assumption.
That convergence is the strongest single piece of evidence in this document.

---

## Three arguments

### 1. The field's binding constraint is measurement, and measurement is the one thing our scale makes *easier*

`open-problems-ranked.md` scores attribution P5·T5·E5 — the only 5/5/5 on the list. Four
documented cases where the outcome metric held while the mechanism broke: refusals down
15.2% at 1.03× perplexity across 11 models and 1,894 prompts `[C]` (2606.09864); specific
instructions dropped entirely under StreamingLLM/SnapKV/TOVA/H2O while LongBench looked
fine `[C]` (2510.00231); single-turn rankings not surviving multi-turn cache reuse `[C]`
(2412.10319); mean-aggregated rankings not surviving worst-case aggregation `[C]`
(2510.13334). The 2026 serving survey names seven KV quantities nobody measures `[C]`
(2607.02574).

Attribution requires a **full-cache oracle on every probe** — you must run the expensive
thing you were trying to avoid. At 300M that is ~600 MB of weights against a 62 GiB fast
tier `[M]`. At 70B it is unaffordable.

**Therefore** small scale is not a compromise for this question, it is the enabling
condition — and `mnemosyne-core` should deliver an instrument with a null distribution,
not a policy implementation.

### 2. A class of published KV-tiering conclusions may be a statement about interconnects, not about language models

`[M]` 2026-07-26, identical probe, two BIOS configurations: at a 16 GiB carve-out,
~185–210 GB/s to a 30 GiB footprint then collapse to 61.3 GB/s; at 96 GB, flat 203–205
GB/s out to ≥62 GiB. The offload/CXL tiering literature is designed around a GPU-HBM-to-
host-DRAM ratio of order 10–50× across PCIe. Retention beats eviction as that ratio falls,
because refetching costs bytes/slow-bandwidth while recompute does not.

**Therefore** if the eviction-versus-retention boundary flips at a ratio of 2–3×, a body
of published design guidance is conditional on a bus. We can vary that ratio; a discrete
GPU cannot vary it at all.

### 3. Most of the survey's highest-pain problems are explicitly *not* ours, and saying so is the most valuable part of this plan

The most tempting item — distributed/disaggregated KV, CXL pooling, shared stores — scores
E=5 on our edge and is disqualified on T: it needs multiple machines, and
`single-device-only` is `[C]` a hard constraint. Agent-memory security is the best-evidenced
pain in the register and is a different research programme. RLVR at our scale actively
regresses: `[C]` (2606.22189) ran our experiment at 135M on one GPU and GSM8K exact match
*fell*.

**Therefore** park them with written un-park triggers, rather than letting them absorb
attention by seniority.

---

## MECE issue tree — the memory problem space

```
LLM memory
├── 1. What is stored
│   ├── 1.1 Parametric (weights)                          — PARK: needs pretraining scale
│   ├── 1.2 Recurrent/activation state                    — partial: recall-cliff arm only
│   ├── 1.3 KV cache                                      — PURSUE
│   ├── 1.4 External retrieval index                      — PARK: not our gap
│   └── 1.5 Cross-session store                           — PARK: different programme
├── 2. Where it lives
│   ├── 2.1 On-device, single tier                        — baseline
│   ├── 2.2 Multi-tier on one device                      — PURSUE (the BIOS-knob arm)
│   └── 2.3 Distributed across machines                   — PARK: no collectives
├── 3. What is discarded, and how it is decided
│   ├── 3.1 Eviction policy design                        — PARK: ~30 exist, no dominance
│   ├── 3.2 Compression/quantization                      — PARK: confounded by bf16
│   └── 3.3 Retention-vs-refetch economics                — PURSUE (follows from 2.2)
└── 4. How any of it is judged
    ├── 4.1 Task-outcome evaluation                       — known insufficient
    ├── 4.2 Mechanism attribution                         — PURSUE (the deliverable)
    └── 4.3 Eval calibration / fault injection            — PURSUE (gates everything)
```

**Pursuing:** 4.2 + 4.3 (attribution and its calibration), 1.3 + 2.2 + 3.3 (the tier-ratio
arm). **Parking:** everything else, above, with reasons.

---

## The three to five questions worth our compute

1. **Does divergence from a full-cache oracle localise to identifiable dropped cache
   entries?** The instrument's foundational question. Testable at 150M on one GPU.
2. **Does the eviction-versus-retention boundary move with the fast/slow bandwidth ratio?**
   The BIOS carve-out is the independent variable. Nobody else can run this.
3. **Can any eval we own detect a memory regression?** Six-fault calibration battery
   (needle absent, needle's KV dropped, uniform eviction, RoPE-phase corruption,
   retrieval-head masking, haystack shuffle). Zero training runs; publishable methodology.
4. **Does decode arithmetic intensity actually equal the GQA group size on gfx1151?**
   Sweep G ∈ {1,2,4,8} against the measured ~105 FLOP/byte ridge. If it does not track,
   the bandwidth model under this entire track is wrong on our hardware.
5. **Does a memory-policy ranking at 20M–300M survive to deployment scale?** The
   `ablation-scale-sufficient` assumption, which everything else rests on.

---

## Folklore — repeated without controlled evidence

- **"3:1 is the right hybrid ratio."** Four labs shipped it; none ablated it against the
  others. Every ablation reporting a quality surface reports a *flat* one.
- **"The model has a 1M context."** Laguna's 1,048,576 is 8192 × 128 exactly — pretraining
  length times YaRN extension factor. Its `attention_factor` matches YaRN's default
  temperature formula to the last digit `[M]`. Inherited convention, not demonstrated
  capability.
- **"MLA gives 93% KV reduction."** Not in the HF reference implementation, which expands
  the latent back to full per-head K and V *before* the cache write `[M]`.
- **"The KV cache is a storage tier."** It is a memo table with no backing store: no fault
  path, no miss signal, no durability contract. Discarding is always legal.
- **"PyramidKV's per-layer budget is what buys the gain."** It degenerates to SnapKV at
  aggressive ratios by the paper's own account.

## Contested — left contested

Whether the KV cache is "memory" at all (serving vs agent-memory literatures, incompatible
vocabulary, rarely cross-citing); whether the hybrid ratio sets a capability ceiling
`[C]` (2507.06457) or only governs how fast long-context ability emerges `[C]` (2606.15378);
whether efficient attention is worth it at all (MiniMax shipped full attention on
reliability grounds; Kimi Linear claims the opposite under matched pretraining); AdamW vs
Muon at scale.

## What is weakest in our own evidence

An adversarial completeness review of the seventeen notes and of `ASSUMPTIONS.md` found
25 overclaims, most of them ours. Acted on: `z13-is-right-instrument` downgraded to
**untested** (it rested on one un-peer-reviewed GitHub issue while being the load-bearing
justification for the entire hardware strategy); `hardware-capacity-ceiling`'s "filled"
claim corrected to a ≥74.40 GiB floor; `decode-intensity-varies-by-layer` retagged `[A]`
because it is a derivation, not a measurement; `gpu-fast-tier-size` and `mnemosyne-separable`
status wording corrected to match what was actually shown.

Still weak and load-bearing: the ≥62 GiB fast tier is **one run per arm**; the ~105
FLOP/byte ridge is a ratio of two single-run numbers of different kinds, neither an
attention kernel; the 32 GiB fault rests on two observations with an untested mechanism;
and the `[A]` sustained-throughput figure that sizes every wall-clock estimate in the
backlog is anchored to a single GEMM microbenchmark.

**Therefore** the Hardware Validation Gate as written in CLAUDE.md is under-specified, and
should be widened before it is closed: add RoPE-at-long-position in bf16, an
attention-kernel roofline (not a GEMM one), and the fp32 discriminator for the 32 GiB fault.

---

## Riskiest assumption

**That distributional divergence from a full-cache oracle measures anything
decision-relevant.** Both independent syntheses named this without prompting. The whole
plan pivots on a differential instrument, and divergence and task accuracy can dissociate
in both directions: a policy can shift the output distribution without flipping any argmax,
or flip one critical token at negligible average KL.

**Next test.** Before building the harness: take a known-good model, drop a *known*
cache entry, and check whether per-token KL against the full-cache reference localises to
that entry and moves only when the recoverable token moves. If it does not, branch 1
collapses and the plan changes — which is what makes it worth running first.
