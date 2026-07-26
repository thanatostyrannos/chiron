---
title: Open problems in LLM memory systems, ranked into an ablation backlog
version: 1.0.0
date: 2026-07-26
---

# Open problems, ranked — the ablation backlog

This note settles what Chiron works on next and, more usefully, what it refuses to work
on. It ranks the open problems in LLM memory systems on three axes — how real the pain
is, whether progress is *testable* on one gfx1151 GPU with a measured ≥62 GiB fast
memory tier and no working multi-GPU, and whether thirty years of storage/caching/
observability is a genuine edge rather than a comforting analogy. The ordering is
opinionated on purpose: it becomes `BACKLOG.md`, and a backlog that hedges is a backlog
that never starts.

---

## How the ranking was made

Three scores, 1–5 each.

| Axis | Question | What a 5 looks like |
|---|---|---|
| **P — pain** | Is this a real, evidenced problem, or a paper-generating problem? | Multiple independent 2026 papers exist *because* the previous generation got it wrong. |
| **T — testability here** | 20M–300M params, one GPU, ≥62 GiB fast tier `[M]`, no collectives. | Runnable this month with committed code and no rented hardware. |
| **E — edge** | Does systems/storage/observability expertise buy a real advantage? | The literature's stated gap is a measurement gap, which is his day job. |

**The sum is not the ordering rule.** A T ≤ 2 is disqualifying regardless of P and E —
an unrunnable experiment has zero information value per dollar, which is the G3 test.
That single rule is what moves the most tempting item on the list (distributed KV
tiering, where E = 5) into the parked section.

---

## The envelope, stated once

Everything below is bounded by six measured facts. They are instrument characterisation,
not results.

| Fact | Value | Tag |
|---|---|---|
| Fast memory tier, flat bandwidth | **≥62 GiB at ~200 GB/s**, upper edge unmeasured | `[M]` 2026-07-26, `notebook/uma-carveout-controls-fast-tier.md`, single run |
| Slow tier beyond the old 30 GiB boundary | 61–114 GB/s (measured at the 16 GiB carve-out, before the BIOS change) | `[M]` same source, single run |
| Single tensors ≥32 GiB | **hang at 0 CPU or fault** — 31 GiB is clean, 32 GiB hangs silently | `[M]` `ASSUMPTIONS.md: large-tensor-fault-32gib` |
| GEMM, bf16, 8192³ | 20.9 TFLOP/s configured | `[M]` `scripts/benchmark_gemm.py` |
| Distributed collectives | incomplete on gfx1151 — single device, always | `[C]` ROCm docs, `ASSUMPTIONS.md: single-device-only` |
| bf16 numerics | **unproven** — the Hardware Validation Gate has not run | `[C]` five documented gfx1151 bf16 bugs (ROCm #6034) |

The last row is a hard gate: no arm below produces evidence until the gate is green.

**Practical consequence of the 32 GiB fault, because it will bite:** a 62 GiB KV cache
cannot be one tensor. Allocate per layer. At 24 layers that is 2.6 GiB each, comfortably
under the cliff, and it happens to be the layout every real implementation uses anyway
(vLLM's pool is per-layer-group because page size in *bytes* differs per layer type —
`CODE_MAP`, `kv_cache_interface.py`).

### The one piece of math that decides the whole ranking

Write the KV cache cost per token:

```
B_tok = 2 · L · H_kv · d_head · s
```

- `2` — one key vector and one value vector, per layer, per token
- `L` — number of layers
- `H_kv` — key/value heads per layer (under GQA, fewer than query heads)
- `d_head` — channels per head
- `s` — bytes per element (2 for bf16)

For the reference model `[M]` (Laguna S 2.1, read from the artifact, not quoted):
`2 · 48 · 8 · 128 · 2 = 196,608 B = 192 KiB/token`. Against a 62 GiB fast tier that is
**≈339,000 tokens** of resident cache — and that is the *upper* bound, since 36 of 48
layers are windowed at 512 (`ASSUMPTIONS.md: kv-per-token-laguna`).

Now the same arithmetic at our own ablation scale. A 300M-class model with `L=24`,
`H_kv=4`, `d_head=64`, bf16: `B_tok = 24,576 B = 24 KiB/token`, so 62 GiB holds
**≈2.7 million tokens** — against ~600 MB of weights. **The KV cache can be 100× the
model.** In production it is usually comparable to or smaller than the weights. That
inverted ratio is the single most unusual thing this lab owns, and the ranking below is
largely a list of experiments that are only cheap in that regime.

Now the part that explains *why* memory is the bottleneck at all. In decode, attention
over `N` cached tokens does, per layer per KV head: `2·d_head` FLOP for the query-key dot
product and `2·d_head` FLOP for the value-weighted sum, per cached token — while reading
`2·d_head·s` bytes of cache. Arithmetic intensity, in FLOP per byte moved:

```
I = 4·d_head / (2·d_head·s) = 2/s
```

`d_head` cancels. **At bf16 that is exactly 1 FLOP per byte, independent of model size,
context length, head count, and batch size.** GQA is the only lever: if `G = H_q/H_kv`
query heads share one KV head, the same bytes serve `G` dot products and `I = 2G/s`.
Laguna's nominal `48/8 = 6` gives `I ≈ 6`.

Compare against this machine's ridge point — the intensity at which compute and bandwidth
are balanced — computed from two `[M]` numbers: `20.9 TFLOP/s ÷ 200 GB/s ≈ 104 FLOP/byte`.

So attention decode sits **17–100× on the memory-bound side of the ridge**. You cannot
buy your way out with FLOPS, and unlike the weight-reading part of decode, you cannot
batch your way out either: every sequence has its own KV cache, so more sequences means
proportionally more bytes. This is the 2019 observation `[C]` (1911.02150) with 2026
numbers, and it is the reason a memory-systems agenda is defensible on a slow GPU.

---

## The ranking

### 1. Attribution: isolate *which mechanism* a cache policy's gain came from

**P 5 · T 5 · E 5**

The literature's own diagnosis, not ours. "The Pitfalls of KV Cache Compression" `[C]`
(2510.00231, 2025) shows five standard policies silently dropping *specific
instructions* while aggregate scores look fine. "When Does Value-Aware KV Eviction Help?"
`[C]` (2605.08234, May 2026) argues task accuracy alone cannot tell you why a selector
worked. "Error Certificates for KV-Cache Eviction via Randomized Design" `[C]`
(2607.21475, Jul 2026) makes the point most sharply — randomization buys *attribution*,
not prediction, separating cache-induced from inherent failures. And the field openly
suspects misattribution in its own canon: several groups argue most of PyramidKV's
reported gain `[C]` (2406.02069) comes from SnapKV's observation window `[C]` (2404.14469)
rather than from the per-layer budget allocation it claims credit for.

**The bridge.** This is enterprise observability. You have spent thirty years
distinguishing "p99 improved" from "which subsystem caused p99 to improve," and the
tooling instinct — trace the request, tag the span, diff against a control — transfers
directly. **Where it breaks, three ways.** First, there is no request id: causality runs
through a continuous attention weight distribution, not a call graph. Second, the
counterfactual is only obtainable by *recomputation with the full cache* — you must run
the expensive thing you were trying to avoid, on every probe. Third, and this is the one
that will surprise him: **instrumentation costs throughput directly.** OLMo-core keeps
metrics as unevaluated device tensors and drains them every 5 steps specifically because
reading a GPU tensor's value stalls the pipeline (`CODE_MAP`, `trainer.py:1037` and
`:1394`). No logging system he has run in production has that property.

**The shape of the arm.** An oracle-diff harness in Mnemosyne: same prompt, same seed,
run once with a full cache and once under policy `P`; log per-token KL divergence between
the two output distributions; attribute each divergence spike to the specific cache
entries the policy dropped. Then ablate the policy into its parts (observation window,
scoring function, budget allocation, sink pinning) and show which part moves the
divergence. This is small-scale-*favoured* work: at 300M you can afford the full-cache
oracle on every probe, which nobody at 70B can.

**Riskiest assumption:** that output-distribution divergence localises to identifiable
cache entries rather than smearing across all of them. Cheapest test that would move it —
a single synthetic needle-retrieval prompt where the correct attribution is known by
construction.

**Follow-on it unlocks:** learned eviction `[C]` (2602.10238, Feb 2026) is the direct
analogue of learned cache replacement, and it is *downstream* of this arm — you cannot
train a policy without a per-decision reward signal, which is exactly what the harness
produces.

---

### 2. Eviction versus retention when the tier ratio is 3×, not 50×

**P 4 · T 4 · E 5**

Eviction-vs-retention is explicitly unresolved. Permanent eviction (H2O `[C]` 2306.14048,
SnapKV `[C]` 2404.14469) cuts capacity irreversibly; full-retention plus tiered fetch
preserves fidelity and cuts bandwidth but not capacity; RocketKV `[C]` (2502.14051, 2025)
claims the two are orthogonal and composable, while a large 2026 tiering literature
treats eviction as the wrong primitive entirely `[C]` (2607.02574, Jun 2026; 2604.26968,
Apr 2026; 2607.18141, Jul 2026; 2512.18194, Dec 2025). The 2026 technique-to-scenario map
`[C]` (2603.20397, Mar 2026) concludes no single method dominates.

**Here is the claim worth making.** `[A]` high confidence: that entire tiering literature
is designed around a fast/slow bandwidth ratio of order 10–50× — GPU HBM against host
DRAM across PCIe — and the ratio, not the algorithm, is what makes "evict and recompute"
beat "retain and fetch." Cheapest test that would move this assumption: tabulate the
assumed tier bandwidths in 2604.26968, 2607.18141 and 2512.18194 and check the spread.

Our machine has a ratio near **2–3×** `[M]` (~200 GB/s fast, 61–114 GB/s beyond the
boundary — though note those two figures come from *different* BIOS configurations of the
same single sweep, so establishing the ratio cleanly under one configuration is step
zero). Write the decision boundary explicitly. Let `r = B_fast / B_slow`. Fetching an
evicted entry back costs `B_tok / B_slow`; recomputing it costs a prefill pass over the
prefix. Retention wins when

```
B_tok / B_slow  <  c_recompute  +  λ · (accuracy lost to eviction)
```

As `r` falls, the left side falls with it, and the inequality flips in favour of
retention. **If that flip is real, a whole class of published design conclusions is
conditional on a hardware ratio rather than on anything about language models** — and
this lab can demonstrate it because the BIOS UMA carve-out is a *knob on r*, already
shown to move the boundary from 30 GiB to ≥62 GiB `[M]`. Datacenter hardware does not
give you that knob.

**The bridge.** Working-set analysis, hit rate versus miss cost, tier sizing. Home turf.
**Where it breaks — and this is the most important break in the whole note:** *a KV cache
miss is not a latency event, it is a correctness event.* There is no fault handler.
FlashInfer's page table cannot even represent a miss (`CODE_MAP`, `decode.py:1239`);
llama.cpp's SWA tier makes out-of-window tokens architecturally unreadable via
`is_masked_swa`, so discarding them is lossless rather than a gamble; SGLang's plain
`RadixCache` has no backing store at all, so the only "reload" is a full prefill
(`CODE_MAP`, `radix_cache.py:565`). The model does not stall, does not retry, and does not error — it
emits a fluent wrong answer. **Hit rate here is an accuracy metric wearing a performance
metric's clothes.** Mooncake makes the inversion explicit and exploits it: `offload_force_evict`
throws bytes away rather than block on writeback (`CODE_MAP`, `master_service.cpp:6382`),
a trade no storage tier is permitted to make.

---

### 3. Does a memory-policy ranking at 20M–300M survive to deployment scale?

**P 3 (field) / 5 (us) · T 5 · E 3**

`ASSUMPTIONS.md: ablation-scale-sufficient` is `untested`, `[A]` medium confidence, and it
is the riskiest assumption in the entire program: if policy rankings do not transfer,
every result this lab produces is a curiosity. Evidence that rank instability is real and
not paranoia: proxy-model work `[C]` (2512.24503, Dec 2025) finds dataset rankings
preserved only under specific learning-rate and batch-size conditions; ATLAS `[C]`
(2605.28079, May 2026) reports seven models shifting 2+ ranks between the 8K–128K and
8K–1M length regimes, with gaps up to 12 positions.

**Decision: this is not a project, it is a rider.** Every arm in #1, #2, #4 and #5 runs at
two scales (roughly 30M and 300M, a 10× span) and reports Spearman rank correlation of
the arm ordering between them. muP `[C]` (2203.03466) is mandatory, not optional — without
it, "policy A beat policy B" is indistinguishable from "policy A happened to be better
tuned." And when IsoFLOP curves get fitted, use Approach 3 with variable projection, not
the parabola: the standard fit is biased even on noise-free data, and the bias sources it
names (narrow grid, off-centre sampling, loss-surface asymmetry) are precisely the
conditions of a 20M–300M sweep `[C]` (2603.22339, Mar 2026; cf. 2203.15556).

**The bridge, and it is a good one:** this is "does the staging environment predict
production?" He has thirty years of scar tissue on exactly that question. **Where it
breaks:** in staging, the failure mode is that you under-provisioned and missed a
bottleneck. Here the failure mode is that the *ordering inverts* — a policy that wins at
30M can lose at 300M because the thing it exploits (attention sparsity, outlier channel
structure, retrieval-head formation) has not emerged yet at the smaller size.

---

### 4. Does the right retention prior change as the context window fills?

**P 4 · T 5 · E 3**

Every eviction policy since StreamingLLM pins a prefix `[C]` (2309.17453, 2023) and most
lean on the lost-in-the-middle U-shape `[C]` (2307.03172, 2023). But the U-shape holds
only up to roughly 50% context occupancy — past that, primacy decays, recency persists,
and the bias becomes distance-based `[C]` (2508.07479, Aug 2025). **Nobody has connected
those two facts.** If the positional value prior is occupancy-dependent, then a fixed
retention prior is wrong at exactly the occupancy where eviction starts mattering, and the
correct policy is a *schedule* over occupancy rather than a constant.

**Why it ranks here:** it is the cheapest genuinely novel experiment on the list. It needs
one small model, a synthetic multi-position retrieval task in the RULER style `[C]`
(2404.06654), and a sweep over occupancy — no new kernels, no training beyond the base
arms. `[A]` medium confidence that the effect survives at 300M; cheapest test is to
reproduce the occupancy-dependent bias itself at our scale before building any policy on
top of it. If it does not reproduce, this arm dies in a day, which is a feature.

**The bridge:** cache admission policy that changes with pressure — segmented LRU,
watermark-triggered behaviour change. SGLang implements exactly that shape, with the whole
replacement-policy surface reduced to one `get_priority(node)` function
(`CODE_MAP`, `evict_policy.py:16`). **Where it breaks:** in a storage cache, pressure
changes *how much* you evict; here the claim is that pressure changes *which entries are
valuable*, because the model's own positional attention behaviour shifts. Value is not a
property of the entry. It is a property of the entry, the query, and the occupancy.

---

### 5. An exact cache in front of a lossy state: two-tier memory as architecture

**P 4 · T 4 · E 4**

The constant-state literature has a capacity wall and knows it. Zoology's MQAR `[C]`
(2312.04927, 2023) isolates the failure as a state-capacity limit, not a training
artifact; the recall–throughput frontier is explicitly a fixed memory budget you spend
`[C]` (2402.18668, 2024); effective state-size `[C]` (2504.19561, Apr 2025) separates
theoretical capacity from utilisation, and a 2025 re-examination `[C]` (2508.19029)
revisits how well associative recall predicts anything. The 2026 answers split: grow the
state sparsely `[C]` (2607.07386, Jul 2026), grow it logarithmically `[C]` (2506.04761),
or bolt a bounded *exact* KV cache onto a delta-rule compressive state `[C]` (2607.02303,
Jul 2026 — single-author, unreplicated). That last one is the Mnemosyne-shaped idea: a
small exact tier in front of a large lossy one.

**CONTESTED, and do not let this note pick a side.** MiniMax abandoned hybrid linear
attention for M2 on reliability grounds; Kimi Linear `[C]` (2510.26692, Oct 2025) claims a
3:1 hybrid matching or beating full attention under matched-scale pretraining. Both are
shipping-product retrospectives with commercial incentives, not controlled ablations.

**The bridge:** L1 in front of a compressed L2. **Where it breaks, and this one is worth
internalising because the intuition is exactly backwards:** the recurrent state is not a
cache with a TTL. Gated DeltaNet's decay is *one scalar per head applied to the entire
K×V matrix* — every stored association is attenuated by the same factor every step,
whether or not anything is written, and there is no per-key TTL even though the kernel
already implements per-channel gates and the layer simply does not pass them (`CODE_MAP`,
`naive.py:54`, `fused_recurrent.py:138-150`). All content-dependent selectivity lives in
the *delta* term, which erases exactly one direction via a read-before-write. So "gating =
selective forgetting" is backwards: **the gate is indiscriminate decay; the delta is the
targeted erase.** And the failure mode is interference, not a miss — a *similar* key
partially clobbers a neighbour's content, because keys are L2-normalised continuous
vectors with no addresses and no lines. Nothing can be paged back in, because the
information was destroyed rather than relocated.

Ranked below #4 only because it requires training real arms at matched budget rather than
probing an existing model, and because the fair 2026 baselines keep moving `[C]`
(2412.06464; 2605.22791; 2603.15569).

---

### 6. A measured cost model for a byte of KV on unified memory

**P 3 · T 5 · E 5**

Low novelty, high leverage, nearly free. Every arm above makes a cost claim, and right now
this lab's cost claims rest on a device-to-device `copy_` microbenchmark, which is the
*best* case and not what an attention kernel achieves. Measure: achieved bandwidth for
paged KV reads at realistic page sizes and layouts (NHD versus HND changes whether a head's
slice of a page is a contiguous burst or a strided gather — `CODE_MAP`, `page.py:403`),
the cost of a cross-tier migration, and the actual decode token rate against the roofline
bound derived above. Publish the table. `[A]` high confidence nobody has published a
KV-cost model for a unified-memory APU part; cheapest test is a literature check before
spending a day on it.

**Where the analogy breaks:** on a discrete GPU, "host memory" and "device memory" are
different address spaces with a bus between them, and the cost model is a transfer cost.
Here they are the same physical DRAM behind a carve-out, so the "tier boundary" is a
*driver policy*, not a physical one — which is why a BIOS setting moved it by 2× `[M]`.
A storage engineer's instinct that tiers are physical is wrong on this machine, and that
is precisely what makes it an interesting instrument.

---

## The second rig, and why it changes the parked list

The brief assumes 20M–300M trained models. That is one rig. There is a second, and it is
sitting unused: **inference-only studies on an off-the-shelf 7–14B model.** At bf16, 7B is
14 GB of weights and 14B is 28 GB — both fit inside the ≥62 GiB fast tier `[M]` with room
for a large KV cache. The roofline upper bound on batch-1 decode, computed from the `[M]`
200 GB/s figure, is ~14 tok/s at 7B and ~7 tok/s at 14B; `[A]` the achievable fraction is
unmeasured and will be lower, because `copy_` bandwidth is not kernel bandwidth — cheapest
test is one decode benchmark. Even at half of that, a 500-prompt × 200-token evaluation is
a few hours.

This matters because the most *painful* problems in the field — instruction dropping under
compression `[C]` (2510.00231), alignment collapse under KV quantization `[C]` (2606.09864,
Jun 2026), governance decay under compaction `[C]` (2606.22528, Jun 2026) — all require
instruction-following capability a 300M model does not have, and all of them are
inference-only. Splitting the backlog by rig converts several hard parks into scheduled
work. Arms #1, #2 and #4 all have inference-rig variants that reach deployment-relevant
scale without a single training run.

---

## Parked, and why

**Distributed / disaggregated KV, CXL pooling, shared stores.** E = 5, T = 1. This is the
most tempting item on the entire list given his background — Mooncake `[C]` (2407.00079),
TraCT `[C]` (2512.18194), HyMCache `[C]` (2607.18141), and the survey that names seven
unmeasured quantities in this exact area `[C]` (2607.02574) are all squarely his
vocabulary. **We cannot run it.** `single-device-only` is `[C]`-supported, not a
preference. Second reason, and it is the more interesting one: our platform *collapses*
the HBM/DRAM boundary these designs are built around, so even with the nodes it would be a
poor testbed. Un-park trigger: rented multi-node hardware with a specific pre-registered
question, gated on approval.

**Hybrid ratio selection (SWA:global, linear:full).** P = 4, T = 3, E = 2. Contested, and
the contest is about *token budget*: 2507.06457 finds recall degrading sharply below 3:1
(a real ceiling), while 2606.15378 `[C]` (Jun 2026) argues configurations converge given
enough training, so the choice governs how fast long-context ability emerges rather than
its ceiling. Token budget is the axis we can least afford to sweep — the entry fee in the
first paper was 72 trained models. Un-park trigger: post-hoc conversion `[C]`
(2606.30562, Jun 2026 — freeze weights, learn layerwise gates to pick which layers keep
full attention) proving out, which would move ratio search *after* pretraining and collapse
the cost.

**Sub-4-bit KV quantization.** P = 4, T = 2, E = 3. KIVI and KVQuant claim near-lossless
2-bit `[C]` (2402.02750; 2401.18079); 2026 work pushes back hard, and alignment collapse
`[C]` (2606.09864) shows perplexity-invisible damage. We should not touch it *yet* for a
specific reason: `bf16-numerics-unproven` is `untested`, so any quantization error we
measure is confounded with hardware error. Second reason, `[A]` medium confidence: the
per-channel key outlier structure these methods exploit may not exist at 300M — cheapest
test is to measure the outlier structure itself before building anything. Un-park trigger:
Hardware Validation Gate green **and** outlier structure confirmed at our scale.

**Agent-memory poisoning and lifecycle security.** P = 5, T = 2 on the training rig,
E = 3. The literature is excellent and new — a six-phase lifecycle threat model `[C]`
(2604.16548, Apr 2026), write-channel taxonomy including compaction-driven writes `[C]`
(2606.04329, Jun 2026), monotonic risk accumulation with exposure length `[C]`
(2605.17830, May 2026) — but the phenomena need agentic capability we cannot train. Note
the adjacent finding that *where* the control plane sits determines which failure modes
are even addressable, with mutation-time placement winning `[C]` (2606.15903, Jun 2026);
that is a design input for Mnemosyne's plug point regardless. Un-park on the inference rig
once #1's harness exists.

**Session-scoped KV lifetime.** The survey names it as an *empty* design point despite
chat and agent workloads obviously wanting it `[C]` (2607.02574). E = 5, T = 2 — it is a
serving-system question and we have no fleet, no multi-tenancy, and no realistic workload
trace. Park with regret; revisit if the lab ever acquires a workload.

**Compaction as a serving problem.** `[C]` (2605.23296, May 2026; 2606.11213; 2605.08580).
Latency, blocking and overlap — his native language — but it optimises a pipeline we do not
run.

---

## Open questions

Testable here: single GPU, 20M–300M params, ≥62 GiB fast tier, no multi-GPU.

1. Does output-distribution KL divergence under a cache policy localise to identifiable
   evicted entries, or does it smear? (Decides whether arm #1 is possible at all.)
2. Under a fast/slow bandwidth ratio of 2–3× rather than 10–50×, does retention-plus-fetch
   beat permanent eviction at matched accuracy? At what ratio does the decision flip?
3. What *is* the clean fast/slow ratio on this machine, measured under one BIOS
   configuration with ≥3 seeds? (The current 2–3× estimate mixes two configurations.)
4. Do eviction-policy rankings at ~30M agree with rankings at ~300M (Spearman ρ), under
   muP, on a fixed synthetic recall suite?
5. Does the occupancy-dependent shift in positional bias (U-shape below ~50%, distance-based
   above) reproduce at 300M — and if so, does an occupancy-scheduled retention prior beat a
   fixed one?
6. How much of SnapKV-family gain is the observation window versus the scoring function
   versus the budget allocation, decomposed at matched budget?
7. At what exact-cache size does a bounded KV tier in front of a delta-rule state recover
   full-attention MQAR accuracy, and how does that size scale with the number of key-value
   pairs to be recalled?
8. Does the per-channel key outlier structure that 2-bit KV quantizers exploit exist at
   20M–300M, or is it an emergent property of scale?
9. What fraction of the 200 GB/s `copy_` bandwidth does a paged attention kernel actually
   achieve on gfx1151, and does NHD versus HND layout change it materially?
10. Is the 32 GiB single-tensor fault a byte limit or an element-count limit? (Allocate
    32 GiB as fp32: same bytes, half the elements.)
11. Does a 300M model trained with a KV cache 100× the size of its weights exhibit
    qualitatively different retrieval behaviour than one trained at production ratios —
    i.e. is our unusual capacity regime a confound or a finding?

---

## Decision / Riskiest assumption / Next test

**Decision.** Run #1 (attribution harness) first, as a Mnemosyne deliverable, with #3
folded in as a two-scale rider on every arm. #2 (tier-ratio inversion) is the lab's most
defensible *contribution* and starts as soon as #1's measurement plumbing exists, because
#2 without attribution instrumentation produces exactly the outcome-without-mechanism
result this lab exists to avoid. #4 runs in parallel as a one-week probe because it is
cheap enough to kill fast. Everything else waits.

**Riskiest assumption.** Not any single item above — it is
`ablation-scale-sufficient`: that memory-policy conclusions at 20M–300M mean anything at
deployment scale. It is currently `[A]` medium confidence with zero evidence, and it is
load-bearing for the entire backlog.

**Next test.** Run the Hardware Validation Gate. Nothing above is evidence until bf16
numerics, determinism and checkpoint round-trip are green — and the gate is cheaper than
any arm on this list.

---

## Sources

All 67 arXiv ids cited in this note were resolved against the live arXiv API on
2026-07-26 — the ones drawn from `research/reference/papers/anchors.bib` were verified
originally by `scripts/verify_papers.py`, and every id here, old and new, was re-checked
by direct API lookup while writing. Titles below are the API's, not anyone's memory.

**Attribution and diagnostics**
2510.00231 *The Pitfalls of KV Cache Compression* (2025) ·
2605.08234 *When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic* (May 2026) ·
2607.21475 *Error Certificates for KV-Cache Eviction via Randomized Design* (Jul 2026) ·
2510.13334 *Taming the Fragility of KV Cache Eviction in LLM Inference* (2025) ·
2602.02199 *More Than a Quick Glance: Overcoming the Greedy Bias in KV-Cache Compression* (Feb 2026) ·
2602.05929 *KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches* (Feb 2026) ·
2412.10319 *SCBench: A KV Cache-Centric Analysis of Long-Context Methods* (2024)

**Eviction, compression, retention**
2306.14048 *H2O: Heavy-Hitter Oracle* (2023) ·
2309.17453 *Efficient Streaming Language Models with Attention Sinks* (2023) ·
2404.14469 *SnapKV* (2024) ·
2406.02069 *PyramidKV* (2024) ·
2504.15364 *KeyDiff* (2025) ·
2502.14051 *RocketKV* (2025) ·
2602.10238 *Learning to Evict from Key-Value Cache* (Feb 2026) ·
2603.20397 *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference* (Mar 2026)

**KV mechanics and quantization**
1911.02150 *Fast Transformer Decoding: One Write-Head is All You Need* (2019) ·
2305.13245 *GQA* (2023) ·
2405.04434 *DeepSeek-V2* / MLA (2024) ·
2401.18079 *KVQuant* (2024) ·
2402.02750 *KIVI* (2024) ·
2412.19442 *A Survey on LLM Acceleration based on KV Cache Management* (2024) ·
2606.19348 *DeepSeek-V4* (2026) ·
2607.08057 *Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization* (Jul 2026) ·
2606.09864 *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (Jun 2026)

**Serving hierarchy and tiering**
2309.06180 *PagedAttention / vLLM* (2023) ·
2312.07104 *SGLang / RadixAttention* (2023) ·
2405.04437 *vAttention* (2024) ·
2407.00079 *Mooncake* (2024) ·
2512.18194 *TraCT: CXL Shared Memory KV Cache at Rack-Scale* (Dec 2025) ·
2604.26968 *Predictive Multi-Tier Memory Management for KV Cache* (Apr 2026) ·
2607.02574 *From Tensor Buffer to Distributed Memory Hierarchy* (Jun 2026) ·
2607.18141 *HyMCache: KV Cache for Multi-Turn LLM Serving with CXL-Hybrid Memory* (Jul 2026)

**Constant-state memory and hybrids**
2312.04927 *Zoology: Measuring and Improving Recall* (2023) ·
2405.21060 *Transformers are SSMs* (2024) ·
2402.18668 *Simple linear attention balances the recall-throughput tradeoff* (2024) ·
2412.06464 *Gated Delta Networks* (2024) ·
2504.19561 *Quantifying Memory Utilization with Effective State-Size* (Apr 2025) ·
2506.04761 *Log-Linear Attention* (2025) ·
2508.19029 *Revisiting associative recall in modern recurrent models* (Aug 2025) ·
2510.26692 *Kimi Linear* (Oct 2025) ·
2507.06457 *A Systematic Analysis of Hybrid Linear Attention* (2025) ·
2603.15569 *Mamba-3* (Mar 2026) ·
2604.03444 *Olmo Hybrid: From Theory to Practice and Back* (Apr 2026) ·
2605.22791 *Gated DeltaNet-2: Decoupling Erase and Write* (May 2026) ·
2606.15378 *Rethinking the Role of Efficient Attention in Hybrid Architectures* (Jun 2026) ·
2606.30562 *Morphing into Hybrid Attention Models* (Jun 2026) ·
2607.02303 *A Hippocampus for Linear Attention* (Jul 2026) ·
2607.07386 *Sparse Delta Memory* (Jul 2026)

**Long context and position**
2307.03172 *Lost in the Middle* (2023) ·
2404.06654 *RULER* (2024) ·
2508.07479 *Positional Biases Shift as Inputs Approach Context Window Limits* (Aug 2025) ·
2601.02872 *LongBench Pro* (Jan 2026) ·
2605.28079 *ATLAS: All-round Testing of Long-context Abilities across Scales* (May 2026)

**Agent memory, compaction, security**
2310.08560 *MemGPT* (2023) ·
2605.17830 *Remembering More, Risking More* (May 2026) ·
2605.23296 *Parallel Context Compaction for Long-Horizon LLM Agent Serving* (May 2026) ·
2605.08580 *Slipstream: Trajectory-Grounded Compaction Validation* (May 2026) ·
2606.04329 *From Untrusted Input to Trusted Memory* (Jun 2026) ·
2606.11213 *Beyond Compaction: Structured Context Eviction for Long-Horizon Agents* (2026) ·
2604.16548 *A Survey on Long-Term Memory Security in LLM Agents* (Apr 2026) ·
2606.15903 *Control-Plane Placement Shapes Forgetting* (Jun 2026) ·
2606.22528 *Governance Decay: How Context Compaction Silently Erases Safety Constraints* (Jun 2026) ·
2607.08032 *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction* (Jul 2026)

**Methodology**
2203.03466 *Tensor Programs V* / muP (2022) ·
2203.15556 *Training Compute-Optimal Large Language Models* (2022) ·
2512.24503 *Can Small Training Runs Reliably Guide Data Curation?* (Dec 2025) ·
2603.22339 *Problems with Chinchilla Approach 2* (Mar 2026)

**Local measurements and code pointers**
`ASSUMPTIONS.md` rows `gpu-fast-tier-size`, `large-tensor-fault-32gib`,
`hardware-capacity-ceiling`, `gemm-throughput-below-reference`, `single-device-only`,
`bf16-numerics-unproven`, `ablation-scale-sufficient`, `kv-per-token-laguna`,
`reference-model` (all `[M]`/`[C]`, 2026-07-24 to 2026-07-26) ·
`notebook/uma-carveout-controls-fast-tier.md` (`[M]`, 2026-07-26, single run per arm) ·
`research/reference/CODE_MAP.md`: vLLM `block_pool.py:647`, `:719`, `kv_cache_utils.py:184`;
SGLang `radix_cache.py:565`, `evict_policy.py:16`; FlashInfer `decode.py:1239`,
`page.py:403`; Mooncake `master_service.cpp:6382`; flash-linear-attention `naive.py:54`,
`fused_recurrent.py:138-150`; llama.cpp-laguna `llama-kv-cache-iswa.cpp:73`;
OLMo-core `trainer.py:1037`, `:1394`.
