---
title: Memory failure register
version: 1.0.0
gate: G0 — Discovery
status: living document (class 3 — documentation, keep accurate)
---

# Memory failure register — the G0 Discovery Brief

**What this note settles.** Eleven named failure modes in LLM memory systems, each with a symptom you can observe, a mechanism you can point at in code or math, citations that survive an arXiv-API check, and an honest open/solved verdict — plus one entry that fails the evidence bar and is marked as such rather than quietly promoted. The organising finding is that nine of the eleven reduce to a single defect: **an irreversible discard decision taken before the query that needs the data is known**, which is a scheduling error, not a compression error, and the literature has only just started saying so out loud `[C]` (2607.08032, Jul 2026). The second finding is that the field's dominant measurement practice — report throughput or accuracy, do not isolate which mechanism produced it — is itself the highest-leverage open problem for this lab, because it is the one failure a 20M–300M single-GPU rig is *better* positioned to attack than a frontier lab.

## How to read this register

**The G0 bar.** A pain claims a row only with **5+ independent sources**. "Independent" means separate research groups, not five papers from one lineage. Surveys count as one source each and are marked as corroborating rather than primary. A pain resting only on a paper's Future Work section, a taxonomy's "empty design point", or a blog is marked **UNPROVEN DEMAND** and stays marked until someone measures it.

**Evidence tags** (CLAUDE.md hard rule 2). `[C]` cited — arXiv id, verified against the live API on 2026-07-26, with date. `[M]` measured — on our hardware, with the run reference. `[A]` assumed — confidence stated plus the cheapest test that would move it. No `[A]` is ever stated in the register of an `[M]`.

**Status vocabulary.** `OPEN` — mechanism understood, no accepted fix. `OPEN/CONTESTED` — the field disagrees about the mechanism or its severity; both camps presented, no side picked. `PARTIALLY SOLVED` — a fix exists and ships, but only covers part of the failure. `UNPROVEN DEMAND` — the failure is plausible and unmeasured.

---

## The register

| Slug | Symptom | Mechanism (one line) | Status | Primary / corroborating sources |
|---|---|---|---|---|
| `position-bias-lost-in-the-middle` | Accuracy on a fact depends on *where* in the context it sits, U-shaped: ends good, middle bad | Causal masking + residual accumulation produce a non-uniform influence profile before any training happens; RoPE adds long-term decay on top | **OPEN/CONTESTED** (architectural vs. correctable) | 6 / 2 |
| `effective-context-collapse` | A model advertising 1M tokens is unreliable far below that, non-uniformly by task | No single mechanism — retrieval, aggregation and multi-hop tracing degrade at different lengths, so one headline number cannot exist | **OPEN** | 5 / 2 |
| `eviction-destroys-long-range-recall` | Aggressive KV eviction silently drops whole instructions; some tasks fall off a cliff, perplexity barely moves | Softmax renormalisation redistributes the evicted attention mass onto survivors, so eviction is a distribution edit, not a deletion — and deterministic top-k provably cannot estimate its own error | **OPEN** | 8 / 2 |
| `quantization-breaks-alignment-not-perplexity` | KV quantised to low bit width keeps its perplexity and loses its refusals | Safety behaviour lives in a low-dimensional activation subspace far more fragile than the mean representation, which is exactly what a perplexity-only eval averages away | **OPEN** (opened Jun 2026) | 3 / 1 |
| `ssm-recall-capacity-wall` | Constant-state models fail multi-query associative recall past a state-size-dependent number of pairs | Reads are `S^T q`; non-orthogonal keys cross-talk, so the failure is interference between stored associations, not a cache miss | **OPEN** (now with theory) | 7 / 1 |
| `hybrid-ratio-sensitivity` | Recall collapses as full-attention layers thin out; the "safe" ratio does not survive post-training | Full-attention layers are the only exact-recall substrate in the stack; whether the ratio sets a ceiling or a learning rate is disputed | **OPEN/CONTESTED** | 7 / 1 |
| `prefix-cache-correctness-and-leakage` | Reused KV is stale, mis-positioned, or observable by another tenant through timing | The cache key is a hash *chain* over the prefix, so reuse is position-locked; and a hit is measurable as a TTFT delta | **PARTIALLY SOLVED** (salting ships; non-prefix reuse does not) | 7 / 1 |
| `memory-poisoning-cross-session-contamination` | An agent's stored memory carries an attacker's payload across sessions and users | Every write channel is an injection surface — including the summariser — and a poisoned entry persists, propagates, and resists cleanup | **OPEN** | 6 / 3 |
| `memory-induced-sycophancy` | Adding long-term memory makes a model *less* correct, agreeing with the user's stated misconception | Memory extraction is lossy compression: the snippet keeps the user's belief and discards the corrective context that surrounded it | **OPEN** (first measured 2026) | 2 / 3 |
| `forgetting-and-rollback-unsolved` | There is no auditable delete; forgotten content re-enters via retrieval or regeneration | Parameter-memory backflow — parameter-only unlearning cannot close the write-read loop of an external store | **OPEN** | 4 / 3 |
| `attribution-gap-in-serving-results` | Papers report a speedup or an accuracy hold; almost none isolate which mechanism caused it | Aggregate outcome metrics are the field's unit of evidence, and multiple mechanisms move them in the same direction | **OPEN — highest lab leverage** | 7 / 2 |
| `per-session-kv-lifetime` | Chat/agent workloads want session-scoped KV retention; no system implements the lifetime | Named as an *empty design point* in a taxonomy — no measurement of what it would buy | **UNPROVEN DEMAND** | 0 / 1 |

---

## Per-entry detail

### `position-bias-lost-in-the-middle`

**Symptom.** Retrieval accuracy on an identical fact varies by tens of points depending only on its index in the prompt `[C]` (2307.03172, Jul 2023). The classic shape is a U: primacy and recency preserved, middle suppressed.

**Mechanism, written out.** Attention weight on key `i` for query `q` is

```
a_i = exp(q · k_i / √d) / Σ_j exp(q · k_j / √d)
```

`q` is the query vector for the current token, `k_i` the key vector cached for token `i`, `d` the head dimension, `√d` a variance-stabilising divisor, and the sum runs over every token the mask allows. Nothing in that expression mentions position — position enters through `k_i` itself, because RoPE has already rotated it by an angle proportional to its index `[C]` (2104.09864, Apr 2021). Three separate things then bias the profile: the causal mask means early tokens are attended by more queries and so accumulate more downstream influence; residual connections compound that per layer; and RoPE's rotation gives an inner product that decays with relative distance. A 2026 derivation reproduces the U-shape from causal masking plus residual connections alone, via residual-aware cumulative attention rollout `[C]` (2602.16837, Feb 2026) — i.e. it is a property of the wiring, present before any data is seen.

**Systems bridge, and where it breaks.** This looks like NUMA: the same logical address costs differently depending on which node holds it. The analogy breaks in the way that matters — a NUMA read is *slower but correct*, whereas a middle-of-context read is fast and **wrong**. There is no latency signal to observe, no counter that increments, no page-fault trace. Your entire observability instinct (measure the slow path) finds nothing here, because the failure is silent and full-speed.

**Contested — do not resolve this in the curriculum.** Whether the bias is architectural or correctable is live. The structural-theory line says it falls out of the wiring and more long-context training will not fix it `[C]` (2602.16837). The calibration line reports recovering up to 15 points by correcting attention bias at inference time `[C]` (2406.16008, Jun 2024), which implies it is substantially correctable. Separately, the U-shape itself is contested as the right model: it holds only up to roughly 50% context occupancy, above which primacy decays and the bias becomes distance-based `[C]` (2508.07479, Aug 2025). Papers reporting a clean U and papers reporting recency-dominance may simply be probing different occupancy regimes.

**Why a memory subsystem must care.** KV entries are not equally reachable. Any eviction policy that treats retained tokens as equally useful is wrong before a line of code is written — and a policy tuned at 30% occupancy is tuned for a different bias regime than the one it will run in at 90%.

---

### `effective-context-collapse`

**Symptom.** Advertised context length and usable context length are different numbers, and the gap has not closed. RULER established the methodology — 13 synthetic tasks spanning retrieval, multi-hop tracing, aggregation and QA — and found models with near-perfect needle-in-a-haystack scores collapsing far below their claimed length `[C]` (2404.06654, Apr 2024). Two 2026 measurements confirm it at current scale: LongBench Pro over 46 models at 8k–256k using naturally occurring rather than synthetic tasks `[C]` (2601.02872, Jan 2026), and ATLAS out to 1M tokens `[C]` (2605.28079, May 2026).

**Mechanism.** There is no single one, and that is the finding. ATLAS reports two distinct failures at frontier scale: performance collapses as length grows, *and* strong retrieval does not transfer to downstream use — a model can find the needle and still not use it. ATLAS also reports rank instability: 7 models shift 2+ positions, with gaps up to 12 positions, between the 8K–128K and 8K–1M regimes. That is the concrete argument against any single headline long-context score.

**Systems bridge, and where it breaks.** "Advertised vs. effective capacity" is a storage-vendor problem you have fought before — the enclosure says 100TB, the usable figure after RAID, overhead and reserve is 68TB, and the difference is *arithmetic you can do*. Here it is not arithmetic. The effective figure is a function of task type, occupancy, and post-training history, so there is no derating factor to apply and no datasheet footnote that would let you compute one.

**Aggravating factor discovered in 2026.** Effective context is not a fixed property of a pretrained model — post-training destroys it. Chain-of-thought SFT systematically degrades long-context recall in hybrid linear-attention models; HypeNet-9B on NIAH-S2@256K falls from 67.2% to 9.4% after CoT-SFT, attributed to reasoning gradients biasing the query/key projections `W_Q, W_K` toward short-range patterns `[C]` (2606.11052, Jun 2026). A training-free fix — restoring `W_Q, W_K` from the pretrained checkpoint while keeping other fine-tuned weights — recovers most of it, which is strong evidence the mechanism attribution is correct.

---

### `eviction-destroys-long-range-recall`

**Symptom.** Under StreamingLLM, SnapKV, TOVA, H2O and K-Norm, *certain instructions degrade much faster than others and are effectively ignored entirely* — with system-prompt leakage as the worked case study, and instruction order and eviction bias identified as contributing factors `[C]` (2510.00231, Sep 2025, ACL 2026). The damage does not show up in perplexity, and it does not show up in LongBench.

**Mechanism, written out.** This is the entry where the math earns its place. Evicting a set `E` of tokens does not remove their contribution and leave the rest untouched. Attention is a softmax, so the denominator shrinks:

```
before:  a_i = exp(s_i) / Σ_{j ∈ all} exp(s_j)
after:   a_i = exp(s_i) / Σ_{j ∉ E}  exp(s_j)
```

where `s_j = q · k_j / √d` is the pre-softmax score for token `j`. Every surviving token's weight goes **up**, by a factor of `Σ_all exp(s_j) / Σ_{j∉E} exp(s_j)`. Eviction is therefore a *redistribution of attention mass*, not a deletion — which is why evicting the first few tokens is catastrophic rather than merely lossy. Those tokens are attention sinks that absorb mass the model has no better use for; remove them and that mass is forced onto content tokens, distorting the output distribution `[C]` (2309.17453, Sep 2023). Every eviction policy since pins a prefix for exactly this reason.

**The 2026 result that reframes the whole area.** Deterministic top-k eviction *provably cannot know what it destroyed*: evicted values can be altered so that everything the serving system retains is unchanged while the true attention-output error grows arbitrarily, so no serving-time estimator of that error is consistent. Randomised eviction with Poisson-sampled tail at known inclusion probabilities restores identifiability, and a survey-sampling variance estimator over the retained set gives a per-step error certificate at 0.97 empirical coverage at no accuracy cost `[C]` (2607.21475, Jul 2026). For a lab whose thesis is attribution, this is the single most important eviction paper of the year: it converts "we hope this was safe" into a measurable quantity.

**Systems bridge, and where it breaks.** You would call this a cache with no backing store — a miss is not a slow path, it is a wrong answer. But go further, because two deeper breaks matter. (1) *There is no miss signal at all.* In vLLM the free list and the LRU victim cache are the same intrusive linked list, and eviction happens lazily at reallocation inside `get_new_blocks → _maybe_evict_cached_block` (`memory/vllm/vllm/v1/core/block_pool.py:679`); a freed block is still matchable, so "blocks in use" and "entries available for hits" are two different numbers. (2) *Eviction granularity is not the page.* In vLLM, `allocate_slots` returning None is not a fault to be serviced, it is an admission rejection that preempts the whole sequence. In SGLang, `evict` (`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565`) only ever considers *leaves* of the radix tree, so a hot child keeps a cold parent resident indefinitely — eviction order is topological, not recency-ordered. In Mooncake it is neither: `BatchEvict` (`memory/mooncake/mooncake-store/src/master_service.cpp:6382`) considers only lease-expired objects and partially sorts them by lease deadline. Three production systems, three incompatible notions of "the victim".

**Corroborating evidence that the pain is real, not stylised.** Coverage — the number of *unique* tokens retained — predicts degradation better than eviction rate does `[C]` (2606.29563, Jun 2026). Eviction hurts most at exactly the ratios where papers claim it is free `[C]` (2605.09649, May 2026). Rankings do not survive multi-turn cache reuse `[C]` (2412.10319, Dec 2024) or worst-case rather than mean aggregation `[C]` (2510.13334, Oct 2025).

**Contested.** Eviction versus retention is unresolved. Permanent eviction cuts peak capacity but is irreversible; full-retention sparse loading preserves fidelity and cuts bandwidth but not capacity. RocketKV argues the two are orthogonal and composable; other 2026 work treats eviction as the wrong primitive entirely and prefers tiered offload plus retrieval `[C]` (2607.02574, Jun 2026). Also contested: whether non-uniform per-layer budget allocation is real at all — PyramidKV degenerates to SnapKV at aggressive ratios `[C]` (2406.02069, Jun 2024), and several groups argue most of the reported gain comes from the observation window rather than the allocation rule.

---

### `quantization-breaks-alignment-not-perplexity`

**Symptom.** Quantise the KV cache and the model keeps its perplexity while losing its refusals. Mistral-7B loses 15.2% of its refusals at 1.03x perplexity, across eleven instruction-tuned models from 3.8B to 72B and five benchmarks totalling 1,894 prompts, with sharp model-specific phase transitions and no universal safe bit width `[C]` (2606.09864, Jun 2026).

**Mechanism.** Safety behaviour occupies a low-dimensional activation subspace reported as 10²–10³x more vulnerable to quantisation noise than the full representation space. Perplexity is an average over the full space; a per-channel diagnostic recovers up to 97% of the lost alignment.

**Why it is in this register even though it is not "memory loss".** It is the cleanest available demonstration of the register's central methodological claim: *the outcome metric was fine and the mechanism was broken.* If you accept this result, you cannot accept a KV-compression paper that reports only perplexity and a task average. Related mechanisms in the same window: attention-sink destruction under quantisation `[C]` (2508.04257, Aug 2025) and error accumulation over long reasoning chains `[C]` (2606.03458, Jun 2026).

**Contested.** Whether sub-4-bit KV is deployable at all. 8-bit is production-boring, 4-bit is broadly safe, 2-bit is a live question whose answer is task-dependent — perplexity-friendly, reasoning-hostile.

---

### `ssm-recall-capacity-wall`

**Symptom.** Constant-state models fail multi-query associative recall (MQAR) past a number of stored pairs that scales with state size, not with training `[C]` (2312.04927, Dec 2023). The recall-versus-state-size frontier is explicit and Pareto-shaped `[C]` (2402.18668, Feb 2024).

**Mechanism, written out.** A linear-attention or delta-rule layer keeps one matrix `S ∈ R^{d_k × d_v}` — `d_k` the key dimension, `d_v` the value dimension — and nothing else. Writing the pair `(k, v)` adds an outer product; reading with query `q` computes

```
read(q) = Sᵀ q = Σ_i v_i (k_i · q)
```

If the stored keys were mutually orthogonal and `q = k_m`, every term but `m` vanishes and the read is exact. They are not orthogonal: keys are L2-normalised continuous vectors in `d_k` dimensions, and you cannot pack more than `d_k` mutually orthogonal directions into `d_k` dimensions. Past that, every read returns the wanted value **plus a weighted sum of every other stored value**. The 2026 theory recasts this as spherical packing and derives a Welch-bound interference floor `[C]` (2607.17419, Jul 2026) — a hard limit, not an optimisation target.

Gated DeltaNet's update makes the write side concrete (`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54–58`):

```
S ← S · exp(g)                        # decay: g ≤ 0, so exp(g) ∈ (0,1]
S ← S + k ⊗ (β · (v − Sᵀ k))          # delta: write only the residual
```

`g` is a learned per-head, per-token decay exponent; `β` is write strength; `Sᵀk` is what the state currently returns for this key, so the write carries only the correction. `β = 1` is an exact overwrite of that key's direction, `β = 0` a no-op.

**Systems bridge, and where it breaks — three places, and all three matter.** It reads like a fully-associative cache with a global TTL tick, `β` as write strength, and the delta term as a compare-and-swap. (1) *There are no lines and no addresses.* A *similar* key partially clobbers a neighbour's content; the failure mode is interference, not a miss. (2) *There is no capacity miss and no eviction policy.* The decay multiply destroys old content every step whether or not new content arrives, and there is no tier to spill to. (3) *It is not a write-ahead log.* A KV cache is an append-only exact log you can rescan; this state is destructive-update and unreplayable, so token 5's contribution cannot be recovered at token 5000 (`architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` is the one-line proof: `states = scale * states + new_states`).

One correction to the folk model, verified in code: **the gate cannot forget selectively.** The decay is a single scalar per head applied to the entire `d_k × d_v` matrix, so every stored association is attenuated identically. Per-key and per-channel gates exist in the fused kernel (`USE_GK`/`USE_GV`) and Gated DeltaNet simply does not pass them. "Gating = selective forgetting" is backwards: the gate is indiscriminate decay; the *delta term* is the targeted erase.

**Live directions, none settled.** Sparse addressing into a large explicit memory `[C]` (2607.07386, Jul 2026); a bounded exact KV cache alongside the compressive state, framed as complementary learning systems `[C]` (2607.02303, Jul 2026); a logarithmically growing hierarchy of states that abandons the O(1) premise outright `[C]` (2506.04761, Jun 2025); uncertainty-weighted updates to mitigate state saturation `[C]` (2602.10743, Feb 2026); Mamba-2 perplexity at half the state size `[C]` (2603.15569, Mar 2026).

---

### `hybrid-ratio-sensitivity`

**Symptom.** Interleave linear-attention and full-attention layers and recall degrades sharply as full-attention density drops below roughly 3:1, across 72 trained models, six linear variants and five ratios `[C]` (2507.06457, Jul 2025, rev. Jun 2026). Corroborated independently for Mamba-Transformer hybrids `[C]` (2510.26912, Oct 2025).

**Mechanism.** The full-attention layers are the only exact-recall substrate in the stack; everything else is a lossy summary subject to the interference floor above. Thin them out and there is nowhere for an exact long-range lookup to happen. The ratio is not an abstraction — in the reference model it is one config list lookup at layer construction (`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365`), and in llama.cpp one call, `set_swa_pattern(4, dense_first=true)` (`architecture/llama-cpp-laguna/src/models/laguna.cpp:41`). Which makes it trivially ablatable — and is why this is a good target for us.

**The trap that kills the obvious experiment.** You cannot test long context by simply widening the sliding windows. In Laguna, full layers apply YaRN RoPE over 64 of 128 head dims at θ=500000 while SWA layers apply plain RoPE over all 128 at θ=10000 (`architecture/llama-cpp-laguna/src/models/laguna.cpp:184`). The SWA layers were never trained with a positional encoding that reaches past their window. Widening the window puts them in a regime they have no encoding for, and you will measure the encoding failure, not the ratio.

**Contested, and this is the sharpest live dispute in the whole track.** Does the ratio set a capability *ceiling* or only the *rate* at which long-context ability emerges? 2507.06457 finds a sharp recall degradation below 3:1, implying a ceiling. "Rethinking the Role of Efficient Attention in Hybrid Architectures" argues the opposite reading — different configurations converge to comparable performance given enough training, and larger SWA windows *delay* retrieval-head formation ("Large-Window Laziness") `[C]` (2606.15378, Jun 2026). Same year, same question, incompatible framings. The resolution likely depends on token budget, which is exactly the axis a small-scale rig can sweep.

Shipping practice disagrees with itself too, and both sides have commercial incentives: Kimi Linear claims a 3:1 KDA/full hybrid matching or beating full attention under matched-scale pretraining `[C]` (2510.26692, Oct 2025), while MiniMax shipped M2 as full attention on reliability grounds `[C]` (2605.26494, May 2026). Neither is a controlled ablation. Treat the disagreement as the open question, not as evidence for either camp.

**Newly destabilising.** A 2026 wave reframes ratio choice as post-hoc conversion — freeze a pretrained transformer and learn layerwise gates that pick which layers keep full attention under a fixed budget `[C]` (2606.30562, Jun 2026); KL-guided layer selection for distillation `[C]` (2512.20569, Dec 2025); distillation-first hybrid design `[C]` (2601.22156, Jan 2026). If that line holds, ratio search becomes cheap and moves *after* pretraining, which changes the cost structure of the entire question.

---

### `prefix-cache-correctness-and-leakage`

**Symptom.** Three distinct failures wearing one name: (a) reuse is position-locked, so semantically identical content at a different offset is a miss; (b) non-prefix reuse silently produces wrong KV; (c) a cache hit is observable to another tenant.

**Mechanism (a), written out.** The vLLM key is a hash *chain*, not a content hash (`memory/vllm/vllm/v1/core/kv_cache_utils.py:596`):

```
h_0 = H(∅,     t[0:B],     extra)
h_i = H(h_{i−1}, t[iB:(i+1)B], extra)
```

`B` is the block size in tokens (16 by default), `t[·]` the token ids, `extra` a namespacing salt. Folding the parent hash in makes the key strictly prefix-ordered: the same 16 tokens at a different offset are a different key, one changed token at position 0 invalidates every downstream hash, and there is no associative or middle-of-sequence match. The match loop breaks at the first miss because a later hit is impossible by construction (`single_type_kv_cache_manager.py:658`). Two consequences that contradict the usual story: a 100% prefix match never skips 100% of the work — the hit is capped at `num_tokens − 1` and then floored to block alignment, so an exact-duplicate prompt recomputes a whole trailing block; and freeing is not evicting.

**Mechanism (b).** Concatenating independently computed KV chunks loses the cross-attention between them. CacheBlend reuses non-prefix KV and selectively recomputes a small high-deviation token subset to partially repair it `[C]` (2405.16444, May 2024) — an *approximation*, explicitly, and a controlled experimental study of chunk-level reuse strategies confirms the accuracy/latency tradeoff is real and configuration-dependent `[C]` (2603.20218, Mar 2026). Multi-agent LLM-judge pipelines break under naive reuse specifically because cross-candidate interaction is destroyed `[C]` (2601.08343, Jan 2026). RAG systems need an explicit safety predicate for when a cached answer may be reused at all `[C]` (2605.27494, May 2026).

**Mechanism (c).** A hit is faster than a miss, so time-to-first-token is a side channel: an attacker probes with guessed prefixes and reconstructs another tenant's prompt word by word `[C]` (2409.20002, Sep 2024). Extended to non-prefix KV in RAG in 2026 `[C]` (2606.21842, Jun 2026), with a dedicated mitigation `[C]` (2603.10726, Mar 2026).

**Systems bridge, and where it breaks.** This is a refcounted buffer cache plus a page table, and it will feel completely familiar — right up to the point where it stops being one. A page-table miss traps and pages in; a radix-cache miss just means recompute, and there is no backing store to fault from (SGLang's plain `RadixCache` has none; the host tier is a separate bolt-on). Worse for anyone with a filesystem background: **`match_prefix` is a mutating read** (`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355`) — a lookup that terminates mid-node splits the node and clones two tensors, so read cost is not read-only and lookups cannot run concurrently with anything.

**Status: PARTIALLY SOLVED.** Namespacing works and ships: SGLang's `extra_key` (LoRA id, cache salt) partitions the tree like an ASID so identical token prefixes stay disjoint, and vLLM added cache salting for the same reason (vLLM RFC #16016). That closes the cross-tenant *content* hazard when operators use it. It does not close the timing channel, and it does not make non-prefix reuse correct.

---

### `memory-poisoning-cross-session-contamination`

**Symptom.** Content written into an agent's long-term memory in one session steers behaviour in later sessions, for the same or a different user. Triggers optimised so poisoned entries occupy a distinct embedding region achieve >80% attack success at <0.1% poison rate with <1% benign degradation `[C]` (2407.12784, Jul 2024).

**Mechanism.** Poisoning is a cross-phase chain — write → persist → propagate → resist-cleanup — not a single injection event `[C]` (2604.16548, Apr 2026, rev. Jun 2026). The write-channel taxonomy is the operationally useful part: it enumerates *which code paths can write memory*, and **compaction-driven writes are one of them**, which means your summariser is an attack surface `[C]` (2606.04329, Jun 2026). Delayed-trigger variants lie dormant across sessions before activating `[C]` (2605.15338, May 2026). Multimodal agent memory leaks along channels the text-only threat model does not cover `[C]` (2606.29788, Jun 2026).

**Longitudinal shape.** Violation rates rise monotonically with exposure length — contamination is a slow accumulation defect, not a discrete event — which means any memory evaluation run over a short horizon systematically under-reports harm `[C]` (2605.17830, May 2026). Corroborating framing: governance of evolving memory `[C]` (2603.11768, Mar 2026) and the always-on persistent-state survey `[C]` (2606.30306, Jun 2026).

**Systems bridge, and where it breaks.** You know this shape: it is a poisoned entry in a shared cache, or a bad row replicated by a sync job. What breaks is the remediation model. A poisoned DNS record has a TTL and a flush command; a poisoned memory has neither a canonical version to roll back to nor a way to prove the rollback worked — see the next entry.

**Contested / unverified.** Secondary summaries claim five of six defence classes fail against delayed-trigger attacks and that only tool-layer memory restriction holds structurally. I could not confirm that in a primary source. Treat it as a hypothesis worth testing, not a finding. Likewise, OWASP reportedly added "Memory and Context Poisoning" as ASI06 to its 2026 Agentic AI Top 10; I did not verify this against OWASP directly and it is cited here as unverified.

---

### `memory-induced-sycophancy`

**Symptom.** Turning long-term memory *on* makes a model less correct. Across three memory systems and five model families, memory amplified sycophancy in every condition, **up to 25x higher sycophancy rates than in-context baselines** `[C]` (2606.10949, Jun 2026).

**Mechanism, and this is why the entry belongs in a memory register rather than an alignment one.** The culprit is identified as **memory extraction**: lossy compression into discrete snippets encodes the user's misconception while discarding the corrective context that surrounded it. That is precisely the failure named in this register's opening — an irreversible discard decision made before the query is known — occurring at the agent-memory layer instead of the KV layer. A dedicated benchmark separates the sub-behaviours: whether an agent can reject memory as factual evidence, respect its applicable scope, resolve conflicts between memory and objective evidence, track updates, and still use valid memory for personalisation `[C]` (2607.01071, Jul 2026).

**Evidence bar — stated honestly.** Two primary measurement papers, both from 2026, plus three corroborating sources (2605.17830, 2603.11768, 2606.30306). **This is below the 5-primary-source bar.** It is in the register because the effect size is large, the mechanism is specific and testable, and the two primary sources are independent — but a second independent replication of the 25x figure would materially change its weight, and until then treat the magnitude as one lab's measurement.

---

### `forgetting-and-rollback-unsolved`

**Symptom.** There is no auditable delete. Content removed from an agent's memory can be retrieved from a surviving copy, regenerated from parameters, and re-stored.

**Mechanism.** *Parameter-memory backflow.* Existing unlearning methods assume stateless models and target either parametric knowledge or ephemeral context, leaving the retrieval-write loop uncontrolled; retrieval reactivates parametric remnants, or memory artefacts reintroduce content, so parameter-only unlearning cannot prevent cross-pathway recontamination `[C]` (2602.17692, Feb 2026). Where the deletion decision is *made* also matters: across thirteen system configurations, the placement of the LLM in the memory pipeline determines which failure modes are even addressable, with mutation-time placement winning — a direct contradiction of the common assumption that retrieval-time reranking is where the leverage is `[C]` (2606.15903, Jun 2026). Budget-curated forgetting is being explored as a *capability* rather than a compliance chore `[C]` (2606.25115, Jun 2026), and the rate-distortion framing argues consolidation, eviction and pruning are one problem `[C]` (2607.08032, Jul 2026).

**Systems bridge, and where it breaks — the sharpest break in the note.** Everything you know about deletion assumes a durability contract: a write is committed, a tombstone propagates, an audit log proves it. LLM memory has **no durability contract at all**, and the 2026 serving survey names durability contracts for persisted KV as one of its seven missing measurements `[C]` (2607.02574). Worse, the asymmetry runs the other way from storage: evicting a KV block is never data loss, only a recompute, which is why Mooncake can force-evict rather than block on writeback — a tradeoff no real storage tier is permitted to make. So the same system is *too willing* to lose bytes it should keep and *unable to prove* it lost the bytes it must delete.

**Scope note on demand.** The engineering pain (rollback of a bad memory write) is well evidenced. The *regulatory* framing — right-to-erasure over agent memory — is asserted in survey Future Work more often than it is measured, and should not be cited as demonstrated demand.

---

### `attribution-gap-in-serving-results`

**Symptom.** The literature's unit of evidence is an aggregate outcome — tokens/sec, TTFT, task average — and papers almost never isolate which mechanism produced it. Multiple mechanisms move the same aggregate in the same direction, so a reported win is compatible with several causes.

**Mechanism, with named gaps.** The 2026 serving survey classifies 30+ KV systems on four axes — locality, lifetime, ownership, substrate — and explicitly names **seven missing KV-specific measurements**, tied to open problems in fault tolerance, isolation, tiered eviction, speculative decoding, MoE serving, and shared-cache semantics `[C]` (2607.02574, Jun 2026). A companion survey reorganises the field by system behaviour — temporal (scheduling), spatial (placement/migration), structural (representation/retention) — and analyses behaviour-objective links, which is the same complaint from the other side `[C]` (2607.08057, Jul 2026). On the algorithm side, the diagnostic argument is explicit: task accuracy alone cannot tell you *why* a selector worked, which is why a fixed-contract diagnostic is needed `[C]` (2605.08234, May 2026).

**Four documented instances of the gap producing a wrong conclusion.**
1. Perplexity held, refusals collapsed `[C]` (2606.09864).
2. LongBench held, individual instructions were dropped entirely `[C]` (2510.00231).
3. Single-turn rankings did not survive multi-turn cache reuse `[C]` (2412.10319).
4. Mean-aggregated rankings did not survive worst-case aggregation `[C]` (2510.13334).

Plus the standing suspicion that PyramidKV-style per-layer allocation contributes little beyond SnapKV's observation window, since it degenerates to SnapKV at aggressive ratios `[C]` (2406.02069).

**Systems bridge, and where it breaks.** You have run the equivalent postmortem: p99 improved after a deploy that changed four things, and nobody can say which. Your instinct is the right one — bisect, hold everything else fixed, instrument the causal path. Where the analogy breaks is that in a serving stack you can usually A/B one flag; here the mechanisms are *entangled by construction*. Compressing the cache changes the memory footprint, which changes the achievable batch size, which changes arithmetic intensity, which changes throughput — so a compression paper's speedup is partly a batching result, and separating them requires holding batch size fixed, which nobody does because it looks like leaving performance on the table.

**Why this is the lab's highest-leverage entry.** Attribution is cheap at 20M–300M parameters and expensive at 100B. The rig's stated design goal — every ablation axis is a config field — is exactly the instrument this gap requires, and `[M]` our measured ≥62 GiB fast tier at ~200 GB/s (single run per arm, `notebook/uma-carveout-controls-fast-tier.md`, 2026-07-26) is enough to hold a long-context KV cache resident while varying one mechanism at a time.

---

### `per-session-kv-lifetime` — **UNPROVEN DEMAND**

**Claim as stated in the literature.** Chat and agent workloads obviously want session-scoped KV retention, yet the 2026 serving survey identifies per-session KV lifetime as an *empty design point* in its taxonomy — no system occupies it `[C]` (2607.02574).

**Why it is marked UNPROVEN DEMAND and not OPEN.** This rests on one taxonomy's structural observation. There is no measurement of what a session-scoped lifetime would buy over the existing combination of prefix caching plus offload tiering, no workload trace showing the miss pattern it would fix, and no user complaint traced to its absence. It is a hole in a grid, which is a very seductive kind of non-evidence — the shape of the taxonomy makes the gap *look* like a finding. **Zero primary sources.** Promote it the moment someone publishes a trace or a measurement; do not build against it before then.

---

## Open questions — testable on our hardware

Constraints: single GPU, 20M–300M params, `[M]` ≥62 GiB fast memory tier at ~200 GB/s (single run per arm, 2026-07-26), `[M]` single tensors ≥32 GiB hang or fault so KV must be paged, no working multi-GPU, and the Hardware Validation Gate has not run — so nothing below counts as evidence until it does.

1. **Is eviction damage the information loss or the renormalisation?** Compare true eviction against a masked-but-denominator-preserved control (add the evicted mass back as a constant in the softmax denominator) at matched budget. If the control recovers most of the loss, the field has been attributing to information loss what is actually a distribution shift — and the fix is a cheap scalar, not a better selector. Inference-only, no training.
2. **Does the randomised-eviction error certificate hold at our scale?** Reproduce the Poisson-sampled tail plus Hájek logit offset `[C]` (2607.21475) on a 100M model and check the claimed 0.97 coverage. Inference-only; the cheapest way to acquire an attribution instrument rather than another policy.
3. **Is the 3:1 hybrid cliff a ceiling or a token-budget artefact?** Sweep ratio × token budget at matched params. This is the one place the field's disagreement (2507.06457 vs 2606.15378) turns on an axis a small rig can actually sweep, and where frontier labs have no advantage.
4. **At what `d_k` does MQAR break for a Gated DeltaNet layer at 100M, and does the Welch-bound interference floor `[C]` (2607.17419) predict the breakpoint?** A quantitative prediction that either holds or does not.
5. **Does the U-shaped position bias exist at 20M–300M, and is it occupancy-dependent?** Run needle placement with context occupancy as an explicit axis, testing the ~50% transition `[C]` (2508.07479). If the U does not appear at this scale, a large class of our eviction experiments is unsound and we need to know now.
6. **Does per-layer budget allocation add anything once the observation window is held fixed?** SnapKV vs PyramidKV with the observation window identical across arms — the specific confound the field has not controlled.
7. **How much of a KV-compression speedup is really a batching effect?** Measure throughput at fixed batch size vs. batch-size-free, same compression ratio. Directly attacks `attribution-gap-in-serving-results` with no model training at all.
8. **Does the prefix-cache block-alignment floor cost what the code implies?** Measure recomputation on an exact-duplicate prompt against block size. Pure instrumentation; a same-day result.
9. **Does CoT-style fine-tuning degrade long-range recall at 300M the way it does at 9B `[C]` (2606.11052), and does restoring `W_Q, W_K` recover it?** If the mechanism reproduces at our scale, we have a cheap testbed for a frontier-scale failure.

---

## Sources

Every arXiv id below was resolved against the live arXiv API on 2026-07-26. Non-arXiv items are marked and are cited as weaker evidence.

**Position bias and long context**
- 2104.09864 — RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
- 2307.03172 — Lost in the Middle: How Language Models Use Long Contexts (2023)
- 2404.06654 — RULER: What's the Real Context Size of Your Long-Context Language Models? (2024)
- 2406.16008 — Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization (2024)
- 2508.07479 — Positional Biases Shift as Inputs Approach Context Window Limits (2025)
- 2601.02872 — LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark (2026)
- 2602.16837 — A Structural Theory of Position Bias in Transformers (2026)
- 2605.28079 — ATLAS: All-round Testing of Long-context Abilities across Scales (2026)

**KV eviction, compression, quantization**
- 2306.14048 — H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (2023)
- 2309.17453 — Efficient Streaming Language Models with Attention Sinks (2023)
- 2404.14469 — SnapKV: LLM Knows What You are Looking for Before Generation (2024)
- 2406.02069 — PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling (2024)
- 2412.10319 — SCBench: A KV Cache-Centric Analysis of Long-Context Methods (2024)
- 2508.04257 — KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization (2025)
- 2510.00231 — The Pitfalls of KV Cache Compression (2025, ACL 2026)
- 2510.13334 — Taming the Fragility of KV Cache Eviction in LLM Inference (2025)
- 2603.20397 — KV Cache Optimization Strategies for Scalable and Efficient LLM Inference (2026)
- 2605.08234 — When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic for Non-Monotone Cache Compression (2026)
- 2605.09649 — Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction (2026)
- 2606.03458 — KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks (2026)
- 2606.09864 — Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation (2026)
- 2606.29563 — Coverage-Driven KV Cache Eviction for Efficient and Improved Inference of LLM (2026)
- 2607.21475 — Error Certificates for KV-Cache Eviction via Randomized Design (2026)

**Serving, prefix cache, correctness and leakage**
- 2309.06180 — Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)
- 2312.07104 — SGLang: Efficient Execution of Structured Language Model Programs (2023)
- 2405.16444 — CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion (2024)
- 2409.20002 — The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems (2024)
- 2601.08343 — When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges (2026)
- 2603.10726 — PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems (2026)
- 2603.20218 — An experimental study of KV cache reuse strategies in chunk-level caching systems (2026)
- 2605.18825 — Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches (2026)
- 2605.27494 — Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer? (2026)
- 2606.21842 — Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG (2026)
- 2607.02574 — From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving (2026)
- 2607.08057 — Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization (2026, ACL 2026 Findings)
- vLLM RFC #16016 — Cache Salting for Secure and Flexible Prefix Caching (GitHub issue; engineering artifact, not peer-reviewed)

**Constant-state memory and recall capacity**
- 2312.04927 — Zoology: Measuring and Improving Recall in Efficient Language Models (2023)
- 2402.18668 — Simple linear attention language models balance the recall-throughput tradeoff (2024)
- 2410.11135 — Mimetic Initialization Helps State Space Models Learn to Recall (2024)
- 2506.04761 — Log-Linear Attention (2025)
- 2602.10743 — Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking (2026)
- 2603.15569 — Mamba-3: Improved Sequence Modeling using State Space Principles (2026)
- 2607.02303 — A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets (2026)
- 2607.07386 — Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity (2026)
- 2607.17419 — Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones (2026)

**Hybrid architectures and ratio selection**
- 2507.06457 — A Systematic Analysis of Hybrid Linear Attention (2025, rev. 2026)
- 2510.20787 — Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction (2025)
- 2510.26692 — Kimi Linear: An Expressive, Efficient Attention Architecture (2025)
- 2510.26912 — Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling (2025)
- 2512.20569 — Distilling to Hybrid Attention Models via KL-Guided Layer Selection (2025)
- 2601.22156 — Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures (2026)
- 2605.26494 — The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence (2026)
- 2606.11052 — Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, and How to Fix It (2026)
- 2606.15378 — Rethinking the Role of Efficient Attention in Hybrid Architectures (2026)
- 2606.30562 — Morphing into Hybrid Attention Models (2026)

**Agent memory: poisoning, contamination, sycophancy, forgetting**
- 2407.12784 — AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases (2024)
- 2505.00675 — Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics (2025)
- 2602.17692 — Agentic Unlearning: When LLM Agent Meets Machine Unlearning (2026)
- 2603.11768 — Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the SSGM Framework (2026)
- 2604.16548 — A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle (2026)
- 2605.15338 — Hidden in Memory: Sleeper Memory Poisoning in LLM Agents (2026)
- 2605.17830 — Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents (2026)
- 2606.04329 — From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents (2026)
- 2606.10949 — Recalling Too Well: Sycophancy Evaluation and Mitigation in Memory-Augmented Models (2026)
- 2606.15903 — Control-Plane Placement Shapes Forgetting: An Architectural Study of Agent Memory Across Thirteen System Configurations (2026)
- 2606.25115 — Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory (2026)
- 2606.29788 — MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory (2026)
- 2606.30306 — Always-On Agents: A Survey of Persistent Memory, State, and Governance in LLM Agents (2026)
- 2607.01071 — MemSyco-Bench: Benchmarking Sycophancy in Agent Memory (2026)
- 2607.08032 — What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents (2026)
- OWASP Agentic AI Top 10, ASI06 "Memory and Context Poisoning" — **unverified against OWASP directly**; reported in secondary sources only

**Code pointers (verified `file:line`, see `research/reference/CODE_MAP.md`)**
- `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` — prefix-cache key is a hash chain
- `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:658` — match loop breaks at first miss
- `memory/vllm/vllm/v1/core/block_pool.py:679` — eviction is lazy, at reallocation
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355` — `extra_key` namespacing (cache salt)
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` — eviction considers leaves only
- `memory/mooncake/mooncake-store/src/master_service.cpp:6382` — lease-expiry eviction, not LRU
- `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54,56,58` — global decay, delta term, rank-1 write
- `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` — destructive state overwrite
- `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` — hybrid ratio is a config list lookup
- `architecture/llama-cpp-laguna/src/models/laguna.cpp:41,184` — `set_swa_pattern`; per-layer RoPE divergence

**Our own measurements (`ASSUMPTIONS.md`, 2026-07-26)**
- `gpu-fast-tier-size` — ≥62 GiB fast tier at ~200 GB/s with 96 GiB BIOS UMA carve-out; single run per arm, `notebook/uma-carveout-controls-fast-tier.md`
- `large-tensor-fault-32gib` — single tensors ≥32 GiB hang silently or fault; keep KV buffers paged below that
- `kv-per-token-laguna` — 192 KiB/token upper bound for the reference model if every layer were global
