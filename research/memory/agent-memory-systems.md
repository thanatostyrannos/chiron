---
title: Agent memory systems
version: 1.0.0
date: 2026-07-26
track: research/memory
---

# Agent memory systems

This note settles three things. **First**, the working/episodic/semantic/procedural
quartet is a *persistence-semantics* taxonomy wearing cognitive-science clothes, and it
is only load-bearing where the four boxes actually differ in retention, supersession
and revision policy — a distinction the 2026 literature is openly split on. **Second**,
the dominant category error in the field is treating working memory as a retrieval
problem when it is a **context-budget allocation problem under an unknown future
query**, and the 2026 compaction wave is the field discovering this the expensive way.
**Third**, the agent layer keeps re-deriving, in text, mechanisms the serving layer
already implements in the KV cache one level down — and it re-derives them *without*
the two properties that make the serving-layer versions work: a resurrection path for
freed-but-not-yet-evicted state, and an accounting of what a mid-prefix edit costs.

---

## 1. The four boxes, and what actually distinguishes them

The standard quartet, stated in the terms a storage engineer can act on:

| Box | What it holds | Retention policy | Read path | Write path |
|---|---|---|---|---|
| **Working** | tokens currently resident in the context window | hard byte-bounded, destroyed at session end | attention, O(1) addressable, zero lookup latency | append (cheap) or rewrite (see §3) |
| **Episodic** | time-indexed records of what happened | decay / archival | similarity or temporal index | append-only log |
| **Semantic** | facts abstracted from episodes | indefinite until superseded | index lookup | read-modify-write with conflict resolution |
| **Procedural** | reusable how-to, i.e. skills | evidence-gated revision | matched on situation | distilled from trajectories |

The framing worth stealing is `[C]` **The Missing Knowledge Layer in Cognitive
Architectures for AI Agents** (arXiv 2604.11364, Apr 2026), which argues that CoALA
`[C]` (arXiv 2309.02427) and JEPA both lack an explicit Knowledge layer with its own
persistence semantics, producing "a category error: systems apply cognitive decay to
factual claims, or treat facts and experiences with identical update mechanics." Its
four layers — Knowledge (indefinite supersession), Memory (Ebbinghaus decay), Wisdom
(evidence-gated revision), Intelligence (ephemeral inference) — are explicitly labelled
engineering constructs justified by persistence requirements, not by neuroscience.

That is the right test. **If your store has one retention policy, you have one memory
type no matter how many tables you have.** Most shipped agent-memory systems fail this
test: a single vector index with a timestamp column, called four things in the docs.

Procedural memory is the box with the best 2026 evidence and the sharpest caveat.
`[C]` The AFTER benchmark (arXiv 2606.23127, Jun 2026; 382 enterprise tasks, six roles,
22 skills) reports a single refinement round improving aggregate performance by 3.7–6.7
points, and skills evolved from *multi-model* execution traces reaching 73.1%
cross-model test accuracy, beating any single-model trace source. `[C]` But the skill
lifecycle study (arXiv 2605.23899, May 2026) finds model-generated skills "beneficial
on average but exhibit non-trivial negative transfer," and — the load-bearing
observation — a model can be a strong skill *extractor* and a weak skill *consumer*,
with utility uncorrelated with model scale or baseline task strength. `[C]` Skill-Pro
(arXiv 2602.01869, Feb 2026) adds score-based maintenance to keep the procedural store
compact without parameter updates. So procedural memory is real, it is measurable, and
it is not monotone: adding skills can make an agent worse.

**Contested, do not resolve it here.** `[C]` *Memory in the Age of AI Agents*
(arXiv 2512.13564, Dec 2025, rev. Jan 2026) states outright that "traditional taxonomies
such as long/short-term memory have proven insufficient" and replaces them with
forms (token-level / parametric / latent) × functions (factual / experiential /
working) × dynamics. `[C]` The rate–distortion view (arXiv 2607.08032, Jul 2026) and
`[C]` the Mar 2026 agent-memory survey (arXiv 2603.07670) both prefer functional and
mechanistic axes and treat the cognitive mapping as decorative. `[C]` Meanwhile
arXiv 2504.15965 and `[C]` arXiv 2605.06716 lean on the cognitive terms structurally.
Use the cognitive words as *labels for retention policies*, never as an argument.

---

## 2. The category error: working memory is a budget, not an index

State the naive framing precisely so its failure is visible. Let the window hold
**B** tokens. At turn *t* the harness holds candidate context objects indexed *i* —
system prompt, tool schemas, user turns, tool results, retrieved notes, plan state —
each of token length **s_i** with future utility **u_i**, and a keep/drop decision
**x_i ∈ {0,1}**:

> maximize Σ_i u_i·x_i   subject to   Σ_i s_i·x_i ≤ B

Symbols: *i* indexes a context object; *s_i* is its length in tokens; *u_i* is how much
having it resident will improve the next decision; *x_i* = 1 means keep it in the
prompt; *B* is the token budget you have actually chosen to spend, which is **not** the
advertised context length. That is a 0/1 knapsack. Retrieval — "find the most similar
memory" — is a *scoring function for u_i*. It is one term in the objective. It is not
the problem.

Three things break the static formulation, and all three are where real systems fail:

1. **u_i is unknown at decision time.** You discard before the query arrives. `[C]`
   arXiv 2607.08032 (Jul 2026) names this as the recurring failure mode shared by KV
   eviction, prompt pruning, recurrent-state bounding and agent consolidation:
   attention and recency signals cause irreversible discard before the query is known.
2. **The cost is delayed and the problem is sequential.** `[C]` OSL-MR
   (arXiv 2606.10616, Jun 2026) formulates retention as constrained stochastic
   optimization with budget feasibility, evidence utility, and *delayed* miss,
   reacquisition, and stale penalties — and proves the multi-step problem NP-hard. Its
   own ablation shows single-step optimization cannot anticipate future demand shifts,
   which is the formal statement of "greedy eviction is wrong."
3. **u is not additive.** Attention is not a sum over independent slots. Utility of a
   block depends on what else is resident, and total occupancy degrades quality by
   itself: `[C]` arXiv 2508.07479 (Aug 2025) shows the lost-in-the-middle U-shape holds
   only up to roughly 50% context occupancy, after which primacy decays and the bias
   becomes distance-based. **The position prior your eviction policy exploits changes
   as the window fills.** A fixed prior is wrong in one of the two regimes.

`[C]` ContextBudget/BACM (arXiv 2604.01664, Apr 2026) is the cleanest direct statement:
context management as a sequential decision problem with an explicit budget constraint,
where the agent assesses remaining budget *before* ingesting an observation and decides
when and how much history to compress. Its RL variant reports over 1.6× gains over
strong baselines at high task complexity, with the advantage *growing as the budget
shrinks* — the signature of a genuine allocation effect rather than a retrieval one.

**Bridge, and where it breaks.** This is admission control plus Denning working-set
theory, and `[C]` *The Missing Memory Hierarchy* (arXiv 2603.09023, Mar 2026) says so
in as many words: "The context window of a large language model is not memory. It is L1
cache." Its production numbers — 857 sessions, 4.45M effective input tokens, 21.8%
structural waste; 1.4M simulated evictions at a 0.0254% fault rate; live deployment over
681 turns cutting context consumption up to 93% (5,038 KB → 339 KB) — are the most
systems-legible measurement in the track. It also reports the expected thrashing
pathology under sustained pressure, which is the detail that makes it credible. Caveat
`[A]`, high confidence: this is a single-system production report, not a controlled
comparison, and "up to 93%" is a best case, not a mean.

The analogy breaks at the fault handler. In demand paging the fault is *transparent* to
the faulting process and serviced by a different, deterministic entity. Here the
"process" and the "pager" are the same stochastic model, the fault must be *chosen* via
a tool call, and each one costs a full turn. There is no MMU, no present bit, and no
dirty bit. `[C]` VISTA (arXiv 2606.30005, Jun 2026) makes the sharpest version of this
point: frontier models are "proprioceptively blind to their own context" — from the
prompt alone they cannot see how large, how old, or how used each block is. Its
training-free interface exposes typed addressable blocks plus a runtime dashboard of
per-block token usage, recency and access history, and lifts Gemini-3-Flash from 22.7%
to 50.7% on LOCA-Bench, with ablations showing the dashboard matters beyond the archive
and recovery tools. If exposing *counters* recovers that much, working memory was never
a retrieval problem — it was an accounting problem with no counters.

---

## 3. The break nobody in the agent literature models: mid-prefix edits are not cheap

This is the most important paragraph in the note, and it lives in the serving layer.

vLLM keys each KV block by a **chain** hash, not a content hash: `hash_block_tokens`
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:596`) computes
`h_j = H(h_{j-1}, tokens[j·P : (j+1)·P], extra_keys)`, where *P* is the block size in
tokens and *h_{j-1}* is the parent block's hash. `find_longest_cache_hit`
(`single_type_kv_cache_manager.py:658`) therefore breaks at the first miss, because a
later hit is impossible by construction. Consequence: **editing any token at position
*p* invalidates every block from ⌊p/P⌋ onward**, so the prefill you pay for a compaction
is not proportional to what you changed, it is proportional to everything after it:

> recompute_tokens = n − ⌊p/P⌋·P

where *n* is total context length. Rewriting the oldest 60% of a 200k-token transcript
does not cost "the summary" — it costs a full prefill of the ~80k surviving tokens plus
the summary. This is a log-structured store with a hash chain: append is cheap,
mid-stream overwrite costs the tail. `[C]` TokenPilot (arXiv 2606.17016, Jun 2026) is
built entirely around this trade-off — "unconstrained sequence mutations alter layouts,
introducing prefix mismatches and cache invalidation" — and reports 56–87% cost
reduction by making prefix stability a first-class constraint. `[C]` Self-GC
(arXiv 2607.00692, Jul 2026) lists "cache-aware commit" as an explicit mechanism
alongside recoverable sidecars and safe commit boundaries. Almost every other compaction
paper prices summarization in *summarizer* tokens and ignores the invalidated prefill.

Four further breaks, each a real transplant opportunity:

- **Free ≠ evict downstairs; free = destroy upstairs.** vLLM's `free_blocks`
  (`block_pool.py:719`) only decrements a refcount and pushes the block back on the LRU
  with contents *and hash* intact, so `touch` (`block_pool.py:702`) can resurrect it on
  a later prefix hit; actual eviction happens lazily in `_maybe_evict_cached_block`
  (`block_pool.py:679`) at reallocation time. Agent-level compaction has no equivalent —
  once the transcript is rewritten the tokens are gone. Self-GC's recoverable sidecars
  and VISTA's full-fidelity archive are both attempts to reintroduce, in text, the
  resurrection path the KV layer already has.
- **Eviction is topologically constrained, not recency-ordered.** SGLang's `evict`
  (`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565`) only ever considers
  *leaves* from an incrementally maintained `evictable_leaves` set, peeling the frontier
  inward, and `inc_lock_ref` walks to the root so one in-flight request pins an
  arbitrarily deep prefix chain. The agent analogue is exact: an active sub-task pins the
  whole plan chain above it. `[C]` CWL (arXiv 2606.11213) re-derives this at the text
  layer as typed, *dependency-linked* episodes with a deterministic LLM-free eviction
  policy over the dependency graph.
- **The hot signal should be a lease with admission control, not a touch.** Mooncake's
  `BatchEvict` (`memory/mooncake/mooncake-store/src/master_service.cpp:6382`) only
  considers objects whose *lease has expired*, and `TryPushPromotionQueue`
  (`master_service.cpp:5211`) gates disk→DRAM promotion behind a count-min-sketch
  TinyLFU threshold so a single cold hit cannot pollute the fast tier. Agent memory
  systems overwhelmingly use recency plus embedding similarity with **no admission
  control at all**. That is a straight transplant and nobody has run it.
- **The one real free lunch: clean pages.** Evicting from a KV cache is
  lossless-with-recompute. Evicting from agent memory is lossy and unrecomputable —
  *unless the effect is already persisted outside the context*. `[C]` CWL's criterion is
  precisely this: aggressively shed action episodes "whose effects are already persisted
  in the environment," preserving user turns and the exploratory context under active
  reasoning. That is the clean/dirty page distinction, and it is the only principled
  zero-cost eviction in the space. CWL reports a single session completing 89 sequential
  tasks across 80M tokens with no measurable accuracy degradation versus per-task
  isolated sessions.

---

## 4. MemGPT → Letta, A-MEM, and the memory-OS designs

**MemGPT** `[C]` (arXiv 2310.08560, Oct 2023) is the origin of the vocabulary: main
context vs. external context, self-editing memory via tool calls, page-fault-style
interrupts on overflow. It earned the analogy and then the analogy overran its evidence.
What it got right: a two-level store with a trap on overflow, and an explicit mediator.
What it cannot deliver: transparency (the model must choose to fault), a dirty bit
(an LLM memory edit is a lossy re-derivation with no checksum), and demotion (evicted
tokens are not moved to a slower tier, they are deleted and must be *recomputed*).

**Letta** continues the line with two additions worth taking seriously. *Memory blocks*
`[C]` (letta.com/blog/memory-blocks, accessed 2026-07-26) make in-context memory typed,
labelled and individually rewritable — which is exactly VISTA's "typed, addressable
blocks" arrived at from the product side rather than the benchmark side. *Sleep-time
compute* `[C]` (arXiv 2504.13171, Apr 2025; letta.com/blog/sleep-time-compute) moves
memory reorganization off the critical path: a separate agent holds the memory-editing
tools and rewrites the primary agent's blocks during idle time. The paper reports ~5×
less test-time compute for equal accuracy on Stateful GSM-Symbolic and Stateful AIME,
up to 13% and 18% accuracy gains from *scaling* sleep-time compute, and 2.5× lower
average cost per query when amortized across related queries — with efficacy correlated
to how predictable the user's query is. This is background compaction: LSM merge,
vacuum, defrag. It is the analogy that holds best. **Where it breaks:** an LSM merge is
*required* to preserve exact key-value contents and can be verified; a sleep-time memory
rewrite is required to preserve nothing, and there is no way to detect that it did not.

**A-MEM** `[C]` (arXiv 2502.12110, Feb 2025) is the structural alternative: Zettelkasten
notes with generated keywords and tags, dynamic linking, and *retroactive refinement of
older memories at write time*. MemGPT is a manually tiered store; A-MEM is a
self-organizing secondary index rebuilt incrementally on write. `[C]` FluxMem
(arXiv 2605.28773, May 2026) pushes the same idea further — memory as a heterogeneous
graph with link repair, interference pruning, granularity alignment, and distillation of
recurrent successful trajectories into reusable procedural circuits. The cost is write
amplification and a much worse blast radius: a poisoned write can rewrite *existing*
memories, which is exactly the write → persist → propagate → resist-cleanup chain in
`[C]` the memory-security survey (arXiv 2604.16548, Apr 2026, rev. Jun 2026), and `[C]`
the write-channel study (arXiv 2606.04329, Jun 2026) names compaction-driven writes as
a first-class attack surface. **Your summarizer is an ingest path.**

**Memory-OS designs.** `[C]` MemOS (arXiv 2507.03724, Jul 2025; short version
arXiv 2505.22101) gives the parametric / activation / plaintext trichotomy and MemCube
units carrying provenance and versioning, with a scheduler migrating content between
tiers. It is the closest thing in the literature to a storage-hierarchy mental model —
and the place the mental model is most misleading. **A storage hierarchy's defining
property is that a datum can live at any level and the level is a performance
decision.** Here the level is a *type* decision: you cannot promote plaintext to
parametric without a training step, you cannot demote parametric to plaintext at all,
and the conversions are lossy and mostly one-way. `[C]` Both arXiv 2505.00675 and
arXiv 2507.03724 include parametric memory in the taxonomy and admit the lifecycle
mismatch. Treat MemOS as a *catalog with a migration policy*, not as a cache hierarchy.

---

## 5. The compaction wave (Feb–Jul 2026), sorted by what it assumes

| Family | Representative | Core assumption | Reported |
|---|---|---|---|
| Summarize | Parallel Context Compaction `[C]` 2605.23296 | lossy summary is acceptable; the problem is that it *blocks* inference for tens of seconds | overlap compaction with inference; control how much you compact |
| Summarize, optimized | ACON `[C]` 2510.00615 | the compression *guideline* can be optimized in natural-language space from failure analysis | 26–54% peak token reduction; up to 46% gain for small LMs by cutting distraction |
| Structured eviction | CWL `[C]` 2606.11213 | typed dependency-linked episodes make a deterministic, LLM-free policy sufficient | 89 tasks / 80M tokens, one session, no measurable degradation |
| Lifecycle governance | Self-GC `[C]` 2607.00692 | context is *objects* with lifetimes; fold/mask/prune plus recoverable sidecars | 43.95% prefix tokens pruned at 84.85% no-impact vs 54.55–69.70% for heuristics; production input tokens −10–15% |
| Explicit budget | BACM `[C]` 2604.01664, OSL-MR `[C]` 2606.10616 | the constraint is the primitive; learn/solve under it | >1.6× at high complexity, advantage grows as budget shrinks |
| Train the model | CompactionRL `[C]` 2607.05378 | compaction should be inside the RL objective, not outside it | GLM-4.5-Air to 66.8% SWE-bench Verified (+7.0), 24.5% Terminal-Bench 2.0 (+3.1) |
| Expose state | VISTA `[C]` 2606.30005 | the policy is already latent; the missing piece is an interface | Gemini-3-Flash 22.7 → 50.7 on LOCA-Bench, training-free |
| Change substrate | AgentOCR `[C]` 2601.04786 | visual tokens are denser than text tokens for history | >95% of text-agent performance at >50% fewer tokens |
| Self-regulation | Focus `[C]` 2601.07190 | a capable model will self-compact if given the tools | 22.7% token reduction at equal accuracy — but N=5 SWE-bench Lite instances; an anecdote |

**Contested, and this is the live question of the track:** where the leverage is.
2604.01664 and 2607.05378 argue you need a *learned policy*. 2606.30005 argues you need
no policy at all, only an *interface* that exposes occupancy. 2606.11213 argues you need
neither — typed structure plus a deterministic rule beats both. These three cannot all
be right, they were published within eight weeks of each other, and none of them
compares against the others. Do not let a curriculum section pick one.

---

## 6. Evaluation is the weakest link, and the gap is shaped like an opening

Five independent 2026 sources converge on the same structural complaint, which is
unusual enough to treat as established:

- `[C]` **Anatomy of Agentic Memory** (arXiv 2602.19320, Feb 2026): benchmarks are
  underscaled, metrics are misaligned with semantic utility, results are
  backbone-dependent, and the latency/throughput cost of memory *maintenance* is
  routinely omitted.
- `[C]` **MemFail** (arXiv 2605.26667, May 2026): formalizes a memory system as the
  composition of summarization, storage and retrieval, then adversarially isolates each
  — because aggregate QA accuracy makes attribution impossible.
- `[C]` **MemTrace** (arXiv 2605.28732, May 2026): turns memory pipelines into
  executable memory-evolution graphs and attributes failures to operation subgraphs;
  finds failures are systematic (information loss, retrieval misalignment) and closes
  the loop for up to 7.62% end-task gain.
- `[C]` **PrecisionMemBench** (arXiv 2605.11325, May 2026): benchmarks score answers,
  not retrieval, so a system that dumps its whole store gets perfect recall while hiding
  precision failure; reports baseline precision clustering at 0.22 and below. Caveat
  `[A]`, high confidence: the paper also ships a competing system that scores perfectly,
  so read the headline as vendor-adjacent and the *methodological* point as sound.
- `[C]` **Control-plane placement** (arXiv 2606.15903, Jun 2026): "production failures
  are predominantly forgetting failures rather than recall failures, yet existing
  benchmarks measure only recall." Across 13 configurations on a 385-case adversarial
  surface: deterministic primitives get 5% on identifier obfuscation and 0%
  cross-lingual; an inscribe-time LLM recovers canonicalization to 100% but gets 0% on
  intent-aware deletion; a mutation-time hook recovers intent-aware deletion (78–85%)
  and lifts overall to 91.7–93.2%. **Where the model sits in the pipeline determines
  which failures are even addressable.**
- `[C]` **Always-On Agents** (arXiv 2606.30306, Jun 2026): across a 435-work coded
  corpus, the literature "concentrates more heavily on accumulating and retrieving state
  than on governing, recovering, or relinquishing it."

For anyone who has run a storage system, a field that benchmarks reads and neglects
deletes is not a mature field. That is the opening.

**Also contested: memory system vs. plain long context.** `[C]` arXiv 2603.04814
(Mar 2026) finds long-context GPT-5-mini achieves *higher* factual recall than a
Mem0-based fact store on LongMemEval and LoCoMo, with the memory system competitive only
on PersonaMemv2 — while giving the memory system a structurally better cost curve
(fixed per-turn read cost after a one-time write, crossing over at roughly ten turns at
100k context). `[C]` Mem0 itself (arXiv 2504.19413) reports the opposite ordering on
LoCoMo. `[C]` And arXiv 2604.11628 (Apr 2026) argues the bottleneck is not architecture
at all but "signal sparsity," with a minimalist retrieve-and-generate baseline beating
hierarchical-summarization systems. Three answers, all 2025–26. Unresolved.

**And the deepest dispute: is any of this memory?** `[C]` arXiv 2604.27707 (Apr 2026)
argues current agentic memory implements *lookup*, not memory — that retrieval
generalizes by similarity to stored cases while weight-based memory generalizes by
applying abstract rules, that conflating them yields a generalization ceiling no
context size or retrieval quality can overcome, and that Complementary Learning Systems
theory says biology solved this by pairing fast exemplar storage with slow weight
consolidation, of which agents implement only the first half. The MemOS/MemCube line
takes the opposite position. Both are live.

---

## 7. What this implies for Mnemosyne

Three design consequences, stated as consequences and not as decisions:

1. **Occupancy telemetry is an interface, not instrumentation.** VISTA's result says the
   per-block token/recency/access counters are what the *policy consumer* needs, whether
   that consumer is a model or a heuristic. Mnemosyne should expose them as a first-class
   read on the cache, not as a logging side-channel. This also satisfies the lab's
   attribution stance, and it is exactly what `Argus` is reserved for.
2. **Measure forgetting, not only recall.** 2606.15903 and 2606.30306 independently say
   the field does not. A supersede/release/purge surface with its own tests is cheap and
   currently uncontested territory.
3. **Price the prefix invalidation.** Any Mnemosyne policy that mutates history must
   report `recompute_tokens`, not just tokens saved. The hash-chain arithmetic in §3 is
   the formula; the vLLM pointers are the reference implementation.

---

## Open questions

Testable at 20M–300M params, single GPU, `[M]` ≥62 GiB fast memory tier at ~200 GB/s
(`notebook/uma-carveout-controls-fast-tier.md`, 2026-07-26), with the `[M]` constraint
that single tensors ≥32 GiB hang or fault (`ASSUMPTIONS.md: large-tensor-fault-32gib`),
so any KV pool must be paged rather than allocated as one buffer.

1. **Is context proprioception emergent or trainable-in?** VISTA's lift is measured on
   frontier backbones. Train two matched ~100M models on identical token budgets, one
   with a synthetic occupancy dashboard block in the prompt, and score on a multi-hop
   synthetic retrieval task across occupancy levels. A null at 100M would say
   proprioception is a scale-gated capability; a positive would say it is an interface
   the whole size range can use, which is a far stronger claim than the paper makes.
2. **What does compaction actually cost in prefill?** No training required. Instrument
   `recompute_tokens` versus edit position for realistic agent transcripts against a
   local paged-KV engine, and turn "compaction is cheap" from an assumption into an
   `[M]`. This is the measurement that connects §3 to every paper in §5.
3. **Does the occupancy-dependent position-bias regime shift reproduce at small
   scale?** If arXiv 2508.07479's ~50%-occupancy transition holds at 20M–300M, then
   occupancy is a mandatory policy input and every fixed position prior is wrong in one
   regime. Cheap: synthetic needle tasks, matched budgets, ≥3 seeds.
4. **Clean-vs-dirty eviction, measured.** Build a synthetic agent task where some
   episode effects are externally persisted and some are not. Does a policy that reads
   that bit beat recency at matched token budget? This isolates CWL's central claim from
   its LLM-annotation machinery.
5. **Admission control transplant.** Does a TinyLFU-style frequency gate on *writes*
   into an agent memory store beat unconditional write-on-observation at matched budget?
   Mooncake does this at the KV tier; no agent-memory system does it at the note tier.
6. **The counterfactual nobody can afford.** With ~62 GiB of fast tier, a small model's
   KV for a long trajectory fits uncompressed. Per-token KV bytes are
   `2 · L · H_kv · d_head · b` — factor 2 for K and V, *L* layers, *H_kv* key/value
   heads after grouping, *d_head* per-head dimension, *b* bytes per element (2 for bf16).
   At a 300M-class config this is tens of KiB per token, so 62 GiB holds millions of
   tokens of exact KV `[A]`, high confidence in the arithmetic, unverified for any
   specific config. That makes it possible to run a compaction policy *and* the
   never-compacted control in the same experiment and measure what the policy cost —
   the attribution measurement the whole §6 literature says is missing, and one that is
   capacity-bound rather than FLOPS-bound, which is precisely what this hardware buys.

---

## Sources

Verified against the arXiv API on 2026-07-26 unless marked otherwise.

**Taxonomy and surveys**
- Memory in the Age of AI Agents — arXiv [2512.13564](https://arxiv.org/abs/2512.13564) (Dec 2025, rev. Jan 2026)
- Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers — arXiv [2603.07670](https://arxiv.org/abs/2603.07670) (Mar 2026)
- Rethinking Memory in LLM based Agents — arXiv [2505.00675](https://arxiv.org/abs/2505.00675) (2025)
- From Human Memory to AI Memory — arXiv [2504.15965](https://arxiv.org/abs/2504.15965) (2025)
- From Storage to Experience — arXiv [2605.06716](https://arxiv.org/abs/2605.06716) (May 2026)
- What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction — arXiv [2607.08032](https://arxiv.org/abs/2607.08032) (Jul 2026)
- A Survey of Context Engineering for Large Language Models — arXiv [2507.13334](https://arxiv.org/abs/2507.13334) (Jul 2025)
- Always-On Agents: Persistent Memory, State, and Governance — arXiv [2606.30306](https://arxiv.org/abs/2606.30306) (Jun 2026)
- The Missing Knowledge Layer in Cognitive Architectures for AI Agents — arXiv [2604.11364](https://arxiv.org/abs/2604.11364) (Apr 2026)
- Cognitive Architectures for Language Agents (CoALA) — arXiv [2309.02427](https://arxiv.org/abs/2309.02427) (2023)

**Systems and lineage**
- MemGPT: Towards LLMs as Operating Systems — arXiv [2310.08560](https://arxiv.org/abs/2310.08560) (2023)
- Sleep-time Compute: Beyond Inference Scaling at Test-time — arXiv [2504.13171](https://arxiv.org/abs/2504.13171) (Apr 2025)
- Letta, *Memory Blocks: The Key to Agentic Context Management* — https://www.letta.com/blog/memory-blocks (accessed 2026-07-26)
- Letta, *Sleep-time Compute* — https://www.letta.com/blog/sleep-time-compute (accessed 2026-07-26)
- A-MEM: Agentic Memory for LLM Agents — arXiv [2502.12110](https://arxiv.org/abs/2502.12110) (2025)
- Rethinking Memory as Continuously Evolving Connectivity (FluxMem) — arXiv [2605.28773](https://arxiv.org/abs/2605.28773) (May 2026)
- MemOS: A Memory OS for AI System — arXiv [2507.03724](https://arxiv.org/abs/2507.03724) (Jul 2025)
- MemOS: An Operating System for Memory-Augmented Generation (short) — arXiv [2505.22101](https://arxiv.org/abs/2505.22101) (May 2025)
- Mem0: Production-Ready AI Agents with Scalable Long-Term Memory — arXiv [2504.19413](https://arxiv.org/abs/2504.19413) (Apr 2025)
- The Missing Memory Hierarchy: Demand Paging for LLM Context Windows — arXiv [2603.09023](https://arxiv.org/abs/2603.09023) (Mar 2026)
- ES-Mem: Event Segmentation-Based Memory for Long-Term Dialogue Agents — arXiv [2601.07582](https://arxiv.org/abs/2601.07582) (Jan 2026)

**Context engineering and compaction**
- Parallel Context Compaction for Long-Horizon LLM Agent Serving — arXiv [2605.23296](https://arxiv.org/abs/2605.23296) (May 2026)
- ACON: Optimizing Context Compression for Long-horizon LLM Agents — arXiv [2510.00615](https://arxiv.org/abs/2510.00615) (Oct 2025)
- Beyond Compaction: Structured Context Eviction (CWL) — arXiv [2606.11213](https://arxiv.org/abs/2606.11213) (2026)
- Self-GC: Self-Governing Context for Long-Horizon LLM Agents — arXiv [2607.00692](https://arxiv.org/abs/2607.00692) (Jul 2026)
- ContextBudget: Budget-Aware Context Management for Long-Horizon Search Agents — arXiv [2604.01664](https://arxiv.org/abs/2604.01664) (Apr 2026)
- Learning What to Remember (OSL-MR) — arXiv [2606.10616](https://arxiv.org/abs/2606.10616) (Jun 2026)
- CompactionRL: RL with Context Compaction for Long-Horizon Agents — arXiv [2607.05378](https://arxiv.org/abs/2607.05378) (Jul 2026)
- LLM Agents Are Latent Context Managers (VISTA) — arXiv [2606.30005](https://arxiv.org/abs/2606.30005) (Jun 2026)
- TokenPilot: Cache-Efficient Context Management for LLM Agents — arXiv [2606.17016](https://arxiv.org/abs/2606.17016) (Jun 2026)
- AgentOCR: Reimagining Agent History via Optical Self-Compression — arXiv [2601.04786](https://arxiv.org/abs/2601.04786) (Jan 2026)
- Active Context Compression (Focus) — arXiv [2601.07190](https://arxiv.org/abs/2601.07190) (Jan 2026)
- Everything is Context: Agentic File System Abstraction — arXiv [2512.05470](https://arxiv.org/abs/2512.05470) (Dec 2025)

**Procedural memory**
- Managing Procedural Memory in LLM Agents (AFTER) — arXiv [2606.23127](https://arxiv.org/abs/2606.23127) (Jun 2026)
- From Raw Experience to Skill Consumption — arXiv [2605.23899](https://arxiv.org/abs/2605.23899) (May 2026)
- Skill-Pro: Learning Reusable Skills from Experience — arXiv [2602.01869](https://arxiv.org/abs/2602.01869) (Feb 2026)
- Procedural Memory Distillation — arXiv [2607.01480](https://arxiv.org/abs/2607.01480) (Jul 2026)
- Learning Hierarchical Procedural Memory for LLM Agents — arXiv [2512.18950](https://arxiv.org/abs/2512.18950) (Dec 2025)

**Evaluation, failure modes, security**
- Anatomy of Agentic Memory — arXiv [2602.19320](https://arxiv.org/abs/2602.19320) (Feb 2026)
- MemFail: Stress-Testing Failure Modes of LLM Memory Systems — arXiv [2605.26667](https://arxiv.org/abs/2605.26667) (May 2026)
- MemTrace: Tracing and Attributing Errors in LLM Memory Systems — arXiv [2605.28732](https://arxiv.org/abs/2605.28732) (May 2026)
- Structured Belief State / PrecisionMemBench — arXiv [2605.11325](https://arxiv.org/abs/2605.11325) (May 2026)
- Control-Plane Placement Shapes Forgetting — arXiv [2606.15903](https://arxiv.org/abs/2606.15903) (Jun 2026)
- LongMemEval-V2 — arXiv [2605.12493](https://arxiv.org/abs/2605.12493) (May 2026)
- MemoryAgentBench / Evaluating Memory in LLM Agents — arXiv [2507.05257](https://arxiv.org/abs/2507.05257) (Jul 2025)
- Contextual Agentic Memory is a Memo, Not True Memory — arXiv [2604.27707](https://arxiv.org/abs/2604.27707) (Apr 2026)
- Back to Basics: Let Conversational Agents Remember with Just Retrieval and Generation — arXiv [2604.11628](https://arxiv.org/abs/2604.11628) (Apr 2026)
- Beyond the Context Window: Fact-Based Memory vs. Long-Context LLMs — arXiv [2603.04814](https://arxiv.org/abs/2603.04814) (Mar 2026)
- MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis — arXiv [2605.03675](https://arxiv.org/abs/2605.03675) (May 2026)
- A Survey on Long-Term Memory Security in LLM Agents — arXiv [2604.16548](https://arxiv.org/abs/2604.16548) (Apr 2026)
- From Untrusted Input to Trusted Memory — arXiv [2606.04329](https://arxiv.org/abs/2606.04329) (Jun 2026)
- Remembering More, Risking More — arXiv [2605.17830](https://arxiv.org/abs/2605.17830) (2026)
- AgentPoison — arXiv [2407.12784](https://arxiv.org/abs/2407.12784) (2024)

**Long-context behaviour**
- Positional Biases Shift as Inputs Approach Context Window Limits — arXiv [2508.07479](https://arxiv.org/abs/2508.07479) (Aug 2025)
- Human-inspired Episodic Memory for Infinite Context LLMs (EM-LLM) — arXiv [2407.09450](https://arxiv.org/abs/2407.09450) (2024)
- Generative Agents: Interactive Simulacra of Human Behavior — arXiv [2304.03442](https://arxiv.org/abs/2304.03442) (2023)

**Code pointers** (all from `research/reference/CODE_MAP.md`, machine-verified at the
revisions pinned in `PROVENANCE.md`)
- `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` — `hash_block_tokens`, the chain hash
- `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:658` — `find_longest_cache_hit`
- `memory/vllm/vllm/v1/core/block_pool.py:702` / `:719` / `:679` — `touch`, `free_blocks`, `_maybe_evict_cached_block`
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` — leaf-constrained `evict`
- `memory/mooncake/mooncake-store/src/master_service.cpp:6382` / `:5211` — lease-expiry `BatchEvict`, TinyLFU-gated `TryPushPromotionQueue`

**Lab measurements**
- `ASSUMPTIONS.md: gpu-fast-tier-size` — `[M]` 2026-07-26, ≥62 GiB flat at ~200 GB/s
- `ASSUMPTIONS.md: large-tensor-fault-32gib` — `[M]` 2026-07-26, single tensors ≥32 GiB hang or fault
