---
title: Glossary — every term this repo uses, defined once
version: 1.0.0
date: 2026-07-26
track: reference (not a module; no exercises, no self-check)
audience: senior systems engineer, new to ML internals
covers: curriculum/ (31 modules), research/memory/ (10 notes), research/notes/ (7 notes), research/synthesis.md
---

# Glossary

Every term used anywhere in this repo, defined once. Written for someone who has thirty
years of distributed systems, storage hierarchy, caching, DR and observability behind him
and none of the ML vocabulary in front of him. The definitions are precise rather than
gentle; where a systems analogy genuinely helps, it is stated and then broken, because the
break is the part that carries information.

## How to read this

**Link convention.** `→ kv-cache-mechanics` means `curriculum/kv-cache-mechanics.md` —
the module that teaches the term properly. `→ RM: kv-serving-hierarchy` means
`research/memory/kv-serving-hierarchy.md`. `→ RN: evaluation-landscape` means
`research/notes/`. Terms with no module link are defined here and nowhere else, usually
because they are lab process rather than subject matter.

**Evidence tags**, per `CLAUDE.md` hard rule 2, apply to every material claim:
`[M]` measured on this lab's hardware (with a source), `[C]` cited (arXiv id or URL),
`[A]` assumed (with confidence and the cheapest test). A number without a tag in this file
is a definition or an identity, not an empirical claim.

**What is deliberately not here.** No arXiv id appears below unless it is in the verified
set in `curriculum/citation-verification.json` (`resolved` + `already_known` = 214 ids,
0 unresolved). Several load-bearing claims in the modules rest on ids in that file's
`unreachable` list (132 ids, never checked); where a glossary entry depends on one, it says
so and omits the id rather than laundering it.

**Three tables come before the alphabet** — the banned vocabulary, the cross-community
collisions, and the notation index — because each of them is a thing you need *before* an
individual definition is safe to use.

---

## Banned vocabulary — hard, and this file enforces it

`research/memory/memory-taxonomy.md` §9 bans a set of terms outright, on the grounds that
each of them names different bytes in two different literatures. The ban is repeated here
because a glossary is exactly where a banned term would sneak back in with a definition
attached.

| Use this | Never write this | Because |
|---|---|---|
| `weights` / parametric store | "long-term memory", "internal memory" | collides with the session store |
| `recurrent-state` | "hidden state memory", "compressive memory" | "compressive" is a property, not the thing |
| `kv-cache` | "activation memory", "working memory", "short-term memory" | all three mean something else to the other camp |
| `retrieval-index` | "external memory", "non-parametric memory" | "external" also describes the session store |
| `session-store` | "agent memory", "persistent memory", "long-term memory" | "agent memory" is the term that collides with `kv-cache` in the wild |

Two standing rules that come with the ban. **When citing a source, name which camp's
vocabulary it uses** — "MemOS's *activation memory* (= our `kv-cache`)". And **never write
"memory" as a bare noun in a hypothesis card**; a hypothesis whose subject is ambiguous
cannot fail cleanly, and a hypothesis that cannot fail is not one.

The banned terms still appear *below*, as entries, marked **BANNED**, with the two meanings
spelled out. You have to be able to read a paper that uses them.

---

## Collisions — one word, two communities, different bytes

The single most expensive thing in this literature. The serving community came from systems
(objects: tensors with a lifetime and an owner; nouns: blocks, pages, prefixes, eviction,
hit rate). The agent-memory community came from cognitive science (objects: episodic,
semantic, procedural, working, long-term; verbs: consolidate, forget, reflect, retrieve).
They describe *the same bytes* and rarely cross-cite `[C]` (2607.02574, 2507.03724).

| Word | Serving literature means | Agent-memory literature means | Our word |
|---|---|---|---|
| working memory | the resident KV cache | the text currently in the prompt | say `kv-cache` or `prompt occupancy` |
| long-term memory | a vector store / offloaded KV | the model weights | say `retrieval-index` or `weights` |
| activation memory | *also* the saved forward-pass tensors needed for backprop | the KV cache (MemOS's tier name) | say `saved activations` or `kv-cache` |
| memory hierarchy | a real tier ladder with a preference order | an aspirational scheduler across parametric/activation/plaintext | say which, and see **memo table** |
| eviction | dropping a block from a pool that can recompute it | forgetting a record with no backup | say `kv-eviction` or `session-store deletion` |
| retrieval | reading K/V out of the cache during attention | ANN top-k over an external index | say `attention read` or `index query` |
| context | three different numbers — see **advertised / admissible / effective context** | usually the prompt | always qualify |
| memory | five mechanically unrelated systems | five mechanically unrelated systems | never bare; name which of the five |

The Feb 2026 paper titled *Agent Memory Below the Prompt*, describing a persistent quantized
KV cache, is the collision in a title. A Jul 2026 rate–distortion framing is the first
serious attempt to unify all of it `[C]` (2607.08032) and it is weeks old; treat the
unification as an open proposal, not the field's position. → memory-taxonomy-for-engineers

---

## Notation index — the symbol collisions, repo-wide

Every module declares its own symbols locally and correctly. Nobody has been positioned to
read all 31 at once, so this is the first cross-module notation table, and it exists because
several of these collisions are between modules that quote each other's numbers.

| Symbol | Meaning | Where |
|---|---|---|
| `T` | tokens resident in context | consistent everywhere — the one symbol you can trust |
| `L` | layers in the stack | consistent, **except** Mamba's own code, where `L` is the intra-chunk decay matrix (flagged in constant-state-memory §3.1) |
| **`c`** | **bytes of KV per token for the WHOLE STACK** (`2·L·n_kv·d_h·b`) | memory-taxonomy-for-engineers §3.1 |
| **`c`** | **bytes of KV per token PER LAYER** (`2·n_kv·d_h·b`) | kv-cache-mechanics, hybrid-attention-and-ratios, quantization, speculative-decoding-and-serving |
| `k` | whole-stack bytes per token (the first `c` above, renamed) | paged-attention-and-prefix-reuse §3.1 |
| `b` | bytes per stored element (bf16 → 2) | Track B and most of Track C |
| `b` | **bits** in the stored payload (`B_e = b/8` is bytes) | quantization §3.1 — flags its own collision |
| `b` | retention budget as a **fraction** of entries kept | measuring-recall-and-memory §3.1 |
| `p` | bytes per stored element (where everything else says `b`) | memory-taxonomy-for-engineers §3.1 |
| `p_j`, `p_full`, `p` | attention weight / output distribution / survival probability | long-context, measuring-memory, measuring-recall |
| `B` | batch (concurrent sequences) | attention-variants, kv-cache-mechanics, constant-state |
| `B` | **block size in tokens** (batch is `R`) | paged-attention-and-prefix-reuse §3.1 |
| `B` | the eviction **budget** — how many past tokens survive | kv-eviction-policies §3.1 |
| `B` | the prompt **token budget** you chose to spend | agent-memory-in-practice §3.1 |
| `P` | parameter count | most modules |
| `P` | **block size in tokens** in the chain-hash formula | agent-memory-in-practice §3.2 |
| `d` | model width (residual stream) | attention-variants, distributed-training-strategies |
| `d` | head dimension | kv-eviction-policies §3.1 |
| `d` | **effect size** | building-an-eval, measuring-memory, measuring-recall |
| `w` | sliding window, in tokens | Track B and C |
| `w` | one real number being quantized | quantization §3.1 — **not flagged there** |
| `H` | attention entropy, in **nats** | long-context-and-effective-context §3.1 |
| `H` | text entropy per token, in **bits** | building-an-eval-you-can-trust §3.1 |
| `H` | heads per layer / the activation second-moment "Hessian" / the chain-hash function | constant-state, quantization, agent-memory |
| `m` | fraction of attention mass removed | long-context §3.1 |
| `m_E` | evicted attention mass (same quantity, different name) | measuring-memory §3.1 |
| `A` | *retained* attention mass — the complement | kv-eviction-policies §3.1 |
| `s` | salience of a target span | measuring-recall §3.1 |
| `s` | YaRN's `factor` (context scale) | long-context §3.1 |
| `s` | the quantization scale | quantization §3.1 |
| `α` | acceptance probability / collective latency / decay scalar / significance level | speculative, distributed, constant-state, building-an-eval |
| `ρ` | prefill FLOPs ratio / inter-arm correlation / KV budget fraction | paged, measuring-memory, building-an-eval |
| `M` | fast-tier bytes / metric-series cardinality / collective message size / SSM transition operator | paged, telemetry, distributed, constant-state |

**Rule of use:** never carry a symbol across a module boundary without re-reading that
module's §3.1. In particular, `c` differs by a factor of exactly `L` between
`memory-taxonomy-for-engineers` (whole stack) and `kv-cache-mechanics` (per layer). For
Laguna-S that is 192 KiB versus 4 KiB `[M]` — a 48× error if you mix them, and both modules
call their version "the module's working unit."

---

# The terms, A–Z

### A

**Ablation** — an experiment that removes or varies exactly one design axis at matched
parameter count and matched token budget, so the difference in outcome is attributable to
that axis. *Bridge:* an A/B test with a controlled blast radius. *Break:* your two arms are
two different trained models, and the seed-to-seed variance between two runs of the *same*
config is frequently larger than the effect you are testing — so an ablation without ≥3
seeds and a stated confidence interval is an anecdote, and the house standard says to label
it as one. → building-an-eval-you-can-trust

**Activation memory** — **BANNED as a memory-tier name.** Two live meanings: (1) the
forward-pass tensors PyTorch retains because the backward pass will need them — the term's
original and correct use, and the largest training-time allocation after the optimizer
state; (2) MemOS's name for the KV cache tier `[C]` (2507.03724). Say `saved activations`
for (1) and `kv-cache` for (2). → tensors-and-autograd §2.5, memory-taxonomy-for-engineers

**Admissible context** — the longest request the serving stack will actually accept once KV
memory has been accounted for; decided by the server at startup, visible in a log line.
Fails loudly (`ValueError`) or silently by auto-fitting downward. → long-context-and-effective-context §1.1

**Admission control** — refusing a request rather than degrading everyone. In vLLM this is
what `allocate_slots` returning `None` means: not a fault to be serviced but a rejection
that preempts the entire request. *Bridge:* a load shedder. *Break:* the shed unit is the
whole sequence, so there is no partial-service mode. → paged-attention-and-prefix-reuse

**Advantage collapse** — in GRPO, when every rollout in a group receives the same reward the
group-normalized advantage is zero for all of them and the batch contributes no gradient;
at small scale, where a model solves either none or all of a group, this is the common case
rather than an edge case. → supervised-and-preference-finetuning §3.8

**Advertised context** — the `max_position_embeddings` field. A claim, not a behaviour; it
can never fail because nothing measures it. Laguna's 1,048,576 is `8192 × 128` exactly —
pretraining length times the YaRN extension factor `[M]` — and never enters the RoPE
frequency computation. → long-context-and-effective-context §1.1

**AdamW** — Adam with weight decay applied directly to the parameter rather than folded into
the gradient `[C]` (1711.05101). Carries two extra full-precision tensors per parameter
(first and second moment), which is why optimizer state, not weights, usually dominates the
training memory bill. → the-training-loop §3.3, loss-and-optimization

**ALiBi** — attention with linear biases: a fixed per-head linear penalty on distance,
added to the attention logit instead of encoding position in the vectors `[C]` (2108.12409).
Superseded in practice by RoPE, kept in the curriculum because it is the cleanest example of
positional information entering as a bias rather than a rotation. → positional-encoding §1.3

**A-MEM** — a self-organizing agent-memory design: Zettelkasten-style notes with
LLM-generated links and retroactive refinement, contrasted against MemGPT's fixed tiers
`[C]` (2502.12110). Its `consolidate_memories` rebuilds the entire collection from scratch
on every consolidation — compaction as a full rewrite. → agent-memory-in-practice

**Arithmetic intensity** — FLOPs performed per byte moved from memory, for a given kernel.
The single number that decides whether an operation is compute-bound or bandwidth-bound.
For attention decode the closed form is `2G/b` where `G` is the GQA group size and `b` is
bytes per stored KV element — so in bf16 (`b=2`) it equals `G` exactly, independent of
context length and of depth. *Bridge:* IOPS versus throughput on a storage device; you size
for whichever the workload actually stresses. *Break:* on a discrete GPU the compute and the
bandwidth are separate resources; on a unified-memory machine they are the same silicon, so
"trade compute for bandwidth" is not always a trade.
→ attention-variants-and-kv-cost §3.6, RM: kv-cache-mechanics

**Argus** — reserved system name for telemetry, currently living inside Themis. Earns its
own package when it has an interface, a lifecycle and plausible extractability, per
`CLAUDE.md`. Named forward in `moe-and-routing` (router health signals) and
`agent-memory-in-practice` (the attribution side-channel). → training-telemetry-as-observability

**Arm** — one configuration in an experiment, named for what it *is*
(`proteus-swa-4to1`, `mnemosyne-h2o`), never `arm_a`. Architecture arms take the
`proteus-` prefix; memory-policy arms take `mnemosyne-`.

**Aspect ratio** — `d_model / L`: how wide a model is relative to how deep, at fixed
parameter count. The `12·L·d²` identity means you can trade depth for width along an
iso-parameter curve, and the trade is not free in either direction.
→ depth-width-and-initialization §3.3

**Attention sink** — the first few token positions, which absorb a large and largely
content-independent share of attention mass; deleting them collapses generation quality even
though they carry no relevant information `[C]` (2309.17453). *Bridge:* a pinned page.
*Break:* nothing marked it as important — the model needs somewhere to dump probability
mass, so the "importance" is an artifact of softmax being forced to sum to one, not a
property of the tokens. → kv-eviction-policies §2.5

**Attribution** — isolating *which mechanism* produced an observed effect, as opposed to
reporting that the effect occurred. Scored P5·T5·E5 in `open-problems-ranked.md` — the only
5/5/5 — and named as this lab's deliverable. The mechanical reason it is hard here: **a
storage tier reports its own misses; a memo table cannot, so the miss signal has to be
manufactured, and manufacturing it is the contribution.** *Bridge:* distributed tracing —
you instrument for the causal chain, not just the SLO. *Break:* there is no request id and
no span boundary inside a forward pass; the only counterfactual available is running the
expensive full-cache reference alongside the policy and diffing.
→ measuring-memory §4.1, research/synthesis.md

**Auxiliary loss (MoE)** — an added loss term that penalizes unbalanced expert load. Its
known defect is that it can be minimized by making the router uninformative, so a falling
aux loss is not evidence of a healthy router. → moe-and-routing

**Aux-loss-free balancing** — replacing the auxiliary loss with a per-expert bias added to
the router score *before* top-k selection and adjusted by a controller, so load is
rebalanced by nudging *selection* while leaving the *combination* weights untouched
`[C]` (2408.15664, 2412.19437). *Bridge:* a load balancer whose health signal is a learned
bias rather than queue depth. *Break:* a mis-balanced expert does not merely add latency —
it permanently changes what that expert learns, so the control loop and the thing it
controls are not separable. The controller has a hard saturation threshold worth knowing
before you tune it. → moe-and-routing

### B

**Batch invariance** — the property that a token's logits do not depend on what else was in
its batch. Not free: it costs a fixed reduction order, and vLLM ships it as an explicit
feature with a price list. Matters here because if it does not hold, an eviction decision
can flip on batch composition alone. → determinism-and-reproducibility §4.2

**Belady's MIN** — the provably optimal cache replacement policy, which requires knowing the
future. *Bridge:* the standard yardstick — every real policy is an approximation and the gap
to MIN is what you tune. *Break:* three ways. (1) A wrong eviction costs *correctness*, not
latency; there is no miss to service, so competitive-ratio framing has no cost function.
(2) The reference string is generated by the process you are perturbing — evicting a token
changes the model's later hidden states, which changes its later queries; there is no "the"
access sequence to be optimal against. (3) Importance is *estimated*, and a Jul 2026 result
argues deterministic top-k eviction cannot estimate its own error consistently at all. That
last citation is among the 132 unverified ids; treat it as directional.
→ memory-failure-modes §2.3, kv-eviction-policies §2.4

**Bits per byte (BPB)** — cross-entropy loss converted to bits and normalized by *source
bytes* rather than tokens, so it is comparable across tokenizers. The right unit when
comparing models with different vocabularies; perplexity is not. → loss-and-optimization

**Block (KV)** — the allocation unit of a paged KV cache: a fixed number of token slots,
contiguous in physical memory, referenced through a block table. `B` in
`paged-attention-and-prefix-reuse` is the block size *in tokens*, not batch.
→ paged-attention-and-prefix-reuse §3.3

**Block table** — request id → ordered list of physical block ids; list index is the logical
block. *Bridge:* a page table, and the analogy is unusually good — vLLM's is literally that
`[C]` (2309.06180), at `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`.
*Break:* there is no fault handler behind it. A page table's whole reason to exist is to
trap on absence; this one cannot, because there is no lower tier to fault to.
→ paged-attention-and-prefix-reuse

**BPE (byte-pair encoding)** — build a vocabulary by repeatedly merging the most frequent
adjacent symbol pair `[C]` (1508.07909). Encoding applies the learned merges greedily, which
is *not* guaranteed to produce the minimal token count — the module gives a counterexample.
→ tokenization

**Bradley–Terry model** — the probabilistic model behind pairwise preference learning: the
probability that response A beats B is a logistic function of their reward difference. The
bridge from "humans ranked these" to "here is a scalar reward."
→ supervised-and-preference-finetuning §3.2

### C

**Capacity factor (MoE)** — the per-expert buffer size as a multiple of the average load;
tokens beyond it are *dropped* (their expert contribution is skipped). *Bridge:* a bounded
queue with tail-drop. *Break:* the dropped unit is not a request you can retry — it is a
token whose representation is silently worse, and nothing in the loss identifies which.
→ moe-and-routing

**Cardinality (telemetry)** — the number of distinct metric series emitted per step. The
analogy to observability cardinality mostly holds; the axis that replaces "tags" is
*per-layer* series, which grow with `L` and turn one drain into `L` device syncs.
→ training-telemetry-as-observability §2.4

**Chain hash** — see **prefix hash chain**.

**Chat template** — the wire format that turns a list of role/content messages into one
token string, shipped as a Jinja template inside the tokenizer artifact. *Bridge:* a
serialization format. *Break:* there is no version negotiation and no content type — the
model was trained against one rendering, and a mismatched template degrades quality with no
error anywhere. → tokenization, supervised-and-preference-finetuning §4.2

**Chinchilla** — the compute-optimal scaling result: for a fixed FLOPs budget, parameters
and training tokens should scale together at roughly 20 tokens per parameter `[C]`
(2203.15556), correcting an earlier parameter-heavy prescription `[C]` (2001.08361).
→ scaling-laws-and-flops-budget

**Chiron** — the lab: repo, docs, curriculum, the umbrella. Named for the centaur who
tutored Achilles, Jason and Asclepius. One of exactly four names in this repo permitted to
be arbitrary.

**Chunked prefill** — processing a long prompt in fixed-size chunks so prefill work can be
interleaved with decode work in one scheduler budget. **Carries a numerics consequence
nobody advertises:** `[M]` on this box (2026-07-26, toy 8-layer stack, `T=512`), fp32
chunked prefill is bit-identical to one-shot prefill, but **bf16 chunked prefill diverges
from layer 3**, reaching final-hidden-state rel-L2 **5.57e-3** with only 44.1% of components
bitwise equal. Both are valid roundings of the same exact quantity. The consequence for this
lab is direct: an oracle-diff instrument that does not pin the prefill schedule across arms
is measuring the scheduler. → memory-taxonomy-for-engineers §3.7

**Compaction (agent memory)** — summarizing or rewriting the middle of a stored history to
free budget. *Bridge:* log compaction in a write-ahead store. *Break:* three ways. It is
lossy with no checksum, so you cannot tell afterwards what was dropped. It rewrites the
*middle* of the log, which in a prefix-cached serving stack invalidates every downstream
block hash and costs a re-prefill of the tail. And the summarizer is an LLM call, which
makes compaction itself a write channel and therefore an attack surface `[C]` (2606.04329).
→ agent-memory-in-practice §2.4, §3.3

**Composition law (recall)** — recovering `Q` independent facts each at salience `s` has the
same survival probability as recovering one fact at salience `s/Q`. The reason multi-fact
recall degrades much faster than single-needle recall, and the reason NIAH flatters a policy.
→ measuring-recall-and-memory §3.3

**Constant state** — see **recurrent state**. Note the qualifier that costs people money:
"constant" is a **decode-time** property. During training and prefill the chunked scan
materializes one fp32 state per chunk, so activation memory there is `O(L / chunk_size)`,
not `O(1)`. → constant-state-memory §3.9

**Contamination** — test items present in training data. *Bridge:* a cache hit you did not
intend. *Break:* it inverts at our scale — a 300M model trained on 2B tokens has so little
capacity that the usual "the model memorized the benchmark" story is weak, while
*n*-gram-overlap detectors have a false-positive floor set by the entropy of the text
itself. → building-an-eval-you-can-trust §2.3, §3.5

**Context parallel** — sharding a sequence across ranks so each holds part of the KV cache;
the one parallelism axis that touches the KV cache directly, and therefore the one Mnemosyne
must be able to name even though we cannot run it. → distributed-training-strategies §4.1

**Context proprioception** — a model's ability to sense how much of its own context budget
is occupied and adjust behaviour accordingly. Open question: emergent with scale, or
trainable in at small scale? A null at 100M would say it is scale-gated.
→ agent-memory-in-practice §8.4

**Continuous batching** — admitting and retiring sequences from a running batch every step
rather than at batch boundaries. One budget, two SLOs (time-to-first-token vs inter-token
latency) pulling in opposite directions. → speculative-decoding-and-serving §3.7

**Coupon-collector problem** — the shape of MoE decode weight traffic: with `E` experts and
`k` active per token, the number of *distinct* experts a batch touches grows like the coupon
collector's expectation, so weight bytes read per step rise sharply with batch size before
saturating. This is why MoE decode has a batch knee that dense decode does not.
→ moe-and-routing

**Cross-entropy** — the training loss: the negative log probability the model assigned to
the token that actually came next, averaged over positions. In nats if you use `ln`, bits if
you use `log₂`. Everything else in training is machinery around this one number.
→ loss-and-optimization

### D

**Decode** — the autoregressive phase: one token at a time, tiny matmuls, the entire KV
cache read every step. Bandwidth-bound by construction. Contrast **prefill**.
→ kv-cache-mechanics §2.3

**Delta rule** — the write rule in DeltaNet-family linear attention: read what the state
currently returns for this key (`hᵀk`), subtract it, and add back `β·(v − hᵀk)` as a rank-1
outer product. `β=1` is an exact overwrite of that key's direction; `β=0` is a no-op. This
— not the gate — is the *targeted* erase in the layer.
`architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56` →
constant-state-memory §3.6

**Determinism** — two properties that get conflated, expensively: *repeatability* (same
inputs, same machine, same bits) and *reproducibility* (same recipe, different machine,
same conclusion). You can have the first and not the second, and only the second is what an
experiment needs. → determinism-and-reproducibility §2.1

**Detection gap** — the interval between a training failure occurring and anything noticing.
The term nobody models and the one that dominates RTO for a one-person lab: `[M]` a 32 GiB
allocation on this machine hangs at 0% CPU with no error and no exit code
(`ASSUMPTIONS.md → large-tensor-fault-32gib`), so an unattended overnight run in that state
has a detection gap measured in hours. The cheapest DR improvement here is a step-progress
watchdog, not faster checkpoints. → checkpointing-and-resumption §3.2, §3.6

**Dilution theorem** — as context length `N` grows with the needle's logit margin `Δ` held
fixed, the attention weight on the needle falls, and the loss is expressible in nats: you
need roughly `ln N` of extra margin to hold position. *Bridge:* signal-to-noise in a
widening search. *Break:* eviction, which shortens the effective `N`, therefore *helps* the
needle — which is why needle-in-a-haystack cannot falsify a heavy-hitter policy.
→ long-context-and-effective-context §3.2

**Disaggregation (prefill/decode)** — running the compute-bound prefill phase and the
bandwidth-bound decode phase on separate hardware, moving the KV cache between them.
→ RM: kv-serving-hierarchy

**DPO (direct preference optimization)** — training directly on preference pairs by
reparameterizing the reward as a log-ratio against a frozen reference policy, deleting the
online reward model and the RL loop. *Bridge:* replacing a live scoring service with a
materialized view. *Break:* the view is the *reference model*, it is a second full copy of
the weights held in memory for the whole run, and the well-documented artifact is that the
chosen response's absolute likelihood *falls* during training — only the margin rises.
→ supervised-and-preference-finetuning §3.3, §3.4

**Drain (telemetry)** — the host-device synchronization that moves accumulated metrics off
the GPU. Cost is `c₀ + c₁·M` for `M` series, amortized over `k` steps. *Bridge:* a scrape
interval. *Break:* a Prometheus scrape does not stall the process it scrapes; this one does,
because reading a metric off a GPU means waiting for the queue to empty.
→ training-telemetry-as-observability §3.2

**Drafter** — the cheap model or head that proposes `K` speculative tokens for the target
model to verify. Taxonomized by who pays the fixed cost: a separate small model, extra heads
on the target, or n-gram lookup. → speculative-decoding-and-serving §2.4

### E

**Effective bits** — the actual bits per stored element once the group scale (and zero-point)
are counted, e.g. 4-bit weights with an fp16 scale per group of 128 cost `4 + 16/128 = 4.125`
bits. The "half-bit tax". It decides whether a quantized model fits, and the scales do
**not** amortize for a KV cache the way they do for weights, because a KV scale tensor grows
with tokens. → quantization §3.3, §3.7

**Effective context** — the longest length at which measured quality still clears a stated
threshold. Decided nowhere; it must be measured, and it is the only one of the three
"context" numbers with no instrumentation anywhere in the stack. Two hazards: a single
threshold induces an unstable ranking across models, and RULER-style reports do not publish
their variance. → long-context-and-effective-context §1.1, §3.4

**Embedding tax** — the fraction of a small model's parameters sitting in the token
embedding and output head. At our scale it is large enough that "matched parameters" is
ambiguous unless you say whether embeddings are counted.
→ depth-width-and-initialization §3.4

**Episodic memory** — time-indexed records of what happened, in the agent-memory
four-box taxonomy. The test that makes the boxes real: **if your store has one retention
policy, you have one memory type no matter how many tables you have.** Most shipped systems
fail it. → agent-memory-in-practice §2.6

**Eviction (KV)** — permanently discarding cache entries under a budget. Note what it is
*not*: it is not a working-set decision, because every decode step reads 100% of the
resident set; and it is not a data-safety decision, because the bytes are recomputable.
Restated precisely: **not "which entries stay resident" but "which entries do we delete
forever and hope we did not need."** → kv-eviction-policies §2.1

**Expert parallel** — sharding MoE experts across ranks. The collective whose shape the
*model* chooses at runtime (an all-to-all sized by the routing decision), which is why its
cost is data-dependent in a way no other parallelism axis is.
→ distributed-training-strategies §3.6

### F

**Fast tier** — the region of the gfx1151 unified-memory pool that sustains full bandwidth.
Lab-specific and load-bearing: `[M]` 2026-07-26 (`notebook/uma-carveout-controls-fast-tier.md`,
**single run per arm — an anecdote by the house standard**) with the BIOS UMA carve-out at
16 GiB the tier ends at a 30 GiB footprint and collapses to 61.3 GB/s; at 96 GB it is flat at
~200 GB/s out to **≥62 GiB** with no degradation. The upper edge is unmeasured — the sweep
hit the 32 GiB single-tensor fault, not a bandwidth knee. **Budget every long-context
experiment against ≥62 GiB.** *Bridge:* the resident set of a storage cache. *Break:* it is
set by a BIOS field, not by a bus, which is precisely why this machine can vary a fast/slow
bandwidth ratio that a discrete GPU cannot vary at all — see **tier-ratio experiment**.

**Fault injection (eval calibration)** — deliberately breaking a known thing and checking
that the eval notices. **An eval you have never seen fail is a decoration.** The lab's
battery is six faults: needle absent, needle's KV dropped, uniform eviction, RoPE-phase
corruption, retrieval-head masking, haystack shuffle. *Bridge:* chaos engineering. *Break:*
in chaos engineering the fault is the thing under test; here the fault is the *ruler*, and
the thing under test is whether your metric has any resolution at all.
→ building-an-eval-you-can-trust §2.6, measuring-memory §2.6

**Fertility** — the exchange rate a tokenizer charges: mean tokens emitted per unit of
source text. *Bridge:* a codec's compression ratio. *Break:* the cost is not storage, it is
every downstream term at once — fertility multiplies through into prefill FLOPs, KV bytes,
and the quadratic attention term. And it is content-dependent (code, prose and digit strings
have very different fertilities), so it is a workload property, not a model constant.
→ tokenization

**FLOPs budget (6ND)** — total training FLOPs ≈ `6 · N · D` for `N` parameters and `D`
tokens: 2 for the forward matmul, 4 for the backward. What it leaves out is attention, which
has no parameters and therefore no `N` — and that omission stops being negligible at long
context. → scaling-laws-and-flops-budget

**Free-row window `r(K)`** — the ratio of the cost of one target forward pass over `K+1`
query rows to the cost over 1. Near 1 while decode is bandwidth-bound (extra query rows ride
along free); it is where `r(K)` departs from 1 that speculation stops paying.
→ speculative-decoding-and-serving §3.4

**FSDP (fully sharded data parallel)** — shard parameters, gradients and optimizer state
across ranks; all-gather the shard you need just before you use it, then discard it.
*Bridge:* sharded replication with lazy hydration. *Break:* the "replica" is reassembled
every layer of every step, so the bandwidth cost is per-layer-per-step rather than per-write.
**Not runnable here:** `[M]` 2026-07-26 on the lab wheel, `torch.distributed.is_available()`
is False, `torch._C._distributed_c10d` does not exist in the build, and the FSDP import
raises `ModuleNotFoundError` — parallelism is design-only, and it fails at *import*, which is
a cleaner constraint than silently half-running. → distributed-training-strategies

### G

**Gated DeltaNet** — a linear-attention layer combining a scalar decay gate with the delta
rule. Its state is one `d_k × d_v` matrix per head. The folk story that "gating = selective
forgetting" is backwards: the shipped layer's gate is a *single scalar* attenuating the whole
matrix indiscriminately; the fused kernel implements per-key and per-channel gates
(`USE_GK`/`USE_GV`) and the layer simply does not pass them. → constant-state-memory §2.4

**Ghost list** — ARC's record of recently-evicted keys, used to adapt the recency/frequency
balance. **No analogue exists here**, and the reason is structural: a ghost list works
because a later reference to an evicted key is observable. Here there is no key, no
reference, and no observation. → kv-eviction-policies §2.4

**GQA (grouped-query attention)** — several query heads share one key/value head, so the KV
cache shrinks by the group size `G = n_q / n_kv` `[C]` (2305.13245). MHA is `G=1`, MQA is
`G = n_q` `[C]` (1911.02150). *Bridge:* deduplication. *Break:* the usually-missed corollary
is that GQA does not only shrink bytes, it *raises decode arithmetic intensity by the same
factor* — two wins from one knob. → attention-variants-and-kv-cost §3.4

**GQA group size** — `G = n_q / n_kv`. Derived, never configured. `[M]` On Laguna-S it is
**6 on the 12 full-attention layers and 9 on the 36 sliding layers**, because query heads are
per-layer (48/72) while `num_key_value_heads` is uniform at 8. Consequence: decode arithmetic
intensity is **not a single number for the model**, and a Mnemosyne cost model keyed on the
top-level `num_attention_heads` is wrong for 75% of layers
(`ASSUMPTIONS.md → laguna-heads-uniform`, `decode-intensity-varies-by-layer`).

**Gradient accumulation** — summing gradients over several microbatches before one optimizer
step, to simulate a larger batch. *Bridge:* batching writes before a commit. *Break:* the
normalization is easy to get wrong (the mean must be over the *total* token count, not per
microbatch), and it is not numerically identical to true data parallelism because the
reduction order differs. → the-training-loop §3.5

**GRPO** — group-relative policy optimization: sample a group of rollouts per prompt,
normalize rewards within the group to form advantages, and skip the value network. See
**advantage collapse** for why it degrades at small scale. Its rollout group is also the most
prefix-shareable workload that exists, which makes it a free Mnemosyne benchmark.
→ supervised-and-preference-finetuning §3.7, §3.10

**GSSS** — the shipped Laguna layer pattern: one Global followed by three Sliding, repeated.
`[M]` 48 layers as 12 `full_attention` + 36 `sliding_attention`, strict GSSS, `sliding_window`
512, read from `config.json` at `b0a9fd7c850e` rather than quoted. The whole hybrid-ratio
research question reduces to what goes in `config.layer_types` —
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` — which
makes it trivially ablatable. → hybrid-attention-and-ratios

### H

**H2O (heavy-hitter oracle)** — evict by accumulated attention mass, keeping the tokens that
have historically received the most `[C]` (2306.14048). `[M]` The accumulator is biased
toward old tokens — a token present for more steps has had more chances to accumulate —
and `kv-eviction-policies` §3.7 quantifies the bias. → kv-eviction-policies

**Half-bit tax** — see **effective bits**.

**Hardware Validation Gate** — the named gate in `CLAUDE.md` that blocks all research runs
until this machine's stack is proven: capacity ceiling, bf16 numerics against fp32,
determinism across repeated runs, bit-exact checkpoint round-trip, hipBLASLt configured,
and a known-good tiny recipe to a published loss target. **It has not run.** Standing
consequence: no number measured on this machine counts as evidence by house standard yet,
including the ones quoted in this glossary — they are instrument-shakedown readings.
`research/synthesis.md` argues the gate as written is under-specified and should be widened
before it is closed (add RoPE-at-long-position in bf16, an *attention-kernel* roofline rather
than a GEMM one, and the fp32 discriminator for the 32 GiB fault).

**Heavy hitter** — a cache entry with high accumulated attention mass. The assumption every
mass-based policy makes is that past mass predicts future mass; the assumption fails exactly
when the next query is about something the earlier queries were not.
→ kv-eviction-policies

**hipBLASLt** — AMD's GEMM library, selected via `TORCH_BLAS_PREFER_HIPBLASLT` and
`HIPBLASLT_TENSILE_LIBPATH`. `[M]` 2026-07-26: on this wheel it is **not** a 5× throughput
cliff (18.6 → 20.9 TFLOPS bf16 at 8192³, +12%) — the widely-cited cliff is **refuted here**.
What it *is* is a numerics control: relative error of a length-1M bf16 weighted sum against
an fp64 reference is **2.01e-3 configured versus 5.60e-3 unconfigured**, ~2.8×, reproduced
across 3 seeds. Every run must record whether it was configured, and any long-context result
taken without it is confounded by arithmetic (`ASSUMPTIONS.md → hipblaslt-config`).

**Hybrid (attention)** — mixing an expensive primitive with a cheap one. "3:1 hybrid"
collapses **four independent decisions**: the efficient primitive (SWA vs constant-state vs
block-sparse), the fusion granularity (inter-layer vs head-wise vs weight-tied), the ratio,
and the cheap layer's own window or state size. Two traps that have produced wrong published
comparisons: *the denominator moves* (Jamba's 1:7 is attention:Mamba, Gemma 3's 5:1 is
local:global), and *the primitive changes what the ratio means* (a 3:1 SWA model still runs
softmax attention everywhere with `O(w)` KV; a 3:1 linear model has no attention at all in
three of four layers). → hybrid-attention-and-ratios §2.1

### I

**Induction head** — an attention head implementing "if the pattern `AB` occurred earlier and
we just saw `A`, predict `B`" — the mechanistic basis of in-context copying. Relevant here
because it means a needle's content is smeared into *later* positions' cache entries, so
dropping only the needle's own entries measures a **lower bound** on its causal contribution.
→ measuring-memory §4.4

**Interference** — the failure mode of a fixed-size associative state: a *similar* key
partially overwrites a neighbour's content, because keys are continuous L2-normalized vectors
with no addressing. *Bridge:* hash collisions. *Break:* a hash collision is detectable and a
chained bucket resolves it; here nothing detects it, there is no bucket, and the corrupted
read is returned with full confidence. The closed form is in constant-state-memory §3.4, and
mutual coherence `μ = max_{i≠j}|k_i·k_j|` is the governing quantity.
→ constant-state-memory

**IsoFLOP** — the experimental design behind scaling laws: hold total FLOPs fixed, sweep the
parameter/token split, find the minimum. The fitting procedure has a known bias worth
reproducing before you budget a real sweep. → scaling-laws-and-flops-budget

### K

**K/N law (retrieval)** — under a fixed prompt budget, the probability that a needed item is
in the injected set is governed by `K` retrieved over `N` stored, and improving the *scorer*
moves it far less than changing the *budget*. The argument that agent memory is an allocation
problem wearing a retrieval problem's clothes. → agent-memory-in-practice §3.4

**KV cache** — for every token in every layer that keeps a cache, one key vector and one
value vector, stored exactly as computed. Nothing is summarized, nothing is merged.
`kv_bytes = 2 · L · n_kv · d_h · b · T`. *Bridge:* a buffer cache — **and this is the analogy
this whole lab is built on breaking.** Five breaks: (1) no backing store, so nothing can be
dirty and "reload" means re-running prefill; (2) no fault path — vLLM's `allocate_slots`
returns `None` and the scheduler preempts the *whole request*; (3) a hit removes compute,
never I/O, and supplies no information the tokens did not already carry; (4) no working set —
every decode step reads 100% of the resident entries, per head, per layer, so there is no
temporal locality and no hit rate to improve; (5) no associative match — reuse is keyed on a
*chain* hash over the whole prefix. → kv-cache-mechanics, memory-taxonomy-for-engineers §2.3

**KV cache, the three budgets** — the organising idea of `kv-cache-mechanics`, with no
analogue in the literature: **residency** (bytes held while the request lives — decides
whether you fit), **read traffic** (bytes into the attention kernel per decode step — sets
tokens/s), and **maintenance traffic** (bytes moved in order to *append* one token's entry —
normally assumed zero, and in the reference implementation everyone benchmarks against it is
the entire cache, twice, every token). *Bridge:* you already price write amplification
separately from read amplification, and you know a log-structured store's compaction traffic
can exceed its user traffic. *Break:* on an SSD write amplification is a property of the
device; here it is a property of *which Python class the serving stack instantiated*, and
nothing in the config, the checkpoint or the API tells you which one you got.
→ kv-cache-mechanics §2.2

### L

**Laguna S 2.1** — the reference model under study: 118B total / 8.5B active MoE,
OpenMDW-1.1, ungated. `[M]` from `config.json` at `b0a9fd7c850e` (2026-07-26): 48 layers,
12 `full_attention` + 36 `sliding_attention` in strict GSSS, `sliding_window` 512,
`num_key_value_heads` 8 (uniform), `head_dim` 128, query heads 48 full / 72 sliding, 256
experts with 10 active. Derived: `c` = **192.0 KiB per token whole-stack, exactly**
(= 4 KiB per token per layer), so 24.0 GiB at 128k context if every layer were global — but
36 of 48 are windowed, giving **6.07 GiB** at 128k, a 4× reduction from one config list.

**Laguna XS 2.1** — the smaller sibling used in `running-laguna-locally`: 40 layers,
10 global / 30 windowed, `hidden_size` 2048, `head_dim` 128, `n_kv` 8, query heads 48/64,
`vocab_size` 100,352, 256 experts with 8 active, max context 262,144. Its per-layer `c` is
identical to S's (4 KiB) because `n_kv` and `d_h` match. **Read the suffix before you reuse a
Laguna number**; S and XS differ in depth, width, active-expert count and advertised context.

**Lethe** — reserved name for the eviction-policy layer inside Mnemosyne. Unused until the
thing grows. Its founding constraint is already recorded: **eviction costs a compaction** —
removing an entry from a contiguous cache is not free, it is a restructuring of the buffer,
and that maintenance traffic is a budget line. → kv-cache-mechanics §4.2

**Logit** — the raw, un-normalized score the model produces for one vocabulary token at one
position, before softmax. *Bridge:* an unnormalized score in a ranking system. *Break:* the
logits tensor is `[batch, positions, vocab]` and is routinely the single largest allocation
in a training step — larger than the weights — which is a shape fact people meet as an OOM
rather than as arithmetic. → loss-and-optimization

**Logit margin (`Δ`)** — how far the needle's pre-softmax score leads the background. The
quantity the dilution theorem consumes. → long-context-and-effective-context §3.1

**Long-term memory** — **BANNED.** Means the model *weights* to the cognitive camp and a
*vector store* to the systems camp. Say `weights` or `retrieval-index`.

**Lost in the middle** — the observation that information placed mid-context is recovered
less reliably than information at either end `[C]` (2307.03172). Presented as **contested**
in this repo: the position-bias literature does not agree on whether the effect is a property
of the architecture, of the training data's document structure, or of the eval's construction.
→ long-context-and-effective-context §2.2

### M

**Maintenance traffic** — see **KV cache, the three budgets**.

**Mamba-2 / SSD** — a state-space model whose inference cache is allocated with shape
`(batch, nheads, headdim, d_state)` — **no sequence-length term at all**
(`architecture/mamba/mamba_ssm/modules/mamba2.py:352`) `[C]` (2312.00752, 2405.21060).
Decode footprint at 1K tokens and at 1M tokens is byte-identical. Its inter-chunk recurrence
is a destructive overwrite, not an append
(`architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80`).

**Memo table** — **the load-bearing correction to your instincts.** A cache stores a *copy*
of something that exists elsewhere; a memo table stores the *result of a pure function* you
would rather not call again. `f(tokens, weights)` is deterministic in exact arithmetic, so a
KV entry is not data — it is **deferred work**. Consequences: durability, writeback,
coherence, checksums, replication and DR are all *inapplicable*, because there is nothing to
be coherent with; eviction is a compute/quality tradeoff, not a data-safety tradeoff; and
Mooncake's `offload_force_evict` may throw bytes away rather than block on writeback
(`memory/mooncake/mooncake-store/src/master_service.cpp:6382`) — a tradeoff no real storage
tier is allowed to make, legal here precisely because there is no data to lose. Contrast
**storage tier**. → memory-taxonomy-for-engineers §2.4

**MemGPT** — the design that set the framing everything in agent memory inherits: main
context vs external context, self-editing memory via tool calls, page-fault-style interrupts
`[C]` (2310.08560). *Bridge:* virtual memory, and the paper says so. *Break:* the "page
fault" is a prompt-level LLM decision, not a trap; there is no hardware to enforce it and no
guarantee it fires.

**MemOS** — the design that partitions memory into parametric / activation / plaintext tiers
with an explicit scheduler, placing the KV cache in "activation" as a first-class scheduled
resource `[C]` (2507.03724). Cited throughout this repo as the *aspiration* of a memory
hierarchy, explicitly not as a description of anything shipping.

**MFU (model FLOPs utilization)** — achieved model FLOPs/s ÷ device peak FLOPs/s. *Bridge:*
CPU utilization. *Break:* both halves are conventions. The numerator uses `6ND`, which omits
attention entirely; the denominator is a vendor peak for a specific dtype and a specific
unit. On this machine the denominator must be patched before MFU means anything, and three
different denominators ship in the reference code.
→ training-telemetry-as-observability §4.3, scaling-laws-and-flops-budget

**MLA (multi-head latent attention)** — compress K and V into a shared low-rank latent per
token and store *that*, with a decoupled rotary key alongside `[C]` (2405.04434). *Bridge:*
storing a compressed representation instead of the raw record. *Break:* it is the *opposite*
arithmetic-intensity trade from GQA — it shrinks bytes while *adding* the decompression
FLOPs, so it pushes a workload back toward compute-bound. **Folklore warning:** `[M]` the
"93% KV reduction" does not hold in the HF reference implementation, which expands the latent
back to full per-head K and V *before* the cache write.
→ attention-variants-and-kv-cost §3.8, §5.3

**Mnemosyne** — the memory subsystem and **the lab's research contribution**: layered memory
and its management — KV cache, eviction/compression policies, tiering, prefix reuse,
attribution instrumentation. Bound by the dependency rule `mnemosyne → torch` only; it never
imports `proteus` or `themis`, enforced by `packages/mnemosyne/pyproject.toml` plus a lint
contract, `[M]` proved red-then-green 2026-07-26. The taxonomy derives *which guarantees it
does not owe*: no durability, no WAL, no checksums, no replication, no coherence protocol —
all inapplicable to a memo table. What it does owe is deterministic reconstruction
(conditionally — only if the prefill schedule is pinned) and **attribution of quality loss**.

**MoE (mixture of experts)** — replace one FFN with `E` independent FFNs and route each token
to `k` of them `[C]` (1701.06538, 2101.03961). *Bridge:* a load balancer in front of a
worker pool. *Break:* three ways — a mis-balanced expert does not just add latency, it
permanently changes what that expert *learns*; the routing decision is part of the model, not
the infrastructure; and bytes scale with *total* parameters while FLOPs scale with *active*
parameters, so the two parameter counts must be tracked separately in every cost model.
→ moe-and-routing

**MQAR (multi-query associative recall)** — the diagnostic task that isolates a state's
capacity limit from a training artifact: store `N` key→value bindings in context, then query
several of them `[C]` (2312.04927). The recall-versus-state-size Pareto frontier is the
capacity-planning statement of the same fact `[C]` (2402.18668). **The lab's actual open
question is the answer in bytes, not in architecture names.**
→ constant-state-memory, measuring-recall-and-memory §3.5

**muP (maximal update parameterization)** — a unit system for hyperparameters under which the
optimal learning rate transfers across model width `[C]` (2203.03466). *Bridge:* dimensional
analysis. *Break:* width transfer is well supported; *depth* transfer is contested and the
module says so. → depth-width-and-initialization §3.7, §3.8

### N

**Needle** — a distinguished, high-salience span planted in a long context that an eval asks
the model to retrieve. **The adverse-selection argument, which is the most important sentence
in the lab's eval story:** a needle is by construction a high-salience span that attracts
attention mass, which is exactly what heavy-hitter eviction *retains* — so
needle-in-a-haystack **structurally cannot fail** for H2O-style policies. Using NIAH to
validate an eviction policy is not a weak test, it is a test with the wrong sign.
→ memory-failure-modes §2.5, measuring-recall-and-memory §2.4

**NIAH (needle in a haystack)** — the eval built from the above. Saturated across models
`[C]` (2404.06654) and, per the adverse-selection argument, unable to falsify the mechanism
this lab studies.

**NoPE** — no positional encoding: rely on the causal mask alone to leak ordering
information. Stated honestly in the module — it works better than it has any right to at
short context and is not a solved answer at long. → positional-encoding §2.9

**Null distribution** — the run-to-run variation of your instrument when nothing you care
about has changed. **The number that must exist before any other number.** The trap:
re-running the same config with a fixed seed and deterministic kernels gives KL exactly zero,
and a degenerate null makes every difference look infinitely significant. So the null must
come from a *nuisance axis* — batch composition, dtype, attention backend, data seed. `[M]`
At the smallest interesting scale the floor is under water: on a 4-layer/4-head/d=256/T=1024
randomly-initialised transformer at three seeds, re-run and batch-composition nulls are
exactly zero, the bf16-vs-fp32 null is **6.6–8.4 × 10⁻⁶ nats**, and the median signal from
evicting one cache entry is **0.7–1.0 × 10⁻⁶ nats** — an **SNR of 0.1 at every seed**. Read
the caveat with it: random weights give near-uniform attention, so this is close to the
smallest obtainable signal; what it establishes is the floor.
→ measuring-memory §2.7

### O

**Oracle diff** — the lab's instrument, and its first Mnemosyne milestone: run the expensive
full-cache reference alongside a policy arm on the same prompts, and attribute the divergence
per token to individual cache decisions. Minimum viable shape: pin prompt, seed, batch
composition, dtype and attention backend; run full-cache → `p_full`; run policy → `p_pol`;
compute per-token `D_KL(p_full ‖ p_pol)` plus the evicted-mass and value-displacement scalars
for every eviction decision; **compute the null on the nuisance axes from the same prompts
before reporting anything**; then report median and p99 of signal, p99 of each null, their
ratio, the rank correlation between the per-decision scalars and the following KL, and the
effective `n` after prompt-level clustering. *Bridge:* shadow traffic against a reference
implementation. *Break:* the reference is not cheaper-and-approximate, it is the *expensive*
thing you were trying to avoid — which is why this is affordable at 300M against a `[M]`
≥62 GiB fast tier and unaffordable at 70B. **This is the argument for the lab's scale.**
→ measuring-memory §4.3, research/synthesis.md

### P

**Paged attention** — allocate the KV cache in fixed-size blocks through an indirection
table instead of one contiguous reservation per sequence `[C]` (2309.06180). *Bridge:*
virtual memory and page tables, and this is the *best* analogy in the whole track — same
motivation (internal fragmentation), same mechanism (indirection), same win (overcommit).
*Break:* four ways, and they are the module. No fault path (a miss restarts the request).
Eviction granularity is the *request*, not the page — and in SGLang it is a *subtree*. The
cache key is a chain over the whole prefix, not a content hash, so identical content at a
different offset is a different key. And discarding is always legal, because the bytes are
recomputable. → paged-attention-and-prefix-reuse

**Parametric memory** — the weights. Two capacity numbers that measure different quantities
and **must not be averaged**: ~2 bits of *useful knowledge* per parameter, and ~3.6 bits as
an upper bound on *unintended memorization* of random data `[C]` (2505.24832; the 2-bit
result's id is among the unverified 132). *Bridge:* a lossily-compressed read-only image — a
compiled binary, not a database. *Break:* a binary has addresses; you can patch one byte and
know what changed. Here the patch is a gradient step whose blast radius is the entire
parameter vector, and the ripple-effect and catastrophic-forgetting results are the empirical
measurement of that blast radius. → memory-taxonomy-for-engineers §2.2

**Perplexity** — `exp(cross-entropy in nats)`; read it as "the effective number of equally
likely next tokens." *Bridge:* a compression ratio, and that reading is exact. *Break:* it is
not comparable across tokenizers (use **bits per byte**), and it is a *mean* over positions —
which is why a policy can drop a specific instruction entirely while perplexity moves 1.03×
`[C]` (2606.09864, 2510.00231). A metric that averages cannot see a rare catastrophic
failure. → loss-and-optimization

**Policy null** — the calibration exercise that should precede reading any eviction paper:
run policies against data with *no structure whatsoever*, where no query-blind policy can
beat random because there is nothing to be right about — then discover that several of them
do, and work out why. → kv-eviction-policies, Exercise B

**Prefill** — consuming the whole prompt in one pass: large matmuls, compute-bound,
`O(T²)` in the attention term. Contrast **decode**. → kv-cache-mechanics §2.3

**Prefix cache** — reusing the KV bytes of a shared prompt prefix across requests.
*Bridge:* a shared read-only cache tier, and the economics are the same. *Break:* the
correctness hazards are not. The key is a *chain*; a change at position 0 invalidates
everything downstream; the match loop breaks at the first miss because a later hit is
impossible by construction. And across tenants the *existence of a hit* is observable in
latency — `[A]` high confidence this is a timing side channel on prompt contents, cheapest
test being a two-request timing probe against a warmed prefix. SGLang's `extra_key`
namespacing (`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355`) is the
mitigation shape, and it is exactly an ASID. Note precisely what changed: the hazard is not
in the *bytes* (still reconstructible, still information-free) but in the *metadata*.
→ paged-attention-and-prefix-reuse, memory-taxonomy-for-engineers §4.3

**Prefix hash chain** — the cache key: `h_j = H(h_{j−1}, tokens[j·P:(j+1)·P], extra_keys)`
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:596`). Strictly prefix-ordered by construction.
*Bridge:* a Merkle chain, and the integrity intuition transfers. *Break:* here the chain is
not for integrity, it is because a KV entry is only valid *at its position under its
history* — so this is not a design choice that could be swapped for content addressing.
Its direct consequence for agent memory: **a mid-history edit costs a re-prefill of the
entire tail.** → agent-memory-in-practice §3.2

**Prematurity penalty** — the cost of choosing the retained set using a query issued *before*
the one that will read it. This is SnapKV's core assumption, stated as a measurable quantity.
→ kv-eviction-policies, Exercise C

**Pre-registration** — committing the G2 hypothesis card (HYPOTHESIS / FOR / BECAUSE /
MEASURED BY / SUCCESS / KILL / COST / RISKIEST) to git *before* the run. Moving a SUCCESS or
KILL threshold after seeing results is a change of standard and must be called out as one.
*Bridge:* a change-control record. *Break:* the artifact being frozen is not the change, it
is the *decision rule* — and the value is entirely in its being unavailable for revision.

**Procedural memory** — reusable how-to distilled from trajectories, in the agent four-box
taxonomy. **Not monotone:** adding skills can make an agent worse, and a model can be a
strong skill *extractor* and a weak skill *consumer*. → agent-memory-in-practice §2.6

**Proteus** — the experimental model architecture: decoder, attention variants, MoE,
positional schemes, and the full config surface. Depends on `torch` and `mnemosyne`, never
the reverse.

### Q

**QK-norm** — normalizing queries and keys before the dot product, which bounds the attention
logit by `√d_head` instead of leaving it unbounded `[C]` (2302.05442). *Bridge:* clamping an
input to keep a downstream stage in range. *Break:* the open question is whether it buys
*loss* or buys *learning rate* — a stability fix and a quality win look identical on a single
curve, and separating them takes a two-axis experiment.
→ normalization-and-activations §2.4

**Quantization** — mapping a real value to a low-bit integer plus a shared scale, and back.
Two decisions people state (bit width, symmetric vs asymmetric) and two they usually do not
(group size, and what the scale's own dtype costs). The group is the only real lever.
*Bridge:* lossy compression with a per-block dictionary. *Break:* weight error is not the
objective — *output* error is, and the two are related through the activation second-moment
matrix, which is why data-aware methods exist at all. → quantization

### R

**RadixAttention** — SGLang's prefix cache as a radix tree over token sequences, with
eviction that considers only *leaves* from an incrementally maintained set
(`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565`). *Bridge:* an LRU.
*Break:* it is **topological, not recency-ordered** — a hot child keeps a cold parent
resident indefinitely, and the topology is prefix order. `[C]` (2312.07104)

**RAG (retrieval-augmented generation)** — retrieve passages from an external index and
prepend them to the prompt. *Bridge:* a read-through cache in front of a document store, and
this is **the one tier where a systems engineer's instincts transfer wholesale** —
transactional, addressable, deletable, with a real backing corpus. *Break:* the retrieved
text becomes *tokens*, so RAG does not add memory, it converts storage into context-window
occupancy at a fixed exchange rate. And there is no miss: the store returns `k` results
whether or not any is relevant, so the failure mode is silent low relevance, not a signalled
absence. **RAG versus long context is left contested** in this repo — the ordering is task-
and scale-dependent and no section may assert a winner.
→ memory-taxonomy-for-engineers §3.6, §5.4

**Rate–distortion view** — the Jul 2026 proposal that KV eviction, prompt pruning, bounded
recurrent state and agent-memory consolidation are *one* problem: what context-derived
information to keep, at what fidelity, under a budget `[C]` (2607.08032). Weeks old; treat
as an open proposal. The cheapest test of whether it has teeth is whether an H2O-style
attention-mass rule transfers to session-store consolidation.

**Reconstructibility** — **the axis that actually partitions the five things called memory.**
Not capacity, not speed, not latency. Ask one question of each tier: *if I delete this, where
do I get it back from?* KV cache → the token sequence, exactly (cost: FLOPs, never
information). Recurrent state → the token sequence (cost: FLOPs). Retrieval index → the
corpus (cost: embedding FLOPs). Weights → the training data plus the training run (cost:
very large, and usually the data is gone). **Session store → nowhere (cost: the information
itself).** Everything Mnemosyne does and does not owe falls out of that table.
→ memory-taxonomy-for-engineers §2.4

**Recurrent state** — one fixed-size tensor per layer per sequence carrying a lossy running
summary of the whole prefix; `state_bytes = L · h · d_hs · d_s · p_s`, with no token term.
*Bridge:* a fixed-size rolling aggregate versus an unbounded log — and it is a fully
associative cache with a global TTL tick and a compare-and-swap. *Break:* it is emphatically
**not a write-ahead log**. A KV cache is an append-only exact record you can re-scan; this is
a destructive rolling aggregate with **no replay**. Token 5's contribution is not recoverable
at token 5000 because it was *destroyed*, not relocated, and there is no lower tier to page
it back from. Its failure mode is **interference, not a miss**.
→ constant-state-memory

**Renormalisation** — after eviction the softmax is recomputed over survivors only, so every
retained weight is inflated by `1/(1 − m_E)` where `m_E` is the evicted attention mass.
**Not a design choice** — it is what softmax does. It means eviction damage decomposes into
*information loss* (the missing `Σ a_j v_j`) and *renormalisation* (the distortion of what
remains), and those two are separately measurable, which is what makes the decomposition an
instrument rather than a description. → kv-eviction-policies §3.2, memory-failure-modes §3.2

**Residency** — see **KV cache, the three budgets**. For a hybrid stack,
`R(T) = c · (L_g · T + L_w · min(T, w))` — a growing term plus a fixed term. `[M]` Laguna-S:
48 KiB/token growing, 72 MiB fixed, with a **byte-parity point at T = L_w·w/L_g = 1536
tokens**. Below 1536 the windowed layers hold *more* bytes than the global ones — worth
internalising, because a great deal of small-scale experimentation happens below it, where
the hybrid saves nothing and its constant term dominates. → kv-cache-mechanics §3.2

**Residual stream** — the running sum that each block reads from and adds back into.
*Bridge:* a shared bus, with a norm as gain control on each tap. *Break:* under pre-norm the
stream's magnitude grows like `√l` with depth, so later layers write into a progressively
louder bus and go quiet — a bus with no arbitration and an ever-rising noise floor.
→ normalization-and-activations §2.1, transformer-forward-pass-by-hand

**Resume equivalence** — the testable property that a run resumed from a checkpoint is
indistinguishable from one that never stopped. *Bridge:* DR failover verification. *Break:*
"indistinguishable" needs a definition and a noise floor — bit-exactness is often
unachievable and usually not required, so the harness must measure the same-run noise floor
*first* or it will report every resume as a failure. → checkpointing-and-resumption §3.4

**Retrieval head** — a small subset of attention heads that implement copy-from-context;
masking them selectively destroys long-context retrieval while leaving general fluency
intact, which makes retrieval-head masking one of the six calibration faults. The arXiv id
used in the modules for this result is among the 132 unverified; treat the mechanism as
well-attested and the specific numbers as unchecked.
→ measuring-recall-and-memory, long-context-and-effective-context

**Retrieval index** — see **RAG**.

**Ridge point** — the arithmetic intensity at which a machine crosses from bandwidth-bound to
compute-bound: `peak FLOP/s ÷ peak bytes/s`. For this box, `20.9e12 / 200e9 ≈ 105 FLOP/byte`,
computed from two `[M]` numbers (`scripts/benchmark_gemm.py`; the fast-tier sweep). Attention
decode delivers 1 (MHA, bf16) to 6 (Laguna's global-layer GQA), so **attention decode on this
machine runs at roughly 1–6% of peak arithmetic** — a pure bandwidth problem, and the single
mental model the whole memory track depends on. **Caveat, stated because `research/synthesis.md`
states it:** this ridge is a ratio of two single-run numbers of different kinds, *neither of
them an attention kernel*, and it does not currently have its own row in `ASSUMPTIONS.md`.
→ attention-variants-and-kv-cost §3.7

**RLVR** — reinforcement learning from verifiable rewards: replace the learned reward model
with a program that checks the answer. `[C]` At our scale it can actively regress — a 135M
single-GPU replication saw GSM8K exact match *fall*.
→ supervised-and-preference-finetuning §2.5

**RoPE (rotary position embedding)** — encode position by *rotating* the query and key
vectors by an angle proportional to position, so the attention logit depends only on the
*relative* offset `[C]` (2104.09864). *Bridge:* phase encoding. *Break:* because RoPE is
applied *before* the cache write, a cached key is `RoPE(RMSNorm(k))` — position is baked into
the bytes. "Just re-position the cache" is not a thing, and that single fact constrains every
KV-reuse design in this repo
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` region).
→ positional-encoding

**RMSNorm** — LayerNorm with the mean-subtraction and the bias deleted: divide by the root
mean square, multiply by a learned gain `[C]` (1910.07467). Nobody missed what was deleted.
→ normalization-and-activations

**RPO / RTO (in training)** — Recovery Point Objective here is **seconds of GPU compute
lost**, `E[lost work] = (k/2)·t_step` for a checkpoint every `k` steps: nothing is
unrecoverable, you are buying back electricity and wall-clock. RTO decomposes as
`T_detect + T_restart + T_load + T_warmup`, and `T_load` — the term the literature optimizes
— is the smallest one here (`[M]` 0.066 s median to load a 403 MB checkpoint, against 0.387 s
to write it). See **detection gap** for the term that dominates.
→ checkpointing-and-resumption §3.2

**RULER** — a generator over 13 synthetic long-context task families with controlled length
and a threshold-based effective-context report `[C]` (2404.06654). Its durable contribution
is the *generator*, not the single number. Two cautions the modules add: the chance level is
not zero for every family, and the report does not carry its own variance.
→ measuring-recall-and-memory §3.6

### S

**Salience** — a target span's mean attention mass in units of the background mean. The
governing variable in the survival law, and the quantity that makes a needle adversely
selected. → measuring-recall-and-memory §3.1

**Scaling law** — an empirical power-law relationship between loss and compute/parameters/
data `[C]` (2001.08361, 2203.15556). *Bridge:* a capacity-planning curve. *Break:* the curve
is fitted, its fitting procedure has a documented bias, and it predicts *loss*, which is not
the thing you ship. → scaling-laws-and-flops-budget

**Seed wall** — the point at which adding eval items stops reducing your error bar because
the dominant variance component is *across training seeds*, not across items. Governed by the
intraclass correlation `ICC = σ²_seed / (σ²_seed + σ²_item)`. *Bridge:* you cannot fix a
systematic error by sampling harder. *Break:* the fix here is expensive in a specific way —
more seeds means more *training runs*, which is the budget line the whole schedule is built
around. → building-an-eval-you-can-trust §3.2

**Semantic memory** — facts abstracted from episodes, in the agent four-box taxonomy; the box
whose update mechanics should *not* be the same as episodic decay, and usually are.
→ agent-memory-in-practice §2.6

**Session store** — what the agent decided to write down: preferences, extracted facts,
trajectories, distilled procedures. **The only one of the five with no reconstruction path**,
and therefore the only one that needs real storage-system machinery — durability, provenance,
integrity, access control, an eviction policy that can be *wrong* in the correctness sense.
*Bridge:* a write-ahead store with schema drift and no compaction policy. *Break:* the writes
originate from model output and from *untrusted input*, so the write path is an attack
surface, and correctness is a property of the state *trajectory* rather than of any record —
which is exactly why record-level database operations do not suffice.
**Not Mnemosyne's.** If a session store ever lands in this repo it gets its own package name,
because a subsystem that owes durability and a subsystem that owes none should not share an
interface. → agent-memory-in-practice, memory-taxonomy-for-engineers §4.1

**SFT (supervised fine-tuning)** — cross-entropy on curated demonstrations, plus a loss mask
that zeroes the prompt tokens. *Bridge:* an access-control list on the gradient. *Break:*
the masked fraction is a real budget line — you pay full forward-pass cost for tokens that
contribute no gradient, and most people never measure their own mask fraction.
→ supervised-and-preference-finetuning §3.1

**Sliding-window attention (SWA)** — each query attends only to the last `w` positions.
**The one lossless eviction in the entire taxonomy:** an out-of-window token is
architecturally *unreadable* — the mask forbids it — so discarding it needs no repair and
costs nothing. **A window is a proof; an eviction policy is a bet.**
→ attention-variants-and-kv-cost §2.3

**SnapKV** — select the retained set using the attention pattern of an *observation window*
of recent queries, then freeze it. Its assumption is that the next query resembles the last
few; see **prematurity penalty** for the measurable version.

**Softmax** — exponentiate scores and divide by their sum, producing weights that sum to
exactly 1. The constraint "sum to 1" is the origin of both **attention sinks** and
**renormalisation** — two apparently unrelated phenomena that are the same arithmetic fact
seen from two sides. → transformer-forward-pass-by-hand

**Speculative decoding** — draft `K` tokens cheaply, verify them in one target forward pass,
commit the accepted prefix plus one. Always commits `a+1` tokens. *Bridge:* read-ahead
prefetch with validation. *Break:* the "prefetch" is a *distributional* claim, and the
verification rule preserves the target's output distribution exactly — so speculation is
lossless in a sense no prefetcher is, while being *useless* whenever `r(K)` departs from 1.
Also: a bad drafter is worse than none. And for this lab specifically, **speculation is a
confound for every eviction experiment**, controlled by one config field.
→ speculative-decoding-and-serving

**Spike detector** — the standard training-telemetry guard: flag a metric more than `z`
standard deviations off a trailing window of `n` steps (128 in both reference
implementations). Has an operating characteristic — a false-positive rate under drift — and
nobody reports it. → training-telemetry-as-observability §3.5

**Storage tier** — the thing a KV cache is **not**. Contrast **memo table**. The one-line
test: *what information is lost if this fails?* If the answer is "none, it costs FLOPs," you
are looking at a memo table, and importing durability machinery is over-engineering.

**Survival law** — the closed form for whether a target span survives a top-`b` retention
rule when background attention masses are `Exponential(1)`: the threshold is the `(1−b)`
quantile, `q(b) = −ln b`, and modelling the target's own mass as a random variable turns the
naive step function (`s > −ln b`) into a smooth curve. → measuring-recall-and-memory §3.2

**SwiGLU** — a gated feed-forward activation: one linear branch gates another elementwise
`[C]` (2002.05202). *Bridge:* a valve rather than a switch. *Break:* it costs a third weight
matrix, so parameter parity against a plain MLP requires shrinking the hidden width by 2/3 —
and "matched parameters" comparisons that skip this are not matched.
→ normalization-and-activations

### T

**Tensor parallel** — shard individual weight matrices across ranks and reduce the partial
results. The axis whose communication volume scales with *tokens* rather than with
parameters, which is what makes it latency-sensitive and intra-node.
→ distributed-training-strategies §3.4

**Themis** — the ablation rig: pre-registration, matched budgets, seeds, execution,
aggregation, reporting. Depends on `torch`, `proteus`, `mnemosyne`.

**Tier-ratio experiment** — the one experiment no discrete-GPU lab can run: sweep the
fast/slow memory bandwidth ratio, which on this platform is a **BIOS setting** rather than a
bus, and see whether the eviction-versus-retention boundary moves with it. The stakes: the
offload/CXL tiering literature is designed around a GPU-HBM-to-host-DRAM ratio of order
10–50× across PCIe; if the boundary flips at a ratio of 2–3×, **a body of published design
guidance is a statement about interconnects rather than about language models.** Note the
prediction, which is what makes it pre-registerable: `memory-taxonomy-for-engineers` §4.4
computes that at our scale the refetch/recompute boundary sits below 2 GB/s at every context
length we can reach, so **a null is the prediction**, and any observed flip is either a real
effect or a harness bug — both worth finding. → research/synthesis.md

**Top-k (routing / retrieval / eviction)** — keeping the `k` highest-scoring items. Its
discontinuity is a live hazard here: under numerical error two nearly-tied scores can swap,
and one swapped eviction decision propagates. → determinism-and-reproducibility §3.7

**TOVA / KeyDiff / ChunkKV / PyramidKV / StreamingLLM** — the eviction-policy field, of which
roughly thirty exist with **no dominance result**. Reference points: StreamingLLM pins the
attention sinks plus a recent window `[C]` (2309.17453); PyramidKV allocates a per-layer
budget and, by the paper's own account, degenerates to SnapKV at aggressive ratios. `[C]` A
Mar 2026 survey concludes no single method dominates (2603.20397). **This is why the lab
builds an instrument rather than a 31st policy.** → kv-eviction-policies §4

### U

**UMA carve-out** — the BIOS "UMA FB Size" field that dedicates part of the 128 GB unified
pool to the iGPU. `[M]` It controls the **fast tier**, not the allocation ceiling: raising it
from 16 GiB to 96 GB moved the fast tier from 30 GiB to ≥62 GiB and the *reported* pool from
82.99 to 107.87 GiB, while allocation-only probes returned ≥100 GiB either way because the
driver oversubscribes into system RAM. Two numbers, two different questions — do not conflate
"what allocates" with "what runs at full bandwidth."
(`ASSUMPTIONS.md → gpu-fast-tier-size`, `hardware-capacity-ceiling`)

### V

**VJP (vector–Jacobian product)** — what reverse-mode autodiff actually computes at each
node: not the Jacobian, but its product with the incoming gradient. *Bridge:* a fold over a
DAG. *Break:* the intermediate values needed for the fold are the *forward* activations, so
the graph's memory cost is a property of the forward pass, not of the backward one — which is
why activation checkpointing (recompute instead of store) is a real lever.
→ tensors-and-autograd §2.4

### W

**Weights** — see **parametric memory**.

**Winner's curse** — the systematic overestimation of an effect size conditional on having
found it significant. At low statistical power the exaggeration (Type M error) is large and
the sign can be wrong (Type S error), which is the arithmetic that kills most small-scale
eval plans. → building-an-eval-you-can-trust §3.4

**Working memory** — **BANNED.** Means the resident KV cache in the serving literature and
the text currently in the prompt in the agent literature. They are related — one is derived
from the other — but they have different sizes, different lifetimes and different failure
modes. Say `kv-cache` or `prompt occupancy`.

**Working set** — the subset of data actively in use. **The analogy that fails hardest in
this field.** A working set implies a hot subset; every decode step reads **100%** of the
resident KV, per head, per layer. There is no temporal locality to exploit and no hit rate to
improve. Everything you know about `LRU`, `ARC`, ghost lists and hit-rate curves derives from
an assumption that does not hold here. → kv-eviction-policies §2.2

### Y

**YaRN** — a RoPE context-extension scheme: scale the low-frequency dials, leave the high
ones, and apply an `attention_factor` temperature to the logit. `[M]` Laguna's
`attention_factor` matches YaRN's default temperature formula to the last digit `[C]`
(2309.00071) — i.e. it was inherited, not tuned. *Bridge:* rescaling a clock domain.
*Break:* the temperature is a *multiplicative* fix for what the dilution theorem shows is an
*additive* deficit in nats, so it cannot be the whole answer.
→ positional-encoding §2.7, long-context-and-effective-context §3.3

### Z

**Zoology** — the harness that introduced MQAR as the capacity diagnostic for
fixed-state architectures `[C]` (2312.04927). → constant-state-memory

---

## Lab process vocabulary

These are not subject matter; they are how this repo works. Defined here because they appear
in module prose without explanation.

| Term | Meaning |
|---|---|
| **G0 … G5** | the operating loop in `OPERATING_INSTRUCTIONS.md`: Discovery → … → evidence-tagging. Each gate has a written checkable artifact. Skips are logged, never silent. |
| **`[M]` / `[C]` / `[A]`** | measured (run id + seed count) / cited (arXiv id or URL + date) / assumed (confidence + cheapest test). **Never state an `[A]` in the register of an `[M]`.** |
| **Record / Register / Documentation** | the three document classes. Records (ADRs, notebook entries) are immutable — corrections are *appended*. Registers (`ASSUMPTIONS.md`, `LOG.md`, the ADR and notebook indexes) append rows and update status; rows are never deleted. Documentation (this file, `curriculum/`, `docs/`) is mutable and you are obligated to keep it accurate. |
| **ADR** | `docs/adr/<slug>.md`, cited by slug. Frozen the moment it reads `Accepted`; the single permitted write is appending a `Superseded by` line to the Status block. |
| **Hypothesis card** | HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST / RISKIEST, committed before the run. |
| **Naming rule** | every identifier states what the thing is or does. No ordering encoded in identifiers, anywhere. Ordering lives in the folder's `README.md`. Versioning is semver in frontmatter or in git tags, never in a filename. |
| **Boundary rule** | `mnemosyne → torch`; `proteus → torch, mnemosyne`; `themis → torch, proteus, mnemosyne`. Enforced by the dependency graph and a lint contract, not by discipline. |
| **Folklore** | a claim repeated without controlled evidence, tracked explicitly in `research/synthesis.md`. Current entries: "3:1 is the right hybrid ratio"; "the model has a 1M context"; "MLA gives 93% KV reduction"; "the KV cache is a storage tier"; "PyramidKV's per-layer budget is what buys the gain". |
| **Contested** | a live dispute the repo refuses to resolve, listed as such so no section quietly picks a winner. |
| **UNPROVEN DEMAND** | a failure-register claim resting only on a Future Work mention rather than a demonstrated failure. |

---

## What this glossary does not settle

Three things, stated so they are not mistaken for settled.

**The `c` collision is unresolved, not merely documented.** Two modules define `c` as
different quantities differing by a factor of `L`, and both call it their working unit. The
notation index above is a workaround; the fix is a repo-wide convention with one of them
renamed, which is a change to written modules and therefore not this file's to make.

**Ten of the thirty-one modules have never been citation-checked.**
`curriculum/citation-verification.json` covers 21 files — all of Tracks A, B and C. Every
Track D, E and F module (`quantization`, `determinism-and-reproducibility`,
`distributed-training-strategies`, `speculative-decoding-and-serving`,
`training-telemetry-as-observability`, `building-an-eval-you-can-trust`,
`checkpointing-and-resumption`, `running-laguna-locally`, `measuring-recall-and-memory`,
`supervised-and-preference-finetuning`) is unverified. Where those modules are the sole
source of a definition above, this file relies on internal consistency, not on a resolved id.

**Nothing measured on this machine is admissible yet.** The Hardware Validation Gate has not
run and `bf16-numerics-unproven` is `untested`. Every `[M]` in this glossary is an
instrument-shakedown reading, correctly tagged and not yet evidence by house standard. That
is not a caveat to be read past; it is the current state of the lab.
