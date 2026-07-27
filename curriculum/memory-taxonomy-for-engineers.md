---
title: Five things called memory — the reconstructibility axis, and where the storage hierarchy stops being a useful lie
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost (Track B), tensors-and-autograd (Track A)
mirrors: research/memory/memory-taxonomy.md
difficulty: 2/5 conceptually, 4/5 in the places your existing instincts fire wrong
time: 2–3 h reading; 2–3 h for the three exercises
bridges_into: kv-cache-mechanics, kv-serving-hierarchy, constant-state-memory, agent-memory-systems
---

# Five things called memory

**Difficulty and time, honestly.** There is almost no new mathematics here — one product
you already derived in `attention-variants-and-kv-cost.md`, two ratios, and one inequality
where a variable cancels. The difficulty is entirely in the *unlearning*, and it is
concentrated in one place: thirty years of storage-hierarchy reflexes will fire correctly
on two of the five things in this module, misleadingly on two more, and catastrophically
on one. Budget 2–3 hours for sections 1–5, and protect the time for Exercise C — it is
twenty lines of code and it settles a question the literature states but does not
demonstrate.

This module teaches what `research/memory/memory-taxonomy.md` surveys. Read that note
first if you have not; this file assumes it and does not repeat its citation apparatus.
Where I refine it — §2.5, §3.7 and §3.8 — I say so and show the working.

---

## 1. What this module settles

**One:** the word "memory" names five mechanically unrelated systems in the LLM
literature, two research communities apply incompatible vocabularies to *the same bytes*,
and until you can name which of the five a paper is talking about you cannot read the
field — so this module fixes the vocabulary and gives each of the five a definition
grounded in what a read and a write physically do. **Two:** the property that actually
partitions them is not capacity, not speed, and not latency, but **reconstructibility** —
*if I delete this, where do I get it back from?* — and that single question sorts the five
into **two** memo tables recoverable by recompute, one derived index rebuildable from a
corpus, one parametric store recoverable only by repeating the training run, and exactly
**one** authoritative tier whose loss is loss of information. **Three:**
the storage-hierarchy analogy, which is the most natural one for you to reach for, breaks
in a specific and load-bearing way — two of the five have **no miss path at all**, and I
show by measurement that for the KV cache the miss path could not be implemented at page
granularity even if someone wanted it, because the repair unit is a suffix, not a block.

That third point is the module. Everything Mnemosyne will or will not owe as a subsystem
falls out of it.

---

## 2. Theory in plain language

### 2.1 Why the vocabulary is broken, and why that is your problem

Two communities converged on the word "memory" from opposite directions and never
reconciled.

The **serving community** came from systems. Its objects are tensors with a lifetime and
an owner. Its nouns are blocks, pages, prefixes, eviction, hit rate, tiering. The 2026
survey that consolidates thirty-plus KV systems calls the thing a *tensor buffer* in its
own title `[C]` (`2607.02574`, Jun 2026).

The **agent-memory community** came from cognitive science. Its objects are episodic,
semantic, procedural, working, long-term. Its verbs are consolidate, forget, reflect,
retrieve. MemOS partitions memory into *parametric / activation / plaintext* and puts the
KV cache in the "activation" tier as a first-class scheduled resource `[C]`
(`2507.03724`, 2025).

Those are the same bytes. One camp calls them a buffer with an owner; the other calls them
a memory tier with a scheduler. They cross-cite rarely.

Worse than two names for one thing is one name for two things. **"Working memory"** in the
agent literature means the text currently in the prompt; in the serving literature the
nearest analogue is the resident KV cache, which is *derived from* that text but is a
different object with a different size, a different lifetime and a different failure mode.
**"Long-term memory"** means the weights to one camp and a vector store to the other. This
is not a hypothetical hazard: a Feb 2026 paper titles a persistent quantized KV cache
"Agent Memory Below the Prompt" `[C]` (`2603.04428`).

A July 2026 paper is the first serious attempt to put all of it under one frame —
rate-distortion, where every compaction decision from KV eviction to prompt pruning to
bounded recurrent state to agent-memory consolidation is one choice about what
context-derived information to keep at what fidelity under a budget `[C]` (`2607.08032`,
Jul 2026). It is three weeks old and has settled nothing. Treat the unification as an open
proposal, not the field's position. A recent scan of the 2026 taxonomy literature turns up
several more consolidations along the same lines — token / intermediate-representation /
parameter is a recurring trichotomy — which tells you the field feels the problem and has
not agreed on the fix.

**Why this is your problem and not a philosophy seminar.** You are about to build a
subsystem named Mnemosyne whose scope is a *contribution claim*. If its scope is "memory,"
it has no scope. If its scope is "the KV cache and the policies over it," it has a
testable boundary, a definable interface, and an acceptance test. The vocabulary work is
scope work.

### 2.2 The five, mechanically

Each of the five gets the same four questions: what does it hold, what does a write
physically do, what does a read physically do, and what can it structurally never do.

**Weights (parametric store).** Holds everything absorbed during training, in
superposition, with no index and no schema. There is no row for "Paris is the capital of
France"; there is a distributed pattern that makes that continuation likely. A write is a
gradient step: **non-local** (one fact perturbs parameters everywhere), non-atomic, no
transaction, no rollback, no read-your-writes. A read is implicit and non-enumerable — you
cannot list what a model knows. What it cannot do: no addressable write at inference time
(targeted editing exists — ROME `[C]` `2202.05262`, MEMIT `[C]` `2210.07229` — but edits
fail to propagate to logical consequences `[C]` (`2307.12976`) and applied sequentially at
scale produce gradual then catastrophic forgetting `[C]` (`2401.07453`)); no delete; no
provenance.

**Recurrent state (SSM / linear attention).** Holds one fixed-size tensor per layer per
sequence: a lossy running summary of everything seen so far. A write is a destructive
read-modify-write in place — decay the whole state by a scalar, then add a rank-1 update.
A read is one small matrix read. What it cannot do: recall a specific past token exactly
(the carry was *destroyed*, not relocated — there is no lower tier to page it back from);
forget selectively in the shipped configuration (the decay is a single scalar over the
whole state; the *delta* term is the targeted erase, not the gate); grow at inference time.
Its failure mode is **interference, not a miss** — a similar key partially clobbers a
neighbour, and nothing detects that this happened.

**KV cache (attention working set).** Holds, for every token in every layer that keeps a
cache, one key vector and one value vector, stored exactly as computed. Nothing is
summarized, nothing is merged. A write is an append, one entry per token per layer, done
by the forward pass. A read is **not a lookup**: there is no address and no key-equality
test. Every decode step performs a softmax-weighted sum over the entire resident set, per
head, per layer. What it cannot do: hold anything not derivable from the tokens; fault;
match associatively; stop growing.

**Retrieval index (RAG).** Holds a document corpus plus a derived index, usually dense
embeddings, often hybridized with BM25 `[C]` (`2005.11401`, 2020). A write is an external,
offline, transactional operation on an ordinary database. A read is top-k by vector
similarity, and then — the part that dominates in practice — the retrieved passages are
**prepended as tokens**. What it cannot do: iterate on a query the model has not yet
formulated; guarantee the retrieved text is used or that it beats a conflicting parametric
belief; represent what was never written down.

**Session store (agent memory).** Holds whatever the agent decided to write down:
preferences, extracted facts, trajectories, distilled procedures. MemGPT set the framing
everything inherits `[C]` (`2310.08560`); A-MEM offers the contrasting self-organizing
design `[C]` (`2502.12110`). A write is durable, cross-session, and — uniquely —
**originates from model output and from untrusted input**. A read is retrieve-then-inject,
same as RAG. What it cannot do: be recomputed; be verified by the system that wrote it;
localize correctness to a record.

### 2.3 The systems bridge, stated in full before I break it

Here is the analogy you will reach for, and it is worth stating in its strongest form
because the strong form is what makes the breaks legible.

> **The bridge.** Five tiers of one hierarchy. Weights are the ROM image — slow to build,
> read-only in production, the widest and least specific store. The retrieval index is
> cold object storage with a real query interface. The session store is the warm durable
> tier — small, authoritative, written by the application. The KV cache is the page cache:
> hot, per-request, evictable. Recurrent state is a small fixed register file. Data
> promotes upward on access and demotes downward under pressure; misses fault to the tier
> below; the whole thing is a working-set problem with an eviction policy.

Every clause of that paragraph is wrong in an instructive way.

**Break 1 — there is no promotion or demotion between any pair.** Not "it is not
implemented yet." There is no shipped system in which a KV entry is demoted to a cheaper
*representation* and faulted back transparently. Offload systems come close, and Mooncake
is the honest exemplar — DRAM over RDMA, local NVMe, shared NVMe, CXL, with a fixed
preference ladder in `SelectBestReplica` and watermark-triggered writeback — but they move
**bytes without changing representation**. That is relocation, not demotion. The
hierarchy framing is a design aspiration (MemOS's scheduler `[C]` `2507.03724`), not a
description of anything running today. `[A]` high confidence; the cheapest falsifier is a
shipped system that changes an entry's representation on demotion and restores it on
access.

**Break 2 — two of the five have no miss path at all.** In vLLM, when the allocator cannot
satisfy a request, `allocate_slots` returns `None`. That is not a fault to be serviced. It
is an admission rejection that **preempts the entire request** — eviction granularity is
the sequence, not the page. For recurrent state it is worse: there is no entry to miss on,
because there is no slot per token. A page fault is a *signal*; these systems emit no
signal, because a signal requires a distinction between "present" and "absent," and
neither of them has one.

**Break 3 — a hit removes compute, never I/O, and never supplies information.** In a page
cache a hit means you did not have to go to disk *for data you did not have*. In a KV cache
a hit means you did not have to recompute something you could have recomputed. The bytes
carry no information the tokens did not already carry. This is why Mooncake's
`offload_force_evict` may throw bytes away rather than block on writeback — a tradeoff no
real storage tier is allowed to make, and it is legal here precisely because there is no
data to lose.

**Break 4 — "dirty" has no referent.** A dirty page is one whose contents differ from the
backing store. There is no backing store. Nothing can be dirty; nothing needs flushing;
coherence is not a property the system can have or lack. Delete every KV byte in the
machine and you have lost exactly zero bits of information and some number of FLOPs.

**Break 5 — the working-set assumption fails.** A working set implies a hot subset. Every
decode step reads 100% of the resident KV, per head, per layer. There is no temporal
locality to exploit and no hit rate to improve. You covered this in
`attention-variants-and-kv-cost.md` §2.1; it is worth re-stating here because it is what
makes "eviction policy" mean something completely different in this field: not *which
entries do we keep resident*, but *which entries do we delete forever and hope we did not
need*.

### 2.4 The property that actually partitions them

Ask one question of each of the five: **if I delete this, where do I get it back from?**

| Tier | Reconstruct from | Cost of loss |
|---|---|---|
| KV cache | the token sequence | FLOPs. Never information. |
| Recurrent state | the token sequence | FLOPs. Never information. |
| Retrieval index | the corpus | embedding FLOPs |
| Weights | the training data + the training run | very large, and usually the data is gone |
| **Session store** | **nowhere** | **the information itself** |

The KV cache and recurrent state are **memo tables**. A memo table is not a cache: a cache
stores a copy of something that exists elsewhere; a memo table stores the *result of a
pure function* you would rather not call again. `f(tokens, weights)` is deterministic in
exact arithmetic, so the entry is not data, it is *deferred work*. Every reflex you own —
durability, writeback, coherence, checksums, replication, DR — is inapplicable, because
there is nothing to be coherent *with*.

Three consequences, and they are the reason the table is worth drawing:

1. **Eviction policy on a memo table is a compute/quality tradeoff, not a data-safety
   tradeoff.** This is exactly why H2O-style eviction `[C]` (`2306.14048`) and attention-sink
   pinning `[C]` (`2309.17453`) are respectable engineering rather than reckless. The worst
   case is a worse answer, never a lost record.
2. **Only the session store needs the machinery of a real storage system** — durability,
   provenance, integrity, access control, an eviction policy that can be *wrong* in the
   correctness sense. Applying that machinery to the KV cache is over-engineering. Omitting
   it from the session store is the bug the 2026 security literature is documenting `[C]`
   (`2604.16548`, `2606.04329`, `2407.12784`).
3. **The five are not tiers of one hierarchy.** See breaks 1 and 2 above.

### 2.5 A refinement to the note, stated explicitly

`research/memory/memory-taxonomy.md` §8 says "two of the five have no miss path at all."
Strictly, **three** of the five have no fault handler: weights have no miss path either —
you cannot fault in a fact.

I do not think the note is wrong; I think it is answering a narrower question and it is the
right narrower question. "Is there a miss path?" is only well-formed for a tier that has
*addressable entries that can be individually absent*. The KV cache has them (a block, a
token's K and V) and refuses to fault on them. Recurrent state does not have them at all,
which is a different and stronger statement. The weights have neither entries nor lookups,
so "miss" is not a well-formed question about them; they are outside the frame rather than
a third instance inside it.

The count that matters for design is therefore: **two tiers have entries and no fault
handler, one tier has no entries, and two tiers have both entries and a real backing
store.** I will use the note's phrasing and this refinement interchangeably.

---

## 3. The math that actually matters

Five cost models, one crossover, one inequality where a variable cancels, and one
measurement that puts a caveat on the word "exactly."

### 3.1 Symbols

| Symbol | Reads as | Source |
|---|---|---|
| `T` | tokens resident in context | runtime |
| `L` | layers in the stack | config |
| `n_kv` | key/value heads per layer | `num_key_value_heads` |
| `n_q` | query heads per layer | `num_attention_heads` |
| `d_h` | width of one attention head | `head_dim` |
| `p` | bytes per stored element (2 for bf16, 4 for fp32, 1 for fp8) | dtype |
| `P` | parameter count (active params, for an MoE) | — |
| `b_w` | bytes per weight element | dtype |
| `h` | SSM heads per layer | `nheads` |
| `d_hs` | SSM head dimension | `headdim` (Mamba-2 default 64) |
| `d_s` | state-space dimension | `d_state` (Mamba-2 default 128) |
| `p_s` | bytes per state element (the boundary pass runs fp32 regardless of model dtype) | — |
| `c` | **bytes of KV per token, whole stack** — the single most useful derived constant | derived |
| `F` | achieved FLOP/s for prefill | measured |
| `BW` | bandwidth of a candidate slow tier | measured |
| `k`, `m` | retrieved passages, tokens per passage | runtime |

### 3.2 The five capacity laws, side by side

```
weights_bytes        =  P × b_w                                   independent of T
weights_information  ≈  P × (2 … 3.6) bits                        independent of T
state_bytes          =  L × h × d_hs × d_s × p_s                  independent of T
kv_bytes(T)          =  2 × L × n_kv × d_h × p × T   ≡  c × T      LINEAR in T
index_bytes          =  N_chunks × d_embed × p  (+ the corpus)     independent of T
session_bytes        =  O(interactions)                            unbounded, off-device
```

Read the right-hand column. **Exactly one of the five grows with context**, and it is the
one that lives in the memory you are compute-bound on. That is the entire reason this
research track exists.

The two capacity numbers for weights measure different things and must not be averaged.
Allen-Zhu and Li fit synthetic (name, attribute, value) tuples and report **~2 bits of
useful knowledge per parameter** across architectures `[C]` (`2404.05405`). Morris et al.
train hundreds of models from 500K to 1.5B params and put the capacity for *unintended
memorization of random data* at **~3.6 bits per parameter**, with a sigmoidal law and a
grokking transition once capacity fills `[C]` (`2505.24832`). The first is knowledge
density; the second is a storage upper bound.

### 3.3 Worked at our own scale, because the ratios are the shock

Take the Proteus shape used in `attention-variants-and-kv-cost.md` §4 — `[A]` medium
confidence, it is a plausible arm and not yet a frozen config: 24 layers, `d_model` 1024,
`n_q` 16, `n_kv` 8, `d_h` 64, bf16, all-global, ~300M params.

```
c = 2 × L × n_kv × d_h × p
  = 2 × 24 × 8 × 64 × 2 B
  = 49,152 B  =  48 KiB per token
```

Weights: `300e6 × 2 B = 600 MB = 572 MiB`.
Parametric information content: `300e6 × 2 bits = 75 MB`; upper bound `300e6 × 3.6 bits =
135 MB`.

Now put the KV cache next to those:

| Context `T` | `kv_bytes` | vs weight **bytes** (572 MiB) | vs weight **information** (135 MB) |
|---|---|---|---|
| 1,024 | 48.0 MiB | 0.08× | 0.37× |
| 12,203 | 572 MiB | **1.00×** | 4.4× |
| 32,768 | 1.50 GiB | 2.7× | 12× |
| 131,072 | 6.00 GiB | 10.7× | 48× |
| 1,354,411 | **62 GiB** | 111× | 493× |

Three checkable numbers fall out and Exercise A reproduces them:

- **`T ≈ 12,200` tokens is where the KV cache outweighs the entire model.** Not "a large
  context." Twelve thousand tokens. A medium document.
- **`T ≈ 1.35 M` tokens fills the `[M]` ≥62 GiB fast tier** measured on this box
  (`notebook/uma-carveout-controls-fast-tier.md`, single run per arm, 2026-07-26).
- At that point the KV cache is **111×** the size of the weights. This is the ratio
  `research/synthesis.md` calls "a ratio no production system ever sees." It is not a
  quirk of our hardware; it is what happens when you hold a big memory next to a small
  model, and it is why this lab's scale is an *enabling condition* for memory research
  rather than a compromise.

For contrast, the reference model: Laguna S 2.1 has `c = 192 KiB/token` exactly `[M]`
(`ASSUMPTIONS.md → kv-per-token-laguna`; `L=48`, `n_kv=8`, `d_h=128`, bf16), so `62 GiB /
192 KiB ≈ 338,600` tokens fills the same tier. But **all** 118B parameters are resident —
active-parameter count governs FLOPs, not bytes — so the weights are 236 GB and the cache
only outweighs them at `236e9 / 196,608 ≈ 1.20 M` tokens, which is essentially its
advertised context. **The bigger the model relative to its per-token cache, the later the
crossover**, and small models enter the KV-dominated regime at trivially short contexts.
Keep the two parameter counts straight: for an MoE, bytes are total params and FLOPs are
active params, and the taxonomy's cost table needs both.

### 3.4 The crossover nobody states: recurrent state vs KV in bytes

The selling point of a constant-state architecture is that `state_bytes` has no `T` in it.
The unstated corollary is that a constant is not the same as a small constant. Set the two
equal:

```
                  state_bytes        L × h × d_hs × d_s × p_s        h × d_hs × d_s × p_s
T*  =  ───────────────────────  =  ──────────────────────────  =  ──────────────────────
                       c              2 × L × n_kv × d_h × p          2 × n_kv × d_h × p
```

`L` cancels when both stacks have the same depth. Below `T*` the constant-state model
holds **more** bytes than the KV model; above it, fewer.

Worked at a matched shape — `d_model` 1024, Mamba-2 expansion 2 so `d_inner = 2048`,
`headdim = 64` giving `h = 32`, `d_state = 128`, and the state carried in fp32 because the
chunked scan's boundary pass runs fp32 regardless of model dtype (`ssd_combined.py:375`
calls `_chunk_state_fwd` with `states_in_fp32=True`):

```
state per layer  = 32 × 64 × 128 × 4 B  = 1,048,576 B = 1.00 MiB
whole stack      = 24 × 1 MiB           = 24 MiB
T*               = 24 MiB / 48 KiB      = 512 tokens
```

In bf16 the state halves and `T* = 256`.

**A constant-state layer at this shape only starts saving memory past ~512 tokens, and it
saves nothing at all at prompt lengths a chat turn actually uses.** That is not an argument
against SSMs — the argument for them is asymptotic and it is overwhelming at 128k, where
KV is 6 GiB and the state is still 24 MiB, a **256× ratio**. It is an argument against
citing "constant state" as if it were free. Below `T*` you are paying a fixed 24 MiB to
avoid a bill you had not yet received.

One more caveat with a price tag: **"constant state" is a decode-time property only.**
During training and prefill the chunked scan materializes one fp32 state per chunk before
the boundary pass runs, so activation memory there is `O(L / chunk_size)`, not `O(1)`.

### 3.5 The inequality where `T` cancels: refetch versus recompute

This is the equation Mnemosyne's tiering arm turns on, and it is worth deriving in full
because the result is counterintuitive twice.

You have evicted a prefix of `T` tokens. Two ways to get it back: **refetch** the bytes
from a slower tier, or **recompute** them by re-running prefill.

```
refetch_seconds    =  c × T / BW
recompute_seconds  =  prefill_FLOPs(T) / F
```

Prefill FLOPs have a term linear in `T` (the weight-matmul work, `2·P` FLOPs per token) and
a term quadratic in `T` (attention scores and the weighted sum, which scale with *query*
heads):

```
prefill_FLOPs(T)  =  2·P·T  +  2·L·n_q·d_h·T²
```

Refetch wins when `c·T/BW < (2·P·T + 2·L·n_q·d_h·T²)/F`. Divide both sides by `T` — the
linear part of the comparison is scale-free:

```
                          c × F
BW*  =  ───────────────────────────────────────
              2·P  +  2·L·n_q·d_h·T
```

**Refetch beats recompute whenever the slow tier is faster than `BW*`.** Two readings:

**First reading — the threshold is tiny.** With `[M]` `F = 20.9 TFLOP/s` bf16 GEMM at 8192³
(`scripts/benchmark_gemm.py`, 2026-07-26) as an optimistic ceiling on `F`:

| `T` | `2P + 2·L·n_q·d_h·T` | `BW*` at peak `F` | `BW*` at 30% of peak `[A]` |
|---|---|---|---|
| → 0 | 6.00e8 | 1.71 GB/s | 0.51 GB/s |
| 32,768 | 2.21e9 | 0.465 GB/s | 0.14 GB/s |
| 131,072 | 7.04e9 | 0.146 GB/s | 0.044 GB/s |

At 128k context, **any tier faster than ~150 MB/s beats recomputing** at this shape. That
includes spinning disks. The reason is that a small model does very little arithmetic per
token relative to the bytes it caches — `c/(2P)` is bytes-per-FLOP, and it is large when
`P` is small.

**Second reading — the threshold falls as context grows.** The `T²` term means recompute
gets *relatively more expensive* the longer the prefix, so refetch wins by a wider margin
at long context, which is precisely the regime where you were tempted to evict. This is the
opposite of the intuition that says "long prefixes are expensive to store, so drop them."

Three caveats, because this equation is easy to over-apply:

- It is a **throughput** comparison, not a latency one. Recompute contends for the same
  compute the decode step needs; refetch contends for bandwidth. On a unified-memory
  machine those are the *same* resource, which is exactly why our hardware is the
  interesting instrument for this question and a discrete GPU is not.
- Recompute also moves bytes: at minimum one read of the weights, `P × b_w = 600 MB`,
  which at `[M]` ~200 GB/s is ~3 ms — negligible against a multi-second refetch, but not
  zero.
- `F` is a peak-GEMM number standing in for achieved prefill throughput. `[A]` low
  confidence in the absolute value, high confidence in the *ordering*; the 30% column is a
  guess and is labelled as one. The cheapest test that would move it is an actual prefill
  throughput measurement at our shape, which is a Hardware Validation Gate item anyway.

And the direction that matters for Laguna: `c = 192 KiB`, active `2P = 1.7e10`, giving
`BW*(T→0) ≈ 0.24 GB/s`. **A bigger model makes refetch more attractive, not less** — and
MLA, by shrinking `c` roughly 3.6× against GQA-8, is the one design choice that pushes back
toward recompute. Nobody frames MLA that way.

### 3.6 The exchange rate: retrieval and session store have no runtime cost of their own

This is the most useful single line in the module for anyone who has built agent memory.

**The retrieval index and the session store do not have a runtime memory cost. They have an
exchange rate into KV bytes and prefill FLOPs.** Recall `n` tokens from either and you pay:

```
kv_cost      =  c × n            bytes, in the fast tier, for the life of the request
prefill_cost =  2·P·n  +  2·L·n_q·d_h·(2·T_ctx·n + n²)   FLOPs, once
```

At the Proteus shape, ignoring the cross-term for a short injection into a short context:

```
1,024 recalled tokens  =  48.0 MiB of KV cache  +  6.14e11 FLOPs
                       ≈  48.0 MiB  +  ~29 ms at peak F  (~98 ms at 30%)
```

A retrieval of `k = 8` passages at `m = 512` tokens each is 4,096 tokens: **192 MiB of KV
cache and ~118 ms of prefill, per call.** That is the number the RAG-versus-long-context
debate is actually about, and it is almost never stated in those units.

The session store adds one cost RAG does not have: **the write path is itself an LLM call.**
Summarization and consolidation cost a forward pass, and doing it synchronously blocks
inference `[C]` (`2605.23296`, 2026).

The design consequence is sharp. Your agent-memory instinct — "storage is cheap, keep
everything, retrieve generously" — is correct about the store and wrong about the
consequence. Every recalled item is converted, at a fixed exchange rate you can compute in
advance, into the one resource that is genuinely scarce. **A memory system that retrieves
more is a memory system that has less context budget left**, and no amount of cheap disk
changes that.

### 3.7 "Exactly recoverable" — measured, and it needs a caveat

The taxonomy's central claim is that the KV cache is recoverable **exactly** by recomputing
from tokens. In exact arithmetic that is trivially true: `K = W_k · RMSNorm(x)` is a pure
function. In floating point it is a claim about kernels and reduction order, and it is
testable in twenty lines.

`[M]` **Measured 2026-07-26 on the Z13** (gfx1151, native Windows, `torch
2.12.0a0+rocm7.13.0a20260313`). Toy stack, no pretrained weights, seeded:
`L=8`, `T=512`, `d_model=256`, 8 heads × 32 dims, LayerNorm + SDPA + GELU MLP, `no_grad`.
Three paths to the same last-token K/V:

- **A** — one-shot prefill of all 512 tokens.
- **B** — prefill 511 tokens, then append token 512 against the cache (the decode shape).
- **C** — prefill 511 tokens in **four chunks of 128**, then append token 512 (the
  chunked-prefill / prefix-cache-hit shape).

Repeatability: three evaluations per process × three separate process invocations; all nine
byte-identical, and the cross-process JSON diffs are empty.

| Config | A vs B (append) | A vs C (chunked prefill) |
|---|---|---|
| **fp32, GPU** | bit-identical, all 8 layers | **bit-identical, all 8 layers** |
| **bf16, GPU, default SDPA** | bit-identical, all 8 layers | **diverges from layer 3**; max abs diff 0.0078125 rising to 0.009765625 by layer 7; only **28.7%** of last-token K/V components bitwise equal at layer 7; final hidden state rel-L2 **5.57e-3**, 44.1% of components bitwise equal |
| **bf16, GPU, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** | diverges at layer 5 (6.1e-05) | diverges from layer **2**; hidden rel-L2 4.89e-3 |
| **bf16, CPU** | **diverges from layer 3** (0.0078125) | diverges from layer 3 |

Read the table carefully, because it says four things:

1. **In fp32 the claim is exactly true**, across all three schedules. Recompute is exact.
2. **In bf16 the claim is true under an identical schedule and false under a different
   one.** Chunked prefill produces a numerically different cache from one-shot prefill —
   about one bf16 ULP per entry at these magnitudes, compounding with depth. Nothing is
   *wrong*; both are valid roundings of the same exact quantity.
3. **The AOTriton flag is a numerics change**, which corroborates `ASSUMPTIONS.md →
   sdpa-is-memory-efficient`, and it is not a free one: with the flag on, even the plain
   append path stops being bit-exact.
4. **The bit-exact append result is a property of this kernel path, not of transformers.**
   On CPU bf16, even A vs B diverges. Do not generalize the GPU row.

**Why this matters more than it looks.** `research/synthesis.md` makes the lab's deliverable
a **full-cache oracle diff**: run the expensive full-cache reference alongside a policy and
attribute the divergence. If the reference and the policy arm are prefilled under different
chunk schedules — and in any realistic harness with prefix caching they will be — then
**part of the measured divergence is the scheduler, not the policy.** At `T=512` and 8
layers that floor is already rel-L2 5.6e-3 on the final hidden state. Any oracle-diff
instrument must either pin the prefill schedule across arms or measure this null first. That
is a direct, measured consequence for the riskiest assumption in the lab's plan, and it came
out of a twenty-line script.

`[A]` medium-high confidence that the same effect appears in recurrent state and is likely
**larger**: Mamba-2's chunked scan carries a per-chunk fp32 state through a sequential
boundary pass, so a state recovered under a different `chunk_size` traverses a different
association order. Untested here. The cheapest test is the same three-path probe against
`ssd_minimal_discrete` with `chunk_size ∈ {64, 256}`.

### 3.8 The repair unit is a suffix, not a block — measured

The taxonomy says the KV cache "cannot fault." That is a statement about what shipped
systems do. Here is the stronger structural statement, which is what actually forecloses
the design: **a demand-paging KV cache could not be built at block granularity even if
someone wanted one.**

The argument: `K` at layer `ℓ`, position `t` is a projection of the residual stream at
`(ℓ, t)`, which depends on the attention output at `(ℓ-1, t)`, which is a weighted sum over
K/V at `(ℓ-1, 0…t)`. So recomputing one entry at depth requires everything causally beneath
and before it. The recomputable unit is therefore *prefix-closed below and suffix-closed
after*: given all K/V for positions `[0, s)` at all layers, you can regenerate positions
`[s, T)` at all layers, and nothing smaller.

`[M]` **Measured 2026-07-26**, same box, fp32 so the answer is structural rather than a
rounding artifact. `L=6`, `T=256`, `d_model=256`, 8 heads × 32; replace the embedding at
position `J=100` with a fresh random vector, re-prefill, and count which cached
`(layer, position)` entries changed. Identical across two separate process invocations.

| Layer | entries changed | first | last | contiguous |
|---|---|---|---|---|
| 0 | **1** | 100 | 100 | yes |
| 1–5 | **156** | 100 | 255 | yes |

156 = `T − J` = 256 − 100. The cone is exactly as predicted: **layer 0 is per-token; every
layer above it is suffix-closed from the perturbed position onward.**

Three consequences:

- **Evicting one block at layer 3 does not cost you one block.** Restoring it costs
  re-prefilling every position from that block's start to the end of the sequence, at every
  layer from 3 up. That is why vLLM preempts the sequence rather than servicing a fault, and
  the code says so plainly: `allocate_slots` returns `None`, the scheduler preempts.
- **H2O-style eviction is irreversible in practice**, not merely lossy in theory. There is
  no cheap repair, so the policy's bet is unhedgeable. This is the honest reason
  "eviction is a quality decision" — not because information is lost, but because the
  recovery is as expensive as never having cached at all.
- **The single exception in the entire taxonomy is a sliding window.** An out-of-window
  token is architecturally *unreadable* — the mask forbids it — so discarding it needs no
  repair and costs nothing. A window is a proof; an eviction policy is a bet. That is
  covered in `attention-variants-and-kv-cost.md` §2.3 and it is the only lossless eviction
  in this field.

---

## 4. Why this matters for Proteus and Mnemosyne

### 4.1 The scope of Mnemosyne, derived rather than asserted

`CLAUDE.md` gives Mnemosyne "layered memory and its management: KV cache, eviction/
compression policies, tiering, prefix reuse, attribution instrumentation." The taxonomy
tells you which of the five that is, and — more usefully — which guarantees it does **not**
owe.

| Guarantee | Owed by Mnemosyne? | Why |
|---|---|---|
| Durability / fsync | **No** | Memo table. Loss costs FLOPs, not bits. |
| Write-ahead log, journal | **No** | Nothing to replay to. The tokens are the log. |
| Checksums / integrity | **No** | A corrupted entry is indistinguishable from a differently-rounded one (§3.7). Detecting corruption means recomputing, which is the thing you were avoiding. |
| Replication | **No** | Single-device, and copies of deferred work have no value. |
| Coherence protocol | **No** | No second copy exists to be coherent with. |
| Access control | **No** *for KV* | Except across tenants sharing a prefix cache — see §4.3. |
| Deterministic reconstruction | **Yes, conditionally** | §3.7: only if the prefill schedule is pinned. |
| Attribution of quality loss | **Yes — this is the contribution** | §4.2. |

Anything on the "No" rows appearing in Mnemosyne is over-engineering imported from the
wrong analogy, and it should be caught in review by asking one question: *what information
is lost if this fails?*

**The session store is not Mnemosyne's.** It is the only tier with real durability,
provenance, integrity, and access-control obligations, and it is a different research
programme (`research/synthesis.md` parks it explicitly, node 1.5 of the issue tree). If a
session store ever lands in this repo it gets its own package name, because a subsystem
that owes durability and a subsystem that owes none should not share an interface. Putting
them together is precisely the category error the vocabulary collision produces.

### 4.2 Why the lab's deliverable is an instrument, restated from the taxonomy

`research/synthesis.md` concludes: *build the instrument, not another policy.* The taxonomy
gives you the mechanical reason, and it is worth being able to state it in one sentence at
a whiteboard:

> **A storage tier reports its own misses; a memo table cannot, so the miss signal has to be
> manufactured, and manufacturing it is the contribution.**

A page cache tells you its hit rate for free — the fault handler is the instrument. Here
there is no fault, no signal, and no distinction the system can observe between "the policy
kept the right entries" and "the policy dropped something that mattered and the model
confabulated smoothly over the gap." The only way to observe the difference is to run the
full-cache oracle and diff. That is expensive at 70B and cheap at 300M against a `[M]` ≥62
GiB fast tier, which is the whole argument for this lab's scale.

§3.7 adds a design constraint to that instrument that was not previously written down:
**pin the prefill schedule across arms, or measure the schedule-only null first.** In bf16
on this box the schedule-only divergence at `T=512, L=8` is rel-L2 5.6e-3 on the final
hidden state, which is not obviously smaller than the effect a mild eviction policy would
produce. An instrument whose noise floor you have not measured is a rumour generator.

### 4.3 Consequences for the config surface and the naming rule

**House vocabulary is enforced in identifiers, not just prose.** Per the NAMING RULE, use
`weights`, `recurrent-state`, `kv-cache`, `retrieval-index`, `session-store`. Never
"activation memory," "working memory," "short-term memory," "long-term memory," "external
memory," or bare "memory." Two standing rules from the note, both of which are cheap and
both of which pay: when citing a source, name which camp's vocabulary it uses ("MemOS's
*activation memory* (= our `kv-cache`)"); and **never write "memory" as a bare noun in a
hypothesis card** — a hypothesis whose subject is ambiguous cannot fail cleanly.

**`c` is a first-class derived quantity.** Bytes-per-token for the whole stack appears in
the tier crossover (§3.3), the state crossover (§3.4), the refetch inequality (§3.5) and
the retrieval exchange rate (§3.6). It should be computed once from config and exposed, not
recomputed inline in four places. Since Laguna varies query heads per layer but not KV heads
`[M]` (`ASSUMPTIONS.md → laguna-heads-uniform`, refuted for query heads only), `c` is exact
and not an estimate — but a hybrid model's `c` is piecewise, and a Mnemosyne cost model keyed
on a single top-level head count is wrong for 75% of Laguna's layers.

**The boundary rule has a reason, and it is this taxonomy.** `mnemosyne → torch`, never
`proteus`. A memo table's interface is `(tokens, weights) → entries`. It needs tensors and
shapes; it does not need the model class. If Mnemosyne needs to import Proteus, what it is
managing is not a memo table over a pure function — it is an implementation detail of one
model, and the contribution claim evaporates. The dependency graph enforces what the
taxonomy implies.

**Prefix sharing is the one place a KV cache acquires a real security property.** Break 4
says nothing can be dirty — true within one tenant. Across tenants, a shared prefix cache
is a shared read-only tier keyed on `hash(parent_hash, tokens)`
(`kv_cache_utils.py:596`), and whether a hit occurred is observable in latency. `[A]`
high confidence that this is a timing side channel on prompt contents; the mechanism is
mechanical rather than cited here, and the cheapest test is a two-request timing probe
against a warmed prefix. SGLang namespaces its tree with an `extra_key` (LoRA id, cache
salt) exactly like an ASID (`radix_cache.py:355`), which is the mitigation shape. Note what
changed: the hazard is not in the *bytes* (still reconstructible, still information-free)
but in the *metadata* — the existence of a hit. That is the one durable exception to "the KV
cache owes no storage-system guarantees," and it is worth an ADR if Mnemosyne ever grows
cross-request sharing.

### 4.4 What the taxonomy says about our hardware bet

`[M]` The ≥62 GiB fast tier at ~200 GB/s means a 300M model can hold a KV cache **111×** its
own weights (§3.3). §3.5 then says the refetch/recompute boundary on this machine sits below
2 GB/s at every context length we can reach. Together those say something specific: **the
tier-ratio experiment `research/synthesis.md` reserves for this box is a study of a boundary
that, at our scale, we predict is not close.** That is a good thing to know before running
it — it means a *null* result is the prediction, and any observed flip of the boundary is
either a real effect or a harness bug, both of which are worth finding. Pre-register it that
way.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`. Read these in the order given — the sequence is the taxonomy.

### 5.1 Weights: the parametric store has no index

| Where | What to look for |
|---|---|
| `training/nanogpt/model.py:126` | `self.transformer = nn.ModuleDict(...)`. This is the entire parametric store: a dict of tensors. Look for what is absent — no key, no schema, no per-fact addressing, no metadata. There is nothing here that could support a `DELETE WHERE`. |
| `training/nanogpt/model.py:150` | `def get_num_params` — the only enumeration the store supports is *counting*. You can ask how many parameters there are; there is no API, and no possible API, that lists what they encode. Hold this next to §3.3: 300M params is 572 MiB of bytes and at most ~135 MB of information. |

### 5.2 Recurrent state: fixed size, destructive write, no slot per token

| Where | What to look for |
|---|---|
| `architecture/mamba/mamba_ssm/modules/mamba2.py:352` | The inference cache allocation: `(batch, nheads, headdim, d_state)`. **No `seqlen` term.** Decode footprint at 1K and 1M tokens is byte-identical. This one line is the whole constant-state claim, and §3.4 is its price. |
| `architecture/mamba/mamba_ssm/modules/mamba2.py:317` | The single-token decode update in plain PyTorch — an in-place `copy_` that decays and accumulates. This is what one autoregressive step does to the state. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` | The inter-chunk recurrence in one line: decay the running carry, add the chunk's contribution. **Destructive overwrite, not append.** There is no version of this line that leaves the old value retrievable. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54` | `h = h.clone() * g[:, :, i].exp()` — one scalar attenuates the entire K×V state. Every stored association decays by the same factor. This is the "gate," and it is indiscriminate. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56` | The delta term: read what the decayed state already returns for this key and subtract it, so the write carries only the residual. This read-before-write is what makes it an overwrite rather than an accumulate — and it is the *only* targeted erase in the layer. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:58` | The write: a rank-1 outer product added into the state. **The only place new information ever enters.** Note there is no addressing — keys are L2-normalized continuous vectors, so a similar key smears into a neighbour's content. That is the interference failure mode, in three lines. |

> **What to take away.** Put lines 54, 56 and 58 side by side and you have a fully-associative
> cache with a global TTL tick and a compare-and-swap. Then notice the three things it is
> missing that every cache you have operated has: an address, a capacity miss, and a tier to
> spill to.

### 5.3 KV cache: an allocator with no fault handler

| Where | What to look for |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `past_key_values.update(key_states, value_states, self.layer_idx)` — the one line where bytes enter the cache. Note what has already happened above it: QK-norm, then RoPE. A cached key is `RoPE(RMSNorm(k))`, which is why "just re-position the cache" is not a thing. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | `config.layer_types[layer_idx]` — which layers keep an unbounded cache and which keep a window. The only lossless eviction in the taxonomy is decided by this list lookup. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97` | `req_to_blocks` — the block table itself: request id → ordered list of blocks, list index = logical block, stored `block_id` = physical frame. A page table, in Python, on the scheduler. |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:340` | **`allocate_slots`, and its return type: `KVCacheBlocks \| None`.** This is Break 2, in a type annotation. `None` is not a fault to be serviced; the caller preempts the request. Read the docstring's block-layout diagram and then ask where the demand-paging path would go. There isn't one, and §3.8 says why there couldn't be. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `get_new_blocks` — pop N blocks off the free queue, evict each one's stale prefix-cache hash entry, bump refcount. **No zeroing.** A freshly allocated block contains a previous request's KV until it is overwritten, and nobody cares, because it is not data. |
| `memory/vllm/vllm/v1/core/block_pool.py:719` | `free_blocks` — freeing is not evicting. Decrement refcount, push back on the queue **with contents and hash intact**, so a later request can resurrect it. The free list and the LRU victim cache are the same linked list. |
| `memory/vllm/vllm/v1/core/block_pool.py:702` | `touch` — the resurrection: increment refcount, unlink from the free queue in O(1). This is the closest thing in the whole system to a cache hit, and note it supplies *no bytes to anyone*; it only prevents work. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens` — the key is `hash(parent_hash, token_ids, extra_keys)`, a **chain**. Strictly prefix-ordered: the same 16 tokens at a different offset are a different key, and one changed token at position 0 invalidates everything downstream. This is §3.8's suffix property, showing up in a hash function. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict` — considers only *leaves*, from an incrementally maintained `evictable_leaves` set, peeling the frontier inward. A hot child keeps a cold parent resident indefinitely. Topological, not recency-ordered — and topology here is again prefix order. |
| `memory/mooncake/mooncake-store/include/replica_selection.h:122` | `SelectBestReplica` — the read-side tier ladder: local DRAM > local NVMe-over-fabric > remote DRAM > remote NOF > local disk > disk. A fixed preference order, not a latency model. This is the closest thing to a real hierarchy anyone ships. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | `BatchEvict` — and the `offload_force_evict` path that discards bytes rather than blocking on writeback. **Read this line and then try to name a storage system allowed to do it.** That inability is Break 3, in production C++. |

### 5.4 Retrieval index: a normal database, and the moment it becomes tokens

| Where | What to look for |
|---|---|
| `memory/a-mem/agentic_memory/retrievers.py:42` | `class ChromaRetriever` — an ordinary vector store. Note what it is: transactional, addressable, deletable, with a real backing corpus. Everything your instincts say about a storage tier is *correct here* and nowhere else in §5.2 or §5.3. |
| `memory/a-mem/agentic_memory/retrievers.py:63` | `add_document` — writes are offline and explicit. Contrast with `modeling_laguna.py:397`, where the "write" is a side effect of a forward pass nobody scheduled. |
| `memory/a-mem/agentic_memory/retrievers.py:93` | `search(query, k)` — top-k by similarity. **There is no miss.** The store always returns `k` results whether or not any of them is relevant; the failure mode is silent low-relevance, not a signalled absence. A second kind of no-miss-path, distinct from §5.3's. |

> **What to take away.** The retrieval index is the only one of the five where a systems
> engineer's instincts transfer wholesale. Then trace what happens to the returned text: it
> becomes prompt tokens, and §3.6's exchange rate converts it into exactly the resource the
> rest of this track is about.

### 5.5 Session store: the only authoritative tier, written by a language model

| Where | What to look for |
|---|---|
| `memory/letta/letta/functions/function_sets/base.py:246` | `core_memory_append` — read the whole function. It is four lines: fetch the block, string-concatenate, write back. No validation, no schema, no conflict detection. The *content* is a tool-call argument produced by the model. |
| `memory/letta/letta/functions/function_sets/base.py:263` | `core_memory_replace`, and the sentence in its docstring: **"To delete memories, use an empty string for new_content."** A destructive, unverified, irreversible write whose only integrity check is that `old_content` occurs in the block. Put that next to `block_pool.py:719`, where freeing a KV block is deliberately non-destructive because it costs nothing to keep. The tier with nothing to lose is careful; the tier with everything to lose is not. |
| `memory/letta/letta/functions/function_sets/base.py:164` | `archival_memory_insert` — unbounded append to the durable tier, again from a tool call. There is no compaction policy here, and no quota. |
| `memory/a-mem/agentic_memory/memory_system.py:159` | `analyze_content` — the keywords, context and tags that will later drive retrieval are **generated by an LLM from the content itself**, with a bare `except` that falls back to `{"keywords": [], "context": "General", "tags": []}`. A schema-inference step whose failure mode is silent and whose input is untrusted. |
| `memory/a-mem/agentic_memory/memory_system.py:233` | `add_note` — the write path: analyze, link to neighbours, insert, and trigger consolidation every `evo_threshold` writes. Note that the write path invokes the model, which is §3.6's asymmetric cost. |
| `memory/a-mem/agentic_memory/memory_system.py:266` | `consolidate_memories` — **rebuilds the entire collection from scratch** on every consolidation. Compaction as a full rewrite. Also, per the 2026 write-channel taxonomy `[C]` (`2606.04329`), this is an attack surface: a summarizer is a write channel. |
| `memory/a-mem/agentic_memory/memory_system.py:398` | `delete` — and this is the point. A `delete` exists here and nowhere else in the taxonomy that means what you think it means. Deleting a KV block loses nothing. Deleting this loses the information. |

> **What to take away.** You have built tiered agent memory. Read these six pointers with the
> question: *which of these operations would I have allowed in a production storage system
> without a review?* The answer is roughly none of them, and the reason is not that these
> projects are careless — it is that the field imported the word "memory" from the tier where
> writes are free and applied it to the tier where they are not.

---

## 6. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats**, from `ASSUMPTIONS.md`. Single tensors **≥32 GiB hang**
silently at 0% CPU (`large-tensor-fault-32gib`, refuted) — keep every buffer under 31 GiB;
none of these exercises comes close. bf16 numerics on gfx1151 are **untested**
(`bf16-numerics-unproven`), which is exactly what Exercise B probes one corner of. The
Hardware Validation Gate has not run, so nothing measured here is evidence by house
standard until it does; these are instrument-shakedown runs and should be labelled as such
in any notebook entry.

Write scratch scripts under `notebook/`. Exercise B is a Hardware Validation Gate candidate
and should migrate into the rig with tests on reuse.

---

### Exercise A — the five-leg budget, and three crossovers you can check

**Goal:** one function per leg, validated against numbers already in this repo, then used to
find three crossings that decide how Mnemosyne is sized.

**Hardware:** none. Pure Python. **Runtime:** 30–40 min to write, under a second to run.

```python
"""Byte budgets for the five things called memory, at one model shape."""
KIB, MIB, GIB = 1024, 1024**2, 1024**3

PROTEUS = dict(L=24, n_kv=8, d_h=64, p=2, n_q=16, P=300_000_000, b_w=2)
LAGUNA  = dict(L=48, n_kv=8, d_h=128, p=2, n_q=48, P=8_500_000_000, b_w=2)
SSM     = dict(L=24, h=32, d_hs=64, d_s=128, p_s=4)          # matched d_model=1024

def kv_bytes_per_token(L, n_kv, d_h, p, **_):
    """c — the constant that shows up in four other formulas."""
    return 2 * L * n_kv * d_h * p

def kv_bytes(T, cfg):            return kv_bytes_per_token(**cfg) * T
def weight_bytes(P, b_w, **_):   return P * b_w
def weight_information_bytes(P, bits_per_param, **_): return P * bits_per_param / 8
def state_bytes(L, h, d_hs, d_s, p_s):  return L * h * d_hs * d_s * p_s

def refetch_beats_recompute_above(T, cfg, flops_per_s):
    """BW* in bytes/s: any tier faster than this beats recomputing the prefix."""
    c = kv_bytes_per_token(**cfg)
    denom = 2 * cfg["P"] + 2 * cfg["L"] * cfg["n_q"] * cfg["d_h"] * T
    return c * flops_per_s / denom
```

**Five assertions that must pass.** Each reproduces a number stated elsewhere in the repo:

1. `kv_bytes_per_token(**LAGUNA) == 192 * KIB` — matches `ASSUMPTIONS.md →
   kv-per-token-laguna`.
2. `kv_bytes_per_token(**PROTEUS) == 48 * KIB` — matches
   `attention-variants-and-kv-cost.md` §4.
3. `state_bytes(**SSM) == 24 * MIB`.
4. `weight_bytes(**PROTEUS) == 600_000_000` (572 MiB).
5. `round(refetch_beats_recompute_above(0, PROTEUS, 20.9e12) / 1e9, 2) == 1.71` — the `[M]`
   GEMM figure from `scripts/benchmark_gemm.py`.

**Deliverable — three crossover token counts and one plot.**

Solve for `T` at each crossing and report it:

| Crossing | Expected answer |
|---|---|
| `kv_bytes(T) == state_bytes(**SSM)` | **512 tokens** |
| `kv_bytes(T) == weight_bytes(**PROTEUS)` | **≈ 12,200 tokens** |
| `kv_bytes(T) == 62 GiB` (the `[M]` fast tier) | **≈ 1,354,400 tokens** |

Then plot, on log-log axes, all five legs against `T ∈ [1, 2^21]`: the four flat lines
(weights bytes, weight information at 2 and 3.6 bits/param, SSM state) and the one sloped
line (KV). Mark the three crossings and the 62 GiB tier line.

**Then answer, in one line each, in your notebook entry.**
(a) At what context does one sequence's KV cache exceed the *information capacity* of the
entire model at 3.6 bits/param, and what does that ratio mean given that the cache contains
no information the tokens do not? (b) Recompute crossing 1 with the SSM state in bf16 — does
your answer to "is constant state cheaper?" change for a 256-token chat turn?

**Check yourself.** Crossing 3 must equal `62 × 1024 × 1024 KiB ÷ 48 KiB`. If it does not,
your `2` (one K *and* one V) is missing or doubled.

---

### Exercise B — is recompute exact? Three prefill schedules, one cache

**Goal:** test §3.7's claim on our instrument, and establish the noise floor that any
oracle-diff attribution instrument has to clear.

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback: `--device cpu`, works
identically and gives a *different* answer, which is the point.** **Runtime:** ~40 s per
configuration on GPU, ~3 min on CPU. Run six configurations (bf16/fp32 × default/AOTriton ×
GPU, plus CPU bf16); budget 15 minutes total.

Build a small stack — LayerNorm, SDPA, GELU MLP, `L=8`, `T=512`, `d_model=256`, 8 heads of
32 — with a fixed seed and `no_grad`. Then produce the last token's K and V three ways:

- **A** — one-shot prefill of all `T` tokens, `is_causal=True`.
- **B** — prefill `T-1`, then append the last token against the cache with an explicit
  all-visible mask.
- **C** — prefill `T-1` **in chunks of 128**, each chunk masked against the accumulated
  cache, then append the last token.

For every layer report `max_abs_diff` and the **fraction of components that are bitwise
equal** between A/B and A/C, plus the same for the final hidden state. Run three repeats in
process and diff across three separate process invocations — a number you cannot reproduce
in a fresh process is an anecdote, per house standard.

The masking detail matters and is the easiest place to get this wrong: when a chunk of `S`
queries is appended to `Tk - S` cached keys, the mask is `key_index <= (Tk - S + query_index)`.
Getting that wrong gives you a model that attends to the future, and it will still run.

**Deliverable — one table and one number.**

1. The per-layer table for at least fp32-GPU, bf16-GPU-default, and bf16-CPU.
2. **The number that matters: the rel-L2 divergence of the final hidden state between A and
   C in bf16.** That is the schedule-only noise floor for an oracle-diff instrument.

**Expected results, so you can tell a bug from a finding** `[M]` (measured on this box
2026-07-26, 3 repeats × 3 processes, byte-identical; `torch 2.12.0a0+rocm7.13.0a20260313`):

| Config | A vs B | A vs C |
|---|---|---|
| fp32 GPU | bit-identical | bit-identical |
| bf16 GPU default | bit-identical | diverges from layer 3, max abs 0.0078125, 28.7% bitwise equal at layer 7, hidden rel-L2 **5.57e-3** |
| bf16 GPU + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | diverges at layer 5 (6.1e-05) | diverges from layer 2, hidden rel-L2 4.89e-3 |
| bf16 CPU | diverges from layer 3 | diverges from layer 3 |

**What a falsification would mean.** If your fp32 run diverges, you have a nondeterministic
kernel and the whole determinism assumption in `CLAUDE.md` needs revisiting. If your bf16 A
vs C run is bit-identical, your chunked path is not actually chunking — check that the cache
is being carried between chunks and that the mask is right. Either way, write it up: this is
a Hardware Validation Gate item in disguise.

---

### Exercise C — map the dependency cone, and derive the repair unit

**Goal:** prove §3.8 rather than believe it. Twenty lines, and it settles whether a
demand-paged KV cache is a missing feature or an impossible one.

**Hardware:** one gfx1151 GPU. **CPU fallback: identical, `device="cpu"`, ~30 s.**
**Runtime:** 15 min to write, under 20 s to run. Use **fp32** so the answer is structural
and not a rounding artifact — this is the one place in the module where bf16 would give you
a wrong answer for an interesting reason.

Prefill `T=256` tokens through `L=6` layers and keep every layer's K and V. Then replace the
embedding at position `J=100` with a fresh random vector, re-prefill, and count — per layer —
how many positions' cached entries changed, where the changed set starts, where it ends, and
whether it is contiguous.

**Deliverable — one table and one derived sentence.**

Expected `[M]` (measured on this box 2026-07-26, fp32, identical across two process
invocations):

| Layer | entries changed | first | last | contiguous |
|---|---|---|---|---|
| 0 | 1 | 100 | 100 | yes |
| 1–5 | 156 | 100 | 255 | yes |

`156 = T − J`. Then write the derived sentence in your own words, and check it against
§3.8: *given all K/V for positions `[0, s)` at all layers, the smallest set you can
regenerate is …*

**Then extend it, and this is the part with research value.** Repeat with the perturbation
applied at a *layer* rather than a token: zero out K/V at layer 3, positions `[100, 116)` —
one 16-token vLLM block — and measure how many entries at layers 4 and 5 are affected.
Report the ratio of *entries you must repair* to *entries you evicted*. Predict it first.

**What a falsification would mean.** If layer 0 shows more than one changed position, your
model has a non-causal path — most likely a mask bug, possibly a normalization computed over
the sequence axis. If layers ≥1 show a *non-contiguous* changed set, something is wrong with
your comparison, not with transformers. If the changed set at layer 1 starts *before*
position `J`, stop and find the leak; you have information flowing backwards in time.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. A colleague proposes adding CRC-32C checksums to Mnemosyne's KV blocks, citing a storage
   system he ran that caught silent corruption this way. Give the one-sentence argument
   against, and then give the one narrow case where he is right.

2. Two of the five have no miss path. Name them, and then explain why "no miss path" means
   something structurally *different* for each of the two.

3. At the Proteus shape (`c` = 48 KiB/token, `P` = 300M), a teammate wants to evict half the
   KV cache at 128k context rather than offload it to a slower tier, arguing that "recompute
   is cheap on a small model." Using §3.5, what bandwidth would the slow tier need to fall
   below before he is right, and what does that number tell you about his argument?

4. Your agent has 40,000 tokens of accumulated session-store notes and you are considering
   raising the retrieval `k` from 5 passages to 15. At the Proteus shape, what does that cost
   you, in the two units that actually matter? Why is "storage is cheap" true and irrelevant?

5. In fp32 the three prefill schedules in Exercise B produce bit-identical caches; in bf16
   the chunked schedule diverges. Does that make the taxonomy's claim — "the KV cache is
   recoverable *exactly* by recomputing from the tokens" — wrong? Answer precisely.

6. Rank the five tiers by "cost of total loss," from cheapest to most expensive, and then
   name the one place where the ranking inverts relative to how much engineering effort the
   field spends on each.

---

## 8. What is still unsolved here

Everything below is testable at 20M–300M params on one gfx1151, inside the `[M]` ≥62 GiB
fast tier at ~200 GB/s, and every item needs a pre-registered hypothesis card before it
runs. The Hardware Validation Gate precedes all of them.

1. **Is a quantized KV cache still a pure memo table?** §2.4 claims the KV cache carries no
   information beyond the tokens. Under 4-bit or 2-bit KV that needs a caveat, and the
   caveat is measurable: compare bit-exact recompute against quantized cached decode and
   report logit divergence. If they differ materially, quantized KV is a lossy *information*
   store and every consequence in §2.4 needs restating for it. Exercise B is the harness;
   this is a two-hour extension of it.

2. **How large is the schedule-only null, at realistic scale?** §3.7 measures rel-L2 5.6e-3
   at `T=512`, `L=8`, on random weights. Nobody has reported this quantity at 24 layers, at
   32k context, on trained weights, in the units an attribution instrument uses. It is the
   noise floor of the lab's primary deliverable and it is currently a single toy datapoint.

3. **Does the same schedule sensitivity afflict recurrent state, and worse?** `[A]`
   medium-high confidence that it does, because the chunked scan runs a sequential fp32
   boundary pass whose association order depends on `chunk_size` (`ssd_combined.py:375`,
   `ssd_state_passing.py:80`). Untested. If a Mamba state is not reproducible across
   `chunk_size`, then "recoverable exactly by recompute" is *less* true for recurrent state
   than for KV, which would invert a row of the §2.4 table.

4. **Where is the recall crossover in bytes, not in architecture names?** At matched params
   and matched tokens, at what `state_bytes` does a constant-state model match a KV model on
   multi-query associative recall? Zoology gives the harness `[C]` (`2312.04927`) and the
   recall-vs-state-size Pareto frontier is the capacity-planning statement of the same fact
   `[C]` (`2402.18668`). The answer in bytes is the number §3.4 is missing.

5. **Do compaction policies transfer across the five legs, as the rate-distortion frame
   claims `[C]` (`2607.08032`)?** Cheapest version: score an H2O-style attention-mass rule
   against plain recency for *session-store* consolidation on a synthetic multi-turn task. If
   a KV policy transfers, the unification has teeth. If not, it is a metaphor. Three weeks
   old; nobody has tried.

6. **Parametric versus context attribution.** Train a ~100M model on synthetic facts, then
   present a conflicting fact in context. Which wins, and can a probe attribute the answer to
   `weights` versus `kv-cache`? This needs no scale and it is the attribution question the
   literature is weakest on.

7. **CONTESTED — RAG versus long context.** One evaluation finds long context generally
   beating RAG on QA while RAG wins on dialogue and general queries `[C]` (`2501.01880`);
   other 2025–2026 work reports the ordering reversing as corpus size grows, with a large
   cost asymmetry in RAG's favour. The honest 2026 summary is that the ordering is task- and
   scale-dependent and hybrids are the practical default. Do not let anyone assert a winner.
   `[A]` low-to-medium confidence that the ordering is even stable within one task family;
   the cheapest test is a fixed-corpus sweep holding the generator constant while varying
   corpus size, which is affordable at our scale.

8. **CONTESTED — is the session store "memory" at all?** "Contextual Agentic Memory is a
   Memo, Not True Memory" argues current systems are externalized note-taking with no
   consolidative process `[C]` (`2604.27707`); "Storage Is Not Memory" makes an adjacent
   retrieval-centred argument `[C]` (`2605.04897`); the MemOS/MemCube line argues the
   opposite, that memory becomes real once it is a scheduled resource spanning plaintext,
   activation and parameter tiers `[C]` (`2507.03724`). All live. Both camps are describing
   the same artifacts.

9. **CONTESTED, and directly load-bearing for Mnemosyne — where does the control plane
   belong?** An architectural study across thirteen configurations finds that *where* the LLM
   sits in the memory pipeline determines which failure modes are even addressable, with
   mutation-time placement winning, contradicting the common assumption that retrieval-time
   reranking is the leverage point `[C]` (`2606.15903`). If that holds, a Mnemosyne plug-point
   at eviction time is better positioned than one at read time, which is not where the
   eviction literature has concentrated.

10. **The hierarchy has never been built.** §2.3 Break 1 says no shipped system demotes a KV
    entry to a *different representation* and faults it back. `[A]` high confidence. The
    cheapest falsifier is one counterexample, and finding one would be more valuable than
    most positive results in this module — it would mean the hierarchy analogy is a
    description rather than an aspiration, and half of this module would need rewriting.

11. **Nobody measures the exchange rate.** §3.6 converts recalled tokens into KV bytes and
    prefill FLOPs at a rate computable from config. As far as the survey pass could establish,
    no agent-memory paper reports its retrieval budget in these units, which means the field
    compares memory systems without normalizing the resource they actually consume. That is a
    methodology contribution available for the cost of an instrument, not an experiment.

---

## Answers to the self-check

**1.** The one-sentence argument: **a KV block has no authoritative copy to be checked
against, so a checksum can only tell you a block changed, not that it is wrong** — and since
§3.7 shows two legitimate computations of the same entry differ in bf16 under different
prefill schedules, "changed" and "corrupt" are not distinguishable by content. Detecting real
corruption would require recomputing, which is the work you cached to avoid. The narrow case
where he is right: bytes that have **left the device** — an offloaded block that crossed a
network or landed on NVMe has been exposed to a channel with its own error model, and there a
checksum protects against a transport fault rather than against a semantic one. Mooncake's
log backend carries CRC-32C records for exactly that reason. In-device, in-process KV needs no
checksum.

**2.** The KV cache and recurrent state. For the **KV cache** the statement is about policy
and mechanism: it *has* addressable entries — a block, a token's K and V — and the system
simply refuses to service their absence; `allocate_slots` returns `None` and the scheduler
preempts the whole sequence (`kv_cache_manager.py:340`). §3.8 shows why the refusal is not
laziness: the repair unit is a suffix, so a per-block fault handler would have to re-prefill
to the end of the sequence anyway. For **recurrent state** the statement is stronger and
structural: there are no entries at all. Nothing is stored per token, so "absent" is not a
state an entry can be in; a token's contribution was destroyed by a decay multiply
(`ssd_state_passing.py:80`), not relocated. The KV cache *could* signal a miss and does not;
recurrent state *has nothing that could miss*.

**3.** From §3.5 at `T = 131,072`: `BW* = c·F / (2P + 2·L·n_q·d_h·T)` = `49,152 × 20.9e12 /
7.04e9` ≈ **0.146 GB/s**, and at a more realistic 30% of peak prefill throughput, ~44 MB/s.
He is right only if the slow tier is slower than about 150 MB/s — slower than a decade-old
hard disk. What the number tells you: **his intuition is inverted.** Recompute is cheap
relative to bytes for a small model at *short* context, and the `T²` term means it gets
steadily worse as context grows, so the long-context case he is worried about is the case
where eviction is least defensible on cost grounds. The remaining honest argument for
eviction is not cost, it is that we do not have a slow tier at all on a single device — which
is a different claim and one the tier-ratio experiment is designed to test.

**4.** Going from `k=5` to `k=15` at, say, 512 tokens per passage adds 5,120 tokens per call.
The two units: **KV bytes** — `5,120 × 48 KiB = 240 MiB` of fast-tier residency, held for the
life of the request; and **prefill FLOPs** — `2 × 300e6 × 5,120 = 3.07e12` FLOPs ≈ 147 ms at
the `[M]` peak GEMM figure, likely 3× that in practice, *per call*. "Storage is cheap" is true
— the 40,000 tokens of notes are a few hundred kilobytes on disk — and irrelevant, because
the store is not the resource being consumed. §3.6: the session store has no runtime memory
cost of its own, only an exchange rate into the one resource that is scarce. A memory system
that retrieves more has less context budget left.

**5.** No — but it needs a qualifier the note does not carry, which is why §3.7 exists. The
claim is about *information*, and it is exactly right at that level: the entries are a
deterministic function of `(tokens, weights)`, they add nothing, and deleting them loses no
bits. The measurement adds that in finite precision the *bit pattern* you recover depends on
the reduction order, hence on the prefill schedule — `[M]` bit-identical in fp32 across all
three schedules, and diverging by roughly one bf16 ULP per entry, compounding with depth, in
bf16. So: **"exactly" is true in exact arithmetic and true in fp32 on this box; in bf16 it is
true up to one ULP under a schedule change.** That is not a correction to the taxonomy, it is
an engineering constraint on anything that *differences* two caches — which is precisely what
the lab's primary deliverable does, and it is why the constraint is worth writing down.

**6.** Cheapest to most expensive: **KV cache** (FLOPs; and §3.5 says those FLOPs are worth
less than a slow disk's bandwidth) ≈ **recurrent state** (FLOPs, same argument) < **retrieval
index** (embedding FLOPs over the corpus — expensive but bounded, and fully automatic) <
**weights** (the training run, plus the training data, which is usually gone) < **session
store** (the information itself; unrecoverable at any price). The inversion: **the field
spends by far the most engineering effort on the cheapest tier.** Thirty-plus KV eviction
policies, four serving systems with paged allocators in this repo's reference library alone,
and a survey classifying thirty-plus KV management systems — all for a tier where the worst
possible outcome is that you recompute something. Meanwhile the one tier with no
reconstruction path gets `core_memory_replace`, whose documented delete procedure is passing
an empty string (`base.py:263`). That mismatch is not irrational — the KV cache is where the
latency is — but it is worth seeing clearly, because it explains why the security literature
lives in the session store and nowhere else, and it tells you which of the five deserves a
review gate.

---

## Sources

**Measured on this machine (`[M]`)**

- **§3.7, Exercise B** — three-path prefill comparison, 2026-07-26, Z13 / gfx1151 / native
  Windows / `torch 2.12.0a0+rocm7.13.0a20260313`. Config: `L=8`, `T=512`, `d_model=256`, 8
  heads × 32 dims, LayerNorm + SDPA + GELU MLP, seeded (`manual_seed(1337)` for weights, `7`
  for inputs), `no_grad`, chunk size 128 for path C. Three repeats per process, three
  process invocations per configuration; all outputs byte-identical, cross-process diffs
  empty. Reported: fp32 bit-identical on all paths; bf16 default A-vs-C diverges from layer
  3 at max abs 0.0078125 (0.009765625 by layer 7), 28.7% bitwise-equal at layer 7, final
  hidden rel-L2 5.571e-3 with 44.1% bitwise equal; bf16 with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` diverges from layer 2 (hidden rel-L2 4.894e-3)
  and A-vs-B diverges at layer 5 (6.104e-05); bf16 CPU diverges from layer 3 on both
  comparisons.
- **§3.8, Exercise C** — dependency-cone probe, 2026-07-26, same box, fp32. Config: `L=6`,
  `T=256`, `d_model=256`, 8 heads × 32, perturbation at position `J=100`. Two process
  invocations, identical. Result: layer 0 → 1 changed position (100); layers 1–5 → 156
  changed positions, contiguous, 100–255.
- `ASSUMPTIONS.md` rows relied on: `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per
  arm), `kv-per-token-laguna` (192 KiB/token exactly), `laguna-heads-uniform` (refuted for
  query heads only), `gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192³),
  `large-tensor-fault-32gib`, `sdpa-is-memory-efficient`, `bf16-numerics-unproven`,
  `torch-build`, `reference-model`, `single-device-only`.
- `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier sweep.
- Code pointers: every `file:line` in §5 was opened and the named symbol confirmed on the
  named line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.

**Internal documents**

- `research/memory/memory-taxonomy.md` — the note this module teaches. §2.5 and §3.7 are
  refinements to it, both flagged in place; no claim here contradicts it.
- `research/synthesis.md` — the attribution-instrument decision that §4.2 derives
  mechanically.
- `curriculum/attention-variants-and-kv-cost.md` — the KV product, `AI = 2G/b`, and the
  windowing argument this module assumes.
- `research/reference/CODE_MAP.md` — the guided tour the §5 pointers are drawn from.

**arXiv (`[C]`)** — all resolved against the live arXiv API on 2026-07-26 via
`research/memory/citation-verification.json`.

*Taxonomy and vocabulary*
- `2607.08032` — *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction
  in LLMs and Agents* (Jul 2026). The unification proposal; three weeks old.
- `2507.03724` — *MemOS: A Memory OS for AI System* (2025). Parametric / activation /
  plaintext; the scheduler framing.
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy* (Jun 2026). The
  serving-side axes.
- `2603.04428` — *Agent Memory Below the Prompt* (Feb 2026). Cited as evidence of the
  vocabulary collision, not for its method.

*Parametric*
- `2404.05405` — *Knowledge Capacity Scaling Laws* (2024). ~2 bits/param.
- `2505.24832` — *How much do language models memorize?* (2025). ~3.6 bits/param — a
  different quantity.
- `2202.05262` — *ROME* (2022). `2210.07229` — *MEMIT* (2022).
- `2307.12976` — *Ripple Effects of Knowledge Editing* (2023).
- `2401.07453` — *Model Editing at Scale leads to Gradual and Catastrophic Forgetting* (2024).

*Recurrent state*
- `2312.04927` — *Zoology* (2023). MQAR as the capacity diagnostic.
- `2402.18668` — *Simple linear attention LMs balance the recall-throughput tradeoff* (2024).

*KV cache*
- `2306.14048` — *H2O* (2023). `2309.17453` — *StreamingLLM / attention sinks* (2023).

*Retrieval*
- `2005.11401` — *Retrieval-Augmented Generation* (2020).
- `2501.01880` — *Long Context vs. RAG for LLMs* (2025).

*Session store*
- `2310.08560` — *MemGPT* (2023). `2502.12110` — *A-MEM* (2025).
- `2605.23296` — *Parallel Context Compaction for Long-Horizon LLM Agent Serving* (2026).
  The synchronous-consolidation cost.
- `2604.16548` — *A Survey on Long-Term Memory Security in LLM Agents* (Apr 2026).
- `2606.04329` — *From Untrusted Input to Trusted Memory* (Jun 2026). Compaction as a write
  channel.
- `2407.12784` — *AgentPoison* (2024).
- `2606.15903` — *Control-Plane Placement Shapes Forgetting* (Jun 2026).
- `2604.27707` — *Contextual Agentic Memory is a Memo, Not True Memory* (Apr 2026).
- `2605.04897` — *Storage Is Not Memory* (May 2026).
