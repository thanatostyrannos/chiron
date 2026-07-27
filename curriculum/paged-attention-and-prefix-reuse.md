---
title: Paged attention and prefix reuse — the page table that has no fault handler
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost (B), tensors-and-autograd (A)
mirrors: research/memory/kv-serving-hierarchy.md
difficulty: moderate-to-hard — the code is easy, the four analogy breaks are not
time: 2.5–3.5 h reading and working the math; 4–5 h for the three exercises. Two weeks at 8 h/wk.
---

# Paged attention and prefix reuse

**Difficulty and time, honestly.** You already know page tables, block allocators, victim
caches, refcounts and shared read-only pages. That knowledge is 80% transferable and the
remaining 20% will actively mislead you, which is worse than knowing nothing. Budget 2.5–3.5
hours for sections 1–5 — most of it is *unlearning*, not learning. The exercises are the
expensive part: A is an hour, B is 1.5–2 hours, C is 1.5–2 hours plus a few minutes of GPU
time. Exercise C produces a number this lab does not have and, as far as this module's search
could establish, nobody has published for gfx1151.

This module **teaches** what `research/memory/kv-serving-hierarchy.md` **surveys**. Read that
note first if you have not. Where I refine it — three places, all from reading the code
directly this session — I say so explicitly and show the line. I found nothing in it that is
wrong.

---

## 1. What this module settles

**One:** PagedAttention does not eliminate over-reservation, it *relocates* it from a
structure costing 192 KiB per token to one costing 0.25 bytes per token, a factor of 786,000 —
and once you see it that way, the block size question becomes an optimisation you can write
down and solve in one line, whereupon it returns 2 tokens against the 16 every production system
ships — an eight-fold miss that localises exactly the cost term byte accounting cannot see. **Two:** the four places the virtual-memory analogy breaks are
not curiosities, they are the load-bearing structure: there is no fault path, eviction
granularity is the request rather than the page, the cache key is a *chain* over the whole
prefix rather than a content hash, and discarding is always legal because the bytes are
recomputable. **Three:** prefix reuse's headline "hit rate" is three different numbers that
diverge by construction, and the divergence has a closed form — the FLOPs actually avoided
interpolate between `φ` and `φ²` in the hit fraction as context grows, which is why a 6.25%
prefix match on a long prompt buys 1.98% of the compute.

---

## 2. Theory in plain language

### 2.1 What paging replaced, and why the replacement was obvious in hindsight

Before `[C]` 2309.06180 (Sep 2023), a serving system reserved a contiguous KV buffer of
`max_seq_len` per sequence at admission time. It had to: attention kernels indexed the cache
with a stride, so `K[t]` had to be at `base + t·stride`. A sequence that would generate 300
tokens still reserved room for 128,000.

Put our reference model's numbers on it. Laguna S 2.1 costs **192 KiB per token** of KV
`[M]` (`ASSUMPTIONS.md → kv-per-token-laguna`; derived in `attention-variants-and-kv-cost.md`
§3.3). Reserving a 128k window is `131,072 × 196,608 B = 24.0 GiB` **per sequence**. Against
our `[M]` ≥62 GiB fast tier (`notebook/uma-carveout-controls-fast-tier.md`, single run per
arm), that is a maximum concurrency of **two**.

PagedAttention's move is the one you would make: fixed-size blocks of `B` tokens, a per-
sequence table mapping logical block index to physical frame, and a kernel taught to walk the
table. Internal fragmentation drops to under one block per sequence. External fragmentation
disappears because all frames are the same size. This is `malloc` beating static partitioning,
and it is the same argument that got made about disk extents in the 1980s.

> **Systems bridge.** `req_to_blocks` is a page table. List index = virtual page number,
> `KVCacheBlock.block_id` = physical frame number
> (`memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`). Worker-side it is flattened
> into a dense `[max_num_reqs, max_num_blocks_per_req]` int32 matrix
> (`memory/vllm/vllm/v1/worker/block_table.py:81`) and memcpy'd to the device every step. The
> attention kernel does the translation itself, one load per KV tile:
> `block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE`
> (`memory/vllm/vllm/v1/attention/ops/triton_unified_attention.py:424`). A software page walk
> with no MMU, no TLB, and no hardware assist of any kind.
>
> Two structural differences before we get to the interesting breaks. First, **the page size
> is denominated in tokens, not bytes** — `page_size_bytes` is a *derived property* of the
> layer's spec (`memory/vllm/vllm/v1/kv_cache_interface.py:109`), and a hybrid SWA/global model
> instantiates several specs with different frame geometry over one physical pool
> (`kv_cache_interface.py:539` for `SlidingWindowSpec`). No single-address-space page table
> ever has to model several page sizes at once. Second, **the translation is not per-access**;
> FlashInfer copies the page table to the host, plans a work partition once, and reuses it for
> every layer of the forward pass. That is a query planner, not an MMU.

### 2.2 What prefix reuse replaced

Once blocks exist, two sequences with the same leading tokens can *point at the same physical
frame*. Nothing else in the stack changes. That is automatic prefix caching, and it is the
single highest-leverage optimisation in production serving, because real traffic is
overwhelmingly repeated system prompts, repeated few-shot blocks, and multi-turn conversations
that resend their own history.

`[C]` 2506.02634 (USENIX ATC'25) is the only public characterisation of this at cloud-provider
scale that this module's search could find. Its reported findings: reuse is heavily skewed
across request categories; single-turn and multi-turn reuse are comparably valuable; and the
cache size needed for a near-ideal hit ratio is *moderate*. That last one matters to us — it
says hit rate saturates, so a capacity-only story is incomplete and routing is the real lever
`[C]` 2602.06502, `[C]` 2605.08581, `[C]` 2607.02525.

> **Systems bridge.** A shared prefix block is a read-only page shared across tenants — page
> dedup, KSM, a shared library's text segment. The refcount and the free list are exactly the
> structures you would build: `FreeKVCacheBlockQueue` is an intrusive doubly-linked list
> threaded through the block objects so a hit can splice one out in O(1)
> (`memory/vllm/vllm/v1/core/kv_cache_utils.py:184`), and `touch` is the splice
> (`memory/vllm/vllm/v1/core/block_pool.py:702`).
>
> **First hazard, and it is the one your instincts will not raise:** a hit is *observable in
> latency from outside the system*. TTFT for a cache hit is measurably lower than for a miss,
> which means an attacker who can time your API can binary-search another tenant's prompt one
> token at a time `[C]` 2411.18191 (InputSnatch). This is now a small subfield with three
> shipped defences in the last year — `[C]` 2508.08438 (SafeKV, selective sharing),
> `[C]` 2603.10726 (PrefixWall, owner-tagged blocks), `[C]` 2605.23640 (CachePrune, token-level
> privacy-aware sharing). The isolation/efficiency frontier is live and there is no agreed
> metric on it. The mechanism for isolation already exists in both engines: vLLM folds
> `extra_keys` into the block hash (`kv_cache_utils.py:596`) and SGLang's `extra_key`
> namespaces the tree "like an ASID" for LoRA ids and cache salts
> (`memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355`). What is unsettled is the
> *policy* of when to salt.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

| Symbol | Reads as |
|---|---|
| `k` | KV bytes retained per token, whole stack — `2·L·H_kv·d·b` from the prereq module |
| `B` | **block size in tokens** (vLLM's `block_size`; FlashInfer's `page_size`). Not batch. |
| `S` | tokens in one sequence's context |
| `S_max` | the reservation length: model max context, or the admission cap |
| `R` | concurrent requests resident in the pool |
| `M` | usable fast-tier bytes — for us `[M]` ≥62 GiB |
| `φ` | prefix hit fraction, `S_hit / S` |
| `ρ` | attention-to-linear FLOPs ratio of a prefill at length `S` (defined in §3.4) |
| `m` | number of sequences sharing one prefix |
| `P` | model parameters (active, for MoE) |
| `d` | head dimension; `b` bytes per element; `H_kv` KV heads; `L` layers |

Notation collision kept deliberately, because the reference code uses it: `B` is the **block
size in tokens** throughout this module. The prereq module used `B` for batch. Here batch is
`R`.

### 3.2 Fragmentation, and where it actually went

**Reservation.** Bytes per sequence = `k · S_max`. Concurrency `R = ⌊M / (k·S_max)⌋`.

**Paging.** Bytes per sequence = `k · B · ⌈S/B⌉`. The waste is bounded by `k·(B−1)` and, for a
sequence length uniform modulo `B`, averages `k·(B−1)/2`.

Worked at Laguna's shape, `M` = 62 GiB, mean live length 2,048 tokens, `B` = 16. Arithmetic
over two `[M]` inputs (`k` and `M`), not a measurement:

| | All-global counterfactual | Actual 3:1 hybrid (`w`=512) |
|---|---|---|
| Reserved per sequence at `S_max`=131,072 | 24.00 GiB | 6.07 GiB |
| Concurrency under reservation | **2** | **10** |
| Paged per sequence at `S`=2,048 (+ waste) | 385.4 MiB | 168.0 MiB |
| Concurrency under paging | **164** | **377** |
| Improvement | **82×** | **37.7×** |

Two things to take from the second column, both first-order for Proteus. The hybrid's windowed
layers contribute a **fixed 72 MiB from token 512 onward that paging cannot recover** — it is
genuinely resident, not over-reserved. So a hybrid model is *already* four times less wasteful
under reservation and correspondingly gains less from paging. **Windowing and paging are
partially substitutable capacity mechanisms**, and if you measure paging's benefit on an
all-global toy model you will overstate it by more than 2× for our reference architecture.

**Now the relocation, which is the sentence to remember.** The block table is not free. At
`R`=256 concurrent requests, `S_max`=131,072, `B`=16, the dense device-side matrix is

```
256 rows × ⌈131072/16⌉ entries × 4 B  =  256 × 8192 × 4  =  8,388,608 B  =  8 MiB
```

copied host-to-device **every decode step** (`worker/block_table.py:81`). And notice: that row
is sized to `S_max` regardless of how long the sequence actually is. At a mean length of 2,048
tokens, 98.4% of every row is unused — *the exact over-reservation pattern paging was
introduced to eliminate, reproduced faithfully in the metadata*.

The difference is unit price:

```
KV per token of context            = 196,608 B
block-table per token of context   = 4 B / 16 tokens = 0.25 B
ratio                              = 786,432 : 1
```

**PagedAttention did not remove over-reservation. It moved it to a structure 786,000× cheaper
per token.** That is a complete and honest description of the mechanism, and it is more useful
than "it eliminates fragmentation," because it tells you where to look when the cheap structure
stops being cheap — which is exactly what happens as `B → 1`.

### 3.3 The block size, derived — and why the derivation is wrong in an instructive way

Write the per-sequence overhead of a choice of `B` as capacity in bytes. Two terms, one rising
and one falling:

```
O(B)  =  k·(B−1)/2   +   4·S_max/B
         ^^^^^^^^^^       ^^^^^^^^^^
         internal          block-table row,
         fragmentation     4 bytes per entry
```

Differentiate, set to zero:

```
dO/dB = k/2 − 4·S_max/B² = 0     ⟹     B* = √(8·S_max / k)
```

The classic square-root / economic-order-quantity shape: the optimum is where a linear cost
crosses a reciprocal one. Substitute:

- **Laguna** (`k` = 196,608 B, `S_max` = 131,072): `B* = √5.333 =` **2.31**
- **A 300M dense ablation arm** (`k` = 24,576 B, `S_max` = 32,768): `B* = √10.67 =` **3.27**

Check it discretely at Laguna's shape:

| `B` | fragmentation | block-table row | total |
|---|---|---|---|
| 1 | 0 | 512 KiB | 512 KiB |
| **2** | 96 KiB | 256 KiB | **352 KiB** |
| 4 | 288 KiB | 128 KiB | 416 KiB |
| 16 | 1,440 KiB | 32 KiB | 1,472 KiB |

**The byte-optimal block size is 2 tokens. Every production system ships 16.** The shipped
value is 4.2× worse on the only cost this model accounts for.

That is not a bug in the systems; it is a missing term in the model. The term is **kernel
throughput**: at `B`=2, one page holds 2 tokens, so a KV tile load is a 2-element gather with a
page-table load in front of it, and the GPU spends its time on address arithmetic and short
bursts rather than on streaming. vLLM's own paper says exactly this — too small and you lose
GPU parallelism, too large and you lose sharing granularity. What the byte model buys you is
the knowledge that **the block size is set entirely by a hardware-dependent throughput effect
that byte accounting cannot see.** Which is a testable claim, on our hardware, and it is
Exercise C.

Note where the second cost term lives physically:

```
run length seen by a kernel loading one KV head's slice of one page
    HND layout  [pages, n_kv_heads, page_size, head_dim] :  B · d · b   bytes  ← grows with B
    NHD layout  [pages, page_size, n_kv_heads, head_dim] :      d · b   bytes  ← independent of B
```

Both layouts are documented in one docstring at
`memory/flashinfer/flashinfer/page.py:403`, and the enum that selects between them is two lines
(`memory/flashinfer/flashinfer/utils.py:50`). **The prediction that falls out: the block size
should matter for bandwidth under HND and not under NHD.** At `d`=128 and bf16, NHD's run is
256 B — already at or above a typical burst — so NHD should be flat in `B`. This is a sharp,
falsifiable, layout-level claim and I have not found it measured anywhere.

### 3.4 Prefix hit rate is three numbers, and the third has a closed form

`[C]` CODE_MAP and the mirror note both warn that a 100% prefix match never skips 100% of the
work. Here is the whole chain of caps, from the code:

1. `max_cache_hit_length = request.num_tokens − 1` — you need one forward pass to get logits
   (`memory/vllm/vllm/v1/core/kv_cache_manager.py:255`).
2. The match loop walks full blocks from token 0 and breaks at the first miss
   (`memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:708`).
3. The result is floored to block alignment.

So `S_hit = ⌊(S−1)/B⌋ · B` for an exact duplicate, and `φ = S_hit/S`.

Now the part that is not in the note. Prefill FLOPs are **not linear in tokens**:

```
prefill_flops(S)  =  c₁·S  +  c₂·S²

  c₁ = 2·P                    linear term: every parameter, two FLOPs, per token
  c₂ = 2·L·n_q·d_h            quadratic term: attention scores and the value sum,
                              summed over positions (4·L·n_q·d_h·S²/2)
```

Skipping the first `S_hit` tokens avoids `c₁·S_hit + c₂·S_hit²` — the *quadratic* part avoided
goes as the **square** of the hit length, because the skipped tokens are the ones with the
shortest contexts. Define `ρ = c₂·S/c₁`, the attention-to-linear ratio of this prefill. Then:

```
                    c₁·φS + c₂·φ²S²        φ + ρ·φ²
flops_avoided(φ) = ─────────────────  =  ────────────
                     c₁·S + c₂·S²           1 + ρ
```

**Read the two limits.** At short context (`ρ → 0`) the FLOPs avoided equal the tokens skipped,
`φ`. At long context (`ρ → ∞`) they fall to `φ²`. Prefix caching's compute saving is *sublinear
in the prefix fraction, and — at a fixed hit fraction — the deficit grows with context length.*
That qualifier is load-bearing; see the next paragraph, and Exercise B, for what happens when the
hit fraction moves too.

Worked, at the plausible 300M ablation shape (`L`=24, `n_q`=16, `d_h`=64, `P`=3e8, so
`c₁`=6e8, `c₂`=49,152), for a 2,048-token shared system prompt in front of a 30,720-token
unique document (`S` = 32,768, `B` = 16):

```
ρ  = 49,152 × 32,768 / 6e8               = 2.684
φ  = 2,048 / 32,768                      = 6.25 %
blocks matched   = 128 of 2,048          = 6.25 %
tokens skipped   = 2,048 of 32,768       = 6.25 %
FLOPs avoided    = (0.0625 + 2.684×0.0625²)/3.684  = 1.98 %
```

**Three metrics, a 3.16× spread, all correct.** This is arithmetic over a hypothetical config,
not a measurement — but the *form* is exact and it explains a production complaint you will
hear: prefix cache dashboards look excellent in tokens and disappointing in TTFT at long
context. They are both telling the truth.

**Where the divergence is worst — and it is not where you would guess.** Subtract the two
metrics and the algebra collapses:

```
absolute gap   =  φ − flops_avoided  =  ρ·φ·(1−φ) / (1 + ρ)          [percentage points]
ratio          =  φ / flops_avoided  =  (1 + ρ) / (1 + ρ·φ)          [× overstatement]
```

Both are increasing in `ρ`, so **both get worse with context length at a fixed hit fraction** —
that part is intuitive. The `φ` dependence is not. The **absolute** gap is `φ(1−φ)`-shaped: it
vanishes at both a cold miss and an exact duplicate and **peaks at `φ = 0.5`**. The **ratio**
does the opposite: it is worst as `φ → 0`, bounded above by `1 + ρ`, and falls to 1 at an exact
duplicate.

Read that as an operational rule. **An exact-duplicate prompt is the case where all three metrics
agree, and it is the case every prefix-cache demo shows you.** The regime where they disagree is
the half-cached long prompt — RAG with a shared instruction header and a long retrieved document,
or an agent whose scratchpad has outgrown its system prompt. Which is the regime that pays for
the feature.

House consequence, stated in the mirror note §9 and now with a formula behind it: **report hit
rate three ways or not at all.** And because the third is predictable from the first, a
measured FLOPs-avoided that departs from `(φ + ρφ²)/(1+ρ)` is a *harness bug detector*, which is
worth more than the metric.

### 3.5 Sharing is the only thing in the stack that raises decode arithmetic intensity

The prereq module derived decode-attention intensity as `2·H_q/(H_kv·b)` = `G` at bf16,
invariant under batch, because each sequence owns its own cache: batching multiplies bytes and
FLOPs equally. There is exactly one exception, and it is this module's subject.

Let `m` sequences share a prefix of `S_p` tokens.

```
Per-sequence block-table walk (what a stock paged decode kernel does):
    bytes = m · k · S_p          FLOPs = m · 4·L·H_q·d·S_p       ⟹  I = 2·H_q/(H_kv·b)

Cascade / shared-prefix kernel (read the shared pages once for all m queries):
    bytes =     k · S_p          FLOPs = m · 4·L·H_q·d·S_p       ⟹  I = m · 2·H_q/(H_kv·b)
```

**Sharing multiplies decode arithmetic intensity by the sharing factor.** At Laguna's global
layers (`I` = 6) and our `[M]` ≈105 FLOP/byte ridge (`ASSUMPTIONS.md`, a ratio of two single-run
numbers — treat as ±30%), you cross the ridge at **m ≈ 18 sequences sharing a prefix**. That is
an entirely ordinary number for a chat service with a system prompt. Decode on the shared
portion becomes compute-bound.

This is not hypothetical plumbing. `memory/flashinfer/flashinfer/cascade.py:226`
(`MultiLevelCascadeAttentionWrapper`) implements it: the top level is planned with
`qo_indptr = [0, batch_size]` so all `m` queries attend the shared pages in one prefill-shaped
matmul, the per-sequence suffixes run at the bottom level, and the two partial results are
combined by `merge_state` (`cascade.py:42`), which merges attention outputs given their
log-sum-exp values — the same online-softmax identity FlashAttention uses to split over keys.

**The break, and it is a bad one:** a stock vLLM paged decode does *not* do this. Each
sequence's kernel invocation walks its own block table and re-reads the shared blocks. Prefix
sharing saves **capacity** and **prefill compute**; it saves **decode bandwidth only if the
kernel has an explicit cascade path**. Every capacity number in §3.2 is real without cascade;
every intensity number in this section requires it. Conflating the two is the most common error
in vendor material on prefix caching, and there is no cascade kernel in our ROCm stack.

---

## 4. The four breaks

The bridge is virtual memory, page tables, and a shared read-only cache tier. It carries you a
long way. Here is precisely where it stops.

### Break 1 — There is no fault path. A miss is not serviced; the request is restarted.

An OS page fault traps, fills the page from a backing store, and **resumes the faulting
instruction**. Nothing is lost. That mechanism does not exist here, at any layer.

Follow it in the code. The scheduler asks for slots; `allocate_slots` returns `None`
(`memory/vllm/vllm/v1/core/sched/scheduler.py:566`). That is not a fault. The loop below it
pops a victim off the running queue (`scheduler.py:603`) and calls `_preempt_request`
(`scheduler.py:1203`), which frees every block the victim owns and then does this:

```python
request.num_computed_tokens = 0            # scheduler.py:1216
```

**The victim's progress is reset to zero and it goes back on the waiting queue.** A request
29,000 tokens into a 32,000-token prefill loses all 29,000. The unit of loss is not a page; it
is everything.

The same absence holds one layer down and in the other engine. SGLang's plain `RadixCache` has
no backing store at all — eviction frees KV slots and the only "reload" is a full prefill
(`radix_cache.py:565`). FlashInfer's page table has no present bit: a page is in the table or it
is unrepresentable. llama.cpp's interleaved-SWA cache has no promote or demote path between its
two tiers.

> **The refinement — and it is the best thing in this module.** There *is* a recovery path, and
> it is an accident. `_preempt_request` frees the victim's blocks via `free_blocks`
> (`block_pool.py:719`), which only decrements the refcount and pushes the block back on the
> free queue **with its hash-table entry still live**. Actual eviction happens later and lazily,
> inside `get_new_blocks → _maybe_evict_cached_block` at the moment of reallocation
> (`block_pool.py:647`, `:679`). And `free` walks the request's blocks in reverse so tail blocks
> sit nearest the front of the eviction order and *head* blocks — the reusable prefix — survive
> longest (`kv_cache_manager.py:563`, and the two-tier reclaim at `block_pool.py:741`).
>
> So when the preempted request is rescheduled, `get_computed_blocks` runs the ordinary prefix
> match (`kv_cache_manager.py:225`) and usually resurrects most of its own KV via `touch`
> (`block_pool.py:702`).
>
> **The fault handler is the victim cache, wired up backwards, best-effort, with no guarantee
> and no accounting.** How often does a preempted request recover, and how many tokens does it
> recover? No shipped system reports it. `prefix_cache_stats.record` does carry a `preempted`
> flag (`kv_cache_manager.py:222`), so the *ingredients* are there — nobody assembles the
> number. Cheapest genuinely-novel measurement in this module.

### Break 2 — Eviction granularity is the request, not the page. And in SGLang it is a subtree.

An OS evicts *a page*, chosen by approximate recency, and the working set degrades gracefully.
Neither engine can do that.

**vLLM: the request.** `_preempt_request` (`scheduler.py:1203`) frees the whole allocation. There
is no "page out the cold quarter of this sequence and keep running." Every capacity-planning
intuition that depends on graceful degradation under pressure is unavailable — the system holds
the whole sequence or drops it.

**SGLang: leaves of a tree, topologically constrained.** `evict()` builds a heap from
`evictable_leaves`, an incrementally maintained set, and peels the frontier inward
(`radix_cache.py:565`). A node is only in that set if it has no unevicted children and no lock
(`_update_leaf_status`, `radix_cache.py:790`). Consequences that no LRU intuition predicts:

- **A hot child keeps a cold parent resident indefinitely.** The parent is not a leaf, so it is
  not a candidate, no matter how old.
- **`inc_lock_ref` walks all the way to the root** (`radix_cache.py:594`), so one in-flight
  request pins an arbitrarily deep chain. This is a pinned dentry chain, not a refcounted page.
- **A "lookup" mutates the tree.** `_match_prefix_helper` calls `_split_node` when a match ends
  mid-node (`radix_cache.py:650`, `:676`), which re-parents a subtree and `.clone()`s two
  tensors. The docstring says so plainly (`radix_cache.py:355`). Read cost is not read-only, and
  it is not concurrency-safe.

**What the free list actually is.** In vLLM there is no separate victim cache: the free list
*is* the cache. Blocks sit on `FreeKVCacheBlockQueue` with contents and hashes intact
(`kv_cache_utils.py:184`), matchable, until someone reallocates them. So three quantities that a
dashboard will happily conflate are all different:

1. blocks with `ref_cnt > 0` — in use by a live request
2. blocks with `ref_cnt == 0` but still in `cached_block_hash_to_block` — free *and* matchable
3. blocks with no hash — the only ones that are genuinely "free memory"

Any panel labelled "cache occupancy" showing one of these is lying about the other two.

> **Where page-granular eviction does exist, and why it is a different contract.**
> `[C]` 2509.04377 (PagedEviction) prunes *within* a sequence at block granularity. But that is
> an accuracy-degrading compression policy — it decides some tokens will never be needed — not a
> capacity mechanism that preserves the computation. Do not file it under the same heading; the
> curriculum's compression module owns it.

### Break 3 — The cache key is a chain over the whole prefix, not a content hash. This is not a choice.

```python
# memory/vllm/vllm/v1/core/kv_cache_utils.py:596
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    ...
    return BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

`get_request_block_hasher` (`kv_cache_utils.py:691`) chops the prompt into `hash_block_size`
chunks and chains them, resuming from `len(request.block_hashes)`. Consequences a cache engineer
feels immediately: the same 16 tokens at a different offset are a **different key**; one changed
token at position 0 invalidates **every** downstream hash; there is no associative match; and the
match loop breaks at the first miss because a later hit is impossible by construction
(`single_type_kv_cache_manager.py:708`).

And vLLM **deliberately refuses de-duplication**. The `BlockHashToBlockMap` docstring
(`block_pool.py:47–51`) says that when a block becomes full and is cached, no check is made for
an identical existing block, because collapsing them would change an already-issued `block_id`
and block-table rows must be strictly append-only. KSM is off the table by invariant.

> **Why this is not conservatism.** The reflex is "they could content-hash and chose not to."
> They could not. `K[i]` and `V[i]` in layer `ℓ` are a function of the residual stream at
> position `i` in layer `ℓ−1`, which is a function of *every token at position ≤ i* through the
> attention below it — plus RoPE at the **absolute** position `i`. A KV block's contents depend
> on its history and on where it sits. **The chain is not a cheap approximation of a content
> hash; it is the only sound key.** This is the deepest break in the module and the one a
> storage engineer is most likely to get backwards.

**Refinement 1 (from the code, this session).** "Breaks at the first miss" is now slightly
stronger than that. After phase 1 stops, `FullAttentionManager` runs a phase-2 probe into the
*first non-full block*, testing its interior hash boundaries high-to-low
(`single_type_kv_cache_manager.py:718`). So the hit can land on a sub-block boundary. The
mirror note's statement is right about the structure; the granularity is finer than it says.

**Refinement 2, and it is a real one.** A sliding-window layer's manager searches **right to
left** and accepts a run of `⌈(w−1)/B⌉` contiguous blocks *anywhere* in the sequence
(`single_type_kv_cache_manager.py:871`, loop at `:919`), filling everything before the run with
`null_block`. **A windowed layer can get a middle-of-sequence hit.** That does not contradict the
chain — it probes the same chained `block_hashes[i]`, so the whole prefix is still proven equal.
What changes is the *residency* requirement: a windowed layer only needs the last `w` tokens
present, because everything older is architecturally unreadable.

The unifying statement, which is a Proteus design consideration and not just a vLLM detail:

> **The key is always chained over the whole prefix. What the layer type sets is how much of
> that prefix has to be resident.** Full attention: all of it. Sliding window: `w` tokens.
> Recurrent/SSM state: all of it again, because the state is a function of everything and cannot
> be reconstructed from a suffix.

**The frontier is trying to break the chain.** Three live attempts, all attacking the same
premise: `[C]` 2405.16444 (CacheBlend, 2024) reuses precomputed KV of non-prefix chunks and
selectively recomputes a small token subset to stitch them, reporting 2.2–3.3× lower TTFT;
`[C]` 2604.13226 (KV Packet, Apr 2026) treats documents as immutable packets with trainable
adapters distilled to bridge context discontinuities; `[C]` 2607.17715 (C²KV, Jul 2026) learns an
explicitly position-agnostic composable KV manifold, claiming up to 17× speedup at long context.
All three are author-reported, none independently replicated, and all carry quality caveats.
Present as contested. But note what a success here would mean: **content-addressable KV**, and
with it dedup, associative lookup, and mid-sequence sharing — the whole toolbox the chain
currently forecloses.

**Hybrid models pay for the chain twice.** A single-block probe packs `(block_hash, group_id)`
into one key and returns `None` if **any** group in that spec group misses (`block_pool.py:198`).
Above that, `HybridKVCacheCoordinator.find_longest_cache_hit` runs an **iterative fixed-point
reconciliation** (`memory/vllm/vllm/v1/core/kv_cache_coordinator.py:685`): each attention type
either accepts the current candidate hit length or reduces it, and any reduction restarts the
sweep, converging because the length decreases monotonically. The reconciled hit is the **min
over groups**. For our 12-global + 36-sliding GSSS reference model `[M]`
(`ASSUMPTIONS.md → reference-model`), that is a first-order property of the arm, not a footnote.

> **And here is the finding.** The coordinator computes the cost of the reconciliation
> explicitly:
>
> ```python
> num_uncached_common_prefix_tokens = longest_hit_length - hit_length   # kv_cache_coordinator.py:813
> ```
>
> That is literally "how many tokens of shared prefix some group had cached and the
> reconciliation threw away." The mirror note's open question 5 asks how much hit rate the
> sliding group costs and says it appears unmeasured in public. **The quantity exists in
> production code today** — it is computed to *pin* the junction against retention sweeps, i.e.
> as a control-loop input, not as a reported metric. That is a refinement of the note, not a
> contradiction, and it makes the experiment cheaper than the note assumed: the instrument is
> already written, it just has no dashboard.

### Break 4 — Discarding is always legal, because the bytes are recomputable.

Evicting a KV block is never data loss. It is a recompute. There is no durability contract, no
writeback requirement, no fsync, no torn-write problem, no replication factor. Mooncake's
`offload_force_evict` throws bytes away rather than block on writeback `[C]` CODE_MAP,
`[C]` 2407.00079. **Every durability guarantee you have ever engineered is optional here**, and
the correctness argument for any Mnemosyne policy is not "did we lose data" but "did we spend
more recomputing than we saved."

Three consequences, in increasing order of how much they should worry you.

**(a) The economics replace the integrity argument.** From the mirror note §5: loading a prefix
from a slower tier beats recomputing it when `BW > F·k/(2·P)`, and `S_p` cancels. At our `[M]`
20.9 TFLOPS that threshold is ≈0.24 GB/s for Laguna's shape and ≈0.86 GB/s for a 300M arm — any
real tier clears it by two orders of magnitude. So "store and reload" always wins over
"recompute" *when a tier exists*. In vLLM's local pool there is no tier, so discard means
recompute, and the question is only how often.

**(b) Because loss is legal, nobody accounts for it.** No shipped system counts discards that
were later recomputed, or the token-cost of doing so. This is precisely the attribution gap this
lab claims as its edge, and `[C]` 2607.02574 (Jun 2026) names measurement gaps of exactly this
kind across the whole field.

**(c) Legality of loss is why nobody checks for *corruption*.** A storage tier has checksums
because loss must be detected. A KV cache has none because loss is legal — but silent corruption
is a completely different failure from loss, and the legality of the one is the historical reason
nobody built the check for the other. `[C]` 2604.17249 (Apr 2026, SECRYPT'26) flips bits in
shared KV blocks and finds **13 of 16 bf16 bit positions produce coherent but altered output,
indistinguishable from a legitimate response**; damage is confined to requests using the affected
prefix and accumulates linearly without decay. A cheap checksum bounds it to one batch.

For us that lands twice. `ASSUMPTIONS.md → bf16-numerics-unproven` is still **untested**, and the
Hardware Validation Gate has not run. If our bf16 path silently perturbs a KV block, the shared-
prefix mechanism *propagates* the perturbation to every request that matched that prefix, and by
2604.17249's finding it will not look like a fault. A cache with no integrity check on hardware
with unproven numerics is a genuinely bad combination, and it is our combination today.

### 4.1 The breaks, consolidated

| Systems instinct | What actually happens | Where to see it |
|---|---|---|
| A miss faults and fills | No fault path. `allocate_slots` → `None` → preempt, `num_computed_tokens = 0` | `sched/scheduler.py:566`, `:1216` |
| …but the OS resumes the instruction | The request restarts from token 0 and hopes the victim cache saves it | `kv_cache_manager.py:225` |
| Eviction granularity is the page | The **request** (vLLM) or the **leaf subtree** (SGLang) | `scheduler.py:1203`, `radix_cache.py:565` |
| Free list ≠ cache | Same intrusive list. Freeing is not evicting; eviction is lazy at reallocation | `kv_cache_utils.py:184`, `block_pool.py:679` |
| Cache key is content-addressed | A **chain** over the whole prefix — and it is the only sound key | `kv_cache_utils.py:596` |
| Identical pages get deduped | Deliberately refused, to keep block tables append-only | `block_pool.py:47` |
| A read is read-only | `match_prefix` splits nodes and clones tensors | `radix_cache.py:355`, `:676` |
| One page size | One per layer type; hybrid runs several geometries over one pool | `kv_cache_interface.py:109`, `:539` |
| Losing data is a bug | Losing data is a **cost**. Discard is always legal | Mooncake `offload_force_evict` |
| …so integrity is handled | No checksums anywhere; 13/16 bf16 bit flips are silent | `[C]` 2604.17249 |
| Sharing a page saves reads | It saves capacity and prefill; it saves decode bandwidth only under a cascade kernel | `flashinfer/cascade.py:226` |

### 4.2 Contested, and left contested

**Is paging even the right primitive?** `[C]` 2405.04437 (vAttention, ASPLOS'25) argues no: GPU
virtual-memory APIs decouple virtual from physical allocation, giving the same defragmentation
while keeping the cache *virtually contiguous*, so stock attention kernels need no rewrite —
reported up to 24% faster prefill and matching the best paged decode. `[C]` 2607.02574 (Jun 2026)
still lists this as unresolved. Do not read vLLM's dominance as evidence; it is evidence about
ecosystem gravity. The meta-point is the interesting one: "of course you page" is the systems
instinct *under dispute*, and it is disputable only because GPU attention kernels, unlike CPU
loads, must be taught about non-contiguity by hand.

Related and newer: `[C]` 2601.13631 (ContiguousKV, Jan 2026) argues the granularity of KV
management should be aligned to algorithmic semantics rather than to a fixed page, reporting
3.85× on the re-prefill phase against an offloading baseline.

---

## 5. Why it matters for Proteus and Mnemosyne

**Mnemosyne's cache interface must be page-table-shaped from the first commit, and on this
machine that may not even be a choice.** `[M]` `ASSUMPTIONS.md → large-tensor-fault-32gib`:
single tensors ≥32 GiB hang the GPU silently at 0% CPU. A 62 GiB KV pool therefore **cannot** be
one contiguous tensor; it must be built from ≤31 GiB shards. That is paged allocation, forced on
us by a driver bug rather than chosen for fragmentation. It is also, as the mirror note's open
question 8 notes, an amusing empirical vindication of PagedAttention over vAttention *on this
platform specifically* — and it is a confound to declare in any arm that claims to speak to the
paging-versus-VMM debate generally.

**Copy SGLang's policy surface wholesale, then widen it twice.** The entire replacement-policy
API is one method (`memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:10`), and LRU, LFU,
FIFO, MRU, FILO, priority-aware and segmented-LRU all fit behind it as tuple-valued comparators
in 65 lines (`:16` through `:49`). That is the cleanest pluggable-policy interface in the
reference library. But `get_priority(node)` alone is under-specified for our purposes: it cannot
express Mooncake's lease expiry (a TTL renewal, not a touch bit) or vLLM's refcount pinning. A
Mnemosyne policy interface that cannot express all three is not general enough to be a
contribution.

**Instrument the three things nobody instruments.** In priority order, cheapest first:

1. **Hit rate three ways** — blocks matched, tokens skipped, FLOPs avoided — with the predicted
   third computed from `(φ + ρφ²)/(1+ρ)` alongside the measured one. The residual is a harness
   check, free.
2. **Preemption recovery rate** — of the `num_computed_tokens` a preempted request lost, how many
   came back from the prefix cache on reschedule. The `preempted` flag already rides on
   `prefix_cache_stats.record` (`kv_cache_manager.py:222`); nobody joins it to the recovery.
3. **Regretted discards** — blocks evicted that were subsequently recomputed, and the token cost.
   Absent from the literature, named as a gap by `[C]` 2607.02574, and the attribution metric the
   lab's synthesis identifies as its deliverable.

**The hybrid reconciliation is a named Proteus arm and it is now cheaper than we thought.** Build
a matched-parameter pair — one all-global control, one 3:1 SWA/global at `w`=512, mirroring
Laguna `[M]` — and measure `num_uncached_common_prefix_tokens` (`kv_cache_coordinator.py:813`) as
a function of the ratio. The prediction is that the sliding group's mid-sequence right-to-left
match (`single_type_kv_cache_manager.py:871`) makes it *more* cacheable than the global group in
some regimes and less in others, and the min-over-groups reconciliation means whichever is worse
sets the arm's hit rate. Two arms, one metric already computed in the engine.

**Page size is a config field, and it is per-layer-type.** Because `page_size_bytes` derives from
the layer's spec, a hybrid Proteus config has more than one. `attention-variants-and-kv-cost.md`
§4 established that a cost model keyed on top-level `num_attention_heads` mis-predicts 75% of
Laguna's layers; the same is true of a cost model with one block size. Build it per-layer-type
from the start; retrofit is worse.

**One thing we cannot do, stated so it does not get attempted.** `[M]`/`[C]`
`ASSUMPTIONS.md → single-device-only` — distributed collectives are incomplete on gfx1151. No
prefill/decode disaggregation arm, no distributed prefix routing, no Mooncake-shaped store. What
we *can* run, because they are single-pool: chunked prefill `[C]` 2403.02310, the phase asymmetry
itself, and every policy question in this module.

---

## 6. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Every pointer below was opened and the named symbol confirmed
on the named line on 2026-07-26 against the revisions in `PROVENANCE.md`.

### 6.1 The page table, in three places at once

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97` | `req_to_blocks` — the page table on the scheduler side. A `defaultdict[str, list[KVCacheBlock]]`. List index is the logical block, `block_id` is the frame. That is all it is. |
| `memory/vllm/vllm/v1/worker/block_table.py:81` | The device-side materialisation: a dense `[max_num_reqs, max_num_blocks_per_req]` int32 buffer. Read line 79 above it too — the row length is `S_max`-sized, which is where §3.2's relocated over-reservation lives. |
| `memory/vllm/vllm/v1/attention/ops/triton_unified_attention.py:424` | The translation, inside the kernel: one `tl.load` of `block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE` per KV tile. This is the software page walk. Compare the cost of this line to a hardware TLB hit and you have the whole argument for vAttention. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:109` | `page_size_bytes` as an abstract property of the *spec*, not a constant. Then `:539` (`SlidingWindowSpec`) and `:567` (`max_admission_blocks_per_request`) — the per-request block cap that makes the windowed tier bounded. |
| `memory/sglang/python/sglang/srt/mem_cache/memory_pool.py:251` | `ReqToTokenPool` — the same idea one level finer: a dense `[max_reqs+1, max_context_len]` int32 tensor mapping (request slot, position) → KV slot. Note the padding row at index 0 that absorbs CUDA-graph dummy writes. |
| `memory/flashinfer/flashinfer/page.py:326` | `get_seq_lens` — three lines, `(num_pages−1)·page_size + last_page_len`, which is internal-fragmentation accounting made executable. Reconstruct §3.2's waste term from it. |
| `memory/flashinfer/flashinfer/page.py:403` | `append_paged_kv_cache` — the docstring states both physical layouts explicitly. This is the source for §3.3's run-length claim and the specification Exercise C tests. |
| `memory/flashinfer/flashinfer/utils.py:50` | `TensorLayout` — the entire layout abstraction is a two-value enum passed to the kernel as an int. There is no richer description anywhere, which is why the layout is a physical requirement and not a view. |

### 6.2 The absent fault handler, and the accidental one

Read these five in order; the story only works in sequence.

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/sched/scheduler.py:566` | `allocate_slots(...)` inside a `while True`. If it returns `None`, control falls to the preemption branch. **This is the entire miss path.** |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:603` | `preempted_req = self.running.pop()` — the victim, under the default policy, is simply the last running request. Not the coldest. Not the largest. The last. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:1203` | `_preempt_request` — frees blocks, sets `PREEMPTED`, prepends to the waiting queue. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:1216` | `request.num_computed_tokens = 0`. Sit with this line. It is the whole of Break 1. |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:563` | `free` — and read the docstring: *"We free the blocks in reverse order so that the tail blocks are evicted first."* The head of the prefix survives longest, which is why the recovery in Break 1 works at all. |
| `memory/vllm/vllm/v1/core/block_pool.py:719` → `:741` | `free_blocks` — decrement refcount; hash-less blocks get `prepend_n` (die first), hashed blocks get `append_n` (linger as cache). Two-tier reclaim in three lines. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` → `:679` | `get_new_blocks` calls `_maybe_evict_cached_block` at *allocation* time. **Freeing is not evicting.** This is the single most consequential non-obvious fact for anyone building a KV dashboard. |

### 6.3 The chain, the match, and the reconciliation

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens(hash_fn, parent_block_hash, curr_block_token_ids, extra_keys)`. Four arguments; the first two make it a chain and the last makes it namespaced. Everything in Break 3 is in this signature. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:691` | `get_request_block_hasher` — incremental chaining, resuming from `len(request.block_hashes)`, only over full blocks. The tail partial block is never keyed. |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:255` | `max_cache_hit_length = request.num_tokens - 1`, with the comment explaining why (you need a forward pass for logits). Cap #1 of the three in §3.4. |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:708` | The phase-1 loop: `for block_hash in islice(...)`, `if not cached_block: break`. Cap #2. The comment above it states the invariant: *"A missing block implies every later block misses too (chained hashes)."* |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:718` | Phase 2, fine-grained: probe the interior boundaries of the first non-full block, high-to-low. This is the refinement to "breaks at the first miss." |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:871` → `:919` | `SlidingWindowManager.find_longest_cache_hit`, and its `for i in range(max_num_blocks - 1, -1, -1)`. **Right to left.** Windowed layers get mid-sequence hits. Read the `null_block` fill on the unmatched positions. |
| `memory/vllm/vllm/v1/core/block_pool.py:198` | `get_cached_block` — packs `(block_hash, group_id)` and returns `None` if any group in the list misses. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:685` | `HybridKVCacheCoordinator.find_longest_cache_hit` — the fixed-point reconciliation. Read the docstring, then watch `curr_hit_length` only ever decrease. |
| `memory/vllm/vllm/v1/core/kv_cache_coordinator.py:813` | `num_uncached_common_prefix_tokens = longest_hit_length - hit_length`. The metric the field says is unmeasured, computed in production for a different purpose. |
| `memory/vllm/vllm/v1/core/block_pool.py:47` | The `BlockHashToBlockMap` docstring, NOTE #1. Five lines explaining why de-duplication is refused. An append-only invariant paid for in physical memory. |

### 6.4 The radix tree, and the policy surface worth copying

| Where | What to look at, and why |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217` | `TreeNode` — the four fields every policy reads (`lock_ref`, `last_access_time`, `hit_count`, `priority`) plus `host_value` for the optional CPU tier. This is the data model a policy interface has to serve. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355` | `match_prefix` — read the docstring in full. It documents `extra_key` as an ASID-style namespace *and* admits the method may mutate the tree. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:650` | `_match_prefix_helper` — one dict hop per node, O(depth) not O(prefix length). Contrast directly with vLLM's O(blocks) walk at `single_type_kv_cache_manager.py:708`. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:676` | `_split_node` — cleaving an extent on partial overwrite, with two `.clone()`s. The filesystem move, in a cache. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict` — a heap over `evictable_leaves` only. Note the re-push of the parent on line 585, gated on it having lost its last child *and* holding no lock. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:790` | `_update_leaf_status` — the incremental maintenance that decides what is even a candidate. This is where the topological constraint lives. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:594` | `inc_lock_ref` — `while node != self.root_node`. One request pins a chain to the root. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:10` → `:49` | The whole policy surface: an ABC with one method, then six strategies as one-to-three-line comparators. `SLRUStrategy` at `:49` is the interesting one — segmented LRU expressed as a two-tuple `(is_protected, last_access_time)`. Copy this shape into Mnemosyne. |

### 6.5 The only place sharing becomes a bandwidth win

| Where | What to look at, and why |
|---|---|
| `memory/flashinfer/flashinfer/cascade.py:226` | `MultiLevelCascadeAttentionWrapper` — the runnable docstring example is the clearest artifact in the reference library for §3.5. Note `qo_indptr_arr[0] = [0, batch_size]`: the top level attends the shared pages with *all* queries at once. |
| `memory/flashinfer/flashinfer/cascade.py:42` | `merge_state(v_a, s_a, v_b, s_b)` — combining two partial attention outputs given their log-sum-exps. This identity is what makes splitting attention over KV segments exact rather than approximate; it is the same algebra as FlashAttention's online softmax. Understand this and the cascade is obvious. |

---

## 7. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats, from `ASSUMPTIONS.md`.** Single tensors **≥32 GiB hang silently at
0% CPU** (`large-tensor-fault-32gib`, refuted) — keep every buffer under 31 GiB. bf16 numerics on
gfx1151 are **untested** (`bf16-numerics-unproven`), so timing claims from these exercises are
usable and accuracy claims are not. The Hardware Validation Gate has not run, so nothing measured
here is evidence by house standard; these are instrument-shakedown runs and must be labelled as
such. Exercise C uses explicit `bmm` rather than `scaled_dot_product_attention` specifically so
the `sdpa-is-memory-efficient` hazard (147.2 bytes/T² retained by default `[M]`) does not apply —
at decode shape the score row is `[1, S]` and costs nothing either way.

Write scratch scripts under `notebook/`. Exercise C is worth a pre-registered hypothesis card
before it runs; A and B are not, because their predictions are closed forms rather than empirical
claims.

---

### Exercise A — the allocator, and the block size that byte accounting recommends

**Goal:** reproduce §3.2 and §3.3 from scratch, and find out for yourself that the byte-optimal
block size is 2 while everyone ships 16.

**Hardware:** none. Pure Python, no torch, no GPU. **Runtime:** 45–60 min to write, <1 s to run.

```python
"""Reservation vs paging, and the block-size optimum that byte accounting sees."""
import math

GIB = 1024 ** 3
KIB = 1024

def kv_bytes_per_token(n_layers_global, n_layers_window, n_kv, head_dim, dtype_bytes):
    """Growing term only (per token, global layers). Window layers are a fixed cost."""
    return 2 * n_layers_global * n_kv * head_dim * dtype_bytes

LAGUNA = dict(L_global=12, L_window=36, n_kv=8, head_dim=128, dtype_bytes=2, window=512)
FAST_TIER = 62 * GIB
S_MAX = 131_072

def residency(seq_len, cfg):
    """Bytes of KV one sequence holds at seq_len tokens, hybrid-aware."""
    per_layer_per_token = 2 * cfg["n_kv"] * cfg["head_dim"] * cfg["dtype_bytes"]
    return (cfg["L_global"] * per_layer_per_token * seq_len
            + cfg["L_window"] * per_layer_per_token * min(seq_len, cfg["window"]))

def concurrency_reserved(cfg, s_max=S_MAX, pool=FAST_TIER):
    return pool // residency(s_max, cfg)

def concurrency_paged(cfg, mean_len, block, pool=FAST_TIER):
    blocks = math.ceil(mean_len / block) * block
    return pool // residency(blocks, cfg)
```

**Deliverable 1 — the concurrency table.** Reproduce §3.2 for both the hybrid config and an
all-global counterfactual (`L_global=48, L_window=0`), at mean length 2,048 and `B`=16.

**Check yourself.** All-global: **2** reserved, **164** paged, **82×**. Hybrid: **10** reserved,
**377** paged, **37.7×**. If your all-global reserved number is not 2, your `2·L·n_kv·d·b` has a
factor wrong; the per-sequence figure must be exactly 24.0 GiB.

**Deliverable 2 — the block-size sweep.** For `B ∈ {1,2,3,4,8,16,32,64,128,256}` compute
`O(B) = k(B−1)/2 + 4·S_max/B` per sequence, print the table, and find the argmin. Then compute
the closed form `B* = √(8·S_max/k)` and confirm the discrete argmin is the nearest integer to it.

**Check yourself.** `k` here is the all-global 192 KiB/token. The minimum is at **B = 2**, total
**352 KiB**; `B*` = **2.31**; `B`=16 costs **1,472 KiB**, i.e. **4.18×** the optimum. Repeat for a
300M ablation shape (`L`=24, `n_kv`=4, `d_h`=64, `S_max`=32,768) and get `B*` = **3.27**.

**Deliverable 3 — one sentence in your notebook entry.** Name the cost term the model above is
missing, and say which exercise measures it. If your answer mentions fragmentation or metadata,
re-read §3.3.

**What a falsification would look like.** There is none — this is closed-form arithmetic. That is
why it is Exercise A and not a hypothesis card. Its job is to make you distrust the answer.

---

### Exercise B — the prefix chain, and hit rate measured three ways

**Goal:** build the chained-hash prefix cache, run a synthetic agent trace through it, and watch
the three hit-rate metrics separate. Then break the chain two ways and see what each break costs.

**Hardware:** none. Pure Python (+ matplotlib for the plot). **Runtime:** 1.5–2 h to write, a few
seconds to run.

```python
"""The chain, the match, and three hit rates that are all correct and all different."""
import hashlib

BLOCK = 16

def chained_hashes(token_ids, block=BLOCK, extra_key=b""):
    """vLLM's hash_block_tokens, in miniature: kv_cache_utils.py:596."""
    out, parent = [], b"\x00" * 32
    for start in range(0, len(token_ids) - block + 1, block):
        chunk = bytes(str(token_ids[start:start + block]), "utf8")
        parent = hashlib.sha256(parent + chunk + extra_key).digest()
        out.append(parent)
    return out

def content_hashes(token_ids, block=BLOCK, extra_key=b""):
    """The WRONG key: content-addressed, parent ignored. For deliverable 3."""
    return [hashlib.sha256(bytes(str(token_ids[s:s + block]), "utf8") + extra_key).digest()
            for s in range(0, len(token_ids) - block + 1, block)]

def match(hashes, cache, num_tokens, block=BLOCK):
    """kv_cache_manager.py:255 cap, then single_type_kv_cache_manager.py:708 loop."""
    max_hit = num_tokens - 1
    n = 0
    for h in hashes[: max_hit // block]:
        if h not in cache:
            break
        n += 1
    return n * block          # already block-aligned

def flops_avoided_fraction(hit, total, c1, c2):
    """(c1*Sh + c2*Sh^2) / (c1*S + c2*S^2)  ==  (phi + rho*phi^2)/(1+rho)."""
    return (c1 * hit + c2 * hit ** 2) / (c1 * total + c2 * total ** 2)

def gap(phi, rho):      return rho * phi * (1 - phi) / (1 + rho)     # percentage points
def ratio(phi, rho):    return (1 + rho) / (1 + rho * phi)           # x overstatement

# 300M ablation shape: L=24, n_q=16, d_h=64, P=3e8
C1, C2 = 2 * 3e8, 2 * 24 * 16 * 64
def rho_at(total):      return C2 * total / C1
```

**Trace 1 — the agent loop.** A warm-up request containing only the 2,048-token system prompt (so
the shared prefix is already resident, as it would be for the second user), then 12 turns; each
turn appends a 256-token user message and a 256-token assistant reply, and turn `n`'s request is
`system + all prior turns + this user message`, i.e. `2048 + (n−1)·512 + 256` tokens. Insert a
fixed 16-token "tool preamble" string at a *different* offset in every turn — you need it for
deliverable 3. Seed your RNG and record the seed.

**Trace 2 — the RAG loop.** The same 2,048-token shared instruction header, then one request per
retrieved document with document lengths sweeping `{256, 1k, 4k, 16k, 30k, 62k}` unique tokens.
This is the trace that lands in the interesting regime; trace 1 does not, and finding out why is
deliverable 1.

**Deliverable 1 — the divergence surface, and where each trace sits on it.** For each request
record blocks matched, tokens skipped, and FLOPs avoided. Then plot the two closed forms from
§3.4 over `φ ∈ (0,1)` at `ρ ∈ {0.1, 1, 10}`:

```
gap(φ,ρ)   = ρ·φ·(1−φ)/(1+ρ)        ratio(φ,ρ) = (1+ρ)/(1+ρ·φ)
```

and mark both traces' operating points on it.

**Predictions, stated before you run.** Blocks-matched and tokens-skipped track each other almost
exactly — they differ only by the `−1` cap and block alignment. FLOPs-avoided sits **below** both.
The absolute gap peaks at **`φ = 0.5`** and vanishes at both ends; the ratio is worst as
**`φ → 0`** and is bounded by `1+ρ`. **So in trace 1 the gap should get *smaller* with turn index
even though `ρ` is growing**, because `φ` is climbing past 0.5 toward 1 faster than `ρ` rises. If
you predicted "the gap widens with context length," you have conflated the two variables — that is
the trap this deliverable exists to spring, and I walked into it while writing this module.

**Check yourself, trace 1.** Turn 1: `S` = 2,304, hit = 2,048, `φ` = 88.89%, `ρ` = 0.1887, FLOPs
avoided = **87.32%**, gap **1.57 pp**, ratio **1.018×**. Turn 12: `S` = 7,936, hit = 7,680,
`φ` = 96.77%, `ρ` = 0.6501, FLOPs avoided = **95.54%**, gap **1.23 pp**, ratio **1.013×**. The gap
*narrows*. **At these shapes the three metrics agree to within 2 points and the metric choice does
not matter** — which is exactly why every prefix-cache demo uses a trace like this one.

**Check yourself, trace 2.** At a 30,720-token document (`S` = 32,768, `φ` = 6.25%, `ρ` = 2.684):
FLOPs avoided = **1.98%**, gap **4.27 pp**, ratio **3.15×** — §3.4's worked case, reproduced by
your code. At a 4,096-token document (`S` = 6,144, `φ` = 33.3%, `ρ` = 0.5033): FLOPs avoided =
**25.89%**, gap **7.44 pp**, ratio **1.29×**. At a 62,464-token document (`S` = 64,512,
`φ` = 3.17%, `ρ` = 5.285): FLOPs avoided = **0.59%**, gap **2.58 pp**, ratio **5.38×**.

**Report the document length that maximises the absolute gap and the one that maximises the
ratio, and confirm they are different requests.** In the sweep above they are the 4k document
(7.44 pp) and the 62k document (5.38×) — the two ends of the range, from one workload. Note also
that the absolute-gap maximum lands at `φ` = 33%, not at the `φ` = 50% the closed form predicts,
because along this sweep `ρ` and `φ` are not independent: lengthening the document raises `ρ` and
lowers `φ` simultaneously. **Explain that discrepancy in one line before you move on** — if you
cannot, you are reading the surface plot from deliverable 1 wrong.

**Deliverable 2 — the closed form, as a harness check.** Assert that your measured FLOPs-avoided
equals `(φ + ρφ²)/(1+ρ)` to within 1e-9 for every turn. If it does not, your FLOP accounting is
wrong, not the theory. **This assertion is the exercise's falsification condition.**

**Deliverable 3 — break the chain, twice, and price each break.**

- **(a) Flip one token at position 0 of the turn-6 request.** Predict, then measure, the hit
  length for turns 6 through 12. **Prediction: exactly 0 for all of them.** One token, six turns
  of cache, gone. This is the chain.
- **(b) Swap `chained_hashes` for `content_hashes` and count the false matches** — blocks that
  match on content but whose prefixes differ (your repeated tool preamble at varying offsets
  guarantees some). Report the count. **Every one of those is a wrong answer**, because the KV of
  those 16 tokens genuinely differs: it depends on the whole prefix through the attention below
  it, and on absolute position through RoPE. This is the exercise that turns "the chain is the
  only sound key" from a claim into a count.

---

### Exercise C — what does non-contiguity actually cost on gfx1151?

**Goal:** measure the term Exercise A's model cannot see. Sweep block size against achieved decode
bandwidth under both physical layouts, and produce the smallest block size at which the gather
overhead falls under 10%. That number is the page size our hardware wants, and it is unpublished
for this silicon.

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback given.** **Runtime:** 1.5–2 h to
write, 3–5 min to run on GPU, 15–25 min on CPU at reduced shapes.

**Pre-register this before you run** (G2 card, `notebook/<slug>.md`). Suggested thresholds:
SUCCESS = the two layouts differ by ≥1.3× at some block size, confirming the run-length mechanism.
KILL = both layouts are flat within ±10% across the whole sweep, meaning block size is
bandwidth-free on this hardware and §3.3's second cost term does not exist here — which would be a
genuinely surprising and reportable finding, and would make `B`=2 defensible for Mnemosyne.

```python
"""Decode attention over a paged KV cache: does block size cost bandwidth, and does layout decide?"""
import torch, time, json, statistics

S, H_KV, D = 65_536, 8, 128          # tokens, kv heads, head dim
DTYPE, ELEM = torch.bfloat16, 2
REPEATS = 5                           # >=3; report median and min, never a single run

def build_pool(block, layout, device):
    n_pages = S // block
    if layout == "NHD":               # [pages, block, heads, dim]
        shape = (n_pages, block, H_KV, D)
    else:                             # HND: [pages, heads, block, dim]
        shape = (n_pages, H_KV, block, D)
    k = torch.randn(shape, dtype=DTYPE, device=device)
    v = torch.randn(shape, dtype=DTYPE, device=device)
    # shuffled page order: the whole point of a block table is that frames are not in order
    order = torch.randperm(n_pages, device=device)
    return k, v, order

def gather_to_dense(pool, order, block, layout):
    pages = torch.index_select(pool, 0, order)          # the page walk, as a gather
    if layout == "NHD":
        return pages.reshape(S, H_KV, D).transpose(0, 1)   # -> [heads, S, dim]
    return pages.permute(1, 0, 2, 3).reshape(H_KV, S, D)

def decode_step(k_dense, v_dense, q, scale):
    scores = torch.bmm(q, k_dense.transpose(1, 2)) * scale     # [heads, 1, S]
    w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, v_dense)                                # [heads, 1, dim]

def measure(block, layout, device, paged=True):
    k, v, order = build_pool(block, layout, device)
    q = torch.randn(H_KV, 1, D, dtype=DTYPE, device=device)
    scale = D ** -0.5
    if not paged:                                   # contiguous baseline: identity order
        order = torch.arange(S // block, device=device)
    def one():
        kd = gather_to_dense(k, order, block, layout)
        vd = gather_to_dense(v, order, block, layout)
        return decode_step(kd, vd, q, scale)
    for _ in range(3):
        one()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        one()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del k, v, q
    if device == "cuda":
        torch.cuda.empty_cache()
    kv_bytes = 2 * S * H_KV * D * ELEM
    med = statistics.median(times)
    return dict(block=block, layout=layout, paged=paged, ms=med * 1e3,
                gb_s=kv_bytes / med / 1e9, all_ms=[t * 1e3 for t in times])

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for layout in ("NHD", "HND"):
    for block in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        rows.append(measure(block, layout, device, paged=True))
        rows.append(measure(block, layout, device, paged=False))
print(json.dumps(rows, indent=1))
```

**Footprint check before you run.** K and V are each `65,536 × 8 × 128 × 2 B` = 134 MB; the gather
materialises a second copy of each. Peak ≈ 540 MB. Far inside the `[M]` ≥62 GiB fast tier and
nowhere near the 31 GiB per-tensor hazard. You could raise `S` to 1M tokens (8.6 GB peak) and still
be safe — worth doing as a second sweep, since run-length effects usually sharpen with footprint.

**CPU fallback.** `S = 8_192`, `DTYPE = torch.float32`, `ELEM = 4`, `REPEATS = 3`. Expect one to
two orders of magnitude less bandwidth. The *shape* of the block-size curve is what transfers; the
absolute GB/s does not.

**Deliverable 1 — two curves and a mechanism.** Plot GB/s against block size, one line per layout,
paged only. **Prediction from §3.3: NHD is flat in block size (its run length is `d·b` = 256 B
regardless), HND rises and plateaus (its run length is `B·d·b`).** If HND rises and NHD is flat,
the run-length mechanism is confirmed on this hardware. If both rise, something other than run
length is driving it — say what, and note that `index_select` cost also falls with fewer, larger
pages, which is a confound you must separate by also reporting the gather time alone.

**Deliverable 2 — the number.** `overhead(B, layout) = t_paged / t_contiguous` at matched bytes.
Report the **smallest block size at which overhead < 1.10** for each layout. That is the
throughput-optimal page size for gfx1151. Compare it to Exercise A's byte-optimal `B* = 2` and to
the industry's 16. Write the three numbers in one line of your notebook entry.

**Deliverable 3 — the honesty check.** Report `all_ms` for at least one configuration, not just the
median. Single-run numbers are anecdotes by house rule, and a previous module in this curriculum
tagged a crash `[M]` on a single non-reproducing observation. If your spread across five repeats
exceeds the effect you are claiming, you have measured nothing — say so and raise `REPEATS`.

**Expected friction, stated so it does not surprise you.** `torch.index_select` on a 4-D tensor may
lower to a copy kernel whose efficiency itself depends on the trailing contiguous extent, which is
exactly the variable under test — so the gather and the attention are not cleanly separable. That is
a real limitation of this harness, not a flaw in the question; report gather-only timings alongside
the total so a reader can see the split. A fused paged-attention kernel would not go through this
path at all, and none is available to us on ROCm — which is itself worth stating in the write-up,
because it bounds how far the result generalises.

---

## 8. Self-check

Answers at the end of the file. Do not scroll.

1. PagedAttention is described as eliminating over-reservation. The block table's rows are sized to
   `max_num_blocks_per_req` regardless of actual sequence length, so at a mean live length of 2,048
   tokens and `S_max` = 131,072 they are 98.4% unused — the same pattern. Is the claim wrong? State
   the correct claim in one sentence with a number in it.

2. A request is preempted 29,000 tokens into a 32,000-token prefill. State (a) what happens to
   `num_computed_tokens`, (b) what happens to the 29,000 tokens of KV, and (c) what the request will
   most likely actually pay when it is rescheduled. Then say why (c) is not guaranteed and what
   number nobody reports.

3. A 2,048-token system prompt is shared; each request appends a unique 30,720-token document. Give
   blocks matched, tokens skipped, and FLOPs avoided, at the 300M shape (`c₁` = 6e8, `c₂` = 49,152,
   `B` = 16). Then say which of the three a TTFT-based SLO tracks.

4. Why is `hash(parent_hash, token_ids, extra_keys)` the *only sound* key rather than a conservative
   choice — and what changes about the *residency* requirement, not the key, when the layer is
   sliding-window rather than full attention?

5. Your Grafana panel says "KV cache: 60% occupied." Name the three distinct quantities that phrase
   could denote in vLLM, cite the line where each lives, and say which one is free memory.

6. Discard is always legal, so no KV cache carries a checksum. Name the failure mode this makes
   undetectable, give the citation, and connect it to a specific row in our `ASSUMPTIONS.md`.

---

## 9. What is still unsolved here

Everything below is testable at 20M–300M params on one gfx1151 device with a `[M]` ≥62 GiB fast
tier and a hard 31 GiB per-buffer limit, unless marked otherwise. Each needs a pre-registered
hypothesis card.

1. **The throughput-optimal page size on gfx1151 is unknown.** §3.3 shows byte accounting says 2,
   the industry ships 16, and the gap is a hardware-dependent kernel effect nobody has measured on
   this silicon. Exercise C. If the answer is "block size is free here," Mnemosyne can run at `B`=2
   and get 4.2× better capacity utilisation than a port of vLLM's defaults would.

2. **Layout is treated as a free choice and is not one.** FlashInfer documents NHD and HND as user
   options (`page.py:403`) while its backends disagree and the wrapper transposes at run time. The
   claim in §3.3 — that page size interacts with bandwidth under HND and not under NHD — is a clean
   prediction I could not find measured anywhere, for any hardware.

3. **Paging versus GPU virtual memory is unresolved.** `[C]` 2405.04437 (vAttention) versus
   `[C]` 2309.06180, with `[C]` 2607.02574 (Jun 2026) still listing it open and `[C]` 2601.13631
   (Jan 2026) arguing for semantics-aligned granularity instead of either. Our `[M]` 32 GiB
   single-tensor fault forces paging on us, which is a platform accident and must be declared as a
   confound rather than reported as a finding.

4. **The hybrid reconciliation cost is computed and never reported.** `kv_cache_coordinator.py:813`
   evaluates `num_uncached_common_prefix_tokens` in production. A matched-parameter
   global-vs-3:1-SWA arm measuring it as a function of ratio and window is, as far as this module's
   search established, unpublished. Cheap for us, directly relevant to Laguna's GSSS pattern `[M]`.

5. **Preemption recovery rate is unmeasured.** Break 1's accidental fault handler works, sometimes.
   Nobody reports how often or how much. The `preempted` flag already rides on
   `prefix_cache_stats.record` (`kv_cache_manager.py:222`).

6. **Regretted discards are unmeasured field-wide.** Named as a gap by `[C]` 2607.02574 and
   identified in `research/synthesis.md` as the lab's deliverable. Note the honest risk the synthesis
   also names: divergence from a full-cache oracle may not measure anything decision-relevant, and
   that must be tested before the harness is built.

7. **Contested: can the chain be broken?** `[C]` 2405.16444 (CacheBlend), `[C]` 2604.13226
   (KV Packet), `[C]` 2607.17715 (C²KV) all claim position-agnostic or composable KV reuse, with
   author-reported speedups of 2.2–3.3×, "near-zero overhead," and up to 17× respectively. None is
   independently replicated and all carry quality caveats. None has been evaluated below 1B
   parameters, which is a gap our scale could fill — a matched-quality small-scale replication of
   CacheBlend's selective-recompute rule would be a real contribution and costs one training run's
   worth of compute at most.

8. **Cascade sharing is available in principle and unmeasured as an intensity claim.** §3.5 predicts
   `I` multiplies by the sharing factor and crosses our ridge at `m ≈ 18`. FlashInfer implements it
   (`cascade.py:226`). There is no ROCm build of it available to us, so this is design-only until
   someone ports the kernel or we rent hardware — which makes it the strongest single argument for a
   costed rental in the backlog.

9. **Prefix-cache privacy versus hit rate has no agreed metric.** `[C]` 2411.18191 established the
   attack; `[C]` 2508.08438, `[C]` 2603.10726 and `[C]` 2605.23640 propose incompatible defences with
   incomparable evaluations (40.6% latency-overhead reduction, 70% higher reuse, 4.5× TTFT and 44%
   higher hit rate — all against different baselines). Not our research programme, but a Mnemosyne
   policy interface that cannot express per-tenant salting is under-specified, and the mechanism is
   one field in the hash (`kv_cache_utils.py:596`).

10. **Production hit-rate distributions are known from one provider.** `[C]` 2506.02634 is the only
    public large-scale characterisation this module found. Every scheduling result in the field
    — `[C]` 2602.06502, `[C]` 2605.08581, `[C]` 2607.02525 — is tuned against traces we cannot see.
    Any hit-rate number we generate from a synthetic trace is a statement about our trace generator,
    and Exercise B should be read that way.

11. **Silent corruption of shared blocks is undefended everywhere.** `[C]` 2604.17249: 13 of 16 bf16
    bit positions produce coherent altered output. No engine in the reference library checksums a KV
    block. Combined with `ASSUMPTIONS.md → bf16-numerics-unproven` this is our most under-examined
    correctness exposure, and the Hardware Validation Gate should acquire a KV-block round-trip
    integrity check on top of its existing checkpoint round-trip check.

---

## Answers to the self-check

**1.** The claim is not wrong, it is imprecise, and the precise version is more useful. **Paging
does not eliminate over-reservation; it relocates it from a structure costing 196,608 bytes per
token of context to one costing 0.25 bytes per token — a factor of 786,432.** The 98.4%-unused
block-table row is real and is exactly the same pattern; it is simply affordable. This matters
practically, because as `B → 1` the cheap structure stops being cheap: at `B`=1 the per-step
host-to-device table copy at `R`=256 and `S_max`=131,072 is 128 MiB rather than 8 MiB, and the
"negligible metadata" framing quietly stops holding.

**2.** (a) `request.num_computed_tokens = 0` (`sched/scheduler.py:1216`) — all progress is discarded
and the request is prepended to the waiting queue. (b) Its blocks are freed via `free_blocks`
(`block_pool.py:719`) in reverse order (`kv_cache_manager.py:563`), so refcounts drop to zero but the
hash-table entries stay live and the head-of-prefix blocks sit furthest from the eviction front. (c)
On reschedule the ordinary prefix match (`kv_cache_manager.py:225`) usually resurrects most of them
via `touch`, so the request typically re-pays a fraction of the 29,000 tokens, not all of them. It is
not guaranteed because eviction is lazy and happens at *someone else's* allocation
(`block_pool.py:647` → `:679`) — under memory pressure the blocks are gone. The unreported number is
the **recovery rate**: tokens recovered ÷ tokens lost, per preemption. The `preempted` flag exists
(`kv_cache_manager.py:222`); nobody joins it to the recovery.

**3.** `S` = 32,768, `S_hit` = 2,048, `B` = 16. Blocks matched = 128 of 2,048 = **6.25%**. Tokens
skipped = 2,048/32,768 = **6.25%**. FLOPs: `ρ` = 49,152 × 32,768 / 6e8 = 2.684, `φ` = 0.0625, so
`(0.0625 + 2.684 × 0.0625²)/(1 + 2.684)` = **1.98%** — a 3.16× shortfall against the other two. A
TTFT SLO tracks the **third**, because TTFT is prefill wall-clock and prefill is FLOPs (until it is
not — at very high hit rates and short prompts the fixed per-request overheads dominate and none of
the three predicts TTFT well). The general shape: FLOPs avoided interpolates from `φ` at short
context to `φ²` at long, so **at a fixed hit fraction, the longer your contexts, the more your
prefix-cache dashboard flatters you.** The two divergence measures peak in different places —
absolute gap `ρφ(1−φ)/(1+ρ)` at `φ` = 0.5, ratio `(1+ρ)/(1+ρφ)` as `φ → 0` — so "which metric
lies most" depends on the workload, not just on the context length.

**4.** Because `K[i]` and `V[i]` in layer `ℓ` are computed from the residual stream at position `i`
in layer `ℓ−1`, which depends through attention on *every* token at position ≤ `i`, and because RoPE
is applied at the **absolute** position `i`. So the bytes in a block are a function of the block's
contents, its position, and its entire history. A content hash would collide two blocks whose stored
tensors genuinely differ — that is a wrong answer, not a conservative miss. The chain is the minimal
sound key, not a cheap approximation of a better one. **What changes for a sliding-window layer is
residency, not the key.** `SlidingWindowManager` still probes the same chained `block_hashes[i]`
(so the whole prefix is still proven equal) but scans right-to-left and accepts a run of
`⌈(w−1)/B⌉` contiguous blocks anywhere, filling the rest with `null_block`
(`single_type_kv_cache_manager.py:871`, `:919`) — because out-of-window tokens are architecturally
unreadable and need not be present. Generalising: the key is always chained over the whole prefix;
the layer type sets how much of that prefix must be **resident**. Full attention: all of it. SWA:
`w` tokens. Recurrent state: all of it again, since the state cannot be rebuilt from a suffix.

**5.** Three quantities: (i) blocks with `ref_cnt > 0`, held by a live request — incremented in
`touch` (`block_pool.py:702`) and `get_new_blocks` (`:647`); (ii) blocks with `ref_cnt == 0` that are
still keyed in `cached_block_hash_to_block`, i.e. on the free queue *and* matchable for a future
prefix hit — put there by `free_blocks` (`:719`, appended at `:742`); (iii) blocks with no hash at
all, prepended to die first (`:741`). **Only (iii) is free memory.** (ii) is simultaneously "free"
and "cache," which is the whole point of `FreeKVCacheBlockQueue` being an intrusive list
(`kv_cache_utils.py:184`) — and eviction of (ii) does not happen when it is freed, it happens lazily
inside someone else's allocation (`:647` → `:679`). A panel reporting any one of the three as "cache
occupancy" is misdescribing the other two.

**6.** **Silent corruption of a shared block.** `[C]` 2604.17249 (Apr 2026) reports that 13 of 16
bf16 bit positions, when flipped in a shared KV block, produce coherent but altered output that is
indistinguishable from a legitimate response, that damage is confined to requests matching that
prefix, and that it accumulates linearly without decay. The reason no engine checks is structural:
storage tiers carry checksums because *loss* must be detected, and in a KV cache loss is legal, so
the integrity machinery was never built — but corruption is not loss, and the legality of one is not
an argument about the other. Our row is `ASSUMPTIONS.md → bf16-numerics-unproven`, still **untested**
with the Hardware Validation Gate unrun. The interaction is the bad part: the prefix-sharing
mechanism is a *propagation* mechanism for exactly this fault, so on our hardware a numerics defect
would not stay local to the request that produced it.

---

## Sources

**Local artifacts and measurements (`[M]`)**

- `ASSUMPTIONS.md` rows: `kv-per-token-laguna` (192 KiB/token exactly), `reference-model` (48 layers,
  12 full + 36 sliding, GSSS, `w`=512), `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per
  arm), `large-tensor-fault-32gib` (**refuted**; ≥32 GiB single tensors hang at 0 CPU),
  `gemm-throughput-below-reference` (20.9 TFLOPS bf16 at 8192³), `sdpa-is-memory-efficient`
  (**refuted by default**; 147.2 vs 6.6 bytes/T²), `bf16-numerics-unproven` (**untested**),
  `single-device-only`, `torch-build` (`2.12.0a0+rocm7.13.0a20260313`).
- `notebook/uma-carveout-controls-fast-tier.md` (2026-07-26) — the fast-tier sweep and the
  unplanned ≥32 GiB fault. Single run per arm; an anecdote by house standard.
- `research/memory/kv-serving-hierarchy.md` v1.0.0 — the note this module teaches. Nothing here
  contradicts it; three refinements are flagged inline (§3.4 phase-2 fine-grained probe, §4 Break 3
  right-to-left SWA matching, §4 Break 3 `num_uncached_common_prefix_tokens` already computed).
- `curriculum/attention-variants-and-kv-cost.md` v1.0.0 — the KV product, `AI = 2G/b`, and the
  ~105 FLOP/byte ridge this module reuses without re-deriving.
- `research/synthesis.md` v1.0.0 — what the lab decided to pursue (attribution instrument, tier-ratio
  arm) and park (distributed KV, eviction-policy design).
- **No number in this module was measured by its author.** Everything tagged with a value is either
  arithmetic over the `[M]` inputs above (and labelled as arithmetic) or a `[C]` citation. The
  exercises are the measurements.

**Code (`file:line`, all opened and confirmed 2026-07-26 against `PROVENANCE.md` revisions)**

- vLLM: `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:97`, `:658`, `:708`, `:718`,
  `:871`, `:919`; `core/block_pool.py:47`, `:198`, `:647`, `:679`, `:702`, `:719`, `:741`;
  `core/kv_cache_utils.py:184`, `:596`, `:691`; `core/kv_cache_manager.py:222`, `:225`, `:255`,
  `:563`; `core/kv_cache_coordinator.py:685`, `:813`; `core/sched/scheduler.py:566`, `:603`,
  `:1203`, `:1216`; `worker/block_table.py:81`;
  `attention/ops/triton_unified_attention.py:424`; `kv_cache_interface.py:109`, `:539`, `:567`.
- SGLang: `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217`, `:355`, `:565`, `:594`,
  `:650`, `:676`, `:790`; `mem_cache/evict_policy.py:10`, `:16`, `:49`; `mem_cache/memory_pool.py:251`.
- FlashInfer: `memory/flashinfer/flashinfer/page.py:326`, `:403`; `flashinfer/utils.py:50`;
  `flashinfer/cascade.py:42`, `:226`.

**arXiv (`[C]`)** — ids resolved against the arXiv abstract pages on 2026-07-26 except where noted
as inherited from `research/memory/kv-serving-hierarchy.md`, whose ids are machine-verified.

*Paging and its alternatives*
- `2309.06180` — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
  (Sep 2023). Block tables; internal/external fragmentation; the block-size trade-off.
- `2405.04437` — *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*
  (2024, ASPLOS'25). Contiguous virtual memory; up to 24% faster prefill.
- `2601.13631` — *ContiguousKV: Accelerating LLM Prefill with Granularity-Aligned KV Cache
  Management* (2026-01-20). 3.85× on re-prefill vs IMPRESS; author-reported.
- `2509.04377` — *PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language
  Model Inference* (2025-09-04). Block-granular pruning as a compression policy.

*Prefix reuse, its structures and its scheduling*
- `2312.07104` — *SGLang: Efficient Execution of Structured Language Model Programs* (Dec 2023).
  RadixAttention. (Inherited.)
- `2506.02634` — *KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large
  Cloud Provider* (2025-06-03, USENIX ATC'25). The only public large-scale reuse characterisation
  this module found.
- `2602.06502` — *DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM
  Serving* (Feb 2026). (Inherited.)
- `2605.08581` — *PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design* (May 2026).
  Demand-aware radix tree. (Inherited.)
- `2607.02525` — *PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving* (2026).
  Incremental radix tree over the pending queue. (Inherited.)

*Breaking the prefix chain*
- `2405.16444` — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*
  (2024-05-26). Non-prefix reuse with selective recompute; 2.2–3.3× lower TTFT, 2.8–5× throughput,
  author-reported.
- `2604.13226` — *KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs* (2026-04-14,
  rev. 2026-04-17). Immutable packets plus distilled adapters.
- `2607.17715` — *C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference*
  (2026-07-20). Position-agnostic composable manifold; up to 17× at long context, author-reported.

*The shared read-only tier's correctness and privacy hazards*
- `2411.18191` — *InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks*
  (2024-11-27). Establishes the attack; no exact success rate in the abstract.
- `2508.08438` — *Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference*
  (2025-08-11, rev. 2026-02-09). SafeKV; up to 40.58% lower TTFT overhead, 2.66× throughput vs full
  isolation.
- `2603.10726` — *PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems*
  (2026-03-11). Owner-tagged blocks; up to 70% higher reuse, 30% lower latency vs user-level
  isolation.
- `2605.23640` — *CachePrune: Privacy-Aware and Fine-Grained KV Cache Sharing for Efficient LLM
  Inference* (2026-05-22). Token-level sharing; 4.5× TTFT reduction, 44% higher hit rate.
- `2604.17249` — *Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems*
  (2026-04-19, rev. 2026-06-07, SECRYPT'26). **13 of 16 bf16 bit positions produce coherent but
  altered output**; damage confined to the affected prefix and accumulating linearly.

*Framing and background*
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management
  for LLM Serving* (Jun 2026). Names paging-vs-VMM unresolved and seven measurement gaps.
  (Inherited.)
- `2407.00079` — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024).
  `offload_force_evict`; discard as a legal operation. (Inherited.)
- `2403.02310` — *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (Mar 2024).
  Chunked prefill; single-pool, therefore runnable by us. (Inherited.)
- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019). Why decode is
  bandwidth-bound. (Inherited.)
- `2305.13245` — *GQA* (2023). The `H_q → H_kv` knob that sets `I` in §3.5. (Inherited.)

**Not cited because unverified.** Several 2026 vendor and blog sources surfaced during search with
concrete-looking prefix-cache hit-rate ranges (60–85% on agent loops, 85–95% cost savings on hits)
and fragmentation-recovery figures (30–50%). None carried an arXiv id or a reproducible harness, so
they are excluded rather than repeated. If you want a hit-rate number for a design document, use
`[C]` 2506.02634 or generate your own with Exercise B and label it as synthetic.
