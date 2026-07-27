---
title: KV cache mechanics — residency, read traffic, and the maintenance budget nobody prices
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
mirrors: research/memory/kv-cache-mechanics.md
prereqs: tensors-and-autograd, transformer-forward-pass-by-hand, attention-variants-and-kv-cost
difficulty: hard — the algebra is still arithmetic, but three budgets have to be held apart at once and the machine disagrees with two of them
time: 4–5 h reading and working the math; 2–3 h for the three exercises (all three were run on the Z13 before shipping; reference numbers are in the text)
---

# KV cache mechanics

**Difficulty and time, honestly.** Nothing here is harder than division. What is hard is
that the phrase "the KV cache costs X" is three different quantities, and every published
number silently picks one. Once you separate them, two of this module's three measurements
come out *against* the naive model, and one of them comes out 3× against it in the
direction nobody expects. Budget 4–5 hours for sections 2–5 with a pen, and 2–3 hours for
the exercises. Exercise B is the one that carries the module's central claim; Exercise C is
the one that produces a number this lab did not have and that the mirrored survey note
lists as an open question.

**Prerequisites, and what this module refuses to re-teach.** You need
`attention-variants-and-kv-cost.md` (Track B) cold: the product `2·L·n_kv·d_h·b`, the
MHA/GQA/MQA/MLA ladder, the windowing split, and the derivation `AI = 2G/b`. All of that is
assumed. This module takes those results as inputs and asks the operational questions Track
B deliberately deferred: *what is the object, how does it grow, what does it cost to
maintain as opposed to to read, and does the machine agree.*

---

## 1. What this module settles

**One:** the KV cache is not one object but `L` independently-shaped objects with a
per-layer-type residency rule, so the cost model is a *sum over layer types* and never a
product with `L` — and the same is true of its arithmetic intensity, which on our reference
model is 6 on twelve layers and 9 on thirty-six. **Two:** there are three budgets, not one —
**residency** (bytes held), **read traffic** (bytes moved to use the cache), and
**maintenance traffic** (bytes moved to append to it) — and the default HuggingFace cache
writes, purely to maintain itself, exactly as many bytes as attention reads to use it, an
amortisation defect that is invisible in every config file and every paper, and that we
measured at **23× wall clock** against a preallocated cache at 8k context `[M]`. **Three:** on this machine the
single-number bandwidth model (total KV bytes ÷ peak GB/s) is optimistic by **1.45× to
3.2×** depending on context length `[M]`, because the windowed layers that hold 1.2% of the
bytes consume 8.4% of the time — and an fp8 cache, which the shape math says should be
strictly better, is **2.9–3.1× slower** than bf16 on our stack `[M]`.

The first is theory you can check with a pen. The second and third are measurements taken
on the Z13 for this module, and both of them contradict the arithmetic in the direction of
"the machine is worse than the model says." That is the useful direction: a cost model that
is optimistic in production is the one that gets you paged.

---

## 2. Theory in plain language

### 2.1 What the object actually is

Not a cache in the sense of one structure with one policy. Per request, the model holds
**2·L tensors** — a K and a V for every attention layer — each shaped

```
[batch, n_kv, tokens_so_far, head_dim]
```

They are created lazily on the first forward pass, appended to exactly once per decode
step, read in full exactly once per decode step per layer, and destroyed when the request
finishes. There is no persistence, no durability contract, no fsync, no recovery, and no
name. In HuggingFace the whole thing is a Python list of layer objects
(`cache_utils.py:1737`); in llama.cpp it is two C++ allocations
(`llama-kv-cache-iswa.cpp:73`); in vLLM it is several block tables over one preallocated
slab (`kv_cache_interface.py:197`). Three data structures, one object model.

> **Systems bridge — the one this module runs on.** Treat it as a **working set with a
> per-token cost function**. Every token admitted to context costs exactly
> `per_token_bytes` of residency for the remaining life of the request. That is not a
> capacity you fill; it is a *rate you commit to on admission*. The planning question is
> therefore not "how big is the cache" but "what slope am I buying, and where does the line
> cross my tier."
>
> **Where it breaks, first cut.** A working-set manager has three moves: admit, decline,
> evict. This one has *one*. You cannot decline a token — the sequence is the sequence. You
> cannot page out, because there is no lower tier that the attention kernel can read from;
> a miss is not slow, it is **unrepresentable** inside the kernel (FlashInfer's page table
> has no present bit — `memory/flashinfer/flashinfer/decode.py:1239`; vLLM's allocator, on failure, preempts
> the entire request rather than faulting — `block_pool.py:647`). So the only real move is
> destroy, and destruction is lossy in accuracy rather than in latency. A working-set
> manager that cannot refuse work and cannot demote is not a manager. It is an accountant.

### 2.2 The three budgets, which almost everyone merges

This is the module's organising idea and it has no analogue in the literature, which
reports capacity and end-to-end throughput and nothing in between.

| Budget | What it is | Units | Who pays it |
|---|---|---|---|
| **Residency** | bytes held while the request lives | bytes | the memory tier — this is the one that decides whether you fit |
| **Read traffic** | bytes moved from memory into the attention kernel, per decode step | bytes/step | the memory bus — this is the one that sets tokens/s |
| **Maintenance traffic** | bytes moved in order to *append* one token's entry | bytes/step | also the memory bus, and it is normally assumed to be zero |

Residency is the only one the shape math in Track B addresses. Read traffic equals
residency once per step, which is why decode is bandwidth-bound. Maintenance traffic
*should* be one token's worth per step and, in the reference implementation everyone
benchmarks against, it is not — it is the entire cache, twice, on every single token.
Section 3.4 does that algebra and Exercise A measures it.

> **Systems bridge.** You already price write amplification separately from read
> amplification on an SSD, and you already know that a log-structured store's compaction
> traffic can exceed its user traffic. Same shape of mistake, same discipline required:
> **the traffic to maintain a structure is a first-class budget line, not an implementation
> detail.**
>
> **Where it breaks.** On an SSD, write amplification is a property of the device and the
> workload. Here it is a property of *which Python class the serving stack instantiated*,
> and both classes ship in the same file with the same public method. Nothing in the model
> config, the checkpoint, or the API surface tells you which one you got.

### 2.3 Two phases with opposite characters

**Prefill** consumes the whole prompt at once: one pass, `T` tokens, large matmuls,
compute-bound, and it *writes* the entire cache. **Decode** produces one token at a time:
tiny matmuls, and it *reads* the entire cache. `[C]` This split is the organising fact of
the modern serving stack (2311.18677, Nov 2023; 2401.09670, Jan 2024) and is why
prefill/decode disaggregation exists.

> **Systems bridge.** Bulk load, then trickle-append. You have built this: an initial import
> followed by a change stream.
>
> **Where it breaks.** In every trickle-append system you have run, the append is cheap
> because it touches only the tail. Here the append is accompanied, on the same step, by a
> **full scan of everything previously appended** — there is no index, no partition pruning,
> no bloom filter, no way to touch less than 100% of the structure. The read is not "hot
> data plus a long tail." It is a table scan, every token, forever.

### 2.4 The heterogeneous-cache object model, and the module's central break

On a hybrid model the cache is heterogeneous by construction. Our reference model, Laguna S
2.1, has 48 layers in a strict `full, sliding, sliding, sliding` pattern — 12 global and 36
windowed at `w = 512` `[M]` (`ASSUMPTIONS.md → reference-model`, read from the shipped
config at revision `b0a9fd7c850e`).

In HuggingFace this is not an abstraction. It is a dictionary lookup and a list
comprehension:

```python
DYNAMIC_LAYER_TYPE_MAPPING = {
    "full_attention": DynamicLayer,
    "sliding_attention": DynamicSlidingWindowLayer,
    ...
}
layers = [DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs) for layer_type in layer_types]
```

— `cache_utils.py:1184` and `cache_utils.py:1737`.

> **Systems bridge.** Two storage classes over one pool, sized independently. You have
> capacity-planned exactly this.
>
> **Where it breaks — and this is the module's headline.** The two classes do not merely
> hold different amounts of data. They have **different arithmetic intensities** — `G = 6`
> on Laguna's global layers, `G = 9` on its sliding ones `[M]`, because query-head count
> varies per layer while KV-head count does not — so a single-number bandwidth model built
> from the top-level `num_attention_heads` is wrong for **36 of 48 layers, i.e. 75% of the
> stack**. That is the break the two-storage-classes analogy hides, and it is real.
>
> It is also not the worst of it. Exercise B shows a *second*, independent
> failure of the single-number model that the `G` story does not predict at all: the
> windowed layers issue reads so small that they never reach streaming bandwidth. At 128k
> context they are **1.2% of the bytes and 8.4% of the time** `[M]`. At 32k they are **4.5%
> of the bytes and 30.5% of the time** `[M]`. Correcting `G` per layer type does not fix
> this; only measuring does.

### 2.5 The property that makes all of this different from storage

A KV entry is **not independently recomputable**. To rebuild layer `l`'s key for token `t`
you need the layer-`l` residual stream at position `t`, which requires every layer below it
at every position `≤ t`, which requires *their* KV entries. So:

- You can recompute a **suffix** if you still hold the prefix, at every layer. (This is what
  chunked prefill does.)
- You **cannot** recompute an interior hole. A gap in the middle of the cache is
  unrecoverable without recomputing everything after it, at every layer.
- The unit of recompute is therefore `(all layers, prefix [0..t])` — never a page, never an
  entry.

> **Where the storage analogy dies completely.** Every tier you have operated has a fault
> granularity equal to its allocation granularity: you lose a block, you re-read a block.
> Here the allocation granularity is one token in one layer and the *recovery* granularity
> is the entire prefix across the entire stack. That asymmetry is why eviction in this
> system is destruction rather than demotion, and it is the fact that the whole of Track C
> is built on.

---

## 3. The math that actually matters

### 3.1 Symbols

Carried from Track B; repeated so the equations are readable standalone.

| Symbol | Reads as | Source |
|---|---|---|
| `T` | tokens currently in context | runtime |
| `L`, `L_g`, `L_w` | total / global / windowed attention layers | `num_hidden_layers`, `layer_types` |
| `n_kv` | key-value heads per layer | `num_key_value_heads` |
| `n_q,l` | query heads in layer `l` | `num_attention_heads` (a per-layer list on Laguna) |
| `G_l` | `n_q,l / n_kv`, the GQA group size in layer `l` | derived, never configured |
| `d_h` | head dimension | `head_dim` |
| `b` | **bytes per stored element** — bf16 → 2, fp8 → 1 | `torch_dtype` / `kv_dtype` |
| `w` | sliding window, in tokens | `sliding_window` |
| `c` | **bytes per token per layer** = `2 · n_kv · d_h · b` | derived; the module's working unit |
| `B` | batch, i.e. concurrent sequences | runtime |

Lowercase `b` is bytes-per-element; uppercase `B` is batch. Deliberate, matching the survey
notes.

`[M]` **For Laguna-S**: `c = 2 × 8 × 128 × 2 = 4096 B = 4 KiB` per token per layer.
Everything below is that one number times a count.

### 3.2 Residency as a sum over layer types

```
R(T)  =  Σ_over_layers  c · min(T, w_l)          w_l = ∞ on a global layer
      =  c · L_g · T        +      c · L_w · w
         \__ grows with T __/      \__ constant in T __/
```

In words: every global layer holds one `c`-byte entry for every token ever seen; every
windowed layer holds at most `w` of them, and stops growing the moment `T` passes `w`.

`[M]` **Laguna-S**: growing term `12 × 4 KiB = 48 KiB/token`; fixed term
`36 × 4 KiB × 512 = 72 MiB`.

| Context `T` | Growing term | Fixed term | `R(T)` | vs all-global |
|---|---|---|---|---|
| 1,536 | 72 MiB | 72 MiB | 144 MiB | 2.0× |
| 32,768 | 1.500 GiB | 72 MiB | **1.570 GiB** | 3.8× |
| 131,072 | 6.000 GiB | 72 MiB | **6.070 GiB** | 4.0× |
| 1,048,576 | 48.00 GiB | 72 MiB | **48.07 GiB** | 4.0× |

Note the **byte-parity point**: `c·L_g·T = c·L_w·w` when `T = L_w·w / L_g = 36×512/12 =
1536`. Below 1536 tokens the windowed layers hold *more* bytes than the global ones. That
is worth internalising because a great deal of small-scale experimentation happens below it,
where the hybrid saves you nothing and its constant term dominates.

`R(131072) = 6,517,948,416 B` is also exactly the `stack_bytes` figure Exercise B allocates
and times, which is a useful arithmetic self-check on the harness.

### 3.3 Read traffic, and the stack's *effective* arithmetic intensity

Track B derived, per layer, `AI = 2·G/b`, which for bf16 is exactly `G`. Independent of `T`,
`d_h`, `L` and batch. That is per layer. The stack does not have a single `G`, so what is
the stack's intensity? Take the ratio of the sums, not the sum of the ratios:

```
                Σ_l 4 · n_q,l · min(T,w_l) · d_h            (FLOPs, all layers)
AI_stack(T)  =  ───────────────────────────────────
                Σ_l 2 · n_kv · d_h · b · min(T,w_l)         (bytes, all layers)
```

Substituting Laguna-S (`n_q = 48` on 12 global layers, `72` on 36 sliding, `d_h = 128`,
`n_kv = 8`, `b = 2`, `w = 512`), for `T ≥ w`:

```
                4·128·(48·12·T + 72·36·512)       576·T + 1,327,104
AI_stack(T)  =  ─────────────────────────────  =  ───────────────────
                4096·(12·T + 36·512)               96·T + 147,456
```

| `T` | `AI_stack` | reading |
|---|---|---|
| 512 | **8.25** | at `T = w` every layer is effectively global; the 72-query-head layers dominate |
| 1,536 | **7.50** | byte parity; exactly midway between 9 and 6 |
| 32,768 | **6.13** | global layers own 96% of the bytes |
| 131,072 | **6.03** | |
| → ∞ | **6.00** | the asymptote is the global layers' `G`, and nothing else |

**This is the quantitative form of the break.** The stack's arithmetic intensity is not a
constant of the architecture; it is a byte-weighted average that **migrates from 8.25 to
6.00 as context grows**, because the mix of bytes shifts toward the layers with the lower
`G`. A Mnemosyne cost model that stores one intensity per model is wrong twice over: wrong
per layer, and wrong as a function of `T`. Store the spec list, compute the ratio.

Against the `[M]` ≈105 FLOP/byte ridge point in `ASSUMPTIONS.md`, `AI_stack` between 6 and
8.25 means decode attention runs at **5.7%–7.9% of peak compute**, and gets *worse* with
context. That number is derived, not measured; Exercise B measures the bandwidth half of it.

### 3.4 Maintenance traffic — the budget that is not in any paper

Two implementations ship in the same HuggingFace file, and the difference is not a tuning
option, it is an asymptotic class.

**Preallocated (`StaticLayer`).** Allocate `[B, n_kv, T_max, d_h]` once
(`cache_utils.py:406`), then write one row per step in place with `index_copy_`
(`cache_utils.py:454`).

```
maintenance per step  =  c bytes            (one token's entry)
maintenance total     =  c · T
peak residency        =  c · T_max          from step one
```

**Grow-by-concatenation (`DynamicLayer`, the default).** One line
(`cache_utils.py:143`):

```python
self.keys = torch.cat([self.keys, key_states], dim=-2)
```

`torch.cat` cannot extend a tensor. It allocates a new one and copies. So at step `t` (cache
currently holds `t` tokens):

```
bytes read   =  c · t          (the old cache)
bytes written =  c · (t+1)     (the new cache)
```

Summed over a `T`-token generation:

```
maintenance written  =  c · T·(T+1)/2   ≈  c·T²/2
maintenance moved    ≈  c · T²
amplification vs preallocated (written)  =  (T+1)/2
```

`[M]` **Worked at Laguna's per-layer shape** (`c = 4096 B`), for one global layer over an
8,192-token generation:

```
preallocated : 8192 × 4096 B                    =   33.55 MB written
concatenated : 4096 × 8192 × 8193 / 2           =  137.46 GB written
amplification                                    =   4096.5×
```

Now the part that makes this more than a curiosity. Compare maintenance-written against the
attention **read** traffic over the same generation, for the same global layer:

```
read traffic over the generation  =  Σ_{t=1..T} c·t  =  c · T·(T+1)/2
maintenance written               =  Σ_{t=1..T} c·t  =  c · T·(T+1)/2
```

They are **identical**, exactly, and the identity does not depend on `c`, on `T`, on the
model, or on the hardware. Counting the concat's read side as well:

> **With a grow-by-concatenation cache, maintaining the cache costs twice the traffic of
> using it. Total decode memory traffic is 3× what the shape math predicts, at every
> context length, forever.**

That is a closed-form result, it is not in the literature, and it falls out in four lines.
`[M]` Exercise A's third deliverable tests it end to end and lands at **2.09×** measured
maintenance-to-read against the predicted 2.00×. Note carefully what Exercise A's headline
23× is *not*: that number compares concat maintenance against **preallocated maintenance**,
which is a different ratio and is much larger because the denominator is a fixed launch cost
rather than a byte cost. Three ratios live in this section — 4096× (bytes, concat vs
preallocated), 23× (wall clock, same comparison), and 2× (traffic, maintenance vs read) —
and conflating any two of them will produce a wrong number that looks authoritative.

> **Systems bridge.** `std::vector::push_back` with `reserve(size()+1)` before every push:
> geometric growth replaced by linear growth, `O(n)` amortised turned into `O(n)` per
> operation. You have found this bug in production code.
>
> **Where it breaks — why doubling is not the fix.** In a vector you would fix this with
> geometric growth and move on. Here nobody does, and the reason is a second constraint you
> do not have in a vector: **`T_max` is unknown and the waste is enormous.** Preallocating
> `T_max` commits `c·L·T_max` bytes per request from step one — at `T_max = 131,072` that is
> 24 GiB for a 48-layer all-global model, or 6.07 GiB for Laguna's hybrid, committed for a
> request that may emit twelve tokens. Nor does geometric growth help as much as you would
> hope: during the realloc the old and new tensors are both resident, so peak transient
> exceeds steady state by the growth factor, and the fault at `[M]` ≥32 GiB per tensor
> (`ASSUMPTIONS.md → large-tensor-fault-32gib`) is a *silent hang at 0% CPU* on this
> machine. The production answer is neither: it is **fixed-size blocks with an indirection
> table**, so growth is `O(1)`, waste is bounded at under one block, and no byte is ever
> copied. That is PagedAttention `[C]` (2309.06180, Sep 2023), and this paragraph is the
> reason it exists. It is the subject of `paged-attention-and-prefix-reuse.md`; read this
> section as that module's motivation.

### 3.5 The preallocated cache's own defect: it reads what it has not written

`StaticLayer.get_mask_sizes` returns `kv_length = self.max_cache_len`
(`cache_utils.py:466`). The tensor handed to attention is the **entire preallocated
buffer**, including rows that have never been written; correctness comes from the mask, not
from the shape. So:

```
read traffic per step, preallocated  =  c · T_max        (constant in T)
read traffic per step, grown         =  c · T            (grows with T)
```

At `T_max = 131,072` and an actual `T = 1,024`, the preallocated cache moves **128× more
bytes per step than it needs to.** `[A]` High confidence in the code reading, medium in the
consequence: a fused kernel that skips fully-masked tiles (FlashAttention-style) avoids
most of it, and on gfx1151 the SDPA backend that runs by default retains the score matrix
anyway (`ASSUMPTIONS.md → sdpa-is-memory-efficient`). The cheapest test is to time a
`StaticCache` decode step at fixed `T` while sweeping `max_cache_len`; a flat line means the
kernel skips, a rising line means it does not.

**Put the two together and the trade is legible:** grow-by-concat has optimal read traffic
and catastrophic maintenance traffic; preallocation has optimal maintenance traffic and
provisioned-not-actual read traffic. Paging is the design that takes the good half of each.

### 3.6 The three budgets side by side

`[M]` for the numbers marked as such; the rest is arithmetic over them. Laguna-S, one
sequence, `T = 32,768`, bf16.

| | Grow-by-concat | Preallocated (`T_max` = 131,072) | Paged (16-token blocks) |
|---|---|---|---|
| Residency | 1.570 GiB | 6.070 GiB | 1.570 GiB + ≤ 48×64 KiB slack |
| Read traffic / step | 1.570 GiB | 6.070 GiB (see §3.5) | 1.570 GiB |
| Maintenance / step | ~3.14 GiB (read + write of the whole cache) | 192 KiB | 192 KiB |
| Peak transient | residency **+ one layer's cache** (~128 MiB here), because the old and new tensors coexist during the copy | none | none |
| Growth failure mode | `OOM` or the ≥32 GiB silent hang | `OOM` at admission | block-pool exhaustion → **preempt the request** |

That last row is the one to sit with. Three implementations, three completely different
failure modes, and the model config is identical in all three.

### 3.7 `b`, and why halving it made things worse on this machine

`b` is a factor in all three budgets simultaneously: fp8 halves residency, halves read
traffic, halves maintenance traffic, and doubles `AI` (`2G/b` with `b = 1`). The shape math
says it is the one reduction technique with no interaction term.

The shape math assumes the attention kernel can **consume** fp8. On gfx1151 it cannot:
`[M]` `torch._scaled_mm` raises *"only supported on CUDA devices with compute capability
>= 9.0 or 8.9, or ROCm MI300+"* (probe recorded in `research/memory/kv-cache-mechanics.md`).
So the cache must be dequantised to bf16 before every attention call, and the traffic
accounting changes completely:

```
bf16 path : read 2 B/elem                                     = 2 B/elem
fp8 path  : read 1 B/elem  +  write 2 B/elem (the bf16 temp)
            +  read 2 B/elem (attention reads the temp)       = 5 B/elem
predicted slowdown  =  5 / 2  =  2.5×
```

`[M]` **Measured: 2.92–3.13× slower**, across two separate processes and three context
lengths (median of 3 and of 5 repeats respectively — six measurements in all), with the
dequantise step alone accounting for **65–66%** of the fp8 path's time. Details and the exact
configuration in §5.4 and Exercise C.

This directly answers open question 2 in `research/memory/kv-cache-mechanics.md` — *"does
the fp8-storage / bf16-compute path actually recover the bandwidth win on this hardware?"* —
and the answer is **no, it loses roughly 3×**. It does not contradict that note; the note
predicted the hazard and asked for the number. Here is the number.

The residency win is still real and still worth having: an fp8 cache is genuinely half the
bytes, and for a *capacity*-bound experiment (which is most of what this lab runs) that is
the whole point. What is dead on this hardware is the *bandwidth* half of the fp8 story.
Those are two different budgets, which is the module's thesis.

---

## 4. Why this matters for Proteus and Mnemosyne

### 4.1 Mnemosyne's cost model is a spec list, not a model object

The boundary rule (CLAUDE.md) says Mnemosyne never imports Proteus. That is usually
described as hygiene; here it is a design constraint with a right answer already visible in
the reference library. vLLM's `AttentionSpec` is a frozen dataclass carrying
`num_kv_heads`, `head_size`, `dtype`, `block_size`, and a `real_page_size_bytes` property
that is *literally* our formula (`kv_cache_interface.py:204`):

```python
return 2 * self.block_size * self.num_kv_heads * head_dim * get_dtype_size(self.dtype)
```

Copy that shape. Mnemosyne's cost model should take `list[LayerCacheSpec]` and nothing else.
Consequences:

- It works on Laguna, on Proteus, and on gpt-oss without knowing what any of them are.
- It is forced to be per-layer, which §3.3 shows is not optional.
- It is testable without a model — which matters, because the acceptance test for
  separability is a clean venv containing only torch.

Three methods, matching §2.2: `residency_bytes(T)`, `read_bytes_per_step(T)`,
`maintenance_bytes_per_step(T)`. If a policy changes any of the three it must report which,
because that is exactly the attribution the field does not do.

### 4.2 The eviction-costs-a-compaction problem, which is Lethe's founding constraint

An eviction policy removes tokens from the middle of a contiguous per-layer tensor. In a
contiguous layout that is a **compaction**: you must build a new tensor and copy the
survivors. So for a policy that evicts a fraction `f` at every step:

```
compaction traffic, per compaction  ≈  2 · c · T      (read all, write the survivors)
read-traffic saving, per step after  =  f · c · T
```

The compaction is a full pass over the cache; the saving is a fraction of a full pass, and it
only accrues on *subsequent* steps. **So a policy that compacts on every step pays 2 and
collects `f`, and loses for every `f < 2` — which is every `f`, since `f ≤ 1`.** Real policies
dodge this in one of three ways: evict rarely (SnapKV-style, once at the end of prefill),
evict in whole blocks so compaction is a pointer update (`block_pool.py:719`), or never
physically remove anything and just mask.
Mnemosyne must pick one deliberately and record which, because the choice determines whether
a reported speedup is a property of the policy or of the layout.

`[A]` Medium confidence in the "any `f < 2`" framing — it assumes compaction is a full copy
and ignores that a fused gather could fold the compaction into the next read. Cheapest test:
implement one policy both ways at 300M and compare traffic counters, not wall clock.

### 4.3 The attribution instrument's own overhead is the harness bug waiting to happen

`research/synthesis.md` commits the lab to a **full-cache oracle diff** as its deliverable:
run the expensive full-cache reference on every probe and measure divergence. If that oracle
runs on HuggingFace's `DynamicCache`, §3.4 says it pays `T/2`-fold maintenance amplification
that the policy arm — if it uses any block-based cache — does not. At `T = 8192` that is a
4096× byte difference on one budget line. Your policy will look brilliant and the finding
will be about `torch.cat`.

Concrete rule for the rig: **the oracle and the policy arm must use the same cache
implementation**, and the harness must assert it. That is a one-line test and it prevents a
class of result that would be very hard to catch afterwards. House rule "if a result looks
too good, suspect the harness" has a specific address in this module.

### 4.4 The tier-ratio experiment is priced in these budgets

Synthesis question 2 — *does the eviction-versus-retention boundary move with the fast/slow
bandwidth ratio* — is the experiment no discrete-GPU lab can run, because our fast tier size
is a BIOS setting `[M]` (`notebook/uma-carveout-controls-fast-tier.md`). Section 2.5 supplies
the constraint that makes it non-trivial: **refetch and recompute have different
granularities.** Refetching costs `c · (tokens refetched)` bytes at slow-tier bandwidth and
can be done per token. Recomputing costs a full prefill over the prefix at *every* layer and
cannot be done per token. So the crossover is not a simple bandwidth comparison; it is:

```
refetch wins  when   c · k / B_slow   <   prefill_time(prefix of length t)
```

for `k` tokens missing at position `t`. The right-hand side is independent of `k`. Therefore
**refetch always wins for small `k`, and the interesting question is only where `k` gets
large** — which happens exactly when the fast tier is small relative to the working set.
That is the shape of the experiment, and it comes out of §2.5 rather than out of any paper.

### 4.5 Config surface

Every quantity in §3 must be a Proteus config field, because the config surface is the
experimental surface:

| Field | Budget it moves | Why it must be explicit |
|---|---|---|
| `layer_types` (explicit list) | residency slope, `AI_stack` | §3.3 — one list controls both, and they move in opposite directions |
| `sliding_window` | the constant term, the byte-parity point | §3.2 — below `T = L_w·w/L_g` the hybrid is a pure cost |
| `kv_dtype` | all three, and `AI` | §3.7 — and on this hardware it moves them in opposite directions |
| `cache_implementation` | maintenance traffic, peak transient, failure mode | §3.6 — **not** currently a standard ablation axis anywhere, and it should be one here |
| `max_cache_len` | preallocated read traffic | §3.5 |

`cache_implementation` is the one to argue for. It is not an architecture field, which is
precisely why it belongs to Mnemosyne rather than Proteus, and it is the axis this module
shows can dominate a measurement.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 The write path, and the two classes that implement it

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `past_key_values.update(key_states, value_states, self.layer_idx)` — the one line in the whole model where bytes enter the cache. Note what has already happened above it: QK-norm at `:390`, RoPE at `:394`. **A cached key is `RoPE(RMSNorm(k))`** — a doubly-transformed quantity, which is why any later re-positioning or requantisation scheme is not operating on the raw projection. |
| `architecture/transformers/src/transformers/cache_utils.py:112` | `class DynamicLayer` — the default. Four lines of state and one `update`. |
| `architecture/transformers/src/transformers/cache_utils.py:143` | `self.keys = torch.cat([self.keys, key_states], dim=-2)` — **the whole of §3.4 is this line.** `torch.cat` allocates and copies; there is no reserve, no capacity, no geometric growth. Read it and then read `:144`, which does it again for V. |
| `architecture/transformers/src/transformers/cache_utils.py:372` | `class StaticLayer` — the alternative, in the same file. Docstring says it exists for `torch.compile`, not for traffic; the traffic win is a side effect nobody advertises. |
| `architecture/transformers/src/transformers/cache_utils.py:406` | `self.keys = torch.zeros(...)` at `max_cache_len` — the whole worst case, allocated on the first forward. |
| `architecture/transformers/src/transformers/cache_utils.py:454` | `self.keys.index_copy_(2, cache_position, key_states)` — in-place, `c` bytes, no allocation. Put `:143` and `:454` side by side; that pair is the module. |
| `architecture/transformers/src/transformers/cache_utils.py:466` | `kv_length = self.max_cache_len` — §3.5. The static cache tells the mask builder its length is the *provisioned* length, so attention is handed the whole buffer every step. |

### 5.2 The heterogeneous cache, in three codebases

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/cache_utils.py:1184` | `DYNAMIC_LAYER_TYPE_MAPPING` — the dispatch table. Nine layer types, including `linear_attention` and `hybrid`, which is a preview of Track C's constant-state module. |
| `architecture/transformers/src/transformers/cache_utils.py:1187` | `"sliding_attention": DynamicSlidingWindowLayer` — and the comment above it, that sliding and chunked attention are *the same cache* and differ only in the mask. Layer type is a cache decision and a mask decision, and they are separable. |
| `architecture/transformers/src/transformers/cache_utils.py:1645` | `get_layer_types_and_kwargs` — the fallback ladder when `layer_types` is absent. Note `:1653`: a bare `sliding_window` field makes **every** layer sliding. A config that omits `layer_types` gets a completely different memory profile from one that specifies it, with no error. |
| `architecture/transformers/src/transformers/cache_utils.py:1737` | The list comprehension that builds a 48-element heterogeneous cache. This is the entire hybrid mechanism at the cache layer. |
| `architecture/transformers/src/transformers/cache_utils.py:240` | `self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]` — the windowed layer's residency rule, and an off-by-one worth noticing: it **stores** `w-1` and **returns** `w`. The retained tensor and the attended tensor are different objects. Any instrumentation that measures "cache size" will read `w-1`; any instrumentation that measures "bytes attended over" will read `w`. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73` | `size_swa` — the same idea in C++ as two physically separate allocations, sized `min(size_base, n_swa·n_seq + n_ubatch)` padded to 256. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:197` | `page_size_bytes` — vLLM's version: one pool, several specs, different page geometry per layer type. |

### 5.3 The formula, and the flag that deletes the windowing saving

This is the most valuable read in the module.

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/kv_cache_interface.py:204` | `real_page_size_bytes` — `2 * block_size * num_kv_heads * head_dim * dtype_size`. Our `c`, times `block_size`, in production code. The `2` is K and V; the branches above it handle nvfp4 and int4 by changing `head_dim`, which is `b` moving in disguise. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:227` | `class FullAttentionSpec` — **read the docstring.** "When hybrid allocator is disabled and the model contains both full attention layers and sliding window attention layers, sliding window attention are regarded as full attention in KV cache manager (blocks are allocated for all tokens), while computed as sliding window attention in model runner." In plain terms: **with one feature flag off, the 4× capacity saving from Laguna's windowing silently does not happen** — the blocks are allocated for every token anyway, and only the mask is windowed. The model is unchanged. The config is unchanged. The residency is 4× higher. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:567` | `max_admission_blocks_per_request` — the fixed term, allocated: `cdiv(sliding_window - 1 + in_flight_tokens, block_size) + 1`. The `- 1` is the same off-by-one as `cache_utils.py:240`; the `+ 1` is because the window need not start on a block boundary. Internal fragmentation, made explicit. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `get_new_blocks` — the allocator. On failure there is no fault and no demotion; the request is preempted. **There is no miss path anywhere in a KV cache.** |
| `memory/vllm/vllm/v1/core/block_pool.py:719` | `free_blocks` — and the reason eviction can be a pointer update rather than a compaction (§4.2). |

**The generalisation, and it is the one to carry:** residency is a property of the
**allocator**, read traffic is a property of the **kernel**, maintenance traffic is a
property of the **cache class**. None of the three is a property of the architecture, and
the config file describes none of them.

### 5.4 Layout, quantisation, and the offload hook

| Where | What to look at, and why |
|---|---|
| `memory/flashinfer/flashinfer/page.py:403` | `append_paged_kv_cache` — the docstring gives the two physical layouts: `[pages, page_size, n_kv_heads, head_dim]` (NHD) versus `[pages, n_kv_heads, page_size, head_dim]` (HND). Identical bytes, different stride order. It decides whether one head's read is a contiguous burst or a strided gather — i.e. whether you get the 150 GB/s of Exercise B or something much worse. `n_kv` is a *layout axis*, not just a count. |
| `architecture/transformers/src/transformers/cache_utils.py:724` | `keys_to_return = torch.cat([dequant_keys, self.keys, key_states], dim=-2)` — HF's quantised cache **dequantises the entire cache on every step** and concatenates three tensors to do it. This is §3.7's 5-bytes-per-element path, in the reference implementation, with a third concat on top. If you benchmark KV quantisation here you will measure the cost of the harness. |
| `architecture/transformers/src/transformers/cache_utils.py:77` | `self.keys = self.keys.to("cpu", non_blocking=True)` — the offload hook, with a `prefetch` counterpart at `:80` and a dedicated stream at `:1260`. Two-tier KV in fourteen lines. Forward reference: the curriculum picks this up in `paged-attention-and-prefix-reuse.md` and the survey behind it is `research/memory/kv-serving-hierarchy.md`; note here only that the granularity is **a whole layer**, and that `offload_only_non_sliding` defaults to true because the windowed layers are too small to be worth moving — which is Exercise B's 1.2%-of-the-bytes result, arrived at by a completely different route. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319` | `attn_rot_k` — a Hadamard rotation applied to K and V before a quantised store, to smear outliers across quantisation blocks. Gated by a runtime heuristic, not by the model config. Another instance of the same lesson. |

---

## 6. Exercises

All three were run on the Z13 for this module, and the reference numbers below are `[M]`
from those runs. Your absolute numbers will differ; the **ratios** are the deliverable, and
each exercise states a prediction the machine is free to falsify. Exercise B falsified one of
mine outright; Exercise C came out 20% worse than predicted, in the same direction. Only
Exercise A went entirely to plan, which is roughly the hit rate you should expect and is why
the predictions are written down before the tables.

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats** (`ASSUMPTIONS.md`): single tensors ≥32 GiB **hang silently at
0% CPU** — keep every buffer under 31 GiB; bf16 numerics on gfx1151 are unproven
(`bf16-numerics-unproven`), so accuracy claims from these exercises are provisional while
*timing* claims are not; `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is deliberately **off**
in `activate-lab.ps1` and these exercises were run with it off. The Hardware Validation Gate
has not run, so nothing here is evidence by house standard — these are instrument-shakedown
runs, and the module labels them as such.

Write scratch scripts under `notebook/`. Exercises B and C are Hardware Validation Gate
candidates and should migrate into the rig with tests on reuse.

---

### Exercise A — price the maintenance budget: `torch.cat` versus preallocation

**Goal:** turn §3.4 from algebra into a curve, and find out how much of a 4096× byte
amplification actually shows up in wall clock.

**Hardware:** one gfx1151 GPU. **CPU fallback:** identical code, set `T` values to
`[256, 512, 1024, 2048]`; the shape of the curve transfers and the runtime stays under two
minutes. **Runtime:** ~2 minutes on GPU at the shapes given.

```python
"""Maintenance traffic: DynamicLayer (torch.cat) vs StaticLayer (index_copy_).
Shapes are Laguna-S per-layer: n_kv=8, head_dim=128, bf16, batch 1."""
import statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, N_KV, D_H, B = torch.bfloat16, 8, 128, 1
C = 2 * N_KV * D_H * 2            # bytes per token per layer = 4096

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def grow_cat(T):                                   # cache_utils.py:143
    k = torch.empty(B, N_KV, 0, D_H, dtype=DT, device=DEV)
    v = torch.empty(B, N_KV, 0, D_H, dtype=DT, device=DEV)
    new = torch.randn(B, N_KV, 1, D_H, dtype=DT, device=DEV)
    sync(); t0 = time.perf_counter()
    for _ in range(T):
        k = torch.cat([k, new], dim=-2)
        v = torch.cat([v, new], dim=-2)
    sync(); return time.perf_counter() - t0

def grow_static(T):                                # cache_utils.py:406 + :454
    k = torch.zeros(B, N_KV, T, D_H, dtype=DT, device=DEV)
    v = torch.zeros(B, N_KV, T, D_H, dtype=DT, device=DEV)
    new = torch.randn(B, N_KV, 1, D_H, dtype=DT, device=DEV)
    pos = torch.zeros(1, dtype=torch.long, device=DEV)
    sync(); t0 = time.perf_counter()
    for i in range(T):
        pos.fill_(i)
        k.index_copy_(2, pos, new)
        v.index_copy_(2, pos, new)
    sync(); return time.perf_counter() - t0

grow_cat(64); grow_static(64)                      # warm the allocator
print(f"{'T':>6} {'cat us/step':>12} {'static us/step':>15} {'wall ratio':>11} {'byte ratio':>11}")
for T in [512, 1024, 2048, 4096, 8192]:
    tc = statistics.median(grow_cat(T) for _ in range(3))
    ts = statistics.median(grow_static(T) for _ in range(3))
    print(f"{T:>6} {tc/T*1e6:>12.1f} {ts/T*1e6:>15.1f} {tc/ts:>11.2f} {(T+1)/2:>11.1f}")
```

**Predictions, stated before you run.**
1. `static us/step` is **flat** in `T` — it moves 4 KiB regardless.
2. `cat us/step` is **linear** in `T` — it moves `c·T`.
3. The wall ratio is **far below** the byte ratio, because the static path is too small to be
   bandwidth-bound at all.

**`[M]` Reference numbers**, Z13 / gfx1151 / native Windows, torch
`2.12.0a0+rocm7.13.0a20260313`, median of 3 runs per cell, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`
unset, 2026-07-26:

| `T` | cat µs/step | static µs/step | wall ratio | byte ratio `(T+1)/2` | cat bytes written |
|---|---|---|---|---|---|
| 512 | 18.1 | 9.4 | 1.94 | 256.5 | 0.54 GB |
| 1,024 | 32.8 | 8.4 | 3.92 | 512.5 | 2.15 GB |
| 2,048 | 56.1 | 10.7 | 5.24 | 1,024.5 | 8.59 GB |
| 4,096 | 131.3 | 9.9 | 13.28 | 2,048.5 | 34.37 GB |
| 8,192 | 234.1 | 10.2 | **23.00** | 4,096.5 | **137.46 GB** |

All three predictions hold. Static is flat at 8.4–10.7 µs/step — it is **launch-bound, not
bandwidth-bound**; appending a token to a preallocated cache is free and its cost is two
kernel launches. Cat is linear beyond `T ≈ 1024` (8× the context gives 7.1× the per-step
time). And the wall ratio is 23×, not 4096×, precisely because the denominator is a fixed
launch cost rather than a byte cost.

**Deliverables — three numbers and one plot.**
1. Plot `cat µs/step` against `T`. **Fit a line and report R² and intercept.** The intercept
   is your platform's per-step launch overhead; ours is ~10 µs, and you can read it directly
   off the static column as a cross-check.
2. The wall ratio at your largest `T`. Report it next to `(T+1)/2` and explain the gap in one
   sentence.
3. **Test §3.4's identity end to end.** Over the whole 8,192-token generation on Laguna's 12
   global layers, compare *measured* maintenance time against *derived* read time:

   ```
   per-layer total       = 234.1 us x 8192     = 1.918 s  [M]  (the T=8192 row)
   measured maintenance  = 12 x 1.918 s        = 23.0 s
   maintenance traffic   = 12 × 2 × c·T²/2     = 3.30 TB  → implies 143 GB/s, sane
   read-for-use traffic  = 12 × c·T²/2         = 1.65 TB  → 11.0 s at the 150 GB/s of Exercise B
   ratio                 = 23.0 / 11.0         = 2.09×
   ```

   §3.4 predicts **exactly 2.00×**. Ours came out at 2.09×. Report yours. A ratio far from 2
   means either your concat is not doing what you think or your read-bandwidth estimate is
   wrong, and the two are distinguishable by checking whether the implied maintenance
   bandwidth (3.30 TB ÷ your time) is physically possible.

**What a falsification would mean.** If `cat µs/step` is flat, your torch build is doing
something clever — an in-place realloc into a caching-allocator block with spare capacity.
Check `k.untyped_storage().data_ptr()` before and after; if it is stable across steps, that
is a genuinely interesting finding and worth a notebook entry, because it would mean the
`O(T²)` result does not hold on this stack.

---

### Exercise B — the two-tier stack: measure how wrong a single-number bandwidth model is

**Goal:** the module's central claim. Measure a decode-shaped attention read at Laguna's two
layer types, build the 48-layer stack from the two measurements, and compare against the
single-number model everyone uses.

**Hardware:** one gfx1151 GPU. **CPU fallback:** set contexts to `[1024, 4096]` and
`iters=5`; expect one to two orders of magnitude less bandwidth, but the *sliding-vs-global*
asymmetry — the thing being tested — survives. **Runtime:** ~3 minutes on GPU.

**Footprint check first:** the largest allocation is `K` at `T = 131072`:
`8 × 131072 × 128 × 2 B = 268 MB`, and `K + V = 537 MB`. Far inside the `[M]` ≥62 GiB fast
tier and far below the 31 GiB per-tensor hazard.

```python
"""Per-layer-type decode reads, and the stack model they imply."""
import json, statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, N_KV, D_H = torch.bfloat16, 8, 128
L_GLOBAL, L_WINDOW, WINDOW = 12, 36, 512          # Laguna-S, [M] from config.json

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def decode_step(K, V, q):
    s = torch.bmm(q, K.transpose(1, 2)) * (D_H ** -0.5)
    w = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, V)

def time_layer(T, G, iters=50):
    K = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    V = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    q = torch.randn(N_KV, G, D_H, dtype=DT, device=DEV)
    for _ in range(5): decode_step(K, V, q)
    sync(); t0 = time.perf_counter()
    for _ in range(iters): decode_step(K, V, q)
    sync(); s = (time.perf_counter() - t0) / iters
    kv_bytes = 2 * N_KV * T * D_H * 2
    del K, V, q
    if DEV == "cuda": torch.cuda.empty_cache()
    return dict(ms=s * 1e3, kv_bytes=kv_bytes, gb_s=kv_bytes / s / 1e9)

REFERENCE_GB_S = 199.9                             # [M] ASSUMPTIONS.md, device-to-device copy
for ctx in [4096, 16384, 32768, 131072]:
    g = statistics.median(time_layer(ctx, 6)["ms"] for _ in range(3))
    s = statistics.median(time_layer(WINDOW, 9)["ms"] for _ in range(3))
    gb = 2 * N_KV * ctx * D_H * 2
    sb = 2 * N_KV * WINDOW * D_H * 2
    stack_ms    = L_GLOBAL * g + L_WINDOW * s
    stack_bytes = L_GLOBAL * gb + L_WINDOW * sb
    naive_ms    = stack_bytes / (REFERENCE_GB_S * 1e9) * 1e3
    print(json.dumps(dict(
        context=ctx, global_ms=round(g, 4), sliding_ms=round(s, 4),
        stack_ms=round(stack_ms, 2), stack_gib=round(stack_bytes / 2**30, 3),
        naive_ms=round(naive_ms, 2), model_is_optimistic_by=round(stack_ms / naive_ms, 2),
        sliding_share_of_bytes=round(L_WINDOW * sb / stack_bytes, 4),
        sliding_share_of_time=round(L_WINDOW * s / stack_ms, 4))))
```

**Predictions, stated before you run.**
1. Achieved bandwidth on the **global** layer approaches the `[M]` ~200 GB/s reference at
   large `T`.
2. Achieved bandwidth on the **sliding** layer is much lower, because 2 MiB is not enough to
   amortise anything.
3. Therefore the sliding layers' share of *time* greatly exceeds their share of *bytes*, and
   the naive model is optimistic. **Prediction 1 is the one that failed.**

**`[M]` Reference numbers**, same configuration as Exercise A, median of 3 per cell:

| Context | global layer (`T`=ctx, `G`=6) | sliding layer (`T`=512, `G`=9) | stack | naive @199.9 GB/s | optimistic by | sliding % bytes | sliding % time |
|---|---|---|---|---|---|---|---|
| 4,096 | 0.126 ms · 133.5 GB/s | 0.080 ms · 26.2 GB/s | 4.39 ms | 1.39 ms | **3.17×** | 27.3% | **65.7%** |
| 16,384 | 0.511 ms · 131.3 GB/s | 0.077 ms · 27.1 GB/s | 8.92 ms | 4.41 ms | **2.02×** | 8.6% | **31.2%** |
| 32,768 | 0.894 ms · 150.1 GB/s | 0.131 ms · 16.1 GB/s | 15.43 ms | 8.44 ms | **1.83×** | 4.5% | **30.5%** |
| 131,072 | 3.598 ms · 149.2 GB/s | 0.111 ms · 19.0 GB/s | 47.15 ms | 32.61 ms | **1.45×** | 1.2% | **8.4%** |

Read the 32,768 row twice. **The windowed layers hold 4.5% of the bytes and consume 30.5% of
the time.** That is a 6.8× penalty in time-per-byte, and no amount of correcting `G` per
layer type predicts it — the `G` correction changes the FLOP side, and this is entirely on
the byte-rate side.

Prediction 1 failed: the global layer tops out at ~150 GB/s, **75% of the 199.9 GB/s
device-to-device copy figure**. That is not a contradiction of `ASSUMPTIONS.md → gpu-fast-tier-size`
— a copy benchmark counts one read plus one write and streams perfectly, while a decode
attention read is a strided gather feeding a matmul with `M = 6`. It does mean the ridge
point in `ASSUMPTIONS.md` (20.9 TFLOP/s ÷ 199.9 GB/s ≈ 105) is built on the wrong
denominator for this workload, which is open question 6 in the mirrored survey note.

**Deliverables — three numbers and one plot.**
1. The `model_is_optimistic_by` column. **One number per context; report all four.**
2. Plot `sliding_share_of_bytes` and `sliding_share_of_time` on the same axes against
   context. The gap between the curves is the module's thesis, drawn.
3. Achieved GB/s on the global layer at your largest context, against the 199.9 GB/s
   reference. Report the ratio and say which of the two numbers you would put in a cost
   model.

**Repeatability, because these numbers carry the module.** The two load-bearing cells were
re-measured in a **third, separate process**: `T = 512, G = 9` came back at 0.1143 and
0.1104 ms (18.3 and 19.0 GB/s) against the table's 0.111 ms, and `T = 131072, G = 6` came
back at 3.5977 ms / 149.2 GB/s against the table's 3.598 ms / 149.2 GB/s — agreement to four
significant figures. The `T = 32768` global cell was noisier: 1.035 ms / 129.6 GB/s on the
recheck versus 0.894 ms / 150.1 GB/s in the table, a 15% spread. Treat the mid-context row as
±15% and the endpoints as solid. `[M]`

**The follow-up that makes this attributive rather than descriptive** — and it is required,
not optional, because a slow number without a mechanism is not a finding. Three candidate
mechanisms explain the sliding layer's 19 GB/s: (a) kernel-launch overhead across the five
or six small kernels this decomposition issues; (b) an `M = 9` GEMM, which is a
matrix-*vector* product and grossly under-occupies the compute units; (c) 2 MiB genuinely
being too few bytes to reach streaming bandwidth.

Establish your launch floor first — time a trivial op on a 64-element tensor. `[M]` Ours is
**10.1 µs** per dependent kernel (median of 5 × 50 iterations, fresh process), which
independently corroborates Exercise A's flat ~10 µs/step static-append column. That gives a
partial attribution straight away:

```
sliding layer measured                  110 µs
  ~6 dependent kernel launches @10.1 µs  ~61 µs   (55%)
  2.10 MB at the 150 GB/s of the global layer  14 µs   (13%)
  unattributed                            ~35 µs   (32%)
```

So it is **mostly (a), partly (c), and about a third unexplained** — most plausibly the
`M = 9` GEMM of (b), but that is a hypothesis, not a measurement.

The obvious discriminator — re-run the same shape through a single
`F.scaled_dot_product_attention` call — **does not work on this machine and the reason is
worth knowing.** With `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset, SDPA on gfx1151 emits
`UserWarning: Flash Efficient attention on Current AMD GPU is still experimental` and falls
back to the math backend, which is *not* fused and additionally materialises the score matrix
(`ASSUMPTIONS.md → sdpa-is-memory-efficient`, `[M]` 147.2 bytes/T² retained). In our attempt
the "fused" arm ran 2–5.6× **slower** than the hand-rolled decomposition, which measures the
fallback, not fusion. **Do not tag that run.** Getting a real fused arm means either turning
the experimental flag on — which is a numerics change and a Hardware Validation Gate item, not
a benchmarking convenience — or writing the kernel. That is §8, item 2, and it is genuinely
open.

---

### Exercise C — does halving `b` halve the time? (It does not.)

**Goal:** answer open question 2 of `research/memory/kv-cache-mechanics.md` on our own
instrument, and separate the *capacity* win of fp8 from the *bandwidth* win.

**Hardware:** one gfx1151 GPU. **CPU fallback:** set contexts to `[4096, 16384]`. Unverified
caveat: if your build does not implement `float8_e4m3fn` conversions on CPU the script will
raise at the first `.to()`, in which case the exercise is GPU-only — **record that as the
finding rather than working around it**, because dtype support per device is exactly the kind
of thing that changes under you between wheels. **Runtime:** ~2 minutes.

```python
"""fp8 storage + bf16 compute vs plain bf16, and where the time goes."""
import json, statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, N_KV, D_H, G = torch.bfloat16, 8, 128, 6

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def bench(fn, iters=50):
    for _ in range(5): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters * 1e3

def attention(K, V, q):
    s = torch.bmm(q, K.transpose(1, 2)) * (D_H ** -0.5)
    w = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.bmm(w, V)

for T in [16384, 65536, 131072]:
    Kb = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    Vb = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    K8, V8 = Kb.to(torch.float8_e4m3fn), Vb.to(torch.float8_e4m3fn)
    q = torch.randn(N_KV, G, D_H, dtype=DT, device=DEV)
    bf16   = statistics.median(bench(lambda: attention(Kb, Vb, q)) for _ in range(5))
    deq    = statistics.median(bench(lambda: (K8.to(DT), V8.to(DT)))  for _ in range(5))
    fp8    = statistics.median(bench(lambda: attention(K8.to(DT), V8.to(DT), q)) for _ in range(5))
    print(json.dumps(dict(T=T, bf16_ms=round(bf16, 4), dequant_ms=round(deq, 4),
                          fp8_path_ms=round(fp8, 4), slowdown=round(fp8 / bf16, 3),
                          dequant_share=round(deq / fp8, 3))))
    del Kb, Vb, K8, V8, q
    if DEV == "cuda": torch.cuda.empty_cache()
```

**Prediction, from the traffic accounting in §3.7:** the fp8 path moves 5 bytes per element
against bf16's 2, so it should be **~2.5× slower** while holding **half** the bytes.

**`[M]` Reference numbers**, same configuration, median of 5 per cell, run in a **fresh
process** from the other two exercises:

| `T` | bf16 attention | dequantise only | fp8 → bf16 → attention | slowdown | dequant's share | bf16 read GB/s |
|---|---|---|---|---|---|---|
| 16,384 | 0.476 ms | 0.944 ms | 1.432 ms | **3.01×** | 65.9% | 140.9 |
| 65,536 | 1.739 ms | 3.362 ms | 5.090 ms | **2.93×** | 66.1% | 154.4 |
| 131,072 | 3.455 ms | 6.559 ms | 10.079 ms | **2.92×** | 65.1% | 155.4 |

An earlier run in a different process, at a different tensor arrangement (`N_KV` folded into
a batch axis) and `REPEATS=3`, gave 3.13× and 3.08× at `T = 16384` and `65536`. **Six
measurements, two processes, three context lengths, one ratio.** That is repeatable enough
to be `[M]` by the standard this curriculum adopted after a previous module tagged a
non-reproducing crash as measured.

The prediction was 2.5× and the measurement is 2.9–3.1×, with the extra plausibly the
conversion kernel's own inefficiency. **The attribution is the valuable part: dequantising
costs twice as much as the entire bf16 attention it feeds.** The traffic model is right about
the mechanism and slightly optimistic about the magnitude.

**Deliverables — two numbers and a decision.**
1. The slowdown ratio at each context. Is it flat in `T`? (Ours is — which says it is a
   traffic effect, not an overhead effect. A ratio that fell with `T` would indicate fixed
   overhead instead, and would change the conclusion.)
2. `dequant_share`. If it is above 50%, the fp8 cost is in materialising the temporary, and
   the fix is kernel fusion, not a different format.
3. **Write one line in your notebook entry:** for a *capacity*-bound experiment, is fp8 KV
   still worth using on this machine? (It is: half the residency for a 3× read cost is a good
   trade when residency is what is binding and you are not measuring tokens/s. Say so
   explicitly, because the reflex from the shape math is the opposite.)

**What a falsification would mean.** If fp8 comes out *faster*, `torch._scaled_mm` or an fp8
attention path became available in your wheel — which would be a change of instrument and a
Hardware Validation Gate re-run, not a happy accident. Record the wheel version.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. A serving engineer tells you their KV cache "uses 6 GiB at 128k context" on Laguna-S and
   that decode therefore costs `6.07 GiB ÷ 200 GB/s ≈ 33 ms/token`. Both halves of that
   sentence use a measured input. Which half is wrong, by roughly how much, and what is the
   mechanism?

2. You are asked for the arithmetic intensity of Laguna-S decode attention. Give the answer
   as a function of context length, and state the two values it lies between and why it moves
   in the direction it does.

3. Over a `T`-token generation with a grow-by-concatenation cache, how does the traffic spent
   *maintaining* the cache compare to the traffic spent *reading* it? Give the ratio and say
   what it depends on.

4. Exercise A measures a 4096× byte amplification and a 23× wall-clock ratio at the same
   `T`. Both numbers are correct. Reconcile them in one sentence.

5. Two arms of an experiment are identical except that one uses `DynamicCache` and the other
   `StaticCache` with `max_cache_len = 131072`. Both run at an actual context of 1,024
   tokens. Name one budget on which each arm is worse than the other, and predict the sign of
   the difference in measured tokens/s.

6. A colleague proposes storing the KV cache in fp8 on this machine, citing `AI = 2G/b` to
   argue it doubles arithmetic intensity. The measurement says it is 3× slower. Is the
   formula wrong?

---

## 8. What is still unsolved here

Everything below is testable at 20M–300M params on one GPU against a `[M]` ≥62 GiB fast
tier, and every item needs a pre-registered hypothesis card before it runs.

1. **The `cache_implementation` axis is not in anyone's ablation design.** Exercise A shows a
   23× wall-clock difference from a choice that appears in no config file, no paper's methods
   section, and no reported result. As far as this survey pass could establish, **no published
   KV-cache paper states which cache class its baseline used.** That is either a large
   confound in the literature or a non-issue because everyone uses paged serving engines — and
   nobody has checked which. Cheapest test: reproduce one published eviction-policy speedup
   under both cache classes and see whether the effect size moves.

2. **A third of the sliding layers' cost is unattributed, and the discriminator is blocked.**
   Exercise B's follow-up pins ~55% of the 110 µs on kernel-launch overhead (`[M]` 10.1 µs per
   dependent launch) and ~13% on bytes, leaving ~32% most plausibly on `M = 9` GEMM
   under-occupancy — a hypothesis, not a measurement. The clean discriminator, a genuinely
   fused attention call, **is not available on this stack**: SDPA falls back to the math
   backend with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset, and turning the flag on is a
   numerics change gated behind the Hardware Validation Gate. This matters beyond bookkeeping:
   if the cost is launch overhead, a fused kernel removes it and the hybrid's wall-clock
   penalty largely disappears; if it is occupancy, the penalty is structural and **the 3:1
   ratio is a worse deal in wall clock than it is in bytes** — which would be a real result
   about hybrid architectures and not just about our instrument. Cheapest path: run the gate's
   numerics suite both ways, then re-run Exercise B with the flag on.

3. **The attention-kernel ridge point is still not measured.** `ASSUMPTIONS.md` carries
   ≈105 FLOP/byte from an 8192³ GEMM over a device-to-device copy. Exercise B shows the
   denominator is wrong for decode attention by 25%. The numerator is wrong too — a
   decode-shaped `M = 6` matmul will not reach 20.9 TFLOP/s. Nobody in the literature reports
   a decode-attention roofline for a consumer APU, and it is the number that actually governs
   every wall-clock estimate in `BACKLOG.md`.

4. **Whether the preallocated cache actually reads its unwritten rows.** §3.5 is a code
   reading (`cache_utils.py:466`), not a measurement. If the backend skips fully-masked tiles
   the defect is theoretical; if not, `StaticCache` is unusable for short-context work at long
   `max_cache_len`. One sweep of `max_cache_len` at fixed `T` settles it, and it directly
   affects whether Mnemosyne's oracle can use `StaticCache`.

5. **Sequence-axis compression is a fourth budget line and nobody has priced its maintenance.**
   DeepSeek-V4's Compressed Sparse Attention folds groups of `m` tokens into one entry `[C]`
   (2606.19348, Apr 2026 — vendor-reported, not independently replicated), and the layer type
   already exists in the HuggingFace dispatch table (`cache_utils.py:1672`). Folding is a
   read-modify-write over the group, so it has a maintenance cost the compression ratio does
   not disclose. The shape math predicts feature-axis and sequence-axis compression multiply
   cleanly on residency; nothing predicts what they do to maintenance.

6. **Contested: contiguous virtual memory versus paging.** PagedAttention `[C]` (2309.06180)
   is treated as settled, but vAttention `[C]` (2405.04437, May 2024, rev. Jan 2025) argues
   the non-contiguous layout is a self-inflicted wound and that CUDA virtual-memory APIs give
   you dynamic allocation *and* a contiguous layout, reporting up to 1.23× over paged
   FlashAttention and FlashInfer. The 2026 line continues — page-aware decode scheduling on
   commodity GPUs `[C]` (2606.26666, Jun 2026) reports 1.04–1.40× purely from *scheduling*
   which kernel sees which page layout. Present as contested. Note the constraint that decides
   it for us: there is no ROCm equivalent of the CUDA VMM path validated on gfx1151, so
   vAttention's design is not currently available on our instrument even if it is right.

7. **Nobody reports the three budgets separately.** The 2026 serving surveys `[C]`
   (2607.02574, Jun 2026; 2607.08057, Jul 2026) enumerate what the field does not measure and
   this decomposition is not in either list, which is either an oversight or evidence that it
   is obvious to practitioners and unwritten. Either way it is a cheap methodological
   contribution: instrument all three counters in Mnemosyne, report all three on every arm,
   and the attribution failure the lab exists to attack gets one axis narrower.

8. **`c` is exact but the *effective* `c` under quantisation is not.** vLLM's
   `unpadded_page_size_bytes` (`kv_cache_interface.py:185`) adds two fp32 scales per token
   per head for per-token-head quantisation modes. At `n_kv = 8` that is 64 bytes per token
   per layer against a 2048-byte int8 payload — a **3.1% overhead** that no compression-ratio
   claim in the literature includes. Sub-4-bit claims should be re-derived with the scale
   tensor counted; at 2 bits the payload is 512 bytes and the same scales are a **12.5%**
   overhead, which is large enough to change a ranking.

---

## Answers to the self-check

**1.** The residency half is right — `R(131072) = 6.070 GiB` exactly, from
`12 × 4 KiB × 131072 + 36 × 4 KiB × 512`. The bandwidth half is wrong by `[M]` **1.45×**:
measured 47.15 ms against the naive 32.61 ms. Two mechanisms stack. First, the achieved
bandwidth on a decode-shaped attention read is ~150 GB/s, not 199.9 — the copy benchmark
streams and counts a write, this is a strided gather feeding an `M = 6` matmul. Second, the
36 windowed layers hold 1.2% of the bytes and take 8.4% of the time, because a 2 MiB read
never reaches streaming rate. Neither mechanism is visible in the shape math, and the error
grows as context *shrinks* — 3.17× at 4k — which is the opposite of most people's intuition.

**2.** `AI_stack(T) = (576·T + 1,327,104) / (96·T + 147,456)` for `T ≥ 512`. It falls
monotonically from **8.25** at `T = 512` to **6.00** asymptotically. It moves *down* because
it is a byte-weighted average of the per-layer intensities (9 on the sliding layers, 6 on the
global ones) and the weight shifts toward the global layers as `T` grows, since only they
accumulate bytes. The mechanically important consequence: intensity is not a constant of the
architecture, and a Mnemosyne cost model must recompute it per context length, not cache it.

**3.** They are **equal**, and the equality is exact and unconditional. Reads for use over the
generation are `Σ_{t=1..T} c·t = c·T(T+1)/2`; bytes written by the concat are the same sum.
Counting the concat's read side too, maintenance is 2× the read traffic and total decode
traffic is **3×** what the shape math predicts. It depends on nothing — not `c`, not `T`, not
the model, not the hardware. It is a property of the data structure alone.

**4.** The static baseline is not bandwidth-bound: it moves 4 KiB per step, which costs two
kernel launches (~10 µs) and no measurable memory time, so the denominator of the wall-clock
ratio is a *fixed* cost while the denominator of the byte ratio is a *byte* cost — the two
ratios are measuring against different things, and the 23× is the honest one for wall clock
while the 4096× is the honest one for the memory bus.

**5.** `DynamicCache` is worse on **maintenance traffic** — at `T = 1024` it has moved ~2.15 GB
per layer against `StaticCache`'s 4.19 MB. `StaticCache` is worse on **residency**: on
Laguna's hybrid it commits `12 × 4 KiB × 131072 + 36 × 4 KiB × 512` = **6.07 GiB** against the
**120 MiB** actually needed at `T = 1024`, a 52× over-commitment — and, per §3.5, plausibly
worse on **read traffic** by 128× on the global layers (52× on the whole stack), since it
hands attention the full provisioned buffer. Predicted sign on tokens/s: at `T = 1024` the
dynamic cache's maintenance is still small in absolute terms (33 µs/step from Exercise A)
while the static cache's read amplification, if real, is 52× on the dominant term — so
**`StaticCache` should be slower here**, and the crossover moves in `DynamicCache`'s favour
only as `T` approaches `max_cache_len`. This is exactly the measurement in unsolved item 4, and if the kernel skips
masked tiles the prediction inverts. Getting the *sign* wrong here is easy, which is the point
of the question.

**6.** No. `AI = 2G/b` is a statement about the bytes the *attention operation* must read, and
it is correct: with an fp8 cache the attention operation reads half the bytes. The formula
does not claim anything about how those bytes get into a form the kernel accepts. On hardware
with fp8 tensor cores the formula and the wall clock agree; on gfx1151, where
`torch._scaled_mm` is unavailable `[M]`, the path is fp8 → bf16 temporary → attention, which
moves 5 bytes per element instead of 2. The formula describes a *lower bound* that requires a
kernel able to realise it, exactly like Track B's `repeat_kv` result: the arithmetic-intensity
number is what the hardware could do, not what your stack does. **Measure the traffic, do not
read it off the algebra** — the same rule as "measure the cache, do not read it off the
config," one level down.

---

## Sources

**Local measurements (`[M]`), all 2026-07-26.** Z13, Ryzen AI Max+ 395, Radeon 8060S
(gfx1151), native Windows, torch `2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0), venv
`C:\venvs\lab`, `HIPBLASLT_TENSILE_LIBPATH` and `TORCH_BLAS_PREFER_HIPBLASLT=1` set per
`scripts/activate-lab.ps1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` **unset**. Shapes are
Laguna-S per-layer: `n_kv=8`, `head_dim=128`, bf16, batch 1. The exercise listings in §6 are
the scripts that produced these numbers; they were run from a scratch directory and are
**not committed**, so by house standard these are instrument-shakedown numbers and not
evidence until the Hardware Validation Gate closes.

- Exercise A, cache growth: median of 3 per cell, `T ∈ {512, 1024, 2048, 4096, 8192}`. At
  `T = 8192`: concat 234.1 µs/step, preallocated 10.2 µs/step, ratio 23.00×.
- Exercise B, per-layer-type reads: median of 3 per cell, contexts
  `{4096, 16384, 32768, 131072}`. Global layer tops out at 149–150 GB/s; sliding layer
  (`T = 512`, `G = 9`) at 16–27 GB/s; naive single-number model optimistic by 1.45×–3.17×.
  Re-measured in a third process (median of 5): `T=512,G=9` → 0.1143 / 0.1104 ms;
  `T=131072,G=6` → 3.5977 ms / 149.2 GB/s; `T=32768,G=6` → 1.035 ms / 129.6 GB/s (15% off the
  table's cell, the only number here with visible run-to-run spread).
- Launch floor, same recheck process: **10.1 µs** per dependent kernel on a 64-element
  in-place add, median of 5 × 50 iterations. Used for the partial attribution in Exercise B.
- **Explicitly not tagged:** an attempt to discriminate launch-bound from bandwidth-bound by
  comparing `F.scaled_dot_product_attention` against the three-op decomposition. That run
  reported an implausible 489 µs launch floor — 48× the figure two independent processes
  agree on — so the whole process is discarded as contaminated, and separately its "fused"
  arm was not fused (SDPA fell back to the math backend, per the emitted `UserWarning`).
  Recorded here rather than deleted, because a previous module in this curriculum tagged a
  non-reproducing observation as `[M]` and the correction is cheaper than the habit.
- Exercise C, fp8: median of 5 per cell in a fresh process, contexts
  `{16384, 65536, 131072}`, plus an earlier independent run at `{16384, 65536}` with a
  different tensor arrangement. Slowdown 2.92×–3.13× across all six; dequantise is 65–66% of
  the fp8 path.

**Repo `[M]` inputs used but not re-measured here:** `ASSUMPTIONS.md` rows `reference-model`
(Laguna-S config at revision `b0a9fd7c850e`: 48 layers, 12 full + 36 sliding, `w`=512,
`n_kv`=8, `head_dim`=128), `kv-per-token-laguna` (192 KiB/token exactly),
`laguna-heads-uniform` (query heads 48/72 per layer type, KV heads uniform),
`gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per arm),
`large-tensor-fault-32gib` (199.9 GB/s device-to-device at 31 GiB; ≥32 GiB hangs),
`gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192³),
`sdpa-is-memory-efficient`, `bf16-numerics-unproven`, `torch-build`.
`notebook/uma-carveout-controls-fast-tier.md`. The fp8 `torch._scaled_mm` unavailability
probe recorded in `research/memory/kv-cache-mechanics.md`.

**Code pointers.** Every `file:line` in §5 was opened and the named symbol confirmed on the
named line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.
Pointers reused from `research/reference/CODE_MAP.md` (machine-verified by
`scripts/generate_code_map.py`): `llama-kv-cache-iswa.cpp:73`, `llama-kv-cache.cpp:319`,
`block_pool.py:647`, `block_pool.py:719`, `memory/flashinfer/flashinfer/decode.py:1239`,
`memory/flashinfer/flashinfer/page.py:403`,
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397`.
Pointers introduced by this module and
verified by reading:
`architecture/transformers/src/transformers/cache_utils.py:77`, `:112`, `:143`, `:240`,
`:372`, `:406`, `:454`, `:466`, `:724`, `:1184`, `:1187`, `:1645`, `:1653`, `:1672`, `:1737`;
`memory/vllm/vllm/v1/kv_cache_interface.py:185`, `:197`, `:204`, `:227`, `:567`.
`scripts/verify_code_pointers.py curriculum/kv-cache-mechanics.md` reports **26/26 resolving**
as of 2026-07-26.

**arXiv (`[C]`).** Every id below appears in `research/memory/citation-verification.json`
(265 ids resolved against the live arXiv API) except `2606.26666`, which was verified by
fetching its arXiv abstract page on 2026-07-26.

- `2309.06180` — *Efficient Memory Management for LLM Serving with PagedAttention*
  (2023-09-12). The answer to §3.4's growth problem.
- `2311.18677` — *Splitwise: Efficient generative LLM inference using phase splitting*
  (2023-11-30). Prefill compute-bound, decode bandwidth-bound.
- `2401.09670` — *DistServe: Disaggregating Prefill and Decoding* (2024-01-18).
- `2405.04437` — *vAttention: Dynamic Memory Management for Serving LLMs without
  PagedAttention* (2024-05-07, rev. 2025-01-29). The contiguous-layout counter-position; up
  to 1.23× over paged FlashAttention/FlashInfer.
- `2606.19348` — *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
  (2026-04-26). Sequence-axis compression. Vendor-reported, not independently replicated.
- `2606.26666` — *PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on
  Commodity GPUs* (2026-06-25, rev. 2026-07-01). 1.04–1.40× from scheduling alone.
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache
  Management for LLM Serving* (2026-06-30).
- `2607.08057` — *Towards Efficient LLM Serving: A Survey on System-Aware KV Cache
  Optimization* (2026-07-09).

**Mirrored note.** `research/memory/kv-cache-mechanics.md` v1.0.0 is the survey this module
teaches. No number here contradicts it. Two of its open questions are partially closed by
§3.7/Exercise C (question 2: the fp8 storage/compute path **loses** ~3× on this hardware) and
by Exercise B (question 6: the attention-read bandwidth is ~150 GB/s, not the 199.9 GB/s the
ridge point uses). Its remaining open questions are carried forward unchanged.
