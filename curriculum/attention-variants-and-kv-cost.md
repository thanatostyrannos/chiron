---
title: Attention variants and KV cost — MHA, MQA, GQA, MLA, and the window
version: 1.0.0
date: 2026-07-26
track: B — Modern architecture
prereqs: tensors-and-autograd, transformer-forward-pass
difficulty: moderate — the arithmetic is easy, the two places it lies to you are not
time: 3–4 h reading and working the math; 1.5–2 h for the three exercises
bridges_into: Track C (memory), specifically kv-cache-mechanics and kv-serving-hierarchy
---

# Attention variants and KV cost

**Difficulty and time, honestly.** The algebra here is arithmetic — multiplication and one
division. What takes the time is unlearning two reflexes: that a bigger model has a bigger
cache (it does not, necessarily), and that a configuration file tells you what a system will
do (it does not, ever). Budget 3–4 hours for sections 1–5 with a pen, and 1.5–2 hours for the
exercises. Exercise B is the one worth protecting time for; it produces a number this lab
currently does not have.

---

## 1. What this module settles

**One:** the four attention variants everyone names — MHA, MQA, GQA, MLA — are not four
architectures but edits to a single product: three of them are just different values of one
integer in it, and the fourth replaces one term, so once you can write that product you can
price any of them on paper in thirty seconds. **Two:** sliding-window
attention edits a *different* factor in the same product — the number of tokens retained, not
the width of each token's entry — which is why windowing and head-sharing compose cleanly and
why "3:1 hybrid" and "GQA-8" are answers to different questions. **Three:** the same head-count
ratio that sets the capacity win also sets the *arithmetic intensity* of decode, exactly and
without approximation, at `2G / dtype_bytes` — so on our machine you can compute, before
allocating anything, how far below peak compute a decode step will run and why more batch will
not rescue it.

This is the bridge module into Track C. Everything downstream — eviction policy, paging,
prefix reuse, offload tiering — is a policy layered on top of the product derived here. If the
product is wrong in your head, every policy conclusion inherits the error.

---

## 2. Theory in plain language

### 2.1 What attention actually does, in one paragraph, mechanically

For each token in the sequence the model computes three vectors from that token's residual
stream: a **query**, a **key**, and a **value**. To produce the output at position `t`, the
model takes token `t`'s query, dot-products it against the key of *every* token at position
≤ `t`, softmaxes those scores into weights, and returns the weighted sum of those tokens'
values. That is the whole operation. Its two costs follow immediately: you must *keep* every
past token's key and value around (that is the KV cache), and you must *read all of them* on
every single step (that is the bandwidth problem).

This is done `n_q` times in parallel per layer with different learned projections — those are
the **heads**. Each head has its own query, key and value subspace of width `d_h`. Heads are
what lets one layer attend to several things at once.

> **Systems bridge.** The KV cache is a per-sequence working set with an append-only write
> pattern and a full-scan read pattern.
>
> **Where it breaks — and this is the whole module in one sentence.** A working set implies a
> hot subset. There is no hot subset here. Every decode step reads 100% of the cache, so
> there is no temporal locality to exploit, no prefetch to issue (token `t+1` depends on token
> `t`, strictly serially), and no hit rate to improve. A cache whose every access is a full
> table scan is a *streaming buffer* wearing a cache's name. The consequence: you cannot make
> it faster by being smarter about *which* entries you touch. You can only make each entry
> smaller, or keep fewer of them. Those are the only two levers, and this module is about
> both.

### 2.2 The problem the variant ladder solves, and what it replaced

The 2017 design (`[C]` 1706.03762) gave every query head its own key head and value head. Call
that **MHA**, multi-head attention. Nobody thought about the cache because nobody was serving
100k-token contexts.

By 2019 the problem was legible. Shazeer's MQA paper (`[C]` 1911.02150, Nov 2019) states it in
the cleanest form anyone has managed: autoregressive decoding reads the entire KV cache from
memory to produce **one** token. The ratio of arithmetic to memory traffic is terrible, the
matrix units sit idle, and the bottleneck is the memory bus. His fix was brutal and simple —
keep `n_q` query heads but only **one** key head and **one** value head, shared by all queries.
That is **MQA**, multi-query attention. Cache shrinks by `n_q`. Quality drops measurably and
training gets less stable.

**GQA** (`[C]` 2305.13245, May 2023) is the compromise that won: partition the `n_q` query
heads into `n_kv` groups, one shared key/value head per group. `n_kv = n_q` recovers MHA;
`n_kv = 1` recovers MQA; anything between is GQA. The paper's second contribution matters as
much as the first: you can *uptrain* an existing MHA checkpoint into GQA at roughly 5% of
pretraining compute, by mean-pooling the key and value projections within each group. That
made the change adoptable without retraining the industry's models, which is why essentially
every open decoder since ships GQA.

**MLA** (`[C]` 2405.04434, May 2024, DeepSeek-V2) changes the shape of the entry rather than
the number of entries. Instead of caching `n_kv` keys and `n_kv` values per token, cache one
low-rank **latent** vector per token plus one small shared rotary key, and reconstruct the
per-head keys and values on the fly with a learned up-projection. DeepSeek's abstract claims
93.3% KV reduction against DeepSeek 67B, 42.5% lower training cost, and 5.76× maximum
generation throughput.

Read the ladder as an economic argument, not a modelling one. Each rung buys memory with
either quality (MQA), a little quality (GQA), or implementation complexity (MLA). None of them
makes the model better at language.

### 2.3 The other axis: sliding window versus global attention

Head-sharing shrinks *each token's* entry. **Sliding-window attention** shrinks *how many
tokens have an entry*. A windowed layer with window `w` can only attend to the last `w`
tokens, so anything older is unreachable and need not be kept.

The modern design does not pick one; it interleaves. Our reference model, Laguna S 2.1, has 48
layers in a strict `full, sliding, sliding, sliding` repeating pattern — 12 global and 36
windowed at `w = 512` `[M]` (read from `config.json` at revision `b0a9fd7c850e`; `ASSUMPTIONS.md
→ reference-model`). Gemma 3 uses 5 local : 1 global with `w = 1024` `[C]` (2503.19786);
gpt-oss alternates 1:1 with `w = 128` `[C]` (2508.10925). Same idea, a 32× spread in window
size, and — see section 8 — very little agreement on why.

> **Systems bridge.** Two layer types with two independently-sized caches looks exactly like a
> two-tier storage hierarchy, and the sizing arithmetic is genuinely capacity planning. In
> llama.cpp's Laguna implementation it is literally two allocations: a full-size cache holding
> only the global layers and a small one sized to `n_swa + n_ubatch`
> (`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73`).
>
> **Where it breaks — three ways, all instructive.**
> 1. **No promotion, no demotion, no miss path.** A layer is bound to one tier forever by its
>    index, decided at load time. Nothing migrates between tiers, ever.
> 2. **Eviction from the small tier is free and lossless.** An out-of-window token is
>    architecturally *unreadable* — the mask forbids it — so discarding it costs exactly
>    nothing. That is true of no storage tier you have ever operated, and it is the reason
>    windowing is a fundamentally different thing from an eviction policy. An H2O or SnapKV
>    eviction is a *bet* that a token will not be needed. A window is a *proof*.
> 3. **The tiers are not numerically interchangeable.** Laguna's global layers apply
>    YaRN-scaled RoPE over 64 of 128 head dimensions at θ=500000; the sliding layers apply
>    plain RoPE over all 128 at θ=10000 `[M]`
>    (`architecture/llama-cpp-laguna/src/models/laguna.cpp:184`, where `n_rot_l` forks on the
>    per-layer SWA bit). You therefore **cannot** "just widen the window" to test long
>    context — the sliding layers were never trained with positional encoding that reaches
>    past 512.

### 2.4 The claim you should be most suspicious of

The KV cache is a property of the **inference path**, not of the architecture. A config file
describes intent. What gets cached is whatever the code calls `cache.update()` with. Section 5
shows two production reference implementations where the code and the config disagree in
opposite directions, and one of them silently deletes the entire advertised benefit of MLA.
Hold that thought while reading the math; the math is correct and it is still not what your
benchmark will measure.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

| Symbol | Reads as | Where it comes from |
|---|---|---|
| `T` | number of tokens currently in context | runtime |
| `L` | number of attention layers in the stack | `num_hidden_layers` |
| `n_q` | **query** heads in one layer | `num_attention_heads` (per-layer list in Laguna) |
| `n_kv` | **key/value** heads in one layer | `num_key_value_heads` |
| `G` | GQA group size, `G = n_q / n_kv` — query heads sharing one KV head | **derived, never configured** |
| `d_h` | width of one head's key/query/value vector | `head_dim` |
| `d` | width of the residual stream | `hidden_size` |
| `b` | **bytes per stored element** — 2 for bf16, 1 for fp8 | `torch_dtype` |
| `B` | batch size, i.e. concurrent sequences | runtime |
| `w` | sliding window, in tokens | `sliding_window` |
| `d_c` | MLA latent rank — width of the compressed KV entry | `kv_lora_rank` |
| `d_r` | MLA's decoupled rotary key width | `qk_rope_head_dim` |
| `P` | total parameter count | — |
| `b_w` | bytes per weight element | — |

Note the notation collision I am keeping deliberately because the survey notes use it:
lowercase `b` is **bytes per element**, uppercase `B` is **batch**. They appear in the same
equations.

### 3.2 One layer of attention, written out

For a single query position with `T` tokens in context, in one layer:

```
scores  = q Kᵀ / √d_h          q is [n_q, d_h],  K is [T, d_h] per KV head
weights = softmax(scores)      over the T context positions
out     = weights V            V is [T, d_h] per KV head
```

Under GQA, query head `i` reads KV head `⌊i / G⌋`. So `G` query heads share one key matrix and
one value matrix. That sharing is the entire mechanism, and it appears in the reference code as
one integer division:

```
self.num_key_value_groups = self.num_heads // config.num_key_value_heads
```
— `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:350`.

### 3.3 The KV product

Per sequence, per token, the bytes the model must retain across the whole stack:

```
per_token_bytes  =  2  ×  L  ×  n_kv  ×  d_h  ×  b
```

Factor by factor, in words:

- **`2`** — one **K**ey vector and one **V**alue vector are stored per token per KV head. This
  comes from the definition of attention; it is not a tunable.
- **`L`** — every layer keeps its own cache. Caches are not shared between layers in a standard
  decoder.
- **`n_kv`** — key/value heads. **Not** query heads.
- **`d_h`** — elements per head.
- **`b`** — bytes per element.

**What is absent is the interesting part.** No `n_q`. No `d` (hidden size). No vocabulary. No
MLP width. No expert count. No parameter count. A 118B mixture-of-experts model and a 300M
dense model with the same `L`, `n_kv`, `d_h`, `b` have **identical** KV cost per token. This is
the single most useful fact in the formula and it is the one that surprises people who have
sized systems by "how big is the model."

**Worked on the reference model** `[M]` (`L=48`, `n_kv=8`, `d_h=128`, bf16 so `b=2`; read from
`research/reference/models/laguna-s/config.json`, revision `b0a9fd7c850e`):

```
per layer, per token = 2 × 8 × 128 × 2 B = 4096 B = 4 KiB
whole stack          = 48 × 4 KiB        = 192 KiB per token
```

At 128k context (131,072 tokens) that is exactly **24.0 GiB** if every layer were global. At
the advertised 1,048,576-token context, **192 GiB**. Against the `[M]` **≥62 GiB** fast memory
tier measured on our Z13 (`notebook/uma-carveout-controls-fast-tier.md`, single run per arm),
the first fits with room and the second is not close.

A caveat our own register raised and then withdrew, worth repeating because the reasoning is
the lesson: Laguna varies query heads per layer (48 on global layers, 72 on sliding ones) and
the top-level `num_attention_heads: 48` is therefore wrong for 36 of 48 layers `[M]`
(`ASSUMPTIONS.md → laguna-heads-uniform`, refuted). That does **not** make the 192 KiB figure
an estimate. `k_proj` and `v_proj` are built from the *global* `num_key_value_heads` field —
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:357` — while
`q_proj` is built from the per-layer count at `:355`. Query heads vary, KV heads do not, and
query heads do not appear in the product. **192 KiB/token is exact, not an upper bound.**

What the per-layer query count *does* control is `G`: 48/8 = **6** on global layers, 72/8 =
**9** on sliding layers. Section 3.6 shows that `G` is the arithmetic intensity of decode.
So per-layer head count controls **speed** and global `n_kv` controls **size**. Two different
questions, two different config fields, routinely conflated into one sentence.

### 3.4 The four variants are four values of one factor

| Variant | Edit | `n_kv` | Bytes vs MHA |
|---|---|---|---|
| MHA | baseline | `n_q` | 1× |
| GQA `[C]` 2305.13245 | share KV across groups | `n_q / G` | `1/G` |
| MQA `[C]` 1911.02150 | one KV head total | `1` | `1/n_q` |
| MLA `[C]` 2405.04434 | replace the whole `2 · n_kv · d_h` term | — | see below |

MLA does not set `n_kv`; it replaces the term. Per token per layer it stores

```
mla_elements_per_token_per_layer  =  d_c  +  d_r
```

— one shared latent of width `d_c` plus one shared rotary key of width `d_r`, for **all** heads
together. `[M]` Verified against a local artifact: `models/kimi-linear-model/config.json` has
`kv_lora_rank = 512` and `qk_rope_head_dim = 64`, and the projection that produces the cached
entry is built with output width exactly `kv_lora_rank + qk_rope_head_dim`
(`models/kimi-linear-model/modeling_kimi.py:362`). So **576 elements per token per layer** =
**1.125 KiB** in bf16.

**Worked comparison at Laguna-S's own shape** (`d_h = 128`, `n_q = 48` on a global layer):

```
MHA   : 2 × 48 × 128 × 2 B = 24 576 B = 24.0 KiB / token / layer
GQA-6 : 2 ×  8 × 128 × 2 B =  4 096 B =  4.0 KiB / token / layer     ← what Laguna ships
MQA   : 2 ×  1 × 128 × 2 B =    512 B =  0.5 KiB / token / layer
MLA   :    (512 + 64) × 2 B =  1 152 B =  1.125 KiB / token / layer  ← Kimi's ranks
```

Across the whole 48-layer stack, treating every layer as global: MHA 1.547 MiB/token (using
Laguna's actual per-layer query counts of 48 and 72, which is why it is not simply 48 × 24 KiB),
GQA-8 192 KiB/token — an **8.25× reduction** — MQA 24 KiB/token, MLA-576 54 KiB/token.

Two things to notice. MQA is *cheaper than MLA* at these ranks. MLA's advertised 93.3% is
against an MHA baseline at DeepSeek-67B's shape, not against GQA-8; against GQA-8 at Laguna's
shape the ratio is 4096/1152 = **3.56×**, which is a real win and not an order of magnitude.
Read vendor compression ratios by asking "against what baseline," every time.

### 3.5 Windowing splits the formula into a growing term and a fixed term

A windowed layer never retains more than `w` tokens. Split the sum by layer type:

```
total(T)  =  (L_global × 2 × n_kv × d_h × b) × T          ← grows linearly with context
          +  (L_window × 2 × n_kv × d_h × b × w)          ← constant, independent of context
```

`[M]` For Laguna-S: the growing term is `12 × 4 KiB = 48 KiB per token`; the fixed term is
`36 × 4 KiB × 512 = 72 MiB`, and it stops growing the instant context passes 512.

| Context | All-global | Actual hybrid | Ratio |
|---|---|---|---|
| 32k | 6.00 GiB | 1.57 GiB | 3.8× |
| 128k | 24.00 GiB | 6.07 GiB | 4.0× |
| 1M | 192.00 GiB | **48.07 GiB** | 4.0× |

The asymptotic saving is `L / L_global` = 48/12 = **4×** — the ratio *plus one*, not the ratio.
A "3:1 hybrid" saves 4×, and if you quote 3× you are wrong by a third. Below `T = w` the hybrid
saves **nothing at all**; the entire benefit is asymptotic. That is the capacity-planning shape:
a constant overhead you pay immediately and a slope reduction you collect later.

1M-token Laguna-S KV lands at 48 GiB, inside our `[M]` ≥62 GiB fast tier with 14 GiB spare. That
is the kind of thing you want to know before anyone reboots into the BIOS.

### 3.6 Decode arithmetic intensity — the derivation that governs everything

**Arithmetic intensity** is FLOPs performed per byte moved from memory. Compare it to the
machine's own ratio (peak FLOP/s ÷ peak byte/s — the roofline **ridge point**). Below the ridge
you are bandwidth-bound and the arithmetic units idle; above it you are compute-bound.

For one decode step, one layer, one sequence, `T` tokens in context:

```
bytes read  =  2 × n_kv × d_h × b × T
```

— the whole cache for that layer: K and V (`2`), for each KV head (`n_kv`), each of width `d_h`,
at `b` bytes per element, for all `T` positions.

```
FLOPs       =  2·n_q·T·d_h      (query · key dot products, one multiply + one add each)
            +  2·n_q·T·d_h      (weighted sum of values, same shape)
            =  4 × n_q × T × d_h
```

Divide. Watch `T`, `d_h` and `L` all cancel:

```
                4 · n_q · T · d_h          2 · n_q         2 · G
AI_attention = ───────────────────────  =  ─────────  =  ─────────
               2 · n_kv · d_h · b · T       n_kv · b          b
```

**For bf16 (`b = 2`), the arithmetic intensity of decode attention is exactly the GQA group
size `G`.** Independent of context length, head dimension, layer count, batch, and model size.
For fp8 (`b = 1`) it is `2G`.

The mechanism in words: `G` query heads share one KV read, so one trip to memory is amortised
over `G` dot products. GQA raises arithmetic intensity by amortising along the **head** axis in
exactly the way batching amortises the weight read along the **request** axis.

### 3.7 The ridge point on our machine, and how far below it we sit

`[M]` From measurements already in `ASSUMPTIONS.md`: 20.9 TFLOP/s bf16 GEMM at 8192³
(`scripts/benchmark_gemm.py`, row `gemm-throughput-below-reference`) ÷ 199.9 GB/s
device-to-device copy bandwidth (row `large-tensor-fault-32gib`, measured on a 31 GiB buffer)
= **≈105 FLOP per byte**.

Caveats stated plainly, because this number gets quoted: the bandwidth figure counts one read
plus one write and a pure-read stream may differ; both are single runs; and the GEMM figure is
itself only 63% of the ~33 TFLOPS cited for this silicon, unexplained. The ridge is ~105 give or
take, not a precision instrument. The gaps below are large enough that precision does not
matter.

| Configuration | `G` | `AI` (bf16) | Fraction of ~105 ridge |
|---|---|---|---|
| MHA | 1 | 1 | 0.96% |
| Laguna global layers | 6 | 6 | 5.7% |
| Laguna sliding layers | 9 | 9 | 8.6% |
| GQA-8 at `n_q`=32 (a plausible Proteus arm) | 4 | 4 | 3.8% |
| MQA at `n_q`=32 | 32 | 32 | 30% |

`[A]` These are derived, not measured — high confidence in the algebra, zero measurements on
this hardware. `ASSUMPTIONS.md → decode-intensity-varies-by-layer` records exactly this status.
Exercise B is the cheapest test that moves it.

**What that means in wall clock.** Decode must stream the whole cache once per token, so at the
`[M]` 199.9 GB/s figure:

```
Laguna-S hybrid @128k :  6.07 GiB = 6.518 GB  ÷ 199.9 GB/s  ≈  32.6 ms/token  ≈ 31 tok/s
Same model, all-global:  24.0 GiB = 25.77 GB  ÷ 199.9 GB/s  ≈  129 ms/token   ≈  8 tok/s
```

That is arithmetic over two measured inputs, not a benchmark. It ignores weight traffic, which
for a 118B-A8.5B MoE dominates until context passes roughly 100k. The systems point survives:
**the hybrid ratio is a bandwidth decision before it is a quality decision.**

### 3.8 MLA's arithmetic intensity, derived — and why it is the opposite trade

MLA is usually sold as a capacity win. In the *absorbed* serving form — where the up-projection
`kv_b_proj` is folded into the query side so the cache holds only the latent — it is also a
large arithmetic-intensity win, and the derivation is worth doing because the answer is
counterintuitive.

Per decode step, one layer, one sequence, absorbed MLA:

```
bytes read  =  (d_c + d_r) × b × T           one latent per token, shared by all heads

FLOPs       =  2·n_q·T·d_c                   query·latent scores (the non-rotary part)
            +  2·n_q·T·d_r                   query·rotary-key scores
            +  2·n_q·T·d_c                   weighted sum over latents
            =  2·n_q·T·(2·d_c + d_r)
```

```
                 2 · n_q · T · (2·d_c + d_r)        2 · n_q · (2·d_c + d_r)
AI_MLA      =  ──────────────────────────────  =  ─────────────────────────
                   (d_c + d_r) · b · T                   (d_c + d_r) · b
```

`[A]` Substituting Kimi's ranks (`d_c = 512`, `d_r = 64`, `b = 2`): `AI_MLA ≈ 1.89 × n_q`. At
`n_q = 32` that is **≈60 FLOP/byte**, against **4** for GQA-8 at the same query-head count — a
**15×** higher arithmetic intensity while the cache is only 3.6× smaller. Medium confidence:
this is derived from the absorbed-form operation counts, not measured, and it ignores the
up-projection FLOPs folded into the query path.

Read the direction of the trade: **MLA buys bytes with FLOPs.** Each query head now does a
512-wide dot product instead of a 128-wide one. On a bandwidth-starved machine that is exactly
the right trade, and it is why MLA looks better the worse your memory system is. It is also why
MLA has a tensor-parallel sharding problem — the latent is shared across heads, so it cannot be
split the way per-head KV can `[C]` (2603.02188, Mar 2026). Irrelevant to us until we leave
single-device, which `ASSUMPTIONS.md → single-device-only` says is not soon.

### 3.9 Why batching fixes the weight term and never the attention term

The weight-read half of decode:

```
AI_weights = (2 × P × B) / (P × b_w) = 2B / b_w   →   for bf16 weights, AI = B
```

— every weight is read once per step regardless of batch, and each of the `B` sequences does
`2P` FLOPs against it. **Batch size *is* the arithmetic intensity of the weight-bound half.**
To reach our ~105 ridge you would need batch ≈105.

The attention half does not move. Each sequence owns its **own** KV cache, so batching
multiplies bytes and FLOPs by the same `B`:

```
AI_attention(B) = (B × 4·n_q·T·d_h) / (B × 2·n_kv·d_h·b·T) = 2G/b     ← B cancels
```

**Weight-bound decode gets better with batch; attention-bound decode does not.** As context
grows the attention term takes over, which is precisely why long-context high-batch serving is
a bandwidth problem that more GPUs do not fix `[C]` (2607.13068, Jul 2026, which formalises
this ridge-point mismatch and argues for decode accelerators with less compute and more cheap
memory).

> **Where the analogy breaks, part two: read amplification is frozen at pretraining time.**
> `G` is baked into the checkpoint's tensor shapes. There is no runtime knob, no re-sharding,
> no index rebuild, no "increase the group size in prod and see." The only post-hoc move is
> uptraining (`[C]` 2305.13245, ~5% of pretrain compute) or conversion (`[C]` 2502.14837 for
> GQA→MLA, recoverable on 0.3–0.6% of the data). This is a storage layout decision that you
> make once, before you have any production traffic to inform it. Nothing in your career has
> that property.

---

## 4. Why it matters for Proteus

**The config surface is the experimental surface.** Every factor in the product above must be a
first-class config field, because every one of them is an ablation axis:

| Proteus config field | Factor it edits | Ablation it enables |
|---|---|---|
| `num_key_value_heads` | `n_kv` | the MHA↔GQA↔MQA ladder, as one integer |
| `num_attention_heads` (or a per-layer list) | `n_q`, hence `G` | decode intensity at fixed cache size |
| `head_dim` | `d_h` | explicit, never derived from `hidden_size / n_q` — Laguna sets 128 where the division gives 64 `[M]` |
| `layer_types` (explicit list) | which layers carry the `T` term | hybrid ratio and **placement** |
| `sliding_window` | `w` | the fixed term; interacts with ratio `[C]` 2606.15378 |
| `kv_dtype` | `b` | fp8 storage halves bytes and doubles `AI` |
| `attention_variant` | the term structure | MLA as a distinct branch, not a head count |

Four consequences for how we build and how we name.

**`G` is derived, so arm names must state both numbers.** `mnemosyne-*` and `proteus-*` arm
names carry information by house rule. `proteus-gqa8` is ambiguous — at `n_q=32` it means
`G=4`, at `n_q=64` it means `G=8`, and those have the same capacity and double the decode
intensity. Name arms with the pair, e.g. `proteus-q32-kv8` — the cache size and the intensity
are then both readable off the name.

**A single-number bandwidth model is wrong for this model class.** Laguna has `G=6` on 12
layers and `G=9` on 36. A Mnemosyne cost model keyed on the top-level `num_attention_heads`
mis-predicts 75% of the layers. The cost model must be per-layer-type from the start; retrofit
is worse than build.

**Windowed layers are not "the cheap layers" without qualification.** They are cheap in *KV
residency* and expensive in *parameters and prefill FLOPs* — Laguna's 36 sliding layers hold
2.27 B attention parameters against the 12 global layers' 0.53 B, because they carry 72 query
heads each `[M]` (`research/notes/transformer-state-of-the-art.md`, computed from `config.json`).
Any matched-budget arm that swaps a global layer for a sliding one and calls the budgets matched
is wrong.

**We are structurally inside the interesting regime, not simulating it.** `[A]` At a plausible
Proteus shape (24 layers, `n_kv=8`, `d_h=64`, bf16, all-global, 300M params — medium confidence,
the cheapest thing that would move it is freezing an actual arm config): weights are 572 MiB and
KV is 48 KiB/token, so the cache outweighs the model at **batch 2 on an 8k context**. At 32k ×
batch 8 the cache is 12 GiB against 0.56 GiB of weights — a 21× ratio, which fits our fast tier
with room to sweep. Small models enter the KV-dominated regime at trivially short contexts,
which is the honest justification for doing memory research at 300M rather than an excuse for it.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 GQA is one integer division and two projection widths

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:350` | `self.num_key_value_groups = self.num_heads // config.num_key_value_heads`. This is `G`, in the code, as an integer division. It is derived at construction, appears in no config file, and is the number that governs decode intensity. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:355` | `q_proj` output width is `self.num_heads * self.head_dim` — the **per-layer** query count (48 or 72). |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:357` | `k_proj` output width is `config.num_key_value_heads * self.head_dim` — the **global** field. Put `:355` and `:357` side by side: the entire MHA→GQA change is that these two lines read different config fields. That asymmetry *is* the KV saving. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `past_key_values.update(key_states, value_states, self.layer_idx)` — the one line where bytes enter the cache. Note what has already happened to `key_states` above it: QK-norm at `:390`, then RoPE at `:394`. **A cached key is `RoPE(RMSNorm(k))`,** which is why any later re-positioning or quantization scheme is operating on a doubly-transformed quantity. |

### 5.2 The place where GQA's bandwidth saving quietly does not happen

This is the most valuable read in the module. The formula says one KV read serves `G` query
heads. Whether that is *true in the kernel* is a runtime decision made three files away.

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:303` | `def repeat_kv` — the function that undoes GQA. It takes `[batch, n_kv, T, d_h]` and returns `[batch, n_q, T, d_h]`. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:311` | `.expand(...)` — a **view**, stride 0 on the new axis, zero bytes copied. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:312` | `.reshape(...)` on that non-contiguous view — this **cannot** be a view, so it materialises. The tensor handed to the matmul is physically `G×` larger than the cache. Exercise C measures this. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:325` | The eager attention path calls `repeat_kv` unconditionally. Eager attention *always* pays the `G×` amplification. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:97` | The SDPA path branches on `num_key_value_groups > 1` … |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:99` | … and on the `False` branch calls `repeat_kv` — the same materialisation. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:102` | … and on the `True` branch passes `enable_gqa=True`, letting the kernel do the sharing with no copy. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:38` | **The condition that decides which branch.** `enable_gqa` is used only when `attention_mask is None` (plus torch ≥ 2.5 and `head_dim ≤ 256`). A comment at `:32` explains why: with a mask, SDPA falls back to the math kernel. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:567` | And here is the sting: sliding-window layers get an explicit mask from `create_sliding_window_causal_mask`. **A windowed layer always has a mask, so it always takes the `repeat_kv` branch.** The 36 layers with the *highest* `G` (9) are the ones structurally guaranteed not to get the kernel-level sharing. |

Do not over-read this — the materialised copy is `G × n_kv × w × d_h` on a windowed layer, so
the absolute cost is bounded by the window, and a fused kernel (FlashAttention/AOTriton) does
not go through this path at all. But the *pattern* is the lesson: **the arithmetic-intensity
number derived in section 3.6 is an upper bound that the kernel may or may not deliver, and
nothing in the config tells you which.**

### 5.3 MLA, and the reference implementation that deletes its own benefit

| Where | What to look at, and why |
|---|---|
| `models/kimi-linear-model/modeling_kimi.py:362` | `kv_a_proj_with_mqa` output width is exactly `kv_lora_rank + qk_rope_head_dim` = 512 + 64 = 576. This is the compressed entry, and 576 × 2 B = 1.125 KiB is the number MLA's marketing is about. |
| `models/kimi-linear-model/modeling_kimi.py:397` | `compressed_kv = self.kv_a_proj_with_mqa(hidden_states)` — the latent, 576 wide, exists here. |
| `models/kimi-linear-model/modeling_kimi.py:401` | `self.kv_b_proj(...)` expands the latent back into full per-head K and V **before** anything is cached. |
| `models/kimi-linear-model/modeling_kimi.py:413` | `past_key_values.update(key_states, value_states, ...)` — and what gets cached is the *expanded* form: 32 heads × (128+64) for K plus 32 × 128 for V = 10,240 elements = **20 KiB per layer per token**, which is **17.8× larger** than the compressed form and **zero saving versus GQA**. |

`[M]` This is verified from the local artifact, not inferred. The advertised MLA win exists only
in a serving engine that caches `compressed_kv` and folds `kv_b_proj` into the query side (the
"absorb" trick). **If you benchmark MLA using an HF reference implementation you will measure
the opposite of the claim.** Generalise it: the KV cache is a property of the inference path,
not of the architecture.

### 5.4 Windowing, in two implementations

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | `self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"`. The entire hybrid mechanism is a list lookup at construction time. Every hybrid-ratio question this lab asks is a question about what goes in that list — which is why it is trivially ablatable. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:366` | `self.sliding_window = config.sliding_window if self.is_local_attention else None` — 512 or `None`. That ternary is the difference between `O(T)` and `O(w)` residency. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:41` | The same decision in C++, as `set_swa_pattern(swa_period, dense_first=true)` — full attention at `il % 4 == 0`. Note the fallback: if the `sliding_window` key is absent the whole hybrid path is skipped and the model is all-full-attention. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73` | `size_swa` — the capacity math, sized to `min(size_base, n_swa·n_seq + n_ubatch)` padded to 256. This is section 3.5's fixed term, allocated. |
| `architecture/llama-cpp-laguna/src/llama-graph.cpp:2891` | `const bool is_swa = hparams.is_swa(il);` — per-layer dispatch selecting which of the two cache contexts receives this layer's writes. |

### 5.5 Two pointers for where this goes next (Track C)

| Where | What to look at, and why |
|---|---|
| `memory/flashinfer/flashinfer/page.py:403` | `append_paged_kv_cache` — the docstring gives the two physical layouts, `[pages, page_size, n_kv_heads, d_h]` (NHD) versus `[pages, n_kv_heads, page_size, d_h]` (HND). Same bytes, different stride order; it decides whether one head's read is a contiguous burst or a strided gather. `n_kv` is a *layout axis*, not just a count. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `get_new_blocks` — the allocator. When it cannot find a block it does not fault; the request is preempted. There is no miss path anywhere in a KV cache, which is the fact Track C is built on. |

---

## 6. Exercises

All three run on the Z13. Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

Standing hardware caveats, from `ASSUMPTIONS.md`: single tensors **≥32 GiB hang** the GPU
silently at 0% CPU (`large-tensor-fault-32gib`, refuted); keep every buffer under 31 GiB. bf16
numerics on gfx1151 are **untested** (`bf16-numerics-unproven`), so any accuracy claim from
these exercises is provisional — timing claims are not affected. The Hardware Validation Gate
has not run, so nothing measured here is evidence by house standard until it does; these are
instrument-shakedown runs and should be labelled as such.

Write scratch scripts under `notebook/`. Exercise B is a Hardware Validation Gate item and
should migrate into the rig with tests when it is reused (house rule: one-off analysis scripts
are exempt from TDD only until reuse).

---

### Exercise A — the KV budget calculator, checked against three known answers

**Goal:** a function you trust, validated against numbers already measured elsewhere in the
repo, then used to answer a question you cannot answer by intuition.

**Hardware:** none. Pure Python, no torch, no GPU. **Runtime:** 20–30 minutes to write,
< 1 second to run.

```python
"""KV residency as a closed form over five config fields."""
GIB = 1024 ** 3
KIB = 1024

def kv_bytes_per_token_per_layer(n_kv: int, head_dim: int, dtype_bytes: int) -> int:
    """One K vector and one V vector, per KV head, per token, in one layer."""
    return 2 * n_kv * head_dim * dtype_bytes

def kv_bytes(context_len, layer_types, n_kv, head_dim, dtype_bytes, window):
    """Total KV residency for one sequence. layer_types is a list of
    'full_attention' / 'sliding_attention', one entry per layer."""
    per_layer = kv_bytes_per_token_per_layer(n_kv, head_dim, dtype_bytes)
    return sum(
        per_layer * (context_len if t == "full_attention" else min(context_len, window))
        for t in layer_types
    )

LAGUNA_S = dict(
    layer_types=["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(48)],
    n_kv=8, head_dim=128, dtype_bytes=2, window=512,
)
```

**Three assertions that must pass** — each reproduces a `[M]` number already in the repo:

1. `kv_bytes_per_token_per_layer(8, 128, 2) * 48 == 192 * KIB` — the exact 192 KiB/token.
2. `kv_bytes(131072, ["full_attention"]*48, 8, 128, 2, 512) == 24 * GIB` — the all-global
   counterfactual at 128k.
3. `kv_bytes(131072, **LAGUNA_S) / GIB` rounds to **6.07** — the actual hybrid, matching
   `research/memory/kv-cache-mechanics.md`.

Then a fourth against a different model: gpt-oss-20b is `L=24` (12 full + 12 sliding at
`w=128`), `n_kv=8`, `d_h=64` `[M]`. Your function should give **2 KiB per layer per token**, so
**48 KiB/token** if every layer were global, a **24 KiB/token** growing term for the 12 real
global layers, a **3 MiB** fixed term, and **3.00 GiB** total at its full 131,072-token context.

**Deliverable — the number to produce.** For each of `{MHA at n_q=48, GQA-8, MQA, MLA-576}`
crossed with `{all-global, 3:1 hybrid w=512}`, print the context length at which one sequence's
KV cache reaches our `[M]` **62 GiB** fast tier. Eight numbers. Then answer, in one line in
your notebook entry: *which single edit buys the most context per byte at Laguna's shape, and
does the hybrid change the ranking?*

**Check yourself.** The all-global GQA-8 answer must be 62 GiB ÷ 192 KiB/token ≈ **338,000
tokens**. If it is not, your `2` is missing or doubled somewhere.

---

### Exercise B — measure decode arithmetic intensity against `AI = 2G/b`

**Goal:** test the module's central prediction on our actual instrument, and move
`ASSUMPTIONS.md → decode-intensity-varies-by-layer` off "derived and never run."

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback given below.**
**Runtime:** 5–10 minutes on GPU; 10–20 minutes on CPU at reduced shapes.

Hold `n_q` fixed and sweep `n_kv`. Cache bytes fall as `1/G`; FLOPs stay constant; so if the
model is right, **wall time should fall roughly linearly in `n_kv`** while achieved bandwidth
stays flat near the `[M]` 199.9 GB/s figure — right up until `G` approaches the ~105 ridge,
which at these shapes it never does.

```python
"""Decode-attention roofline sweep: does AI = 2G/b hold on gfx1151?"""
import torch, time, json

N_Q, HEAD_DIM = 32, 128
CONTEXT, BATCH = 65536, 4
DTYPE = torch.bfloat16
BYTES_PER_ELEM = 2

def one_decode_step(K, V, q, scale):
    """K,V: [B*n_kv, T, d]  q: [B*n_kv, G, d]  -> one token's attention output."""
    scores = torch.bmm(q, K.transpose(1, 2)) * scale          # [B*n_kv, G, T]
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(weights, V)                              # [B*n_kv, G, d]

def measure(n_kv, device, iters=20):
    G = N_Q // n_kv
    shape = (BATCH * n_kv, CONTEXT, HEAD_DIM)
    K = torch.randn(shape, dtype=DTYPE, device=device)
    V = torch.randn(shape, dtype=DTYPE, device=device)
    q = torch.randn(BATCH * n_kv, G, HEAD_DIM, dtype=DTYPE, device=device)
    scale = HEAD_DIM ** -0.5
    for _ in range(3):
        one_decode_step(K, V, q, scale)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one_decode_step(K, V, q, scale)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / iters

    kv_bytes = 2 * BATCH * n_kv * CONTEXT * HEAD_DIM * BYTES_PER_ELEM
    flops = 4 * BATCH * N_Q * CONTEXT * HEAD_DIM
    del K, V, q
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(n_kv=n_kv, G=G, ms=seconds * 1e3,
                gb_s=kv_bytes / seconds / 1e9,
                tflop_s=flops / seconds / 1e12,
                predicted_ai=2 * G / BYTES_PER_ELEM,
                measured_ai=flops / kv_bytes)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'n_kv':>5} {'G':>3} {'ms':>8} {'GB/s':>8} {'TFLOP/s':>9} {'AI_pred':>8} {'AI_meas':>8}")
for n_kv in [1, 2, 4, 8, 16, 32]:
    r = measure(n_kv, device)
    print(f"{r['n_kv']:>5} {r['G']:>3} {r['ms']:>8.2f} {r['gb_s']:>8.1f} "
          f"{r['tflop_s']:>9.3f} {r['predicted_ai']:>8.1f} {r['measured_ai']:>8.1f}")
```

**CPU fallback.** Set `CONTEXT = 4096`, `BATCH = 1`, `DTYPE = torch.float32`,
`BYTES_PER_ELEM = 4` and `iters = 5`. The predicted intensity becomes `2G/4 = G/2`, so the
column you compare against changes — which is itself the point of the exercise. Expect
one-to-two orders of magnitude less bandwidth; the *shape* of the curve is what transfers.

**Footprint check before you run.** At `n_kv=32`, `BATCH=4`, `CONTEXT=65536`: each of K and V
is `4 × 32 × 65536 × 128 × 2 B` = 2.15 GB, total footprint 4.3 GB. Comfortably inside the
`[M]` ≥62 GiB fast tier and far below the 31 GiB per-tensor hazard. Do not raise `BATCH` past
16 without redoing that arithmetic.

**Deliverable — three numbers and one plot.**
1. The `AI_pred` / `AI_meas` columns must agree exactly for every row. They are both
   closed forms; a mismatch means your byte or FLOP accounting is wrong, not the hardware.
2. Plot `ms` against `n_kv`. **Prediction: a straight line through the origin.** Report the
   R² of a linear fit and the intercept. A non-zero intercept is fixed per-call overhead — name
   it in your notebook entry.
3. The `GB/s` column against the `[M]` 199.9 GB/s reference. **Prediction: roughly flat.**
   Where it is not flat, say which end and offer a mechanism (small-`n_kv` rows move only
   134 MB and are latency-bound, not bandwidth-bound; that is the expected deviation).

**What a falsification would mean.** If time does *not* scale with `n_kv`, the attention path is
not bandwidth-bound at these shapes and the whole `AI = 2G/b` framing needs a caveat on this
hardware — which is a finding worth a notebook entry either way. Pre-register the SUCCESS and
KILL thresholds before you run: this is a G2 hypothesis card, not a script.

---

### Exercise C — prove that `repeat_kv` materialises, and price it

**Goal:** turn section 5.2 from a code reading into a measurement. Show that the theoretical
`G×` bandwidth saving is conditional on a kernel path, and measure what it costs when the
condition fails.

**Hardware:** one gfx1151 GPU. **CPU fallback:** works identically with
`torch.cuda.max_memory_allocated()` replaced by tensor-storage accounting, which is the more
honest measurement anyway. **Runtime:** 15 minutes to write, under a minute to run.

```python
"""Does GQA's saving survive the kernel? Two questions: does repeat_kv copy,
and what does the copy cost."""
import torch

n_kv, G, T, d = 8, 4, 8192, 128
device = "cuda" if torch.cuda.is_available() else "cpu"
k = torch.randn(1, n_kv, T, d, dtype=torch.bfloat16, device=device)

expanded = k[:, :, None, :, :].expand(1, n_kv, G, T, d)     # modeling_laguna.py:311
reshaped = expanded.reshape(1, n_kv * G, T, d)              # modeling_laguna.py:312

print("cache storage bytes      ", k.untyped_storage().nbytes())
print("after .expand()          ", expanded.untyped_storage().nbytes(), "  <- a view")
print("after .reshape()         ", reshaped.untyped_storage().nbytes(), "  <- ?")
print("shares memory with cache ", reshaped.data_ptr() == k.data_ptr())
print("amplification factor     ",
      reshaped.untyped_storage().nbytes() / k.untyped_storage().nbytes())
```

Then price it. Time `scaled_dot_product_attention` two ways at decode shape
(`q` of length 1, `T` keys): once with `enable_gqa=True` and `attn_mask=None`, once with an
explicit sliding-window additive mask and `repeat_kv`-materialised K and V — which is exactly
what `sdpa_attention.py:38` forces a windowed layer to do. Record `torch.cuda.max_memory_allocated()`
around each.

**Deliverable — two numbers.**
1. The amplification factor. **Prediction: exactly `G`.** If it prints `1.0`, your torch build
   made `reshape` a view, which would be news — check `reshaped.is_contiguous()` and
   `expanded.stride()` before believing it.
2. The peak-memory ratio and the time ratio between the two SDPA paths. **Prediction:** peak
   memory ratio ≈ `G` for the KV tensors; the time ratio is the open question and is the number
   worth writing down, because nothing in the literature reports realized-versus-theoretical
   arithmetic intensity for the masked GQA path.

**Expected friction, stated so it does not surprise you.** `enable_gqa` requires torch ≥ 2.5
(`sdpa_attention.py:38`); our build is `2.12.0a0+rocm7.13.0a20260313` `[M]`, so it is available,
but the ROCm SDPA backend may not implement it and may fall back to the math kernel. **A
fallback is a result.** Record which backend ran (`torch.backends.cuda.flash_sdp_enabled()` and
friends) alongside the timing; a timing number without the backend recorded is uninterpretable.
Flash Attention 2 is unavailable on gfx1151, so do not expect the fused path.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. Model X is a 118B mixture-of-experts with 256 experts and a 100k vocabulary. Model Y is a
   300M dense model. Both have 48 layers, 8 KV heads, head dimension 128, bf16. Which has the
   larger KV cache per token, and by how much?

2. You double `n_q` and hold `n_kv`, `d_h`, `L`, `b` fixed. State what happens to (a) KV bytes
   per token, (b) decode-attention arithmetic intensity, (c) attention parameter count, and
   (d) prefill FLOPs.

3. Laguna-S is described as a "3:1 hybrid." At 1M tokens its KV cache is 4× smaller than an
   all-global equivalent, not 3×. Where does the extra factor come from, and at what context
   length does the hybrid start saving anything at all?

4. Batching from 1 to 64 sequences raises the arithmetic intensity of the weight-read half of
   decode by 64×. It does nothing at all to the attention half. Why — in one sentence about
   ownership?

5. You implement MLA in Proteus by porting the HF Kimi reference, run a KV-residency benchmark,
   and measure a cache *larger* than your GQA-8 baseline. The house rule says "if a result looks
   too good, suspect the harness." Does the converse apply here — is this a harness bug?

6. Our ridge point is `[M]` ≈105 FLOP/byte. At bf16, what group size `G` would make decode
   attention compute-bound, what `n_kv` does that imply at `n_q = 32`, and why is that not a
   design anyone ships?

---

## 8. What is still unsolved here

The honest frontier, drawn from `research/memory/` and `research/notes/`. Everything here is
testable at 20M–300M params on one GPU with a `[M]` ≥62 GiB fast tier, and every item needs a
pre-registered hypothesis card before it runs.

1. **`AI = 2G/b` has never been measured on this hardware.** `ASSUMPTIONS.md →
   decode-intensity-varies-by-layer` is marked *refuted analytically; unmeasured*. It is a
   derivation, not a measurement, and the derivation assumes the kernel honours the grouping.
   Exercise B is the cheapest test in this module and it validates the instrument at the same
   time.

2. **Nobody reports realized versus theoretical arithmetic intensity.** Section 5.2 shows a
   production reference implementation where masked layers silently forfeit the kernel-level
   saving. The literature reports the closed form and the end-to-end throughput and nothing in
   between. That gap is exactly the attribution failure this lab exists to attack, and it is
   measurable with Exercise C plus a per-layer counter.

3. **Where is the `G` cliff at our scale?** GQA's quality claims come from uptraining 7B–70B
   checkpoints `[C]` (2305.13245). At 20M–300M the per-head capacity is far smaller, so the
   group size at which quality breaks may be much lower. If so, small-scale ablations
   systematically *understate* how good GQA is — a confound affecting every arm we run, not
   just GQA arms.

4. **MLA versus GQA below 300M is untested in public.** MLA's wins are reported at 100B+ MoE
   scale by the labs that ship MLA. The fair framing is matched-KV-*budget*, not matched-rank,
   and the conversion recipe `[C]` (2502.14837, Feb 2025) makes the arm cheap. As far as the
   anchoring survey pass could establish, this is a genuinely available experiment rather than a
   replication.

5. **Two orthogonal axes exist as of 2026 and most experiments vary only one.** MLA compresses
   along the *feature* axis; DeepSeek-V4's Compressed Sparse Attention and Heavily Compressed
   Attention compress along the *sequence* axis, folding groups of tokens into one entry,
   reported at ~7% of V3.2's KV size at 1M context `[C]` (2606.19348, Apr 2026 — vendor-reported,
   not independently replicated). Both are single factors in the same product, so the null
   hypothesis is clean multiplication and any interaction is a finding.

6. **The query-head axis is a separate budget nobody ablates jointly with GQA.** Sparse Query
   Attention reduces *query* heads to cut prefill FLOPs rather than KV bytes — the mirror image
   of GQA — reporting up to 3× throughput in compute-bound phases with minimal quality impact at
   small scale `[C]` (2510.01817, Oct 2025). Since `G = n_q/n_kv`, moving `n_q` changes decode
   intensity *without* changing cache size. A two-dimensional `n_q × n_kv` sweep at matched
   parameters is, as far as I can find, unpublished.

7. **The variant space did not stop at MLA.** Grouped-head Latent Attention `[C]` (2506.17286,
   Jun 2025) and Group-Query Latent Attention `[C]` (2605.15250, May 2026) both try to keep MLA's
   latent compression while recovering a GQA-style execution path — GQLA explicitly frames this
   as exposing two algebraically equivalent decode paths over the same weights so one checkpoint
   can be optimal on different hardware. If that line holds, "which variant" stops being a
   pretraining commitment, which changes the cost structure of this whole question.

8. **Contested: is the hybrid ratio a capability ceiling or a training-speed knob?**
   `[C]` 2507.06457 (72 trained models) finds recall degrades sharply once full-attention layers
   thin below 3:1 — a ceiling. `[C]` 2606.15378 (Jun 2026) finds different hybrids converge to
   comparable long-context performance given enough training, with the efficient-attention design
   controlling only how *fast* the capability emerges. Same year, same question, incompatible
   framings; the resolution plausibly depends on token budget, which is precisely the axis a
   small rig can attack.

9. **Contested, and counterintuitive: bigger windows can hurt.** Two independent 2026 results
   report that larger sliding windows *degrade* long-context ability, because a short window
   forces the model to train its long-range machinery rather than lean on local attention
   ("Large-Window Laziness", `[C]` 2606.15378; stochastic windows beating fixed ones, `[C]`
   2509.24552). For a caching engineer this is the sharpest place the tiering metaphor dies:
   **increasing the fast tier's capacity degrades the slow tier's behaviour**, because the two
   tiers are co-trained and the cheap one crowds out the expensive one's learning signal. There
   is no storage-hierarchy analogue.

10. **Contested: is hybridization worth it at all?** MiniMax shipped M2 and M2.5 as full-attention
    plain-MHA/GQA models on reliability grounds `[C]` (2605.26494, May 2026); Kimi Linear claims
    the opposite under matched-scale pretraining `[C]` (2510.26692, Oct 2025). Both are shipping
    product retrospectives with commercial incentives, neither is a controlled academic ablation.
    Present as contested.

11. **What is the ridge point for the attention *kernel*, not for GEMM?** Our ~105 FLOP/byte uses
    an 8192³ GEMM and a copy benchmark. A decode-shaped attention roofline would give the number
    that actually governs, and would resolve whether the unexplained 63%-of-cited GEMM shortfall
    `[M]` (`ASSUMPTIONS.md → gemm-throughput-below-reference`) also afflicts attention.

---

## Answers to the self-check

**1.** They are identical — 192 KiB per token each. The KV product `2 · L · n_kv · d_h · b`
contains no parameter count, no hidden size, no vocabulary and no expert count. The MoE's 118B
of weights are shared across every concurrent request; its KV cache is private per sequence and
is the *same size* as the 300M model's. This is why the crossover into the KV-dominated regime
happens at *shorter* contexts for the small model, not longer.

**2.** (a) **Unchanged** — `n_q` does not appear in the product. (b) **Doubles** —
`AI = 2·n_q/(n_kv·b)`, so `G` doubles and intensity with it. (c) **Increases** — `q_proj` and
`o_proj` both scale with `n_q · d_h` (see `modeling_laguna.py:355`), so the layer gets
materially bigger. (d) **Roughly doubles** for the attention score/output matmuls, since prefill
FLOPs scale with `n_q`. The pair (a)+(b) is the useful one: you can buy decode speed with
parameters and prefill compute, at zero cache cost. That is the trade Laguna makes by putting
72 query heads on its sliding layers.

**3.** The asymptotic saving is `L / L_global`, and 48/12 = 4. The "3:1" names the ratio of
sliding layers to global layers, so the denominator of the saving is *the ratio plus one*. Below
`T = w = 512` the hybrid saves **nothing** — a windowed layer holds `min(T, w)` tokens, which is
`T` while `T ≤ w`. The whole benefit is asymptotic, and there is a fixed 72 MiB you pay from
token 512 onward regardless.

**4.** Weights are shared across all sequences in the batch and read once per step; KV caches
are private per sequence, so batching multiplies the attention term's bytes and FLOPs by the
same `B` and the ratio is invariant. Ownership is the whole answer: **shared bytes amortise,
private bytes do not.**

**5.** No — the harness is telling you the truth. Look at `models/kimi-linear-model/modeling_kimi.py:401`
and `:413`: the reference implementation runs `kv_b_proj` to expand the latent into full per-head
K and V *before* it calls `past_key_values.update`, so it caches 20 KiB per layer per token
against the compressed form's 1.125 KiB — 17.8× worse, and worse than GQA-8's 4 KiB. The
advertised saving requires the absorb trick, which lives in the serving engine, not the model
class. The general form of the lesson: **measure the cache, do not read it off the config.** The
"suspect the harness" rule cuts both ways, and the way you tell them apart is by reading the code
path, not by re-running the benchmark.

**6.** At bf16 `AI = G`, so you would need `G ≳ 105`, which at `n_q = 32` implies `n_kv ≲ 0.3` —
i.e. fewer than one KV head, which is not a thing. Even MQA at `n_q = 32` reaches only `G = 32`,
about 30% of the ridge. The honest reading: **decode attention cannot be made compute-bound by
head sharing on this machine at any realistic query-head count.** To get there you would need
either `n_q` in the hundreds with `n_kv = 1` (which is MQA's quality problem multiplied), or a
different term structure entirely — which is precisely what MLA does, reaching `AI ≈ 1.89·n_q`
≈ 60 at `n_q = 32` by making every head read the *same* wide latent (section 3.8). That is the
strongest argument for MLA that nobody makes in the marketing, because it is an argument about
bandwidth-starved hardware rather than about quality.

---

## Sources

**Local artifacts and measurements (`[M]`)**

- `research/reference/models/laguna-s/config.json` at revision `b0a9fd7c850e`
  (`research/reference/PROVENANCE.md`) — `num_hidden_layers` 48, `num_key_value_heads` 8,
  `head_dim` 128, `sliding_window` 512, `max_position_embeddings` 1048576, `torch_dtype`
  bfloat16, `layer_types` = 12 `full_attention` + 36 `sliding_attention` in a strict
  `full,sliding,sliding,sliding` pattern, `num_attention_heads_per_layer` ∈ {48, 72}.
- `research/reference/models/kimi-linear-model/config.json` — `kv_lora_rank` 512,
  `qk_rope_head_dim` 64.
- `ASSUMPTIONS.md` rows: `reference-model`, `kv-per-token-laguna`, `laguna-heads-uniform`,
  `gpu-fast-tier-size`, `large-tensor-fault-32gib`, `gemm-throughput-below-reference`,
  `hipblaslt-config`, `decode-intensity-varies-by-layer`, `bf16-numerics-unproven`,
  `single-device-only`, `torch-build`.
- `notebook/uma-carveout-controls-fast-tier.md` — ~200 GB/s flat to ≥62 GiB, single run per arm,
  2026-07-26.
- `research/memory/kv-cache-mechanics.md` — the source derivation this module teaches; all shape
  math here is consistent with it and no number contradicts it.
- `research/memory/hybrid-architectures.md`, `research/notes/transformer-state-of-the-art.md` —
  the hybrid ledger and the per-layer parameter accounting.
- Code pointers: every `file:line` reference in section 5 was opened and the named symbol
  confirmed on the named line on 2026-07-26, against the revisions in `PROVENANCE.md`.

**arXiv (`[C]`)**

- `1706.03762` — *Attention Is All You Need* (2017). The MHA baseline.
- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019-11-06). MQA;
  the origin of the decode-is-bandwidth-bound argument.
- `2305.13245` — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints* (2023-05-22). The `n_kv` knob and the ~5%-of-pretraining uptraining recipe.
- `2405.04434` — *DeepSeek-V2* (2024-05-07). MLA; the 93.3% / 42.5% / 5.76× claims.
- `2502.14837` — *Towards Economical Inference: Enabling MLA in Any Transformer-based LLMs*
  (2025-02-20). GQA→MLA conversion at 0.3–0.6% of the data.
- `2503.19786` — *Gemma 3 Technical Report* (2025-03-25). 5:1 local:global, `w`=1024.
- `2506.17286` — *GTA: Grouped-head latenT Attention* (2025-06-15, verified by fetching the
  arXiv abstract page 2026-07-26).
- `2507.06457` — *A Systematic Analysis of Hybrid Linear Attention* (2025-07-08, rev. 2026-06-24).
  Recall collapse below 3:1.
- `2508.10925` — *gpt-oss-120b & gpt-oss-20b Model Card* (2025-08-08). 1:1 alternating, `w`=128.
- `2509.24552` — *Short window attention enables long-term memorization* (2025-09-29, rev.
  2026-05-04). Stochastic window size.
- `2510.01817` — *Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism
  with Query Heads Reduction* (2025-10). The mirror image of GQA.
- `2510.26692` — *Kimi Linear: An Expressive, Efficient Attention Architecture* (2025-10-30).
- `2603.02188` — *Multi-Head Low-Rank Attention* (2026-03-02). MLA's tensor-parallel sharding
  bottleneck.
- `2605.15250` — *GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model
  Decoding* (2026-05-14, verified by fetching the arXiv abstract page 2026-07-26).
- `2605.26494` — *The MiniMax-M2 Series* (2026-05-26). The full-attention counter-position.
- `2606.15378` — *Rethinking the Role of Efficient Attention in Hybrid Architectures* (2026-06-13).
  Large-Window Laziness.
- `2606.19348` — *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*
  (2026-04-26). Sequence-axis compression. Vendor-reported.
- `2607.13068` — *The Economics of AI Decoding Chips: Rebalancing Compute, Capacity, and
  Bandwidth for Efficient LLM Inference* (2026-07-10). Formalises the ridge-point mismatch.

Every id above other than the two marked as page-verified this session appears in
`research/memory/citation-verification.json` or `research/reference/papers/anchors.bib`, both
resolved against the live arXiv API.
