---
title: KV eviction policies — deciding what to forget before you know what you will be asked
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost, kv-cache-mechanics, tensors-and-autograd
mirrors: research/memory/kv-compression-and-eviction.md
difficulty: hard — the algebra is four lines; the discipline of not believing a policy comparison is the hard part
time: 4–5 h reading and working the math; 2–3 h for the three exercises (Exercise C needs a trained nanoGPT checkpoint you already have from Track A)
bridges_into: kv-serving-hierarchy, kv-attribution-instrument
---

# KV eviction policies

**Difficulty and time, honestly.** Nothing here is mathematically hard. The whole
theoretical spine is a four-line algebraic identity you can rederive on a napkin, and
the ten policies are each one scoring function. What is hard — genuinely hard, and the
reason this module exists in a memory track rather than an inference track — is
un-learning the reflex that a cache policy comparison means what it appears to mean.
Budget 4–5 hours for sections 1–5 with a pen. The exercises are 2–3 hours and Exercise B
is the one to protect time for: it produces a table that will change how you read every
eviction paper afterwards.

You have run LRU. You have tuned ARC. You know what a working set is. All of that is
load-bearing here and none of it is re-explained. What gets explained is the four places
it stops being true.

---

## 1. What this module settles

**One:** every KV eviction policy in the literature is the same object — a scoring
function that picks a subset `S` of past tokens to keep — and the exact error that
eviction inflicts on the model's output is a four-line identity with two factors, one of
which every attention-score policy optimises and the other of which almost none of them
looks at. **Two:** the caching intuition transfers further than you would expect (budget
is cache size, the attention sink is a pinned page, ChunkKV is a cache line) and then
breaks in four specific places, of which the deepest is that ARC's ghost lists have no
analogue here — you cannot learn from a bad eviction, because a bad eviction produces no
observable event. **Three:** on our hardware the eviction-versus-retention argument is
decidable rather than doctrinal, because capacity and bandwidth bind at different context
lengths and both numbers are measured — and the arithmetic says bandwidth binds
first by about 3×, which means eviction has to justify itself on quality-per-byte-read
rather than on capacity relief.

This module mirrors `research/memory/kv-compression-and-eviction.md`. That note surveys;
this one teaches, and adds three things the note does not have: a measured calibration of
the note's own recommended instrument (§3.6), a null-distribution table showing that
policy rankings appear on data with no structure in it at all (§6, Exercise B), and the
Belady framing that makes the oracle experiment obvious (§2.4).

---

## 2. Theory in plain language

### 2.1 The object, stated once

You have `T` tokens of context. Each has a key vector and a value vector cached, per
layer, per KV head. You have decided — because of capacity, or bandwidth, or both — that
you will keep only `B` of them. An **eviction policy** is a function that picks which `B`.

```
S = f(everything you know at decision time),    |S| = B
```

Every policy in §4 differs only in what goes inside `f` and *when* `f` runs. That is the
whole taxonomy. Resist the urge to memorise ten mechanisms; memorise one function
signature and ten arguments to it.

> **Systems bridge.** This is a replacement policy over a fixed-capacity cache, and a
> large amount of what you know applies directly:
>
> | Caching concept | KV eviction equivalent |
> |---|---|
> | cache size | budget `B` |
> | replacement policy | the scoring function |
> | pinned page | the attention sink (StreamingLLM's first four tokens) |
> | cache line / block size | ChunkKV's chunk granularity |
> | two-level index (coarse filter over fine) | RocketKV's two stages |
> | admission control from object metadata | L2-norm and KeyDiff scoring |
> | LFU counter | H2O's accumulated attention |
> | segmented LRU / protected segment | heavy hitters plus a recent window |
>
> Use all of it. The rest of this section is the four places it costs you.

### 2.2 Break 1 — there is no miss, so there is no hit rate

Every cache you have operated makes eviction a bet on **latency**. Get it wrong and you
pay a refill: slower, but correct, and — critically — *counted*. Hit rate is a first-class
observable, it is on a dashboard somewhere, and the entire discipline of cache tuning is
built on the fact that you can measure how often you were wrong.

Here, eviction is a bet on **correctness**. The evicted KV entry is the only existing
encoding of that token *in this context*; there is no lower tier holding a copy, and
"reconstructing" it means re-running prefill over the entire prefix. Inside the attention
kernel a miss is not slow, it is **unrepresentable**: FlashInfer's page table has no
present bit, so a page is in `kv_indices` or the token does not exist
(`memory/flashinfer/flashinfer/decode.py:1239`). vLLM's allocator, when it cannot find a
block, does not fault — it preempts the whole request
(`memory/vllm/vllm/v1/core/block_pool.py:647`).

The operational consequence is the one to internalise: **a wrong eviction produces a
fluent, confident, wrong answer, and emits no signal anywhere in the stack.** There is no
counter that goes up. There is no latency spike. Perplexity may not move
`[C]` [2510.00231](https://arxiv.org/abs/2510.00231) — that paper's worked case is a
multi-instruction prompt where a *specific* instruction is dropped entirely while the
aggregate benchmark looks fine, with system-prompt leakage as the demonstration.

Contrast this with what a real KV cache eviction looks like one layer up the stack. vLLM's
prefix cache evicts blocks, and that eviction *is* a latency bet with a legible cost: the
block leaves the hash table at `_maybe_evict_cached_block`
(`memory/vllm/vllm/v1/core/block_pool.py:679`), and until it does, a zero-refcount block
is still matchable and can be resurrected by `touch`
(`memory/vllm/vllm/v1/core/block_pool.py:702`). A miss there costs a recompute you can
price. Same words, entirely different contract. §5 makes you read both.

### 2.3 Break 2 — the working set is defined by a request that has not arrived

A buffer cache's frequency counters measure the same access stream that is going to
continue. Stationarity is a defensible prior; that is *why* LFU works at all.

Here the decision and the access come from different distributions, and the gap is
structural. Prefill compresses the context using statistics from **the prompt attending to
itself**. The thing that later reads the cache is **the answer attending to the prompt**.
Those are different query distributions, and they are adversarially different exactly when
the question is not predictable from the document's surface statistics — which is the case
users care about.

Write it as the field's actual failure, because once you see it the taxonomy collapses:

```
every policy computes    S  = f(K, V, prompt)      chosen once, serves every future query
the quantity that matters is  S*(q)                 the best subset for the query that arrives
```

This makes it a **robust-optimisation** problem — pick one subset that is good against a
distribution of unknown future queries — and not a ranking problem. Nine of the ten
canonical policies never write down that objective.

### 2.4 Break 3 — ARC's ghost lists have no analogue, and Belady's MIN half does

This is the deepest break and the one that most directly shapes what this lab should
build.

**ARC works because it learns from its own mistakes.** It keeps two ghost lists, B1 and
B2, holding the *metadata* of recently evicted entries. When a request hits a ghost entry,
ARC learns that its recency/frequency balance was wrong in a specific direction and shifts
its target size `p`. The ghost list is cheap because metadata is small, and it is
informative because a ghost hit is an observable event.

The KV analogue would require observing that the current query *would have* attended
strongly to a token you dropped. Computing that requires `q_t · k_i` — which requires
`k_i`, which you deleted. **There is no cheap metadata that answers the question, because
the metadata that answers it is the key itself.** You cannot build a ghost list without
keeping half the cache.

And here is the payoff of taking that sentence literally: *if you keep the keys and drop
only the reads*, you have a ghost list, and you have re-invented the retention camp. Quest
keeps per-page key bounds as summaries `[C]`
[2406.10774](https://arxiv.org/abs/2406.10774); SparQ keeps a subset of key channels to
approximate scores `[C]` [2312.04985](https://arxiv.org/abs/2312.04985). `[A]` High
confidence in this framing as a teaching device (it is mine, not a paper's): **sparse
attention is eviction with ghost lists, and the price of the ghost list is that you never
get the capacity back.** That single sentence explains why RocketKV's headline numbers are
"400× compression" and "32.6% peak memory reduction" in the same abstract `[C]`
[2502.14051](https://arxiv.org/abs/2502.14051) — most of the 400× is bandwidth, not bytes
held.

**Belady's MIN transfers halfway, and the half that transfers is the experiment nobody
runs.** In caching, MIN is uncomputable online but perfectly computable *offline* from a
trace, which is why you can always report how far your policy sits from optimal. The KV
analogue of the trace is the future attention weights, and those are also computable
offline — you just run the full cache alongside the compressed one. So an oracle
*ranking* is available, cheaply, at our scale.

The half that does not transfer: an oracle *ranking* is not an oracle *policy*. MIN is
optimal because cache misses are additive and independent. The eviction error (§3.3) is
not additive — the compressed output `o_S` appears inside its own error term, so dropping
token `i` changes the cost of dropping token `j`. Top-`B` by true future attention is the
right *diagnostic* ceiling and is **not** the optimum. H2O's own formulation acknowledges
the shape of this by casting selection as dynamic submodular maximisation with a guarantee
only "under mild assumptions" `[C]`
[2306.14048](https://arxiv.org/abs/2306.14048).

The lab-relevant conclusion: **compute the attention-oracle at every budget you test.** The
gap between your policy and the oracle is *policy headroom*; the gap between the oracle
and the full cache is *budget headroom*. Reporting one number without the other tells you
nothing about which to work on. As far as this pass could establish, no eviction paper
reports both. Exercise B builds it.

### 2.5 Break 4 — granularity carries semantics, and pinning is an arithmetic artifact

Two smaller breaks, stated quickly because you will hit them immediately.

**A cache line is a pure performance construct.** False sharing costs throughput and
nothing else. A KV "line" — ChunkKV's chunk `[C]`
[2502.00299](https://arxiv.org/abs/2502.00299) — costs *accuracy*, in both directions: one
high-scoring token drags its dull neighbours in and consumes budget, and one chunk holding
the single needed fact but otherwise unremarkable gets dropped whole.

**The attention sink is pinned for a reason you cannot discover by profiling.**
StreamingLLM's first few tokens are retained not because they are hot but because the
softmax denominator must sum to 1, and heads that want to attend to nothing need somewhere
to dump the mass `[C]` [2309.17453](https://arxiv.org/abs/2309.17453) — the paper says
they attract strong scores "even if they are not semantically important." `[C]`
[2410.10781](https://arxiv.org/abs/2410.10781) gives the empirical account of when sinks
emerge as a function of optimisation, data, and architecture. No access-frequency profile
will tell you that this page must be pinned; it is a property of the arithmetic, not of
the workload.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

Carried over from `attention-variants-and-kv-cost` §3, plus five new ones.

| Symbol | Reads as | Shape |
|---|---|---|
| `d` | head dimension — width of one head's vectors | scalar (128 on Laguna `[M]`, 64 in the exercises) |
| `q_t` | the **query** at decode step `t`: what the model is looking for *now* | vector, length `d` |
| `k_i` | the cached **key** of past token `i`: what it is, as a thing to be looked up | vector, length `d` |
| `v_i` | the cached **value** of past token `i`: what gets returned if it is looked up | vector, length `d` |
| `s_i` | raw score of token `i` against `q_t` | scalar |
| `a_i` | normalised attention **weight** on token `i` | scalar in `[0,1]`, `Σ_i a_i = 1` |
| `o_t` | the head's output at step `t` | vector, length `d` |
| **`B`** | the **budget** — how many past tokens survive | integer, `B ≤ T` |
| **`S`** | the **retained set** — which `B` survive | index set |
| **`A`** | **retained attention mass** — the fraction of softmax weight inside `S` | scalar in `[0,1]` |
| **`o_S`** | the head's output computed from `S` alone | vector, length `d` |
| **`u`** | the **lost signal**, `Σ_{i∉S} a_i v_i` | vector, length `d` |

The three lines you already know:

```
s_i = (q_t · k_i) / sqrt(d)         dot product, scaled so variance stays ~1 as d grows
a_i = exp(s_i) / Σ_j exp(s_j)       softmax: exponentiate, divide by the total
o_t = Σ_i a_i · v_i                 output is a weighted average of the values
```

And one decomposition to hold, because two policies exploit only half of it:

```
q_t · k_i  =  ‖q_t‖ · ‖k_i‖ · cos θ_i        magnitude × magnitude × alignment
```

### 3.2 Renormalisation is not a design choice

Run the softmax over `S` alone and you get, mechanically:

```
weight on token i, post-eviction  =  exp(s_i) / Σ_{j∈S} exp(s_j)  =  a_i / A
```

because the numerator is unchanged and the denominator lost the evicted terms. Therefore

```
o_S  =  (1/A) · Σ_{i∈S} a_i · v_i
```

Nobody chose to renormalise. It is what a softmax over fewer keys *does*. And it is the
single most important structural difference from every cache you have run:

> **Dropping a cache line does not change the value of the other lines. Dropping a KV
> entry increases every surviving token's weight, by exactly `1/A`.** There is no analogue
> anywhere in storage. It is why "we only dropped 5% of the attention mass" is not the
> reassurance it sounds like — §3.4 makes that arithmetic.

### 3.3 The exact error identity — four lines, derive it yourself

Start from the definition and split the sum:

```
o_t  =  Σ_{i∈S} a_i v_i  +  Σ_{i∉S} a_i v_i        split the sum at S
     =  A · o_S          +  Σ_{i∉S} a_i v_i        because Σ_{i∈S} a_i v_i = A · o_S
```

Subtract `o_S` from both sides and use `Σ_{i∉S} a_i = 1 − A`:

```
o_t − o_S  =  (A − 1)·o_S + Σ_{i∉S} a_i v_i
           =  Σ_{i∉S} a_i · (v_i − o_S)                                   [EXACT]
```

Then one triangle inequality:

```
‖o_t − o_S‖  ≤  (1 − A) · max_{i∉S} ‖v_i − o_S‖                          [BOUND]
```

`[C]` The same upper-bound framing is the stated theoretical contribution of Ada-KV, which
"establish[es] a theoretical loss upper bound between pre- and post-eviction attention
output" and then allocates budget across heads to minimise it
[2407.11550](https://arxiv.org/abs/2407.11550). The survey note lists Ada-KV only as a
per-head allocation rule; the bound is the more portable half.

**Two things fall out, and they organise everything after this.**

**(a) The error is a product of two independent factors.** How much mass you dropped,
`(1−A)`, and how far the dropped *values* sit from the retained average. Every
attention-score policy in §4 optimises the first and ignores the second entirely. That gap
is not hypothetical: `[C]` [2606.03928](https://arxiv.org/abs/2606.03928) (Jun 2026)
reports that a small fraction of value states carry abnormally large magnitudes — large
`‖v_i − o‖` — and that evicting them drives the model into repetitive reasoning loops.

**(b) Dropping redundant tokens is free, regardless of their attention weight.** If
`v_i = o_S`, the term vanishes. Read that against KeyDiff, which scores tokens by
*distinctiveness of the key* `[C]`
[2504.15364](https://arxiv.org/abs/2504.15364). Distinctiveness in **key** space governs
whether a token *gets* attention. Distinctiveness in **value** space governs whether
dropping it *hurts*. They are two different spaces and there is no reason for them to
agree. That is the sharpest available statement of why the whole attention-free-scoring
family is on thin ice, and it is one line of algebra.

### 3.4 The arithmetic, on four tokens, by hand

Four past tokens. Attention weights `a = [0.50, 0.30, 0.15, 0.05]`. Scalar values (take
`d = 1` so you can do it in your head) `v = [1, 1, 1, −10]` — one value outlier on the
*least*-attended token.

```
o_t  = 0.50(1) + 0.30(1) + 0.15(1) + 0.05(−10)  =  0.95 − 0.50  =  0.45
```

Evict token 4. It has the lowest attention weight, so **every score-based policy in this
module evicts it first.**

```
S = {1,2,3},   A = 0.50 + 0.30 + 0.15 = 0.95      dropped 5% of the mass
o_S = (0.50 + 0.30 + 0.15) / 0.95 = 0.95 / 0.95   =  1.00
error = o_t − o_S = 0.45 − 1.00 = −0.55
```

Check against the identity: `Σ_{i∉S} a_i (v_i − o_S) = 0.05 × (−10 − 1.00) = −0.55`. ✓
Check against the bound: `(1 − A) · max‖v_i − o_S‖ = 0.05 × 11 = 0.55`. ✓ (Tight here only
because exactly one token was dropped.)

**Dropping 5% of the attention mass moved the output by 122% of its own magnitude.**

Now change one number. Same weights, `v = [1, 1, 1, 0.5]`:

```
o_t = 0.975,   o_S = 1.00,   error = −0.025      →  2.6% relative
```

**Identical policy, identical budget, identical dropped mass — 22× different error.** The
policy cannot tell these two cases apart, because it never looked at `v`.

### 3.5 Two vectors, exactly — the split that is actually an instrument

Go back to the exact identity and pull the two pieces apart rather than bounding them:

```
o_t − o_S  =  Σ_{i∉S} a_i v_i  −  (1 − A) · o_S
           =        u          −  (1 − A) · o_S                          [EXACT]
```

- **`u = Σ_{i∉S} a_i v_i`** is the **lost signal**: the contribution the dropped tokens
  would have made.
- **`(1 − A)·o_S`** is the **renormalisation kickback**: the amount by which the surviving
  tokens got over-weighted to fill the gap.

Both are computable, both are one vector. The error magnitude follows from three scalars
by the law of cosines:

```
‖o_t − o_S‖²  =  ‖u‖²  +  ((1−A)‖o_S‖)²  −  2 · (1−A) · ‖u‖ · ‖o_S‖ · cos∠(u, o_S)
```

Log `‖u‖`, `(1−A)‖o_S‖`, and `cos∠(u, o_S)` per head per step and you can reconstruct the
exact error, attribute it to *lost information* versus *over-weighting*, and never
materialise a `T × T` matrix to do it. In the hand example above: `u = 0.05 × (−10) =
−0.50`, kickback `= 0.05 × 1.00 = 0.05`, and `|−0.50 − 0.05| = 0.55`. ✓

Note what this costs you at serving time: **nothing, and also everything.** `o_S` and `A`
are free if you have the full cache to compare against, and unavailable if you do not —
computing `A` requires the attention weights of tokens you deleted. So this is an
**offline attribution instrument**, which is precisely what `research/synthesis.md`
decided the lab should build, and not a production monitor.

### 3.6 `[M]` The measurement that says the standard decomposition is not an instrument

`research/memory/kv-compression-and-eviction.md` §6 recommends logging `(1 − A)` and
`max‖v_i − o‖` separately, on the grounds that no paper reports the two factors of the
bound. The recommendation is right in spirit and I think it names the wrong two
quantities. Here is the working; disagree with it if the numbers do not reproduce.

**`[M]`** Exercise A, run on this machine 2026-07-26, torch `2.12.0a0+rocm7.13.0a20260313`,
**CPU**, fp32, seed 1337, `H=8` heads, `T=1024` context, `d=64`, 64 independent contexts,
top-`B`-by-attention selection, budgets `{1,2,5,10,25,50}%`, 3072 samples per arm.
**Byte-identical output on two consecutive runs of the final script.** The script *is*
Exercise A — retest it there.

| Check | Gaussian values | 1% of value rows scaled 20× |
|---|---|---|
| worst relative residual of the exact identity | 2.196e-06 | 2.976e-06 |
| violations of the `(1−A)·max` bound (out of 3072) | 0 | 0 |
| mean tightness `err / [(1−A)·max]` | 0.129 | 0.013 |

The bound is a bound. It is never violated, and it is loose by roughly 8× on well-behaved
data and **75×** once value outliers exist — the regime the survey cites as the motivating
failure. A predictor that is wrong by 75× is not an instrument.

Worse, and this is the part worth arguing about — **within a fixed budget, `(1−A)` carries
essentially no information about the error at all:**

`[M]` within-budget R² of a linear fit of `‖o_t − o_S‖` on each candidate predictor:

| Budget | `(1−A)` | `(1−A)·max‖v−o_S‖` | `Σ_{i∉S} a_i‖v_i−o_S‖` | `(1−A)·‖o_S‖` |
|---|---|---|---|---|
| **Gaussian values** ||||
| 1% | 0.0028 | 0.0791 | 0.1882 | **0.9722** |
| 5% | 0.0124 | 0.0212 | 0.0322 | **0.8900** |
| 25% | 0.2451 | 0.2204 | 0.2480 | **0.7356** |
| 50% | 0.5066 | 0.4949 | 0.5109 | **0.6079** |
| **1% of value rows × 20** ||||
| 1% | 0.0003 | 0.0017 | 0.9649 | **0.9992** |
| 5% | 0.0011 | 0.0003 | 0.4355 | **0.9942** |
| 25% | 0.0460 | 0.0331 | 0.0499 | **0.9437** |
| 50% | 0.1757 | 0.1576 | 0.2181 | **0.7587** |

Three readings, in order of importance.

**One — the pooled number lies.** Pooled over all budgets, `(1−A)` scores R² = 0.764 on
the Gaussian arm, which looks respectable. Within any single budget it scores 0.003. The
pooled figure is entirely between-budget variance: you swept the thing that drives both
sides. **This is the standard inflation you get from regressing on a variable you also
swept**, and it is exactly the trap an eviction paper falls into when it plots degradation
against compression ratio across budgets and calls the correlation an explanation. The
operational regime is fixed-budget — you set the budget — so the fixed-budget column is
the one that matters.

**Two — the mechanism.** At a fixed budget with top-`B` selection, `A` is nearly constant
across heads and contexts, because it is determined by how much of a softmax's mass sits
in its top `B` entries and that is stable across similar score distributions. A predictor
with no variance predicts nothing. Meanwhile `‖o_S‖` swings widely, because renormalising
by `1/A` amplifies whatever noise survived.

**Three — what to log instead.** `(1−A)·‖o_S‖` — the renormalisation kickback of §3.5 —
reaches R² 0.97–0.999 at aggressive budgets and decays to 0.61–0.76 at 50%, which is
exactly what the two-term split predicts: as the budget grows, `o_S → o_t`, the kickback
shrinks, and the residual error becomes the lost-signal term `u`. The three-scalar
reconstruction of §3.5 is exact to 1.1e-06 in fp32 at every budget in both arms, as it must
be.

**The honest caveat, stated before you build anything on this.** This is **synthetic
isotropic Gaussian data, not a trained model.** It says nothing about whether real
attention behaves this way; what it establishes is that the *instrument* recommended in
§6 of the survey note has near-zero resolving power on data where the exact answer is
known. That is a calibration result, which is the only kind of result a harness is allowed
to produce before it is calibrated. Exercise C is the test on real attention, and it has a
KILL threshold. `[A]` Medium confidence that the ranking survives on a trained model; the
mechanism (fixed `B` ⇒ near-fixed `A`) is architecture-independent, but heads in a real
model have far more heterogeneous score distributions, which should give `A` back some
variance.

**Proposed fold-back into the survey note** (documentation class, mutable): replace §6's
"log `A` and dropped-value distance" with "log `‖u‖`, `(1−A)‖o_S‖`, and `cos∠(u, o_S)`;
the three reconstruct the error exactly and separate lost information from over-weighting."

### 3.7 `[M]` H2O's accumulator is biased toward old tokens, and here is by how much

H2O scores token `i` at step `t` by `h_i = Σ_{τ≤t} a_{τ,i}` — the running sum of attention
weight it has received `[C]` [2306.14048](https://arxiv.org/abs/2306.14048). The survey
note calls the resulting age bias "arithmetic, not a hypothesis." Make it arithmetic.

Assume the **null hypothesis that no token is more important than any other**: at each
step `τ`, attention is uniform over the `τ` available positions, so token `i` receives
`1/τ`. Then

```
h_i  =  Σ_{τ=i}^{t} 1/τ  =  H_t − H_{i−1}    where H_n = Σ_{k=1}^{n} 1/k  (harmonic number)
     ≈  ln(t / (i−1))
```

`[M]` Computed exactly (Exercise A prints it), `t = 4096`:

| token `i` | `h_i` under the uniform null |
|---|---|
| 1 | 8.8951 |
| 16 | 5.5769 |
| 256 | 2.7747 |
| 2048 | 0.6935 |
| 4000 | 0.0240 |

```
h_1 / h_4000  =  371×
```

**On data where every token is equally important by construction, H2O's score spans a
371× range driven purely by position.** It is unnormalised LFU where age is a head start,
and the obvious repair — divide by the number of opportunities `t − i + 1` — turns a sum
into a mean and is not what the paper does. Note the design consequence: H2O's "recent
window" term, usually described as an extra heuristic, is better read as **a correction
for an arithmetic artifact of its own accumulator.** Without it the newest tokens can never
compete.

Whether importance persists at all is contested and should stay contested. Scissorhands
argues yes `[C]` [2305.17118](https://arxiv.org/abs/2305.17118); LazyEviction documents
*Token Importance Recurrence* — a large proportion of tokens regaining high attention after
many decoding steps — for long chain-of-thought `[C]`
[2506.15969](https://arxiv.org/abs/2506.15969). They may be describing different regimes
rather than disagreeing.

### 3.8 The attention-free scorers are claims about a correlation coefficient

Since `s_i = ‖q_t‖ · ‖k_i‖ · cos θ_i / sqrt(d)` and `‖q_t‖` is common to all `i` at a given
step, ranking tokens by score is ranking them by `‖k_i‖ · cos θ_i`. Two policies drop the
query entirely and rank on the key alone.

**L2-norm** `[C]` [2406.11430](https://arxiv.org/abs/2406.11430) computes
`‖k_i‖₂ = sqrt(Σ_c k_i[c]²)` and keeps the **low**-norm keys. The paper's own framing is
the thesis of this whole module said out loud: the influence of a KV pair "is potentially
determined by the key embedding itself before being queried" (verbatim). Note the
direction — naively, a *large* norm should produce a large score. It does not, empirically,
because in trained decoders `cos θ` anti-correlates with `‖k‖` strongly enough to flip the
product.

**KeyDiff** `[C]` [2504.15364](https://arxiv.org/abs/2504.15364) scores by cosine distance
from the mean key direction `c`: `score_i = 1 − (k_i·c)/(‖k_i‖‖c‖)`, and keeps the most
distinctive.

Both are `O(d)` per token, need no attention matrix, and are therefore the only family
compatible with fused kernels. And both are, stripped of narrative, **claims that a
correlation coefficient measured on particular trained checkpoints has a particular sign.**
A correlation coefficient is exactly the kind of thing that can differ at 300M parameters,
which is why `research/memory/kv-compression-and-eviction.md` open question 1 is a Spearman
rank correlation on a model we train ourselves, and why a null result there would be a real
finding rather than a failed experiment.

KeyDiff's failure mode is a clean inversion worth carrying: a name repeated twenty times
through a document is by construction *not* distinctive and gets evicted early; a typo or a
stray identifier is maximally distinctive and gets pinned. `[A]` Medium-high confidence this
is real and measurable; cheapest test is a needle task where the needle is a *repeated*
phrase versus a *unique* one at matched budget.

> **Systems bridge and its break.** This is admission control from object metadata, before
> a single request has been seen. It works when metadata correlates with popularity, and
> fails on precisely the requests that deviate from population statistics — which are the
> requests users care about. The break specific to KV: dedup is safe in storage because the
> canonical copy survives behind a pointer. Here the "duplicate" you drop is *gone*, and
> near-duplicate **keys** carry different **values**.

### 3.9 The budget arithmetic on our machine — capacity and bandwidth bind at different lengths

Two measured inputs from `ASSUMPTIONS.md`: the fast tier is **≥62 GiB** and it runs flat at
**~200 GB/s** (`gpu-fast-tier-size`, `notebook/uma-carveout-controls-fast-tier.md`, single
run per arm). Take a plausible Proteus shape — 24 layers, `n_kv = 8`, `d_h = 64`, bf16 —
which from the KV product `2 · L · n_kv · d_h · b` gives:

```
2 × 24 × 8 × 64 × 2 B  =  49,152 B  =  48 KiB per token
```

**Capacity bound** — the context length at which one sequence's cache fills the fast tier:

```
62 GiB / 48 KiB  =  66,571,993,088 / 49,152  =  1,354,411 tokens
```

**Bandwidth bound** — decode re-reads the whole cache once per emitted token, so a target
generation rate is a byte budget per step:

```
199.9 GB/s ÷ 10 tok/s = 19.99 GB/step  ÷  48 KiB/token  =    406,698 tokens
199.9 GB/s ÷ 30 tok/s =  6.66 GB/step  ÷  48 KiB/token  =    135,566 tokens
```

| Constraint | Binds at |
|---|---|
| capacity (62 GiB fast tier) | 1,354,411 tokens |
| bandwidth at 10 tok/s | 406,698 tokens |
| bandwidth at 30 tok/s | 135,566 tokens |

`[A]` This is arithmetic over two `[M]` inputs and one assumed model shape, not a
benchmark — high confidence in the arithmetic, and the shape is `[A]` medium confidence
until an arm config is frozen. Rerun it with your own shape in Exercise A.

**The conclusion is specific and it is not the field's default.** On this machine, at this
scale, **bandwidth binds roughly 3.3× earlier than capacity.** Eviction relieves both
(fewer entries held and fewer bytes read); sparse retention relieves only bandwidth but
keeps the option of re-deciding per query. Since capacity is not the binding constraint
until context lengths we will never train at, **eviction cannot be justified here on
capacity grounds and must win on quality-per-byte-read against a retention baseline that
never destroys anything.** Set your default baseline accordingly.

Now run the same arithmetic for a discrete card — 20 GiB of HBM at `[A]` ~1 TB/s (a generic
figure, not measured here, and the only unmeasured input in this section):

```
capacity :  20 GiB / 48 KiB                       =    436,906 tokens
bandwidth:  (1 TB/s ÷ 10 tok/s) / 48 KiB          =  2,034,505 tokens
```

**The ordering inverts.** On a discrete card capacity binds 4.7× *before* bandwidth; on ours
bandwidth binds 3.3× before capacity. That is a 15× swing in the ratio, and it is the reason
a published "eviction beats retention" result may be a statement about a bus rather than
about language models. `research/synthesis.md` makes this the lab's distinctive experiment:
the fast/slow bandwidth ratio here is a BIOS setting rather than a bus, so we can move the
crossover and a discrete-GPU lab cannot move it at all.

### 3.10 `[M]`-derived: the attention matrix is a hardware constraint here, not just a kernel one

`[C]` [2606.26472](https://arxiv.org/abs/2606.26472) states that ranking by attention
weight "prohibits the use of fused kernels in production inference by forcing the model to
materialize the attention matrix." On our machine that constraint has a number attached.

The score matrix for one layer, batch 1, is `[1, n_h, T, T]` at 2 bytes per element:

```
bytes  =  2 · n_h · T²
```

`ASSUMPTIONS.md → large-tensor-fault-32gib` `[M]`: a **32 GiB single tensor hard-hangs the
GPU at 0% CPU with no error** and 36 GiB raises `hipErrorLaunchFailure`. Solving
`2 · n_h · T² = 32 GiB`:

| heads | `T` at which one score matrix hits the 32 GiB hang |
|---|---|
| 6 (nanoGPT shakespeare_char) | 53,510 |
| 8 | 46,341 |

So on this instrument, an H2O- or SnapKV-style policy at batch 1 and eight heads cannot see
past **~46k tokens** in a single materialised score tensor, and the failure mode is a silent
hang rather than an OOM. Design around it — chunk the score computation over the key axis —
rather than discovering it eleven minutes into a run.

There is a second, sharper trap here, and it is reflexive. `ASSUMPTIONS.md →
sdpa-is-memory-efficient` `[M]`: on this build `F.scaled_dot_product_attention` **retains
the score matrix by default** (147.2 bytes/T² retained, versus 6.6 with
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), and `flash_sdp_enabled()` returns `True`
either way. Consequence:

> **Our default configuration hides the very constraint that governs the field.** If you
> prototype an attention-score policy against default SDPA on this machine it will appear
> free, because the memory-hungry path was already running. Turn on the memory-efficient
> path — which you should, for an 18× activation reduction — and the incompatibility that
> the literature is organised around reappears. Record which path ran; a timing or memory
> number without it is uninterpretable.

---

## 4. The policies, as arguments to one function

Compressed to what each one *assumes*, since §3 gives you the machinery to price the
assumption yourself. Full treatment, evidence and breaks: the mirrored survey note.

| Policy | Assumes | `f` sees | Runs |
|---|---|---|---|
| **StreamingLLM** `[C]` [2309.17453](https://arxiv.org/abs/2309.17453) | importance is positional: sinks + recent window | position only | streaming |
| **H2O** `[C]` [2306.14048](https://arxiv.org/abs/2306.14048) | importance persists: past attention predicts future | accumulated `a` | decode |
| **Scissorhands** `[C]` [2305.17118](https://arxiv.org/abs/2305.17118) | the persistence hypothesis, named | `a` | decode |
| **TOVA** `[C]` [2401.06104](https://arxiv.org/abs/2401.06104) | last query's attention suffices | `a` at last step | decode |
| **FastGen** `[C]` [2310.01801](https://arxiv.org/abs/2310.01801) | heads have distinct, profilable patterns | per-head profile | prefill |
| **SnapKV** `[C]` [2404.14469](https://arxiv.org/abs/2404.14469) | the prompt's last `w` queries proxy the real query | `a` in an observation window | end of prefill |
| **PyramidKV** `[C]` [2406.02069](https://arxiv.org/abs/2406.02069) | the *budget* should shrink with depth | SnapKV + layer index | end of prefill |
| **Ada-KV** `[C]` [2407.11550](https://arxiv.org/abs/2407.11550) | the budget should vary per **head**, minimising the §3.3 bound | `a` per head | end of prefill |
| **ChunkKV** `[C]` [2502.00299](https://arxiv.org/abs/2502.00299) | importance lives in semantic chunks, not tokens | chunk-summed `a` | end of prefill |
| **L2-norm** `[C]` [2406.11430](https://arxiv.org/abs/2406.11430) | importance is intrinsic to the key | `‖k_i‖` | write time |
| **KeyDiff** `[C]` [2504.15364](https://arxiv.org/abs/2504.15364) | importance is geometric distinctiveness | `cos(k_i, mean k)` | write time |
| **EpiKV** `[C]` [2606.26472](https://arxiv.org/abs/2606.26472) | importance is the representation change a token induces | forward-pass deltas | write time |
| **FastKV** `[C]` [2502.01068](https://arxiv.org/abs/2502.01068) | importance stabilises with depth | activations at a TSP layer | **inside the forward pass** |
| **RocketKV** `[C]` [2502.14051](https://arxiv.org/abs/2502.14051) | coarse eviction and fine sparse read compose | stage 1: prompt; stage 2: `q_t` | both |
| **KVzip** `[C]` [2505.23416](https://arxiv.org/abs/2505.23416) | keep whatever reconstructs the context | reconstruction loss | offline |
| **KVpop** `[C]` [2607.05061](https://arxiv.org/abs/2607.05061) | decide *later*: defer `k` steps to see more context | lagged attention | deferred |

Four notes that are worth more than the table.

**FastKV is not an eviction policy and cannot be hosted by a cache interface.** A token not
propagated past its TSP layer never has its KV *computed* for the layers above; those bytes
were not evicted, they were never produced. Any interface claiming to host FastKV behind a
`score(...) -> subset` hook is lying about its boundary. The systems reading is predicate
pushdown in a query plan — and the break is that a pushed-down predicate is *provably* safe
(the discarded rows could not have affected the result) while TSP is a guess with no
after-the-fact check.

**RocketKV is the most instructive entry in the list**, because its architecture is a tacit
admission. Stage 1 is query-blind and permanent; stage 2 is query-*aware* and per-step. The
design says: keep the blind stage conservative, and get the real value from **deferring**
the fine decision until the query exists. That is the interface lesson of §5, discovered by
someone building a system.

**KVpop converts prematurity into a latency knob** `[C]`
[2607.05061](https://arxiv.org/abs/2607.05061), deferring the keep/drop decision a fixed
number of steps to exploit near-future context. This is a write-back cache with a staging
buffer: pay `k` steps of extra residency, buy `k` steps of extra information. It is the one
idea in the field that transfers from systems cleanly and completely, and the cleanest
available answer to "you must decide before the query is known" — *decide later*.

**2026's live trend is stochastic retention.** Two independent lines argue that
*deterministic* top-`B` is the mistake: VaSE `[C]`
[2606.03928](https://arxiv.org/abs/2606.03928) (Jun 2026) with value-aware stochastic
eviction, and Nexus Sampling `[C]`
[2606.23961](https://arxiv.org/abs/2606.23961) (22 Jun 2026), which combines iterative
attention scoring with probabilistic retention to preserve "subtly important" tokens and
reports within 1% of dense attention at 80% cache reduction. The caching analogue is
randomised admission (TinyLFU's doorkeeper, and the coin-flip in CLOCK-Pro variants), and
the motivation is the same: a deterministic ranking makes a *systematic* error on the tail,
where a randomised one makes an *unbiased* one. This is newer than the survey note's
framing and worth folding back.

---

## 5. Why it matters for Proteus and Mnemosyne

### 5.1 A single hook cannot express this field

`score(keys, values, attention) -> subset` cannot host FastKV (it edits the forward pass),
RocketKV stage 2 (it needs `q_t` at read time), or KVpop's deferred scorer (it needs a
staging buffer and a flush trigger). The minimum honest surface is **three plug points,
distinguished by when they run relative to the query** — which is the only axis that
matters, per §2.3:

| Plug point | Runs | Inputs available | Hosts | Cost |
|---|---|---|---|---|
| **write-time admission** | as each token's KV is produced | `k_i`, `v_i` only — `O(d)`, no attention matrix | L2-norm, KeyDiff, EpiKV | free; fused-kernel safe |
| **deferred eviction** | after a bounded staging delay | prompt-side or `k`-step-lagged attention | H2O, SnapKV, PyramidKV, Ada-KV, ChunkKV, KVpop | `k` tokens of extra residency; needs the score matrix |
| **read-time selection** | per decode step | the actual `q_t` | Quest, SparQ, RocketKV stage 2 | no capacity relief; per-step selection cost |

This is a decision to make before any policy is coded, and it belongs in
`docs/adr/mnemosyne-cache-plugpoints.md` rather than in a docstring. Note also the boundary
rule from `CLAUDE.md`: Mnemosyne never imports Proteus. All three plug points above are
expressible against a model-agnostic cache handle. **FastKV is not**, which is a clean
demonstration that the boundary is doing real work rather than being decoration — the thing
it excludes is exactly the thing that is not a memory policy.

### 5.2 The lab is building the instrument, not a 31st policy

`research/synthesis.md` parks eviction-policy design (issue-tree node 3.1) on the grounds
that ~30 policies exist and the Mar 2026 survey scores five families against seven
deployment scenarios and finds **no method dominates** `[C]`
[2603.20397](https://arxiv.org/abs/2603.20397). This module is not a contradiction of that
decision; it is the prerequisite for it. You cannot build an attribution instrument for
eviction damage without knowing exactly what eviction does, and §3.3–3.6 is that.

What the instrument needs from this module, concretely:

1. **Three scalars per head per step** — `‖u‖`, `(1−A)‖o_S‖`, `cos∠(u, o_S)` (§3.5). Exact
   reconstruction, no `T×T` materialisation, and it separates lost information from
   over-weighting. Do not log `A` alone; §3.6 measured its resolving power at 0.003.
2. **The attention-oracle at every budget** (§2.4). Policy headroom and budget headroom are
   different numbers and a single accuracy figure conflates them.
3. **A null distribution** — the same harness on structureless data (Exercise B). A policy
   ranking that reproduces on Gaussian noise is not a finding about language models.
4. **Two compression numbers, always** — bytes held and bytes read. Any Mnemosyne metric
   reporting one number is unreportable (§2.4, RocketKV's 400× vs 32.6%).

### 5.3 Three hardware constraints that are design inputs, not surprises

- **Layout decides where the 32 GiB hang lives.** At 48 KiB/token, a cache allocated as one
  tensor for the whole stack hits 32 GiB at 699,050 tokens; allocated per layer at
  2 KiB/layer/token it hits it at 16.7M. Per-layer allocation is not a style preference, it
  is a 24× headroom decision, and the failure it avoids is a silent hang.
- **Score-matrix chunking is mandatory above ~46k tokens** (§3.10) for any attention-score
  policy at eight heads.
- **The baseline is retention, not eviction** (§3.9), because bandwidth binds 3.3× before
  capacity does at our scale. An eviction arm that does not beat a matched-bytes-read sparse
  read is not interesting here even if it beats the full cache on cost.

---

## 6. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers pin to the revisions in `PROVENANCE.md`.

Read these in the order given. The order is the argument.

### 6.1 First, a real eviction policy — so you know what you are giving up

| Where | What to look at, and why |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `LRUStrategy.get_priority` returns `node.last_access_time`. The **entire** replacement-policy surface of a production serving engine is one function returning a sort key. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:21` | `LFUStrategy` returns `(hit_count, last_access_time)` — LFU with LRU tie-break, as a two-tuple. Note the field it reads: `hit_count`. **That field can exist because hits are observable.** Hold that thought for §6.2. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:49` | `SLRUStrategy` — segmented LRU with a protected segment, the closest thing in the file to "heavy hitters plus a recent window." Six policies live in 65 lines; the mechanism is never the hard part in either field. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict` — a heap built fresh per call from `evictable_leaves`, popping lowest priority, freeing KV slots, re-pushing a parent once it loses its last child. Eviction is **topologically constrained**: only leaves are candidates, so a hot child keeps a cold parent resident forever. No LRU chain you have run has that property. |
| `memory/vllm/vllm/v1/core/block_pool.py:679` | `_maybe_evict_cached_block` — the only place a block leaves the hash table, called **lazily at reallocation**, not at free. |
| `memory/vllm/vllm/v1/core/block_pool.py:702` | `touch` — a matched block is unlinked from the free queue in O(1) and its refcount bumped. This is the resurrection path: a zero-refcount block is still matchable. |
| `memory/vllm/vllm/v1/core/block_pool.py:719` | `free_blocks` — hash-less blocks prepended (die first), hashed blocks appended (linger as reuse candidates). A two-tier reclaim written in eight lines. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:5211` | `TryPushPromotionQueue` — TinyLFU-style count-min-sketch admission before a disk-tier object may enter DRAM. **A promotion path.** Note it exists at all, then note that nothing in §6.2 has one. |

**What to take away:** every one of these policies reads a field that was written by an
observed event — `last_access_time`, `hit_count`, a sketch counter. That is the resource
token-level KV eviction does not have.

### 6.2 Then, where a token-level policy would have to live — and why it cannot live there

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:328` | `attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling` — the `[B, n_h, T, T]` score matrix, materialised. This is the tensor §3.10 prices at 46,341 tokens. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:332` | `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)` — **the only line in the whole forward pass where `a_i` exists as a value.** Every attention-score policy in §4 must hook here or reconstruct this tensor. Note `dtype=torch.float32`: the softmax is upcast, so the scores you would score with are fp32 even in a bf16 model. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:337` | `return attn_output, attn_weights` — the eager path hands the weights back, and the caller throws them away. At `T = 46,341` and eight heads that is a **32 GiB** tensor per layer, computed and discarded, every forward pass. The information every attention-score policy needs is produced and dropped on the floor; the cost of "using" it is not computing it, it is *keeping* it. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:90` | The return annotation: `-> tuple[torch.Tensor, None]`. **The incompatibility that governs the entire field is visible as a type.** The SDPA path structurally cannot return attention weights, so H2O, SnapKV, PyramidKV, Ada-KV and ChunkKV cannot be implemented behind it. |
| `architecture/transformers/src/transformers/integrations/sdpa_attention.py:91` | `if kwargs.get("output_attentions", False):` → a warning telling you to switch to eager. This is the fused-kernel constraint, in a log line, for anyone who did not read the type signature. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `past_key_values.update(key_states, value_states, self.layer_idx)` — the one line where bytes enter the cache, and therefore **the write-time admission plug point** of §5.1. Note what has already happened above it: QK-norm at `:390`, RoPE at `:394`. A cached key is `RoPE(RMSNorm(k))`, so L2-norm and KeyDiff are scoring a doubly-transformed quantity, not the raw projection. |

**The read to do slowly.** Open `modeling_laguna.py:332` and `sdpa_attention.py:90` side by
side. One materialises `a` and hands it to you; the other's type says `None`. The entire
attention-score-versus-attention-free dispute of the 2026 literature is the choice between
those two lines, and it is a *systems* choice — kernel fusion — masquerading as a modelling
choice.

### 6.3 Two contrasts that sharpen the definition of eviction

| Where | What to look at, and why |
|---|---|
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` | The entire inter-chunk recurrence: decay the carry, add the chunk contribution. The state is **destructively overwritten**, so there is nothing to evict and nothing to attribute back to a specific past token. Not a cache with a harsh policy — a different data structure. Contrast with `architecture/mamba/mamba_ssm/modules/mamba2.py:352`, where `ssm_state` is allocated with **no `seqlen` dimension at all**. |
| `training/nanogpt/model.py:306` | `generate` re-runs the full prefix per sampled token — quadratic and cacheless by design. Useful as the degenerate control: a system with **no** KV cache has no eviction problem and pays for it in compute. Every policy in §4 is a point on the line between this and the full cache. |

---

## 7. Exercises

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

Standing caveats from `ASSUMPTIONS.md`, restated because they bite here specifically:
single tensors **≥32 GiB hang silently at 0% CPU** (`large-tensor-fault-32gib`) — §3.10 says
a score matrix reaches that at T≈46k; bf16 numerics on gfx1151 are **unproven**
(`bf16-numerics-unproven`), so run Exercises A and B in fp32, where they belong anyway; and
the Hardware Validation Gate has not run, so nothing here is evidence by house standard.
These are instrument-shakedown runs and should be labelled as such in your notebook entry.

Write scratch scripts under `notebook/`. Exercises A and B share a harness and should
migrate into `packages/mnemosyne/` with tests on second use — that harness is the
attribution instrument of §5.2, arriving early.

---

### Exercise A — the error identity, and what actually predicts damage

**Goal.** Establish that the identity is exact, that the bound is a bound, and then find out
which candidate instrument has any resolving power at a fixed budget. Reproduce the `[M]`
table in §3.6, then break it.

**Hardware:** none — pure CPU, fp32, no GPU, no model. **Runtime:** ~30 min to write,
**~5 s to run** `[M]` (5.2 s wall for both arms plus the §3.7 and §3.9 extras, 4 threads,
this machine, 2026-07-26). **CPU is the primary path**, not a fallback — there is nothing
here a GPU would help with.

```python
"""Exercise A: the exact eviction-error identity and its candidate instruments."""
import math
import torch

SEED, H, T, D, N_CONTEXTS = 1337, 8, 1024, 64, 64
BUDGET_FRACS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]

def make_context(gen, outlier_frac, outlier_scale):
    q = torch.randn(H, D, generator=gen)
    K = torch.randn(H, T, D, generator=gen)
    V = torch.randn(H, T, D, generator=gen)
    if outlier_frac > 0.0:
        idx = torch.randperm(T, generator=gen)[: max(1, int(outlier_frac * T))]
        V[:, idx, :] *= outlier_scale
    return q, K, V

def attention(q, K, V):
    s = torch.einsum("hd,htd->ht", q, K) / math.sqrt(D)
    a = torch.softmax(s, dim=-1)
    return a, torch.einsum("ht,htd->hd", a, V)

def evict(a, V, keep_idx):
    a_keep = torch.gather(a, 1, keep_idx)
    A = a_keep.sum(dim=1)
    V_keep = torch.gather(V, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))
    return A, torch.einsum("hb,hbd->hd", a_keep, V_keep) / A.unsqueeze(-1)

def r2(y, x):
    x, y_c = x - x.mean(), y - y.mean()
    beta = (x * y_c).sum() / (x * x).sum()
    return float(1.0 - (y_c - beta * x).pow(2).sum() / y_c.pow(2).sum())

def run(outlier_frac, outlier_scale, label):
    gen = torch.Generator().manual_seed(SEED)
    rows = []
    for _ in range(N_CONTEXTS):
        q, K, V = make_context(gen, outlier_frac, outlier_scale)
        a, o = attention(q, K, V)
        for frac in BUDGET_FRACS:
            B = max(1, int(frac * T))
            keep = a.topk(B, dim=1).indices
            A, o_S = evict(a, V, keep)
            drop = torch.ones(H, T, dtype=torch.bool).scatter_(1, keep, False)
            err = (o - o_S).norm(dim=-1)
            dist = (V - o_S.unsqueeze(1)).norm(dim=-1)
            a_drop = torch.where(drop, a, torch.zeros(1))
            one_minus_A = 1.0 - A
            u = torch.where(drop.unsqueeze(-1), a.unsqueeze(-1) * V, torch.zeros(1)).sum(1)
            rows.append(torch.stack([
                err,                                                  # 0 actual error
                one_minus_A,                                          # 1 dropped mass
                one_minus_A * torch.where(drop, dist, torch.zeros(1)).max(1).values,  # 2
                (a_drop * dist).sum(1),                               # 3 L1 mass-weighted
                one_minus_A * o_S.norm(dim=-1),                       # 4 renorm kickback
                u.norm(dim=-1),                                       # 5 lost signal
                (u * o_S).sum(-1) / (u.norm(dim=-1) * o_S.norm(dim=-1)),  # 6 cosine
                torch.full_like(err, frac),                           # 7 budget
            ], dim=-1))
    R = torch.cat(rows, 0)
    err, fr = R[:, 0], R[:, 7]
    recon = (R[:, 5] ** 2 + R[:, 4] ** 2 - 2 * R[:, 4] * R[:, 5] * R[:, 6]).clamp_min(0).sqrt()
    print(f"\n=== {label} ===")
    print(f"3-scalar reconstruction, max rel error  "
          f"{float(((recon - err).abs() / err).max()):.3e}")
    print(f"bound violations                        "
          f"{int((err > R[:, 2] + 1e-4).sum())} / {R.shape[0]}")
    print(f"{'budget':>8}{'(1-A)':>10}{'(1-A)max':>10}{'L1-mass':>10}{'(1-A)|o_S|':>12}")
    for frac in BUDGET_FRACS:
        m = fr == frac
        print(f"{frac:>7.0%}" + "".join(
            f"{r2(err[m], R[m, c]):>10.4f}" if c != 4 else f"{r2(err[m], R[m, c]):>12.4f}"
            for c in (1, 2, 3, 4)))

torch.set_num_threads(4)
print(f"torch {torch.__version__} seed {SEED} H={H} T={T} D={D} contexts={N_CONTEXTS}")
run(0.0, 1.0, "Gaussian values")
run(0.01, 20.0, "1% of value rows scaled 20x")
```

**Deliverables — four numbers and one judgement.**

1. **The reconstruction residual.** Must be `< 1e-5`. It is guaranteed by algebra, so a
   larger number means your gather indices or your masking are wrong, not that the maths is.
   This is the harness self-check; it cannot fail for interesting reasons.
2. **Bound violations: exactly 0** out of 3072 per arm. Same reasoning.
3. **The within-budget R² table.** Compare against §3.6. `[M]` The reference run gives
   `(1−A)` = 0.0028 and `(1−A)‖o_S‖` = 0.9722 at the 1% budget on the Gaussian arm. If you
   get materially different numbers on the same seed and shape, one of us has a bug and it
   is worth finding out which.
4. **Then break it.** Change `outlier_scale` to 1.0 and `outlier_frac` to 0.5 (half the
   values scaled by nothing — i.e. back to the null). Then try `outlier_frac=0.30,
   outlier_scale=3.0`. **Find a configuration where `(1−A)` recovers R² > 0.5 within a fixed
   budget, or convince yourself none exists and say why.** Write the answer in your notebook
   entry in one sentence.

**Also print the two derived quantities** you will need later: the H2O harmonic-null score
ladder at `t=4096` (§3.7, `h_1/h_4000` must come out at 371) and the capacity/bandwidth
crossover for **your** intended Proteus shape (§3.9). Both are ten lines of arithmetic and
both go in `ASSUMPTIONS.md` as derived, not measured.

---

### Exercise B — the policy null: calibrate before you conclude

**Goal.** Run five eviction policies against data that contains **no structure whatsoever**,
where the correct answer is known: no query-blind policy should be able to beat random,
because there is nothing to be right about. Then discover that three of them do, and work
out why.

This is the single most useful hour in the module. Every eviction paper you read afterwards
gets measured against this table.

**Hardware:** CPU, fp32. **Runtime:** ~20 min to write, **a few seconds to run** (it is a
subset of Exercise A's 5.2 s `[M]`). A GPU variant is pointless at these shapes.

Extend Exercise A's harness with five selectors at each budget:

```python
knorm = K.norm(dim=-1)                                     # [H, T]
cand = {
    "random":        torch.rand(H, T, generator=gen).topk(B, dim=1).indices,
    "l2-keep-low":   (-knorm).topk(B, dim=1).indices,      # the published L2 rule
    "l2-keep-high":  knorm.topk(B, dim=1).indices,         # its inverse
    "recent-window": torch.arange(T - B, T).expand(H, B).contiguous(),
    "oracle-topk":   a.topk(B, dim=1).indices,             # perfect query knowledge
}
```

and for each record: mean relative output error `‖o − o_S‖/‖o‖`, retained mass `A`, the
**effective retained count** `1 / Σ_i (a_i/A)²` (the participation ratio — how many tokens
the surviving weight is really spread over), and `‖o_S‖/‖o‖`.

**`[M]` Reference output** from this machine (same config as §3.6 — CPU, fp32, seed 1337,
`H=8, T=1024, D=64`, 64 contexts; identical on two consecutive runs):

| budget | random | l2-keep-low | l2-keep-high | recent-window | oracle-topk |
|---|---|---|---|---|---|
| **mean relative output error** ||||||
| 1% | 8.542 | **7.703** | 9.443 | 8.381 | 5.995 |
| 5% | 4.100 | **3.690** | 4.574 | 4.121 | 2.344 |
| 25% | 1.710 | **1.649** | 1.716 | 1.698 | 0.644 |
| 50% | 0.997 | 1.055 | **0.913** | 0.992 | 0.243 |
| **effective retained count** ||||||
| 1% | 5.6 | 6.6 | 4.6 | 5.6 | 9.0 |
| 50% | 197.4 | 226.9 | 179.9 | 198.1 | 279.3 |
| **‖o_S‖ / ‖o‖** ||||||
| 1% | 8.571 | 7.707 | 9.506 | 8.411 | 6.528 |
| 50% | 1.407 | 1.306 | 1.470 | 1.405 | 1.175 |

**Deliverables — three findings, each of which can be wrong.**

1. **"L2-keep-low beats random by 10% at every budget up to 25%."** On data with no
   structure. Reproduce it, then explain it. The explanation is in the table: read the
   error column against the `‖o_S‖/‖o‖` column and notice they are nearly the same numbers.
   Then read the effective-retained-count row. **Write down the mechanism in one sentence
   before reading the answer in §10 (self-check 4).**
2. **The ordering inverts between 25% and 50%.** `l2-keep-high` goes from worst to best;
   `l2-keep-low` from best to worst. So "policy X beats policy Y" is not even a
   *budget-stable* statement on structureless data. Report the crossover budget you measure.
3. **Oracle headroom.** At 1% the oracle sits at 5.995 against the best heuristic's 7.703 —
   a 1.28× policy gap — while the oracle itself is 6× worse than the full cache. **Policy
   headroom is small; budget headroom is enormous.** Compute both ratios at every budget and
   state which one you would work on. This is the number §2.4 argues nobody reports.

**Pre-register before you run** (G2 card, house rule). SUCCESS: the reference numbers
reproduce to three significant figures on the same seed. KILL: if `random` wins at every
budget, your selectors are not doing what you think and the null is uninformative — debug
rather than interpret.

**What this exercise licenses you to say.** Nothing about language models. What it licenses
is a *veto*: any published comparison whose margins are inside the margins in this table has
not demonstrated that its policy knows anything about importance. Keep the table.

---

### Exercise C — the prematurity penalty on a real model

**Goal.** Measure the module's central claim on real attention: how much does it cost to
choose the retained set using a query issued *before* the one that reads it? This is
SnapKV's assumption, measured, with a KILL condition that would make SnapKV free.

**Prerequisite:** the trained nanoGPT `shakespeare_char` checkpoint from Track A
(`the-training-loop`) — 6 layers, 6 heads, 384 channels, `block_size=256`,
`training/nanogpt/config/train_shakespeare_char.py:22`, published best val loss 1.4697
(`training/nanogpt/README.md:51`). If you do not have it, the CPU recipe is
`training/nanogpt/README.md:85` (4 layers, 128 channels, `block_size=64`, 2000 iters,
published target 1.88); at `block_size=64` cap `Δ` at 16.

**Hardware:** GPU or CPU; the forward passes are trivial. **Runtime:** ~45 min to write,
**2–5 min to run** on a checkpoint you already have. `[A]` If you must train first, budget
20–60 min on the Z13 for the GPU recipe — no measurement of nanoGPT wall-clock on this
machine exists yet, so treat that as a guess and record the real number.

**Method.**

1. Load the checkpoint. Register a forward hook on each block's `c_attn`
   (`training/nanogpt/model.py:56` — `q, k, v = self.c_attn(x).split(...)`) and capture
   `q, k, v` reshaped to `[B, n_h, T, d_h]`.
   **Do not** flip `attn.flash = False` to get the score matrix:
   `training/nanogpt/model.py:45` sets `flash` at construction and
   `training/nanogpt/model.py:46` only registers the causal `bias` buffer when it is
   `False`, so a post-hoc flip raises `AttributeError` on `self.bias`. Computing attention
   yourself from the captured `q, k, v` is both simpler and reuses Exercise A's code
   unchanged.
2. Fix a decision position `p = 128` and a budget `B`. For each layer and head, build `S`
   from the attention row at `p` — this is the "prompt-end proxy," SnapKV with `w = 1`.
3. Evaluate the resulting error at query position `p + Δ` for
   `Δ ∈ {1, 2, 4, 8, 16, 32, 64}`, using the causal mask so only positions `≤ p` are
   candidates in both arms.
4. Compare against the **matched** arm: `S` chosen from the attention row at `p + Δ` itself,
   restricted to the same candidate set.

```
prematurity_penalty(Δ)  =  mean_err[ S from row p,  evaluated at row p+Δ ]
                         ─────────────────────────────────────────────────
                           mean_err[ S from row p+Δ, evaluated at row p+Δ ]
```

**Deliverables — one plot and two numbers.**

1. **Plot penalty against `Δ`, one line per layer**, at budgets 5% and 25%.
   **Prediction:** monotone increasing and saturating. A non-monotone curve is interesting
   and should be reported, not smoothed.
2. **The penalty at `Δ = 64`, 5% budget, averaged over layers and heads.** This is the
   headline number. **KILL:** if it is `≤ 1.05`, then at this scale queries are effectively
   interchangeable, SnapKV's proxy costs nothing here, and small-scale eviction experiments
   cannot detect the prematurity failure at all — which is a finding about our *rig*,
   directly relevant to `ASSUMPTIONS.md → ablation-scale-sufficient`, and more valuable
   than a confirmation would be.
3. **The §3.6 instrument, re-checked on real attention.** Compute the within-budget R² of
   `(1−A)` and `(1−A)‖o_S‖` on these real heads. **Prediction:** `(1−A)` recovers some
   power relative to the synthetic 0.003, because real heads have heterogeneous score
   distributions, but stays below `(1−A)‖o_S‖`. **KILL:** if `(1−A)` wins on real
   attention, §3.6's recommendation is a synthetic-data artifact and this module needs an
   appended correction. Say so plainly if it happens.

**Two confounds to state in your notebook entry before you run.** A char-level model on
Shakespeare has a very local attention profile, which will *understate* the prematurity
penalty relative to a document-QA workload. And `block_size = 256` means `Δ = 64` is a
quarter of the whole context, so "distant query" here is not distant in any absolute sense.
Both push the measurement toward the KILL threshold, so a *positive* result is strong and a
null result is weak. Say which one you got.

---

## 8. Self-check

Answers in §10. Do not scroll.

1. You evict the single lowest-attention token out of 4,096, dropping 0.01% of the attention
   mass. Give a concrete construction in which the head's output changes by more than 100%
   of its own magnitude, and state which of the two factors in §3.3 you exploited.

2. You are asked to add a hit-rate metric to a KV eviction policy, so operations can alert
   on cache pressure. Explain in two sentences why the request is not merely hard but
   ill-posed — and then name the one thing you could keep that would make an ARC-style ghost
   list possible, and what it would cost.

3. `research/memory/kv-compression-and-eviction.md` recommends logging retained attention
   mass `A` per layer and head. §3.6 measures its within-budget R² at 0.003 on synthetic
   data. Is the note wrong? Answer with a distinction, not a verdict.

4. A colleague reports that on their benchmark, keeping low-L2-norm keys beats random
   selection by 9% at a 5% budget, and cites this as replication of
   `[C]` [2406.11430](https://arxiv.org/abs/2406.11430). Using only Exercise B's table,
   state the objection in one sentence.

5. Our fast tier is `[M]` ≥62 GiB at ~200 GB/s. A Proteus arm costs 48 KiB/token of KV.
   Your collaborator proposes an eviction policy to "make long context fit." At what context
   length does capacity actually bind, at what context length does bandwidth bind if you
   want 10 tokens/second, and what does the gap between those two numbers imply about which
   family of technique you should baseline against?

6. H2O keeps heavy hitters *plus a recent window*, and the recent window is usually described
   as a separate heuristic for locality. Give the arithmetic reason the window is better read
   as a bug fix, and state the one-line change that would remove the need for it.

---

## 9. What is still unsolved here

Everything below is testable at 20M–300M params on one GPU with a `[M]` ≥62 GiB fast tier,
and every item needs a pre-registered hypothesis card. Ordered by information per GPU-hour.

1. **Which error factor binds, on a real model?** §3.6 says `(1−A)` is uninformative and
   `(1−A)‖o_S‖` is nearly sufficient — on synthetic isotropic data. Nobody has reported the
   decomposition at all, on any data. Exercise C step 3 is the cheapest version; the full
   version instruments H2O, SnapKV, L2 and KeyDiff at matched budget and regresses downstream
   degradation on each factor. This is the highest-value item in the module and it is one
   forward pass plus bookkeeping.

2. **What is the oracle headroom, and does anyone's policy occupy it?** §2.4. Report policy
   headroom (policy → attention-oracle) and budget headroom (oracle → full cache) at every
   budget. Cheap at our scale, unaffordable at 70B, and as far as this pass could establish,
   unpublished. Note the honest caveat: the attention-oracle is a ranking, not the optimum,
   because the objective is non-additive — so it is a *diagnostic ceiling* and quantifying
   the gap between it and the true combinatorial optimum is itself open.

3. **Does the L2/attention anti-correlation exist at 20M–300M?** Spearman rank correlation
   between `−‖k_i‖₂` and mean received attention, per layer and per head, on a model we
   trained. One forward pass. A null result kills every attention-free scorer as a testable
   object in this lab, and is publishable-grade negative evidence.

4. **Is deferral monotone, and where is the knee?** KVpop `[C]`
   [2607.05061](https://arxiv.org/abs/2607.05061) defers the keep/drop decision `k` steps.
   `k` is a staging-buffer size in tokens, so this is a pure capacity-versus-quality curve of
   exactly the shape a write-back cache tuning study produces. A monotone curve with a sharp
   knee turns "decide later" from a research question into a tunable.

5. **Does the pyramid survive a hybrid SWA/global stack?** PyramidKV's depth profile is a
   measured property of uniform-global networks. Our reference model is 12 full + 36 sliding
   in strict GSSS `[M]`. Measure per-layer attention entropy split by layer type. If the
   pyramid is an artifact of uniform stacks, the allocation rule is mis-specified for the
   model class we study.

6. **Is cross-layer index reuse compatible with depth-varying attention?** ChunkKV asserts
   the retained *set* is stable with depth; PyramidKV asserts attention *changes
   qualitatively* with depth. Both can be true — set stability and concentration change are
   different quantities — but nobody has measured them in one harness. Jaccard overlap of
   retained index sets between adjacent layers, split by layer type and budget.

7. **Where does capacity stop binding and bandwidth start, measured rather than derived?**
   §3.9 derives the crossover from two single-run `[M]` inputs. Measuring it means sweeping
   the BIOS carve-out, which is the one experiment `research/synthesis.md` identifies as
   unavailable to a discrete-GPU lab. The `≥32 GiB` single-tensor hang `[M]` constrains cache
   layout and must be designed around, not discovered mid-run.

8. **Is deterministic top-`B` the actual mistake?** Two independent 2026 lines — VaSE `[C]`
   [2606.03928](https://arxiv.org/abs/2606.03928) and Nexus Sampling `[C]`
   [2606.23961](https://arxiv.org/abs/2606.23961) — replace deterministic ranking with
   probabilistic retention. If a randomised policy dominates its deterministic parent at
   matched budget across several scorers, that is a statement about the *form* of the
   decision rather than the score, and it would be the first thing in this field that
   generalises across policies.

**Six disputes that should stay disputes** (from the mirrored note, unchanged): eviction vs
retention `[C]` [2603.20397](https://arxiv.org/abs/2603.20397) finds no method dominates;
attention-score vs attention-free scoring, where cheapness and robustness are in direct
tension `[C]` [2606.26472](https://arxiv.org/abs/2606.26472) vs `[C]`
[2510.00231](https://arxiv.org/abs/2510.00231); whether non-uniform budget allocation is real
at all (per-layer `[C]` [2406.02069](https://arxiv.org/abs/2406.02069) vs per-head `[C]`
[2407.11550](https://arxiv.org/abs/2407.11550) vs `[C]`
[2509.09754](https://arxiv.org/abs/2509.09754)); whether importance persists `[C]`
[2305.17118](https://arxiv.org/abs/2305.17118) vs `[C]`
[2506.15969](https://arxiv.org/abs/2506.15969); whether benchmark choice decides the winner
`[C]` [2412.10319](https://arxiv.org/abs/2412.10319), `[C]`
[2510.13334](https://arxiv.org/abs/2510.13334); and whether training-free is even the right
constraint `[C]` [2502.11089](https://arxiv.org/abs/2502.11089).

---

## 10. Answers to the self-check

**1.** Set the evicted token's value vector far from the retained average. Concretely:
`a_4096 = 0.0001`, `v_4096 = −10000·e₁`, all other values ≈ `e₁`. Then
`Σ_{i∉S} a_i(v_i − o_S) ≈ 0.0001 × (−10001) ≈ −1.0`, against `‖o_t‖ ≈ 1`. You exploited the
**second** factor, `max‖v_i − o_S‖`, which no attention-score policy examines. This is not a
contrived corner: `[C]` [2606.03928](https://arxiv.org/abs/2606.03928) reports that real
models carry a small fraction of value states with abnormally large magnitudes, and that
evicting them causes catastrophic repetitive-loop failures.

**2.** A hit rate requires a miss to be an observable event. Here a "miss" is the model
attending to a token that no longer exists, which is unrepresentable inside the attention
kernel — the softmax simply runs over the surviving keys and produces a well-formed answer —
so there is no event to count, and the only way to know you were wrong is to also run the
uncompressed model, at which point you have not saved anything. The one thing you could keep
is **the keys** (or a lossy summary of them: page-level min/max bounds, a subset of channels,
a low-bit quantization), because `q_t · k_i` is what a "would have hit" test needs. That
costs you roughly half the cache and turns the policy into sparse attention — which is
exactly what Quest `[C]` [2406.10774](https://arxiv.org/abs/2406.10774) and SparQ `[C]`
[2312.04985](https://arxiv.org/abs/2312.04985) are, and why they relieve bandwidth but not
capacity.

**3.** The distinction is between a **bound** and an **instrument**. As a bound, `(1−A) ·
max‖v_i − o_S‖` is correct — 0 violations in 6,144 synthetic samples `[M]` — and the note's
argument that the field optimises one factor and ignores the other is right and important.
As an instrument for attribution, the pair has near-zero resolving power at a fixed budget,
because `A` barely varies once `B` is fixed and the max-based bound is loose by 8–75×. The
note names the right *problem* and the wrong *quantities*. The repair keeps the note's
structure and swaps the scalars: log `‖u‖`, `(1−A)‖o_S‖`, `cos∠(u, o_S)`, which reconstruct
the error exactly. Caveat carried forward: the measurement is synthetic, and Exercise C step
3 is its test on real attention with an explicit KILL.

**4.** On structureless Gaussian data with no importance signal to detect at all,
`l2-keep-low` beats random by 10% at a 5% budget `[M]` — so a 9% margin is inside the null
and demonstrates nothing about keys carrying importance. The mechanism is that keeping
low-norm keys retains a *flatter* weight distribution (effective retained count 28.4 versus
random's 22.8 at 5%), which makes `o_S` a better-conditioned average and shrinks the
renormalisation kickback — an artifact of §3.2's `1/A` amplification, not a property of
trained networks. The correct ask is a null run at matched shapes, and a margin that clears
it.

**5.** Capacity binds at `62 GiB / 48 KiB = 1,354,411` tokens. Bandwidth at 10 tok/s binds
at `(199.9 GB/s ÷ 10) / 48 KiB = 406,698` tokens — **3.3× earlier**. Since capacity is not
the binding constraint at any context length we will train at, "make long context fit" is
solving the problem we do not have. The baseline to beat is therefore **retention** —
query-aware sparse read at matched bytes-read, which destroys nothing and re-decides every
step — and an eviction policy has to beat that on quality per byte read, not on bytes held.
Report both numbers for any compression claim, always (§2.4).

**6.** `h_i = Σ_{τ≤t} a_{τ,i}` is a sum over *opportunities*, and token 1 has had `t`
opportunities while token `t−1` has had one. Under a uniform-attention null where every
token is equally important by construction, `h_i ≈ ln(t/(i−1))`, which at `t = 4096` gives
`h_1/h_4000 = 371×` `[M]`. Recent tokens can therefore never accumulate enough score to
compete, no matter how important they are, so the recent window is not a locality heuristic —
it is a floor that prevents the accumulator's age bias from evicting everything new. The
one-line change that removes the need: normalise by the number of opportunities,
`h_i / (t − i + 1)`, turning unnormalised LFU into mean-rate LFU. That it is not what the
paper does is worth noticing, and whether it helps is a cheap arm.

---

## 11. Sources

**Local measurements and artifacts (`[M]`)**

- Exercise A / B probe, this machine, 2026-07-26: torch `2.12.0a0+rocm7.13.0a20260313`,
  **CPU**, fp32, seed 1337, `H=8`, `T=1024`, `d=64`, 64 contexts, budgets
  `{1,2,5,10,25,50}%`, 3,072 samples per arm; byte-identical output on two consecutive runs
  of the final script. Produces: the identity residual (2.196e-06 / 2.976e-06), zero bound
  violations, the within-budget R² table of §3.6, the policy-null table of Exercise B, and
  the H2O harmonic-null ladder (`h_1/h_4000 = 371.2`). Synthetic Gaussian data — a
  calibration result, not a statement about trained models.
- `ASSUMPTIONS.md` rows: `gpu-fast-tier-size` (≥62 GiB flat at ~200 GB/s, single run per
  arm), `large-tensor-fault-32gib` (≥32 GiB single tensors hang at 0% CPU or fault),
  `sdpa-is-memory-efficient` (147.2 → 6.6 bytes/T² with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), `kv-per-token-laguna` (192 KiB/token exact),
  `reference-model` (48 layers, 12 full + 36 sliding, strict GSSS, `w=512`),
  `bf16-numerics-unproven`, `single-device-only`, `ablation-scale-sufficient`, `torch-build`.
- `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier bandwidth sweep.
- `research/memory/kv-compression-and-eviction.md` — the note this module mirrors. §3.6 above
  proposes a refinement to its §6 instrumentation recommendation; nothing else here
  contradicts it.
- `research/memory/kv-cache-mechanics.md`, `research/synthesis.md` — the KV product and the
  lab's build/park decisions.
- `research/reference/CODE_MAP.md` and `PROVENANCE.md` — every `file:line` in §6 was opened
  and the named symbol confirmed on the named line on 2026-07-26.

**arXiv (`[C]`)** — every id below appears in
`research/memory/kv-compression-and-eviction.md`'s source list, resolved against the live
arXiv API on 2026-07-26, except `2606.23961`, verified by fetching its abstract page this
session.

- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (Nov 2019). Decode is bandwidth-bound.
- `2305.17118` — *Scissorhands* (May 2023). The persistence-of-importance hypothesis.
- `2306.14048` — *H2O: Heavy-Hitter Oracle* (Jun 2023). The accumulator; submodular framing.
- `2309.17453` — *Efficient Streaming Language Models with Attention Sinks* (Sep 2023).
- `2310.01801` — *FastGen: Adaptive KV Cache Compression* (Oct 2023). Per-head profiles.
- `2312.04985` — *SparQ Attention: Bandwidth-Efficient LLM Inference* (Dec 2023).
- `2401.06104` — *Transformers are Multi-State RNNs (TOVA)* (Jan 2024).
- `2404.14469` — *SnapKV: LLM Knows What You are Looking for Before Generation* (Apr 2024).
- `2406.02069` — *PyramidKV* (Jun 2024). Per-layer budget allocation.
- `2406.10774` — *Quest: Query-Aware Sparsity* (Jun 2024). Page-level key summaries.
- `2406.11430` — *A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression* (Jun 2024).
- `2407.11550` — *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation* (Jul 2024). The eviction-loss upper bound between pre- and post-eviction attention output.
- `2410.10781` — *When Attention Sink Emerges in Language Models* (Oct 2024).
- `2412.10319` — *SCBench: A KV Cache-Centric Analysis of Long-Context Methods* (Dec 2024).
- `2502.00299` — *ChunkKV* (Feb 2025). Chunk granularity and layer-wise index reuse.
- `2502.01068` — *FastKV* (Feb 2025). Token-Selective Propagation; decoupling.
- `2502.11089` — *Native Sparse Attention* (Feb 2025). Sparsity trained in.
- `2502.14051` — *RocketKV* (Feb 2025). Two stages; 400× vs 32.6%.
- `2504.15364` — *KeyDiff* (Apr 2025). Geometric distinctiveness.
- `2505.23416` — *KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction* (May 2025).
- `2506.15969` — *LazyEviction* (Jun 2025). Token Importance Recurrence.
- `2509.09754` — *LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation* (Sep 2025).
- `2510.00231` — *The Pitfalls of KV Cache Compression* (Sep 2025, rev. May 2026; ACL 2026). Silent instruction dropping.
- `2510.13334` — *Taming the Fragility of KV Cache Eviction (DefensiveKV)* (Oct 2025). Worst-case aggregation.
- `2603.20397` — *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference* (Mar 2026). No method dominates.
- `2606.03928` — *Value-Aware Stochastic KV Cache Eviction (VaSE)* (Jun 2026). Value-magnitude outliers; repetitive-loop failure.
- `2606.23961` — *Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction Under Fixed Budgets* (22 Jun 2026). Probabilistic retention; within 1% of dense at 80% reduction.
- `2606.26472` — *Epiphany-Aware KV Cache Eviction Without the Attention Matrix (EpiKV)* (Jun 2026). The fused-kernel incompatibility, stated.
- `2607.05061` — *KVpop: Key-Value Cache Compression with Predictive Online Pruning* (Jul 2026). The delayed scorer.
