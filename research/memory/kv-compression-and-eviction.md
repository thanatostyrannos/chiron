---
title: KV compression and eviction — what each policy assumes about token importance, and where that assumption breaks
version: 1.0.0
status: living
date: 2026-07-26
class: documentation (mutable — obligated to stay accurate)
---

# KV compression and eviction

**Orientation.** This note settles what the ten canonical KV-reduction policies actually
*assume* about which tokens matter, reduces every one of those assumptions to a single
piece of algebra that says exactly when it fails, and shows that nine of the ten share
one structural defect: they commit to a retained subset before the query that will read
it exists. It settles that the eviction-vs-retention argument is a capacity-vs-bandwidth
argument in disguise, and that on our hardware — 62 GiB fast tier, ~200 GB/s `[M]` — the
two constraints bind at different context lengths, so the choice is measurable rather
than doctrinal. It does not settle which policy wins; the 2026 literature is explicit
that none dominates `[C]` [2603.20397](https://arxiv.org/abs/2603.20397), and this note
presents six live disputes as disputes.

---

## 1. The arithmetic, written out

One attention head, one decode step *t*. Symbols, all of them:

| Symbol | Is | Shape / value |
|---|---|---|
| `d` | head dimension — the width of one head's vectors | scalar; 128 in our reference model `[M]` (ASSUMPTIONS: `kv-per-token-laguna`) |
| `q_t` | the **query** vector the model emits at step *t*: "what am I looking for right now" | vector of length `d` |
| `k_i` | the **key** vector cached for past token *i*: "what I am, as something to be looked up" | vector of length `d` |
| `v_i` | the **value** vector cached for past token *i*: "what gets returned if you look me up" | vector of length `d` |
| `s_i` | raw score of past token *i* against the current query | scalar |
| `a_i` | normalised attention **weight** on past token *i* | scalar in `[0,1]`, and `Σ_i a_i = 1` |
| `o_t` | the head's output at step *t* | vector of length `d` |

```
s_i = (q_t · k_i) / sqrt(d)              dot product, scaled so variance stays ~1 as d grows
a_i = exp(s_i) / Σ_j exp(s_j)            softmax: exponentiate, then divide by the total
o_t = Σ_i a_i · v_i                      the output is a weighted average of the values
```

`q_t · k_i` is `Σ_{c=1..d} q_t[c] · k_i[c]` — sum of elementwise products. Equivalently
`‖q_t‖·‖k_i‖·cos θ`: magnitude times magnitude times alignment. Hold that decomposition;
two of the policies below exploit only the magnitude half.

**Every policy in this note does the same thing:** choose a subset `S` of past positions
with `|S| = B` (the *budget*), throw the rest away, and run the softmax over `S` alone.

### The exact error identity — the spine of the whole note

Let `A = Σ_{i∈S} a_i` be the **retained attention mass**: the fraction of the softmax
weight that survives eviction. The compressed output renormalises over what is left:

```
o_t^S = (1/A) · Σ_{i∈S} a_i · v_i
```

Then, by pure algebra (substitute `o_t = A·o_t^S + Σ_{i∉S} a_i v_i` and collect):

```
o_t − o_t^S  =  Σ_{i∉S} a_i · (v_i − o_t^S)                              [exact]

‖o_t − o_t^S‖  ≤  (1 − A) · max_{i∉S} ‖v_i − o_t^S‖                      [triangle ineq.]
```

This is derived here, not cited — reproduce it in four lines. Two things fall out, and
they organise everything that follows.

**First: eviction reweights what remains.** Dropping a cache line in a buffer cache does
not change the value of the other lines. Dropping a KV entry shrinks the softmax
denominator, so *every retained token's weight goes up*. There is no analogue anywhere in
storage. This is why "we only dropped 5% of the attention mass" is not the reassurance it
sounds like.

**Second: the error is a product of two independent factors** — the dropped mass `(1−A)`
and how far the dropped *values* sit from the retained average. **Every attention-score
policy below optimises the first factor and ignores the second entirely.** That gap is
not theoretical: `[C]` [2606.03928](https://arxiv.org/abs/2606.03928) (Jun 2026) reports
that a small fraction of value states carry abnormally large magnitudes — large
`‖v_i − o‖` — and that evicting them causes catastrophic failure in which the model enters
repetitive reasoning loops.

### Why this is a bandwidth problem before it is a capacity problem

Decode is memory-bandwidth-bound, not FLOPS-bound `[C]`
[1911.02150](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) — the single mental model
to hold. You re-read the *entire* KV cache once per generated token. Our reference model
costs 2 · 48 layers · 8 KV heads · 128 dims · 2 bytes = **192 KiB per token**, exactly
`[M]` (ASSUMPTIONS: `kv-per-token-laguna`; the 24 GiB at 128k below is the all-global
bound — real residency is ~4× lower because 36/48 layers are windowed at 512).
At 128k context that is 24 GiB. Against our measured ~200 GB/s `[M]`
(`notebook/uma-carveout-controls-fast-tier.md`), reading it once takes ~0.13 s, i.e.
~7.8 tokens/second — arithmetic from two measured inputs, not a benchmark. For a storage
engineer: it is a full table scan per emitted row.

---

## 2. Where the caching analogy pays, and the four places it breaks

It pays a lot. Budget is cache size, eviction policy is replacement policy, the attention
sink is a pinned page, ChunkKV is a cache line, RocketKV is a two-level index. Use all of
it. Then internalise the four breaks, because they are where the intuition costs you.

**Break 1 — there is no miss path and no backing store.** Every cache you have operated
makes eviction a bet on *latency*: get it wrong, pay a refill. Here eviction is a bet on
*correctness*. The evicted KV entry is the only existing encoding of that token in this
context; reconstructing it means re-running prefill over the whole prefix. There is no
"present bit," no fault, and — this is the operationally important part — **no hit-rate
metric**. A wrong eviction produces a fluent, confident, wrong answer with no error
signal anywhere in the stack. Contrast vLLM's prefix cache, where a miss is a legible,
costed recompute (`memory/vllm/vllm/v1/core/block_pool.py:679`, `_maybe_evict_cached_block`
— eviction happens lazily at reallocation and a zero-refcount block is still matchable;
see `research/reference/CODE_MAP.md`).

**Break 2 — the working set is defined by a request that has not arrived.** A buffer
cache's frequency counters measure the same access stream that will continue; stationarity
is a defensible prior. Here, prefill compresses the *context* using statistics from the
prompt attending to itself, and the thing that later reads the cache is the *answer*
attending to the prompt. Different distributions, and adversarially so when the question
is unrelated to the document's surface statistics.

**Break 3 — some entries are load-bearing for reasons unrelated to their content.**
Attention sinks (§3.1) are pinned not because they are hot but because the softmax must
sum to 1 and heads that want to attend to nothing need somewhere to dump the mass. You
cannot discover that pin by profiling access frequency; it is an artifact of the
arithmetic.

**Break 4 — granularity is not free of semantics.** A cache line is a pure performance
construct: false sharing costs throughput. A KV "line" (ChunkKV, §3.5) costs *accuracy*,
because the tokens dragged in or dropped together carry meaning.

---

## 3. The policies, one at a time

Format for each: **assumes** → **mechanism** → **evidence** → **breaks**.

### 3.1 StreamingLLM — importance is positional `[C]` [2309.17453](https://arxiv.org/abs/2309.17453) (Sep 2023)

**Assumes** importance is a function of position alone: a fixed prefix plus a recent
window is all you ever need. **Mechanism:** keep the first few tokens (four, in the
paper's ablation) plus the most recent `L`; discard the middle permanently. The paper's
own finding is that plain window attention *fails* once text exceeds the cache, and that
retaining initial tokens "will largely recover the performance," because they act as an
attention **sink** — attracting strong scores "even if they are not semantically
important" `[C]` (verbatim). `[C]`
[2410.10781](https://arxiv.org/abs/2410.10781) (Oct 2024) gives the empirical account of
when sinks emerge as a function of optimisation, data, and architecture.
**Evidence** `[C]`: stable language modelling to 4M tokens; up to 22.2x speedup over
sliding-window recomputation.

**Breaks:** the paper is explicit that this does **not** extend effective context — it
buys stable streaming, not recall. Anything the query needs from the discarded middle is
unrecoverable. And `[C]` [2510.00231](https://arxiv.org/abs/2510.00231) (rev. May 2026,
ACL 2026) finds StreamingLLM among five methods under which specific instructions in a
multi-instruction prompt are **completely ignored**, with system-prompt leakage as the
worked case. Systems reading: a ring buffer with a pinned header, where the header is
pinned to satisfy an allocator invariant rather than because it is hot.

### 3.2 H2O — importance persists `[C]` [2306.14048](https://arxiv.org/abs/2306.14048) (Jun 2023)

**Assumes** the *persistence of importance*: accumulated past attention predicts future
attention. **Mechanism:** score token *i* at step *t* by `h_i = Σ_{τ≤t} a_{τ,i}` — the
running sum of attention weight token *i* has received — keep the top-`B` "heavy hitters"
plus a recent window. Formulated as dynamic submodular maximisation with a guarantee
under mild assumptions `[C]`. **Evidence** `[C]`: 20% heavy hitters; up to 29x throughput
over DeepSpeed Zero-Inference and HF Accelerate and 3x over FlexGen on OPT-6.7B/30B; up to
1.9x lower latency at equal batch size.

**Breaks, four of them, precisely:**

1. **The accumulator is structurally biased toward old tokens.** `h_i` is a *sum* over
   opportunities, and token 1 has had `t` opportunities while token `t−1` has had one.
   This is arithmetic, not a hypothesis: unnormalised LFU where age is a head start.
2. **Persistence is empirically contested.** `[C]`
   [2506.15969](https://arxiv.org/abs/2506.15969) (LazyEviction, ACL 2026) documents
   *Token Importance Recurrence* — a large proportion of tokens regain high attention
   after many decoding steps — which is precisely the negation of Scissorhands' persistence
   hypothesis `[C]` [2305.17118](https://arxiv.org/abs/2305.17118).
3. **It optimises only `(1−A)`.** See §1. `[C]`
   [2606.03928](https://arxiv.org/abs/2606.03928) reports H2O significantly lagging the
   full cache on nearly every reasoning dataset tested, producing traces that occasionally
   fail to terminate.
4. **It needs the attention matrix.** `[C]`
   [2606.26472](https://arxiv.org/abs/2606.26472) (Jun 2026) states that ranking by
   attention weight "prohibits the use of fused kernels in production inference by forcing
   the model to materialize the attention matrix" — i.e. H2O is incompatible with the
   FlashAttention-class kernels every serving stack actually runs.

### 3.3 SnapKV — the end of the prompt proxies the query `[C]` [2404.14469](https://arxiv.org/abs/2404.14469) (Apr 2024)

**Assumes** the last few query positions of the prompt predict the queries generation will
issue — the title says it: "LLM Knows What You are Looking for Before Generation."
**Mechanism:** at the end of prefill, take an observation window of the last `w` query
rows (`w ≈ 32` in the default configuration), sum their attention to all prior keys, apply
a 1-D pooling over the position axis so that selected positions arrive in contiguous runs
rather than shredded singletons, and keep the top-`B` per head. **Evidence** `[C]`:
3.6x generation speed and 8.2x memory efficiency at 16K input; up to 380K context on one
A100-80GB; comparable across 16 long-sequence datasets.

**Breaks:** SnapKV does not remove the prematurity flaw — it *narrows the window of
prematurity* from "before the prompt" to "before generation," which is why it works so
much better than H2O on benchmarks whose question sits at the end of the prompt. Move the
question to the front, or issue a *second, different* question against the same compressed
cache, and the proxy is measuring boilerplate. Multi-turn and shared-context reuse is
exactly what `[C]` [2412.10319](https://arxiv.org/abs/2412.10319) (SCBench) was built to
expose. It is also prefill-only: tokens generated afterward are never re-scored, and `[C]`
[2606.03928](https://arxiv.org/abs/2606.03928) notes SnapKV cannot be used as a
decoding-phase evictor at all, because its window statistics require real-time window
construction. And it is on the `[C]` [2510.00231](https://arxiv.org/abs/2510.00231)
instruction-dropping list. Systems reading: readahead keyed on the last few accesses —
sound when the tail is representative of the future stream, and the tail of a chat prompt
usually is not.

### 3.4 PyramidKV — the *budget* should vary with depth `[C]` [2406.02069](https://arxiv.org/abs/2406.02069) (Jun 2024)

**Assumes** "Pyramidal Information Funneling": attention is scattered in lower layers,
consolidates in the middle, and concentrates on sinks/massive activations in higher layers
— so lower layers deserve more cache and higher layers less. **Mechanism:** non-uniform
per-layer budget allocation; token selection *within* a layer remains SnapKV's.
**Evidence** `[C]`: matches full-cache performance at 12% retention on LongBench; at 0.7%
retention beats other compressors by up to 20.5 absolute accuracy points on TREC;
128 KV entries gives LLaMA-3-70B 100.0 on needle-in-a-haystack.

**Breaks:** (a) it inherits every SnapKV failure and adds an allocation rule on top — and
by the paper's own account it *converges to SnapKV* as the compression ratio rises, which
means the allocation is second-order exactly where the pressure is highest. (b) Whether
non-uniform allocation is real at all is contested: Ada-KV allocates per **head** `[C]`
[2407.11550](https://arxiv.org/abs/2407.11550), LAVa gives a different per-layer rule `[C]`
[2509.09754](https://arxiv.org/abs/2509.09754), and several groups attribute most of the
reported gain to the observation window rather than the allocation. (c) The pyramid is a
measured property of particular trained networks, not a law. `[A]` medium confidence the
profile differs on a hybrid SWA/global stack like our reference model (12 full + 36
sliding, strict GSSS `[M]`); cheapest test is per-layer attention entropy split by layer
type on our own models — see Open questions. Systems reading: per-tier capacity
allocation. Break: in a storage hierarchy you measure per-tier miss rate and reallocate
online; here the miss rate is unobservable and the split comes from an offline profile.

### 3.5 ChunkKV — importance is a property of chunks, not tokens `[C]` [2502.00299](https://arxiv.org/abs/2502.00299) (Feb 2025)

**Assumes** token-granular selection shreds phrases and produces "fragmented context";
importance lives in semantic chunks. **Mechanism:** partition into chunks, score chunks,
retain whole chunks. Plus **layer-wise index reuse** — because retained indices are highly
similar across adjacent layers, compute the selection once and reuse it downstream.
**Evidence** `[C]`: +26.5% throughput from index reuse; up to 8.7% precision over prior
SOTA at matched ratio, on LongBench, NIAH, GSM8K and JailbreakV; KV cache stated as up to
70% of inference memory.

**Breaks:** this is the closest thing in the literature to a systems idea done right —
exploit spatial locality, raise granularity, amortise metadata — and it breaks in exactly
the classic way. Fixed-length chunks are **cache lines, and false sharing is now an
accuracy bug in both directions**: one high-scoring token drags in its neighbours and
consumes budget, and one chunk containing the single needed fact but otherwise dull gets
dropped wholesale. Second, index reuse across layers is in tension with PyramidKV's core
observation: ChunkKV asserts the retained *set* is stable with depth while PyramidKV
asserts attention *changes qualitatively* with depth. Both can be true — set stability and
concentration change are different quantities — but nobody has measured them in one
harness. `[A]` medium confidence they conflict at aggressive budgets; cheap Jaccard-overlap
test in Open questions. Third, chunk scores still come from attention, so the prematurity
flaw and the fused-kernel incompatibility both carry over.

### 3.6 L2-norm strategies — importance is intrinsic to the key `[C]` [2406.11430](https://arxiv.org/abs/2406.11430) (Jun 2024)

This one states the thesis of the whole note out loud, so quote it: the finding "indicates
that the influence of a KV pair is potentially determined by the key embedding itself
before being queried" `[C]` (verbatim).

**Assumes** exactly that. **Mechanism:** compute `‖k_i‖₂ = sqrt(Σ_{c=1..d} k_i[c]²)` — the
Euclidean length of the key vector, no query involved — and keep the *low*-norm keys.
Recall `s_i = ‖q‖·‖k_i‖·cos θ / sqrt(d)`: a large norm can only produce a large score if
the angle cooperates, and empirically in trained decoders it does not. **Evidence** `[C]`:
50% KV reduction on language modelling and NIAH, 90% on passkey retrieval, without accuracy
loss; and — the reason anyone deploys it — it needs no attention scores, so it is
FlashAttention-compatible.

**Breaks:** (a) the correlation is an unexplained empirical regularity of specific trained
networks, not a property of attention. `[A]` high confidence it is not guaranteed at
20M–300M params; cheapest test is a per-layer rank correlation on our own model, and a
null result is itself a finding (see Open questions). (b) Cheapness is bought with
robustness: `[C]` [2510.00231](https://arxiv.org/abs/2510.00231) finds K-Norm among the
methods that cause instructions to be silently dropped. (c) It is *maximally*
query-agnostic, which is both the feature (O(d) per token, computable at write time,
streaming, no attention matrix) and the defect (it cannot possibly know which of two
equally low-norm tokens the question needs). Systems reading: admission control from object
metadata before a single request has been seen. Works when metadata correlates with
popularity; fails on precisely the requests that deviate from population statistics — and
those are the requests users care about.

### 3.7 KeyDiff — importance is geometric distinctiveness `[C]` [2504.15364](https://arxiv.org/abs/2504.15364) (Apr 2025)

**Assumes**, verbatim, that "geometrically distinctive keys during LLM inference tend to
have high attention scores." **Mechanism:** score each key by cosine *distance* from an
anchor direction `c` (the mean key direction): `score_i = 1 − (k_i·c)/(‖k_i‖·‖c‖)`. Keep
the most distinctive. Attention-free, so FlashAttention-compatible. **Evidence** `[C]`:
under 0.04% performance gap at an 8K cache budget (~23% KV reduction) on LongBench with
Llama 3.1-8B and 3.2-3B; near-baseline on Math500 with DeepSeek-R1-Distill-Llama-8B; up to
30% lower end-to-end latency.

**Breaks:** distinctive is not the same as needed, and the failure is a clean inversion.
A name repeated twenty times through a document is by construction *not* distinctive and
gets evicted early; a typo, a stray identifier, or a formatting artifact is maximally
distinctive and gets pinned. `[A]` medium-high confidence this is a real and measurable
inversion; cheapest test is a needle task where the needle is a *repeated* phrase versus a
*unique* phrase at matched budget. Note also that ~23% KV reduction is a mild budget
compared to the 90%+ regimes elsewhere in this note, so the sub-0.04% gap is a
low-compression number and should not be read against PyramidKV's 0.7%. Systems reading:
dissimilarity-based sampling / dedup. Break: dedup is safe because the canonical copy
survives behind a pointer; here the "duplicate" you drop is gone, and near-duplicate keys
carry *different values*.

### 3.8 FastKV — importance stabilises with depth `[C]` [2502.01068](https://arxiv.org/abs/2502.01068) (Feb 2025)

**Assumes** token importance stabilises in later layers, so full-context compute is only
needed up to some layer. **Mechanism:** run full context to a **Token-Selective Propagation
(TSP)** layer, then forward only the most informative tokens onward; from those, select
KV entries to cache independently. The actual contribution is the **decoupling**, stated
verbatim: prior prefill accelerators "inadvertently tie the prefill compute reduction to
the decoding KV budget," and FastKV makes TSP rate and KV retention rate two independent
knobs. **Evidence** `[C]`: up to 1.82x prefill and 2.87x decode speedup versus full
context, matching decoding-only baselines on accuracy.

**Breaks:** (a) the stabilisation depth is a per-model hyperparameter of the same family
as PyramidKV's pyramid, fit to particular checkpoints. (b) **Irreversibility moves out of
the cache and into the compute graph.** A token not propagated past layer `ℓ` never has
its KV computed for layers `ℓ+1..L`; those bytes were not evicted, they were never
produced. This has a direct consequence for Mnemosyne: FastKV *cannot* be expressed as a
policy sitting behind a cache interface — it changes the forward pass, and any interface
that claims to host it is lying about its boundary. (c) TSP runs during prefill, so the
prematurity flaw is intact. Systems reading: predicate pushdown in a query plan. Break: a
pushed-down predicate is *provably* safe — the discarded rows could not have affected the
result. TSP is a guess, and there is no way to check it after the fact.

### 3.9 RocketKV — eviction and sparse attention are complementary `[C]` [2502.14051](https://arxiv.org/abs/2502.14051) (Feb 2025)

**Assumes** the two camps are composable rather than rival. **Mechanism:** stage 1 does
coarse **permanent** eviction over input tokens (a strengthened SnapKV); stage 2 does
fine-grained top-`k` sparse attention *at read time*, approximating attention scores via
both head-dimension and sequence-dimension reduction. **Evidence** `[C]`: up to 400x
compression ratio, up to 3.7x end-to-end speedup, up to 32.6% peak decode memory reduction,
negligible long-context accuracy loss.

**Breaks, and this is the most instructive entry in the list:** stage 2 is query-*aware*
and stage 1 is not, so the architecture is a tacit admission that the query-blind stage
must stay conservative and the real value comes from **deferring** the fine decision until
the query exists. That is the interface lesson (§5). But stage 1 is still irreversible —
whatever it drops, stage 2 can never select, so the compression ratios multiply while the
risk is inherited wholesale from the query-blind half. Read the two headline numbers
together: **400x "compression" against 32.6% peak memory reduction.** They are an order of
magnitude apart because most of the 400x is *bandwidth* (how much you read per step), not
*capacity* (how many bytes you hold). That single pair of numbers is the entire
eviction-versus-retention debate, visible inside one paper. Systems reading: a coarse
filter over a fine index. Break: a Bloom filter has one-sided error — no false negatives.
Stage-1 eviction produces false negatives, and nothing downstream can detect them.

### 3.10 Inference-time sparse attention — discard nothing, read less

The other camp: keep the whole cache and select what to *read* per step, so importance is
re-estimated against the actual `q_t` every time. Query prematurity dissolves; capacity
relief disappears. Representative work: Quest `[C]`
[2406.10774](https://arxiv.org/abs/2406.10774), SparQ `[C]`
[2312.04985](https://arxiv.org/abs/2312.04985), MInference's three per-head sparse patterns
for prefill `[C]` [2407.02490](https://arxiv.org/abs/2407.02490), Multipole Attention `[C]`
[2506.13059](https://arxiv.org/abs/2506.13059), and the trained variants — NSA `[C]`
[2502.11089](https://arxiv.org/abs/2502.11089) and DeepSeek Sparse Attention in V3.2 `[C]`
[2512.02556](https://arxiv.org/abs/2512.02556), extended by V4's Compressed/Heavily
Compressed Attention `[C]` [2606.19348](https://arxiv.org/abs/2606.19348). UNIQUE `[C]`
[2605.27740](https://arxiv.org/abs/2605.27740) (May 2026) pushes a universal top-`k`
formulation for both training-free inference and sparsity-aware training.

The best current evaluation is *The Sparse Frontier* `[C]`
[2504.17768](https://arxiv.org/abs/2504.17768), and its findings are load-bearing:
at matched isoFLOPS "larger sparse models outperform smaller dense ones," improving the
Pareto frontier; fine-grained sparsity in **prefill** "remains impractical — due to both
the cost of estimation and the lack of sparse kernels," while in **decode** token-to-page
selection is feasible with higher sparsity tolerance; the right selection unit
(global-to-token vs block-to-block) is **task-dependent**, so no single configuration is
universal; and "longer sequences tolerate higher sparsity," evaluated to sparsity 0.95 at
up to 128K.

**Breaks:** it does not bound memory. On our machine the fast tier is **62 GiB** `[M]` and
single tensors ≥32 GiB hang outright `[M]` — retention-only means the cache must fit or be
offloaded across a boundary whose cost you now own (that is the serving-hierarchy note's
territory). And selection is itself an approximation whose per-step errors compound over
long generations, which is the mechanism both camps blame for reasoning-workload failures.

---

## 4. The recurring flaw, stated precisely — and what 2026 did about it

Strip the mechanisms away and every policy in §3.1–3.9 computes

```
S = f(K, V, prompt)      chosen once, then used to serve every future query
```

whereas the quantity that matters is `S*(q)`, the best subset **for the query that
actually arrives**. That makes this a *robust*-optimisation problem — pick one subset good
against a distribution of unknown future queries — and not a ranking problem. Seen that
way, the taxonomy collapses:

- **H2O** substitutes the *past* query stream for the future one.
- **SnapKV / PyramidKV / ChunkKV / FastKV / RocketKV-stage-1** substitute a *point estimate*
  of the query (the observation window) for the distribution.
- **L2-norm / KeyDiff** drop the query from the objective entirely and rank by input
  statistics.

None of them writes down the objective. Four 2026 lines finally do:

**(a) State the query-agnostic objective correctly.** KVzip `[C]`
[2505.23416](https://arxiv.org/abs/2505.23416) (May 2025) scores a KV pair by whether the
model can *reconstruct the original context* from the retained cache — a sufficiency
criterion rather than a query proxy — reporting 3–4x KV reduction and ~2x lower
FlashAttention decode latency with negligible loss across QA, retrieval, reasoning and code
on LLaMA3.1 / Qwen2.5 / Gemma3 to 170K context. Fast KVzip `[C]`
[2601.17668](https://arxiv.org/abs/2601.17668) (Jan 2026) cuts its cost with gated
eviction. CapKV `[C]` [2604.25975](https://arxiv.org/abs/2604.25975) (Apr 2026) makes the
same move formally: under a linear-Gaussian surrogate of attention it derives a
closed-form mutual-information objective for the retained subset, shows "a wide range of
existing eviction strategies can be interpreted as different approximations of the same
capacity-maximization principle," and selects by log-determinant via statistical leverage
scores — which is exactly the tool you reach for when an unknown future query must be
served from a subspace. `[C]` [2607.08032](https://arxiv.org/abs/2607.08032) (Jul 2026)
generalises further, unifying KV eviction, prompt compression, recurrent-state bounding and
agent-memory consolidation as one rate-distortion problem and naming this precise failure:
irreversible discard before the query is known.

**(b) Learn the policy instead of hand-writing the heuristic.** `[C]`
[2602.10238](https://arxiv.org/abs/2602.10238) (Feb 2026) trains lightweight per-head RL
agents to rank tokens by predicted future usefulness — the direct analogue of learned cache
replacement. LKV `[C]` [2605.06676](https://arxiv.org/abs/2605.06676) learns head-wise
budgets and token selection end to end.

**(c) Convert prematurity into a latency knob — the one idea that transfers cleanly from
systems.** KVpop `[C]` [2607.05061](https://arxiv.org/abs/2607.05061) (Jul 2026)
supervises the keep-or-drop decision directly against a future-attention target computed
without materialising dense attention maps, and adds a **delayed** scorer that defers
scoring for a fixed number of steps to exploit near-future context, reporting 98% of
full-attention performance on Qwen3-4B at 75% compression and 97% at 88%. Deferral is a
write-back cache with a staging buffer: pay `k` steps of extra residency, buy `k` steps of
extra information. It is the cleanest available answer to "decide before the query is
known" — *decide later*.

**(d) Score without the attention matrix, but not from input statistics.** EpiKV `[C]`
[2606.26472](https://arxiv.org/abs/2606.26472) (Jun 2026) scores tokens by the **change in
the model's internal representation** they induce, read straight from the forward pass with
no attention matrix and negligible extra state — no training, no classifier, no custom
kernel, usable in unmodified FlashAttention stacks — reporting 72% on MATH-500 at a
4096-token cache, 37% on AIME-2024 at 8192, up to 2.8x speedup, and 16x longer feasible
context than attention-based scoring.

Also new in this window and worth knowing: reasoning-model eviction became the dominant
subthread, because policies tuned for long *input* degrade on long chain-of-thought
*output* — ForesightKV `[C]` [2602.03203](https://arxiv.org/abs/2602.03203), LookaheadKV
`[C]` [2603.10899](https://arxiv.org/abs/2603.10899), MomentKV `[C]`
[2606.01563](https://arxiv.org/abs/2606.01563), VaSE `[C]`
[2606.03928](https://arxiv.org/abs/2606.03928), ReasonAlloc `[C]`
[2606.11164](https://arxiv.org/abs/2606.11164), ThinKV `[C]`
[2510.01290](https://arxiv.org/abs/2510.01290). Diagnostics over leaderboards: `[C]`
[2605.08234](https://arxiv.org/abs/2605.08234) argues task accuracy alone cannot tell you
*why* a selector worked. Joint eviction+quantization under one rate-distortion budget:
RDKV `[C]` [2605.08317](https://arxiv.org/abs/2605.08317), EvicPress `[C]`
[2512.14946](https://arxiv.org/abs/2512.14946). Reuse of compressed caches across requests:
C²KV `[C]` [2607.17715](https://arxiv.org/abs/2607.17715) (Jul 2026).

---

## 5. Six live disputes — presented as contested, not resolved

1. **Eviction vs retention.** RocketKV `[C]` [2502.14051](https://arxiv.org/abs/2502.14051)
   claims composability; VaSE `[C]` [2606.03928](https://arxiv.org/abs/2606.03928) reports
   eviction methods often *underperform* selection methods that keep the full cache, while
   also beating the strongest selection method with its own eviction scheme; the Mar 2026
   survey `[C]` [2603.20397](https://arxiv.org/abs/2603.20397) scores five families against
   seven deployment scenarios and finds **no method dominates**. Capacity and bandwidth are
   different budgets; the answer is deployment-specific.
2. **Attention-score vs attention-free scoring.** Attention-free (L2, KeyDiff, EpiKV) is
   the only option under fused kernels `[C]`
   [2606.26472](https://arxiv.org/abs/2606.26472) — but `[C]`
   [2510.00231](https://arxiv.org/abs/2510.00231) finds K-Norm among the silent
   instruction-droppers. Cheapness and robustness are in tension and the field has not
   settled it.
3. **Is non-uniform budget allocation real?** PyramidKV (per layer) vs Ada-KV (per head)
   `[C]` [2407.11550](https://arxiv.org/abs/2407.11550) vs LAVa `[C]`
   [2509.09754](https://arxiv.org/abs/2509.09754) disagree on the rule, and PyramidKV
   degenerates to SnapKV at aggressive ratios.
4. **Does importance persist?** Scissorhands `[C]`
   [2305.17118](https://arxiv.org/abs/2305.17118) says yes; LazyEviction's Token Importance
   Recurrence `[C]` [2506.15969](https://arxiv.org/abs/2506.15969) says no for long CoT.
   Both may be describing different regimes rather than disagreeing.
5. **Benchmark choice decides the winner.** LongBench/NIAH rankings do not survive
   multi-turn cache reuse `[C]` [2412.10319](https://arxiv.org/abs/2412.10319),
   instruction-following stress `[C]` [2510.00231](https://arxiv.org/abs/2510.00231), or
   worst-case rather than mean aggregation `[C]`
   [2510.13334](https://arxiv.org/abs/2510.13334). Treat any single-benchmark ranking as an
   anecdote — house rule, and the literature agrees.
6. **Is training-free the right constraint?** NSA `[C]`
   [2502.11089](https://arxiv.org/abs/2502.11089) and DSA `[C]`
   [2512.02556](https://arxiv.org/abs/2512.02556) argue sparsity should be *trained in*,
   which would make the entire training-free eviction literature a local optimum imposed by
   a research convenience.

---

## 6. Consequences for Mnemosyne's interface

Three consequences follow directly and should be settled before any policy is coded.

**A single `score(keys, values, attention) -> subset` hook cannot express this field.**
It cannot host FastKV (which alters the forward pass), RocketKV stage 2 (which needs `q_t`
at read time), or KVpop's delayed scorer (which needs a staging buffer and a flush
trigger). The minimum honest surface is three plug points, distinguished by *when they run
relative to the query*:

| Plug point | Runs | Inputs available | Hosts |
|---|---|---|---|
| **write-time admission** | as each token's KV is produced | `k_i`, `v_i` only — O(d), no attention matrix | L2-norm, KeyDiff, EpiKV |
| **deferred eviction** | after a bounded staging delay | prompt-side or `k`-step-lagged attention | H2O, SnapKV, PyramidKV, ChunkKV, KVpop |
| **read-time selection** | per decode step | the actual `q_t` | Quest, SparQ, RocketKV stage 2 |

**Instrument the two error factors, because nobody reports them.** The exact identity in
§1 says degradation is `(1 − A) × max‖v_i − o‖`. Every paper reports downstream accuracy;
none reports retained attention mass `A` and dropped-value distance separately. Logging
both, per layer and per head, turns "the policy helped" into "*this* factor was the
binding one" — precisely the attribution gap this lab exists to attack, and it costs two
scalars per step.

**Treat "compression ratio" as two numbers, always.** RocketKV's 400x-vs-32.6% gap (§3.9)
is not sloppiness; it is the capacity/bandwidth distinction. Any Mnemosyne metric that
reports one number is unreportable.

---

## Open questions

Testable on our hardware: single GPU, 20M–300M params, 0.5–5B tokens, **62 GiB fast tier
at ~200 GB/s** `[M]`, single tensors **≥32 GiB hang** `[M]`, no working multi-GPU
`[C]`/`[M]`, and bf16 numerics unproven until the Hardware Validation Gate runs `[M]` —
nothing below counts as evidence until it does. All arms matched on params and tokens, ≥3
seeds, CIs reported; single-seed results labelled as anecdotes.

1. **Does the L2/attention anti-correlation exist at our scale?** Measure Spearman rank
   correlation between `−‖k_i‖₂` and mean received attention, per layer and per head, on a
   model we trained ourselves. One forward pass over a held-out set; no training beyond
   having the model. A null result kills every attention-free scorer as a testable object
   in this lab and is itself publishable-grade negative evidence.
2. **Which error factor binds?** Instrument `A` and `max‖v_i − o‖` for H2O, SnapKV, L2 and
   KeyDiff at matched budget, and regress downstream degradation on each. Nobody has
   reported this decomposition. Highest information-per-GPU-hour item on the list.
3. **Is deferral monotone, and where is the knee?** Delay the keep/drop decision by `k`
   decode steps and sweep `k`. `k` is a staging-buffer size in tokens, so this is a pure
   capacity-vs-quality curve of exactly the shape a write-back cache tuning study produces.
   If the curve is monotone with a sharp knee, "decide later" becomes a tunable rather than
   a research question.
4. **Does the pyramid survive a hybrid SWA/global stack?** Our reference architecture is
   12 full + 36 sliding in strict GSSS `[M]`. Measure per-layer attention entropy split by
   layer type. If the pyramid is an artifact of uniform-global stacks, PyramidKV's
   allocation rule is mis-specified for the model class we actually study.
5. **Chunk granularity vs false sharing.** Sweep chunk size 1→64 at matched budget on a
   needle task where the needle is (a) a unique string and (b) a repeated common phrase.
   One harness tests ChunkKV's semantic-unit claim and KeyDiff's distinctiveness assumption
   against each other.
6. **Is cross-layer index reuse compatible with depth-varying attention?** Jaccard overlap
   of retained index sets between adjacent layers, split by layer type and by budget.
   ChunkKV and PyramidKV make claims in tension; this measures the tension directly.
7. **Where does capacity stop binding and bandwidth start?** With 62 GiB and ~200 GB/s
   `[M]`, derive then *measure* the crossover context length at which eviction (capacity)
   or sparse retrieval (bandwidth) is the binding constraint for a 20M–300M model. The
   ≥32 GiB single-tensor hang `[M]` constrains cache layout and must be designed around,
   not discovered mid-run — the failure presents at 0% CPU with no error.
8. **Does eviction damage instruction-following before it damages perplexity at small
   scale?** Replicating `[C]` [2510.00231](https://arxiv.org/abs/2510.00231) in miniature
   needs an instruction-tuned 300M model, so this is the most expensive item here. Worth it
   if it lands: a cheap small-scale testbed for the failure mode that standard evaluation
   provably misses.

---

## Sources

Every arXiv id below was resolved against the live arXiv API on 2026-07-26. Where a title
has drifted across versions, the current API title is used.

**Canonical eviction / compression policies**
- `[C]` H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models — [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) (Jun 2023)
- `[C]` Efficient Streaming Language Models with Attention Sinks (StreamingLLM) — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) (Sep 2023)
- `[C]` SnapKV: LLM Knows What You are Looking for Before Generation — [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) (Apr 2024)
- `[C]` PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling — [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) (Jun 2024)
- `[C]` ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference — [arXiv:2502.00299](https://arxiv.org/abs/2502.00299) (Feb 2025)
- `[C]` KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments — [arXiv:2504.15364](https://arxiv.org/abs/2504.15364) (Apr 2025)
- `[C]` A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression — [arXiv:2406.11430](https://arxiv.org/abs/2406.11430) (Jun 2024)
- `[C]` FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration — [arXiv:2502.01068](https://arxiv.org/abs/2502.01068) (Feb 2025) — note the title changed across versions; cite by id
- `[C]` RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression — [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) (Feb 2025)
- `[C]` Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time — [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) (May 2023)
- `[C]` Transformers are Multi-State RNNs (TOVA) — [arXiv:2401.06104](https://arxiv.org/abs/2401.06104) (Jan 2024)
- `[C]` Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen) — [arXiv:2310.01801](https://arxiv.org/abs/2310.01801) (Oct 2023)
- `[C]` Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference — [arXiv:2407.11550](https://arxiv.org/abs/2407.11550) (Jul 2024)
- `[C]` LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation — [arXiv:2509.09754](https://arxiv.org/abs/2509.09754) (Sep 2025)

**Query-agnostic objectives, learned and deferred policies (2025–2026)**
- `[C]` KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction — [arXiv:2505.23416](https://arxiv.org/abs/2505.23416) (May 2025)
- `[C]` Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction — [arXiv:2601.17668](https://arxiv.org/abs/2601.17668) (Jan 2026)
- `[C]` Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective (CapKV) — [arXiv:2604.25975](https://arxiv.org/abs/2604.25975) (Apr 2026)
- `[C]` Learning to Evict from Key-Value Cache — [arXiv:2602.10238](https://arxiv.org/abs/2602.10238) (Feb 2026)
- `[C]` LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction — [arXiv:2605.06676](https://arxiv.org/abs/2605.06676) (Apr 2026)
- `[C]` KVpop — Key-Value Cache Compression with Predictive Online Pruning — [arXiv:2607.05061](https://arxiv.org/abs/2607.05061) (Jul 2026)
- `[C]` Epiphany-Aware KV Cache Eviction Without the Attention Matrix (EpiKV) — [arXiv:2606.26472](https://arxiv.org/abs/2606.26472) (Jun 2026)
- `[C]` Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction — [arXiv:2509.10798](https://arxiv.org/abs/2509.10798) (Sep 2025)
- `[C]` Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query — [arXiv:2505.20334](https://arxiv.org/abs/2505.20334) (May 2025)
- `[C]` Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction — [arXiv:2605.09649](https://arxiv.org/abs/2605.09649) (May 2026)
- `[C]` In-context KV-Cache Eviction for LLMs via Attention-Gate — [arXiv:2410.12876](https://arxiv.org/abs/2410.12876) (Oct 2024)
- `[C]` C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference — [arXiv:2607.17715](https://arxiv.org/abs/2607.17715) (Jul 2026)

**Reasoning-model eviction (the 2026 subthread)**
- `[C]` LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning — [arXiv:2506.15969](https://arxiv.org/abs/2506.15969) (Jun 2025)
- `[C]` ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution — [arXiv:2602.03203](https://arxiv.org/abs/2602.03203) (Feb 2026)
- `[C]` LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation — [arXiv:2603.10899](https://arxiv.org/abs/2603.10899) (Mar 2026)
- `[C]` MomentKV: Closing the Directional Gap in KV Cache Eviction for Long-Context Inference — [arXiv:2606.01563](https://arxiv.org/abs/2606.01563) (Jun 2026)
- `[C]` Value-Aware Stochastic KV Cache Eviction for Reasoning Models (VaSE) — [arXiv:2606.03928](https://arxiv.org/abs/2606.03928) (Jun 2026)
- `[C]` ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models — [arXiv:2606.11164](https://arxiv.org/abs/2606.11164) (Jun 2026)
- `[C]` ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models — [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) (Oct 2025)

**Sparse attention (the retention camp)**
- `[C]` Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference — [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) (Jun 2024)
- `[C]` SparQ Attention: Bandwidth-Efficient LLM Inference — [arXiv:2312.04985](https://arxiv.org/abs/2312.04985) (Dec 2023)
- `[C]` MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention — [arXiv:2407.02490](https://arxiv.org/abs/2407.02490) (Jul 2024)
- `[C]` Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) (Feb 2025)
- `[C]` The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs — [arXiv:2504.17768](https://arxiv.org/abs/2504.17768) (Apr 2025)
- `[C]` Multipole Attention for Efficient Long Context Reasoning — [arXiv:2506.13059](https://arxiv.org/abs/2506.13059) (Jun 2025)
- `[C]` UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training — [arXiv:2605.27740](https://arxiv.org/abs/2605.27740) (May 2026)
- `[C]` DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models — [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) (Dec 2025)
- `[C]` DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence — [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) (2026)

**Evaluation, diagnostics, unification, and the bandwidth premise**
- `[C]` Fast Transformer Decoding: One Write-Head is All You Need — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) (Nov 2019)
- `[C]` The Pitfalls of KV Cache Compression — [arXiv:2510.00231](https://arxiv.org/abs/2510.00231) (Sep 2025, rev. May 2026; ACL 2026)
- `[C]` SCBench: A KV Cache-Centric Analysis of Long-Context Methods — [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) (Dec 2024)
- `[C]` Taming the Fragility of KV Cache Eviction in LLM Inference (DefensiveKV) — [arXiv:2510.13334](https://arxiv.org/abs/2510.13334) (Oct 2025)
- `[C]` When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic for Non-Monotone Cache Compression — [arXiv:2605.08234](https://arxiv.org/abs/2605.08234) (May 2026)
- `[C]` KV Cache Optimization Strategies for Scalable and Efficient LLM Inference — [arXiv:2603.20397](https://arxiv.org/abs/2603.20397) (Mar 2026)
- `[C]` What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents — [arXiv:2607.08032](https://arxiv.org/abs/2607.08032) (Jul 2026)
- `[C]` When Attention Sink Emerges in Language Models: An Empirical View — [arXiv:2410.10781](https://arxiv.org/abs/2410.10781) (Oct 2024)
- `[C]` RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache — [arXiv:2605.08317](https://arxiv.org/abs/2605.08317) (May 2026)
- `[C]` EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving — [arXiv:2512.14946](https://arxiv.org/abs/2512.14946) (Dec 2025)
- `[C]` KVCompose: Efficient Structured KV Cache Compression with Composite Tokens — [arXiv:2509.05165](https://arxiv.org/abs/2509.05165) (Sep 2025)
- `[C]` OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference — [arXiv:2510.07651](https://arxiv.org/abs/2510.07651) (Oct 2025)
- `[C]` SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference — [arXiv:2606.31145](https://arxiv.org/abs/2606.31145) (Jun 2026)

**Lab-internal measurements**
- `[M]` `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB flat at ~200 GB/s),
  `large-tensor-fault-32gib` (≥32 GiB single tensors hang or fault),
  `kv-per-token-laguna` (192 KiB/token exact), `reference-model` (48 layers,
  12 full + 36 sliding, strict GSSS), `single-device-only`, `bf16-numerics-unproven`.
- `[M]` `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier bandwidth sweep.
- `[C]`/`[M]` `research/reference/CODE_MAP.md` — verified `file:line` pointers, used here
  for the vLLM prefix-cache eviction path (`block_pool.py:679`, `_maybe_evict_cached_block`).
