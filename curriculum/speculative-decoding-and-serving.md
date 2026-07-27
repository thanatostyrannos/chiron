---
title: Speculative decoding and serving — three ways to put rows into the same matmul
version: 1.0.0
date: 2026-07-26
track: F — Inference
mirrors: research/notes/inference-and-quantization.md §4–§5
prereqs: attention-variants-and-kv-cost (B), kv-cache-mechanics (C), paged-attention-and-prefix-reuse (C)
difficulty: moderate — one page of algebra, but the roofline prediction and the machine disagree by 8× until you find the missing bytes, and finding them is the module
time: 3–4 h reading and working the math; 3–4 h for the three exercises. All three were run on the Z13 before shipping; reference numbers are in the text.
---

# Speculative decoding and serving

**Difficulty and time, honestly.** The arithmetic here is easier than in
`kv-cache-mechanics.md` — there is one new equation and it is a geometric series. What is
hard is that this module's central prediction, derived from the roofline you already have,
is **wrong by 8× on the default software path of this machine** — it predicts sixteen free
rows and you get two — and the whole value of the module is in the four lines of byte
accounting that explain the gap and the one environment variable that closes it. Budget 3–4 hours for sections 1–5. Exercise A is 45 minutes and
carries the finding; Exercise B is an hour; Exercise C runs in under a second and produces
a result that contradicts the standard analysis in the speculative-decoding literature.

**What this module refuses to re-teach.** From `attention-variants-and-kv-cost.md` you need
`AI_attention = 2G/b`, `AI_weights = 2B/b_w`, and the `[M]` ≈105 FLOP/byte ridge point. From
`kv-cache-mechanics.md` you need the three budgets (residency, read traffic, maintenance
traffic) and `c = 2·n_kv·d_h·b` bytes per token per layer. From
`paged-attention-and-prefix-reuse.md` you need the block allocator, the absent fault path,
and the fact that preemption resets a request to zero. All of that is assumed and none of it
is repeated.

This module **teaches** what `research/notes/inference-and-quantization.md` §4–§5
**surveys**. Read that note first if you have not. I refine it in four places — §3.4 (the
batch/context frontier, derived rather than cited), §3.6 (the lookahead cost is two orders of
magnitude smaller than the drafter's own cache, which the note does not mention), §3.7 (do not
over-read vLLM's shipped depth table), and §5.4 (the per-position acceptance metric both
engines log is a survival function, not a rate) — and I say so at each. I found nothing in it
that is wrong.

---

## 1. What this module settles

**One:** speculation, batching, and chunked prefill are the *same operation* — adding rows
to a matmul that was going to read those bytes anyway — and they differ only in which byte
stream the extra rows leave unchanged, which is why batching can never move the attention
term and speculation always can; at 32 matched query rows against 32k of context, `[M]`
speculation costs 1.22× a single row and batching costs 21.02×, a **15.6× gap on identical
FLOPs**. **Two:** the number of free rows is `I*/I₁ − 1` where `I*` is the machine's ridge
point and `I₁` the single-row intensity, and on this machine that predicts 104 free rows on
the weight stream and 16 on a Laguna global attention layer — predictions I measured at **64**
and **16**, the second right *only* once the attention kernel stops materialising its score
matrix, which on gfx1151 it does by default, costing **30 bytes of traffic per score element**
and collapsing the free window from 16 rows to 2 `[M]`. **Three:** the acceptance-length model
everyone uses — i.i.d. acceptance with probability `α`, mean length `(1−α^{K+1})/(1−α)` —
understates the real mean by **16–26%** on three different token streams `[M]`, because
acceptance is strongly positively autocorrelated within a draft, and vLLM's own telemetry
already logs the survival curve that shows it while labelling it a per-position rate.

The first is theory. The second and third are measured on the Z13 for this module, and the
second is the reason a serving result from this machine has to declare its attention backend
before it means anything.

---

## 2. Theory in plain language

### 2.1 The actual problem: a serial dependency chain over a bus you cannot fill

Decode reads 100% of the weights and 100% of that sequence's KV cache to produce one token,
and does a trivial amount of arithmetic with what it drags in. Track B put a number on it:
a Laguna global layer runs decode attention at 6 FLOP per byte against a `[M]` ≈105 FLOP/byte
ridge — **5.7% of the machine's compute, structurally, not because anything is tuned badly**.

You have three moves against read amplification, and you have used all three in your career:
add an index, add a cache, or prefetch. Section 2 of the mirrored survey note explains why
none of them is available. There is no index — attention is defined as a scan over every
position. There is no cache tier below the KV cache — a miss is not slow, it is
unrepresentable inside the kernel (`paged-attention-and-prefix-reuse.md`, Break 1). And there
is nothing to prefetch, because token `t+1` is a function of token `t`, so the address of the
next work item is not known until the current work item retires.

That last sentence is the whole setup. **A strictly serial dependency chain on a machine with
idle execution units is the exact precondition for speculative execution**, and the response
here is the same response a CPU designer had in 1993: guess the chain, execute ahead, check,
squash on a miss.

> **Systems bridge — and it is unusually exact.** A branch predictor, an out-of-order window,
> and a squash on misprediction. The drafter is the predictor, the target model's forward pass
> is the execution window, and the rejection sampler is the retire stage.
>
> **First break, and it is in your favour.** A CPU squash is a correctness-neutral performance
> event only because the architectural state was never committed. Here the guarantee is
> stronger and stranger: the verification step *defines* the output, so a wrong guess costs
> nothing in quality even in principle. `[C]` Leviathan et al. (2211.17192, Nov 2022) and Chen
> et al. (2302.01318, Feb 2023) independently prove that a modified rejection-sampling
> acceptance rule makes the output distribution **exactly** the target model's. "Lossless" here
> is a theorem, not a benchmark result. You have never had a speculation mechanism with that
> property.
>
> **Second break, and it is against you.** A mispredicted branch costs a pipeline flush — a
> fixed hardware resource, already paid for. A mispredicted draft costs **KV cache blocks**,
> which are the scarcest resource in the system and are shared with every other tenant. The
> speculative state is not a fixed register file; it is a variable share of your working set.
> That trade has no analogue in CPU speculation, and §3.6 prices it.

### 2.2 Why verifying K+1 tokens costs almost the same as decoding 1

Here is the mechanism, in the order the bytes move.

A drafter — cheap, by construction — proposes `K` tokens. The target model then runs **one**
forward pass whose input is `K+1` positions: the last committed token plus the `K` guesses.
That forward pass reads the weights once and reads the KV cache once, exactly as a
single-token decode would, and performs `K+1` times as much arithmetic against the bytes it
read. The output is `K+1` sets of logits, one per position, which is precisely what you need
to check every guess simultaneously.

Then a rejection rule walks the `K+1` positions, accepts the longest correct prefix of the
draft, and commits **exactly one extra token** — either the corrected token at the first
mismatch ("recovered"), or a fresh sample from the target's own distribution at position
`K+1` if everything was accepted ("bonus"). This is why the commit count is `a+1` where `a`
is the accepted length, and why vLLM's mean-acceptance-length metric is literally
`1 + accepted/drafts` (`memory/vllm/vllm/v1/spec_decode/metrics.py:114`).

> **Systems bridge.** Read-ahead. You issued one large sequential read instead of `N` small
> dependent ones, and you threw away whatever you over-read. The economics are identical:
> read-ahead pays exactly when the marginal byte is cheaper than the marginal round trip.
>
> **Where it breaks.** In storage, read-ahead over-reads *bytes* and the waste is bandwidth.
> Here speculation over-computes *FLOPs* and the waste is compute you were not using anyway —
> so the waste is genuinely free until the moment you cross the ridge point, at which
> instant it becomes the dominant cost with no warning. There is no gradual degradation. The
> next section makes that a number.

### 2.3 The unification: three mechanisms, one matmul

This is the idea to carry out of the module. Every throughput technique in a modern serving
stack adds **rows** to a matrix multiplication that was going to read its operands regardless.
They differ only in where the rows come from, and therefore in which byte stream stays
constant.

| Mechanism | Rows added | Weight bytes | KV bytes | Weight AI | Attention AI |
|---|---|---|---|---|---|
| Decode, 1 sequence | 1 | `P·b_w` | `c·L·T` | `2/b_w` | `2G/b` |
| **Batching**, `B` sequences | `B` | unchanged | **×B** | `2B/b_w` | `2G/b` — **unchanged** |
| **Speculation**, `K+1` positions | `K+1` | unchanged | unchanged | `2(K+1)/b_w` | `2G(K+1)/b` |
| **Chunked prefill**, `C` tokens | `C` | unchanged | unchanged | `2C/b_w` | `2G·C/b` |

Read the KV column. Batching is the only one of the three that multiplies it, because **each
sequence owns its own cache**; the other two add rows that attend to a cache that is already
being read. That single structural fact is why batching provably cannot raise the attention
term and speculation always can, and it is the analytical claim the mirrored survey note is
built on.

Now read rows three and four together. A speculative verify over `K+1` positions and a
chunked prefill of `C` tokens have **identical arithmetic-intensity structure**. They are the
same operation. The only difference is epistemic:

> **Speculation is chunked prefill of tokens you have not observed yet, discounted by the
> probability that you guessed right.**

vLLM's scheduler says exactly this, in a comment, and then builds the whole scheduler around
it (`memory/vllm/vllm/v1/core/sched/scheduler.py:430`):

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the
> `num_computed_tokens` and `num_tokens_with_spec`. […] This is general enough to cover
> chunked prefills, prefix caching, speculative decoding […]

> **Systems bridge.** A single admission-controlled work queue with one resource — a per-step
> token budget (`scheduler.py:447`) — and heterogeneous work items that all cost the same unit.
> This is the cleanest scheduler design you will read this year and it will feel familiar
> immediately.
>
> **Where it breaks, and this is the one that surprises infrastructure people.** The queue has
> no backpressure. When the block allocator cannot satisfy a request, the scheduler does not
> stall it and does not fault a page — it **preempts** a victim and sets
> `request.num_computed_tokens = 0` (`scheduler.py:1216`). Overload does not degrade latency
> gracefully; it converts completed work into wasted work. Every admission-control intuition
> you have is calibrated on systems where the queue absorbs the overload. Here the queue
> *destroys* it.

### 2.4 The drafter taxonomy, read as "who pays the fixed cost"

Ordered by how much of the target model they reuse. Every row is a different answer to the
question: where does the prediction come from, and what did it cost to build?

| Family | Drafter | Reuses | Fixed cost | Anchor |
|---|---|---|---|---|
| Independent draft model | a separate small LM | tokenizer only | train/obtain a second model | `[C]` 2211.17192, 2302.01318 |
| Self-draft heads | extra LM heads on the target | all target features | train the heads | `[C]` Medusa 2401.10774 |
| Feature-conditioned AR draft | 1-layer transformer over target hidden states | hidden states + tokenizer | train a 1-layer model per target | `[C]` EAGLE 2401.15077, EAGLE-2 2406.16858, EAGLE-3 2503.01840 |
| MTP heads | extra-token heads trained in | the pretraining objective | pay at pretraining | `[C]` DeepSeek-V3 2412.19437 |
| Block-diffusion draft | small masked-diffusion model | target hidden states + embeddings | train a per-target block drafter | `[C]` DFlash 2602.06036 |
| Model-free | n-gram / suffix match over history | nothing | **zero** | llama.cpp `ngram-*`, no paper |

> **Systems bridge.** This is a cache-warming ladder. At the top you build a whole second
> index; at the bottom you exploit locality that is already in the request stream. You have
> made this decision before and the answer was usually "the free one is startlingly good on
> real traffic."
>
> **Where it breaks.** A warm cache is *correct* regardless of how it was built. A drafter's
> value is entirely in its hit rate, which is workload-dependent and unbounded below — and a
> bad drafter is worse than none, because you still pay for its forward pass and for the
> lookahead slots. The floor is not zero benefit; it is negative benefit. §3.5 gives the
> break-even.

Two structural notes that only fall out of reading the code:

- **EAGLE's entire difference from a plain draft model, in vLLM, is one boolean.**
  `memory/vllm/vllm/v1/spec_decode/eagle.py:20` is `pass_hidden_states_to_model=True` and the
  class body is otherwise empty. Everything else — the K-step autoregressive draft loop, the
  KV cache for the drafter, the vocabulary mapping — is shared machinery in
  `llm_base_proposer.py`.
- **The autoregressive-versus-parallel split is one `if`.**
  `memory/vllm/vllm/v1/spec_decode/llm_base_proposer.py:619` short-circuits to a single
  sampling call when `self.parallel_drafting` is set (DFlash, MTP, Medusa heads); otherwise
  `:682` runs `for token_index in range(self.num_speculative_tokens - 1)`, a genuine
  `K−1`-iteration serial loop with a forward pass in each. §3.3 shows that this `if` is the
  difference between `K·c_d` and `c_d` in the denominator of the speedup, which is the entire
  practical argument for block drafters.

### 2.5 Laguna DFlash: a drafter that writes into a KV cache it does not own

`[C]` DFlash (2602.06036, Feb 2026; ICML 2026) replaces autoregressive drafting with **block
diffusion**: the drafter emits an entire block of `K` tokens in one forward pass under a
non-causal mask, conditioned on features extracted from the target. Authors claim over 6×
lossless acceleration and up to 2.5× more speedup than EAGLE-3. Not independently replicated;
treat the numbers as claims.

The mechanism, read from the Laguna llama.cpp branch, is more interesting than the abstract
and is why an inference module belongs next to the memory track:

- The drafter is **not** a small model that shares a tokenizer. It *requires* a
  `target_layers` array in its GGUF metadata and sets its encoder input width to
  `len(target_layers) × n_embd` (`architecture/llama-cpp-laguna/src/models/dflash.cpp:10`,
  `:14`). It is a function of several *named layers* of one specific target.
- The decoder is dual-mode by batch type (`dflash.cpp:153`). An **embd batch** projects the
  fused target features and **injects them straight into the draft model's own K/V cache**; a
  **token batch** then attends over `[committed, MASK, MASK, …]` to emit the block.
- The block size is a trained property read from a `dflash.block_size` GGUF key, defaulting to
  16 (`common/speculative.cpp:945`), and `--spec-draft-n-max` is clamped to `block_size − 1`
  because the input literally is `[id_last, <mask> × (block_size−1)]` (`:958`–`:963`).
- One `llama_decode` builds every sequence's noise block into a single batch (`:1178`), and
  the block is read greedily at noise positions `1..n−1`, stopping early if the top
  candidate's probability falls below `p_min` (`:1207`).

> **Systems bridge.** A cache used as an inter-process communication channel rather than as a
> memo table. You have seen this: shared memory that started as a cache and became an ABI.
>
> **Where it breaks — and this is a taxonomy problem, not a metaphor problem.** Every other
> entry in `research/memory/memory-taxonomy.md` classifies memory by *reconstructibility*: a
> KV entry is a memo you could recompute. A DFlash draft-cache entry is not a memo of anything
> the draft model computed — it is a projection of the *target's* hidden states, written in by
> a different model. It cannot be recomputed by the owner of the cache. That row does not
> exist in the taxonomy and should.

---

## 3. The math that actually matters

### 3.1 Symbols, every one translated

| Symbol | Reads as | Source |
|---|---|---|
| `K` | draft length: speculative tokens proposed per cycle | vLLM `num_speculative_tokens`; llama.cpp `--spec-draft-n-max` |
| `a` | **realised accept length** in one cycle: drafted tokens accepted before the first rejection, `0 ≤ a ≤ K` | runtime |
| `α` | per-token acceptance probability, in the i.i.d. idealisation | fitted, never configured |
| `τ̄` | **mean tokens committed per verify cycle** = `1 + E[a]` | `metrics.py:114` |
| `R` | **query rows in the step's matmul** = `B·(K+1)` + chunked-prefill tokens | derived |
| `c_d` | cost of producing one draft, in units of one ordinary target decode step | measured |
| `r(K)` | cost of one target forward over `K+1` positions ÷ cost over 1 position | measured; §3.4 |
| `I₁` | arithmetic intensity of a decode step with **one query row per sequence**, FLOP/byte | derived; §3.4 |
| `I*` | the machine's **ridge point**, FLOP/byte — `[M]` ≈105 here | `ASSUMPTIONS.md` |
| `B` | batch, i.e. concurrent sequences | runtime |
| `G` | GQA group size `n_q/n_kv` | derived; 6 on Laguna global, 9 on sliding `[M]` |
| `b`, `b_w` | bytes per KV element / per weight element | config |
| `T` | tokens in context | runtime |
| `P` | parameters read per forward pass (active params for an MoE) | config |
| `c` | KV bytes per token per layer = `2·n_kv·d_h·b` | from `kv-cache-mechanics.md` |

Lowercase `b` is bytes-per-element, uppercase `B` is batch, matching the rest of Track C.

### 3.2 The commit count, and why it is always `a+1`

Whatever happens, a verify cycle commits `a + 1` tokens. If `a < K` the first rejected
position is replaced by a token sampled from the **residual** distribution
`max(0, p_target − p_draft)`, renormalised — vLLM's "recovered" token
(`memory/vllm/vllm/v1/sample/rejection_sampler.py:930`). If `a = K` all guesses were right and
you additionally get a free sample at position `K+1` from the target's own logits, which the
forward pass already produced — the "bonus" token (`rejection_sampler.py:839`).

So the baseline is 1 token per target forward, and speculation gives `a+1` for the same
forward. Everything else is bookkeeping.

**The acceptance rule itself is two lines of Triton**
(`memory/vllm/vllm/v1/sample/rejection_sampler.py:829`):

```python
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

In words: draw `u ~ U[0,1)`; accept the drafted token `x̂` if `p_target(x̂)/p_draft(x̂) ≥ u`,
i.e. with probability `min(1, p/q)`. That is the Leviathan rule verbatim, and the file cites
the paper (`:41`).

**A detail with real consequences.** When the drafter returns no probability distribution —
every n-gram drafter, and any greedy drafter — the kernel sets `draft_prob = 1`
(`rejection_sampler.py:816`) and the residual becomes the target distribution with the drafted
token zeroed (`:913`–`:918`). This is not an approximation. A deterministic drafter *is* the
point mass `q = δ_{x̂}`, so `q(x̂) = 1` exactly, and the rule specialises correctly. Two things
follow, and both are useful:

```
for a deterministic drafter:   P(accept x̂)  =  p_target(x̂)
                                       α    =  E[ p_target(drafted token) ]
```

**The acceptance rate of a deterministic drafter is exactly the target model's own
probability mass on the token the drafter guessed.** No free parameters. That is the cleanest
definition of drafter quality in the whole subject, and it means "acceptance rate" is a joint
property of the drafter *and* the target's confidence — a target that has been made more
confident (lower temperature, more training, a sharper distribution) raises `α` for free.

`[M]` **One caveat on "lossless", from the same file's docstring** (`:53`–`:55`): the bonus
token supports top-p/top-k, while "spec decode does not support these sampling strategies."
The distributional guarantee holds against the target's **raw** distribution. Layer a
truncation sampler on top and you are outside the theorem. This is exactly the kind of thing
that turns a proof into a quiet behaviour change in production.

### 3.3 Expected accepted length, and the speedup

Under the standard i.i.d. idealisation — every drafted token accepted independently with
probability `α` — the accept length is geometric, truncated at `K`:

```
P(a ≥ i)  =  α^i                    for i = 0 … K
E[a]      =  Σ_{i=1..K} α^i  =  α·(1 − α^K)/(1 − α)
τ̄         =  1 + E[a]        =  (1 − α^{K+1})/(1 − α)
```

In words: the chance of getting at least `i` tokens right is `α` multiplied by itself `i`
times; the expected number right is that summed over `i`; and the tokens you actually commit
is one more than that. Sanity: `α = 1` gives `τ̄ = K+1`; `α = 0` gives `τ̄ = 1`.

Now the cost. Measure everything in units of one ordinary single-token target decode step:

```
                              τ̄                                    (1 − α^{K+1})/(1 − α)
speedup, AUTOREGRESSIVE  =  ─────────────         speedup, BLOCK =  ─────────────────────
                            K·c_d + r(K)                                c_d + r(K)
```

- `K·c_d` — an autoregressive drafter (a draft model, EAGLE) runs `K` sequential forward
  passes; `llm_base_proposer.py:682` is that loop.
- `c_d` — a block drafter (Medusa heads, MTP, DFlash) produces all `K` in one pass;
  `llm_base_proposer.py:619` is that branch.
- `r(K)` — what one verify pass over `K+1` rows costs relative to one row. The next section is
  entirely about this term, because it is the one everybody assumes is 1.

**Worked, with the numbers Exercise C measures.** A model-free n-gram drafter over source
code: `c_d ≈ 0` (it is a `bytes.rfind`), `K = 8`, measured `τ̄ = 4.96` `[M]`. Note `K = 8`
means the verify pass is **9 rows**. `[M]` On the fused attention path at 32k context those
nine rows cost the same as one, so `r(8) = 1.0` and the speedup is **4.96×**. On this
machine's *default* attention path the fitted curve gives `t(9)/t(1) = 3.32`, so the speedup
is **1.49×**. Same drafter, same acceptance rate, same model, same GPU. The entire difference
is an environment variable.

### 3.4 `r(K)`, the free-row window, and where it comes from

A verify pass over `K+1` rows reads the same bytes as a one-row pass and does `K+1` times the
FLOPs. So its intensity is `(K+1)·I₁`, and under the roofline it stays free until that
crosses the ridge:

```
r(K)  =  max( 1 ,  (K+1)·I₁ / I* )

                         I*
K_free  =  ───────────────────────────  −  1
                        I₁
```

In words: **the number of rows you get for free is the ridge point divided by the intensity
of one row, minus one.** Every symbol in it is something you already measured.

`I₁` differs per byte stream, so evaluate it per stream and take the binding one.

| Byte stream | `I₁` | `K_free = 105/I₁ − 1` |
|---|---|---|
| Weights, batch 1, bf16 | `2/b_w` = 1 | **104** |
| Weights, batch 8 | 8 | 12.1 |
| Weights, batch 32 | 32 | 2.3 |
| KV, Laguna **global** layer (`G`=6) | 6 | **16.5** |
| KV, Laguna **sliding** layer (`G`=9) | 9 | 10.7 |
| KV, MHA (`G`=1) | 1 | 104 |
| KV, a hypothetical MQA Proteus arm (`G`=48) | 48 | 1.2 |

`[A]` Derived, high confidence in the algebra, and Exercise A measures two of these rows.

Three readings, in increasing order of usefulness.

**First: the binding stream flips with context.** At short context the weight read dominates
the bytes and the free window is enormous; at long context the KV read dominates and the
window is set by `G`. On Laguna specifically the **sliding** layers bind first (`K_free` 10.7
versus 16.5), which is the same 36-of-48-layers minority that
`kv-cache-mechanics.md` Exercise B found consuming 30% of the time for 4.5% of the bytes. The
windowed layers keep turning out to be the constraint.

**Second: GQA and speculation are complements on latency and competitors for the same
headroom.** Both multiply the attention numerator. A model with aggressive GQA reaches the
ridge at a smaller `K`, so **the more aggressive your GQA, the less speculation depth is
free**. An MQA Proteus arm would have `K_free ≈ 1`, which is to say speculation would be
almost worthless to it as a bandwidth play. That is a Proteus design consideration, it is
derivable in one line, and I have not seen it stated anywhere.

**Third: batching cannot close the window, and there is a floor.** Compute the blended
single-row intensity of a whole step:

```
             B · ( 2·P  +  4·L·n_q·d_h·T )        (FLOPs: one row per sequence)
I₁(B, T)  =  ──────────────────────────────
                P·b_w  +  B·c·L·T                 (bytes: weights once, caches B times)
```

Read `I₁(B, T)` as the intensity of a step in which each of `B` sequences contributes exactly
one query row. Speculation then multiplies the numerator by `K+1` and leaves the denominator
alone, so the step's intensity is `(K+1)·I₁` and `K_free = I*/I₁ − 1` as before.

`[A]` For the 300M Proteus placeholder used elsewhere in the curriculum (`P` = 3e8, `L` = 24,
`n_q` = 16, `n_kv` = 8, `d_h` = 64, bf16, all-global, so `c·L` = 49,152 B/token — medium
confidence, it is a placeholder config, and the cheapest fix is freezing an arm config):

| `B` | `T` | `I₁` | `K_free` | note |
|---|---|---|---|---|
| 1 | 1,024 | 1.08 | 96 | weights dominate the bytes |
| 1 | 32,768 | 1.73 | 60 | |
| 8 | 1,024 | 5.59 | 17.8 | |
| 8 | 32,768 | 2.27 | 45.3 | |
| 32 | 1,024 | 10.1 | 9.4 | batching has eaten the window |
| 32 | 32,768 | 2.34 | 43.9 | batching has barely touched it |
| → ∞ | 32,768 | **2.37** | **43.3** | the asymptote |

The last row is the point. As `B → ∞` the weight bytes vanish from the denominator and
`I₁ → (2P + 4·L·n_q·d_h·T)/(c·L·T)`, which for this config is 2.37 FLOP/byte — **2.3% of the
ridge**. No amount of batching can push a long-context decode step above that, so **a
~43-row free window survives at any batch size at 32k context**. At 1k context the same
sweep collapses from 96 to 9.4.

That is `[C]` MagicDec's argument (2408.11049, Aug 2024) — at long context and large batch,
decode returns to memory-bound and speculation pays again — derived from first principles
with our own ridge point. It is also the analytical resolution of the contested question in
§4 of the survey note, and it comes out on MagicDec's side *for the long-context regime this
lab studies* and against it at short context. Both halves of the folklore are right about a
different regime.

`[M]` **A capacity check, because this lab is capacity-bound.** At `T` = 32,768 the 300M
placeholder holds 1.611 GB of KV per sequence, so against the measured ≥62 GiB fast tier
(`ASSUMPTIONS.md → gpu-fast-tier-size`) the maximum batch is `(62·2³⁰ − 6e8)/1.611e9 ≈ 40`.
The `B = 32` rows above are near our ceiling; the `B → ∞` row is arithmetic, not an operating
point.

**Sensitivity, stated plainly.** `I*` ≈ 105 is a ratio of two single-run numbers
(20.9 TFLOP/s ÷ 199.9 GB/s), and `kv-cache-mechanics.md` Exercise B measured only ~150 GB/s
on decode-shaped attention reads, which would put the ridge at ~139 and every `K_free` above
33% higher. Treat these as order-of-magnitude, which is all they need to be — the numbers
being compared differ by factors of 10.

### 3.5 Break-even, and why a bad drafter is worse than none

Set speedup > 1 and solve. For a block drafter with `r(K) = 1`:

```
(1 − α^{K+1})/(1 − α)  >  1 + c_d
```

At `K = 4` and `c_d = 0.1` (the drafter costs a tenth of a target step) the left side is
`1 + α + α² + α³ + α⁴` and the condition is `α + α² + α³ + α⁴ > 0.1`, i.e. `α > 0.093`.
Trivially satisfiable. At `c_d = 0.5` you need `α > 0.39`. For an *autoregressive* drafter the
denominator is `K·c_d + 1` = 3.0 at the same `c_d = 0.5`, needing `α > 0.72`.

**That factor of `K` in the denominator is the entire argument for block drafters**, and it is
why Medusa, MTP and DFlash exist. It is also why the cheapest drafter in the taxonomy — a
model-free n-gram matcher with `c_d ≈ 0` — is competitive out of all proportion to its
sophistication: its break-even acceptance is essentially zero.

### 3.6 What speculation costs in bytes — and the term the survey's framing hides

The mirrored survey note says speculation "shrinks the batch a given memory budget supports."
That is directionally right, and I want to refine it with numbers, because the magnitude
inverts the intuition.

**Cost 1 — lookahead reservation.** Every running request must hold slots for tokens that may
be rejected. vLLM reserves `num_lookahead_tokens` per request
(`memory/vllm/vllm/v1/core/sched/scheduler.py:242`, set at `:253`, passed to the allocator at
`:569`). Relative capacity cost:

```
lookahead bytes / live bytes  =  K / (T + K)
```

`[M]` At Laguna-S's 192 KiB/token (`ASSUMPTIONS.md → kv-per-token-laguna`), `B` = 8, `K` = 7:
**10.5 MiB** of reservation against **48.0 GiB** of live KV at 32k context — **0.021%**. At
`T` = 512 it is 1.3%; at `T` = 128 it is 5.2%.

So the survey's framing is right in the regime it is usually said about (short chat turns) and
almost exactly zero in the regime this lab studies. **Speculation is memory-cheap precisely
where it is bandwidth-valuable.** That inversion is worth internalising and I have not seen it
written down.

`[M]` **A detail worth the read:** DFlash needs `K+1` lookahead slots rather than `K`, because
it does in-fill decoding rather than next-token sampling and therefore issues a query for the
last sampled token *plus* one per draft token (`scheduler.py:256`–`:260`). One extra slot per
request per step; a good example of a serving-layer cost that is a property of the drafter's
*decoding scheme*, invisible in its paper.

**Cost 2 — the drafter's own KV cache, and it is ~100× larger.** This is the term the "shrinks
the batch" framing hides entirely. Any drafter that is a transformer keeps its own cache:

```
k_draft / k_target  =  (L_d · n_kv,d) / (L · n_kv)
```

`[M]` EAGLE-3's draft is documented as a **one-layer** transformer
(`architecture/llama-cpp-laguna/docs/speculative.md:19`); DFlash's "uses several transformer
layers" with no number given (`docs/speculative.md:60`), and llama.cpp exposes a separate
`--spec-draft-type-k`/`-v` for it (`docs/speculative.md:259`), which tells you it is a
first-class cache with its own dtype. `[A]` If a 1-layer draft matches Laguna's KV geometry,
`k_draft/k_target` = 1/48 = 2.1%, which at `B` = 8 and `T` = 32,768 is **1.0 GiB** — a hundred
times the lookahead reservation. Medium confidence: the actual draft head counts are a
property of specific checkpoints I have not read.

**Cost 3 — slot churn.** Draft slots are allocated before verification and freed on rejection,
every cycle. At block granularity the churn is bounded at one block per request per cycle,
which is why this is a scheduling annoyance rather than a fragmentation problem — see
`paged-attention-and-prefix-reuse.md` §3.2.

**The rule for Mnemosyne:** budget the drafter's cache, not the lookahead. Everyone quotes the
lookahead because it appears in a scheduler config; the cache is two orders of magnitude
bigger and appears nowhere.

### 3.7 Continuous batching: one budget, two SLOs pulling apart

`[C]` Iteration-level batching originates with Orca (Yu et al., OSDI 2022 — no arXiv id; cited
by venue). Stop scheduling *requests*, schedule *iterations*: a finished sequence leaves the
batch immediately and a waiting one joins on the next forward pass.

The scheduling identity is one line:

```
B·(K+1)  +  C_prefill   ≤   N          N = max_num_batched_tokens  (scheduler.py:447)
```

— decodes, speculative tokens and chunked prefill all spend the same currency. Chunked prefill
is simply what happens when a prefill does not fit the remaining budget and is truncated
(`scheduler.py:511`), and `long_prefill_token_threshold` (`:509`) caps any single prefill so
one 128k prompt cannot starve every decode in the batch. `[C]` This is Sarathi-Serve's
stall-free batching (2403.02310, Mar 2024), contested against prefill/decode disaggregation
(`[C]` 2401.09670 DistServe, `[C]` 2311.18677 Splitwise), with 2026 work still trying to
unify the positions (`[C]` 2508.01989). **Leave it contested.**

**The TTFT/TPOT tension, with a number.** Prefill rows and decode rows cost the same in the
budget but not in wall clock, because prefill rows come with `2P` FLOPs each and no marginal
byte. `[A]` At the 300M placeholder and our `[M]` 20.9 TFLOP/s, a 512-token prefill chunk is
`512 × 6e8 / 20.9e12` = **14.7 ms** of arithmetic, against a `T` = 32,768 decode step of
2.211 GB ÷ 199.9 GB/s = **11.1 ms**. Adding that chunk more than doubles the step, and every
decoding request in the batch pays it. To hold TPOT inflation under 10% you need

```
C  <  0.1 · t_decode · FLOPS / (2·P)  =  0.1 × 11.1e-3 × 20.9e12 / 6e8  ≈  39 tokens
```

`[A]` Medium confidence — it assumes the prefill GEMM reaches peak, which at 512 rows is
plausible and at 39 rows is not (Exercise A measures exactly this curve). But the shape is
right and the conclusion is actionable: **`long_prefill_token_threshold` should be set from
the free-row window, not from a throughput target.** That is a serving-config number you can
derive from two measurements you already have.

**Speculation depth is a runtime function of load, not a constant.** `[M]` vLLM builds a dense
batch-size → `K` lookup table from a `num_speculative_tokens_per_batch_size` config
(`scheduler.py:245` → `memory/vllm/vllm/v1/spec_decode/dynamic/utils.py:77`), consulted per
step at `scheduler.py:1124`. The worked example in that file's own docstring is
`[(1,16,3),(32,128,2)]` — K=3 for batch 1–16, K=2 for batch 32–128 (`utils.py:107`).

Our §3.4 derivation predicts `K_free` = 5.6 at B=16 and 2.3 at B=32 on the *weight* stream.
The shipped table has the same **shape** — `K` decreasing in `B` — and the same order of
magnitude. **Do not over-read the numerical agreement.** That table was tuned on hardware
whose ridge point is roughly three times ours, which should make its free window *larger*, not
equal; so either their defaults are conservative for reasons unrelated to the roofline
(acceptance, memory, latency variance) or the agreement is coincidence. The shape is evidence;
the numbers are not.

---

## 4. Why this matters for Proteus and Mnemosyne

### 4.1 Speculation is a fourth budget, and Mnemosyne must name it

`kv-cache-mechanics.md` §2.2 established three budgets: residency, read traffic, maintenance
traffic. Speculation adds a fourth that is structurally different from all three:

> **Speculative reservation** — bytes held for tokens that may never be committed.

It is not residency, because the tokens may never exist. It is not read traffic, because
nothing reads it until verification. It is not maintenance, because it is not amortising a
structure. It is the only budget line in the system whose *expected* value depends on a model
quality parameter (`α`). Mnemosyne's cost-model interface — `residency_bytes(T)`,
`read_bytes_per_step(T)`, `maintenance_bytes_per_step(T)` — needs a fourth method,
`reserved_bytes_per_step(T, K)`, and it should return the drafter cache term as well as the
lookahead term, because §3.6 shows the drafter cache is the larger one by two orders of
magnitude.

This stays inside the boundary rule: it is a function of `list[LayerCacheSpec]` plus two
integers, and needs nothing from Proteus.

### 4.2 Speculation is a confound for every eviction experiment, and the control is one field

Any eviction or compression policy measured under a speculative decoder is measured against
(a) a smaller byte budget, (b) a different query shape — `K+1` query rows per KV read rather
than 1 — and (c) a cache whose tail churns every cycle. All three change the thing being
measured. The rig control is trivial and must be enforced:

**Pin `K = 0` for all Mnemosyne policy ablations, and record `spec_type` and `K` in every run
header regardless.** The header schema in `research/notes/inference-and-quantization.md`
already has the fields; the rig must assert the value rather than just log it. This is the
same class of harness bug as `kv-cache-mechanics.md` §4.3 — the oracle and the policy arm
disagreeing about their cache implementation — and it deserves the same one-line test.

### 4.3 The hypothesis this module exists to produce

§3.2 established that for a deterministic drafter `α = E[p_target(drafted token)]`. Acceptance
is therefore a functional of the target's **full output distribution**, not of its argmax.
Now recall `[C]` 2606.09864 (Jun 2026): KV quantisation costs Mistral-7B 15.2% of its refusals
at 1.03× perplexity — an outcome metric that looks fine while the distribution has moved.

Put the two together:

> **An eviction or compression policy that preserves perplexity can still destroy acceptance
> rate, because acceptance depends on the probability mass at the drafted token and perplexity
> does not resolve it.**

If that is true, `Δα` is a **cheaper and more sensitive probe of distributional damage than
perplexity**, it is free to compute (the rejection sampler already produces it), and it gives
Mnemosyne an attribution instrument that the compression literature does not use. If it is
false — if `Δα` tracks `Δppl` linearly — that is also worth knowing and kills a line of work
in an afternoon. This is the highest novelty-per-GPU-hour item this module produces and it
belongs in the ablation backlog as a pre-registered hypothesis card.

Note the honest risk: `α` also depends on the *drafter*, so a change in `α` under eviction
confounds "the target's distribution moved" with "the drafter's inputs moved" whenever the
drafter reads target features (EAGLE, DFlash). Use a model-free n-gram drafter for the probe
and the confound disappears — its inputs are token ids only.

### 4.4 The draft cache has no policy anywhere

Every engine in the reference library gives the drafter a real KV cache with a real dtype
(`docs/speculative.md:259` exposes `--spec-draft-type-k`), and **not one of them applies an
eviction, compression or offload policy to it.** It is allocated, filled, and freed with the
request. At 2.1% of the target's cache for a 1-layer drafter that is defensible; at DFlash's
"several layers" it is 5–10% of the most contended resource in the system, managed by nothing.
This is a small, concrete, unclaimed piece of ground for Mnemosyne, and the interface for it
already exists because the draft cache is a cache like any other — which is exactly what the
`mnemosyne → torch` boundary rule was designed to make possible.

### 4.5 Config surface

The config surface is the experimental surface. These fields belong to Mnemosyne and Themis,
not to Proteus, because none of them is an architecture property:

| Field | What it moves | Why explicit |
|---|---|---|
| `spec_type` | `c_d`, `α`, drafter cache size | §2.4 — the taxonomy is an ablation axis |
| `num_speculative_tokens` (`K`) | `τ̄`, `r(K)`, reservation | §3.3–§3.6; must be pinned to 0 for policy arms |
| `parallel_drafting` | `K·c_d` vs `c_d` in the denominator | §3.3 — the single largest term in the speedup |
| `draft_kv_dtype` | drafter cache bytes | §3.6 cost 2 — nobody ablates this |
| `max_num_batched_tokens` (`N`) | the whole scheduling identity | §3.7 |
| `long_prefill_token_threshold` | TTFT/TPOT split | §3.7 — derive it, do not guess it |
| `attention_backend` + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `r(K)`, and with it the entire speculation-versus-batching advantage: **3.17× → 15.64×** `[M]` | §5.5 and Exercises A and B. **This is not a tuning flag; it is the independent variable.** |

The last row is the one to argue for. On this machine, on identical shapes, the attention
backend changes the central quantity of this module by a factor of five to six — and it is set
by an environment variable that appears in no config file and that
`torch.backends.cuda.flash_sdp_enabled()` reports as `True` either way.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md` (`memory/vllm` @ `0934b267906f`, `architecture/llama-cpp-laguna` @
`04b2b72cb540`).

### 5.1 The verify rule — read this first, it is 40 lines

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/sample/rejection_sampler.py:40` | The class docstring. It cites 2211.17192 and defines the three output classes — **accepted + recovered + bonus** — which is the vocabulary the rest of the field uses loosely and this file uses precisely. Read `:53`–`:55` for the top-p/top-k caveat that voids the losslessness theorem. |
| `memory/vllm/vllm/v1/sample/rejection_sampler.py:829` | `accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob` — the entire acceptance rule, one line. Everything above it is index arithmetic. |
| `memory/vllm/vllm/v1/sample/rejection_sampler.py:816` | `if NO_DRAFT_PROBS: draft_prob = 1`. The deterministic-drafter specialisation. Not a hack — for `q = δ_{x̂}`, `q(x̂) = 1` exactly, so the rule is still the theorem. §3.2. |
| `memory/vllm/vllm/v1/sample/rejection_sampler.py:930` | `prob = tl.maximum(target_prob - draft_prob, 0.0)` — the residual distribution the recovered token is drawn from. The `NO_DRAFT_PROBS` branch at `:913` instead masks out the drafted id, which is the same thing for a point mass. |
| `memory/vllm/vllm/v1/sample/rejection_sampler.py:839` | `if not rejected:` → append the bonus token. This is the `+1` in `τ̄ = 1 + E[a]`, and it is only reached on a full accept. |

### 5.2 The drafters, in increasing order of coupling to the target

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/spec_decode/ngram_proposer.py:207` | `_find_longest_matched_ngram_and_propose_tokens` — the model-free drafter, a Numba-jitted KMP prefix-function search over the reversed token history. Read the comment block at `:230`–`:241`: it flips the sequence to turn "longest matching suffix" into "longest matching prefix." **Zero training, zero weights, zero KV cache.** Exercise C reimplements its semantics in nine lines. |
| `memory/vllm/vllm/v1/spec_decode/medusa.py:56` | `torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)` — and now notice what is *not* here. The Medusa paper's contribution is **tree attention** over several candidate continuations `[C]` (2401.10774); vLLM's proposer takes the top-1 from each head and returns a flat `[batch, num_heads]` chain. The shipped implementation is not the paper. Worth knowing before you cite a Medusa speedup. |
| `memory/vllm/vllm/v1/spec_decode/eagle.py:20` | `pass_hidden_states_to_model=True`. The entire class. EAGLE's difference from a draft model, in this stack, is one constructor argument; everything else is `SpecDecodeBaseProposer`. |
| `memory/vllm/vllm/v1/spec_decode/llm_base_proposer.py:619` | `if self.num_speculative_tokens == 1 or self.parallel_drafting:` — the one-pass branch. DFlash, MTP and Medusa land here. |
| `memory/vllm/vllm/v1/spec_decode/llm_base_proposer.py:682` | `for token_index in range(self.num_speculative_tokens - 1):` — the serial branch, one forward pass per iteration, each feeding the previous token back in at `:686`. Put `:619` and `:682` side by side: that pair is the `c_d` versus `K·c_d` distinction of §3.3, and it is the largest single term in the speedup formula. |
| `architecture/llama-cpp-laguna/src/models/dflash.cpp:10` | `throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata")` — the drafter is a function of *named layers* of one specific target, not a small model that shares a tokenizer. `:14` sets the encoder width to `len(target_layers) × n_embd`. |
| `architecture/llama-cpp-laguna/src/models/dflash.cpp:153` | The dual-mode comment, then the code: an **embd batch** injects projected target features straight into the draft model's own K/V cache; a **token batch** attends over `[committed, MASK…]`. This is a KV cache used as an inter-model channel — §2.5, and the taxonomy row that does not exist. |
| `architecture/llama-cpp-laguna/common/speculative.cpp:945` | `block_size = 16`, read from a `dflash.block_size` GGUF key; `:958`–`:963` clamps `--spec-draft-n-max` to `block_size − 1` with a warning. **`K` is a trained property of the drafter, not a serving knob.** |
| `architecture/llama-cpp-laguna/common/speculative.cpp:1178` | `common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, …)` — every drafting sequence's noise block built into one batch, then a single `llama_decode` at `:1187`. `:1207` reads the block greedily at noise positions and stops early below `p_min`. |
| `architecture/llama-cpp-laguna/docs/speculative.md:190` | The full `--spec-type` catalogue: `draft-simple`, `draft-eagle3`, `draft-dflash`, `draft-mtp`, and **four** model-free n-gram families. `:150` notes `ngram-mod` is ~16 MB, constant memory and complexity, with a **single hash pool shared across all server slots** so different requests draft for each other. That last property has no equivalent in any trained drafter. |

### 5.3 The scheduler — one budget, no phases

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/sched/scheduler.py:430` | The comment quoted in §2.3. Read it twice. It is the clearest statement in the reference library of why prefill, decode and speculation are one mechanism. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:447` | `token_budget = self.max_num_scheduled_tokens` — the single resource the whole scheduler spends. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:509` | `long_prefill_token_threshold` clamps one prefill; `:511` then clamps to the remaining budget. Chunked prefill is two lines and no abstraction. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:242` | `self.num_lookahead_tokens = 0`, then `:253` sets it to `K` for EAGLE and draft-model paths — and `:256`–`:260` sets it to **`K+1`** for DFlash, with a comment explaining that in-fill decoding needs a query for the last sampled token as well. A serving cost that is a property of the drafter's decoding scheme. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:569` | `num_lookahead_tokens=self.num_lookahead_tokens` — where the reservation actually reaches the block allocator. Follow it into `paged-attention-and-prefix-reuse.md`'s `allocate_slots`. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:1124` | `num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]` — depth chosen per step by array lookup on the current batch size. |
| `memory/vllm/vllm/v1/spec_decode/dynamic/utils.py:77` | `build_dynamic_sd_schedule_lookup` — the table builder, and at `:107` the worked example `[(1,16,3),(32,128,2)]` in a comment. §3.7. |
| `memory/vllm/vllm/v1/core/sched/scheduler.py:1216` | `request.num_computed_tokens = 0` — the preemption path. Under memory pressure caused *by your lookahead reservations*, this is what happens to somebody else's 29,000 tokens of prefill. |

### 5.4 Telemetry — and the metric everyone reads wrong

This is the highest-value read in the module for someone with an observability background.

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/spec_decode/metrics.py:41` | `observe_draft`. Then read `:46`–`:49` very carefully: `for i in range(num_accepted_tokens): num_accepted_tokens_per_pos[i] += 1`. Position `i`'s counter is incremented whenever the accept length **exceeded** `i` — not when position `i` was accepted given that it was reached. |
| `memory/vllm/vllm/v1/spec_decode/metrics.py:117` | `acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts`, logged at `:127` as "Per-position acceptance rate". **It is not a per-position rate. It is the survival function `S(i) = P(a > i)`.** The conditional per-position rate is `S(i)/S(i−1)`, which you have to compute yourself, and Exercise C shows the two differ by 12 percentage points at position 1 on real data. |
| `memory/vllm/vllm/v1/spec_decode/metrics.py:114` | `mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)` with the comment "Conventionally, mean acceptance length includes the bonus token." This is `τ̄` and it is the numerator of the speedup. |
| `architecture/llama-cpp-laguna/tools/server/server-context.cpp:3879` | The C++ accumulation — `for (i = 0; i < ids.size()-1 …) n_accepted_per_pos[i]++` — byte-for-byte the same semantics as `metrics.py:46`. Printed at `:620` divided by `n_draft_verif_steps`, i.e. by the number of drafts, so it is the same survival function. Alongside it at `:616`, `mean_acc_len = 1.0 + n_draft_accepted/n_draft_verif_steps`. **Two independent engines converged on the same three metrics and on the same misleading label; match the metrics, fix the label.** |
| `architecture/llama-cpp-laguna/docs/speculative.md:401` | The per-implementation statistics block, including `#gen drafts` vs `#acc drafts` vs `#gen tokens` vs `#acc tokens`. Four counters, three ratios, and only one of them is the number that predicts speedup. |

### 5.5 The thing that decides `r(K)` on this machine

There is no line to point at, which is the lesson. `r(K)` — the cost of a `K+1`-row verify —
is a property of the attention kernel that ran, not of anything in a config file, and on
gfx1151 the kernel that runs by default materialises its score matrix
(`ASSUMPTIONS.md → sdpa-is-memory-efficient`). Section 6 measures the consequence: the free
window collapses from the roofline's 16 rows to 2, and `torch.backends.cuda.flash_sdp_enabled()`
returns `True` the whole time. The only honest signal is the stderr `UserWarning`.

**The generalisation to carry:** residency is a property of the allocator, read traffic of the
kernel, maintenance traffic of the cache class — and now, `r(K)` is a property of the kernel's
*intermediate* materialisation, which no interface in the stack reports.

---

## 6. Exercises

All three were run on the Z13 for this module and the reference numbers below are `[M]` from
those runs. Your absolute numbers will differ; the **ratios** are the deliverable. Each
exercise states a prediction before the table, and one of the three predictions failed.

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats** (`ASSUMPTIONS.md`): single tensors ≥32 GiB **hang silently at 0%
CPU** — every buffer below is far under that; bf16 numerics on gfx1151 are unproven
(`bf16-numerics-unproven`), so *timing* claims here are sound and *accuracy* claims are not;
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is deliberately **off** in `activate-lab.ps1` and
Exercise A deliberately runs both ways because the flag is this module's independent variable.
The Hardware Validation Gate has not run, so none of this is evidence by house standard —
these are instrument-shakedown runs and the module labels them as such.

Write scratch scripts under `notebook/`. Exercises A and B are Hardware Validation Gate
candidates and migrate into the rig with tests on reuse.

---

### Exercise A — how many rows are free, on each byte stream?

**Goal:** measure `r(K)` directly. This is the module's central quantity and §3.4's
`K_free = I*/I₁ − 1` is the prediction under test.

**Hardware:** one gfx1151 GPU. **CPU fallback:** set `d=2048`, `T=4096`, rows to
`[1,2,4,8,16,32]`; the curve shape transfers, absolute bandwidth does not. **Runtime:** ~4
minutes on GPU, ~3 on CPU. Largest allocation is the 8192² weight matrix at 134 MB.

```python
"""How many extra rows does a matmul give you for free?
Part 1: the weight stream (one bf16 matrix, R input rows).
Part 2: the KV stream, Laguna-S global-layer geometry, R query rows vs ONE cache."""
import json, statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, N_KV, D_H, G = torch.bfloat16, 8, 128, 6       # [M] ASSUMPTIONS.md -> reference-model
def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters

def weight_stream(rows, d=8192):
    W = torch.randn(d, d, dtype=DT, device=DEV); wb = d * d * 2
    for R in rows:
        X = torch.randn(R, d, dtype=DT, device=DEV)
        s = statistics.median(timeit(lambda: X @ W) for _ in range(3))
        print(json.dumps(dict(part="weight", R=R, ms=round(s*1e3, 4),
                              gb_s=round(wb/s/1e9, 1),
                              tflops=round(2*R*d*d/s/1e12, 3))), flush=True)

def kv_stream(T, rows):
    K = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    V = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    kvb = 2 * N_KV * T * D_H * 2
    def attn(q):
        s = torch.bmm(q, K.transpose(1, 2)) * (D_H ** -0.5)
        return torch.bmm(torch.softmax(s.float(), -1).to(DT), V)
    for R in rows:
        q = torch.randn(N_KV, G * R, D_H, dtype=DT, device=DEV)
        s = statistics.median(timeit(lambda: attn(q)) for _ in range(3))
        print(json.dumps(dict(part="kv", T=T, R=R, ms=round(s*1e3, 4),
                              gb_s_kv_only=round(kvb/s/1e9, 1),
                              score_elems=N_KV*G*R*T)), flush=True)

weight_stream([1, 2, 4, 8, 16, 32, 64, 128, 256])
kv_stream(32768, [1, 2, 4, 8, 16, 32, 64])
```

**Predictions, stated before you run.**
1. The weight stream is **flat to R ≈ 104** (`I₁` = 1 at bf16, batch 1), then linear.
2. The KV stream is **flat to R ≈ 16** (`I₁` = `G` = 6), then linear.
3. Both knees land where `(R)·I₁ = 105`. **Prediction 2 is the one that failed, badly.**

`[M]` **Reference numbers.** Z13 / gfx1151 / native Windows, torch
`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, bf16, hipBLASLt configured
(`HIPBLASLT_TENSILE_LIBPATH` set, `TORCH_BLAS_PREFER_HIPBLASLT=1`), median of 3 × 30
iterations after 5 warmups, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset, 2026-07-26.

**Part 1 — weight stream** (8192×8192 bf16 = 134.2 MB):

| `R` | ms | GB/s | TFLOP/s | vs `R`=1 |
|---|---|---|---|---|
| 1 | 0.6592 | 203.6 | 0.204 | 1.00 |
| 2 | 0.6617 | 202.8 | 0.406 | 1.00 |
| 4 | 0.6706 | 200.1 | 0.801 | 1.02 |
| 8 | 0.6681 | 200.9 | 1.607 | 1.01 |
| 16 | 0.6692 | 200.6 | 3.209 | 1.02 |
| 32 | 0.7026 | 191.0 | 6.113 | 1.07 |
| **64** | **0.7259** | 184.9 | 11.833 | **1.10** |
| 128 | 1.2591 | 106.6 | 13.644 | 1.91 |
| 256 | 1.8285 | 73.4 | 18.791 | 2.77 |

**Prediction 1 holds.** Sixty-four rows for the price of one, within 10%. The knee sits
between 64 and 128 and the roofline said 105 — **the first local validation of the ≈105 ridge
point on a workload shape rather than on a peak-FLOPS benchmark.** Note also that the 8192³
GEMM figure of 20.9 TFLOP/s only reappears at `R` = 256 (18.8, 90% of it); at `R` = 64 the same
hardware delivers 11.8. Skinny GEMMs do not reach peak, which is why `r(K)` must be measured
rather than assumed even on the stream where the prediction works.

**Part 2 — KV stream**, `T` = 32,768, two independent processes:

| `R` | run 1, ms | run 2, ms | vs `R`=1 | roofline says |
|---|---|---|---|---|
| 1 | 0.8542 | 0.8956 | 1.00 | 1.00 |
| 2 | 1.0451 | 1.1534 | **1.29** | 1.00 |
| 4 | 1.6574 | 1.7107 | **1.91** | 1.00 |
| 8 | 2.7909 | 2.9097 | **3.25** | 1.00 |
| 16 | 4.8140 | 4.8896 | 5.46 | 1.00 |
| 32 | 8.4638 | 8.5477 | 9.54 | 1.83 |
| 64 | 15.7111 | 15.8222 | 17.67 | 3.66 |

**Prediction 2 failed by 8×.** The free window is 2 rows, not 16 — and at `R` = 16 the pass
costs 5.46× a single row where the roofline said 1.00. Least squares over run 2:

```
t(R)  ≈  0.847 ms  +  0.2362 ms · R          (R = 1 … 64)
```

**The attribution, which is the actual exercise.** A slope means bytes. The intermediate score
matrix has `n_kv · G · R · T` = 1.573 × 10⁶ · `R` elements, and the naive path traverses it
about seven times — bmm output in bf16, a scale, a cast to fp32, softmax read and write, a cast
back, and the second bmm's read. At the `[M]` 199.9 GB/s reference the measured slope is
`0.2362 ms × 199.9 GB/s` = 47.2 MB per row, which over 1.573 × 10⁶ elements is

```
≈ 30 bytes of memory traffic per score-matrix element        [M]
```

Add that to the KV read and the path is bandwidth-bound at 167–199 GB/s at **every** `R`.
Nothing is wrong with the machine. **The roofline model was undercounting bytes**, because it
assumes the score matrix never leaves registers — which is exactly what a fused
FlashAttention-style kernel guarantees and this one does not.

**Now close the loop.** Rerun part 2 with `F.scaled_dot_product_attention`, once with the flag
off and once on:

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = '1'   # a NUMERICS change; see activate-lab.ps1
```

`[M]` Same configuration, `T` = 32,768, `n_q` = 48, `n_kv` = 8, `enable_gqa=True`:

| `R` | manual bmm+softmax | SDPA, flag **off** | SDPA, flag **on** |
|---|---|---|---|
| 1 | 0.8956 | 28.05 | **3.769** |
| 2 | 1.1534 | 27.81 | 4.117 |
| 4 | 1.7107 | 28.45 | 4.160 |
| 8 | 2.9097 | 30.09 | 4.110 |
| 16 | 4.8896 | 34.45 | **4.148** |
| 32 | 8.5477 | 54.20 | 5.121 |
| 64 | 15.8222 | 93.66 | 5.816 |

Three findings, in order of importance.

1. **With the fused kernel the free window is exactly what the roofline predicted.** 3.769 →
   4.148 ms from `R` = 1 to `R` = 16 is +10%; the knee is between 16 and 32; §3.4 said 16.5.
   **`K_free = I*/I₁ − 1` is confirmed on the attention stream to within one row, once the
   kernel stops materialising bytes the model does not count.**
2. **The default path is 7.4× to 16.1× slower than the fused one** and is what you get unless
   you set an environment variable that appears in no config file. Any speculative-decoding
   result from this machine that does not state its attention backend is uninterpretable.
3. **The fused kernel is 4.2× *slower* than the naive one at `R` = 1**, and only wins past
   `R` ≈ 14. So the flag is not a free win: on this stack it buys a flat `r(K)` at the price of
   a much worse `r(0)`. Which one you want depends entirely on whether you are speculating.

**The mechanism behind finding 3, probed — and the probe came back split.** `[M]` Same
configuration, `R` = 1, `T` = 32,768, fused path, fresh process:

| case | `n_q` | `n_kv` | ms | KV *stored* | GB/s if stored | GB/s if expanded to `n_q` heads |
|---|---|---|---|---|---|---|
| `gqa48` (`enable_gqa=True`) | 48 | 8 | 4.376 | 134.2 MB | 30.7 | **184.0** |
| `expand48` (K/V pre-expanded) | 48 | 48 | 4.833 | 805.3 MB | **166.6** | 166.6 |
| `mha8` | 8 | 8 | 3.366 | 134.2 MB | 39.9 | 39.9 |

Read the first two rows: **`gqa48` and `expand48` take the same time to within 10% while
nominally differing by 6× in stored bytes**, and `expand48` is demonstrably bandwidth-bound at
166.6 GB/s. The straightforward reading is that the fused path reads the cache once per *query*
head — i.e. it materialises the GQA expansion and the GQA **bandwidth** saving does not exist
here, only the capacity saving.

The third row is the control and it does **not** cooperate. The deliverable below predicted
`mha8` would be ~6× faster than `gqa48` if the expansion were real; it is 1.30× faster. But
`mha8` runs at 39.9 GB/s — a fifth of this machine's bandwidth — so it is not bandwidth-bound
at all. With eight heads and one query row there is not enough parallelism to saturate the
memory system, and head count confounds the comparison. `[A]` **Medium confidence in the
expansion mechanism**, on the strength of rows 1–2 and with row 3 explicitly inconclusive
rather than supporting. The clean test is to repeat rows 1–2 at `R` = 16, where both are
comfortably bandwidth-bound; that is fifteen minutes and it would earn an `ASSUMPTIONS.md` row,
because if it holds it touches every arithmetic-intensity claim in Tracks B and C on this
machine.

**Deliverables — four numbers and one plot.**
1. `R` at which the weight stream first exceeds 1.1× its `R`=1 time. Report next to `105/I₁`.
2. The fitted slope of the KV curve in ms/row, converted to bytes per score element. Ours is
   30. If yours is far from 30, your intermediate dtype path differs — print
   `s.dtype` inside `attn` and check.
3. The ratio `t_default(R) / t_fused(R)` at `R` = 1 and `R` = 64. Ours: 7.4 and 16.1.
4. Reproduce the three-shape probe above, then extend it to `R` = 16 and report whether
   `gqa48` and `expand48` still agree once both are bandwidth-bound. State plainly whether your
   run confirms, refutes, or (like ours) half-confirms the expansion mechanism.

**Plot** `t(R)` for all three kernels on log-log axes. The flat segment is the free window,
drawn.

---

### Exercise B — speculation versus batching, at matched query rows

**Goal:** the module's central claim, measured. Two arms with **identical FLOPs and identical
query-row counts**, differing only in how many distinct KV caches those rows read.

**Hardware:** one gfx1151 GPU. Largest allocation is the batch arm at `R`=32: 32 × 134 MB =
4.3 GB, well inside the `[M]` ≥62 GiB fast tier and far below the 31 GiB per-tensor hazard.
**CPU fallback:** `T=4096`, rows `[1,2,4,8]`. **Runtime:** ~6 minutes on GPU.

```python
"""spec  arm: 1 sequence, R query rows, ONE cache of T tokens.
   batch arm: R sequences, 1 query row each, R caches of T tokens.
   Identical FLOPs. The only difference is how many caches the rows read."""
import json, statistics, time, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT, N_KV, D_H, G, T = torch.bfloat16, 8, 128, 6, 32768
def sync():
    if DEV == "cuda": torch.cuda.synchronize()
def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters
def attn(q, K, V):
    s = torch.bmm(q, K.transpose(1, 2)) * (D_H ** -0.5)
    return torch.bmm(torch.softmax(s.float(), -1).to(DT), V)

for R in [1, 2, 4, 8, 16, 32]:
    K = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    V = torch.randn(N_KV, T, D_H, dtype=DT, device=DEV)
    q = torch.randn(N_KV, G * R, D_H, dtype=DT, device=DEV)
    spec = statistics.median(timeit(lambda: attn(q, K, V)) for _ in range(3)) * 1e3
    del K, V, q
    if DEV == "cuda": torch.cuda.empty_cache()

    K = torch.randn(R * N_KV, T, D_H, dtype=DT, device=DEV)
    V = torch.randn(R * N_KV, T, D_H, dtype=DT, device=DEV)
    q = torch.randn(R * N_KV, G, D_H, dtype=DT, device=DEV)
    batch = statistics.median(timeit(lambda: attn(q, K, V)) for _ in range(3)) * 1e3
    del K, V, q
    if DEV == "cuda": torch.cuda.empty_cache()

    print(json.dumps(dict(R=R, spec_ms=round(spec, 4), batch_ms=round(batch, 4),
                          batch_over_spec=round(batch / spec, 2))), flush=True)
```

**Predictions.**
1. The batch arm is **linear in `R` from `R`=1**, at constant achieved GB/s. Bytes and FLOPs
   both scale with `R`, so intensity is invariant — `AI_attention(B) = 2G/b`, `B` cancels.
2. The spec arm is flatter, so `batch_over_spec` grows with `R`.
3. If the spec arm were flat (as the roofline claims), `batch_over_spec` would equal `R`.

`[M]` **Reference numbers**, same configuration as Exercise A, `T` = 32,768, default
(unfused) attention path:

| `R` | spec ms | batch ms | batch ÷ spec | if spec were free |
|---|---|---|---|---|
| 1 | 0.8542 | 0.9478 | 1.11 | 1 |
| 2 | 1.0451 | 1.7572 | 1.68 | 2 |
| 4 | 1.6574 | 3.4250 | 2.07 | 4 |
| 8 | 2.7909 | 6.9049 | 2.47 | 8 |
| 16 | 4.8140 | 13.8153 | 2.87 | 16 |
| 32 | 8.4638 | 26.7898 | **3.17** | 32 |

Batch-arm achieved bandwidth: 141.6, 152.8, 156.8, 155.5, 155.4, 160.3 GB/s — **flat**.
Prediction 1 holds exactly: batching buys nothing on the attention term and the machine
agrees to within noise across a 32× range.

Prediction 3 is the interesting one. **The theory says speculation should beat batching by
32× at matched rows; the machine delivers 3.17×** — and Exercise A already told you the whole
of the shortfall is the score-matrix traffic, which is `O(R·T)` in the spec arm and equally
`O(R·T)` in the batch arm. The two arms have the same score bytes; they differ only in KV
bytes. So the measured ratio is exactly

```
(KV + score)_batch      R·kv  +  R·s          R·(kv + s)
──────────────────  =  ────────────────  =  ──────────────      → R only when s → 0
(KV + score)_spec        kv   +  R·s           kv + R·s
```

At `R`=32, `kv` = 134 MB and `s` ≈ 47 MB/row: `32×181 / (134 + 1510)` = **3.52**, against a
measured 3.17. The model predicts the shortfall to 10%. That is the exercise: not that the
theory was wrong, but that a two-line correction to the byte count reproduces the machine.

**Now the same sweep on the fused kernel.** Prediction, which was written down before the run:
the spec arm goes flat, the batch arm stays linear, and `batch_over_spec` approaches `R`.

`[M]` Same configuration, both arms on `F.scaled_dot_product_attention` with
`enable_gqa=True`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `T` = 32,768, a third fresh
process:

| `R` | spec ms | batch ms | batch ÷ spec | spec vs its own `R`=1 | batch vs its own `R`=1 |
|---|---|---|---|---|---|
| 1 | 4.1884 | 3.7891 | 0.90 | 1.00 | 1.00 |
| 2 | 4.1070 | 5.4793 | 1.33 | 0.98 | 1.45 |
| 4 | 4.2122 | 9.7940 | 2.33 | 1.01 | 2.58 |
| 8 | 4.1086 | 19.5497 | 4.76 | 0.98 | 5.16 |
| 16 | 4.1416 | 38.5981 | 9.32 | 0.99 | 10.19 |
| 32 | 5.0912 | 79.6507 | **15.64** | **1.22** | **21.02** |

**The prediction holds and this is the module's cleanest single result.** Thirty-two
speculative rows cost 1.22× one row. Thirty-two batched rows cost 21.02×. At matched query
rows and matched FLOPs, speculation beats batching by **15.6×**, and the spec arm's flat
segment reproduces Exercise A's independently in a third process (4.11–4.21 ms there,
3.77–4.15 ms in Exercise A — an 11% spread between processes, well inside the effect).

Two honest notes. The batch arm is slightly *super*linear (21.0× at 32× the rows), which is
unexplained and probably an allocator or residency effect at 4.3 GB of live caches; it flatters
the ratio by about 5%. And `batch_over_spec` reaches 15.6 rather than the ideal 32 because the
spec arm has entered its knee at `R` = 32 — exactly where §3.4 said it would, at
`K_free ≈ 16.5`.

**Deliverables — three numbers.**
1. `batch_over_spec` at your largest `R`, on both backends. Ours: 3.17 default, 15.64 fused.
2. The batch arm's achieved GB/s across the default-backend sweep. A flat line is the claim
   "batching cannot raise attention intensity," measured. If it rises, your caches are being
   shared somewhere.
3. The spec arm's `t(R)/t(1)` at `R` = 16 on both backends. Ours: 5.46 default, 0.99 fused.
   **That pair of numbers is the whole module in two measurements.**

---

### Exercise C — acceptance is a survival curve, not a rate

**Goal:** measure `α` for a model-free drafter on real token streams, and test the i.i.d.
assumption that every speedup formula in the literature is built on. **No model, no GPU, no
tokenizer.**

**Hardware:** any CPU. **Runtime:** under one second per stream. This exercise has no GPU path
and needs none.

```python
"""Replay vLLM's n-gram drafter (ngram_proposer.py:207) over a real byte stream and
collect the survival curve vLLM logs as its per-position acceptance rate."""
import sys, time
from pathlib import Path
K, MIN_N, MAX_N, WINDOW, SAMPLES = 8, 3, 8, 4096, 20000

def draft_at(data: bytes, t: int, k: int) -> bytes:
    lo = max(0, t - WINDOW)
    for n in range(min(MAX_N, t - lo), MIN_N - 1, -1):
        j = data.rfind(data[t - n:t], lo, t - 1)      # latest earlier occurrence
        if j != -1:
            return data[j + n: j + n + k]
    return b""

data = Path(sys.argv[1]).read_bytes()
acc_per_pos = [0] * K; n_drafts = n_acc = n_drafted = 0
start, stop = MAX_N + 1, len(data) - K - 1
step = max(1, (stop - start) // SAMPLES)
t0 = time.perf_counter()
for t in range(start, stop, step):
    d = draft_at(data, t, K)
    if not d: continue
    n_drafts += 1; n_drafted += len(d)
    a = 0
    for i, tok in enumerate(d):
        if data[t + i] != tok: break
        a += 1
    n_acc += a
    for i in range(a): acc_per_pos[i] += 1

print(f"drafts {n_drafts}  accepted {n_acc}/{n_drafted} = {n_acc/n_drafted:.4f}"
      f"  ({time.perf_counter()-t0:.2f}s)")
print(f"mean acceptance length (metrics.py:114) = {1 + n_acc/n_drafts:.4f}")
prev = 1.0
for i in range(K):
    s = acc_per_pos[i] / n_drafts
    print(f"  i={i}  S[i]=P(a>i)={s:.4f}   conditional={s/prev:.4f}")
    prev = s
a0 = acc_per_pos[0] / n_drafts
print(f"geometric model with alpha={a0:.4f} predicts {(1-a0**(K+1))/(1-a0):.4f}")
```

**Predictions.**
1. Acceptance is far higher on source code than on prose — that is why n-gram drafting is
   recommended for code editing (`docs/speculative.md:169`).
2. The conditional per-position acceptance rate is roughly **flat**, because the i.i.d. model
   that every speedup formula uses assumes exactly that. **This prediction failed on all three
   streams.**

`[M]` **Reference numbers.** Python 3.12, byte tokens, `K`=8, n-gram ∈ [3,8], 4,096-token
lookback, ~20,000 sampled positions per stream, three separate processes, 0.16–0.30 s each,
2026-07-26. The scheduler.py row was run twice in separate processes and reproduced to every
digit printed (deterministic by construction — no RNG anywhere).

| Stream | drafts | acceptance rate | `τ̄` measured | `τ̄` geometric | ratio |
|---|---|---|---|---|---|
| `sched/scheduler.py` (Python) | 18,696 | 0.4950 | **4.9603** | 4.0667 | **1.220** |
| `speculative.cpp` (C++) | 18,337 | 0.5129 | **5.1034** | 4.3866 | **1.163** |
| `kv-cache-mechanics.md` (prose) | 17,372 | 0.2275 | **2.8202** | 2.2308 | **1.264** |

Prediction 1 holds: code accepts at 0.50–0.51, prose at 0.23.

**Prediction 2 failed, and this is the finding.** The conditional acceptance rate
`S(i)/S(i−1)` *rises monotonically with position* on every stream:

| position `i` | scheduler.py | speculative.cpp | prose |
|---|---|---|---|
| 0 | 0.7805 | 0.8041 | 0.5539 |
| 1 | 0.8494 | 0.8536 | 0.6644 |
| 2 | 0.8631 | 0.8554 | 0.7039 |
| 3 | 0.8685 | 0.8642 | 0.7454 |
| 4 | 0.8776 | 0.8887 | 0.7723 |
| 5 | 0.8796 | 0.8794 | 0.8001 |
| 6 | 0.8722 | 0.8935 | 0.8123 |
| 7 | 0.8756 | 0.8940 | **0.8308** |

**Acceptance is strongly positively autocorrelated within a draft: having been right so far is
evidence that you will keep being right.** The mechanism is not mysterious — a draft that
survives its first token was probably drawn from a genuinely repeated region — but its
consequence is quantitative and unflattering to the standard analysis:

> The geometric model `τ̄ = (1 − α^{K+1})/(1 − α)`, fitted at `α = S(0)`, **understates the true
> mean acceptance length by 16–26%** on real streams. Every speedup formula in the
> speculative-decoding literature is built on that model.

The direction matters: the standard analysis is **pessimistic**, so real drafters do better
than their fitted `α` predicts, and — more usefully — **the optimal `K` is larger than the
i.i.d. model says**, because the marginal value of draft position `i` is not decaying as fast
as `α^i`. That is a free, falsifiable, cheap-to-check correction to a design rule.

`[A]` **The honest caveat, and it is real.** These are *byte* tokens, not BPE tokens, and a
byte-level match of length `n` is a weaker match than a subword match of length `n`. The
absolute `α` values are therefore not comparable to any published number. What transfers is
the **shape** — a rising conditional curve — and that shape has a mechanism (repetition
structure in the source) that a BPE tokenizer does not remove. Medium confidence that it
survives tokenization; the cheapest test is to rerun with a real tokenizer over the same files.

**Deliverables — three numbers.**
1. The measured/geometric ratio on a code stream and a prose stream. Ours: 1.22 and 1.26.
2. The conditional acceptance rate at `i`=0 and `i`=7. If the second is not higher, your
   stream has no repetition structure and you should say so.
3. Compute the §3.3 speedup at your measured `τ̄`, once with `r(K)` = 1 (fused path, measured
   flat to 16 rows) and once with the `r(8)` = 3.32 that Exercise A's fit gives for nine rows
   on the default path. Ours: **4.96× versus 1.49×**. That gap is the attention backend, and it
   is larger than the gap between any two drafters in the §2.4 taxonomy.

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. Batching multiplies both bytes and FLOPs in the attention term, so it cannot raise decode
   arithmetic intensity. Name the one mechanism in the reference library that *is* an exception,
   say why it does not contradict the algebra, and say why we cannot run it.

2. A dashboard reports, for a `K`=8 drafter: "draft acceptance rate 0.4950" and "mean
   acceptance length 4.9603." Are those consistent? Show the arithmetic. Then say what the
   panel labelled "per-position acceptance rate: 0.78, 0.85, 0.86, …" actually is, cite the
   line, and give the formula for the quantity a reader almost certainly wants instead.

3. Laguna-S has 12 global layers at `G`=6 and 36 sliding layers at `G`=9. Using the `[M]` ≈105
   ridge, give `K_free` for each, say which binds, and say in one sentence why that is the same
   minority of layers that `kv-cache-mechanics.md` Exercise B found anomalous.

4. An n-gram drafter returns token ids and no probabilities. Is the output distribution still
   exactly the target model's? Cite the two lines that decide it, and give the closed form for
   that drafter's acceptance probability.

5. You are told speculation "shrinks the batch a given memory budget supports." Quantify it for
   Laguna-S at `K`=7, `B`=8, at `T`=32,768 and at `T`=128. Then name the memory cost that
   framing omits and give its order of magnitude relative to the one it names.

6. Exercise A measured the free row window on the KV stream at 2 with one attention backend and
   16 with another, on identical shapes and the same GPU. Explain the 8× in terms of bytes, and
   say what that implies for the field's practice of reporting speculative-decoding speedups.

---

## 8. What is still unsolved here

### 8.0 First, what this module could **not** do on this machine

Said plainly, because a module that teaches a technique as if we can run it is worse than one
that says we cannot.

- **No end-to-end speculative decoding was run.** Everything measured here is a
  microbenchmark of the *shape* of a verify pass. There is no validated vLLM build for gfx1151
  in this repo, no Laguna weights on disk (`ASSUMPTIONS.md → reference-model` records a
  **config** fetched at revision `b0a9fd7c850e`, not weights), and Laguna S 2.1 at 118B would
  not fit in bf16 regardless. Every `α`, every acceptance length, and every speedup for a
  *trained* drafter in this module is `[C]`, cited, never measured here.
- **No trained drafter exists locally**, so §4.3's `Δα` hypothesis is stated, costed, and
  un-run. Exercise C measures `α` for a model-free drafter only, on byte tokens.
- **No disaggregated arms, ever, until we rent.** `ASSUMPTIONS.md → single-device-only`:
  collectives are incomplete on gfx1151. Prefill/decode disaggregation (`[C]` 2401.09670,
  `[C]` 2311.18677) is design-only. Do not simulate it and report the simulation as evidence.
- **No cascade / shared-prefix kernel and no tree-attention mask on ROCm.** The
  intensity-multiplying mechanism in `paged-attention-and-prefix-reuse.md` §3.5 and the tree
  drafting of `[C]` 2401.10774 both need kernels we do not have. Self-check 1 turns on this.
- **No FP8 or FP4 arithmetic.** `torch._scaled_mm` is unsupported here `[M]`, so any
  quantization × speculation result from this machine is a memory-economics result only.
- **The Hardware Validation Gate has not run.** Nothing in this module is evidence by house
  standard. Timing claims are sound; anything touching numerics is provisional
  (`bf16-numerics-unproven`), and that includes every number taken with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, which is an experimental kernel and therefore a
  numerics change.

### 8.1 The frontier

Everything below is testable at 20M–300M params on one gfx1151 device with a `[M]` ≥62 GiB
fast tier and a hard 31 GiB per-buffer limit, unless marked otherwise. Each needs a
pre-registered hypothesis card.

1. **Settled here, and it should go in the register: speculation raises attention intensity
   and batching does not.** Survey open question #3 asked whether speculation raises attention
   arithmetic intensity by `(K+1)` as derived. `[M]` On the fused path it does — flat to
   `R` = 16, knee where `K_free = I*/G − 1` said, 15.6× better than batching at matched rows.
   On the default path it does not, and the reason is 30 bytes per score element. Both halves
   belong in `ASSUMPTIONS.md`; the second is the one that changes how this lab reports results.

2. **Does `enable_gqa` undo the GQA bandwidth saving on the fused ROCm path?** `[M]` `gqa48`
   and `expand48` agree to 10% at 6× different stored bytes, which says yes; the `mha8` control
   is inconclusive because at eight heads the kernel is occupancy-bound at 39.9 GB/s rather
   than bandwidth-bound. `[A]` Medium confidence. Repeat rows 1–2 of the probe at `R` = 16
   where both are bandwidth-bound — fifteen minutes. If it holds, the GQA *bandwidth* saving
   does not exist on this stack, only the capacity saving, and that touches every
   arithmetic-intensity claim in Tracks B and C on this machine.

3. **Acceptance autocorrelation is unmeasured for trained drafters.** Exercise C shows a
   strongly rising conditional curve for a model-free drafter. Nobody reports the conditional
   curve for EAGLE, Medusa, MTP or DFlash — both engines log the survival function and neither
   differentiates it. If the effect is as large there, the field's optimal-`K` guidance is
   systematically too small. The data is already being collected; only the division is missing.

4. **Does eviction change `α`?** §4.3. Highest novelty-per-GPU-hour item here. A model-free
   drafter removes the confound and makes it a one-afternoon experiment at 300M.

5. **The draft KV cache has no policy in any engine.** Nobody evicts it, compresses it, offloads
   it, or reports its size. At DFlash's several layers it is a non-trivial share of the
   contended resource. Unclaimed ground for Mnemosyne.

6. **Does an n-gram drafter beat a trained one at ablation scale?** Survey open question #5. At
   20M–300M the target is so cheap that a trained drafter's fixed cost may never amortise, and
   `c_d ≈ 0` for the n-gram path makes its break-even acceptance essentially zero (§3.5). If the
   free drafter wins at our scale, every small-scale speculative-decoding ablation in the
   literature carries a confound.

7. **Tree drafting is probably a bad trade on a bandwidth-poor machine, and nobody has checked.**
   A tree multiplies verify rows without multiplying accepted tokens proportionally. Exercise A
   says rows past the free window cost real time, and our free window is small. `[C]` 2607.06763
   (Jul 2026) is the current entry; vLLM's Medusa proposer ships a flat chain rather than the
   paper's tree (`medusa.py:56`), which may be exactly this reasoning, undocumented.

8. **Contested, and left contested: does speculation survive high batch?** `[C]` MagicDec
   (2408.11049) and `[C]` 2310.18813 say yes at long context; `[C]` 2504.17674 reports 25.65%
   *more* energy at batch 128; `[C]` 2508.08192 is a production account of making it work rather
   than an argument that it should. §3.4 derives our own answer — a ~43-row free window survives
   any batch at 32k, collapsing to ~9 at 1k — but that is arithmetic over two single-run
   measurements, not an experiment.

9. **Not testable here: disaggregation.** `[C]` DistServe (2401.09670) / Splitwise (2311.18677)
   versus `[C]` Sarathi-Serve (2403.02310) is the live scheduling debate, and every disaggregated
   arm requires collectives. `ASSUMPTIONS.md → single-device-only` says we have none. **Design
   only.** Do not simulate it and report the simulation as evidence.

10. **Quantization × speculation is one object and is treated as two.** `[C]` 2607.04244 (Jul
    2026) is the first entry to take the interaction seriously: a quantized drafter has lower
    acceptance but costs less per draft, and the product is not obviously monotone in bit width.
    Two axes, one matched token budget, one afternoon at our scale — but see the standing caveat
    that `torch._scaled_mm` is unsupported on gfx1151, so we would be measuring memory economics
    only.

11. **Our ridge point is a ratio of two single-run numbers and everything here hangs off it.**
    Survey open question #7 asks for the ridge of the *attention kernel* rather than of an 8192³
    GEMM. Exercise A part 1 is the first evidence that ≈105 is approximately right on the weight
    stream. The attention-stream equivalent is now within reach precisely because the fused path
    exists: the knee between `R`=16 and `R`=32 in the fused column *is* that measurement, and
    tightening the sweep to `R` ∈ {16,20,24,28,32} would pin it in four minutes.

---

## Answers to the self-check

**1.** **Cascade / shared-prefix attention** — `memory/flashinfer/flashinfer/cascade.py:226`
(`MultiLevelCascadeAttentionWrapper`), covered in `paged-attention-and-prefix-reuse.md` §3.5.
It does not contradict the algebra; it changes the premise. The algebra assumes each sequence
owns its own cache, so `B` multiplies bytes. A cascade kernel reads the *shared* pages once for
all `m` sequences, so bytes stop scaling with `m` on the shared portion and intensity becomes
`m · 2G/b`. That is sharing, not batching — the mechanism is prefix reuse, and the win exists
only on the shared prefix. We cannot run it: there is no cascade kernel in our ROCm stack, which
makes it the strongest single argument for a costed rental in the backlog.

**2.** Consistent. `K` = 8, so drafted tokens = 8 × drafts; acceptance rate = `E[a]/K`; mean
length = `1 + E[a]` (`metrics.py:114`). Therefore `mean = 1 + K × rate` = `1 + 8 × 0.4950` =
**4.9600**, matching 4.9603 to rounding. The "per-position acceptance rate" panel is **not** a
per-position rate: `metrics.py:46`–`:49` increments position `i`'s counter whenever the accept
length *exceeded* `i`, and `:117` divides by `num_drafts`, so the logged vector is the survival
function `S(i) = P(a > i)`. The quantity a reader wants — the probability that position `i` is
accepted *given that the draft reached it* — is `S(i)/S(i−1)`, with `S(−1) = 1`. On the
Exercise C data those differ by 12 percentage points at `i` = 1 (0.663 versus 0.849), and the
difference is not noise; it is the whole autocorrelation finding.

**3.** `K_free = 105/G − 1`: global layers `105/6 − 1` = **16.5**, sliding layers `105/9 − 1` =
**10.7**. **The sliding layers bind**, because a higher `G` means fewer bytes per FLOP and
therefore less headroom before the verify pass goes compute-bound. It is the same minority of
layers because both effects come from the same source: those 36 layers hold very little data
(512 tokens each) and issue very small reads, so they are the layers whose cost is dominated by
something other than streaming KV — in Exercise B of `kv-cache-mechanics.md` it was per-launch
overhead on a 2 MiB read; here it is that their arithmetic intensity is already 50% higher.
Small tiers behave badly, twice, for two unrelated reasons.

**4.** **Yes, exactly the target's distribution.** `rejection_sampler.py:816` sets
`draft_prob = 1` when no draft distribution is supplied, and `:913`–`:918` builds the residual
as the target distribution with the drafted id masked out. That is not an approximation: a
deterministic drafter *is* the point mass `q = δ_{x̂}`, for which `q(x̂) = 1` and the residual
`max(0, p − q)` renormalises to `p` with `x̂` removed. Substituting into the rule at `:829`:

```
P(accept)  =  min(1, p_target(x̂) / 1)  =  p_target(x̂)          →    α = E[ p_target(x̂) ]
```

**A deterministic drafter's acceptance probability is exactly the target model's own
probability mass on the guessed token** — which is why `α` rises when the target gets more
confident, and why it is a probe of the target's distribution rather than only of the drafter.

**5.** Lookahead reservation is `K/(T+K)` of live KV. At `T` = 32,768: `7/32,775` = **0.021%**,
i.e. 10.5 MiB against 48.0 GiB `[M]` at 192 KiB/token. At `T` = 128: `7/135` = **5.2%**. So the
claim is materially true only at short context and is essentially zero in the long-context
regime this lab studies — speculation is memory-cheap exactly where it is bandwidth-valuable.
**The omitted cost is the drafter's own KV cache**, `k_draft/k_target = (L_d·n_kv,d)/(L·n_kv)`.
`[A]` For a one-layer EAGLE-3 drafter matching Laguna's KV geometry that is 1/48 = 2.1%, or
about **1.0 GiB** at `B`=8, `T`=32,768 — roughly **100×** the lookahead reservation it is
usually contrasted with. Budget the cache, not the slots.

**6.** The 8× is bytes the roofline does not count. The model charges a verify pass with the KV
read only, `2·n_kv·T·d_h·b`, which is invariant in `R`. An unfused implementation additionally
materialises the score matrix, `n_kv·G·R·T` elements, and traverses it about seven times — `[M]`
30 bytes of traffic per element, a measured slope of 47.2 MB per query row against a 134 MB KV
read. So the true byte count is `kv + s·R`, the intensity stops growing at `R ≈ kv/s ≈ 3`, and
the free window collapses from 16 to 2. A fused kernel keeps the scores in registers, `s → 0`,
and the roofline is right again — measured flat to `R` = 16, exactly as predicted.

**The implication for the field is uncomfortable and worth saying plainly.** A speculative
decoding speedup is a ratio in which `r(K)` is the denominator, and `r(K)` is set by an
implementation detail of the attention kernel that no paper reports and no config file records.
On identical shapes and identical hardware, this module measured the same quantity as 2 and as
16 depending on one environment variable. **Any reported speculative-decoding speedup that does
not state its attention backend is not a measurement of speculative decoding.** It is a
measurement of the backend. Our own house rule follows directly: `attention_backend` and the
AOTriton flag go in the run header, and a run without them is not evidence.

---

## Sources

**Local measurements (`[M]`, this session, 2026-07-26)**

All GPU numbers: Z13, Radeon 8060S (gfx1151), native Windows 11, torch
`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, bf16, hipBLASLt configured
(`HIPBLASLT_TENSILE_LIBPATH` set, `TORCH_BLAS_PREFER_HIPBLASLT=1`), median of 3 runs of 20–30
timed iterations after 5 warmups. Shapes stated in each table. Every table was produced in a
fresh process, and the two load-bearing sweeps (KV stream, Exercise A part 2) were repeated in
a second fresh process and agreed to within 5%.

- Weight-stream free window: 64 rows within 10% of the 1-row time; knee between 64 and 128
  against a roofline prediction of 105.
- KV-stream free window, default backend: 2 rows; `t(R) ≈ 0.847 + 0.2362·R` ms at `T`=32,768;
  ≈30 bytes of traffic per score-matrix element.
- KV-stream free window, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`: flat 3.769 → 4.148 ms
  from `R`=1 to `R`=16, knee between 16 and 32, against a roofline prediction of 16.5.
  Independently reproduced in a third process at 4.109 → 4.212 ms over the same range.
- Default SDPA versus fused SDPA: 7.4× at `R`=1, 16.1× at `R`=64.
- Batching arm, default backend: linear in `R` at 141.6–160.3 GB/s across `R` = 1…32.
- Speculation versus batching at matched rows, fused backend, `R`=32: 5.09 ms versus 79.65 ms,
  a ratio of **15.64**; spec `t(32)/t(1)` = 1.22, batch `t(32)/t(1)` = 21.02.
- GQA probe at `R`=1, fused backend: `gqa48` 4.376 ms, `expand48` 4.833 ms, `mha8` 3.366 ms.
- n-gram acceptance survival, three streams, byte tokens, `K`=8: `τ̄` = 4.960 / 5.103 / 2.820
  against geometric predictions 4.067 / 4.387 / 2.231; conditional acceptance rises
  monotonically with position on all three.

**`ASSUMPTIONS.md` rows relied on**: `gemm-throughput-below-reference` (20.9 TFLOP/s bf16),
`gpu-fast-tier-size` (≥62 GiB at ~199.9 GB/s), `large-tensor-fault-32gib`,
`sdpa-is-memory-efficient`, `kv-per-token-laguna` (192 KiB/token), `reference-model`,
`laguna-heads-uniform` (`G` = 6 global / 9 sliding), `hipblaslt-config`,
`bf16-numerics-unproven`, `single-device-only`, `torch-build`.

**Code, at the revisions in `research/reference/PROVENANCE.md`**

- `memory/vllm` @ `0934b267906f`: `vllm/v1/sample/rejection_sampler.py:40`, `:816`, `:829`,
  `:839`, `:930`; `vllm/v1/spec_decode/ngram_proposer.py:207`; `vllm/v1/spec_decode/medusa.py:56`;
  `vllm/v1/spec_decode/eagle.py:20`; `vllm/v1/spec_decode/llm_base_proposer.py:619`, `:682`;
  `vllm/v1/spec_decode/metrics.py:41`, `:114`, `:117`; `vllm/v1/spec_decode/dynamic/utils.py:77`,
  `:107`; `vllm/v1/core/sched/scheduler.py:242`, `:253`, `:256`, `:430`, `:447`, `:509`, `:511`,
  `:569`, `:1124`, `:1216`.
- `architecture/llama-cpp-laguna` @ `04b2b72cb540`: `src/models/dflash.cpp:10`, `:14`, `:153`;
  `common/speculative.cpp:945`, `:958`, `:1178`, `:1207`; `tools/server/server-context.cpp:616`,
  `:620`; `docs/speculative.md:19`, `:60`, `:150`, `:167`, `:190`, `:259`, `:401`.
- `memory/flashinfer/flashinfer/cascade.py:226` (referenced from the prereq module).

**Cited work.** Every arXiv id below appears in the verified source list of
`research/notes/inference-and-quantization.md`, which records that 66 candidate ids were
queried against the live arXiv API on 2026-07-26 with 0 unresolved. Resolving an id proves the
paper exists, not that it supports the claim beside it.

*Speculative decoding*
- `2211.17192` — *Fast Inference from Transformers via Speculative Decoding* (2022-11-30). The acceptance rule.
- `2302.01318` — *Accelerating LLM Decoding with Speculative Sampling* (2023-02-02).
- `2401.10774` — *Medusa* (2024-01-19). `2401.15077` — *EAGLE* (2024-01-26).
- `2406.16858` — *EAGLE-2* (2024-06-24). `2503.01840` — *EAGLE-3* (2025-03-03).
- `2412.19437` — *DeepSeek-V3 Technical Report* (2024-12-27). MTP heads.
- `2602.06036` — *DFlash: Block Diffusion for Flash Speculative Decoding* (2026-02-05). ICML 2026.
- `2408.11049` — *MagicDec* (2024-08-20). `2310.18813` — *The Synergy of Speculative Decoding and Batching* (2023-10-28).
- `2504.17674` — *Energy Considerations of LLM Inference and Efficiency Optimizations* (2025-04-24).
- `2508.08192` — *Efficient Speculative Decoding for Llama at Scale* (2025-08-11).
- `2607.06763` — *Trees from Marginals: Autoregressive drafting with factorized priors* (2026-07-07).
- `2607.04244` — *Quantize the Target, Quantize the Drafter* (2026-07-05).
- `2401.07851` and `2411.13157` — the two surveys, for orientation only.

*Serving and scheduling*
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — **no arXiv id**; cited by venue.
- `2309.06180` — *PagedAttention / vLLM* (2023-09-12).
- `2403.02310` — *Sarathi-Serve* (2024-03-04). `2401.09670` — *DistServe* (2024-01-18).
- `2311.18677` — *Splitwise* (2023-11-30). `2508.01989` — *Prefill-Decode Aggregation or Disaggregation?* (2025-08-04).
- `2607.13068` — *The Economics of AI Decoding Chips* (2026-07-10).

*Referenced for the §4.3 hypothesis*
- `2606.09864` — *Alignment Collapse Under KV Cache Quantization* (2026-06-01).
- `2305.13245` — *GQA* (2023-05-22).
