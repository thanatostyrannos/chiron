---
title: Long context and effective context — advertised capacity, usable capacity, and why the gap has no alarm
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: attention-variants-and-kv-cost, positional-encoding
mirrors: research/memory/long-context-behavior.md §7, research/notes/evaluation-landscape.md §1–3
worked_example: poolside Laguna S 2.1 on the Z13, read from the committed artifact and our own measurements
---

# Long context and effective context

## What this module settles

**Advertised context is a number someone typed into a config file; usable context is what
the memory system will admit; effective context is a measurement against a threshold — and
these are three different numbers that the field routinely reports as one.** The gap between
them is not a single defect but five distinguishable mechanisms, exactly one of which
(capacity) has an error path in any code you will read; the other four fail silently, and
one of them — softmax dilution — is forced by arithmetic rather than by any implementation
choice, which means it cannot be engineered away inside the current attention primitive.
By the end you will be able to compute, from three numbers you already have, how much of
Laguna's advertised 1,048,576-token context our machine can actually hold, state the extra
attention-logit margin (in nats) that a token needs to survive at 1M keys versus 8K, and
say precisely why a benchmark that reports one "effective context length" produces a model
ranking that changes when you change the length grid.

**Prerequisites, and this module does not re-teach them.** From
`attention-variants-and-kv-cost.md`: the KV product `2 · L · n_kv · d_h · b` bytes/token,
GQA, sliding windows, decode arithmetic intensity. From `positional-encoding.md`: RoPE as a
bank of dials, YaRN's per-band ramp and its `attention_factor`, partial rotary, and the
crucial fact that RoPE phase is baked into the cached key before the write. This module is
the *behavioural* half: what happens to model quality as the sequence gets long, and how
you measure it without fooling yourself.

**Relationship to the survey note.** This module teaches
`research/memory/long-context-behavior.md` §7 ("Effective vs advertised context") and its
five-mechanism breakdown. It does not contradict it. Three things here go beyond it, and
each is flagged where it appears: the dilution arithmetic in §3.2, the observation in §3.3
that YaRN's temperature is a *multiplicative* correction for an *additive* deficit, and the
eviction-sharpening result in §4.2 that turns the note's "NIAH is adversely selected"
argument from an intuition into arithmetic.

---

## 1. Theory in plain language

### 1.1 Three numbers, all called "context"

| Name | What it is | Where it is decided | Who can observe it | Fails how? |
|---|---|---|---|---|
| **Advertised** | `max_position_embeddings` — a field | model card / config JSON | anyone, by reading a file | never; it is a claim, not a behaviour |
| **Admissible** | the longest request the serving stack will accept, after KV memory is accounted | the server, at startup | the operator, in a log line | loudly — `ValueError`, or a silent downward *auto-fit* |
| **Effective** | the longest length at which measured quality still clears a threshold | nowhere; it must be measured | only someone who runs an eval | silently, smoothly, with no counter anywhere |

Laguna advertises 1,048,576. That number is `8192 × 128` — the pretraining length times
YaRN's `factor` — and `positional-encoding.md` §2.7 already showed it never enters the RoPE
frequency computation at all (`modeling_rope_utils.py:402`). It is an arithmetic consequence
of a rescaling decision. Nobody measured anything to produce it.

The second number is the one your instincts are already good at. It is capacity planning,
and §3.5 does the arithmetic for our machine. The third number is the one this module is
really about, because it is the one with no instrumentation.

### 1.2 The bridge: 90% full and 100% fragmented

You have owned this problem. A 4 TB volume advertises 4 TB. The filesystem takes some.
Reserved blocks take some. At 90% full, the allocator starts placing files wherever it can,
and by the time the free-extent histogram is all 4 KB fragments, a sequential read of a
large file becomes a random-read workload. Every byte is still there. Every read still
returns correct data. The nameplate number is still true. And the system is unusable for
the job it was bought for.

That is the right first-order picture of a long context. The tokens are all there. Attention
can still reach every one of them — the mask permits it, the KV entries are resident, no
lookup fails. And the model's ability to *use* what is there has degraded to a fraction of
what the number on the box implies.

**Now the three places the analogy breaks, because that is where the teaching is.**

**Break 1: there is no defragmentation operation.** On a disk you run a compaction pass:
read the extents, write them back contiguously, update the metadata. Three candidate
"defrags" exist here and each fails for a different, instructive reason.

- *Reorder the prompt* — put the important material at the beginning and the end, where
  §2.2's position bias says accuracy is highest. This works, and it is what every prompt
  engineer has rediscovered. But it requires knowing which material is important, which is
  the question you were asking the model. It is a defrag that needs an oracle over the very
  relevance ranking you are trying to compute.
- *Re-pack the KV cache* — evict the dead entries and close the gaps. From
  `positional-encoding.md` §2.4: RoPE phase is written into the key *before* the cache write,
  so a cached key carries its absolute position the way a cache line would carry a physical
  address. Re-packing without recomputation changes every surviving token's effective
  distance and corrupts the data silently. You can renumber (`renumber_compact`, which is
  StreamingLLM's deliberate choice `[C]`
  ([2309.17453](https://arxiv.org/abs/2309.17453), 2023)) or preserve (`preserve_original`),
  and both are self-consistent, but neither is a defrag: one lies about distances, the other
  leaves the holes.
- *Compact into a summary* — replace 100K tokens with a 2K-token précis. This is real and
  widely deployed, and it is not a defrag; it is a lossy re-encode with no inverse. A disk
  defrag is information-preserving by definition. If yours were not, you would call it data
  loss.

**Break 2: the degradation is smooth and unannounced.** Fragmentation is measurable — you
can walk the free-extent histogram, you can alarm on it, and there is a well-defined moment
when an allocation fails. The long-context failure has no threshold and no counter. Quality
falls off gradually with length; nothing in the model, the runtime, or the kernel emits a
signal. FlashInfer's page table cannot even *represent* a miss
(`memory/flashinfer/flashinfer/page.py:326` reconstructs sequence length from the page table
and there is no "absent" state). The model returns a fluent, confident, wrong answer at
exactly the same latency as a right one.

**Break 3: the failure mode is inverted.** A fragmented disk degrades *throughput* and
preserves *correctness*. This degrades *correctness* and preserves throughput. Every
observability instinct you have — latency histograms, error rates, saturation — is pointed
at the wrong axis. The thing that gets worse is the only thing you are not measuring.

The cleanest published demonstration of break 3 is, appropriately, a monitoring result.
`[C]` ([2605.12366](https://arxiv.org/abs/2605.12366), May 2026) puts frontier models to work
as classifiers watching for dangerous actions in agent transcripts and finds they miss those
actions **2× to 30× more often** when the action occurs after 800K tokens of benign activity
than when it occurs in isolation. Read that as a systems statement rather than a safety one:
**the detector's recall degrades as a function of how much benign traffic preceded the
event.** You have never operated a monitoring system with that property — `grep` does not get
worse as the logfile grows — and every runbook you have written assumes it cannot happen.

> One place the analogy holds exactly, and it is worth keeping: **occupancy matters.** A
> filesystem behaves differently past a fill ratio, not just past an absolute size. `[C]`
> ([2508.07479](https://arxiv.org/abs/2508.07479), Aug 2025) finds the classic U-shaped
> position bias holds only up to roughly **50% context occupancy**; above that, primacy
> weakens, recency persists, and the bias becomes distance-based. So the same 100K-token
> prompt may sit in two different bias regimes depending on the *configured* window it is
> being measured against. Whether the mechanism tracks the configured limit or the absolute
> length is not settled — see §7.

### 1.3 What replaced what

The measurement discipline has a short, legible history and you should know which artifact
you are quoting.

- **Needle-in-a-haystack (2023, no canonical paper).** Insert "the magic number is 7492"
  into a long document, ask for it back. It became the industry's long-context claim
  because it produces a green heatmap.
- **RULER (2024)** `[C]` ([2404.06654](https://arxiv.org/abs/2404.06654)) replaced it as a
  *methodology*: 13 synthetic task families at controlled length, plus the threshold
  definition of effective context. Its headline finding was that nearly all of 17 models
  claiming ≥32K score near-perfectly on vanilla NIAH and only about half hold up on the
  harder families at 32K. **A benchmark everyone passes cannot rank anything.**
- **NoLiMa (2025)** `[C]` ([2502.05167](https://arxiv.org/abs/2502.05167)) showed *why* NIAH
  passes: it is a lexical-match test. Remove the literal overlap between question and needle
  and 11 of 13 models claiming ≥128K drop below half their own short-context baseline at
  32K; GPT-4o falls from 99.3% to 69.7%.
- **The 2026 stack** replaced RULER as a *result*: LongBench Pro `[C]`
  ([2601.02872](https://arxiv.org/abs/2601.02872), Jan 2026, 46 models, naturally occurring
  tasks) and ATLAS `[C]` ([2605.28079](https://arxiv.org/abs/2605.28079), May 2026, 26 models
  on an 8K–1M grid). The gap did not close. Two more are worth knowing by name: HELMET `[C]`
  ([2410.02694](https://arxiv.org/abs/2410.02694), Oct 2024) states directly that NIAH does
  not reliably predict downstream performance and that its seven application categories have
  *low mutual correlation*; PredicateLongBench `[C]`
  ([2607.08284](https://arxiv.org/abs/2607.08284), Jul 2026) attacks the complaint that
  existing suites measure only the average case, replacing the needle with a
  predicate-satisfaction search over the whole input.

Cite RULER for harness design. Cite something from 2026 for numbers. And treat any
model-card long-context claim as vendor-run: a lead, never evidence.

---

## 2. The five mechanisms, and which of them can even be seen

`research/memory/long-context-behavior.md` §7 lists five causes of the advertised/effective
gap. Here they are as an engineer's fault table — what each predicts, how to tell it from
the others, and whether anything in the stack will tell you it is happening.

| # | Mechanism | What it predicts | Discriminating test | Signal exists? |
|---|---|---|---|---|
| 1 | **Position bias** | accuracy depends on *where* in the input the evidence sits, not just how long the input is | sweep needle position at fixed length; look for U-shape or recency ramp | none |
| 2 | **Attention dilution** | accuracy falls with the *number of competing keys* even at fixed evidence position and fixed distance | hold position and distance fixed, pad with irrelevant tokens | none |
| 3 | **Untrained positional phase** | failure appears past the pretraining length and is repaired by rescaling — PI `[C]` ([2306.15595](https://arxiv.org/abs/2306.15595)) or YaRN `[C]` ([2309.00071](https://arxiv.org/abs/2309.00071)) | compare at `L_train` vs `2·L_train` with and without rescaling | none |
| 4 | **Numerics** | error grows with *absolute* position, not distance; dtype-dependent | run the same prompt in bf16 and fp32 | none |
| 5 | **Capacity** | the request is refused, or the server quietly shortens its own limit | read the startup log | **yes — this is the only one** |

Read that last column twice. Four of the five mechanisms have no counter, no log line, no
exception, and no metric anywhere in any serving stack you will read. The fifth is
implemented as a binary search and prints a friendly message. The field's instrumentation is
concentrated entirely on the one failure that already announces itself.

Mechanisms 3 and 4 are taught in `positional-encoding.md` (§2.6 for untrained phase, §2.10
for the dtype floor) and are not repeated. Mechanism 1 gets §2.2 below; mechanism 2 gets the
mathematics in §3.2 because it is the one that cannot be fixed by better training.

### 2.1 A worked distinction: length versus distance versus occupancy

Three variables get conflated in almost every paper, and separating them is most of what a
good long-context experiment does.

- **Length** `N` — how many keys the softmax runs over.
- **Distance** `δ` — how far the evidence sits from the query.
- **Occupancy** — `N` as a fraction of the configured window.

A single "accuracy at 128K" number moves all three at once. Any experiment that does not
hold two fixed while varying the third produces a result you cannot attribute. This is the
same discipline as the confounded `partial_rotary_factor` sweep in `positional-encoding.md`
§2.8: if two things moved, you measured neither.

### 2.2 Position bias, stated as contested

`[C]` **Lost in the Middle** ([2307.03172](https://arxiv.org/abs/2307.03172), 2023):
accuracy is highest when the relevant information is at the start or the end of the input
and degrades sharply in the middle. This is the most-cited long-context result in existence
and it is *not* settled as to cause or curability.

- **Architectural.** `[C]` ([2602.16837](https://arxiv.org/abs/2602.16837), Feb 2026, rev.
  May 2026) derives U-shaped influence profiles from causal masking plus residual
  connections. If that is right, more long-context training will not remove it.
- **Correctable.** `[C]` ([2406.16008](https://arxiv.org/abs/2406.16008), Jun 2024) recovers
  up to **15 percentage points** by calibrating the attention bias at inference time.
- **Partly correctable, and that is not enough.** `[C]`
  ([2606.27793](https://arxiv.org/abs/2606.27793), Jun 2026) tests debiasing directly on a
  one-pass attention-sorting task: on one model debiasing produces *identical* results to
  the uncalibrated baseline, and where it helps it still trails iterative re-ordering by
  **14.84 pp**. Their conclusion — repeated reordering buys something that bias correction
  does not — is the sharpest available evidence that position bias is a symptom rather than
  the disease.
- **Occupancy-dependent.** `[C]` ([2508.07479](https://arxiv.org/abs/2508.07479), Aug 2025),
  per §1.2.

Do not let a curriculum assert one of these. What you *can* assert: the bias is real,
reproducible, and large enough that where you put the evidence changes the answer — which
is why every eval in §6 sweeps position, and why "average accuracy at length L" hides the
effect it is supposed to measure.

### 2.3 The training-side answers, and the one measurement rule they all agree on

Everything above is inference-time. The other half of the field attacks the gap during
training, and the four current positions are worth knowing because they constrain what our
own runs should look like.

- **Continued pretraining at long window.** `[C]` ProLong
  ([2410.02660](https://arxiv.org/abs/2410.02660), Oct 2024): Llama-3-8B, 40B tokens, 64K then
  512K sequence length, long data mixed with high-quality short data or general ability
  regresses. Their headline methodological finding is the rule this module is built on —
  **they explicitly reject perplexity and bare needle-in-a-haystack as progress signals** —
  and their second is *train beyond your evaluation length*.
- **Schedule the window rather than fixing it.** `[C]` SkyLadder
  ([2503.15450](https://arxiv.org/abs/2503.15450), Mar 2025, rev. Dec 2025): under a *fixed
  token budget*, models pretrained with shorter windows beat their long-context counterparts,
  and scheduling the window upward gives up to 3.7% benchmark gain and 22% faster training at
  1B/3B over 100B tokens. Stage 1 is a schedule, not a constant.
- **Randomise the positions instead of the data.** `[C]` Randomized YaRN
  ([2606.23687](https://arxiv.org/abs/2606.23687), Jun 2026): sample YaRN position encodings
  from a range *larger* than the training sequences actually span, train entirely under 8K,
  improve 16K–128K results.
- **Fix the supervision, not the window.** `[C]`
  ([2605.10544](https://arxiv.org/abs/2605.10544), May 2026) argues the real defect is a
  token-level supervision mismatch — under packed training with document masking, each target
  token's *actually usable* context stays short even though the sequences are long. They
  report +10.69 RULER and +10.09 NoLiMa on **Qwen2.5-0.5B**, which is the closest thing to
  our scale that anyone has published on this question (§6, Exercise C).

> **Systems bridge.** This is a rolling upgrade with a canary: you change the addressing
> scheme, re-warm on representative traffic, and validate at a length beyond the one you
> intend to serve.
>
> **Where it breaks:** there is no rollback. The extension overwrites the weights in place,
> and a model that lost short-context ability during stage 2 does not fail a health check —
> it reports fine on perplexity, which is exactly the signal ProLong tells you not to trust.

---

## 3. The math that actually matters

### 3.1 Symbols, all of them

| Symbol | In words |
|---|---|
| `N` | number of keys the softmax normalises over — the context length at this query |
| `ℓ_j` | the pre-softmax attention logit for key `j` (after scaling by `1/√d_h`, after any YaRN temperature) |
| `p_j` | the attention weight on key `j`; `p_j = e^{ℓ_j} / Σ_k e^{ℓ_k}`, and `Σ_j p_j = 1` exactly |
| `Δ` | the **logit margin** of a distinguished key (the "needle") over the background: `Δ = ℓ_needle − ℓ_background` |
| `H` | attention entropy in **nats**, `H = −Σ_j p_j ln p_j`; maximum is `ln N` when all weights are equal |
| `t` | YaRN's `attention_factor`; the logit is multiplied by `t²` (see `positional-encoding.md` §2.7) |
| `s` | YaRN's `factor` — the context scale factor. Laguna full-attention layers: 128 |
| `m` | the fraction of total attention *mass* removed by an eviction policy |
| `T` | context length in tokens, when talking about bytes rather than about softmax |
| `b` | bytes per stored element (bf16: 2) |

One convention, stated once: **entropy and margins are in nats** (natural log) throughout,
because `e` is what softmax uses. Divide by `ln 2 = 0.6931` for bits. Mixing the two is the
single most common arithmetic error in this material.

### 3.2 The dilution theorem — why length costs you, in nats

Set up the cleanest possible case. One key is relevant. `N − 1` keys are background, all at
the same logit. The relevant key leads by margin `Δ`. Then

```
p_needle  =  e^Δ / ( e^Δ + (N − 1) )
```

**In words:** the needle's share of the attention budget is its exponentiated margin divided
by that plus one unit for every competing key. Solve for the margin required to hold a share
`p`:

```
Δ(N, p)  =  ln( (N − 1) · p / (1 − p) )
```

**In words:** *the logit margin needed to keep a fixed share of attention grows like the
natural log of the number of competitors.* Computed `[M]` (deterministic arithmetic,
2026-07-26; reproduce it with three lines of `math`):

| `N` | uniform weight `1/N` | max entropy `ln N` (nats) | `Δ` for 50% | `Δ` for 90% | `Δ` for 99% |
|---|---|---|---|---|---|
| 512 | 1.95e-3 | 6.238 | 6.236 | 8.434 | 10.831 |
| 4 096 | 2.44e-4 | 8.318 | 8.318 | 10.515 | 12.912 |
| **8 192** | 1.22e-4 | 9.011 | **9.011** | 11.208 | 13.606 |
| 32 768 | 3.05e-5 | 10.397 | 10.397 | 12.594 | 14.992 |
| 131 072 | 7.63e-6 | 11.784 | 11.784 | 13.981 | 16.379 |
| **1 048 576** | 9.54e-7 | 13.863 | **13.863** | 16.060 | 18.458 |

Read three things off this table.

**(a) The cost of length is logarithmic, which is the good news.** Doubling the context costs
`ln 2 = 0.693` nats of extra margin. That is why long context works at all. If the cost were
linear nothing past a few thousand tokens would function.

**(b) Going from Laguna's training length to its advertised length costs exactly `ln 128`.**
`ln(1048576) − ln(8192) = ln 128 = 4.852` nats `[M]`. (Using `ln(N−1)` instead gives 4.852151
— the same number to four figures.) Remember this constant; §3.3 shows it appearing twice
more, from a completely different direction.

**(c) Softmax cannot abstain.** `Σ p_j = 1` is not a design choice, it is the definition. There
is no way for attention to say "none of these keys are relevant"; the mass gets allocated
whether or not anything deserves it. `[C]` ([2506.16640](https://arxiv.org/abs/2506.16640),
Jun 2025, rev. Mar 2026; ICLR 2026) states the asymptotic version: softmax attention entropy
approaches the maximum `Θ(log n)` as sequence length grows — complete dispersion — while
α-entmax, which can assign *exactly zero*, keeps entropy bounded at `O(log s)` in the support
size `s`. They report 1000× length extrapolation on synthetic tasks by making that one
change. `[C]` ([2602.15028](https://arxiv.org/abs/2602.15028), Feb 2026) reaches the same
diagnosis empirically over ~29,000 instances from 1K to 256K, naming "attention dilution
under context scaling" an inherent limitation of soft attention in fixed-capacity
transformers.

> **Systems bridge.** A mandatory allocator with no admission control and no backpressure.
> The closest thing you have run is a weighted fair scheduler that is not allowed to idle:
> every tick, 100% of the quantum is distributed among the runnable set, and adding a
> thousand idle-but-runnable threads dilutes everyone.
>
> **Where it breaks:** your scheduler has a run queue you can inspect and a priority you can
> set. Here the "priority" is a learned dot product, the allocation is recomputed from
> scratch every token at every layer at every head, and nothing persists or accumulates. You
> cannot pin a token, you cannot nice it down, and there is no queue-depth metric — the
> allocation exists only inside one kernel invocation and is discarded.

**What bounds the margin?** The table says how much margin you need. It does not say how much
you can get. In Laguna, queries and keys are RMS-normalised per head before RoPE
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:368` — `q_norm`,
and `:369` — `k_norm`). RMSNorm fixes the RMS of the vector and then applies a learned
per-channel gain, so `|q|` and `|k|` are set by the *gains*, not by the data. With head
dimension `d_h`, unit-RMS vectors and scalar gains `g_q`, `g_k`, the largest attainable logit
is on the order of

```
ℓ_max  ≈  |q| · |k| / √d_h  =  (√d_h · g_q)(√d_h · g_k) / √d_h  =  √d_h · g_q · g_k
```

**In words:** *QK-norm converts "how sharply can this head discriminate" from a property of
the activations into a property of two learned scalars.* At `d_h = 128`, `√d_h = 11.3`, so
unit gains cap the margin near 11 nats — enough for 50% share at 8K, not enough for 99% share
at 1M. `[A]` This is a derivation, not a measurement, and the confidence is medium: real
heads use per-channel gains and the bound is loose. But it yields a falsifiable prediction —
**models trained for longer context should show larger QK-norm gains in the layers that carry
long-range work** — and the cheapest test is to read the gain norms out of any two
checkpoints of the same architecture trained at different lengths. Nobody appears to have
published it. See §7.

### 3.3 YaRN's temperature is a multiplicative fix for an additive deficit

`positional-encoding.md` §2.7 established the two halves of YaRN: the per-band frequency ramp,
and the attention temperature `attention_factor = 0.1·ln(s) + 1`, applied to `cos`/`sin` and
therefore to the logit **squared**. For Laguna: `t = 1.4852030263919618`, `t² = 2.2058280296`.

The paper's justification for the temperature is entropy: attending over `s`× more keys raises
softmax entropy and needs counteracting. §3.2 lets us check that claim quantitatively, which
as far as we can find nobody does.

The deficit is **additive** in nats. Going from `L_orig` to `s · L_orig` costs exactly

```
Δ_extra  =  ln( s · L_orig ) − ln( L_orig )  =  ln s        = 4.852 nats at s = 128
```

The fix is **multiplicative**. Every logit — hence every margin — is scaled by `t²`. So a
token whose baseline margin was `Δ₀` at the training length now has margin `t²·Δ₀` at the
extended length, and it keeps its share if and only if

```
t² · Δ₀  ≥  Δ₀ + ln s        ⟺        Δ₀  ≥  ln s / (t² − 1)
```

Substituting Laguna's numbers `[M]` (deterministic arithmetic, 2026-07-26):

```
Δ*  =  ln 128 / (2.2058280296 − 1)  =  4.852030 / 1.205828  =  4.0238 nats
```

**The break-even margin is 4.02 nats.** Tokens that were already discriminated by more than
about 4 nats get *over*-corrected — their margin grows faster than dilution erodes it, so the
softmax sharpens beyond where it was. Tokens below 4 nats get *under*-corrected and lose
share. And 4 nats is not an extreme number: `e⁴ = 54.6`, so it is a needle holding **52%**
against 50 competitors and **0.66%** against 8,191 of them `[M]`. Most token pairs in a long
context are below it, which means the temperature's *typical* effect is to under-correct.

| baseline margin `Δ₀` | after `×t²` | needed (`Δ₀ + 4.852`) | verdict |
|---|---|---|---|
| 2.0 | 4.41 | 6.85 | under-corrected |
| 4.0238 | 8.876 | 8.876 | exactly break-even |
| 6.0 | 13.23 | 10.85 | over-corrected |
| 9.011 | 19.88 | 13.86 | over-corrected |

> **Systems bridge.** This is proportional control applied to a constant offset error. You
> have watched this before in any P-only controller: it never lands, and its steady-state
> error scales with the setpoint. The fix a control engineer reaches for is an integral term
> — here, an *additive* logit bias of `ln s`, which is a one-line change and which nobody in
> the literature seems to have tried.
>
> **Where it breaks:** a controller has a measured error signal. Here there is no measurement
> of `Δ₀` at all — it is a property of the trained weights that varies per head, per layer,
> per token pair. `attention_factor` is applied blind to all 12 full-attention layers, all
> their heads, and every token pair in them.

`[A]` **This is our derivation, medium confidence, and it is a hypothesis rather than a
result.** It assumes the "one needle against uniform background" model of §3.2, which real
attention distributions do not obey — actual logit distributions are heavy-tailed and there
are attention sinks holding large mass at position 0. What it does establish is that the
temperature's functional form and the dilution requirement's functional form *disagree*, so
one cannot be the correct compensation for the other except at a single crossing point. That
is a cheap thing to test: `positional-encoding.md` §7 item 1 already asks for the
"YaRN with `attention_factor = 1.0`" arm; this section adds a third arm — **YaRN with an
additive `+ln s` logit bias instead of a multiplicative temperature** — and a prediction
about which tokens each one helps.

### 3.4 The threshold definition, and why one number induces an unstable ranking

RULER defines **effective context length** as *the longest length at which the model still
exceeds a fixed reference score* — concretely, Llama-2-7B's 85.6 at 4K `[C]`
([2404.06654](https://arxiv.org/abs/2404.06654)). Under that rule:

| model | claimed | effective (RULER) |
|---|---|---|
| GPT-4 | 128K | 64K |
| Llama-3.1-70B | 128K | 64K |
| Command-R-plus | 128K | 32K |
| Qwen2-72B | 128K | 32K |
| Yi-34B | 200K | 32K |
| Mixtral-8x22B | 64K | 32K |
| Gemini-1.5-Pro | 1M | >128K |

The definition is a good one and it has a failure mode you can compute. Model the score curve
locally as linear in `log₂ L`:

```
score(L)  ≈  τ  −  k · ( log₂ L  −  log₂ L* )
```

where `k` is the **slope in score-points per doubling of length**, `τ` the threshold, and
`L*` the true crossing. If the measured score carries noise of standard deviation `σ` points,
the induced uncertainty in the reported effective length is

```
δ( log₂ L* )  =  σ / k        ⟹        L*  is uncertain by a factor of  2^(σ/k)
```

**In words:** *the length error is the score error divided by the steepness of the curve.*
Numbers, so it is not abstract:

| slope `k` (points/doubling) | noise `σ` = 2 points | factor of uncertainty in `L*` |
|---|---|---|
| 16 | 0.125 doublings | 1.09× |
| 8 | 0.25 | 1.19× |
| 4 | 0.5 | **1.41×** |
| 2 | 1.0 | **2.0×** |
| 1 | 2.0 | **4.0×** |

`[M]` deterministic arithmetic. **House rule that falls out of it:** never report an effective
context length without the slope at the crossing and the seed variance. If `k < 4σ`, the
single number is not a measurement, and quoting it is the same error as reporting a p99 from
three samples. Report the whole curve; derive the number from it; state the factor.

This is not a theoretical worry. `[C]` **ATLAS** ([2605.28079](https://arxiv.org/abs/2605.28079),
May 2026) evaluated 26 models on an 8K–1M grid and found that **7 models shift by ≥2 rank
positions** between the 8K–128K regime and the 8K–1M regime, with individual gaps up to
**12 positions**, and that the two taxonomy layers share only **61% of cross-model variance**.
The ordering — the only thing an ablation ever produces — is a function of the length grid you
chose. `[C]` LongBench Pro ([2601.02872](https://arxiv.org/abs/2601.02872), Jan 2026) reaches
the same place from the realistic-task direction, and adds that effective context is
**cross-lingually misaligned** — the same model has different effective contexts in different
languages, which is by itself fatal to the one-number framing.

### 3.5 Capacity: the one mechanism with an error path, computed for our machine

Now the part your instincts are already right about. From
`attention-variants-and-kv-cost.md` and `ASSUMPTIONS.md → kv-per-token-laguna`, Laguna's KV
cost is exactly `2 · 8 · 128 · 2 B = 4 KiB` per token per layer, uniformly, because
`num_key_value_heads` is 8 on every layer. The hybrid splits it:

```
12 full-attention layers   :  12 × 4 KiB  =  48 KiB per token,  grows with context
36 sliding layers (512 win):  36 × 4 KiB × 512  =  72 MiB total,  constant
```

`[M]` computed 2026-07-26 from `models/laguna-s/config.json` (`:41` window, `:15` KV heads,
`:16` head_dim, `:13` layer count) against the measured fast tier:

| context `T` | KV total | % of the `[M]` ≥62 GiB fast tier |
|---|---|---|
| 8 192 (training length) | 0.445 GiB | 0.7% |
| 32 768 | 1.570 GiB | 2.5% |
| 131 072 | 6.070 GiB | 9.8% |
| 262 144 | 12.07 GiB | 19.5% |
| 524 288 | 24.07 GiB | 38.8% |
| **1 048 576 (advertised)** | **48.07 GiB** | **77.5%** |

KV alone crosses 62 GiB at about **1.35 M tokens** — comfortably past the advertised context.
Which sounds fine until you put the weights in first.

**The number that actually binds.** Laguna-S is 118B parameters. Weights are resident before a
single token of KV exists:

| weight dtype | weights | fast tier left (of 62 GiB) | context that fits | % of advertised |
|---|---|---|---|---|
| bf16 | 219.7 GiB | — | 0 | does not load at all |
| int8 | 109.9 GiB | — | 0 | exceeds the fast tier |
| **4-bit** | **54.95 GiB** | **7.05 GiB** | **≈152 000 tokens** | **14.5%** |

`[M]` deterministic arithmetic over two measured inputs (the ≥62 GiB fast tier from
`notebook/uma-carveout-controls-fast-tier.md`, and the per-token KV cost above), 2026-07-26.

**So: advertised 1,048,576. Capacity-usable on this machine at 4-bit, ~152,000 — 14.5%. And
that is before a single quality measurement.** Apply even the most generous published
effective/advertised ratio from the RULER table (GPT-4: 50%) and the honest number for what
this machine can *use* is in the tens of thousands. That is the "90% full and fragmented"
bridge with real numbers in it.

Three caveats, because this number will get quoted:

1. The ≥62 GiB fast tier is a **floor, not an edge** — the sweep in
   `notebook/uma-carveout-controls-fast-tier.md` stopped for an unrelated reason and never
   found the degradation point. There is also a slower tier beyond it (the pool reports
   107.87 GiB), so you can exceed the fast tier and take a bandwidth hit rather than a
   failure. That is a *tiering* decision, and it is exactly the axis `research/synthesis.md`
   says is our comparative advantage.
2. It is **one run per arm** — an anecdote by the house standard.
3. `[M]` single tensors ≥32 GiB hang or fault on this machine
   (`ASSUMPTIONS.md → large-tensor-fault-32gib`), so a 48 GiB KV cache **cannot be one
   allocation**. It must be chunked into at least two. This is the single place where the
   disk analogy holds without qualification: an allocator constraint forces fragmentation on
   you, and the fragmentation is real, and you have to manage it.

### 3.6 The numerics floor nobody budgets for

One more piece of arithmetic that connects dilution to dtype, and that is cheap to check.

bfloat16 has an 8-bit significand, so its machine epsilon is `2⁻⁸ = 0.00390625` and its
half-ulp is `2⁻⁹ = 0.001953`. Consider the attention output `o = Σ_j p_j v_j` accumulated
**sequentially in bf16**. A term whose magnitude is below a half-ulp of the running sum
contributes *nothing at all* — it rounds away. So a needle must carry more than about 0.2% of
the total attention mass to survive a naive bf16 accumulator, regardless of how many keys
there are:

| `N` | uniform weight | survival threshold (mass) | needle must be … × the average weight |
|---|---|---|---|
| 8 192 | 1.22e-4 | 0.001953 | **16×** |
| 131 072 | 7.63e-6 | 0.001953 | **256×** |
| 1 048 576 | 9.54e-7 | 0.001953 | **2 048×** |

`[M]` deterministic arithmetic 2026-07-26. Production kernels defend against this by
accumulating in fp32 even when the inputs are bf16 — which is exactly the kind of thing that
is *usually* true and that `ASSUMPTIONS.md → bf16-numerics-unproven` says we have not verified
on gfx1151. Exercise B measures it in about a minute and puts a number on the row. Note the
interaction, which is the point: **dilution makes weights small, and small weights are what
low-precision accumulation destroys.** The two mechanisms multiply. `[C]`
([2411.13476](https://arxiv.org/abs/2411.13476), Nov 2024) documents the sibling effect on the
RoPE side — bf16 breaks RoPE's relative-position property with error that accumulates as
context grows.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 This module supplies the outcome metric; the lab's contribution is the attribution metric

`research/synthesis.md` commits the lab to **building the instrument, not another eviction
policy**, on the grounds that the field has ~30 policies and no dominance result, while
attribution scores 5/5/5 on pain, tractability and our edge. Effective-context measurement is
the *outcome* half of that instrument: it is what an oracle-diff has to be correlated against
before anyone believes the oracle-diff means anything.

Concretely, the deliverable pairing is:

- **Outcome:** a scaled-RULER accuracy surface over (length × evidence position × task
  family), with the threshold-derived effective length *and its slope*, per §3.4.
- **Attribution:** per-token KL against a full-cache oracle, with a seed-to-seed null
  distribution.
- **The result nobody publishes:** the correlation between them. If distributional divergence
  and task accuracy do not track at our scale, that is `research/synthesis.md`'s named
  riskiest assumption failing, and it is worth writing up on its own.

### 4.2 Eviction is a context-shortening operation, and that is why NIAH cannot fail

Here is the arithmetic behind a claim the survey notes assert qualitatively.

`research/notes/evaluation-landscape.md` §2(e) argues that NIAH is *adversely selected* for
memory research: the needle is a high-salience, low-frequency span that attracts attention
mass, and heavy-hitter eviction (`[C]` H2O, [2306.14048](https://arxiv.org/abs/2306.14048)) or
observation-window selection (`[C]` SnapKV,
[2404.14469](https://arxiv.org/abs/2404.14469)) is precisely the family that keeps
high-attention tokens. §3.2 turns that from an intuition into two lines.

**Effect one — renormalisation.** Drop a set of keys carrying total attention mass `m`. The
survivors' weights are all divided by `(1 − m)`:

```
p_needle'  =  p_needle / (1 − m)
```

**Effect two, and this is the bigger one — the dilution requirement itself falls.** After
eviction the softmax runs over `N' < N` keys, so the margin required to hold a given share
drops from `ln(N−1)` to `ln(N'−1)`. Evicting 90% of the *keys* returns `ln 10 = 2.303` nats of
margin to everything that survives.

**Therefore an eviction policy that keeps the needle makes the needle easier to retrieve than
it was with the full cache.** Not "does no harm" — actively easier. Concretely: at
`N = 131072` with a needle at margin `Δ₀ = 6` nats, evicting 90% of the keys raises the
needle's share from **0.31% to 3.0%** — very nearly 10×, because `e^Δ₀ ≪ N'` keeps the
denominator dominated by the key count `[M]`. A NIAH score can *rise* under aggressive
compression while every diffuse, distributed, low-salience dependency in the context is
destroyed. `[A]` High confidence in the arithmetic; the empirical question is
whether the effect is large enough to observe, and the cheapest test is the one
`evaluation-landscape.md` already specifies: run H2O and SnapKV at a severe budget against
standard S-NIAH and against a NoLiMa-style associative needle at matched length, and watch
the gap. If NIAH goes *up* while the associative needle goes down, the point is made in one
plot.

This is also why `research/synthesis.md`'s corroborating citations look the way they do: `[C]`
([2510.00231](https://arxiv.org/abs/2510.00231), rev. 2026) finds StreamingLLM/SnapKV/TOVA/H2O
silently dropping *specific instructions* while aggregate LongBench scores look fine; `[C]`
([2510.13334](https://arxiv.org/abs/2510.13334), Oct 2025) finds that worst-case rather than
mean aggregation changes the ranking of eviction policies outright; and `[C]` SCBench
([2412.10319](https://arxiv.org/abs/2412.10319), Dec 2024) finds single-turn rankings not
surviving multi-turn cache reuse. Three independent ways for the outcome metric to hold while
the mechanism breaks.

### 4.3 What Mnemosyne must expose, as an interface requirement

Three fields, each justified by a section above, each cheap, each of which converts a silent
failure into a legible one.

1. **`occupancy`, not just `bytes_used`.** Position bias is occupancy-dependent (§1.2), so
   any retention policy tuned at one occupancy is mistuned at another. The register already
   contains the stronger form of this claim:
   `research/memory/open-problems-ranked.md` argues the correct policy is a *schedule over
   occupancy* rather than a constant. A policy cannot be a schedule over a quantity it cannot
   see. Report `N_resident / N_configured` and `N_resident / N_trained` — both, because §7
   says we do not know which one the bias tracks.
2. **`keys_evicted` and `mass_evicted` as separate counters.** §4.2 shows they have different
   effects: key count moves the dilution floor, mass moves the renormalisation. A single
   "compression ratio" conflates them and makes the two mechanisms unattributable — the exact
   error `positional-encoding.md` §2.8 warns about on a different axis.
3. **`position_policy`, mandatory, no default.** Inherited unchanged from
   `positional-encoding.md` §3.2: `preserve_original` or `renumber_compact`, declared per
   policy. A policy that does not state one is not runnable. This is the difference between a
   silent corruption and a config error.

None of these require Mnemosyne to know anything about Proteus, so none of them threaten the
package boundary.

### 4.4 What this hands to Themis

The eval configuration surface, stated as fields because the config surface is the
experimental surface:

```yaml
long_context_eval:
  lengths_as_multiple_of_training_length: [0.5, 1, 2, 4, 8, 16, 32]
  evidence_positions_fraction: [0.0, 0.25, 0.5, 0.75, 1.0]
  threshold_reference: own_baseline_at_0.5x     # never another model's score
  report:
    - accuracy_surface           # length x position, per task family
    - effective_length           # derived, never primary
    - slope_at_threshold         # points per doubling; gate on k >= 4*sigma
    - seed_variance              # >= 3 seeds
  control_arms:                  # each MUST behave as stated or the eval is void
    - evidence_absent            # -> chance
    - haystack_shuffled          # -> no change for a pure retrieval task
    - evidence_kv_dropped        # -> large drop
```

The `control_arms` block is not optional politeness. `research/notes/evaluation-landscape.md`
§6.2 makes the case in the language you already use: **you do not trust an alarm you have
never seen fire.** An eval that survives its own fault injections without moving is measuring
something other than memory and should be retired rather than reported. That note records
`[A]` high confidence that no KV-compression paper in the last twelve months reports a
needle-removed control. A fourth arm belongs there once we can run it: masking the
*retrieval heads* `[C]` ([2404.15574](https://arxiv.org/abs/2404.15574), Apr 2024) should
produce a targeted drop on retrieval tasks and leave the others intact — a mechanism ablation
rather than a data ablation.

Two statistics requirements ride along, and they are the difference between an ablation and a
rumour. `[C]` ([2406.10229](https://arxiv.org/abs/2406.10229), Jun 2024) finds seed variance
and non-monotonicity at small scale large enough that most differences are not meaningful,
and `[C]` ([2411.00640](https://arxiv.org/abs/2411.00640), Nov 2024) supplies the framework —
question-level clustering, paired analysis across arms, power analysis *before* the run.
State the **minimum detectable effect** in the pre-registration card. A 100-item suite cannot
resolve a 3-point difference, and the §3.4 slope gate is the length-axis version of the same
discipline.

---

## 5. Read the code

Paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 The advertised number, and where it is conspicuously not used

| Where | What to look for |
|---|---|
| `models/laguna-s/config.json:17` | `"max_position_embeddings": 1048576`. The entire "1M context" claim is this line. Note it sits between `head_dim` and `attention_bias` with no comment, no provenance, and no unit beyond "tokens". |
| `architecture/transformers/src/transformers/modeling_rope_utils.py:402` | `if factor is None:` — the *only* place `max_position_embeddings` enters the RoPE computation, and Laguna sets `factor` explicitly so this branch never runs. The advertised number does not participate in the math it appears to describe. |
| `models/laguna-s/config.json:41` | `"sliding_window": 512`, against the 1,048,576 above. Two numbers three orders of magnitude apart in the same file, describing the same model. 36 of 48 layers obey the small one. |

### 5.2 Where "usable" is computed — the one place with an error path

This is the most valuable reading in the module, because it is the systems half of the
argument written out by people who had to make it work.

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1932` | `_estimate_max_model_len_from_groups` — a **binary search** over context length, whose predicate is "does the KV cache for this length fit in available memory". The usable context is not derived in closed form; it is *searched for*. Read `fits()` at `:1943` and note it mutates the global config and restores it in a `finally`. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:1967` | `_auto_fit_max_model_len` — runs only when `max_model_len` is set to `-1`, takes the minimum across workers, and raises if not even one token fits (`:2006`). |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:2022` | The `logger.info_once(` call in the else-branch; its format string, on the following line, reads *"Auto-fit max_model_len: reduced from %d to %d to fit in available GPU memory (%s GiB available for KV cache)"*. **This is the entire user-visible signal that advertised context and usable context differ.** One `info_once`, at startup, at INFO. Everything in §1–§3 of this module is invisible; this one line is not. |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:854` | `check_enough_kv_cache_memory` — the ENOSPC path, for the case where the user *did* pin `max_model_len`. Contrast the two functions: one silently shrinks the claim, the other refuses to start. Both are defensible; note that only one leaves a trace in a dashboard. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:590` | `SlidingWindowSpec.max_memory_usage_bytes` — sizes the windowed tier at `min(sliding_window − 1 + in_flight_tokens, max_model_len)` tokens, rounded up to blocks **plus one** (read the comment at `:585`: the window may not start on a block boundary). This is `attention-variants-and-kv-cost.md` §3.5's "growing term plus fixed term" as production capacity planning, and it is why a hybrid model's usable context is not a single division. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:109` | `page_size_bytes` is a per-spec property, so a hybrid model has **several block geometries over one physical pool**. Your page-table intuition assumes one page size per address space; this violates it deliberately. |

### 5.3 Fragmentation, literally

| Where | What to look for |
|---|---|
| `memory/flashinfer/flashinfer/page.py:326` | `get_seq_lens` — `(num_pages - 1) * page_size + last_page_len`, three lines that are the entire internal-fragmentation accounting. Bounded waste of under one page per request, exactly as in a block-allocated filesystem. This is the *good* kind of fragmentation: measurable, bounded, and with a formula. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `get_new_blocks` — pop from the free queue, evict the stale hash entry, bump the refcount. No zeroing, no search, no compaction pass. There is no defragmenter in this allocator because fixed-size pages make external fragmentation impossible — which is the trade paged attention exists to make. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict` — eviction is constrained to *leaves* of the prefix tree, so a hot child pins a cold parent indefinitely. This is a capacity policy with a topological constraint, and it has no analogue in a disk allocator. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | `BatchEvict` — lease-expiry-based reclamation across a real storage hierarchy. Read the surrounding code for the line that matters philosophically: evicting a KV block is never data loss, only a recompute, so this tier is allowed to throw bytes away rather than block on writeback. No storage system you have run is permitted that. |

### 5.4 The smallest possible version of the whole problem

| Where | What to look for |
|---|---|
| `training/nanogpt/model.py:173` | `assert t <= self.config.block_size` — advertised context as a bounds check. Hard, loud, correct. |
| `training/nanogpt/model.py:314` | `idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]` — **the entire usable-context policy of a language model in one line**: silently drop the oldest tokens at sample time. No log, no counter, no metric, no way for the caller to learn it happened. Every failure mode in this module is a more sophisticated version of this line. |
| `training/nanogpt/train.py:216` | `estimate_loss` — a Monte-Carlo mean over `eval_iters` random batches. Look at what is *not* measured: nothing about length, nothing about position, nothing that could distinguish "learned the data" from "learned the last 64 tokens of the data". Your scaled-RULER harness in Exercise C is the thing that has to be added here. |

### 5.5 What you will not find

Spend ten minutes trying to find, in vLLM, SGLang, FlashInfer, llama.cpp or transformers, a
single counter, metric, log line or assertion that concerns model *quality* as a function of
context length. There is none. Every long-context number in every serving stack is a capacity
number. That absence is the module's thesis, and looking for it yourself is more convincing
than being told.

---

## 6. Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU). Activate with
`. .\scripts\activate-lab.ps1` from the repo root. Exercise A needs no GPU at all; B has a
CPU path that is part of the experiment; C has a published CPU recipe.

**ROCm/WSL2 caveat:** do not run these under WSL2. `[C]` ROCm issue #6022 clamps the ROCm pool
to the `.wslconfig` value, and Exercise B is a numerics measurement whose whole point is the
wheel and the device. Native Windows, `C:\venvs\lab`, pinned wheel
`torch 2.12.0a0+rocm7.13.0a20260313` `[M]` (`ENVIRONMENT.md`).

---

### Exercise A — the dilution ledger, and the eviction paradox

**Difficulty 2/5. Writing: 45–60 min. Runtime: seconds. No GPU.**

Reproduce every number in §3.2, §3.3 and §3.6 from scratch, then push one step past them.

**Build.** Plain Python, `math` and `matplotlib`. Put it under `notebook/` while it is a
one-off.

1. **The margin table.** Implement `delta(N, p) = ln((N-1)*p/(1-p))` and print §3.2's table.
   Check `Δ(8192, 0.5) = 9.0108` and `Δ(1048576, 0.5) = 13.8629`.
2. **The `ln s` identity.** Show numerically that
   `Δ(s·L, p) − Δ(L, p) → ln s` for any `p`, and that it equals `4.852` for `s = 128`. This is
   the one result in the module that makes YaRN's `factor` and the dilution cost the same
   quantity, and it should surprise you slightly.
3. **The break-even margin.** Parse `attention_factor` out of
   `research/reference/models/laguna-s/config.json` (do not hardcode it), square it, and solve
   `Δ* = ln(factor)/(t²−1)`. You should get **4.0238**. Then plot, for `Δ₀` on 0–15 nats,
   the two curves `t²·Δ₀` and `Δ₀ + ln 128`, and mark the crossing.
4. **The eviction paradox.** For `N = 131072` and a needle at `Δ₀ = 6` nats, plot `p_needle`
   as a function of the fraction of *keys* evicted, from 0 to 0.999, under the assumption
   that the needle is always retained and the evicted keys are drawn from the background. The
   curve must go **up**: 0.31% at no eviction, 2.99% at 90% `[M]`. Report the eviction
   fraction at which `p_needle` first exceeds 0.5 — expect it between 99.5% and 99.9%, and
   note *why* it is that extreme: dilution is logarithmic, so you have to remove almost
   everything to move the share by an order of magnitude.
5. **The bf16 survival threshold.** Print §3.6's table and add a column: at what `N` does the
   *uniform* weight drop below the bf16 half-ulp entirely (i.e. a perfectly flat attention
   distribution becomes unrepresentable term-by-term)?

**Deliverable — three numbers and two plots.** The numbers: `Δ*` (expect 4.0238), the eviction
fraction at which the needle crosses 50% share, and the `N` from step 5. The plots: the
crossing plot from step 3, and the eviction-paradox curve from step 4.

**How this can fail, which is the point.** If your `Δ*` is not 4.02, you have either used
`t` instead of `t²` (the temperature is applied to `cos`/`sin`, hence to both `q` and `k` —
`positional-encoding.md` §2.7) or mixed nats and bits. If your eviction curve goes *down*, you
have renormalised over the original `N` instead of the survivors.

---

### Exercise B — does the needle survive the accumulator?

**Difficulty 3/5. Writing: 60–90 min. Runtime: 2–5 minutes including torch import.**

§3.6 says a needle carrying less than ~0.2% of the attention mass is destroyed by a
sequential bf16 accumulator, and that production kernels are *supposed* to accumulate in fp32.
`ASSUMPTIONS.md → bf16-numerics-unproven` says we have never checked that on gfx1151. Check it.

**Build.** One script, both devices. Keep two things separate that are easy to conflate, and
the conflation is exactly how this lab previously tagged a non-reproducing observation as
`[M]` (see `curriculum/README.md`, "Findings the exercises produced"):

- **seed variation** — three *different* seeds, which tells you whether an effect is a
  property of the arithmetic or of one unlucky draw;
- **determinism** — the *same* seed run twice on the same device, which tells you whether the
  kernel is reproducible at all.

Steps:

1. Fix `d = 128`, seed a CPU generator, draw `V ∈ R^{N×d}` in **float64**.
2. Build logits: zero everywhere, `Δ` on index 0, with `Δ = ln((N−1)p/(1−p))` chosen so the
   needle holds exactly mass `p`. Softmax in float64. This gives you an exact reference
   `o_ref = p @ V`, and the needle's exact contribution `p₀·v₀`.
3. Cast `p` and `V` to bfloat16, compute `o = p @ V` on the target device, cast back to
   float64, and report (a) the relative L2 error `‖o − o_ref‖ / ‖o_ref‖`, and (b) the
   projection of the error onto `p₀·v₀`, normalised — i.e. **what fraction of the needle's
   contribution was lost**.
4. Sweep `N ∈ {8192, 131072, 1048576}` and `p ∈ {0.5, 0.01}`, on `cpu` and on `cuda`
   (ROCm presents as `cuda`), three seeds each. Draw `V` **once per seed** and use the same
   bf16 tensor on both devices, or you are comparing two different random matrices.
5. Separately, run the GPU arm twice on one seed and compare with `torch.equal`.

**Deliverable — a 12-row table, a CPU/GPU ratio, and a determinism verdict.** For each
(device, `N`, `p`): relative error, needle-loss fraction, spread across seeds; then the
GPU-to-CPU error ratio per cell; then one boolean for bitwise determinism.

**What to expect, and how to read each outcome.**

- Relative error around `2–3e-3` on both devices, flat in `N`: **the matmul accumulates in
  fp32** and you are seeing bf16 *input* rounding (half-ulp `2⁻⁹ = 1.95e-3`) rather than
  accumulator loss. §3.6's table then describes the worst case rather than the shipped case.
- **GPU error rising with `N` while CPU error stays flat:** the accumulator, or the split-k
  partials, are losing precision at large reduction depth. This is the §3.6 mechanism showing
  up in a real kernel, and it is what we measured — see below.
- CPU and GPU differ by more than an order of magnitude: a hardware-specific numerics defect.
  Stop, write it up as a Hardware Validation Gate finding, and do not proceed to Exercise C.
- Same-seed repeats not bit-identical: a nondeterminism finding, which matters *more* than the
  numerics one, because determinism is a Hardware Validation Gate item and the entire ablation
  programme assumes seeded runs reproduce.

**What we measured, so you know what you are checking against.** `[M]` 2026-07-26, one
session, `torch 2.12.0a0+rocm7.13.0a20260313`, gfx1151, native Windows, shapes exactly as
above, three independent seeds (1337/1338/1339) per cell:

| | CPU rel. err. | GPU rel. err. | ratio |
|---|---|---|---|
| `N = 8192`, `p = 0.5` | 2.21–2.45e-3 | 2.21–2.45e-3 | **1.00×** (bit-identical to CPU) |
| `N = 131072`, `p = 0.5` | 2.12–2.27e-3 | 2.02–2.20e-3 | ~0.95× |
| `N = 1048576`, `p = 0.5` | 1.85–1.91e-3 | **5.31–5.91e-3** | **~3×** |
| `N = 1048576`, `p = 0.01` | 2.21–2.68e-3 | 4.42–5.30e-3 | ~2× |

The CPU error is flat in `N`, as bf16 input rounding predicts. The GPU error is flat up to
131K and roughly triples at 1M (per-seed ratios 3.03×, 2.79×, 3.18×), and the sign of the
needle-loss term flips from negative to positive — consistent with reduced-precision partial
sums at large reduction depth rather than with input rounding. All three seeds agree in every
cell, so it is not one unlucky draw.

**Replication, because the tag depends on it.** Both `N = 1048576` rows were re-run in a
**fresh process**, with `V` drawn once per seed and shared bit-identically between CPU and
GPU. All six cells reproduced to every printed digit — at `p = 0.5`: 1.8486/5.6046,
1.9052/5.3079, 1.8623/5.9139 e-3 (ratios 3.03×, 2.79×, 3.18×); at `p = 0.01`: 2.6848/4.4245,
2.2090/4.9975, 2.5305/5.2985 e-3 (ratios 1.65×, 2.26×, 2.09×). Running the GPU arm twice on
the same seed gave `torch.equal == True` in all six cases, so **the kernel is deterministic
and the effect is not a scheduling artefact**. That is the difference between this and the
crash that `curriculum/README.md` records as having been tagged `[M]` without a repeatable
basis.

**What that is and is not.** It **is** repeatable across seeds and across processes, with
same-seed bitwise determinism. It is **not** yet a characterised defect: we have not varied
the accumulation dtype
explicitly, not tested `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`,
not tried a non-power-of-two `N`, and not established whether it tracks `N` or the tile
count. Treat it as an open Hardware Validation Gate item with a measured effect size, not as
a diagnosis — and note that a 3× error inflation at 1M keys is small in absolute terms
(5e-3 relative) and would only matter where the answer hinges on a term of that size, which
§3.6 says is exactly the long-context needle case.

> **Resolved the same day — and the cause is not the GPU.** `[M]` 2026-07-26,
> `scripts/measure_bf16_reduction_error.py` plus an environment matrix at
> `N = 1,048,576`, seed 1337, identical CPU reference (`1.8486e-3`) in every run:
>
> | GPU path | rel. err. | vs CPU |
> |---|---|---|
> | `HIPBLASLT_TENSILE_LIBPATH` + `TORCH_BLAS_PREFER_HIPBLASLT` **set** | `2.0079e-3` | **1.09×** |
> | both **unset** | `5.6046e-3` | **3.03×** |
>
> `5.6046e-3` is this table's own number to five significant figures, so the measurement
> above was taken **without `scripts/activate-lab.ps1`**. The effect is real and
> repeatable; the attribution was wrong. It is not *GPU vs CPU* — it is
> **hipBLASLt configured vs not**, and with the lab's standard environment the GPU is
> within 9% of the CPU at 1M keys.
>
> Two corollaries. First, `hipBLASLt` is a **numerics** control on this machine, not the
> +12% throughput tweak `ASSUMPTIONS.md` originally recorded — it is worth ~2.8× in
> long-reduction accuracy, and every run must therefore record whether it was configured.
> Second, `allow_bf16_reduced_precision_reduction` — named above as untested — was tested:
> toggling it changes **exactly zero bits** in either configuration. It is inert on this
> stack, which is why it could never have been the explanation.
>
> The methodology here was right and is worth copying: replicate in a fresh process, share
> inputs bit-identically, check same-seed determinism, and **write down what you did not
> test**. That last list is what made the cause findable in one attempt — the missing
> variable was sitting in it.

**Shapes, stated so this is retestable.** `V` is `[N, 128]`; at `N = 1048576` that is 1.07 GB
in float64 and 0.27 GB in bf16, well inside the `[M]` ≥62 GiB fast tier and far below the
`[M]` 32 GiB single-tensor fault. Windows has only ~31.6 GB of visible system RAM with the
96 GB UMA carve-out in place, so free the float64 tensors between sweeps or you will page.

**CPU fallback:** the CPU arm is half the experiment, not a fallback. If the GPU is
unavailable, run the CPU arm alone and note that the comparison — which is the actual finding
— is missing.

**Trap:** do not use `F.scaled_dot_product_attention` for this. `[M]`
`ASSUMPTIONS.md → sdpa-is-memory-efficient` shows SDPA on this wheel retains the score matrix
by default (147.2 bytes/T² vs 6.6 with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), so at
`N = 1048576` you would be asking for a `T²` allocation and would hit a silent hang at 0 CPU
rather than a measurement. An explicit `p @ V` matmul is what you want here anyway: it isolates
the accumulator.

---

### Exercise C — a scaled RULER at nanoGPT scale, with the control arms that make it an eval

**Difficulty 4/5. Writing: one evening. Compute: `[A]` 3–6 h GPU for the full grid, medium
confidence — the basis is nanoGPT's published ~3 min/run on one A100
(`training/nanogpt/README.md:51`) scaled by our `[M]` 20.9 TFLOPS bf16, which is a poor
estimator for a launch-bound 10.6M model. Time one cell of the grid and re-plan.**

The question, and it is the fourth open question in the survey note: **does the
effective/advertised gap reproduce at all below 300M parameters?** Every measurement of it
(RULER, LongBench Pro, ATLAS) is at frontier scale. There is one recent data point from below
1B — `[C]` ([2605.10544](https://arxiv.org/abs/2605.10544), May 2026) moves RULER by +10.69
and NoLiMa by +10.09 points on **Qwen2.5-0.5B** with a training-side intervention, which
establishes that these benchmarks have dynamic range at 0.5B. Our scale is 2–20× smaller
again. If the gap does not appear here, this rig cannot study the phenomenon, and that is a
publishable negative about our own methodology.

**Pre-register first.** This is a real experiment, not a demo: G2 hypothesis card verbatim,
committed before the run, per `CLAUDE.md → Experimental standards`.

**Build.**

1. **Model.** Reuse the `rope_full` checkpoint from `positional-encoding.md` Exercise 3 if you
   have it (6 layers / 6 heads / 384 channels, `block_size` 256, ~10.6M params on
   `shakespeare_char`). If not, train it — 3 seeds, matched tokens.
2. **A generator, not a dataset.** Build S-NIAH in the model's own vocabulary: a haystack of
   sampled text, and a needle of the form `<key> is <value>` where key and value are drawn
   from a **held-out symbol inventory** that never appears in training. At character level
   this means reserving character n-grams; check the reservation, do not assume it
   (`evaluation-landscape.md` §5, "harness leakage").
3. **The grid.** Lengths at `{0.5, 1, 2, 4}× block_size` — you will need the extension
   machinery from `positional-encoding.md` Exercise 3 to run past 256 at all. Evidence
   positions at `{0, 0.25, 0.5, 0.75, 1.0}` of the way through. 3 seeds. Score by
   **per-token log-likelihood of the correct value**, not exact match: at this scale a
   discontinuous metric manufactures cliffs `[C]`
   ([2304.15004](https://arxiv.org/abs/2304.15004), Apr 2023) and destroys statistical power.
4. **The three control arms**, run at every length:
   - *evidence absent* — delete the needle. Score must fall to the model's prior over the
     value symbols. Compute that chance level in closed form and check against it.
   - *haystack shuffled* — shuffle the distractor sentences. A pure retrieval task should not
     move; if it does, you are measuring discourse structure, not recall.
   - *evidence KV dropped* — run with the needle present but its KV entries masked out. Score
     must fall to the absent arm. This is the arm that proves the score depends on *those*
     cache entries, and it is the one nobody publishes.
5. **The analysis.** Produce the accuracy surface, then derive: effective length against your
   own `0.5×` baseline as the threshold reference, the slope `k` in points per doubling at the
   crossing, and the seed standard deviation `σ`. Apply the §3.4 gate: if `k < 4σ`, report the
   curve and **refuse to report a single number**.

**Deliverable — one heatmap, one curve, and a gated number.** The heatmap: score over
(length × position), which is where you will see position bias if it exists at 10M params. The
curve: score versus length with 3-seed CIs, with the three control arms overlaid. The number:
`L_eff` with its uncertainty factor `2^(σ/k)`, or an explicit refusal.

**What to expect, honestly.** Three outcomes and all three are informative. (i) The model is
at chance everywhere, including inside its training length — the task is too hard at 10M
params, and the finding is about the rig. In that case fall back to **MQAR** `[C]`
(Zoology, [2312.04927](https://arxiv.org/abs/2312.04927), Dec 2023), which has a published
shape to calibrate a harness against (attention solves it at model dimension 64 across
lengths) — reproduce the published shape before trusting any number your own harness emits.
(ii) It solves the task inside the training length
and collapses immediately past it — you have measured mechanism 3 (untrained positional phase)
and *not* the effective-context gap, which is a distinct thing. (iii) It degrades smoothly
within the training length as you pad the context with more distractors — that is mechanism 2,
dilution, in isolation, and it is the interesting outcome, because it is the one that says the
phenomenon reproduces at our scale.

**CPU fallback:** the published CPU recipe (4 layers / 4 heads / 128 channels, `block_size` 64,
2000 iters, target val loss 1.88 `[M]`, `training/nanogpt/README.md:85`). Scale lengths to
`{32, 64, 128, 256}` and keep the five positions. Curve *shapes* are the deliverable, not
absolute scores, so the CPU version answers the same question at lower resolution.

**Trap:** when you extend past `block_size` you must extend the mask *and* `position_ids`, and
it is easy to extend one. Assert `position_ids.max() == context_len - 1` before every eval
batch — the same miniature of the eviction hazard flagged in `positional-encoding.md` §3.2.

---

## 7. Self-check

1. A model card says "1M context". Give three different numbers that phrase could denote,
   say where each is decided, and say which of the three has an error path in a production
   serving stack.

2. Attention over 8,192 keys gives a needle 50% of the mass. The context is extended to
   1,048,576 keys with the needle's raw logit unchanged. How many nats of additional margin
   does it need to hold 50%, and what is that number in terms of Laguna's config?

3. YaRN multiplies every logit by `t² = 2.2058` on the full-attention layers. For which tokens
   does that over-correct dilution, and for which does it under-correct? Give the break-even
   margin and the one-line reason the two do not match in general.

4. An eviction policy drops 95% of KV entries and NIAH accuracy goes *up*. Give the two
   independent mechanisms that predict this, and say what it implies about using NIAH to
   validate a memory policy.

5. You measure a model's long-context score at 4 lengths and it crosses your threshold between
   32K and 64K. The curve falls 3 points per doubling and your 3-seed standard deviation is
   2 points. What is the honest uncertainty on the effective context length, and what should
   you report?

6. Laguna advertises 1,048,576 tokens. Our fast tier is `[M]` ≥62 GiB. Compute the context
   length that fits in KV alone, then the context length that fits once 4-bit weights are
   resident, and explain why the second number is not simply the first minus a constant.

7. Name the one line of code in the reference library that is the entire user-visible signal
   that advertised and usable context differ, and the one line in nanoGPT that is the same
   policy with no signal at all.

---

## 8. What is still unsolved here

**1. Whether the effective/advertised gap reproduces below 300M parameters.** Every
measurement of it is at frontier scale. `[C]` ([2605.10544](https://arxiv.org/abs/2605.10544),
May 2026) shows RULER and NoLiMa both have dynamic range at 0.5B, which brackets us from
above but does not answer it. Exercise C is the experiment; a negative result is a result
about our own methodology and is worth writing up with equal care.

**2. Whether "effective context" is a single number at all.** ATLAS's rank instability `[C]`
([2605.28079](https://arxiv.org/abs/2605.28079)) — 7 of 26 models shifting ≥2 positions, gaps
up to 12, 61% shared variance across taxonomy layers — undermines the one-number framing that
RULER's threshold rule encourages. LongBench Pro's cross-lingual misalignment `[C]`
([2601.02872](https://arxiv.org/abs/2601.02872)) says the same thing from a different axis. No
replacement framing has been adopted.

**3. Whether position bias is architectural, correctable, or a symptom.** Three live positions
as of this writing: derived from causal masking plus residuals `[C]`
([2602.16837](https://arxiv.org/abs/2602.16837)); recoverable by up to 15 pp with inference-time
calibration `[C]` ([2406.16008](https://arxiv.org/abs/2406.16008)); and insufficient-when-corrected,
trailing iterative reordering by 14.84 pp `[C]`
([2606.27793](https://arxiv.org/abs/2606.27793), Jun 2026). All three are recent and they do not
reduce to each other.

**4. Whether occupancy is measured against the configured window or the trained one.** `[C]`
([2508.07479](https://arxiv.org/abs/2508.07479)) reports the bias regime shifting at ~50%
occupancy. Occupancy against *what* is load-bearing and, as far as we can tell, unaddressed:
if it is the configured `max_position_embeddings`, then changing a config field changes the
bias regime of an unchanged prompt, which is testable in an afternoon on any model with an
adjustable window. If it is absolute length, the finding is really about length and the word
"occupancy" is misleading. `[A]` We do not know, and it directly determines whether
Mnemosyne's occupancy metric should be normalised by the configured or the trained length —
which is why §4.3 requires both.

**5. Whether QK-norm gains bound the achievable logit margin, and whether long-context
training grows them.** §3.2's derivation says the margin ceiling is roughly `√d_h · g_q · g_k`,
which makes QK-norm — introduced as a *stability* measure — silently also a *long-context
capacity* parameter. `[A]` Medium confidence, unpublished as far as we can find. The cheapest
test needs no training: read the `q_norm`/`k_norm` gain norms out of two public checkpoints of
one architecture family trained at different context lengths and compare, layer by layer.

**6. Whether YaRN's temperature should be additive rather than multiplicative.** §3.3's
derivation says the deficit is `+ln s` nats and the fix is `×t²`, so they can agree at exactly
one margin (4.02 nats for Laguna). `[A]` Our derivation, medium confidence, untested. The arm
is one config field — an additive logit bias — and it composes with the
"YaRN with `attention_factor = 1.0`" attribution arm that `positional-encoding.md` §7 already
asks for. Nobody publishes either.

**7. Whether sparse attention removes the dilution mechanism or relocates it.** `[C]`
([2506.16640](https://arxiv.org/abs/2506.16640)) replaces softmax with α-entmax to get genuinely
zero weights and bounded entropy, reporting 1000× extrapolation on synthetic tasks. If
dilution is the dominant mechanism, that should transfer to natural long-context tasks; the
published evidence is on synthetics. The competing possibility is that the entropy bound just
moves the failure into the support-selection step, which would show up as a *new* silent
failure rather than none.

**8. Why the bf16 weighted-sum error triples at 1M keys on this GPU and not on this CPU.**
`[M]` Exercise B, run while writing this module: identical bf16 inputs, `[1048576, 128]`,
relative L2 error 1.85e-3 on CPU against 5.60e-3 on gfx1151 — ratio 3.03× on seed 1337, 2.79×
and 3.18× on two more seeds — reproducing bit-for-bit in a fresh process, with same-seed GPU
repeats bitwise identical, while the same comparison at `N = 8192` is bit-identical between
the two devices. Something in the reduction changes
between 131K and 1M and we do not know what. Candidates, none tested: reduced-precision
partial sums, a split-k tiling change at large reduction depth, or a different kernel being
selected. The discriminating experiments are cheap (toggle
`allow_bf16_reduced_precision_reduction`, use a non-power-of-two `N`, sweep `N` finely across
the transition) and none of them has been run. Until they are, this is a measured effect with
no mechanism, which is the honest label. More broadly:
`ASSUMPTIONS.md → bf16-numerics-unproven` is still `untested`, the Hardware Validation Gate
has not run, and the ≥62 GiB fast tier is one run per arm — so no long-context *result* from
this machine is evidence yet. That is the correct standing state, not a complaint.

**9. The one that is not on arXiv.** Chroma Research's "Context Rot" (Hong, Troynikov, Huber,
Jul 2025) reports that coherent, well-structured input degrades attention *more* than shuffled
input — which, if true, inverts the "haystack shuffled" control arm in §4.4 and Exercise C from
a null-expectation control into a signal. It is an industry technical report with no arXiv id
and no independent replication we can find. Cite it as such or not at all, and note that our
own control arm would detect it if it is real: if shuffling the haystack *improves* the score,
that is the Context Rot claim reproducing in our harness.

---

## 9. Answers to the self-check

**1.** (a) **Advertised** — `max_position_embeddings`, decided by whoever wrote
`config.json:17`; for Laguna it is `8192 × 128`, the pretraining length times YaRN's `factor`.
(b) **Admissible** — the longest request the server accepts after KV memory accounting, decided
at server startup; vLLM either binary-searches it
(`memory/vllm/vllm/v1/core/kv_cache_utils.py:1932`) or raises (`:854`). (c) **Effective** — the
longest length at which measured quality clears a threshold, decided nowhere and only knowable
by running an eval. Only (b) has an error path. (a) is a claim; (c) fails silently and
smoothly.

**2.** `ln(1048576) − ln(8192) = ln 128 = 4.852` nats `[M]`. In config terms it is exactly
`ln(factor)` — the natural log of YaRN's scale factor — because the advertised length *is*
`factor × original_max_position_embeddings`. The needle's share, if the raw logit does not
change, falls from 50% to `e^9.011 / (e^9.011 + 1048575) ≈ 0.8%`.

**3.** Break-even at `Δ* = ln 128 / (t² − 1) = 4.852030 / 1.205828 = 4.0238` nats `[M]`. Tokens
with baseline margin **above** 4.02 nats are over-corrected (the multiplicative gain adds more
margin than dilution removes, so the softmax sharpens beyond its training behaviour); tokens
**below** are under-corrected. They cannot match in general because the dilution deficit is
*additive* in nats and fixed at `ln s` for every token, while the temperature is a
*multiplicative* gain on a per-token margin that varies by head, layer and token pair — a
proportional correction for a constant offset.

**4.** (i) **Renormalisation:** removing keys carrying mass `m` multiplies every survivor's
weight by `1/(1−m)`. (ii) **Reduced dilution:** the softmax now runs over `N' < N` keys, and the
margin required to hold a given share is `ln(N'−1)`, so dropping 95% of the keys returns
`ln 20 = 3.0` nats to everything retained. Both push in the same direction, and heavy-hitter
policies retain the needle by construction because the needle is high-attention by
construction. It implies NIAH **cannot fail** for this policy family — it is adversely selected
for the mechanism under test, so passing it is not evidence, and the only useful version is a
NoLiMa-style associative needle plus the `evidence_kv_dropped` control arm.

**5.** `δ(log₂ L*) = σ/k = 2/3 = 0.67` doublings, so `L*` is uncertain by a factor of
`2^0.67 = 1.59×` — the honest statement is "somewhere between roughly 28K and 71K". Since
`k = 3 < 4σ = 8`, the §3.4 gate says do not report a single number at all: report the curve, the
slope, the seed variance, and the interval. A single "effective context: 48K" here is the same
error as quoting a p99 from three samples.

**6.** KV alone: `(62 GiB − 72 MiB) / 48 KiB ≈ 1.35 M tokens` `[M]`, using 48 KiB/token for the
12 global layers and a constant 72 MiB for the 36 windowed ones. With 4-bit weights:
`118e9 × 0.5 B = 54.95 GiB` resident first, leaving 7.05 GiB, so
`(7.05 GiB − 72 MiB) / 48 KiB ≈ 152 000` tokens — **14.5% of advertised** `[M]`. It is not
"the first number minus a constant" for two reasons: the relationship between weights and
context is a subtraction in *bytes* but the conversion to *tokens* is a division, so a fixed
byte cost removes a length-independent number of tokens only after dividing by the per-token
rate; and the 72 MiB windowed term is constant, not proportional, so it does not scale out of
the ratio. The deeper reason is that the fast tier is a floor, not an edge: exceeding it
degrades bandwidth rather than failing, so the "limit" is a performance cliff whose location we
have measured from one side only.

**7.** The signal: the `logger.info_once` at
`memory/vllm/vllm/v1/core/kv_cache_utils.py:2022`, whose message is
*"Auto-fit max_model_len: reduced from %d to %d to fit in available GPU memory"*. The same
policy with no signal:
`training/nanogpt/model.py:314` —
`idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]`,
which silently drops the oldest tokens every sampling step.

---

## 10. Sources

**Read from the repo's own artifacts** `[M]`, revisions pinned in
`research/reference/PROVENANCE.md`. Paths relative to `research/reference/`:

- `models/laguna-s/config.json` — `num_hidden_layers` (:13), `num_key_value_heads` (:15),
  `head_dim` (:16), `max_position_embeddings` (:17), `sliding_window` (:41),
  `rope_parameters` (:42-58).
- `architecture/transformers/src/transformers/modeling_rope_utils.py:402` — the `factor is
  None` fallback, the only use of `max_position_embeddings` in the RoPE path.
- `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:368` and `:369`
  — `q_norm` / `k_norm`, the bound on the achievable logit margin (§3.2).
- `memory/vllm/vllm/v1/core/kv_cache_utils.py` — `check_enough_kv_cache_memory` (:854),
  `_estimate_max_model_len_from_groups` (:1932), the `fits` predicate (:1943),
  `_auto_fit_max_model_len` (:1967), the "not enough memory" raise (:2006), the
  "reduced from %d to %d" log line (:2022).
- `memory/vllm/vllm/v1/kv_cache_interface.py` — `page_size_bytes` as a per-spec property (:109),
  `SlidingWindowSpec.max_memory_usage_bytes` (:590).
- `memory/vllm/vllm/v1/core/block_pool.py:647` — `get_new_blocks`, allocation without compaction.
- `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` — leaf-constrained eviction.
- `memory/mooncake/mooncake-store/src/master_service.cpp:6382` — `BatchEvict`, lease-based
  reclamation across a real hierarchy.
- `memory/flashinfer/flashinfer/page.py:326` — `get_seq_lens`, internal-fragmentation accounting.
- `training/nanogpt/model.py:173` (bounds assert), `:314` (silent left-crop);
  `training/nanogpt/train.py:216` (`estimate_loss`); `training/nanogpt/README.md:51` (GPU
  target), `:85` (CPU recipe and its 1.88 target).
- `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s), `large-tensor-fault-32gib`,
  `bf16-numerics-unproven`, `kv-per-token-laguna`, `sdpa-is-memory-efficient`,
  `ablation-scale-sufficient`, `hipblaslt-config`.
- `notebook/uma-carveout-controls-fast-tier.md`, `ENVIRONMENT.md`.
- Notes this module must stay consistent with: `research/memory/long-context-behavior.md` §7,
  `research/notes/evaluation-landscape.md` §1–3 and §6, `research/memory/open-problems-ranked.md`,
  `research/synthesis.md`.

**Cited** `[C]`. Every arXiv id below resolved against arxiv.org on 2026-07-26.

- [2104.09864](https://arxiv.org/abs/2104.09864) — RoFormer / RoPE (2021). Background only;
  taught in `positional-encoding.md`.
- [2306.14048](https://arxiv.org/abs/2306.14048) — H2O: Heavy-Hitter Oracle for Efficient
  Generative Inference (Jun 2023).
- [2306.15595](https://arxiv.org/abs/2306.15595) — Position Interpolation (Jun 2023).
- [2307.03172](https://arxiv.org/abs/2307.03172) — Lost in the Middle (2023).
- [2309.00071](https://arxiv.org/abs/2309.00071) — YaRN (2023).
- [2309.17453](https://arxiv.org/abs/2309.17453) — StreamingLLM / attention sinks (2023).
- [2304.15004](https://arxiv.org/abs/2304.15004) — Are Emergent Abilities of LLMs a Mirage?
  (Apr 2023). Why discontinuous metrics manufacture cliffs at small scale.
- [2312.04927](https://arxiv.org/abs/2312.04927) — Zoology / MQAR (Dec 2023).
- [2404.06654](https://arxiv.org/abs/2404.06654) — RULER (Apr 2024). The effective-context
  threshold definition.
- [2404.14469](https://arxiv.org/abs/2404.14469) — SnapKV (Apr 2024).
- [2404.15574](https://arxiv.org/abs/2404.15574) — Retrieval Head Mechanistically Explains
  Long-Context Factuality (Apr 2024).
- [2406.10229](https://arxiv.org/abs/2406.10229) — Quantifying Variance in Evaluation
  Benchmarks (Jun 2024).
- [2406.16008](https://arxiv.org/abs/2406.16008) — Found in the Middle (Jun 2024).
- [2410.02660](https://arxiv.org/abs/2410.02660) — How to Train Long-Context Language Models
  (Effectively) (Oct 2024). ProLong; rejects perplexity and bare NIAH as progress signals.
- [2410.02694](https://arxiv.org/abs/2410.02694) — HELMET (Oct 2024).
- [2411.00640](https://arxiv.org/abs/2411.00640) — Adding Error Bars to Evals (Nov 2024).
- [2411.13476](https://arxiv.org/abs/2411.13476) — When Precision Meets Position: BFloat16
  Breaks Down RoPE in Long-Context Training (Nov 2024).
- [2412.10319](https://arxiv.org/abs/2412.10319) — SCBench (Dec 2024). Rankings do not survive
  multi-turn cache reuse.
- [2502.05167](https://arxiv.org/abs/2502.05167) — NoLiMa (Feb 2025; ICML 2025).
- [2503.15450](https://arxiv.org/abs/2503.15450) — SkyLadder (Mar 2025, rev. Dec 2025).
  Context-window scheduling during pretraining.
- [2506.16640](https://arxiv.org/abs/2506.16640) — Long-Context Generalization with Sparse
  Attention (Jun 2025, rev. Mar 2026; ICLR 2026). Softmax entropy `Θ(log n)` versus α-entmax
  `O(log s)`.
- [2508.07479](https://arxiv.org/abs/2508.07479) — Positional Biases Shift as Inputs Approach
  Context Window Limits (Aug 2025).
- [2510.00231](https://arxiv.org/abs/2510.00231) — The Pitfalls of KV Cache Compression
  (Oct 2025, rev. 2026).
- [2510.13334](https://arxiv.org/abs/2510.13334) — DefensiveKV (Oct 2025). Worst-case versus
  mean aggregation changes policy rankings.
- [2601.02872](https://arxiv.org/abs/2601.02872) — LongBench Pro (Jan 2026).
- [2602.15028](https://arxiv.org/abs/2602.15028) — Long Context, Less Focus: A Scaling Gap in
  LLMs Revealed through Privacy and Personalization (Feb 2026). Attributes degradation to
  "attention dilution under context scaling" as an inherent limit of soft attention in
  fixed-capacity transformers; ~29,000 instances, 1K–256K.
- [2602.16837](https://arxiv.org/abs/2602.16837) — A Structural Theory of Position Bias in
  Transformers (Feb 2026, rev. May 2026).
- [2605.10544](https://arxiv.org/abs/2605.10544) — Where Does Long-Context Supervision Actually
  Go? Effective-Context Exposure Balancing (May 2026). RULER +10.69 / NoLiMa +10.09 at
  Qwen2.5-0.5B; the only sub-1B evidence we have that these instruments have dynamic range.
- [2605.12366](https://arxiv.org/abs/2605.12366) — Classifier Context Rot: Monitor Performance
  Degrades with Context Length (May 2026). Frontier models miss dangerous actions **2× to 30×
  more often** when they occur after 800K tokens of benign activity than in isolation. The
  cleanest available statement that the failure is a *monitoring* failure.
- [2605.28079](https://arxiv.org/abs/2605.28079) — ATLAS (May 2026). Rank instability across
  length grids.
- [2606.27793](https://arxiv.org/abs/2606.27793) — Position Bias Correction is Insufficient for
  One-Pass Attention Sorting (Jun 2026). Debiasing identical to baseline on one model, 14.84 pp
  behind iterative sorting on another.
- [2606.23687](https://arxiv.org/abs/2606.23687) — Randomized YaRN (Jun 2026).
- [2607.08284](https://arxiv.org/abs/2607.08284) — PredicateLongBench (Jul 2026). Worst-case
  difficulty axes rather than average-case retrieval.
- Chroma Research, "Context Rot: How Increasing Input Tokens Impacts LLM Performance" (Hong,
  Troynikov, Huber, Jul 2025) — **industry technical report, no arXiv id**; cited for its claim
  only, and see §8 item 9.
- ROCm issue #6022 (librocdxg VRAM mapping under WSL2) — GitHub issue, not a paper.
