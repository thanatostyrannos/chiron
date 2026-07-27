---
title: Memory failure modes — the incident register, and why none of it pages you
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
mirrors: research/memory/memory-failure-register.md
prereqs: attention-variants-and-kv-cost, tensors-and-autograd, transformer-forward-pass-by-hand
difficulty: 3/5 conceptually, 4/5 methodologically — the math is one page, the epistemology is the hard part
time: 3–4 h reading; 2–3 h exercises (Exercise A is 20 min of compute, Exercise B needs a ~500 MB download)
---

# Memory failure modes

## 1. What this module settles

**One:** there are eleven named ways an LLM memory system fails, each with a mechanism you can
point at in code or in three lines of algebra, and nine of them collapse to a single defect —
*an irreversible discard decision taken before the query that needs the data is known* — which
makes them a scheduling problem, not a compression problem. **Two:** every one of them is silent
in the specific sense your career has trained you to fear most: no counter increments, no
latency percentile moves, the answer arrives fast, fluent, and wrong, and the only oracle is the
counterfactual you deliberately did not compute. **Three:** the benchmark the field uses to
certify eviction policies is adversely selected against the exact mechanism it is meant to test
— a needle is a maximally salient span, heavy-hitter eviction retains maximally salient spans,
so passing needle-in-a-haystack at any deployable budget is close to zero evidence of safety,
and that is the single strongest argument for this lab building an instrument instead of a
thirty-first eviction policy.

This module **teaches** `research/memory/memory-failure-register.md`, which **surveys**. Read the
register first; it is the evidence. This is the explanation, the arithmetic, and the argument
about what to do next. Where I extend the register I say so; where I would amend it I say so and
show my working (§8.6).

---

## 2. Theory in plain language

### 2.1 The bridge: this is an incident register, and you have run this meeting

You have kept the postmortem catalogue. Symptom, mechanism, blast radius, status, owner, action
items, and a standing rule that the register is append-only because the value of a postmortem is
that nobody tidied it afterwards. `research/memory/memory-failure-register.md` is exactly that
artefact for LLM memory. It has the same columns and the same discipline:

| Your postmortem catalogue | The memory failure register |
|---|---|
| Symptom as observed by a customer | Symptom as observed on a benchmark |
| Root cause, one line | Mechanism, one line, pointing at code or math |
| Severity | Status: `OPEN` / `OPEN/CONTESTED` / `PARTIALLY SOLVED` / `UNPROVEN DEMAND` |
| Evidence: dashboards, traces, the one log line | Evidence: `[C]` arXiv id, `[M]` our measurement, `[A]` assumption |
| "We think it's the load balancer" gets flagged as unconfirmed | 5-independent-primary-source bar; below it, marked `UNPROVEN DEMAND` |
| Append-only | Append-only |

Two details of that table repay attention because they are not decoration. The **five-primary-
source bar** exists because the field's failure claims propagate by citation the way a bad
root-cause propagates by Slack thread: one paper's Future Work becomes a survey's open problem
becomes a grant proposal's motivation, and nobody ever measured it. Exactly one row in the
register fails the bar and is kept, marked, precisely so the shape of non-evidence stays visible
— `per-session-kv-lifetime`, which rests entirely on a taxonomy noticing an empty cell in its own
grid `[C]` (2607.02574). A hole in a table looks like a finding and is not one.

The second is **`OPEN/CONTESTED`**, which has no analogue in your catalogue and should. It means
two competent groups disagree about the mechanism, both published this year, and the register
declines to pick. Two of the eleven rows sit there. Resist the urge to resolve them in your head;
the disagreement is usually a clue about a hidden variable, and in one case (§2.6) that hidden
variable is a knob a small rig can sweep and a frontier lab cannot be bothered to.

### 2.2 Where the bridge breaks — three places, and all three are the teaching

**Break 1 — there is no alarm, and there is no place an alarm could attach.**

In every system you have operated, a failure leaves a signal *somewhere*: a non-2xx, a p99, a
queue depth, a retry counter, a checksum mismatch, a page fault, a stack trace. Your entire
instinct — measure the slow path, alert on the derivative, bisect the deploy — is built on the
premise that failure is *observable in the machine's own terms*.

A KV cache has no miss. Not "a rare miss", not "a miss with a fast fallback": the concept does not
exist. Nothing is demand-paged, there is no backing store, and no code path anywhere asks "was
this token available?" When a token's key and value are gone, attention simply runs over a
shorter list and returns a fluent answer computed from less evidence, at exactly the same speed.
There is no counter to increment because there is no code site at which the absence is noticed.

The nearest thing your field has to this is **gray failure** — Huang et al., HotOS 2017, the
observation that a system's own view of its health can differ from the client's, so it stays
"up" while being useless (non-arXiv, cited as the framing, not as evidence). The 2026 production
study of an LLM agent runtime escalates gray failure by one rung and names the rung
`[C]` (2606.14589, Jun 2026): across eight weeks, 22 incidents with full postmortems, one
meta-pattern — *a failure whose error signal never reaches a human in actionable form* —
recurred at least 28 times, and its worst class is **"fail-plausible"**: the model does not
merely swallow the error, it converts the error into a confident narrative and hands it to the
user. That is a real incident register, from a real production system, and it is the closest
published thing to the postmortem catalogue you would write yourself.

> **The consequence for you, stated bluntly.** Your observability instinct will find nothing
> here, and it will find nothing *quickly and cheaply*, which is worse — you will conclude the
> system is healthy. The metric that would catch each of these failures is, in almost every case,
> a metric nobody runs, because running it costs the resource the mechanism was introduced to
> save.

**Break 2 — the postmortem has no ground truth, and obtaining it costs the thing you were saving.**

When your replicated store returns a stale row, you can reconstruct what should have happened:
the WAL exists, the primary's copy exists, the timestamps exist. When a compressed KV cache
returns a worse answer, the correct answer is *whatever the model would have produced with the
full cache*, and the only way to know it is to run the full cache. Attribution requires paying
the exact bill you were trying to avoid, on every probe.

That is why attribution research is structurally cheap at 300M and structurally unaffordable at
70B, and it is the load-bearing justification for this lab's scale. `[M]` Our fast tier is ≥62
GiB at ~200 GB/s (`ASSUMPTIONS.md → gpu-fast-tier-size`, `notebook/uma-carveout-controls-fast-tier.md`,
single run per arm, 2026-07-26); a 300M model's weights are ~600 MB. The oracle and the
experiment fit side by side with room to spare. Nobody serving a frontier model can do this, and
nobody needs to be cleverer than us to do it — they just cannot afford it.

**Break 3 — most of the evidence in this register is somebody else's telemetry, and some of it
is adversely selected.**

Your postmortems rest on your own instrumentation, which you trust because you built it. This
register rests on published benchmark numbers, and §2.5 shows that the most-cited of those
benchmarks is structurally incapable of failing for the most-published class of mechanism. You
would never accept a vendor's own load test as proof their fix worked. That is what most of this
literature is.

### 2.3 The organising finding: one defect wearing eleven names

Nine of the eleven rows reduce to the same sentence: **an irreversible discard decision, taken
before the query that needs the data is known.** The 2026 rate–distortion paper is the first to
say it out loud and unify consolidation, eviction, and pruning as one problem `[C]` (2607.08032,
Jul 2026).

> **Systems bridge — Belady's MIN.** You already know that optimal cache replacement requires
> knowledge of the future, that MIN is unimplementable online, and that every real policy is an
> approximation whose gap to MIN is the thing you are actually tuning. LLM memory is that, with
> the future being "which tokens the model will want to attend to."
>
> **Where it breaks, and it breaks three ways.**
> 1. **A wrong eviction costs correctness, not latency.** MIN's competitive-ratio framing assumes
>    that being wrong means servicing a miss. There is no miss to service. The cost function is
>    not "extra milliseconds", it is "a different answer", and you cannot integrate it over a
>    trace the way you integrate miss cost.
> 2. **The reference string is generated by the process you are perturbing.** Belady's analysis
>    assumes a fixed access sequence, independent of your policy. Here, evicting a token changes
>    the model's subsequent hidden states, which changes its subsequent queries, which changes
>    what it would have accessed. It is a closed loop. There is no "the" access sequence to be
>    optimal against.
> 3. **"Importance" is estimated, and the 2026 result is that it *cannot* be estimated
>    consistently.** Deterministic top-k eviction provably cannot know what it destroyed: an
>    adversary can alter the evicted values so that everything the serving system retains is
>    unchanged while the true attention-output error grows without bound, so no serving-time
>    estimator of that error is consistent `[C]` (2607.21475, Jul 2026). Randomising the tail —
>    Poisson-sample it at known inclusion probabilities and apply one logit offset inside the
>    softmax (the Hájek correction) — restores identifiability and gives a per-step error
>    certificate with 0.97 empirical coverage at no accuracy cost. Read that as: **you cannot get
>    an error bar out of a deterministic policy, at any price; you can get one nearly free out of
>    a randomised one.** For an engineer who has spent a career on sampled telemetry, that
>    sentence should land hard.

That paper deserves one more paragraph, because of *how* it reports. Its authors pre-registered
seven claims and lost three: question-aware eviction at 25–50% budgets is nearly free; output
log-probability predicts failure better than the certificate does; certificate-gated budget
escalation adds nothing. What survived is attribution — the certificate separates cache-induced
failures from inherent ones at AUC 0.73–0.75 against 0.47–0.54 for output confidence `[C]`
(2607.21475). A paper that publishes its own dead hypotheses is the standard this lab has already
committed to (CLAUDE.md → Experimental standards). It is rare enough to be worth naming.

### 2.4 The eleven, triaged by "would anything you run catch it?"

The register's table, re-sorted by **where the discard decision is taken**, with a column the
register does not have: the cheapest metric that would actually detect the failure. That column
is the point of the module.

| Slug | Discard happens at | Status | Would your dashboard catch it? | Cheapest detector |
|---|---|---|---|---|
| `eviction-destroys-long-range-recall` | cache-eviction time | OPEN | No | Full-cache oracle diff, per-token |
| `quantization-breaks-alignment-not-perplexity` | quantization time | OPEN | No — perplexity holds | Refusal rate; per-channel error probe |
| `prefix-cache-correctness-and-leakage` | reuse time | PARTIALLY SOLVED | Partly — TTFT variance is visible | TTFT distribution by tenant; salt audit |
| `ssm-recall-capacity-wall` | architecture time, every step | OPEN | No | MQAR at swept pair count |
| `hybrid-ratio-sensitivity` | architecture time, once | OPEN/CONTESTED | No | Recall-vs-distance curve, not perplexity |
| `position-bias-lost-in-the-middle` | never — it is a read bias | OPEN/CONTESTED | No | Accuracy vs. depth at fixed occupancy |
| `effective-context-collapse` | never — emergent | OPEN | No | RULER-style task battery, per length |
| `memory-induced-sycophancy` | agent-memory extraction time | OPEN (below evidence bar) | No | Agreement rate vs. ground truth, memory on/off |
| `memory-poisoning-cross-session-contamination` | write time, any channel | OPEN | No | Provenance tag per memory write |
| `forgetting-and-rollback-unsolved` | delete time | OPEN | No — there is no delete receipt | Post-delete retrieval probe |
| `attribution-gap-in-serving-results` | measurement time | OPEN — highest lab leverage | N/A — it *is* the dashboard failure | Fixed batch size, one mechanism at a time |
| `per-session-kv-lifetime` | — | UNPROVEN DEMAND | — | A workload trace; there isn't one |

Ten "no"s. That column is the module's thesis in one glance, and it is why the systems analogy
that opened this section is a scaffold rather than an answer.

Three of these rows deserve their mechanism spelled out in words before we do the algebra.

**`eviction-destroys-long-range-recall`.** Under StreamingLLM, SnapKV, TOVA, H2O and K-Norm,
*specific instructions* degrade far faster than the average and some are effectively ignored
entirely; the worked case is system-prompt leakage, with instruction order and eviction bias
identified as contributing factors `[C]` (2510.00231, Sep 2025, ACL 2026). Perplexity does not
move. LongBench does not move. One instruction out of eight silently stops being obeyed. If you
have ever shipped a config change that silently disabled one firewall rule out of eight while all
health checks stayed green, this is that, and there is no `diff` you can run.

**`quantization-breaks-alignment-not-perplexity`.** Quantise the KV cache and Mistral-7B loses
15.2% of its refusals at 1.03× perplexity — eleven instruction-tuned models from 3.8B to 72B,
five benchmarks, 1,894 prompts, sharp model-specific phase transitions, no universal safe bit
width `[C]` (2606.09864, Jun 2026). Safety behaviour occupies a low-dimensional activation
subspace reported as 10²–10³× more sensitive to quantisation noise than the mean representation;
perplexity averages over the full space, which is precisely where the damage is not. This is the
cleanest available demonstration of the register's central methodological claim: **the outcome
metric was fine and the mechanism was broken.**

**`attribution-gap-in-serving-results`.** The field's unit of evidence is an aggregate — tokens/s,
TTFT, task average — and papers almost never isolate which mechanism produced it. The 2026
serving survey classifies 30+ KV systems on four axes and names **seven KV-specific measurements
nobody makes** `[C]` (2607.02574); a companion survey reorganises the field by system behaviour
and reaches the same complaint from the other direction `[C]` (2607.08057). Four documented cases
where the gap produced a wrong conclusion: perplexity held while refusals collapsed `[C]`
(2606.09864); LongBench held while individual instructions were dropped `[C]` (2510.00231);
single-turn rankings did not survive multi-turn cache reuse `[C]` (2412.10319); mean-aggregated
rankings did not survive worst-case aggregation `[C]` (2510.13334).

> **A fifth case, not in the register v1.0.0, and it is the strongest one.** A May 2026 study runs
> seven eviction policies — LRU, H2O, SnapKV, StreamingLLM, Ada-KV, QUEST, Random — under one
> shared globally-capped decode-time harness and finds they share a prompt-boundary
> vulnerability: **without structural protection they all collapse to near-zero quality
> (F1 ≤ 0.064) on six pure-transformer models.** Reserving 10% of the cache at boundaries recovers
> 69–90% of quality; once boundaries are guarded, scoring differences are secondary — simplified
> scoring reaches within 0.02 F1 of plain LRU at K=32 `[C]` (2605.18053, May 2026). If that
> replicates, then the *scoring function* — the thing essentially every eviction paper is about —
> is a second-order term, and the first-order term is a structural rule (pin the sinks, pin the
> boundary) that most papers treat as a boilerplate implementation detail and do not ablate. I am
> folding this into the module rather than the register because I cannot edit a class-2 register
> from here; it belongs in `attribution-gap-in-serving-results` as a fifth documented instance and
> arguably as a sixth open question.

### 2.5 The NIAH adverse-selection argument, in full

This is the most important idea in the module and the one you should be able to reconstruct on a
whiteboard.

**What the benchmark is.** Needle-in-a-haystack: take a long, homogeneous, irrelevant context (the
haystack — typically Paul Graham essays or repeated filler), insert one short factual sentence
(the needle — "the best thing to do in San Francisco is eat a sandwich in Dolores Park"), ask a
question whose answer is the needle, and score retrieval accuracy on a grid of context length ×
insertion depth. It is the standard smoke test for long context and it appears in the evaluation
section of most KV-compression papers.

**What heavy-hitter eviction is.** H2O keeps the tokens with the largest accumulated attention
mass — the "heavy hitters" — plus a recent window `[C]` (2306.14048, Jun 2023). SnapKV scores
tokens by the attention they receive from an observation window at the *end* of the prompt, i.e.
from the question, and keeps the top-k `[C]` (2404.14469, Apr 2024). StreamingLLM keeps the first
few tokens plus a recent window `[C]` (2309.17453, Sep 2023). Every one of them is a top-k
selection under a salience score.

**The argument.** A needle is, by construction, the single most salient span in the context:

- Under a **question-aware** scorer (SnapKV, QUEST, and most 2025–2026 policies), the score *is*
  the attention from the question. The needle is the only span in the whole context semantically
  related to the question. Its score is at or near the maximum **by the eval's design**.
- Under a **question-agnostic** accumulated-mass scorer (H2O), the needle is a distributionally
  odd span inserted into a homogeneous haystack. Low-probability, locally surprising tokens
  attract attention. Its rank is high for a different reason and the same effect.

So the eval's target is placed exactly where the policy's retention set is. Detection requires the
needle to be *evicted*, and the eval's construction makes that the last thing that happens.

**Write it as a power calculation, because that is what it is.** Let `T` be context length, `k`
the retained budget in tokens, `ρ = k/T` the budget fraction, and `R` the rank of the target span
under the policy's scoring function (rank 1 = highest score). A top-k policy evicts the target iff
`R > k`, so:

```
detection  ⟺  R > ρ·T        ⟺        ρ < R / T
```

The eval detects a bad policy only in the budget regime below `R/T`. That is a step function in
`R`, and `R` is set by the *eval author*, not by the policy author. Numbers, at `T = 32,768`:

| Target | Plausible rank `R` | Detected only when budget `ρ <` | Deployed budgets |
|---|---|---|---|
| NIAH needle (question-aware scorer) | ~10 | 0.03% | 10–50% |
| NIAH needle (accumulated mass) | ~100 | 0.31% | 10–50% |
| A buried instruction, mid-prompt | ~8,000 | 24.4% | 10–50% |
| A low-salience aggregation target | ~20,000 | 61.0% | 10–50% |

`[A]` The rank column is illustrative, medium confidence — the *shape* is the claim, the specific
ranks are what Exercise B measures. The table says the thing worth internalising: **at any budget
anyone actually deploys, NIAH is a test the policy passes by construction, while the failure mode
that is documented in the literature — a dropped instruction `[C]` (2510.00231) — sits four orders
of magnitude away on the same axis.**

**Say the strong form and the honest form, and pick.** The strong form, as `research/synthesis.md`
states it, is that NIAH "structurally cannot fail" for heavy-hitter policies. The honest form is
that it is a **low-power test with adversely-selected targets**, not an impossibility theorem:
published work does report failure bands at intermediate depths for SnapKV and Ada-KV at long
contexts, where a needle that does not rank in the top-k is evicted regardless of budget headroom.
I would defend the honest form and I would defend the operational conclusion unchanged: *a NIAH
pass at a deployable budget carries almost no information about eviction safety, and reporting one
as evidence of safety is a methodological error.*

> **Systems bridge.** You have seen this exact failure: benchmarking a cache with a synthetic
> workload whose working set fits inside the cache. Every policy looks perfect; the benchmark is
> measuring the workload, not the policy.
>
> **Where it breaks — and this is why it is worse here.** With a cache you can *compute* the
> working set size before you run, compare it to capacity, and know immediately that your
> benchmark is void. Here, "salience" is defined by the model's own attention distribution, which
> you cannot know without a forward pass, and which differs per layer, per head, per query. There
> is no offline check that tells you your eval is adversely selected. **You have to measure the
> target's rank, and nobody does.**

**The generalisation, which is the actually useful output.** *Any eval whose target is the most
salient span in the context is adversely selected against any policy that retains salient spans.*
This is a property of the **eval–policy pair**, not of either alone. Corollaries:

1. NIAH is fine for testing *positional* failures (§3.5) and useless for testing *salience-based*
   retention.
2. It becomes informative again against a policy that does **not** score by salience — random
   eviction, uniform eviction, block-granular eviction, quantisation. Which is precisely why
   quantisation papers that report NIAH are less compromised than eviction papers that do.
3. The fix is not a better needle. The fix is **rank-stratified targets**: place the required fact
   at controlled attention-mass percentiles — bottom decile, median, top decile — and report
   accuracy as a function of the target's rank rather than of its depth. As far as I can find,
   nobody publishes this, and it costs one extra forward pass per prompt to construct. That is a
   methodology contribution available to a one-person lab this quarter.

### 2.6 The two contested rows, and why you should not resolve them

**Position bias.** Retrieval accuracy on an identical fact varies by tens of points depending only
on its index in the prompt, in a U — primacy and recency preserved, middle suppressed `[C]`
(2307.03172, Jul 2023). Is the U architectural or correctable? The structural-theory line derives
U-shaped influence profiles from causal masking plus residual connections alone, via
residual-aware cumulative attention rollout — i.e. before any data is seen, so more long-context
training will not remove it `[C]` (2602.16837, Feb 2026). The calibration line recovers up to 15
points by correcting attention bias at inference time, which implies it is substantially
correctable `[C]` (2406.16008, Jun 2024). Separately the *shape itself* is contested: the U holds
only to roughly 50% context occupancy, above which primacy decays and the bias becomes
distance-based `[C]` (2508.07479, Aug 2025). Papers reporting a clean U and papers reporting
recency dominance may be probing different occupancy regimes — a hidden variable, and a cheap one
to sweep.

**Hybrid ratio.** Does the full-attention:linear ratio set a capability *ceiling* or only the
*rate* at which long-context ability emerges? 72 trained models across six linear variants and
five ratios find recall degrading sharply below ~3:1 — a ceiling `[C]` (2507.06457, Jul 2025, rev.
Jun 2026), corroborated independently for Mamba-Transformer hybrids `[C]` (2510.26912, Oct 2025).
The opposite reading: different configurations converge to comparable performance given enough
training, and larger SWA windows *delay* retrieval-head formation — "Large-Window Laziness" `[C]`
(2606.15378, Jun 2026). Same year, same question, incompatible framings. The hidden variable is
almost certainly token budget, which is exactly the axis a 20M–300M rig can sweep and a frontier
lab will not.

Do not resolve either in the curriculum. Note the hidden variable and put it in the backlog.

---

## 3. The math that actually matters

Four pieces of algebra. Each is short, each has a number attached, and each changes what you would
build.

### 3.1 Eviction is a distribution edit, not a deletion

Attention weight on cached token `i` for the current query:

```
a_i = exp(s_i) / Σ_{j ∈ retained} exp(s_j)        where  s_j = (q · k_j) / √d
```

| Symbol | Reads as |
|---|---|
| `q` | the query vector of the token being generated right now, one per head |
| `k_j` | the key vector cached for past token `j` |
| `d` | head dimension; `√d` is a variance-stabilising divisor so scores do not blow up with width |
| `s_j` | the pre-softmax score of past token `j` for this query — a raw affinity, unnormalised |
| `a_i` | the fraction of this head's output that comes from token `i`'s value vector |
| `E` | the evicted set |
| `Z_full = Σ_{all j} exp(s_j)` | the softmax denominator with nothing evicted |
| `Z_keep = Σ_{j ∉ E} exp(s_j)` | the denominator after eviction |

Now the point. Evicting `E` does not remove `E`'s contribution and leave everything else alone,
because the denominator shrinks:

```
before:   a_i = exp(s_i) / Z_full
after:    a_i = exp(s_i) / Z_keep          with  Z_keep < Z_full
```

Every survivor's weight goes **up**, by exactly

```
A  =  Z_full / Z_keep  =  1 / m          where  m = Z_keep/Z_full  is the retained attention mass
```

Call `A` the **amplification factor**. Keep 90% of the mass and every survivor is scaled by 1.111.
Keep 50% and every survivor is scaled by 2.0. Eviction is a *redistribution of attention mass*,
not a deletion — which is why removing the first few tokens is catastrophic rather than merely
lossy: they are attention sinks that absorb mass the model has no better use for, and removing
them forces that mass onto content tokens `[C]` (2309.17453). Every eviction policy since pins a
prefix for exactly this reason.

### 3.2 The decomposition: information loss versus renormalisation

This is not in the register and it falls out of §3.1 in two lines. Define three outputs for one
head, one query:

```
out_full     = Σ_{all j}  (e_j / Z_full)  v_j              the oracle
out_evict    = Σ_{j ∉ E}  (e_j / Z_keep)  v_j              what a real cache returns
out_massfix  = Σ_{j ∉ E}  (e_j / Z_full)  v_j              the denominator-preserved control
```

where `e_j = exp(s_j)` and `v_j` is token `j`'s value vector. `out_massfix` keeps exactly the same
tokens as `out_evict` — identical information loss — and differs only in that the denominator was
not shrunk. Then, exactly:

```
out_full − out_massfix  =  Σ_{j ∈ E} (e_j / Z_full) v_j          ← pure information loss
out_evict               =  A · out_massfix                        ← pure common-mode gain
```

So the total error splits into a **dropped-contribution term** and a **scalar gain of `A` on
everything that survived**. Two different defects, two different fixes: the first needs a better
selector, the second needs one float.

That second term is the interesting one, because a common-mode gain does not cancel. In `D`
dimensions with many small dropped weights, the dropped-contribution term is governed by the
*L2* norm of the discarded weight vector — many small errors in random directions partially
cancel — while the gain term scales the whole output coherently. The prediction is that the gain
term dominates the error norm. It does:

`[M]` **Computed 2026-07-26, CPU numpy, `T=4096` cached tokens, `D=128` value dimension, 512
independent (score, value) draws per seed, seeds 0/1/2, Zipf score profile with exponent
`α = 1.0` plus N(0, 0.5) jitter, oracle top-k retention. Values reported as mean ± half-range
across the three seeds.** Relative error is `‖out − out_full‖ / ‖out_full‖`.

| tokens kept | mass kept `m` | `A = 1/m` | rel. err, true eviction | rel. err, denominator-preserved | ratio |
|---|---|---|---|---|---|
| 50% | 0.9351 ± 0.0003 | 1.069 | 0.0703 ± 0.0003 | 0.0107 ± 0.0002 | **0.152 ± 0.002** |
| 25% | 0.8574 ± 0.0006 | 1.166 | 0.1677 ± 0.0009 | 0.0204 ± 0.0005 | **0.122 ± 0.002** |
| 10% | 0.7540 ± 0.0011 | 1.326 | 0.3289 ± 0.0019 | 0.0365 ± 0.0008 | **0.111 ± 0.002** |
| 5% | 0.6757 ± 0.0014 | 1.480 | 0.4842 ± 0.0028 | 0.0535 ± 0.0012 | **0.111 ± 0.002** |
| 2% | 0.5723 ± 0.0019 | 1.747 | 0.7549 ± 0.0053 | 0.0863 ± 0.0020 | **0.114 ± 0.002** |
| 1% | 0.4946 ± 0.0023 | 2.022 | 1.0321 ± 0.0094 | 0.1225 ± 0.0026 | **0.119 ± 0.001** |

**Read the arithmetic check first**, because it proves the decomposition rather than the code. At
10% retention, `A − 1 = 0.326`, and `‖out_massfix‖ ≈ ‖out_full‖` to within 3.6%, so a pure gain
of `A` should produce a relative error of about 0.326. Measured: 0.329. The entire error is the
gain term to two significant figures.

**What this does and does not license.** It licenses: on this synthetic distribution, ~89% of the
attention-output error norm under oracle top-k eviction is the softmax renormalisation, not the
lost information, and the split is stable across two orders of magnitude of budget. It does **not**
license any claim about a real model — the value vectors here are i.i.d. Gaussian, and in a real
model the discarded values are correlated with each other and possibly with the answer direction,
which is exactly the case where the dropped-contribution term stops cancelling. `[A]` Medium
confidence that the direction of the effect survives to real attention; the cheapest test that
would move it is Exercise A's real-model extension, and this is register open question 1.

Two more results from the same run, same config, one seed (seed 0, 256 trials) — **anecdotes by
the house standard, labelled as such**:

- **Peakedness is the whole story.** At 10% retention, sweeping the Zipf exponent `α`: at `α=0.5`
  (near-flat attention) the retained mass is 0.351 and eviction error is 1.716 — larger than the
  signal. At `α=2.0` the top 10% of tokens hold 99.87% of the mass and eviction error is 0.0013.
  Same policy, same budget, three orders of magnitude of damage, decided entirely by how peaked
  the attention distribution happens to be for that query. **This is the mechanism behind
  "eviction is task-dependent" and behind the coverage finding** — that the number of *unique*
  tokens retained predicts degradation better than the eviction rate does `[C]` (2606.29563, Jun
  2026). Eviction rate is a bytes metric; coverage is a distinct-keys metric; the attention
  profile is what converts between them, and it is per-query.
- **Sink eviction is almost pure renormalisation, as the decomposition predicts.** Force-evicting
  the `n` highest-mass tokens with value vectors set to zero (a caricature of an attention sink:
  absorbs mass, contributes nothing) at 10% retention: `n=4` gives `A = 1.97` and eviction error
  0.975, while the denominator-preserved control gives 0.096. Ratio 0.098. At `n=64`, `A = 4.49`,
  errors 3.30 vs 0.344. `[A]` Reading, medium confidence: **StreamingLLM's prefix pin is a
  denominator fix in disguise** — it works because it preserves `Z`, not because the first four
  tokens contain information.

**Why you cannot simply apply the fix, and why that is the interesting part.** To compute
`out_massfix` at serving time you need `Z_full`, which requires the scores of the tokens you
threw away, for *this* query. You knew the discarded mass at eviction time, for the query you had
*then*. The discarded mass is query-dependent, so the correction must be *estimated* — which is
precisely the problem `[C]` (2607.21475) solves with Poisson sampling and the Hájek offset. The
two results are the same result approached from opposite ends: the decomposition says the
denominator is where the damage is; the randomised design says a sampled tail is what lets you
estimate the denominator with a confidence interval. That is a design, not a coincidence, and it
is the most concrete thing in this module for Mnemosyne.

### 3.3 The constant-state capacity wall, and the fact that surprises people

A linear-attention or delta-rule layer keeps one matrix `S ∈ R^{d_k × d_v}` and nothing else.
`d_k` is the key dimension, `d_v` the value dimension. Writing the pair `(k, v)` adds an outer
product `k vᵀ`. Reading with query `q`:

```
read(q) = Sᵀ q = Σ_i v_i (k_i · q)
```

In words: the read returns a weighted sum of *every* value ever written, weighted by how much its
key resembles the query. If the stored keys were mutually orthogonal and `q = k_m`, every term
but `m` would vanish and the read would be exact. They are not orthogonal — keys are
L2-normalised continuous vectors in `d_k` dimensions, and you cannot pack more than `d_k` mutually
orthogonal directions into `d_k` dimensions.

**Do the arithmetic for the interference.** For `N` stored pairs with roughly random unit keys,
`k_i · k_m ≈ N(0, 1/d_k)` for `i ≠ m`. So

```
read(k_m)  =  v_m  +  Σ_{i ≠ m} v_i (k_i · k_m)
              ↑signal        ↑interference
```

The interference term is a sum of `N−1` independent contributions each of expected squared norm
`‖v‖²/d_k`, so its norm is about `‖v‖·√((N−1)/d_k)`. The signal-to-interference ratio is

```
SIR  ≈  √( d_k / (N − 1) )
```

**Set SIR = 1 and you get the capacity: `N ≈ d_k` stored associations.** At `d_k = 128`, about 128
pairs before interference equals signal. This is the closed form behind the empirical
recall-versus-state-size frontier `[C]` (2312.04927, Dec 2023; 2402.18668, Feb 2024), and the 2026
theory recasts it as spherical packing with a Welch-bound interference floor — a hard limit, not
an optimisation target `[C]` (2607.17419, Jul 2026).

**The surprising consequence, which is the reason this is in the math section.** Capacity in
*associations* is governed by `d_k` alone. State *bytes* are `d_k × d_v`. So **doubling `d_v`
doubles your memory footprint and buys exactly zero additional retrievable associations** — it
only widens each stored value. If you are designing a constant-state layer and you want recall,
the budget goes into `d_k`. That fact is invisible in any "state size in KB" comparison, and
"state size in KB" is how the entire hybrid literature reports its memory axis.

> **Systems bridge.** A fully-associative cache with a global TTL tick, `β` as write strength, and
> the delta term as a compare-and-swap. **Three breaks, all load-bearing.** (1) *There are no
> lines and no addresses* — a *similar* key partially clobbers a neighbour's content, so the
> failure is interference, not a miss. (2) *There is no capacity miss and no eviction policy* —
> the decay multiply destroys old content every step whether or not new content arrives, and there
> is no tier to spill to. (3) *It is not a write-ahead log* — a KV cache is an append-only exact
> log you can rescan; this state is destructive-update and unreplayable, so token 5's contribution
> cannot be recovered at token 5000.
>
> **And one correction to the folk model, verified in code:** the gate cannot forget selectively.
> The decay is a single scalar per head applied to the entire `d_k × d_v` matrix, so every stored
> association is attenuated identically. "Gating = selective forgetting" is backwards: the gate is
> indiscriminate decay, and the *delta term* is the targeted erase. See §5.4.

### 3.4 The prefix-cache hash chain, and why the same property enables the attack

vLLM's prefix-cache key is a hash **chain**, not a content hash:

```
h_0 = H(∅,       t[0:B],           extra)
h_i = H(h_{i−1}, t[iB:(i+1)B],     extra)
```

`B` is the block size in tokens (16 by default), `t[·]` the token ids in that block, `extra` a
namespacing salt (LoRA id, tenant salt). Folding the parent hash in makes the key strictly
prefix-ordered. Three consequences, all mechanical:

1. **The same 16 tokens at a different offset are a different key.** There is no associative match
   and no middle-of-sequence match. The match loop breaks at the first miss because a later hit is
   impossible by construction (§5.3).
2. **A 100% prefix match never skips 100% of the work.** The hit is capped at `num_tokens − 1`
   (you need a forward pass to produce logits) and then floored to block alignment. Arithmetic: an
   exact-duplicate prompt of 1,024 tokens caps at 1,023, floors to `⌊1023/16⌋ × 16 = 1,008`, so
   **16 tokens — a whole block — are recomputed.** A 1,000-token prompt caps at 999, floors to 992,
   recomputing 8. Worst case is exactly `B` tokens and it happens whenever the prompt length is a
   multiple of the block size.
3. **The chain makes the timing attack incremental, which is what makes it feasible.** A hit is
   faster than a miss, so TTFT is a side channel `[C]` (2409.20002, Sep 2024). Because the key is
   prefix-ordered, an attacker can confirm block 1 before guessing block 2: recovering `n` blocks
   with `V` candidates each costs `O(n·V)` probes, **linear**, not `O(V^n)`. The property that
   makes reuse correct is the property that makes the attack cheap. Extended to non-prefix KV in
   RAG in 2026 `[C]` (2606.21842), with a dedicated mitigation `[C]` (2603.10726).

**Status: PARTIALLY SOLVED, and be precise about which part.** Namespacing ships and works —
SGLang's `extra_key` partitions the radix tree like an ASID so identical token prefixes stay
disjoint across tenants, and vLLM added cache salting for the same reason (vLLM RFC #16016;
GitHub engineering artifact, not peer-reviewed). That closes the cross-tenant *content* hazard
**when operators use it**. It does not close the timing channel, and it does not make non-prefix
reuse correct — concatenating independently computed KV chunks loses the cross-attention between
them, and CacheBlend repairs it only *approximately*, by selectively recomputing a small
high-deviation subset `[C]` (2405.16444, May 2024).

### 3.5 The evaluation-power arithmetic, restated as something you can act on

From §2.5: a top-k policy hides a target of rank `R` at any budget `ρ ≥ R/T`. Turn it around into
a design rule. If you want an eval with power against budget `ρ*`, the target must be *evictable*
at that budget, which means its rank must be **worse** than the retained set:

```
R  >  ρ* · T
```

Rank 1 is the most salient token, so "worse rank" means a numerically larger `R`. At `T = 32,768`
and `ρ* = 0.10` the threshold is `R > 3,277`: the target must sit **outside the top 10% of the
salience distribution**. A needle is the opposite of that by construction. An instruction buried
at 5% depth in a long prompt plausibly satisfies it, which is precisely the failure the literature
actually documents `[C]` (2510.00231).

**This is a two-line calculation that no KV-compression paper I can find performs**, and it is the
difference between an eval that can fail and an eval that cannot. Exercise B measures `R`.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 The decision this module supports

`research/synthesis.md` says: **ship an attribution instrument as the lab's deliverable, and add
no new eviction policy to a field that has ~30 of them and no dominance result.** This module is
the argument for the first half. You cannot ship a policy you cannot certify; nothing currently
shipped certifies; and the most-used certification instrument is adversely selected against the
mechanism it certifies. Building policy number 31 would be measured with the same broken ruler
that produced the previous thirty.

### 4.2 Six concrete design consequences for Mnemosyne

**(1) Mnemosyne must synthesise the counters the architecture refuses to emit.** There is no miss
signal, so build one: an **eviction journal**, append-only, one record per discard decision —
`(step, layer, head, position, score, rank, budget, retained, discarded_mass)`. Note that
`discarded_mass` is the `Z_full − Z_keep` from §3.2 and is free at eviction time because you
already computed the scores to make the decision.

> **Systems bridge.** This is an audit log for a control-plane decision, not for the data. You
> have built these. **Where it breaks:** the audit log is larger than the thing being audited —
> one record per token per head per layer per step is `O(L · n_kv · T)` per step, which at
> Laguna's shape is 48 × 8 = 384 records per token per step. You must sample. **And here is the
> convergence that makes this a design rather than a compromise:** the sampling you need for
> observability is the *same* Poisson sampling with known inclusion probabilities that the error
> certificate needs `[C]` (2607.21475). One mechanism, two payoffs. Design the journal as a
> sampled tail from the start, not as a full log you later downsample.

**(2) The denominator-preserved control is a first-class arm, not a debugging flag.** §3.2 says
~89% of the error norm on synthetic data is renormalisation. If that survives to a real model,
then the field has been building better *selectors* for a problem whose dominant term is a
*scalar*. Name the arm for the hypothesis, per the naming rule: `mnemosyne-denominator-preserved`
against `mnemosyne-h2o` at matched budget, with `mnemosyne-full-cache` as the oracle. Pre-register
it; the KILL condition is "the control recovers <20% of the loss," which would refute the
mechanism cleanly.

**(3) Every eval gets calibrated by fault injection before it is allowed to certify anything.**
The synthesis specifies a six-fault battery (needle absent, needle's KV dropped, uniform eviction,
RoPE-phase corruption, retrieval-head masking, haystack shuffle). Add **fault zero: measure the
target's salience rank** (§3.5). An eval that cannot detect a fault you injected on purpose cannot
detect one you did not. This is the standard your DR practice already holds — an untested restore
is not a backup — applied to measurement instead of storage.

**(4) The package boundary is under real pressure here, and you must not relieve it.** Attribution
instrumentation wants the model's attention weights, which makes `import proteus` inside
`mnemosyne/` feel natural. It is the one shortcut that destroys the result (CLAUDE.md → Boundary
rule). Design the interface so the instrument consumes **scores as data** — a tensor and a shape
contract — never a model handle. If the eviction journal can be replayed from a recorded score
stream produced by any model, Mnemosyne is separable; if it needs a live Proteus object, it is an
implementation detail.

**(5) The cost model must be per-layer-type, and failures are too.** Laguna is 12 global + 36
sliding at `w=512` `[M]` (`ASSUMPTIONS.md → reference-model`). A windowed layer's discard is
*lossless* — the mask makes out-of-window tokens architecturally unreadable, so discarding them is
a proof, not a bet (see `attention-variants-and-kv-cost.md` §2.3). A global layer's discard is a
bet. **An eviction policy applied uniformly across a hybrid stack is applying a bet-shaped
mechanism to proof-shaped layers**, and any aggregate "compression ratio" that averages the two is
meaningless. Mnemosyne's budget allocation must be aware of layer type from the first commit;
retrofitting it is worse than building it.

**(6) The trap that would kill the obvious hybrid experiment.** You cannot test long context by
widening the sliding windows. Position enters attention through the key vector itself, because
rotary embedding has already rotated it by an angle proportional to its index `[C]` (2104.09864,
Apr 2021) — and Laguna's global layers apply YaRN-scaled RoPE over 64 of 128 head dims at
θ=500,000 while SWA layers apply plain RoPE over all 128 at θ=10,000 (§5.5). The SWA layers were
never trained with a positional encoding that reaches past their window. Widen the window and you
will measure the encoding failure, not the ratio.

### 4.3 Our own instrument is an instance of the module's theme

Two rows in `ASSUMPTIONS.md` are, structurally, entries in this register:

- `[M]` **`large-tensor-fault-32gib`**: a 31 GiB buffer copies cleanly at 199.9 GB/s; a 32 GiB
  buffer **hard-hangs for 11 minutes at 0% CPU** with host free RAM draining to 5 GB, and had to be
  force-killed. *A hang at 0 CPU is silent.* Process liveness checks pass. GPU utilisation is the
  metric that would catch it, and it is not the one anybody watches on a workstation.
- `[M]` **`sdpa-is-memory-efficient` — refuted by default**: `F.scaled_dot_product_attention`
  retains the score matrix at **147.2 bytes/T²** by default and 6.6 with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, while `flash_sdp_enabled()` returns `True` **either
  way** — the API reports what is *permitted*, not what *ran*. That is a health check that lies,
  which is the purest form of the failure this module is about.

**These two compose into a specific hazard for the very harness this module tells you to build.**
The 147.2 bytes/T² figure was measured at `B=4, nh=8`, so the per-`(B·nh)`-unit coefficient is
`147.2/32 = 4.6` bytes/T². A full-cache oracle over one 8-head layer at batch 1 therefore retains
`8 × 4.6 = 36.8` bytes/T², and hits the 31 GiB per-tensor hazard at

```
T = √( 31 × 2^30 / 36.8 )  ≈  30,000 tokens
```

`[A]` High confidence in the arithmetic, medium in the transfer — it is a derivation over two `[M]`
inputs measured at a different shape, not a measurement at oracle shape, and the cheapest test is
to run it. The operational point stands regardless: **your attribution harness is the process most
likely to trip the silent hang, because it is the one materialising T² scores.** Set the AOTriton
flag for oracle runs, chunk the oracle over the key axis, and put a GPU-utilisation watchdog on
long runs. Note that the flag is experimental and therefore a numerics change — the Hardware
Validation Gate must run numerics both ways before it becomes a default.

### 4.4 What this module says to leave alone

Agent-memory security is the best-evidenced pain in the register — poisoning at >80% attack
success with <0.1% poison rate `[C]` (2407.12784, Jul 2024); a write-channel taxonomy in which
**your summariser is an attack surface** `[C]` (2606.04329, Jun 2026); dormant delayed-trigger
variants `[C]` (2605.15338, May 2026); violation rates rising monotonically with exposure length,
so short-horizon evaluations systematically under-report harm `[C]` (2605.17830, May 2026). It is
also a different research programme requiring multi-session agent infrastructure this lab does
not have. Park it with a written un-park trigger. The same goes for distributed/disaggregated KV:
highest interest, disqualified by `single-device-only`.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run `scripts/fetch_reference.sh`
first. Line numbers are pinned to the revisions in `PROVENANCE.md`. Every pointer below was opened
and the named symbol confirmed on the named line on 2026-07-26.

Read these as **evidence for register entries**, not as a tour. Each row states what to look for
and what it proves.

### 5.1 There is no miss signal — four pointers that prove it

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:184` | `class FreeKVCacheBlockQueue`. An intrusive doubly-linked list threaded through the `KVCacheBlock` objects themselves. **Look for what is absent:** there is no separate victim cache. The free list *is* the LRU victim cache — freed blocks keep their contents and their hash. |
| `memory/vllm/vllm/v1/core/block_pool.py:719` | `free_blocks`. It decrements `ref_cnt` and pushes back onto the queue. **It does not remove the hash-table entry.** A zero-refcount block is still matchable. "Freeing is not evicting" is a line of code, not a slogan. |
| `memory/vllm/vllm/v1/core/block_pool.py:702` | `touch`. On a prefix hit, the block is spliced out of the free queue in O(1) and resurrected. This is why "blocks in use" and "entries available for hits" are two different numbers, and why any dashboard reporting one as the other is wrong. |
| `memory/vllm/vllm/v1/core/block_pool.py:679` | `_maybe_evict_cached_block` — **the only place a block leaves the hash table**, called lazily from `get_new_blocks` at *reallocation* time. Eviction happens when someone needs memory, not when something becomes cold. Ask yourself: at what instant would you emit an "evicted" metric here, and what would its timestamp mean? |

**The question to answer before moving on:** where would you add a counter that increments when a
token the model *would have attended to* is no longer resident? Work through it honestly. There is
no such site, because nothing in the code ever forms the counterfactual.

### 5.2 Three production systems, three incompatible definitions of "the victim"

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/block_pool.py:679` | vLLM: **lazy, at reallocation, LRU-ordered over refcount-zero blocks.** Granularity is the block; when allocation fails the whole request is preempted, so the *effective* eviction granularity is the sequence. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | SGLang `evict`: a heap built fresh on every call from `evictable_leaves` — **only leaves of the radix tree are ever candidates.** A hot child keeps a cold parent resident indefinitely. Eviction order is *topological*, not recency-ordered. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `LRUStrategy.get_priority` returns `node.last_access_time`. The entire replacement-policy surface is one function; LFU, FIFO, MRU and segmented-LRU sit in the same short file as tuple-valued comparators. **The policy is trivially swappable and the topological constraint above is not.** That asymmetry is the finding. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | Mooncake `BatchEvict`: only objects whose **lease has expired** are candidates, then an `nth_element` partial sort on lease deadline. Not LRU at all. The "hot" signal is a TTL renewal on `Get`, not a touch bit. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:5211` | `TryPushPromotionQueue` — disk→DRAM promotion gated by a count-min-sketch frequency threshold (TinyLFU-style admission) then by the DRAM high-watermark. **A single cold hit is designed to lose.** Contrast with the vLLM `touch` above, where one hit resurrects immediately. |

**What this proves.** "Eviction rate" is not comparable across systems, because "victim" is not the
same object. Any cross-system claim about eviction — and the serving literature is full of them —
is comparing three different mechanisms under one word.

### 5.3 Prefix reuse: position-locked by construction

| Where | What to look for |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)`. **The parent hash is an argument.** That single parameter is what makes the key a chain and reuse position-locked (§3.4). |
| `memory/vllm/vllm/v1/core/single_type_kv_cache_manager.py:658` | `FullAttentionManager.find_longest_cache_hit`. Walks block hashes from token 0 and **breaks at the first miss** — a later hit is impossible by construction, so this is not an optimisation. |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:225` | `get_computed_blocks`. Read the comment immediately below the docstring: *"When all tokens hit the cache, we must recompute the last token."* This is the `num_tokens − 1` cap from §3.4, in the code, with the reason. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355` | `match_prefix`, and read the whole docstring. `extra_key` namespaces the tree **like an ASID** — identical token prefixes with different LoRA ids or cache salts are kept disjoint and never share nodes. This is the shipped mitigation for cross-tenant content leakage, and the docstring is its clearest statement anywhere. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:217` | `class TreeNode`. Note the four fields every policy reads — `lock_ref`, `last_access_time`, `hit_count`, `creation_time` — and `children` being a `defaultdict(TreeNode)`, which means a bare `node.children[k]` on a miss silently *creates* a phantom node. |

> **The break that will surprise you most:** `match_prefix` is a **mutating read**. A lookup that
> terminates mid-node calls `_split_node`, which cleaves the extent and clones two tensors. Read
> cost is not read-only and lookups cannot run concurrently with anything. Every filesystem
> instinct you have says a lookup is safe to parallelise. Here it is a write.

### 5.4 Constant state: the destructive update, in two repositories

| Where | What to look for |
|---|---|
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:54` | `h = h.clone() * g[:, :, i].exp()[..., None, None]` — **one scalar multiplies the entire `d_k × d_v` state.** Every stored association is attenuated identically. There is no per-key TTL. |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:56` | `b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)` — the delta term: read what the *decayed* state already returns for this key and subtract it, so the write carries only the residual. **This is the targeted erase, and it is the only content-dependent selectivity in the layer.** |
| `architecture/flash-linear-attention/fla/ops/gated_delta_rule/naive.py:58` | `h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)` — a rank-1 outer product. The only place new information enters memory, ever. |
| `architecture/mamba/mamba_ssm/ops/triton/ssd_state_passing.py:80` | `states = scale * states + new_states`. One line, and it is the proof that the carry is destructively overwritten rather than appended to. Token 5's contribution cannot be recovered at token 5000 — not "expensively", *not at all*. |

**Look at the ordering at `:54` and `:56`.** Decay is applied *before* the residual is computed, so
the correction is measured against the post-decay state. `β = 1` is an exact overwrite of that
key's direction; `β = 0` is a no-op.

### 5.5 The architecture-time discard, and the experimental trap

| Where | What to look for |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:365` | `self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"`. The entire hybrid mechanism is a list lookup at construction. Every hybrid-ratio question the lab asks is a question about what goes in that list — which is what makes `hybrid-ratio-sensitivity` cheaply ablatable. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:41` | `hparams.set_swa_pattern(swa_period, /*dense_first=*/true)` — the same decision in C++, full attention at `il % 4 == 0`. Note the fallback: absent the `sliding_window` key, the whole hybrid path is skipped and the model is all-full-attention. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | `const int n_rot_l = is_swa_il ? hparams.n_rot_swa : n_rot;` and the four lines under it. **This is the trap.** SWA layers get plain RoPE with YaRN `ext_factor`, `beta_fast`, `beta_slow` forced to zero; full layers get the YaRN long-context schedule. The two layer types are not numerically interchangeable, so you cannot widen the window to test long context. |

---

## 6. Exercises

Three. All three run on CPU; none needs the GPU, which is deliberate — the point of this module is
measurement, not throughput, and a CPU result you can reproduce on a train is worth more than a GPU
result you run once.

Activate first, in PowerShell, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

Standing caveats from `ASSUMPTIONS.md`: single tensors **≥32 GiB hang silently at 0% CPU**
(`large-tensor-fault-32gib`); bf16 numerics on gfx1151 are **untested** (`bf16-numerics-unproven`);
the Hardware Validation Gate has not run, so nothing measured here is evidence by house standard
until it does. Write scratch scripts under `notebook/`; on reuse they migrate into the rig and
acquire tests.

**Pre-register before you run.** Each exercise below states a prediction and a falsification. Write
the G2 hypothesis card first (HYPOTHESIS / FOR / BECAUSE / MEASURED BY / SUCCESS / KILL / COST /
RISKIEST) and commit it. Moving a threshold after seeing results is a change of standard and must
be called out as one.

---

### Exercise A — Split eviction damage into information loss and renormalisation

**Goal.** Reproduce §3.2's decomposition, then break it on a real model. This is register open
question 1, and it is the cheapest experiment in the entire memory track.

**Hardware.** CPU, numpy only. **Runtime:** ~4 minutes for part 1 at three seeds; ~10 minutes for
part 2 if you already have a small model locally.

**Part 1 — the synthetic decomposition (must reproduce the table in §3.2).**

```python
"""Eviction error = dropped contribution + a scalar gain on the survivors."""
import numpy as np

T, D, TRIALS = 4096, 128, 512          # cached tokens, value dim, draws per seed

def split(keep_frac: float, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    k = max(1, int(round(keep_frac * T)))
    ev = np.zeros(TRIALS); mf = np.zeros(TRIALS); mk = np.zeros(TRIALS)
    for t in range(TRIALS):
        rank = rng.permutation(T) + 1
        s = alpha * np.log(T / rank) + rng.normal(0.0, 0.5, T)   # Zipf-ish scores
        V = rng.normal(0.0, 1.0, (T, D))                          # value vectors
        e = np.exp(s - s.max()); Zf = e.sum()
        out_full = (e / Zf) @ V
        keep = np.argpartition(-s, k - 1)[:k]                     # oracle top-k
        Zk = e[keep].sum()
        out_evict   = (e[keep] / Zk) @ V[keep]                    # what a cache returns
        out_massfix = (e[keep] / Zf) @ V[keep]                    # denominator preserved
        n = np.linalg.norm(out_full) + 1e-12
        ev[t] = np.linalg.norm(out_evict   - out_full) / n
        mf[t] = np.linalg.norm(out_massfix - out_full) / n
        mk[t] = Zk / Zf
    return mk.mean(), ev.mean(), mf.mean()

print(f"{'keep':>6} {'mass':>8} {'A':>7} {'err_evict':>10} {'err_massfix':>12} {'ratio':>7}")
for kf in [0.50, 0.25, 0.10, 0.05, 0.02, 0.01]:
    rows = np.array([split(kf, 1.0, s) for s in (0, 1, 2)])
    mk, ev, mf = rows.mean(axis=0)
    print(f"{kf:>6.2f} {mk:>8.4f} {1/mk:>7.3f} {ev:>10.4f} {mf:>12.4f} {mf/ev:>7.3f}")
```

**Deliverable — three numbers and one check.**
1. The `ratio` column. **Prediction: 0.10–0.16 across the whole budget range.** If you get ~1.0,
   your `out_massfix` is dividing by `Zk` somewhere.
2. **The arithmetic check that proves the decomposition rather than the code.** At `keep=0.10`,
   verify that `err_evict ≈ A − 1` to two significant figures (predicted 0.326, measured 0.329
   here). If that identity fails, the two error terms are not decomposing the way §3.2 claims and
   the algebra is wrong, not the code.
3. Sweep `alpha ∈ {0.5, 1.0, 2.0, 3.0}` at `keep=0.10` and report retained mass. **Prediction: at
   `alpha=2.0` the top 10% of tokens hold >99% of the mass and eviction error falls below 0.01.**
   Write one line in your notebook entry naming what real-world quantity `alpha` stands for.

**Part 2 — the falsification, on a real model.** Take any small causal LM you can run on CPU (GPT-2
124M is enough; `output_attentions=True` gives you the weights). For one layer, one head, one
query position over a 1,024-token context: extract the real pre-softmax scores and the real value
vectors, and rerun the three-way comparison with them in place of the synthetic draws.

**Prediction:** the ratio rises — real value vectors are correlated, so the dropped-contribution
term stops cancelling. **Falsification and why it matters either way:** if the ratio stays near
0.11 on real attention, the renormalisation term dominates in practice and a one-float correction
recovers most of the loss, which would be a genuinely important negative result about thirty
published selectors. If the ratio goes to 0.8, information loss dominates and better selectors
*are* the right lever — also worth knowing, and it closes register open question 1 in the other
direction. **An exercise that cannot fail teaches nothing; this one can go either way and both
outcomes change the plan.**

**GPU note, and it is the module's own hazard.** Do not be tempted to run the oracle at long
context with default SDPA on gfx1151. §4.3: the score matrix is retained at 147.2 bytes/T² by
default `[M]`, and a single-layer 8-head oracle at batch 1 reaches the 31 GiB per-tensor hazard
around T ≈ 30,000 `[A]`. The failure is a silent hang at 0% CPU, not an exception.

---

### Exercise B — Measure the needle's salience rank, and locate the eval's blind spot

**Goal.** Turn §2.5 from an argument into a number. Produce the rank `R` of a needle token under
two scoring functions, and read off the budget at which any top-k policy would evict it.

**Hardware.** CPU. **Runtime:** ~15 minutes including a ~500 MB model download the first time.
Part 1 needs no download at all.

**Part 1 — the detection-threshold curve (no model, no download, 2 minutes).**

Plot `ρ_detect = R / T` against `R` for `T = 32,768`, and mark the deployed budget band (10–50%).
Then answer, in one line: *how salient may a target be — expressed as a rank threshold — before a
benchmark run at a 10% budget loses all power against it?* Remember rank 1 is most salient, so the
answer is the rank the target must be **worse** than. (Check: 3,277; anything ranked 1…3,277 is
retained and therefore undetectable.) Mark on the plot where a top-10 needle, a top-100 needle, and
a rank-8,000 buried instruction fall. **Deliverable: one plot and one integer.**

**Part 2 — the measurement (needs a small model).**

Build a mini-haystack at the model's context limit (1,024 for GPT-2): filler text, one needle
sentence inserted at depth `d ∈ {10%, 30%, 50%, 70%, 90%}`, and a question appended at the end.
Run one forward pass with `output_attentions=True`. For each layer and head compute two scores per
context token:

- **question-aware** (SnapKV-like): mean attention received from the last 16 query positions;
- **accumulated mass** (H2O-like): total attention received from all query positions.

Then report, for each scoring function, the **percentile rank of the needle's tokens** among all
context tokens, and — as the control — the percentile rank of a randomly chosen filler token and
of an *instruction* sentence placed at 5% depth ("Answer in exactly three words.").

**Deliverable — three percentiles and a verdict.**
1. Needle percentile under the question-aware scorer. **Prediction: top decile (rank percentile
   ≥ 90) in a majority of heads.**
2. Needle percentile under accumulated mass. **Prediction: above median, but weaker and noisier
   than (1).**
3. Instruction percentile. **Prediction: materially lower than the needle's under the
   question-aware scorer, because the question does not mention it.**

**Falsification, and it is a real one.** If the needle's rank is near the median under both
scorers at this scale, the adverse-selection argument does not hold at 124M and §2.5 needs a
correction appended to this module naming the scale at which it fails. Do not quietly drop it —
write it up. Note honestly that GPT-2 at 1,024 tokens is a weak stand-in for a 32k-context
instruct model, and that a negative result here is evidence about small models, not about the
argument in general. **Say which of those you measured.**

---

### Exercise C — Three definitions of "the victim", one trace

**Goal.** Show that the eviction-rate metric is not comparable across systems, by implementing
§5.2's three victim rules over one synthetic request trace and measuring **coverage** — the number
of *unique* tokens retained — rather than the eviction rate.

**Hardware.** CPU, pure Python. **Runtime:** ~30 minutes to write, seconds to run.

Build a trace of 200 requests over a shared prefix tree: 5 system prompts of 256 tokens each,
each with 40 conversations that extend it by 64–512 tokens, arriving in a shuffled order with a
fixed seed. Fix the cache capacity to 40% of the total distinct tokens. Then implement three
reclaim rules:

1. **vLLM-like** — LRU over refcount-zero blocks, freed blocks stay matchable until reallocation
   (`block_pool.py:679`, `:719`).
2. **SGLang-like** — only *leaves* are evictable; a node with a live child is never a candidate
   (`radix_cache.py:565`).
3. **Mooncake-like** — lease of `L` steps, renewed on access; only expired objects are candidates,
   ranked by lease deadline (`master_service.cpp:6382`).

**Deliverable — a 3 × 2 table and one sentence.** For each rule report (a) **coverage**: unique
tokens resident at the end of the trace, as a fraction of distinct tokens seen; (b) **shared-prefix
survival**: what fraction of the five system prompts is still fully resident.

**Prediction.** The leaves-only rule keeps the most shared prefix (parents are structurally
protected) and the LRU rule keeps the least, while their *eviction rates* — bytes reclaimed per
step — are within a few percent of one another. **The point being made is that eviction rate is
the metric the systems report and coverage is the metric that predicts damage `[C]` (2606.29563).**

**Falsification.** If coverage is within 5% across all three rules, the "three incompatible
victims" break in §5.2 is a distinction without a difference *at this trace shape*, and you should
say so and characterise which trace shape would separate them (hint: vary the fan-out per system
prompt). **A null result here is a real finding about when the systems' differences matter.**

---

## 7. Self-check

Answers at the end of the file. Do not scroll.

1. Your eviction policy scores 99% on needle-in-a-haystack at a 10% KV budget across 4k–128k
   context. Your colleague proposes shipping it. State precisely what the benchmark result
   licenses you to claim, and give the arithmetic that bounds it.

2. You evict the four attention-sink tokens at the start of the context. They hold 30% of the
   attention mass and their value vectors contribute almost nothing to the output. Decompose the
   resulting error using §3.2. Which term dominates, what is the amplification factor, and why can
   you not simply apply the obvious fix at serving time?

3. Three serving systems run the same workload and all report a 30% eviction rate. Which one is
   most likely still holding an intact shared system prompt, and why is its behaviour not a bug?

4. A client sends the exact same 1,024-token prompt twice to vLLM with prefix caching on and block
   size 16. How many tokens are recomputed on the second request, and what are the **two
   independent** reasons it is not zero?

5. You are designing a constant-state layer and have a budget of 65,536 floats for the recurrent
   state. Option 1 is `d_k=256, d_v=256`. Option 2 is `d_k=512, d_v=128`. Same bytes. Which gives
   better multi-query associative recall, by roughly how much, and what is the one-line argument?

6. An agent in production gets steadily worse over six weeks. No code changed, no model changed,
   error rate is flat, latency is flat, and the eval suite is green. Name three register entries
   consistent with this and give the single cheapest measurement that separates them.

---

## 8. What is still unsolved here

### 8.1 The register's own scorecard

Nine `OPEN`, of which two are `OPEN/CONTESTED`; one `PARTIALLY SOLVED`; one `UNPROVEN DEMAND`.
Nothing here is a solved problem with an engineering workaround you have not heard of. That is
unusual and worth sitting with — you are not late to this field.

### 8.2 The honest weak spots inside the register

- **`memory-induced-sycophancy` is below the evidence bar and is in anyway.** Two primary
  measurement papers `[C]` (2606.10949, Jun 2026; 2607.01071, Jul 2026) plus three corroborating
  sources, against a 5-primary-source standard. The register keeps it because the effect size is
  large (up to 25× higher sycophancy with memory on than in-context baselines) and the mechanism
  is specific and testable. **Treat the 25× as one lab's measurement until independently
  replicated.** The register says this itself; do not let it harden into a fact by repetition,
  which is exactly how the failure modes in §2.1 propagate.
- **`per-session-kv-lifetime` has zero primary sources.** It is a hole in a taxonomy's grid `[C]`
  (2607.02574). Promote it the moment someone publishes a trace; do not build against it.
- **One unverified claim is flagged and should stay flagged.** The register reports, and could not
  confirm in a primary source, that five of six defence classes fail against delayed-trigger
  poisoning attacks and that only tool-layer memory restriction holds structurally; likewise the
  OWASP ASI06 "Memory and Context Poisoning" item is cited from secondary sources only. Both are
  hypotheses worth testing, not findings.

### 8.3 The deepest unsolved thing: there is no durability contract

Everything you know about deletion assumes one: a write commits, a tombstone propagates, an audit
log proves it. LLM memory has **no durability contract at all**, and the 2026 serving survey names
durability contracts for persisted KV as one of its seven missing measurements `[C]` (2607.02574).
Parameter-memory backflow means parameter-only unlearning cannot close the retrieval-write loop —
retrieval reactivates parametric remnants, or memory artefacts reintroduce content `[C]`
(2602.17692, Feb 2026). Across thirteen system configurations, *where* the deletion decision is
made determines which failure modes are even addressable, with mutation-time placement winning —
a direct contradiction of the common assumption that retrieval-time reranking is the leverage
point `[C]` (2606.15903, Jun 2026).

> **The sharpest break in the whole module.** The asymmetry runs the *opposite* way from storage.
> Evicting a KV block is never data loss, only a recompute, which is why Mooncake can force-evict
> rather than block on writeback — a trade no real storage tier is permitted to make. So the same
> system is *too willing* to lose bytes it should keep and *unable to prove* it lost the bytes it
> must delete. Your entire mental model of durability is inverted on both axes at once.

### 8.4 Contested and left contested

Eviction versus retention: permanent eviction cuts peak capacity but is irreversible;
full-retention sparse loading preserves fidelity and cuts bandwidth but not capacity. RocketKV
argues they are orthogonal and composable; other 2026 work treats eviction as the wrong primitive
entirely and prefers tiered offload plus retrieval `[C]` (2607.02574). Also contested: whether
non-uniform per-layer budget allocation is real at all — PyramidKV degenerates to SnapKV at
aggressive ratios by its own account `[C]` (2406.02069, Jun 2024), and several groups argue the
gain comes from the observation window rather than the allocation rule. And whether sub-4-bit KV
is deployable: 8-bit is production-boring, 4-bit broadly safe, 2-bit a live question whose answer
is task-dependent — perplexity-friendly, reasoning-hostile `[C]` (2606.03458, Jun 2026;
2508.04257, Aug 2025).

### 8.5 Three papers not in the register v1.0.0, folded in here

I found these while writing and they are additions, not corrections. They should be appended to
the register when it is next revised.

1. `[C]` **2605.18053** (May 2026) — *Protection Is (Nearly) All You Need*. Seven policies under one
   harness collapse to F1 ≤ 0.064 without structural protection; 10% boundary reservation recovers
   69–90%; scoring differences are second-order once boundaries are guarded. Belongs in
   `attribution-gap-in-serving-results` as a fifth documented instance, and it is the strongest of
   the five.
2. `[C]` **2605.26667** (May 2026) — *MemFail*. A diagnostic benchmark that decomposes agent memory
   into summarisation / storage / retrieval and builds five adversarial datasets across four tasks
   to attribute a wrong answer to a specific operation. This is the agent-memory analogue of the
   instrument this lab wants to build for KV, and it is prior art worth reading before building.
3. `[C]` **2606.14589** (Jun 2026) — *When Errors Become Narratives*. Eight weeks, 22 incidents, a
   five-class taxonomy of silent failures in a production LLM agent runtime, and the "fail-plausible"
   class. The only genuine production postmortem catalogue in the literature, and the closest thing
   to the artefact a systems engineer would produce.

### 8.6 Where I would amend the register, with my working

**Proposed twelfth entry: `eval-adverse-selection`.**

*Symptom:* memory-policy evaluations report high scores on benchmarks whose targets are the exact
spans the policy is designed to retain, so the score is uninformative about the policy.

*Mechanism:* §2.5 and §3.5 — detection requires target rank `R > ρ·T`, and the eval author sets
`R` near 1.

*Evidence, assessed honestly against the register's own bar.* The **symptom** — that aggregate
long-context evaluations under-report compression damage — clears the bar comfortably: RULER `[C]`
(2404.06654), SCBench `[C]` (2412.10319), worst-case aggregation `[C]` (2510.13334), the pitfalls
paper `[C]` (2510.00231), ATLAS `[C]` (2605.28079), LongBench Pro `[C]` (2601.02872), and
protection-dominates `[C]` (2605.18053) — seven, from independent groups. The **salience-rank
mechanism** is *ours* and I have found no primary source that states it. It is therefore `[A]`,
medium confidence, and the cheapest test that would move it is Exercise B — one forward pass per
prompt.

*Proposed status:* `OPEN` on the symptom, `[A]` on the mechanism, with the mechanism promoted or
dropped on Exercise B's result. **I am not claiming the register is wrong.** The register
acknowledges NIAH's weakness in two places (`effective-context-collapse` notes models with
near-perfect NIAH collapsing on RULER; the CoT entry uses NIAH-S2 as a *symptom* measure). What is
missing is the mechanism and the design rule that follows from it, and a mechanism is exactly what
the register's own format demands per row.

### 8.7 The frontier, in one list, all testable here

1. **Is eviction damage the information loss or the renormalisation?** Exercise A, part 2.
2. **Does the randomised-design error certificate hold at 100M?** `[C]` (2607.21475) claims 0.97
   coverage. Inference-only. The cheapest way to acquire an attribution *instrument* rather than
   another policy.
3. **Does structural protection dominate scoring at our scale?** `[C]` (2605.18053) at 7B; nobody
   has run it at 300M, and the small-scale answer bears on whether our ablations transfer.
4. **What is the needle's salience rank, and does a rank-stratified eval separate policies that
   NIAH cannot?** Exercise B, then the eval design.
5. **Is the 3:1 hybrid cliff a ceiling or a token-budget artefact?** `[C]` 2507.06457 vs 2606.15378
   — the one place the field's disagreement turns on an axis a small rig can sweep.
6. **Does the U-shaped position bias exist at 20M–300M, and is it occupancy-dependent?** If the U
   does not appear at this scale, a large class of our eviction experiments is unsound and we need
   to know before running them, not after `[C]` (2508.07479).
7. **At what `d_k` does MQAR break for a Gated DeltaNet layer at 100M, and does §3.3's `N ≈ d_k`
   prediction hold?** A quantitative prediction that either holds or does not `[C]` (2607.17419).
8. **How much of a KV-compression speedup is really a batching effect?** Fixed batch size vs.
   batch-size-free at the same compression ratio. Directly attacks the attribution gap with no
   training at all.
9. **Does the prefix-cache block-alignment floor cost what §3.4's arithmetic implies?** Pure
   instrumentation; a same-day result.
10. **Does CoT-style fine-tuning degrade long-range recall at 300M the way it does at 9B, and does
    restoring `W_Q, W_K` from the pretrained checkpoint recover it?** `[C]` (2606.11052, Jun 2026)
    reports HypeNet-9B falling from 67.2% to 9.4% on NIAH-S2@256K after CoT-SFT, with a
    training-free fix. If the mechanism reproduces at our scale we have a cheap testbed for a
    frontier-scale failure.

**And the one that is upstream of all of them** — this lab's riskiest assumption, from
`research/synthesis.md`: *that distributional divergence from a full-cache oracle measures anything
decision-relevant.* Divergence and task accuracy can dissociate in both directions: a policy can
shift the output distribution without flipping any argmax, or flip one critical token at negligible
average KL. Exercise A's result sharpens this rather than resolving it — a common-mode gain is the
*most benign* kind of divergence for an argmax, so a large error norm from renormalisation might
matter far less than a small error norm from a dropped instruction. **Norm is not the metric;
decision flips are.** Test that before building the harness, not after.

---

## Answers to the self-check

**1.** It licenses almost nothing about eviction safety. The arithmetic: a top-k policy evicts a
target of rank `R` only when the budget `ρ < R/T`. Under a question-aware scorer the needle's rank
is near 1 by the eval's construction — it is the only span related to the question — so at
`T = 32,768` you would need `ρ < ~0.03%` for the eval to have any chance of failing. You ran it at
10%, three orders of magnitude above the detection threshold. The correct claim is: *"the policy
retains high-salience question-relevant spans," which we already knew because that is what the
policy is.* What it does not test is whether a low-salience span that the answer depends on — a
buried instruction at rank ~8,000, detectable only below `ρ = 24%` — survives. That is the failure
the literature actually documents `[C]` (2510.00231).

**2.** Amplification `A = 1/m = 1/0.70 = 1.43`. Using §3.2: the information-loss term
`out_full − out_massfix = Σ_{j∈E}(e_j/Z_full)v_j` is small because the sinks' value vectors are
near-zero — you lose 30% of the *mass* but almost none of the *content*. The renormalisation term
scales everything that survives by 1.43, a 43% common-mode gain on the head's entire output. The
second term dominates overwhelmingly; measured on the synthetic caricature (`[M]` §3.2, one seed),
evicting 4 zero-value sinks gives error 0.975 against 0.096 for the denominator-preserved control.
You cannot apply the fix at serving time because `out_massfix` needs `Z_full` for *this* query,
and the discarded scores are query-dependent — you knew the discarded mass for the query you had
at eviction time, not for the one you have now. It must be *estimated*, which is exactly what the
Poisson-sampled tail plus Hájek logit offset does `[C]` (2607.21475). Corollary worth stating:
StreamingLLM's prefix pin is, mechanically, a denominator fix.

**3.** The SGLang-style radix cache (`radix_cache.py:565`). Its `evict` only ever considers *leaves*
of the tree, so a system prompt sitting near the root with any live descendant is structurally
un-evictable — parents are protected by topology, not by policy. That is not a bug: it is a
different and arguably better answer to "what is the victim," and it makes shared prefixes durable
for free. The deeper point is that all three systems can report the same eviction rate while
retaining completely different *sets*, which is why coverage — unique tokens retained — is the
metric that predicts damage `[C]` (2606.29563) and eviction rate is the one everyone reports.

**4.** Sixteen tokens — one full block. Reason one: `get_computed_blocks` caps the hit at
`num_tokens − 1 = 1,023` because you need a forward pass to produce logits for the next token
(`kv_cache_manager.py:225`). Reason two: the hit is then floored to block alignment,
`⌊1023/16⌋ × 16 = 1,008`, so the entire trailing block is recomputed
(`single_type_kv_cache_manager.py:658`). The two reasons are independent — the first costs one
token, the second rounds that up to a block — and the worst case occurs exactly when the prompt
length is a multiple of the block size, which is a fact your benchmark's prompt-length choice will
silently determine.

**5.** Option 2, and by roughly `√2` in signal-to-interference ratio. §3.3: capacity in retrievable
associations is set by `d_k` alone, because interference comes from non-orthogonality of *keys* in
`d_k` dimensions, with `SIR ≈ √(d_k/(N−1))` and the capacity point at `N ≈ d_k`. Option 1 gives
`N ≈ 256`; option 2 gives `N ≈ 512`. `d_v` only widens each stored value; it buys zero additional
associations while consuming exactly as many bytes. One-line argument: **you cannot pack more than
`d_k` mutually orthogonal directions into `d_k` dimensions, and `d_v` is not a dimension you are
packing into.** Note the implication for the literature: "state size in KB" is the wrong axis, and
it is the axis the hybrid papers report.

**6.** Three candidates. (a) `memory-poisoning-cross-session-contamination` — violation rates rise
*monotonically with exposure length*, so slow accumulation with no discrete event is its signature
`[C]` (2605.17830). (b) `memory-induced-sycophancy` — the store has accumulated snippets encoding
the user's beliefs while discarding the corrective context that surrounded them, so the agent
agrees more over time `[C]` (2606.10949). (c) `forgetting-and-rollback-unsolved` — a bad entry was
"deleted" and keeps returning through a surviving copy or regeneration `[C]` (2602.17692). The
cheapest measurement that separates them: **run the same fixed probe set with the memory store
disabled and with it enabled, on the same model, same day.** If accuracy recovers with memory off,
it is (b) or (c) and not model drift; then diff the retrieved snippets against their source
documents — poisoning shows content with no legitimate provenance, sycophancy shows faithful
snippets missing their corrective context, and failed deletion shows content you already removed.
The reason this is cheap and nobody runs it is the module's whole theme: the metric that would
catch it costs the resource the feature was introduced to save.

---

## Sources

**Our own measurements and artifacts (`[M]`)**

- §3.2 decomposition — computed 2026-07-26, CPU numpy under `torch`-free Python in the lab venv,
  `T=4096`, `D=128`, 512 draws per seed, seeds 0/1/2, Zipf exponent `α=1.0` plus N(0,0.5) jitter,
  oracle top-k retention. Table reports mean ± half-range across seeds. The `α` sweep and the sink
  variant are single-seed (seed 0, 256 draws) and are labelled anecdotes in the text.
- `ASSUMPTIONS.md` rows: `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per arm),
  `large-tensor-fault-32gib` (≥32 GiB single tensors hang silently at 0% CPU),
  `sdpa-is-memory-efficient` (refuted by default; 147.2 vs 6.6 bytes/T²),
  `bf16-numerics-unproven`, `reference-model`, `kv-per-token-laguna`, `single-device-only`,
  `torch-build`.
- `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier measurement, 2026-07-26.
- `research/memory/memory-failure-register.md` v1.0.0 — the note this module teaches.
- `research/synthesis.md` — the decision this module supports.
- Code pointers: every `file:line` in §5 was opened and the named symbol confirmed on the named
  line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.

**arXiv (`[C]`)** — every id below appears in `research/memory/citation-verification.json`
(resolved against the live arXiv API on 2026-07-26) except the three in §8.5, which were verified
by fetching their arXiv abstract pages this session.

- `2104.09864` — RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
- `2306.14048` — H2O: Heavy-Hitter Oracle for Efficient Generative Inference (2023)
- `2307.03172` — Lost in the Middle: How Language Models Use Long Contexts (2023)
- `2309.17453` — Efficient Streaming Language Models with Attention Sinks (2023)
- `2312.04927` — Zoology: Measuring and Improving Recall in Efficient Language Models (2023)
- `2402.18668` — Simple linear attention language models balance the recall-throughput tradeoff (2024)
- `2404.06654` — RULER: What's the Real Context Size of Your Long-Context Language Models? (2024)
- `2404.14469` — SnapKV: LLM Knows What You are Looking for Before Generation (2024)
- `2405.16444` — CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion (2024)
- `2406.02069` — PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling (2024)
- `2406.16008` — Found in the Middle: Calibrating Positional Attention Bias (2024)
- `2407.12784` — AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases (2024)
- `2409.20002` — The Early Bird Catches the Leak: Timing Side Channels in LLM Serving (2024)
- `2412.10319` — SCBench: A KV Cache-Centric Analysis of Long-Context Methods (2024)
- `2507.06457` — A Systematic Analysis of Hybrid Linear Attention (2025, rev. 2026)
- `2508.04257` — KVSink: Preservation of Attention Sinks in KV Cache Quantization (2025)
- `2508.07479` — Positional Biases Shift as Inputs Approach Context Window Limits (2025)
- `2510.00231` — The Pitfalls of KV Cache Compression (2025, ACL 2026)
- `2510.13334` — Taming the Fragility of KV Cache Eviction in LLM Inference (2025)
- `2510.26912` — Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall (2025)
- `2601.02872` — LongBench Pro (2026)
- `2602.16837` — A Structural Theory of Position Bias in Transformers (2026)
- `2602.17692` — Agentic Unlearning: When LLM Agent Meets Machine Unlearning (2026)
- `2603.10726` — PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems (2026)
- `2605.15338` — Hidden in Memory: Sleeper Memory Poisoning in LLM Agents (2026)
- `2605.17830` — Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents (2026)
- `2605.18053` — Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction (2026-05-18)
- `2605.26667` — MemFail: Stress-Testing Failure Modes of LLM Memory Systems (2026-05-26)
- `2605.28079` — ATLAS: All-round Testing of Long-context Abilities across Scales (2026)
- `2606.03458` — KVarN: Variance-Normalized KV-Cache Quantization (2026)
- `2606.04329` — From Untrusted Input to Trusted Memory: Memory Poisoning Attacks in LLM Agents (2026)
- `2606.09864` — Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation (2026)
- `2606.10949` — Recalling Too Well: Sycophancy Evaluation and Mitigation in Memory-Augmented Models (2026)
- `2606.11052` — Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall (2026)
- `2606.14589` — When Errors Become Narratives: A Longitudinal Taxonomy of Silent Failures in a Production LLM Agent Runtime (2026-06-12)
- `2606.15378` — Rethinking the Role of Efficient Attention in Hybrid Architectures (2026)
- `2606.15903` — Control-Plane Placement Shapes Forgetting (2026)
- `2606.21842` — Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG (2026)
- `2606.29563` — Coverage-Driven KV Cache Eviction (2026)
- `2607.01071` — MemSyco-Bench: Benchmarking Sycophancy in Agent Memory (2026)
- `2607.02574` — From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving (2026)
- `2607.08032` — What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction (2026)
- `2607.08057` — Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization (2026)
- `2607.17419` — Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones (2026)
- `2607.21475` — Error Certificates for KV-Cache Eviction via Randomized Design (2026)

**Non-arXiv, cited as framing or engineering artifacts, not as evidence**

- Belady, *A study of replacement algorithms for a virtual-storage computer*, IBM Systems Journal,
  1966 — MIN and the future-knowledge requirement.
- Huang et al., *Gray Failure: The Achilles' Heel of Cloud-Scale Systems*, HotOS 2017 —
  differential observability.
- vLLM RFC #16016 — Cache Salting for Secure and Flexible Prefix Caching (GitHub issue).
- OWASP Agentic AI Top 10, ASI06 "Memory and Context Poisoning" — **unverified against OWASP
  directly**; reported in secondary sources only, and repeated here with the register's own flag
  intact.
