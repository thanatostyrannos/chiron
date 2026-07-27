---
title: Measuring memory — attribution, null distributions, and evals that can fail
version: 1.0.0
date: 2026-07-26
track: C — Memory (the deep track)
prereqs: tensors-and-autograd, transformer-forward-pass-by-hand, attention-variants-and-kv-cost, kv-cache-mechanics
difficulty: hard — the math is easy, learning to distrust your own instrument is not
time: 4–5 h reading and working the arithmetic; 3–4 h for the three exercises
mirrors: research/memory/open-problems-ranked.md, research/notes/evaluation-landscape.md
---

# Measuring memory

**Difficulty and time, honestly.** Nothing in section 3 is harder than a logarithm and a
weighted average. What makes this module hard is that it asks you to stop trusting numbers
you produced yourself, and that is slower than learning a formula. Budget 4–5 hours for
sections 1–5 with a pen and a terminal, and 3–4 hours for the exercises. Exercise B is the
one to protect time for: it is a working miniature of the instrument this lab intends to
ship, and it produces a number — the noise floor — that nothing downstream is interpretable
without.

This module mirrors `research/memory/open-problems-ranked.md` §1 and
`research/notes/evaluation-landscape.md`. It teaches what those notes survey. Where I
sharpen or extend them I say so explicitly and show the working; there is one place
(section 3.4) where I think the notes understate a problem, and one (section 4.4) where I
add a hazard they do not mention.

---

## 1. What this module settles

**One:** a memory-policy result is evidence only if it separates *outcome* from
*attribution*, and at our scale the only instrument that does that is differential — a
full-cache oracle diffed against the policy, with a measured null distribution — because an
accuracy metric needs a competent model while a divergence metric needs only a reference
run. **Two:** every metric must first be shown to *fail* under deliberately injected faults,
because an eval you have never seen fire is a decoration, and the six-fault battery is the
calibration procedure that turns a decoration into an instrument. **Three:** instrumentation
here is not free in the way it is free in production — the counterfactual costs a second
full forward pass, the probe stalls the GPU pipeline (`[M]` reading one scalar off the
device costs ~230× reading the same scalar off a host tensor on this machine, section 6),
and the attribution signal you most want forces the attention kernel onto its memory-hungry
path — so the measurement plan is a capacity budget, not an afterthought.

**A finding this module's own exercises produced, folded back in.** `[M]` Exercise B, three
seeds: at the floor case (a randomly-initialised 4-layer model, T = 1024, one cache entry
dropped) the **bf16-versus-fp32 noise floor is ~10× the eviction signal**, giving a
signal-to-noise ratio of 0.1 at every seed. Whether a trained model clears that floor is now
the cheapest decision-changing experiment in the memory track — see §2.7, Exercise B, and
§8 item 3.

---

## 2. Theory in plain language

### 2.1 The one distinction that organises everything

Every measurement answers one of two questions:

- **Outcome:** did the system produce the right answer / the higher throughput?
- **Attribution:** did it do so *by the mechanism you claim*?

For a memory-systems lab only the second is evidence. A model can answer a retrieval
question by recalling a specific cached token, by lexical overlap between query and needle,
by a prior over plausible answers, or by an artifact of how the haystack was built. All four
produce the same score.

The field's own record shows this is not a fastidiousness problem. Four documented cases
where the outcome metric held and the mechanism was broken:

| Outcome that held | Mechanism that broke | Source |
|---|---|---|
| Perplexity within 1.03× | 15.2% of refusals lost, 11 models, 1,894 prompts | `[C]` 2606.09864 (Jun 2026) |
| LongBench aggregate | *specific instructions* dropped entirely under StreamingLLM / SnapKV / TOVA / H2O | `[C]` 2510.00231 (2025) |
| Single-turn policy ranking | ranking did not survive multi-turn cache reuse | `[C]` 2412.10319 (2024) |
| Mean-aggregated ranking | ranking inverted under worst-case aggregation | `[C]` 2510.13334 (2025) |

And the standing suspicion inside the canon: several groups argue most of PyramidKV's
reported gain `[C]` (2406.02069) comes from SnapKV's observation window `[C]` (2404.14469)
rather than from the per-layer budget allocation it claims credit for — it degenerates to
SnapKV at aggressive ratios by the paper's own account.

> **Systems bridge — and this is the bridge you should be able to teach back.**
> This is the SLO-versus-trace distinction. *"p99 improved"* is an outcome.
> *"p99 improved because the read path started hitting the page cache"* is attribution. You
> have thirty years of scar tissue on the first without the second: the deploy changed four
> things, the graph went down and to the right, and two quarters later the real cause
> surfaces as an incident. Your instinct — bisect, hold everything else fixed, tag the span,
> diff against a control — is exactly the right instinct and it transfers.
>
> It breaks in three specific places, and the breaks compound. The rest of section 2 is
> those three breaks.

### 2.2 Break one — there is no request id

A distributed trace works because causality is *discrete and nameable*. A request enters,
touches a bounded set of named components, and each touch is a span with a parent. Spans
are disjoint, they nest, and their durations sum to the total. That structure is what lets
you say "62% of p99 is in the auth call."

Inside a transformer, causality runs through a **continuous attention distribution**. There
is no call graph. The closest thing to a span is an attention weight `a_j` — the softmax
share that query position `t` places on cached key `j` — and it is tempting to read that as
"token `j` was responsible for `a_j` of this answer." **Do not.** Three reasons, in
increasing order of how much they cost you:

1. **Weights are not effects.** The attention *output* is `Σ_j a_j v_j`. If every value
   vector `v_j` in the context were identical, every weight distribution would give the same
   output and no token would be causally responsible for anything. Section 3.2 makes this
   exact: the error from evicting a set is attention mass **times** a value-difference term,
   and mass alone tells you nothing.
2. **Weights do not compose across layers.** A trace's spans are additive because time is
   additive. Layer `l`'s attention output goes through an MLP, a nonlinearity, and a residual
   add before layer `l+1` attends over it. Multiplying weights down the stack to get a "total
   attribution" is a linearity assumption you have not earned. This is precisely why
   attention-rollout methods are contested rather than standard.
3. **The information has already been copied.** By the time you are at layer 12, the needle's
   content is not only in the needle position's cache entry — earlier layers copied it into
   the residual stream of *later* positions, whose own K and V at layer 12 therefore carry
   it. This is the documented behaviour of induction and retrieval heads `[C]` (2209.11895,
   2404.15574). Section 4.4 turns this into a concrete warning about the fault battery.

What *does* exist is coarse, head-level localisation: retrieval heads are identifiable and
maskable `[C]` (2404.15574), which gives you something like a per-subsystem tag rather than
a per-request id. Whether that signature is present at 300M is an open question, listed in
section 8.

### 2.3 Break two — the counterfactual requires running the expensive thing

In production the control arm is free: it is the other half of the fleet, serving real
traffic, right now. You get the counterfactual for the price of a routing rule.

Here the control arm is **the same model, on the same prompt, with the full cache** — which
is exactly the configuration your policy exists to avoid. You cannot get it from traffic.
You must run it, and you must run it *on every probe*, because the counterfactual is
per-token, not per-deploy.

Two consequences that people skip:

**The oracle harness saves no memory at all.** To be the oracle you must retain the full
cache. So a run that measures a policy's *quality cost* cannot simultaneously measure its
*capacity benefit* — the capacity benefit is definitionally absent from that run. Two
configurations, two runs, and the throughput number from the oracle run is not the policy's
throughput. `research/notes/evaluation-landscape.md` §6.3 makes the affordability argument
correctly (at 300M you can afford the oracle; at 70B you cannot) but does not say this out
loud, so say it out loud in every write-up: **quality and cost come from different runs and
must never be reported from the same one.**

**Small scale is the enabling condition, not a compromise.** `[M]` A 300M model is ~600 MB
of bf16 weights against our measured ≥62 GiB fast tier
(`ASSUMPTIONS.md → gpu-fast-tier-size`, `notebook/uma-carveout-controls-fast-tier.md`,
single run per arm). Two caches plus two models is trivially resident. This is the same
inverted ratio the rest of Track C keeps hitting — the KV cache can be ~100× the weights at
our scale — and here it buys us the one thing frontier labs cannot afford: the full
counterfactual, every time.

### 2.4 Break three — the probe stalls the pipeline

This is the break that will surprise you most, because no logging system you have operated
has this property.

GPU execution is asynchronous. Your Python process enqueues kernels and runs ahead; the
device drains the queue behind it. That gap is what keeps the GPU fed. **Reading a value
off a device tensor collapses the gap** — `.item()`, `float(t)`, `t.cpu()`, `if loss >
threshold:`, `print(t)`, and a dozen other innocuous constructs all force the host to wait
for every previously enqueued kernel to retire, then copy four bytes back.

`[M]` On this machine — `torch 2.12.0a0+rocm7.13.0a20260313`, gfx1151, native Windows,
fp32, three independent runs on 2026-07-26, full config in section 6 — with an *empty*
queue, nothing to wait for:

| Operation | Cost |
|---|---|
| `.item()` on a **host** fp32 scalar tensor | **0.25 µs** (tight: 0.18–0.34) |
| bare `torch.cuda.synchronize()` | **3–7 µs** across three runs |
| `.item()` on a one-element **device** tensor | **52–58 µs** in two clean runs; **136 µs** in one contended run |
| `t.cpu()` on the same tensor | 63 µs |
| `float(t)` on the same tensor | 59 µs |

**The robust number is the ratio, not the level.** Measured within the same interleaved
block — so machine-level drift cancels — reading a scalar off the device costs a median
**236×** (range 169–335) what reading the same scalar off a host tensor costs. The absolute
microsecond figure moved by 2.6× between runs when the machine was busy, which is itself the
lesson of section 6 and the reason it is reported as a range.

`.item()`, `.cpu()` and `float()` all land within ~10% of each other in every run, which
says the cost is the **device-to-host round trip**, not anything special about `.item()`.
That comparison is within-run and therefore drift-invariant; treat it as established.
Treat the absolute level as ~55 µs on a quiet machine and materially worse on a busy one.

Now price a plausible instrument. A per-layer, per-token telemetry read on a 24-layer model
during decode, drained eagerly, at 55 µs:

```
24 layers × 55 µs = 1.3 ms of probe cost per decoded token
```

Against a decode step of ~3 ms (a generous figure for a 300M model at this bandwidth), that
is **~44% of throughput spent on observability** — and ~110% if the machine is in the state
run 2 caught it in. Derived from `[M]` inputs, not measured end to end.

The fix is the one OLMo-core ships, and reading it is the point of section 5:
metrics are recorded as **unevaluated device tensors** (`trainer.py:1037`), accumulated in a
plain dict, and drained in a single host-device sync every `metrics_collect_interval` steps
(`trainer.py:200`, default **5**; the modulo gate is `trainer.py:1514`; the drain is
`trainer.py:1394`). Same 24 counters, drained once per 5 steps:

```
55 µs / (5 × 3 ms) = 0.37% overhead
```

**A 120× reduction in observability cost from one design decision** — keep the counter on
the device and batch the read. Note that the 120× is `24 syncs/step ÷ 0.2 syncs/step` and is
therefore independent of the per-sync cost, so it survives the instability in that figure. That is write-combining, and you already know it as deferring
a cache-line writeback until eviction. What is new is the direction of the cost: the
expensive part is not the I/O, it is the *synchronisation*, and the batching exists to avoid
a stall rather than to amortise a write.

There is a second-order version that is worse. Instrumentation that changes memory footprint
changes achievable batch size, which changes arithmetic intensity, which changes throughput.
So a "compression speedup" measured with attribution logging enabled is partly a batching
result, and separating them requires holding batch size fixed — which nobody does, because
it looks like leaving performance on the table `research/memory/memory-failure-register.md`
(`attribution-gap-in-serving-results`). Your Heisenberg problem in production was tolerable
because a StatsD counter does not change the shape of the workload. Here it does.

### 2.5 A miss is not a latency event

Restating the break that governs everything else in Track C, because it is what makes
measurement hard rather than merely expensive: **a KV cache miss is a correctness event, not
a latency event.** There is no fault handler, no retry, no error, and no counter that
increments. FlashInfer's page table cannot even *represent* a miss — a page is in the CSR
index or it does not exist (`memory/flashinfer/flashinfer/decode.py:1239`). The model does
not stall. It emits a fluent wrong answer at full speed.

Your entire observability reflex — instrument the slow path, alert on the error rate — finds
exactly nothing here. Hit rate is an accuracy metric wearing a performance metric's clothes.

### 2.6 Fault injection as calibration — an eval you have never seen fail is a decoration

You do not trust an alert you have never seen fire. You inject the fault and watch the pager.
Do the same to every eval before it is allowed to certify anything.

The battery, from `research/notes/evaluation-landscape.md` §6.2:

| Injected fault | Implementation | Expected response |
|---|---|---|
| **Needle absent** | delete the injected span | score → chance (proves the eval reads the needle, not the prior) |
| **Needle unreachable** | drop exactly the KV entries spanning the needle | large drop (proves the score depends on *those* entries) |
| **Random capacity loss** | evict `p%` of entries uniformly | monotone degradation curve; gives a sensitivity slope |
| **Position corruption** | re-pack entries so RoPE phase no longer matches position | large drop |
| **Mechanism ablation** | mask retrieval heads `[C]` (2404.15574) | targeted drop on retrieval tasks only |
| **Distribution shuffle** | shuffle haystack sentence order | *no* drop for a true retrieval task; a drop means you were measuring discourse structure |

An eval that survives all six without moving is measuring something other than memory and
should be retired rather than reported.

> **Systems bridge, and the break is subtle.** This is chaos engineering, and the discipline
> is the one you already have. **Where it breaks:** in production, you inject a fault and the
> *system* is under test — you learn whether the system survives. Here the *metric* is under
> test, and the system is broken only so you can find out whether the metric notices. You are
> calibrating the instrument, not the plant. That inversion changes what a "pass" means: a
> chaos experiment passes when nothing breaks; a calibration experiment passes when the
> number **moves**, and a flat line is the failure.

The sharpest instance of a decoration is needle-in-a-haystack. A needle is a low-frequency,
semantically anomalous, high-salience span. It **attracts attention mass**. Heavy-hitter
eviction `[C]` (2306.14048) and observation-window selection `[C]` (2404.14469) are exactly
the policies that retain high-attention tokens. So a policy can shed 90% of the cache,
destroy ordinary long-range dependence, and still pass NIAH. `research/notes/evaluation-landscape.md`
§2(e) states this as `[A]` high confidence and names the cheapest test; Exercise C makes the
selection mechanism quantitative at toy scale so you know what magnitude to expect before you
spend a day on the real thing.

Independent corroboration that the defect is real and not stylised: NoLiMa rebuilds NIAH so
the needle shares minimal literal overlap with the question, and 11 of 13 models claiming
≥128K drop below 50% of their own short-context baseline at 32K, with GPT-4o falling
99.3% → 69.7% `[C]` (2502.05167). Same task, same lengths; delete the string match and
two-thirds of the capability evaporates.

### 2.7 The null distribution — the number that must exist before any other number

In production you do not report a 3% latency improvement without knowing the run-to-run
variance. Same rule. But here the noise floor is generated by something unfamiliar, and
getting it wrong is the most likely way this lab produces a confident wrong result.

**Re-running the same config is usually not a null.** With greedy inference, fixed seed, and
deterministic kernels, two runs of the same configuration produce bit-identical logits and a
KL of exactly zero. A degenerate null makes every difference look infinitely significant.

So the null must come from a **nuisance axis** — something you do not care about that
nevertheless perturbs the output:

| Nuisance axis | Why it perturbs | Why it is the right null |
|---|---|---|
| Batch composition / padding | changes reduction order inside kernels | you will compare arms at different batch sizes eventually |
| dtype (bf16 vs fp32) | different accumulation precision | `bf16-numerics-unproven` is `untested` on this machine `[C]` |
| Attention backend | different kernel, different math | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` changes which path runs `[M]` |
| Data seed / prompt sample | item-level variance | this is the one the statistics literature means |

The third row is a live hazard for us specifically. `[M]` `ASSUMPTIONS.md →
sdpa-is-memory-efficient`: with the flag off, SDPA retains **147.2 bytes/T²**; with it on,
**6.6** — and `torch.backends.cuda.flash_sdp_enabled()` returns `True` either way, so the
API reports what is *permitted*, not what *ran*. Different kernel, different numerics,
different null. **Every number this instrument produces must carry the attention backend
alongside it, or it is uninterpretable.**

And the question that makes this a lab-specific research item rather than hygiene: **is the
bf16 null larger than the eviction signal?** If it is, no bf16 eviction result from this
machine means anything, and we would rather know that in an afternoon than after a
three-week sweep.

`[M]` **At the smallest interesting scale, it is.** Exercise B, run here at three seeds on a
4-layer / 4-head / d=256 / T=1024 randomly-initialised transformer: re-run and
batch-composition nulls are **exactly zero**, the bf16-versus-fp32 null is **6.6–8.4 × 10⁻⁶
nats**, and the median signal from evicting one cache entry is **0.7–1.0 × 10⁻⁶ nats** — an
**SNR of 0.1 at every seed.** Read the caveat with the number: random weights give
near-uniform attention, so a single-entry drop is close to the smallest signal obtainable,
and a trained model concentrating mass on one retrieval-relevant entry would give a larger
one. What this establishes is the **floor**, and the floor is under water. Full config, the
retest discipline, and the interpretation are in Exercise B and section 6.

---

## 3. The math that actually matters

### 3.1 Symbols

| Symbol | Reads as |
|---|---|
| `p_full(v)` | probability the model assigns to vocabulary token `v` at this position **with the full cache** |
| `p_pol(v)` | the same probability **under policy P** |
| `D_KL(p‖q)` | Kullback–Leibler divergence of `q` from reference `p`, in nats |
| `a_j` | attention weight (softmax share) that the current query places on cached position `j` |
| `v_j` | the value vector cached at position `j` |
| `E` | the set of positions the policy evicted |
| `K` | the retained set, `all \ E` |
| `m_E` | **evicted attention mass**, `Σ_{j∈E} a_j` — a number in [0, 1] |
| `v̄_E`, `v̄_K` | attention-weighted mean value vector over `E` and over `K` |
| `σ` | per-item standard deviation of a metric |
| `σ_d` | standard deviation of the *paired difference* between two arms |
| `d` | the effect size you want to detect |
| `ρ` | correlation between the two arms' per-item scores |
| `c`, `k`, `t` | probe cost, drain interval in steps, step duration |

### 3.2 The exact error of an eviction, in one line

This is the module's centrepiece and it takes four lines to derive. Attention output with the
full cache, for one head at one query position:

```
o_full = Σ_{j ∈ all} a_j v_j          with  Σ_j a_j = 1
```

Evict `E`. The softmax is recomputed over the survivors only, so every retained weight is
inflated by the same factor:

```
â_j = a_j / (1 − m_E)      for j ∈ K
o_pol = Σ_{j ∈ K} â_j v_j = v̄_K
```

Split `o_full` by set, using `v̄_E = (1/m_E) Σ_{j∈E} a_j v_j` and
`v̄_K = (1/(1−m_E)) Σ_{j∈K} a_j v_j`:

```
o_full = m_E · v̄_E + (1 − m_E) · v̄_K
```

Subtract:

```
o_full − o_pol = m_E · v̄_E + (1 − m_E) · v̄_K − v̄_K
               = m_E · (v̄_E − v̄_K)
```

**The attention-output error from evicting a set is exactly the evicted attention mass times
the difference between the mean evicted value and the mean retained value.** Exact, not a
bound, for one head at one query position.

Four things fall out, and they are the practical content of this module:

1. **Attention mass alone is not error.** If the evicted values happen to resemble the
   retained ones (`v̄_E ≈ v̄_K`), evicting high-mass entries is *free*. This is the formal
   reason value-aware eviction is a live question rather than an obvious improvement
   `[C]` (2605.08234, May 2026), and it is why every heavy-hitter policy is optimising one
   factor of a product of two.
2. **Low-mass entries with atypical values can hurt.** The product is symmetric. A policy
   that ranks purely on `a_j` cannot see the second factor at all — a fact the 2026 work that
   scores tokens by *representation change* rather than by attention weight is built on
   `[C]` (2606.26472, Jun 2026, verified against the arXiv listing 2026-07-26).
3. **It gives you the two scalars Mnemosyne should log**, per layer, per head, per step:
   `m_E` and `‖v̄_E − v̄_K‖₂`. Two floats. Their product bounds the per-head attention-output
   error *before* any KL is computed, at a cost of two reductions over tensors you already
   have.
4. **It does not compose.** This is the error in one layer's attention output, pre-projection.
   Subsequent layers are nonlinear, so it does not give you a bound on the final logits. It is
   a diagnostic, not a certificate — which is exactly why getting an actual per-step
   *certificate* required randomisation rather than better bookkeeping `[C]` (2607.21475,
   Jul 2026): deterministic top-k eviction provably cannot estimate its own error, because
   evicted values can be altered so that everything the server retains is unchanged while the
   true error grows arbitrarily.

**Worked renormalisation arithmetic**, because the inflation factor surprises people.
`1/(1 − m_E)`:

| Evicted mass `m_E` | Every survivor's weight is multiplied by |
|---|---|
| 0.01 | 1.010 |
| 0.10 | 1.111 |
| 0.30 | 1.429 |
| 0.50 | 2.000 |

Evicting the attention sinks at the start of the sequence is catastrophic not because those
tokens carry information but because they absorb mass the model has no better use for; remove
them and that mass is *forced onto content tokens* `[C]` (2309.17453). Eviction is a
distribution edit, not a deletion. Every policy since pins a prefix for this reason.

**The control this suggests is cheap and nobody runs it.** Compare true eviction against a
*masked-but-denominator-preserved* control: drop the values but add `m_E` back as a constant
in the softmax denominator, so survivors are not inflated. If the control recovers most of
the loss, the field has been attributing to information loss what is actually a distribution
shift, and the fix is a scalar rather than a better selector. Listed as open question 1 in
`research/memory/memory-failure-register.md`; inference-only; no training.

### 3.3 Per-token KL, and the two-line arithmetic that shows why it can lie

The differential instrument's metric:

```
D_KL(p_full ‖ p_pol) = Σ_v p_full(v) · ln( p_full(v) / p_pol(v) )
```

- `Σ_v` — sum over the whole vocabulary at this position.
- `p_full(v)` — reference probability, from the full-cache run.
- `p_pol(v)` — probability under the policy.
- The result is in **nats**; divide by `ln 2 = 0.6931` for bits.

**Why forward KL, with `p_full` as the reference.** The sum is weighted by `p_full`, so it
punishes the policy for putting *low* probability where the oracle put mass — options the
policy destroyed. Reverse KL (`p_pol` as reference) would punish the policy for inventing
mass the oracle did not have. Both are defensible; they are different questions, and the
choice is a decision to record in the pre-registration card rather than a detail.

**Now the arithmetic that shows the risk.** Two toy distributions over two outcomes.

*Case 1 — big divergence, no behavioural change.* `p_full = (0.9, 0.1)`,
`p_pol = (0.8, 0.2)`:

```
0.9 · ln(0.9/0.8) = 0.9 × 0.117783 = +0.106005
0.1 · ln(0.1/0.2) = 0.1 × (−0.693147) = −0.069315
D_KL = 0.036690 nats  (0.0529 bits)
```

Argmax unchanged. Greedy decoding produces the identical token. Accuracy: untouched.

*Case 2 — tiny divergence, the answer flips.* `p_full = (0.51, 0.49)`,
`p_pol = (0.49, 0.51)`:

```
0.51 · ln(0.51/0.49) = 0.51 × 0.040005 = +0.020403
0.49 · ln(0.49/0.51) = 0.49 × (−0.040005) = −0.019602
D_KL = 0.000801 nats
```

Argmax **flipped**, at a divergence **46× smaller** than the case where nothing happened.

That pair of calculations is the riskiest assumption in the entire lab, made concrete.
`research/synthesis.md` names it without prompting: *"distributional divergence and task
accuracy can dissociate in both directions."* The field has two current answers within three
months of each other and they disagree on method — a fixed-contract diagnostic `[C]`
(2605.08234) versus error certificates via randomised design `[C]` (2607.21475) — and
neither validates against the other. **Do not adopt either wholesale.** Report divergence
*and* downstream synthetic accuracy, and report their rank correlation as a first-class
result. If they do not correlate at our scale, that is the finding.

The mitigation that follows directly from the arithmetic: **KL is the wrong summary near a
decision boundary.** Log the per-token *margin* — `p_full(top1) − p_full(top2)` — alongside
KL, and slice divergence by margin. Case 2 is only dangerous where the margin is small, and
the margin is free to compute.

### 3.4 Statistical power — the arithmetic that kills most small-scale eval plans

Sample size for a two-arm comparison, normal approximation, unpaired:

```
n  =  2 · (z_{α/2} + z_β)² · σ² / d²        per arm
```

- `z_{α/2} = 1.960` at α = 0.05 two-sided; `z_β = 0.8416` at 80% power. Sum = 2.8016,
  squared = **7.849**.
- `σ` — per-item standard deviation. For a binary correct/incorrect item at accuracy ≈ 0.5,
  `σ² = p(1−p) = 0.25`.
- `d` — the difference you want to detect. Take 3 accuracy points, `d = 0.03`.

```
n = 2 × 7.849 × 0.25 / 0.0009 = 4,361 items per arm
```

**Four thousand three hundred items per arm to resolve three points of accuracy.** A
100-item eval suite cannot see a 3-point difference, and reporting one from 100 items is how
a null becomes a win. This arithmetic takes ninety seconds and almost nobody does it
`[C]` (2411.00640 gives the framework; 2406.10229 measures how bad seed variance already is
at small scale).

Now pair the arms — same items, same seeds, both policies — and use the difference:

```
σ_d² = σ_A² + σ_B² − 2ρ·σ_A·σ_B
n    = (z_{α/2} + z_β)² · σ_d² / d²
```

At `ρ = 0.9`, `σ_A = σ_B = 0.5`:

```
σ_d² = 0.25 + 0.25 − 2(0.9)(0.25) = 0.05
n    = 7.849 × 0.05 / 0.0009 = 436 distinct items
```

**Pairing buys 10×** at ρ = 0.9. That is the quantitative case for the paired design, and
what must be held identical for the pairing to be valid is everything except the policy:
same prompts, same sampling seed, same batch composition, same dtype, same attention backend.
Break any of those and ρ collapses toward zero and you are back to 4,361.

**Where I extend the mirror note.** `research/notes/evaluation-landscape.md` §6.3 argues the
divergence instrument's advantage is that *"an accuracy metric requires a competent model; a
divergence metric requires only a reference run."* That is right, and there is a second,
larger advantage it does not quantify: **KL is continuous and per-token, so it is not floored
by Bernoulli variance and `n` is measured in tokens rather than items.** One 1,000-token
prompt yields 1,000 paired observations. The honest caveat is that those observations are
correlated within a prompt, so the effective `n` is smaller than the raw count and the
standard error must be cluster-robust at the prompt level — the question-level clustering
`[C]` 2411.00640 prescribes. State the effective `n`, not the raw one.

**Minimum detectable effect**, which every pre-registration card in this lab should carry
and currently does not:

```
MDE = (z_{α/2} + z_β) · σ_d / √n = 2.8016 · σ_d / √n
```

At `σ_d = 0.01` nats and an effective `n = 1,000`: `MDE ≈ 0.00089` nats. Compare that against
the two-outcome arithmetic in 3.3 — an argmax flip cost 0.0008 nats — and you can see the
instrument is, on paper, just barely able to resolve a single decision flip. That is a
warning, not a reassurance.

### 3.5 The probe budget

Overhead of an eagerly-drained probe, as a fraction of useful work:

```
overhead = c / (k · t)
```

- `c` — cost of one probe (one host-device sync).
- `k` — drain interval in steps.
- `t` — duration of one step.

`[M]` With `c = 55 µs` (quiet machine, section 6) and `t = 1.33 ms` at the benchmark's
shape:

| `k` | overhead |
|---|---|
| 1 | 4.1% |
| 5 | 0.83% |
| 20 | 0.21% |

Solve for the interval that keeps you under 1%: `k ≥ c / (0.01 · t) = 55 / 13.3 = 4.1`, so
**`k ≥ 5`**. OLMo-core's default of 5 (`trainer.py:200`) lands exactly on that boundary,
which is a small satisfying sign the upstream default was chosen for a reason. Run it with
run 2's contended figures (`c = 136 µs`, `t = 2.16 ms`) and you need `k ≥ 7` — so the
default is correct on a quiet machine and marginal on a busy one, which is the honest way to
read a default you did not measure yourself.

The number that matters more is per-layer telemetry during decode, from section 2.4:
24 layers × 55 µs = 1.3 ms per token against a ~3 ms step, **~44%** if drained eagerly,
**0.37%** if buffered on device and drained every 5 steps.

### 3.6 The instrument that trips the machine's known fault

`[M]` inputs: `large-tensor-fault-32gib` (single tensors ≥32 GiB hang at 0% CPU or fault; 31
GiB is clean) and `sdpa-is-memory-efficient` (the default SDPA path on this build retains
147.2 bytes/T², i.e. it materialises the score matrix).

To log per-entry attention mass you need the score matrix. Materialised, for one layer:

```
score_bytes = B · n_heads · T² · dtype_bytes
```

At `B = 1`, `n_heads = 8`, bf16:

| `T` | one layer's score matrix |
|---|---|
| 8,192 | 1.0 GiB |
| 32,768 | 16.0 GiB |
| **46,341** | **32.0 GiB — the fault threshold** |

Solve it: `16·T² = 2³⁵` gives `T² = 2³¹` and `T = 46,341`. Note that `2³¹` elements per head
is exactly the 32-bit-index boundary, which independently corroborates the `[A]` overflow
hypothesis already recorded in `ASSUMPTIONS.md → large-tensor-fault-32gib`.

So: **the attribution probe you most want is the one that trips this machine's silent-hang
fault, at a context length inside the range we intend to study.** And the failure presents as
0% CPU with no error, so it stalls a long run rather than crashing it. Derived from two `[M]`
rows, tagged `[A]` high confidence; the cheapest test is to run it deliberately at
`T = 48,000` in a throwaway process and watch.

The design consequence: **never materialise the full score matrix for telemetry.** Reduce
inside the kernel or in chunks. `m_E` is a sum over an index set and `v̄_E` is a weighted
mean — both are reductions, and a reduction never needs the full matrix resident. This is the
same instinct as computing a percentile with a sketch instead of sorting the log.

---

## 4. Why it matters for Proteus and Mnemosyne

### 4.1 The deliverable is the instrument, not another policy

`research/synthesis.md` states it once and it governs this whole track: *"Build the
instrument, not another policy."* The field has ~30 eviction policies and no dominance
result `[C]` (2603.20397, Mar 2026, concludes no single method dominates). It has one 5/5/5
on the ranked backlog — attribution — because the pain is evidenced, it is testable on one
gfx1151 GPU, and thirty years of observability is a genuine edge rather than a comforting
analogy.

### 4.2 The telemetry contract, written as an interface obligation

Section 3.2 says exactly what to log. Section 2.4 says exactly how. Together they specify
Mnemosyne's eviction plug point:

```
On every eviction decision, per layer, per head, per step, emit:
    step, layer_idx, head_idx           small ints, host-side, free
    |E|                                 int
    m_E                                 device scalar tensor, UNEVALUATED
    ||v̄_E − v̄_K||₂                      device scalar tensor, UNEVALUATED
    attention_backend                   string, recorded once per run, not per step
    dtype                               string, recorded once per run
Drain: one host-device sync every k steps. k is a config field, default 5.
Never: .item(), float(), print(), or a Python conditional on any of these tensors
       inside the step.
```

Four properties of that contract worth stating explicitly:

- **It is defined over tensors and integers, not over Proteus objects.** That is not
  fastidiousness, it is the boundary rule: `mnemosyne → torch`, never `mnemosyne → proteus`
  (CLAUDE.md). An instrument that only works against our model is an implementation detail;
  one that can be pointed at a different model is a contribution.
- **`attention_backend` is not optional.** Section 2.7: the same code path on this machine
  runs two different kernels with two different numerics depending on an environment
  variable, and the API that reports which one is *permitted* lies about which one *ran*
  `[M]`.
- **The drain interval is a config field**, because the config surface is the experimental
  surface, and "how much did observability cost" is itself a measurable ablation axis
  (Exercise A).
- **Two scalars, not a distribution.** Resist logging the attention histogram. Section 3.6 is
  why.

### 4.3 The oracle-diff harness, as the first Mnemosyne milestone

Minimum viable shape:

1. Fix prompt, seed, batch composition, dtype, and attention backend.
2. Run once with the full cache → `p_full` per position; keep on device.
3. Run once under policy P → `p_pol` per position.
4. Compute per-token `D_KL(p_full ‖ p_pol)`, the per-token margin, and the two scalars from
   §3.2 for every eviction decision. All on device; one drain at the end.
5. Compute the null on the nuisance axes of §2.7, from the *same* prompts.
6. Report: median and p99 of the signal, p99 of each null, the ratio, the Spearman
   correlation between per-decision `m_E · ‖v̄_E − v̄_K‖` and the KL at the positions that
   follow, and the effective `n` after prompt-level clustering.

Step 5 before step 6 is the discipline. The lab has already tagged one crash `[M]` that did
not reproduce (`curriculum/README.md`, the hipBLASLt segfault, refuted). A null distribution
is the structural version of the fix.

### 4.4 A hazard the mirror notes do not name: the needle-drop fault is a lower bound

Fault #2 in the battery is *"drop exactly the KV entries spanning the needle; expect a large
drop."* That is right as a direction and wrong as a clean ablation, and the reason matters.

In a causal transformer, position `t`'s residual stream at layer `l` can already contain
information attended in from position `s < t` at layers `< l`. Position `t`'s **own** K and V
at layer `l` are computed from that residual stream. So the needle's content is *smeared
across every subsequent position's cache entries at higher layers* — this is precisely the
copying behaviour that induction and retrieval heads implement `[C]` (2209.11895,
2404.15574).

Consequence: dropping only the needle's own entries removes **direct** readability from those
positions and leaves the copies intact. The observed score drop is therefore a **lower bound**
on the needle's causal contribution, and the size of the gap is unknown.

Two things follow. First, if your needle-drop fault produces a *small* drop, that is not
automatically an indictment of the eval — it may be a measurement of how much the model
already copied forward, which is an interesting quantity nobody reports. Second, the fault
becomes clean only if it is **layer-resolved**: drop the needle's entries at layers `0..l`
and sweep `l`. The curve of score against `l` is a direct measurement of the depth at which
the information stops being local to its own position. `[A]` high confidence in the mechanism
(it is a straightforward consequence of causal attention plus the residual stream, and the
copying heads are well documented); zero confidence about the magnitude at 300M. Cheapest
test: the layer sweep above, inference-only, on the inference rig.

### 4.5 The Hardware Validation Gate is a prerequisite of the instrument, not hygiene

The oracle diff is a difference between two runs. If the runs are not bit-reproducible, the
difference is noise plus signal with no way to separate them. So:

- **Determinism across repeated runs with a fixed seed** is not a nice-to-have; it is what
  makes the null degenerate-by-design and forces you to build it from nuisance axes
  deliberately rather than accidentally.
- **Checkpoint round-trip bit-exactness** is what lets the oracle run and the policy run come
  from the same weights.
- **bf16 numerics** decide whether the instrument works at all in bf16
  (`bf16-numerics-unproven`, `untested`).
- **Both attention paths** must be characterised, since the flag is a numerics change `[M]`.

`research/synthesis.md` already argues the gate as written is under-specified and should be
widened. This module adds one item to that list: **measure the null distribution on each
nuisance axis as part of the gate**, because it is nearly free once the determinism check is
running and it is load-bearing for every downstream result.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Line numbers are pinned to the revisions in
`PROVENANCE.md`.

### 5.1 What a production KV cache actually measures about itself

This is the most valuable read in the module, because the field list *is* the argument.

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/core/kv_cache_metrics.py:46` | `class KVCacheMetricsCollector` — the entire observability surface of vLLM's KV cache, in 97 lines. Read the whole file; it is short. |
| `memory/vllm/vllm/v1/core/kv_cache_metrics.py:49` | `def __init__(self, sample_rate: float = 0.01)` — **production KV observability is 1% sampled.** Sampling is correct for latency distributions and useless for causal attribution: the block you need to explain one wrong answer is, with 99% probability, not instrumented. |
| `memory/vllm/vllm/v1/core/kv_cache_metrics.py:24` | `access_history: deque[int] = deque(maxlen=4)` — reuse-gap history is bounded at four. A block accessed forty times is structurally unrepresentable. Bounded cardinality, exactly like a label set you cap to keep a time-series database alive. |
| `memory/vllm/vllm/v1/metrics/stats.py:162` | `class KVCacheEvictionEvent` — **three fields: `lifetime_seconds`, `idle_seconds`, `reuse_gaps_seconds`.** All time. Not one field about what the block *contained* or whether any answer changed. The attribution gap, in a dataclass. |
| `memory/vllm/vllm/v1/metrics/stats.py:115` | `class PrefixCacheStats` — hits, queries, requests, and a separate set of counters for preempted requests. Note that "hits" is denominated in *tokens*, which is a choice with consequences (next row). |
| `memory/vllm/vllm/v1/core/kv_cache_manager.py:225` | `get_computed_blocks` — the hit numerator is produced here, and it is capped at `num_tokens − 1` (you need a forward pass to get logits) and then floored to block alignment. So an *exact duplicate prompt* recomputes a whole trailing block and reports less than 100%. Your hit rate has a scheduler decision baked into its numerator. |
| `memory/vllm/vllm/v1/core/block_pool.py:679` | `_maybe_evict_cached_block` — eviction happens **lazily, at reallocation**, not when a block is freed. A zero-refcount block is still matchable. Therefore "blocks in use" and "entries available for hits" are two different numbers and neither is "cache occupancy." |
| `memory/vllm/vllm/v1/core/block_pool.py:717` | `self.metrics_collector.on_block_accessed(block)` — inside `touch`, which runs only on a prefix hit. So the access counter counts *hits*, not reads. The attention kernel reads every resident block on every decode step and increments nothing. |

### 5.2 Where the probe changes the system

| Where | What to look at, and why |
|---|---|
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355` | `match_prefix` — read the docstring, which openly admits the method may mutate the tree. A lookup that terminates mid-node calls `_split_node`, re-parents a subtree and clones two tensors. **The probe is a write.** Your instinct that a read-only lookup is safe to instrument freely does not hold. |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:565` | `evict` — note `start_time = time.perf_counter()` four lines in: eviction is timed, and the timing is host-side. Note also that the heap is built from *leaves only*, so eviction order is topological, not recency-ordered — a hot child pins a cold parent forever. "LRU hit rate" does not mean what it means in a buffer cache. |
| `memory/sglang/python/sglang/srt/mem_cache/evict_policy.py:16` | `class LRUStrategy(EvictionStrategy)` — the entire replacement-policy surface is one `get_priority(node)` function, and LRU is a one-line return of `last_access_time`. This is the natural **fault-injection plug point**: a `RandomStrategy` and a `WorstCaseStrategy` are each three lines, and injecting them is how you find out whether your metric notices. |
| `memory/mooncake/mooncake-store/src/master_service.cpp:6382` | `MasterService::BatchEvict` — the working set is defined by **leases**, not recency. An object is not evictable until its lease expires; `Get` renews it. So "hit rate" here is measured against a TTL policy, and comparing it to an LRU hit rate from another system is comparing two different quantities with the same name. |
| `memory/flashinfer/flashinfer/decode.py:1239` | `plan` — the CSR page table `(indptr, indices, last_page_len)`. There is no present bit and no fault handler; a miss is **unrepresentable**. There is no counter to increment because there is no event to count. |
| `memory/flashinfer/flashinfer/decode.py:1481` | `indptr_host = indptr.to("cpu")` — a device-to-host copy of the page table, already in the critical path, once per plan rather than per access. If you need a host-side probe, this is the kind of place to piggyback on: a sync that is already happening is free. |

### 5.3 Telemetry that does not cost throughput

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/train/trainer.py:1037` | `record_metric` — metrics are buffered as **detached device tensors** in a per-step `OrderedDict`, deliberately not evaluated at record time. This is the whole trick. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1394` | `_log_metrics` — read the comment above `move_metrics`: *"if training on GPU and `bookkeeping_device` is CPU, this triggers host-device sync… we prefer to do that early and then finish processing the metrics in a separate thread."* One sync, then a background thread does the reduction and the callback fan-out. |
| `training/olmo-core/src/olmo_core/train/trainer.py:200` | `metrics_collect_interval: int = 5` — the drain interval, as a config field with a docstring. Compare against your own measured per-probe cost in Exercise A. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1514` | `if first_batch or self.global_step % self.metrics_collect_interval == 0:` — the modulo gate. Four lines of control flow are the entire difference between 42% overhead and 0.35%. |
| `training/olmo-core/src/olmo_core/train/trainer.py:1517` | `torch.cuda.set_sync_debug_mode("warn")` — **the tool you did not know you had.** PyTorch will warn on every unintended host-device sync. OLMo-core enables it immediately after the drain, so any *other* sync in the step gets caught. Use this in Exercise A. |
| `training/olmo-core/src/olmo_core/utils.py:749` | `cuda_sync_debug_mode` — the same thing as a context manager, so you can wrap a suspect block instead of the whole run. |
| `training/nanogpt/train.py:216` | `estimate_loss` — the counter-example, and it is instructive rather than wrong. |
| `training/nanogpt/train.py:225` | `losses[k] = loss.item()` — **one host-device sync per eval iteration**, and `eval_iters` defaults to 200, so a single `estimate_loss()` call performs 200 syncs. At the `[M]` ~55 µs measured here that is ~11 ms of pure stall per eval. Negligible against a 200-batch eval, and it would be catastrophic in the training step. Instrumentation cost is a *rate*, not a constant. |
| `training/nanogpt/README.md:51` | `1.4697` — the published validation loss for the shakespeare_char config, and the Hardware Validation Gate's target. Read it together with `train.py:216`: the number is a Monte Carlo mean over 200 random batches, so it has a standard error, and "reproducing it" means landing inside roughly a hundredth, not matching four decimals. A gate threshold without a variance estimate is a coin flip. |

### 5.4 The instrumentation-versus-throughput conflict, in the model code

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:328` | `attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling` — the score matrix, materialised, in the **eager** path. `B · n_heads · T · T` elements. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:332` | `softmax(..., dtype=torch.float32)` — the softmax is computed in fp32 even in a bf16 model. Relevant to §2.7: your bf16 null is not uniformly bf16. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:337` | `return attn_output, attn_weights` — this is the **only** path that hands you the attention distribution. A fused kernel never materialises it, so asking for attention weights forces the slow, memory-hungry path. The attribution probe and the efficient kernel are in direct conflict, and §3.6 prices the conflict on this machine. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | `key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)` — the one line where bytes enter the cache, and therefore the natural plug point for Mnemosyne's write-side instrumentation. Note what has already happened above it: QK-norm at `:390`, RoPE at `:394`. A cached key is `RoPE(RMSNorm(k))`, so anything you log about "the key" is a log about a doubly-transformed quantity. |

---

## 6. Exercises

All three run on the Z13. Activate first, in PowerShell, dot-sourced so the variables
survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing caveats, from `ASSUMPTIONS.md`.** Single tensors ≥32 GiB hang the GPU silently at
0% CPU (`large-tensor-fault-32gib`); keep every buffer under 31 GiB. bf16 numerics on gfx1151
are `untested` (`bf16-numerics-unproven`), so exercises default to fp32 and Exercise B treats
bf16 as an *experimental variable* rather than a default. The Hardware Validation Gate has
not run, so nothing measured here is evidence by house standard until it does — these are
instrument-shakedown runs and should be labelled as such in the notebook.

Write scratch scripts under `notebook/`. Exercise B is the seed of a rig component and
acquires tests when it is reused (house rule: one-off analysis scripts are exempt from TDD
only until reuse).

---

### Exercise A — price the probe, then find the syncs you did not know you had

**Goal:** produce your own version of the module's `[M]` numbers, and learn the tool that
finds accidental synchronisation.

**Hardware:** one gfx1151 GPU, native Windows. **CPU fallback:** runs unchanged and is
*informative by contrast* — on CPU there is no asynchronous queue, so `.item()` is nearly
free and the curve should be flat. That contrast is the point: the entire break in §2.4
exists because of the accelerator.

**Runtime:** ~3 minutes on GPU after torch imports; ~2 minutes on CPU.

```python
"""What does a telemetry read cost, and how fast does it amortise?"""
import random, statistics, time, torch

M, N_ITERS, N_TRIALS, SEED = 1024, 200, 9, 1337
DTYPE = torch.float32                      # bf16 numerics are unproven on this machine
dev = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(M, M, dtype=DTYPE, device=dev)
b = torch.randn(M, M, dtype=DTYPE, device=dev)

def micro(fn, reps=500, trials=9):
    out = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps * 1e6)   # microseconds
    return statistics.median(out), min(out), max(out)

def trial(drain_every):
    acc = torch.zeros((), dtype=DTYPE, device=dev)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_ITERS):
        acc = acc + (a @ b)[0, 0]
        if drain_every and (i + 1) % drain_every == 0:
            _ = acc.item()                                     # <- the probe
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1e3          # ms per iteration

for _ in range(50):                                            # warm up: kernels load once
    _ = (a @ b)[0, 0].item()
if dev == "cuda":
    torch.cuda.synchronize()
    t = torch.zeros((), dtype=DTYPE, device=dev)
    for name, fn in [("bare synchronize", torch.cuda.synchronize),
                     ("idle .item()",     lambda: t.item()),
                     ("idle .cpu()",      lambda: t.cpu()),
                     ("idle float(t)",    lambda: float(t))]:
        med, lo, hi = micro(fn)
        print(f"{name:<18} {med:7.2f} us  [{lo:.2f}, {hi:.2f}]")

DRAINS = [None, 1, 2, 5, 10, 25, 50, 200]
order = [d for d in DRAINS for _ in range(N_TRIALS)]
random.Random(SEED).shuffle(order)                             # randomise arm order
samples = {str(d): [] for d in DRAINS}
for d in order:
    samples[str(d)].append(trial(d))
for d in DRAINS:
    v = samples[str(d)]
    print(f"drain_every={str(d):>4}  median {statistics.median(v):.4f} ms/iter "
          f"[{min(v):.4f}, {max(v):.4f}]  stdev {statistics.stdev(v):.4f}")
```

**Why the arm order is shuffled, and why you should care.** My first attempt ran the arms in
a fixed order and the `never` arm — which must be fastest — came out *slower* than
`drain_every=200`. The first arm eats residual warm-up drift. Fixed-order microbenchmarks
manufacture that artifact routinely, and a shuffled order plus a reported standard deviation
is the cheapest defence. Shuffling is still not enough for the micro-arms: my second run
disagreed with the first by 2.6× on the `.item()` figure because the *machine* was busier,
and shuffling within a run cannot fix a run-level shift. The fix that worked was
**interleaving** — run all four micro-arms round-robin inside each timed block and report the
within-block ratio to the host-tensor control, which is drift-invariant by construction.
Add that variant yourself; it is fifteen lines and it is the difference between a number and
an anecdote.

**Part 2 — the sync detector.** Add this and re-run a short loop:

```python
torch.cuda.set_sync_debug_mode("warn")     # olmo-core does this at trainer.py:1517
```

Now go find three constructs in code you have already written that sync without looking like
they do. Candidates worth trying: `if loss > 1.0:` on a device tensor; `print(t)`;
`t.tolist()`; `torch.nonzero(mask)` (output shape is data-dependent, so it must sync);
`tensor[mask]`; `len(t[t > 0])`; and any `assert` on a device scalar.

**Deliverables — four numbers and one curve.**

1. Median cost of a bare `torch.cuda.synchronize()` and of an idle `.item()`, in µs, with
   min/max. **Reference `[M]` (mine, section 6): 3–7 µs and 52–58 µs on a quiet machine.**
   If you get ~136 µs, check what else is running before you write it down.
2. **The ratio `idle .item()` ÷ `host .item()`, measured inside the same block.**
   **Reference `[M]`: median 236, range 169–335.** This is the number that survived three
   runs; the absolute microseconds did not.
3. The difference between `drain_every=1` and `drain_every=None` — that difference *is* the
   per-probe cost per iteration. Check it against your isolated `.item()` figure. They should
   agree within noise; if they do not, say which is wrong and why.
4. The smallest drain interval at which overhead is under 1% of the unprobed time. Compare
   with OLMo-core's default of 5 (`trainer.py:200`).
5. Plot median ms/iter against drain interval with min/max whiskers. **Prediction: monotone
   decreasing, asymptoting to the `None` arm.**

**What a falsification would mean, and it is likely — it happened to me.** At this shape the
per-probe cost is about 4% of a 1.33 ms step and the trial-to-trial spread is of the same
order, so deliverable 5's curve may simply not resolve. On my randomised-order run the
medians came out `never` 2.16, `k=1` 2.87, `k=2` 5.11, `k=5` 7.53, `k=10` 2.28 ms — flatly
non-monotone, with per-arm standard deviations up to 11.7 ms. **The honest answer there is
"below the noise at this shape," not "no cost."** Fix it by making the loop cheaper per
iteration (drop `M` to 256) so the probe dominates, or by raising `N_ITERS`, or by
interleaving as above. Reporting "not resolvable at this shape" is a correct result;
reporting a number you cannot resolve is how a curriculum acquires an `[M]` that does not
reproduce, which has happened here once already (`curriculum/README.md`, the refuted
hipBLASLt segfault).

---

### Exercise B — build the oracle-diff instrument and measure its noise floor

**Goal:** a working miniature of Mnemosyne's deliverable, and the number without which none
of its outputs are interpretable. This is the exercise that matters.

**Hardware:** one gfx1151 GPU. **CPU fallback:** identical code, drop `T` to 256 and layers
to 2. **Runtime:** 45–90 minutes to write; under 2 minutes to run on GPU, ~5 on CPU.

**No download required.** You will use a randomly-initialised transformer. That sounds like a
cheat and it is not: a random model still produces a well-defined output distribution, and
the four quantities this exercise measures — the nulls — are properties of the *hardware and
the kernels*, not of the weights. The one thing a random model cannot tell you is whether the
signal *localises*, and that is exactly why the exercise asks you to compute the localisation
baseline instead: with near-uniform attention over `T` positions, dropping one entry should
move KL by `O(1/T)` and should show **no** dependence on position. Any trained model must beat
that baseline for the instrument to be worth anything, and you now know the number it has to
beat.

```python
"""Oracle-diff KL: the signal, and the four nulls it has to clear."""
import torch, torch.nn.functional as F

torch.manual_seed(1337)
dev = "cuda" if torch.cuda.is_available() else "cpu"
L, H, D, T, V = 4, 4, 256, 1024, 4096          # layers, heads, model dim, ctx, vocab

def build(dtype):
    return torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(D, H, 4 * D, dropout=0.0,
                                         batch_first=True, norm_first=True),
        num_layers=L).to(device=dev, dtype=dtype).eval()

def causal_mask(dtype, drop=None):
    m = torch.triu(torch.full((T, T), float("-inf"), device=dev, dtype=dtype), diagonal=1)
    if drop is not None:
        m[:, drop] = float("-inf")             # nobody may attend to this position
    return m

@torch.no_grad()
def logits_of(model, x, head, drop=None):
    out = model(x, mask=causal_mask(x.dtype, drop))
    return head(out.to(head.weight.dtype))[:, -1, :].float()

def kl(p_logits, q_logits):
    """Forward KL(p_full || p_pol) in nats. F.kl_div(input, target, log_target=True)
    computes sum target*(log target - input), i.e. KL(target || input)."""
    p = F.log_softmax(p_logits, -1)
    q = F.log_softmax(q_logits, -1)
    return F.kl_div(q, p, log_target=True, reduction="sum").item()

fp32 = build(torch.float32)
head = torch.nn.Linear(D, V, bias=False).to(device=dev, dtype=torch.float32)
x = torch.randn(1, T, D, device=dev)

ref = logits_of(fp32, x, head)

# --- the four nulls -------------------------------------------------------
null = {}
null["repeat"] = kl(ref, logits_of(fp32, x, head))
with torch.no_grad():                                       # same content, batch 8
    xb = x.repeat(8, 1, 1)
    ob = fp32(xb, mask=causal_mask(xb.dtype))
    null["batch"] = kl(ref, head(ob)[0:1, -1, :].float())
bf16 = build(torch.bfloat16)
bf16.load_state_dict({k: v.to(torch.bfloat16) for k, v in fp32.state_dict().items()})
null["bf16"] = kl(ref, logits_of(bf16, x.to(torch.bfloat16), head))
# 4th null: re-run this script with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 in a
#           SEPARATE process and diff the two `ref` tensors.

# --- the signal -----------------------------------------------------------
positions = list(range(1, T - 1, max(1, T // 64)))          # NOT 0: masking column 0
sig = {p: kl(ref, logits_of(fp32, x, head, drop=p))         # leaves row 0 all -inf
       for p in positions}

print("nulls (nats):", {k: f"{v:.3e}" for k, v in null.items()})
vals = sorted(sig.values())
med = vals[len(vals) // 2]
print(f"signal median {med:.3e}   p99 {vals[int(0.99*len(vals))]:.3e}"
      f"   min {vals[0]:.3e}   max {vals[-1]:.3e}")
print(f"SNR (median signal / largest null) = {med / max(max(null.values()), 1e-30):.1f}")
print("first 8 (position, KL):", [(p, f"{sig[p]:.2e}") for p in positions[:8]])
```

**Three bugs this code has already had, so you do not rediscover them.** `positions` must
start at 1, not 0 — masking column 0 leaves row 0 entirely `-inf` and the softmax returns
NaN. The mask must be built in the model's dtype, or the bf16 arm errors. And the output
head must be applied to a tensor cast to the head's dtype, for the same reason. Each cost me
a run; none is interesting.

**Deliverables — four numbers, one ratio, one plot.** I ran this at three seeds (1337, 4242,
90210) on the config above; my numbers are in the right-hand column so you have something to
disagree with.

| # | Deliverable | Mine `[M]`, 3 seeds |
|---|---|---|
| 1 | `null["repeat"]`, in nats | **exactly 0.0** at all three seeds |
| 2 | `null["batch"]`, in nats | **exactly 0.0** at all three seeds |
| 3 | `null["bf16"]`, in nats | **6.6e-06 / 8.4e-06 / 7.9e-06** |
| 4 | signal median (64 dropped positions) | **7.0e-07 / 8.2e-07 / 1.0e-06** |
| 5 | **SNR** = median signal ÷ largest null | **0.1 at all three seeds** |

Then interpret, and this is the part that matters:

1. **A degenerate repeat-null is the finding, not a bug.** Zero means reruns are not a null
   and every real null must come from a nuisance axis. That sentence is the whole of §2.7.
2. **The batch-composition null was also zero** at this shape — batch 8 of identical rows
   gave bit-identical logits for row 0. Good news, and *only* for this shape; re-check it at
   a shape where padding is involved before relying on it.
3. **SNR = 0.1 means the bf16 numerical noise is ~10× the single-entry eviction signal.**
   At random init, at this scale, **single-entry eviction is undetectable in bf16.** Note
   carefully what this does and does not say: random weights give near-uniform attention, so
   dropping one of 1,024 entries is close to the *smallest possible* signal, and a trained
   model concentrating mass on one retrieval-relevant entry would produce a larger one. What
   it does establish is the **floor**, and the floor is above the noise. This belongs in
   `ASSUMPTIONS.md` as a new row and it is the most decision-relevant number in the module.
4. **Watch for a negative KL.** At seed 4242 one probe returned **−1.7e-07 nats**. KL cannot
   be negative; a negative value is floating-point cancellation, i.e. **the noise floor
   announcing itself in a form you cannot argue with.** If your run produces one, you have
   independently confirmed deliverable 5 without needing the bf16 arm at all.
5. **Plot signal KL against dropped position. It is not flat.** I predicted flat before
   running — near-uniform attention, `O(1/T)` per entry — and that prediction was wrong.
   Position 1 gave the largest KL at every seed (1.3e-05 / 5.5e-06 / 9.9e-06) and the curve
   decays with position. The mechanism is §4.4 in miniature: an early entry is upstream of
   every later position's residual stream at every layer, so removing it perturbs far more of
   the computation than removing a late one. That is causal-mask-induced primacy appearing
   **before any training**, which is what the structural theory of position bias predicts
   `[C]` (2602.16837, Feb 2026). Reproduce the decay; it is the cheapest thing in this module
   that connects a measurement to a paper.
6. **Compute Spearman ρ between the dropped position's attention mass and its KL.**
   Prediction with random weights: near zero, because mass is near-uniform while KL is
   position-structured — so ρ against *mass* should be weak even though ρ against *position*
   is strong. That dissociation is §3.2's point (`mass` is one factor of two) shown with your
   own data. When you later run this on a trained model, ρ against mass is the number that
   must move for the `m_E · ‖v̄_E − v̄_K‖` decomposition to be useful in practice.

**One free by-product you should not miss.** Running this on gfx1151 emits
`UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it
with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on stderr. That is `ASSUMPTIONS.md →
sdpa-is-memory-efficient` reproducing itself live: the memory-efficient path did **not** run,
and the only signal saying so is a warning on stderr — not an API you can query. Capture
stderr in your harness.

**Stretch, if a small trained checkpoint is available locally.** Repeat with a real model and
compare: SNR, the position curve, and ρ. Then answer the lab's riskiest question — does the
divergence *localise* to identifiable evicted entries, or does it smear? A smear kills backlog
item #1 and changes the plan, which is exactly why it is worth running first.

**Failure modes to expect and what they mean.** If `null["batch"]` comes out at the same order
as the signal, your kernels' reduction order is batch-dependent and every cross-batch-size
comparison in this lab is invalid — a real finding, and cheap. If `null["bf16"]` is
`inf` or `nan`, you have found a numerics bug worth a `BLOCKERS.md` entry. If the signal is
*zero* for every position, check that your mask is actually being applied; a silently ignored
`mask` argument is the single most common bug in this exercise.

---

### Exercise C — fault-inject a retrieval metric until it fails

**Goal:** make the adverse-selection claim quantitative — a needle only modestly more salient
than background text survives aggressive heavy-hitter eviction, so NIAH structurally cannot
fail for the policies we most want to break — and derive the salience threshold analytically
*before* measuring it.

**Hardware:** none. Pure Python plus numpy. **Runtime:** 20 minutes to write, ~15 s for the
main table and under a minute for everything.

This is a **simulation of the selection mechanism, not a measurement of a model.** It can show
that a metric's pass criterion is correlated with what a policy retains. It cannot tell you the
magnitude at real scale. Label it that way in the notebook entry.

**Setup.** A context of `T` positions. Background attention masses are drawn i.i.d. from
`Exponential(1)` (a defensible stand-in: attention mass over ordinary text is heavy-tailed and
positive). A needle is inserted with mass `s ×` the background mean. A policy retains a budget
`b` of entries. The eval "passes" if the needle survives.

**Do the arithmetic before you run it.** Top-`b` retention keeps the needle iff its mass
exceeds the `(1 − b)` quantile of the background, where `b` is the budget *fraction*. For
`Exponential(1)` the `q` quantile is `−ln(1 − q)`, so the survival threshold is

```
s* = −ln(1 − (1 − b)) = −ln(b)
```

**The salience a needle needs to survive a heavy-hitter policy is minus the log of the budget
fraction.** That is the whole exercise in one line, and it is worth staring at:

| Budget kept | `s* = −ln(b)` |
|---|---|
| 50% | 0.69 |
| 25% | 1.39 |
| 10% | **2.30** |
| 5% | 3.00 |
| 1% | 4.61 |

**A 100× compression only requires a needle 4.6× more salient than average text.** Write the
10% prediction — `s* ≈ 2.30` — down before running the Monte Carlo. Then:

```python
import numpy as np

rng = np.random.default_rng(1337)
T, TRIALS = 4096, 20_000

def pass_rate(policy, s, budget_frac):
    keep = int(T * budget_frac)
    hits = 0
    for _ in range(TRIALS):
        mass = rng.exponential(1.0, T)
        needle = rng.integers(0, T)
        mass[needle] = s                                   # salience in units of the mean
        if policy == "heavy_hitter":
            survived = needle in np.argpartition(-mass, keep)[:keep]
        elif policy == "uniform":
            survived = needle in rng.choice(T, keep, replace=False)
        elif policy == "recency":
            survived = needle >= T - keep
        hits += bool(survived)
    return hits / TRIALS

for s in [1.0, 1.5, 2.0, 2.3, 3.0, 5.0, 10.0]:
    row = [f"{pass_rate(p, s, 0.10):.3f}" for p in ("heavy_hitter", "uniform", "recency")]
    print(f"s={s:>5}  heavy_hitter={row[0]}  uniform={row[1]}  recency={row[2]}")
```

**Deliverables — three checks and one threshold.** My run, `T = 4096`, 20,000 trials, seed
1337, is in the right-hand column.

| # | Deliverable | Prediction | Mine `[M]` |
|---|---|---|---|
| 1 | **Harness check:** `uniform` pass rate, every `s` | exactly the budget, 0.100 | 0.098–0.108 |
| 2 | **Second anchor:** `recency` pass rate, every `s` | 0.100 (needle position is uniform) | 0.097–0.102 |
| 3 | **The threshold:** `heavy_hitter` at `s = 2.3`, budget 10% | ≈0.5 from `s* = −ln(0.10)` | **0.453** |
| 4 | `heavy_hitter` at `s = 2.0` and `s = 3.0` | below and above the step | **0.000** and **1.000** |

Deliverables 1 and 2 are what make this an exercise rather than a demo: both are known in
closed form, so a harness bug announces itself instead of producing a plausible wrong graph.

**The result is sharper than "biased," and that is the finding.** The transition from 0.000 to
1.000 happens between `s = 2.0` and `s = 3.0`. At `T = 4096` the binomial spread around the
threshold is tiny relative to its mean, so the pass criterion is effectively a **step function
in salience** with the step at `−ln(b)`. NIAH is therefore not a metric that is *somewhat*
favourable to heavy-hitter eviction — it is a metric that is **trivially passed or trivially
failed** with the switch sitting at a very modest salience. A metric with no dynamic range
cannot rank anything, which is exactly `research/notes/evaluation-landscape.md` §2(a) arriving
from a different direction.

5. **The gap that is the actual claim.** At `s = 10` (a salient NIAH needle) versus `s = 1.2`
   (a NoLiMa-style associative needle with no lexical anchor), report `heavy_hitter` pass rate
   at budgets of 50%, 25%, 10% and 5%. **Mine `[M]`:**

   | Budget | `s = 10` | `s = 1.2` | gap |
   |---|---|---|---|
   | 50% | 1.000 | 1.000 | 0.000 |
   | 25% | 1.000 | 0.000 | **1.000** |
   | 10% | 1.000 | 0.000 | **1.000** |
   | 5% | 1.000 | 0.000 | **1.000** |

   The gap does not *grow* with compression, as I expected before running it — it **jumps to 1
   the moment `−ln(b)` crosses the associative needle's salience** (between 50% and 25%
   budget, since `−ln(0.5) = 0.69 < 1.2 < 1.39 = −ln(0.25)`) and stays pinned there. So the
   right report is not a slope, it is a **budget at which the associative needle dies while the
   lexical one is untouched** — and you can predict that budget in closed form as
   `b = e^{−s}`. That is the shape to look for on real hardware
   (`research/notes/evaluation-landscape.md` open question 1).

**Then apply the fault battery to the metric itself.** Needle absent (`s` such that the needle
is indistinguishable — set `mass[needle]` from the background draw): pass rate must fall to the
policy's blind rate, `b/T`. If your metric does not floor there, it is reading something other
than the needle. Distribution shuffle (permute `mass`): must not change anything, since the
model has no positional term — a change means your harness has one you did not intend.

**What would falsify the module's framing.** If `heavy_hitter` and `uniform` pass rates were
similar at realistic salience, the adverse-selection argument against NIAH would be weak and
`research/notes/evaluation-landscape.md` §2(e) would need softening. It is not weak in
simulation — but note honestly what the simulation assumed: an `Exponential(1)` background and
a needle whose salience is a single scalar multiple of the mean. Both are stipulations, not
measurements. **The whole result is a consequence of the tail of the background distribution**,
and if real attention mass over text is heavier-tailed than exponential the threshold `−ln(b)`
moves. Measuring the actual background distribution of per-token attention mass on a real model
is a one-day job on the inference rig and would turn this simulation into a prediction.

---

## 7. Self-check

Answers at the end. Do not scroll.

1. You run an eviction arm. Per-token KL against the full-cache oracle is 0.030 nats, well
   above your measured seed-to-seed null of 0.002. Synthetic recall accuracy is unchanged
   within its confidence interval. State precisely what you have evidence for, what you do
   not, and the one extra quantity that would resolve it.

2. Your needle-retrieval eval scores the same whether the needle is present or deleted. Name
   what it has been measuring, and give the cheapest next check.

3. A policy evicts a set carrying 12% of the attention mass at a given layer and head. How
   much attention-output error does that imply?

4. You add per-head attention-mass logging to a 24-layer model and run at `T = 48,000`,
   batch 1, 8 heads, bf16. Predict what happens on this machine, and say at what `T` it starts.

5. Pairing arms bought 10× in sample size at ρ = 0.9. List everything that must be held
   identical for that ρ to survive, and name the one item on your list that is easiest to
   break by accident on this machine.

6. Someone asks for "the prefix cache hit rate." Give two defensible numerators from vLLM's
   code, say which question each answers, and explain why "blocks in use" is not the
   denominator for either.

---

## 8. What is still unsolved here

Everything below is testable at 20M–300M params on one GPU with a `[M]` ≥62 GiB fast tier, or
on the inference rig (a 7–14B off-the-shelf model resident in that tier). Each needs a
pre-registered hypothesis card before it runs.

1. **Does oracle-diff KL localise to identifiable evicted entries, or does it smear?** The
   foundational question of the whole instrument, and the lab's stated first test
   (`research/synthesis.md`, `open-problems-ranked.md` Q1). If it smears, backlog item #1
   collapses and the plan changes — which is what makes it worth running first.

2. **Does divergence track task accuracy?** Contested, and the two 2026 answers do not
   validate against each other: a fixed-contract diagnostic `[C]` (2605.08234, May 2026)
   versus error certificates via randomised design `[C]` (2607.21475, Jul 2026). Section 3.3's
   two-line arithmetic shows both dissociation directions are possible; nobody has measured
   which dominates in practice, at any scale.

3. **Where does the bf16 noise floor sit relative to a real eviction signal at 300M?**
   Added by this module, and **partly answered**: `[M]` at random init, 4 layers, T = 1024,
   three seeds, the bf16 null is ~10× the median single-entry eviction signal (§2.7,
   Exercise B). What is unresolved is whether a *trained* model's signal clears it — the
   floor case says no, but a retrieval head concentrating mass on one entry could move the
   signal by orders of magnitude. The next test is the same script against a trained
   checkpoint, and it is an afternoon. If the signal does not clear the floor, **the
   oracle-diff instrument must run in fp32**, which roughly doubles its memory cost and
   changes the capacity arithmetic for every arm in the backlog.

4. **Is fault-injection calibration practised anywhere?** `research/notes/evaluation-landscape.md`
   marks this `[A]` high confidence: the note's author searched twelve months of eviction
   papers and found none reporting a needle-removed control. I searched again on 2026-07-26 and
   found nothing contradicting it. This is a publishable methodological result *before* any
   research arm runs — which is unusual and worth exploiting.

5. **How much of a needle's content has already been copied forward by the time you drop its
   cache entries?** Section 4.4. The needle-drop fault is a lower bound and nobody quantifies
   the gap. The layer-resolved sweep that would measure it is inference-only and cheap.

6. **How much of the SnapKV-family gain is the observation window versus the scoring function
   versus the budget allocation?** `[C]` 2406.02069 / 2404.14469. The specific confound the
   field has not controlled, and the first real use for the harness.

7. **Nobody has published the background distribution of per-token attention mass.** Exercise
   C shows the salience a needle needs to survive top-`b` eviction is `s* = −ln(b)` — but only
   under an `Exponential(1)` background, which is a stipulation. The threshold, and therefore
   the entire adverse-selection argument against NIAH, is a function of that distribution's
   tail. Measuring it on a real model at a few context lengths is a one-day job on the
   inference rig and would convert a simulation into a prediction. It is also, as far as I can
   establish, unpublished.

8. **Attention weights are used as attribution throughout the literature and are not
   attribution.** Section 3.2 shows the error is a product of two factors and every
   heavy-hitter policy optimises one. The 2026 line that scores by representation change
   instead `[C]` (2606.26472, Jun 2026) is a direct response; it is new, unreplicated, and
   should be treated as a lead.

9. **There is no published cost model for GPU-resident telemetry.** Section 3.5's arithmetic
   is two `[M]` numbers and a division, and one of those numbers moved by 2.6× between runs
   on the same machine on the same day (section 6) — so even the *input* to the cost model is
   a research question here, not a datasheet lookup. The 2026 serving survey names seven
   unmeasured KV-specific quantities `[C]` (2607.02574, Jun 2026); observability cost is not
   on that list either.

10. **Minimum detectable effect is essentially never reported.** The framework exists
   `[C]` (2411.00640) and small-scale variance is documented as large enough to make
   differences meaningless `[C]` (2406.10229). Our own pre-registration cards do not carry an
   MDE field yet; §3.4 is the arithmetic that should go in one.

11. **Contested: mean versus worst-case aggregation.** `[C]` 2510.13334 shows worst-case
    aggregation changes the *ranking* of eviction policies. Since a ranking is the only thing
    an ablation produces, this is not a reporting preference — it is a choice that determines
    the conclusion, and it must be pre-registered.

12. **Contested: does the ranking survive scale?** `[C]` 2512.24503 finds proxy-model rankings
    preserved only under specific LR/batch conditions; `[C]` ATLAS (2605.28079) finds 7 of 26
    models shifting ≥2 rank positions between length regimes. `ASSUMPTIONS.md →
    ablation-scale-sufficient` is `untested` and load-bearing for the entire backlog. The
    lab's answer is a rider, not a project: every arm runs at ~30M and ~300M under muP
    `[C]` (2203.03466) and reports Spearman ρ of the arm ordering.

---

## Answers to the self-check

**1.** You have evidence that the policy **changed the model's output distribution**, by an
amount fifteen times your noise floor. You do **not** have evidence that it changed behaviour,
nor that it did not: section 3.3 shows a 0.037-nat divergence with no argmax change and a
0.0008-nat divergence that flipped the answer, so the two metrics dissociate in both
directions and neither result constrains the other. You also do not know *where* the
divergence came from. The one extra quantity that resolves the behavioural question is the
**per-token margin** `p_full(top1) − p_full(top2)`: slice the KL by margin and ask whether the
divergence is concentrated on low-margin positions. Divergence on high-margin tokens is
cosmetic; the same divergence on low-margin tokens is a coin flip you just biased. (For the
*where*, the additional quantity is `m_E · ‖v̄_E − v̄_K‖` per eviction decision, correlated
against the KL at subsequent positions.)

**2.** It has been measuring the model's **prior** — the answer is guessable from the question,
the haystack's construction, or lexical overlap, without reading the needle at all. This is
fault #1 in the battery and it is the first one to run precisely because it is the cheapest
and it invalidates everything downstream. Cheapest next check: compute the **chance level in
closed form** for your task and compare. If the needle-absent score equals chance, the eval is
fine and you deleted the wrong span. If the needle-absent score is well above chance, the eval
is answering from the prior and must be rebuilt with a high-entropy value (a random symbol, so
priors cannot help) *and* no lexical overlap between query and needle — both, because either
one alone leaves a shortcut open `[C]` (2502.05167).

**3.** **You cannot say.** The error is `m_E · (v̄_E − v̄_K)` — evicted mass times the
difference between the mean evicted value and the mean retained value. Twelve percent is one
factor of a product; if the evicted values look like the retained ones the error is near zero,
and if they are orthogonal to them it is 12% of the output's magnitude. Reporting evicted mass
as if it were error is exactly the attribution mistake this module exists to prevent, and it is
the formal reason value-aware eviction is an open question rather than an obvious win. Log
`‖v̄_E − v̄_K‖₂` next to `m_E`; it costs one reduction over a tensor you already have.

**4.** The score matrix materialises at `B · n_heads · T² · 2` bytes = `16 T²`. At
`T = 48,000` that is 36.9 GiB **in a single tensor**, past the `[M]` 32 GiB fault threshold.
The documented failure mode is not an exception — it is a **hang at 0% CPU with host free RAM
draining**, so a long run stalls silently rather than crashing (`ASSUMPTIONS.md →
large-tensor-fault-32gib`). It starts at `T = √(2³¹) ≈ 46,341`. Two lessons: the attribution
probe you most want is the one that trips this machine's worst fault, and the fix is to reduce
inside the kernel or in chunks rather than materialising — `m_E` is a sum and `v̄_E` is a
weighted mean, and neither needs the full matrix resident.

**5.** Prompts, prompt order, sampling seed, batch size **and batch composition** (padding
changes reduction order), dtype, attention backend, the kernel-selection environment
(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`), torch/ROCm build, and the model checkpoint
bit-for-bit. **The easiest to break by accident on this machine is the attention backend**,
because `torch.backends.cuda.flash_sdp_enabled()` returns `True` whether or not the
memory-efficient path actually ran — the API reports what is permitted, not what executed, and
the only honest signal is a stderr warning `[M]` (`ASSUMPTIONS.md →
sdpa-is-memory-efficient`). Two arms run days apart in shells with different environment
variables are not paired, and nothing in the output will tell you.

**6.** Two defensible numerators, both in the code:
(a) **`PrefixCacheStats.hits`** (`memory/vllm/vllm/v1/metrics/stats.py:115`), fed from
`get_computed_blocks` (`memory/vllm/vllm/v1/core/kv_cache_manager.py:225`) — *tokens whose
prefill the scheduler skipped*. This answers **"how much compute did the cache save?"** and it
is capped at `num_tokens − 1` and floored to block alignment, so an exact-duplicate prompt
never reports 100%.
(b) **`on_block_accessed`** counts (`memory/vllm/vllm/v1/core/block_pool.py:717`), which fire
inside `touch`, i.e. only when a freed-but-still-hashed block is resurrected by a hit. This
answers **"how often did the LRU tail pay for itself?"** — a *retention-policy* question, not a
compute-saving one.
"Blocks in use" is the denominator for neither, because `_maybe_evict_cached_block`
(`memory/vllm/vllm/v1/core/block_pool.py:679`) runs lazily at reallocation rather than at free
time: a block with `ref_cnt == 0` is not in use and is still matchable. "In use" and "available
for hits" are two disjoint-ish, overlapping populations, and dividing hits by either one
produces a ratio that answers no question anyone asked. The general lesson is the one from
Mooncake (`memory/mooncake/mooncake-store/src/master_service.cpp:6382`): three production
systems have three incompatible definitions of "the victim," so "hit rate" is not portable
across them and comparing published hit rates is comparing different quantities with the same
name.

---

## Sources

### Local measurements produced for this module

Environment, identical for all three runs: `torch 2.12.0a0+rocm7.13.0a20260313` (HIP 7.2.0),
AMD Radeon 8060S (gfx1151), native Windows 11 build 26200, Python 3.12.10, `C:\venvs\lab`,
2026-07-26. Workload: fp32 throughout (bf16 numerics are `untested` on this machine).
Micro-benchmark arms measure a single operation on a one-element scalar tensor with an empty
command queue, after ≥2,000 warm-up calls per path.

- **Run 1** — fixed arm order, micro-arms 500 reps × 7 trials; loop arm `M = 1024` square
  matmul, `N_ITERS = 200`, 7 trials.
- **Run 2** — randomised arm order, micro-arms 500 reps × 9 trials, loop arm 9 trials.
- **Run 3** — arms **interleaved** round-robin, 400 reps × 15 blocks, micro-arms only, plus a
  host-tensor control arm inside every block.

| Quantity | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|
| bare `torch.cuda.synchronize()` | 6.8 µs | 4.7 µs | 3.3 µs | `[M]` single-digit µs; level unstable |
| `.item()` on a device scalar | 51.9 µs | 135.7 µs | 57.5 µs | `[M]` **52–58 µs quiet**, 136 µs contended |
| `t.cpu()` on a device scalar | — | 122.1 µs | 63.0 µs | tracks `.item()` in both runs |
| `float(t)` on a device scalar | — | 120.1 µs | 58.7 µs | tracks `.item()` in both runs |
| `.item()` on a **host** scalar | — | 0.386 µs | 0.252 µs | `[M]` sub-microsecond, tight |
| **device ÷ host `.item()`, within block** | — | — | **median 236 (169–335)** | `[M]` the drift-invariant figure |

**What is promoted to `[M]` and what is not.** The device-versus-host ratio is `[M]`: it is
measured inside the same timed block, so any machine-level drift divides out, and it is tight
(stdev 7.1 µs on a 57.5 µs median in run 3). The claim that `.item()`, `.cpu()` and `float()`
cost the same thing is `[M]`: they land within ~10% of each other in both runs that measured
all three, which establishes the cost as the device-to-host round trip rather than anything
specific to `.item()`. The **absolute microsecond level is deliberately reported as a range**,
because run 2 disagreed with runs 1 and 3 by 2.6× and its whole loop arm was ~60% slower,
including a 24.8 ms outlier — evidence of machine contention rather than of a property of the
op. Every derived overhead figure in sections 2.4 and 3.5 is computed at 55 µs and is a
derivation, not a measurement.

**Not promoted, and labelled an anecdote:** the drain-interval curve. At `M = 1024` the loop
runs 1.3–2.2 ms/iteration and the per-probe cost is ~4% of that, inside the trial-to-trial
spread. Run 1's fixed order made the `never` arm look slower than `drain_every=200` (an
ordering artifact); run 2's randomised order produced a flatly non-monotone curve with per-arm
standard deviations up to 11.7 ms. **The correct report is "not resolvable at this shape,"**
and Exercise A says so.

**Retest instructions.** Rerun on a quiet machine with nothing else on the GPU, prefer the
interleaved design, and report the within-block device ÷ host ratio rather than the absolute
level. If the ratio moves outside 169–335, the finding has changed and this table should be
superseded rather than edited.

#### Oracle-diff noise floor (Exercise B, run as written)

Same environment. Shape: `torch.nn.TransformerEncoder`, 4 layers, 4 heads, `d_model = 256`,
`dim_feedforward = 1024`, `norm_first=True`, `dropout=0.0`, **randomly initialised**, eval
mode, causal mask, `T = 1024`, batch 1, vocab 4096 via an fp32 `Linear(256, 4096, bias=False)`
head. Metric: forward `D_KL(p_full ‖ p_pol)` in nats on the final position's next-token
distribution. Eviction = masking one column of the attention mask, sampled at 64 positions
`range(1, 1023, 16)`. Seeds 1337 / 4242 / 90210.

| Quantity | 1337 | 4242 | 90210 |
|---|---|---|---|
| `null["repeat"]` | 0.0 | 0.0 | 0.0 |
| `null["batch"]` (batch 8, identical rows) | 0.0 | 0.0 | 0.0 |
| `null["bf16"]` (bf16 weights + activations vs fp32) | 8.372e-06 | 6.552e-06 | 7.936e-06 |
| signal, median over 64 positions | 8.177e-07 | 7.038e-07 | 1.048e-06 |
| signal, max (always at position 1) | 1.330e-05 | 5.530e-06 | 9.862e-06 |
| signal, min | 1.865e-08 | **−1.669e-07** | 3.019e-07 |
| **SNR** = median signal ÷ largest null | **0.1** | **0.1** | **0.1** |

`[M]` **The SNR of 0.1 reproduced at all three seeds**, and the within-seed forward pass is
deterministic (repeat-null exactly zero), so the variation above is across seeds, not across
reruns. A negative KL at seed 4242 is mathematically impossible and is floating-point
cancellation at the noise floor.

**What this does and does not establish.** It establishes that on this machine, at this
shape, **bf16 numerical noise exceeds the divergence produced by evicting a single cache
entry by roughly an order of magnitude.** It does **not** establish anything about a trained
model: random initialisation gives near-uniform attention, so a one-of-1024 drop is close to
the smallest signal the instrument can be asked to see. Recommended `ASSUMPTIONS.md` row:
`oracle-diff-bf16-noise-floor`, status `supported (floor case only)`, with the trained-model
comparison as the test that would extend it. That row is not written by this module — a
curriculum module should not mutate a register while sibling work is in flight.

**A free reproduction.** Every run emitted
`UserWarning: Mem Efficient attention on Current AMD GPU is still experimental. Enable it with
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` on stderr, independently reproducing
`ASSUMPTIONS.md → sdpa-is-memory-efficient` — and confirming that the only honest signal for
which attention path ran is a stderr warning, not an API.

#### Adverse-selection simulation (Exercise C, run as written)

Pure numpy, no GPU. `T = 4096`, 20,000 Monte Carlo trials per cell, `numpy.default_rng(1337)`,
background masses `Exponential(1)`, needle mass `s ×` the background mean, budget 10% unless
stated. 14.8 s for the seven-row table.

| `s` | heavy_hitter | uniform | recency |
|---|---|---|---|
| 1.0 | 0.000 | 0.100 | 0.100 |
| 1.5 | 0.000 | 0.100 | 0.098 |
| 2.0 | 0.000 | 0.098 | 0.102 |
| **2.3** | **0.453** | 0.099 | 0.097 |
| 3.0 | 1.000 | 0.103 | 0.101 |
| 5.0 | 1.000 | 0.102 | 0.099 |
| 10.0 | 1.000 | 0.108 | 0.099 |

The two analytic anchors held (`uniform` and `recency` both ≈ 0.100 at every `s`), and the
heavy-hitter crossing landed at 0.453 against the closed-form prediction of ≈0.5 at
`s* = −ln(0.10) = 2.303`. **This is a simulation of a selection rule under a stipulated
background distribution, not a measurement of a model**, and it is labelled that way in the
exercise. The one thing it does establish rigorously is the closed form `s* = −ln(b)` and the
step-function shape of the pass criterion, both of which are properties of the rule and the
distribution rather than of any hardware.

### Local artifacts and prior measurements

- `ASSUMPTIONS.md` rows: `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s, single run per arm),
  `large-tensor-fault-32gib` (≥32 GiB single tensors hang at 0% CPU; 31 GiB clean),
  `sdpa-is-memory-efficient` (147.2 vs 6.6 bytes/T²; `flash_sdp_enabled()` returns True either
  way), `bf16-numerics-unproven` (`untested`), `gemm-throughput-below-reference`
  (20.9 TFLOP/s bf16 at 8192³), `torch-build`, `ablation-scale-sufficient`,
  `decode-intensity-varies-by-layer`, `single-device-only`.
- `notebook/uma-carveout-controls-fast-tier.md` — the fast-tier sweep, single run per arm.
- `research/memory/open-problems-ranked.md` §1 and §2 — the attribution arm and the
  eviction-versus-retention boundary; this module teaches §1.
- `research/notes/evaluation-landscape.md` — the evaluation half of the memory track; §1, §2,
  §5 and §6 are the direct source for sections 2.1, 2.6 and 2.7 here.
- `research/memory/memory-failure-register.md` — `attribution-gap-in-serving-results`,
  `eviction-destroys-long-range-recall`, `prefix-cache-correctness-and-leakage`.
- `research/synthesis.md` — the decision to build the instrument rather than another policy,
  and the riskiest assumption restated in §3.3 here.
- `curriculum/attention-variants-and-kv-cost.md` — the KV product and decode arithmetic
  intensity, assumed throughout and not re-derived.
- Code pointers: every `file:line` in section 5 was opened and the named construct confirmed
  on the named line on 2026-07-26, against the revisions in `research/reference/PROVENANCE.md`.

### arXiv `[C]`

Ids drawn from the mirror notes were resolved against the live arXiv API on 2026-07-26 by
those notes' authors; the two marked below were additionally verified by fetching the arXiv
listing while writing this module.

- `1911.02150` — *Fast Transformer Decoding: One Write-Head is All You Need* (2019). The
  origin of decode-is-bandwidth-bound.
- `2203.03466` — *Tensor Programs V* / muP (2022). Mandatory for two-scale riders.
- `2209.11895` — *In-context Learning and Induction Heads* (2022). Copying behaviour, §4.4.
- `2304.15004` — *Are Emergent Abilities of Large Language Models a Mirage?* (2023).
  Discontinuous metrics manufacture cliffs.
- `2306.00946` — *Exposing Attention Glitches with Flip-Flop Language Modeling* (2023).
- `2306.14048` — *H2O: Heavy-Hitter Oracle* (2023).
- `2307.03172` — *Lost in the Middle* (2023).
- `2309.17453` — *Efficient Streaming Language Models with Attention Sinks* (2023). Sinks and
  the renormalisation argument.
- `2312.04927` — *Zoology: Measuring and Improving Recall* (2023). 82% of a perplexity gap on
  associative-recall tokens.
- `2404.06654` — *RULER* (2024). Generator methodology and the effective-context threshold.
- `2404.14469` — *SnapKV* (2024). Observation-window selection.
- `2404.15574` — *Retrieval Head Mechanistically Explains Long-Context Factuality* (2024).
  Head-level localisation and the maskable-mechanism fault.
- `2406.02069` — *PyramidKV* (2024). Degenerates to SnapKV at aggressive ratios.
- `2406.10229` — *Quantifying Variance in Evaluation Benchmarks* (2024).
- `2410.02694` — *HELMET* (2024). NIAH does not predict downstream performance.
- `2410.05229` — *GSM-Symbolic* (2024). Perturbation twins as a method.
- `2411.00640` — *Adding Error Bars to Evals* (2024). Clustering, paired analysis, power.
- `2412.10319` — *SCBench* (2024). Rankings do not survive multi-turn cache reuse.
- `2502.05167` — *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (2025).
- `2510.00231` — *The Pitfalls of KV Cache Compression* (2025). Specific instructions dropped
  while LongBench held.
- `2510.13334` — *Taming the Fragility of KV Cache Eviction in LLM Inference* (2025).
  Worst-case aggregation changes the ranking.
- `2512.24503` — *Can Small Training Runs Reliably Guide Data Curation?* (Dec 2025).
- `2602.16837` — *A Structural Theory of Position Bias in Transformers* (Feb 2026, rev.
  May 2026). **Verified against the arXiv listing 2026-07-26.** Derives the U-shape from
  causal masking plus residual connections alone — i.e. before any training, which is what
  Exercise B's position curve reproduces at random init.
- `2603.10899` — *LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future
  without Generation* (Mar 2026). **Verified against the arXiv listing 2026-07-26.** Cited only
  as evidence that oracle-guided importance labelling is an active 2026 line; a search snippet
  suggested it reports full-cache-versus-compressed *logit* KL as an evaluation metric, and
  fetching the paper showed it does not — its KL is a training loss over attention importance
  scores. Recorded here because the near-miss is instructive: a secondary summary asserted a
  citation-worthy fact that the primary source does not contain.
- `2603.20397` — *KV Cache Optimization Strategies for Scalable and Efficient LLM Inference*
  (Mar 2026). No single method dominates.
- `2605.08234` — *When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic*
  (May 2026).
- `2605.28079` — *ATLAS: All-round Testing of Long-context Abilities across Scales* (May 2026).
  Rank instability across length regimes.
- `2606.09864` — *Alignment Collapse Under KV Cache Quantization* (Jun 2026). Perplexity held,
  refusals collapsed.
- `2606.26472` — *Epiphany-Aware KV Cache Eviction Without the Attention Matrix* (Jun 2026).
  **Verified against the arXiv listing 2026-07-26.** Scores tokens by representation change
  rather than attention weight; new and unreplicated, treated as a lead.
- `2606.29914` — *MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory
  Evaluation* (Jun 2026). One-variable-at-a-time control finding most reported memory gains
  were confounds.
- `2607.02574` — *From Tensor Buffer to Distributed Memory Hierarchy* (Jun 2026). Seven
  unmeasured KV quantities.
- `2607.08032` — *What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction*
  (Jul 2026).
- `2607.08284` — *Understanding Axes of Difficulty For Long Context Tasks Via
  PredicateLongBench* (Jul 2026). Worst-case difficulty axes.
- `2607.21475` — *Error Certificates for KV-Cache Eviction via Randomized Design* (Jul 2026).
  Deterministic top-k cannot estimate its own error; randomisation restores identifiability.
