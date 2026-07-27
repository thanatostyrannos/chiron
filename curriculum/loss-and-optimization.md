---
title: Loss and optimization — cross-entropy, perplexity, the logits allocation, AdamW, schedules, clipping
slug: loss-and-optimization
version: 1.0.0
date: 2026-07-26
track: A — Foundations
owner: curriculum-author
prereqs: tensors-and-autograd, transformer-forward-pass-by-hand, the-training-loop
---

# Loss and optimization

## What this module settles

**One:** next-token cross-entropy is one equation — `logsumexp(z) − z_y` — and every
number a training run reports (loss, perplexity, bits-per-byte, the gate target 1.4697)
is that equation with a different unit conversion applied, so you can convert between
them on paper and check anyone's claim. **Two:** the two largest memory objects in a
training step are the logits tensor (sized by *vocabulary*, not by parameters, and the
first thing on this machine to hit the `[M]` 32 GiB single-tensor fault) and the
optimizer state (exactly two extra full-precision copies of every parameter), and both
are computed, not measured — you size them before you allocate. **Three:** the learning
rate schedule and the gradient clip are the two controls that decide whether a run
produces a model or a number-shaped artifact, and both have arithmetic you can do in
your head that folklore usually replaces with a shrug.

---

## Theory in plain language

### What the loss is, and what it replaced

A language model is a classifier with one class per vocabulary entry. For every position
in the sequence it emits `V` real numbers — **logits** — and the training objective is:
make the logit of the token that actually came next as large as possible relative to all
the others. That is cross-entropy, and cross-entropy for a language model is not a
modelling choice so much as an accounting identity: minimizing it is *exactly* minimizing
the number of bits a perfect arithmetic coder would need to transmit the corpus given the
model. Loss in nats, divided by `ln 2`, is bits. Train a language model and you have
built a compressor; the loss is its compression ratio.

What it replaced is not the objective — maximum likelihood has been the objective since
Shannon — but the *workarounds* for computing it. When vocabularies were large relative
to hardware, the softmax denominator over `V` classes was the expensive part, and the
field built machinery to avoid it: hierarchical softmax (a tree over the vocabulary),
noise-contrastive estimation and sampled softmax (score the true token against `k`
random negatives instead of all `V`), and adaptive softmax (frequency-tiered heads).
All of those are approximations that trade gradient fidelity for memory. They were
displaced by dense softmax on GPUs, and then — as vocabularies climbed back to 100k+ and
sequence lengths to 100k+ — the same pressure came back, but the 2024–2026 answer is
different in kind. `[C]` Cut Cross-Entropy (arXiv 2411.09009, Nov 2024) and `[C]` Liger
Kernel's fused linear cross-entropy (arXiv 2410.10989, Oct 2024) compute the **exact**
loss without ever materializing the `T × V` logits tensor, by fusing the output
projection into the loss kernel and streaming the log-sum-exp. The old answer was
"approximate the loss to fit in memory." The new answer is "the loss was never the
problem; materializing the intermediate was." That distinction is the whole of §*Why the
logits tensor is the largest single allocation* below, and it is a distinction a storage
engineer will recognize instantly: this is write amplification at the last hop, fixed by
not writing.

### What the optimizer is

Given the gradient of the loss with respect to every parameter, the optimizer decides how
far to move. Four generations, each fixing a specific defect of the last:

- **SGD.** Step in the direction of the negative gradient, scaled by a learning rate.
  Zero extra state. Defect: gradients are noisy per-batch, and a single global step size
  is wrong for parameters whose gradients differ in scale by orders of magnitude — which,
  in a transformer, they do (an embedding row for a rare token versus a norm scale).
- **Momentum.** Keep an exponentially weighted moving average of the gradient and step
  along that. One extra copy of the parameters. Fixes the noise; does not fix the scale
  problem.
- **Adam.** Keep *two* EWMAs — of the gradient and of the squared gradient — and divide
  the first by the square root of the second. Two extra copies. The division makes each
  coordinate's step size roughly scale-invariant: a parameter with tiny but consistent
  gradients moves as far as one with large gradients. This is the change that made deep
  transformers trainable without per-layer LR hand-tuning.
- **AdamW.** Same as Adam except weight decay is applied to the weights directly rather
  than added into the gradient, so the decay does not get divided by the second-moment
  estimate. One line of code; a materially different regularizer.

**Systems bridge.** The optimizer is a stateful stream processor: one record per step,
per-key (per-parameter) state, an EWMA update per key, no cross-key interaction.
`m` and `v` are exactly the kind of per-key aggregate you would keep in a Flink or Kafka
Streams state store.

**Where it breaks, and this is the part that matters for DR.** In a stream processor,
state is *derivable from the log* — lose it, replay from the last offset, rebuild it.
Optimizer state is not. Replaying the same training data against the current weights does
not reconstruct `m` and `v`, because those moments were accumulated against *different*
weights at every past step; the input stream and the state are not independent. There is
no recompute path. This has a direct operational consequence: a checkpoint that saves
weights but not optimizer state is not a backup, it is a fork. Resuming from it produces a
different model than the uninterrupted run would have, and the divergence is invisible —
the loss curve looks fine. Treat optimizer state as the only truly non-reconstructible
data in the system, which is the inverse of how you would rank it by size.

A second break worth naming, since "two extra copies" invites the wrong intuition: those
copies are not replicas. Three copies of a parameter in an FSDP checkpoint would give you
two-fault tolerance. Here, `θ`, `m`, and `v` are three *different* quantities that happen
to have the same shape. Losing `v` is not a redundancy event; it is data loss.

### What the schedule and the clip are

The learning rate schedule is the step-size-over-time policy: ramp up (**warmup**), hold
or decay through the bulk of training, and decay to near zero at the end. The gradient
clip is a hard ceiling on the total size of one update.

**Systems bridge.** Warmup is TCP slow start: do not send at line rate before you know
what the path can absorb. Gradient clipping is a rate limiter — a token bucket with the
clip threshold as the bucket size.

**Where warmup breaks the analogy.** Slow start probes for a capacity that exists
independently of the probing; the pipe's bandwidth-delay product is a property of the
network, and measuring it does not change it. Warmup is not measuring a fixed capacity —
it is *manufacturing* the quantity it appears to be probing. Adam's second-moment estimate
`v` starts at zero and is meaningless for the first few dozen steps; warmup exists in
large part so that the denominator has enough samples to mean something by the time the
numerator is allowed to matter. The "capacity" is created by the ramp. §*Learning-rate
schedules* makes this exact with a number.

**Where clipping breaks the analogy, and this one has bitten people.** A rate limiter
throttles the offending flow and leaves the others alone, and the throttled work is
*deferred*, not destroyed. Global-norm gradient clipping computes **one** scalar over
every parameter tensor in the model and scales **all** of them by the same factor. One
exploding tensor throttles every other tensor in the model, because they shared a bus.
And the discarded magnitude never arrives later — the step is simply smaller than the
optimizer asked for, permanently. §*Gradient clipping* does the arithmetic; it is worse
than most people assume.

---

## The math that actually matters

### Cross-entropy for next-token prediction

For one sequence position, the model produces a logit vector `z ∈ ℝ^V`. The softmax turns
it into a probability distribution, and the loss is the negative log-probability of the
observed next token:

```
p_i = exp(z_i) / Σ_{j=1..V} exp(z_j)

L   = −log p_y  =  log( Σ_{j=1..V} exp(z_j) )  −  z_y
                   \_______ logsumexp(z) _____/
```

| Symbol | In words |
|---|---|
| `V` | vocabulary size — the number of distinct tokens the tokenizer can emit. `100352` on the reference model `[M]` |
| `z` | the **logits**: one unnormalized score per vocabulary entry, for this one position |
| `z_i` | the score the model assigned to vocabulary entry `i` |
| `y` | the index of the token that actually came next — the label |
| `z_y` | the score the model assigned to the correct answer |
| `p_i` | the probability the model assigns to entry `i` after normalization |
| `L` | the loss at this position, in **nats** (natural-log units) |
| `logsumexp(z)` | the log of the normalizing denominator — the "log partition function" |

The right-hand form is the one implementations use: `logsumexp(z) − z_y`. No softmax
tensor is ever needed.

**Two structural facts fall out of that equation, and both have systems consequences.**

*Shift invariance.* Add any constant `c` to every logit and `L` is unchanged:
`logsumexp(z+c) = logsumexp(z) + c`, and `z_y + c` grows by the same `c`. So the loss
constrains only `V−1` degrees of freedom per position; the overall magnitude of the logits
is unconstrained and free to drift. In practice it drifts upward, which is a numerics
hazard (overflow in bf16, and quantization headroom lost downstream). The standard fix is
**z-loss**: add `α · logsumexp(z)²` to the objective, pinning the free degree of freedom
near zero without changing any of the differences the loss cares about. `[C]` ST-MoE
(arXiv 2202.08906) is the canonical source; olmo-core computes it from the logsumexp the
CE already had to compute.

*The gradient has the same shape as the logits.* Differentiating `L` with respect to `z`:

```
∂L/∂z_i  =  p_i − 1[i = y]
```

In words: the gradient of the loss with respect to each logit is *the probability the
model assigned to that token*, minus one if that token was the right answer. This is the
cleanest gradient in deep learning and it is worth memorizing, but the operationally
important part is the shape: `∂L/∂z` is a dense `T × V` tensor, exactly as large as the
logits themselves. Backward through cross-entropy does not shrink anything. That is why
the peak allocation is a multiple of `T × V`, not `T × V` once.

### Perplexity, bits, and bits-per-byte

**Perplexity** is the exponential of the mean cross-entropy in nats:

```
PPL = exp( (1/N) Σ_{n=1..N} L_n )
```

where `N` is the number of scored positions and `L_n` is the loss at position `n`. Read it
as an **effective branching factor**: the number of equally-likely options the model is
hedging across per token. If the model were uniform over the whole vocabulary, `p_i = 1/V`
for all `i`, so `L = −log(1/V) = log V` and `PPL = V` exactly.

Worked, on our reference vocabulary:

```
uniform over V = 100352:  L = ln(100352) = 11.5164 nats = 16.61 bits, PPL = 100352
```

That is the ceiling — a model that has learned nothing. Every nat below it is real
information. Some conversions to keep in your head:

```
L = 3.0 nats  →  PPL = e^3.0 = 20.09     (hedging across ~20 tokens)
L = 2.0 nats  →  PPL = e^2.0 =  7.39     (hedging across ~7.4)
```

One nat is a factor of `e` in branching factor. This is why loss differences that look
tiny are not: a 0.05-nat improvement is a 5.1% reduction in effective branching factor
(`e^0.05 = 1.0513`), everywhere, on every token.

**Bits and bits-per-byte.** Divide nats by `ln 2 = 0.693147` to get bits. Perplexity is
per *token*, which makes it useless for comparing models with different tokenizers — a
tokenizer that emits fewer, larger tokens gets a worse per-token perplexity for the same
actual compression. The tokenizer-independent version is **bits-per-byte**:

```
BPB = (total nats over the corpus) / (ln 2 × total bytes of the raw text)
```

Worked on the nanoGPT gate recipe, whose target is a published number we will have to
reproduce `[C]` (`training/nanogpt/README.md:51`):

```
val loss 1.4697 nats/char
bits/char = 1.4697 / 0.693147 = 2.1204 bits
chars are 1 byte here, so BPB = 2.1204
PPL = e^1.4697 = 4.348   (out of a 65-symbol vocabulary)
compression vs 8-bit ASCII = 8 / 2.1204 = 3.77×
```

That agrees with the "~2.12 bits/char" in `CODE_MAP`, which is the point of doing it: the
gate number is checkable arithmetic, not a magic constant. Note also what `CODE_MAP` says
about it — `estimate_loss` is a Monte Carlo mean over 200 random batches
(`training/nanogpt/train.py:216`), so the bar is landing within roughly 0.01 of 1.4697,
not reproducing four decimals.

**And the honest caveat, which our own survey already carries.** Perplexity is dominated
by high-frequency local prediction. `[C]` Zoology (arXiv 2312.04927) attributes **82% of a
2.1-point Pile perplexity gap to tokens requiring associative recall** — so perplexity
does carry the memory signal, diluted roughly 1:5 against everything else. `[C]` Alignment
collapse under KV quantization (arXiv 2606.09864, Jun 2026) reports refusal-rate loss at
perplexity deltas small enough that PPL-only evaluation misses it entirely. For the
memory track specifically, `research/notes/evaluation-landscape.md` sets the standard:
report per-token loss **sliced to the answer span**, never the corpus mean. That is an
interface requirement on the loss function, not a reporting convention — see
§*Why it matters for Proteus*.

### Why the logits tensor is the largest single allocation

The logits are the only activation in the network whose width is `V` rather than `d`. For
the reference model, `V/d = 100352/3072 = 32.7`; for a 300M ablation model at `d = 768`
with the same tokenizer, `V/d = 130.7`. The last layer's output is over a hundred times
wider than every other tensor in the stack, and there is one of them per token in the
microbatch.

```
logits_bytes = T × V × b
```

| Symbol | In words |
|---|---|
| `T` | tokens in the microbatch — `batch_size × sequence_length`, all of them scored |
| `V` | vocabulary size |
| `b` | bytes per element (bf16 → 2, fp32 → 4) |

The arithmetic, at fp32 (which is what the loss actually runs in — see the code pointer):

| `T` | `V` | `T × V` elements | fp32 bytes |
|---|---|---|---|
| 16,384 | 50,257 | 823,410,688 | **3.07 GiB** |
| 65,536 | 50,257 | 3,293,642,752 | **12.27 GiB** |
| 16,384 | 100,352 | 1,644,167,168 | **6.13 GiB** |
| 131,072 | 100,352 | 13,153,337,344 | **49.00 GiB** |

But one copy is not what you pay. The standard (non-fused) path holds, concurrently:

```
bf16 logits from the output projection     T × V × 2
fp32 upcast inside the loss                T × V × 4      ← cross_entropy_loss.py:35
gradient w.r.t. the fp32 logits            T × V × 4      ← same shape, always
gradient w.r.t. the bf16 logits            T × V × 2
                                          ------------
                                           T × V × 12 bytes
```

So the honest coefficient is **~12 bytes per logit element**, not 4. At `T = 16,384` and
`V = 100,352`:

```
1,644,167,168 × 12 = 19,730,006,016 B = 18.375 GiB
```

Eighteen gigabytes, for a microbatch of sixteen thousand tokens, on a model whose
*parameters* in bf16 are 0.6 GB and whose *optimizer state* is 4.8 GB. The logits are
larger than the entire rest of the training step, and nothing about the model size caused
it.

**Against our measured hardware, this is the binding constraint, not a curiosity.**
`[M]` A 31 GiB buffer copies cleanly at 199.9 GB/s; a **32 GiB buffer hard-hangs at 0 CPU
with no error** and a 36 GiB buffer raises `hipErrorLaunchFailure`
(`ASSUMPTIONS.md: large-tensor-fault-32gib`, 2026-07-26). The single largest tensor in the
list above is the fp32 logits at `T × V × 4`. Solving for the token count that reaches
each threshold at `V = 100352`:

```
32 GiB / 4 B = 8,589,934,592 elements  →  T = 85,598 tokens   (the measured fault)
 8 GiB / 4 B = 2,147,483,648 elements  →  T = 21,399 tokens   (the config-validator guard)
```

`research/notes/pretraining-recipes.md` §9 already prescribes the guard —
`assert T_micro × V × 4 ≤ 8 GiB` in the config validator, and chunk the CE over the
sequence axis. What the arithmetic above adds is the *shape of the trap*: the obvious way
to exploit a `[M]` ≥62 GiB fast tier is to raise the microbatch, and raising the
microbatch is precisely what walks the logits tensor into a silent hang. The failure
presents as a stalled job at 0% GPU utilization, not as an OOM.

**One more consequence, specific to ablation scale.** Forward FLOPs for the body are
about `2 · N_nonemb · T`; the output projection adds `2 · T · d · V` on top. At
`T = 16,384`, `d = 768`, `V = 100,352`:

```
head:  2 × 16,384 × 768 × 100,352 = 2.53e12 FLOPs
body at N = 300M: 2 × 3.0e8 × 16,384 = 9.83e12 FLOPs   → head is 26% on top of the body
body at N =  20M: 2 × 2.0e7 × 16,384 = 6.55e11 FLOPs   → head is 3.85× the entire body
```

At the bottom of our ablation ladder with a frontier tokenizer, the output layer is
four times the compute of the model it sits on, and the dominant memory object besides.
This is the same conclusion `research/notes/transformer-state-of-the-art.md` reaches from
the parameter side (two untied `100352 × d` matrices are >50% of a 300M budget), arrived
at independently from the activation side. Vocabulary is not a detail to settle later.

### SGD → Adam → AdamW, with the state cost written out

Notation: `θ` is a parameter, `g` its gradient at the current step, `η` the learning rate,
`t` the step index (1-based), `λ` the weight decay coefficient, `ε` a small constant to
avoid division by zero.

```
SGD                 θ ← θ − η·g

Momentum            m ← μ·m + g
                    θ ← θ − η·m

Adam                m ← β₁·m + (1−β₁)·g          first moment: EWMA of the gradient
                    v ← β₂·v + (1−β₂)·g²         second moment: EWMA of the squared gradient
                    m̂ ← m / (1 − β₁ᵗ)            bias correction
                    v̂ ← v / (1 − β₂ᵗ)
                    θ ← θ − η · m̂ / (√v̂ + ε)

AdamW               θ ← θ − η·λ·θ                decoupled decay, applied to θ directly
                    (then the Adam update above, unchanged)
```

| Symbol | In words |
|---|---|
| `μ` | momentum coefficient, classically 0.9 — how much of the old average survives |
| `β₁` | decay rate of the gradient average. 0.9 means a ~10-step effective window |
| `β₂` | decay rate of the squared-gradient average. **0.95** in LLM practice, not the 0.999 library default — LLM gradients are noisy and a shorter window tracks better |
| `g²` | elementwise square, so `v` is a per-coordinate estimate of gradient magnitude |
| `√v̂` | per-coordinate RMS gradient — the denominator that makes the step scale-invariant |
| `ε` | floor on that denominator. Not a formality: `[C]` arXiv 2509.02046 shows published optimizer rankings *flip* when `ε` is tuned rather than defaulted |
| `λ` | weight decay strength, ~0.1 on matrices, 0 on norms, biases and (usually) embeddings |

**Why bias correction exists, as arithmetic.** `m` and `v` are initialized to zero, so
early on they are biased toward zero. At `t = 1` with `β₁ = 0.9`:

```
m₁ = 0.9·0 + 0.1·g = 0.1·g          — ten times too small
m̂₁ = 0.1·g / (1 − 0.9¹) = 0.1·g / 0.1 = g     — corrected
```

And with `β₂ = 0.95`, `v₁ = 0.05·g²`, so `v̂₁ = 0.05·g²/0.05 = g²` and `√v̂₁ = |g|`.
Substituting into the update:

```
θ ← θ − η · g / (|g| + ε)  ≈  θ − η · sign(g)
```

**Adam's very first step moves every single parameter by exactly the learning rate,
regardless of how large or small its gradient was.** That is not folklore, it is two lines
of algebra, and it is the real reason warmup exists. With nanoGPT's `learning_rate = 6e-4`
and an initialization standard deviation of 0.02, step one without warmup would change
every weight in the model by `6e-4 / 0.02 = 3%` of its typical magnitude, in a direction
determined by a single noisy minibatch. With a 2000-step linear warmup, step one uses
`6e-4 × 1/2001 = 3.0e-7`, a 0.0015% perturbation. Same optimizer, four orders of magnitude
difference in the damage one bad batch can do.

**Why AdamW's decoupling is not cosmetic.** The pre-2017 way to get weight decay was to
add `λθ` into the gradient before the moments. Then the decay term flows through the
`/√v̂` normalization, and the effective decay applied to coordinate `i` becomes:

```
η·λ·θ_i / (√v̂_i + ε)
```

— inversely proportional to that coordinate's RMS gradient. Coordinates with small
gradients get *more* decay; coordinates with large gradients get almost none. The
regularizer becomes a function of gradient history, which is not what anyone intended.
`[C]` Loshchilov & Hutter (arXiv 1711.05101) decouple it: apply `−η·λ·θ` directly, outside
the normalization. In olmo-core that is one line (`adamw.py:29`), and it is the entire
difference between the two algorithms.

**The memory arithmetic, per parameter.** This is the "two extra copies" claim, made
precise. Mixed-precision training with an fp32 master copy:

| Component | bytes/param | Why it exists |
|---|---|---|
| bf16 parameter (compute copy) | 2 | what the matmuls read |
| bf16 gradient | 2 | what backward writes |
| fp32 master parameter | 4 | bf16 has ~8 bits of mantissa; small updates would be lost to rounding |
| `m`, fp32 | 4 | **extra copy 1** |
| `v`, fp32 | 4 | **extra copy 2** |
| **total** | **16** | |

| Optimizer / precision | bytes/param | At 300M params |
|---|---|---|
| fp32 SGD | 8 | 2.4 GB |
| fp32 SGD + momentum | 12 | 3.6 GB |
| fp32 AdamW | 16 | 4.8 GB |
| bf16 mixed + fp32 master + fp32 moments | **16** | **4.8 GB (4.47 GiB)** |
| bf16 mixed + fp32 master + bf16 moments | 12 | 3.6 GB |
| `[C]` 8-bit moments (arXiv 2110.02861) | 10 | 3.0 GB |
| `[C]` Adafactor, factored `v`, no `m` (arXiv 1804.04235) | ~8 | ~2.4 GB |

Two things in that table are counterintuitive and worth stopping on. First, **fp32 AdamW
and bf16-mixed AdamW cost the same 16 bytes per parameter.** Mixed precision does not save
optimizer memory; it saves activation memory and bandwidth. People routinely expect the
former and are then confused by their own profiler. Second, Adafactor's trick is worth
understanding even if we never use it: instead of storing `v` elementwise for an `m × n`
matrix, store its row sums and column sums and reconstruct a rank-1 approximation. For
Laguna's dense MLP matrix at `3072 × 12288`:

```
elementwise v: 3072 × 12288 = 37,748,736 values
factored v:    3072 + 12288 =     15,360 values
reduction:                          2458×
```

**Where this lands for us.** `[M]` 4.8 GB of optimizer state at the very top of our
declared ablation box, against a `[M]` ≥62 GiB fast tier
(`notebook/uma-carveout-controls-fast-tier.md`). Optimizer state is ~7% of the fast tier.
**Optimizer memory is not our constraint, and most published advice about it is written
for people for whom it is.** The logits tensor is our constraint. Read the two sections
above in that order and the priority is obvious.

### Learning-rate schedules

**Linear warmup**, exactly as nanoGPT implements it (`train.py:234`):

```
η_t = η_peak · (t + 1) / (W + 1)        for t < W
```

`W` is the warmup length in steps; `t` is the current step. The `+1`s make step 0 nonzero.

**Cosine decay** (`train.py:239–242`):

```
τ    = (t − W) / (T_total − W)                    fraction of post-warmup training elapsed, 0→1
η_t  = η_min + ½·(1 + cos(π·τ))·(η_peak − η_min)
```

`T_total` is the total step count and `η_min` the floor (conventionally `η_peak/10`). The
coefficient `½(1 + cos πτ)` runs smoothly from 1 at `τ=0` to 0 at `τ=1`.

**Warmup-stable-decay (WSD)** — warmup, then hold `η_peak` flat, then decay linearly to
near zero over the last `f` fraction. olmo-core implements it in three branches
(`scheduler.py:208`), with `decay_fraction` defaulting to 0.1.

**The difference, as one number.** Integrate the learning rate over the run (take
`η_min = 0` for clarity). Cosine:

```
mean η / η_peak = ∫₀¹ ½(1 + cos πτ) dτ = ½·(1 + 0) = 0.50
```

WSD with a 10% decay tail, decaying linearly to zero:

```
mean η / η_peak = 0.9·1 + 0.1·½ = 0.95
```

**WSD delivers 1.9× the integrated learning rate of cosine over the same number of
steps.** That single ratio explains both observed behaviours: loss during WSD's stable
phase sits *higher* than the cosine curve (bigger steps, more bouncing across the valley),
and the final loss after the decay often lands *lower* (more distance covered along the
valley before settling). `[C]` The river-valley picture (arXiv 2410.05192, Oct 2024) is
the most-cited mechanistic account; `[C]` arXiv 2602.06797 (Feb 2026) derives power-decay
and WSD as optima under functional scaling laws, which is the closest thing to a
justification rather than a picture; `[C]` arXiv 2601.09000 (Jan 2026) shows WSD's
behaviour is not transformer-specific, which weakens architecture-flavoured explanations.

**The structural argument, which matters more here than the loss argument.** A cosine
schedule is a function of `T_total`. That makes a cosine run a **fixed-size job**: stop
early and you have a model trained under the wrong schedule; extend it and you must
restart. WSD's stable phase has no `T_total` in it, so any checkpoint on the trunk is a
legitimate branch point — run the short decay from it and you have a finished model at
that budget. The systems reading is a base image with layered commits.

Cost arithmetic for a budget ladder with `K` budget points, in units of one full run:

```
cosine:  K full runs                              = K
WSD:     1 trunk + K decays at 10% each           = 1 + 0.1K
K = 6:   6.0 vs 1.6                               → 3.75× cheaper
```

One clarification, because it is easy to over-claim and
`research/notes/pretraining-recipes.md` §2 states this for "an IsoFLOP ladder or an N-arm
architecture sweep": **the saving is along the budget axis.** A different architecture is a
different trunk — `proteus-swa-4to1` and `proteus-dense` cannot share one. What they can
each do is amortize their own budget ladder over one trunk apiece. That is still the
difference between a sweep that fits our wall-clock budget and one that does not, but the
factor is per-arm, not across arms.

Contested, and worth flagging: `[C]` arXiv 2603.16127 (Mar 2026) reports that pretraining
*without* LR decay produces a better base for supervised fine-tuning. If a checkpoint's
destination is more training rather than evaluation, the decay may be counterproductive.
Unresolved.

### Gradient clipping

**Global-norm clipping**, the near-universal default:

```
G = sqrt( Σ_p ‖g_p‖²  )        over every parameter tensor p in the model
if G > c:  g_p ← g_p · (c / G)  for every p
```

`c` is the clip threshold, conventionally 1.0. `‖g_p‖` is the Frobenius norm of one
tensor's gradient. `G` is one scalar for the entire model.

**The property nobody states, in arithmetic.** Suppose 99 parameter tensors each have
gradient norm 1.0, and one tensor — say one attention head's query projection, mid-spike —
has norm 100.

```
G     = sqrt(99 × 1² + 100²) = sqrt(10,099) = 100.494
scale = c / G = 1.0 / 100.494 = 0.00995
```

Every healthy tensor's update is multiplied by 1/100. The step still happens; the loss
still decreases slightly; nothing is logged as an error. One misbehaving tensor has
silently cancelled the step for the entire model, and the *only* observable is the
pre-clip norm. This is why the telemetry schema in `research/notes/pretraining-recipes.md`
§9 carries `grad_norm_preclip`, `grad_norm_postclip` and `clip_fraction` as separate
fields — post-clip norm alone is uninformative, because when clipping is active it is
always exactly `c`.

**Second-order effect: clipping silently overrides your schedule.** When the clip is
active, the update magnitude is `η · c / G`, so the effective learning rate is inversely
proportional to the gradient norm. During a stretch of clipped steps you are not running
the schedule you configured; you are running one dictated by the gradient norms. A run
that clips on 40% of steps has an effective LR curve you never chose and did not log.

**Why clip at all — the theory.** `[C]` Pascanu et al. (arXiv 1211.5063, 2012) introduced
it for exploding gradients in RNNs, as a heuristic. `[C]` Zhang et al. (arXiv 1905.11881,
2019) gave it a justification: under `(L₀, L₁)`-smoothness — where the local smoothness
constant grows with the gradient norm rather than being globally bounded — clipped
gradient descent provably converges faster than any fixed-step-size method. Transformer
loss surfaces empirically look like that. So clipping is not only a safety rail; in the
regime we actually train in, it is part of the algorithm.

**The 2025–2026 alternatives, none settled.** `[C]` AdaGC (arXiv 2502.11034) clips
per-tensor rather than globally, precisely to avoid the throttle-everyone problem above.
`[C]` ZClip (arXiv 2504.02507) replaces the fixed threshold with a z-score against the
running distribution of gradient norms, on the grounds that a static threshold and a
drifting distribution inevitably diverge. `[C]` arXiv 2510.01578 (Oct 2025) argues for
smooth per-layer gradient *shaping* instead of a hard threshold. And the pragmatic
production answer is not clipping at all but **skipping**: reject the whole step when loss
or gradient norm exceeds a running threshold. olmo-core's `SkipStepOptimizer` uses a
128-step rolling window and a 6-sigma factor (`skip_step_optimizer.py:37`), and implements
the branch as a multiply by `0.0` or `1.0` rather than an `if` — because reading the
decision on the host would force a host-device sync and stall the pipeline
(`skip_step_optimizer.py:86`). That is a control-plane/data-plane separation you will
recognize, executed for an unusual reason.

---

## Why it matters for Proteus

**The config surface is the experimental surface** (house rule), so every knob named above
is a field, and this module is where their defaults get justified rather than inherited.

| Config field | Default and why |
|---|---|
| `loss.implementation` | `default` \| `fused_linear` \| `chunked`. Not a performance flag — it decides whether the largest tensor in the step exists at all |
| `loss.max_logit_bytes` | Validator assertion, `T × V × 4 ≤ 8 GiB`. `[M]` derived from the 32 GiB single-tensor fault with a 4× margin |
| `loss.z_loss_multiplier` | `1e-4` or off. Pins the shift-invariant degree of freedom; matters more if we add MoE routing |
| `loss.reduction` + `loss.token_mask` | **Must support `none` plus a mask.** `research/notes/evaluation-landscape.md` requires per-token loss sliced to the answer span for memory work; a loss function that only returns a scalar cannot produce the metric the memory track needs |
| `optim.betas` | `(0.9, 0.95)` — not the library default |
| `optim.eps` | Tunable, not defaulted `[C]` (arXiv 2509.02046) |
| `optim.weight_decay` | `0.1`, matrices only |
| `optim.state_dtype` | fp32 default; bf16 available. olmo-core exposes exactly this (`adamw.py:168`) |
| `schedule.type` | `wsd` default, `cosine` available for replication of published recipes |
| `schedule.warmup_fraction` / `schedule.decay_fraction` | `0.01–0.02` / `0.1` |
| `optim.max_grad_norm` | `1.0`, with pre- and post-clip norms logged separately |
| `optim.skip_step.*` | rolling window and sigma factor |

**Three couplings specific to this lab.**

*The logits tensor and the KV cache draw on the same pool.* `research/memory/kv-cache-mechanics.md`
sizes long-context experiments against the `[M]` ≥62 GiB fast tier. So does the loss. On a
unified-memory machine there is no separate activation budget to hide in, and the logits
tensor reaches the `[M]` 32 GiB fault before the KV cache does. Any Mnemosyne experiment
that trains rather than only decodes has to budget both.

*The loss function is the memory track's instrument.* Every eviction-policy or
compression experiment is ultimately scored by a delta in per-token loss on specific
tokens. If `loss.reduction="none"` with a mask is not in the interface from the start, the
attribution experiments this lab claims as its comparative advantage cannot be run — you
will have an outcome number and no mechanism, which is the exact failure the house
standards name.

*Perplexity alone will not settle any memory question.* `[C]` The 82%/2.1-point Zoology
result (arXiv 2312.04927) says corpus-mean perplexity carries the recall signal at roughly
1:5 dilution. A 5× dilution against a small effect at 20M–50M parameters and three seeds
is a null result waiting to happen. Plan for sliced metrics, and read
`research/notes/evaluation-landscape.md` before designing any arm.

---

## Read the code

Paths are relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Read in the order given — nanoGPT first because the
whole story fits on two screens, olmo-core second because it is what the story looks like
after it has met production.

### The whole loop, small enough to hold in your head

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:186` | `logits = self.lm_head(x)` — the allocation. One line creates the largest tensor in the step. Note it is unconditional in the training branch |
| `training/nanogpt/model.py:187` | `F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ...)` — the entire loss. The `view(-1, ...)` is the `[B, S, V] → [T, V]` flatten that makes `T` the number that matters |
| `training/nanogpt/model.py:190` | The inference branch: `self.lm_head(x[:, [-1], :])` — only the last position. Same tensor, `T=1` instead of `T=16384`. This one contrast explains why serving and training have completely different memory profiles at the head |
| `training/nanogpt/model.py:263` | `configure_optimizers` — the whole optimizer setup in 25 lines |
| `training/nanogpt/model.py:270` | `decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]` — the "which parameters get weight decay" rule, stated as a dimensionality test. Every matmul weight decays; every norm scale and bias does not. Three lines, and it is the convention every open recipe uses |
| `training/nanogpt/model.py:282` | `use_fused = fused_available and device_type == 'cuda'`. On ROCm, torch reports `'cuda'` for HIP, so the fused path **is** selected on gfx1151. `[A]` medium confidence that the fused AdamW kernel is correct there; `bf16-numerics-unproven` is open. Cheapest test: one step, fused vs unfused, compare parameters bit-exactly |
| `training/nanogpt/train.py:231` | `get_lr` — cosine-with-warmup in twelve lines, no scheduler object. Lines `:234` (warmup) and `:241` (the cosine coefficient) are the two formulas from §*Learning-rate schedules*, verbatim |
| `training/nanogpt/train.py:301` | `loss = loss / gradient_accumulation_steps` — mean-of-means accumulation. Compare with olmo-core, which divides by the whole-batch token count *before* backward; `CODE_MAP` explains why the difference is not cosmetic when microbatches are ragged |
| `training/nanogpt/train.py:307` | `if grad_clip != 0.0:` then `:309` `clip_grad_norm_(model.parameters(), grad_clip)` — note the ordering: after all micro-batches have accumulated, before the step. Clipping a partial accumulation would clip the wrong quantity |

### The same ideas, production-hardened

| Where | What to look at, and why |
|---|---|
| `training/olmo-core/src/olmo_core/nn/functional/cross_entropy_loss.py:35` | `logits = logits.float()` — **the single most expensive line in this module.** An unconditional fp32 upcast of the largest tensor in the model, doubling it. It is correct (bf16 log-sum-exp over 100k terms is not trustworthy) and it is why the coefficient in §*Why the logits tensor* is 12 bytes per element and not 4 |
| `training/olmo-core/src/olmo_core/nn/functional/cross_entropy_loss.py:41` | `z_squared = logits.logsumexp(-1).pow(2)` — z-loss computed from the log-partition function the CE already needed. Read this next to the shift-invariance argument above; the code makes it concrete that z-loss constrains the one degree of freedom CE ignores |
| `training/olmo-core/src/olmo_core/nn/lm_head.py:268` | `logits = None` — in the `fused_linear` branch the logits are *never materialized*. This is the Cut-Cross-Entropy idea in the codebase, and the line to point at when someone says the memory cost is inherent |
| `training/olmo-core/src/olmo_core/nn/lm_head.py:269` | The fused call: the output projection weight goes *into* the loss function rather than being applied before it |
| `training/olmo-core/src/olmo_core/nn/lm_head.py:278` | `accum_dtype=torch.float32,  # <Liger-Kernel issue link>` — evidence that the fused path was once numerically wrong and needed an explicit fp32 accumulator. Relevant to us: it is a Triton kernel of uncertain gfx1151 status, and `bf16-numerics-unproven` is open |
| `training/olmo-core/src/olmo_core/optim/adamw.py:29` | `p.mul_(1 - step_factor * (lr * weight_decay))` — decoupled weight decay, the entire AdamW-vs-Adam difference, one line, applied to `p` and not to `grad` |
| `training/olmo-core/src/olmo_core/optim/adamw.py:32` | `exp_avg.lerp_(grad, ...)` then `:33–34` the second moment — the two EWMAs from the math section, in-place |
| `training/olmo-core/src/olmo_core/optim/adamw.py:41` | `denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)` — note `eps` is added *after* the bias correction, which is not what the paper's pseudocode literally says and changes the early-step behaviour slightly. Worth noticing that implementations differ here |
| `training/olmo-core/src/olmo_core/optim/adamw.py:43` | `update = -step_size * torch.div(exp_avg, denom)` — the step |
| `training/olmo-core/src/olmo_core/optim/adamw.py:168` and `:169` | `state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)` / `state["exp_avg_sq"] = ...` — **the two extra copies, allocated.** The `dtype=self.dtype` is the memory knob from the table in §*SGD → Adam → AdamW* |
| `training/olmo-core/src/olmo_core/optim/scheduler.py:170` | `decay_fraction: Optional[float] = 0.1` — the WSD default |
| `training/olmo-core/src/olmo_core/optim/scheduler.py:208` | `WSD.get_lr` — three branches (warmup / flat / decay) and nothing else. Compare against nanoGPT's cosine and notice what is *absent*: no dependence on `t_max` in the middle branch. That absence is the trunk-and-branch property |
| `training/olmo-core/src/olmo_core/optim/scheduler.py:227` | `_linear_decay(initial_lr, t_max - current, decay, self.decay_min_lr)` — the tail |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:514` | `optim_step` — the canonical ordering: clip (and record the norm), then set LR per param group, then `optim.step()`. Note the LR is applied per group, which is where muP's per-tensor learning rates would live |
| `training/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py:611` | `_clip_grad_norm` — split into `:622` `get_total_norm` and `:637` `clip_grads_with_norm_`, deliberately, so the norm can be logged and reduced across ranks before it is applied. The single-call convenience version is at `train/train_module/train_module.py:393`; compare them |
| `training/olmo-core/src/olmo_core/optim/skip_step_optimizer.py:37` | `rolling_interval_length: int = 128, sigma_factor: int = 6` — the spike detector's whole configuration |
| `training/olmo-core/src/olmo_core/optim/skip_step_optimizer.py:86` | `get_step_factor` — returns a device tensor of `1.0` or `0.0`, multiplied into the update, so the skip decision never crosses to the host. The comment says why explicitly |

---

## Exercises

All three assume `. .\scripts\activate-lab.ps1` first (native Windows, gfx1151, one GPU;
the script sets `HIPBLASLT_TENSILE_LIBPATH` and `TORCH_BLAS_PREFER_HIPBLASLT`). Put
scratch scripts under `notebook/`; per house rules they are exempt from TDD only while
they stay one-off and reproducible from committed config.

**Hard safety constraint for all three.** `[M]` A single 32 GiB tensor hard-hangs this
machine at 0% CPU with no error and requires a force-kill
(`ASSUMPTIONS.md: large-tensor-fault-32gib`). Cap any single allocation at **24 GiB**.
Separately, `[M]` `ENVIRONMENT.md` reports **32 GB of system RAM** after the 96 GB UMA
carve-out — so the CPU fallbacks must stay small, and 8 GiB is a sane host-side cap.

### Measure the bytes-per-logit coefficient, and find your own guard threshold

**What you build.** A script that, for a sweep of microbatch token counts `T` at a fixed
`V`, runs a forward-and-backward through `F.cross_entropy` on a synthetic logits tensor
and records `torch.cuda.max_memory_allocated()`. Then divide peak bytes by `T × V` and
plot the coefficient.

```python
import torch, torch.nn.functional as F, json
V, dev = 100352, "cuda"          # CPU fallback: dev="cpu", and stop the sweep at T=8192
rows = []
for T in [1024, 2048, 4096, 8192, 16384, 32768]:
    if T * V * 4 > 24 * 2**30:   # the 32 GiB fault; do not go near it
        break
    torch.cuda.reset_peak_memory_stats()
    z = torch.randn(T, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(0, V, (T,), device=dev)
    F.cross_entropy(z.float(), y).backward()
    peak = torch.cuda.max_memory_allocated()
    rows.append({"T": T, "elems": T * V, "peak_bytes": peak,
                 "bytes_per_elem": peak / (T * V)})
    del z, y; torch.cuda.empty_cache()
print(json.dumps(rows, indent=2))
```

**What you should get.** A `bytes_per_elem` that converges to roughly **12** as `T` grows
(2 for the bf16 logits + 4 for the fp32 upcast + 4 for the fp32 gradient + 2 for the bf16
gradient), with the small-`T` rows inflated by fixed overhead. If you get something
materially different, the interesting question is *which* of the four terms your PyTorch
build actually keeps alive — `torch.cuda.memory_snapshot()` will tell you.

**Then do the useful half.** Repeat with the loss chunked over the token axis
(`for chunk in z.split(4096): loss += F.cross_entropy(chunk.float(), y_chunk, reduction="sum")`,
then divide by `T` at the end) and confirm the peak flattens instead of growing linearly.
Report the largest `T` your machine sustains at each setting, and the `T` at which
`T × V × 4` crosses 8 GiB. That last number is the config-validator constant this lab
should actually use, measured rather than assumed.

**Runtime.** `[A]` 10–20 minutes GPU including the chunked variant. CPU fallback: same
script with `dev="cpu"`, `V=32000`, sweep stopping at `T=8192`, and peak measured with
`tracemalloc` or `psutil.Process().memory_info().rss`; `[A]` ~5 minutes.

**Why it is worth your evening.** This produces the first `[M]` number this lab has for
the loss-side memory ceiling, and `ASSUMPTIONS.md` currently has no row for it. If the
coefficient is 12, propose the row. If it is not, you have found something about this
PyTorch build that matters more.

### Weigh the optimizer, and catch Adam's first step in the act

**What you build.** A script that instantiates a nanoGPT-scale model, then measures
optimizer state exactly — not by profiling, but by summing the state tensors, which works
identically on CPU and GPU:

```python
def optimizer_state_bytes(opt):
    return sum(t.numel() * t.element_size()
               for s in opt.state.values() for t in s.values()
               if torch.is_tensor(t))

n_params = sum(p.numel() for p in model.parameters())
# after exactly one step:
print(name, optimizer_state_bytes(opt) / n_params, "bytes/param")
```

Run it for `SGD`, `SGD(momentum=0.9)`, `AdamW`, and `AdamW` with bf16 state, and check the
four numbers against the table in §*SGD → Adam → AdamW*: expect approximately **0, 4, 8,
4** extra bytes per parameter for the optimizer state itself (the parameter and gradient
copies are counted separately — verify that too). On GPU, cross-check against
`torch.cuda.max_memory_allocated()` and reconcile any discrepancy; the discrepancy is
usually the allocator's block rounding and is worth seeing once.

**Then the second half, which is the real point.** Verify the algebra from
§*SGD → Adam → AdamW*: set all gradients to a known tensor `g`, take exactly one AdamW step
with `lr=1e-3`, `weight_decay=0`, and measure `(θ_after − θ_before)`. It should equal
`−1e-3 × sign(g)` to within `ε`, **regardless of the magnitude of `g`**. Scale `g` by
`1000×` and confirm the step does not change. That is the fact that justifies warmup, and
seeing it on your own hardware is worth more than reading it here.

**Runtime.** `[A]` 10 minutes GPU, 10 minutes CPU. The CPU fallback is the same script with
`device="cpu"` and no changes — the state-summing measurement is device-agnostic by
construction.

### WSD versus cosine on the gate recipe, with the LR integral checked

**What you build.** Two runs of nanoGPT's `train_shakespeare_char` config (6 layers,
6 heads, 384 channels, `block_size` 256, 5000 iters, `warmup_iters` 100 — read the config
at `training/nanogpt/config/train_shakespeare_char.py`), matched seed and iteration count,
differing only in the schedule.

The stock schedule is cosine. `get_lr` is defined at `train.py:231`, *after* the
configurator runs at `train.py:77`, so a config file cannot override it — you must edit
`train.py`. Replace the body of `get_lr` with WSD:

```python
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    decay_start = lr_decay_iters - int(0.1 * lr_decay_iters)   # 10% decay tail
    if it >= lr_decay_iters:
        return min_lr
    if it < decay_start:
        return learning_rate                                    # the stable phase
    frac = (lr_decay_iters - it) / (lr_decay_iters - decay_start)
    return min_lr + frac * (learning_rate - min_lr)
```

Keep the diff; it is the experiment's design freeze. Log `lr` every step in both runs.

**What you produce.** Three checkable numbers and one plot.

1. **The integrated LR ratio.** Sum the logged `lr` over both runs and divide. Predicted
   from §*Learning-rate schedules*: with `min_lr = learning_rate/10` as this config sets,
   cosine's mean is `0.55 × η_peak` and WSD's is about `0.95 × η_peak`, so expect a ratio
   near **1.7**. If your measured ratio disagrees, your patch is wrong — check it before
   you interpret any loss.
2. **The crossover.** Plot both validation curves. The prediction is that WSD's trunk sits
   *above* cosine for most of the run and crosses below it during the decay tail. If it
   never crosses, that is a real finding at this scale and should be written up as one.
3. **The gate number in three units.** From the cosine run's best validation loss,
   compute perplexity (`exp(L)`), bits per character (`L / ln 2`), and the compression
   ratio against 8-bit characters. Check against `1.4697 nats → 4.348 PPL → 2.1204
   bits/char → 3.77×`. Landing within ~0.01 nats of 1.4697 is the bar
   (`training/nanogpt/README.md:51`); four decimals is not, because `estimate_loss` is a
   200-batch Monte Carlo mean.

**Runtime, honestly.** The published GPU figure is ~3 minutes on an A100. `[A]` On the
8060S, expect **30–90 minutes per run**, low confidence — nobody has measured this recipe
on this machine, and doing so is literally open question #1 in
`research/notes/pretraining-recipes.md`. Record tokens/s while you are there; that single
number replaces the `[A]` 6 TFLOP/s assumption the entire wall-clock plan rests on. Set
`compile = False` on the first attempt; `[A]` torch.compile on gfx1151 is unproven and a
compile failure would be indistinguishable from a schedule bug.

**CPU fallback.** nanoGPT's published CPU configuration — 4 layers, 128 channels,
`block_size` 64, 2000 iterations, target validation loss **1.88**
(`training/nanogpt/README.md:85`) — with the same two schedules. `[A]` 15–40 minutes per
run. The crossover prediction is the same; the absolute numbers are not comparable across
the two configurations, which is itself the lesson about matched budgets.

---

## Self-check

1. A model reports validation loss 2.35 nats on a 100,352-token vocabulary. What is its
   perplexity, what is that in bits per token, and how far is it from the
   learned-nothing baseline?

2. You are training a 50M-parameter model with `V = 100352`, `batch_size = 8`,
   `sequence_length = 4096`, bf16 with an fp32 master copy and fp32 AdamW moments.
   Rank these four by size: parameters, gradients, optimizer state, logits (including the
   logits' own gradient). Give the numbers.

3. Two runs are identical except one uses Adam with `weight_decay=0.01` passed as an L2
   term added into the gradient, and the other uses AdamW with the same coefficient.
   Which parameters end up more strongly decayed in each case, and why?

4. Your run logs `grad_norm_postclip = 1.0` on 100% of steps and the loss is decreasing.
   Is the training healthy? What single additional metric would tell you, and what would a
   bad value look like?

5. You want to compare six token budgets for the same architecture. Under a cosine
   schedule this costs six full runs. Under WSD it costs one trunk plus six short decays.
   Now you want to compare `proteus-dense` against `proteus-swa-4to1` across those same
   six budgets. What does WSD cost you now, and what is the saving?

6. Why does the very first Adam step move every parameter by approximately the learning
   rate regardless of gradient magnitude, and what does that imply about the *minimum*
   sensible warmup length?

---

## What is still unsolved here

Not "areas for future work" — places where the map genuinely ends, drawn from the survey
notes and from our own register.

**Whether AdamW is necessary at all, or just a well-tuned proxy for clipping.** `[C]`
"Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective Learning
Rates" (arXiv 2605.17787, May 2026) reports that simple clipping mechanisms which stabilize
SGD at large learning rates recover most of Adam's advantage — the validation-loss gap on a
1B LLaMA-class model shrinking from >50% to ~3.5%. If that holds, the standard story
("adaptive per-coordinate scaling is what makes transformers trainable") is partly a story
about gradient clipping wearing a different hat, and the 16-bytes-per-parameter tax is
partly optional. Too new to be settled; directly testable at our scale.

**Whether any optimizer beats AdamW at our scale, and whether the question is even
well-posed.** `research/notes/pretraining-recipes.md` §1 lays out the live disagreement:
`[C]` arXiv 2509.02046 finds matrix-preconditioned optimizers' margin shrinking with scale
(1.4× at 0.1B → 1.1× at 1.2B) and attributes published wins to weak AdamW baselines and
untuned `ε`; `[C]` arXiv 2607.20548 finds SOAP and Muon consistently ahead at
multi-billion scale and large batch. `[C]` arXiv 2602.07712 argues the comparison
methodology itself is ill-conditioned. Contested; do not pick a side in a config default.

**Why WSD works.** The river-valley account `[C]` (arXiv 2410.05192) is a picture that
fits, not a derivation. `[C]` arXiv 2602.06797 derives WSD-like schedules as optima under
functional scaling laws, which is progress; `[C]` arXiv 2601.09000 shows the phenomenon is
not transformer-specific, which rules out architecture-flavoured explanations but does not
supply a replacement. And `[C]` arXiv 2603.16127 argues the decay should sometimes be
skipped entirely. We use WSD because it is operationally superior (branchable) and
empirically fine, which is a good reason and not the same as understanding it.

**What the right clipping primitive is.** Global-norm clipping has the throttle-everyone
pathology demonstrated in §*Gradient clipping*, and the field's responses — per-tensor
`[C]` (arXiv 2502.11034), z-score `[C]` (arXiv 2504.02507), functional shaping `[C]`
(arXiv 2510.01578), skip-step — have not converged. Worse for our purposes: almost nobody
reports `clip_fraction`, so the published comparisons cannot be re-analyzed. This is a
measurement gap, not just a design gap, and it is cheap to close locally.

**Whether the fused/cut cross-entropy path is numerically equivalent on our hardware.**
`[C]` Cut Cross-Entropy (arXiv 2411.09009) and Liger's fused linear CE (arXiv 2410.10989)
claim exactness, and the fp32-accumulator workaround visible at `nn/lm_head.py:278` is
evidence that exactness needed defending at least once. On gfx1151, with
`ASSUMPTIONS.md: bf16-numerics-unproven` still open and Triton kernel support uncertain,
we have no basis to assume it. This blocks the cheapest fix to our largest allocation.

**Perplexity's relationship to anything we care about.** `[C]` Zoology's 82%-of-a-2.1-point
attribution (arXiv 2312.04927) is the strongest available evidence that perplexity carries
recall signal, and `research/notes/evaluation-landscape.md` open question #5 notes that
*no source demonstrates an MQAR capacity curve quantitatively predicting a downstream
curve*. So we know perplexity is diluted; we do not know the dilution factor for the
specific interventions Mnemosyne will make, and there is no published calibration to
borrow.

**And locally: none of this is measured.** The Hardware Validation Gate has not run.
`bf16-numerics-unproven` is open, and cross-entropy's log-sum-exp over 100k terms plus
AdamW's `√v` are both on the list of ops it covers. No loss number from this machine is
evidence yet.

---

## Answers

**1.** `PPL = e^2.35 = 10.49` — the model is effectively hedging across about ten and a
half tokens per position. In bits: `2.35 / 0.693147 = 3.390` bits per token. The
learned-nothing baseline is `ln(100352) = 11.5164` nats (16.61 bits, PPL 100,352), so the
model has removed `11.5164 − 2.35 = 9.17` nats, or `16.61 − 3.39 = 13.22` bits, per token.
Framed as compression: it has cut the branching factor by a factor of `100352 / 10.49 =
9566`.

**2.** `T = 8 × 4096 = 32,768` tokens.

```
parameters:      5.0e7 × 2 B (bf16)                    = 100 MB
                 + 5.0e7 × 4 B (fp32 master)           = 200 MB
gradients:       5.0e7 × 2 B                           = 100 MB
optimizer state: 5.0e7 × 8 B (m and v, fp32)           = 400 MB
logits:          32,768 × 100,352 × 12 B               = 39.5 GB (36.7 GiB)
```

The logits are roughly **49× everything else combined** (39.5 GB against 0.8 GB). The
configuration is legal — the largest *single* tensor is the fp32 logits at
`32,768 × 100,352 × 4 = 13.2 GB (12.25 GiB)`, well under the `[M]` 32 GiB fault, and the
36.7 GiB total fits inside the `[M]` ≥62 GiB fast tier. But it leaves under half the tier
for activations, and the trap is one doubling away: raise `sequence_length` to 8192 and
the fp32 logits alone become **24.5 GiB**, close enough to the measured fault that the
next increment hangs the machine silently. Chunk the loss, or assert the guard.

**3.** With L2-in-the-gradient, the decay term `λθ` is added to `g` before the moments,
so it passes through the `/(√v̂ + ε)` normalization; the effective decay on coordinate `i`
is `η·λ·θ_i / (√v̂_i + ε)`, which is *larger* for parameters with small RMS gradients.
Rarely-updated parameters get decayed hardest — the opposite of the intent. AdamW applies
`−η·λ·θ` directly (`adamw.py:29`), so every parameter is decayed by the same *relative*
amount `η·λ` regardless of its gradient history. The two are only equivalent when `√v̂` is
uniform across coordinates, which it never is.

**4.** You cannot tell, and the fact that the loss is decreasing is not reassurance.
`grad_norm_postclip = 1.0` on every step means the clip is active on every step, so the
post-clip norm is a constant by construction and carries no information. The metric you
need is **`grad_norm_preclip`**. If it is, say, 1.5, you are clipping mildly and the
schedule is roughly the one you configured. If it is 100, then per §*Gradient clipping*
your effective learning rate is `η/100` and you are running a schedule you never chose; if
it is drifting upward over time you are watching a slow instability that clipping is
masking rather than fixing. `clip_fraction` is the companion metric — it should be a small
minority of steps, not all of them.

**5.** Two trunks, twelve decays: `2 × (1 + 6 × 0.1) = 3.2` full-run-equivalents, against
`2 × 6 = 12` for cosine — a **3.75× saving**, unchanged per arm. The saving does *not*
compound across arms, because a different architecture is a different trunk: `proteus-dense`
and `proteus-swa-4to1` cannot share one. WSD amortizes the **budget** axis, never the
architecture axis. This is the clarification flagged against
`research/notes/pretraining-recipes.md` §2 in the schedules section above.

**6.** At `t = 1`, with moments initialized to zero, `m₁ = (1−β₁)·g` and `v₁ = (1−β₂)·g²`.
Bias correction divides each by exactly `(1−β₁)` and `(1−β₂)` respectively, giving
`m̂₁ = g` and `√v̂₁ = |g|`. The update is therefore `−η·g/(|g|+ε) ≈ −η·sign(g)`: the
magnitude of `g` cancels completely. The implication for warmup is that the minimum
sensible length is set by how many steps `v` needs to become a meaningful estimate of the
gradient's second moment, which is on the order of `1/(1−β₂)` steps — 20 steps at
`β₂ = 0.95`, 1000 at `β₂ = 0.999`. Anything shorter and the denominator is still dominated
by whichever minibatches happened to arrive first. Note that this couples warmup length to
`β₂`, which almost no recipe states explicitly: shortening `β₂`'s window shortens the
warmup you need. In practice recipes use 1–2% of total steps, comfortably above that floor.

---

## Sources

`[C]` arXiv ids verified against the live arXiv API on 2026-07-26; resolution proves the
paper exists, not that it supports the claim beside it.

**Loss and the logits allocation.** 2411.09009 — Cut Your Losses in Large-Vocabulary
Language Models (Nov 2024). 2410.10989 — Liger Kernel: Efficient Triton Kernels for LLM
Training (Oct 2024). 2202.08906 — ST-MoE (Feb 2022; z-loss). 2312.04927 — Zoology:
Measuring and Improving Recall in Efficient Language Models (Dec 2023). 2606.09864 —
alignment degradation under KV quantization (Jun 2026; via
`research/notes/evaluation-landscape.md`). 2410.02660 — How to Train Long-Context Language
Models (Effectively) (Oct 2024; rejects perplexity as a progress signal).

**Optimizers.** 1412.6980 — Adam: A Method for Stochastic Optimization (Dec 2014).
1711.05101 — Decoupled Weight Decay Regularization (Nov 2017). 1804.04235 — Adafactor
(Apr 2018). 2110.02861 — 8-bit Optimizers via Block-wise Quantization (Oct 2021).
2509.02046 — Fantastic Pretraining Optimizers and Where to Find Them (Sep 2025).
2605.17787 — Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective
Learning Rates (May 2026). 2602.07712 — Towards Robust Scaling Laws for Optimizers
(Feb 2026). 2607.20548 — SOAP, Muon, and Beyond (Jul 2026; via
`research/notes/pretraining-recipes.md`). 1910.02054 — ZeRO (Oct 2019; optimizer-state
sharding, design-only here per `single-device-only`).

**Schedules.** 2410.05192 — Understanding Warmup-Stable-Decay Learning Rates: A River
Valley Loss Landscape Perspective (Oct 2024). 2508.01483 — Training Dynamics of the
Cooldown Stage in WSD (Aug 2025). 2601.09000 — Universal Dynamics of Warmup Stable Decay
(Jan 2026). 2602.06797 — Optimal Learning-Rate Schedules under Functional Scaling Laws
(Feb 2026). 2603.16127 — Pre-training without LR decay enhances SFT (Mar 2026; via
`research/notes/pretraining-recipes.md`). 2404.06395 — MiniCPM (Apr 2024; the WSD origin
in an open recipe). 2203.15556 — Training Compute-Optimal Large Language Models (Mar 2022).
2001.08361 — Scaling Laws for Neural Language Models (Jan 2020).

**Clipping and stability.** 1211.5063 — On the difficulty of training Recurrent Neural
Networks (Nov 2012). 1905.11881 — Why gradient clipping accelerates training (May 2019).
2502.11034 — AdaGC (Feb 2025). 2504.02507 — ZClip (Apr 2025). 2510.01578 — Gradient
Shaping Beyond Clipping (Oct 2025). 2312.16903 — Spike No More (Dec 2023). 2309.14322 —
Small-scale proxies for large-scale Transformer training instabilities (Sep 2023).

**Context.** 1706.03762 — Attention Is All You Need (Jun 2017). 1710.03740 — Mixed
Precision Training (Oct 2017). 2005.14165 — Language Models are Few-Shot Learners
(May 2020). 2203.03466 — Tensor Programs V (Mar 2022; muP, and where per-group LRs land in
`optim_step`). 1812.06162 — An Empirical Model of Large-Batch Training (Dec 2018).
2410.21676 — How Does Critical Batch Size Scale in Pre-training? (Oct 2024).

**Local and measured.** `ASSUMPTIONS.md` — `large-tensor-fault-32gib` (32 GiB hang),
`gpu-fast-tier-size` (≥62 GiB at ~200 GB/s), `hipblaslt-config`,
`gemm-throughput-below-reference` (20.9 TFLOP/s bf16 at 8192³),
`bf16-numerics-unproven`, `kv-per-token-laguna`. `ENVIRONMENT.md` (2026-07-26; 32 GB system
RAM after the carve-out, torch `2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, driver
`32.0.23033.5002`). `notebook/uma-carveout-controls-fast-tier.md`.
`research/reference/CODE_MAP.md`. `research/notes/pretraining-recipes.md`,
`research/notes/transformer-state-of-the-art.md`, `research/notes/evaluation-landscape.md`,
`research/memory/kv-cache-mechanics.md`.
