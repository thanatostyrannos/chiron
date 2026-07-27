---
title: Normalization and activations — LayerNorm to RMSNorm, placement, QK-norm, SwiGLU
version: 1.0.0
date: 2026-07-26
track: B — Modern architecture
prereqs: transformer-forward-pass-by-hand (soft — this module re-derives the residual stream from scratch, but the block structure is assumed). Also assumes you can read PyTorch and have run `scripts/activate-lab.ps1`. Reads well immediately before or after attention-variants-and-kv-cost, which shares the QK-norm-before-RoPE finding.
difficulty: moderate. The math is high-school algebra plus one Jacobian; the hard part is holding four ideas apart that the field routinely conflates.
time: 3–4 hours reading and working the arithmetic by hand. Exercises add roughly 3 hours of writing and analysis plus ~35 minutes of unattended machine time (all three runtimes measured on this Z13, not estimated).
---

# Normalization and activations

## What this module settles

Normalization and activation choices in a 2026 decoder are, with one arguable exception,
**stability controls rather than quality features** — they widen the range of learning rates,
depths and precisions in which a model trains at all, and they mostly do not move the loss at
a learning rate where both arms already converge. The three decisions the word
"normalization" hides — *what function*, *where in the block*, and *on which quantity* — have
completely different evidential status: the function is settled (RMSNorm), the quantity is
settled in practice (queries and keys, per head, before RoPE), and the placement is
genuinely unresolved with at least seven positions shipping in 2026. The way you tell a
stability fix from a quality win is that a stability fix moves the **maximum stable learning
rate** and leaves the loss-at-fixed-LR alone; almost nobody reports that frontier, which is
why the literature reads as if these changes were free wins.

---

## Theory in plain language

### The residual stream is a shared bus, and a norm is gain control on the tap

A decoder layer does not transform its input and pass the result on. It *adds* to a running
vector. Write `x_l` for that vector at layer `l`; a pre-norm block is

```
x_{l+1} = x_l + Attention(Norm(x_l))
x_{l+2} = x_{l+1} + MLP(Norm(x_{l+1}))
```

Every layer reads the whole vector and writes a correction back onto it. In the code this is
literally two lines — `research/reference/training/nanogpt/model.py:104` is
`x = x + self.attn(self.ln_1(x))`.

**The systems bridge.** This is a shared bus that ~100 devices read from and write to, with
no arbitration and no addressing: each writer adds its contribution to the same wire, and
each reader takes an automatic-gain-controlled copy of whatever is currently on it. The norm
is the AGC on the reader's tap. It exists because there is no protocol keeping the bus level
in range — nothing bounds how large `x_l` gets, and a sublayer whose input scale drifts by
100x behaves like a different function.

**Where the bridge breaks, and this is the whole of the placement argument.** The AGC is on
the *tap*, not on the *bus*. `Norm(x_l)` is bounded; `x_l` itself is not, and it keeps
growing with depth. So the raw signal that the last layer's output projection and the
language-model head actually see has a magnitude nobody controls. Every argument about
pre-norm versus post-norm versus peri-norm is an argument about where else to put a gain
stage so the bus itself stays in range — and each answer costs something on the gradient
path, which has no analogue in a hardware bus at all, because a hardware bus does not have to
propagate anything backwards.

### LayerNorm to RMSNorm: what got deleted, and why nobody missed it

LayerNorm `[C]` ([1607.06450](https://arxiv.org/abs/1607.06450), Jul 2016) does two things to
a vector: it subtracts the mean (re-centering) and divides by the standard deviation
(re-scaling), then applies a learned per-channel gain and bias. RMSNorm `[C]`
([1910.07467](https://arxiv.org/abs/1910.07467), Oct 2019) deletes the mean subtraction and
the bias, keeping only "divide by the root-mean-square, then apply a learned gain."

The paper's stated argument was that re-centering is not what buys the benefit — re-scaling
is — and it reported run-time reductions of **7%–64%** across the models it tested `[C]`
(1910.07467 abstract). That number is from the RNN era and does not transfer; on a modern
decoder the FLOP saving is a rounding error (worked below). RMSNorm won because it is
strictly simpler and measured no worse, which is a perfectly good reason and not the one
usually given.

### Placement: same sublayer, same norm, four different wirings

Take one sublayer `F` (attention or MLP), one norm `N`, and one residual add. Three
arrangements are in production right now, and a fourth is historical but keeps getting
confused with one of the three:

| Name | Wiring | Shipping in |
|---|---|---|
| **Pre-norm** | `x + F(N(x))` | Laguna `[M]`, Qwen3 `[C]` ([2505.09388](https://arxiv.org/abs/2505.09388)) |
| **Post-norm, inside the residual** | `x + N(F(x))` | OLMo 2 `[C]` ([2501.00656](https://arxiv.org/abs/2501.00656)) |
| **Peri-norm / sandwich** | `x + N(F(N(x)))` | Gemma 3 `[C]` ([2503.19786](https://arxiv.org/abs/2503.19786)), analysed in `[C]` ([2502.02732](https://arxiv.org/abs/2502.02732), ICML 2025) |

Note that "post-norm" in 2026 does **not** mean the original 2017 wiring `N(x + F(x))`, which
puts the norm *outside* the skip connection and is the thing that needed learning-rate warmup
to not diverge `[C]` ([2002.04745](https://arxiv.org/abs/2002.04745)). OLMo 2's post-norm
keeps the norm inside the skip. If you read two papers using "post-norm" for these two
different things — and you will — the arithmetic in the next section tells them apart
immediately: the 2017 version has no un-normalized identity path, the 2025 version does.

Two more positions exist and are worth knowing because they are attacking the same problem
from different sides:

- **Mix-LN** `[C]` ([2412.13795](https://arxiv.org/abs/2412.13795), Dec 2024): post-norm on
  the early layers, pre-norm on the deep ones, on the argument that pre-norm starves deep
  layers of gradient and post-norm starves early ones.
- **Depth-scaled gain**, where the block's second norm is initialised near `1/√L`, motivated
  by the *curse of depth* result `[C]` ([2502.05795](https://arxiv.org/abs/2502.05795),
  NeurIPS 2025).

And two that question the premise:

- **DyT** `[C]` ([2503.10622](https://arxiv.org/abs/2503.10622), Mar 2025, CVPR 2025) replaces
  the norm entirely with `tanh(αx)` for a learned scalar `α`, on the observation that a
  trained LayerNorm's input-output map already looks like a squashing S-curve. No reduction,
  no cross-channel coupling, elementwise.
- **TaperNorm** `[C]` ([2602.10408](https://arxiv.org/abs/2602.10408), Feb 2026, rev. May
  2026) removes internal norms gradually during training and reports small validation-loss
  increases, but finds the **final** norm before the LM head is not removable: it anchors the
  scale of the pre-logit representation, and without it the model inflates logit magnitudes
  instead of adjusting representations. They fold the tapered ops into adjacent projections
  for up to 1.18x decode throughput.

Seven positions. None of them has been compared head-to-head at a matched token budget above
roughly 3B parameters, and `[C]` ([2603.15389](https://arxiv.org/abs/2603.15389), Mar 2026)
reports that **sparsity interacts with the depth effect** — so the answer for a dense model
is not automatically the answer for an MoE. Our reference model is a sparse pre-norm model,
i.e. exactly the configuration the depth literature is most suspicious of. Treat pre-norm in
Laguna as an inherited default, not a demonstrated choice.

### QK-norm: a second norm, on a completely different quantity

The block norms operate on the residual stream at full width (3072 in Laguna). QK-norm is a
*separate* RMSNorm applied to the query and key vectors of each attention head, at head width
(128 in Laguna), after the projection and **before** RoPE. Origin `[C]`
([2010.04245](https://arxiv.org/abs/2010.04245)); scaled up in ViT-22B `[C]`
([2302.05442](https://arxiv.org/abs/2302.05442)).

What it fixes: the attention logit is `q·k/√d`, and nothing in the architecture bounds `‖q‖`
or `‖k‖`. A run can drift into a regime where logits are large, softmax saturates, the
gradient through the softmax collapses, and a loss spike or a silently dead head follows.
QK-norm bounds both vectors, which bounds the logit — and the bound is a closed-form constant
you can compute from `head_dim`, derived below.

> **Systems bridge, and where it breaks.** QK-norm is a rate limiter on a queue whose
> overflow is silent. The bridge breaks because the "overflow" is not dropped work — it is a
> *gradient* that stops flowing, so the symptom appears thousands of steps later as a flat
> loss curve rather than at the moment of saturation. There is no backpressure signal to
> observe and no counter to scrape. That is why the fix is architectural rather than a
> monitored threshold, and it is the general shape of every stability control in this module.

### Activations: from a switch to a valve

The original transformer FFN was `W2 · ReLU(W1 x)` — a per-channel switch that passes
positive values and zeroes negative ones. Two things changed.

**Smoothing.** ReLU's derivative is discontinuous at zero and identically zero for negative
inputs, so a unit that goes negative stops receiving gradient. GELU `[C]`
([1606.08415](https://arxiv.org/abs/1606.08415)) and SiLU/Swish `[C]`
([1710.05941](https://arxiv.org/abs/1710.05941)) replace the switch with a smooth,
*non-monotone* curve that dips slightly below zero before flattening, so negative-input units
keep a small live gradient.

**Gating.** GLU-style layers `[C]` ([2002.05202](https://arxiv.org/abs/2002.05202)) split the
FFN's first projection into two, run the activation on one branch, and multiply. SwiGLU is
`down( SiLU(gate(x)) ⊙ up(x) )` — the `⊙` is elementwise multiply. Read as a valve: the
`gate` branch produces a per-channel opening coefficient and the `up` branch produces the
content being passed. The Laguna implementation is one line,
`research/reference/models/laguna-s/modeling_laguna.py:140`.

**Where the valve analogy breaks:** the coefficient is not in `[0,1]`. `SiLU(z) → z` for large
positive `z`, so a "wide open" valve has *gain*, not just transparency, and the output grows
as the product of two unbounded linear functions — quadratically in the input. That is not a
detail; it is the mechanism behind both gpt-oss's activation clamp and the 2026 papers
proposing SwiGLU replacements, and it is measurable in ten lines (Exercise A).

### Stability fix or quality win: the test

This is the load-bearing methodological point of the module.

Every change here is reported as "improves training." Two very different things get that
label:

- A **quality win** moves the converged loss at a learning rate where *both* arms are stable.
- A **stability fix** does not move that loss. It moves the *maximum learning rate at which
  the arm is stable at all* — and since practitioners then run at the higher LR, the change
  shows up as better loss-per-wall-clock, which reads like quality in a results table.

The discriminating experiment is a two-dimensional sweep, not a single-point comparison: run
both arms across a learning-rate ladder and report (i) the highest LR at which each arm
survives, and (ii) the loss at the highest LR where *both* survive. If (ii) is flat and (i)
differs, you have a stability control. Almost no paper in this area reports both axes, which
is why the adoption rate of QK-norm outruns its published evidence.

Rating the four topics of this module against that test:

| Change | Status |
|---|---|
| RMSNorm over LayerNorm | Settled, and neither a stability fix nor a quality win — a simplification measured no worse |
| Norm placement | Genuinely contested; the published mechanisms are all stability/variance arguments |
| QK-norm | Stability control, demonstrated at high LR. Quality effect at conservative LR is small and sometimes negative |
| SwiGLU over ReLU FFN | The closest thing here to a quality win — but it was reported at matched parameters on a fixed recipe, not against an LR frontier |

---

## The math that actually matters

Notation, once. `x` is one token's activation vector at one position, with `d` components
written `x_1 … x_d`. `⊙` is elementwise multiplication. `‖x‖₂` is the ordinary Euclidean
length, `√(x_1² + … + x_d²)`. All norms here operate **along the feature axis of a single
token** — never across tokens, never across the batch. That is the property that makes them
safe for autoregressive decode: token `t`'s normalization does not depend on token `t+1`, so
the same computation is valid during prefill and during single-token decode.

### LayerNorm, symbol by symbol

```
μ    = (1/d) · Σ_i x_i                        the mean:  add up all d components, divide by d
σ²   = (1/d) · Σ_i (x_i − μ)²                 the variance: mean of the squared deviations
x̂_i  = (x_i − μ) / √(σ² + ε)                  centre, then divide by the standard deviation
y_i  = g_i · x̂_i + b_i                        learned per-channel gain g and bias b
```

`ε` (epsilon) is a small constant added before the square root so that an all-zero vector
does not divide by zero. nanoGPT uses `1e-5`
(`research/reference/training/nanogpt/model.py:27`); Laguna uses `1e-6` (`config.json`,
`rms_norm_eps`). `g` and `b` are each `d` numbers learned by gradient descent, so LayerNorm
costs `2d` parameters per norm site.

### RMSNorm, and the one thing it deletes

```
RMS(x) = √( (1/d) · Σ_i x_i² )                root of the mean of the squares — no μ anywhere
y_i    = g_i · x_i / √( RMS(x)² + ε )         divide by that, then apply the learned gain
```

Three differences, and only the third is interesting:

1. No `μ`. One reduction over `d` instead of two moments.
2. No bias `b`. `d` parameters per site instead of `2d`.
3. **`x_i` is not re-centred.** Before the gain is applied, the output is a pure *positive
   rescaling* of the input: signs, zeros and the direction of `x` are all preserved, and only
   its length changes. (The learned per-channel gain then does rotate the direction — but it
   is the same gain for every token, so the token-dependent part of the operation is purely
   radial. LayerNorm's centering is not.)

The code is `research/reference/models/laguna-s/modeling_laguna.py:58-63`, and the fp32
upcast on line 60 is the whole numerics story (Exercise A).

**The parameter arithmetic, in full, so you can see it does not matter.** Laguna has 48
layers × 2 block norms + 1 final norm = 97 norm sites at width 3072. LayerNorm would cost
`97 × 2 × 3072 = 595,968` parameters; RMSNorm costs `97 × 3072 = 297,984`. The saving is
298k parameters out of 117.5 B — 0.00025%. The QK-norms add `48 layers × 2 (q and k) × 128 =
12,288` more. **Nobody chose RMSNorm to save parameters.**

**The FLOP arithmetic, also in full, also to show it does not matter.** The extra work
LayerNorm does per token per site is one more reduction over `d` (`d−1` adds) plus `d`
subtractions, so roughly `2d` operations. Over 97 sites and an 8192-token sequence:
`97 × 8192 × 2 × 3072 ≈ 4.88 GFLOP`. Laguna's forward pass over the same sequence costs
roughly `2 × 8.1e9 active params × 8192 tokens ≈ 132.7 TFLOP`. The ratio is **0.0037%**. A
fused kernel computes both moments in one pass anyway, so even the extra memory pass largely
disappears.

So: RMSNorm is not a performance optimization on a modern decoder. It is a deletion that was
measured not to hurt. Any curriculum or blog post that tells you RMSNorm was adopted "because
it's faster" is repeating a 2019 RNN result out of context.

### Worked arithmetic on four numbers

Take `x = (2, −1, 0, 3)`, so `d = 4`. Ignore `ε` and set the gain to 1.

*LayerNorm.*
```
μ  = (2 + (−1) + 0 + 3) / 4 = 4/4 = 1
centred:  (2−1, −1−1, 0−1, 3−1) = (1, −2, −1, 2)
σ² = (1² + (−2)² + (−1)² + 2²) / 4 = (1 + 4 + 1 + 4)/4 = 10/4 = 2.5
σ  = √2.5 = 1.5811
LN(x) = (1/1.5811, −2/1.5811, −1/1.5811, 2/1.5811)
      = (0.6325, −1.2649, −0.6325, 1.2649)
```

*RMSNorm.*
```
RMS(x)     = √((2² + (−1)² + 0² + 3²)/4) = √((4 + 1 + 0 + 9)/4) = √3.5 = 1.8708
RMSNorm(x) = (2/1.8708, −1/1.8708, 0/1.8708, 3/1.8708)
           = (1.0690, −0.5345, 0.0000, 1.6036)
```

These are not the same vector and not even close. LayerNorm moved the zero to `−0.6325`;
RMSNorm left it at zero. LayerNorm's output is orthogonal to the all-ones direction by
construction; RMSNorm's is not.

**The invariant they share.** Check the lengths:
```
‖LayerNorm(x)‖₂²  = 0.4000 + 1.6000 + 0.4000 + 1.6000 = 4.0
‖RMSNorm(x)‖₂²    = 1.1428 + 0.2857 + 0.0000 + 2.5715 = 4.0
```
Both are exactly `d`. That is not a coincidence and it is the single most useful fact in this
module: **an RMS-normalized vector has L2 length exactly `√d`, always.** For `d = 128` that
is `11.3137`, whatever the input was. Everything downstream — the QK-norm logit bound, the
key-norm eviction consequence, the scale-invariance argument — falls out of that one line.

### The gradient consequence: a norm deletes one degree of freedom

Write RMSNorm without gain or epsilon as `y = √d · x / ‖x‖₂` (identical to the formula above,
since `RMS(x) = ‖x‖₂/√d`). Its Jacobian — the matrix of partial derivatives `∂y_i/∂x_j` — is

```
J = (√d / ‖x‖₂) · ( I − x̂ x̂ᵀ )        where x̂ = x/‖x‖₂ is the unit vector along x
```

In words: identity, minus the projection onto the direction of `x` itself, all scaled by
`√d/‖x‖₂`. Now apply it to `x`:

```
J x = (√d/‖x‖₂) · ( x − x̂ (x̂ · x) ) = (√d/‖x‖₂) · ( x − x̂‖x‖₂ ) = (√d/‖x‖₂) · ( x − x ) = 0
```

**The Jacobian annihilates the radial direction.** Any gradient signal that would have
changed the *length* of `x` is discarded exactly. Three consequences that matter:

1. A sublayer behind a norm cannot see, and cannot be trained by, the magnitude of its input.
   This is why the residual stream's steady growth with depth is invisible to each sublayer —
   and why fixing it requires a wiring change rather than a training signal.
2. A weight matrix whose output feeds *directly* into a norm becomes scale-invariant:
   `Norm(Wx) = Norm(αWx)` for any `α > 0`. Its gradient is therefore orthogonal to `W`
   itself, so under plain SGD `‖W‖` only ever grows (each update adds a perpendicular
   component, and `‖W + ΔW‖² = ‖W‖² + ‖ΔW‖²`), and the *effective* step size on the unit
   sphere shrinks like `1/‖W‖²`. Weight decay is what holds that in check. This is the
   mechanism behind the notorious LR-and-weight-decay coupling in normed networks.
3. Point 2 applies with full force to `q_proj` and `k_proj` under QK-norm, since their
   outputs feed straight into `q_norm`/`k_norm`
   (`research/reference/models/laguna-s/modeling_laguna.py:421-422`). Turning QK-norm on
   changes what the weight decay on those two matrices *means*. If you ablate QK-norm without
   re-tuning weight decay, you are running a confounded arm.

### Pre-norm: why the stream grows as √l and the deep layers go quiet

Assume each sublayer's output `F(Norm(x_l))` has unit RMS (which pre-norm approximately
enforces, since its input is normalized) and is roughly uncorrelated with `x_l`. Then

```
x_{l+1} = x_l + f_l                       with RMS(f_l) ≈ 1 and f_l ⊥ x_l
⟹  RMS(x_{l+1})² ≈ RMS(x_l)² + 1          variances of independent terms add
⟹  RMS(x_l) ≈ √(1 + l)                    starting from RMS(x_0) = 1
```

So the bus level grows as the **square root of depth**. At `l = 48` that is `√49 = 7.0`.
Now the important part — the *relative* size of what each layer contributes:

```
contribution ratio at layer l  =  RMS(f_l) / RMS(x_l)  ≈  1 / √l
```

At `l = 1` it is 1.0. At `l = 24` it is `1/√24 = 0.204`. At `l = 48` it is
`1/√48 = 0.1443`. **The 48th layer of a 48-layer pre-norm stack is writing a correction
worth about 14% of the bus level it read.** Its Jacobian with respect to the block input is
`I + J_F`, and as `J_F` shrinks relative to `I`, the block converges to the identity — which
is the *curse of depth* claim `[C]` ([2502.05795](https://arxiv.org/abs/2502.05795)) in one
line: deep pre-norm layers contribute progressively less and are correspondingly hard to
train.

`[M]` **Measured, 2026-07-26, this machine, CPU, deterministic (seeded generator, seed 0),
d=512, L=48, batch 256** — a linear surrogate stack, one run, script in Exercise B:

| l | measured RMS(x_l) | √(1+l) | measured contribution ratio | 1/√l |
|---|---|---|---|---|
| 1 | 1.416 | 1.414 | 1.0002 | 1.0000 |
| 4 | 2.228 | 2.236 | 0.4978 | 0.5000 |
| 12 | 3.600 | 3.606 | 0.2877 | 0.2887 |
| 24 | 5.011 | 5.000 | 0.2043 | 0.2041 |
| 48 | 6.995 | 7.000 | 0.1441 | 0.1443 |

The derivation reproduces to three significant figures. This is a linear surrogate, not a
trained transformer — it demonstrates the *mechanism*, not the empirical magnitude in a real
model, where sublayer outputs are neither unit-RMS nor independent of the stream.

### The other three wirings, and where the surrogate honestly runs out

Four wirings, stated precisely, because two of them are routinely both called "post-norm":

```
pre        :  f = F(N(x))     ;  x ← x + f          Laguna, Qwen3
post_res   :  f = N(F(x))     ;  x ← x + f          OLMo 2 — norm INSIDE the skip
post_2017  :  f = F(x)        ;  x ← N(x + f)       the original — norm OUTSIDE the skip
peri       :  f = N(F(N(x)))  ;  x ← x + f          Gemma 3
```

`[M]` Same surrogate, same seed, RMS(x_48) and contribution ratio at l=48:

| wiring | sublayer gain 1.0 | | sublayer gain 2.0 | |
|---|---|---|---|---|
| | RMS(x_48) | contrib@48 | RMS(x_48) | contrib@48 |
| pre | 6.995 | 0.1441 | **13.871** | 0.1450 |
| post_res | 6.995 | 0.1443 | **6.995** | 0.1443 |
| peri | 6.995 | 0.1443 | **6.995** | 0.1443 |
| post_2017 | **1.000** | **1.0048** | **1.000** | **2.0150** |

Three readings, and the third is the important one.

**post_2017 is the odd one out, and that is the warmup story.** Normalizing *outside* the skip
pins the stream at exactly 1.000 at every depth — the bus never grows — but the contribution
ratio stays at ~1.0 (and rises to ~2.0 when the sublayer gain does). The sublayer's write is
always the same size as the whole bus, so there is no un-normalized identity path, and the
backward pass picks up a `1/σ` factor at every layer. That is the configuration that needed
learning-rate warmup `[C]` ([2002.04745](https://arxiv.org/abs/2002.04745)). OLMo 2's
post-norm is **not** this.

**post_res and peri cap the growth that pre-norm does not.** At unit gain all three are
identical, because the sublayer output is already unit-RMS and normalizing it changes nothing.
Turn the sublayer gain to 2 and pre-norm's stream grows to 13.871 — matching the prediction
`√(1 + 48·2²) = 13.892` — while post_res and peri stay pinned at 6.995, matching `√49 = 7.000`.
Normalizing the sublayer's *output* caps each write at unit size no matter what the sublayer
wants to emit, so stream growth is `√L` regardless of gain. That is exactly the peri-LN
argument `[C]` ([2502.02732](https://arxiv.org/abs/2502.02732)): pre-LN's hidden-state
variance grows without bound and produces massive activations; peripheral placement controls
it. Massive activations are a documented, separate phenomenon — up to 100,000x the typical
activation magnitude, near-constant across inputs, acting as implicit bias terms `[C]`
([2402.17762](https://arxiv.org/abs/2402.17762), COLM 2024) — and they are what makes
low-precision inference and quantization hard.

**post_res and peri are indistinguishable in this surrogate, and that is a finding about the
surrogate.** Their forward statistics are identical to four decimal places in every column.
The real difference between them is that peri's sublayer sees a *normalized* input while
post_res's sees the raw growing stream — and a linear map's output direction does not depend
on its input scale, so a linear surrogate cannot express that difference at all. Whatever
separates OLMo 2 from Gemma 3 lives in the nonlinearity, the attention softmax, and the
optimizer. **Know where your toy stops.** It reproduces four mechanisms and it cannot
adjudicate between two of the four positions; a real comparison needs the two-axis LR sweep
from the previous section on a trained model.

### QK-norm: from an unbounded logit to a bound of `√d_head`

The attention logit between query `q` and key `k`, both of width `d = head_dim`, is

```
logit = (q · k) / √d              q · k = Σ_i q_i k_i, the dot product
```

The `1/√d` is there because a dot product of two `d`-dimensional random unit-variance vectors
has standard deviation `√d`, so dividing restores unit scale. Cauchy–Schwarz gives
`|q · k| ≤ ‖q‖₂ · ‖k‖₂`, so

```
|logit| ≤ ‖q‖₂ · ‖k‖₂ / √d
```

**Without QK-norm, nothing bounds `‖q‖₂` or `‖k‖₂`.** They are outputs of unconstrained
linear projections; they can and do drift upward during training.

**With QK-norm**, using the invariant established above — an RMS-normalized vector has length
exactly `√d` — and writing `g_max` for the largest absolute value in the learned gain vector:

```
‖RMSNorm_g(q)‖₂ ≤ g_max · √d
|logit| ≤ (g_max √d)(g_max √d) / √d = g_max² · √d
```

For Laguna's `head_dim = 128` and gains near 1, that is `√128 = 11.31`. **The logit range is
now an architectural constant you can compute from the config, not a runtime property you
have to monitor.**

Why the bound matters, in numbers. Softmax over `n` competing keys, where the winner leads by
a margin `Δ`, gives the winner probability roughly `p = 1 / (1 + (n−1)e^{−Δ})`, and the
gradient flowing back through the softmax to that logit is proportional to `p(1−p)`. Take
`n = 1024`:

| margin Δ | `(n−1)·e^{−Δ}` | `p` | `p(1−p)` — the gradient scale |
|---|---|---|---|
| 11.31 (QK-normed bound) | 1.25e−2 | 0.9877 | **1.22e−2** |
| 30 | 9.57e−11 | 1 − 9.6e−11 | **9.6e−11** |
| 60 | 8.96e−24 | 1 − 9.0e−24 | **9.0e−24** |

Going from a bounded margin of 11.31 to an unbounded one that reaches 30 costs **eight orders
of magnitude of gradient** through that head (1.22e−2 → 9.6e−11); at 60 it is twenty-one
orders (1.22e−2 → 9.0e−24). bf16 has fp32's
exponent range so `9e−24` is representable — the number is not the problem. The problem is
that it is negligible against every other term in the sum, so the head has silently stopped
learning while every metric you scrape looks fine. That is the "silent overflow" the systems
bridge was pointing at.

**The honest evidential status.** The published ablations show QK-norm buys headroom *at high
learning rate*; at conservative learning rates the un-normed model is sometimes marginally
better. At least one 2026 model (Cohere's Tiny Aya) dropped it on the grounds that it
interacts badly with long context. And it is structurally incompatible with MLA, which
deliberately never materializes full-width keys — a conflict serious enough that a 2026 paper
exists purely to reconcile them `[C]`
([2606.16310](https://arxiv.org/abs/2606.16310), Jun 2026), reporting under 2% decode latency
overhead to 256k context. The community treating it as non-negotiable enough to redesign
around is evidence about adoption, not about quality.

### SwiGLU: the parameter-parity arithmetic and the quadratic tail

```
SiLU(z) = z · σ(z)          where σ(z) = 1/(1 + e^{−z}), the logistic sigmoid
SwiGLU FFN(x) = W_down · ( SiLU(W_gate x) ⊙ (W_up x) )
```

Three symbols to place: `W_gate` and `W_up` both map width `d → m`; `W_down` maps `m → d`;
`m` is the FFN inner width (`intermediate_size` in the config).

**SiLU is non-monotone.** Its derivative is `σ(z)·(1 + z(1 − σ(z)))`, which is negative for
`z` below about `−1.278`, where SiLU reaches its minimum of about `−0.278`. So a small
negative input produces a small negative output *and* a live gradient — the property ReLU
lacks.

**The parameter-parity arithmetic.** An ungated FFN of inner width `m_u` costs `2·d·m_u`
parameters (two matrices). A gated FFN of inner width `m_g` costs `3·d·m_g` (three matrices).
The classic ungated width was `m_u = 4d`. Setting them equal:

```
3 d m_g = 2 d (4d)  ⟹  m_g = (2/3)·4d = 8d/3 ≈ 2.667 d
```

That is where the famous `8/3` factor comes from. It is a *parameter-parity* convention, not
a modelling result, and it exists so that a SwiGLU-vs-ReLU comparison is not secretly a
comparison of two different parameter counts.

**Laguna does not follow it, and the config says so.** `[M]` from
`research/reference/models/laguna-s/config.json`: `hidden_size = 3072`,
`intermediate_size = 12288 = 4 × 3072`. So the dense MLP (present only on layer 0, per
`mlp_only_layers: [0]`) is gated *and* full 4d width:

```
3 × 3072 × 12288 = 113,246,208 parameters
```

against `3 × 3072 × 8192 = 75,497,472` under the 8/3 convention — **1.5x the
parameter-parity width**. And the experts are the other direction:
`moe_intermediate_size = 1024 = d/3`, so each expert is `3 × 3072 × 1024 = 9,437,184`
parameters, and `256 experts × 47 MoE layers × 9,437,184 = 113.55 B` — which reproduces the
published routed-expert total, so the reading is right. In MoE models the expert width is set
by a sparsity/granularity argument, and the parameter-parity convention is simply gone.

One more config-reading lesson: `hidden_act` does **not** appear in `config.json` at all. It
comes from the dataclass default at
`research/reference/models/laguna-s/configuration_laguna.py:147`, which is `"silu"`. You
cannot determine this model's activation function from its config file.

**The quadratic tail.** For large positive `z`, `σ(z) → 1`, so `SiLU(z) → z`. Therefore for
large positive gate values,

```
SwiGLU ≈ (W_gate x) ⊙ (W_up x)
```

— a product of two unbounded linear functions, i.e. **quadratic growth in the input**. This
is the destabilizing mechanism named explicitly by `[C]`
([2605.25704](https://arxiv.org/abs/2605.25704), May 2026, PowLU), which argues SwiGLU's
quadratic-like behaviour "enlarges output ranges and worsens outliers, particularly in
low-precision training," and proposes a rational-power replacement tested at 7.9B and 124B.

gpt-oss patches the same problem by clamping instead of replacing `[M]`
(`research/reference/architecture/gpt-oss/gpt_oss/torch/model.py:249-256`, with
`swiglu_limit: float = 7.0` at `:20`). Their variant clamps the gate branch above at 7 and
the linear branch to `[−7, 7]`, uses `σ(1.702·z)` (which makes SiLU approximate GELU), and
adds 1 to the linear branch. That gives a **hard, compile-time-known output ceiling**:

```
7 · σ(1.702 × 7) · (7 + 1) = 7 × 0.9999933 × 8 = 55.9996
```

`[M]` **Measured, 2026-07-26, gfx1151, 1e6 samples per row, plain SwiGLU with a symmetric
`limit = 7` clamp** (Exercise A produces this table):

| input std | max abs SwiGLU | max abs clamped | max ReLU² | SwiGLU p99.99 |
|---|---|---|---|---|
| 1.0 | 12.3 | 12.3 | 20.0 | 6.90 |
| 2.0 | 48.6 | 43.7 | 89.2 | 28.27 |
| 4.0 | 245.1 | 49.0 | 295.5 | 115.33 |
| 8.0 | 780.5 | 49.0 | 1304.8 | 461.64 |

Read that table as the definition of a stability fix. At input std 1.0 — the normal operating
regime — the clamp is **completely inert**: 12.3 versus 12.3, bit-identical behaviour. At std
4.0 it cuts the extreme by 5x. The clamp does nothing at all until the model is already in
trouble. Compare to a design change that alters the function everywhere, and the difference
between a stability control and a modelling choice becomes concrete.

(The clamped ceiling saturates at 49.0 in this table rather than 56.0 because the measurement
uses plain SiLU rather than gpt-oss's `σ(1.702z)` and omits their `+1` on the linear branch:
`7 · σ(7) · 7 = 7 × 0.99909 × 7 = 48.96`. Both are closed-form; check the arithmetic yourself
rather than trusting either number.)

**2026 alternatives, all contested and none independently replicated.** xIELU `[C]`
([2411.13010](https://arxiv.org/abs/2411.13010), Nov 2024, rev. Jan 2025) derives an
activation by integrating trainable affine transforms of ELU and reports lower loss than both
ReLU² and SwiGLU at matched compute on 1.1B and 3B Llama models — but with a task split
(better on factual recall and structured reasoning, worse on linguistic understanding) that
should make you suspicious of any single headline number. PowLU is above. Neither has been
replicated outside its originating group as far as I can find.

---

## Why it matters for Proteus

Every item in this module is a config field, and per the house rule the config surface *is*
the experimental surface. The fields this module argues Proteus needs:

| Field | Values | Why it exists |
|---|---|---|
| `norm_function` | `rmsnorm` \| `layernorm` \| `dyt` | The one settled axis; keep `layernorm` and `dyt` as controls, not as candidates |
| `norm_placement` | `pre` \| `post_in_residual` \| `peri` \| `mix` | Four implementable arms against seven shipping positions, and no matched-budget head-to-head exists. This is a real experiment |
| `norm_gain_init` | `1.0` \| `depth_scaled` (`1/√L`) \| `zero_centered` | Gemma 3 stores `(1 + w)` with `w` initialised to zero — see the code section |
| `norm_reduction_dtype` | `fp32` \| `native` | Directly gates `ASSUMPTIONS.md: bf16-numerics-unproven`, which lists RMSNorm by name |
| `qk_norm` | `off` \| `pre_rope` \| `post_rope` | The ordering matters for Mnemosyne — see below |
| `activation` | `swiglu` \| `gelu_ffn` \| `relu2` \| `swiglu_clamped` | `swiglu_clamped` needs a companion `activation_clamp: float` |
| `ffn_width_rule` | `parity_8_3` \| `explicit` | Laguna uses explicit 4d; the parity rule is what makes a gated/ungated comparison fair |

Three connections that are specific to this lab rather than generic:

**1. The bf16 gate.** `ASSUMPTIONS.md: bf16-numerics-unproven` is `untested` and names
matmul / softmax / **RMSNorm** / attention. That row is not abstract: the fp32 upcast at
`modeling_laguna.py:60` exists precisely because the mean-of-squares reduction over `d`
accumulates error, and Exercise A measures how much on our silicon. Until that runs, no
result from this machine involving a norm counts as evidence.

**2. QK-norm changes what is in the KV cache, and one popular eviction policy may not survive
it.** Laguna applies QK-norm *before* RoPE (`modeling_laguna.py:421-425`), so a cached key is
`RoPE(RMSNorm_g(k))`. RoPE is a rotation, and rotations preserve L2 length. Therefore

```
‖k_cached‖₂ = ‖ g_k ⊙ (k / RMS(k)) ‖₂  ∈  [ min|g_k| · √d ,  max|g_k| · √d ]
```

If the learned gain vector `g_k` is close to uniform, **every cached key has almost the same
L2 norm, regardless of the token**. The L2-norm KV eviction family `[C]`
([2406.11430](https://arxiv.org/abs/2406.11430), Jun 2024, EMNLP 2024) scores tokens by
exactly that quantity — it reports that low key-L2 predicts high attention and retains on
that basis, claiming 50% cache reduction on language modelling and 90% on passkey retrieval.
`[A]` **Medium confidence: on a QK-normed model, L2-norm key scoring loses most of its
dynamic range and degenerates toward a constant.** Cosine-distinctiveness scoring (KeyDiff
`[C]` [2504.15364](https://arxiv.org/abs/2504.15364)) is scale-invariant and should be
unaffected — which would make the choice between two "attention-free" eviction scorers
architecture-dependent, a distinction the eviction literature does not currently draw.
Cheapest test: read the 96 `q_norm.weight` / `k_norm.weight` gain vectors out of a QK-normed
checkpoint and report `max|g| / min|g|` per layer; if it is near 1, the claim holds. **Note
that the local Laguna clones carry git-LFS pointers, not weights** (135 bytes per shard), so
this needs either a single-shard fetch or a smaller QK-normed model — cost it before running.

**3. QK-norm confounds weight decay.** Per the Jacobian result above, `q_proj` and `k_proj`
become scale-invariant when QK-norm is on. A `qk_norm: off` arm run at the same weight decay
as the `qk_norm: pre_rope` arm is not a controlled comparison. Whatever the Themis config for
this ablation looks like, weight decay on those two matrices has to be either swept or
explicitly argued.

---

## Read the code

Paths are relative to `research/reference/`. Clones are gitignored — run
`scripts/fetch_reference.sh` first. Read in this order; it is a deliberate progression from
the 2019 baseline to the 2026 positions.

**The baseline you are moving away from.**

| Where | What to look at |
|---|---|
| `training/nanogpt/model.py:18` | `class LayerNorm` — the whole 2019 recipe in ten lines, and the docstring tells you why it exists at all: PyTorch's `nn.LayerNorm` cannot be built without a bias. |
| `training/nanogpt/model.py:23` and `:24` | `weight` (ones) and `bias` (zeros) — the `2d` parameters. Compare with `models/laguna-s/modeling_laguna.py:55`, which has only the weight. The deletion is visible as one missing line. |
| `training/nanogpt/model.py:27` | `F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)` — note `eps` is hardcoded, and note there is no dtype management at all. That is the thing 2026 codebases all changed. |
| `training/nanogpt/model.py:104` | `x = x + self.attn(self.ln_1(x))` — the canonical pre-norm block on one line. Read this before any of the placement variants; everything below is a rearrangement of this line. |
| `training/nanogpt/model.py:83` | `self.gelu = nn.GELU()` inside a plain two-matrix MLP (`c_fc` at `:82` is `4 * n_embd` wide). This is the ungated FFN whose parameter count the `8/3` rule is matching. |

**RMSNorm as shipped, and the numerics.**

| Where | What to look at |
|---|---|
| `models/laguna-s/modeling_laguna.py:49` | `class LagunaRMSNorm`. The docstring says "equivalent to T5LayerNorm" — a hint that this formulation predates the RMSNorm paper's adoption in decoders. |
| `models/laguna-s/modeling_laguna.py:60` | `hidden_states = hidden_states.to(torch.float32)`. **The single most important line in this module.** The reduction on the next line is what needs the precision; Exercise A measures by how much. |
| `models/laguna-s/modeling_laguna.py:61-62` | `pow(2).mean(-1, keepdim=True)` then `rsqrt(variance + eps)` — the formula from the math section, verbatim, with no mean subtraction anywhere. Note the variable is *named* `variance` and is not one; it is the mean of squares. |
| `models/laguna-s/modeling_laguna.py:63` | `return self.weight * hidden_states.to(input_dtype)` — cast back to bf16 *before* the gain multiply. Contrast `olmo2/modeling_olmo2.py:65`, which is `(self.weight * hidden_states).to(input_dtype)` — gain applied in fp32, then cast. Two frontier models, same line, different order. Nobody has published which is better. |

**The three shipping placements, side by side in one checkout.** This is the most valuable
half hour in the module: three files, three different wirings, all frontier models, all
readable in fifteen lines each. (The fourth wiring, 2017 post-norm, is not in any of these
checkouts — it exists only in the surrogate of Exercise B.)

| Where | What to look at |
|---|---|
| `models/laguna-s/modeling_laguna.py:486-487` | Pre-norm constructs **two** norms: `input_layernorm`, `post_attention_layernorm`. The second name is a trap — it is the norm *before the MLP*, not after the attention output. |
| `models/laguna-s/modeling_laguna.py:500-512` | The pre-norm forward: `residual = hidden_states` → norm → sublayer → `residual + hidden_states`. Note the residual is captured *before* the norm. |
| `architecture/transformers/src/transformers/models/olmo2/modeling_olmo2.py:303-304` | Post-norm constructs `post_attention_layernorm` and `post_feedforward_layernorm` — and crucially there is **no** `input_layernorm`. That absence is the entire difference. |
| `architecture/transformers/src/transformers/models/olmo2/modeling_olmo2.py:326-327` | `hidden_states = self.post_attention_layernorm(hidden_states)` then `residual + hidden_states`. The sublayer runs on the raw stream and its *output* is normalized — inside the skip, which is what distinguishes this from 2017 post-norm. |
| `architecture/transformers/src/transformers/models/gemma3/modeling_gemma3.py:401-404` | Peri-norm constructs **four** norms per block: input, post-attention, pre-feedforward, post-feedforward. Count them and the sandwich is obvious. |
| `architecture/transformers/src/transformers/models/gemma3/modeling_gemma3.py:427` and `:431` and `:433` | The forward: normalize in, run sublayer, normalize out, add. Twice. |
| `architecture/transformers/src/transformers/models/gemma3/modeling_gemma3.py:140` and `:149` | A detail worth stealing: Gemma initialises the gain to **zeros** and applies `(1.0 + weight)`. Same function at init, but the parameter's natural scale is now centred on zero, which changes what weight decay does to it. Laguna initialises to ones and applies `weight` directly. |

**QK-norm.**

| Where | What to look at |
|---|---|
| `models/laguna-s/modeling_laguna.py:398-399` | `q_norm`/`k_norm` built at width `config.head_dim` (128), **not** `hidden_size`. A per-head norm, not a residual-stream norm. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:368` | The same construction in the upstream `transformers` copy, at a different line number. Worth knowing both exist: the model-repo copy and the library copy drift. |
| `models/laguna-s/modeling_laguna.py:421-422` | Applied after the `view/transpose` into head layout, so it normalizes each head's 128-dim slice independently. |
| `models/laguna-s/modeling_laguna.py:425` | `apply_rotary_pos_emb` on the *next* line. The ordering is norm-then-rotate, which is what makes the cached key a doubly-transformed quantity — see the Mnemosyne point above. |

**Activations.**

| Where | What to look at |
|---|---|
| `models/laguna-s/modeling_laguna.py:134-136` | Three matrices — `gate_proj`, `up_proj`, `down_proj` — all `bias=False`. The three-matrix shape *is* the gating. |
| `models/laguna-s/modeling_laguna.py:140` | `down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The `*` is the valve. |
| `models/laguna-s/configuration_laguna.py:147` | `hidden_act: str = "silu"` — a dataclass default. Grep `config.json` for `hidden_act` and you will find nothing. The shipped config does not state the model's activation function. |
| `architecture/gpt-oss/gpt_oss/torch/model.py:20` | `swiglu_limit: float = 7.0` sitting in the model config next to `head_dim` — a numerical-stability constant promoted to a first-class architecture hyperparameter. |
| `architecture/gpt-oss/gpt_oss/torch/model.py:249-256` | The clamped SwiGLU. Note three deviations from the standard form in six lines: `alpha=1.702` (which makes `z·σ(1.702z)` approximate GELU), an **asymmetric** clamp on the gate branch (`min=None, max=limit`) versus symmetric on the linear branch, and `+1` added to the linear branch so a zero gate output still passes signal. None of the three is explained anywhere in the file. |

---

## Exercises

All three run on the Z13 (gfx1151, native Windows, one GPU). Activate with
`. .\scripts\activate-lab.ps1` from PowerShell.

> **`[M]` Environment caveat, measured 2026-07-26 on this machine.** If you invoke Python from
> the Bash tool rather than PowerShell, set `HIPBLASLT_TENSILE_LIBPATH` with **Windows path
> separators**. A POSIX-style value (`/c/venvs/lab/...`) segfaults inside `hipblaslt` on the
> first `addmm` — the crash is in `hipblasLtMatmulAlgoGetHeuristic`, exit code 139, with no
> Python-level exception. Minimal repro: one `torch.addmm` in bf16 on `cuda`. The same call
> with `C:\venvs\lab\...` succeeds. This is a stronger failure than the documented ~5x
> throughput cliff (`ASSUMPTIONS.md: hipblaslt-config`) and is worth adding to the Hardware
> Validation Gate.

Work in a scratch directory outside `research/reference/` (those are read-only upstream
clones). If any of these produces a decision-changing result, it stops being an exercise and
becomes a pre-registered `notebook/<slug>.md` with a full G2 hypothesis card.

### Exercise A — Measure what the fp32 upcast actually buys, and what the SwiGLU clamp costs

**Difficulty: easy. Runtime: under 2 minutes GPU, under 4 minutes CPU. Writing it: 30–40
minutes.**

Two measurements in one script, because they are the same question — how an unbounded
quantity behaves in low precision.

*Part one.* Implement RMSNorm twice: once computing the mean-of-squares in the input dtype
(`naive`), once upcasting to fp32 for the reduction and casting back at the end (`upcast`,
i.e. what `modeling_laguna.py:58-63` does). At `d = 3072`, generate fp64 reference inputs at
scales `1e−2, 1, 1e2, 1e4, 1e6`, cast to bf16, run both, and report the **maximum relative
error against the fp64 reference**. Then, separately, report the relative error of the
mean-of-squares reduction alone, comparing a bf16 reduction against an fp32 reduction against
the fp64 truth.

*Part two.* Draw `1e6` samples of a gate value and a linear value at input standard deviations
`1, 2, 4, 8`. Compute plain SwiGLU (`SiLU(g)·u`), clamped SwiGLU (`g` clamped above at 7, `u`
clamped to `±7`), and ReLU². Report the max absolute value and the 99.99th percentile of each.

**What you should find** `[M]` (measured here 2026-07-26, gfx1151, one run each, seed 0):

- End-to-end max relative error is roughly `1.0e−2` naive and `7.2e−3` with the upcast — only
  a ~30% improvement, because the *output* is bf16 either way and bf16's own representation
  error (`2^-8 ≈ 3.9e−3` per element) dominates.
- The reduction alone tells a different story: `1.5e−3` to `2.6e−3` relative error in bf16
  versus `1.7e−5` to `1.7e−4` in fp32 — **one to two orders of magnitude**. That is what line
  60 is protecting, and it is a claim about the *sum over d*, so it gets worse as `d` grows.
- The clamp is bit-identical to plain SwiGLU at input std 1.0 (12.3 vs 12.3) and cuts the
  extreme 5x at std 4.0 (245.1 → 49.0). See the table in the math section.

**The point:** the folk claim "you must upcast RMSNorm to fp32 or bf16 breaks it" is directionally
right and quantitatively sloppy. Say instead: the upcast buys two orders of magnitude on the
reduction, and the output precision is bounded by bf16 regardless. Write your own numbers down —
they feed `ASSUMPTIONS.md: bf16-numerics-unproven`, which names RMSNorm explicitly.

**CPU fallback:** identical, set `device='cpu'`, drop the sample count in part two to `1e5`.
The numbers should agree to within run-to-run variation; if they *do not*, you have found a
gfx1151 numerics bug and that is a far more valuable result than the exercise.

### Exercise B — Reproduce the √l stream growth and separate the four placements

**Difficulty: easy-moderate. Runtime: under 30 seconds, CPU only. Writing it: 45–60 minutes.**

Build a linear surrogate of a residual stack: `L = 48` "sublayers", each a random matrix
`W ~ N(0, 1/d)` at `d = 512`, so each sublayer's output has unit RMS when its input does.
Start from an RMS-normalized batch of 256 vectors. Implement **four** wirings — and get the
two post-norms right, because that distinction is half the point:

```
pre        :  f = RMSNorm(x) @ W            ;  x ← x + f
post_res   :  f = RMSNorm(x @ W)            ;  x ← x + f        (OLMo 2)
post_2017  :  f = x @ W                     ;  x ← RMSNorm(x + f)
peri       :  f = RMSNorm(RMSNorm(x) @ W)   ;  x ← x + f        (Gemma 3)
```

At each layer record `RMS(x)` and the contribution ratio `RMS(f)/RMS(x)`. Plot both against
layer index for all four. Then re-run everything with the sublayer matrices scaled by `2.0`,
and re-run `pre` with them scaled by `1/√L` (depth-scaled gain).

**What you should find** `[M]` (measured here 2026-07-26, CPU, seeded generator, seed 0, one
run per arm — full table in the math section):

- Pre-norm `RMS(x_l)` tracks `√(1+l)` to three significant figures, and the contribution ratio
  tracks `1/√l`, reaching `0.1441` at `l = 48` against a prediction of `0.1443`.
- **At unit gain, `pre`, `post_res` and `peri` are all identical** — 6.995 at `l = 48`. The
  extra norm does nothing when the sublayer output is already unit-RMS.
- At gain 2.0 they split: `pre` grows to 13.871 (prediction `√(1+48·4) = 13.892`) while
  `post_res` and `peri` both stay at 6.995 (prediction `√49 = 7.000`).
- `post_2017` pins `RMS(x)` at exactly 1.000 at every depth *and* holds the contribution ratio
  at ~1.0 (gain 1) or ~2.0 (gain 2) — the sublayer write is always as large as the whole bus.
- Depth-scaled gain `1/√48` flattens pre-norm growth from 6.995 to 1.418 at `l = 48`.

**The point, and the honest limit.** You have derived, predicted and measured the mechanism
behind four norm-placement positions in a script with no training loop, no loss and no
optimizer. And you have found that the surrogate **cannot distinguish `post_res` from `peri`
at all** — identical to four decimal places in every column — because a linear map's output
direction does not depend on its input scale, which is the only thing those two wirings
disagree about. Whatever separates OLMo 2 from Gemma 3 lives in the nonlinearity and the
optimizer, not in the forward statistics of a linear stack. Knowing which of your questions a
cheap experiment can answer, and which it silently cannot, is the skill being exercised here.

**GPU version:** set `device='cuda'` and raise `d` to 4096 and the batch to 4096 to see
whether the fits hold at realistic width. Still under a minute.

### Exercise C — The two-axis test: does QK-norm buy loss, or buy learning rate?

**Difficulty: moderate. Machine time: ~21 minutes GPU or ~17 minutes CPU, both measured below,
plus dataset prep and process startup — call it 30 minutes unattended either way. Setup and
analysis: 1.5–2 hours the first time, and that is the real cost.**

This is the experiment the module argues the literature is missing, at the smallest scale that
can show it.

Copy nanoGPT into a scratch directory and prepare the `shakespeare_char` dataset per
`research/reference/training/nanogpt/README.md`. Patch `CausalSelfAttention` to apply an
RMSNorm over `head_dim` to `q` and `k` after the reshape into head layout — about six lines,
and `models/laguna-s/modeling_laguna.py:398-399,421-422` is the reference implementation.
Gate it on a config flag so both arms come from one file.

Run a learning-rate ladder — `1e−3, 2e−3, 4e−3, 8e−3, 1.6e−2, 3.2e−2` — for **400 iterations
each**, both arms, at the published 6-layer/384-channel/256-context config
(`research/reference/training/nanogpt/config/train_shakespeare_char.py:22`). You are not
training to convergence; you are finding where each arm blows up, and divergence shows up in
the first few hundred steps. Record, per run: whether the loss went NaN or exceeded its
initial value, and the loss at step 400.

Report two numbers per arm:
1. **Maximum stable learning rate** — the highest rung that did not diverge.
2. **Loss at step 400 at the highest LR where *both* arms survived.**

**Predicted result** `[A]`, medium confidence, from the published pattern: QK-norm's max
stable LR is one to two rungs higher, and the loss at the common LR is within noise. If you
see that, you have measured a stability control. If the loss at the common LR is
*meaningfully* better for QK-norm, you have found something the ablation literature says
should not happen at conservative LR, and it is worth pre-registering properly and running
with ≥3 seeds before believing it. **One seed is an anecdote by this lab's own standard** —
this exercise is a method demonstration, not a result.

**Runtime, measured, not estimated** `[M]` (2026-07-26, this machine): a 10.80M-parameter
model of exactly that shape at batch 64, bf16 autocast, runs at **259.4 ms/step** on the
8060S, so 400 iterations is **1.73 minutes** and the full 12-run ladder is about **21
minutes** plus process startup. For reference, the full published 5000-iteration run to the
1.4697 target (`research/reference/training/nanogpt/README.md:51`) is **21.6 minutes** on this
GPU against ~3 minutes on an A100.

**CPU fallback** `[M]`: the published CPU recipe
(`research/reference/training/nanogpt/README.md:85` — 4 layers, 4 heads, 128 channels,
block 64, batch 12, target loss 1.88) runs at **204.4 ms/step**, so 400 iterations is **1.4
minutes** and the 12-run ladder is about **17 minutes**. The absolute LR values shift at the
smaller width — that is expected and is exactly what μP exists to fix `[C]`
([2203.03466](https://arxiv.org/abs/2203.03466)) — but the *ordering* between arms is the
result you want, and it survives.

**Caveat you should hit and should not paper over:** `F.scaled_dot_product_attention` on this
stack emits "Flash / Mem Efficient attention on Current AMD GPU is still experimental"
warnings `[M]` and Flash Attention 2 is unavailable on gfx1151. Both arms use the same
backend, so the comparison is internally valid, but a max-stable-LR number measured under a
fallback attention kernel is not automatically transferable to another machine. Say so when
you write it up.

---

## Self-check

1. RMSNorm removes the mean subtraction and the bias from LayerNorm. Compute the parameter and
   FLOP savings for Laguna specifically, then state in one sentence why neither is the reason
   RMSNorm won.

2. `x = (2, −1, 0, 3)`. Compute `LayerNorm(x)` and `RMSNorm(x)` with unit gain and no bias.
   Then compute `‖LayerNorm(x)‖₂` and `‖RMSNorm(x)‖₂`. Why are they equal, and what does that
   fact let you predict about attention logits in a QK-normed model?

3. Three 48-layer stacks start from a unit-RMS residual stream: pre-norm `x + F(N(x))`,
   OLMo-2-style post-norm-in-residual `x + N(F(x))`, and 2017 post-norm `N(x + F(x))`. Assume
   each sublayer emits unit-RMS output when handed a unit-RMS input. Without running anything,
   give `RMS(x_48)` and the layer-48 contribution ratio `RMS(f)/RMS(x)` for each. Which one is
   the *curse of depth*, which one needed warmup, and what would you have to change about the
   sublayer before the first two stopped looking identical?

4. You run two 100M models to convergence, one with QK-norm and one without, both at
   `lr = 3e−4`, and the losses are identical to three decimal places. A colleague concludes
   QK-norm is useless. What is wrong with the conclusion, and what single additional axis
   would settle it?

5. Laguna sets `intermediate_size = 12288` with `hidden_size = 3072` and uses a three-matrix
   gated MLP. Compute the dense MLP's parameter count, compute what it would be under the
   `8/3` convention, and explain in one sentence what the `8/3` convention is *for* — and why
   its absence here is not a bug.

6. Mnemosyne is considering an eviction policy that ranks cached tokens by the L2 norm of
   their key vector. Proteus has QK-norm enabled before RoPE. State the concern precisely, in
   terms of the invariant from question 2, and give the cheapest measurement that would
   confirm or kill it.

---

## What is still unsolved here

**Norm placement has seven shipping answers and zero matched-budget head-to-heads above ~3B.**
Pre-norm (Laguna, Qwen3), post-norm-in-residual (OLMo 2), peri-norm (Gemma 3), depth-scaled
gain, Mix-LN, DyT, and TaperNorm are all live in 2026. Each is supported by a *mechanism*
argument about forward-pass statistics or gradient flow, and each mechanism is real — Exercise
B reproduces four of them. What is missing is a controlled comparison at matched tokens and
matched parameters with the LR frontier reported for each arm. `[C]`
([2603.15389](https://arxiv.org/abs/2603.15389)) further reports that sparsity interacts with
the depth effect, so the dense answer may not be the MoE answer, and Proteus is planned as a
sparse model.

**Whether normalization is needed at all is now a live question rather than a rhetorical
one.** DyT `[C]` ([2503.10622](https://arxiv.org/abs/2503.10622)) replaces the norm with an
elementwise `tanh(αx)` and matches on vision and language benchmarks; TaperNorm `[C]`
([2602.10408](https://arxiv.org/abs/2602.10408)) removes internal norms progressively with
small loss increases and 1.18x decode throughput, while finding the *final* norm irreplaceable
because it anchors pre-logit scale. Neither has been tested at frontier decoder scale. If
either holds up, the entire placement debate becomes a question about one norm rather than
`2L+1` of them.

**QK-norm's quality-versus-stability status is unresolved and its long-context interaction is
worse than unresolved — it is barely studied.** The published ablations put the benefit at
high learning rate; at least one 2026 model dropped it citing long-context degradation; and it
is structurally incompatible with MLA, which motivated a dedicated 2026 reconciliation `[C]`
([2606.16310](https://arxiv.org/abs/2606.16310)). Nobody has published the max-stable-LR
frontier that would settle the first question, which is why Exercise C is a method
demonstration rather than a replication.

**Activation stability is an open 2026 front with no independent replications.** SwiGLU's
quadratic tail is real and measurable (Exercise A), and there are at least three responses in
play: clamp it (gpt-oss, `swiglu_limit = 7.0`), replace it with a bounded-growth function
(PowLU `[C]` [2605.25704](https://arxiv.org/abs/2605.25704)), or replace it with a trainable
integral-derived function (xIELU `[C]`
[2411.13010](https://arxiv.org/abs/2411.13010)). Each is reported by its originating group
only. **Nobody has published a controlled comparison of clamping the activation versus
normalizing the QK logits** — two clamps on two different unbounded quantities in the same
block, addressing the same failure mode. That is a cheap ablation on our rig and it is
genuinely open.

**The attribution problem is the deepest one, and it is methodological rather than
architectural.** Every change in this module is reported as an improvement, and the reports do
not separate "moved the loss" from "moved the safe operating envelope." That is the same
weakness this lab has already named in the KV-compression literature — reporting that
something helped without isolating which mechanism did it. The two-axis sweep is not
expensive at our scale. It is simply not the convention, and running it is a small, real,
available contribution.

**Two things are unmeasured on our own instrument, and both gate everything above.**
`ASSUMPTIONS.md: bf16-numerics-unproven` is still `untested` and names RMSNorm by name;
Exercise A is a direct contribution to closing it. And the interaction between QK-norm and
key-norm-based eviction (`[A]`, medium confidence, above) is, as far as I can find, unpublished
in either direction — the eviction literature and the architecture literature do not cite each
other on this point. It is checkable from a checkpoint's weights alone, which makes it the
cheapest genuinely-new result in this module.

---

## Answers to the self-check

**1.** Parameters: LayerNorm would cost `97 sites × 2 × 3072 = 595,968`; RMSNorm costs
`97 × 3072 = 297,984`. The saving is ~298k out of 117.5 B, or 0.00025%. FLOPs: the extra work
is about `2d` operations per token per site, so `97 × 8192 × 2 × 3072 ≈ 4.88 GFLOP` against a
forward pass of roughly `2 × 8.1e9 × 8192 ≈ 132.7 TFLOP` — 0.0037%. Neither is why it won.
RMSNorm won because it is a strictly simpler function that was measured to cost nothing in
quality; the 7%–64% speedup in the original paper `[C]`
([1910.07467](https://arxiv.org/abs/1910.07467)) was measured on 2019-era RNNs and does not
transfer to a decoder where norms are a rounding error.

**2.** `μ = 1`, centred `(1, −2, −1, 2)`, `σ² = 2.5`, `σ = 1.5811`, so
`LayerNorm(x) = (0.6325, −1.2649, −0.6325, 1.2649)`. `RMS(x) = √3.5 = 1.8708`, so
`RMSNorm(x) = (1.0690, −0.5345, 0.0000, 1.6036)`. Both have `‖·‖₂² = 4.0`, i.e. length
`√d = 2`. They are equal because both operations divide by a quantity proportional to the
length of the (respectively centred or raw) vector, and normalizing by the root-mean-square of
`d` components forces the L2 length to exactly `√d`. The prediction: in a QK-normed model the
query and key vectors have fixed length `√d_head` (up to the learned gain), so by
Cauchy–Schwarz the logit `q·k/√d` is bounded by `g_max² · √d_head` — for Laguna's
`head_dim = 128`, about `11.31` at unit gain. Attention logits become an architectural
constant rather than a runtime property.

**3.** Pre-norm: `RMS(x_48) ≈ √49 = 7.0`, contribution ratio `≈ 1/√48 = 0.144`. Post-norm-in-
residual: **the same**, 7.0 and 0.144 — the stream is still a sum of 48 unit-RMS independent
terms, and normalizing the sublayer's output rather than its input does not change that.
2017 post-norm: `RMS(x_48) = 1.0` exactly, because the stream is renormalized every layer,
with contribution ratio `≈ 1.0`. `[M]` measured 6.995 / 6.995 / 1.000 respectively.

The `0.144` shared by the first two is the *curse of depth* `[C]`
([2502.05795](https://arxiv.org/abs/2502.05795)): a layer writing 14% of the bus level has a
block Jacobian approaching the identity, so it contributes little and trains poorly. The 2017
number is the opposite pathology and is why that wiring needed warmup — the sublayer's write
is always as large as the whole stream and there is no un-normalized identity path, so the
gradient picks up a `1/σ` factor at every layer on the way back. OLMo 2's post-norm keeps the
norm *inside* the skip and does not inherit this.

To separate pre from post-norm-in-residual you must break the assumption that the sublayer
emits unit RMS: give it a gain. `[M]` at sublayer gain 2.0 the pre-norm stream reaches 13.871
(prediction `√(1+48·4) = 13.892`) while post-norm-in-residual stays at 6.995, because
normalizing the *output* caps every write at unit size regardless of what the sublayer wanted
to emit. That is the entire variance-control argument, and it is also why a linear surrogate
cannot tell post-norm-in-residual apart from peri-norm at all.

**4.** The conclusion tests only one axis. A stability control is not supposed to move the loss
at a learning rate where both arms are already stable — that is the definition of a stability
control, and `3e−4` is conservative. The missing axis is the learning rate itself: sweep both
arms up the LR ladder and report the highest rung each survives. If QK-norm survives one or two
rungs higher, it is buying headroom, and since a practitioner will then *use* the higher LR, it
buys loss-per-wall-clock even though it buys nothing at fixed LR. Exercise C is exactly this
measurement.

**5.** Gated MLP with three matrices: `3 × 3072 × 12288 = 113,246,208` parameters. Under the
`8/3` convention the inner width would be `8 × 3072 / 3 = 8192`, giving
`3 × 3072 × 8192 = 75,497,472`. Laguna is 1.5x that. The `8/3` convention exists to make a
gated-versus-ungated comparison *parameter-matched*: an ungated FFN at width `4d` costs
`2·d·4d = 8d²`, and a gated FFN at width `8d/3` costs `3·d·8d/3 = 8d²` — identical. Its absence
in Laguna is not a bug because Laguna is not running that comparison; in an MoE the expert
width is set by a sparsity/granularity argument instead (`moe_intermediate_size = 1024 = d/3`,
256 experts, top-10), and the single dense layer-0 MLP just takes the conventional `4d` width
with no ungated baseline to be parity-matched against. The convention only earns its keep
inside a controlled ablation — which is exactly where Proteus will need it, and why
`ffn_width_rule` is a config field rather than a constant.

**6.** The concern: QK-norm forces every normalized key to length exactly `√d_head` before the
gain, and RoPE is a rotation, which preserves L2 length. So the cached key's norm is
`‖g_k ⊙ (k/RMS(k))‖₂`, which lies in `[min|g_k|·√d, max|g_k|·√d]` — its entire dynamic range
across tokens is controlled by the *learned per-channel gain*, not by the token. If `g_k` is
near-uniform, all cached keys have nearly identical L2 norm and an L2-norm eviction score `[C]`
([2406.11430](https://arxiv.org/abs/2406.11430)) has almost no signal to rank on. Cheapest
measurement: load only the 48 `k_norm.weight` tensors from a QK-normed checkpoint — 128 floats
each, trivial — and report `max|g_k| / min|g_k|` per layer. A ratio near 1 kills the policy for
that model; a large ratio means the gain itself is doing the discrimination, which is a
different and also interesting result. Note the local Laguna clones hold git-LFS pointers
rather than weights, so this needs a deliberate single-shard fetch or a smaller QK-normed
model. Scale-invariant scorers such as KeyDiff `[C]`
([2504.15364](https://arxiv.org/abs/2504.15364)) are not affected by this at all.

---

## Sources

**Code read locally** (relative to `research/reference/`; clones gitignored, rebuild with
`scripts/fetch_reference.sh`, revisions in `PROVENANCE.md`): `models/laguna-s/config.json`,
`configuration_laguna.py`, `modeling_laguna.py`, `model.safetensors.index.json`;
`architecture/transformers/src/transformers/models/{laguna,olmo2,gemma3}/modeling_*.py`;
`architecture/gpt-oss/gpt_oss/torch/model.py`; `training/nanogpt/{model.py,train.py,README.md,
config/train_shakespeare_char.py}`.

**Lab documents this module must stay consistent with:**
`research/notes/transformer-state-of-the-art.md` (sections on norms and activations),
`research/notes/pretraining-recipes.md` (the stability-controls glossary and the bf16 gate),
`research/memory/kv-compression-and-eviction.md` (attention-free eviction scoring),
`research/reference/CODE_MAP.md`, `research/reference/papers/README.md`, `ASSUMPTIONS.md`.

**Papers.** Every arXiv id below was resolved against arxiv.org on 2026-07-26, either by this
module's own fetch or by `research/reference/papers/README.md` on the same date.

*Normalization.* [1607.06450](https://arxiv.org/abs/1607.06450) Layer Normalization ·
[1910.07467](https://arxiv.org/abs/1910.07467) Root Mean Square Layer Normalization ·
[2002.04745](https://arxiv.org/abs/2002.04745) On Layer Normalization in the Transformer
Architecture · [2412.13795](https://arxiv.org/abs/2412.13795) Mix-LN ·
[2501.00656](https://arxiv.org/abs/2501.00656) 2 OLMo 2 Furious ·
[2502.02732](https://arxiv.org/abs/2502.02732) Peri-LN ·
[2502.05795](https://arxiv.org/abs/2502.05795) The Curse of Depth in Large Language Models ·
[2503.10622](https://arxiv.org/abs/2503.10622) Transformers without Normalization (DyT) ·
[2602.10408](https://arxiv.org/abs/2602.10408) Gated Normalization Removal and Scale Anchoring
in Pre-Norm Transformers (TaperNorm) ·
[2603.15389](https://arxiv.org/abs/2603.15389) When Does Sparsity Mitigate the Curse of Depth
in LLMs

*QK-norm and attention stability.*
[2010.04245](https://arxiv.org/abs/2010.04245) Query-Key Normalization for Transformers ·
[2302.05442](https://arxiv.org/abs/2302.05442) Scaling Vision Transformers to 22 Billion
Parameters · [2606.16310](https://arxiv.org/abs/2606.16310) QK-Normed MLA ·
[2402.17762](https://arxiv.org/abs/2402.17762) Massive Activations in Large Language Models

*Activations.* [1606.08415](https://arxiv.org/abs/1606.08415) Gaussian Error Linear Units ·
[1710.05941](https://arxiv.org/abs/1710.05941) Searching for Activation Functions (Swish) ·
[2002.05202](https://arxiv.org/abs/2002.05202) GLU Variants Improve Transformer ·
[2411.13010](https://arxiv.org/abs/2411.13010) Deriving Activation Functions Using Integration
(xIELU) · [2605.25704](https://arxiv.org/abs/2605.25704) PowLU

*Models cited for their configuration choices.*
[2503.19786](https://arxiv.org/abs/2503.19786) Gemma 3 ·
[2505.09388](https://arxiv.org/abs/2505.09388) Qwen3

*Memory-track connections.*
[2406.11430](https://arxiv.org/abs/2406.11430) A Simple and Effective L2 Norm-Based Strategy for
KV Cache Compression · [2504.15364](https://arxiv.org/abs/2504.15364) KeyDiff

*Methodology.* [2203.03466](https://arxiv.org/abs/2203.03466) Tensor Programs V (μP)
