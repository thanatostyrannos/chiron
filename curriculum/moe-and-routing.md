---
title: MoE and routing — the load balancer that changes what its backends become
version: 1.0.0
track: B — Modern architecture
written: 2026-07-26
prereqs: transformer-forward-pass, attention-variants
---

# MoE and routing

## What this module settles

A sparse Mixture-of-Experts layer replaces one wide feed-forward network with many
narrow ones plus a learned dispatcher, so that parameter count and per-token compute
stop being the same number — and once they are decoupled, "how big is this model" has
two answers that differ by an order of magnitude. The dispatcher is a load balancer
you already know how to reason about, right up to the point where it isn't: an
overloaded backend in your world serves requests more slowly, whereas an overloaded
expert **permanently changes what it learns to be**, which makes imbalance an
irreversible allocation of representational capacity rather than a transient queueing
problem. By the end you will be able to read Laguna's twelve-line router, state
exactly which of its five knobs are forced and which are inherited convention, derive
the hard saturation threshold of its load-balancing controller from the code, and
demonstrate on your own machine that the industry-standard auxiliary loss can be
driven to its theoretical minimum without improving balance at all.

---

## Theory in plain language

### What problem this solves, and what it replaced

A transformer layer is attention followed by a feed-forward network (FFN). The FFN is
where most of the parameters live and where most of the per-token FLOPs go. It is a
fixed pipeline: every token pays for every parameter.

That coupling is the constraint MoE breaks. `[C]` The 2017 sparsely-gated MoE layer
(1701.06538, 2017-01-23) proposed replacing one FFN with *E* independent FFNs
("experts") and a small trained network (the "router" or "gate") that picks *k* of them
per token. Only the chosen *k* run. Total parameters scale with *E*; per-token compute
scales with *k*. Set *E* = 256 and *k* = 10 and you hold roughly 25× the parameters
while doing roughly the same arithmetic per token.

What it replaced, concretely: the previous way to add capacity was to make the FFN
wider or the model deeper, both of which raise per-token cost linearly. MoE is the
first mechanism that adds capacity at sublinear compute cost. The bill is not waived,
it is **moved** — from FLOPs to memory capacity and memory bandwidth. For a lab whose
hardware is capacity- and bandwidth-bound rather than FLOPs-bound, that relocation is
the whole story, and it is why this module sits in the architecture track but keeps
pointing at the memory track.

### The systems bridge

A top-*k* router is a **content-addressed layer-7 load balancer** in front of a pool of
*E* identically-shaped backends. Per request (per token) it scores every backend, picks
the *k* highest, fans the request out to all *k*, and returns a weighted merge of the
responses. The score is a dot product against a learned per-backend key vector, so it
is consistent hashing where the hash function is trained rather than fixed.

Three things about that analogy are load-bearing, and you should lean on them:

- **Imbalance is the central operational problem**, and the fix is a per-backend weight
  adjustment applied *before* selection. Megatron's update rule is literally weighted
  round-robin with an integral term: `b += u * sign(mean_load - load)`
  (`architecture/megatron-lm/megatron/core/transformer/moe/moe_utils.py:1165`).
- **Tail latency is set by the hottest backend, not the average.** The layer cannot
  finish until every expert's matrix multiply finishes. On one device that is a ragged
  batch of *E* GEMMs whose largest tile sets the kernel's runtime.
- **Capacity is a real configured quantity with an overflow policy.** Megatron exposes
  `moe_expert_capacity_factor` and a `drop_policy` of `probs` or `position`
  (`architecture/megatron-lm/megatron/core/transformer/transformer_config.py:903`).

### Where the analogy breaks — three places, all of them the point

**Break one: an overloaded expert does not get slower, it becomes a different expert.**
The router's dispatch decision *is* the training signal for the expert. An expert
receiving 40× its share of tokens receives 40× the gradient and generalizes; an expert
receiving none receives no gradient and stays at its random initialization forever.
There is no drain-and-restart. Load imbalance is not a latency problem that resolves
when traffic drops — it is a permanent allocation of the model's representational
budget, decided early and self-reinforcing (more traffic → better expert → more
traffic). `[C]` This failure has been named since the original MoE paper (1701.06538)
and was the motivating problem for Switch Transformers `[C]` (2101.03961, 2021-01-11).

This break has a corollary that catches systems people: **the load-balancing controller
runs during training and is then frozen.** In Laguna the correction bias is
`requires_grad=False` and has no update rule at inference at all
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:170`).
What ships is a deploy-time snapshot of a control loop that no longer runs. You are
shipping the PID controller's last output as a constant.

**Break two: a dropped token returns no error.** An overloaded web backend returns 503
and the client retries. A token that exceeds an expert's capacity has its routing
probability zeroed and its mask cleared
(`architecture/megatron-lm/megatron/core/transformer/moe/moe_utils.py:958`); it then
passes through that layer on its residual connection alone. No exception, no counter in
the loss, nothing in the output distribution that says "this activation is missing a
layer's worth of FFN." It is a correctness fault that presents as a slightly worse
model. Your entire instinct that failures are observable does not transfer.

**Break three: total parameters are not a working set you can page.** Here is the
second bridge, and it also breaks. You will want to read total-vs-active parameters as
dataset-on-disk vs working-set-in-RAM, with the router as a prefetcher. It is a
reasonable first pass — expert offloading is a real research area `[C]` (2502.05370,
2025-02-07). But a KV cache entry and an expert weight sit on opposite sides of the
axis `research/memory/memory-taxonomy.md` uses to partition memory:
**reconstructibility**. Evicting a KV entry costs a recompute; the information is
recoverable exactly from the tokens. Evicting an expert weight costs a *fetch from a
slower tier*, and it cannot be recomputed at all. So the expert-offload literature and
the KV-eviction literature share a vocabulary — prefetch, hit rate, tiering — while
solving genuinely different problems. On this machine the break is sharper still:
unified memory means there is no separate weight tier, so the hot-expert set and the KV
cache contend for the same measured **≥62 GiB fast tier at ~200 GB/s** `[M]`
(`notebook/uma-carveout-controls-fast-tier.md`, 2026-07-26, single run per arm).

---

## The math that actually matters

Notation, fixed once. Every symbol is translated.

| Symbol | Words |
|---|---|
| *x* | the hidden state of one token, a vector of length *d* (Laguna-S: *d* = 3072) |
| *E* | number of routed experts (Laguna-S: 256) |
| *k* | experts selected per token, "top-k" (Laguna-S: 10) |
| *W*<sub>r</sub> | router weight matrix, shape *E* × *d*; row *i* is expert *i*'s learned key vector |
| *z* | router logits, a vector of length *E*; *z*<sub>i</sub> = row *i* of *W*<sub>r</sub> dotted with *x* |
| *s*<sub>i</sub> | expert *i*'s **score** after the nonlinearity (sigmoid or softmax) |
| *b*<sub>i</sub> | expert *i*'s **correction bias** — a learned-by-control-rule scalar, not by gradient |
| *w*<sub>j</sub> | the **combination weight** applied to the *j*-th selected expert's output |
| *f*<sub>i</sub> | fraction of dispatched (token, expert) assignments that landed on expert *i*; Σ *f*<sub>i</sub> = 1 |
| *P*<sub>i</sub> | mean over tokens of the router's probability for expert *i*; Σ *P*<sub>i</sub> = 1 |
| *T* | tokens in the batch |
| *u* | bias update rate (Megatron default 1e-3, `transformer_config.py:800`) |

### Scoring: softmax versus sigmoid, and why it is not a free choice

**Softmax** turns logits into a probability distribution:

&nbsp;&nbsp;&nbsp;&nbsp;*p*<sub>i</sub> = exp(*z*<sub>i</sub>) / Σ<sub>j</sub> exp(*z*<sub>j</sub>)

In words: exponentiate every logit, then divide each by the total, so the *E* scores sum
to exactly 1. The experts compete for a fixed probability budget.

**Sigmoid** scores each expert independently:

&nbsp;&nbsp;&nbsp;&nbsp;*s*<sub>i</sub> = 1 / (1 + exp(−*z*<sub>i</sub>))

In words: squash each logit on its own into the open interval (0, 1). Nothing sums to
anything. There is no budget.

The consequence is visible in the derivatives. For softmax:

&nbsp;&nbsp;&nbsp;&nbsp;∂*p*<sub>i</sub> / ∂*z*<sub>j</sub> = *p*<sub>i</sub> (δ<sub>ij</sub> − *p*<sub>j</sub>)

where δ<sub>ij</sub> is 1 when *i* = *j* and 0 otherwise. So for *i* ≠ *j* the derivative
is −*p*<sub>i</sub>*p*<sub>j</sub>, which is **not zero**: nudging expert *j*'s logit
changes expert *i*'s score. For sigmoid:

&nbsp;&nbsp;&nbsp;&nbsp;∂*s*<sub>i</sub> / ∂*z*<sub>j</sub> = *s*<sub>i</sub>(1 − *s*<sub>i</sub>) when *i* = *j*, and **exactly 0** otherwise.

Fully decoupled. In load-balancer terms: under softmax, "prefer backend 7" is not
expressible — every adjustment reweights the entire pool. Under sigmoid it is a clean
per-backend offset.

This is why Megatron **refuses at config-validation time** to combine an expert bias
with softmax scoring: *"Expert bias for aux-loss-free routing only supports 'sigmoid'
and 'sqrtsoftplus' score functions"*
(`architecture/megatron-lm/megatron/core/transformer/transformer_config.py:2289`) `[C]`.
So "Laguna uses sigmoid" and "Laguna uses aux-loss-free balancing" are **one decision
stated twice**, and an ablation arm that varies one without the other is not a valid
arm.

There is a second, purely arithmetic reason worth carrying, because it tells you where
the controller's gain comes from. Under softmax the scores sum to 1, so the **mean
score is 1/*E*** and shrinks as you add experts: 1/64 = 0.0156 at *E* = 64, and
1/256 = 0.0039 at *E* = 256. A bias update rate *u* tuned at *E* = 64 is four times too
coarse at *E* = 256. Worse, softmax scores sharpen over the course of training, so the
controller's effective gain drifts by orders of magnitude during a single run. Under
sigmoid the score scale is (0, 1) regardless of *E* and regardless of training progress.
`[A]` High confidence in the arithmetic; the claim that this is *the designers'* reason
is not documented anywhere and is mine.

### Selection and combination are deliberately different quantities

Laguna's router, read line by line
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:178-189`):

1. *z* = *W*<sub>r</sub> *x*, **computed in fp32** regardless of model dtype (`:178`).
2. Optional softcapping *z* ← tanh(*z*/*c*) · *c*, **skipped because *c* = 0** (`:180-181`).
3. *s* = sigmoid(*z*) (`:183`).
4. Selection score *t*<sub>i</sub> = *s*<sub>i</sub> + *b*<sub>i</sub> (`:185`).
5. Selected set = indices of the top *k* values of *t* (`:186`).
6. Combination weights gathered from ***s***, **not from *t*** (`:187`).
7. Normalize: *w*<sub>j</sub> = *s*<sub>i(j)</sub> / Σ<sub>j'</sub> *s*<sub>i(j')</sub> (`:188`).

Step 6 is the one to internalize. The bias shifts **who gets picked**; it never touches
**how much their output counts**. In load-balancer terms: the health-check weight
changes the routing decision but is not allowed to distort the response body. If the
bias were also applied to the combination weight, the balancing controller would be
directly editing the model's output, and every rebalancing step would be a quality
regression.

Step 7 costs something real. Sigmoid scores carry *absolute magnitude* — "this token
needs expert 41 a lot" versus "a little" — and dividing by the sum discards it. After
normalization Σ *w*<sub>j</sub> = 1 by construction for every token in the corpus, so the
routed branch always contributes a convex combination with a fixed total gain. The model
cannot express "this token barely needs the sparse FFN." Whether that matters is
untested publicly and is among the cheapest ablations available (`norm_topk_prob`).

### The bias controller, and its hard saturation threshold

The update rule (`moe_utils.py:1163-1165`), in words: average the token counts across
experts, subtract each expert's own count to get an error signal, take only the **sign**
of that error, and step the bias by ±*u*.

&nbsp;&nbsp;&nbsp;&nbsp;*b*<sub>i</sub> ← *b*<sub>i</sub> + *u* · sign( mean(load) − load<sub>i</sub> )

This is a bang-bang integral controller. It has no proportional term: an expert that is
1% overloaded and an expert that is 400% overloaded get the identical −*u* step.

Now derive the failure mode. Expert *i* outranks expert *j* exactly when

&nbsp;&nbsp;&nbsp;&nbsp;*s*<sub>i</sub> + *b*<sub>i</sub> > *s*<sub>j</sub> + *b*<sub>j</sub>&nbsp;&nbsp;⟺&nbsp;&nbsp;*b*<sub>i</sub> − *b*<sub>j</sub> > *s*<sub>j</sub> − *s*<sub>i</sub>

Because the bias is added to the **sigmoid output** and not to the logits (`:185`),
every *s* lies strictly inside (0, 1). Therefore the right-hand side lies strictly
inside (−1, 1). So:

> **If max<sub>i</sub> *b*<sub>i</sub> − min<sub>j</sub> *b*<sub>j</sub> ≥ 1.0, then expert *i* outranks expert *j* for
> every possible token, regardless of content. Routing has become content-independent
> for that pair.**

That is a derivation from the code, not a literature claim, and it hands you a complete
saturation detector: one scalar gauge per layer, `spread(e_score_correction_bias)`, with
a red line at exactly 1.0.

How fast can you get there? Each step moves each bias by exactly *u*, so the spread grows
by **at most 2*u* per step** (one bias up, one down). Starting from the zeros
initialization (`:170`), reaching spread 1.0 requires at least

&nbsp;&nbsp;&nbsp;&nbsp;*T*<sub>min</sub> = 1 / (2*u*) = 1 / (2 × 10<sup>−3</sup>) = **500 steps**

of consistently one-directional pressure. Saturation is therefore always preceded by a
long, visible, monotone drift — which is exactly what makes it a good gauge.

> **Refinement of an existing lab document.** `research/notes/moe-routing-and-failure-modes.md`
> gives this bound as 1/*u* ≈ 1000 steps. That figure correctly bounds the travel of a
> *single* bias element across distance 1.0; the *spread* is a difference of two elements
> moving in opposite directions, so the correct bound for the spread gauge is 1/(2*u*).
> Both statements are true of different quantities. Use 1/(2*u*) when setting an alert
> threshold — it is the conservative one.

### The auxiliary loss, and how to minimize it without doing anything

Switch Transformers' load-balancing loss `[C]` (2101.03961), implemented at
`architecture/megatron-lm/megatron/core/transformer/moe/moe_utils.py:56`:

&nbsp;&nbsp;&nbsp;&nbsp;*L*<sub>aux</sub> = α · *E* · Σ<sub>i</sub> *f*<sub>i</sub> · *P*<sub>i</sub>

In words: multiply each expert's *actual dispatch share* by its *average router
probability*, sum over experts, scale by the expert count and a coefficient α. At
perfect balance *f*<sub>i</sub> = *P*<sub>i</sub> = 1/*E*, so

&nbsp;&nbsp;&nbsp;&nbsp;*L*<sub>aux</sub>/α = *E* · Σ<sub>i</sub> (1/*E*)(1/*E*) = *E* · *E* · (1/*E*²) = **1**

Now the part nobody writes down. *f*<sub>i</sub> is a **hard count** from a top-k
selection: it is not differentiable, so no gradient flows through it. The optimizer sees
a function of *P* alone. Consider scaling every router logit by a factor γ > 0:

- **Selection is bit-identical.** Softmax is monotone and γ > 0 preserves order, so
  top-k(softmax(γ*z*)) = top-k(*z*). Therefore *f* is **exactly unchanged**.
- **P flattens.** As γ → 0, softmax(γ*z*) → uniform, so *P*<sub>i</sub> → 1/*E*.
- **The loss falls to exactly 1.** *L*<sub>aux</sub>/α → *E* · Σ<sub>i</sub> *f*<sub>i</sub> · (1/*E*) = Σ<sub>i</sub> *f*<sub>i</sub> = 1.

And at γ = 1 with any router worth the name, *P* is positively correlated with *f*
(the router routes to what it scores highly), so Σ *f*<sub>i</sub>*P*<sub>i</sub> > 1/*E* and
*L*<sub>aux</sub>/α > 1. So there is a smooth, monotone, always-available path that drives
the auxiliary loss to its perfect-balance value **while changing the load distribution
by exactly zero**. That is the degenerate minimum, and it is precisely what
Sigma-MoE-Tiny reported observing in layer 0 of a top-1-of-96 model — probabilities
optimized toward uniformity while dispatch fractions stayed highly non-uniform `[C]`
(2512.16248, 2025-12-18). You will reproduce it in the first exercise.

Two consequences.

- The true infimum is lower still: minimizing a linear function over the probability
  simplex puts all of *P* on the *least*-loaded expert, giving *L*<sub>aux</sub>/α =
  *E* · min<sub>i</sub> *f*<sub>i</sub> < 1. But reaching that vertex requires huge logits, which
  reorders top-k and changes *f*. `[A]` High confidence that the flat-*P* plateau is the
  reachable, stable solution and the vertex is not; this stability claim is reasoned,
  not measured.
- **Router z-loss pushes in the same direction.** z-loss = mean(logsumexp(*z*)²)
  (`moe_utils.py:146`) has gradient 2·logsumexp(*z*)·softmax(*z*)<sub>i</sub> with respect to
  *z*<sub>i</sub>, so it shrinks the largest logits hardest — it flattens *P*. `[A]` Medium-high
  confidence that combining z-loss with an auxiliary loss therefore makes the degenerate
  solution *cheaper* rather than more expensive. Untested. Cheapest test: log the aux
  loss, the z-loss, MaxVio, and the mean top-k score gap on the same axes during a run
  and check whether the two losses fall together while MaxVio does not move.

### Capacity factors and dropped tokens

Capacity exists for exactly one reason: fixed tensor shapes. Ragged expert batches mean
ragged GEMMs; a rectangle means a batched GEMM and a fixed-size all-to-all.

&nbsp;&nbsp;&nbsp;&nbsp;capacity = ceil( (*T* · *k* / *E*) · cf )

read directly off `moe_utils.py:990` (the call passes `num_tokens = T * router_topk`)
and `moe_utils.py:256` (the formula). In words: work out the average number of
assignments each expert would get under perfect balance, multiply by the capacity factor
cf, round up.

Worked example. *T* = 8192 tokens, *E* = 64, *k* = 8, cf = 1.25.

- Total assignments = *T* · *k* = 65,536.
- Mean per expert = 65,536 / 64 = **1024**.
- capacity = ceil(1024 × 1.25) = **1280**.
- Suppose the hottest expert actually wants 1900 assignments. Then
  MaxVio = (1900 − 1024) / 1024 = **0.855**, and 1900 − 1280 = **620 assignments are
  dropped** from that one expert.
- 620 / 65,536 = **0.95%** of all assignments silently vanish. They do not vanish
  uniformly; they concentrate on exactly the tokens the hot expert wanted.
- With `pad_to_capacity` set, the expert GEMM runs on *E* × capacity = 64 × 1280 =
  **81,920 rows**. That allocation is fixed by *T*, *E*, *k* and cf alone — 81,920/65,536
  = 1.25 = cf exactly — so **1 − 1/cf = 20% of expert compute is allocated to padding
  before you look at the data at all**. Imbalance only makes it worse: with 620 dropped,
  just 64,916 of the 81,920 rows carry a real assignment, so actual waste is 20.8%.

`[C]` MegaBlocks (2211.15841, 2022-11-29) removed the hyperparameter entirely by
reformulating the MoE FFN as block-sparse matmuls, which is why "dropless" is the modern
default. **Laguna's reference implementations are dropless by construction** — the
HuggingFace path loops over hit experts and `index_add_`s their outputs
(`modeling_laguna.py:219`), and the llama.cpp path builds a `build_moe_ffn` node graph
with no capacity concept at all
(`architecture/llama-cpp-laguna/src/models/laguna.cpp:285`).

### Total versus active parameters, computed not quoted

All figures below are arithmetic on `research/reference/models/laguna-s/config.json` at
revision `b0a9fd7c850e` `[M]`.

One routed expert is a SwiGLU FFN: gate, up, and down projections between *d* = 3072 and
*d*<sub>ff</sub> = 1024.

&nbsp;&nbsp;&nbsp;&nbsp;3 × 3072 × 1024 = **9,437,184 parameters per expert**

- Per MoE layer: 256 × 9,437,184 = **2,415,919,104**
- Across 47 MoE layers (layer 0 is dense, `mlp_only_layers: [0]`):
  47 × 2,415,919,104 = **113,548,197,888 ≈ 113.5 B routed parameters**
- Active routed per token: 10/256 of that = **4,435,476,480 ≈ 4.44 B**
- Routed-block sparsity: 10 / 256 = **3.91%**

Adding attention, embeddings, the shared experts, the routers and the dense layer gives
**117.5 B total / 8.41 B active, i.e. 7.16% of parameters active** `[M]`
(`research/notes/moe-routing-and-failure-modes.md`, arithmetic on the same artifact).

Two rows to read together:

- Active FFN width per MoE layer: (10 + 1) × 1024 = **11,264**
- Dense layer 0's FFN width: **12,288**

**A Laguna MoE layer activates 92% of the FFN width of its own dense layer, and in
exchange holds 23.4× more FFN parameters** (257 experts' worth held, 11 activated). That
is the sparsity trade with no press release attached: essentially the same activated
FLOPs as a conventional dense FFN, twenty-three times the parameters, the entire cost
moved from compute to memory.

**Granularity** is *d*<sub>ff,dense</sub> / *d*<sub>ff,expert</sub> = 12288 / 1024 = **12** `[M]`.
`[C]` DeepSeekMoE (2401.06066, 2024-01-11) argues finer experts increase the number of
expressible combinations; the fine-grained scaling law puts granularity in the exponent
`[C]` (2402.07871, 2024-02-12). The combinatorial claim is easy to check: choosing 10 of
256 gives C(256,10) = **2.79 × 10¹⁷** distinct routes per token per layer.

### Decode weight traffic is a coupon-collector problem

At batch size *B*, with a balanced router, each token independently selects expert *i*
with probability *k*/*E*. The probability that expert *i* is missed by all *B* tokens is
(1 − *k*/*E*)<sup>*B*</sup>, so the expected number of **distinct** experts whose weights
must be read in one decode step is

&nbsp;&nbsp;&nbsp;&nbsp;*N*(*B*) = *E* · ( 1 − (1 − *k*/*E*)<sup>*B*</sup> )

In words: every expert is either touched or not; the expected count of touched ones is
the expert count times the probability that at least one token picked it. Assumes a
perfectly balanced router and independent routing across tokens — both false in
practice, and the direction of the error matters (real routing is temporally correlated
within a sequence, so the real curve sits *below* this prediction; that is a testable
prediction, not a caveat).

For Laguna-S (*E* = 256, *k* = 10), with amortized per-token expert reads = *N*(*B*)/*B*:

| *B* | *N*(*B*) distinct experts | experts-worth of weights read **per token** |
|---:|---:|---:|
| 1 | 10.0 | 10.00 |
| 8 | 69.9 | 8.74 |
| 16 | 120.8 | 7.55 |
| 32 | 184.5 | 5.77 |
| 64 | 236.1 | 3.69 |
| 128 | 254.4 | 1.99 |
| 256 | 256.0 | 1.00 |

Setting *B* = *E*/*k* gives (1 − *k*/*E*)<sup>*E*/*k*</sup> ≈ 1/*e*, so *N* ≈ 0.632·*E*.
**The knee sits at *B* ≈ *E*/*k***, which for Laguna-S is **25.6 tokens**. A batch of
just 32 concurrent sequences already touches over 70% of all experts on every decode
step. MoE decode efficiency is therefore a step function of batch size in a way dense
decode is not, and the step is at a computable place.

---

## Why it matters for Proteus

Every knob named above is a config field, and by house rule the config surface *is* the
experimental surface. The useful partition is not "which knobs exist" but **which are
forced, which are inherited, and which are actually open**.

**Forced — treat as constraints, do not vary casually.**

- *Sigmoid scoring paired with a selection bias.* One decision, not two (Megatron raises
  on the illegal combination, `transformer_config.py:2289`). Any sweep is 2×N over
  {sigmoid+bias, softmax+aux} × other-axis, never 2×2×N.
- *fp32 router logits.* Both HuggingFace (`modeling_laguna.py:178`) and Megatron
  (`router.py:104`, and `router.py:259` for the bias itself, with the explicit comment
  *"to avoid routing errors when updating the expert_bias"*) refuse bf16 here. This is
  **the one place in the network where every reference implementation refuses bf16**,
  which lands directly on `bf16-numerics-unproven` in `ASSUMPTIONS.md` — currently
  untested on gfx1151.

**Inherited convention with no published ablation — prime targets.**

- `moe_routed_scaling_factor: 2.5` (`config.json:210`, applied at
  `modeling_laguna.py:250`). The same constant as DeepSeek-V3 `[C]` (2412.19437,
  2024-12-27). Nobody has published 1.0 versus 2.5 at matched everything. If it is a
  no-op, that is a publishable negative result about a number sitting in four frontier
  configs.
- `norm_topk_prob: true` (`config.json:25`). Tests whether discarding routing-confidence
  magnitude costs anything.
- Granularity 12. Compare against Qwen3-Next's 10 and Kimi Linear's 9 `[M]`, and against
  gpt-oss-20b's granularity of **1** paired with `router_aux_loss_coef: 0.9` `[M]` —
  three orders of magnitude above everyone else. Coarse experts and an aggressive balance
  penalty travel together; fine experts and a bias controller travel together. That
  co-variation is a hypothesis about mechanism, and it is testable at our scale.

**Genuinely open, and the most interesting thing about Laguna.**

`router_aux_loss_coef: 0.0` (`config.json:26`; the class default is 0.001 at
`configuration_laguna.py:113`). DeepSeek-V3, the model that popularized "aux-loss-free",
**kept a small sequence-wise auxiliary loss anyway** `[C]` (2412.19437). Laguna does not.
That is a strictly more aggressive published position, and the axis between them —
0.0 vs 1e-4 vs 1e-3 vs 1e-2, bias controller on in all arms — has never been run at a
scale we can afford.

**Where this reaches Mnemosyne.** Three hooks.

- *The expert working set and the KV cache are the same ≥62 GiB* `[M]`. On unified memory
  there is no separate weight tier, so how much fast tier a KV policy gets is a function
  of routing balance. No discrete-GPU paper has to model that interaction; for us it is a
  natural experiment, not a limitation.
- *Expert weights fail the reconstructibility test that KV entries pass.* If Mnemosyne's
  interface is to stay honest and general, expert offload is the boundary case that tests
  whether the abstraction is real or just vocabulary.
- *Router telemetry is the MoE analogue of cache-hit-rate instrumentation, and the
  literature does not report it.* Per-layer per-expert token counts, bias spread, and
  top-k score gaps are cheap JSONL and are exactly the attribution signals the memory
  track identified as the field's weak spot. Argus should carry them from the first MoE
  arm, not after the first confusing result.

**Sizing note for this machine.** Laguna-S's 113.5 B routed parameters are ~227 GB at
bf16 — it will not run here in any precision above 4-bit, and not comfortably even then.
Laguna-XS at 33.4 B total `[M]` is ~66.8 GB at bf16, which sits *just above* the measured
≥62 GiB fast tier and just under the 128 GB pool; at 8-bit it is ~33.4 GB and fits
comfortably. Plan inference experiments accordingly, and keep every individual tensor
under 32 GiB (`large-tensor-fault-32gib` `[M]`: a 32 GiB buffer hard-hangs at 0 CPU with
no error).

---

## Read the code

All paths relative to `research/reference/`. Revisions are pinned in `PROVENANCE.md`.
Read in this order.

### The router itself — twelve lines that contain the whole design

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:170` | `e_score_correction_bias` declared with `requires_grad=False` and zeros init. Confirm for yourself that no optimizer will ever touch it — this is a control variable smuggled into a parameter list, and it is why the shipped model carries a frozen controller output. |
| `.../modeling_laguna.py:178` | `.float()` on the router matmul. The one place the network refuses bf16. Note this is unconditional, not config-gated. |
| `.../modeling_laguna.py:181` | The softcapping line, `tanh(z/c) * c`. **This is dead code as shipped.** See the trap below. |
| `.../modeling_laguna.py:183` | `torch.sigmoid(router_logits)` — independent scores, no shared budget. |
| `.../modeling_laguna.py:185` | The bias added to the score. Check the type: it is added to the *sigmoid output*, which is what bounds the scores in (0,1) and gives you the spread-1.0 red line. |
| `.../modeling_laguna.py:187` | Combination weights gathered from the **unbiased** scores. Compare against line 186, which selects from the biased ones. The two-line gap is the entire selection/combination separation. |
| `.../modeling_laguna.py:219` | The expert loop: `for expert_idx in expert_hit`. Confirm there is no capacity check anywhere in it — this path is dropless by construction. |
| `.../modeling_laguna.py:250` | `hidden_states * self.routed_scaling_factor` — the ×2.5, applied to the routed branch only. Line 245 computes the shared expert and line 251 adds it at 1.0×. |

> **Trap, and a correction to a lab document.** `research/reference/CODE_MAP.md:52` lists
> the softcapping line as *"a guard against the runaway-confidence failure mode that
> causes expert collapse."* The mechanism is present in code but **disabled in both
> shipped checkpoints** — `models/laguna-s/config.json:261` sets
> `moe_router_logit_softcapping: 0.0`, `laguna-xs/config.json` omits the key, and the
> class default is 0.0 at
> `architecture/transformers/src/transformers/models/laguna/configuration_laguna.py:128`.
> Laguna as shipped has **no router-logit guard, no z-loss, and no auxiliary loss** —
> nothing at all standing between the router and runaway confidence except fp32 logits.
> `CODE_MAP.md` is class-3 documentation and should be amended; this repeats the
> discrepancy already recorded in `research/notes/moe-routing-and-failure-modes.md` so the
> amendment stays traceable.

### The trap in the shipped auxiliary loss

| Where | What to look at, and why |
|---|---|
| `.../modeling_laguna.py:596` | `load_balancing_loss_func` — the inherited Switch implementation. |
| `.../modeling_laguna.py:632` | It applies **softmax** to the gate logits. Laguna's router is sigmoid. |
| `.../modeling_laguna.py:634` | It re-runs top-k on those softmax probabilities, and never sees `e_score_correction_bias`. |

With `router_aux_loss_coef: 0.0` this is harmless in the shipped checkpoint. It is *not*
harmless for anyone who fine-tunes a Laguna derivative and turns the coefficient on: they
will be regularizing a counterfactual softmax router toward balance while the real
sigmoid-plus-bias router does something else entirely. A live trap in code, not a
hypothetical.

### The reference implementations, for contrast

| Where | What to look at, and why |
|---|---|
| `architecture/megatron-lm/megatron/core/transformer/moe/moe_utils.py:710` | `topk_routing_with_score_function` — the generalized router. One function, every score-function variant. |
| `.../moe/moe_utils.py:854` | The `sigmoid`/`sqrtsoftplus` branch; line 863 adds the bias, 865 gathers unbiased scores, 868 normalizes. Structurally identical to Laguna's. Two independent implementations agreeing is worth more than either alone. |
| `.../moe/moe_utils.py:56` | `switch_load_balancing_loss_func`. Its docstring writes out *f*<sub>i</sub> and *P*<sub>i</sub> explicitly — the clearest statement of the formula in any codebase. |
| `.../moe/moe_utils.py:146` | `z_loss_func`. Laguna has no equivalent. |
| `.../moe/moe_utils.py:1136` | `get_updated_expert_bias`; line 1165 is the sign rule itself. Notice `torch.no_grad()` and the all-reduce — this is a control loop bolted alongside the optimizer, not inside it. |
| `.../moe/moe_utils.py:958` | `apply_router_token_dropping`. Read lines 1001–1010 for the two drop policies, then 1013–1019 for the padding branch. Confirm the dropped token gets no error path. |
| `.../moe/moe_utils.py:241` | `get_capacity`; the formula is line 256 and the caller at line 990 shows `num_tokens` is already *T*·*k*. |
| `architecture/megatron-lm/megatron/core/transformer/transformer_config.py:2284` | The validation that refuses softmax + expert bias. This is the cleanest evidence that the pairing is forced. |
| `architecture/megatron-lm/megatron/core/transformer/moe/router.py:259` | `_maintain_float32_expert_bias`, with the comment naming the failure it prevents. |
| `architecture/gpt-oss/gpt_oss/torch/model.py:315` | The opposite design: top-k **first** (line 315), softmax over the *k* survivors **second** (line 316). No bias, no shared expert, coarse experts. Read it to see that Laguna's arrangement is a choice, not a necessity. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:285` | `build_moe_ffn` — the same router as a ggml node graph, with `ffn_exp_probs_b` (the correction bias) passed in at line 290. Line 300 builds the always-on shared expert and line 308 adds the two branches. A third independent implementation. |

---

## Exercises

Each produces a number or a plot you can check. Each has a CPU fallback. Activate the
lab first: `. .\scripts\activate-lab.ps1`.

### `router-clone-and-the-degenerate-aux-loss`

**Difficulty: low. Time: 45–75 minutes including reading. Runs entirely on CPU.**

Reimplement `LagunaTopKRouter` from scratch in about 25 lines (do not copy it — type it
from the equations above, then diff against `modeling_laguna.py:173-191`). Then prove
the degenerate minimum.

Sweep γ over a log range in [0.01, 2.0]. At each γ: scale the router logits by γ, compute
the Switch auxiliary loss *L*<sub>aux</sub>/α = *E*·Σ *f*<sub>i</sub>*P*<sub>i</sub>, and compute the
fraction of tokens whose top-k selected set is identical to the γ = 1 selection.

```python
import torch
torch.manual_seed(1337)
E, k, T, d = 64, 8, 4096, 256

# A deliberately UNBALANCED router: expert affinity has a heavy head.
W = torch.randn(E, d) * 0.02
W[:4] *= 6.0                       # four experts that win almost everything
x = torch.randn(T, d)
z = (x @ W.T).float()              # router logits, fp32 as the reference impls do

def switch_aux(z_scaled, k):
    P = torch.softmax(z_scaled, dim=-1)          # [T, E] router probabilities
    sel = torch.topk(z_scaled, k, dim=-1).indices
    f = torch.zeros(z_scaled.shape[1])
    f.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel()))
    f = f / f.sum()                              # dispatch fraction, sums to 1
    return z_scaled.shape[1] * (f * P.mean(0)).sum().item(), sel, f

base_loss, base_sel, base_f = switch_aux(z, k)
print(f"gamma=1.000  aux={base_loss:.4f}  MaxVio={(base_f.max()*E-1).item():.3f}")
for g in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    loss, sel, f = switch_aux(z * g, k)
    agree = (sel == base_sel).all(dim=-1).float().mean().item()
    print(f"gamma={g:<6.3f} aux={loss:.4f}  selection-agreement={agree:.4f}  "
          f"MaxVio={(f.max()*E-1).item():.3f}")
```

**What to check.** Three numbers. As γ → 0.01 the auxiliary loss converges to
**1.0000** (its perfect-balance value); `selection-agreement` stays at **1.0000** at
every γ; and MaxVio is **identical at every γ**. The loss went to its minimum and the
imbalance did not move at all. Plot aux-loss and MaxVio against γ on the same axes if you
want the picture.

**Extension worth twenty more minutes.** Add the z-loss, mean(logsumexp(*z*)²), to the
same table and confirm it also falls as γ shrinks. That is the `[A]` claim from the math
section becoming a measurement.

### `bias-controller-drift-and-the-spread-gauge`

**Difficulty: medium. Time: 60–90 minutes. Runs entirely on CPU (seconds of compute).**

Simulate the sign-rule controller against two synthetic routers and find out whether the
spread gauge warns you before the loss would.

- **Balanceable arm:** router logits drawn i.i.d. per step. Balance is achievable.
- **Unbalanceable arm:** a fixed subset of 6 experts whose logits sit far above the rest,
  a structure the controller cannot fix without saturating. This is a synthetic stand-in
  for the extreme-sparsity regime where the bias runaway was reported `[C]` (2512.16248).

Run both for 20,000 steps with *E* = 64, *k* = 2, *u* = 1e-3, *T* = 512 tokens per step,
using sigmoid scores and Laguna's exact selection rule (bias on selection only). Log per
step: `spread = b.max() - b.min()`, and `MaxVio = (max_load - mean_load) / mean_load`.

Report three numbers: the step at which spread first exceeds **0.3**, the step at which
it first reaches **1.0** (or "never"), and the MaxVio at each of those steps.

**What to check.** First, the arithmetic bound must hold: spread cannot reach 1.0 before
step **500** = 1/(2*u*), in either arm. If your simulation reaches it sooner, your update
rule is wrong — most likely you applied the bias to the logits instead of to the sigmoid
output. Second, the substantive question: in the unbalanceable arm, does spread cross 0.3
**before** MaxVio degenerates? That is the lab's open question #1 in miniature, and a
synthetic answer is not a real answer — but it tells you whether the gauge is worth
wiring into Argus before you spend GPU hours finding out.

**Deliberate limitation, state it in your write-up.** This simulates the *controller*
against a *fixed* score distribution. Real training has the router weights moving too, so
the controller is chasing a moving target. The synthetic can show the gauge is
mechanically sound; it cannot show the gauge is predictive.

### `decode-batch-knee-vs-coupon-collector`

**Difficulty: medium-high. Time: 2–3 hours including the first-run debugging you will
actually hit.**

Measure the decode batch-size knee on the Z13 and compare it against
*N*(*B*) = *E*(1 − (1 − *k*/*E*)<sup>*B*</sup>).

Build a synthetic MoE FFN: *E* = 128 experts, *d* = 1024, *d*<sub>ff</sub> = 2048, bf16.
Per expert that is 3 × 1024 × 2048 × 2 bytes = **12.6 MB**; the full expert bank is
**1.61 GB** — far larger than any cache, far below the ≥62 GiB fast tier `[M]`, and far
below the 32 GiB single-tensor hazard `[M]`. Route with a uniform-random top-*k* = 8 and
sweep *B* ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}.

At each *B* record: measured distinct experts touched, predicted *N*(*B*), wall-clock per
step (median of ≥20 timed reps after ≥5 warmup reps), and derived tokens/s.

**GPU (gfx1151, native Windows) caveats — read before running.**

- `. .\scripts\activate-lab.ps1` first. hipBLASLt config is worth ~12% GEMM on this wheel
  `[M]` (`ASSUMPTIONS.md: hipblaslt-config`) — not the 5× cliff the issue tracker
  reported, but free.
- **Warm up properly.** The first HIP call JIT-compiles kernels; separately, the first
  `torch.unique`-style op on ROCm can cost ~200 ms of rocPRIM compilation. Discard at
  least the first five reps or your *B* = 1 point will be nonsense.
- `torch.cuda.synchronize()` before every timer stop. Without it you are timing kernel
  launches.
- bf16 numerics on gfx1151 are **unproven** (`ASSUMPTIONS.md: bf16-numerics-unproven`).
  This exercise measures *time*, not values, so unvalidated bf16 is legitimate here —
  but do not let any correctness claim leak out of it.
- Our measured ceiling is 20.9 TFLOPS bf16 at 8192³ `[M]`, which is 63% of the figure
  cited for this silicon and currently unexplained. Use *our* number in any roofline.

**CPU fallback.** Set *E* = 32, *d* = 256, *d*<sub>ff</sub> = 512, fp32, *B* up to 128, 5
timed reps. The absolute throughput is meaningless; the *shape* of the curve and the
distinct-expert count are not, and the coupon-collector comparison works identically.
Roughly 5 minutes.

**What to check.** Two things. (1) Measured distinct-expert count should track
*N*(*B*) closely, because your synthetic router *is* uniform — if it does not, your
sampling is wrong, and you have found a bug rather than a finding. (2) The tokens/s curve
should show a knee near *B* = *E*/*k* = **16** for these settings (at *B* = 16 the
prediction is *N* = 82.4, i.e. 64% of all experts already touched). Report the batch size
at which measured distinct experts first exceeds 90% of *E* — solve
(1 − 8/128)<sup>*B*</sup> = 0.1, giving *B* = ln(0.1)/ln(0.9375) = **35.7, so *B* ≈ 36** —
and the batch size at which tokens/s stops improving materially.

**The real finding is the gap you create next.** Re-run with a *correlated* router —
sample expert affinity from a smoothly drifting vector rather than i.i.d. per token, so
adjacent tokens route similarly — and confirm the measured distinct-expert count falls
*below* the coupon-collector prediction. That gap is the quantity a real MoE serving
system actually lives on, and it is a per-workload constant nobody publishes. If you get
a clean number here, it belongs in `notebook/` as a pre-registered follow-up.

---

## Self-check

1. Laguna adds the correction bias to the sigmoid *output* rather than to the router
   logits. Name one thing this buys that biasing the logits would not.
2. You are handed a training run whose Switch auxiliary loss has fallen smoothly from 1.4
   to 1.02 over 5,000 steps. What is the single cheapest additional metric that tells you
   whether balance actually improved, and what would it look like if it had not?
3. Your MoE serving deployment runs at batch size 4 and you are considering raising it to
   64 to improve throughput. Using *E* = 256, *k* = 10, estimate the change in expert
   weight bytes read *per generated token*, and state the one assumption most likely to
   make your estimate optimistic.
4. A colleague proposes a clean 2×2×3 ablation: {sigmoid, softmax} × {bias controller on,
   off} × {*E* = 16, 64, 256}. What is wrong with the design, and what is the largest
   valid design in its place?
5. With *T* = 4096, *E* = 32, *k* = 4 and cf = 1.0, how many assignments can each expert
   accept? If `pad_to_capacity` is enabled, what fraction of the expert GEMM is spent on
   padding, and does that fraction depend on how imbalanced the router actually is?
6. Laguna ships with softcapping disabled, no z-loss, and `router_aux_loss_coef: 0.0`.
   Name the one remaining mechanism in the shipped code that acts against router
   numerical failure, and name the failure it does *not* protect against.

---

## What is still unsolved here

Drawn from `research/notes/moe-routing-and-failure-modes.md` and the papers register.
These are contested in the literature as of July 2026, and the module presents them
contested rather than resolved.

**Is the bias controller sufficient on its own?** DeepSeek reports better perplexity and
10–20× better global load balance than an auxiliary loss `[C]` (2408.15664, 2024-08-28),
with a theoretical framing as a primal-dual method carrying monotonic-improvement and
logarithmic-regret results `[C]` (2512.03915, 2025-12-03). Against that: Sigma-MoE-Tiny
reports the bias running away in lower layers at extreme sparsity (top-1 of 96) `[C]`
(2512.16248, 2025-12-18), and DeepSeek-V3 itself retained a small sequence-wise
auxiliary loss `[C]` (2412.19437, 2024-12-27). Newer work proposes replacing the sign
rule outright: φ-Balancing derives an EMA-based routing adjustment from convex duality,
explicitly arguing that mini-batch statistics are the wrong signal `[C]` (2605.15403,
2026-05-14). Laguna's `0.0` is the most aggressive shipped position on this axis.

**Does expert specialization exist at all?** DeepSeekMoE `[C]` (2401.06066) and
knowledge-attribution work `[C]` (2505.24593, 2025-05-30) say yes. Two 2026 papers argue
that because the router is a linear map, hidden-state similarity alone explains expert
usage — so specialization is a property of the representation space rather than of the
routing architecture `[C]` (2604.09780, 2026-04-10; 2605.12476, 2026-05-12). Both
positions are live and both are recent.

**"Balanced" is not "good", and nobody has a metric for the difference.** Holding a
trained model fixed and enumerating equal-compute alternative routes shows the trained
route is near-optimal on confident tokens and close to uninformative on the hard ones
that matter for reasoning `[C]` (2605.07260, 2026-05-08). A load-balance metric cannot
see this. Neither can a perplexity number. This is exactly the attribution gap this lab
exists to attack, and there is no accepted instrument for it.

**Does routing need a router?** Autonomy-of-Experts removes it entirely — experts
pre-compute internal activations and self-select by activation norm `[C]` (2501.13074,
2025-01-22). Routing-free MoE is a separate 2026 line `[C]` (2604.00801, 2026-04-01). If
either holds up, most of this module describes a transitional design.

**Optimal sparsity has no successor law.** `[C]` 2501.12370 (2025-01-21) gives an optimal
sparsity under fixed compute; `[C]` 2508.18672 (2025-08-26) argues it must be set jointly
by active FLOPs and tokens-per-parameter and differs for reasoning tasks; `[C]` 2603.21862
(2026-03-23) fits holistic laws over hundreds of models and finds the near-optimal band
*widens* with scale. And a warning aimed squarely at 20M–300M sweeps: the standard
IsoFLOP parabola fit is systematically biased even on noise-free data `[C]` (2603.22339,
2026-03-21). Do not fit one without reading that paper first.

**Unmeasured on our own hardware, and cheap.** Whether bf16 flips router top-k decisions
on gfx1151. Every reference implementation refuses bf16 for router logits; nobody has
checked what the disagreement rate actually is on this silicon. Run a trained small MoE's
router in fp32 and bf16 over identical hidden states and count top-k disagreements per
layer. If the rate is non-trivial it is a Hardware Validation Gate item rather than a
curiosity, and it is the single cheapest experiment in this module.

**Not in scope here, and blocked.** Expert parallelism, all-to-all dispatch, node-limited
routing and the entire distributed half of the MoE literature are design-only for us —
`torch._C._distributed_c10d` is incomplete on gfx1151 `[C]` (`ASSUMPTIONS.md:
single-device-only`). That is less of a loss than it sounds. On one device, capacity is a
kernel-shape question and imbalance shows up as wasted FLOPs and a longer critical-path
GEMM rather than as a network straggler — which removes the communication confound that
makes published capacity-factor comparisons hard to read.

---

## Answers

**1.** Bounded scores, and therefore a computable saturation threshold. Adding the bias
after the sigmoid means every score lies in (0, 1), so a bias spread of 1.0 is exactly the
point at which one expert outranks another for every possible token. Biasing the logits
instead leaves the score gaps unbounded and scale-drifting, so there is no fixed red line
to alert on. (A second acceptable answer: it keeps the bias out of the combination
weights, which are gathered from the unbiased scores at `modeling_laguna.py:187`.)

**2.** MaxVio, or equivalently the per-expert token-count distribution — one histogram per
layer per logging interval. The failure case is unmistakable: the auxiliary loss
converges toward exactly 1.0 while MaxVio does not move. That is the degenerate minimum,
reachable by shrinking router logit magnitude, which flattens the probabilities *P*
without changing the top-k ordering and therefore without changing the dispatch fractions
*f* at all. You reproduce this in the first exercise.

**3.** *N*(4) = 256(1 − 0.9609⁴) = 37.7 distinct experts, so 37.7/4 = **9.4
experts-worth per token**. *N*(64) = 236.1, so 236.1/64 = **3.69 experts-worth per
token** — a **2.6× reduction** in expert weight bytes per generated token. The assumption
most likely to make this optimistic is the independence of routing across tokens. Real
routing is temporally correlated within a sequence, so fewer distinct experts get touched
than predicted, which — read carefully — makes the *actual* amortization **better** than
predicted, not worse. The optimism is elsewhere: this counts only expert weights and
ignores that at *B* = 64 the KV cache is also 64× larger and contending for the same
≥62 GiB fast tier `[M]`.

**4.** The `{softmax} × {bias on}` cell is not a legal configuration — Megatron raises at
config validation (`transformer_config.py:2289`) because a per-expert additive offset is
not a separable per-backend control under a shared probability budget. Sigmoid-with-bias
and softmax-with-aux-loss are each one decision, not two crossed decisions. The largest
valid design is **2×3**: {sigmoid + bias, softmax + aux loss} × {*E* = 16, 64, 256} at
matched active parameters.

**5.** capacity = ceil((*T*·*k*/*E*) · cf) = ceil((4096 × 4 / 32) × 1.0) = **512
assignments per expert**. With padding, the expert GEMM runs *E* × capacity = 32 × 512 =
16,384 rows against *T*·*k* = 16,384 real assignments, so the padding fraction is
1 − 1/cf = **0%** at cf = 1.0 — in the sense that no *extra* rows are allocated. It does
**not** depend on the actual imbalance: the padded shape is fixed by *T*, *E*, *k* and cf
alone. What imbalance changes is how many of those 16,384 rows are wasted on *dropped*
tokens versus real ones, and that cost is invisible in the shape. This is the point of
the question: the FLOP cost of a capacity factor is deterministic and budgetable; the
*quality* cost is data-dependent and unlogged.

**6.** The remaining mechanism is **fp32 router logits** — the unconditional `.float()`
at `modeling_laguna.py:178`, mirrored by Megatron's `moe_router_dtype` and its fp32
expert bias (`router.py:104`, `router.py:259`). It protects against precision-induced
top-k boundary flips, where a rounding error in a small score gap sends a token to a
different expert. It does **not** protect against runaway confidence: nothing in the
shipped configuration bounds the magnitude of the router logits, because softcapping is
off (`config.json:261`), there is no z-loss, and the auxiliary loss coefficient is 0.0.
Precision is guarded; magnitude is not.
