---
title: Depth, width, and initialization — how to spend a parameter budget and how to start
version: 1.0.0
date: 2026-07-26
track: B — Modern architecture
prereqs: transformer-forward-pass, scaling-laws-and-flops-budget, attention-variants-and-kv-cost
reading_difficulty: 3 / 5
exercise_difficulty: 2 / 5, 3 / 5, 4 / 5
time: ~3 h to read and work the arithmetic; ~1 h + ~1.5 h + ~1 h GPU (~2 h CPU) for the three exercises
---

# Depth, width, and initialization

## 1. What this module settles

A parameter budget `N` does not determine a model: the same `N` buys a 6-layer model
1024 wide or a 96-layer model 256 wide, and those two models have identical parameter
counts, **1.4× different FLOPs per token, and 4× different KV bytes per token** — so
"matched parameters" is not a matched budget, and the aspect ratio is a memory-systems
decision before it is a modelling one. Initialization is not a warm-up detail you grow
out of after a hundred steps: the residual stream is a shared accumulator whose variance
grows linearly in depth, and the `1/√(2L)` output-projection scaling that every
production codebase applies is the fix, with the deep-layer degeneracy that appears when
you skip it being permanent rather than transient. muP is the tool that makes a 20M-param
tuning sweep say anything at all about a 300M-param model — it is the control that
converts "policy A beat policy B" from an ambiguous statement into a comparison, and
this lab's riskiest assumption (`ASSUMPTIONS.md: ablation-scale-sufficient`) cannot be
attacked without it.

---

## 2. Theory in plain language

### 2.1 The question

You have decided to train a model of roughly `N` parameters. Nothing has told you its
shape. The two primary dials are:

- **depth** `L` — how many decoder blocks are stacked;
- **width** `d` — the size of the residual stream (`hidden_size`, `n_embd`, `d_model`;
  all the same thing).

The ratio `d/L` is the **aspect ratio**. GPT-2 Small is `768/12 = 64`. GPT-3 175B is
`12288/96 = 128`. Our reference model, Laguna S 2.1, is `3072/48 = 64` `[M]` (read from
`research/reference/models/laguna-s/config.json`). A 20M-param model you might train
this week could reasonably be anywhere from 10 to 200.

**What this replaced.** Before 2020 the shape was pure convention: 12 layers, 768 wide,
FFN 4× the residual stream, because that is what the previous paper did. `[C]` Kaplan et
al. ([2001.08361](https://arxiv.org/abs/2001.08361), Jan 2020) gave that convention an
excuse by reporting that loss depends only weakly on shape at fixed parameter count —
the "shape doesn't matter, only `N` does" result that most practitioners still quote.
That licence has been eroded steadily since. `[C]`
[2006.12467](https://arxiv.org/abs/2006.12467) (Jun 2020) derived a depth-to-width
interaction specific to self-attention rather than to networks in general; `[C]`
[2207.10551](https://arxiv.org/abs/2207.10551) (Jul 2022) showed across ten
architectures that inductive bias changes scaling *exponents*, so a ranking at one size
need not survive to another; `[C]` MobileLLM
([2402.14905](https://arxiv.org/abs/2402.14905), Feb 2024) found deep-and-thin
consistently better below 1B parameters — which is exactly our ablation box; and `[C]`
[2606.18246](https://arxiv.org/abs/2606.18246) (Jun 2026) attacks the premise that width
should even be *constant* across depth, reporting that widening the first and last
blocks and narrowing the middle beats a uniform-width baseline at lower FLOPs and lower
memory. The current state is that shape matters, the direction depends on scale, and
nobody has published the answer for 20M–300M with our constraints.

> **Systems bridge.** Depth is pipeline stages; width is per-stage fan-out. A deep,
> narrow model is a long pipeline of small stages: low per-stage work, a long serial
> dependency chain, poor arithmetic intensity per kernel, and — as we will compute in
> §3.3 — a bigger resident cache. A wide, shallow model is a short pipeline of large
> stages: big GEMMs, better hardware utilisation, smaller cache.
>
> **Where it breaks, and this is the part worth keeping.** In a pipeline, adding a stage
> adds latency and nothing else; the data that comes out is the same data. In a residual
> network, every block *writes into a shared accumulator*, so adding a block changes the
> statistics of what every downstream block reads (§3.6). The failure mode is not
> slowness — it is that the deep blocks quietly stop contributing (`[C]` the "curse of
> depth", [2502.05795](https://arxiv.org/abs/2502.05795), NeurIPS 2025) and you pay for
> parameters that do nothing. There is no queue depth to watch, no error, no dropped
> request. Second break: you cannot rebalance stages at runtime. The shape is frozen at
> `__init__` and the optimizer's stable learning rate is a function of it, so getting it
> wrong means re-running, not reconfiguring.

### 2.2 Initialization: the starting state you cannot roll back

Every weight matrix has to start somewhere. The obvious choices are wrong in
instructive ways: all zeros makes every unit in a layer compute the same thing forever
(symmetry never breaks, gradients are identical); too large and activations saturate or
explode; too small and the signal dies before it reaches the last layer.

**What it replaced, in order.** (1) A fixed small constant — `std = 0.02`, GPT-2's
value, still the default in nanoGPT (`model.py:164`) and still the default in Laguna's
config `[M]` (`configuration_laguna.py:149`). (2) Fan-in-aware schemes (LeCun, Glorot,
Kaiming) that make the initialization a function of the layer's shape rather than a
constant. (3) Depth-aware schemes that additionally account for how many layers write
into the residual stream — GPT-2's `0.02/√(2L)` on output projections
(`model.py:145`), DeepNorm `[C]` ([2203.00555](https://arxiv.org/abs/2203.00555)).
(4) muP `[C]` ([2203.03466](https://arxiv.org/abs/2203.03466)), which makes
initialization and learning rate co-vary with width so that the *optimum* transfers.

> **Systems bridge.** Init is the cold state of a cache, or the initial replica
> placement of a freshly deployed cluster. **Where it breaks:** a cold cache self-heals
> — you pay a warm-up period and then the steady state is the same regardless of how you
> started. A badly scaled initialization does not self-heal. It changes which optimum
> the run converges to, and in the deep-network case it can leave whole blocks
> permanently near-identity. The state is not a cache; it is more like a schema you
> chose at table-creation time.

### 2.3 Residual scaling: the shared accumulator

A pre-norm decoder block is:

```
x ← x + attention(norm(x))
x ← x + mlp(norm(x))
```

`x` is the **residual stream**. Every sublayer reads a normalized copy of it and adds
its output back. With `L` blocks there are `2L` writes into the same accumulator.

> **Systems bridge.** This is a shared bus, or a write-combining accumulator that every
> stage adds into. If each of `2L` stages contributes an independent quantity of unit
> magnitude, the accumulator's magnitude grows like `√(2L)` — the standard random-walk
> result.
>
> **Where it breaks.** A hardware accumulator has a fixed width and overflow is loud:
> you get a flag, a trap, a wrapped value. Here nothing signals. The stream just gets
> larger, the pre-norm divides by a larger number, each layer's *relative* contribution
> shrinks as `1/√l`, and the deep layers converge toward the identity function. You
> observe this as a loss curve that is fine but slightly worse than it should be, on a
> model whose last third of blocks are doing nothing. `[C]`
> [2502.05795](https://arxiv.org/abs/2502.05795) measures this on 130M–1B models; `[C]`
> Peri-LN ([2502.02732](https://arxiv.org/abs/2502.02732), ICML 2025) attacks the same
> phenomenon by adding output norms rather than by rescaling init; `[C]`
> [2603.15389](https://arxiv.org/abs/2603.15389) (Mar 2026) reports that sparsity
> interacts with it, so the dense answer may not be the MoE answer — and Laguna is a
> sparse pre-norm model, i.e. the exact configuration this literature is most suspicious
> of.

### 2.4 muP: a unit system for hyperparameters

Under standard parameterization, the learning rate that is best at width 256 is not the
one that is best at width 1024. So a sweep at small scale tells you about small scale
and nothing else, and a two-arm comparison at small scale is confounded by which arm
happened to be closer to its own optimum.

muP re-scales three things as functions of width — initialization variance, per-tensor
learning rate, and the attention logit scale — so that the optimal learning rate becomes
approximately width-invariant. Tune at 20M, transfer to 300M.

> **Systems bridge.** muP is dimensional analysis. Reporting a raw learning rate across
> widths is like reporting raw p99 latency across queue depths: the number moves for
> reasons unrelated to the change you are studying. Normalize the units and the
> comparison becomes meaningful.
>
> **Where it breaks (three places).** First, classic muP normalizes **width only**. It
> says nothing about depth, batch size, training duration, data mixture, or sequence
> length — and depth is precisely the other half of this module. Second, its derivation
> is an infinite-width limit at finite step count, while a real run is the opposite:
> finite width, very many steps. Third, and least comfortable, **the active ingredient
> is disputed** — `[C]` [2510.19093](https://arxiv.org/abs/2510.19093) (Oct 2025) argues
> decoupled weight decay does most of the stabilising work that muP is credited with,
> and `[C]` [2605.21486](https://arxiv.org/abs/2605.21486) (May 2026) finds the
> embedding-layer learning rate disproportionately decisive. Use muP as a control; do
> not present it as a solved one.

---

## 3. The math that actually matters

### 3.1 Parameter count from a config, exactly

Symbols, every one translated:

| Symbol | Words | Config field |
|---|---|---|
| `L` | number of decoder blocks stacked | `num_hidden_layers` / `n_layer` |
| `d` | width of the residual stream — the vector each token carries between blocks | `hidden_size` / `n_embd` / `d_model` |
| `n_q` | number of query heads per layer | `num_attention_heads` |
| `n_kv` | number of key/value heads per layer (≤ `n_q` under GQA) | `num_key_value_heads` |
| `d_h` | width of one attention head's vectors | `head_dim` (or `d/n_q` if derived) |
| `F` | inner width of the feed-forward network | `intermediate_size` |
| `V` | vocabulary size — rows in the embedding table | `vocab_size` |
| `t` | 1 if input and output embeddings share one matrix, else 0 | `tie_word_embeddings` |
| `G` | GQA group size, `n_q / n_kv` — query heads per KV head | derived |

One block, with biases omitted (no modern decoder uses them) and RMSNorm counted:

```
q_proj      d × (n_q  · d_h)
k_proj      d × (n_kv · d_h)
v_proj      d × (n_kv · d_h)
o_proj      (n_q · d_h) × d
──────────────────────────────────────────────
attention   2 · d · d_h · (n_q + n_kv)

gate_proj   d × F      up_proj   d × F      down_proj   F × d
──────────────────────────────────────────────
SwiGLU MLP  3 · d · F

two RMSNorm gains                      2 · d
```

Whole model:

```
N_total = (2 − t) · V · d                                  ← embeddings + readout
        + L · [ 2·d·d_h·(n_q + n_kv) + 3·d·F + 2·d ]        ← the trunk
        + d                                                 ← final norm
```

Read this out loud once: **the trunk is linear in `L`, and every term in it is linear in
`d`, multiplied by another quantity that is conventionally also proportional to `d`.**
That is where the next identity comes from.

### 3.2 The `12·L·d²` identity, and the iso-parameter family

Apply the two conventions almost every dense decoder follows: heads tile the residual
stream (`n_q · d_h = d`), and the FFN is sized to `F = 4d` for a two-matrix ReLU/GELU
MLP or `F = (8/3)·d` for a three-matrix SwiGLU. That second convention exists *precisely*
to keep the parameter count identical `[C]`
([2002.05202](https://arxiv.org/abs/2002.05202)):

```
GELU MLP    : 2 · d · 4d      = 8 d²
SwiGLU MLP  : 3 · d · (8/3)d  = 8 d²      ← same, by construction
```

With MHA (`n_kv = n_q`, so `G = 1`): attention = `2 · d · d_h · 2 n_q = 4d²`. Total per
block `12 d²`, and dropping the `O(d)` norm terms:

```
N_trunk ≈ 12 · L · d²
```

Under GQA the attention term shrinks to `d²·(2 + 2/G)`, so the constant becomes
`10 + 2/G`: 12 at `G=1`, 10.25 at `G=8`, → 10 as `G → ∞` (MQA). Useful sanity number:
**the whole GQA ladder moves the parameter count by at most 17%,** which is why GQA is
sold as a KV-bytes change and not a parameter change `[C]`
([2305.13245](https://arxiv.org/abs/2305.13245)).

**Check the identity against real code.** nanoGPT's `shakespeare_char` config is `L=6`,
`d=384`, `n_head=6`, `bias=False`, GELU MLP at `4d`, `V=65`, `block_size=256`, tied
embeddings (`model.py:138`):

```
12 · L · d²          = 12 · 6 · 147,456   = 10,616,832
+ 2 RMSNorm-equivalents per block, 6 × 2 × 384 =  4,608
+ final norm                                   =    384
+ token embedding    65 × 384                  = 24,960   (tied, so lm_head is free)
+ position embedding 256 × 384                 = 98,304
────────────────────────────────────────────────────────
total                                          = 10,745,088
non-embedding, as nanoGPT defines it (minus wpe)= 10,646,784
```

`get_num_params()` (`model.py:150`) subtracts only the position embedding and keeps the
token embedding, because tying means those weights *are* the readout matrix. Exercise A
makes you reproduce this to the parameter.

**Now the family that makes the point.** Hold `L · d²` fixed at 6,291,456, so
`N_trunk = 75,497,472` exactly in all three:

| Shape | `L` | `d` | aspect ratio `d/L` | `N_trunk` |
|---|---|---|---|---|
| wide-shallow | 6 | 1024 | 170.7 | 75,497,472 |
| middle | 24 | 512 | 21.3 | 75,497,472 |
| deep-thin | 96 | 256 | 2.7 | 75,497,472 |

Three models, identical trunk parameter count, aspect ratios 64× apart. If parameter
count were the whole story these would be interchangeable. They are not, and the next
two subsections say exactly how much they differ, in numbers you can check.

A practical wrinkle worth noticing: exact iso-parameter matching requires `L · d²` to be
constant, and `d` is in practice constrained to multiples of 64 or 128 for tensor-core
and hipBLASLt tile alignment. There is no integer point between `(24, 512)` and
`(6, 1024)` at a nice width — `L = 12` demands `d = 724.08`. **Your matched-budget grid
is quantised by kernel geometry**, which is a constraint no paper mentions and which you
should state in any pre-registration.

### 3.3 What the aspect ratio actually costs

**FLOPs per token.** The standard accounting `[C]` (PaLM Appendix B,
[2204.02311](https://arxiv.org/abs/2204.02311); implemented at nanoGPT
`model.py:296`):

```
flops_per_token = 6·N  +  12·L·n_q·d_h·T
                  ↑        ↑
                  |        the attention score/aggregate matmuls
                  the dense matmuls
```

The `6` is `2 FLOPs per multiply-accumulate × (1 forward + 2 backward)`. The second term
is the two attention matmuls (`Q·Kᵀ` and `A·V`), each `T·d` MACs per token per layer, at
2 FLOPs each, times 3 for forward+backward: `12 · L · d · T` when heads tile the stream.
`T` is the sequence length (replace it with the window `w` on a sliding-window layer).

At `T = 1024`, with `6·N_trunk = 452,984,832`:

| Shape | `12·L·d·T` | as % of `6N` | total FLOPs/token |
|---|---|---|---|
| `L=6, d=1024` | 75,497,472 | 16.7% | 528,482,304 |
| `L=24, d=512` | 150,994,944 | 33.3% | 603,979,776 |
| `L=96, d=256` | 301,989,888 | 66.7% | 754,974,720 |

**The deep-thin model costs 1.43× the FLOPs per token of the wide-shallow one at
identical parameter count and identical context.** The attention term scales as `L·d`,
and along an iso-`L·d²` curve `L·d = √(L) · √(L·d²)` — so it grows as `√L`. Doubling
depth at fixed `N` costs you 41% more attention FLOPs.

**KV bytes per token — the one that matters for this lab.**
`research/memory/kv-cache-mechanics.md` derives, and I am not restating differently:

```
per_token_bytes = 2 · L · n_kv · d_h · b
```

where `b` is bytes per stored element (2 for bf16). **Hidden size does not appear.** That
is correct and it is the most useful fact in that note. But `n_kv · d_h` is not free in
practice: under the tiling convention `n_q · d_h = d` and `n_kv = n_q / G`, so
`n_kv · d_h = d / G` and

```
per_token_bytes = 2 · L · (d / G) · b
```

Along the iso-parameter curve (`G = 1`, bf16):

| Shape | KV bytes/token | at 32k context |
|---|---|---|
| `L=6, d=1024` | `2·6·1024·2` = 24,576 B = **24 KiB** | 0.75 GiB |
| `L=24, d=512` | `2·24·512·2` = 49,152 B = **48 KiB** | 1.50 GiB |
| `L=96, d=256` | `2·96·256·2` = 98,304 B = **96 KiB** | 3.00 GiB |

**A 4× spread in resident cache at identical parameter count.** Depth is a KV-cache
lever, and it is not in anyone's list of KV-cache levers — which are conventionally
`n_kv` (GQA/MQA), `d_h`, `b` (quantization), and eviction. The aspect ratio sits
upstream of all of them.

The two statements are consistent: the KV formula genuinely does not contain `d`, and
depth moves KV cost only because the *convention* couples `d_h` and `n_kv` to `d`. Break
the coupling and depth stops mattering — which is exactly what our reference model does.
`[M]` Laguna sets `head_dim: 128` explicitly while `hidden_size / num_attention_heads =
3072/48 = 64`, so `n_q · d_h = 6144 = 2d` on its full-attention layers and `9216 = 3d`
on its sliding ones. Its attention block is *wider than its residual stream*. Any
arithmetic that assumes head tiling is wrong for this model. State which convention you
are using, every time.

### 3.4 The embedding tax, and why "matched parameters" is ambiguous

The embedding term `(2 − t)·V·d` is linear in `d` and **independent of `L`**. At
frontier scale it is a rounding error; at 20M–300M it dominates. Same three shapes,
GPT-2's `V = 50,257`, untied (`t = 0`):

| Shape | embeddings `2·V·d` | trunk | total | embeddings as % of total |
|---|---|---|---|---|
| `L=6, d=1024` | 102,926,336 | 75,497,472 | 178,423,808 | **57.7%** |
| `L=24, d=512` | 51,463,168 | 75,497,472 | 126,960,640 | **40.5%** |
| `L=96, d=256` | 25,731,584 | 75,497,472 | 101,229,056 | **25.4%** |

Three models with *identical trunks* whose total parameter counts differ by 76%. If your
matched-budget rule says "matched parameters" without saying which count, the rule is
vacuous. **House rule for Proteus arms: match non-embedding parameters, and state `V`
and the tying flag in the pre-registration.** This is the same trap
`research/notes/transformer-state-of-the-art.md` §7 flags from the other direction — at
`V = 100,352` and `d = 768`, untied embeddings are 154,140,672 parameters, more than half
of a 300M budget and larger than an entire 20M model.

Vocabulary is a scaling-law variable, not a preprocessing detail `[C]`
([2407.13623](https://arxiv.org/abs/2407.13623), NeurIPS 2024) — and tying `[C]`
([1608.05859](https://arxiv.org/abs/1608.05859)) is a modelling constraint with a real
cost, not a free deduplication, because the input table wants vectors that compose well
under addition into the residual stream while the output table wants vectors that
separate well under dot product.

One more number worth carrying, because it connects parameter counting directly to the
memory track: `[C]` [2505.24832](https://arxiv.org/abs/2505.24832) (May 2025) estimates
GPT-style models store roughly **3.6 bits per parameter** of memorised content. At
75.5M trunk parameters that is ~34 MB of parametric memory — against a KV cache that, at
32k context and the deep-thin shape, is 3 GiB. Parametric memory and contextual memory
are not remotely the same size, and this is the arithmetic that says so.

### 3.5 Initialization variance, derived

Take a linear layer `y = W x`, with `W` of shape `d_out × d_in`, entries drawn
independently with mean 0 and variance `σ²`, and `x` entries independent with variance
`v`. The `i`-th output is `y_i = Σ_j W_ij x_j`, a sum of `d_in` independent
zero-mean products, so:

```
Var(y_i) = Σ_j Var(W_ij · x_j) = d_in · σ² · v
```

In words: the output variance is the input variance multiplied by the fan-in and by the
weight variance. To keep the signal from growing or shrinking as it passes through the
layer, set `d_in · σ² = 1`:

```
σ = 1 / √d_in                      ← fan-in ("LeCun") init
```

The backward pass runs the same argument on `Wᵀ`: the gradient with respect to `x` has
variance `d_out · σ² · Var(gradient wrt y)`, giving `σ = 1/√d_out`. You cannot satisfy
both unless the layer is square, so Glorot averages them (`σ² = 2/(d_in + d_out)`) and
Kaiming picks fan-in with a factor 2 to compensate for ReLU zeroing half the mass
(`σ² = 2/d_in`).

Read this in real code: olmo-core's `fan_in` method is exactly `std = 1/√d_in` per
tensor (`init.py:73`, applied at `init.py:156`, `:165`, `:176`). Contrast nanoGPT
(`model.py:164`) and Laguna (`modeling_laguna.py:552`), which both use a flat `0.02`
regardless of fan-in. At `d = 384`, `1/√384 = 0.0510`; at `d = 3072`,
`1/√3072 = 0.0180`. So `0.02` is the fan-in value at `d = 2500` (`1/√2500 = 0.02`
exactly), is **2.5× smaller** than fan-in prescribes at `d = 384`, and is slightly
larger than it prescribes at `d = 3072`. A constant that is correct at one width is
wrong on both sides of it, in opposite directions — which is one concrete reason the
"same learning rate works at every size" assumption fails.

### 3.6 Residual accumulation and the `1/√(2L)` rule

Let `s²` be the variance of one sublayer's output. Because pre-norm re-standardises the
sublayer's *input*, `s²` does not depend on how large the stream has already grown. With
`2L` sublayers writing into the stream and their contributions roughly uncorrelated:

```
Var(x_L) ≈ Var(x_0) + 2·L·s²
```

In words: the residual stream's variance grows **linearly in depth**, so its magnitude
grows as `√L`. With `Var(x_0) = s² = 1`: at `L = 6` the final variance is 13
(std 3.6); at `L = 48` it is 97 (std 9.8); at `L = 96` it is 193 (std 13.9).

The fix is to shrink each sublayer's *output projection* so its contribution is
`s²/(2L)`:

```
σ_out = σ_base / √(2L)
```

Then the total added variance is `2L · s²/(2L) = s²` — **constant in depth.** That is
the whole derivation, and it is three lines of real code:

- nanoGPT `model.py:145`: `std=0.02/math.sqrt(2 * config.n_layer)`, applied to every
  parameter whose name ends in `c_proj.weight` — i.e. both the attention output
  projection and the MLP down projection.
- olmo-core `init.py:167`: `std = std / (2 * num_blocks) ** 0.5` for the `llama`
  method, and the identical line at `training/olmo-core/src/olmo_core/nn/attention/__init__.py:1143` for `w_out`.

Arithmetic, so the numbers are in your hands: `0.02/√12 = 0.005774` at `L=6`;
`0.02/√96 = 0.002041` at `L=48`; `0.02/√192 = 0.001443` at `L=96`.

**There is a second, different rule shipping in the same file, and the difference
matters.** olmo-core's `llama_depth` method (`init.py:169`,
`training/olmo-core/src/olmo_core/nn/attention/__init__.py:1145`) divides by `√(2·(block_idx + 1))` — the *layer index*, not
the layer count. Sum the contributions: block `l` adds `2 · s²/(2(l+1)) = s²/(l+1)`, and
`Σ_{l=0}^{L-1} 1/(l+1) = H_L ≈ ln L + 0.577`. So:

| Rule | Total residual variance added | at `L=48` |
|---|---|---|
| none | `2·L·s²` | 96 s² |
| `1/√(2L)` (global) | `s²` | 1.0 s² |
| `1/√(2(l+1))` (per-index) | `H_L · s²` | 4.45 s² |

`[A]` Medium confidence, from this derivation and not from a measurement: the per-index
rule bounds the growth but does not flatten it, and it makes late blocks contribute less
than early ones — which is the same direction as the curse-of-depth failure `[C]`
([2502.05795](https://arxiv.org/abs/2502.05795)) rather than against it. **Cheapest test
that would move this:** Exercise B, extended to a third arm. Do not take the ordering on
my word; the point of writing the sum out is that you can now check it in twenty minutes.

### 3.7 muP in three rules, with the arithmetic

Let `d₀` be the **base width** — the width you actually tune at — and `d` the target
width. Define the width multiplier:

```
m = d / d₀
```

The practical AdamW form of the rules (read Table 3 of `[C]`
[2203.03466](https://arxiv.org/abs/2203.03466) for the exact per-tensor statement; do
not reconstruct it from memory, mine included):

| Tensor class | Init std | Adam LR | Forward multiplier |
|---|---|---|---|
| input embedding | unchanged | unchanged | ×1 |
| hidden matrices (`d × d`-ish) | `σ₀ / √m` | `η₀ / m` | ×1 |
| readout / `lm_head` | `σ₀ / m` | `η₀ / m` | **×`1/m`** |
| attention logits | — | — | scale by `√d_h₀ / d_h`, not `1/√d_h` |

In words, one row at a time:

- **Hidden matrices.** Widening by `m` means each output coordinate sums `m` times as
  many inputs, so init variance must fall by `m` (std by `√m`) to keep the forward
  variance fixed — that is §3.5. The learning-rate rule is *not* the same argument.
  Under Adam the update magnitude is approximately `η` per coordinate regardless of
  gradient scale, because `m/(√v + ε)` normalises it away. So a layer with `m` times as
  many inputs produces an output change `m` times larger for the same `η`, and you must
  divide `η` by `m` to keep the *change in the layer's output* constant. This is why muP
  under SGD and muP under Adam have different exponents, and why quoting "the muP LR
  rule" without saying which optimizer is meaningless.
- **Readout.** Its output feeds a softmax over `V` logits, whose scale must not drift
  with width, hence the explicit `1/m` output multiplier applied in the forward pass —
  a *multiplier*, not just an init change, because the layer keeps training.
- **Attention logits.** Standard attention divides `q·k` by `√d_h`, which is correct when
  `q` and `k` are independent random vectors: a sum of `d_h` independent products has
  standard deviation `√d_h`. After training under muP, `q` and `k` become *aligned* —
  correlated at `Θ(1)` — so the sum grows like `d_h`, not `√d_h`, and the correct divisor
  is `d_h`. Normalised to agree at base width: `scale(d_h) = √d_h₀ / d_h`.

**Worked, so the numbers are concrete.** Base `d₀ = 256` with 8 heads (`d_h₀ = 32`),
target `d = 1024` with 8 heads (`d_h = 128`), so `m = 4`:

```
hidden init std    0.02   → 0.02/√4  = 0.01
hidden Adam LR     1e-2   → 1e-2/4   = 2.5e-3
readout multiplier 1.0    → 1/4      = 0.25
attention scale    1/√32 = 0.17678 (base, both schemes agree)
                   standard at d_h=128:  1/√128    = 0.08839
                   muP      at d_h=128:  √32/128   = 0.04419      ← exactly 2× smaller
```

**A practical note that catches people.** If you widen by *adding heads at fixed
`head_dim`* — which is what most codebases do, and what Laguna's explicit
`head_dim: 128` forces `[M]` — then `d_h` never changes, the attention-logit rule is a
no-op, and muP reduces to the init and LR rules. Knowing which of your dials moves `d_h`
is the difference between implementing muP and implementing two thirds of it.

**How you verify it works: the coordinate check.** Instrument the per-tensor
`‖ΔW‖ / ‖W‖` ratio (update-to-parameter) and the per-layer activation RMS, run 10–20
steps at two widths, and confirm neither drifts with width. If the update-to-parameter
ratio at width 1024 is 4× the one at width 256, your muP is wrong regardless of what the
loss curve says. Those two fields already exist in this lab's telemetry schema —
`update_to_param_ratio_p50` and `update_to_param_ratio_max` in
`research/notes/pretraining-recipes.md` §9 — and this is what they are for.

### 3.8 Depth transfer: the contested part

Classic muP is a **width** theory. Optimal hyperparameters do not automatically transfer
across depth, and the literature has not converged on the fix:

- `[C]` Tensor Programs VI / Depth-μP ([2310.02244](https://arxiv.org/abs/2310.02244),
  Oct 2023) extends the framework to depth for residual networks with a per-branch
  multiplier and per-layer LR scaling.
- `[C]` CompleteP ([2505.01618](https://arxiv.org/abs/2505.01618), May 2025, Cerebras)
  claims depth-wise HP transfer **and** non-lazy learning in all layers, 12–34% compute
  efficiency over the prior state of the art, and — directly relevant here — that it
  "enables a wider range of model width/depth ratios to remain compute-efficient". If
  that holds, aspect ratio becomes a free variable you tune for hardware rather than a
  constrained one.
- `[C]` [2512.22382](https://arxiv.org/abs/2512.22382) (Apple, ICLR 2026) is the
  practical superset: transfer across modules, width, depth, batch size and duration,
  reporting transfer to a ~14,000× larger FLOP budget.
- `[C]` [2603.00541](https://arxiv.org/abs/2603.00541) (Feb 2026, rev. May 2026) gives a
  spectral framework for joint width–depth scaling and argues that transformer blocks —
  which apply **two or more** transformations per residual branch — sit in a regime
  where the single-transformation prescriptions that much earlier theory assumed
  actually fail to transfer.
- `[C]` [2604.27077](https://arxiv.org/abs/2604.27077) (Apr 2026) tackles the same
  problem for normalized (nGPT-style) transformers, which do not transfer under the
  original design.
- `[C]` For MoE, none of the above is sufficient: routing and sparsity sit outside
  classic muP theory ([2508.09752](https://arxiv.org/abs/2508.09752),
  [2605.14200](https://arxiv.org/abs/2605.14200)). **A `proteus-moe-*` arm muP'd with
  plain Tensor Programs V is not muP'd.**

Present this as contested in any write-up. What is *not* contested is the negative
result: run a depth sweep under plain width-muP and assume the LR transfers, and you
have an uncontrolled experiment.

---

## 4. Why it matters for Proteus

**The config surface is the experimental surface** (house rule), so here is the slice of
it this module owns, and what each field does to the experiment:

| Field | What it moves | Trap |
|---|---|---|
| `num_hidden_layers` | trunk params linearly; attention FLOPs as `L`; **KV bytes as `L`** | the only field that moves KV cost without touching an attention knob |
| `hidden_size` | trunk params quadratically; embedding params linearly | quantised to multiples of 64/128 by kernel geometry |
| `head_dim` | KV bytes linearly; attention params | if set explicitly, breaks head tiling and every convention-based formula |
| `num_key_value_heads` | KV bytes linearly; arithmetic intensity `G` | the *only* head field in the KV formula |
| `num_attention_heads` | attention params and prefill FLOPs, **not** KV bytes | `[M]` varies per layer in Laguna (48 full / 72 sliding) |
| `intermediate_size` | trunk params linearly | the `8/3` SwiGLU convention exists to preserve `12Ld²` |
| `vocab_size`, `tie_word_embeddings` | up to 58% of total params at our scale | decides whether "matched params" means anything |
| `initializer_range` | forward variance at step 0 | a constant `0.02` is not width-aware |
| init method (`normal` / `llama` / `llama_depth` / `fan_in`) | residual variance growth with depth | two of the four shipping rules give different depth behaviour (§3.6) |
| muP base shape (`base_width`, `base_head_dim`) | whether any cross-scale claim is valid | must be recorded in every run's config hash |

Four consequences specific to this lab:

1. **Aspect ratio is a Mnemosyne axis.** `research/memory/open-problems-ranked.md`
   observes that at our scale the KV cache can be 100× the model. §3.3 shows the shape
   choice moves KV bytes per token by 4× at fixed parameter count. Against the `[M]`
   ≥62 GiB fast tier measured on the Z13 (`notebook/uma-carveout-controls-fast-tier.md`,
   2026-07-26, single run per arm), the deep-thin 75M-param shape at 32k context needs
   3.00 GiB of cache and the wide-shallow one 0.75 GiB. Either fits — but the *ratio*
   between "cache" and "model" that this lab's whole thesis rests on is under your
   control through a field nobody thinks of as a memory field.
2. **muP is a precondition, not a nicety.** `open-problems-ranked.md` item 3 requires
   every arm to run at ~30M and ~300M and report Spearman rank correlation of the arm
   ordering. Without muP, a rank inversion is indistinguishable from one scale being
   better tuned. `ASSUMPTIONS.md: ablation-scale-sufficient` is `untested` and is named
   the riskiest assumption in the program. This module's Exercise C is the cheapest
   thing that moves it.
3. **The 32 GiB single-tensor fault is a vocabulary problem, not a shape problem.**
   `[M]` A 32 GiB buffer hard-hangs on this machine at 0% CPU with no error
   (`ASSUMPTIONS.md: large-tensor-fault-32gib`). The cross-entropy logits tensor is
   `T_micro × V × 4` bytes and is sized by vocabulary and microbatch — not by `L` or `d`.
   Widening the model does not put you near the cliff; widening the vocab or the
   microbatch does. Assert `T_micro × V × 4 ≤ 8 GiB` in the config validator.
4. **Our anchor's own initialization is not what a modern recipe would choose.** `[M]`
   Laguna S 2.1 is 48 layers at width 3072 and its HuggingFace `_init_weights` applies a
   flat `std = self.config.initializer_range` = 0.02 with **no fan-in scaling and no
   depth scaling** (`modeling_laguna.py:552`; default at `configuration_laguna.py:149`).
   Be careful what you conclude: that code path runs only for from-scratch
   initialization, not for loading the released checkpoint, so it tells you what a
   *retrain* of this architecture in `transformers` would do — not what Poolside did.
   Treat it as an inherited default in the reference implementation, and as an argument
   for making the init method an explicit Proteus config field rather than a constant.

---

## 5. Read the code

Paths are relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. Read in this order.

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:109`<br>`class GPTConfig` | Six dataclass fields are the entire model surface: `block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`. Note what is *derived* rather than configurable — head dim is `n_embd//n_head`, FFN width is hardwired to `4*n_embd`. Every convention §3.2 relies on is baked in here rather than stated. |
| `training/nanogpt/model.py:150`<br>`def get_num_params` | The parameter count with its one subtlety spelled out in the docstring: position embeddings are subtracted, token embeddings are not, *because weight tying makes them the readout matrix*. This is the ambiguity of §3.4 sitting in six lines of production code. |
| `training/nanogpt/model.py:162`<br>`def _init_weights` | The pre-muP world in five lines: every Linear and every Embedding gets `std=0.02`, independent of fan-in, width, or depth. This is the thing §3.5 and §3.7 exist to replace. Note that biases are zeroed and norms are untouched. |
| `training/nanogpt/model.py:145`<br>`std=0.02/math.sqrt(2 * config.n_layer)` | The `1/√(2L)` residual fix, applied by *name matching* on `c_proj.weight` after the generic init has already run. Two things to notice: it is a second pass that overwrites the first, and the selector is a string suffix — rename a projection and the fix silently stops applying. |
| `training/nanogpt/model.py:296`<br>`flops_per_token = 6*N + 12*L*H*Q*T` | The FLOP model of §3.3, verbatim. Confirm for yourself that `H*Q` is `n_embd`, i.e. the term is `12·L·d·T`, and that this is what makes deep-thin more expensive per token at matched `N`. Note the surrounding `estimate_mfu` divides by a hardcoded A100 `312e12`, which is meaningless on gfx1151. |
| `training/nanogpt/transformer_sizing.ipynb:75`<br>`def params():` | A per-component parameter ledger — `embedding/position`, `embedding/token`, `attention/kqv`, `attention/proj`, `mlp/ffw`, `mlp/proj` — that reconciles against the model's own count. This is the shape of the table Exercise A asks you to produce, written by the author of the model. |
| `training/olmo-core/src/olmo_core/nn/transformer/init.py:46`<br>`class InitMethod` | Four named initialization schemes as a config enum — `normal`, `normalized`, `llama`, `llama_depth`, plus `fan_in`. Read the docstrings first: this is the clearest statement anywhere in the reference library that initialization is an ablation axis and not a constant. |
| `training/olmo-core/src/olmo_core/nn/transformer/init.py:73`<br>`std = 1/√d_in` | The fan-in rule of §3.5, as documentation, then applied at `:156`, `:165`, `:176` — note it is computed *per tensor* from `in_features`, so `w1`, `w2`, `w3` of the same MLP get three different standard deviations. |
| `training/olmo-core/src/olmo_core/nn/transformer/init.py:167`<br>`std = std / (2 * num_blocks) ** 0.5` | The global `1/√(2L)` rule. Compare with the very next branch. |
| `training/olmo-core/src/olmo_core/nn/transformer/init.py:169`<br>`std = std / (2 * (block_idx + 1)) ** 0.5` | The per-layer-index variant. Two rules, adjacent lines, different asymptotics (§3.6). Nothing in the file says which to prefer — that is the honest state of the art, exposed as a config choice. |
| `training/olmo-core/src/olmo_core/nn/attention/__init__.py:1143`<br>`std = std / (2 * num_blocks) ** 0.5` | The same depth division applied to the attention output projection, confirming the `2L` in the denominator counts *sublayers* (attention + MLP), not blocks. |
| `training/olmo-core/src/olmo_core/nn/transformer/model.py:982`<br>`def num_flops_per_token` | Production-grade FLOP accounting: delegated per block and summed, rather than a closed form. Read it against nanoGPT's one-liner — the delegation is what lets a hybrid stack with per-layer attention types report a correct number. |
| `models/laguna-s/modeling_laguna.py:552`<br>`std = self.config.initializer_range` | A 2026 48-layer frontier architecture initializing with a flat 0.02 and no depth or fan-in scaling. Before concluding anything: this path runs on from-scratch construction, not on `from_pretrained`. |
| `models/laguna-s/configuration_laguna.py:149`<br>`initializer_range: float = 0.02` | The constant itself, still carrying GPT-2's value into a model with 48 layers and width 3072. |

---

## 6. Exercises

All three run inside the lab venv. Activate with:

```powershell
cd C:\projects\School\chiron
. .\scripts\activate-lab.ps1
```

**gfx1151 caveats that apply to all three.** `ASSUMPTIONS.md: bf16-numerics-unproven` is
still open and the Hardware Validation Gate has not run, so **do every variance and
initialization measurement in fp32** — a variance ratio is exactly the kind of quantity a
bf16 bug corrupts silently. Keep every individual tensor under 32 GiB
(`large-tensor-fault-32gib`; a hang here presents at 0% CPU with no error). Runtime
estimates below derive from the `[A]` 6 TFLOP/s sustained-training figure in
`research/notes/pretraining-recipes.md` §5, which is itself unverified — treat them as
order-of-magnitude.

### Exercise — `count_params_from_config`: reproduce a real parameter count from arithmetic alone

**Difficulty 2/5. ~1 hour. CPU-only; no GPU needed at all.**

Write a function that takes `(L, d, n_q, n_kv, d_h, F, V, tied, block_size, mlp_kind)`
and returns a dict of per-component parameter counts plus a total, following §3.1.

1. Instantiate nanoGPT's `GPT` with the `shakespeare_char` config
   (`L=6, n_head=6, d=384, block_size=256, bias=False`, `vocab_size=65`) and compare
   your closed form against `model.get_num_params(non_embedding=False)`.
   **Pass condition: exact equality, to the parameter.** The expected value is
   `10,745,088`; non-embedding as nanoGPT defines it is `10,646,784`. If you are off,
   the missing term is almost always the RMSNorm/LayerNorm gains or the position table.
2. Sweep the iso-parameter family `L·d² = 6,291,456` over
   `(L, d) ∈ {(6,1024), (24,512), (96,256)}` and produce one table with four columns:
   non-embedding params, embedding fraction at `V ∈ {65, 8192, 50257, 100352}`,
   KV bytes per token in bf16, and attention FLOP fraction at `T ∈ {256, 1024, 8192}`.
3. **Plot** aspect ratio (x, log scale) against KV bytes per token and against total
   FLOPs per token (two y-series). You should reproduce the 4× KV spread and the 1.43×
   FLOP spread from §3.3.

**Checkable output:** the exact-match assertion in step 1, and a plot whose KV curve is a
straight line of slope −1 in `d` on log-log axes.

**CPU fallback:** this *is* the CPU version. It never allocates a device tensor.

### Exercise — `residual_variance_vs_depth`: measure the accumulator growing

**Difficulty 3/5. ~1.5 h. GPU ~5 min of compute; CPU ~10 min.**

Instantiate nanoGPT's `GPT` at fixed width `d = 384` and depths
`L ∈ {2, 4, 8, 16, 32, 64}`, in **fp32**, with `torch.manual_seed` fixed. Register a
forward hook on each block that records the RMS of the residual stream (`x`) leaving it.
Run one forward pass on a batch of random token ids, `block_size = 64`, batch 8. No
training, no backward.

Three arms:

- **scaled** — the model as shipped (nanoGPT already applies `0.02/√(2L)` to `c_proj`).
- **unscaled** — after construction, re-initialize every `c_proj.weight` with
  `std=0.02`, removing the fix.
- **per-index** — re-initialize block `l`'s `c_proj` weights with `0.02/√(2(l+1))`, the
  olmo-core `llama_depth` rule.

Produce two plots: (a) residual RMS versus block index, one curve per depth, per arm;
(b) **final-block RMS versus `L` on log-log axes**, one line per arm.

**Predictions to check, from §3.6.** On plot (b): the unscaled arm should have slope
≈ 0.5 (RMS growing as `√L`); the scaled arm should be approximately flat (slope ≈ 0); the
per-index arm should sit between them, growing like `√(ln L)` — over `L = 2 → 64` that is
a factor of `√(H_64/H_2) ≈ √(4.74/1.5) ≈ 1.78`, versus `√32 ≈ 5.66` for the unscaled
arm. Report the fitted slopes with an R². If the scaled arm's slope is not within about
±0.05 of zero, look at whether dropout is on (set `dropout=0.0`) and whether you are
measuring before or after the final `ln_f`.

**Why this is worth the 90 minutes:** it converts §3.6 from an assertion into a number
you generated, and it is the only exercise here that produces evidence about the `[A]`
claim in §3.6 that the per-index rule does not flatten the growth.

**CPU fallback:** identical, add `device='cpu'`. The largest tensor is
`8 × 64 × 384` floats. Roughly 2× slower and still under ten minutes.

### Exercise — `lr_transfer_across_width`: does the optimum move, and does muP hold it still?

**Difficulty 4/5. ~1 h GPU wall-clock for the sweep plus ~1 h to write it; ~2 h on CPU with the reduced grid.**

This is the experiment that decides whether anything else this lab measures at 20M
transfers to 300M. It is also the one most likely to produce a noisy null, so read the
caveats before you run it.

Setup: nanoGPT on `shakespeare_char`, `n_layer=4`, `block_size=128`, `batch_size=32`,
1500 iterations, `dropout=0.0`, `weight_decay=0.0`, AdamW `β=(0.9, 0.95)`, warmup 100
iters then constant LR (no decay — you are locating an optimum, not producing a model).

Grid: widths `d ∈ {128, 512}` (so `m = 4`), learning rates log-spaced over five points
from `1e-3` to `3e-2`, two parameterizations, **three seeds** (house minimum; a
single-seed result is an anecdote and must be labelled one).

- **standard parameterization** — identical LR at both widths, `std=0.02` everywhere.
- **muP** — at the target width, multiply hidden-matrix init std by `1/√m`, hidden-matrix
  LR by `1/m`, and apply a `1/m` forward multiplier on `lm_head`. Keep the embedding LR
  and the readout LR rules as written in §3.7. Because nanoGPT widens by keeping
  `n_head` fixed and growing `n_embd`, `d_h` *does* change here, so the attention-logit
  rule fires: replace `1/√d_h` with `√d_h₀/d_h`. (`d₀=128` with 4 heads → `d_h₀=32`;
  `d=512` → `d_h=128`.)

Record for each run: best validation loss, and the per-tensor update-to-parameter ratio
`‖ΔW‖/‖W‖` at steps 10, 100 and 1000.

**Checkable outputs.**
1. A plot of validation loss versus LR, four curves (2 widths × 2 parameterizations),
   with seed error bars. **Success for the muP arm:** `argmin` LR at `d=512` within one
   grid step of `argmin` at `d=128`. **Expected for the SP arm:** the optimum moves down
   by roughly a factor of `m = 4`, i.e. two grid steps on this grid.
2. The **coordinate check**: under muP the median update-to-parameter ratio should be
   approximately equal at both widths; under SP it should differ by roughly `m`. This
   diagnostic is more reliable at 1500 iterations than the loss curve is, and it is the
   thing to trust if the two disagree.

**Caveats you must state in the write-up.** At 4 layers and 1500 iterations the loss
signal is weak; three seeds is the floor, not comfort. `weight_decay=0` is deliberate —
`[C]` [2510.19093](https://arxiv.org/abs/2510.19093) argues decoupled weight decay does
much of the work attributed to muP, so leaving WD on would confound the arms. That
choice also means you are testing muP *without* its disputed co-factor, which is a
narrower claim than "muP works". Say so.

**CPU fallback:** widths `{64, 256}`, three LRs (`3e-3`, `1e-2`, `3e-2`), 600 iterations,
`block_size=64`, `batch_size=16`, two seeds — and label the result an anecdote. ~2 hours.

**If you only do one thing:** run the coordinate check alone at 20 steps, both widths,
both parameterizations. That is under five minutes and it catches an incorrect muP
implementation, which is by far the most common outcome of a first attempt.

---

## 7. Self-check

1. Two models have identical non-embedding parameter counts: `L=12, d=768` and
   `L=48, d=384`. Which has the larger KV cache per token under the head-tiling
   convention, and by what factor? Which has more attention FLOPs per token at fixed
   context length, and by what factor?

2. You are asked to run a "matched-parameter" ablation comparing an aspect ratio of 20
   against an aspect ratio of 160 at roughly 100M parameters, with `V = 50,257` and
   untied embeddings. State the specific way this comparison can be rigged without
   anyone lying, and the one sentence you must add to the pre-registration to close it.

3. nanoGPT applies `std = 0.02/√(2·n_layer)` to parameters whose names end in
   `c_proj.weight`. Why `2·n_layer` rather than `n_layer`, and what happens to the
   residual stream's variance at `L=48` if you delete that line?

4. Under muP with Adam, hidden-matrix learning rate scales as `1/m` where `m` is the
   width multiplier. Give the reason this exponent differs from the SGD case, in terms of
   what Adam does to the update magnitude.

5. Your Proteus config sets `hidden_size: 1024`, `num_attention_heads: 8`, and
   `head_dim: 128`. You muP the model from a base width of 512 (also 8 heads,
   `head_dim: 128`). Which of the four muP rules in §3.7 has no effect, and why?

6. A colleague reports that a deep model's last twelve layers have near-identity
   Jacobians and concludes the model is over-parameterized. Give the alternative
   explanation from this module, and name the cheapest measurement that distinguishes
   the two.

---

## 8. What is still unsolved here

1. **The optimal aspect ratio at 20M–300M is unknown, and the literature points in
   opposite directions.** `[C]` Kaplan ([2001.08361](https://arxiv.org/abs/2001.08361))
   says shape barely matters; `[C]` MobileLLM
   ([2402.14905](https://arxiv.org/abs/2402.14905)) says deep-and-thin wins below 1B;
   `[C]` [2207.10551](https://arxiv.org/abs/2207.10551) says architecture changes the
   scaling exponent so any single-scale answer is suspect; `[C]`
   [2606.18246](https://arxiv.org/abs/2606.18246) (Jun 2026) says constant width across
   depth is the wrong premise entirely. Contested; a matched-`N` three-point sweep at 50M
   with ≥3 seeds is inside this lab's budget and would be a real contribution rather
   than a replication.

2. **Depth transfer for hyperparameters is unresolved.** Depth-μP
   `[C]` ([2310.02244](https://arxiv.org/abs/2310.02244)), CompleteP `[C]`
   ([2505.01618](https://arxiv.org/abs/2505.01618)), CompleteHP `[C]`
   ([2512.22382](https://arxiv.org/abs/2512.22382)) and the 2026 spectral treatment
   `[C]` ([2603.00541](https://arxiv.org/abs/2603.00541)) do not agree on the depth rule,
   and the last of these argues that transformers — with two or more transformations per
   residual branch — are in the regime where the earlier prescriptions fail. Nothing in
   the reference library implements any of them, so an ablation here starts with an
   implementation.

3. **Whether muP's active ingredient is muP.** `[C]`
   ([2510.19093](https://arxiv.org/abs/2510.19093)) credits decoupled weight decay;
   `[C]` ([2605.21486](https://arxiv.org/abs/2605.21486)) credits the embedding learning
   rate. Exercise C deliberately holds weight decay at zero to isolate one of these, and
   that means it cannot answer the practical question — which is what happens with weight
   decay on, as every real recipe has it.

4. **muP for MoE is a different parameterization and we do not have one.** `[C]`
   ([2508.09752](https://arxiv.org/abs/2508.09752),
   [2605.14200](https://arxiv.org/abs/2605.14200)). Any `proteus-moe-*` arm inherits this
   gap, and the honest position is that a sparse arm tuned under width-muP is untuned.

5. **The curse of depth has three competing fixes and no head-to-head.** Init rescaling
   (§3.6), norm placement (`[C]` Peri-LN,
   [2502.02732](https://arxiv.org/abs/2502.02732)), and depth-scaled norm gains `[C]`
   ([2502.05795](https://arxiv.org/abs/2502.05795)) all target the same mechanism.
   Sparsity interacts `[C]` ([2603.15389](https://arxiv.org/abs/2603.15389), Mar 2026),
   so the dense answer may not be the MoE answer. `research/notes/transformer-state-of-the-art.md`
   §11 lists this as contested and it stays contested here.

6. **The aspect-ratio ↔ KV-cost tradeoff has no small-scale study I could find.** The
   closest published work is `[C]`
   ([2510.18245](https://arxiv.org/abs/2510.18245), Oct 2025, rev. May 2026, ICLR 2026),
   which conditions a Chinchilla-style law on hidden size, MLP-to-attention allocation,
   and GQA for inference efficiency and reports 42% higher inference throughput than
   LLaMA-3.2 at matched accuracy. It does not treat depth as an explicit KV-budget lever
   at ablation scale. `[A]` Low-to-medium confidence that this is a genuine hole rather
   than a search failure on my part — the cheapest way to find out is to read
   2510.18245's related-work section before designing an arm.

7. **Our own scale-transfer assumption is untested.** `ASSUMPTIONS.md:
   ablation-scale-sufficient` is `[A]` medium confidence with zero evidence, and
   `research/memory/open-problems-ranked.md` names it the riskiest assumption in the
   program. Everything in this module is machinery for attacking it. Nothing in this
   module attacks it on its own.

8. **A methodological hazard that lands directly on a shape sweep.** `[C]`
   ([2603.22339](https://arxiv.org/abs/2603.22339), Mar 2026) shows the standard
   IsoFLOP parabola fit is biased even on noise-free data, and the bias sources it names
   — narrow grid, off-centre sampling, loss-surface asymmetry — are exactly the
   conditions of a three-point aspect-ratio sweep at one scale. If you fit anything, use
   Approach 3 with variable projection and report an interval.

---

## 9. Answers to the self-check

**1.** `L=12, d=768`: `12 · 768² = 7,077,888`; `L=48, d=384`: `48 · 147,456 =
7,077,888`. Matched, as claimed.

- KV bytes per token `= 2·L·d·b` under head tiling with `G=1`, bf16 (`b=2`):
  `2·12·768·2 = 36,864 B` versus `2·48·384·2 = 73,728 B`. **The deep model is 2×**, and
  the factor is exactly `L₂/L₁ = 4` divided by `d₁/d₂ = 2`.
- Attention FLOPs per token `= 12·L·d·T`, which contains the same `L·d` product, so it is
  **also 2×** in favour of the shallow model, at any fixed `T`. Total FLOPs are less
  lopsided because `6N` is identical. `N_trunk = 12 · 7,077,888 = 84,934,656`, so
  `6N = 509,607,936`; at `T = 1024` the attention terms are `12·12·768·1024 =
  113,246,208` and `12·48·384·1024 = 226,492,416`. Totals: `622,854,144` versus
  `736,100,352` FLOPs per token — **1.18× overall**, against 2× on the attention term
  alone and 2× on KV bytes. The lesson is that FLOPs hide the shape difference and the
  cache does not.

**2.** The rig is the embedding table. At `V = 50,257` untied, embeddings cost
`2·V·d`, which is linear in `d` and independent of `L`. So the wide arm carries far more
embedding parameters, and if you match *total* parameters, the wide arm gets a much
smaller trunk and loses for a reason that has nothing to do with aspect ratio. Nobody has
to lie: "matched parameters" is simply ambiguous. The sentence to add: **"Arms are
matched on non-embedding parameter count; `vocab_size = 50257` and
`tie_word_embeddings = false` are identical across arms and total parameter counts
therefore differ, reported per arm."**

**3.** `2·n_layer` because there are **two** sublayers per block — attention and MLP —
each of which performs one write into the residual stream, so `L` blocks produce `2L`
writes. The variance contributions add, so dividing each contribution's standard
deviation by `√(2L)` divides its variance by `2L` and makes the total added variance
`2L · s²/(2L) = s²`, independent of depth. Delete the line and the added variance becomes
`2L·s²`: at `L=48` that is `96·s²` instead of `s²`, i.e. the residual stream's standard
deviation is about `√96 ≈ 9.8×` larger at the top of the stack than the fix intends.
Practically, this pushes the pre-norm denominator up, shrinks each layer's relative
contribution, and is the mechanism behind late-layer degeneracy.

**4.** Under SGD the update is `η · gradient`, so the update magnitude inherits the
gradient's scale, and the muP exponent has to correct for how the gradient scales with
width. Under Adam the update is `η · m/(√v + ε)`, and that ratio is approximately
unit-magnitude per coordinate **regardless of the gradient's scale** — Adam has already
normalised it away. So the remaining width dependence is purely structural: a layer with
`m` times as many inputs, each moving by `≈η`, produces an output change `m` times
larger. Dividing `η` by `m` restores a `Θ(1)` change in the layer's output. Different
optimizer, different residual width-dependence, different exponent.

**5.** The **attention-logit rule** has no effect. `head_dim` is set explicitly to 128 at
both the base width and the target width, so `d_h₀ = d_h = 128`, and
`√d_h₀ / d_h = 1/√d_h` — the muP scale and the standard `1/√d_h` scale are the same
number. You widened by adding heads (well, by widening `hidden_size` with `head_dim`
pinned), not by widening heads. The init, LR and readout-multiplier rules all still
apply. This is the Laguna-style configuration, and it is the common case in practice.

**6.** The alternative explanation is the **curse of depth**: under pre-norm without
adequate residual scaling, the residual stream's variance grows as `L`, so the pre-norm
divides by a progressively larger quantity and each block's contribution shrinks as
`1/√l` — the late blocks are near-identity because of the *parameterization*, not because
the capacity is unused. The cheapest distinguishing measurement is the one from Exercise
B: log the residual-stream RMS per block index. If RMS grows monotonically as roughly
`√l` across the stack, you are looking at the parameterization; if RMS is flat and the
late blocks are still near-identity, the over-parameterization reading survives. A second
cheap probe: re-run with the `1/√(2L)` scaling applied (or with Peri-LN) and see whether
the late-layer Jacobians move.

---

## Sources

**Local, read directly** (paths relative to repo root; clones gitignored, rebuild with
`scripts/fetch_reference.sh`, revisions in `research/reference/PROVENANCE.md`):
`research/reference/training/nanogpt/model.py`, `config/train_shakespeare_char.py`,
`train.py`, `transformer_sizing.ipynb`;
`research/reference/training/olmo-core/src/olmo_core/nn/transformer/init.py`,
`.../nn/transformer/model.py`, `.../nn/attention/__init__.py`;
`research/reference/models/laguna-s/config.json`, `configuration_laguna.py`,
`modeling_laguna.py`; `research/reference/CODE_MAP.md`;
`research/reference/papers/README.md`.

**Lab records used as `[M]`:** `ASSUMPTIONS.md`
(`gpu-fast-tier-size` ≥62 GiB at ~200 GB/s; `large-tensor-fault-32gib`;
`kv-per-token-laguna` 192 KiB/token; `laguna-heads-uniform`; `reference-model`),
`notebook/uma-carveout-controls-fast-tier.md`, `ENVIRONMENT.md` (torch
`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0, gfx1151, native Windows).

**Consistent with, and not restating:** `research/memory/kv-cache-mechanics.md` (the
`2·L·n_kv·d_h·b` formula and the `2G/dtype_bytes` intensity result),
`research/memory/open-problems-ranked.md` (items on cross-scale rank correlation),
`research/notes/pretraining-recipes.md` §3 (muP), §5 (wall-clock), §9 (telemetry
schema), `research/notes/transformer-state-of-the-art.md` §6 (Laguna's aspect ratio and
parameter ledger), §7 (the embedding tax), §11 (contested norm placement).

**Papers.** Each id below was resolved against arxiv.org on 2026-07-26, or was already
resolved in `research/reference/papers/README.md`. Resolution proves the paper exists,
not that it supports the claim beside it.

Shape and scaling — [2001.08361](https://arxiv.org/abs/2001.08361) Scaling Laws for
Neural Language Models · [2006.12467](https://arxiv.org/abs/2006.12467) The
Depth-to-Width Interplay in Self-Attention ·
[2207.10551](https://arxiv.org/abs/2207.10551) Scaling Laws vs Model Architectures ·
[2402.14905](https://arxiv.org/abs/2402.14905) MobileLLM ·
[2606.18246](https://arxiv.org/abs/2606.18246) Variable-Width Transformers ·
[2510.18245](https://arxiv.org/abs/2510.18245) Scaling Laws Meet Model Architecture ·
[2203.15556](https://arxiv.org/abs/2203.15556) Chinchilla ·
[2603.22339](https://arxiv.org/abs/2603.22339) Problems with Chinchilla Approach 2 ·
[2204.02311](https://arxiv.org/abs/2204.02311) PaLM (FLOP accounting).

Architecture conventions — [2002.05202](https://arxiv.org/abs/2002.05202) GLU Variants ·
[2305.13245](https://arxiv.org/abs/2305.13245) GQA ·
[2407.13623](https://arxiv.org/abs/2407.13623) Scaling Laws with Vocabulary ·
[1608.05859](https://arxiv.org/abs/1608.05859) Using the Output Embedding.

Initialization, depth, stability — [2203.00555](https://arxiv.org/abs/2203.00555)
DeepNet · [2502.05795](https://arxiv.org/abs/2502.05795) The Curse of Depth ·
[2502.02732](https://arxiv.org/abs/2502.02732) Peri-LN ·
[2603.15389](https://arxiv.org/abs/2603.15389) When Does Sparsity Mitigate the Curse of
Depth · [2312.16903](https://arxiv.org/abs/2312.16903) Spike No More ·
[2309.14322](https://arxiv.org/abs/2309.14322) Small-scale proxies for large-scale
Transformer training instabilities.

Hyperparameter transfer — [2203.03466](https://arxiv.org/abs/2203.03466) Tensor Programs
V (muP) · [2310.02244](https://arxiv.org/abs/2310.02244) Tensor Programs VI (Depth-μP) ·
[2505.01618](https://arxiv.org/abs/2505.01618) CompleteP ·
[2512.22382](https://arxiv.org/abs/2512.22382) Completed Hyperparameter Transfer ·
[2603.00541](https://arxiv.org/abs/2603.00541) Spectral Condition for μP under
Width–Depth Scaling · [2604.27077](https://arxiv.org/abs/2604.27077) Learning Rate
Transfer in Normalized Transformers ·
[2510.19093](https://arxiv.org/abs/2510.19093) Weight Decay may matter more than muP ·
[2605.21486](https://arxiv.org/abs/2605.21486) Quantifying Hyperparameter Transfer and
the Importance of Embedding Layer Learning Rate ·
[2508.09752](https://arxiv.org/abs/2508.09752) μ-Parametrization for Mixture of Experts ·
[2605.14200](https://arxiv.org/abs/2605.14200) From muP to the Maximally Scale-Stable
Parameterization.

Capacity — [2505.24832](https://arxiv.org/abs/2505.24832) How much do language models
memorize? (~3.6 bits per parameter).
