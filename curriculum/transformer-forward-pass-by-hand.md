---
title: The transformer forward pass, by hand
version: 1.0.0
date: 2026-07-26
track: A — Foundations
prereqs: tensors-and-autograd
estimated_effort: 5–6 hours (1.5 h reading, 4 h exercises)
---

# The transformer forward pass, by hand

## What this module settles

One decoder block is a function from a `(T, d_model)` tensor to a `(T, d_model)` tensor,
and by the end of this module you can compute one on paper for a four-channel toy config
and check every intermediate against PyTorch to six decimal places. You will know exactly
which tensors are transient and which two — **K** and **V** — are the only ones that must
survive to the next token, because that boundary is the entire interface Mnemosyne owns.
You will also know why the attention matrix you compute by hand does not exist in memory
when the fast kernel runs, which is the root cause of the attribution problem that the
memory track is organised around.

---

## Theory in plain language

### What a decoder block is

A decoder block preserves shape. Tokens in, same number of tokens out, same width. That
single fact is what makes a stack of them possible: forty-eight blocks in Laguna-S are
forty-eight applications of the same signature, `(T, d) -> (T, d)` `[M]` (48 layers, read
from the shipped config; see `ASSUMPTIONS.md → reference-model`).

Inside, exactly two things happen, and everything else is plumbing:

1. **Mixing across positions.** Attention. Token 5's output can depend on tokens 0–5.
   This is the only place in the whole architecture where information crosses the time
   axis.
2. **Mixing across channels.** The MLP. Applied identically and independently to every
   position — token 5's MLP cannot see token 4. It is a per-row function.

The norms and the residual additions are not part of the computation in any interesting
sense; they are what makes a deep stack trainable at all. Get the two mixing steps right
and you understand the block.

### What it replaced, and why that matters here

Before 2017 the standard sequence model was recurrent: a fixed-size hidden state carried
forward one token at a time, each step overwriting the state with a function of the old
state and the new token. You already own this mental model — it is a **rolling aggregate**
over a stream. Constant memory, constant per-token cost, and lossy by construction,
because everything the model will ever know about token 5 has to survive inside a
fixed-width vector until it is needed.

Attention replaced the rolling aggregate with something closer to a **full table scan over
an append-only log with a learned similarity predicate**. Every past token keeps its own
row. The current token forms a query, scores it against every stored key, and reads back a
weighted average of the stored values. Nothing is compressed away; nothing is overwritten.

That is the bridge. Here is where it breaks, and the break is the reason this lab exists:

- **There is no index.** Every query touches every row. Not "in the worst case" — always,
  by definition. A cache whose every read is a full scan has no temporal locality to
  exploit, so the ordinary caching intuitions (working set, hit rate, prefetch) do not
  transfer. See `research/memory/kv-cache-mechanics.md`, section 3.
- **There is no miss path.** A row is present or the token does not exist. Inside the
  attention kernel a miss is not slow — it is unrepresentable.
- **The log is strictly append-only and immutable per token.** Once token 5's K and V are
  written they are never updated. This is unusually convenient and it is why "eviction"
  in this domain means destruction, not demotion.

The constant-state models (Mamba, Gated DeltaNet) are the pendulum swinging back to the
rolling aggregate with better machinery. That is Track C's `constant-state-memory.md`. You
need this module first, because the SSM literature is written as a contrast against
exactly the computation below.

### The residual stream

A useful frame, and one that will pay off later. The `(T, d_model)` tensor threading
through the whole network is a **bus**. Each sublayer reads the bus, computes a
contribution, and *adds* it back. Nothing is replaced.

```
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Bridge: additive contributions to a shared bus, like a set of processes accumulating into
one shared counter array. Where it breaks: there is no addressing and no ownership. Every
sublayer writes into the same `d_model`-dimensional vector space with no allocation scheme,
so two layers can and do use overlapping directions for unrelated purposes. Interference
between them is not a bug to be fixed; it is the operating regime. That is one reason
"which head does what" is an empirical question with no reliable answer — see the open
problems at the end.

---

## The math that actually matters

### Notation, with every symbol in words

| Symbol | Words | Toy value | nanoGPT shakespeare-char `[M]` |
|---|---|---|---|
| `T` | number of tokens in the sequence right now | 3 | 256 (`block_size`) |
| `d_model` | width of the residual stream; channels per token | 4 | 384 (`n_embd`) |
| `n_q` | number of query heads | 2 | 6 (`n_head`) |
| `n_kv` | number of key/value heads (`= n_q` for MHA) | 2 | 6 |
| `d_h` | channels per head; here `d_model / n_q` | 2 | 64 |
| `d_ff` | width of the MLP's hidden layer | 4 | 1536 (`4 * n_embd`) |
| `V` | vocabulary size | 6 | 65 |
| `B` | batch size (sequences processed together) | 1 | 64 |
| `γ` | learned per-channel gain in RMSNorm, one scalar per channel | all 1 | learned |
| `ε` | small constant inside the norm's square root, guards divide-by-zero | see below | `1e-5` |

The nanoGPT column is `[M]`, read from
`training/nanogpt/config/train_shakespeare_char.py:22` (`n_layer=6, n_head=6, n_embd=384`)
and line 19 (`block_size = 256`).

Throughout I use the **row-vector convention**: `x` is a row of shape `(1, d_model)` and a
projection is `x @ W` with `W` of shape `(in, out)`. PyTorch's `nn.Linear` stores its
weight as `(out, in)` and computes `x @ weight.T`. So to load the matrices below into a
`nn.Linear` you must transpose them. This is not pedantry — it is the single most common
reason a hand-check disagrees with the framework, and Exercise one will catch it.

### The toy configuration, in full

Embedding table `E`, shape `(V=6, d_model=4)`:

```
E = [[ 0, 0, 0, 0],
     [ 1, 0, 1, 0],
     [ 0, 2, 0, 0],
     [ 1, 1, 0, 2],
     [-1, 0, 1, 0],
     [ 0, 0, 2, 1]]
```

Projection matrices, all `(4, 4)` except the MLP's, which are `(4, d_ff=4)` and
`(d_ff=4, 4)`:

```
W_Q = [[1,0,0,1],      W_K = [[ 1, 0,1,0],      W_V = I (4x4 identity)
       [0,1,1,0],             [ 0, 1,0,1],
       [1,0,0,-1],            [-1, 0,1,0],
       [0,1,-1,0]]            [ 0,-1,0,1]]

W_O = [[1,0, 0,1],     W_gate = I    W_up = [[0,1,0,0],   W_down = [[ 1, 0,1,0],
       [0,1, 1,0],                          [1,0,0,0],             [ 0, 1,0,1],
       [0,1,-1,0],                          [0,0,0,1],             [-1, 0,1,0],
       [1,0, 0,-1]]                         [0,0,1,0]]             [ 0,-1,0,1]]
```

`W_V = I` is chosen so the weighted-sum step is legible by eye — the attention output is
literally a weighted average of the normalized inputs. In a real model `W_V` is learned and
mixes channels; nothing else changes.

Both RMSNorm gains are `γ = [1,1,1,1]`. Input token ids: `[3, 1, 2]`.

All numbers below are `[M]` computed in fp64 NumPy with `ε = 0`, deterministic and
reproducible by the script you write in Exercise one. Where PyTorch's default `ε = 1e-6`
changes a number, I say so.

### Step: embedding lookup

Input is `(T,) = (3,)` integers. Output is `(T, d_model) = (3, 4)` floats.

This is a **gather**, not a matmul. `E[3]`, `E[1]`, `E[2]` — three row reads.

```
X = [[1, 1, 0, 2],     <- token id 3
     [1, 0, 1, 0],     <- token id 1
     [0, 2, 0, 0]]     <- token id 2
```

Two things worth internalising. First, the embedding table is usually the largest single
parameter tensor in a small model and is touched by exactly `T` row reads per forward pass
— it is sparse-access, unlike every other weight. Second, at this point the model has no
idea what order the tokens are in. Order is injected later, by RoPE (skipped in this toy —
see the note below) or by an additive position embedding as in nanoGPT
(`training/nanogpt/model.py:178`, `pos_emb = self.transformer.wpe(pos)`).

### Step: RMSNorm

```
RMSNorm(x)_i = γ_i · x_i / sqrt( (1/d) · Σ_j x_j²  +  ε )
```

In words: square every channel of this one token's vector, take the mean over channels,
add a tiny epsilon, take the square root — that is the root-mean-square of the vector —
then divide every channel by it and multiply channel `i` by its learned gain `γ_i`.

It normalizes **per token, across channels**. Batch and sequence position never interact.
Unlike LayerNorm `[C]` (1607.06450, Jul 2016) it does not subtract the mean and has no
bias, which is one fewer reduction and one fewer parameter vector; `[C]` RMSNorm
(1910.07467, Oct 2019) showed the re-centring term was doing almost none of the work.

Arithmetic, row by row:

| Row | `x` | `Σx²` | mean = `Σx²/4` | rms | `x / rms` |
|---|---|---|---|---|---|
| 0 | `[1,1,0,2]` | 6 | 1.5 | 1.224745 | `[0.816497, 0.816497, 0, 1.632993]` |
| 1 | `[1,0,1,0]` | 2 | 0.5 | 0.707107 | `[1.414214, 0, 1.414214, 0]` |
| 2 | `[0,2,0,0]` | 4 | 1.0 | 1.000000 | `[0, 2, 0, 0]` |

Call this `H`, shape `(3, 4)`.

Note that row 2 is unchanged — its RMS was already exactly 1. And note `ε`: with `ε = 0`
row 2 is exactly `[0,2,0,0]`; with `ε = 1e-6` (the class default at
`modeling_laguna.py:48`) it is `[0, 1.999999, 0, 0]`. A five-hundred-nanounit difference
that will propagate all the way to the block output. Hold that thought for the self-check.

**Where it matters for numerics:** production RMSNorm upcasts to fp32 before squaring and
casts back afterwards — `modeling_laguna.py:58`, `hidden_states.to(torch.float32)`. Summing
squares in bf16 across 4096 channels is how you lose the norm.

### Step: QKV projection

Three linear maps of the *same* normalized input:

```
Q = H @ W_Q        (T, d_model) @ (d_model, n_q·d_h)  -> (T, n_q·d_h) = (3, 4)
K = H @ W_K        (T, d_model) @ (d_model, n_kv·d_h) -> (T, n_kv·d_h) = (3, 4)
V = H @ W_V        same shape as K
```

Then reshape to split the head axis out: `(T, n_q, d_h) = (3, 2, 2)`, and transpose so the
head axis is outer: `(n_q, T, d_h)`. Head `h` owns columns `[h·d_h : (h+1)·d_h]`. The heads
never interact until the concat at the end — a decoder layer is `n_q` independent attention
computations that happen to be packed into one matmul for throughput.

```
Q = [[0.816497,  2.449490, -0.816497, 0.816497],
     [2.828427,  0,         0,        0       ],
     [0,         2,         2,        0       ]]

K = [[0.816497, -0.816497,  0.816497, 2.449490],
     [0,         0,         2.828427, 0       ],
     [0,         2,         0,        2       ]]

V = H
```

Check one entry by hand: `Q[0,1] = H[0] · W_Q[:,1] = 0.816497·0 + 0.816497·1 + 0·0 +
1.632993·1 = 2.449490`. ✓

**GQA in one line.** If `n_kv < n_q`, `W_K` and `W_V` are *narrower* — shape
`(d_model, n_kv·d_h)` — and each KV head is shared by `G = n_q / n_kv` query heads. In
code that is `repeat_kv` (`modeling_laguna.py:303`), which broadcasts rather than copies.
Nothing else in the block changes. Query-head count does not appear in the KV-cache size
formula at all; only `n_kv` does. `[M]` This is why Laguna's advertised
`num_attention_heads: 48` being wrong for 36 of 48 layers does *not* invalidate the KV
arithmetic (`ASSUMPTIONS.md → laguna-heads-uniform`, `kv-per-token-laguna`).

**Where RoPE goes, and why you should care now.** This toy omits rotary position
embedding, which gets its own module. But note the ordering in production code
(`modeling_laguna.py:389` then `:394` then `:397`): QK-norm is applied, *then* RoPE rotates
Q and K, *then* the rotated K and V are written to the cache. **The cache holds post-RoPE
keys.** Any Mnemosyne scheme that evicts, compacts, or re-packs cache slots is therefore
operating on tensors that already have absolute position baked in — you cannot move a
token to a different slot and expect it to mean the same thing. `[C]` RoFormer
(2104.09864, Apr 2021).

### Step: attention scores, and the scale factor

For each head, with `q_t` the query row for token `t` and `k_s` the key row for token `s`:

```
S[t,s] = (q_t · k_s) / sqrt(d_h)
```

In words: the raw score of token `t` attending to token `s` is the dot product of `t`'s
query with `s`'s key, divided by the square root of the head dimension. `S` has shape
`(n_q, T, T)`.

**Why `sqrt(d_h)` and not `d_h`.** Treat the components of `q` and `k` as independent with
mean 0 and variance 1. Their product has mean 0 and variance 1, and the dot product is a
sum of `d_h` such products, so `Var(q·k) = d_h` and the standard deviation is `sqrt(d_h)`.
Dividing by `sqrt(d_h)` returns the *spread* of the scores to roughly 1 regardless of head
width. Dividing by `d_h` would shrink the spread toward zero as heads got wider, flattening
every attention distribution toward uniform. `[C]` The original argument is a footnote in
1706.03762 (Jun 2017) and it is a variance argument, not an empirical one.

The consequence of getting it wrong is softmax saturation. `[M]` Computed on head 0, row 2
of this toy, varying only the scale:

| scale | softmax row 2 | entropy (nats) |
|---|---|---|
| `1/sqrt(2)` = 0.707107 (correct) | `[0.017284, 0.054843, 0.927874]` | 0.298822 |
| 1.0 | `[0.003501, 0.017923, 0.978576]` | 0.113072 |
| 2.0 | `[0.000013, 0.000335, 0.999652]` | 0.003175 |
| 4.0 | `[0, 0, 1]` | 0.000002 |

At scale 4 the distribution is a hard argmax. A hard argmax has zero gradient with respect
to the losing scores, so training stalls. This is the mechanism behind QK-norm, which
normalizes q and k to fixed length before the dot product so logits cannot run away
regardless of what the weights do (`modeling_laguna.py:368`).

Head 0, raw scores `q·kᵀ` and after scaling by 0.707107:

```
raw                            scaled
[-1.333333, 0, 4.898979]       [-0.942809, 0, 3.464102]
[ 2.309401, 0, 0       ]       [ 1.632993, 0, 0       ]
[-1.632993, 0, 4.000000]       [-1.154701, 0, 2.828427]
```

Verify `S_raw[0,2] = q_0 · k_2 = [0.816497, 2.449490] · [0, 2] = 4.898979`. ✓

### Step: the causal mask

Token `t` may not read token `s > t`. The mask is `(T, T)`, added to the scores before
softmax:

```
M[t,s] = 0     if s <= t
       = -inf  if s > t

M = [[0, -inf, -inf],
     [0,  0,   -inf],
     [0,  0,    0  ]]
```

Adding `-inf` makes `exp` of that entry exactly 0, so the entry contributes nothing to
either the numerator or the denominator of the softmax. The mask is *additive*, not
multiplicative — a multiplicative zero mask would zero the numerator but leave the
denominator wrong.

Two implementation details that bite:

- Production code does not use `-inf`. It uses `torch.finfo(dtype).min`
  (`masking_utils.py:608–610`), which for bf16 is `-3.3895e38`. Reason: a row that is
  *entirely* masked — which happens with sliding windows and with padding — produces
  `exp(-inf - -inf) = NaN` under the max-subtraction trick, whereas finite `min` produces a
  uniform-ish garbage row that at least does not poison the whole tensor. The Triton kernel
  handles the same case explicitly (`triton_attention_helpers.py:431`).
- nanoGPT does it the textbook way, `masked_fill(..., float('-inf'))`
  (`training/nanogpt/model.py:68`), which is clearer and is one of several reasons the
  nanoGPT slow path is the right place to read this.

A sliding-window mask is the same predicate with one extra clause: `kv_idx > q_idx - w`
(`masking_utils.py:99`). That one line is 36 of Laguna's 48 layers.

### Step: softmax

```
A[t,s] = exp(S'[t,s]) / Σ_u exp(S'[t,u])          where S' = S + M
```

In words: exponentiate every score in the row, then divide each by the row's total, so the
row sums to exactly 1. `A` has shape `(n_q, T, T)` and every row is a probability
distribution over the tokens this token is allowed to see.

Nobody computes it that way. `exp(large)` overflows, so every implementation subtracts the
row max first, which is algebraically identical because the constant cancels:

```
A[t,s] = exp(S'[t,s] - m_t) / Σ_u exp(S'[t,u] - m_t)     where m_t = max_u S'[t,u]
```

Worked in full for head 0, row 2. Masked scores `[-1.154701, 0, 2.828427]`, max = 2.828427:

```
shifted:  [-3.983128, -2.828427, 0]
exp:      [ 0.0186273,  0.0591059, 1]
sum:      1.0777332
divide:   [0.017284, 0.054842, 0.927874]
```

The full attention matrices:

```
head 0                                        head 1
[[1,        0,        0       ],              [[1,        0,        0       ],
 [0.836579, 0.163421, 0       ],               [0.5,      0.5,      0       ],
 [0.017284, 0.054843, 0.927874]]               [0.053990, 0.928995, 0.017015]]
```

Three observations you should be able to make from those numbers alone:

1. **Row 0 is `[1, 0, 0]` in both heads and always will be.** The first token can only
   attend to itself, so softmax over a single element is 1 by construction, *regardless of
   content*. Any eviction policy that ranks tokens by accumulated attention mass therefore
   receives a large, content-free vote for token 0 from every single query. That is not the
   whole story of attention sinks, but it is the arithmetic floor under it. `[C]`
   StreamingLLM (2309.17453, Sep 2023) is the paper that made evicting position 0 a known
   way to collapse a model.
2. **Head 1 row 1 is exactly `[0.5, 0.5]`.** Its query vector is `[0, 0]`, so every score is
   0 and softmax returns the uniform distribution. A zero query means "no preference" and
   yields the plain average of everything visible. Useful intuition: softmax attention has
   no way to express "attend to nothing" — the row must sum to 1. This is exactly the
   constraint that Softpick and the gated-attention line are attacking; see the open
   problems.
3. **Row sums are 1.000000 to fp64.** Cheapest possible assertion in any attention test.

### Step: the weighted sum

```
O[t] = Σ_s A[t,s] · V[s]        shape (n_q, T, d_h)
```

In words: token `t`'s output for this head is the average of all visible value vectors,
weighted by the attention it assigned to each.

Head 0, row 1: `0.836579·[0.816497, 0.816497] + 0.163421·[1.414214, 0] =
[0.683064 + 0.231112, 0.683064] = [0.914176, 0.683064]`. ✓

```
head 0 O                     head 1 O
[[0.816497, 0.816497],       [[0,        1.632993],
 [0.914176, 0.683064],        [0.707107, 0.816497],
 [0.091671, 1.869859]]        [1.313797, 0.088166]]
```

### Step: concat heads, output projection

Concatenate along the channel axis — `(n_q, T, d_h) -> (T, n_q·d_h) = (3, 4)` — then one
more linear map back to `d_model`:

```
concat = [[0.816497, 0.816497, 0,        1.632993],
          [0.914176, 0.683064, 0.707107, 0.816497],
          [0.091671, 1.869859, 1.313797, 0.088166]]

attn_out = concat @ W_O
         = [[2.449490, 0.816497,  0.816497, -0.816497],
            [1.730673, 1.390171, -0.024043,  0.097679],
            [0.179837, 3.183656,  0.556062,  0.003506]]
```

`W_O` is where the heads finally mix. Until this matmul each head's output occupies its own
disjoint slice of channels; `W_O` is the only thing that lets head 0's finding influence
the same output channel as head 1's.

In code the concat is not a `cat` — it is a transpose plus a view
(`training/nanogpt/model.py:72`, `modeling_laguna.py:414`), because the heads were only
ever a reshape of one contiguous buffer.

**Laguna does one extra thing here that is not in the standard recipe.** Between the concat
and `o_proj` it multiplies by a per-head gate: `softplus(x @ W_g)`, one scalar per head
(`modeling_laguna.py:416`, `:370`). Softplus is positive and unbounded, so this is a learned
per-head, per-token volume knob on the attention output — and unlike softmax it is *not*
constrained to sum to anything. Worth knowing that it is there before you diff Proteus
against Laguna and wonder where the extra tensor came from.

### Step: the residual add

```
X1 = X + attn_out

X1 = [[3.449490, 1.816497, 0.816497, 1.183503],
      [2.730673, 1.390171, 0.975957, 0.097679],
      [0.179837, 5.183656, 0.556062, 0.003506]]
```

Note `X`, not `H`. The residual is added to the **pre-norm** input. That is the whole
pre-norm design: the normalization applies to the sublayer's input only, and the bus itself
is never normalized in-line. It is why the residual stream's magnitude grows with depth —
each layer adds, nothing rescales — and it is the mechanism behind the "massive
activations" literature in the open problems below.

### Step: RMSNorm again, then the MLP

Second norm, same formula, on `X1`:

```
mean squares: 4.316497, 2.587795, 6.802962

H2 = [[1.660310, 0.874317, 0.392996, 0.569644],
      [1.697480, 0.864178, 0.606689, 0.060721],
      [0.068949, 1.987409, 0.213194, 0.001344]]
```

The MLP here is **SwiGLU**, the variant Laguna and Llama use
(`modeling_laguna.py:159`):

```
MLP(h) = ( SiLU(h @ W_gate) ⊙ (h @ W_up) ) @ W_down
SiLU(z) = z · sigmoid(z) = z / (1 + e^-z)
```

In words: project up to `d_ff` twice, in parallel, with two different matrices. Pass one
through SiLU. Multiply the two elementwise — that is the "gated" part, one branch
modulating the other. Project back down to `d_model`. Three matrices instead of two, which
is why SwiGLU models use `d_ff ≈ (8/3)·d_model` rather than `4·d_model`, to keep the
parameter count matched. `[C]` GLU Variants (2002.05202, Feb 2020) is four pages and is the
entire justification; it is an empirical result with no theory behind it, which the paper
says outright.

nanoGPT uses the older form — one up-projection, GELU, one down-projection
(`training/nanogpt/model.py:82–84`). Both are in the reference tree; read both.

```
G = H2 @ W_gate = H2            (W_gate = I here)
U = H2 @ W_up   = [[0.874317, 1.660310, 0.569644, 0.392996],
                   [0.864178, 1.697480, 0.060721, 0.606689],
                   [1.987409, 0.068949, 0.001344, 0.213194]]

SiLU(G)         = [[1.395125, 0.616956, 0.234620, 0.363821],
                   [1.434719, 0.607978, 0.392640, 0.031282],
                   [0.035663, 1.747864, 0.117917, 0.000672]]

SiLU(G) ⊙ U     = [[1.219781, 1.024338, 0.133650, 0.142980],
                   [1.239853, 1.032030, 0.023841, 0.018978],
                   [0.070876, 0.120514, 0.000158, 0.000143]]

mlp_out         = [[1.086131, 0.881357, 1.353431, 1.167318],
                   [1.216011, 1.013051, 1.263694, 1.051008],
                   [0.070718, 0.120371, 0.071035, 0.120657]]
```

Check one SiLU: `SiLU(1.660310) = 1.660310 / (1 + e^-1.660310) = 1.660310 / 1.190085 =
1.395125`. ✓

### Step: the second residual, and the block output

```
X2 = X1 + mlp_out

X2 = [[4.535621, 2.697854, 2.169928, 2.350822],
      [3.946684, 2.403222, 2.239651, 1.148688],
      [0.250555, 5.304027, 0.627097, 0.124163]]
```

Shape `(3, 4)`, identical to `X`. Feed it to the next block.

`[M]` The same block in PyTorch with `ε = 1e-6` in both norms gives
`[[4.535619, 2.697853, 2.169927, 2.350822], ...]` — a max absolute difference of
**2.672e-6** from the `ε = 0` hand computation. With `ε` matched at `1e-6` on both sides,
fp64 NumPy and fp64 PyTorch agree to **4.44e-16**, i.e. two ulps. Both numbers matter for
Exercise one.

### The complete shape table

Toy config, then nanoGPT shakespeare-char at `B=64` for scale.

| Tensor | Shape (toy) | Shape (nanoGPT) | fp32 bytes (nanoGPT) |
|---|---|---|---|
| token ids | `(1, 3)` | `(64, 256)` | 131 KB (int64) |
| `X` after embedding | `(3, 4)` | `(64, 256, 384)` | 25.2 MB |
| `H` after norm | `(3, 4)` | `(64, 256, 384)` | 25.2 MB |
| fused QKV | `(3, 12)` | `(64, 256, 1152)` | 75.5 MB |
| `Q`, per head | `(2, 3, 2)` | `(64, 6, 256, 64)` | 25.2 MB |
| `K`, per head | `(2, 3, 2)` | `(64, 6, 256, 64)` | 25.2 MB |
| `V`, per head | `(2, 3, 2)` | `(64, 6, 256, 64)` | 25.2 MB |
| **`S` / `A` scores** | `(2, 3, 3)` | `(64, 6, 256, 256)` | **100.7 MB** |
| `O` per head | `(2, 3, 2)` | `(64, 6, 256, 64)` | 25.2 MB |
| concat | `(3, 4)` | `(64, 256, 384)` | 25.2 MB |
| MLP hidden | `(3, 4)` | `(64, 256, 1536)` | 100.7 MB |
| block output | `(3, 4)` | `(64, 256, 384)` | 25.2 MB |

`[M]` Byte counts are arithmetic on those shapes at 4 bytes per element.

**Read the bold row.** The attention score matrix is `B · n_q · T²` elements. Everything
else in the block is linear in `T`. For this config the score matrix equals the size of
K+V at `T = 2·d_model/n_q = 128`, and at `T = 256` it is already twice K+V *per layer*.
That quadratic term is the entire reason FlashAttention exists, and it is why exercise
three is worth your evening.

### Parameters and FLOPs for one block

Parameters: `4 · d_model²` for Q/K/V/O (MHA) `+ 3 · d_model · d_ff` for SwiGLU
`+ 2 · d_model` for the two norm gains. Toy: `64 + 48 + 8 = 120`. nanoGPT
(GELU MLP, so `2 · d_model · d_ff`, plus LayerNorm weight and bias):
`589,824 + 1,179,648 + 1,536 = 1,771,008` per block, times 6 blocks — and this config leaves
`bias=True` on every `nn.Linear` (`training/nanogpt/model.py:116`), which adds a further
3,456 per block that the formula above silently drops. Counting parameters is a place to be
pedantic, not approximate.

Forward FLOPs, ignoring elementwise: `2·T·(4·d_model²)` for the projections
`+ 2·T·(3·d_model·d_ff)` for the MLP `+ 4·T²·d_model` for the two attention matmuls. The
first two terms are linear in `T`, the third is quadratic. The `6·N·D` training-budget rule
comes out of this and gets its own module.

---

## Why it matters for Proteus

**Every step above is a config field, and the config surface is the experimental surface.**

| Step | Proteus config axis | What an ablation on it tests |
|---|---|---|
| embedding | `vocab_size`, weight tying | tokenizer module; not memory-relevant |
| norm | type (RMS/LN), placement (pre/peri/hybrid), `eps`, fp32 upcast | training stability at depth; see open problems |
| Q/K/V | `n_q`, `n_kv`, `head_dim`, QK-norm on/off | **`n_kv` is the only one that changes KV bytes** |
| scores | scale, softcap | logit blowup; interacts with QK-norm |
| mask | `layer_types`, `sliding_window` | **the hybrid-ratio question, and the one place eviction is free** |
| softmax | softmax vs rectified/gated variants | attention sinks; contested |
| `W_O` | attention gating on/off | Laguna ships it, the standard recipe does not |
| MLP | `d_ff`, activation, dense vs MoE | parameter budget; MoE module |

Three connections that are load-bearing for this lab specifically.

**One: the K/V write is the Mnemosyne interface.** Look at `modeling_laguna.py:397` —
`past_key_values.update(key_states, value_states, self.layer_idx)`. Everything upstream of
that line is Proteus. Everything downstream — where those bytes live, how long, in what
precision, evicted by what policy — is Mnemosyne. The house boundary rule
(`mnemosyne → torch` only, never `proteus`) is enforceable precisely because this seam is
one function call on two tensors of shape `(B, n_kv, T, d_h)`. If a proposed policy needs
anything else from the forward pass, that is a signal the seam is in the wrong place, and
it should surface as an ADR rather than as an import.

**Two: the mask decides whether eviction is lossy.** On a sliding-window layer, a token
outside the window is *architecturally unreadable* — the mask sets it to `finfo.min` before
softmax. Discarding it is lossless. On a global layer, the same discard is a guess about
future queries. Same bytes, same data structure, completely different correctness contract,
decided by one boolean per layer. `[M]` Laguna: 12 global, 36 windowed at `w=512`
(`ASSUMPTIONS.md → reference-model`). The two-tier cache this produces in llama.cpp
(`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73`) looks like hot/cold storage
and is not: there is no promotion, no demotion, and no miss path.

**Three: `A` is the signal every attention-based eviction policy wants, and the fast path
never produces it.** `[C]` H2O (2306.14048, Jun 2023) ranks tokens by accumulated attention
mass; SnapKV (2404.14469, Apr 2024) uses the last ~32 query rows. Both need `A`. But
`F.scaled_dot_product_attention` and every fused kernel compute the softmax **online**,
tile by tile, and never materialize the `(T, T)` matrix — see
`triton_attention_helpers.py:418`. So instrumenting attention mass costs you either the slow
path (quadratic memory, which exercise three will show you cannot afford) or a custom
kernel-level reduction. `[C]` KeyDiff (2504.15364, Apr 2025) exists to avoid the problem
entirely by scoring on key geometry instead — and `[C]` The Pitfalls of KV Cache Compression
(2510.00231, Sep 2025) finds attention-free scoring among the methods that silently drop
instructions. That tension is unresolved and it is downstream of an implementation detail
in this module.

---

## Read the code

Paths are relative to `research/reference/`. Run `scripts/fetch_reference.sh` first if the
clones are not materialised.

### Read first: nanoGPT, the whole block in fifteen lines

| Pointer | What to look at |
|---|---|
| `training/nanogpt/model.py:103` | `Block.forward` — the entire pre-norm residual structure, two lines. Read this before anything else in the tree. |
| `training/nanogpt/model.py:177` | The embedding gather and, on line 178, the additive learned position embedding — the pre-RoPE way of injecting order. |
| `training/nanogpt/model.py:27` | LayerNorm via `F.layer_norm`. Compare to RMSNorm below and note what is missing: the mean subtraction. |
| `training/nanogpt/model.py:56` | One fused `c_attn` producing Q, K, V in a single `(B, T, 3C)` matmul, then split. This is why real code has no three separate projections. |
| `training/nanogpt/model.py:57` | The reshape-and-transpose that creates the head axis: `(B,T,C) -> (B,nh,T,hs)`. This line is the one people get wrong. |
| `training/nanogpt/model.py:67` | Scores and the `1/sqrt(d_h)` scale, spelled out. |
| `training/nanogpt/model.py:68` | The causal mask as `masked_fill(-inf)` against a precomputed `tril` buffer (allocated at line 49). |
| `training/nanogpt/model.py:69` | `F.softmax(att, dim=-1)` — note `dim=-1`, over keys, not queries. |
| `training/nanogpt/model.py:71` | `y = att @ v`, the weighted sum. |
| `training/nanogpt/model.py:72` | The head concat, done as transpose + `view`, not `cat`. |
| `training/nanogpt/model.py:64` | The fast path. Same call, but `scaled_dot_product_attention` — and here the `(T,T)` matrix on lines 67–71 never exists. Flip `self.flash = False` to get it back; that is what exercise three does. |
| `training/nanogpt/model.py:82` | The older MLP: up 4x, GELU, down. Contrast with SwiGLU. |

### Read second: Laguna, what actually ships

| Pointer | What to look at |
|---|---|
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:56` | RMSNorm forward. Line 58 upcasts to fp32 *before* squaring, line 59 is the mean of squares, line 60 is `rsqrt`. Three lines, and the upcast is the part that matters on bf16 hardware. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:385` | Q projection and the reshape to `(..., n_heads, head_dim)` in one expression. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:389` | QK-norm: RMSNorm applied at `head_dim` width to the queries, before RoPE. Not in the 2017 recipe; standard by 2026. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:394` | RoPE applied to Q and K — after the norm, before the cache write. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:397` | **The Mnemosyne seam.** `past_key_values.update(...)`. Post-RoPE keys go in here and never come back out modified. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:303` | `repeat_kv` — GQA implemented as an expand + reshape, so `G` query heads share one physical KV head with no copy. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:328` | Scores times `scaling`, where `scaling = head_dim**-0.5` is set once at line 351. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:330` | The mask as an **addition**, not a `masked_fill`. Compare with nanoGPT line 68 and note that both are correct. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:332` | Softmax forced to fp32 and cast back. The one place in the block where precision is non-negotiable. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:334` | `A @ V`. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:416` | The softplus per-head gate. Read it, then look at line 370 to see it is config-driven. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:424` | `o_proj`. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:159` | SwiGLU in one line. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:450` | The decoder layer: residual saved, norm, attention, add at line 462, then the same shape again for the MLP. |
| `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:547` | The embedding lookup in the production model — one line, same gather. |

### Read third: the mask as a predicate, not a tensor

| Pointer | What to look at |
|---|---|
| `architecture/transformers/src/transformers/masking_utils.py:80` | The whole causal rule: `return kv_idx <= q_idx`. Everything else is materialization. |
| `architecture/transformers/src/transformers/masking_utils.py:99` | The sliding-window overlay: `kv_idx > q_idx - sliding_window`, ANDed with the causal rule. |
| `architecture/transformers/src/transformers/masking_utils.py:610` | Boolean mask converted to float using `torch.finfo(dtype).min` (line 608), **not** `-inf`. Ask yourself why before reading the explanation above. |

### Read fourth: what the kernel actually does

| Pointer | What to look at |
|---|---|
| `memory/vllm/vllm/v1/attention/ops/triton_attention_helpers.py:418` | `softmax_step` — online softmax over one tile. The docstring is the clearest short statement of the algorithm in the tree. |
| `memory/vllm/vllm/v1/attention/ops/triton_attention_helpers.py:428` | The running maximum across tiles, and line 431's guard for a fully-masked row. |
| `memory/vllm/vllm/v1/attention/ops/triton_attention_helpers.py:433` | `P = tl.exp(S - m_j[:, None])` — the max-subtraction you did by hand, but against a max that is still provisional. |
| `memory/vllm/vllm/v1/attention/ops/triton_unified_attention.py:562` | `acc = acc * alpha[:, None]` — the accumulator rescale that fixes up earlier tiles when the running max moves. This line is why `A` never exists as a tensor, and therefore why H2O-style scoring is not free. |

---

## Exercises

All three run under the lab venv. Activate with `. .\scripts\activate-lab.ps1` from the
repo root (dot-sourced, PowerShell). Platform caveats that apply to every exercise on this
machine:

- **Native Windows only.** `[C]` WSL2 clamps the ROCm pool to the `.wslconfig` value
  (ROCm #6022, `ASSUMPTIONS.md → native-windows-over-wsl2`).
- **Keep any single tensor under 32 GiB.** `[M]` A 32 GiB buffer hard-hangs at 0% CPU with
  no error and a 36 GiB one raises `hipErrorLaunchFailure`
  (`ASSUMPTIONS.md → large-tensor-fault-32gib`, 2026-07-26). Exercise three is the only one
  that gets anywhere near this, and it is capped accordingly.
- **No result from this machine is evidence yet.** `[M]` The Hardware Validation Gate has
  not run. Exercise two produces one row of the evidence that gate needs; it does not
  discharge it.

### Exercise: compute the toy block by hand, then break PyTorch's tie

**Difficulty 2/5. 60–90 minutes of work; runs in under a second. CPU only — no GPU needed
or wanted.**

Do the arithmetic above on paper first, at least through `attn_out`. Then write a script
(~70 lines) that:

1. Builds the toy block in raw PyTorch ops — no `nn.Module` — in `float64`, using the
   matrices from this module. Reproduce every intermediate table above.
2. Builds a second implementation using `nn.Linear` and
   `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. Remember that
   `nn.Linear.weight` is `(out, in)`, so you must `copy_(W.T)`.
3. Asserts the two agree.

**What to produce.** Two numbers:

- max abs difference between your two implementations with `eps` matched — target
  `< 1e-14`; `[M]` the reference run gets **4.44e-16**;
- max abs difference between an `eps = 0` implementation and an `eps = 1e-6` one — `[M]`
  the reference run gets **2.672e-6** on the block output.

Then answer, in two sentences in your notes: which of those two numbers is the right
tolerance for a regression test on this block, and what does the other one tell you?

**Extension worth ten more minutes.** Change `n_kv` from 2 to 1 by making both query heads
read head 0's K and V. `[M]` Head 0's attention matrix is unchanged; head 1's row 2 moves
from `[0.053990, 0.928995, 0.017015]` to `[0.613383, 0.193309, 0.193309]`. Confirm that the
KV bytes halved and the query-head count did not change.

**Failure mode to expect.** If your max difference is around `1e-1` rather than `1e-15`, you
transposed a weight or you split heads before the projection instead of after. Print `Q` and
compare against the table.

### Exercise: the bf16 error budget, stage by stage, on gfx1151

**Difficulty 3/5. 90 minutes of work; runs in under a minute on GPU, about two minutes on
CPU.**

This one feeds a real open row in the register: `bf16-numerics-unproven` is `untested`, and
`[C]` five critical bf16 bugs are documented on this silicon.

Build the same block at realistic width — `d_model=384, n_q=6, d_h=64, d_ff=1536, T=256,
B=8` — with seeded random weights (`torch.manual_seed`, at least 3 seeds; a single seed is
an anecdote by house standard). Then run the identical computation three ways:

- fp64 on CPU — the reference;
- fp32 on GPU;
- bf16 on GPU, with the accumulation dtype left at whatever the op defaults to.

After each of seven stages — RMSNorm, QKV projection, `QKᵀ·scale`, mask+softmax, `A@V`,
`o_proj`, MLP — report max absolute error and max relative error against the fp64
reference, casting up for the comparison.

**What to produce.** A 7-row × 2-column table per dtype, plus one sentence on whether error
compounds across stages or resets. For calibration: `[M]` `torch.finfo(torch.bfloat16).eps`
is `0.0078125` — bf16 carries about 3 significant decimal digits, so a per-stage relative
error near `1e-2` is *expected*, not a bug. What you are looking for is (a) any stage whose
error is orders of magnitude worse than that, (b) fp32-on-GPU disagreeing with fp64-on-CPU
by more than ~`1e-6`, which would indicate a kernel problem rather than a precision one, and
(c) NaN or Inf anywhere.

**Also measure this:** force the softmax to run in bf16 rather than fp32 and re-report. The
production code upcasts (`modeling_laguna.py:332`) and you should be able to say why with a
number instead of a citation.

**CPU fallback.** Set `device='cpu'` and drop the fp32-on-GPU arm. PyTorch supports bf16 on
CPU, so you still get the precision answer. Be explicit in your notes that this answers "is
bf16 wide enough for this block" and **not** "are the gfx1151 kernels correct" — those are
different questions and only the GPU run touches the second.

**ROCm caveat.** Do not chase absolute throughput here; `[M]` this wheel reaches 20.9
TFLOPS bf16 at 8192³, 63% of the figure cited for the silicon and unexplained
(`ASSUMPTIONS.md → gemm-throughput-below-reference`). This exercise measures correctness,
not speed.

### Exercise: find the T² term, then watch it disappear

**Difficulty 3/5. 90–120 minutes of work; runs in about three minutes.**

Take nanoGPT's `CausalSelfAttention` and run it both ways: `self.flash = False` (the manual
path, lines 67–71) and `self.flash = True` (SDPA, line 64). Fix `B=8, n_q=6, d_model=384`
and sweep `T ∈ {128, 256, 512, 1024, 2048}`.

For each `(T, path)` record `torch.cuda.max_memory_allocated()` around a single forward
pass, resetting with `torch.cuda.reset_peak_memory_stats()` first. Also compute the analytic
prediction: score-matrix bytes `= B · n_q · T² · 4`, K+V bytes `= 2 · B · T · d_model · 4`,
residual bytes `= B · T · d_model · 4`.

**What to produce.** One plot, log-log, with four series: measured peak (manual), measured
peak (SDPA), predicted score-matrix bytes, predicted K+V bytes. Plus two numbers:

- the `T` at which the score matrix overtakes K+V. `[M]` The prediction is
  `T = 2·d_model/n_q = 128` for this config — verify it lands there;
- the ratio of manual peak to SDPA peak at `T = 2048`.

**The point.** The manual curve should track `T²`; the SDPA curve should not, because the
fused kernel never materializes the matrix you computed by hand. That is the whole content
of the exercise, and it is why the `A` tensor is unavailable to any eviction policy running
on a real serving stack.

**Safety cap, and it is a real one.** `[M]` Single tensors ≥32 GiB hang this machine
silently at 0% CPU. At `B=8` the largest score matrix in the sweep is
`8·6·2048²·4 = 805 MB`, comfortably safe. If you raise `B` to 64, `T=4096` produces a
24 GiB tensor — under the cliff but not by much. Do not go past `B=64, T=4096` on the manual
path without reading that assumption row first.

**CPU fallback.** `torch.cuda.max_memory_allocated` has no CPU equivalent. Instead register
a `forward` hook on each submodule that sums `tensor.numel() * tensor.element_size()` for
every output tensor, and plot that. You lose allocator behaviour (caching, fragmentation,
transient buffers) but you keep the shape arithmetic and both target numbers, and the SDPA
contrast still shows up because the hook never sees a `(T,T)` tensor. Cap the sweep at
`T = 1024` on CPU or it gets slow.

---

## Self-check

1. List every tensor in the block whose shape contains `T`, and say which of them must
   survive to the next decoding step. Give the shapes for the toy config.
2. The scale is `1/sqrt(d_h)`. Using the numbers in this module, state what happens to the
   attention distribution and to the gradient if you use `4` instead, and explain why
   `1/d_h` would be wrong in the opposite direction.
3. Row 0 of every attention matrix is exactly `[1, 0, ..., 0]` regardless of content. What
   does that imply for an eviction policy that ranks tokens by summed attention mass across
   all queries?
4. Laguna writes post-RoPE keys into the cache (`modeling_laguna.py:394` then `:397`).
   Suppose Mnemosyne evicts token 5 and compacts the cache so that token 6 now occupies
   slot 5. Exactly what breaks, and what would have to be true for it not to?
5. Matched-`eps` fp64 agreement between two implementations is `4.44e-16`; `eps = 0` versus
   `eps = 1e-6` is `2.672e-6`. Which do you use as the tolerance in a regression test, and
   why is "the tightest one that passes today" the wrong answer?
6. Where does the query-head count `n_q` appear in the formula for KV-cache bytes per token?
   Where does it appear in the arithmetic intensity of decode attention?

---

## What is still unsolved here

This module presents the forward pass as settled engineering. Most of it is. These parts
are not, and three of them are live disputes as of July 2026.

**Whether softmax's sum-to-one constraint is correct.** Every attention row must sum to 1,
so a head with nothing useful to attend to must still put its mass somewhere — and it
overwhelmingly chooses the first token. `[C]` StreamingLLM (2309.17453, Sep 2023) documented
that evicting those sink tokens collapses the model. Since then the field has split.
`[C]` Softpick (2504.20966, Apr 2025) replaces softmax with a rectified, non-sum-to-one
function and reports no sinks and no massive activations. `[C]` Gated Attention (2505.06708,
May 2025) reports the same outcome from a different direction, via an output gate — which is
notable because Laguna ships exactly such a gate (`modeling_laguna.py:416`) and nobody has
published what it does to Laguna's sinks. Against both, `[C]` "Attention Sinks Are Provably
Necessary in Softmax Transformers" (2603.11487, Mar 2026) argues sinks implement a required
conditional no-op, and `[C]` 2603.17771 (Mar 2026) argues massive activations act as
gradient regulators, i.e. that removing them costs you something in training dynamics.
`[C]` 2605.08504 (May 2026) localizes massive activations to a single layer. **Contested;
do not treat sink removal as an improvement.** For this lab the practical consequence is
narrower and firm: any eviction policy must pin a prefix, and the reason is arithmetic, not
empirical.

**Where normalization belongs.** Pre-norm is the default and is what this module teaches,
but the residual stream grows without bound with depth because nothing rescales it. `[C]`
Peri-LN (2502.02732, Feb 2025) and `[C]` HybridNorm (2503.04598, Mar 2025) both propose
placements between pre- and post-norm and both report gains; `[C]` the pre-vs-post analysis
that established pre-norm (2002.04745, Feb 2020) predates all of it. Not settled, and it is
cheap to ablate at our scale.

**Whether the attention matrix should be observable.** This is the one that matters most for
Mnemosyne. The fast kernel computes softmax online and never materializes `A`
(`triton_attention_helpers.py:418`), so the signal that `[C]` H2O (2306.14048, Jun 2023) and
`[C]` SnapKV (2404.14469, Apr 2024) rank tokens by is not available at serving time without
paying for it. `[C]` KeyDiff (2504.15364, Apr 2025) avoids `A` entirely; `[C]` The Pitfalls
of KV Cache Compression (2510.00231, Sep 2025) finds attention-free scoring among the
methods that silently drop instructions. There is no published account of what a
kernel-level attention-mass reduction actually costs in throughput, which makes it an
available experiment rather than a replication — see `research/memory/open-problems-ranked.md`.

**What the block computes.** There is no theory that predicts which head learns what, how
many heads a given width needs, or what the right head-to-KV-head ratio is. Those are chosen
empirically and then inherited. `research/memory/hybrid-architectures.md` makes the same
point about interleaving ratios: several published ratios were inherited rather than
ablated. Treat every head-count and ratio number you read, including Laguna's, as a
measurement of one training run, not a law.

**Whether the distortion introduced here has a nameable objective.** `[C]` A Rate–Distortion
View of Memory Compaction (2607.08032, Jul 2026) argues that KV eviction, prompt compression,
recurrent-state bounding and agent-memory consolidation are one problem under a resource
budget. The forward pass is where the information that later gets discarded is created, but
nobody writes down the distortion measure at this level. That is the theoretical gap the
memory track is aimed at.

**And on our own hardware:** `[M]` bf16 numerics on gfx1151 are unproven, the Hardware
Validation Gate has not run, and no number this machine produces counts as evidence until it
does. Exercise two exists to start closing that.

**Deliberately out of scope here, each with its own module:** RoPE and positional schemes;
dropout; MoE routing in place of the dense MLP; the final norm and LM head; the backward
pass; and the KV cache as an object with a lifetime, which is Track C.

---

## Answers to the self-check

**1.** Tensors containing `T`, toy config: `X (3,4)`, `H (3,4)`, `Q (2,3,2)`, `K (2,3,2)`,
`V (2,3,2)`, `S`/`A (2,3,3)`, `O (2,3,2)`, concat `(3,4)`, `attn_out (3,4)`, `X1 (3,4)`,
`H2 (3,4)`, MLP hidden `(3,4)`, `mlp_out (3,4)`, `X2 (3,4)`. Only **K and V** must survive —
shapes `(n_kv, T, d_h)` each, so `(2,3,2)` here. Everything else is recomputed from the
residual stream on the next step and is discarded. Note `A` is *not* kept, which is the
point made in the unsolved section. In nanoGPT nothing at all is kept, because training
keeps no state between steps (`training/nanogpt/model.py:56` recomputes K and V every
forward).

**2.** At scale 4 the softmax row becomes `[0, 0, 1]` with entropy `2e-6` nats — a hard
argmax. The gradient of the loss with respect to the losing logits is proportional to their
softmax probability, which is zero to machine precision, so those scores stop receiving a
learning signal and training stalls. `1/d_h` fails in the opposite direction: score standard
deviation scales as `sqrt(d_h)`, so dividing by `d_h` leaves the spread proportional to
`1/sqrt(d_h)`, shrinking toward zero as heads widen, and every distribution flattens toward
uniform. `sqrt(d_h)` is the unique power that makes the spread invariant to head width.

**3.** It means the policy receives a large, content-free vote for token 0 from every query
in every head, so a naive attention-mass ranking will always rank position 0 first
regardless of whether it carries information. Any policy built on that signal is partly
measuring an artifact of the causal mask plus the sum-to-one constraint. That is why
StreamingLLM-derived policies pin a fixed prefix explicitly rather than trusting the score
— and why "our policy learned to keep the sink tokens" is not evidence that the policy
works.

**4.** RoPE encodes *absolute* position as a rotation applied to K before storage, and
attention recovers *relative* position from the difference between the query's rotation and
the key's. Moving token 6's stored key into slot 5 does not change the rotation already
baked into it, but if any part of the system infers position from slot index — the mask
(`masking_utils.py:80` compares `kv_idx` to `q_idx`), a sliding-window bound
(`masking_utils.py:99`), a page table, or a re-application of RoPE — then position and
content now disagree. For compaction to be safe, either every position-dependent consumer
must read position from a stored per-slot position id rather than from the slot index, or
the keys must be de-rotated and re-rotated on move, which costs a pass over the cache and
loses precision. Sparse-but-not-compacted layouts (keep the slot, mark it dead) sidestep the
whole problem at the cost of fragmentation.

**5.** Use `1e-5`-ish — a tolerance derived from the largest *legitimate* source of
disagreement you have identified, which here is the `eps` convention at `2.672e-6`, with
headroom. `4.44e-16` is the agreement of two implementations that made identical
convention choices; a test at that tolerance is not testing the block, it is testing that
nobody changed `eps`, and it will fail the first time someone legitimately switches from
`1e-6` to `1e-5` or reorders two floating-point operations. "The tightest one that passes
today" encodes an accident of the current implementation as a specification. The test should
state a tolerance you can defend from the arithmetic, and a comment saying which effect it
is sized for.

**6.** `n_q` appears **nowhere** in the KV-bytes formula: `per_token_bytes = 2 · L · n_kv ·
d_h · b` (`research/memory/kv-cache-mechanics.md`). It appears in the *arithmetic intensity*
of decode attention, which is `2G/b` where `G = n_q/n_kv` — for bf16, exactly the GQA group
size. So query-head count controls decode **speed** and KV-head count controls decode
**size**, and they are routinely conflated. `[M]` On Laguna the two config fields disagree
per layer: `G = 6` on global layers and `9` on sliding ones, while `n_kv` is a uniform 8
(`ASSUMPTIONS.md → laguna-heads-uniform`, `decode-intensity-varies-by-layer`).
